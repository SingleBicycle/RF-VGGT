from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points, read_depth
from data.rf_utils import list_npz_ids, pack_angular_rf_npz, pack_raw_rf_npz


def _conf_get(conf, name, default=None):
    if isinstance(conf, dict):
        return conf.get(name, default)
    return getattr(conf, name, default)


def _normalize_scene_roots(scene_roots: list[str] | str | Path) -> list[Path]:
    if isinstance(scene_roots, (str, Path)):
        raw = str(scene_roots)
        if "," in raw:
            candidates = [Path(item.strip()) for item in raw.split(",") if item.strip()]
        else:
            path = Path(raw)
            if path.is_file():
                candidates = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
            elif (path / "cameras.npz").is_file():
                candidates = [path]
            else:
                candidates = sorted(child for child in path.iterdir() if (child / "cameras.npz").is_file())
    else:
        candidates = [Path(item) for item in scene_roots]

    scene_paths = [path for path in candidates if (path / "cameras.npz").is_file()]
    if not scene_paths:
        raise ValueError(f"No scene roots with cameras.npz found from {scene_roots}")
    return scene_paths


def _frame_id_from_relpath(relpath: Any, fallback: int) -> int:
    if isinstance(relpath, bytes):
        relpath = relpath.decode("utf-8")
    try:
        return int(Path(str(relpath)).stem)
    except ValueError:
        return int(fallback)


class RFSceneDataset(BaseDataset):
    """Dataset for the scene-local RGB/RF layout backed by cameras.npz."""

    def __init__(
        self,
        common_conf,
        scene_roots: list[str] | str,
        require_rf: bool = True,
        rf_subdir: str = "rf_angular_images_gaussian/npz",
        rf_pack_mode: str = "dense_sparse_mask",
        use_raw_rf_paths: bool = False,
        raw_rf_subdir: str = "rf",
        raw_rf_top_k: int = 64,
        raw_rf_sort_by: str = "pdp_power",
        depth_dir: str | None = None,
        world_points_dir: str | None = None,
        point_mask_dir: str | None = None,
        depth_mask_dir: str | None = None,
        sampling_strategy: str = "mixed",
        nearby_window: int = 10,
        wide_baseline_prob: float = 0.3,
        random_seed: int | None = None,
        len_train: int | None = None,
        **kwargs,
    ):
        super().__init__(common_conf=common_conf)

        self.training = bool(_conf_get(common_conf, "training", True))
        self.inside_random = bool(_conf_get(common_conf, "inside_random", False))
        self.allow_duplicate_img = bool(_conf_get(common_conf, "allow_duplicate_img", True))
        self.depth_dir = depth_dir or _conf_get(common_conf, "depth_dir", "depths")
        self.world_points_dir = world_points_dir or _conf_get(common_conf, "world_points_dir", None)
        self.point_mask_dir = point_mask_dir or _conf_get(common_conf, "point_mask_dir", None)
        self.depth_mask_dir = depth_mask_dir or _conf_get(common_conf, "depth_mask_dir", None)
        self.load_depth = bool(_conf_get(common_conf, "load_depth", False) or depth_dir is not None)
        self.load_world_points = bool(_conf_get(common_conf, "load_world_points", False) or self.world_points_dir is not None)
        self.load_point_masks = bool(_conf_get(common_conf, "load_point_masks", False) or self.point_mask_dir is not None)
        self.load_depth_masks = bool(_conf_get(common_conf, "load_depth_masks", False) or self.depth_mask_dir is not None)

        self.require_rf = require_rf
        self.rf_subdir = rf_subdir
        self.rf_pack_mode = rf_pack_mode
        self.rf_feature_key = _conf_get(common_conf, "rf_feature_key", "angular_image")
        self.use_raw_rf_paths = bool(use_raw_rf_paths or _conf_get(common_conf, "use_raw_rf_paths", False))
        self.raw_rf_subdir = raw_rf_subdir
        self.raw_rf_top_k = int(raw_rf_top_k)
        self.raw_rf_sort_by = raw_rf_sort_by
        self.sampling_strategy = sampling_strategy
        self.nearby_window = int(nearby_window)
        self.wide_baseline_prob = float(wide_baseline_prob)
        self.rng = np.random.default_rng(random_seed)

        self.scenes = []
        for scene_root in _normalize_scene_roots(scene_roots):
            scene = self._load_scene(scene_root)
            if scene["valid_ids"]:
                self.scenes.append(scene)

        if not self.scenes:
            raise ValueError("No RF scenes have valid frames after RF filtering")
        self.len_train = int(len_train) if len_train is not None else len(self.scenes)

    def _load_scene(self, scene_root: Path) -> dict[str, Any]:
        cameras_path = scene_root / "cameras.npz"
        cameras = np.load(cameras_path)
        image_relpaths = cameras["images"]
        image_ids = [_frame_id_from_relpath(relpath, idx) for idx, relpath in enumerate(image_relpaths)]
        row_by_frame_id = {frame_id: row for row, frame_id in enumerate(image_ids)}
        image_id_set = set(image_ids)

        angular_rf_ids = list_npz_ids(scene_root / self.rf_subdir)
        raw_rf_ids = list_npz_ids(scene_root / self.raw_rf_subdir) if self.use_raw_rf_paths else set()

        valid_ids = set(image_id_set)
        if self.require_rf:
            valid_ids &= angular_rf_ids
            if self.use_raw_rf_paths:
                valid_ids &= raw_rf_ids
        valid_ids = sorted(valid_ids)

        depth_path_by_id = self._index_paths(scene_root, self.depth_dir) if self.load_depth else {}
        world_points_path_by_id = self._index_paths(scene_root, self.world_points_dir) if self.load_world_points else {}
        point_mask_path_by_id = self._index_paths(scene_root, self.point_mask_dir) if self.load_point_masks else {}
        depth_mask_path_by_id = self._index_paths(scene_root, self.depth_mask_dir) if self.load_depth_masks else {}

        extrinsics = cameras["extrinsics"].astype(np.float32)
        if extrinsics.shape[-2:] == (4, 4):
            extrinsics = extrinsics[:, :3, :4]
        intrinsics = cameras["intrinsics"].astype(np.float32)
        image_size = cameras["image_size"].astype(np.int64) if "image_size" in cameras.files else None
        camera_centers = self._camera_centers(extrinsics)

        return {
            "scene_root": scene_root,
            "seq_name": scene_root.name,
            "image_relpaths": image_relpaths,
            "row_by_frame_id": row_by_frame_id,
            "valid_ids": valid_ids,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "image_size": image_size,
            "camera_centers": camera_centers,
            "depth_path_by_id": depth_path_by_id,
            "world_points_path_by_id": world_points_path_by_id,
            "point_mask_path_by_id": point_mask_path_by_id,
            "depth_mask_path_by_id": depth_mask_path_by_id,
        }

    def _index_paths(self, scene_root: Path, dirname: str | None) -> dict[int, Path]:
        if dirname is None:
            return {}
        root = scene_root / dirname
        if not root.is_dir():
            return {}
        paths: dict[int, Path] = {}
        for suffix in ("*.npy", "*.npz", "*.exr", "*.png"):
            for path in root.glob(suffix):
                try:
                    paths[int(path.stem)] = path
                except ValueError:
                    continue
        return paths

    def _camera_centers(self, extrinsics: np.ndarray) -> np.ndarray:
        centers = []
        for extri in extrinsics:
            R = extri[:3, :3]
            t = extri[:3, 3]
            centers.append(-(R.T @ t))
        return np.asarray(centers, dtype=np.float32)

    def _sample_ids(self, scene: dict[str, Any], img_per_seq: int) -> np.ndarray:
        valid_ids = np.asarray(scene["valid_ids"], dtype=np.int64)
        if img_per_seq <= 0:
            raise ValueError(f"img_per_seq must be positive, got {img_per_seq}")
        if len(valid_ids) == 0:
            raise ValueError(f"Scene {scene['scene_root']} has no valid ids")

        strategy = self.sampling_strategy
        if strategy == "mixed":
            r = self.rng.random()
            if r < self.wide_baseline_prob:
                strategy = "wide"
            elif r < self.wide_baseline_prob + 0.4:
                strategy = "nearby"
            else:
                strategy = "uniform"

        if strategy == "uniform":
            replace = img_per_seq > len(valid_ids) and self.allow_duplicate_img
            return self.rng.choice(valid_ids, size=img_per_seq, replace=replace)

        if strategy == "nearby":
            anchor = int(self.rng.choice(valid_ids))
            nearby = valid_ids[np.abs(valid_ids - anchor) <= self.nearby_window]
            if len(nearby) == 0:
                nearby = valid_ids
            replace = img_per_seq > len(nearby) and self.allow_duplicate_img
            sampled = self.rng.choice(nearby, size=max(img_per_seq - 1, 0), replace=replace)
            return np.asarray([anchor, *sampled.tolist()], dtype=np.int64)[:img_per_seq]

        if strategy == "wide":
            return self._sample_wide_ids(scene, valid_ids, img_per_seq)

        raise ValueError(f"Unknown RFSceneDataset sampling_strategy '{self.sampling_strategy}'")

    def _sample_wide_ids(self, scene: dict[str, Any], valid_ids: np.ndarray, img_per_seq: int) -> np.ndarray:
        if img_per_seq == 1:
            return np.asarray([int(self.rng.choice(valid_ids))], dtype=np.int64)
        if img_per_seq > len(valid_ids) and self.allow_duplicate_img:
            valid_ids = self.rng.choice(valid_ids, size=img_per_seq, replace=True)

        selected = [int(self.rng.choice(valid_ids))]
        centers = scene["camera_centers"]
        for _ in range(1, min(img_per_seq, len(valid_ids))):
            remaining = [int(frame_id) for frame_id in valid_ids if int(frame_id) not in selected]
            if not remaining:
                break
            selected_rows = [scene["row_by_frame_id"][frame_id] for frame_id in selected]
            remaining_rows = [scene["row_by_frame_id"][frame_id] for frame_id in remaining]
            dists = np.linalg.norm(
                centers[np.asarray(remaining_rows)][:, None, :] - centers[np.asarray(selected_rows)][None, :, :],
                axis=-1,
            )
            selected.append(remaining[int(np.argmax(dists.min(axis=1)))])

        while len(selected) < img_per_seq:
            selected.append(int(self.rng.choice(valid_ids)))
        return np.asarray(selected[:img_per_seq], dtype=np.int64)

    def _resize_image_intrinsics(
        self,
        image: Image.Image,
        intrinsics: np.ndarray,
        target_image_shape: np.ndarray,
        image_size_wh: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_h, target_w = int(target_image_shape[0]), int(target_image_shape[1])
        if image_size_wh is not None:
            orig_w, orig_h = int(image_size_wh[0]), int(image_size_wh[1])
        else:
            orig_w, orig_h = image.size

        image = image.resize((target_w, target_h), Image.BILINEAR)
        K = intrinsics.copy().astype(np.float32)
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)
        K[0, 0] *= scale_x
        K[0, 2] *= scale_x
        K[1, 1] *= scale_y
        K[1, 2] *= scale_y
        return np.asarray(image, dtype=np.uint8), K

    def _load_depth(self, path: Path, target_image_shape: np.ndarray) -> np.ndarray:
        if path.suffix == ".npy":
            depth = np.load(path).astype(np.float32)
        elif path.suffix == ".npz":
            with np.load(path) as data:
                key = "depth" if "depth" in data else "depths" if "depths" in data else "arr_0"
                depth = data[key].astype(np.float32)
        else:
            depth = read_depth(str(path)).astype(np.float32)
        target_h, target_w = int(target_image_shape[0]), int(target_image_shape[1])
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth[~np.isfinite(depth)] = 0.0
        return depth.astype(np.float32)

    def _load_optional_array(self, path: Path, target_image_shape: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            with np.load(path) as data:
                preferred = ("world_points", "points", "point_mask", "mask", "depth_mask", "arr_0")
                key = next((name for name in preferred if name in data), data.files[0])
                arr = data[key]
        else:
            arr = read_depth(str(path))
        arr = np.asarray(arr)
        target_h, target_w = int(target_image_shape[0]), int(target_image_shape[1])
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        if arr.ndim == 2:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=interp)
        elif arr.ndim == 3:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=interp)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if is_mask:
            return arr.astype(bool)
        return arr.astype(np.float32)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = int(self.rng.integers(0, len(self.scenes)))
        elif seq_index is None:
            seq_index = 0
        scene = self.scenes[int(seq_index) % len(self.scenes)]

        if img_per_seq is None:
            img_per_seq = 1
        frame_ids = np.asarray(ids, dtype=np.int64) if ids is not None else self._sample_ids(scene, img_per_seq)
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        extrinsics = []
        intrinsics = []
        rf_frames = []
        raw_paths = []
        raw_masks = []
        raw_globals = []
        raw_ranges = []
        raw_los = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        depth_masks = []

        has_depth = self.load_depth and all(int(frame_id) in scene["depth_path_by_id"] for frame_id in frame_ids)
        has_world_points = self.load_world_points and all(
            int(frame_id) in scene["world_points_path_by_id"] for frame_id in frame_ids
        )
        has_point_masks = self.load_point_masks and all(
            int(frame_id) in scene["point_mask_path_by_id"] for frame_id in frame_ids
        )
        has_depth_masks = self.load_depth_masks and all(
            int(frame_id) in scene["depth_mask_path_by_id"] for frame_id in frame_ids
        )
        has_angular_rf = all((scene["scene_root"] / self.rf_subdir / f"{int(frame_id):06d}.npz").is_file() for frame_id in frame_ids)
        has_raw_rf = self.use_raw_rf_paths and all(
            (scene["scene_root"] / self.raw_rf_subdir / f"{int(frame_id):06d}.npz").is_file() for frame_id in frame_ids
        )
        if self.require_rf and not has_angular_rf:
            raise FileNotFoundError(f"Selected ids {frame_ids.tolist()} include frames without angular RF in {scene['scene_root'] / self.rf_subdir}")
        if self.require_rf and self.use_raw_rf_paths and not has_raw_rf:
            raise FileNotFoundError(f"Selected ids {frame_ids.tolist()} include frames without raw RF in {scene['scene_root'] / self.raw_rf_subdir}")

        for frame_id in frame_ids:
            frame_id = int(frame_id)
            row = scene["row_by_frame_id"][frame_id]
            image_relpath = scene["image_relpaths"][row]
            if isinstance(image_relpath, bytes):
                image_relpath = image_relpath.decode("utf-8")
            image_path = scene["scene_root"] / str(image_relpath)
            image = Image.open(image_path).convert("RGB")
            image_np, K = self._resize_image_intrinsics(
                image,
                scene["intrinsics"][row],
                target_image_shape,
                scene["image_size"],
            )

            extri = scene["extrinsics"][row].astype(np.float32)
            images.append(image_np)
            extrinsics.append(extri)
            intrinsics.append(K)

            if has_angular_rf:
                rf_frames.append(
                    pack_angular_rf_npz(
                        scene["scene_root"] / self.rf_subdir / f"{frame_id:06d}.npz",
                        pack_mode=self.rf_pack_mode,
                        angular_key=self.rf_feature_key,
                    )
                )

            if has_raw_rf:
                path_features, path_mask, global_features, range_m, los_range_m = pack_raw_rf_npz(
                    scene["scene_root"] / self.raw_rf_subdir / f"{frame_id:06d}.npz",
                    top_k=self.raw_rf_top_k,
                    sort_by=self.raw_rf_sort_by,
                )
                raw_paths.append(path_features)
                raw_masks.append(path_mask)
                raw_globals.append(global_features)
                raw_ranges.append(range_m)
                raw_los.append(los_range_m)

            if has_depth:
                depth = self._load_depth(scene["depth_path_by_id"][frame_id], target_image_shape)
                world, cam, mask = depth_to_world_coords_points(depth, extri, K)
                depths.append(depth)
                cam_points.append(cam)
                world_points.append(world)
                point_masks.append(mask)

            if has_world_points:
                explicit_world = self._load_optional_array(
                    scene["world_points_path_by_id"][frame_id],
                    target_image_shape,
                    is_mask=False,
                )
                world_points[-1:] = [explicit_world] if world_points else []
                if not has_depth:
                    world_points.append(explicit_world)

            if has_point_masks:
                explicit_mask = self._load_optional_array(
                    scene["point_mask_path_by_id"][frame_id],
                    target_image_shape,
                    is_mask=True,
                )
                point_masks[-1:] = [explicit_mask] if point_masks else []
                if not has_depth:
                    point_masks.append(explicit_mask)

            if has_depth_masks:
                depth_masks.append(
                    self._load_optional_array(
                        scene["depth_mask_path_by_id"][frame_id],
                        target_image_shape,
                        is_mask=True,
                    )
                )

        batch = {
            "seq_name": scene["seq_name"],
            "scene_root": str(scene["scene_root"]),
            "ids": frame_ids.astype(np.int64),
            "frame_num": len(frame_ids),
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }
        if rf_frames:
            batch["rf"] = np.stack(rf_frames)

        if raw_paths:
            batch["rf_paths"] = np.stack(raw_paths)
            batch["rf_path_mask"] = np.stack(raw_masks)
            batch["rf_global"] = np.stack(raw_globals)
            batch["rf_path_range_m"] = np.stack(raw_ranges)
            batch["rf_los_range_m"] = np.asarray(raw_los, dtype=np.float32)

        if has_depth:
            batch.update(
                {
                    "depths": depths,
                    "cam_points": cam_points,
                }
            )
        if world_points:
            batch["world_points"] = world_points
        if point_masks:
            batch["point_masks"] = point_masks
        if depth_masks:
            batch["depth_masks"] = depth_masks

        return batch
