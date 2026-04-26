import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from data.rf_utils import list_npz_ids, pack_angular_rf_npz, pack_raw_rf_npz
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def parse_args():
    parser = argparse.ArgumentParser(description="Quick RF encoder ablation on a single scene with camera loss.")
    parser.add_argument("--scene-root", type=str, required=True, help="Path to scene folder, e.g. dataset/AI53_001_new")
    parser.add_argument(
        "--rf-feature-root",
        type=str,
        default=None,
        help="Legacy root containing RF angular features, e.g. dataset/rf_feature_gaussian",
    )
    parser.add_argument("--rf-feature-key", type=str, default="angular_image")
    parser.add_argument("--rf-layout", choices=["legacy", "scene_local"], default="scene_local")
    parser.add_argument("--rf-subdir", type=str, default="rf_angular_images_gaussian/npz")
    parser.add_argument("--rf-pack-mode", choices=["angular_image", "dense_sparse_mask"], default="dense_sparse_mask")
    parser.add_argument("--use-raw-rf-paths", action="store_true")
    parser.add_argument("--raw-rf-subdir", type=str, default="rf")
    parser.add_argument("--raw-rf-top-k", type=int, default=64)
    parser.add_argument("--raw-rf-sort-by", choices=["pdp_power", "per_path_gain_db"], default="pdp_power")
    parser.add_argument("--rf-latent-grid", type=str, default="4,8")
    parser.add_argument("--rf-drop-prob", type=float, default=0.0)
    parser.add_argument("--fair-valid-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--methods",
        type=str,
        default="rgb_only,rf_cnn,rf_shallow_vit,rf_hybrid_vit",
        help="Comma-separated methods: rgb_only, rf_cnn, rf_shallow_vit, rf_hybrid_vit, rf_hybrid_vit_path",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=336)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to load with strict=False")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_grid(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--rf-latent-grid must be 'H,W', got {value}")
    return parts[0], parts[1]


def image_ids_from_cameras(cameras: np.lib.npyio.NpzFile) -> tuple[list[int], dict[int, int]]:
    ids = []
    row_by_id = {}
    for row, relpath in enumerate(cameras["images"]):
        if isinstance(relpath, bytes):
            relpath = relpath.decode("utf-8")
        try:
            frame_id = int(Path(str(relpath)).stem)
        except ValueError:
            frame_id = row
        ids.append(frame_id)
        row_by_id[frame_id] = row
    return ids, row_by_id


def select_frame_ids(valid_ids: List[int], num_frames: int) -> np.ndarray:
    valid_ids = sorted(valid_ids)
    if num_frames > len(valid_ids):
        raise ValueError(f"Requested {num_frames} frames, but only {len(valid_ids)} valid frames are available")
    positions = np.linspace(0, len(valid_ids) - 1, num_frames, dtype=int)
    return np.asarray([valid_ids[pos] for pos in positions], dtype=np.int64)


def load_and_resize_image(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_w, target_h), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0
    return image


def resize_intrinsics(K: np.ndarray, orig_w: int, orig_h: int, target_h: int, target_w: int) -> np.ndarray:
    K = K.copy().astype(np.float32)
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K


def angular_rf_dir(args, scene_root: Path) -> Optional[Path]:
    if args.rf_layout == "scene_local":
        return scene_root / args.rf_subdir
    if args.rf_feature_root is None:
        return None
    return Path(args.rf_feature_root) / scene_root.name / "npz"


def angular_rf_path(args, scene_root: Path, frame_id: int) -> Path:
    directory = angular_rf_dir(args, scene_root)
    if directory is None:
        raise ValueError("--rf-feature-root is required when --rf-layout legacy and an RF method is requested")
    return directory / f"{frame_id:06d}.npz"


def raw_rf_path(args, scene_root: Path, frame_id: int) -> Path:
    return scene_root / args.raw_rf_subdir / f"{frame_id:06d}.npz"


def valid_ids_for_run(args, scene_root: Path, image_ids: list[int], methods: list[str]) -> list[int]:
    image_id_set = set(image_ids)
    needs_rf = any(method != "rgb_only" for method in methods)
    needs_raw = args.use_raw_rf_paths or any(method == "rf_hybrid_vit_path" for method in methods)

    directory = angular_rf_dir(args, scene_root)
    angular_ids = list_npz_ids(directory) if directory is not None else set()
    use_rf_valid = needs_rf or (args.fair_valid_ids and bool(angular_ids))

    if use_rf_valid:
        if not angular_ids:
            raise ValueError(f"No angular RF npz files found for layout={args.rf_layout} at {directory}")
        valid_ids = image_id_set & angular_ids
        if needs_raw:
            raw_ids = list_npz_ids(scene_root / args.raw_rf_subdir)
            valid_ids &= raw_ids
    else:
        valid_ids = image_id_set

    valid_ids = sorted(valid_ids)
    if not valid_ids:
        raise ValueError("No valid frame ids after applying RF availability filters")
    return valid_ids


def build_scene_batch(
    scene_root: Path,
    args,
    frame_ids: np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
    load_rf: bool,
    load_raw_rf: bool,
) -> Dict[str, torch.Tensor]:
    cameras = np.load(scene_root / "cameras.npz")
    image_ids, row_by_id = image_ids_from_cameras(cameras)
    rows = np.asarray([row_by_id[int(frame_id)] for frame_id in frame_ids], dtype=np.int64)
    extrinsics = cameras["extrinsics"][rows, :3, :4].astype(np.float32)
    intrinsics = cameras["intrinsics"][rows].astype(np.float32)
    image_relpaths = cameras["images"][rows]
    orig_w, orig_h = int(cameras["image_size"][0]), int(cameras["image_size"][1])

    images = []
    intrinsics_resized = []
    rf_frames = []
    raw_paths = []
    raw_masks = []
    raw_globals = []

    for frame_id, image_relpath, K in zip(frame_ids, image_relpaths, intrinsics):
        if isinstance(image_relpath, bytes):
            image_relpath = image_relpath.decode("utf-8")
        image = load_and_resize_image(scene_root / str(image_relpath), target_h, target_w)
        images.append(image.transpose(2, 0, 1))
        intrinsics_resized.append(resize_intrinsics(K, orig_w, orig_h, target_h, target_w))

        if load_rf:
            rf_frames.append(
                pack_angular_rf_npz(
                    angular_rf_path(args, scene_root, int(frame_id)),
                    pack_mode=args.rf_pack_mode,
                    angular_key=args.rf_feature_key,
                )
            )

        if load_raw_rf:
            path_features, path_mask, global_features = pack_raw_rf_npz(
                raw_rf_path(args, scene_root, int(frame_id)),
                top_k=args.raw_rf_top_k,
                sort_by=args.raw_rf_sort_by,
            )
            raw_paths.append(path_features)
            raw_masks.append(path_mask)
            raw_globals.append(global_features)

    images = torch.from_numpy(np.stack(images)).unsqueeze(0).to(device)
    extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).to(device)
    intrinsics = torch.from_numpy(np.stack(intrinsics_resized)).unsqueeze(0).to(device)
    point_masks = torch.ones((1, len(frame_ids), target_h, target_w), dtype=torch.bool, device=device)

    batch = {
        "images": images,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": point_masks,
    }
    if rf_frames:
        rf = torch.from_numpy(np.stack(rf_frames)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        batch["rf"] = rf
    if raw_paths:
        batch["rf_paths"] = torch.from_numpy(np.stack(raw_paths)).unsqueeze(0).to(device)
        batch["rf_path_mask"] = torch.from_numpy(np.stack(raw_masks)).unsqueeze(0).to(device)
        batch["rf_global"] = torch.from_numpy(np.stack(raw_globals)).unsqueeze(0).to(device)

    _ = image_ids  # keep local row/id validation explicit above
    return batch


def build_method_model(
    method: str,
    target_h: int,
    target_w: int,
    checkpoint: Optional[str],
    device: torch.device,
    rf_in_chans: int,
    rf_latent_grid: tuple[int, int],
    rf_drop_prob: float,
    rf_path_feature_dim: Optional[int] = None,
    rf_global_feature_dim: Optional[int] = None,
) -> VGGT:
    aggregator_kwargs = {
        "depth": 4,
        "num_heads": 6,
        "patch_embed": "conv",
    }

    enable_rf = method != "rgb_only"
    use_raw_rf_paths = method == "rf_hybrid_vit_path"
    if method == "rf_shallow_vit":
        rf_encoder_type = "shallow_vit"
    elif method in {"rf_hybrid_vit", "rf_hybrid_vit_path"}:
        rf_encoder_type = "hybrid_vit"
    elif method == "rf_cnn":
        rf_encoder_type = "cnn"
    elif method == "rgb_only":
        rf_encoder_type = "cnn"
    else:
        raise ValueError(f"Unknown method: {method}")

    model = VGGT(
        img_size=(target_h, target_w),
        patch_size=14,
        embed_dim=384,
        aggregator_kwargs=aggregator_kwargs,
        enable_camera=True,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_rf=enable_rf,
        rf_encoder_type=rf_encoder_type,
        rf_in_chans=rf_in_chans,
        rf_encoder_hidden_dim=192,
        rf_latent_grid=rf_latent_grid,
        rf_img_size=(90, 360),
        rf_fusion_num_heads=6,
        rf_fuse_special_tokens=True,
        rf_fusion_scale_init=1e-3,
        rf_drop_prob=rf_drop_prob,
        use_raw_rf_paths=use_raw_rf_paths,
        rf_path_feature_dim=rf_path_feature_dim,
        rf_global_feature_dim=rf_global_feature_dim,
    ).to(device)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        state = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[{method}] loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

    return model


def compute_camera_loss_local(predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pred_pose_encodings = predictions["pose_enc_list"]
    gt_extrinsics = batch["extrinsics"]
    gt_intrinsics = batch["intrinsics"]
    image_hw = batch["images"].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics,
        gt_intrinsics,
        image_hw,
        pose_encoding_type="absT_quaR_FoV",
    )

    if "point_masks" in batch:
        valid_batch_mask = batch["point_masks"][:, 0].sum(dim=[-1, -2]) > 100
    else:
        valid_batch_mask = torch.ones(gt_extrinsics.shape[0], dtype=torch.bool, device=gt_extrinsics.device)
    n_stages = len(pred_pose_encodings)
    total_loss_T = total_loss_R = total_loss_FL = 0.0

    for stage_idx, pred_pose_stage in enumerate(pred_pose_encodings):
        stage_weight = 0.6 ** (n_stages - stage_idx - 1)
        if valid_batch_mask.sum() == 0:
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            pred_valid = pred_pose_stage[valid_batch_mask]
            gt_valid = gt_pose_encoding[valid_batch_mask]
            loss_T_stage = (pred_valid[..., :3] - gt_valid[..., :3]).abs().clamp(max=100).mean()
            loss_R_stage = (pred_valid[..., 3:7] - gt_valid[..., 3:7]).abs().mean()
            loss_FL_stage = (pred_valid[..., 7:] - gt_valid[..., 7:]).abs().mean()
        total_loss_T = total_loss_T + stage_weight * loss_T_stage
        total_loss_R = total_loss_R + stage_weight * loss_R_stage
        total_loss_FL = total_loss_FL + stage_weight * loss_FL_stage

    total_loss_T = total_loss_T / n_stages
    total_loss_R = total_loss_R / n_stages
    total_loss_FL = total_loss_FL / n_stages
    total_camera_loss = total_loss_T + total_loss_R + 0.5 * total_loss_FL

    return {
        "loss_camera": total_camera_loss,
        "loss_T": total_loss_T,
        "loss_R": total_loss_R,
        "loss_FL": total_loss_FL,
    }


def run_method(method: str, batch: Dict[str, torch.Tensor], args, rf_in_chans: int) -> Dict:
    device = torch.device(args.device)
    set_seed(args.seed)
    rf_path_dim = int(batch["rf_paths"].shape[-1]) if method == "rf_hybrid_vit_path" and "rf_paths" in batch else None
    rf_global_dim = int(batch["rf_global"].shape[-1]) if method == "rf_hybrid_vit_path" and "rf_global" in batch else None
    model = build_method_model(
        method,
        args.image_height,
        args.image_width,
        args.checkpoint,
        device,
        rf_in_chans=rf_in_chans,
        rf_latent_grid=parse_grid(args.rf_latent_grid),
        rf_drop_prob=args.rf_drop_prob,
        rf_path_feature_dim=rf_path_dim,
        rf_global_feature_dim=rf_global_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_inputs = {"images": batch["images"]}
    if method != "rgb_only":
        model_inputs["rf"] = batch.get("rf")
    if method == "rf_hybrid_vit_path":
        if "rf_paths" not in batch or "rf_path_mask" not in batch or "rf_global" not in batch:
            raise ValueError("rf_hybrid_vit_path requires --use-raw-rf-paths or available raw RF npz files")
        model_inputs.update(
            {
                "rf_paths": batch["rf_paths"],
                "rf_path_mask": batch["rf_path_mask"],
                "rf_global": batch["rf_global"],
            }
        )

    losses: List[float] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(**model_inputs)
        loss_dict = compute_camera_loss_local(predictions, batch)
        loss = loss_dict["loss_camera"]
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(f"[{method}] step={step:03d} loss_camera={loss_value:.6f}")

    return {
        "method": method,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_curve": losses,
    }


def main():
    args = parse_args()
    if args.image_height % 14 != 0 or args.image_width % 14 != 0:
        raise ValueError("image-height and image-width must both be divisible by patch_size=14")

    scene_root = Path(args.scene_root)
    device = torch.device(args.device)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    needs_rf = any(method != "rgb_only" for method in methods)
    needs_raw = args.use_raw_rf_paths or any(method == "rf_hybrid_vit_path" for method in methods)
    rf_in_chans = 8 if args.rf_pack_mode == "dense_sparse_mask" else 3

    cameras = np.load(scene_root / "cameras.npz")
    image_ids, _ = image_ids_from_cameras(cameras)
    valid_ids = valid_ids_for_run(args, scene_root, image_ids, methods)
    frame_ids = select_frame_ids(valid_ids, args.num_frames)
    batch = build_scene_batch(
        scene_root=scene_root,
        args=args,
        frame_ids=frame_ids,
        target_h=args.image_height,
        target_w=args.image_width,
        device=device,
        load_rf=needs_rf,
        load_raw_rf=needs_raw,
    )

    results = []
    for method in methods:
        if method != "rgb_only" and "rf" not in batch:
            raise ValueError(f"Method {method} requires RF features")
        results.append(run_method(method, batch, args, rf_in_chans=rf_in_chans))

    summary = {
        "scene_root": str(scene_root),
        "rf_feature_root": str(args.rf_feature_root) if args.rf_feature_root is not None else None,
        "rf_layout": args.rf_layout,
        "rf_subdir": args.rf_subdir,
        "rf_pack_mode": args.rf_pack_mode,
        "use_raw_rf_paths": bool(needs_raw),
        "frame_ids": frame_ids.tolist(),
        "num_frames": args.num_frames,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "steps": args.steps,
        "methods": results,
    }

    print("\nSummary")
    for item in results:
        print(
            f"{item['method']}: initial={item['initial_loss']:.6f} "
            f"final={item['final_loss']:.6f} best={item['best_loss']:.6f}"
        )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
