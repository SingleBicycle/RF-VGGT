# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from pathlib import Path

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .rf_utils import pack_angular_rf_npz, pack_raw_rf_npz
from .track_util import *
from .augmentation import get_image_augmentation


def _conf_get(conf, name, default=None):
    if isinstance(conf, dict):
        return conf.get(name, default)
    return getattr(conf, name, default)


def _convert_optional_sequence_to_tensor(data, dtype=None):
    if data is None:
        return None

    if torch.is_tensor(data):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return None
        if torch.is_tensor(data[0]):
            tensor = torch.stack(list(data), dim=0)
        else:
            tensor = torch.from_numpy(np.stack(data))
    else:
        return None

    if tensor.ndim == 4 and tensor.shape[-1] <= 8:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()

    if dtype is not None:
        tensor = tensor.to(dtype)

    return tensor.contiguous()


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config
        self.rf_feature_root = _conf_get(common_config, "rf_feature_root", None)
        self.rf_layout = _conf_get(common_config, "rf_layout", "legacy")
        self.rf_subdir = _conf_get(common_config, "rf_subdir", "rf_angular_images_gaussian/npz")
        self.rf_feature_key = _conf_get(common_config, "rf_feature_key", "angular_image")
        self.rf_pack_mode = _conf_get(common_config, "rf_pack_mode", "dense_sparse_mask")
        self.allow_missing_rf = bool(_conf_get(common_config, "allow_missing_rf", False))
        self.use_raw_rf_paths = bool(_conf_get(common_config, "use_raw_rf_paths", False))
        self.raw_rf_subdir = _conf_get(common_config, "raw_rf_subdir", "rf")
        self.raw_rf_top_k = int(_conf_get(common_config, "raw_rf_top_k", 64))
        self.raw_rf_sort_by = _conf_get(common_config, "raw_rf_sort_by", "pdp_power")
        self.allow_missing_raw_rf = bool(_conf_get(common_config, "allow_missing_raw_rf", False))

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples

    def _resolve_scene_root(self, seq_name: str, scene_root=None):
        if scene_root is not None:
            return Path(scene_root)
        if self.rf_feature_root is None:
            return None
        return Path(self.rf_feature_root) / seq_name

    def _load_rf_sequence_from_disk(self, seq_name: str, ids: torch.Tensor, scene_root=None):
        if self.rf_layout == "legacy" and self.rf_feature_root is None:
            return None
        if self.rf_layout == "scene_local" and scene_root is None and self.rf_feature_root is None:
            return None

        rf_frames = []
        for frame_id in ids.tolist():
            if self.rf_layout == "legacy":
                rf_path = Path(self.rf_feature_root) / seq_name / "npz" / f"{int(frame_id):06d}.npz"
            elif self.rf_layout == "scene_local":
                resolved_scene_root = self._resolve_scene_root(seq_name, scene_root)
                if resolved_scene_root is None:
                    return None
                rf_path = resolved_scene_root / self.rf_subdir / f"{int(frame_id):06d}.npz"
            else:
                raise ValueError(f"Unsupported rf_layout '{self.rf_layout}'")

            if not rf_path.is_file() and self.allow_missing_rf:
                return None
            rf_frames.append(
                pack_angular_rf_npz(
                    rf_path,
                    pack_mode=self.rf_pack_mode,
                    angular_key=self.rf_feature_key,
                )
            )

        return np.stack(rf_frames)

    def _load_raw_rf_sequence_from_disk(self, seq_name: str, ids: torch.Tensor, scene_root=None):
        if not self.use_raw_rf_paths:
            return None

        if self.rf_layout == "scene_local":
            resolved_scene_root = self._resolve_scene_root(seq_name, scene_root)
        else:
            resolved_scene_root = self._resolve_scene_root(seq_name, scene_root)
        if resolved_scene_root is None:
            return None

        path_features = []
        path_masks = []
        global_features = []
        range_meters = []
        los_ranges = []
        for frame_id in ids.tolist():
            rf_path = resolved_scene_root / self.raw_rf_subdir / f"{int(frame_id):06d}.npz"
            if not rf_path.is_file() and self.allow_missing_raw_rf:
                return None
            features, mask, global_feat, range_m, los_range_m = pack_raw_rf_npz(
                rf_path,
                top_k=self.raw_rf_top_k,
                sort_by=self.raw_rf_sort_by,
            )
            path_features.append(features)
            path_masks.append(mask)
            global_features.append(global_feat)
            range_meters.append(range_m)
            los_ranges.append(los_range_m)

        return {
            "rf_paths": np.stack(path_features),
            "rf_path_mask": np.stack(path_masks),
            "rf_global": np.stack(global_features),
            "rf_path_range_m": np.stack(range_meters),
            "rf_los_range_m": np.asarray(los_ranges, dtype=np.float32),
        }

    def _stack_optional_tensor(self, batch: dict, key: str, dtype=np.float32):
        value = batch.get(key)
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and (len(value) == 0 or any(v is None for v in value)):
            return None
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.from_numpy(np.stack(value).astype(dtype))
        return tensor.contiguous()

    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]
        scene_root = batch.get("scene_root")

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types. Geometry keys are optional
        # for camera-only RF scenes.
        depths = self._stack_optional_tensor(batch, "depths", np.float32)
        extrinsics = self._stack_optional_tensor(batch, "extrinsics", np.float32)
        intrinsics = self._stack_optional_tensor(batch, "intrinsics", np.float32)
        cam_points = self._stack_optional_tensor(batch, "cam_points", np.float32)
        world_points = self._stack_optional_tensor(batch, "world_points", np.float32)
        point_masks = self._stack_optional_tensor(batch, "point_masks", bool)
        depth_masks = self._stack_optional_tensor(batch, "depth_masks", bool)
        ids = torch.from_numpy(np.asarray(batch["ids"], dtype=np.int64))    # Frame indices sampled from the original sequence


        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
        }
        optional_tensors = {
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "depth_masks": depth_masks,
        }
        sample.update({key: value for key, value in optional_tensors.items() if value is not None})
        if scene_root is not None:
            sample["scene_root"] = str(scene_root)

        rf = _convert_optional_sequence_to_tensor(batch.get("rf"), dtype=torch.get_default_dtype())
        if rf is None:
            rf = _convert_optional_sequence_to_tensor(
                self._load_rf_sequence_from_disk(seq_name, ids, scene_root=scene_root),
                dtype=torch.get_default_dtype(),
            )
        if rf is not None:
            sample["rf"] = rf

        if self.use_raw_rf_paths:
            raw_rf_batch = {
                "rf_paths": batch.get("rf_paths"),
                "rf_path_mask": batch.get("rf_path_mask"),
                "rf_global": batch.get("rf_global"),
                "rf_path_range_m": batch.get("rf_path_range_m"),
                "rf_los_range_m": batch.get("rf_los_range_m"),
            }
            if raw_rf_batch["rf_paths"] is None or raw_rf_batch["rf_path_mask"] is None or raw_rf_batch["rf_global"] is None:
                loaded_raw = self._load_raw_rf_sequence_from_disk(seq_name, ids, scene_root=scene_root)
                if loaded_raw is not None:
                    raw_rf_batch = loaded_raw

            rf_paths = _convert_optional_sequence_to_tensor(raw_rf_batch.get("rf_paths"), dtype=torch.get_default_dtype())
            rf_path_mask = _convert_optional_sequence_to_tensor(raw_rf_batch.get("rf_path_mask"), dtype=torch.bool)
            rf_global = _convert_optional_sequence_to_tensor(raw_rf_batch.get("rf_global"), dtype=torch.get_default_dtype())
            rf_path_range_m = _convert_optional_sequence_to_tensor(raw_rf_batch.get("rf_path_range_m"), dtype=torch.get_default_dtype())
            rf_los_range_m = _convert_optional_sequence_to_tensor(raw_rf_batch.get("rf_los_range_m"), dtype=torch.get_default_dtype())
            if rf_paths is not None and rf_path_mask is not None and rf_global is not None:
                sample["rf_paths"] = rf_paths
                sample["rf_path_mask"] = rf_path_mask
                sample["rf_global"] = rf_global
                if rf_path_range_m is not None:
                    sample["rf_path_range_m"] = rf_path_range_m
                if rf_los_range_m is not None:
                    sample["rf_los_range_m"] = rf_los_range_m

        # --- Track Processing (if enabled) ---
        if self.load_track:
            if "point_masks" not in sample or "world_points" not in sample or "depths" not in sample:
                raise KeyError("Track loading requires depth/world point geometry, but this batch is camera-only.")
            if batch["tracks"] is not None:
                # Use pre-computed tracks from the dataset
                tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
                track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

                # Sample a subset of tracks randomly
                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    # If we have enough tracks, sample without replacement
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    # If not enough tracks, sample with replacement (allow duplicates)
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
                                                    (self.track_num,),
                                                    dtype=torch.int64,
                                                    device=valid_indices.device)]

                # Extract the sampled tracks and their masks
                tracks = tracks[:, sampled_indices, :]
                track_vis_mask = track_vis_mask[:, sampled_indices]
                track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

            else:
                # Generate tracks on-the-fly using depth information
                # This creates synthetic tracks based on the 3D information available
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics, intrinsics, world_points, depths, point_masks, images,
                    target_track_num=self.track_num, seq_name=seq_name
                )

            # Add track information to the sample dictionary
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask
            sample["track_positive_mask"] = track_positive_mask

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
