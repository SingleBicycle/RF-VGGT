# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.models.rf_modules import RFScaleHead
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        aggregator_kwargs=None,
        enable_rf=False,
        rf_encoder_type="cnn",
        rf_in_chans=8,
        rf_encoder_hidden_dim=256,
        rf_latent_grid=(4, 8),
        rf_img_size=(90, 360),
        rf_fusion_num_heads=8,
        rf_fuse_special_tokens=True,
        rf_fusion_scale_init=1e-3,
        rf_drop_prob=0.0,
        use_raw_rf_paths=False,
        rf_path_feature_dim=None,
        rf_path_token_count=16,
        rf_global_feature_dim=None,
        rf_global_token_count=1,
        rf_fusion_layers="pre",
        rf_fusion_every_n_blocks=0,
        use_rf=None,
        rf=None,
    ):
        super().__init__()

        if use_rf is not None:
            enable_rf = bool(use_rf)
        rf_cfg = rf
        if rf_cfg is not None:
            enable_rf = bool(_cfg_get(rf_cfg, "enabled", enable_rf))
            rf_encoder_type = _cfg_get(rf_cfg, "encoder_type", rf_encoder_type)
            rf_in_chans = int(_cfg_get(rf_cfg, "in_channels", rf_in_chans))
            rf_img_size = _cfg_get(rf_cfg, "angular_size", rf_img_size)
            adapter_cfg = _cfg_get(rf_cfg, "adapter", None)
            rf_fusion_num_heads = int(
                _cfg_get(adapter_cfg, "num_heads", _cfg_get(_cfg_get(rf_cfg, "scene_sync", None), "num_heads", rf_fusion_num_heads))
            )
            rf_fuse_special_tokens = bool(_cfg_get(adapter_cfg, "rf_fuse_special_tokens", rf_fuse_special_tokens))
            rf_fusion_scale_init = float(_cfg_get(adapter_cfg, "fusion_scale_init", rf_fusion_scale_init))
            use_raw_rf_paths = bool(_cfg_get(rf_cfg, "use_paths", use_raw_rf_paths))
            path_cfg = _cfg_get(rf_cfg, "path_encoder", None)
            rf_path_feature_dim = int(_cfg_get(path_cfg, "path_dim", rf_path_feature_dim or 17))
            rf_path_token_count = int(_cfg_get(path_cfg, "max_path_tokens", rf_path_token_count))
            rf_global_feature_dim = int(_cfg_get(rf_cfg, "global_dim", rf_global_feature_dim or 7))

        aggregator_kwargs = {} if aggregator_kwargs is None else dict(aggregator_kwargs)
        rf_aggregator_kwargs = {
            "enable_rf": enable_rf,
            "rf_encoder_type": rf_encoder_type,
            "rf_in_chans": rf_in_chans,
            "rf_encoder_hidden_dim": rf_encoder_hidden_dim,
            "rf_latent_grid": rf_latent_grid,
            "rf_img_size": rf_img_size,
            "rf_fusion_num_heads": rf_fusion_num_heads,
            "rf_fuse_special_tokens": rf_fuse_special_tokens,
            "rf_fusion_scale_init": rf_fusion_scale_init,
            "rf_drop_prob": rf_drop_prob,
            "use_raw_rf_paths": use_raw_rf_paths,
            "rf_path_feature_dim": rf_path_feature_dim,
            "rf_path_token_count": rf_path_token_count,
            "rf_global_feature_dim": rf_global_feature_dim,
            "rf_global_token_count": rf_global_token_count,
            "rf_fusion_layers": rf_fusion_layers,
            "rf_fusion_every_n_blocks": rf_fusion_every_n_blocks,
            "rf_final_config": rf_cfg,
        }
        for key in list(rf_aggregator_kwargs):
            if key in aggregator_kwargs:
                rf_aggregator_kwargs[key] = aggregator_kwargs.pop(key)

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **aggregator_kwargs,
            **rf_aggregator_kwargs,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # RF-only metric-scale head: predicts log absolute scale from RF paths/global (no images).
        # Resolves the metric-scale ambiguity that image-only feed-forward reconstruction cannot.
        self.rf_scale_head = (
            RFScaleHead(path_dim=int(rf_path_feature_dim or 17), global_dim=int(rf_global_feature_dim or 7))
            if enable_rf
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        rf: torch.Tensor = None,
        rf_paths: torch.Tensor = None,
        rf_path_mask: torch.Tensor = None,
        rf_global: torch.Tensor = None,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            rf (torch.Tensor, optional): RF angular images with shape [S, C_rf, H_rf, W_rf] or [B, S, C_rf, H_rf, W_rf].
            rf_paths (torch.Tensor, optional): Raw RF path features with shape [S, K, F] or [B, S, K, F].
            rf_path_mask (torch.Tensor, optional): Raw RF path mask with shape [S, K] or [B, S, K].
            rf_global (torch.Tensor, optional): Raw RF global features with shape [S, G] or [B, S, G].
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if rf is not None and len(rf.shape) == 4:
            rf = rf.unsqueeze(0)

        if rf_paths is not None and len(rf_paths.shape) == 3:
            rf_paths = rf_paths.unsqueeze(0)

        if rf_path_mask is not None and len(rf_path_mask.shape) == 2:
            rf_path_mask = rf_path_mask.unsqueeze(0)

        if rf_global is not None and len(rf_global.shape) == 2:
            rf_global = rf_global.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            rf=rf,
            rf_paths=rf_paths,
            rf_path_mask=rf_path_mask,
            rf_global=rf_global,
        )

        predictions = {"rf_aux": getattr(self.aggregator, "latest_rf_aux", {})}

        # RF-only metric-scale prediction (feed-forward). Absent when RF is not provided (RF-off),
        # which is exactly the point: without RF there is no per-sample metric-scale mechanism.
        if self.rf_scale_head is not None and rf_paths is not None:
            predictions["rf_log_scale"] = self.rf_scale_head(rf_paths, rf_path_mask, rf_global)

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
