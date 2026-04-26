# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, Sequence, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.models.rf_modules import (
    AngularRFEncoderV2,
    DEFAULT_RAW_RF_GLOBAL_FEATURE_DIM,
    DEFAULT_RAW_RF_PATH_FEATURE_DIM,
    RFGlobalTokenEncoderV2,
    RFGatedAdapterV2,
    RFPathSetEncoderV2,
    RFSceneSyncTransformer,
    build_rf_encoder,
    RFGatedTokenFusion,
    RFGlobalTokenEncoder,
    RFPathSetEncoder,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_tuple2(value, default):
    if value is None:
        value = default
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def resolve_rf_adapter_layers(
    num_blocks: int,
    layers: Union[str, Sequence[int]] = "every_4_plus_last",
    every: int = 4,
    include_last: bool = True,
) -> List[int]:
    if layers is None or layers is False:
        return []
    if isinstance(layers, str):
        layers = layers.lower()
        if layers in {"none", "off", "disabled"}:
            return []
        if layers == "every_4_plus_last":
            ids = list(range(max(every - 1, 0), num_blocks, max(every, 1)))
            if include_last and num_blocks > 0 and (num_blocks - 1) not in ids:
                ids.append(num_blocks - 1)
            return sorted(set(i for i in ids if 0 <= i < num_blocks))
        if layers == "last":
            return [num_blocks - 1] if num_blocks > 0 else []
        raise ValueError(f"Unsupported rf_adapter_layers mode: {layers}")
    return sorted(set(int(i) for i in layers if 0 <= int(i) < num_blocks))


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
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
        rf_final_config=None,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)
        self.enable_rf = enable_rf
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.rf_fuse_special_tokens = bool(rf_fuse_special_tokens)
        self.rf_drop_prob = float(rf_drop_prob)
        self.use_raw_rf_paths = bool(use_raw_rf_paths)
        self.rf_fusion_layers = rf_fusion_layers
        self.rf_fusion_every_n_blocks = int(rf_fusion_every_n_blocks or 0)
        self.rf_final_config = rf_final_config
        final_encoder_type = _cfg_get(rf_final_config, "encoder_type", rf_encoder_type)
        self.rf_final_enabled = bool(self.enable_rf and str(final_encoder_type).lower() == "final_v2")
        self.latest_rf_aux: Dict[str, torch.Tensor] = {}

        if self.enable_rf and self.rf_final_enabled:
            self._build_final_rf_modules(
                embed_dim=embed_dim,
                hidden_dim=rf_encoder_hidden_dim,
                num_heads=rf_fusion_num_heads,
                num_register_tokens=num_register_tokens,
                fallback_rf_img_size=rf_img_size,
                fallback_in_chans=rf_in_chans,
            )
        elif self.enable_rf:
            self.rf_encoder = build_rf_encoder(
                encoder_type=rf_encoder_type,
                in_chans=rf_in_chans,
                hidden_dim=rf_encoder_hidden_dim,
                embed_dim=embed_dim,
                latent_grid=rf_latent_grid,
                rf_img_size=rf_img_size,
            )
            self.rf_fusion = RFGatedTokenFusion(
                embed_dim=embed_dim,
                num_heads=rf_fusion_num_heads,
                fusion_scale_init=rf_fusion_scale_init,
                dropout=0.0,
            )
            self.rf_path_encoder = (
                RFPathSetEncoder(
                    feature_dim=rf_path_feature_dim or DEFAULT_RAW_RF_PATH_FEATURE_DIM,
                    embed_dim=embed_dim,
                    hidden_dim=rf_encoder_hidden_dim,
                    num_latents=rf_path_token_count,
                    num_heads=rf_fusion_num_heads,
                )
                if self.use_raw_rf_paths
                else None
            )
            self.rf_global_encoder = (
                RFGlobalTokenEncoder(
                    feature_dim=rf_global_feature_dim or DEFAULT_RAW_RF_GLOBAL_FEATURE_DIM,
                    embed_dim=embed_dim,
                    hidden_dim=rf_encoder_hidden_dim,
                    num_tokens=rf_global_token_count,
                )
                if (self.use_raw_rf_paths or rf_global_feature_dim is not None)
                else None
            )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self._finish_final_rf_adapters(rf_fusion_num_heads, num_register_tokens)

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def _build_final_rf_modules(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_register_tokens: int,
        fallback_rf_img_size,
        fallback_in_chans: int,
    ) -> None:
        rf_cfg = self.rf_final_config
        angular_cfg = _cfg_get(rf_cfg, "angular_encoder", {}) or {}
        path_cfg = _cfg_get(rf_cfg, "path_encoder", {}) or {}
        sync_cfg = _cfg_get(rf_cfg, "scene_sync", {}) or {}
        adapter_cfg = _cfg_get(rf_cfg, "adapter", {}) or {}

        self.rf_use_angular = bool(_cfg_get(rf_cfg, "use_angular", True))
        self.rf_use_paths = bool(_cfg_get(rf_cfg, "use_paths", True))
        self.rf_use_global = bool(_cfg_get(rf_cfg, "use_global", True))
        rf_embed_dim = int(_cfg_get(rf_cfg, "embed_dim", embed_dim) or embed_dim)
        if rf_embed_dim != embed_dim:
            raise ValueError(f"Final RF embed_dim ({rf_embed_dim}) must match VGGT embed_dim ({embed_dim}).")

        angular_size = _as_tuple2(_cfg_get(rf_cfg, "angular_size", fallback_rf_img_size), fallback_rf_img_size)
        angular_grid = _as_tuple2(_cfg_get(rf_cfg, "angular_token_grid", (9, 36)), (9, 36))
        in_channels = int(_cfg_get(rf_cfg, "in_channels", fallback_in_chans) or fallback_in_chans)

        self.rf_encoder = (
            AngularRFEncoderV2(
                in_channels=in_channels,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                angular_size=angular_size,
                angular_token_grid=angular_grid,
                append_summary_token=bool(_cfg_get(angular_cfg, "append_summary_token", True)),
            )
            if self.rf_use_angular
            else None
        )
        self.rf_path_encoder = (
            RFPathSetEncoderV2(
                path_dim=int(_cfg_get(path_cfg, "path_dim", DEFAULT_RAW_RF_PATH_FEATURE_DIM)),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                max_path_tokens=int(_cfg_get(path_cfg, "max_path_tokens", 64)),
                num_layers=int(_cfg_get(path_cfg, "num_layers", 2)),
                num_heads=int(_cfg_get(path_cfg, "num_heads", num_heads)),
                dropout=float(_cfg_get(path_cfg, "dropout", _cfg_get(adapter_cfg, "dropout", 0.1))),
                range_bins=int(_cfg_get(path_cfg, "range_bins", 64)),
                range_min=float(_cfg_get(path_cfg, "range_min", 0.0)),
                range_max=float(_cfg_get(path_cfg, "range_max", 20.0)),
                hist_sigma=float(_cfg_get(path_cfg, "hist_sigma", 0.15)),
                path_delay_index=int(_cfg_get(path_cfg, "path_delay_index", 0)),
                path_amplitude_index=_cfg_get(path_cfg, "path_amplitude_index", None),
                delay_to_meter=float(_cfg_get(path_cfg, "delay_to_meter", 1.0)),
            )
            if self.rf_use_paths
            else None
        )
        self.rf_global_encoder = (
            RFGlobalTokenEncoderV2(
                feature_dim=int(_cfg_get(rf_cfg, "global_dim", DEFAULT_RAW_RF_GLOBAL_FEATURE_DIM)),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
            )
            if self.rf_use_global
            else None
        )
        self.rf_scene_sync = (
            RFSceneSyncTransformer(
                embed_dim=embed_dim,
                num_layers=int(_cfg_get(sync_cfg, "num_layers", 2)),
                num_heads=int(_cfg_get(sync_cfg, "num_heads", num_heads)),
                mlp_ratio=float(_cfg_get(sync_cfg, "mlp_ratio", 4.0)),
                dropout=float(_cfg_get(sync_cfg, "dropout", 0.1)),
                append_scene_token=bool(_cfg_get(sync_cfg, "append_scene_token", True)),
            )
            if bool(_cfg_get(sync_cfg, "enabled", True))
            else None
        )

        self.rf_adapter_enabled = bool(_cfg_get(adapter_cfg, "enabled", True))
        self.rf_adapter_layer_ids = resolve_rf_adapter_layers(
            self.depth if hasattr(self, "depth") else 0,
            layers=_cfg_get(adapter_cfg, "layers", "every_4_plus_last"),
            every=int(_cfg_get(adapter_cfg, "every", 4)),
            include_last=bool(_cfg_get(adapter_cfg, "include_last", True)),
        )
        # depth is assigned later in __init__; rebuild the ids below after depth is set.
        self._rf_adapter_cfg_cache = dict(
            layers=_cfg_get(adapter_cfg, "layers", "every_4_plus_last"),
            every=int(_cfg_get(adapter_cfg, "every", 4)),
            include_last=bool(_cfg_get(adapter_cfg, "include_last", True)),
            late_adapter=bool(_cfg_get(adapter_cfg, "late_adapter", True)),
            token_type_gating=bool(_cfg_get(adapter_cfg, "token_type_gating", True)),
            use_rf_confidence_bias=bool(_cfg_get(adapter_cfg, "use_rf_confidence_bias", True)),
            fusion_scale_init=float(_cfg_get(adapter_cfg, "fusion_scale_init", 0.0)),
            gate_bias_init=float(_cfg_get(adapter_cfg, "gate_bias_init", -2.0)),
            dropout=float(_cfg_get(adapter_cfg, "dropout", 0.1)),
        )
        self.rf_adapters = nn.ModuleDict()
        self.rf_late_adapter = None

    def _finish_final_rf_adapters(self, num_heads: int, num_register_tokens: int) -> None:
        if not self.rf_final_enabled:
            return
        cfg = self._rf_adapter_cfg_cache
        self.rf_adapter_layer_ids = resolve_rf_adapter_layers(
            self.depth,
            layers=cfg["layers"],
            every=cfg["every"],
            include_last=cfg["include_last"],
        )
        if not self.rf_adapter_enabled:
            self.rf_adapter_layer_ids = []
        self.rf_adapters = nn.ModuleDict(
            {
                str(i): RFGatedAdapterV2(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    dropout=cfg["dropout"],
                    num_register_tokens=num_register_tokens,
                    token_type_gating=cfg["token_type_gating"],
                    use_rf_confidence_bias=cfg["use_rf_confidence_bias"],
                    fusion_scale_init=cfg["fusion_scale_init"],
                    gate_bias_init=cfg["gate_bias_init"],
                )
                for i in self.rf_adapter_layer_ids
            }
        )
        self.rf_late_adapter = (
            RFGatedAdapterV2(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                dropout=cfg["dropout"],
                num_register_tokens=num_register_tokens,
                token_type_gating=cfg["token_type_gating"],
                use_rf_confidence_bias=cfg["use_rf_confidence_bias"],
                fusion_scale_init=cfg["fusion_scale_init"],
                gate_bias_init=cfg["gate_bias_init"],
            )
            if self.rf_adapter_enabled and cfg["late_adapter"]
            else None
        )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        rf: torch.Tensor = None,
        rf_paths: torch.Tensor = None,
        rf_path_mask: torch.Tensor = None,
        rf_global: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            rf (torch.Tensor, optional): RF angular images with shape [B, S, C_rf, H_rf, W_rf].
            rf_paths (torch.Tensor, optional): Raw RF path features with shape [B, S, K, F].
            rf_path_mask (torch.Tensor, optional): Raw RF valid path mask with shape [B, S, K].
            rf_global (torch.Tensor, optional): Raw RF global scalar features with shape [B, S, G].

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        rf_tokens = None
        rf_pack = None
        self.latest_rf_aux = {}
        if self.enable_rf:
            if self.rf_final_enabled:
                rf_pack = self._encode_rf_final(
                    rf,
                    rf_paths=rf_paths,
                    rf_path_mask=rf_path_mask,
                    rf_global=rf_global,
                    B=B,
                    S=S,
                )
                if rf_pack is not None:
                    rf_tokens = rf_pack["tokens_flat"]
            else:
                rf_tokens = self._build_rf_tokens(B, S, rf, rf_paths, rf_path_mask, rf_global)
            if rf_tokens is not None and self.training and self.rf_drop_prob > 0:
                if torch.rand((), device=images.device) < self.rf_drop_prob:
                    rf_tokens = None
                    rf_pack = None

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, P, C = tokens.shape

        if rf_tokens is not None and not self.rf_final_enabled:
            tokens = self._fuse_rf_tokens(tokens, rf_tokens, B, S, P, C)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    if self.rf_final_enabled and rf_pack is not None and (global_idx - 1) in self.rf_adapter_layer_ids:
                        tokens = self._apply_rf_adapter(str(global_idx - 1), tokens, rf_pack, B, S, P, C)
                        if len(global_intermediates) > 0:
                            global_intermediates[-1] = tokens.view(B, S, P, C)
                    elif (
                        rf_tokens is not None
                        and self.rf_fusion_every_n_blocks > 0
                        and global_idx % self.rf_fusion_every_n_blocks == 0
                    ):
                        tokens = self._fuse_rf_tokens(tokens, rf_tokens, B, S, P, C)
                        if len(global_intermediates) > 0:
                            global_intermediates[-1] = tokens.view(B, S, P, C)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        if self.rf_final_enabled and rf_pack is not None and self.rf_late_adapter is not None and output_list:
            tokens = self._apply_rf_adapter("late", tokens, rf_pack, B, S, P, C)
            latest = output_list[-1]
            output_list[-1] = torch.cat([latest[..., :C], tokens.view(B, S, P, C)], dim=-1)

        if self.rf_final_enabled:
            self.latest_rf_aux = self._finalize_rf_aux(rf_pack)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _encode_rf_final(
        self,
        rf,
        rf_paths=None,
        rf_path_mask=None,
        rf_global=None,
        B=None,
        S=None,
    ) -> Optional[Dict[str, Any]]:
        token_list = []
        conf_list = []
        type_list = []
        aux: Dict[str, Any] = {}
        device = None

        if rf is not None and self.rf_encoder is not None:
            if rf.ndim != 5:
                raise ValueError(f"Expected RF tensor with shape [B, S, C, H, W], got {rf.shape}")
            if B is not None and rf.shape[:2] != (B, S):
                raise ValueError(f"RF tensor batch/sequence dims {rf.shape[:2]} do not match {(B, S)}")
            tokens, conf, angular_aux = self.rf_encoder(rf)
            token_list.append(tokens)
            conf_list.append(conf)
            type_list.append(torch.zeros(conf.shape, dtype=torch.long, device=conf.device))
            aux.update(angular_aux)
            device = tokens.device

        if rf_paths is not None and self.rf_path_encoder is not None:
            if rf_paths.ndim != 4:
                raise ValueError(f"Expected rf_paths with shape [B, S, K, F], got {rf_paths.shape}")
            if B is not None and rf_paths.shape[:2] != (B, S):
                raise ValueError(f"rf_paths batch/sequence dims {rf_paths.shape[:2]} do not match {(B, S)}")
            tokens, conf, path_aux = self.rf_path_encoder(rf_paths, rf_path_mask)
            token_list.append(tokens)
            conf_list.append(conf)
            type_list.append(torch.ones(conf.shape, dtype=torch.long, device=conf.device))
            aux.update(path_aux)
            device = tokens.device

        if rf_global is not None and self.rf_global_encoder is not None:
            if rf_global.ndim != 3:
                raise ValueError(f"Expected rf_global with shape [B, S, G], got {rf_global.shape}")
            if B is not None and rf_global.shape[:2] != (B, S):
                raise ValueError(f"rf_global batch/sequence dims {rf_global.shape[:2]} do not match {(B, S)}")
            tokens, conf, global_aux = self.rf_global_encoder(rf_global)
            token_list.append(tokens)
            conf_list.append(conf)
            type_list.append(torch.full(conf.shape, 2, dtype=torch.long, device=conf.device))
            aux.update(global_aux)
            device = tokens.device

        if not token_list:
            return None

        rf_tokens = torch.cat(token_list, dim=2)
        rf_conf = torch.cat(conf_list, dim=2)
        rf_type_ids = torch.cat(type_list, dim=2)
        if self.rf_scene_sync is not None:
            rf_tokens, rf_conf, sync_aux = self.rf_scene_sync(rf_tokens, rf_conf, rf_type_ids)
            aux.update(sync_aux)

        aux["rf_token_confidence"] = rf_conf
        aux["rf_token_confidence_mean"] = rf_conf.mean()
        return {
            "tokens": rf_tokens,
            "conf": rf_conf,
            "tokens_flat": rf_tokens.reshape(rf_tokens.shape[0] * rf_tokens.shape[1], rf_tokens.shape[2], rf_tokens.shape[3]),
            "conf_flat": rf_conf.reshape(rf_conf.shape[0] * rf_conf.shape[1], rf_conf.shape[2]),
            "aux": aux,
            "device": device,
        }

    def _vggt_token_type_ids(self, B: int, S: int, P: int, device: torch.device) -> torch.Tensor:
        token_type_ids = torch.full((B * S, P), 2, dtype=torch.long, device=device)
        if P > 0:
            token_type_ids[:, 0] = 0
        reg_end = min(P, self.patch_start_idx)
        if reg_end > 1:
            token_type_ids[:, 1:reg_end] = 1
        return token_type_ids

    def _apply_rf_adapter(self, adapter_key: str, tokens, rf_pack: Dict[str, Any], B: int, S: int, P: int, C: int):
        if adapter_key == "late":
            adapter = self.rf_late_adapter
        else:
            adapter = self.rf_adapters[str(adapter_key)]
        if adapter is None:
            return tokens
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        rf_tokens = rf_pack["tokens_flat"]
        rf_conf = rf_pack["conf_flat"]
        token_type_ids = self._vggt_token_type_ids(B, S, P, tokens.device)
        if self.rf_fuse_special_tokens:
            return adapter(tokens, rf_tokens, rf_conf, token_type_ids=token_type_ids)
        special_tokens = tokens[:, : self.patch_start_idx]
        patch_tokens = tokens[:, self.patch_start_idx :]
        patch_type_ids = token_type_ids[:, self.patch_start_idx :]
        patch_tokens = adapter(patch_tokens, rf_tokens, rf_conf, token_type_ids=patch_type_ids)
        return torch.cat([special_tokens, patch_tokens], dim=1)

    def _finalize_rf_aux(self, rf_pack: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        aux: Dict[str, torch.Tensor] = {}
        if rf_pack is not None:
            aux.update(rf_pack.get("aux", {}))

        gate_values: Dict[str, List[torch.Tensor]] = {
            "adapter_gate_mean": [],
            "adapter_gate_camera_mean": [],
            "adapter_gate_register_mean": [],
            "adapter_gate_patch_mean": [],
        }
        if hasattr(self, "rf_adapters"):
            for adapter in self.rf_adapters.values():
                for key in gate_values:
                    if key in adapter.last_aux:
                        gate_values[key].append(adapter.last_aux[key])
        if self.rf_late_adapter is not None:
            for key in gate_values:
                if key in self.rf_late_adapter.last_aux:
                    gate_values[key].append(self.rf_late_adapter.last_aux[key])

        ref = None
        if rf_pack is not None:
            ref = rf_pack["conf_flat"]
        for key, values in gate_values.items():
            if values:
                aux[key] = torch.stack([v.to(device=values[0].device) for v in values]).mean()
            elif ref is not None:
                aux[key] = ref.new_zeros(())
        if "rf_token_confidence_mean" not in aux and ref is not None:
            aux["rf_token_confidence_mean"] = ref.mean()
        return aux

    def _build_rf_tokens(
        self,
        B: int,
        S: int,
        rf: torch.Tensor = None,
        rf_paths: torch.Tensor = None,
        rf_path_mask: torch.Tensor = None,
        rf_global: torch.Tensor = None,
    ) -> torch.Tensor | None:
        rf_token_list = []
        if rf is not None:
            if rf.ndim != 5:
                raise ValueError(f"Expected RF tensor with shape [B, S, C, H, W], got {rf.shape}")
            if rf.shape[:2] != (B, S):
                raise ValueError(
                    f"RF tensor batch/sequence dims {rf.shape[:2]} do not match image dims {(B, S)}"
                )
            rf_token_list.append(self.rf_encoder(rf.reshape(B * S, rf.shape[2], rf.shape[3], rf.shape[4])))

        if self.use_raw_rf_paths and rf_paths is not None:
            if self.rf_path_encoder is None:
                raise RuntimeError("use_raw_rf_paths=True but rf_path_encoder is not initialized")
            if rf_paths.ndim != 4:
                raise ValueError(f"Expected rf_paths with shape [B, S, K, F], got {rf_paths.shape}")
            if rf_paths.shape[:2] != (B, S):
                raise ValueError(f"rf_paths batch/sequence dims {rf_paths.shape[:2]} do not match {(B, S)}")
            flat_mask = None
            if rf_path_mask is not None:
                if rf_path_mask.ndim != 3 or rf_path_mask.shape[:2] != (B, S):
                    raise ValueError(f"Expected rf_path_mask with shape [B, S, K], got {rf_path_mask.shape}")
                flat_mask = rf_path_mask.reshape(B * S, rf_path_mask.shape[2])
            rf_token_list.append(
                self.rf_path_encoder(
                    rf_paths.reshape(B * S, rf_paths.shape[2], rf_paths.shape[3]),
                    flat_mask,
                )
            )

        if rf_global is not None:
            if self.rf_global_encoder is None:
                raise RuntimeError("rf_global was provided, but rf_global_encoder is not initialized")
            if rf_global.ndim != 3:
                raise ValueError(f"Expected rf_global with shape [B, S, G], got {rf_global.shape}")
            if rf_global.shape[:2] != (B, S):
                raise ValueError(f"rf_global batch/sequence dims {rf_global.shape[:2]} do not match {(B, S)}")
            rf_token_list.append(self.rf_global_encoder(rf_global.reshape(B * S, rf_global.shape[2])))

        if not rf_token_list:
            return None
        return torch.cat(rf_token_list, dim=1)

    def _fuse_rf_tokens(self, tokens, rf_tokens, B, S, P, C):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if self.rf_fuse_special_tokens:
            return self.rf_fusion(tokens, rf_tokens)

        special_tokens = tokens[:, : self.patch_start_idx]
        patch_tokens = tokens[:, self.patch_start_idx :]
        patch_tokens = self.rf_fusion(patch_tokens, rf_tokens)
        return torch.cat([special_tokens, patch_tokens], dim=1)

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
