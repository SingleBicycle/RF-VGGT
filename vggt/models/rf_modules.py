import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import PatchEmbed
from vggt.layers.block import Block

DEFAULT_RAW_RF_PATH_FEATURE_DIM = 17
DEFAULT_RAW_RF_GLOBAL_FEATURE_DIM = 7


def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if len(x) != 2:
        raise ValueError(f"Expected a 2-tuple, got {x}")
    return (int(x[0]), int(x[1]))


class RFCNNEncoder(nn.Module):
    """
    Lightweight convolutional encoder for RF angular images.

    The encoder consumes RF tensors with shape [B, C, H, W] and returns a
    small set of latent RF tokens with shape [B, K, D].
    """

    def __init__(
        self,
        in_chans: int = 3,
        hidden_dim: int = 256,
        embed_dim: int = 1024,
        latent_grid: tuple[int, int] = (2, 4),
    ) -> None:
        super().__init__()

        latent_grid = _to_2tuple(latent_grid)
        self.latent_grid = latent_grid
        self.num_latents = latent_grid[0] * latent_grid[1]

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(latent_grid)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        x = self.stem(rf)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return self.norm(x)


class RFShallowViTEncoder(nn.Module):
    """
    Shallow ViT encoder for RF angular images.

    A small patch embedder converts the RF angular image into tokens, then a
    short transformer stack models long-range angular relations before the
    tokens are pooled into a compact latent set.
    """

    def __init__(
        self,
        in_chans: int = 3,
        rf_img_size: Union[int, Sequence[int]] = (90, 360),
        patch_size: Union[int, Sequence[int]] = (10, 10),
        token_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        embed_dim: int = 1024,
        latent_grid: Union[int, Sequence[int]] = (2, 4),
        conv_stem: bool = False,
        conv_stem_out_chans: int = 64,
    ) -> None:
        super().__init__()

        rf_img_size = _to_2tuple(rf_img_size)
        patch_size = _to_2tuple(patch_size)
        latent_grid = _to_2tuple(latent_grid)

        stem_layers = []
        patch_in_chans = in_chans
        patch_img_size = rf_img_size
        if conv_stem:
            stem_layers.extend(
                [
                    nn.Conv2d(in_chans, conv_stem_out_chans, kernel_size=7, stride=2, padding=3),
                    nn.GroupNorm(8, conv_stem_out_chans),
                    nn.GELU(),
                ]
            )
            patch_in_chans = conv_stem_out_chans
            patch_img_size = (rf_img_size[0] // 2, rf_img_size[1] // 2)
        self.conv_stem = nn.Sequential(*stem_layers) if stem_layers else nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=patch_img_size,
            patch_size=patch_size,
            in_chans=patch_in_chans,
            embed_dim=token_dim,
            flatten_embedding=False,
        )
        self.grid_size = self.patch_embed.patches_resolution
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(token_dim)
        self.pool = nn.AdaptiveAvgPool2d(latent_grid)
        self.proj = nn.Linear(token_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(rf)
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.pos_embed[:, : H * W]

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return self.out_norm(x)


def build_rf_encoder(
    encoder_type: str = "cnn",
    in_chans: int = 3,
    hidden_dim: int = 256,
    embed_dim: int = 1024,
    latent_grid: Union[int, Sequence[int]] = (2, 4),
    rf_img_size: Union[int, Sequence[int]] = (90, 360),
) -> nn.Module:
    encoder_type = encoder_type.lower()

    if encoder_type == "cnn":
        return RFCNNEncoder(
            in_chans=in_chans,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            latent_grid=latent_grid,
        )

    if encoder_type == "shallow_vit":
        return RFShallowViTEncoder(
            in_chans=in_chans,
            rf_img_size=rf_img_size,
            patch_size=(10, 10),
            token_dim=384,
            depth=4,
            num_heads=6,
            embed_dim=embed_dim,
            latent_grid=latent_grid,
            conv_stem=False,
        )

    if encoder_type == "hybrid_vit":
        return RFShallowViTEncoder(
            in_chans=in_chans,
            rf_img_size=rf_img_size,
            patch_size=(5, 10),
            token_dim=384,
            depth=4,
            num_heads=6,
            embed_dim=embed_dim,
            latent_grid=latent_grid,
            conv_stem=True,
            conv_stem_out_chans=64,
        )

    if encoder_type in {"final_v2", "angular_v2"}:
        return AngularRFEncoderV2(
            in_channels=in_chans,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            angular_size=rf_img_size,
            angular_token_grid=latent_grid,
        )

    raise ValueError(f"Unsupported RF encoder type: {encoder_type}")


class RFGatedTokenFusion(nn.Module):
    """
    Fuse RF latent tokens into RGB patch tokens with gated cross-attention.

    The fusion scale is configurable and defaults to a small value so pretrained
    RGB-only checkpoints remain behaviorally stable before RF fine-tuning.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        fusion_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.rf_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))

    def forward(self, tokens: torch.Tensor, rf_tokens: torch.Tensor) -> torch.Tensor:
        if rf_tokens is None or rf_tokens.numel() == 0:
            return tokens
        rgb_query = self.rgb_norm(tokens)
        rf_kv = self.rf_norm(rf_tokens)
        rf_context, _ = self.cross_attn(rgb_query, rf_kv, rf_kv, need_weights=False)
        rf_context = self.out_proj(rf_context)
        rf_context = self.dropout(rf_context)
        gate = torch.sigmoid(self.gate(torch.cat([rgb_query, rf_context], dim=-1)))
        return tokens + self.fusion_scale * gate * rf_context

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "fusion_scale"
        if key in state_dict and state_dict[key].shape == (1,) and self.fusion_scale.shape == ():
            state_dict[key] = state_dict[key].reshape(())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class RFPathSetEncoder(nn.Module):
    """
    Encode a padded set of raw RF path features into a fixed latent token set.

    Absolute tx/rx positions are intentionally not part of the expected feature
    vector; this encoder only consumes deterministic per-path features produced
    by data.rf_utils.pack_raw_rf_npz.
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        num_latents: int = 16,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        first_linear = nn.Linear(feature_dim, hidden_dim) if feature_dim is not None else nn.LazyLinear(hidden_dim)
        self.input_proj = nn.Sequential(
            first_linear,
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 4, embed_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.latent_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, path_features: torch.Tensor, path_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if path_features.ndim != 3:
            raise ValueError(f"Expected path_features [B, K, F], got {path_features.shape}")
        B, K, _ = path_features.shape
        if K == 0:
            return path_features.new_zeros(B, self.latent_queries.shape[1], self.latent_queries.shape[2])

        path_features = torch.nan_to_num(path_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if path_mask is None:
            valid_mask = torch.ones(B, K, dtype=torch.bool, device=path_features.device)
        else:
            valid_mask = path_mask.to(device=path_features.device, dtype=torch.bool)
            if valid_mask.shape != (B, K):
                raise ValueError(f"Expected path_mask shape {(B, K)}, got {valid_mask.shape}")

        all_invalid = ~valid_mask.any(dim=1)
        safe_valid_mask = valid_mask.clone()
        if all_invalid.any():
            safe_valid_mask[all_invalid, 0] = True

        x = self.input_proj(path_features)
        x = x.masked_fill(~safe_valid_mask[..., None], 0.0)
        key_padding_mask = ~safe_valid_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        queries = self.latent_queries.expand(B, -1, -1)
        latents, _ = self.latent_attn(queries, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        latents = self.out_norm(latents)
        if all_invalid.any():
            latents = latents.masked_fill(all_invalid[:, None, None], 0.0)
        return latents


class RFGlobalTokenEncoder(nn.Module):
    """Encode per-frame raw RF global scalar features into one or more tokens."""

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        num_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        first_linear = nn.Linear(feature_dim, hidden_dim) if feature_dim is not None else nn.LazyLinear(hidden_dim)
        self.mlp = nn.Sequential(
            first_linear,
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * self.num_tokens),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        if global_features.ndim != 2:
            raise ValueError(f"Expected global_features [B, G], got {global_features.shape}")
        x = torch.nan_to_num(global_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = self.mlp(x).view(x.shape[0], self.num_tokens, self.embed_dim)
        return self.norm(x)


def circular_pad_azimuth(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Pad only the azimuth axis circularly and the elevation axis with zeros."""
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="constant", value=0.0)
    return x


class CircularAzimuthConv2d(nn.Module):
    """Conv2d wrapper that treats width/azimuth as circular without wrapping height."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, bias: bool = True) -> None:
        super().__init__()
        kernel = _to_2tuple(kernel_size)
        self.pad_h = kernel[0] // 2
        self.pad_w = kernel[1] // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(circular_pad_azimuth(x, self.pad_h, self.pad_w))


def _gn(num_channels: int) -> nn.GroupNorm:
    groups = min(8, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(eps)


def _normalize_map_per_frame_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(x.ndim - 2, x.ndim))
    min_v = x.amin(dim=dims, keepdim=True)
    max_v = x.amax(dim=dims, keepdim=True)
    return (x - min_v) / (max_v - min_v).clamp_min(eps)


class AngularRFEncoderV2(nn.Module):
    """
    Semantic angular RF encoder with dense/sparse/confidence stems, circular
    azimuth convolutions, angular positional features, and token confidence.
    """

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        angular_size: Union[int, Sequence[int]] = (90, 360),
        angular_token_grid: Union[int, Sequence[int]] = (9, 36),
        append_summary_token: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 8:
            raise ValueError("AngularRFEncoderV2 expects the finalized 8-channel RF layout.")
        self.embed_dim = int(embed_dim)
        self.angular_size = _to_2tuple(angular_size)
        self.angular_token_grid = _to_2tuple(angular_token_grid)
        self.append_summary_token = bool(append_summary_token)

        stem_dim = max(24, min(96, hidden_dim // 3 if hidden_dim >= 96 else hidden_dim))

        def stem(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                CircularAzimuthConv2d(in_ch, stem_dim, kernel_size=5, stride=2),
                nn.GELU(),
                CircularAzimuthConv2d(stem_dim, stem_dim, kernel_size=3, stride=1),
                nn.GELU(),
                _gn(stem_dim),
            )

        self.dense_stem = stem(3)
        self.sparse_stem = stem(3)
        self.confidence_stem = stem(2)

        fused_dim = stem_dim * 3
        pyramid_dim = max(hidden_dim, fused_dim)
        self.block1 = nn.Sequential(
            CircularAzimuthConv2d(fused_dim, pyramid_dim, kernel_size=3, stride=1),
            nn.GELU(),
            _gn(pyramid_dim),
        )
        self.block2 = nn.Sequential(
            CircularAzimuthConv2d(pyramid_dim, pyramid_dim, kernel_size=3, stride=2),
            nn.GELU(),
            _gn(pyramid_dim),
        )
        self.block3 = nn.Sequential(
            CircularAzimuthConv2d(pyramid_dim, pyramid_dim, kernel_size=3, stride=2),
            nn.GELU(),
            _gn(pyramid_dim),
        )
        self.token_proj = nn.Sequential(nn.Linear(pyramid_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.angular_pos_mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        conf_hidden = max(32, hidden_dim // 4)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(4, conf_hidden),
            nn.GELU(),
            nn.Linear(conf_hidden, 1),
        )

    def _angular_pos(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gh, gw = self.angular_token_grid
        rows = torch.arange(gh, device=device, dtype=dtype)
        cols = torch.arange(gw, device=device, dtype=dtype)
        phi = -math.pi / 2 + (rows + 0.5) * math.pi / float(gh)
        theta = -math.pi + (cols + 0.5) * (2.0 * math.pi) / float(gw)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")
        pos = torch.stack(
            [
                torch.sin(theta_grid),
                torch.cos(theta_grid),
                torch.sin(phi_grid),
                torch.cos(phi_grid),
            ],
            dim=-1,
        )
        return pos.reshape(gh * gw, 4)

    def forward(self, rf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rf.ndim != 5:
            raise ValueError(f"Expected rf [B, S, 8, H, W], got {rf.shape}")
        B, S, C, H, W = rf.shape
        if C < 8:
            raise ValueError(f"Expected at least 8 RF channels, got {C}")

        x = torch.nan_to_num(rf.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(B * S, C, H, W)
        dense_rf = x[:, 0:3]
        sparse_rf = x[:, 3:6]
        mask_count_rf = x[:, 6:8]

        dense_feat = self.dense_stem(dense_rf)
        sparse_feat = self.sparse_stem(sparse_rf)
        confidence_feat = self.confidence_stem(mask_count_rf)
        fused = torch.cat([dense_feat, sparse_feat, confidence_feat], dim=1)
        fused = self.block3(self.block2(self.block1(fused)))

        pooled = F.adaptive_avg_pool2d(fused, self.angular_token_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        pos = self.angular_pos_mlp(self._angular_pos(tokens.device, tokens.dtype)).unsqueeze(0)
        tokens = tokens + pos

        pooled_mask = F.adaptive_avg_pool2d(mask_count_rf[:, 0:1], self.angular_token_grid).flatten(2).transpose(1, 2)
        pooled_count = F.adaptive_avg_pool2d(mask_count_rf[:, 1:2], self.angular_token_grid).flatten(2).transpose(1, 2)
        dense_energy = torch.linalg.norm(dense_rf, dim=1, keepdim=True)
        sparse_energy = torch.linalg.norm(sparse_rf, dim=1, keepdim=True)
        pooled_dense = F.adaptive_avg_pool2d(dense_energy, self.angular_token_grid).flatten(2).transpose(1, 2)
        pooled_sparse = F.adaptive_avg_pool2d(sparse_energy, self.angular_token_grid).flatten(2).transpose(1, 2)
        conf_features = torch.cat([pooled_mask, pooled_count, pooled_dense, pooled_sparse], dim=-1)
        confidence = torch.sigmoid(self.confidence_mlp(conf_features)).squeeze(-1)

        confidence = confidence.reshape(B, S, -1)
        tokens = tokens.reshape(B, S, -1, self.embed_dim)
        summary = (tokens * confidence[..., None]).sum(dim=2) / confidence.sum(dim=2, keepdim=True).clamp_min(1e-6)
        summary_confidence = confidence.mean(dim=2, keepdim=True)
        aux = {
            "angular_confidence": confidence,
            "angular_summary": summary,
        }
        if self.append_summary_token:
            tokens = torch.cat([tokens, summary.unsqueeze(2)], dim=2)
            confidence = torch.cat([confidence, summary_confidence], dim=2)
        return tokens, confidence, aux


class RFPathSetEncoderV2(nn.Module):
    """Masked set encoder for raw RF paths with summary and range histogram tokens."""

    def __init__(
        self,
        path_dim: int = DEFAULT_RAW_RF_PATH_FEATURE_DIM,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        max_path_tokens: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        range_bins: int = 64,
        range_min: float = 0.0,
        range_max: float = 20.0,
        hist_sigma: float = 0.15,
        path_delay_index: int = 0,
        path_amplitude_index: Optional[int] = None,
        delay_to_meter: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_path_tokens = int(max_path_tokens)
        self.range_bins = int(range_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.hist_sigma = float(hist_sigma)
        self.path_delay_index = int(path_delay_index)
        self.path_amplitude_index = None if path_amplitude_index is None else int(path_amplitude_index)
        self.delay_to_meter = float(delay_to_meter)

        self.per_path_mlp = nn.Sequential(
            nn.Linear(path_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.summary_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.hist_proj = nn.Sequential(
            nn.Linear(range_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _soft_histogram(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(
            self.range_min,
            self.range_max,
            self.range_bins,
            device=values.device,
            dtype=values.dtype,
        )
        sigma = max(self.hist_sigma, 1e-4)
        kernel = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / sigma) ** 2)
        hist = (kernel * weights.unsqueeze(-1)).sum(dim=-2)
        return hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        rf_paths: torch.Tensor,
        rf_path_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rf_paths.ndim != 4:
            raise ValueError(f"Expected rf_paths [B, S, K, F], got {rf_paths.shape}")
        B, S, K, Fdim = rf_paths.shape
        K_out = self.max_path_tokens
        x = torch.nan_to_num(rf_paths.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if rf_path_mask is None:
            valid = torch.ones(B, S, K, dtype=torch.bool, device=x.device)
        else:
            valid = rf_path_mask.to(device=x.device, dtype=torch.bool)
            if valid.shape != (B, S, K):
                raise ValueError(f"Expected rf_path_mask {(B, S, K)}, got {valid.shape}")

        x = x[:, :, :K_out]
        valid = valid[:, :, :K_out]
        if x.shape[2] < K_out:
            pad_k = K_out - x.shape[2]
            x = F.pad(x, (0, 0, 0, pad_k))
            valid = F.pad(valid, (0, pad_k), value=False)

        flat_x = x.reshape(B * S, K_out, Fdim)
        flat_valid = valid.reshape(B * S, K_out)
        any_valid = flat_valid.any(dim=1)
        safe_valid = flat_valid.clone()
        if (~any_valid).any():
            safe_valid[~any_valid, 0] = True

        path_tokens = self.per_path_mlp(flat_x)
        path_tokens = path_tokens.masked_fill(~safe_valid[..., None], 0.0)
        path_tokens = self.set_encoder(path_tokens, src_key_padding_mask=~safe_valid)
        path_tokens = path_tokens.masked_fill(~flat_valid[..., None], 0.0)

        summary = _masked_mean(path_tokens, flat_valid, dim=1)
        summary = self.summary_proj(summary)

        full_paths = torch.nan_to_num(rf_paths.float(), nan=0.0, posinf=0.0, neginf=0.0)
        full_valid = rf_path_mask.to(device=x.device, dtype=torch.bool) if rf_path_mask is not None else torch.ones(B, S, K, device=x.device, dtype=torch.bool)
        delay = full_paths[..., self.path_delay_index].clamp_min(0.0) * self.delay_to_meter
        weight = full_valid.to(delay.dtype)
        if self.path_amplitude_index is not None and self.path_amplitude_index < full_paths.shape[-1]:
            amp = full_paths[..., self.path_amplitude_index].abs()
            amp = amp / amp.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            weight = weight * amp
        hist = self._soft_histogram(delay.reshape(B * S, K), weight.reshape(B * S, K))
        hist_token = self.hist_proj(hist)

        tokens = torch.cat([path_tokens, summary[:, None], hist_token[:, None]], dim=1)
        summary_conf = any_valid.to(dtype=tokens.dtype).unsqueeze(-1)
        confidence = torch.cat([flat_valid.to(dtype=tokens.dtype), summary_conf, summary_conf], dim=1)
        tokens = tokens.reshape(B, S, K_out + 2, self.embed_dim)
        confidence = confidence.reshape(B, S, K_out + 2)
        aux = {
            "path_confidence": confidence,
            "path_range_histogram": hist.reshape(B, S, self.range_bins),
        }
        return tokens, confidence, aux


class RFGlobalTokenEncoderV2(nn.Module):
    """Per-frame global RF scalar token encoder."""

    def __init__(
        self,
        feature_dim: int = DEFAULT_RAW_RF_GLOBAL_FEATURE_DIM,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    def forward(self, rf_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rf_global.ndim != 3:
            raise ValueError(f"Expected rf_global [B, S, G], got {rf_global.shape}")
        B, S, G = rf_global.shape
        x = torch.nan_to_num(rf_global.float(), nan=0.0, posinf=0.0, neginf=0.0)
        tokens = self.mlp(x.reshape(B * S, G)).reshape(B, S, 1, -1) + self.type_embedding
        confidence = torch.ones(B, S, 1, device=rf_global.device, dtype=tokens.dtype)
        return tokens, confidence, {"global_confidence": confidence}


class RFSceneSyncTransformer(nn.Module):
    """Synchronize RF tokens across frames before fusion into VGGT tokens."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 128,
        num_token_types: int = 8,
        append_scene_token: bool = True,
        use_confidence_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_frames = int(max_frames)
        self.append_scene_token = bool(append_scene_token)
        self.use_confidence_embedding = bool(use_confidence_embedding)
        self.frame_embedding = nn.Embedding(max_frames, embed_dim)
        self.type_embedding = nn.Embedding(num_token_types, embed_dim)
        self.confidence_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.scene_rf_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.scene_rf_token, std=0.02)

    def forward(
        self,
        rf_tokens: torch.Tensor,
        rf_confidence: torch.Tensor,
        rf_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rf_tokens.ndim != 4:
            raise ValueError(f"Expected rf_tokens [B, S, K, D], got {rf_tokens.shape}")
        B, S, K, D = rf_tokens.shape
        if rf_confidence.shape != (B, S, K):
            raise ValueError(f"Expected rf_confidence {(B, S, K)}, got {rf_confidence.shape}")
        if rf_type_ids is None:
            rf_type_ids = torch.zeros(B, S, K, dtype=torch.long, device=rf_tokens.device)
        else:
            rf_type_ids = rf_type_ids.to(device=rf_tokens.device, dtype=torch.long).clamp_min(0)

        frame_ids = torch.arange(S, device=rf_tokens.device).clamp(max=self.max_frames - 1)
        x = rf_tokens + self.frame_embedding(frame_ids).view(1, S, 1, D)
        x = x + self.type_embedding(rf_type_ids.clamp(max=self.type_embedding.num_embeddings - 1))
        if self.use_confidence_embedding:
            x = x + self.confidence_mlp(rf_confidence.to(dtype=x.dtype).unsqueeze(-1).clamp(0.0, 1.0))

        x = x.reshape(B, S * K, D)
        conf = rf_confidence.reshape(B, S * K)
        scene = self.scene_rf_token.expand(B, -1, -1).to(dtype=x.dtype)
        x = torch.cat([scene, x], dim=1)
        key_padding_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), conf <= 0], dim=1)
        all_invalid = key_padding_mask[:, 1:].all(dim=1)
        if all_invalid.any():
            key_padding_mask[all_invalid, 1] = False
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.out_norm(x)
        scene_out = x[:, 0]
        token_out = x[:, 1:].reshape(B, S, K, D)
        if self.append_scene_token:
            scene_broadcast = scene_out[:, None, None, :].expand(B, S, 1, D)
            token_out = torch.cat([token_out, scene_broadcast], dim=2)
            scene_conf = (rf_confidence.sum(dim=(1, 2)) > 0).to(dtype=rf_confidence.dtype)[:, None, None].expand(B, S, 1)
            rf_confidence = torch.cat([rf_confidence, scene_conf], dim=2)
        return token_out, rf_confidence, {"scene_rf_token": scene_out}


class RFGatedAdapterV2(nn.Module):
    """Token-type-aware gated cross-attention adapter from RF tokens into VGGT tokens."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_register_tokens: int = 4,
        token_type_gating: bool = True,
        use_rf_confidence_bias: bool = True,
        fusion_scale_init: float = 1e-3,
        gate_bias_init: float = -2.0,
        min_effective_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_register_tokens = int(num_register_tokens)
        self.token_type_gating = bool(token_type_gating)
        self.use_rf_confidence_bias = bool(use_rf_confidence_bias)
        self.min_effective_scale = float(min_effective_scale)

        self.vggt_norm = nn.LayerNorm(embed_dim)
        self.rf_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.token_type_embedding = nn.Embedding(3, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))
        self.last_aux: Dict[str, torch.Tensor] = {}

        final_gate = self.gate[-1]
        nn.init.constant_(final_gate.bias, float(gate_bias_init))
        nn.init.normal_(self.output_proj.weight, std=1e-5)
        nn.init.zeros_(self.output_proj.bias)

    def _infer_token_types(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        token_types = torch.full((B, N), 2, dtype=torch.long, device=device)
        if N > 0:
            token_types[:, 0] = 0
        reg_end = min(N, 1 + self.num_register_tokens)
        if reg_end > 1:
            token_types[:, 1:reg_end] = 1
        return token_types

    def forward(
        self,
        vggt_tokens: torch.Tensor,
        rf_tokens: Optional[torch.Tensor],
        rf_conf: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rf_tokens is None or rf_tokens.numel() == 0:
            self.last_aux = {}
            return vggt_tokens
        B, N, D = vggt_tokens.shape
        K = rf_tokens.shape[1]
        q = self.vggt_norm(vggt_tokens)
        kv = self.rf_norm(rf_tokens)
        if rf_conf is None:
            rf_conf = torch.ones(B, K, dtype=q.dtype, device=q.device)
        else:
            rf_conf = rf_conf.to(device=q.device, dtype=q.dtype)

        key_padding_mask = rf_conf <= 0
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False

        attn_mask = None
        key_padding_mask_arg = key_padding_mask
        if self.use_rf_confidence_bias:
            safe_conf = rf_conf.clamp_min(1e-4)
            safe_conf = safe_conf.masked_fill(rf_conf <= 0, 1e-4)
            bias = torch.log(safe_conf).unsqueeze(1).expand(B, N, K)
            bias = bias.masked_fill((rf_conf <= 0).unsqueeze(1), -1e4)
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0)
            key_padding_mask_arg = None

        context, _ = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask_arg,
            attn_mask=attn_mask,
            need_weights=False,
        )
        if token_type_ids is None:
            token_type_ids = self._infer_token_types(B, N, q.device)
        else:
            token_type_ids = token_type_ids.to(device=q.device, dtype=torch.long).clamp(0, 2)
        type_emb = self.token_type_embedding(token_type_ids) if self.token_type_gating else torch.zeros_like(q)
        gate = torch.sigmoid(self.gate(torch.cat([q, context, type_emb], dim=-1)))
        projected = self.dropout(self.output_proj(context))
        scale = self.fusion_scale + self.min_effective_scale
        out = vggt_tokens + scale * gate * projected

        with torch.no_grad():
            camera_mask = token_type_ids == 0
            register_mask = token_type_ids == 1
            patch_mask = token_type_ids == 2

            def mean_or_zero(mask: torch.Tensor) -> torch.Tensor:
                if mask.any():
                    return gate[mask].mean()
                return gate.new_zeros(())

            self.last_aux = {
                "adapter_gate_mean": gate.mean().detach(),
                "adapter_gate_camera_mean": mean_or_zero(camera_mask).detach(),
                "adapter_gate_register_mean": mean_or_zero(register_mask).detach(),
                "adapter_gate_patch_mean": mean_or_zero(patch_mask).detach(),
            }
        return out


class RFScaleHead(nn.Module):
    """Predict per-sample log metric scale from RF measurements ONLY (no image features).

    Rationale: RF per-path delays encode absolute metric range (range = c * delay), so the
    absolute scene scale is recoverable from RF alone. Keeping this head blind to images is
    deliberate: it makes the metric-scale prediction attributable to RF, so an RGB-only
    (RF-off) model has no mechanism to recover per-sample metric scale.

    Input:
        rf_paths      [B, S, K, F]  packed per-path features (feature 0 encodes delay)
        rf_path_mask  [B, S, K]     bool/float validity of each path
        rf_global     [B, S, G]     per-frame global RF scalars
    Output:
        log_scale     [B]           predicted log of the metric scale factor
    """

    def __init__(self, path_dim: int = 17, global_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.path_mlp = nn.Sequential(
            nn.Linear(path_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # Initialized so the head starts near log_scale ~= 0 (scale ~= 1) and learns the offset.
        self.log_scale_bias = nn.Parameter(torch.zeros(1))

    def forward(self, rf_paths, rf_path_mask=None, rf_global=None):
        if rf_paths.dim() == 3:  # [S, K, F] -> [1, S, K, F]
            rf_paths = rf_paths.unsqueeze(0)
        B, S, K, _ = rf_paths.shape
        if rf_path_mask is None:
            m = rf_paths.new_ones(B, S, K, 1)
        else:
            if rf_path_mask.dim() == 2:
                rf_path_mask = rf_path_mask.unsqueeze(0)
            m = rf_path_mask.to(rf_paths.dtype).unsqueeze(-1)
        pf = self.path_mlp(rf_paths.float())                 # [B,S,K,H]
        pf = (pf * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)  # masked mean over paths -> [B,S,H]
        if rf_global is None:
            gf = pf.new_zeros(B, S, pf.shape[-1])
        else:
            if rf_global.dim() == 2:
                rf_global = rf_global.unsqueeze(0)
            gf = self.global_mlp(rf_global.float())          # [B,S,H]
        feat = torch.cat([pf, gf], dim=-1)                   # [B,S,2H]
        log_s_per_frame = self.head(feat).squeeze(-1)        # [B,S]
        return log_s_per_frame.mean(dim=1) + self.log_scale_bias  # [B]
