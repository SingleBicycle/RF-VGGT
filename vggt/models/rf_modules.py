from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from vggt.layers import PatchEmbed
from vggt.layers.block import Block


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

    raise ValueError(f"Unsupported RF encoder type: {encoder_type}")


class RFGatedTokenFusion(nn.Module):
    """
    Fuse RF latent tokens into RGB patch tokens with gated cross-attention.

    The fusion scale is initialized to zero so pretrained RGB-only checkpoints
    remain behaviorally stable before RF fine-tuning.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
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
        self.fusion_scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_tokens: torch.Tensor, rf_tokens: torch.Tensor) -> torch.Tensor:
        rgb_query = self.rgb_norm(rgb_tokens)
        rf_kv = self.rf_norm(rf_tokens)
        rf_context, _ = self.cross_attn(rgb_query, rf_kv, rf_kv, need_weights=False)
        rf_context = self.out_proj(rf_context)
        gate = torch.sigmoid(self.gate(torch.cat([rgb_query, rf_context], dim=-1)))
        return rgb_tokens + self.fusion_scale * gate * rf_context
