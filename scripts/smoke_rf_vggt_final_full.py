#!/usr/bin/env python
import argparse
import math
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.models.vggt import VGGT
from loss import MultitaskLoss


def _override_for_smoke(cfg):
    cfg["img_size"] = 28
    cfg["patch_size"] = 14
    cfg["embed_dim"] = 64
    model = cfg["model"]
    model["img_size"] = 28
    model["patch_size"] = 14
    model["embed_dim"] = 64
    model["enable_camera"] = True
    model["enable_depth"] = True
    model["enable_point"] = True
    model["enable_track"] = False
    model["aggregator_kwargs"] = {
        "depth": 4,
        "num_heads": 4,
        "patch_embed": "conv",
        "num_register_tokens": 2,
        "aa_order": ["frame", "global"],
        "aa_block_size": 1,
    }
    model["rf"]["embed_dim"] = 64
    model["rf"]["angular_token_grid"] = [3, 6]
    model["rf"]["path_encoder"]["max_path_tokens"] = 8
    model["rf"]["path_encoder"]["num_heads"] = 4
    model["rf"]["scene_sync"]["num_heads"] = 4
    model["rf"]["scene_sync"]["num_layers"] = 1
    model["rf"]["adapter"]["num_heads"] = 4
    model["rf"]["adapter"]["every"] = 2
    cfg["loss"]["rf_angular"]["max_points_per_view"] = 256
    cfg["loss"]["rf_path"]["max_points_per_view"] = 256
    return cfg


def _make_batch(device):
    B, S, H, W, K = 1, 2, 28, 28, 8
    images = torch.rand(B, S, 3, H, W, device=device)
    rf = torch.rand(B, S, 8, 90, 360, device=device)
    rf[:, :, 6] = (rf[:, :, 6] > 0.35).float()
    rf[:, :, 7] = torch.log1p(torch.rand(B, S, 90, 360, device=device) * 8.0)
    rf_paths = torch.rand(B, S, K, 17, device=device)
    rf_paths[..., 0] = torch.linspace(0.5, 8.0, K, device=device).view(1, 1, K)
    rf_path_mask = torch.ones(B, S, K, dtype=torch.bool, device=device)
    rf_global = torch.rand(B, S, 7, device=device)

    extrinsics = torch.eye(4, device=device)[:3].view(1, 1, 3, 4).repeat(B, S, 1, 1)
    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, S, 1, 1)
    intrinsics[:, :, 0, 0] = 20.0
    intrinsics[:, :, 1, 1] = 20.0
    intrinsics[:, :, 0, 2] = (W - 1) / 2.0
    intrinsics[:, :, 1, 2] = (H - 1) / 2.0

    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    depth = torch.ones(B, S, H, W, device=device) * 2.0
    x = (xx.view(1, 1, H, W) - intrinsics[:, :, 0, 2, None, None]) / intrinsics[:, :, 0, 0, None, None] * depth
    y = (yy.view(1, 1, H, W) - intrinsics[:, :, 1, 2, None, None]) / intrinsics[:, :, 1, 1, None, None] * depth
    world_points = torch.stack([x, y, depth], dim=-1)
    point_masks = torch.ones(B, S, H, W, dtype=torch.bool, device=device)

    return {
        "images": images,
        "rf": rf,
        "rf_paths": rf_paths,
        "rf_path_mask": rf_path_mask,
        "rf_global": rf_global,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depths": depth,
        "world_points": world_points,
        "point_masks": point_masks,
        "depth_masks": point_masks,
    }


def _assert_finite(name, value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"{name} contains NaN/Inf"
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(f"{name}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_finite(f"{name}.{idx}", item)


def _has_nonzero_grad(module):
    for param in module.parameters():
        if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config/rf_vggt_final_full.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r") as f:
        cfg = _override_for_smoke(yaml.safe_load(f))

    device = torch.device(args.device)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("_target_", None)
    model = VGGT(**model_cfg).to(device)
    dim_in = 2 * int(model_cfg["embed_dim"])
    model.camera_head = CameraHead(dim_in=dim_in, num_heads=4).to(device)
    model.depth_head = DPTHead(
        dim_in=dim_in,
        output_dim=2,
        activation="exp",
        conf_activation="expp1",
        features=32,
        out_channels=[32, 64, 64, 64],
        intermediate_layer_idx=[0, 1, 2, 3],
    ).to(device)
    model.point_head = DPTHead(
        dim_in=dim_in,
        output_dim=4,
        activation="inv_log",
        conf_activation="expp1",
        features=32,
        out_channels=[32, 64, 64, 64],
        intermediate_layer_idx=[0, 1, 2, 3],
    ).to(device)
    model.train()
    loss_cfg = dict(cfg["loss"])
    loss_cfg.pop("_target_", None)
    loss_fn = MultitaskLoss(**loss_cfg).to(device)
    batch = _make_batch(device)

    outputs = model(
        images=batch["images"],
        rf=batch["rf"],
        rf_paths=batch["rf_paths"],
        rf_path_mask=batch["rf_path_mask"],
        rf_global=batch["rf_global"],
    )
    assert "pose_enc" in outputs and "depth" in outputs and "world_points" in outputs and "rf_aux" in outputs
    assert outputs["rf_aux"], "rf_aux is empty for full RF forward"

    optional_outputs = model(images=batch["images"], rf=batch["rf"], rf_paths=None, rf_path_mask=None, rf_global=None)
    assert "pose_enc" in optional_outputs
    rgb_outputs = model(images=batch["images"], rf=None, rf_paths=None, rf_path_mask=None, rf_global=None)
    assert "pose_enc" in rgb_outputs

    loss_dict = loss_fn(outputs, batch)
    required_loss_keys = ["loss_total", "loss_camera", "loss_depth", "loss_point", "loss_rf_angular", "loss_rf_path"]
    for key in required_loss_keys:
        assert key in loss_dict, f"missing {key}"
    for key, value in outputs.items():
        _assert_finite(f"output.{key}", value)
    for key, value in loss_dict.items():
        _assert_finite(f"loss.{key}", value)

    total_loss = loss_dict["loss_total"]
    model.zero_grad(set_to_none=True)
    total_loss.backward()

    agg = model.aggregator
    grad_checks = {
        "rf_angular_encoder": agg.rf_encoder,
        "rf_path_encoder": agg.rf_path_encoder,
        "rf_scene_sync": agg.rf_scene_sync,
        "rf_adapters": agg.rf_adapters,
        "camera_head": model.camera_head,
        "depth_head": model.depth_head,
        "point_head": model.point_head,
    }
    for name, module in grad_checks.items():
        assert module is not None and _has_nonzero_grad(module), f"missing nonzero gradient for {name}"

    printable = {}
    for key, value in {**loss_dict, **outputs["rf_aux"]}.items():
        if torch.is_tensor(value) and value.numel() == 1:
            printable[key] = float(value.detach().cpu())
    for key in sorted(printable):
        value = printable[key]
        assert math.isfinite(value), f"{key} is not finite"
        print(f"{key}: {value:.6f}")

    print("smoke_rf_vggt_final_full: ok")


if __name__ == "__main__":
    main()
