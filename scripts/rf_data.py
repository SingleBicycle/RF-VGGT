"""Shared cross-scene data/model/eval utilities for RF-VGGT experiments.

Train on the 16 scenes in RF_SCENES_train, evaluate cross-scene on the 4 unseen
scenes in RF_SCENES_val. This replaces the old N=2 held-out-FRAME protocol with a
true held-out-SCENE protocol (distributional generalization of the RF->metric-scale map).
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "training"))

TRAIN_ROOT = "/DATA/zihao/projects/rf_vggt/RF_SCENES_train"
VAL_ROOT = "/DATA/zihao/projects/rf_vggt/RF_SCENES_val"
CKPT = REPO / "ckpts" / "vggt1b_336.pt"

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


def build_cfg(overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training" / "config")):
        cfg = compose(config_name="rf_vggt_final_full", overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def make_dataset(cfg, scene_root, sampling="uniform", inside_random=False):
    base = OmegaConf.create(OmegaConf.to_container(cfg.data.train.dataset.dataset_configs[0], resolve=True))
    base.scene_roots = scene_root
    base.sampling_strategy = sampling
    common = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    common.inside_random = inside_random
    ds = instantiate(base, common_conf=common)  # RFSceneDataset
    return ds


def to_tensor_sample(raw):
    s = {}
    imgs = torch.from_numpy(np.stack(raw["images"]).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    s["images"] = imgs
    if "rf" in raw:
        s["rf"] = torch.from_numpy(np.stack(raw["rf"]).astype(np.float32)).permute(0, 3, 1, 2).contiguous()  # (S,8,90,360)
    for k, dt in [("rf_paths", np.float32), ("rf_path_mask", bool), ("rf_global", np.float32),
                  ("rf_path_range_m", np.float32), ("rf_los_range_m", np.float32),
                  ("extrinsics", np.float32), ("intrinsics", np.float32),
                  ("depths", np.float32), ("cam_points", np.float32), ("world_points", np.float32),
                  ("point_masks", bool)]:
        if k in raw:
            s[k] = torch.from_numpy(np.stack(raw[k]).astype(dt) if isinstance(raw[k], list) else np.asarray(raw[k]).astype(dt))
    return s


def get_batch(ds, scene_idx, frame_ids, device, want_angular=True):
    raw = ds.get_data(seq_index=scene_idx, ids=list(frame_ids))
    s = to_tensor_sample(raw)
    if not want_angular:
        s.pop("rf", None)
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in s.items()}
    batch["gt_metric_depth"] = batch["depths"].clone()
    ne, ncp, nwp, nd, ms = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"], cam_points=batch["cam_points"],
        world_points=batch["world_points"], depths=batch["depths"], point_masks=batch["point_masks"])
    batch["extrinsics"], batch["cam_points"], batch["world_points"], batch["depths"] = ne, ncp, nwp, nd
    batch["metric_scale"] = ms
    batch["seq_name"] = raw.get("seq_name")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_model(cfg, use_rf, device, angular_encoder=None):
    mcfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    mcfg.use_rf = bool(use_rf)
    if angular_encoder is not None:
        mcfg.rf.encoder_type = angular_encoder
    model = instantiate(mcfg).to(device)
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    return model


# Trainable groups. "frozen" = adapters + heads only (backbone frozen, original proof setup).
# "partial" = also unfreeze the last N aggregator blocks (closer to full training).
HEAD_RF = ["rf_scale_head", "rf_scene_sync", "rf_adapter", "rf_encoder", "rf_path", "rf_global",
           "camera_head", "depth_head", "point_head", "output_proj", "fusion_scale", "gate"]


def set_trainable(model, mode="frozen", unfreeze_last=4):
    for n, p in model.named_parameters():
        train = any(t in n for t in HEAD_RF)
        if mode == "partial" and ".blocks." in n:
            # unfreeze the last `unfreeze_last` transformer blocks of the aggregator (both frame & global)
            try:
                blk = int(n.split(".blocks.")[1].split(".")[0])
                # aggregator has frame_blocks/global_blocks of depth 24; unfreeze top ones
                if blk >= 24 - unfreeze_last:
                    train = True
            except (ValueError, IndexError):
                pass
        p.requires_grad_(train)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def eval_scene_depth(model, batch, use_rf, const_scale=None):
    """Per-frame metric-depth metrics on one window. Returns lists for aggregation."""
    model.eval()
    y = model(images=batch["images"], rf=batch.get("rf") if use_rf else None,
              rf_paths=batch.get("rf_paths") if use_rf else None,
              rf_path_mask=batch.get("rf_path_mask") if use_rf else None,
              rf_global=batch.get("rf_global") if use_rf else None)
    depth = y["depth"].float(); depth = depth[..., 0] if depth.dim() == 5 else depth
    unit_pred = depth[0]; gt = batch["gt_metric_depth"][0].float()
    m = (batch["point_masks"][0].bool() & (gt > 0)) if "point_masks" in batch else (gt > 0)
    if use_rf and y.get("rf_log_scale") is not None:
        scale = float(torch.exp(y["rf_log_scale"].float()).reshape(-1)[0])
        recovery = scale / float(batch["metric_scale"].mean().clamp_min(1e-6))
    else:
        scale = float(const_scale) if const_scale is not None else 1.0
        recovery = None
    metric_pred = unit_pred * scale
    out = dict(absrel_sd=[], absrel_si=[], rmse_sd=[], scale_factor=[], recovery=recovery, scale=scale)
    for s in range(unit_pred.shape[0]):
        ms = m[s]
        if ms.sum() < 100:
            continue
        p, g = metric_pred[s][ms], gt[s][ms]
        out["absrel_sd"].append(((p - g).abs() / g.clamp_min(1e-3)).mean().item())
        out["rmse_sd"].append(torch.sqrt(((p - g) ** 2).mean()).item())
        out["scale_factor"].append((torch.median(p) / torch.median(g)).item())
        r = torch.median(g) / torch.median(unit_pred[s][ms]).clamp_min(1e-6)
        out["absrel_si"].append(((unit_pred[s][ms] * r - g).abs() / g.clamp_min(1e-3)).mean().item())
    return out


def val_windows(ds, n_per=3, stride=None):
    """Deterministic eval windows per val scene: contiguous blocks of n_per frames."""
    stride = stride or n_per
    per_scene = []
    for si, s in enumerate(ds.scenes):
        vids = list(s["valid_ids"])
        wins = [vids[i:i + n_per] for i in range(0, len(vids) - n_per + 1, stride)]
        per_scene.append((si, s["seq_name"], wins))
    return per_scene
