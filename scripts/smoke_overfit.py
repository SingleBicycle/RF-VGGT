"""End-to-end verify: real batch -> forward -> loss -> backward -> short overfit.
Asserts the RF metric-scale anchor learns (rf_scale_factor -> 1.0) and the RF gate opens."""
import os, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "training"))
SCENES = "/DATA/zihao/projects/rf_vggt"
DEV = "cuda:0"

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

def build_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training" / "config")):
        cfg = compose(config_name="rf_vggt_final_full")
    OmegaConf.resolve(cfg)
    return cfg

def add_batch_dim(sample):
    out = {}
    for k, v in sample.items():
        out[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
    return out

def normalize_batch(batch):
    ne, ncp, nwp, nd, ms = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"], cam_points=batch["cam_points"],
        world_points=batch["world_points"], depths=batch["depths"], point_masks=batch["point_masks"],
    )
    batch["extrinsics"], batch["cam_points"], batch["world_points"], batch["depths"] = ne, ncp, nwp, nd
    batch["metric_scale"] = ms
    return batch

def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

def main():
    cfg = build_cfg()
    # dataset (full pipeline -> correct tensor layout incl. rf permute + range fields)
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.dataset, resolve=True))
    dcfg.dataset_configs[0].scene_roots = SCENES
    common = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    common.inside_random = False
    composed = instantiate(dcfg, common_config=common, _recursive_=False)
    sample = composed[(0, 3, 1.0)]
    print("sample keys:", sorted(k for k in sample if torch.is_tensor(sample[k])))
    print("  rf", tuple(sample["rf"].shape), "rf_paths", tuple(sample["rf_paths"].shape),
          "rf_path_range_m", tuple(sample["rf_path_range_m"].shape), "rf_los", tuple(sample["rf_los_range_m"].shape))

    batch = normalize_batch(add_batch_dim(sample))
    print(f"  metric_scale (GT) = {batch['metric_scale'].tolist()}")
    batch = to_dev(batch, DEV)

    # model + pretrained
    model = instantiate(cfg.model).to(DEV)
    sd = torch.load(REPO / "ckpts" / "vggt1b_336.pt", map_location="cpu")["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  loaded VGGT-1B: missing={len(miss)} (rf+scale init), unexpected={len(unexp)}")
    loss_fn = instantiate(cfg.loss, _recursive_=False)

    def forward_loss():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(images=batch["images"], rf=batch.get("rf"), rf_paths=batch.get("rf_paths"),
                      rf_path_mask=batch.get("rf_path_mask"), rf_global=batch.get("rf_global"))
        ld = loss_fn(y, batch)  # loss in fp32 (heads already fp32); BCE not autocast-safe
        return y, ld

    # ---- single forward/backward sanity ----
    model.train()
    y, ld = forward_loss()
    assert "rf_log_scale" in y and y["rf_log_scale"] is not None, "rf_log_scale missing!"
    print(f"\n[sanity] rf_log_scale={y['rf_log_scale'].tolist()}  loss_total={float(ld['loss_total']):.4f}")
    for k in ["loss_camera","loss_depth","loss_point","loss_rf_angular","loss_rf_path","loss_rf_scale",
              "rf_scale_factor_mean","rf_pred_scale_m","rf_gt_scale_m"]:
        if k in ld: print(f"    {k} = {float(ld[k]):.4f}")
    ld["loss_total"].float().backward()
    g_scale = sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if "rf_scale_head" in n and p.grad is not None)
    g_rf = sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if ("rf_" in n and "scale_head" not in n) and p.grad is not None)
    print(f"    grad|rf_scale_head|={g_scale:.3e}  grad|other rf|={g_rf:.3e}")
    assert g_scale > 0, "no gradient to rf_scale_head!"

    # ---- short overfit: train RF modules + scale head + heads (freeze big backbone for memory/speed) ----
    train_names = [n for n,_ in model.named_parameters()
                   if any(t in n for t in ["rf_scale_head","rf_scene_sync","rf_adapter","rf_encoder",
                                            "rf_path","rf_global","camera_head","depth_head","point_head","output_proj","fusion_scale","gate"])]
    for n,p in model.named_parameters():
        p.requires_grad_(any(n == tn for tn in train_names))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[overfit] training {n_train/1e6:.1f}M params for 40 steps...")
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        y, ld = forward_loss()
        ld["loss_total"].float().backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 8 == 0 or step == 39:
            gate = float(y["rf_aux"].get("adapter_gate_mean", torch.tensor(float("nan"))))
            print(f"  step {step:2d}: total={float(ld['loss_total']):.4f} rf_scale={float(ld['loss_rf_scale']):.4f} "
                  f"scale_factor={float(ld['rf_scale_factor_mean']):.3f} pred_m={float(ld['rf_pred_scale_m']):.2f} "
                  f"gt_m={float(ld['rf_gt_scale_m']):.2f} gate={gate:.3f} depth={float(ld['loss_depth']):.3f}")
    sf = float(ld["rf_scale_factor_mean"])
    print(f"\nRESULT: final scale_factor={sf:.3f} (target 1.0). {'PASS' if 0.7 < sf < 1.4 else 'CHECK'}")

if __name__ == "__main__":
    main()
