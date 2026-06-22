"""Controlled A/B: RF-on vs RF-off, identical init/data/schedule, only RF differs.
Held-out FRAME split (contiguous block) per scene. Reports metric-scale + scale-dependent vs
scale-invariant depth error, and dumps arrays for the figures. Single GPU, fine-tunes
RF modules + scale head + DPT/camera heads (backbone frozen) for a short proof run."""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "training"))
SCENES_PARENT = "/DATA/zihao/projects/rf_vggt"
SCENE_NAMES = ["AI53_001_Blender", "AI53_002_Blender"]
OUT = REPO / "results" / "ab"
OUT.mkdir(parents=True, exist_ok=True)

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

TRAIN_FRAMES = list(range(0, 80))     # contiguous-block split: train 0..79
VAL_FRAMES = list(range(80, 100))     #                          val   80..99 (spatially extrapolated)


def build_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training" / "config")):
        cfg = compose(config_name="rf_vggt_final_full")
    OmegaConf.resolve(cfg)
    return cfg


def make_dataset(cfg):
    base = OmegaConf.create(OmegaConf.to_container(cfg.data.train.dataset.dataset_configs[0], resolve=True))
    base.scene_roots = SCENES_PARENT
    common = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    common.inside_random = False
    ds = instantiate(base, common_conf=common)  # RFSceneDataset
    return ds


def to_tensor_sample(raw):
    """Replicate ComposedDataset tensor conversion for an RFSceneDataset.get_data() output."""
    s = {}
    imgs = torch.from_numpy(np.stack(raw["images"]).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    s["images"] = imgs
    rf = torch.from_numpy(np.stack(raw["rf"]).astype(np.float32)).permute(0, 3, 1, 2).contiguous()  # (S,8,90,360)
    s["rf"] = rf
    for k, dt in [("rf_paths", np.float32), ("rf_path_mask", bool), ("rf_global", np.float32),
                  ("rf_path_range_m", np.float32), ("rf_los_range_m", np.float32),
                  ("extrinsics", np.float32), ("intrinsics", np.float32),
                  ("depths", np.float32), ("cam_points", np.float32), ("world_points", np.float32),
                  ("point_masks", bool)]:
        if k in raw:
            s[k] = torch.from_numpy(np.stack(raw[k]).astype(dt) if isinstance(raw[k], list) else np.asarray(raw[k]).astype(dt))
    return s


def get_batch(ds, scene_idx, frame_ids, device):
    raw = ds.get_data(seq_index=scene_idx, ids=list(frame_ids))
    s = to_tensor_sample(raw)
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in s.items()}
    batch["gt_metric_depth"] = batch["depths"].clone()  # metric GT depth (before unit-normalization)
    ne, ncp, nwp, nd, ms = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"], cam_points=batch["cam_points"],
        world_points=batch["world_points"], depths=batch["depths"], point_masks=batch["point_masks"])
    batch["extrinsics"], batch["cam_points"], batch["world_points"], batch["depths"] = ne, ncp, nwp, nd
    batch["metric_scale"] = ms
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_model(cfg, use_rf, device):
    mcfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    mcfg.use_rf = bool(use_rf)
    model = instantiate(mcfg).to(device)
    sd = torch.load(REPO / "ckpts" / "vggt1b_336.pt", map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    return model


TRAINABLE = ["rf_scale_head", "rf_scene_sync", "rf_adapter", "rf_encoder", "rf_path", "rf_global",
             "camera_head", "depth_head", "point_head", "output_proj", "fusion_scale", "gate"]


def set_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad_(any(t in n for t in TRAINABLE))
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def eval_scene(model, ds, cfg, use_rf, scene_idx, frames, device, const_scale=None, n_per=3):
    """Eval metric depth on held-out frames. Returns metrics + arrays for viz."""
    model.eval()
    abs_rel_sd, abs_rel_si, rmse_sd, scale_factors = [], [], [], []
    scale_recovery = []  # exp(predicted log_scale) / true metric_scale on held-out frames (RF-on)
    sample_depth = None
    pred_ranges_all, rf_ranges_all = [], []
    for i in range(0, len(frames), n_per):
        fids = frames[i:i + n_per]
        if len(fids) < 1:
            continue
        b = get_batch(ds, scene_idx, fids, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(images=b["images"], rf=b.get("rf") if use_rf else None,
                      rf_paths=b.get("rf_paths") if use_rf else None,
                      rf_path_mask=b.get("rf_path_mask") if use_rf else None,
                      rf_global=b.get("rf_global") if use_rf else None)
        depth = y["depth"].float()  # [1,S,H,W,1] unit-scale
        if depth.dim() == 5:
            depth = depth[..., 0]
        unit_pred = depth[0]  # [S,H,W]
        gt = b["gt_metric_depth"][0].float()  # [S,H,W] metric
        if "point_masks" in b:
            m = b["point_masks"][0].bool() & (gt > 0)
        else:
            m = gt > 0
        # metric scale per frame
        if use_rf and "rf_log_scale" in y and y["rf_log_scale"] is not None:
            scale = torch.exp(y["rf_log_scale"].float()).reshape(1).item()
            # direct scale-head recovery on held-out frames: predicted avg_scale vs true avg_scale
            scale_recovery.append(scale / float(b["metric_scale"].mean().clamp_min(1e-6)))
        else:
            scale = float(const_scale) if const_scale is not None else 1.0
        metric_pred = unit_pred * scale
        for s in range(unit_pred.shape[0]):
            ms_ = m[s]
            if ms_.sum() < 100:
                continue
            p = metric_pred[s][ms_]; g = gt[s][ms_]
            # scale-dependent
            abs_rel_sd.append(((p - g).abs() / g.clamp_min(1e-3)).mean().item())
            rmse_sd.append(torch.sqrt(((p - g) ** 2).mean()).item())
            scale_factors.append((torch.median(p) / torch.median(g)).item())
            # scale-invariant: align pred to gt by median ratio (best per-frame scale)
            r = (torch.median(g) / torch.median(unit_pred[s][ms_]).clamp_min(1e-6))
            p_si = unit_pred[s][ms_] * r
            abs_rel_si.append(((p_si - g).abs() / g.clamp_min(1e-3)).mean().item())
            if sample_depth is None:
                sample_depth = dict(pred=metric_pred[s].cpu().numpy(), gt=gt[s].cpu().numpy(),
                                    mask=ms_.cpu().numpy(), unit=unit_pred[s].cpu().numpy(),
                                    img=b["images"][0, s].cpu().numpy())
            # range histograms (metric pred point ranges vs RF ranges*c0)
            pr = metric_pred[s][ms_]
            pred_ranges_all.append(pr.flatten().cpu().numpy())
        if use_rf and "rf_path_range_m" in b:
            rr = b["rf_path_range_m"][0][b["rf_path_mask"][0].bool()] * cfg.loss.rf_path.range_to_depth_scale
            rf_ranges_all.append(rr.flatten().cpu().numpy())
    res = dict(
        abs_rel_scale_dep=float(np.mean(abs_rel_sd)) if abs_rel_sd else None,
        abs_rel_scale_inv=float(np.mean(abs_rel_si)) if abs_rel_si else None,
        rmse_scale_dep_m=float(np.mean(rmse_sd)) if rmse_sd else None,
        scale_factor_med=float(np.median(scale_factors)) if scale_factors else None,
        scale_factor_abs_err=float(np.median(np.abs(np.array(scale_factors) - 1.0))) if scale_factors else None,
        scale_recovery_med=float(np.median(scale_recovery)) if scale_recovery else None,
        scale_recovery_abs_err=float(np.median(np.abs(np.array(scale_recovery) - 1.0))) if scale_recovery else None,
    )
    arrays = dict(sample_depth=sample_depth,
                  pred_ranges=np.concatenate(pred_ranges_all) if pred_ranges_all else np.array([]),
                  rf_ranges=np.concatenate(rf_ranges_all) if rf_ranges_all else np.array([]))
    return res, arrays


def train(method, steps, n_frames, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    use_rf = (method == "rf_on")
    cfg = build_cfg()
    ds = make_dataset(cfg)
    # restrict to train frames for sampling integrity (val frames never seen in training)
    train_scene_frames = []
    for s in ds.scenes:
        tv = [f for f in s["valid_ids"] if f in set(TRAIN_FRAMES)]
        train_scene_frames.append(tv)
    model = build_model(cfg, use_rf, device)
    params = set_trainable(model)
    # Scale head is tiny and must converge fast -> its own higher LR.
    scale_params = [p for n, p in model.named_parameters() if p.requires_grad and "rf_scale_head" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "rf_scale_head" not in n]
    groups = [{"params": other_params, "lr": 2e-4}]
    if scale_params:
        groups.append({"params": scale_params, "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    loss_fn = instantiate(cfg.loss, _recursive_=False)
    rng = np.random.default_rng(seed)
    history = []
    metric_scales_seen = []
    print(f"\n=== TRAIN {method}: {sum(p.numel() for p in params)/1e6:.0f}M params, {steps} steps ===")
    model.train()
    for step in range(steps):
        scene_idx = int(rng.integers(0, len(ds.scenes)))
        pool = train_scene_frames[scene_idx]
        fids = sorted(rng.choice(pool, size=min(n_frames, len(pool)), replace=False).tolist())
        b = get_batch(ds, scene_idx, fids, device)
        metric_scales_seen.append(float(b["metric_scale"].mean()))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(images=b["images"], rf=b.get("rf") if use_rf else None,
                      rf_paths=b.get("rf_paths") if use_rf else None,
                      rf_path_mask=b.get("rf_path_mask") if use_rf else None,
                      rf_global=b.get("rf_global") if use_rf else None)
        ld = loss_fn(y, b)
        opt.zero_grad(set_to_none=True)
        ld["loss_total"].float().backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            gate = float(y["rf_aux"].get("adapter_gate_mean", torch.tensor(float("nan")))) if use_rf else float("nan")
            sf = float(ld.get("rf_scale_factor_mean", torch.tensor(float("nan"))))
            history.append(dict(step=step, loss_total=float(ld["loss_total"]),
                                loss_depth=float(ld["loss_depth"]), loss_rf_scale=float(ld.get("loss_rf_scale", 0.0)),
                                scale_factor=sf, gate=gate))
            print(f"  {method} step {step:3d}: total={float(ld['loss_total']):.3f} "
                  f"rf_scale={float(ld.get('loss_rf_scale',0.0)):.3f} scale_factor={sf:.3f} gate={gate:.3f}")
    const_scale = float(np.mean(metric_scales_seen))  # best blind constant for RF-off
    return model, ds, cfg, history, const_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    summary = {}
    for method in ["rf_on", "rf_off"]:
        model, ds, cfg, history, const_scale = train(method, args.steps, args.frames, args.device)
        evals, arrays = {}, {}
        for si, sname in enumerate(SCENE_NAMES):
            if si >= len(ds.scenes):
                continue
            res, arr = eval_scene(model, ds, cfg, method == "rf_on", si, VAL_FRAMES, args.device, const_scale=const_scale)
            evals[sname] = res
            arrays[sname] = arr
            sr = res.get("scale_recovery_med")
            sr_s = f"{sr:.3f}" if sr is not None else "n/a"
            print(f"  [eval {method}/{sname}] scale_recovery(RF->scale)={sr_s} scale_factor={res['scale_factor_med']:.3f} "
                  f"AbsRel_sd={res['abs_rel_scale_dep']:.3f} AbsRel_si={res['abs_rel_scale_inv']:.3f} RMSE={res['rmse_scale_dep_m']:.2f}m")
        summary[method] = dict(history=history, const_scale=const_scale, eval=evals)
        np.savez(OUT / f"arrays_{method}.npz",
                 **{f"{sn}__{k}": (v if not isinstance(v, dict) else np.array([0]))
                    for sn, a in arrays.items() for k, v in a.items() if not isinstance(v, dict)},
                 **{f"{sn}__sample_{kk}": vv for sn, a in arrays.items()
                    if a.get("sample_depth") for kk, vv in a["sample_depth"].items()})
        del model; torch.cuda.empty_cache()
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n===== A/B SUMMARY =====")
    for m in ["rf_on", "rf_off"]:
        for sn, r in summary[m]["eval"].items():
            print(f"  {m:7s} {sn}: scale_factor={r['scale_factor_med']:.3f} |sf-1|={r['scale_factor_abs_err']:.3f} "
                  f"AbsRel_sd={r['abs_rel_scale_dep']:.3f} AbsRel_si={r['abs_rel_scale_inv']:.3f}")
    print(f"\nsaved -> {OUT}/summary.json , arrays_*.npz")


if __name__ == "__main__":
    main()
