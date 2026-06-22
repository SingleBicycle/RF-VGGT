"""Comprehensive RF-contribution validation for RF-VGGT.

Same architecture (use_rf=True, RFScaleHead present) throughout; variants differ ONLY in which RF
inputs are real vs zeroed (textbook modality ablation). Plus inference-time controls on the trained
full model: cross-scene RF shuffle (decisive 'reads RF content' test), noise-RF, zero-RF.

Metrics per variant/scene on held-out frames (80..99): metric scale-factor, RF->scale recovery,
AbsRel scale-dependent vs scale-invariant. Saves results/validate/summary.json + figures.
"""
import os, sys, json, argparse, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_ab as A

OUT = A.REPO / "results" / "validate"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0"

MODALITY_VARIANTS = {            # which RF modalities are REAL (rest zeroed), at train AND eval
    "full":        {"angular", "paths", "global"},
    "paths_only":  {"paths"},
    "angular_only":{"angular"},
    "global_only": {"global"},
    "none":        set(),        # all RF zeroed -> same-arch RF-off
}


def gate_rf(b, feed, perturb=None, noise_std=1.0):
    """Return (rf, rf_paths, rf_path_mask, rf_global); ablated modalities ZEROED (same shape)."""
    rf       = b["rf"].clone()        if "angular" in feed else torch.zeros_like(b["rf"])
    rf_paths = b["rf_paths"].clone()  if "paths"   in feed else torch.zeros_like(b["rf_paths"])
    rf_mask  = b["rf_path_mask"]      if "paths"   in feed else torch.zeros_like(b["rf_path_mask"])
    rf_glob  = b["rf_global"].clone() if "global"  in feed else torch.zeros_like(b["rf_global"])
    if perturb == "noise":
        if "angular" in feed: rf = rf + noise_std * torch.randn_like(rf)
        if "paths" in feed:   rf_paths = rf_paths + noise_std * torch.randn_like(rf_paths)
        if "global" in feed:  rf_glob = rf_glob + noise_std * torch.randn_like(rf_glob)
    return rf, rf_paths, rf_mask, rf_glob


def train_variant(name, feed, steps, n_frames, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = A.build_cfg(); ds = A.make_dataset(cfg)
    train_pool = [[f for f in s["valid_ids"] if f in set(A.TRAIN_FRAMES)] for s in ds.scenes]
    model = A.build_model(cfg, use_rf=True, device=device)   # SAME arch for every variant
    params = A.set_trainable(model)
    scale_p = [p for nm, p in model.named_parameters() if p.requires_grad and "rf_scale_head" in nm]
    other_p = [p for nm, p in model.named_parameters() if p.requires_grad and "rf_scale_head" not in nm]
    opt = torch.optim.AdamW([{"params": other_p, "lr": 2e-4}] + ([{"params": scale_p, "lr": 1e-3}] if scale_p else []), weight_decay=0.01)
    loss_fn = A.instantiate(cfg.loss, _recursive_=False)
    rng = np.random.default_rng(seed); hist = []
    model.train()
    print(f"\n=== TRAIN [{name}] feed={sorted(feed) or 'none'} : {steps} steps ===")
    for step in range(steps):
        si = int(rng.integers(0, len(ds.scenes)))
        fids = sorted(rng.choice(train_pool[si], size=min(n_frames, len(train_pool[si])), replace=False).tolist())
        b = A.get_batch(ds, si, fids, device)
        rf, rfp, rfm, rfg = gate_rf(b, feed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(images=b["images"], rf=rf, rf_paths=rfp, rf_path_mask=rfm, rf_global=rfg)
        ld = loss_fn(y, b)
        opt.zero_grad(set_to_none=True); ld["loss_total"].float().backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0 or step == steps - 1:
            gate = float(y["rf_aux"].get("adapter_gate_mean", torch.tensor(float("nan"))))
            hist.append(dict(step=step, gate=gate, scale_factor=float(ld.get("rf_scale_factor_mean", torch.tensor(float("nan"))))))
            print(f"  [{name}] {step:3d}: total={float(ld['loss_total']):.3f} rf_scale={float(ld.get('loss_rf_scale',0.0)):.3f} "
                  f"sf={float(ld.get('rf_scale_factor_mean',float('nan'))):.3f} gate={gate:.3f}")
    return model, ds, cfg, hist


@torch.no_grad()
def eval_variant(model, ds, cfg, feed, device, perturb=None, n_per=3):
    model.eval()
    sd, si_, sf, rec = [], [], [], []
    for scene_idx, sname in enumerate(A.SCENE_NAMES):
        if scene_idx >= len(ds.scenes): continue
        a_sd, a_si, a_sf, a_rec = [], [], [], []
        for i in range(0, len(A.VAL_FRAMES), n_per):
            fids = A.VAL_FRAMES[i:i+n_per]
            b = A.get_batch(ds, scene_idx, fids, device)
            rf, rfp, rfm, rfg = gate_rf(b, feed, perturb=perturb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(images=b["images"], rf=rf, rf_paths=rfp, rf_path_mask=rfm, rf_global=rfg)
            depth = y["depth"].float();  depth = depth[..., 0] if depth.dim() == 5 else depth
            unit_pred = depth[0]; gt = b["gt_metric_depth"][0].float()
            m = (b["point_masks"][0].bool() & (gt > 0)) if "point_masks" in b else (gt > 0)
            scale = float(torch.exp(y["rf_log_scale"].float()).reshape(1)) if y.get("rf_log_scale") is not None else 1.0
            a_rec.append(scale / float(b["metric_scale"].mean().clamp_min(1e-6)))
            mp = unit_pred * scale
            for s in range(unit_pred.shape[0]):
                ms = m[s]
                if ms.sum() < 100: continue
                p, g = mp[s][ms], gt[s][ms]
                a_sd.append(((p-g).abs()/g.clamp_min(1e-3)).mean().item())
                a_sf.append((torch.median(p)/torch.median(g)).item())
                r = torch.median(g)/torch.median(unit_pred[s][ms]).clamp_min(1e-6)
                a_si.append(((unit_pred[s][ms]*r - g).abs()/g.clamp_min(1e-3)).mean().item())
        sd.append((sname, float(np.mean(a_sd)), float(np.mean(a_si)), float(np.median(a_sf)), float(np.median(a_rec))))
    return {sn: dict(abs_rel_sd=v1, abs_rel_si=v2, scale_factor=v3, scale_recovery=v4) for sn, v1, v2, v3, v4 in sd}


@torch.no_grad()
def shuffle_test(model, ds, device, n_per=3):
    """Cross-scene RF shuffle: images from scene A, RF from scene B. Predicted scale should follow B."""
    model.eval(); out = {}
    if len(ds.scenes) < 2: return out
    for a, b_ in [(0, 1), (1, 0)]:
        recs_self, recs_other = [], []
        for i in range(0, len(A.VAL_FRAMES), n_per):
            fids = A.VAL_FRAMES[i:i+n_per]
            ba = A.get_batch(ds, a, fids, device)   # images + true scale of scene a
            bb = A.get_batch(ds, b_, fids, device)  # RF from scene b
            feed = {"angular", "paths", "global"}
            rf, rfp, rfm, rfg = gate_rf(bb, feed)    # RF from scene b
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(images=ba["images"], rf=rf, rf_paths=rfp, rf_path_mask=rfm, rf_global=rfg)
            scale = float(torch.exp(y["rf_log_scale"].float()).reshape(1))
            recs_self.append(scale / float(ba["metric_scale"].mean()))    # vs image-scene (A)  -> should be OFF
            recs_other.append(scale / float(bb["metric_scale"].mean()))   # vs RF-scene (B)     -> should be ~1
        out[f"img{A.SCENE_NAMES[a]}_rf{A.SCENE_NAMES[b_]}"] = dict(
            pred_over_imageScene=float(np.median(recs_self)),
            pred_over_rfScene=float(np.median(recs_other)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--frames", type=int, default=2)
    args = ap.parse_args()
    results = {"modality": {}, "history": {}, "controls": {}, "shuffle": {}}
    for name, feed in MODALITY_VARIANTS.items():
        torch.cuda.empty_cache(); gc.collect()
        model, ds, cfg, hist = train_variant(name, feed, args.steps, args.frames, DEV)
        results["modality"][name] = eval_variant(model, ds, cfg, feed, DEV)
        results["history"][name] = hist
        for sn, r in results["modality"][name].items():
            print(f"  [eval {name}/{sn}] scale_factor={r['scale_factor']:.3f} recovery={r['scale_recovery']:.3f} "
                  f"AbsRel_sd={r['abs_rel_sd']:.3f} AbsRel_si={r['abs_rel_si']:.3f}")
        if name == "full":
            # inference-time controls on the trained full model (before freeing it)
            results["controls"]["full"] = results["modality"]["full"]
            results["controls"]["zero_rf"] = eval_variant(model, ds, cfg, set(), DEV)
            results["controls"]["noise_rf"] = eval_variant(model, ds, cfg, feed, DEV, perturb="noise")
            results["shuffle"] = shuffle_test(model, ds, DEV)
            print(f"  [shuffle] {json.dumps(results['shuffle'])}")
        del model; torch.cuda.empty_cache(); gc.collect()
        with open(OUT / "summary.json", "w") as f:   # checkpoint progress after each variant
            json.dump(results, f, indent=2)
    with open(OUT / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n===== VALIDATION SUMMARY (held-out frames) =====")
    print(f"{'variant':14s} {'scene':6s} {'scaleFac':>8s} {'recovery':>8s} {'AbsRel_sd':>9s} {'AbsRel_si':>9s}")
    for name in MODALITY_VARIANTS:
        for sn, r in results["modality"][name].items():
            print(f"{name:14s} {sn.replace('AI53_','').replace('_Blender',''):6s} {r['scale_factor']:8.3f} "
                  f"{r['scale_recovery']:8.3f} {r['abs_rel_sd']:9.3f} {r['abs_rel_si']:9.3f}")
    print("\nControls on trained FULL model:")
    for k in ["full", "zero_rf", "noise_rf"]:
        for sn, r in results["controls"][k].items():
            print(f"  {k:9s} {sn.replace('AI53_','').replace('_Blender',''):6s} scaleFac={r['scale_factor']:.3f} AbsRel_sd={r['abs_rel_sd']:.3f}")
    print("\nCross-scene RF shuffle (pred scale / image-scene vs / RF-scene):")
    for k, v in results["shuffle"].items():
        print(f"  {k}: over_imageScene={v['pred_over_imageScene']:.3f}  over_rfScene={v['pred_over_rfScene']:.3f}")
    print(f"\nsaved -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
