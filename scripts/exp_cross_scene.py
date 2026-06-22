"""Cross-scene RF-VGGT driver (Exp A / C / D).

Train on the 16 RF_SCENES_train scenes, evaluate on the 4 unseen RF_SCENES_val scenes.
One invocation == one config. Parameterized by:
  --method     rf_on | rf_off
  --modality   full | paths | global | angular | none   (which RF inputs are REAL at train+eval)
  --angular_encoder final_v2 | cnn | shallow_vit
  --mode       frozen | partial   (partial = also unfreeze last 4 aggregator blocks)
  --steps --frames --lr --seed --tag
  --eval_controls  (also eval zero-RF / noise-RF / cross-scene-shuffle on the trained model)
Outputs results/cross/<tag>.json + results/cross/<tag>_arrays.npz (per-window arrays for stats).
"""
import os, sys, json, argparse, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D
from hydra.utils import instantiate

OUT = D.REPO / "results" / "cross"
OUT.mkdir(parents=True, exist_ok=True)

MODALITY = {
    "full": {"angular", "paths", "global"}, "paths": {"paths"}, "global": {"global"},
    "angular": {"angular"}, "none": set(),
}


def gate_rf(b, feed, perturb=None, noise_std=1.0, zero_delay=False):
    rf = b["rf"].clone() if "angular" in feed and "rf" in b else (torch.zeros_like(b["rf"]) if "rf" in b else None)
    rfp = b["rf_paths"].clone() if "paths" in feed else torch.zeros_like(b["rf_paths"])
    rfm = b["rf_path_mask"] if "paths" in feed else torch.zeros_like(b["rf_path_mask"])
    rfg = b["rf_global"].clone() if "global" in feed else torch.zeros_like(b["rf_global"])
    if zero_delay:
        rfp = rfp.clone(); rfp[..., 0] = 0.0   # remove delay (=range) feature
    if perturb == "noise":
        if rf is not None and "angular" in feed: rf = rf + noise_std * torch.randn_like(rf)
        if "paths" in feed: rfp = rfp + noise_std * torch.randn_like(rfp)
        if "global" in feed: rfg = rfg + noise_std * torch.randn_like(rfg)
    return rf, rfp, rfm, rfg


def train(args, device):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    use_rf = (args.method == "rf_on")
    feed = MODALITY[args.modality] if use_rf else set()
    cfg = D.build_cfg()
    ds = D.make_dataset(cfg, D.TRAIN_ROOT, sampling="mixed", inside_random=False)
    model = D.build_model(cfg, use_rf=use_rf, device=device,
                          angular_encoder=(args.angular_encoder if use_rf else None))
    params = D.set_trainable(model, args.mode, unfreeze_last=args.unfreeze_last)
    scale_p = [p for n, p in model.named_parameters() if p.requires_grad and "rf_scale_head" in n]
    other_p = [p for n, p in model.named_parameters() if p.requires_grad and "rf_scale_head" not in n]
    groups = [{"params": other_p, "lr": args.lr}] + ([{"params": scale_p, "lr": args.lr * 5}] if scale_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    loss_fn = instantiate(cfg.loss, _recursive_=False)
    rng = np.random.default_rng(args.seed)
    seen_scales, hist = [], []
    nsc = len(ds.scenes)
    print(f"=== TRAIN tag={args.tag} method={args.method} mod={args.modality} enc={args.angular_encoder} "
          f"mode={args.mode} steps={args.steps} params={sum(p.numel() for p in params)/1e6:.0f}M ===", flush=True)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        si = int(rng.integers(0, nsc))
        vids = np.asarray(ds.scenes[si]["valid_ids"])
        fids = sorted(rng.choice(vids, size=min(args.frames, len(vids)), replace=False).tolist())
        b = D.get_batch(ds, si, fids, device, want_angular=use_rf)
        seen_scales.append(float(b["metric_scale"].mean()))
        rf, rfp, rfm, rfg = gate_rf(b, feed) if use_rf else (None, None, None, None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(images=b["images"], rf=rf, rf_paths=rfp, rf_path_mask=rfm, rf_global=rfg)
        ld = loss_fn(y, b)
        opt.zero_grad(set_to_none=True); ld["loss_total"].float().backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            gate = float(y["rf_aux"].get("adapter_gate_mean", torch.tensor(float("nan")))) if use_rf else float("nan")
            sf = float(ld.get("rf_scale_factor_mean", torch.tensor(float("nan"))))
            hist.append(dict(step=step, loss_total=float(ld["loss_total"]), loss_depth=float(ld["loss_depth"]),
                             loss_rf_scale=float(ld.get("loss_rf_scale", 0.0)), scale_factor=sf, gate=gate))
            print(f"  step {step:4d} total={float(ld['loss_total']):.3f} depth={float(ld['loss_depth']):.3f} "
                  f"rf_scale={float(ld.get('loss_rf_scale',0.0)):.3f} sf={sf:.3f} gate={gate:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    const_scale = float(np.mean(seen_scales))
    return model, ds, cfg, hist, const_scale, feed


@torch.no_grad()
def evaluate(model, cfg, use_rf, feed, device, const_scale, perturb=None, zero_delay=False, shuffle=False):
    """Cross-scene eval on the 4 val scenes. Returns aggregate + per-scene + per-window arrays."""
    dv = D.make_dataset(cfg, D.VAL_ROOT, sampling="uniform")
    per_scene_wins = D.val_windows(dv, n_per=3)
    rows = []  # (scene, recovery, absrel_sd_mean, absrel_si_mean, scale_factor_med)
    scenes_meta = [(si, name) for si, name, _ in per_scene_wins]
    # cache windows + RF for shuffle (need another scene's RF)
    cached = {}
    for si, name, wins in per_scene_wins:
        cached[si] = [D.get_batch(dv, si, w, device, want_angular=use_rf) for w in wins]
    model.eval()
    for si, name, wins in per_scene_wins:
        for wi, b in enumerate(cached[si]):
            if use_rf:
                if shuffle:
                    other = (si + 1) % len(per_scene_wins)
                    bo = cached[other][wi % len(cached[other])]
                    rf, rfp, rfm, rfg = gate_rf(bo, feed)  # RF from a DIFFERENT scene
                    true_scale_rf = float(bo["metric_scale"].mean())
                else:
                    rf, rfp, rfm, rfg = gate_rf(b, feed, perturb=perturb, zero_delay=zero_delay)
                    true_scale_rf = None
                y = model(images=b["images"], rf=rf, rf_paths=rfp, rf_path_mask=rfm, rf_global=rfg)
            else:
                y = model(images=b["images"])
            depth = y["depth"].float(); depth = depth[..., 0] if depth.dim() == 5 else depth
            unit_pred = depth[0]; gt = b["gt_metric_depth"][0].float()
            m = (b["point_masks"][0].bool() & (gt > 0)) if "point_masks" in b else (gt > 0)
            if use_rf and y.get("rf_log_scale") is not None:
                scale = float(torch.exp(y["rf_log_scale"].float()).reshape(-1)[0])
                rec = scale / float(b["metric_scale"].mean().clamp_min(1e-6))
                rec_rf = (scale / true_scale_rf) if true_scale_rf else None
            else:
                scale = const_scale; rec = scale / float(b["metric_scale"].mean()); rec_rf = None
            mp = unit_pred * scale
            sd, sif, sfac = [], [], []
            for s in range(unit_pred.shape[0]):
                ms = m[s]
                if ms.sum() < 100: continue
                p, g = mp[s][ms], gt[s][ms]
                sd.append(((p - g).abs() / g.clamp_min(1e-3)).mean().item())
                sfac.append((torch.median(p) / torch.median(g)).item())
                r = torch.median(g) / torch.median(unit_pred[s][ms]).clamp_min(1e-6)
                sif.append(((unit_pred[s][ms] * r - g).abs() / g.clamp_min(1e-3)).mean().item())
            if sd:
                rows.append(dict(scene=name, recovery=rec, recovery_rf=rec_rf, absrel_sd=float(np.mean(sd)),
                                 absrel_si=float(np.mean(sif)), scale_factor=float(np.median(sfac))))
    return aggregate(rows)


def aggregate(rows):
    if not rows:
        return dict(n=0)
    def arr(k): return np.array([r[k] for r in rows if r[k] is not None], dtype=np.float64)
    agg = dict(n=len(rows),
               absrel_sd_mean=float(arr("absrel_sd").mean()), absrel_sd_std=float(arr("absrel_sd").std()),
               absrel_si_mean=float(arr("absrel_si").mean()), absrel_si_std=float(arr("absrel_si").std()),
               recovery_mean=float(arr("recovery").mean()), recovery_std=float(arr("recovery").std()),
               scale_factor_med=float(np.median(arr("scale_factor"))))
    rf = arr("recovery_rf")
    if rf.size: agg["recovery_rf_mean"] = float(rf.mean())
    per = {}
    for name in sorted(set(r["scene"] for r in rows)):
        sr = [r for r in rows if r["scene"] == name]
        per[name] = dict(n=len(sr),
                         absrel_sd=float(np.mean([r["absrel_sd"] for r in sr])),
                         absrel_si=float(np.mean([r["absrel_si"] for r in sr])),
                         recovery_mean=float(np.mean([r["recovery"] for r in sr])),
                         recovery_std=float(np.std([r["recovery"] for r in sr])),
                         scale_factor=float(np.median([r["scale_factor"] for r in sr])))
    agg["per_scene"] = per
    agg["rows"] = rows
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="rf_on", choices=["rf_on", "rf_off"])
    ap.add_argument("--modality", default="full", choices=list(MODALITY))
    ap.add_argument("--angular_encoder", default="final_v2")
    ap.add_argument("--mode", default="frozen", choices=["frozen", "partial"])
    ap.add_argument("--unfreeze_last", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval_controls", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()
    model, ds, cfg, hist, const_scale, feed = train(args, args.device)
    out = dict(config=vars(args), history=hist, const_scale=const_scale)
    out["eval"] = evaluate(model, cfg, args.method == "rf_on", feed, args.device, const_scale)
    e = out["eval"]
    print(f"\n[EVAL {args.tag}] recovery={e.get('recovery_mean'):.3f}±{e.get('recovery_std'):.3f} "
          f"AbsRel_sd={e.get('absrel_sd_mean'):.3f} AbsRel_si={e.get('absrel_si_mean'):.3f} "
          f"sf={e.get('scale_factor_med'):.3f}", flush=True)
    for name, r in e["per_scene"].items():
        print(f"    {name}: rec={r['recovery_mean']:.3f}±{r['recovery_std']:.3f} sd={r['absrel_sd']:.3f} si={r['absrel_si']:.3f}")
    if args.eval_controls and args.method == "rf_on":
        out["controls"] = {
            "zero_rf": evaluate(model, cfg, True, set(), args.device, const_scale),
            "noise_rf": evaluate(model, cfg, True, feed, args.device, const_scale, perturb="noise"),
            "no_delay": evaluate(model, cfg, True, feed, args.device, const_scale, zero_delay=True),
            "shuffle": evaluate(model, cfg, True, feed, args.device, const_scale, shuffle=True),
        }
        for k, v in out["controls"].items():
            extra = f" rec_vs_rfscene={v.get('recovery_rf_mean'):.3f}" if v.get("recovery_rf_mean") else ""
            print(f"  [control {k}] AbsRel_sd={v['absrel_sd_mean']:.3f} recovery={v['recovery_mean']:.3f}{extra}", flush=True)
    out["wall_sec"] = time.time() - t0
    # split heavy rows into a side file
    rows = out["eval"].pop("rows", [])
    np.savez(OUT / f"{args.tag}_arrays.npz", rows=json.dumps(rows))
    if "controls" in out:
        for v in out["controls"].values(): v.pop("rows", None)
    with open(OUT / f"{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT}/{args.tag}.json  ({out['wall_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
