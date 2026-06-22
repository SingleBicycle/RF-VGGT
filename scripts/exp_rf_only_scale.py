"""Exp B: image-blind RF -> metric-scale recovery, cross-scene (16 train -> 4 val).

Step 1: build a window cache (rf_paths/mask/global/range_m/los/metric_scale) from the
        RFSceneDataset (no images fed to any predictor).
Step 2: fit/eval:
   - analytic    : log metric_scale = a*log(median path range) + b  (least squares on train)
   - analytic_los: log metric_scale = a*log(LOS range) + b
   - deepsets / pointnet / settransformer : learned path-set encoders
Reports cross-scene scale recovery (mean exp(pred-true)->1), |log| error, per-scene stats.
Also a range-removal control: zero the delay feature (index 0) at train+eval -> recovery collapses.
"""
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D
from scale_encoders import ScalePredictor

OUT = D.REPO / "results" / "rf_only"
OUT.mkdir(parents=True, exist_ok=True)
DELAY_IDX = 0  # feature 0 of rf_paths encodes delay (=> range); see RFScaleHead / pack_raw_rf_npz


def build_cache(split_root, n_per, n_windows, seed, deterministic):
    cfg = D.build_cfg()
    ds = D.make_dataset(cfg, split_root, sampling="uniform")
    rng = np.random.default_rng(seed)
    P, M, G, RG, LOS, MS, SID = [], [], [], [], [], [], []
    if deterministic:
        windows = []
        for si, s in enumerate(ds.scenes):
            vids = list(s["valid_ids"])
            for i in range(0, len(vids) - n_per + 1, n_per):
                windows.append((si, vids[i:i + n_per]))
    else:
        windows = []
        for _ in range(n_windows):
            si = int(rng.integers(0, len(ds.scenes)))
            vids = np.asarray(ds.scenes[si]["valid_ids"])
            fids = sorted(rng.choice(vids, size=min(n_per, len(vids)), replace=False).tolist())
            windows.append((si, fids))
    for si, fids in windows:
        b = D.get_batch(ds, si, fids, "cpu", want_angular=False)
        P.append(b["rf_paths"][0].numpy()); M.append(b["rf_path_mask"][0].numpy())
        G.append(b["rf_global"][0].numpy()); RG.append(b["rf_path_range_m"][0].numpy())
        LOS.append(b["rf_los_range_m"][0].numpy()); MS.append(float(b["metric_scale"].mean()))
        SID.append(si)
    return dict(paths=np.stack(P), mask=np.stack(M), glob=np.stack(G), range_m=np.stack(RG),
                los=np.stack(LOS), metric_scale=np.asarray(MS, np.float32),
                scene_id=np.asarray(SID, np.int64),
                scene_names=[s["seq_name"] for s in ds.scenes])


def cache(n_per=3, n_train=1280, seed=0, force=False):
    ftr, fva = OUT / "cache_train.npz", OUT / "cache_val.npz"
    if force or not ftr.exists():
        print("building train cache..."); c = build_cache(D.TRAIN_ROOT, n_per, n_train, seed, deterministic=False)
        np.savez(ftr, **{k: v for k, v in c.items() if k != "scene_names"}, scene_names=np.array(c["scene_names"]))
    if force or not fva.exists():
        print("building val cache..."); c = build_cache(D.VAL_ROOT, n_per, 0, seed + 1, deterministic=True)
        np.savez(fva, **{k: v for k, v in c.items() if k != "scene_names"}, scene_names=np.array(c["scene_names"]))
    tr = dict(np.load(ftr, allow_pickle=True)); va = dict(np.load(fva, allow_pickle=True))
    print(f"train windows={len(tr['metric_scale'])} val windows={len(va['metric_scale'])} "
          f"(val scenes={list(va['scene_names'])})")
    return tr, va


def metrics(pred_log, true_log, scene_id, scene_names):
    pred_log, true_log = np.asarray(pred_log), np.asarray(true_log)
    ratio = np.exp(pred_log - true_log)
    res = dict(recovery_mean=float(ratio.mean()), recovery_median=float(np.median(ratio)),
               recovery_std=float(ratio.std()), abs_log_err=float(np.abs(pred_log - true_log).mean()),
               mae_m=float(np.abs(np.exp(pred_log) - np.exp(true_log)).mean()))
    per = {}
    for si in np.unique(scene_id):
        r = ratio[scene_id == si]
        per[str(scene_names[si])] = dict(recovery_mean=float(r.mean()), recovery_std=float(r.std()), n=int(r.size))
    res["per_scene"] = per
    return res


def fit_analytic(tr, va, feat_key):
    def feat(c):
        if feat_key == "los":
            f = np.log(np.clip(c["los"].mean(1), 1e-3, None))           # mean LOS range per window
        else:
            rng = c["range_m"]; mask = c["mask"]
            mr = np.where(mask, rng, np.nan)
            f = np.log(np.clip(np.nanmedian(mr.reshape(mr.shape[0], -1), axis=1), 1e-3, None))
        return np.nan_to_num(f, nan=0.0)
    xtr, ytr = feat(tr), np.log(np.clip(tr["metric_scale"], 1e-3, None))
    A = np.stack([xtr, np.ones_like(xtr)], 1)
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    xva = feat(va); pred = coef[0] * xva + coef[1]
    return metrics(pred, np.log(np.clip(va["metric_scale"], 1e-3, None)), va["scene_id"], va["scene_names"]) | dict(coef=coef.tolist())


# Feature-index groups within the 17-dim path vector (see pack_raw_rf_npz):
#   0 delay(range) | 1,2,5 power/amplitude | 3,4 cir re/im | 6,7 phase | 8-15 AoA/AoD angles | 16 rank
DROP = {
    "none": ([], []),                                   # full
    "delay": ([0], []),                                 # remove only the delay feature
    "range_cues": ([0, 1, 2, 3, 4, 5], [0, 1]),         # remove delay + all power/amplitude + range-correlated globals
    "angles_only": ([0, 1, 2, 3, 4, 5, 6, 7, 16], [0, 1, 2, 3, 4, 5, 6]),  # keep ONLY AoA/AoD (bearing); zero all globals
}


def train_learned(tr, va, pool, steps, lr, device, seed, drop="none"):
    torch.manual_seed(seed); np.random.seed(seed)
    def t(c, k): return torch.from_numpy(c[k]).to(device)
    Ptr, Mtr, Gtr = t(tr, "paths").float(), t(tr, "mask").bool(), t(tr, "glob").float()
    ytr = torch.log(t(tr, "metric_scale").float().clamp_min(1e-3))
    Pva, Mva, Gva = t(va, "paths").float(), t(va, "mask").bool(), t(va, "glob").float()
    pf_idx, gf_idx = DROP[drop]
    if pf_idx or gf_idx:
        Ptr, Pva, Gtr, Gva = Ptr.clone(), Pva.clone(), Gtr.clone(), Gva.clone()
        for i in pf_idx: Ptr[..., i] = 0.0; Pva[..., i] = 0.0
        for i in gf_idx: Gtr[..., i] = 0.0; Gva[..., i] = 0.0
    model = ScalePredictor(path_dim=Ptr.shape[-1], global_dim=Gtr.shape[-1], pool=pool).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = Ptr.shape[0]; rng = np.random.default_rng(seed); bs = 64
    model.train()
    for step in range(steps):
        idx = rng.choice(n, size=min(bs, n), replace=False)
        pred = model(Ptr[idx], Mtr[idx], Gtr[idx])
        loss = torch.nn.functional.smooth_l1_loss(pred, ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Pva, Mva, Gva).cpu().numpy()
    return metrics(pred, np.log(np.clip(va["metric_scale"], 1e-3, None)), va["scene_id"], va["scene_names"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force_cache", action="store_true")
    args = ap.parse_args()
    tr, va = cache(force=args.force_cache)
    res = {}
    res["analytic_median_range"] = fit_analytic(tr, va, "range")
    res["analytic_los"] = fit_analytic(tr, va, "los")
    for pool in ["mean", "max", "settransformer"]:
        runs = [train_learned(tr, va, pool, args.steps, args.lr, args.device, seed=s) for s in range(args.seeds)]
        agg = dict(recovery_mean=float(np.mean([r["recovery_mean"] for r in runs])),
                   recovery_mean_std=float(np.std([r["recovery_mean"] for r in runs])),
                   abs_log_err=float(np.mean([r["abs_log_err"] for r in runs])),
                   mae_m=float(np.mean([r["mae_m"] for r in runs])),
                   per_scene=runs[0]["per_scene"], seeds=args.seeds)
        name = {"mean": "deepsets", "max": "pointnet", "settransformer": "settransformer"}[pool]
        res[name] = agg
        print(f"[{name:14s}] recovery={agg['recovery_mean']:.3f}±{agg['recovery_mean_std']:.3f} "
              f"abs_log_err={agg['abs_log_err']:.3f} mae={agg['mae_m']:.2f}m")
    # range-removal controls on the best learned encoder (settransformer), avg over seeds
    for drop in ["delay", "range_cues", "angles_only"]:
        runs = [train_learned(tr, va, "settransformer", args.steps, args.lr, args.device, seed=s, drop=drop)
                for s in range(args.seeds)]
        res[f"settransformer_drop_{drop}"] = dict(
            recovery_mean=float(np.mean([r["recovery_mean"] for r in runs])),
            recovery_mean_std=float(np.std([r["recovery_mean"] for r in runs])),
            recovery_window_std=float(np.mean([r["recovery_std"] for r in runs])),
            abs_log_err=float(np.mean([r["abs_log_err"] for r in runs])),
            mae_m=float(np.mean([r["mae_m"] for r in runs])))
    for k in ["analytic_median_range", "analytic_los"]:
        print(f"[{k:20s}] recovery={res[k]['recovery_mean']:.3f} abs_log_err={res[k]['abs_log_err']:.3f} mae={res[k]['mae_m']:.2f}m")
    for drop in ["delay", "range_cues", "angles_only"]:
        v = res[f"settransformer_drop_{drop}"]
        print(f"[ST drop={drop:12s}] recovery={v['recovery_mean']:.3f} window_std={v['recovery_window_std']:.3f} "
              f"abs_log_err={v['abs_log_err']:.3f}  (range-removal)")
    with open(OUT / "summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
