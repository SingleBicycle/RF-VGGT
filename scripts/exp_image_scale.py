"""Exp B2 (rigor): IMAGE-ONLY learned scale head vs RF-only scale head, cross-scene.

Motivated by AMB3R (arXiv:2511.20343): a learned scale head on frozen VGGT recovers metric
scale from IMAGE priors alone. A reviewer will ask whether RF actually beats a learned
image-only scale predictor (not just the best constant). This trains an MLP on pooled
frozen-VGGT image features -> log metric scale on 16 train scenes, evaluates cross-scene on
the 4 unseen val scenes, and compares against the RF-only predictor from Exp B.

If image-only generalizes POORLY across scenes (scale is image-ambiguous) while RF-only
generalizes, that is the decisive proof the metric scale must come from RF.
"""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D

OUT = D.REPO / "results" / "rf_only"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract(model, ds, windows, device):
    feats, ms, sid = [], [], []
    model.eval()
    for si, fids in windows:
        b = D.get_batch(ds, si, fids, device, want_angular=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            toks, _ = model.aggregator(b["images"])   # list of [B,S,N,2D]
        g = toks[-1].float().mean(dim=(1, 2))[0]       # pooled global descriptor [2D]
        feats.append(g.cpu().numpy()); ms.append(float(b["metric_scale"].mean())); sid.append(si)
    return np.stack(feats), np.asarray(ms, np.float32), np.asarray(sid, np.int64)


def windows_for(ds, n_per, n, seed, deterministic):
    rng = np.random.default_rng(seed)
    if deterministic:
        w = []
        for si, s in enumerate(ds.scenes):
            vids = list(s["valid_ids"])
            for i in range(0, len(vids) - n_per + 1, n_per):
                w.append((si, vids[i:i + n_per]))
        return w
    w = []
    for _ in range(n):
        si = int(rng.integers(0, len(ds.scenes)))
        vids = np.asarray(ds.scenes[si]["valid_ids"])
        w.append((si, sorted(rng.choice(vids, size=min(n_per, len(vids)), replace=False).tolist())))
    return w


def train_eval(Xtr, ytr, Xva, yva, sid_va, snames, device, steps=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Xtr).float().to(device); ytr = torch.from_numpy(np.log(np.clip(ytr, 1e-3, None))).float().to(device)
    Xva = torch.from_numpy(Xva).float().to(device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed); n = Xtr.shape[0]
    net.train()
    for s in range(steps):
        idx = rng.choice(n, size=min(64, n), replace=False)
        loss = torch.nn.functional.smooth_l1_loss(net(Xtr[idx]).squeeze(-1), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xva).squeeze(-1).cpu().numpy()
    true_log = np.log(np.clip(yva, 1e-3, None))
    ratio = np.exp(pred - true_log)
    per = {}
    for k in np.unique(sid_va):
        r = ratio[sid_va == k]; per[str(snames[k])] = dict(recovery_mean=float(r.mean()), recovery_std=float(r.std()))
    return dict(recovery_mean=float(ratio.mean()), recovery_std=float(ratio.std()),
                abs_log_err=float(np.abs(pred - true_log).mean()),
                mae_m=float(np.abs(np.exp(pred) - np.exp(true_log)).mean()), per_scene=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n_train", type=int, default=1280)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = args.device
    cfg = D.build_cfg()
    model = D.build_model(cfg, use_rf=False, device=dev)
    dtr = D.make_dataset(cfg, D.TRAIN_ROOT); dva = D.make_dataset(cfg, D.VAL_ROOT)
    wtr = windows_for(dtr, 3, args.n_train, 0, False)
    wva = windows_for(dva, 3, 0, 1, True)
    print(f"extracting image features: {len(wtr)} train, {len(wva)} val windows...", flush=True)
    Xtr, ytr, _ = extract(model, dtr, wtr, dev)
    Xva, yva, sid = extract(model, dva, wva, dev)
    snames = [s["seq_name"] for s in dva.scenes]
    np.savez(OUT / "image_feats.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, sid=sid, snames=np.array(snames))
    runs = [train_eval(Xtr, ytr, Xva, yva, sid, snames, dev, seed=s) for s in range(args.seeds)]
    res = dict(recovery_mean=float(np.mean([r["recovery_mean"] for r in runs])),
               recovery_mean_std=float(np.std([r["recovery_mean"] for r in runs])),
               recovery_window_std=float(np.mean([r["recovery_std"] for r in runs])),
               abs_log_err=float(np.mean([r["abs_log_err"] for r in runs])),
               mae_m=float(np.mean([r["mae_m"] for r in runs])), per_scene=runs[0]["per_scene"])
    print(f"[image_only_scale_head] recovery={res['recovery_mean']:.3f}±{res['recovery_mean_std']:.3f} "
          f"window_std={res['recovery_window_std']:.3f} abs_log_err={res['abs_log_err']:.3f} mae={res['mae_m']:.2f}m")
    # merge into rf_only summary for the comparison table
    summ = json.load(open(OUT / "summary.json")) if (OUT / "summary.json").exists() else {}
    summ["image_only_scale_head"] = res
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    json.dump(res, open(OUT / "image_scale.json", "w"), indent=2)
    print("saved -> results/rf_only/image_scale.json (+merged into summary.json)")


if __name__ == "__main__":
    main()
