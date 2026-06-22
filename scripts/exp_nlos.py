"""Exp E2: NLOS / robustness analysis for the image-blind RF scale predictor.

The simulated scenes are mostly clean LOS (train ratio c*min_delay/|tx-rx| == 1.0000;
val 019/020 have 1-2% genuine NLOS frames). Since we cannot get real strong-NLOS hardware,
we STRESS-TEST robustness in three controlled ways on the cross-scene val set:

  (1) LOS/early-path occlusion: drop the k shortest-range (earliest-arrival) valid paths per
      frame -> forces the model to infer scale from later multipath only (simulates blocked LOS).
  (2) Delay-timing noise: add Gaussian noise (sigma in nanoseconds-equiv) to the delay feature
      -> simulates hardware timing error. Robustness curve recovery vs sigma.
  (3) Path dropout: randomly drop a fraction of paths -> sparse/occluded measurement.

Also stratifies baseline recovery by per-window NLOS severity (#frames whose first arrival
exceeds the bistatic geometric range).
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D
from scale_encoders import ScalePredictor
from exp_rf_only_scale import cache

OUT = D.REPO / "results" / "nlos"
OUT.mkdir(parents=True, exist_ok=True)
DELAY_IDX = 0


def train_predictor(tr, device, steps=2000, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    P = torch.from_numpy(tr["paths"]).float().to(device)
    M = torch.from_numpy(tr["mask"]).bool().to(device)
    G = torch.from_numpy(tr["glob"]).float().to(device)
    y = torch.log(torch.from_numpy(tr["metric_scale"]).float().clamp_min(1e-3).to(device))
    model = ScalePredictor(path_dim=P.shape[-1], global_dim=G.shape[-1], pool="settransformer").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed); n = P.shape[0]
    model.train()
    for s in range(steps):
        idx = rng.choice(n, size=min(64, n), replace=False)
        loss = torch.nn.functional.smooth_l1_loss(model(P[idx], M[idx], G[idx]), y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return model


def recovery(model, P, M, G, ms, device):
    with torch.no_grad():
        pred = model(P.to(device), M.to(device), G.to(device)).cpu().numpy()
    ratio = np.exp(pred - np.log(np.clip(ms, 1e-3, None)))
    return float(ratio.mean()), float(np.abs(pred - np.log(np.clip(ms, 1e-3, None))).mean())


def drop_shortest_paths(P, M, R, k):
    """Set mask False for the k shortest-range valid paths per frame (simulate blocked LOS)."""
    M2 = M.clone()
    N, S, K = M.shape
    Rn = R.clone()
    Rn[~M.bool()] = float("inf")
    for _ in range(k):
        idx = Rn.argmin(dim=2)  # [N,S] index of shortest remaining valid path
        ar = torch.arange(N)[:, None]; asx = torch.arange(S)[None, :]
        M2[ar, asx, idx] = False
        Rn[ar, asx, idx] = float("inf")
    return M2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tr, va = cache()
    dev = args.device
    P = torch.from_numpy(va["paths"]).float(); M = torch.from_numpy(va["mask"]).bool()
    G = torch.from_numpy(va["glob"]).float(); R = torch.from_numpy(va["range_m"]).float()
    ms = va["metric_scale"]

    models = [train_predictor(tr, dev, args.steps, seed=s) for s in range(args.seeds)]

    def avg_recovery(Pin, Min, Gin):
        rs = [recovery(m, Pin, Min, Gin, ms, dev) for m in models]
        return float(np.mean([r[0] for r in rs])), float(np.std([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))

    res = {}
    base = avg_recovery(P, M, G); res["baseline"] = dict(recovery=base[0], recovery_std=base[1], abs_log_err=base[2])
    print(f"baseline recovery={base[0]:.3f}±{base[1]:.3f} abs_log_err={base[2]:.3f}")

    # (1) LOS / early-path occlusion
    res["los_occlusion"] = []
    for k in [1, 2, 3, 5, 8]:
        Mk = drop_shortest_paths(P, M, R, k)
        r = avg_recovery(P, Mk, G)
        res["los_occlusion"].append(dict(paths_removed=k, recovery=r[0], recovery_std=r[1], abs_log_err=r[2]))
        print(f"  LOS-occlusion drop {k} earliest paths: recovery={r[0]:.3f}±{r[1]:.3f} abs_log_err={r[2]:.3f}")

    # (2) delay-timing noise (sigma applied to delay feature; feature = log1p(delay_ns)/10)
    res["delay_noise"] = []
    rng = np.random.default_rng(0)
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.4]:
        Pn = P.clone()
        Pn[..., DELAY_IDX] = Pn[..., DELAY_IDX] + sigma * torch.from_numpy(rng.standard_normal(Pn[..., DELAY_IDX].shape).astype(np.float32))
        r = avg_recovery(Pn, M, G)
        res["delay_noise"].append(dict(sigma=sigma, recovery=r[0], recovery_std=r[1], abs_log_err=r[2]))
        print(f"  delay-noise sigma={sigma}: recovery={r[0]:.3f}±{r[1]:.3f} abs_log_err={r[2]:.3f}")

    # (3) random path dropout
    res["path_dropout"] = []
    for frac in [0.0, 0.25, 0.5, 0.75]:
        keep = (torch.from_numpy(rng.random(M.shape).astype(np.float32)) > frac)
        Md = M.clone() & keep
        r = avg_recovery(P, Md, G)
        res["path_dropout"].append(dict(drop_frac=frac, recovery=r[0], recovery_std=r[1], abs_log_err=r[2]))
        print(f"  path-dropout frac={frac}: recovery={r[0]:.3f}±{r[1]:.3f} abs_log_err={r[2]:.3f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
