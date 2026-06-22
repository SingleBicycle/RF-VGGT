"""Exp B3 (capstone): the scale-ambiguity test — why RF is necessary even though a learned
image-only scale head also works on in-distribution sim scenes (the AMB3R effect).

Physics: a GLOBAL SIMILARITY SCALING (world x k, camera baseline x k) leaves every rendered
image PIXEL-IDENTICAL but multiplies every RF time-of-flight delay by k. Therefore an
image-only scale head CANNOT distinguish a scene from its k-times copy (its prediction is
invariant -> recovery = 1/k), whereas an RF scale head reads the scaled delays and tracks k.

We simulate the similarity scaling on the cross-scene val set for k in {0.5, 1, 2}:
  - target metric_scale  *= k
  - RF delay feature + range  scaled by k (exact); dB gain / path-loss features offset by
    20*log10(k) (exact in dB domain); amplitude-log features held fixed (CONSERVATIVE for RF).
  - image features UNCHANGED (images are invariant to similarity scaling).
Then compare image-only vs RF-only scale recovery. This is the cleanest statement of the
paper's thesis: image-only feed-forward reconstruction is metrically scale-ambiguous; RF resolves it.
"""
import os, sys, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D
from scale_encoders import ScalePredictor
from exp_rf_only_scale import cache

OUT = D.REPO / "results" / "rf_only"
C0 = 299792458.0
DEV = "cuda:0"


def rescale_rf(paths, glob, range_m, k):
    """Apply a physically-consistent global similarity scaling (factor k) to the RF features."""
    P = paths.copy(); G = glob.copy()
    rng = np.clip(range_m * k, 1e-6, None)
    delay_ns = rng / C0 * 1e9
    P[..., 0] = np.log1p(np.maximum(delay_ns, 0.0)) / 10.0          # delay feature (exact)
    db_off = 20.0 * np.log10(max(k, 1e-6)) / 100.0                  # free-space path-loss change in dB/100
    P[..., 2] = P[..., 2] - db_off                                  # per-path gain_db/100 (exact in dB domain)
    G[..., 0] = G[..., 0] - db_off                                  # path_loss_db/100
    G[..., 1] = G[..., 1] - db_off                                  # total_path_gain/100
    return P, G


def train_rf(tr, seed, steps=2000, scale_aug=False):
    torch.manual_seed(seed)
    P0 = tr["paths"]; R0 = tr["range_m"]; G0 = tr["glob"]
    M = torch.from_numpy(tr["mask"]).bool().to(DEV)
    ms0 = tr["metric_scale"]
    m = ScalePredictor(path_dim=P0.shape[-1], global_dim=G0.shape[-1], pool="settransformer").to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); rng = np.random.default_rng(seed); n = P0.shape[0]
    Pt = torch.from_numpy(P0).float().to(DEV); Gt = torch.from_numpy(G0).float().to(DEV)
    yt = torch.log(torch.from_numpy(ms0).float().clamp_min(1e-3).to(DEV))
    m.train()
    for s in range(steps):
        idx = rng.choice(n, size=64, replace=False)
        if scale_aug:
            # similarity-scale augmentation: physically rescale RF delays/range/dB by random k, target by k
            k = np.exp(rng.uniform(np.log(0.3), np.log(3.0), size=len(idx))).astype(np.float32)
            Pk, Gk = P0[idx].copy(), G0[idx].copy()
            rngm = np.clip(R0[idx] * k[:, None, None], 1e-6, None)
            Pk[..., 0] = np.log1p(np.maximum(rngm / C0 * 1e9, 0.0)) / 10.0
            doff = (20.0 * np.log10(np.clip(k, 1e-6, None)) / 100.0)[:, None, None]
            Pk[..., 2] -= doff; Gk[..., 0] -= doff[:, :, 0]; Gk[..., 1] -= doff[:, :, 0]
            pin = torch.from_numpy(Pk).float().to(DEV); gin = torch.from_numpy(Gk).float().to(DEV)
            tgt = yt[idx] + torch.from_numpy(np.log(k)).float().to(DEV)
            loss = torch.nn.functional.smooth_l1_loss(m(pin, M[idx], gin), tgt)
        else:
            loss = torch.nn.functional.smooth_l1_loss(m(Pt[idx], M[idx], Gt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); return m


def train_img(F, seed, steps=3000):
    torch.manual_seed(seed)
    X = torch.from_numpy(F["Xtr"]).float().to(DEV); y = torch.log(torch.from_numpy(F["ytr"]).float().clamp_min(1e-3).to(DEV))
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6); Xn = (X - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 128),
                              torch.nn.GELU(), torch.nn.Linear(128, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3); rng = np.random.default_rng(seed); n = X.shape[0]
    net.train()
    for s in range(steps):
        idx = rng.choice(n, size=64, replace=False)
        loss = torch.nn.functional.smooth_l1_loss(net(Xn[idx]).squeeze(-1), y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval(); return net, mu, sd


def main():
    tr, va = cache()
    F = dict(np.load(OUT / "image_feats.npz", allow_pickle=True))
    seeds = 3
    rf_models = [train_rf(tr, s, scale_aug=False) for s in range(seeds)]
    rf_aug_models = [train_rf(tr, s, scale_aug=True) for s in range(seeds)]
    img_models = [train_img(F, s) for s in range(seeds)]
    Pva, Mva, Gva, Rva = va["paths"], va["mask"], va["glob"], va["range_m"]
    msva = va["metric_scale"]; Xva = torch.from_numpy(F["Xva"]).float().to(DEV); yva = F["yva"]

    def rf_recovery(models, Pk, Gk, tgt_log):
        out = []
        for m in models:
            with torch.no_grad():
                pr = m(torch.from_numpy(Pk).float().to(DEV), torch.from_numpy(Mva).bool().to(DEV),
                       torch.from_numpy(Gk).float().to(DEV)).cpu().numpy()
            out.append(np.exp(pr - tgt_log))
        return np.concatenate(out)

    res = {"k_sweep": []}
    for k in [0.5, 1.0, 2.0]:
        Pk, Gk = rescale_rf(Pva, Gva, Rva, k)
        tgt_log = np.log(np.clip(msva * k, 1e-3, None))
        rf_rec = rf_recovery(rf_models, Pk, Gk, tgt_log)
        rf_aug_rec = rf_recovery(rf_aug_models, Pk, Gk, tgt_log)
        img_rec = []
        for net, mu, sd in img_models:
            with torch.no_grad():
                pr = net(((Xva - mu) / sd)).squeeze(-1).cpu().numpy()
            img_rec.append(np.exp(pr - tgt_log))
        img_rec = np.concatenate(img_rec)
        ale = lambda r: float(np.abs(np.log(np.clip(r, 1e-6, None))).mean())
        row = dict(k=k, rf_recovery=float(rf_rec.mean()), rf_abs_log_err=ale(rf_rec),
                   rf_aug_recovery=float(rf_aug_rec.mean()), rf_aug_abs_log_err=ale(rf_aug_rec),
                   img_recovery=float(img_rec.mean()), img_abs_log_err=ale(img_rec))
        res["k_sweep"].append(row)
        print(f"k={k}:  RF(aug) rec={row['rf_aug_recovery']:.3f}(|log|{row['rf_aug_abs_log_err']:.3f})  "
              f"RF(no-aug) rec={row['rf_recovery']:.3f}(|log|{row['rf_abs_log_err']:.3f})  "
              f"IMAGE rec={row['img_recovery']:.3f}(|log|{row['img_abs_log_err']:.3f})")
    json.dump(res, open(OUT / "scale_ambiguity.json", "w"), indent=2)
    print("\nReading: under similarity scaling k, images are identical -> image head recovery≈1/k (fails);")
    print("RF head tracks the scaled delays -> recovery≈1. This is why RF is necessary, not just sufficient.")
    print("saved -> results/rf_only/scale_ambiguity.json")


if __name__ == "__main__":
    main()
