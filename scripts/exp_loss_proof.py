"""Loss-level proofs that RF participates and drives the metric-scale loss.

(1) INPUT-GRADIENT SALIENCY: gradient of the scale loss w.r.t. each RF input feature.
    If RF participates in the loss, |d loss / d rf_input| is non-zero and concentrated on the
    range-bearing features (delay idx0, power idx1/2/5), not the bearing (AoA/AoD idx8-15).

(2) SHUFFLED-RF (broken-correspondence) CONTROL: train the SAME image-blind scale head but with
    RF features randomly permuted vs their targets (RF no longer matches the scene). The training
    objective can only be reduced via the genuine RF->scale structure; if it were a memorized prior,
    cross-scene recovery would be unchanged. We compare correct-RF vs shuffled-RF on cross-scene val.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rf_data as D
from scale_encoders import ScalePredictor
from exp_rf_only_scale import cache

OUT = D.REPO / "results" / "rf_only"
DEV = "cuda:0"
FEAT_NAMES = ["delay(range)", "pdp_power", "gain_db", "cir_re", "cir_im", "cir_absLog",
              "phase_sin", "phase_cos", "aoa_s1", "aoa_s2", "aoa_c1", "aoa_c2",
              "aod_s1", "aod_s2", "aod_c1", "aod_c2", "rank"]


def train(tr, seed, steps=2000, shuffle=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    P = torch.from_numpy(tr["paths"]).float().to(DEV); M = torch.from_numpy(tr["mask"]).bool().to(DEV)
    G = torch.from_numpy(tr["glob"]).float().to(DEV); y = torch.log(torch.from_numpy(tr["metric_scale"]).float().clamp_min(1e-3).to(DEV))
    if shuffle:  # break RF<->target correspondence (fixed permutation of RF rows vs targets)
        perm = rng.permutation(P.shape[0])
        P, M, G = P[perm], M[perm], G[perm]
    m = ScalePredictor(path_dim=P.shape[-1], global_dim=G.shape[-1], pool="settransformer").to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); n = P.shape[0]
    m.train(); last = 0.0
    for s in range(steps):
        idx = rng.choice(n, size=64, replace=False)
        loss = torch.nn.functional.smooth_l1_loss(m(P[idx], M[idx], G[idx]), y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last = float(loss)
    m.eval(); return m, last


def val_recovery(m, va):
    with torch.no_grad():
        pr = m(torch.from_numpy(va["paths"]).float().to(DEV), torch.from_numpy(va["mask"]).bool().to(DEV),
               torch.from_numpy(va["glob"]).float().to(DEV)).cpu().numpy()
    tl = np.log(np.clip(va["metric_scale"], 1e-3, None))
    return float(np.exp(pr - tl).mean()), float(np.abs(pr - tl).mean())


def saliency(m, va):
    """Mean |d loss / d input feature| over val windows, per path-feature index and per global index."""
    P = torch.from_numpy(va["paths"]).float().to(DEV).requires_grad_(True)
    G = torch.from_numpy(va["glob"]).float().to(DEV).requires_grad_(True)
    M = torch.from_numpy(va["mask"]).bool().to(DEV)
    y = torch.log(torch.from_numpy(va["metric_scale"]).float().clamp_min(1e-3).to(DEV))
    loss = torch.nn.functional.smooth_l1_loss(m(P, M, G), y)
    gP, gG = torch.autograd.grad(loss, [P, G])
    # average |grad| over valid paths only, per feature
    w = M.float().unsqueeze(-1)
    pf = (gP.abs() * w).sum(dim=(0, 1, 2)) / w.sum().clamp_min(1.0)   # [17]
    gf = gG.abs().mean(dim=(0, 1))                                    # [7]
    return pf.cpu().numpy(), gf.cpu().numpy()


def main():
    tr, va = cache()
    seeds = 3
    # correct vs shuffled
    correct = [train(tr, s, shuffle=False) for s in range(seeds)]
    shuf = [train(tr, s, shuffle=True) for s in range(seeds)]
    rc = [val_recovery(m, va) for m, _ in correct]
    rs = [val_recovery(m, va) for m, _ in shuf]
    res = dict(
        correct_train_loss=float(np.mean([l for _, l in correct])),
        correct_val_recovery=float(np.mean([r[0] for r in rc])),
        correct_val_abs_log_err=float(np.mean([r[1] for r in rc])),
        shuffled_train_loss=float(np.mean([l for _, l in shuf])),
        shuffled_val_recovery=float(np.mean([r[0] for r in rs])),
        shuffled_val_abs_log_err=float(np.mean([r[1] for r in rs])),
    )
    print("=== (2) Shuffled-RF (broken RF<->target) control ===")
    print(f"  correct RF : train_loss={res['correct_train_loss']:.4f}  val recovery={res['correct_val_recovery']:.3f}  abs_log_err={res['correct_val_abs_log_err']:.3f}")
    print(f"  shuffled RF: train_loss={res['shuffled_train_loss']:.4f}  val recovery={res['shuffled_val_recovery']:.3f}  abs_log_err={res['shuffled_val_abs_log_err']:.3f}")

    # saliency on the correct model
    pf, gf = saliency(correct[0][0], va)
    order = np.argsort(pf)[::-1]
    print("\n=== (1) Input-gradient saliency  |d loss / d rf_path_feature|  (top 8) ===")
    for i in order[:8]:
        print(f"  idx{i:2d} {FEAT_NAMES[i]:12s}: {pf[i]:.4e}")
    res["path_feature_saliency"] = {FEAT_NAMES[i]: float(pf[i]) for i in range(len(pf))}
    res["global_feature_saliency"] = [float(x) for x in gf]
    # range-vs-angle aggregate
    range_idx = [0, 1, 2, 5]; angle_idx = [8, 9, 10, 11, 12, 13, 14, 15]
    res["saliency_range_cues"] = float(pf[range_idx].sum())
    res["saliency_angle_cues"] = float(pf[angle_idx].sum())
    print(f"\n  Σ|grad| range/power cues (idx 0,1,2,5) = {res['saliency_range_cues']:.4e}")
    print(f"  Σ|grad| angle cues     (idx 8-15)      = {res['saliency_angle_cues']:.4e}")
    print(f"  range/angle ratio = {res['saliency_range_cues']/max(res['saliency_angle_cues'],1e-12):.1f}x")

    json.dump(res, open(OUT / "loss_proof.json", "w"), indent=2)
    print(f"\nsaved -> results/rf_only/loss_proof.json")


if __name__ == "__main__":
    main()
