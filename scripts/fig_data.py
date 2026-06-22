"""Data-foundation figures: RF physics (LOS ratio), cross-scene scale spread, RF-only scatter."""
import sys, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
FIGS = REPO / "results" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
C = 299792458.0


def cam_centers(extr):
    return np.array([-(e[:3, :3].T @ e[:3, 3]) for e in extr])


def survey(split):
    root = f"/DATA/zihao/projects/rf_vggt/RF_SCENES_{split}"
    scenes = sorted(d for d in glob.glob(root + "/AI53_*_Blender") if os.path.isfile(d + "/cameras.npz"))
    ratios, med_depth, names = [], [], []
    for s in scenes:
        cam = np.load(s + "/cameras.npz"); extr = cam["extrinsics"].astype(np.float64)
        for f in sorted(glob.glob(s + "/rf/*.npz")):
            d = np.load(f); dly = d["cir_delays"]; dly = dly[dly > 0]
            if dly.size == 0: continue
            ratios.append(C * dly.min() / max(np.linalg.norm(d["tx_position"] - d["rx_position"]), 1e-9))
        meds = [np.median(np.load(df)[np.isfinite(np.load(df)) & (np.load(df) > 0)]) for df in sorted(glob.glob(s + "/depths/*.npy"))[:30]]
        med_depth.append(np.median(meds)); names.append(os.path.basename(s).replace("AI53_", "").replace("_Blender", ""))
    return np.array(ratios), med_depth, names


def main():
    rt, dt, nt = survey("train"); rv, dv, nv = survey("val")

    # Fig 0a: LOS ratio histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    allr = np.concatenate([rt, rv])
    ax[0].hist(allr, bins=60, color="#48a")
    ax[0].axvline(1.0, ls="--", c="k", lw=1.5, label="ideal LOS=1.0")
    ax[0].set_xlabel("c·min(delay) / ‖tx−rx‖"); ax[0].set_ylabel("# frames")
    ax[0].set_title(f"RF first-arrival = exact bistatic range\n(mean={allr.mean():.5f}, {len(allr)} frames, 20 scenes)")
    ax[0].legend()

    # Fig 0b: per-scene scale spread
    x = np.arange(len(nt))
    ax[1].bar(x, dt, color="#2a7", label="train (16)")
    ax[1].bar(np.arange(len(nv)) + len(nt) + 0.5, dv, color="#d83", label="val (4, unseen)")
    ax[1].set_xticks(list(x) + list(np.arange(len(nv)) + len(nt) + 0.5))
    ax[1].set_xticklabels(nt + nv, rotation=90, fontsize=7)
    ax[1].set_ylabel("median scene depth (m)")
    ax[1].set_title(f"Cross-scene scale varies {max(dt+dv)/min(dt+dv):.1f}× → metric scale must be read from RF")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig0_data_foundation.png", dpi=130); plt.close(fig)

    # Fig 6: Exp B pred-vs-true scatter (settransformer) — recompute on cache
    try:
        import torch, rf_data as D
        from scale_encoders import ScalePredictor
        from exp_rf_only_scale import cache
        tr, va = cache()
        dev = "cpu"
        P = torch.from_numpy(tr["paths"]).float(); M = torch.from_numpy(tr["mask"]).bool(); G = torch.from_numpy(tr["glob"]).float()
        y = torch.log(torch.from_numpy(tr["metric_scale"]).float().clamp_min(1e-3))
        m = ScalePredictor(path_dim=P.shape[-1], global_dim=G.shape[-1], pool="settransformer")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        rng = np.random.default_rng(0); n = P.shape[0]
        m.train()
        for s in range(2000):
            idx = rng.choice(n, size=64, replace=False)
            loss = torch.nn.functional.smooth_l1_loss(m(P[idx], M[idx], G[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = np.exp(m(torch.from_numpy(va["paths"]).float(), torch.from_numpy(va["mask"]).bool(),
                            torch.from_numpy(va["glob"]).float()).numpy())
        true = va["metric_scale"]; sid = va["scene_id"]; snames = list(va["scene_names"])
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        cols = ["#2a7", "#d83", "#48a", "#a4a"]
        for k in np.unique(sid):
            ax.scatter(true[sid == k], pred[sid == k], s=18, alpha=0.7, color=cols[k % 4],
                       label=str(snames[k]).replace("AI53_", "").replace("_Blender", ""))
        lim = [0, max(true.max(), pred.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="pred=GT")
        ax.set_xlabel("GT metric scale (m)"); ax.set_ylabel("RF-predicted metric scale (m)")
        ax.set_title("Image-blind RF → metric scale (4 unseen scenes)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGS / "fig6_rf_only_scatter.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("scatter skipped:", e)
    print(f"data figures -> {FIGS}/  (LOS mean over {len(allr)} frames = {allr.mean():.5f})")


if __name__ == "__main__":
    main()
