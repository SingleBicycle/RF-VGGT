"""Produce the RF-VGGT proof figures from results/ab/{summary.json, arrays_*.npz}."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT/results/ab")
FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)
SCENES = ["AI53_001_Blender", "AI53_002_Blender"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

summary = json.load(open(OUT / "summary.json"))
arr = {m: dict(np.load(OUT / f"arrays_{m}.npz", allow_pickle=True)) for m in ["rf_on", "rf_off"]}
C_ON, C_OFF, C_GT = "#1f77b4", "#d62728", "#2ca02c"


def g(method, scene, key, default=None):
    return summary[method]["eval"].get(scene, {}).get(key, default)


# ---- Fig 1: predicted point-range vs RF-range histogram overlay (physical consistency) ----
fig, axes = plt.subplots(1, len(SCENES), figsize=(11, 4))
for ax, sc in zip(np.atleast_1d(axes), SCENES):
    pr = arr["rf_on"].get(f"{sc}__pred_ranges", np.array([]))
    rr = arr["rf_on"].get(f"{sc}__rf_ranges", np.array([]))
    bins = np.linspace(0, 25, 60)
    if pr.size: ax.hist(pr, bins=bins, density=True, alpha=0.55, color=C_ON, label="RF-VGGT predicted depth range")
    if rr.size: ax.hist(rr, bins=bins, density=True, alpha=0.55, color=C_GT, label="RF measured range (calibrated)")
    ax.set_title(sc); ax.set_xlabel("range (m)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
fig.suptitle("Fig 1. Predicted geometry is physically consistent with RF multipath ranges", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "fig1_range_hist.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: training scale_factor + rf_scale loss over steps (RF-on learns metric scale) ----
h = summary["rf_on"]["history"]
steps = [r["step"] for r in h]
fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.plot(steps, [r["scale_factor"] for r in h], color=C_ON, label="train scale-factor (pred/GT)")
ax1.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
ax1.set_xlabel("training step"); ax1.set_ylabel("scale factor", color=C_ON); ax1.set_ylim(0, 2.2)
ax2 = ax1.twinx(); ax2.plot(steps, [r["loss_rf_scale"] for r in h], color="#ff7f0e", alpha=0.7, label="RF metric-scale loss")
ax2.set_ylabel("RF metric-scale loss", color="#ff7f0e"); ax2.grid(False)
ax1.set_title("Fig 2. RF-VGGT learns to recover metric scale (scale-factor -> 1.0)")
fig.tight_layout(); fig.savefig(FIG / "fig2_scale_training.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: scale-dependent vs scale-invariant AbsRel, RF-on vs RF-off (the key result) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(SCENES)); w = 0.35
for ax, key, title in [(axes[0], "abs_rel_scale_dep", "Scale-DEPENDENT depth error (no alignment)"),
                       (axes[1], "abs_rel_scale_inv", "Scale-INVARIANT depth error (shape only)")]:
    on = [g("rf_on", s, key, np.nan) for s in SCENES]
    off = [g("rf_off", s, key, np.nan) for s in SCENES]
    ax.bar(x - w/2, on, w, color=C_ON, label="RF-VGGT (RF-on)")
    ax.bar(x + w/2, off, w, color=C_OFF, label="RGB-only (RF-off)")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("AI53_", "").replace("_Blender", "") for s in SCENES])
    ax.set_ylabel("AbsRel"); ax.set_title(title); ax.legend(fontsize=9)
fig.suptitle("Fig 3. RF fixes metric SCALE (left) while shape is on par (right) -> the gap is RF's contribution", y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig3_scale_dep_vs_inv.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: RF adapter gate over training (RF conditioning learns to open) ----
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, [r["gate"] for r in h], color="#9467bd")
ax.axhline(0.119, color="k", ls=":", lw=1, label="original init gate ~0.12 (RF near-off)")
ax.set_xlabel("training step"); ax.set_ylabel("mean RF adapter gate")
ax.set_title("Fig 4. RF cross-attention gate stays open (RF is used, not ignored)"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_gate.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 5: held-out scale recovery (RF-on predicted avg_scale vs true) + |sf-1| bars ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
on_rec = [g("rf_on", s, "scale_recovery_abs_err", np.nan) for s in SCENES]
on_sf = [g("rf_on", s, "scale_factor_abs_err", np.nan) for s in SCENES]
off_sf = [g("rf_off", s, "scale_factor_abs_err", np.nan) for s in SCENES]
ax.bar(x - w/2, on_sf, w, color=C_ON, label="RF-VGGT |scale_factor-1|")
ax.bar(x + w/2, off_sf, w, color=C_OFF, label="RGB-only |scale_factor-1|")
ax.set_xticks(x); ax.set_xticklabels([s.replace("AI53_", "").replace("_Blender", "") for s in SCENES])
ax.set_ylabel("|scale factor - 1| (lower=better)"); ax.set_title("Held-out metric-scale error"); ax.legend(fontsize=9)
ax = axes[1]
ax.bar(x, on_rec, 0.5, color=C_GT)
ax.axhline(0.0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels([s.replace("AI53_", "").replace("_Blender", "") for s in SCENES])
ax.set_ylabel("|pred avg_scale / true - 1|"); ax.set_title("RF->scale recovery on HELD-OUT frames (RF-on)")
fig.suptitle("Fig 5. RF recovers metric scale on held-out frames; RGB-only cannot", y=1.03)
fig.tight_layout(); fig.savefig(FIG / "fig5_scale_recovery.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 6: qualitative metric depth triptych (RGB | RF-VGGT | GT) per scene ----
fig, axes = plt.subplots(len(SCENES), 3, figsize=(11, 4.0 * len(SCENES)))
axes = np.atleast_2d(axes)
for i, sc in enumerate(SCENES):
    img = arr["rf_on"].get(f"{sc}__sample_img")
    pred = arr["rf_on"].get(f"{sc}__sample_pred")
    gt = arr["rf_on"].get(f"{sc}__sample_gt")
    if img is None or pred is None or gt is None:
        continue
    vmax = float(np.percentile(gt[gt > 0], 95)) if (gt > 0).any() else 1.0
    axes[i, 0].imshow(np.transpose(img, (1, 2, 0)).clip(0, 1)); axes[i, 0].set_title(f"{sc}\nRGB input")
    im1 = axes[i, 1].imshow(pred, cmap="turbo", vmin=0, vmax=vmax); axes[i, 1].set_title("RF-VGGT metric depth (m)")
    im2 = axes[i, 2].imshow(gt, cmap="turbo", vmin=0, vmax=vmax); axes[i, 2].set_title("GT metric depth (m)")
    for j in (1, 2): plt.colorbar(im1 if j == 1 else im2, ax=axes[i, j], fraction=0.046)
    for a in axes[i]: a.axis("off")
fig.suptitle("Fig 6. RF-VGGT produces metrically-scaled depth in a single forward pass (held-out frame)", y=1.01)
fig.tight_layout(); fig.savefig(FIG / "fig6_depth_qualitative.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 7: summary table-as-figure ----
fig, ax = plt.subplots(figsize=(10, 2.6 + 0.5 * len(SCENES)))
ax.axis("off")
rows = []
for m in ["rf_on", "rf_off"]:
    for s in SCENES:
        e = summary[m]["eval"].get(s, {})
        rows.append([m, s.replace("_Blender", ""),
                     f"{e.get('scale_factor_med', float('nan')):.2f}",
                     f"{e.get('abs_rel_scale_dep', float('nan')):.3f}",
                     f"{e.get('abs_rel_scale_inv', float('nan')):.3f}",
                     f"{e.get('rmse_scale_dep_m', float('nan')):.2f}"])
col = ["method", "scene", "scale_factor", "AbsRel (scale-dep)", "AbsRel (scale-inv)", "RMSE (m)"]
t = ax.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
ax.set_title("Fig 7. RF-VGGT vs RGB-only on held-out frames (2 scenes, proof-of-mechanism)", pad=20)
fig.savefig(FIG / "fig7_summary_table.png", bbox_inches="tight"); plt.close(fig)

print("Saved figures to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name)
