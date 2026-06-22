"""Figures for the RF-contribution validation (per-modality ablation, shuffle, robustness)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT/results/validate")
FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
R = json.load(open(OUT / "summary.json"))
SCENES = [s for s in ["AI53_001_Blender", "AI53_002_Blender"] if any(s in R["modality"].get(v, {}) for v in R["modality"])]
short = lambda s: s.replace("AI53_", "").replace("_Blender", "")
VARS = [v for v in ["full", "paths_only", "global_only", "angular_only", "none"] if v in R["modality"]]
x = np.arange(len(SCENES)); w = 0.8 / max(len(VARS), 1)
colors = {"full": "#1f77b4", "paths_only": "#2ca02c", "global_only": "#9467bd", "angular_only": "#ff7f0e", "none": "#d62728"}

# Fig A: per-modality |scale_factor - 1| and AbsRel scale-dependent
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, key, title, fn in [(axes[0], "scale_factor", "Metric scale error  |scale_factor - 1|", lambda r: abs(r["scale_factor"] - 1)),
                           (axes[1], "abs_rel_sd", "Depth error (scale-dependent AbsRel)", lambda r: r["abs_rel_sd"])]:
    for i, v in enumerate(VARS):
        vals = [fn(R["modality"][v][s]) for s in SCENES]
        ax.bar(x + (i - (len(VARS)-1)/2) * w, vals, w, color=colors.get(v), label=v)
    ax.set_xticks(x); ax.set_xticklabels([short(s) for s in SCENES]); ax.set_title(title); ax.legend(fontsize=8)
fig.suptitle("Per-modality ablation: PATHS (ToF) drives metric scale; angular/none cannot recover it", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "valA_modality.png", bbox_inches="tight"); plt.close(fig)

# Fig B: cross-scene RF shuffle — predicted scale follows the RF scene, not the image scene
if R.get("shuffle"):
    pairs = list(R["shuffle"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xp = np.arange(len(pairs))
    over_img = [R["shuffle"][p]["pred_over_imageScene"] for p in pairs]
    over_rf = [R["shuffle"][p]["pred_over_rfScene"] for p in pairs]
    ax.bar(xp - 0.2, over_img, 0.4, color="#d62728", label="pred scale / IMAGE-scene (should be off)")
    ax.bar(xp + 0.2, over_rf, 0.4, color="#2ca02c", label="pred scale / RF-scene (should be ~1)")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(xp); ax.set_xticklabels([p.replace("AI53_", "").replace("_Blender", "").replace("img", "img:").replace("_rf", "  rf:") for p in pairs], fontsize=8)
    ax.set_ylabel("ratio"); ax.set_title("Cross-scene RF shuffle: predicted scale tracks the RF, not the image\n(proves the scale head reads RF content, not image cues)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "valB_shuffle.png", bbox_inches="tight"); plt.close(fig)

# Fig C: inference-time robustness controls on the trained FULL model
if R.get("controls"):
    ctrl = ["full", "zero_rf", "noise_rf"]
    cc = {"full": "#1f77b4", "zero_rf": "#d62728", "noise_rf": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, title, fn in [(axes[0], "sf", "|scale_factor - 1|", lambda r: abs(r["scale_factor"] - 1)),
                               (axes[1], "sd", "AbsRel (scale-dependent)", lambda r: r["abs_rel_sd"])]:
        for i, c in enumerate(ctrl):
            if c not in R["controls"]: continue
            vals = [fn(R["controls"][c][s]) for s in SCENES]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=cc[c], label=c)
        ax.set_xticks(x); ax.set_xticklabels([short(s) for s in SCENES]); ax.set_title(title); ax.legend(fontsize=9)
    fig.suptitle("Inference-time controls on trained model: removing/corrupting RF destroys metric scale", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "valC_controls.png", bbox_inches="tight"); plt.close(fig)

# Fig D: RF->scale recovery per modality (held-out)
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, v in enumerate(VARS):
    vals = [R["modality"][v][s]["scale_recovery"] for s in SCENES]
    ax.bar(x + (i - (len(VARS)-1)/2) * w, vals, w, color=colors.get(v), label=v)
ax.axhline(1.0, color="k", ls="--", lw=1, label="perfect recovery")
ax.set_xticks(x); ax.set_xticklabels([short(s) for s in SCENES])
ax.set_ylabel("pred avg_scale / true"); ax.set_title("Held-out RF->scale recovery by modality (1.0 = perfect)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "valD_recovery.png", bbox_inches="tight"); plt.close(fig)

print("Saved validation figures to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name)
