"""Aggregate all cross-scene results into tables, paired significance tests, and figures."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
CROSS = REPO / "results" / "cross"
FIGS = REPO / "results" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)


def load(tag):
    p = CROSS / f"{tag}.json"
    return json.load(open(p)) if p.exists() else None


def rows(tag):
    p = CROSS / f"{tag}_arrays.npz"
    if not p.exists():
        return []
    return json.loads(str(np.load(p, allow_pickle=True)["rows"]))


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    bs = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_test(a, b, n=20000, seed=0):
    """Paired permutation test on mean(a-b) (a=rf_off err, b=rf_on err): is improvement>0 significant?"""
    rng = np.random.default_rng(seed); d = np.asarray(a) - np.asarray(b)
    obs = d.mean(); cnt = 0
    for _ in range(n):
        s = rng.choice([1, -1], size=len(d))
        if (s * d).mean() >= obs:
            cnt += 1
    return float(obs), float((cnt + 1) / (n + 1))


def fmt(v, f="{:.3f}"):
    return f.format(v) if v is not None else "n/a"


def main():
    out = {}
    # ---- Exp A: main cross-scene ----
    print("\n================ Exp A: cross-scene RF-on vs RF-off (16 train -> 4 unseen val) ================")
    print(f"{'config':24s} {'recovery':>12s} {'AbsRel_sd':>10s} {'AbsRel_si':>10s} {'scale_fac':>9s}")
    A = {}
    for tag in ["A_rfon_frozen_s42", "A_rfon_frozen_s43", "A_rfoff_frozen_s42", "A_rfoff_frozen_s43",
                "A_rfon_partial_s42", "A_rfoff_partial_s42"]:
        d = load(tag)
        if not d: continue
        e = d["eval"]; A[tag] = e
        rec = f"{e.get('recovery_mean',0):.3f}±{e.get('recovery_std',0):.3f}" if "recovery_mean" in e else "n/a(const)"
        print(f"{tag:24s} {rec:>12s} {fmt(e.get('absrel_sd_mean')):>10s} {fmt(e.get('absrel_si_mean')):>10s} {fmt(e.get('scale_factor_med')):>9s}")
    out["expA"] = A

    # ---- per-scene headline ----
    don, doff = load("A_rfon_frozen_s42"), load("A_rfoff_frozen_s42")
    if don and doff:
        print("\n-- per val scene (frozen, seed42): recovery | AbsRel_sd  [RF-on vs RF-off] --")
        for sc in sorted(don["eval"]["per_scene"]):
            r1 = don["eval"]["per_scene"][sc]; r0 = doff["eval"]["per_scene"][sc]
            print(f"  {sc:20s} RF-on rec={r1['recovery_mean']:.3f} sd={r1['absrel_sd']:.3f} | RF-off sd={r0['absrel_sd']:.3f}")

    # ---- statistical significance: paired per-window AbsRel_sd, rf_on vs rf_off ----
    print("\n================ Statistical significance (paired per-window AbsRel_sd) ================")
    for reg in ["frozen", "partial"]:
        for seed in ["s42", "s43"]:
            ron, roff = rows(f"A_rfon_{reg}_{seed}"), rows(f"A_rfoff_{reg}_{seed}")
            if ron and roff and len(ron) == len(roff):
                on = np.array([r["absrel_sd"] for r in ron]); off = np.array([r["absrel_sd"] for r in roff])
                obs, p = perm_test(off, on); m, lo, hi = boot_ci(off - on)
                print(f"  {reg:7s} {seed}: n={len(on)}  RF-off={off.mean():.3f}  RF-on={on.mean():.3f}  "
                      f"improvement={m:.3f} [95% CI {lo:.3f},{hi:.3f}]  perm p={p:.2e}")
                out[f"sig_{reg}_{seed}"] = dict(n=len(on), rfoff=float(off.mean()), rfon=float(on.mean()),
                                                improvement=m, ci=[lo, hi], p=p)

    # ---- Exp C: modality ablation (prefer partial-mode C2_* if present, else frozen C_*) ----
    partial_mods = all((CROSS / f"C2_{m}.json").exists() for m in ["paths", "global", "angular", "none"])
    reg = "partial" if partial_mods else "frozen"
    pre = "C2_" if partial_mods else "C_"
    full_tag = f"A_rfon_{reg}_s42"; off_tag = f"A_rfoff_{reg}_s42"
    print(f"\n================ Exp C: per-modality ablation (cross-scene, {reg} regime) ================")
    print(f"{'modality':16s} {'recovery':>10s} {'AbsRel_sd':>10s} {'AbsRel_si':>10s}")
    C = {}
    mods = [("full", full_tag), ("paths", f"{pre}paths"), ("global", f"{pre}global"),
            ("angular", f"{pre}angular"), ("none", f"{pre}none"), ("RF-off", off_tag)]
    for name, tag in mods:
        d = load(tag)
        if not d: continue
        e = d["eval"]; C[name] = e
        print(f"{name:16s} {fmt(e.get('recovery_mean')):>10s} {fmt(e.get('absrel_sd_mean')):>10s} {fmt(e.get('absrel_si_mean')):>10s}")
    out["expC_modality"] = C; out["expC_modality_regime"] = reg

    # ---- Controls (from the well-calibrated PARTIAL model) ----
    d = load("A_rfon_partial_s42")
    if d and "controls" in d:
        print("\n================ Exp C: inference-time controls + cross-scene shuffle (trained full model) ================")
        print(f"{'control':12s} {'AbsRel_sd':>10s} {'recovery':>10s} {'rec_vs_RFscene':>14s}")
        full = d["eval"]
        print(f"{'full':12s} {fmt(full.get('absrel_sd_mean')):>10s} {fmt(full.get('recovery_mean')):>10s} {'-':>14s}")
        for k, v in d["controls"].items():
            print(f"{k:12s} {fmt(v.get('absrel_sd_mean')):>10s} {fmt(v.get('recovery_mean')):>10s} {fmt(v.get('recovery_rf_mean')):>14s}")
        out["expC_controls"] = {"full": full, **d["controls"]}

    # ---- Exp D: angular encoder ----
    print("\n================ Exp D: angular conditioning encoder ================")
    print(f"{'encoder':16s} {'recovery':>10s} {'AbsRel_sd':>10s} {'AbsRel_si':>10s}")
    Dd = {}
    for name, tag in [("AngularRFEncoderV2", "A_rfon_frozen_s42"), ("CNN (RF-Pose-style)", "D_cnn"),
                      ("ShallowViT (RadarFormer-style)", "D_vit")]:
        d = load(tag)
        if not d: continue
        e = d["eval"]; Dd[name] = e
        print(f"{name:30s} {fmt(e.get('recovery_mean')):>10s} {fmt(e.get('absrel_sd_mean')):>10s} {fmt(e.get('absrel_si_mean')):>10s}")
    out["expD_encoder"] = Dd

    # ---- Exp B / NLOS passthrough ----
    rfo = REPO / "results" / "rf_only" / "summary.json"
    if rfo.exists(): out["expB"] = json.load(open(rfo))
    amb = REPO / "results" / "rf_only" / "scale_ambiguity.json"
    if amb.exists(): out["expB3_scale_ambiguity"] = json.load(open(amb))
    nl = REPO / "results" / "nlos" / "summary.json"
    if nl.exists(): out["expE_nlos"] = json.load(open(nl))

    json.dump(out, open(REPO / "results" / "consolidated.json", "w"), indent=2)
    print(f"\nsaved -> results/consolidated.json")
    make_figs(out)


def make_figs(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: per-scene scale recovery RF-on vs RF-off
    don, doff = load("A_rfon_frozen_s42"), load("A_rfoff_frozen_s42")
    if don:
        scs = sorted(don["eval"]["per_scene"])
        on = [don["eval"]["per_scene"][s]["recovery_mean"] for s in scs]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(scs))
        ax.bar(x, on, 0.5, label="RF-on (RF-predicted scale)", color="#2a7")
        ax.axhline(1.0, ls="--", c="k", lw=1, label="perfect (=1.0)")
        ax.set_xticks(x); ax.set_xticklabels([s.replace("AI53_", "").replace("_Blender", "") for s in scs])
        ax.set_ylabel("scale recovery (pred/GT → 1.0)"); ax.set_title("Cross-scene metric-scale recovery on 4 unseen scenes")
        ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig1_cross_scene_recovery.png", dpi=130); plt.close(fig)

    # Fig 2: AbsRel scale-dep vs scale-inv, RF-on vs RF-off
    if don and doff:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = ["AbsRel scale-dep\n(metric)", "AbsRel scale-inv\n(shape)"]
        on = [don["eval"]["absrel_sd_mean"], don["eval"]["absrel_si_mean"]]
        off = [doff["eval"]["absrel_sd_mean"], doff["eval"]["absrel_si_mean"]]
        x = np.arange(2); w = 0.35
        ax.bar(x - w/2, off, w, label="RF-off (RGB only)", color="#c44")
        ax.bar(x + w/2, on, w, label="RF-on (RF-VGGT)", color="#2a7")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("AbsRel (lower=better)")
        ax.set_title("RF fixes metric scale; shape stays on par"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / "fig2_absrel.png", dpi=130); plt.close(fig)

    # Fig 3: modality ablation
    C = out.get("expC_modality", {})
    if C:
        names = [n for n in ["full", "paths", "global", "angular", "none", "RF-off"] if n in C]
        sd = [C[n].get("absrel_sd_mean") for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, sd, color=["#2a7", "#5a9", "#7b8", "#d83", "#c44", "#a44"])
        ax.set_ylabel("AbsRel scale-dep"); ax.set_title("Per-modality ablation: range-bearing RF carries the metric scale")
        fig.tight_layout(); fig.savefig(FIGS / "fig3_modality.png", dpi=130); plt.close(fig)

    # Fig 4: Exp B encoder comparison (recovery + abs_log_err)
    B = out.get("expB", {})
    if B:
        names = ["analytic_los", "analytic_median_range", "deepsets", "pointnet", "settransformer"]
        names = [n for n in names if n in B]
        rec = [B[n]["recovery_mean"] for n in names]
        ale = [B[n]["abs_log_err"] for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(names, rec, color="#48a"); ax[0].axhline(1.0, ls="--", c="k", lw=1)
        ax[0].set_ylabel("scale recovery → 1.0"); ax[0].set_title("Image-blind RF scale recovery"); ax[0].tick_params(axis="x", rotation=30)
        ax[1].bar(names, ale, color="#a64"); ax[1].set_ylabel("|log scale| error (lower=better)")
        ax[1].set_title("RF-only scale accuracy"); ax[1].tick_params(axis="x", rotation=30)
        fig.tight_layout(); fig.savefig(FIGS / "fig4_encoder_compare.png", dpi=130); plt.close(fig)

    # Fig 7: scale-ambiguity capstone (RF vs image-only under similarity rescaling)
    SA = out.get("expB3_scale_ambiguity", {})
    if SA:
        ks = [r["k"] for r in SA["k_sweep"]]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(ks, [r["rf_aug_recovery"] for r in SA["k_sweep"]], "o-", color="#2a7", label="RF scale head (physical ToF)")
        ax.plot(ks, [r["img_recovery"] for r in SA["k_sweep"]], "s--", color="#c44", label="image-only scale head")
        ax.plot(ks, [1.0 / k for k in ks], ":", color="#888", label="image limit = 1/k (scale-blind)")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("similarity rescaling factor k  (images are pixel-identical ∀k)")
        ax.set_ylabel("metric-scale recovery → 1.0")
        ax.set_title("Scale-ambiguity test: images can't see k, RF can")
        ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig7_scale_ambiguity.png", dpi=130); plt.close(fig)

    # Fig 8: RF-only vs image-only scale head (in-distribution)
    B = out.get("expB", {})
    if B and "image_only_scale_head" in B:
        names = ["analytic_los", "deepsets", "pointnet", "settransformer", "image_only_scale_head"]
        names = [n for n in names if n in B]
        labels = {"analytic_los": "analytic(LOS)", "deepsets": "DeepSets", "pointnet": "PointNet",
                  "settransformer": "SetTransf.", "image_only_scale_head": "IMAGE-only"}
        ale = [B[n]["abs_log_err"] for n in names]
        cols = ["#aaa", "#48a", "#48a", "#48a", "#c44"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar([labels[n] for n in names], ale, color=cols)
        ax.set_ylabel("|log scale| error (cross-scene)")
        ax.set_title("Image-only matches RF in-distribution — but fails under scale shift (see fig7)")
        fig.tight_layout(); fig.savefig(FIGS / "fig8_rf_vs_image.png", dpi=130); plt.close(fig)

    # Fig 5: NLOS / robustness curves
    N = out.get("expE_nlos", {})
    if N:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        lo = N["los_occlusion"]
        ax[0].errorbar([0] + [r["paths_removed"] for r in lo],
                       [N["baseline"]["abs_log_err"]] + [r["abs_log_err"] for r in lo], marker="o")
        ax[0].set_xlabel("# earliest paths removed (LOS blockage)"); ax[0].set_ylabel("|log scale| err"); ax[0].set_title("NLOS / LOS-occlusion stress")
        dn = N["delay_noise"]
        ax[1].errorbar([r["sigma"] for r in dn], [r["abs_log_err"] for r in dn], marker="o", color="#a64")
        ax[1].set_xlabel("delay-feature noise σ"); ax[1].set_ylabel("|log scale| err"); ax[1].set_title("Timing-noise robustness")
        pd = N["path_dropout"]
        ax[2].errorbar([r["drop_frac"] for r in pd], [r["abs_log_err"] for r in pd], marker="o", color="#4a8")
        ax[2].set_xlabel("path dropout fraction"); ax[2].set_ylabel("|log scale| err"); ax[2].set_title("Sparse-measurement robustness")
        fig.tight_layout(); fig.savefig(FIGS / "fig5_robustness.png", dpi=130); plt.close(fig)

    print(f"figures -> {FIGS}/")


if __name__ == "__main__":
    main()
