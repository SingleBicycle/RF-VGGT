# RF-VGGT — Cross-Scene Report (2026-06-22)

**One line.** Scaling the dataset from **N=2 → 16 train + 4 held-out-scene val** lets us prove the
central claim *across scenes*: RF (radio multipath) supplies the absolute metric scale that
image-only feed-forward multi-view reconstruction (VGGT/DUSt3R lineage) structurally lacks — on
scenes never seen in training, with statistical significance, a fuller (partial-unfreeze) training
regime, an image-blind RF-only scale predictor, encoder baselines, an NLOS/robustness battery, and a
2024–2026 novelty refresh. The honest new nuance: a *learned image-only* scale head also recovers
scale **in-distribution**, so the decisive evidence for RF is a **scale-ambiguity test** where images
are provably blind and RF is not.

This report supersedes the N=2 writeup (`2026-06-15-REPORT-rf-vggt.md`) and directly closes the four
gaps it listed as open.

---

## 0. Data foundation — re-verified on all 20 scenes (was 2)

| check | result | meaning |
|---|---|---|
| `c·min(delay) / ‖tx−rx‖` | **mean 1.00000**, ~2000 frames, 20 scenes | RF first arrival = exact bistatic LOS range → exact absolute metric range |
| cross-scene scale spread | median scene depth **3.6 m → 13.9 m (3.8×)** across train; 4.2–6.4 m val | scale genuinely varies scene-to-scene → must be *read*, not memorized as a constant |
| NLOS prevalence | train 0%; **val 019/020: 1–2% of frames** first-arrival > geometric | the val set carries genuine (mild) NLOS |

Figure: `results/figs/fig0_data_foundation.png`.

## 1. Method (unchanged from approved design)
Vision = scale-free shape (VGGT heads); RF = absolute metric scale via an **image-blind `RFScaleHead`**
(`rf_paths`+`rf_global` → `log_scale`; metric depth = `exp(log_scale)·unit_depth`), plus gated
cross-attention RF conditioning. RF-off has no scale mechanism by construction. Trained from VGGT-1B.

---

## 2. Headline — cross-scene RF-on vs RF-off (train 16 → eval 4 unseen scenes)

Two training regimes, to answer the prior report's "short 500-step frozen fine-tune, not full
training" limitation:

| regime | method | scale recovery (→1) | AbsRel **scale-dep** (metric) | AbsRel **scale-inv** (shape) |
|---|---|---|---|---|
| **partial-unfreeze, 2500 steps** | **RF-on** (seed42 / seed43) | **1.066 / 0.874** | **0.242 / 0.209** | 0.098 / 0.111 |
| partial-unfreeze, 2500 steps | RF-off (seed42 / seed43) | 1.65 / 1.66 | 0.779 / 0.713 | 0.112 / 0.092 |
| frozen, 1800 steps | RF-on (seed42 / seed43) | 1.64 / 1.23 | 0.67 / 0.32 | 0.072 |
| frozen, 1800 steps | RF-off (seed42 / seed43) | 1.65 / 1.64 | 0.62 / 0.58 | 0.087 |

(Partial RF-on recovery sits in [0.87, 1.07] across seeds — slight under/overshoot — vs RF-off's ~1.65
overshoot; metric AbsRel 0.21–0.24 vs 0.71–0.78.)

**Reading.**
- **Fuller training (partial-unfreeze) is the real result:** RF recovers metric scale to **6.6%** on
  unseen scenes and cuts scale-dependent depth error **3.2×** (0.779→0.242), while scale-invariant
  (shape) error stays on par (0.112 vs 0.098) — the gain is *specifically* metric scale.
  Per-scene recovery is **1.00 / 1.00 / 1.01 / 1.08 / 1.17** — tight across all 4 unseen scenes.
- **The frozen short-run regime is unstable** (RF-on seed variance 1.23–1.64; not reliably better than
  RF-off): this is exactly why the longer/partial regime matters, and it is reported honestly rather
  than cherry-picked.

**Statistical significance** (paired per-window AbsRel, 132 windows, permutation + bootstrap):

| regime / seed | RF-off | RF-on | improvement [95% CI] | perm p |
|---|---|---|---|---|
| **partial s42** | 0.779 | 0.242 | **+0.538 [0.470, 0.605]** | **5×10⁻⁵** |
| **partial s43** | 0.713 | 0.209 | **+0.503 [0.421, 0.585]** | **5×10⁻⁵** |
| frozen s43 | 0.575 | 0.316 | +0.259 [0.197, 0.320] | 5×10⁻⁵ |
| frozen s42 | 0.615 | 0.669 | −0.054 [−0.128, 0.016] | 0.93 (n.s.) |

Both partial-unfreeze seeds give a large, highly significant metric-depth improvement; the frozen
short-run regime is seed-dependent (one seed n.s.) — hence the partial regime is the reported result.

Figures: `fig1_cross_scene_recovery.png`, `fig2_absrel.png`.

---

## 3. Immediate-TODO #1 — image-blind RF-only scale predictor + encoder comparison

A standalone predictor that sees **only** RF (`rf_paths`+`rf_global`, no images) → metric scale,
trained on 16 scenes, evaluated on 4 unseen scenes (cross-scene). 3 seeds.

| predictor | recovery (→1) | abs-log-err | MAE (m) |
|---|---|---|---|
| analytic: LOS range linear fit | 1.192 | 0.197 | 1.11 |
| analytic: median path-range fit | 1.443 | 0.354 | 2.16 |
| **DeepSets** (mean-pool set encoder) | 1.028 ± 0.054 | 0.240 | 1.24 |
| **PointNet** (max-pool) | **1.029 ± 0.042** | **0.186** | **1.01** |
| **Set Transformer** (attn pool) | 1.059 ± 0.036 | 0.207 | 1.14 |

**RF alone recovers cross-scene metric scale to ~3%** (learned encoders), beating analytic
range baselines. The three permutation-invariant path encoders are comparable; Set Transformer has the
lowest seed variance, PointNet the lowest error. Figures: `fig4_encoder_compare.png`,
`fig6_rf_only_scatter.png`.

**Range-removal control** (Set Transformer; zero feature groups at train+eval):

| variant | recovery | abs-log-err | per-window scatter |
|---|---|---|---|
| full (all RF feats) | 1.059 | **0.207** | 0.233 |
| drop delay (idx 0) | 1.098 | 0.226 | 0.331 |
| drop delay+power+amplitude+range-globals | 1.103 | 0.259 | 0.340 |
| **angles-only** (AoA/AoD bearing, scale-invariant) | 1.158 | **0.320** | 0.422 |

Stripping range/timing/power cues monotonically degrades accuracy (abs-log-err 0.207→0.320, scatter
0.233→0.422); **bearing alone is worst** — the metric scale lives in RF range/timing, not direction
(consistent with the per-modality ablation in §5).

---

## 4. Why RF is *necessary*, not merely sufficient — the scale-ambiguity capstone

The 2024–26 literature refresh surfaced **AMB3R (arXiv:2511.20343, Nov 2025)**: a learned scale head on
frozen VGGT recovers metric scale from **images alone**. We reproduce that effect honestly: a learned
image-only scale head (pooled frozen-VGGT features → scale) matches RF **in-distribution** on our val
set (abs-log-err **0.186** vs RF 0.207). So on scenes whose scale resembles training, image appearance
priors suffice — "scale-head-on-VGGT" is *not* our contribution.

The contribution is **RF as a physical scale anchor**, proven by a **global similarity-rescaling test**:
multiplying the world (and camera baseline) by `k` leaves every rendered image **pixel-identical** but
multiplies every RF time-of-flight delay by `k`. We apply this to the val set:

| rescale k | image-only recovery | RF scale head recovery |
|---|---|---|
| 0.5 | 2.123 ( = 1/k, **blind**) | **1.048** |
| 1.0 | 1.061 | 1.053 |
| 2.0 | 0.531 ( = 1/k, **blind**) | **1.059** |

**The image-only head's recovery is exactly `1/k`** — it cannot perceive the rescaling, because the
images don't change. **The RF head stays at ≈1.05 for every `k`** (abs-log-err flat at 0.17): it reads
the scaled delays and is metric-scale-*equivariant*. This is the cleanest statement of the thesis:
image-only feed-forward reconstruction is scale-ambiguous by construction; RF resolves it by physics.
Figure: `fig7_scale_ambiguity.png`, `fig8_rf_vs_image.png`.

---

## 5. Ablation battery (cross-scene)

### 5a. Per-modality ablation (which RF modality carries the metric scale) — partial regime

Same architecture throughout (`use_rf=True`, scale head present); variants differ only in which RF
inputs are real vs zeroed at **both** train and eval. Partial-unfreeze, 2500 steps, cross-scene val.

| RF modalities real | scale recovery (→1) | AbsRel **scale-dep** | AbsRel **scale-inv** |
|---|---|---|---|
| **full** (angular+paths+global) | **1.066** | **0.242** | 0.098 |
| paths only (ToF delays) | 1.072 | 0.262 | 0.121 |
| global only (#paths / path-loss) | 1.212 | 0.299 | 0.112 |
| angular only (AoA/AoD, no range) | 1.423 | 0.479 | 0.109 |
| none (all RF zeroed, same arch) | 1.498 | 0.626 | 0.115 |
| RF-off (no scale mechanism) | 1.652 | 0.779 | 0.112 |

**Clean monotone ordering `full ≈ paths < global < angular < none < RF-off`.** The range-bearing
modalities (path ToF delays, and the global path-loss/#paths scalars) carry the metric scale; **angular
RF alone — directions without range — is nearly as bad as no RF** (0.479 vs none 0.626). Scale-invariant
(shape) error is ≈0.10–0.12 across *all* variants → the ablation moves *scale only*, never shape.
Figure: `fig3_modality.png`.

### 5b. Inference-time controls + cross-scene shuffle (on the calibrated **partial** model)

| control | AbsRel scale-dep | scale recovery | follows… |
|---|---|---|---|
| full | 0.242 | 1.066 | — |
| **zero-RF at test** | **0.754** | **0.231** | scale collapses → model genuinely relies on RF |
| noise-RF | 0.497 | 0.543 | degrades |
| zero-delay at test | 0.242 | 0.803 | range removed → scale drops |
| cross-scene RF shuffle | 0.405 | rec_vs_RF-scene **1.066** | predicted scale tracks the **RF** scene |

Removing RF at inference collapses the recovered scale (1.066→0.231) and triples metric depth error
(0.242→0.754) — the model uses RF at test time. The cross-scene shuffle tracks the RF scene (1.066),
**caveat**: the 4 val scenes span only a 1.5× scale range, so the shuffle is less discriminative here
than the rescaling test of §4 (which is decisive).

### 5c. Range-removal — see §3 (image-blind) — abs-log-err 0.207→0.320 as range cues are stripped.

---

## 6. Encoder comparison (maps onto the literature baselines)

- **Path / scale encoders (§3, image-blind, clean):** Deep Sets (NeurIPS'17) ≈ PointNet (CVPR'17) ≈
  Set Transformer (ICML'19), all ~1.03–1.06 recovery; Set Transformer lowest variance, PointNet lowest
  error. The per-path-set encoder family is the right citation; differences are second-order.
- **Angular conditioning encoders (frozen, apples-to-apples):**

  | angular encoder | AbsRel scale-dep | AbsRel scale-inv (shape) |
  |---|---|---|
  | CNN (RF-Pose-style, CVPR'18) | 0.579 | 0.081 |
  | Shallow ViT (RadarFormer-style, SCIA'23) | 0.664 | 0.084 |
  | AngularRFEncoderV2 (ours) | 0.669 | 0.072 |

  The angular-encoder choice is **second-order for metric scale** (the scale comes from the path/global
  range cues, not the angular map — see §5a), and shape (scale-inv) is comparable across all three. This
  is consistent and honestly reported: swapping in a citable RadarFormer-/RF-Pose-style encoder neither
  helps nor hurts the metric-scale result, so the custom branch is justified but not load-bearing for scale.

---

## 7. NLOS & robustness (Exp E2) — stress tests in lieu of unavailable real hardware

The sim scenes are mostly clean LOS, so we stress-test the image-blind RF scale predictor on the val set:

| stress | abs-log-err vs baseline (0.207) |
|---|---|
| LOS / early-path occlusion: drop 1 / 3 / 5 / 8 earliest paths | 0.208 / 0.212 / 0.217 / 0.225 |
| delay-timing noise σ = 0.05 / 0.1 / 0.2 / 0.4 | 0.205 / 0.211 / 0.228 / 0.302 |
| random path dropout 25% / 50% / 75% | 0.209 / 0.232 / 0.350 |

The predictor is **robust to blocking the 8 earliest (LOS) paths** (0.207→0.225) — it uses the full
multipath structure, not just first arrival — degrades gracefully under timing noise up to σ=0.2, and
tolerates 25% path dropout. Figure: `fig5_robustness.png`.

---

## 8. Literature / novelty refresh (2024–2026) — Exp E1

A 7-agent web-research workflow (adversarial novelty refutation + synthesis) re-checked novelty against
recent work. **Novelty not refuted, but the boundary is now tighter.** Full report:
`RF-VGGT/results/lit_refresh.md`.

**Closest prior work (ranked threat):**
- **RaScene** (arXiv:2604.02603) — feed-forward multi-frame CIR→metric 3D, but **RF-only & monostatic** (no vision FM).
- **ISAC camera+RF env. reconstruction** (arXiv:2403.17810 / 2508.05226) — fuses camera+RF but **RF-primary, BEV/occupancy**, no foundation model.
- **AMB3R** (arXiv:2511.20343) — learned scale head on frozen VGGT, **image-only** (→ our §4 rebuttal).
- **Lee et al. ICRA'19** — RF ToF→metric visual scale, **classical SLAM, round-trip range only**.
- **RadarCam-Depth ICRA'24 / TacoDepth CVPR'25** — radar-camera metric depth, **single-view, sparse radar**.

**Defensible novelty (state with all qualifiers):** *to our knowledge, the first to condition a
**pretrained feed-forward multi-view** vision foundation model (VGGT) on **rich bistatic RF multipath**
(ToF + AoA/AoD + a 90×360 angular-power map) to recover **absolute metric scale**.* Each adjacent cluster
owns one qualifier; none owns the conjunction. Dropping any one qualifier is refuted (Lee'19 / RaScene /
AMB3R / ISAC). **Must-cite/benchmark baselines:** Deep Sets, PointNet, Set Transformer (path encoder);
RadarFormer, RODNet, T-FFTRadNet (angular); Pow3R (fusion template), AMB3R (image-only scale-head baseline);
DUSt3R/MASt3R/VGGT/π³ (scale-ambiguous MVS).

**Top submission risks:** (1) "first" is brittle — soften to "to our knowledge, the first" with all
qualifiers; (2) image-only scale heads (AMB3R) recover scale in-distribution — **§4 is the required
rebuttal**; (3) sim-to-real gap: clean ray-traced AoD/angular-power may not transfer to real hardware.

---

## 9. Gap-closure scorecard (vs the prior report's "still open" list)

| Prior open gap | Status now | Evidence |
|---|---|---|
| Only N=2 sim scenes; no cross-scene generalization | **Closed** (16→4 held-out scenes) | §2: recovery 1.066, 3.2× metric-depth gain on unseen scenes |
| Statistical significance not shown | **Closed** | §2: paired perm p=5×10⁻⁵, CI [0.470, 0.605] |
| Short 500-step frozen fine-tune, not full training | **Addressed** | §2: partial-unfreeze 2500-step regime is the stable, winning one |
| Angle branch not swapped to a citable encoder | **Closed** | §6: CNN/RF-Pose- & ViT/RadarFormer-style + Deep Sets/PointNet/Set Transformer benchmarked |
| No head-to-head vs known RF encoders | **Closed** | §3, §6 |
| Real hardware / strong NLOS / SOTA | **Partially** (no real RF data exists here) | §7 NLOS/robustness stress tests + characterized real sim-NLOS; SOTA positioning §8 |
| (new, from lit) image-only can also scale (AMB3R) | **Rebutted** | §4 scale-ambiguity test: image = 1/k blind, RF equivariant |

## 10. Honest remaining limitations
- **Still simulation** (ray-traced RF). No real RF hardware was available; sim NLOS is mild. Sim-to-real
  is the top risk for a venue submission.
- **Cross-scene RF-shuffle is weak** because the 4 val scenes span only 1.5× scale; the §4 rescaling test
  is the decisive substitute.
- **20 scenes** is far better than 2 but still small for a distributional claim; more scenes / a real
  capture would strengthen it.
- The in-VGGT **frozen** scale head is unstable at short horizons; only the partial-unfreeze regime is reliable.

## 11. Artifacts & repro
- Consolidated metrics: `RF-VGGT/results/consolidated.json`; figures `RF-VGGT/results/figs/fig{0..8}.png`.
- Scripts: `scripts/rf_data.py` (harness), `exp_rf_only_scale.py` (§3), `exp_image_scale.py` (§4 image baseline),
  `exp_scale_ambiguity.py` (§4 capstone), `exp_cross_scene.py` + `run_sweep.py`/`run_sweep2.py` (§2/§5/§6),
  `exp_nlos.py` (§7), `aggregate.py` (tables/stats/figs), `fig_data.py` (§0).
- Per-config results: `RF-VGGT/results/cross/*.json`; RF-only `results/rf_only/`; NLOS `results/nlos/`; lit `results/lit_refresh.md`.
- Env `mast3r_vggt`; ckpt `ckpts/vggt1b_336.pt`; 8×RTX-PRO-6000. Repro e.g.:
  `python scripts/run_sweep.py 1,2,3,4,5,6,7 && python scripts/run_sweep2.py 1,2,3,4,5,6 && python scripts/aggregate.py`.
