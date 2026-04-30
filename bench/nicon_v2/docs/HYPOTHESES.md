# nicon_v2 — Falsifiable Hypotheses (after Codex round 1)

Every hypothesis is uniformly structured so it can be unambiguously **accepted**, **rejected**, or **inconclusive** from a single benchmark run. Hypotheses are landed in `docs/IMPLEMENTATION_LOG.md` once their cohort row is in `benchmark_runs/`.

## Common decision rule

For a hypothesis with primary metric `M`, control variant `C` and treatment variant `T` evaluated on a cohort of `D` datasets at seeds `S`:

* **Effect.** `Δ_med = median over (dataset, seed) of (M_T − M_C) / M_C`. The threshold `θ` is given per-hypothesis below.
* **Significance.** Paired Wilcoxon signed-rank test on the `D × |S|` paired differences `M_T − M_C`. Two-sided. We require `p < 0.05`. (When `D × |S| < 8`, e.g. smoke runs, we report the test result but exempt the run from the significance gate.)
* **Per-dataset safety.** No single dataset regresses by > 5 % rmsep vs the strongest accepted prior variant *and* no dataset regresses by > 10 % vs any cohort reference (PLS / paper-Ridge / TabPFN / CNN).
* **Multiple-comparison correction.** When several hypotheses are tested in the same phase (e.g. H6 + H7 in Phase 1b) we apply Holm correction across them; we declare a hypothesis ACCEPTED only if its corrected `p < 0.05`.

A hypothesis is **ACCEPTED** iff `Δ_med ≤ −θ`, the corrected p < 0.05, and the safety gate holds. **REJECTED** iff `Δ_med > −θ` or any safety gate fails. **INCONCLUSIVE** iff the smoke-run `n` is too small.

## H1 — Linear regression head [W1, severity C]

* **Treatment.** NICON-baseline with the final `Dense(1, sigmoid)` replaced by `Dense(1, identity)`.
* **Control.** NICON-baseline (upstream).
* **Metric / threshold.** rmsep, `θ = 0.03` (3 % median improvement).
* **Cohort.** Curated 39 datasets × seeds {0, 1, 2}.
* **Reject if.** `Δ_med > −0.03` or paired Wilcoxon p ≥ 0.05.

## H2 — Single-activation backbone [W2]

* **Treatment.** H1 + GELU + LayerNorm throughout (no SELU, no AlphaDropout, no BatchNorm).
* **Control.** H1.
* **Metric / threshold.** rmsep, `θ = 0.02`.
* **Reject if.** `Δ_med > −0.02` or p ≥ 0.05.

## H3 — Small-kernel + GAP backbone (Cui-Fearn 2018) [W3, W4]

* **Treatment.** 4-block backbone with kernels `(7, 5, 3, 3)`, channels `(16, 32, 64, 128)`, max-pool 2 + GELU + LayerNorm + spatial-dropout 0.2 + GAP head + linear projection.
* **Control.** Phase 1b accepted variant (H1 + H2 + H5 + H6 + H7 — concat-deriv + Bjerrum + C-Mixup on the original NICON backbone).
* **Metric / threshold.** rmsep, `θ = 0.05`.
* **Reject if.** `Δ_med > −0.05`, p ≥ 0.05, or any short-spectrum dataset (DIESEL/Beer/CORN) regresses > 10 %.

## H4 — Norm robustness vs batch size [W8]

* **Treatment.** Same backbone as H3, swapping LayerNorm ↔ BatchNorm ↔ GroupNorm at batch sizes {8, 16, 32, 64}.
* **Control.** LayerNorm at the same batch.
* **Metric / threshold.** rmsep; **claim**: at batch ≤ 16, LayerNorm wins by `θ = 0.02`; at batch ≥ 32, all three within `0.02`.
* **Reject if.** BatchNorm wins on ≥ 60 % of datasets at batch = 16 by > 2 %.

## H5 — Concat-derivatives input (Mishra/Passos 2022) [W5]

* **Treatment.** Front-end concatenates `[raw, 1st-SG, 2nd-SG]` as 3 channels (window 11, polynomial order 2).
* **Control.** Phase 1a accepted variant (single channel raw).
* **Metric / threshold.** rmsep, `θ = 0.05`.
* **Reject if.** `Δ_med > −0.02`, p ≥ 0.05, or DIESEL/COLZA/AMYLOSE do not improve.

## H6 — Bjerrum-style EMSC-parameter augmentation [W6]

* **Treatment.** Per-batch random offset (U[−σ_u, σ_u]), slope (U[−σ_s, σ_s]·wavelength), multiplicative (U[1−σ_m, 1+σ_m]), with amplitudes scaled to `0.05 · range(x_train)` for offset/slope and `0.05` for the multiplicative factor.
* **Control.** Phase 1a + H5 (concat-deriv only).
* **Metric / threshold.** rmsep, `θ = 0.04`.
* **Reject if.** `Δ_med > −0.02` or p ≥ 0.05.

## H7 — C-Mixup label-aware mixing (Yao 2022) [W6]

* **Treatment.** C-Mixup with fold-locally-tuned `σ_y` (grid `{0.05, 0.1, 0.2, 0.5, 1.0} · std(y_A)`).
* **Control A.** Phase 1a + H5 + H6 (no mixup).
* **Control B.** Same + vanilla mixup (uniform `j`).
* **Metric / threshold.** rmsep, `θ = 0.02` (vs Control A); C-Mixup beats vanilla mixup on ≥ 60 % of datasets.
* **Reject if.** `Δ_med > 0` against Control A, *or* C-Mixup wins ≤ 50 % vs vanilla mixup.

## H8 — Multi-scale Inception block (Zhang 2019) [W7]

* **Treatment.** Phase-1c backbone + 1 Inception block (parallel 1, 3, 5 + pool branches at 64 ch) before GAP.
* **Control.** Phase-1c accepted variant.
* **Metric / threshold.** rmsep on `n_train > 500` sub-cohort, `θ = 0.03`; on `n_train ≤ 500` sub-cohort no significant regression (Wilcoxon p > 0.05 *or* `Δ_med ≥ −0.01`).
* **Reject if.** `Δ_med > −0.03` on the large sub-cohort, *or* the small sub-cohort regresses significantly.

## H9 — Deep ensembles (RMSEP) and conformal UQ (calibration) — split [W9, W15]

* **H9-acc (Phase 3a).** Treatment = 5-seed ensemble of Phase-1c best. Control = single net at seed 0. Metric = rmsep, `θ = 0.02`.
* **H9-uq (Phase 3b).** Treatment = split-conformal calibration on `C` (20 % of train). Metrics:
  * `coverage_90 ∈ [0.85, 0.95]` on ≥ 80 % of cohort datasets;
  * `interval_score_90` improves vs a fixed-σ Gaussian baseline by `θ_is = 0.05` (median across cohort);
  * no rmsep regression vs H9-acc.
* **Reject H9-uq if.** Coverage outside `[0.85, 0.95]` on > 20 % of cohort or interval_score median worse.

## H10 — Learnable EMSC / SG (Helin 2022) [W10]

* **Treatment.** Learnable EMSC layer (reference vector + Vandermonde basis) + learnable SG window (constrained Conv1D with smoothness penalty), replacing the fixed concat-deriv front.
* **Control.** Phase-1c accepted variant (deterministic concat-deriv).
* **Metric / threshold.** rmsep, equivalence band `|Δ_med| ≤ 0.02`; *and* per-dataset regressions ≤ 2 %.
* **Reject if.** Median rmsep > control by ≥ 0.02 *or* > 30 % of datasets regress > 2 %.

## H11 — TabPFN-as-head [open]

* **Treatment.** Phase-1c backbone, GAP features fed to TabPFN-v2 instead of dense head.
* **Control.** Phase-3a (5-ensemble Phase-1c).
* **Metric.** rmsep on `n_train ≤ 200` sub-cohort, `θ = 0.01`.
* **Reject if.** `Δ_med > 0` or any dataset regresses > 5 %.

## H12 — Hybrid stack (nicon_v2 → AOM-PLS meta) [open]

* **Treatment.** OOF predictions from nicon_v2-best stacked with AOM-PLS-best via Ridge meta.
* **Control A.** nicon_v2-best (alone).
* **Control B.** AOM-PLS-best (alone).
* **Metric.** rmsep across the curated cohort, `θ = 0.01` vs the better of A, B.
* **Reject if.** `Δ_med > −0.01` against the better parent.

## Stopping criterion (Prompt.md restated)

Loop terminates when **median rmsep of nicon_v2-best vs `aom_ridge_curated_best` ≤ −0.02 on the 39-dataset curated cohort**, paired Wilcoxon p < 0.05, ≥ 50 % wins per dataset, and no dataset regresses > 10 % vs any cohort reference. Equivalent or stricter criteria for classification will be defined when classification iterations land. A **scientific-success fallback** (`nicon_v2-best > NICON-baseline ∧ DECON-baseline` on ≥ 75 % of cohort, p < 0.05) is also reported in the manuscript even if the leaderboard gate is not cleared.
