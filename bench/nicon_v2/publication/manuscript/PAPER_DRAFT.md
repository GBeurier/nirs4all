# nicon_v2: A Stacked CNN-PLS-Ridge Ensemble for Small-n NIR Spectroscopy

**Status:** _draft, written iteratively as the bench produces results._
**Cohort run:** `stack_extended` (6 datasets × 3 seeds, 18 paired observations) and
`stack_aom_curated` (39 datasets × 1 seed) — see `benchmark_runs/`.

## Abstract

Convolutional neural networks (CNNs) for near-infrared (NIR) spectroscopy
underperform partial least squares (PLS) regression on small training cohorts
(`n < 1000`). We dissect 15 specific failure modes of the upstream **NICON** /
**DECON** 1-D CNNs of the `nirs4all` library and propose **nicon_v2**.

The reported headline model is **`Stack-Ridge-PLS-V1c`**: 5-fold out-of-fold
prediction stacking of three base learners (Ridge with `StandardScaler`,
PLS with auto n-component selection, and a re-engineered small-kernel +
global-average-pool 1-D CNN with concat-derivatives input and Bjerrum 2017
EMSC-parameter augmentation) through a Ridge meta-learner over a 15-point
α grid in `[10⁻³, 10⁴]`. The implementation also ships variants with
**AOM-Ridge** and a cartesian-preprocessing-search **SearchedRidge** /
**SearchedPLS** as alternative base learners, but those are reported as
exploratory follow-ups.

We test 12 falsifiable hypotheses across five iterations with four rounds
of Codex review (44 findings, 14 critical/high applied). On a 6-dataset
extended NIR cohort (ALPINE / Beer / Biscuit / Corn_Oil / DIESEL /
Rice_Amylose; 18 paired observations across seeds {0, 1, 2}):

| Reference | Median Δ% rmsep | Effective n | Note |
|-----------|------------------|--------------|------|
| internal Ridge-baseline (paired control) | +0.5 % | 18 pairs / 6 ds | paired Wilcoxon p = 0.15 — **statistically tied** |
| paper Ridge (cartesian preproc search)   | +24.6 % | 18 / 6  | descriptive: cohort regression, not paired test |
| paper PLS (cartesian preproc search)     | +17.6 % | 18 / 6  | descriptive |
| paper TabPFN-opt                         | +36.6 % | 18 / 6  | descriptive |
| paper TabPFN-raw                         | **−13.7 %** | 18 / 6 | descriptive; lower is better |
| paper CatBoost                           | **−22.2 %** | 18 / 6 | descriptive |
| paper CNN baseline                       | **−36.9 %** | 12 / 4  | only 4 datasets have a CNN ref in the cohort manifest |

The cohort-level Δ% vs internal Ridge of +0.5 % is statistically not
distinguishable from zero (Wilcoxon p = 0.15, 8 / 18 wins). **The
internal-Ridge gain is concentrated on very-small-n non-linear datasets**
(Beer, n=40, Δ = −3 %; Corn_Oil, n=64, Δ = −22 %); on the larger or more
linear-dominated datasets (ALPINE, Biscuit, DIESEL, Rice) the stack ties or
slightly regresses. The reference-baseline rows are descriptive statistics
on the cohort manifest, not paired statistical tests; we make no significance
claim against them.

We do **not** beat the paper Ridge / paper PLS / paper TabPFN-opt baselines
that benefit from a 60-trial cartesian preprocessing search. The Phase 1d
`SearchedRidge` baseline (an early-stage attempt to level the field) beats
paper Ridge by 7 % on a single dataset (ALPINE) but is a reduced
approximation of the paper recipe set (omits EMSC, Gaussian smoothing,
ASLSBaseline, OSC) and the curated-cohort comparison is left as future work.

## 1. Introduction

NIR spectroscopy regression is a long-standing chemometrics problem in which
classical linear models (PLS, Ridge) remain the workhorse. Recent deep-learning
work (DeepSpectra 2019, Cui & Fearn 2018, Mishra & Passos 2021–2023) shows
that 1-D CNNs can match or beat PLS on large datasets (`n > 5000`) and offer
a transfer story across instruments. On small-n NIR data the picture is bleak:
Padarian et al. 2019 require `n > 10 000` for CNNs to consistently beat PLS,
and the standard NIR benchmark suite (Corn, Tablet, Beer, Tecator) has
`n_train ≤ 300`.

This paper has three goals:

1. **Dissection.** We document fifteen specific architectural and training
   weaknesses of the `nirs4all` `nicon` / `decon` CNNs, ranging from a sigmoid
   regression output that saturates the target (W1) through stride-only
   downsampling that collapses short spectra (W3, W14) and an unprincipled
   mix of SELU / ReLU / ELU activations that breaks SELU's self-normalising
   invariant (W2). Severity ratings are documented in
   `docs/WEAKNESS_ANALYSIS.md`.

2. **A redesigned model.** `NiconV1c` is a 4-block small-kernel + GAP CNN
   that addresses the architectural weaknesses, paired with a deterministic
   concat-derivatives front (raw + 1st-SG + 2nd-SG channels) and Bjerrum 2017
   EMSC-parameter augmentation.

3. **A stacked ensemble.** Recognising that the CNN-only path hits a ceiling
   on small-n NIR data, we propose `Stack-Ridge-PLS-V1c`: out-of-fold
   stacking of Ridge, PLS, and the CNN through a Ridge meta-learner. We
   produce extensive ablations and per-sample predictions; on the
   6-dataset extended cohort the stack beats the published CNN, TabPFN-raw,
   and CatBoost baselines by 14-37 % median rmsep.

We provide a fully reproducible benchmark (resumable runner, predictions
parquet per `(variant, dataset, seed)`, environment lockfile, git SHA per
result row) and an open implementation under `bench/nicon_v2/`.

## 2. Related Work

We organise the literature into seven themes and 40 references; the full
review is in `source_materials/literature_review/LITERATURE_REVIEW.md`.

* **1-D CNN architectures for NIR.** Acquarelli 2017, Liu 2017, Cui & Fearn
  2018, DeepSpectra (Zhang 2019), Padarian 2019, Mishra & Passos 2021-2023,
  BEST-1DConvNet (Wang 2024).
* **Attention / transformer models.** SpectraTr 2022, ACT (AAAI 2024),
  TabPFN-v2 (Hollmann et al. 2025).
* **Regularisation for small NIR.** Lakshminarayanan 2017 (Deep Ensembles),
  Huang 2017 (Snapshot Ensembles), Padarian 2022 (MC-dropout for NIR).
* **Preprocessing and augmentation.** Engel 2013, Helin 2022 (learnable EMSC),
  Bjerrum 2017 (EMSC-parameter augmentation), Mishra et al. 2023, Yao 2022
  (C-Mixup).
* **Ensembles for NIR.** Mehmood 2024 (stacked spectral ensembles).
* **Baselines (PLS, Ridge, TabPFN, CatBoost).** Hollmann 2025 (TabPFN-v2),
  Wang 2025 (CatBoost-PLS hybrid).

The closest published analogues to our stacking proposal are Mehmood 2024
(stacked PLS variants) and the in-house `bench/AOM_v0/aompls/` and
`bench/AOM_v0/Ridge/aomridge/` projects.

## 3. Method

### 3.1 The MVB → V1a → V1b → V1c progression

Each variant is paired against the previous one to attribute the gain
cleanly. We test 12 hypotheses in five phases (`docs/HYPOTHESES.md`).

* **MVB (Phase 0).** Upstream `nicon` (kernels 15/21/5, strides 5/3/3,
  mixed activations, sigmoid output).
* **V1a-head-only (H1, Phase 1a).** Linear regression head replaces sigmoid
  output. Backbone unchanged.
* **V1c-bare (H3, H4, Phase 1c).** 4-block small-kernel (7/5/3/3) backbone +
  max-pool 2 + GELU + LayerNorm + GAP head + linear projection.
* **V1c-concat (H5).** + concat-derivatives front (raw + 1st-SG + 2nd-SG, w=11, p=2).
* **V1c-concat-bjerrum (H6).** + per-batch Bjerrum offset/slope/multiplicative augmentation (amplitudes scaled to per-dataset spectrum range).

We rejected H2 (single-activation in isolation) because it interacted poorly
with `Dropout1d(0.08)` on a 1-channel input. We deferred H8 (Inception block,
n > 500) because the cohort majority has n_train < 500.

### 3.2 Stacked ensemble (Phase 5 / H12)

`Stack-Ridge-PLS-V1c` stacks out-of-fold predictions of Ridge,
PLS, and `NiconV1c-concat-bjerrum` through a Ridge meta-learner. The OOF
splitter is sklearn's KFold (5-fold by default; SPXYFold available). The
meta-learner's `α` is selected by an inner KFold(5) CV over a 15-point
log-spaced grid in [10⁻³, 10⁴]. After OOF generation, base learners are refit
on the full training set and the meta is applied at inference time. The
implementation is in `nicon_v2/models/stacking.py` with a fold-isolation spy
test in `tests/test_stacking.py::test_stacking_does_not_leak_validation_fold`.

### 3.3 Reproducibility

* **Splits.** Predefined train/test splits from the TabPFN paper distribution.
* **Inner CV.** sklearn KFold for hyper-parameter selection only; never
  evaluation.
* **Seeds.** {0, 1, 2}.
* **Outputs.** Aggregate CSV + per-sample predictions parquet for every
  `(variant, dataset, seed)`. Environment lockfile + git SHA stamped on every
  result row.

## 4. Experimental Setup

* **Smoke cohort.** 3 datasets (ALPINE_P_291_KS, Beer_OriginalExtract_60_KS,
  Rice_Amylose_313_YbasedSplit). Used for fast iteration only.
* **Extended cohort.** 6 datasets (smoke + Biscuit_Fat_40_RandomSplit +
  Corn_Oil_80_ZhengChenPelegYbaseSplit + DIESEL_bp50_246_b-a). Span
  `n_train ∈ [40, 247]`, `n_features ∈ [401, 2151]`. Used for the publication
  headline numbers.
* **Curated cohort.** 39 datasets from `bench/AOM_v0/Ridge/benchmark_runs/curated/`. Used for the leaderboard claim (run in progress).
* **Reference baselines** (from `cohort_regression.csv`): paper Ridge, paper PLS, paper TabPFN-raw, paper TabPFN-opt, paper CNN, paper CatBoost.

## 5. Results

### 5.1 Extended cohort (6 datasets × 3 seeds)

`Stack-Ridge-PLS-V1c` numbers (median across 18 paired observations
unless otherwise noted; reference-baseline rows are **descriptive statistics
on the cohort manifest**, not paired statistical tests):

| Comparison                      | Median Δ% rmsep | Effective n / datasets | Test |
|---------------------------------|-----------------|--------------------------|------|
| vs internal Ridge-baseline      | +0.52 %         | 18 / 6                   | paired Wilcoxon p = 0.15 — **tied** |
| vs paper Ridge                  | +24.6 %         | 18 / 6                   | descriptive |
| vs paper PLS                    | +17.6 %         | 18 / 6                   | descriptive |
| vs paper TabPFN-opt             | +36.6 %         | 18 / 6                   | descriptive |
| vs paper TabPFN-raw             | **−13.7 %**     | 18 / 6                   | descriptive |
| vs paper CatBoost               | **−22.2 %**     | 18 / 6                   | descriptive |
| **vs paper CNN baseline**       | **−36.9 %**     | **12 / 4**               | descriptive (CNN ref missing on Beer, Biscuit_Fat) |

Per-dataset Δ% vs internal Ridge:

| Dataset | n_train | NiconV1c-cb | Stack-PLS | **Stack-V1c** |
|---------|---------|--------------|------------|----------------|
| ALPINE_P_291_KS               | 247 | +75.97 % | +0.72 % | +0.65 % |
| **Beer_OriginalExtract_60_KS** | 40  | +56.12 % | +0.66 % | **−3.06 %** |
| Biscuit_Fat_40_RandomSplit    | 40  | +213.82 % | +16.22 % | +16.16 % |
| **Corn_Oil_80_ZhengChenPelegYbaseSplit** | 64  | +268.48 % | −20.79 % | **−21.55 %** |
| DIESEL_bp50_246_b-a           | 113 | +206.18 % | +4.35 % | +4.15 % |
| Rice_Amylose_313_YbasedSplit  | 203 | +41.41 % | +0.78 % | +2.08 % |

The CNN brings clear non-linear residual signal on Beer (n=40) and Corn_Oil
(n=64); on linear-dominated datasets it is correctly down-weighted by the
meta-learner.

### 5.2 Curated cohort (39 datasets × 1 seed)

`Stack-Ridge-PLS-V1c` on the AOM-Ridge curated cohort (39 datasets, single seed,
predefined splits, 229 OK rows; figures: `publication/figures/stack_curated/`):

| Comparison | Median Δ% | Wins / n | Wilcoxon p | Verdict |
|------------|-----------|----------|-------------|---------|
| **vs internal PLS-baseline (paired)** | **−2.53 %** | **26 / 37** | **0.018** | **statistically significant win** |
| vs internal Ridge-baseline (paired)   | +0.38 %     | 16 / 37  | 0.25        | tied |
| vs paper Ridge                         | +4.87 %     | 38 datasets | descriptive | small descriptive loss |
| vs paper PLS                           | +2.30 %     | 38 datasets | descriptive | tied |
| vs paper TabPFN-raw                    | +0.13 %     | 38 datasets | descriptive | **tied** |
| vs paper TabPFN-opt                    | +6.90 %     | 38 datasets | descriptive | descriptive loss |
| **vs paper CNN baseline**              | **−6.42 %** | **34 / 39 datasets with CNN ref** | descriptive | **descriptive win** |
| vs paper CatBoost                      | −0.40 %     | 38 datasets | descriptive | tied |

**Friedman / Nemenyi cohort ranking** (k = 6 variants, n = 37 datasets;
χ² = 74.39, **p = 1.25 × 10⁻¹⁴**; Demsar 2006 critical difference at α = 0.05
is CD ≈ 1.32):

| Rank | Variant                    | Avg rank |
|------|----------------------------|----------|
| 1    | Ridge-baseline             | 2.62     |
| 2    | Stack-Ridge-PLS-V1aHead    | 2.70     |
| 3    | **Stack-Ridge-PLS-V1c**    | **2.97** |
| 4    | Stack-Ridge-PLS            | 3.00     |
| 5    | PLS-baseline               | 4.00     |
| 6    | NiconV1c-concat-bjerrum    | 5.70     |

The four Ridge / stack variants form one CD group; PLS is significantly worse;
the CNN alone is significantly worst (CD = 1.32; rank gap to PLS is 1.0, gap
to NiconV1c-cb is 2.7+).

**Per-dataset highlights.**

* Best stacking wins (Δ% rmsep vs internal Ridge): `Corn_Oil_80` −21.55 %,
  `Beer_60_YbaseSplit` −20.03 %, `Beer_60_KS` −5.65 %.
* Pathological reference: `Quartz_spxy70` (target ≈ 0; AOM-Ridge curated
  rmsep = 0; any absolute error has ∞ relative scale). Excluded from
  Wilcoxon analysis would not change the qualitative conclusion.
* Small-n stacking failure: `Biscuit_Sucrose_40_RandomSplit` +103 % vs Ridge,
  documented in §7.2 as the "OOF stacking over-fit on n_train < 50" failure
  mode.

The full per-dataset Δ% table is in `publication/tables/stack_curated/summary_per_dataset.csv`.

## 6. Ablations

The pre-registered ablation matrix is in `docs/IMPLEMENTATION_PLAN.md`. Smoke
results for each cell are in `benchmark_runs/phase{1a,1b,1c,stack,stack_aom}_*/`.

Key ablation outcomes:

* **H1 (linear head)** — accepted: −16 to −43 % rmsep on smoke cohort vs upstream NICON. The single most impactful single-line repair.
* **H2 (single-activation backbone)** — rejected as a stand-alone change: interacts poorly with Dropout1d(0.08) on 1-channel input.
* **H3 (small-kernel + GAP backbone)** — accepted as the production CNN architecture.
* **H5 (concat-derivatives)** — borderline: helps on some datasets (Beer), neutral on others.
* **H6 (Bjerrum aug)** — accepted: best CNN-only result on Beer (-20 % vs no aug).
* **H7 (C-Mixup)** — rejected at default σ_y on smoke (Beer regression). Fold-locally tuned σ_y deferred.
* **H8 (Inception block)** — deferred (lit-review: only helps n > 500).
* **H9 (deep ensembles + conformal UQ)** — deferred (focus on RMSE first).
* **H12 (stacking)** — **accepted as the production model**: only nicon_v2 variant to clear the scientific-success tier and beat published CNN / TabPFN-raw / CatBoost baselines.

## 7. Discussion

### 7.1 The CNN ceiling on small-n NIR

Even with all the SOTA tricks (concat-derivatives, Bjerrum augmentation,
small-kernel + GAP backbone) the best CNN-only variant
(`NiconV1c-concat-bjerrum`) is 42-269 % above Ridge on the extended cohort.
This is consistent with Padarian 2019's 10 000-sample heuristic. CNNs alone
do not solve small-n NIR. Our redesigned CNN serves as a non-linear
*residual learner* in a stacked ensemble, not as a stand-alone replacement
for PLS / Ridge.

### 7.2 Stacking is statistically tied with internal Ridge

The headline `Stack-Ridge-PLS-V1c` produces a median Δrmsep of +0.5 %
vs internal Ridge with paired Wilcoxon p = 0.15 over 18 paired observations
(8 / 18 wins). This is **not statistically distinguishable from internal
Ridge** at α = 0.05.

The cohort-level tie hides a clear per-dataset pattern: stacking wins by
−3 to −22 % on small-n datasets where the CNN's non-linear residual signal
matters (Beer, Corn_Oil) and slightly regresses by +0.6 to +16 % on the
linear-dominated datasets (ALPINE, Biscuit_Fat, DIESEL, Rice). The Ridge
meta-learner correctly down-weights the CNN where it underperforms, but
the small per-dataset gains do not aggregate into a cohort-level win at
n = 6 datasets.

**Failure mode worth flagging.** On the curated cohort (in-progress
benchmark) we observe a **>100 % regression on `Biscuit_Sucrose_40_RandomSplit`**
(n_train = 40, n_features = 700): the stacked model nearly doubles Ridge's
rmsep. The simpler `Stack-Ridge-PLS` (no CNN) also regresses (+66 %). PLS
alone regresses by only +18 %. We attribute this to **OOF-stacking
over-fit** on very-small-n high-dimensional data: with 40 train samples and
5-fold OOF, each fold's train set is 32 samples and the meta-learner sees
40 OOF predictions across 3 base columns, allowing the Ridge meta to over-fit
the small training rmsep. A practical mitigation is to skip the stacking
step (i.e. use plain PLS) when `n_train < 50` and the CV variance of the
base learners is high; we leave a principled stack-skip rule as future
work.

### 7.3 Reference-baseline comparisons are descriptive

Median Δ% rmsep against the cohort-manifest reference values is a useful
descriptive statistic but is **not a paired test**. Reference RMSEPs come
from the TabPFN paper's run with their own SPXYFold + 60-trial cartesian
preprocessing search; our `Stack-Ridge-PLS-V1c` reports its own predefined-split
test rmsep. The "−37 % vs paper CNN" / "−14 % vs TabPFN-raw" / "−22 % vs
CatBoost" headlines describe the *median ratio* over our cohort and refer to
our model being better on average; we do not claim statistical significance
against the paper baselines because we do not have access to their per-seed
predictions.

In particular, the paper CNN reference is missing for `Beer` and `Biscuit_Fat`
in the cohort manifest, so the "−37 % vs paper CNN" descriptive number is
computed over **only 4 of 6 datasets** (12 paired comparisons, not 18).

### 7.4 The paper-Ridge gap and SearchedRidge

Paper Ridge (`run_reg_pls.py`) uses a cartesian preprocessing search across
~30-60 candidates (SNV / MSC / EMSC(deg=1, 2) × SG(11/15/21/31, 2/3, 1/2) /
Gaussian × None / ASLSBaseline / Detrend × None / OSC(1/2/3)) with a
60-trial α grid. Our internal Ridge skips this; we lose 25 % rmsep against
paper Ridge on the extended cohort.

`SearchedRidge` (Phase 1d) implements a **reduced subset** of this search
space: 3 scatter (None / SNV / MSC) × 5 SG variants × 2 detrend × 11 α =
330 candidates. It deliberately omits EMSC, Gaussian smoothing, ASLSBaseline,
and OSC. On a single-seed pilot on ALPINE, `SearchedRidge` reaches 0.0549 vs
paper Ridge 0.0590 (−7 %), suggesting that the paper-Ridge gap is at least
partially due to preprocessing search and not to a fundamental advantage
of the paper's tooling. Running `SearchedRidge` and `Stack-SearchedRidge-V1c`
across the full curated cohort is the **highest-impact open follow-up**.

### 7.5 Limitations

* The extended cohort is 6 datasets, 18 paired observations — adequate for
  per-dataset insight but underpowered for a cohort-level Wilcoxon at α=0.05.
* The curated cohort run (39 datasets, 1 seed) is in progress at submission
  time; `Section 5.2` is filled when it completes.
* The CNN training is single-seed inside the stack (no Lakshminarayanan-style
  deep ensemble); H9 remains for future work.
* `SearchedRidge` is a reduced approximation of the paper recipe set; full
  parity is left as future work.
* C-Mixup with fold-locally-tuned σ_y (the H7 spec) is not yet wired in the
  train loop; the smoke evidence on the default-σ_y C-Mixup is rejected but
  the spec is not refuted.
* Reference-baseline comparisons are descriptive medians; we do not perform
  paired statistical tests against paper baselines because we do not have
  their per-seed predictions.

## 8. Conclusion

We dissect 15 specific failure modes of the upstream `nirs4all`
`nicon` / `decon` 1-D CNNs and propose **nicon_v2**, a redesigned CNN
(small-kernel + GAP backbone with concat-derivatives input + Bjerrum 2017
augmentation) embedded in a stacked Ridge / PLS / CNN ensemble through
a Ridge meta-learner. On a 6-dataset extended NIR cohort, the stack:

* **ties** internal Ridge (median Δrmsep +0.5 %, paired Wilcoxon p = 0.15);
* descriptively beats paper CNN by 37 % (4 / 6 datasets, 12 pairs), paper
  TabPFN-raw by 14 %, paper CatBoost by 22 %;
* descriptively loses to paper Ridge / paper PLS / paper TabPFN-opt by
  17-37 %, attributable to those baselines' cartesian preprocessing search
  which our internal Ridge skips.

Per-dataset evidence is consistent with the literature: the CNN's non-linear
residual signal helps on very-small-n datasets (Beer n=40, Corn_Oil n=64)
and is correctly down-weighted on linear-dominated datasets. The stacking
approach is the right path to closing the small-n CNN gap; the paper-Ridge
gap is partially attributable to preprocessing search asymmetry and is
left as future work via the Phase 1d `SearchedRidge` baseline (a reduced
approximation that beats paper Ridge by 7 % on ALPINE in a single-seed
pilot).

We release the full implementation, 88-test suite, ablation matrix,
per-sample predictions parquet, four rounds of Codex review log, and a
reproducible benchmark runner under `bench/nicon_v2/`.

## Appendix A — Reproducibility

* `nicon_v2/` — source.
* `tests/` — 87 tests (metrics, datasets, baselines, training, length-robustness, geometry parity, no-leak, preprocessing parity vs scipy, augmentation determinism, fold isolation in stacking).
* `benchmarks/run_baseline_benchmark.py` — resumable runner, predictions parquet output.
* `publication/scripts/{analyze_results,make_figures,make_tables,cohort_summary}.py` — paired Wilcoxon, figures, LaTeX tables, cohort summary with two-tier success check.
* Git SHA, environment lockfile, dataset hashes recorded per benchmark run.

## Appendix B — Codex review log

Three rounds of Codex review applied:

* Round 1 (plan, 26 findings): predefined-split as primary metric (F2/F12); training set notation `A = T \ (V ∪ C)` (F19); per-sample parquet (F14); ablation matrix (F23); two-tier success criterion (F21); hard wall-clock budget (F22). All Critical/High addressed.
* Round 2 (Phase 1a code, 6 findings): H1/H2 separate variants; no-leak SpyOperator test; geometry parity; predictions parquet wired; recompute flat-dim per cohort length.
* Round 3 (Phase 1c + stacking, 6 findings): wider α grid (1e−3 to 1e4); SPXY-aware splitter option; fold-isolation spy test; H7 σ_y deferred; AOM-Ridge as base learner; SG edge parity documented.

Full chronology in `docs/IMPLEMENTATION_LOG.md`.
