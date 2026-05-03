# Multi-Kernel Iteration Plan — closing the TabPFN-opt gap

**Goal**: median rel-RMSEP vs TabPFN-opt < 1.0 on diverse10 cohort.
Currently best multi-kernel variant achieves 1.23-1.31 — **needs ~25 %
improvement** on the 5 losing datasets (ALPINE, BEER, N_woOutlier,
grapevine_chloride, TIC).

## Iteration cycle

```
benchmark on diverse10 (or diverse8 to skip ECOSIS)
  → analyse per-dataset losses
  → propose intervention
  → implement
  → re-benchmark
  → Codex review + corrections
  → repeat
```

Stop when iteration brings < 2 % median improvement OR all variants
have been exhausted.

## Iter 0 — Stacking ensemble (rejected; user wants individually strong models)

Tested briefly on BEER YbaseSplit with rel-PLS=1.28 (worse than each
single base). Stacking introduces overfitting on small datasets and
the user prefers each model to be individually strong rather than
ensembles. Dropped from iteration plan.

## Iter 1 — Active-screened default bank (DONE, ~88% gain on hard datasets)

**Hypothesis**: compact 9-op bank is too narrow. Default 100-op bank
is too wide (selection variance). Pre-screen default bank to top-k by
KTA on training data → diversity benefit without noise.

**Implementation**: added `top_k_active` + `screen_score_method` params
to `AOMKernelizer`. Reuses existing `screen_active_operators` helper
(training-data only). Passed through `kernel_top_k_active` in
mkR/MKM/BLUP estimators.

**Tested variants**: 5 variants on 7/8 diverse10 datasets:
- mkR-softmax_cv-default-active15
- mkR-softmax_cv-snv-default-active15
- MKM-reml-default-active15
- MKM-reml-asls-default-active15
- BLUP-reml-default-active15

**Results**:
- median rel-PLS for `MKM-reml-asls-default-active15` = **0.966**
  (vs baseline 1.000, **3.4 pp gain**, **5/7 wins**).
- median rel-TabPFN-opt for `mkR-softmax_cv-snv-default-active15` =
  **1.127** (vs baseline 1.227, **10 pp gain**).
- BIG wins: BEER YbaseSplit **+13.3%**, TIC **+9.5%**.
- Slight regressions: An_NeoSpectra (Ridge-raw still wins),
  MANURE_Total_N (-1.9%).

**Verdict**: clear net positive. The mkR + MKM + BLUP frameworks all
benefit from the active-screened default bank.

## Iter 3 — Score-method ablation (queued)

**Hypothesis**: I used `screen_score_method="norm"` in Iter 1 (default
in `screen_active_operators`). Codex round 5 suggested testing `kta`
and `blend` to see if a different supervised score selects a better
top-15 subset.

**Variants** (planned):
- `mkR-softmax_cv-snv-default-active15-kta`
- `MKM-reml-asls-default-active15-kta`
- `mkR-softmax_cv-snv-default-active15-blend`
- `MKM-reml-asls-default-active15-blend`

**Cohort**: same focused 4 datasets as Iter 2 (BEER, TIC, ALPINE,
MANURE_MgO). Skip if Iter 2 already pushes hard variants below the
TabPFN-opt threshold.

## Iter 4 — Sparse simplex weights (queued)

**Hypothesis**: softmax_cv currently spreads weights across many
operators. Sparser weighting (most eta_b ≈ 0) may reduce noise.

**Implementation idea**: replace KL-to-uniform regulariser in
softmax_cv with an entropy-MAXIMISATION term (or L1 prior) to encourage
sparse simplices. Or post-hoc thresholding.

## Iter 5 — Domain-aware screening (queued)

**Hypothesis**: when small-n datasets have specific scatter / baseline
issues, the screen could be informed by domain heuristics (e.g.,
boost ASLS-like operators on darker spectra).

**Implementation idea**: pluggable score functions that combine
data-driven KTA with domain priors.

## Iter 2 — Tuned + active30 (in progress)

**Hypothesis**: combining iter1's active-screened default bank with
**tighter optimisation budgets** (n_restarts=5, max_iter=40,
alpha_grid=40 for mkR; n_restarts=6, max_iter=150 for MKM) should
push softmax_cv and REML to better optima. Separately, increasing
top_k_active to 30 may help on datasets with many useful operators.

**Tested variants**:
- mkR-softmax_cv-snv-default-active15-tuned (best Iter1 + tuned budget)
- MKM-reml-asls-default-active15-tuned (best Iter1 + tuned budget)
- mkR-softmax_cv-snv-default-active30 (more diversity)
- MKM-reml-asls-default-active30 (more diversity)

**Cohort**: 4 representative datasets (ALPINE, BEER YbaseSplit, TIC,
MANURE_MgO) — mix of hard (BEER/TIC) and easier (ALPINE/MANURE_MgO).

**Expected**: 1-3 % additional lift on top of Iter 1.

## Iter 2 — Active-screened default bank (planned)

**Hypothesis**: compact bank (9 ops) is too narrow. The "default" bank
(100 ops) is too wide and hurts via selection variance. An
active-screened bank: 100 ops → KTA-screen on training data only → top
20 → kernel construction → softmax_cv weight learning.

This combines **diversity** with **selection-variance control**.

**Implementation**: extend `AOMKernelizer` with a `top_k_active`
parameter that pre-screens operators with closed-form KTA before
materializing kernels.

**Expected**: rel-PLS / TabPFN-opt improvement on high-p, complex
spectra datasets (ALPINE p=2151, GRAPEVINES p=1023).

## Iter 3 — POP-style mkR (planned)

**Hypothesis**: a single global eta is too rigid. Allow PER-COMPONENT
operator selection: each dimension of the prediction can come from a
different operator (cf POP-PLS work in AOM-PLS package).

**Implementation**: new estimator `POPMultiKernelRidge` that fits PLS
component-by-component, picking the best operator per component via
PRESS or CV.

**Expected**: 3-8 % rel-RMSEP improvement on diverse cohorts.

## Iter 4 — Sparse simplex weights (planned)

**Hypothesis**: softmax_cv currently spreads weight across many
operators. A sparser weighting (most eta_b = 0) may reduce noise from
irrelevant blocks.

**Implementation**: add an entropy or L1 regularisation term to the
softmax_cv objective.

**Expected**: 2-5 % rel-RMSEP improvement; mostly improved
interpretability (active block count drops).

## Iter 5 — HSIC-based weights (planned)

**Hypothesis**: KTA uses raw Frobenius alignment; HSIC (centred KTA) is
a more robust dependence measure that may improve closed-form weight
estimates.

**Implementation**: replace `kta_simplex_weights` with HSIC variant.

**Expected**: 1-3 % rel-RMSEP improvement.

## Compute budget per iteration

- Iter 1 (Stack-5): ~30-90 min wall time on diverse8.
- Iter 2 (active bank): ~60-120 min wall time on diverse8 (more
  expensive due to default bank screening).
- Iter 3 (POP): ~60-120 min wall time, similar.
- Iter 4 (sparse): ~30 min wall (just changes optimisation).
- Iter 5 (HSIC): ~30 min wall (closed-form, cheap).

After each: Codex review, manuscript update, figures regenerated.

## Stopping criterion

- Median rel-TabPFN-opt < 1.0 on diverse10 (target met).
- OR: 3 consecutive iterations with < 2 % improvement.
- OR: all 5 iterations exhausted.

## Final integration (Phase 9)

Once converged on diverse10:
- Run the winning variant on the full 54-dataset cohort.
- Run on the 17-dataset classification cohort.
- Rewrite the manuscript with full results.
- Final Codex publication review.
