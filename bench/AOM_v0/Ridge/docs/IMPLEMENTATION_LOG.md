# AOM-Ridge Implementation Log

This log is append-only. Each phase should add:

```text
date
phase
files changed
tests run
Codex review prompt used
findings fixed
findings deferred
```

## 2026-04-29: Planning Documents Created

Created the AOM-Ridge documentation scaffold under `bench/AOM_v0/Ridge`.

Key decisions:

- phase 1 is strict-linear only;
- `selection="superblock"` is the primary AOM-Ridge model;
- `selection="global"` is a required baseline;
- active-superblock and nonlinear branch kernels are later phases;
- CV kernels, block scales, means, and fitted preprocessors must be fold-local;
- implementation should be self-contained under `bench/AOM_v0/Ridge`.

## 2026-04-29: Phases 1-7 implemented (claude pilot)

Files added:

- `aomridge/__init__.py`, `kernels.py`, `solvers.py`, `selection.py`,
  `estimators.py`
- `tests/conftest.py`, `tests/test_ridge_kernel_equivalence.py`,
  `test_ridge_solvers.py`, `test_ridge_estimators.py`,
  `test_ridge_cv_no_leakage.py`, `test_ridge_selection.py`
- `benchmarks/__init__.py`, `run_aomridge_benchmark.py`,
  `summarize_aomridge_results.py`

Phase summary:

- Phase 1 — strict-linear kernel utilities (`kernels.py`). `K_b = Xc A^T A Xc^T`,
  superblock `K = sum_b s_b^2 ...`, `U = sum_b s_b^2 A^T A Xc^T`, RMS block
  scaling, explicit-superblock helper for tests.
- Phase 2 — dual Ridge solvers (`solvers.py`). Trace-relative alpha grid,
  Cholesky path with adaptive jitter, eigendecomposition path, vectorised
  alpha-path solver for fast CV.
- Phase 3 — `AOMRidgeRegressor(selection="superblock")` with sklearn-style
  `fit/predict/score`, `coef_` of shape `(p, q)`, identity-only matches
  sklearn `Ridge`, dual matches explicit concatenated Ridge.
- Phase 4 — fold-local CV (`selection.cv_score_alphas`). `cv` accepts an
  integer (KFold) or any sklearn-compatible splitter (`cv=SPXYFold(...)`),
  per the user request. SpyOperator-based no-leakage tests verify per-fold
  centering and that validation rows never enter operator fits or kernel
  construction.
- Phase 5 — `selection="global"` evaluates every `(operator, alpha)`
  pair via fold-local CV and refits the chosen pair on full calibration
  data.
- Phase 6 — `selection="active_superblock"` screens operators with
  normalised scores `||s_b A_b Xc^T Yc||_F^2`, retains identity, prunes
  redundant operators by response cosine, and feeds the surviving subset
  into the superblock model.
- Phase 7 — smoke benchmark runner (`benchmarks/run_aomridge_benchmark.py`)
  with `SPXYFold` as default inner CV; resumable per-row CSV with the
  documented schema. Summariser computes median relative RMSEP vs Ridge-raw,
  wins, failures, and timings.

Tests: 45 / 45 passing.

Acceptance command:

```
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
```

Smoke benchmark (3 datasets, SPXYFold CV) ran end-to-end, 12 result rows.

Codex review: not yet invoked. Run after this commit:

```
codex exec --skip-git-repo-check \
  --output-last-message /tmp/aomridge_codex_math.md \
  "$(cat bench/AOM_v0/Ridge/prompts/codex_review_prompts/math_review.md)" \
  </dev/null
```

Codex math review (2026-04-29):

- Medium: `resolve_operator_bank` did not dedupe duplicate
  `IdentityOperator` instances supplied by the user.
- Low: `_eigh_solve` clips all negative eigenvalues, not only tiny ones —
  fine for the PSD kernels AOM-Ridge produces, surprising for indefinite
  inputs.

Fixed:

- `resolve_operator_bank` now keeps only the first identity and prepends one
  if absent. New test `test_resolve_bank_dedupes_duplicate_identity` covers
  this.
- `_eigh_solve` carries an explicit docstring describing the PSD-only
  contract and pointing indefinite callers at the Cholesky path.

Tests after Codex round 1: 46 / 46 passing.

Findings deferred:

- Phase 8 (nonlinear branch kernels) intentionally not started.

## 2026-04-29: Codex code + test review round 1

Code review:

- High: ``active_superblock`` leaked target via full-data screening before CV.
- Medium: ``active_top_m`` not enforced when identity pre-kept.
- Low: ``operator_scores`` dict overwrote duplicate names.

Test review:

- High: block-scale leakage spy weak; folded-clone identity not directly
  asserted; active CV no-leak coverage missing.

Fixed:

- New ``cv_score_active_alphas`` / ``select_alpha_active`` screen the active
  subset *inside* every fold; estimator wires the CV path through it before
  computing the final active subset on full calibration data.
- ``screen_active_operators`` validates ``top_m >= 1`` and short-circuits
  when the identity-kept list already meets the cap.
- ``operator_scores`` is now a list of records ``{index, name, best_rmse}``.
- New tests: ``FitOnceOperator`` raises if any clone is fitted twice;
  ``CountColsSpy`` asserts that no operator ``apply_cov`` ever sees the
  full sample count; ``select_alpha_active`` is exercised with the same
  spy.

Smoke benchmark after round 1: best variant per dataset still
``AOMRidge-global-compact`` (3% on AMYLOSE, -35% on BEER, 2% on ALPINE).
Superblock dominated by alpha=495 (ALPINE) — clear over-regularisation
from RMS block scaling.

## 2026-04-29: Codex backlog round 1 received (12 items)

Saved to ``docs/CODEX_BACKLOG_2026-04-29.md``. Highest-leverage items:

1. ``scale_power`` block weighting (gamma=0/0.5/1).
2. Adaptive alpha grid with boundary expansion.
3. Pooled-MSE / 1-SE selection rule.
4. Family-balanced active screening with KTA + family quotas.
5. Run smoke variants on `default` / `family_pruned` / `response_dedup`.
6. Fold-local feature standardisation (StandardScaler-equivalent for identity).
7. Strict-linear multi-bank stacking.

Codex confirmed no sign / centering / U / beta bugs.

## 2026-04-29: Iter 1 implementation (items #1, #2, #12)

- ``compute_block_scales_from_xt`` accepts ``block_scaling="scale_power"``
  with parameter ``scale_power`` ∈ [0, 2]. ``scale_power=0`` ≡ ``"none"``;
  ``scale_power=1`` ≡ ``"rms"``.
- New ``alpha_at_boundary`` helper + ``_select_alpha_with_expansion``
  loops the CV when the optimum hits a grid edge, expanding the bracket
  by 3 decades on the relevant side (max 2 expansions by default).
- Diagnostics: ``alpha_index``, ``alpha_at_boundary``, ``grid_expansions``,
  ``cv_min_score``, ``scale_power``.
- Benchmark schema: added ``relative_rmsep_vs_paper_ridge`` (the actual
  reference, not the local Ridge-raw which was confusingly named).

Iter 1 smoke results (3 datasets, % vs ``ref_rmse_ridge`` from TabPFN paper;
negative = beats paper Ridge HPO + preprocessing):

- ALPINE: superblock-rms +16.1%, **superblock-none +0.9%** (alpha 495 → 0.028).
- AMYLOSE: superblock-rms +65.5%, superblock-none +30%, **global +2.9%**.
- BEER: superblock-none +29%, **global -35%**.

Conclusion: ``block_scaling="none"`` resolves the over-regularisation
diagnosed by Codex. AOM-Ridge now matches paper Ridge on ALPINE
(within 1 pt) and beats it on BEER. Still loses on AMYLOSE.

## 2026-04-29: Iter 2 implementation (items #4, #6)

- New ``aomridge.preprocessing`` module with ``fit_feature_scaler`` /
  ``apply_feature_scaler`` for fold-local feature standardisation.
- Estimator parameter ``x_scale ∈ {none, center, feature_std, feature_rms}``
  threaded through CV, screening, and final refit. The fitted ``coef_`` is
  back-mapped to the original feature space (``coef_proc / x_scale``) so
  ``predict(X)`` operates on raw inputs without remembering scales.
- New equivalence test ``test_feature_std_matches_standard_scaler_ridge``:
  identity bank + ``x_scale="feature_std"`` matches sklearn
  ``Pipeline(StandardScaler, Ridge)`` to floating-point precision.
- ``screen_active_operators`` accepts ``score_method ∈ {norm, kta, blend}``
  and ``max_per_family``. KTA = kernel-target alignment; ``blend`` sums
  min-max-normalised norm + KTA scores.
- New tests: family-quota enforcement, KTA / blend score paths run on a
  small bank.

Tests after iter 2: 53 / 53 passing.

Bench iter 2: in progress (3 datasets, 8 variants including
``stdscale`` and ``family-balanced active``).

