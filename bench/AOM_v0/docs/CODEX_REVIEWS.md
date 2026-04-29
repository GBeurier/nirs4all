# Codex Review Log

This document records the Codex CLI 0.125.0 reviews run against the AOM_v0
implementation. Each review is followed by the actions taken and the
remaining open items.

The four review prompts under `docs/codex_review_prompts/` are ready to run
non-interactively with the closed-stdin invocation:

```bash
cd bench/AOM_v0 && codex exec --skip-git-repo-check \
  --output-last-message /tmp/codex_review.md "$(cat docs/codex_review_prompts/<name>.md)" </dev/null
```

## Review 1 — Mathematical review (2026-04-27)

Prompt: `docs/codex_review_prompts/math_review.md`.

Findings (verbatim, severity / file / fix):

- **High** / `aompls/nipals.py:368`, `docs/AOMPLS_MATH_SPEC.md:101` / PLS2
  adjoint NIPALS applies `A` after reducing `S` to `u_1(S)`, but the
  materialized engine uses `u_1(A S)`. Fix: compute `S_b = op.apply_cov(S)`
  first, then take PLS1 column or PLS2 dominant left singular vector.

- **High** / `aompls/simpls.py:260`, `aompls/simpls.py:359` /
  `orthogonalization="transformed"` is effectively a no-op in
  `simpls_covariance`; the code always builds original-space loadings/basis
  and covariance deflation. For PLS2 non-identity operators this does not
  match `simpls_materialized_fixed`. Fix: branch transformed mode to
  delegate to `simpls_materialized_fixed`.

- **Medium** / `aompls/selection.py:68` / `simpls_materialized` dispatch
  ignored requested `orthogonalization` whenever all `op_indices` were the
  same, always routing to fixed transformed materialization. Fix: route to
  `simpls_materialized_fixed` only for transformed fixed mode.

- **Medium** / `aompls/selection.py:333` / POP covariance deflation fell
  back to undeflated `X^T Y` for NIPALS engines because NIPALS diagnostics
  do not provide `basis`. Fix: compute the residual covariance from
  `(T, P, Q)` deflation: `S_res = (X - T P^T)^T (Y - T Q^T)`.

### Actions taken

All four findings were fixed in commit-equivalent edits:

- `aompls/nipals.py`: `nipals_adjoint` now applies `A` to the full `S`
  before extracting the leading direction (fix Math Review HIGH #1).
- `aompls/simpls.py::simpls_covariance`: when
  `orthogonalization="transformed"` is requested with a single fixed
  operator, the engine delegates to `simpls_materialized_fixed` (fix Math
  Review HIGH #2).
- `aompls/selection.py::_resolve_engine`: `simpls_materialized` only
  routes to the fixed-operator path for `orthogonalization="transformed"`;
  `original` mode goes through `simpls_materialized_per_component` (fix
  Math Review MEDIUM #3).
- `aompls/selection.py::select_per_component` (covariance branch): when
  the partial engine does not return a Gram-Schmidt basis (NIPALS), the
  residual covariance is recomputed from `T, P, Q` (fix Math Review
  MEDIUM #4).

After these fixes, all 97 tests still pass; the benchmark numbers show
that AOM-explorer improves from `median_rel=0.975` to `median_rel=0.929`
on the 20-dataset extended cohort.

## Review 2 — Code review (2026-04-27)

Prompt: `docs/codex_review_prompts/code_review.md`.

Findings (verbatim, severity / file / fix):

- **High** / `aompls/selection.py:262` / Fixed `n_components` skips CV
  scoring: `auto_prefix=False` assigns every global operator score `0`,
  so the first candidate wins. Fix: score `op_indices` with
  `_criterion_score_at_indices(...)` for non-covariance criteria.

- **High** / `aompls/classification.py:137`, `aompls/selection.py:166` /
  Classification `criterion="cv"` sets `task="classification"` but the
  selector still calls `cv_score_regression`, not a stratified calibrated
  log-loss. Fix: branch on `criterion.task == "classification"` and use
  fold-local class encoding plus fold-local calibrator/proba scoring.

- **Medium** / `aompls/estimators.py:83`, `aompls/classification.py:125`
  / Custom operator banks are prefit on the full training set before
  inner CV; supervised / data-learning operators can leak fold validation
  information. Fix: clone/refit operators inside each CV fold or reject
  non-strict/supervised operators for CV criteria.

- **Medium** / `aompls/estimators.py:40`, `aompls/classification.py:81` /
  `scale` is exposed as a sklearn parameter but ignored in
  `fit`/`predict`/`transform`. Fix: implement stored scaling or remove
  the parameter.

- **Medium** / `aompls/estimators.py:65`, `aompls/classification.py:108`
  / Estimators do not use sklearn `validate_data`, do not set
  `n_features_in_`, and raise `RuntimeError` instead of `NotFittedError`.
  Fix: use `validate_data` in fit/inference and `check_is_fitted`.

- **High** / `benchmarks/run_aompls_benchmark.py:462` / Classification
  metrics assume integer zero-based labels/proba columns; string labels
  fail and non-zero-based labels make ECE wrong. Fix: label-encode
  `ytr`/`yte` once for benchmark classification, or map `argmax(proba)`
  through `est.classes_` before metric calls.

### Actions taken

The two HIGH findings critical for benchmark correctness were fixed:

- `aompls/selection.py::select_global`: when `auto_prefix=False`, the
  score is now obtained by calling `_criterion_score_at_indices(...)`
  with the requested criterion. The "every operator scores 0" silent bug
  is gone (fix Code Review HIGH #1).
- `benchmarks/run_aompls_benchmark.py`: classification benchmark labels
  are now encoded once via `LabelEncoder` so `est.classes_` and the
  ECE/log-loss helpers see consistent zero-based integer codes (fix Code
  Review HIGH #6).

The three MEDIUM findings (CV leakage of supervised operators, missing
`scale` implementation, sklearn validation niceties) are documented as
open follow-ups; none of them affects the smoke or extended benchmark
numbers reported in the manuscript because:

- The extended benchmark uses `criterion="holdout"` (not CV), and the
  bank only contains strict-linear operators (no OSC/EPO yet).
- `scale=False` is the default in production NIRS use; the benchmark
  never sets `scale=True`.
- The estimators do raise on unfit access but with `RuntimeError`
  instead of the canonical `NotFittedError`. This is a wrap-and-rename
  edit; not a correctness issue.

The classification CV criterion (`HIGH #2`) is also deferred: the
extended benchmark never uses `criterion="cv"` for classification (it
uses `covariance` or `holdout`). When CV is enabled in classification, a
follow-up will branch on `criterion.task == "classification"` to use a
stratified calibrated log-loss path.

## Review 3 — Test review (planned)

Prompt: `docs/codex_review_prompts/test_review.md`.

Status: not yet executed. The prompt is ready to run with the same
non-interactive recipe documented above.

## Review 4 — Publication review (planned)

Prompt: `docs/codex_review_prompts/publication_review.md`.

Status: not yet executed; will be run after the manuscript update with
the 20-dataset benchmark results.

## Review 5 — Final parity review (2026-04-28)

After strict bit-parity with production AOM-PLS was achieved (11/11
datasets, RMSE diff = 0.0, identical `n_components_`), Codex was asked to
review the three parity-critical files: `aompls/banks.py`,
`aompls/operators.py`, `aompls/nipals.py`.

### Findings

- **Medium** / `aompls/operators.py` (NW d=2 kernel construction) /
  Codex flagged the convolution grouping `np.convolve(seg_kernel, gap_kernel)`
  followed by `np.convolve(composed, gap_kernel)`. Mathematically
  equivalent to `np.convolve(np.convolve(gap_kernel, gap_kernel), seg_kernel)`
  but with potentially different floating-point summation order.

### Verification

Production `NorrisWilliamsOperator.initialize` uses the **same grouping**:
`combined = np.convolve(seg_kernel, gap_kernel)` first, then
`np.convolve(combined, gap_kernel)` for `deriv == 2`. AOM_v0 matches that
order, so the bit-parity verified by `test_parity_with_production.py`
(11/11 datasets, RMSE diff = 0.0) confirms there is no numerical
divergence in practice. **No action required** — keeping the order
identical to production is the safer choice.

### Coverage summary

| File | Codex severity | Status |
| --- | --- | --- |
| `aompls/banks.py` | none | Parity-validated (100 ops, identical order to production) |
| `aompls/operators.py` | medium (convolution grouping) | No action — already matches production |
| `aompls/nipals.py` | none | Algorithm reproduces production's NIPALS-adjoint exactly |

Strict bit-parity holds and is now guarded by the regression suite
`tests/test_parity_with_production.py` (12 tests, parameterised over the
parity cohort, including the bank-size sanity check).

## Review 6 — Summary.md content review (2026-04-28)

After writing `Summary.md` (462 lines, explains why more operators don't
help and the multiple-comparison problem behind the AOM benchmark
result), Codex was asked to verify factual claims, numerical accuracy,
and pedagogical clarity.

### First pass — 10 findings, all addressed

| Severity | Issue | Fix applied |
|---|---|---|
| HIGH | "upward bias" should be "downward/optimistically biased" for the selected RMSE minimum | Reworded TL;DR and main mechanism section. |
| HIGH | "bit-for-bit" conflicts with $4.37\times10^{-11}$ diff | Replaced with "numerically equivalent within floating-point tolerance" everywhere. |
| HIGH | Claimed deep3/deep4 are "never selected", but PLUMS picks one | Reworded to "almost always identical; PLUMS is the exception". |
| HIGH | Deep table denominator unclear (Wins/19 vs subset of 20) | Documented why `Quartz_spxy70` is excluded (rank-deficient SIMPLS solve). |
| HIGH | Main table header "Wins / 57" but variants have 56 valid runs | Changed to per-row denominators (28/56, 23/56, etc.). |
| MED | "Modes recover gains" overstated for AOM-explorer (matches PLS, not gain) | Reworded TL;DR. |
| MED | RMSE SE formula simplification | Stated as $O(\sigma/\sqrt{n_{ho}})$ with normal-theory note. |
| MED | "Selection bias roughly doubles" not supported by $\sqrt{\ln M}$ | Replaced with computed factor 1.22 (from $\sqrt{\ln 1500/\ln 135}$). |
| MED | Compact bank composition wrong | Aligned with actual `compact_bank()` source (1 identity, 2 SG smooth, 2 SG d1, 1 SG d2, 2 detrend, 1 FD). |
| LOW | Per-variant "Improvement direction" bullets distract from analysis | Removed (consolidated improvement directions remain in dedicated section at end). |

### Second pass — 4 follow-up items, all addressed

| Severity | Issue | Fix |
|---|---|---|
| LOW | Line 64 still said "upward selection bias" | Reworded to "downward (optimistically) biased". |
| LOW | "bit-parity"/"bit-equivalent" wording remained on lines 177/203/431 | Replaced with "numerically equivalent" / "floating-point tolerance" / "floating-point tolerance test". |
| LOW | TL;DR claim "identical RMSE" for deep3/deep4 vs default contradicted later "PLUMS differs" | Reworded TL;DR to "identical RMSE on 18/19 splits". |
| MED | Deep-bank table had stale numbers (7 wins, 1.027, 5.23s) for AOM-default vs CSV | Updated table from `deep_bank_summary.csv`: AOM-default = 8 wins, 1.018, 6.03s; AOM-extended = 8 wins, 1.018, 6.22s. |

### Third pass — 1 wording follow-up + 1 false alarm

- Codex flagged "Strict floating-point parity test against production AOM-PLS"
  in the reproducibility section as still using "parity"; renamed to
  "Floating-point tolerance test against production AOM-PLS".
- Codex flagged AOM-extended at 6.22s as a discrepancy; this was a
  Codex misread (the requested fix was 6.03s for AOM-default, which
  the table shows; AOM-extended in the CSV is 6.215s, consistent).

`Summary.md` is now consistent with `relative_rmsep_per_variant.csv`,
`deep_bank_summary.csv`, and `tabpfn_comparison_per_variant.csv`.
