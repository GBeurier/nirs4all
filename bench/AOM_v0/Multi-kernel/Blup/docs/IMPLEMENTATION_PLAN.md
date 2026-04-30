# BLUP — AOM Multi-Kernel BLUP / E-BLUP: Implementation Plan

`BLUP` (Best Linear Unbiased Prediction) is the **prediction layer** that
sits on top of MKM. Once variance components `(sigma_b^2)_b, sigma_e^2`
are estimated by REML, BLUP gives:

- the **per-block contribution** `u_b` (random-effect prediction);
- the **per-individual decomposition** of total prediction across blocks;
- (optionally) prediction variance and shrinkage diagnostics.

In strict statistical terminology, **BLUP** assumes variances are known;
when variances are estimated, the prediction is **E-BLUP** (Empirical
BLUP). The estimator is `AOMMultiKernelBLUP` and wraps an
`AOMMultiKernelMixedModel` instance under the hood.

## Scope Boundary

```text
bench/aom_v0/Multi-kernel/Blup/
  blup/
    __init__.py
    estimator.py           # AOMMultiKernelBLUP wrapper around MKM
    decomposition.py       # predict_components, train_decompose, contribution_table
    diagnostics.py         # shrinkage diagnostics, prediction-variance approximations
  tests/
    conftest.py
    test_blup_decomposition.py
    test_blup_predict_total.py
    test_blup_no_leakage.py
    test_blup_synthetic.py
  benchmarks/
    __init__.py
    run_blup_benchmark.py
    summarize_blup_results.py
  benchmark_runs/
  docs/
    IMPLEMENTATION_PLAN.md  (this file)
    BLUP_MATH_SPEC.md
    TEST_PLAN.md
    BENCHMARK_PROTOCOL.md
    CODEX_REVIEW_WORKFLOW.md
    IMPLEMENTATION_LOG.md
    codex_review_prompts/
      math_review.md
      code_review.md
      test_review.md
      publication_review.md
  prompts/
  publication/
```

The package depends on `mkm` for variance estimation and on
`aompls` for operators / banks.

## Phase Gates

### Phase 0 — Synthetic ground-truth

`tests/synthetic.py`: simulate `y = X_f beta + sum_b u_b + e` with KNOWN
`u_b` per individual. Verify BLUP `hat u_b` correlates strongly with truth
when SNR is high.

### Phase 1 — `estimator.py`

`AOMMultiKernelBLUP` mostly delegates to `AOMMultiKernelMixedModel`:

```python
AOMMultiKernelBLUP(
    operator_bank="compact",
    method="reml",
    n_restarts=10,
    fixed_effects="intercept",
    kernel_center=True,
    kernel_normalize="trace",
    branch_preproc="none",
    feature_scaling="center",
    random_state=0,
)
```

API:
- `fit(X, y, X_fixed=None) -> self`.
- `predict(X) -> ndarray` — same as MKM.
- `predict_components(X, X_fixed_test=None) -> dict[str, ndarray]`:
  returns `{"fixed": ..., "<op_name>": ..., "total": ...}`.
- `train_decompose() -> dict[str, ndarray]` — same on training data.
- `contribution_table(X) -> pandas.DataFrame` — long-format contributions
  per individual per block.
- `score`, `get_params`, `set_params` (sklearn).

Stored: same as MKM plus
- `u_components_train_` — dict `block_name -> (n_train,)` array of
  `hat u_b` values on training data.
- `dual_alpha_` — `V^-1 (y - X_f beta_hat)` precomputed for prediction.

### Phase 2 — `decomposition.py`

```python
def predict_components(
    K_blocks_cross: list[ndarray],   # (n_test, n_train) per block
    X_fixed_test: ndarray | None,    # (n_test, r)
    sigma2_blocks: ndarray,
    beta_fixed: ndarray,
    dual_alpha: ndarray,             # V^-1 r
    block_names: list[str],
) -> dict[str, ndarray]
```

Returns:

```python
{
    "fixed": X_fixed_test @ beta_fixed,    # (n_test,)
    "<op_1>": sigma_1^2 * K_1_cross @ dual_alpha,
    ...
    "<op_B>": sigma_B^2 * K_B_cross @ dual_alpha,
    "total": sum of the above,
}
```

Critical invariant:

```text
predict_components(X)["total"] == predict(X)   (within fp tolerance)
```

### Phase 3 — `diagnostics.py`

- `shrinkage_factor(K_b_train, sigma_b^2, sigma_e^2)`: scalar measure of
  how strongly each block is shrunk toward zero.
- `prediction_variance_diagonal(K_blocks_test_train, V_inv_block_kernels, ...)`:
  approximation to `Var(hat y_*) = X_*^T M^-1 X_* (variance of fixed effects)
  + sum_b sigma_b^2 [K_b_test_test - K_b_*train V^-1 K_b_train_*]`.
  (Approximation: ignores cross-block covariance terms, which are zero by
  the model's independence assumption.)

### Phase 4 — Smoke benchmark

`benchmarks/run_blup_benchmark.py` mirrors MKM but adds per-block
contribution table to the output. Useful for Figure-style diagnostics in
the publication.

### Phase 5 — Reintroduce promising preprocessing

Same as MKM: enable branch_preproc.

### Phase 6 — Full 57-dataset benchmark

Same variants as MKM. Output table includes `relative_contribution_per_block`
per dataset (one row per block per dataset). Lets us identify which AOM
operators consistently explain variance across datasets.

## Required Test Commands

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM:bench/aom_v0/Multi-kernel/Blup \
  pytest bench/aom_v0/Multi-kernel/Blup/tests -q
```

## Non-Goals For First Implementation

- BLUP with full prediction-interval coverage check;
- Joint multi-trait BLUP;
- BLUP with frequentist debiasing for variance-component estimation error;
- POP-style BLUP variants;
- classification BLUP-DA.

## Notes on Naming

- `BLUP` (Best Linear Unbiased Prediction) — variances assumed known.
- `E-BLUP` (Empirical BLUP) — variances estimated; what we actually compute.
- We use `BLUP` as the package name for brevity but document E-BLUP in the
  manuscript.
