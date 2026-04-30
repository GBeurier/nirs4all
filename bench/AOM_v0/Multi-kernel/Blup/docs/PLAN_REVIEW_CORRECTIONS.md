# BLUP Plan Review Corrections (Codex roadmap review, 2026-04-30)

Source: `/tmp/codex_blup_roadmap.md`.

## High-Severity Findings

### H1. Prediction-variance formula wrong for total BLUP prediction

**Issue**: `BLUP_MATH_SPEC.md:128` and `IMPLEMENTATION_PLAN.md:131` omit
posterior cross-block covariance terms; also risk double-scaling
`sigma_b^2` by defining `k_b(x_*, x_*) = sigma_b^2 K_b(...)` then
multiplying by `sigma_b^2` outside.

**Fix**: define target first (latent mean / observed `y_*` / block-specific
`u_b`). For latent total prediction error with known variances:

```text
c_*  = sum_b sigma_b^2 k_b*           # (n,)  scaled cross-kernel sum
v_** = sum_b sigma_b^2 K_b(x_*, x_*)  # scalar
M    = X_f^T V^-1 X_f

MSE(hat y_*_latent) = v_**
                   - c_*^T V^-1 c_*
                   + (x_f* - X_f^T V^-1 c_*)^T M^-1 (x_f* - X_f^T V^-1 c_*)
```

For observed-response variance, add `sigma_e^2`. If shipping the
single-block heuristic only, **rename it as such** and document that it
is not the variance of the total BLUP prediction.

### H2. Fixed-effect API mismatch breaks decomposition identity

**Issue**: BLUP `predict(X)` vs `predict_components(X, X_fixed_test=...)`.
With non-intercept fixed effects, the two paths can use different
designs, breaking `predict_components["total"] == predict()`.

**Fix**: every prediction-facing method accepts the same fixed-effect
argument:

```python
predict(X, X_fixed_test=None)
predict_components(X, X_fixed_test=None)
contribution_table(X, X_fixed_test=None)
```

Internally, share one `_predict_parts(X, X_fixed_test)` function used by
both.

## Medium-Severity Findings

### M1. mkR equivalence under-specified

**Issue**: `BLUP_MATH_SPEC.md:95` says fixed-theta equivalence uses
`eta_b = sigma_b^2, alpha = sigma_e^2`. mkR projects to simplex.

**Fix**: state both parameterisations. Tests must either bypass simplex
projection or co-scale `alpha` by `s = sum_b sigma_b^2`.

### M2. Shrinkage diagnostic is single-kernel heuristic

**Issue**: `BLUP_MATH_SPEC.md:107` formula
`sigma_b^2 lambda_max(K_b) / (sigma_b^2 lambda_max(K_b) + sigma_e^2)`
ignores other blocks.

**Fix**: rename as "single-block retention heuristic" or replace with:

```text
S_b = sigma_b^2 K_b P    (P from MKM; P = V^-1 - V^-1 X_f M^-1 X_f^T V^-1)
edf_b = tr(S_b)          # effective degrees of freedom for block b
```

Optionally also report `edf_b / rank_eff(K_b)` and the spectral norm of
the residualised smoother.

### M3. `alpha_dual` naming drift

**Fix**: standardise on `alpha_dual_` everywhere (matches MKM). Compute
during MKM/BLUP fit from the same Cholesky solve used for REML; assert
predict-time decomposition never refactorises `V`.

### M4. `predict_components` return shape fragile

**Issue**: Flat dict `{"fixed": ..., "<op_name>": ..., "total": ...}` can
collide if a block is named `"fixed"` or `"total"`; duplicate operator
names overwrite values.

**Fix**: structured object:

```python
{
    "fixed": ndarray,                                 # (n_test,)
    "random": OrderedDict[block_name, ndarray],       # one per block
    "total": ndarray,                                 # (n_test,) — sum
}
```

For tabular output, columns: `component_type` ("fixed" / "random"),
`block_name`, `contribution`, `contribution_norm`,
`contribution_relative`.

### M5. Synthetic test thresholds inconsistent with spec

**Issue**: `TEST_PLAN.md:10` relaxes R1/R2 to `corr > 0.6`,
inactive ratio `< 0.5`. Spec asks for stricter (`corr > 0.8`, ratio
`< 0.2`).

**Fix**: align thresholds. Use `corr > 0.8`, ratio `< 0.2` at high SNR
on R1; for R3 only test correlated-pair sum (not individual).

### M6. Benchmark "ship" criteria overstate

**Fix**: BLUP success = MKM prediction equality + decomposition identity
+ no leakage + stable block summaries + useful plots. Repeated-refit
stability only on same held-out samples or OOF predictions; aggregate
aligned blocks before per-block stability claims.

### M7. Phase ordering: diagnostics before benchmarks

**Fix**: reorder gates:
1. Finalise MKM exposed state and BLUP API.
2. Implement fixed-theta decomposition.
3. Identity / no-leakage / fixed-effect tests.
4. Diagnostics (corrected formulas).
5. Benchmark.

Defer `prediction_variance_diagonal` until target and formula are
corrected.

## No Blocker

Core E-BLUP mean formula is correct:

```text
alpha_dual = V^-1 (y - X_f hat beta)
hat u_b_*  = sigma_b^2 K_b_* alpha_dual
```

## Action Items

| # | Severity | Action | Status |
|---|----------|--------|--------|
| H1 | High | Fix prediction-variance formula or rename heuristic | applied to spec |
| H2 | High | Unify fixed-effect API on `predict` and `predict_components` | applied to plan |
| M1 | Med | State both eta parameterisations | applied to spec |
| M2 | Med | Replace shrinkage with `edf_b = tr(S_b)` or rename as heuristic | applied to spec |
| M3 | Med | Standardise on `alpha_dual_` naming | applied to plan |
| M4 | Med | Use structured dict (fixed/random/total) | applied to spec/plan |
| M5 | Med | Tighten synthetic test thresholds | applied to test plan |
| M6 | Med | Reframe benchmark success criteria | applied to benchmark protocol |
| M7 | Med | Reorder phase gates | applied to plan |
