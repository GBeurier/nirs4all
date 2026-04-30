# MKM — AOM Multi-Kernel Mixed Model: Implementation Plan

`MKM` is the **probabilistic** counterpart of `mkR`. Each AOM operator block
becomes a random effect `u_b ~ N(0, sigma_b^2 K_b)` with covariance set by
the centred / trace-normalised AOM block kernel. Variances are estimated
by REML (or ML) maximum-likelihood, providing interpretable variance
contributions per block.

## Scope Boundary

```text
bench/aom_v0/Multi-kernel/MkM/
  mkm/
    __init__.py
    kernelizer.py         # mirror of mkR kernelizer (centered + trace-normed)
    likelihood.py         # ML / REML log-likelihood + gradient for V = sum_b sigma_b^2 K_b + sigma_e^2 I
    optimisation.py       # L-BFGS-B on log-variances, multi-start, bounds, convergence diagnostics
    estimator.py          # AOMMultiKernelMixedModel sklearn estimator
    diagnostics.py        # variance contributions, log-likelihood profile, kernel alignment
  tests/
    conftest.py
    test_kernelizer.py
    test_single_kernel_reml.py
    test_multi_kernel_reml.py
    test_estimator_predict.py
    test_estimator_no_leakage.py
    synthetic.py
  benchmarks/
    __init__.py
    run_mkm_benchmark.py
    summarize_mkm_results.py
  benchmark_runs/                          # (created at runtime)
  docs/
    IMPLEMENTATION_PLAN.md         (this file)
    MKM_MATH_SPEC.md
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
  publication/                              # (manuscript scaffolding)
```

The package depends on `aompls` (operators, banks) and **may** import
`aomridge.kernelizer` if mkR ships first; otherwise MkM ships its own
copy of the kernelizer.

## Phase Gates

### Phase 0 — Synthetic ground-truth

`tests/synthetic.py`: simulate `y = X_f beta + sum_b u_b + e` with known
variances. Three regimes:
- **R1 (clean)**: 1 active block of 4, `sigma_b^2 = 1`, `sigma_e^2 = 1`,
  `n = 200`, `B = 4`. MKM should recover `sigma_1^2` near 1 and others near 0.
- **R2 (mixture)**: 3 active blocks, varied variances.
- **R3 (correlated)**: 2 quasi-identical blocks (`A_2 = A_1 + small noise`)
  to stress identifiability.

Acceptance: on R1, REML recovers `sigma_active^2 / sum sigma^2 in [0.6, 1.2]`
of the truth.

### Phase 1 — `kernelizer.py`

Same API as mkR kernelizer: centred + trace-normalised block kernels with
fold-local statistics. May `import` from `Ridge.aomridge.kernelizer` once
that ships, with a documented adapter in `mkm.kernelizer`. **No code copy
duplication** — adapter only.

### Phase 2 — `likelihood.py`

For variance vector `theta = (log sigma_1^2, ..., log sigma_B^2, log sigma_e^2)`:

```text
V(theta) = sum_b exp(theta_b) * K_b + exp(theta_e) * I
beta_hat = (X_f^T V^-1 X_f)^-1 X_f^T V^-1 y
r = y - X_f beta_hat
ell_ML(theta)   = -0.5 [ logdet(V) + r^T V^-1 r + n log(2 pi) ]
ell_REML(theta) = -0.5 [ logdet(V) + logdet(X_f^T V^-1 X_f) + r^T V^-1 r + (n-r) log(2 pi) ]
```

Single Cholesky factorisation of `V` per evaluation.
- `compute_logdet_and_solve(V_blocks, theta) -> (logdet, V_inv_y, V_inv_Xf, ...)`.
- `neg_log_reml(theta, V_blocks, y, X_f) -> float`.
- `neg_log_reml_grad(theta, ...) -> ndarray` (analytic gradient using
  `tr(P dV/dtheta_j) - r^T V^-1 dV/dtheta_j V^-1 r` form).

### Phase 3 — `optimisation.py`

`fit_variance_components(K_blocks, y, X_f, method, n_restarts, bounds, tol)`:
- Multi-start with `n_restarts=10`, default initial `theta_0 ~ N(0, I)` in log space.
- L-BFGS-B with bounds `theta_b in [-15, 15]`.
- Convergence: gradient norm `< 1e-5` or change `< 1e-7`.
- Reports best run + all restart endpoints (used to detect multimodality).
- Detects boundary solutions (`theta_b at bound`) and reports
  `boundary_components`.

### Phase 4 — `estimator.py` — `AOMMultiKernelMixedModel`

```python
AOMMultiKernelMixedModel(
    operator_bank="compact",
    method="reml",                 # or "ml"
    optimizer="lbfgs",
    n_restarts=10,
    fixed_effects="intercept",     # or callable, or None
    kernel_center=True,
    kernel_normalize="trace",
    bounds_log_var=(-15.0, 15.0),
    branch_preproc="none",
    feature_scaling="center",
    random_state=0,
    jitter=1e-8,
    verbose=0,
)
```

API:
- `fit(X, y, X_fixed=None) -> self`.
- `predict(X, X_fixed_test=None) -> ndarray` — total prediction (mean only).
- `predict_components(X) -> dict` — per-block contributions (delegated to
  `Blup` package; MKM only ships `predict()` and `decompose_train()`).
- `score(X, y) -> float` — R^2.
- Stored: `sigma2_blocks_`, `sigma2_residual_`, `relative_contributions_`,
  `beta_fixed_`, `alpha_dual_` (= `V^-1 (y - X_f beta_hat)`),
  `log_likelihood_`, `converged_`, `boundary_components_`,
  `optimisation_diagnostics_`.

### Phase 5 — Smoke benchmark

`benchmarks/run_mkm_benchmark.py` runs on 3 datasets, reports
`sigma_b^2`, `relative_contribution_b`, RMSEP, fit_time. Compare with mkR
softmax_cv (same kernels, different optimisation criterion).

### Phase 6 — Reintroduce promising preprocessing

Add `branch_preproc` (SNV, MSC, ASLS) to MKM kernelizer (fold-local) and
re-run smoke.

### Phase 7 — Full 57-dataset benchmark

Same as mkR but with MKM variants:
- `MKM-reml-compact-none`
- `MKM-reml-compact-snv`
- `MKM-reml-default-none`

Output: per-dataset table of `(sigma_b^2)_b`, RMSEP, alignment_max,
fit_time. Compare with mkR, AOM-Ridge, AOM-PLS, TabPFN.

## Required Test Commands

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM \
  pytest bench/aom_v0/Multi-kernel/MkM/tests -q
```

## Non-Goals For First Implementation

- Multi-trait / multi-output Y;
- per-individual random effects (e.g. plant, batch — outside AOM scope);
- frequentist confidence intervals on `sigma_b^2` (deferred to Round 2);
- Bayesian MCMC sampling;
- POP-MKM variants;
- classification MKM-DA.
