# AOM-Ridge Implementation Plan

## Scope Boundary

Implementation lives under:

```text
bench/AOM_v0/Ridge
```

The implementation may import operators and banks from `bench/AOM_v0/aompls`,
but it must not modify production `nirs4all` or the validated AOM-PLS code
unless explicitly requested.

## Target Layout

```text
bench/AOM_v0/Ridge/
  aomridge/
    __init__.py
    kernels.py
    solvers.py
    selection.py
    estimators.py
    branches.py
    diagnostics.py
    metrics.py
  tests/
    test_ridge_kernel_equivalence.py
    test_ridge_solvers.py
    test_ridge_estimators.py
    test_ridge_selection.py
    test_ridge_cv_no_leakage.py
  benchmarks/
    run_aomridge_benchmark.py
    summarize_aomridge_results.py
```

## Phase Gates

### Phase 0: Orientation

Read the Ridge docs plus:

```text
bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md
bench/AOM_v0/aompls/operators.py
bench/AOM_v0/aompls/banks.py
```

Append a short entry to `docs/IMPLEMENTATION_LOG.md`.

### Phase 1: Strict-Linear Kernel Core

Implement in `aomridge/kernels.py`:

- `as_2d_y(Y)`;
- `resolve_operator_bank(operator_bank, p)`;
- `clone_operator_bank(operators, p=None)`;
- `fit_operator_bank(operators, X, Y=None)`;
- `compute_block_scales_from_xt(Xt, operators, block_scaling="rms")`;
- `metric_times_xt(Xt, operators, block_scales)`;
- `linear_operator_kernel_train(Xc, operators, block_scales)`;
- `linear_operator_kernel_cross(X_left_c, U_train)`;
- `explicit_superblock(Xc, operators, block_scales)` for tests only.

Acceptance:

- kernel equals explicit concatenated superblock;
- identity-only kernel equals `Xc Xc^T`;
- no dense `p x p` materialization except small explicit tests.

### Phase 2: Dual Ridge Solvers

Implement in `aomridge/solvers.py`:

- `make_alpha_grid(K, n_grid=50, low=-6, high=6)`;
- `solve_dual_ridge(K, Y, alpha, method="auto")`;
- `predict_dual(K_cross, dual)`;
- optional eigen path for fast alpha CV.

Acceptance:

- Cholesky path matches `np.linalg.solve`;
- eigen path matches Cholesky;
- jitter/symmetrization behavior is tested.

### Phase 3: Superblock Estimator

Implement `AOMRidgeRegressor(selection="superblock")`.

Required:

- sklearn-like `fit`, `predict`, `score`, `get_params`, `set_params`;
- `center=True`;
- `operator_bank="compact"`;
- `block_scaling="rms"`;
- `alphas="auto"`;
- `coef_` has shape `(p, q)`;
- univariate `predict(X)` returns `(n,)`.

Acceptance:

- identity-only matches sklearn Ridge;
- superblock dual matches explicit concatenated Ridge;
- `Xc @ coef_ == K @ dual_coef_`.

### Phase 4: Fold-Local CV

Implement:

- `cv_score_superblock(...)`;
- `select_alpha_superblock(...)`;
- no global kernel slicing;
- fold-local means, operator clones, operator fits, and block scales.

Acceptance:

- spy test proves validation rows are not observed during fold fitting;
- CV returns finite scores;
- repeated fits with a custom mutable bank do not reuse stale state.

### Phase 5: Global Hard Selection

Implement `selection="global"`:

- resolve bank with identity exactly once;
- evaluate every `(operator, alpha)` by fold-local CV;
- select mean validation RMSE minimum;
- refit selected pair on full calibration data.

### Phase 6: Active Superblock

Implement `selection="active_superblock"`:

- compute fold-local block scales;
- score operators with `||s_b A_b Xc^T Yc||_F^2`;
- always retain identity when possible;
- prune redundant operators by response cosine;
- fit superblock Ridge on active operators only.

### Phase 7: Branch Kernels

Optional after strict-linear models pass:

- branch protocol with `fit/transform`;
- raw, SNV, MSC branches;
- materialized branch kernels;
- no original-space coefficient unless proven linear.

## Required Test Commands

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
```

Focused:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest \
  bench/AOM_v0/Ridge/tests/test_ridge_kernel_equivalence.py \
  bench/AOM_v0/Ridge/tests/test_ridge_solvers.py -q
```

## Non-Goals For First Implementation

- POP-Ridge;
- MKL weight optimization;
- classification Ridge-DA;
- production porting;
- refactoring AOM-PLS.

