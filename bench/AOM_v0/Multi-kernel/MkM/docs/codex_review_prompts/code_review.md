# Codex Code Review: MKM

Review code correctness, sklearn compliance, and robustness.

Read:

```text
bench/aom_v0/Multi-kernel/MkM/docs/MKM_MATH_SPEC.md
bench/aom_v0/Multi-kernel/MkM/mkm/kernelizer.py
bench/aom_v0/Multi-kernel/MkM/mkm/likelihood.py
bench/aom_v0/Multi-kernel/MkM/mkm/optimisation.py
bench/aom_v0/Multi-kernel/MkM/mkm/estimator.py
bench/aom_v0/Multi-kernel/MkM/tests/test_estimator_predict.py
bench/aom_v0/Multi-kernel/MkM/tests/test_estimator_no_leakage.py
```

Check:

1. Sklearn API compliance:
   - `__init__` only stores hyperparameters, no fitting work.
   - `get_params` / `set_params` work; estimator clonable via `sklearn.base.clone`.
   - `fit(X, y)` returns `self`.
   - Validation of `X.shape[0] == y.shape[0]`.
   - 1D `y` accepted; predicted output preserves 1D shape.

2. Numerical robustness:
   - Cholesky failure handled with adaptive jitter (mirroring AOM-Ridge).
   - All operations use `np.float64`.
   - `eps = 1e-12` for trace normalisation.
   - No division by zero.

3. Optimiser:
   - Random restarts seeded reproducibly with `random_state`.
   - L-BFGS-B `maxiter=200`, `tol=1e-8`.
   - Best restart selected by lowest `-ell_REML` (not by gradient norm).
   - Convergence flag stored (boolean).

4. No data leakage:
   - Operator fits use only training data.
   - Trace normalisation uses training kernel only.
   - Cross kernels use stored training statistics.

5. CV path (when applicable):
   - Branch-preproc selection uses fold-local fits.
   - SpyKernelizer test passes.

6. Determinism:
   - Same `random_state` produces identical estimates.
   - No dependency on operator iteration order in dict (use lists).

7. Memory:
   - Block kernels stored only when needed (not duplicated per restart).

8. Logging:
   - `verbose=0` silent;
   - `verbose>=1` prints final REML, variances, fit time;
   - `verbose>=2` prints per-restart endpoints.

Return findings ordered by severity with file and line references.
