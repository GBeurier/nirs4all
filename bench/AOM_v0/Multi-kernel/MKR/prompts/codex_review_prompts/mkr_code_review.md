# Codex Code Review: mkR

Read:

```text
bench/aom_v0/Multi-kernel/MKR/aomridge/kernelizer.py
bench/aom_v0/Multi-kernel/MKR/aomridge/weights.py
bench/aom_v0/Multi-kernel/MKR/aomridge/mkr_estimator.py
bench/aom_v0/Multi-kernel/MKR/aomridge/diagnostics.py
bench/aom_v0/Multi-kernel/MKR/tests/test_mkr_estimator.py
bench/aom_v0/Multi-kernel/MKR/tests/test_mkr_no_leakage.py
```

Check:

1. Sklearn API:
   - `__init__` only stores hyperparameters.
   - `get_params`/`set_params` complete.
   - `fit(X, y) -> self`.
   - `score(X, y)` finite.
2. Numerical robustness: Cholesky path with adaptive jitter; eigendecomposition
   path for alpha-grid sweeps.
3. softmax_cv:
   - L-BFGS-B on theta in unconstrained R^B.
   - Inner CV uses fold-local kernels (no leakage).
   - Cache inner Cholesky factorisations across folds (one factorisation
     per `(theta, alpha)` evaluation per fold).
   - `n_restarts` random Dirichlet starts.
4. KTA simplex weights match existing `mkl.learn_block_weights` API but
   uses the **centred + trace-normalised** kernels from kernelizer.
5. `coef_` recovery only when no nonlinear branch and all blocks
   strict-linear. Otherwise `coef_ = None` and `predict` uses dual.
6. Diagnostics: pairwise alignment matrix returned with proper shape and
   bounds.
7. Determinism: same `random_state` reproduces softmax_cv result.
8. Memory: kernel block list uses `O(B n^2)`; large default banks (100+)
   should fall back to top_k pruning before kernel materialisation.
9. Branch-preproc fits are fold-local.

Return findings ordered by severity.
