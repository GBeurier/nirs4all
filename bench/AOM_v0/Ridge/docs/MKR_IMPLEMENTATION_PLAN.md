# mkR — AOM Multi-Kernel Ridge: Implementation Plan

`mkR` extends the existing `AOM-Ridge` infrastructure with **explicit per-block
kernel weights** and **trace-normalized centered kernels**. Unlike the existing
`selection="mkl"` mode (which learns KTA-aligned simplex weights and applies
`s_b` block scales), mkR exposes the weights as first-class hyperparameters
that can be set manually, fixed uniform, or learned by closed-form alignment
or by gradient-based optimisation of the inner-CV loss.

## Scope Boundary

Implementation lives **inside** the existing AOM-Ridge package:

```text
bench/AOM_v0/Ridge/
  aomridge/
    kernelizer.py      # NEW — centered + trace-normalized block kernels
    weights.py         # NEW — uniform / manual / kta / softmax_cv strategies
    mkr_estimator.py   # NEW — AOMMultiKernelRidge sklearn estimator
    diagnostics.py     # NEW — kernel alignment matrix + stability metrics
    # existing files: kernels.py, mkl.py, solvers.py, selection.py,
    # estimators.py, branches.py, classification.py, cv.py, preprocessing.py
  tests/
    test_mkr_kernelizer.py   # NEW
    test_mkr_weights.py      # NEW
    test_mkr_estimator.py    # NEW
    test_mkr_equivalences.py # NEW — uniform vs superblock, kta vs existing mkl
  benchmarks/
    run_mkr_benchmark.py     # NEW
```

The existing `selection="mkl"` mode is **kept untouched** and serves as a
baseline for benchmark comparison. mkR is a **separate estimator**
(`AOMMultiKernelRidge`) that lives next to `AOMRidgeRegressor`.

## Mathematical Difference Versus Existing AOM-Ridge

| Aspect | Existing AOM-Ridge | New mkR |
|--------|-------------------|---------|
| Kernel form | `K = sum_b s_b^2 K_b` | `K_eta = sum_b eta_b * K_b_norm` |
| Block weighting | RMS-normalised scales `s_b` (data-dependent) | Explicit weights `eta_b >= 0`, often `sum eta_b = 1` |
| Kernel pre-processing | None per-block (sum first) | **Each `K_b` centered (`H K_b H`) and trace-normalised (`n / tr(K_b)`)** |
| Weight learning | superblock (no learning), mkl (KTA simplex top-k) | uniform, manual, kta_simplex, softmax_cv (gradient on inner CV RMSE) |
| Diagnostic | block importance | kernel alignment matrix + per-fold weight stability |

Trace normalisation makes weights `eta_b` directly comparable across blocks
and is a prerequisite for the MKM / BLUP probabilistic interpretation.

## Phase Gates

### Phase 0 — Synthetic ground-truth bed

- `tests/synthetic.py`: generate `X`, blocks `Z_b = X A_b^T`, simulate
  `y = sum_{b in active} Z_b beta_b + e` with controlled SNR.
- Verify uniform mkR ≥ raw Ridge baseline, oracle weights ≥ uniform.
- Verify mkR with `kta` weights tags active blocks.

### Phase 1 — `kernelizer.py`

Implement `AOMKernelizer` (sklearn-style fit/transform):

- `fit(X_train, y_train=None)`: store `n_train`, fit operator bank, compute raw
  block kernels `K_b = X A_b^T A_b X^T`, apply centering `K_b_c = H K_b H` with
  `H = I - (1/n) 1 1^T` (training-mean), apply trace normalisation
  `K_b_norm = (n / tr(K_b_c)) * K_b_c`. Store `block_means` (the `(1/n) K_b 1`
  rows used for cross-kernel centering).
- `transform(X_test)`: compute raw cross kernels `K_b_cross = X_test U_train_b`,
  apply double-centering using stored train means, apply stored trace
  scales. Returns list of `(n_test, n_train)` cross kernels.

Acceptance:
- For one-block-only kernels, `mean(K_train_b_norm) == 0`,
  `trace(K_train_b_norm)/n == 1`.
- Cross kernel shape `(n_test, n_train)` and centering uses **only** training
  statistics (no leakage spy test).

### Phase 2 — `weights.py`

Implement weight strategies:

- `uniform_weights(B) -> np.ndarray`: `1/B`.
- `manual_weights(values, B) -> np.ndarray`: validation + simplex projection.
- `kta_simplex_weights(K_blocks, y, top_k) -> np.ndarray`: closed-form KTA on
  centered/normalised kernels.
- `softmax_cv_weights(K_blocks, y, alpha_grid, cv, n_iter) -> np.ndarray`:
  parameterise `eta = softmax(theta)`, optimise total CV RMSE via L-BFGS-B
  with finite-difference gradient (or analytic if cheap). Caches inner
  Cholesky factorisations across folds.

Acceptance:
- Uniform returns simplex `1/B`.
- Manual rejects negative entries; renormalises to simplex.
- KTA reproduces existing `mkl.learn_block_weights` modulo the centering
  difference (closed-form check on synthetic data).
- softmax_cv on noiseless oracle synthetic data converges close to oracle
  weights.

### Phase 3 — `mkr_estimator.py`

Implement `AOMMultiKernelRidge`:

```python
AOMMultiKernelRidge(
    operator_bank="compact",
    kernel_center=True,
    kernel_normalize="trace",
    weight_strategy="uniform",  # uniform/manual/kta/softmax_cv
    weight_init=None,
    weight_top_k=None,
    weight_n_restarts=3,
    alphas="auto",
    alpha_grid_size=50,
    alpha_low=-6.0, alpha_high=6.0,
    cv=5,
    cv_kind="kfold",  # or "spxy" / "spxy_repeated"
    scoring="rmse_pooled",
    branch_preproc="none",
    feature_scaling="center",
    one_se_rule=False,
    random_state=0,
)
```

Required: sklearn-like `fit`, `predict`, `score`, `get_params`, `set_params`.
Stores `eta_`, `alpha_`, `dual_coef_`, `coef_` (when computable in original
space — i.e. when no nonlinear branch is used), `kernel_alignment_matrix_`,
`weight_stability_` (per-fold weight variance).

Coefficient computation, when no nonlinear branch:

```text
U_eta = sum_b eta_b * tau_b * A_b^T A_b X^T   (with tau_b = trace scale)
beta = U_eta @ (K_eta + alpha I)^-1 (y - y_mean)
```

Acceptance:
- `weight_strategy="uniform"` matches superblock-Ridge with `block_scaling="none"`
  on identity-only operator (sanity).
- `predict` agrees between dual `K_cross @ C` and primal `X @ coef + intercept`.
- No leakage spy test passes on `cv=5`.

### Phase 4 — `diagnostics.py`

- `kernel_alignment_matrix(K_blocks)`: pairwise normalised Frobenius inner
  product, returns `(B, B)` matrix.
- `weight_stability(per_fold_weights)`: returns mean, std, and a stability
  score per block.
- `block_contribution(K_blocks, eta, alpha, y)`: per-block contribution to
  predicted variance.

### Phase 5 — Smoke benchmark

`benchmarks/run_mkr_benchmark.py` mirrors `run_aomridge_benchmark.py` but
adds variants:

- `mkR-uniform-compact`
- `mkR-kta-compact`
- `mkR-softmax_cv-compact`
- `mkR-softmax_cv-default`
- with branch preproc (`none`, `snv`, `msc`, `asls`)

Default cohort: 3 datasets (ALPINE/AMYLOSE/BEER) for smoke; 12 datasets for
extended. Output CSV with columns:
`dataset, variant, eta_*, alpha, kernel_alignment_max, weight_std, rmsep,
mae, r2, ref_rmse_*, fit_time_s`.

### Phase 6 — Reintroduce promising preprocessing

When smoke is stable: enable `branch_preproc` to chain a fitted preprocessor
(SNV, MSC, ASLS, OSC, EMSC1) **before** the operator bank, mirroring the
AOM-Ridge championship variant. Tests verify branch fits are fold-local.

### Phase 7 — Full 57-dataset benchmark

Run mkR against `all57_cohort.csv` with 3 best variants, generate
`relative_rmsep_per_variant.csv`, compare with AOM-Ridge `mkl` baseline.

## Required Test Commands

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge \
  pytest bench/AOM_v0/Ridge/tests -q -k mkr
```

## Non-Goals For First Implementation

- POP-style per-component variants;
- classification mkR-DA (deferred to Round 2);
- multi-output mkR (Y must be `(n,)` or `(n, 1)`);
- automatic operator bank selection;
- gradient-based eta optimisation with analytic gradient (start with
  finite-difference; switch to analytic only if convergence is too slow).
