# MKM — Test Plan

## Test Targets

| File | Coverage |
|------|----------|
| `tests/test_kernelizer.py` | Centred + trace-normalised block kernels, fold-local statistics, no leakage on cross-kernels. |
| `tests/test_likelihood.py` | `neg_log_reml(theta)` and `neg_log_ml(theta)` numerical correctness vs. brute-force `np.linalg.slogdet` + `np.linalg.solve`. Gradient agrees with finite-difference within `1e-4` per-component on synthetic. |
| `tests/test_single_kernel_reml.py` | Single-kernel REML with `K_g + sigma_e^2 I`. Verify `(sigma_g^2, sigma_e^2)` recovered on synthetic R1. Verify mean prediction equivalent to `Ridge(alpha = sigma_e^2 / sigma_g^2 * tau)`. |
| `tests/test_multi_kernel_reml.py` | R1, R2, R3 synthetic: variance components estimated, relative ordering correct, boundary diagnostic flags collapse. |
| `tests/test_estimator_predict.py` | sklearn API contract, `predict` shape, mean-zero residual on training (to numerical tolerance), `score(X, y)` returns finite R². |
| `tests/test_estimator_no_leakage.py` | SpyKernelizer-based test: validation rows never enter likelihood evaluation, kernel construction, or operator fits in any CV fold. |
| `tests/test_at_fixed_theta_equiv_mkr.py` | Predict at fixed `theta` matches mkR predict with `eta = sigma_b^2, alpha = sigma_e^2`. |

## Acceptance

- All tests pass under `pytest -q`.
- `test_likelihood.py` finite-difference gradient check: `<= 1e-4` relative
  error per component for `n=50, B=3` random `theta`.
- `test_single_kernel_reml.py`: estimated `sigma_b^2` within `[0.5, 2.0]`
  of truth on R1.
- `test_multi_kernel_reml.py`: `relative_contribution_active in [0.4, 0.9]`
  on R1, ordering correct on R2.
- `test_estimator_no_leakage.py`: spy assertion never raised.

## Test Commands

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM \
  pytest bench/aom_v0/Multi-kernel/MkM/tests -q
```

## Synthetic Recipes (mkm/tests/synthetic.py)

- **R1**: 1 active block, `sigma_active^2 = 1`, `sigma_e^2 = 1`, `n=200, p=400, B=4`.
- **R2**: 3 active blocks, mix of variances.
- **R3**: 2 quasi-identical blocks (kernel alignment > 0.95).
