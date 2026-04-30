# mkR — Test Plan

## Test Targets

| File | Coverage |
|------|----------|
| `tests/test_mkr_kernelizer.py` | Centring (`K_b @ 1 == 0`), trace norm (`tr(K_b)/n == 1`), cross kernels via training stats only, no leakage of test centring. |
| `tests/test_mkr_weights.py` | Uniform = `1/B`. Manual rejects negatives, projects to simplex. KTA reproduces existing `mkl.learn_block_weights` result modulo centring. softmax_cv on noiseless oracle data converges close to oracle within tolerance. |
| `tests/test_mkr_estimator.py` | Sklearn API contract (`get_params`/`set_params`), `eta_` simplex, `coef_` shape `(p,)` when no nonlinear branch, primal `X@coef + intercept` ≈ dual `K@C` on train and test. Multi-output not supported. |
| `tests/test_mkr_equivalences.py` | Uniform mkR with strict-linear identity-only bank ≡ sklearn `Ridge` (when alpha grid normalised). Uniform mkR ≡ AOM-Ridge superblock at `block_scaling="none"` modulo trace normalisation. KTA mkR gives same ranking of blocks as existing `mkl` mode. |
| `tests/test_mkr_no_leakage.py` | SpyOperator-based test: validation rows never enter operator fits, kernel centring, or trace normalisation in any CV fold. |
| `tests/test_mkr_diagnostics.py` | Alignment matrix is symmetric, diagonal = 1, off-diagonal in `[-1, 1]`. Stability score is in `[0, 1]`. |

## Acceptance

- All tests pass under `pytest -q`.
- No leakage spy test passes with strict assertion (validation indices were never seen).
- softmax_cv test runs in `< 30 s` on `n=200, B=4` synthetic.

## Test Commands

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR \
  pytest bench/aom_v0/Multi-kernel/MKR/tests -q -k mkr
```

## Synthetic Data Recipes (see Ridge/aomridge/tests/synthetic.py)

- **Synth-A** (oracle): `n=200, p=400, B=4`, only block 1 active, SNR=5, `random_state=0`.
- **Synth-B** (mixture): `n=200, p=400, B=4`, blocks {1, 2, 3} active, SNR=3.
- **Synth-C** (correlated): `n=200, p=400, B=4`, blocks 1, 2 quasi-identical, SNR=3.
