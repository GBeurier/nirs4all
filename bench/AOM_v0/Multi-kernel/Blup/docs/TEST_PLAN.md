# BLUP — Test Plan

## Test Targets

| File | Coverage |
|------|----------|
| `tests/test_blup_decomposition.py` | `predict_components` returns dict with right keys (`fixed`, `<op_name>` per block, `total`). All values shape `(n_test,)`. `total == sum(others)` to `< 1e-10`. |
| `tests/test_blup_predict_total.py` | `predict(X) == predict_components(X)["total"]` for any `X`. |
| `tests/test_blup_no_leakage.py` | Spy test: `predict_components` uses only training stats stored at fit-time. |
| `tests/test_blup_synthetic.py` | On synthetic R1, R2, R3 (same as MKM): `corr(hat u_active, u_active_truth) > 0.6` per active block; `||hat u_inactive||_2 / ||hat u_active||_2 < 0.5`. |
| `tests/test_blup_at_fixed_theta_equiv_mkr.py` | At fixed `theta`, BLUP `predict` equals mkR predict with `eta = sigma_b^2, alpha = sigma_e^2`. |

## Acceptance

- All tests pass under `pytest -q`.
- Decomposition sum invariant holds to `<= 1e-10` on `n=200, p=400, B=4`
  synthetic data.
- Per-block correlation tests pass on R1, R2; R3 only sum-of-pair test
  passes (individual contributions not identifiable).

## Test Commands

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM:bench/aom_v0/Multi-kernel/Blup \
  pytest bench/aom_v0/Multi-kernel/Blup/tests -q
```
