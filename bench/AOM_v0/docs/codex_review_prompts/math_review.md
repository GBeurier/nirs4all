# Codex Math Review Prompt

You are reviewing `bench/AOM_v0` as an external mathematical reviewer.

Read:

- `docs/AOMPLS_MATH_SPEC.md`
- `aompls/operators.py`
- `aompls/nipals.py`
- `aompls/simpls.py`
- `aompls/selection.py`
- `tests/test_operators.py`
- `tests/test_nipals.py`
- `tests/test_simpls.py`

Check:

1. Matrix conventions: `X_b = X A_b^T` and `X_b^T Y = A_b X^T Y`.
2. Whether `transform`, `apply_cov`, `adjoint_vec`, and `matrix` use compatible
   orientations.
3. NIPALS materialized vs adjoint equivalence.
4. SIMPLS materialized vs covariance equivalence.
5. Coefficient formula `B = Z (P^T Z)^+ Q^T`.
6. `orthogonalization="transformed"` vs `"original"` semantics.
7. PLS1 and PLS2 shape and SVD conventions.
8. Classification coding and probability calibration leakage risk.

Return findings ordered by severity with file and line references. If a formula
is wrong, give the corrected formula and a minimal test that would catch it.
