# Codex Math Review: BLUP

Review mathematical correctness of the BLUP / E-BLUP implementation.

Read:

```text
bench/aom_v0/Multi-kernel/Blup/docs/BLUP_MATH_SPEC.md
bench/aom_v0/Multi-kernel/Blup/blup/decomposition.py
bench/aom_v0/Multi-kernel/Blup/blup/estimator.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_decomposition.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_predict_total.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_synthetic.py
```

Verify:

```text
alpha_dual    = V^-1 (y - X_f hat beta)
hat u_b       = sigma_b^2 K_b alpha_dual         (training)
hat u_b_*     = sigma_b^2 K_b_* alpha_dual       (test)
hat y_*       = X_*f hat beta + sum_b hat u_b_*
predict       == sum predict_components          (decomposition identity)
```

Check:

1. `alpha_dual` is precomputed at fit time and reused at predict time.
2. Cross kernel `K_b_*` uses centring with **training** mean only.
3. Each contribution is a linear function of the test sample (when blocks
   are strict-linear).
4. `predict_components` keys match operator-bank block names.
5. `predict_components` is shape-compatible with `predict` (both `(n_test,)`).
6. Sum identity tested to `< 1e-10` absolute tolerance.
7. Boundary case: when `sigma_b^2 = 0`, the corresponding contribution is
   exactly zero (multiplied by zero, no NaN).
8. Highly aligned blocks (`align > 0.95`) reported in diagnostics; sum of
   their contributions is consistent with truth, individuals may not be.

Return findings ordered by severity with file and line references.
