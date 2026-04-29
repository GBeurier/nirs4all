# Codex Math Review: AOM-Ridge

Review mathematical correctness after kernel/solver implementation.

Read:

```text
bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md
bench/AOM_v0/Ridge/docs/PLAN_REVIEW_CORRECTIONS.md
bench/AOM_v0/Ridge/aomridge/kernels.py
bench/AOM_v0/Ridge/aomridge/solvers.py
bench/AOM_v0/Ridge/tests/test_ridge_kernel_equivalence.py
bench/AOM_v0/Ridge/tests/test_ridge_solvers.py
```

Verify:

```text
K_b = X A_b^T A_b X^T
K_super = sum_b s_b^2 X A_b^T A_b X^T
U = sum_b s_b^2 A_b^T A_b X^T
K = X U
C = (K + alpha I)^-1 Y
beta = U C
X beta = K C
```

Also verify:

- `s_b^2` is used in the metric;
- identity is included exactly once;
- alpha is positive;
- multi-output `Y` is handled correctly.

Return findings first, ordered by severity.

