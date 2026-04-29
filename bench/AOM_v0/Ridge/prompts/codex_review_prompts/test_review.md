# Codex Test Review: AOM-Ridge

Review the AOM-Ridge test suite before benchmarks.

Read:

```text
bench/AOM_v0/Ridge/docs/TEST_PLAN.md
bench/AOM_v0/Ridge/tests/
bench/AOM_v0/Ridge/aomridge/
```

Assess whether tests catch:

- using `A A^T` instead of `A^T A`;
- missing `s_b^2`;
- assigning wide coefficients to `coef_`;
- slicing globally centered kernels during CV;
- leaking validation data into block RMS scales;
- reusing mutable operator instances across folds;
- failing to keep identity in the bank;
- active-superblock duplicate pruning bugs;
- multi-output shape regressions.

Return missing tests first, ordered by severity.

