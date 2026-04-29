# Codex Code Review: AOM-Ridge

Review AOM-Ridge code after estimator and selection implementation.

Read:

```text
bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md
bench/AOM_v0/Ridge/docs/AOM_RIDGE_API.md
bench/AOM_v0/Ridge/aomridge/
bench/AOM_v0/Ridge/tests/
```

Focus on:

- `coef_` must be original-space `(p, q)`, not wide-space;
- `predict` must use train means and stored original coefficient;
- CV must rebuild fold-local means, block scales, operators, and kernels;
- `selection="global"` must score every `(operator, alpha)` candidate;
- active-superblock screening must use normalized scores;
- diagnostics must be JSON-serializable;
- custom banks must not be mutated across repeated fits;
- operator instances must be cloned/fresh-fit per CV fold;
- one-dimensional and multi-output targets must both work.

Return findings first, ordered by severity, with concrete fixes.

