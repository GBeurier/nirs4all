# Codex Test Review: MKM

Read:

```text
bench/aom_v0/Multi-kernel/MkM/docs/TEST_PLAN.md
bench/aom_v0/Multi-kernel/MkM/tests/test_kernelizer.py
bench/aom_v0/Multi-kernel/MkM/tests/test_likelihood.py
bench/aom_v0/Multi-kernel/MkM/tests/test_single_kernel_reml.py
bench/aom_v0/Multi-kernel/MkM/tests/test_multi_kernel_reml.py
bench/aom_v0/Multi-kernel/MkM/tests/test_estimator_predict.py
bench/aom_v0/Multi-kernel/MkM/tests/test_estimator_no_leakage.py
bench/aom_v0/Multi-kernel/MkM/tests/test_at_fixed_theta_equiv_mkr.py
bench/aom_v0/Multi-kernel/MkM/tests/synthetic.py
```

Verify coverage:

1. **Kernel construction**: tested on `n=20, B=3` synthetic. Centring,
   trace norm, cross-kernel correctness.
2. **Likelihood**: numerical correctness vs brute force (`np.linalg.slogdet`)
   for at least 5 random `theta`. Gradient finite-diff agreement.
3. **Single-kernel REML**: variance recovered on R1 within `[0.5, 2.0]`.
4. **Multi-kernel REML**: R1, R2, R3 acceptance criteria from MATH_SPEC met.
5. **Multi-restart**: at least one test verifies that different starts
   converge to comparable optima OR that one restart dominates.
6. **Boundary detection**: synthetic with truth `sigma_b^2 = 0` triggers
   boundary diagnostic.
7. **No leakage**: SpyKernelizer test exists, validation rows never used.
8. **At-fixed-theta equivalence with mkR**: bullet 7 of TEST_PLAN.md.
9. **Sklearn API**: `clone(estimator)` works, `fit().predict()` returns
   correct shape, `score(X, y)` returns finite R².

Identify:
- missing tests (which axes of the algorithm are not exercised);
- tests that pass trivially (e.g. tolerance too loose);
- redundant tests;
- tests that depend on implementation details rather than behaviour.

Return findings ordered by severity with concrete additions/changes.
