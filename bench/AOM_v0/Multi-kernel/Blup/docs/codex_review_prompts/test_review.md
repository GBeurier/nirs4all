# Codex Test Review: BLUP

Read:

```text
bench/aom_v0/Multi-kernel/Blup/docs/TEST_PLAN.md
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_decomposition.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_predict_total.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_no_leakage.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_synthetic.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_at_fixed_theta_equiv_mkr.py
```

Verify:

1. Decomposition sum identity tested for **train**, **test**, and **random
   unseen** data.
2. Per-block contribution norm tested on R1 (active vs inactive ratio).
3. Correlated-blocks test (R3) verifies sum-of-pair recovers truth.
4. No-leakage test: `predict_components(X_test)` uses only `(X_train_stats,
   alpha_dual, sigma_b^2, beta_fixed)` stored at fit time. Spy test
   asserts no recompute against training data.
5. Equivalence with mkR at fixed theta verified.
6. Sklearn `clone(BLUP_estimator)` preserves all hyperparameters.

Identify missing, trivial, or redundant tests. Return findings ordered by
severity.
