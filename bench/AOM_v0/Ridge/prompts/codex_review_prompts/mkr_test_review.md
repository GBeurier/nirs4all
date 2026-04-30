# Codex Test Review: mkR

Read:

```text
bench/AOM_v0/Ridge/docs/MKR_TEST_PLAN.md
bench/AOM_v0/Ridge/tests/test_mkr_kernelizer.py
bench/AOM_v0/Ridge/tests/test_mkr_weights.py
bench/AOM_v0/Ridge/tests/test_mkr_estimator.py
bench/AOM_v0/Ridge/tests/test_mkr_equivalences.py
bench/AOM_v0/Ridge/tests/test_mkr_no_leakage.py
bench/AOM_v0/Ridge/tests/test_mkr_diagnostics.py
```

Verify coverage of:

1. Centring + trace normalisation invariants on synthetic.
2. Cross-kernel centring uses training-side moments only.
3. Weight strategies: uniform, manual (with negative input rejection),
   kta, softmax_cv.
4. Equivalence: uniform mkR with identity-only bank ≡ sklearn `Ridge`
   modulo constant alpha rescale.
5. softmax_cv on noiseless oracle synthetic converges close to oracle
   (within `< 0.1` L1 from oracle simplex).
6. Primal/dual agreement on `predict`.
7. SpyOperator no-leakage test asserts validation rows never enter
   operator fits, kernel construction, or trace normalisation.
8. Multi-output not supported (graceful error message).

Identify missing tests, trivial-pass tests, and tests testing implementation
detail. Return findings ordered by severity.
