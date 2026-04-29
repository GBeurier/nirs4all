# Codex Test Review Prompt

You are reviewing the test coverage of `bench/AOM_v0`.

Read:

- `docs/AOMPLS_MATH_SPEC.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/BENCHMARK_PROTOCOL.md`
- all files in `tests/`

Check whether tests cover:

1. Operator linearity, adjoint, covariance identity, and explicit matrix parity.
2. Identity-only equivalence to standard PLS.
3. Single fixed operator equivalence between materialized and fast engines.
4. Global vs per-component selection invariants.
5. `orthogonalization="transformed"` and `"original"`.
6. PLS1 and PLS2.
7. Classifier binary and multiclass probability behavior.
8. Leakage prevention in CV and fitted operators.
9. Torch/numpy parity.
10. Benchmark output schema and resumability.

Return missing tests as a prioritized checklist. For the top five missing tests,
write a short pytest-style pseudocode sketch.
