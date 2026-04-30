# Codex Roadmap Review: mkR (Multi-Kernel Ridge)

Review the mkR roadmap **before** code starts.

Read:

```text
bench/AOM_v0/Ridge/docs/MKR_IMPLEMENTATION_PLAN.md
bench/AOM_v0/Ridge/docs/MKR_MATH_SPEC.md
bench/AOM_v0/Ridge/docs/MKR_TEST_PLAN.md
bench/AOM_v0/Ridge/docs/MKR_BENCHMARK_PROTOCOL.md
bench/AOM_v0/Ridge/aomridge/mkl.py
bench/AOM_v0/Ridge/aomridge/kernels.py
bench/AOM_v0/Ridge/aomridge/estimators.py
```

The new mkR estimator extends the existing AOM-Ridge by:

1. Adding **centred + trace-normalised** block kernels (replacing `s_b` RMS scales).
2. Adding **explicit per-block weights** `eta_b` (uniform / manual / kta /
   softmax_cv).
3. Shipping a separate sklearn estimator `AOMMultiKernelRidge` instead of
   adding a selection mode to `AOMRidgeRegressor`.

Review for:

- **Mathematical correctness** — the centred/trace-normalised kernel formulas
  in MKR_MATH_SPEC.md.
- **Cross-kernel double-centring** — does the spec correctly use only
  training-side mu_b and nu_b? Identify any leakage path.
- **Phase ordering** — Phase 0 (synthetic), 1 (kernelizer), 2 (weights),
  3 (estimator), 4 (diagnostics), 5 (benchmark), 6 (preprocessing reintro),
  7 (full benchmark). Are there missing intermediate gates?
- **API ambiguity** — `weight_strategy` semantics, default `top_k`,
  interaction between `kernel_normalize="trace"` and explicit user-supplied
  weights.
- **Weak tests** — equivalence claims need quantified tolerances.
- **Benchmark overclaiming** — stop conditions; comparison to existing
  AOM-Ridge `mkl` mode (which already does kta-simplex weights, but with
  `s_b^2` scales not trace normalisation).

Return findings first, ordered by severity, with concrete fixes.
