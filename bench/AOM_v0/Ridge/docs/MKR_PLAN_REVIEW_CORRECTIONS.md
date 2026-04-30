# mkR Plan Review Corrections (Codex roadmap review, 2026-04-30)

Source: `/tmp/codex_mkr_roadmap.md` (Codex roadmap review of
`MKR_IMPLEMENTATION_PLAN.md`, `MKR_MATH_SPEC.md`, `MKR_TEST_PLAN.md`,
`MKR_BENCHMARK_PROTOCOL.md`).

## High-Severity Findings (must fix before Phase 1 acceptance)

### H1. softmax_cv inner-CV leakage path

**Issue**: Passing globally centred + trace-normalised `K_blocks` into the
inner CV would leak inner-validation rows through `mu_b`, `nu_b`, `tau_b`,
feature scaling, and any branch preprocessing. The kernels know the
inner-validation rows even before the inner split.

**Fix**: ``softmax_cv_weights`` must accept either
- a `kernelizer_factory` + raw `X`, raw `y`, and a CV splitter, **refitting
  the kernelizer fold-locally**, OR
- pre-built kernels with a documented disclaimer that this is "outer-CV-only"
  protection (acceptable when outer test set is held out and kernelizer is
  fitted on outer training only).

The default v1 ships option 2 (fast, simple) with a disclaimer; an "exact"
mode (option 1) is reserved for Round 2.

### H2. Cross-kernel centring needs explicit notation

**Issue**: Formula in `MKR_MATH_SPEC.md:63` is right but ambiguous when
implemented as a batched `H_test K H_train` (which would use test-batch
moments instead of training moments).

**Fix**: name the centring quantities explicitly in the spec:

```text
r_*     = (1/n) K_*^raw 1_n              # per-test-row mean (computed from cross kernel
                                          # against training)
c_train = (1/n) 1_n^T K_train^raw         # training row mean (= column mean)
nu      = (1/n^2) 1_n^T K_train^raw 1_n   # training global mean

K_*^c   = K_*^raw - 1_* c_train - r_* 1_n^T + nu 1_* 1_n^T
```

**Test addition**: batch-invariance — `transform([x])[0]` must equal the
row of `transform([x, x2, ...])` corresponding to `x` (within fp tolerance).

### H3. Uniform-equivalence claim was wrong for multi-block "none"

**Issue**: `MKR_MATH_SPEC.md:200` (Equivalence claim 1) said uniform mkR with
strict-linear bank ≡ AOM-Ridge superblock with `block_scaling="none"`. This
is only true when there is one block or all block traces are equal.

**Fix**: For strict-linear centred blocks:

```text
K_rms = p * sum_b tau_b K_b_raw = p * B * K_uniform_trace
```

So uniform-trace mkR ≡ AOM-Ridge `block_scaling="rms"` superblock with
alpha rescaled by `p * B`. Tests should target this. Keep
`block_scaling="none"` only as a one-block / identity-only sanity check.

### H4. Existing mkl baseline was mischaracterised

**Issue**: The roadmap says existing `selection="mkl"` applies `s_b` scales,
but `estimators.py:719` and `selection.py:997` actually pin MKL scales to
ones.

**Fix**: Update `MKR_BENCHMARK_PROTOCOL.md` to say mkR is benchmarked
against the actual `AOMRidgeRegressor(selection="mkl")` raw-KTA baseline.
KTA weights are scale-invariant, so a separate `mkl_rms` baseline is not
useful unless implemented intentionally.

## Medium-Severity Findings

### M1. API semantics ambiguity

- `MKR_MATH_SPEC.md:81` says "we do not require `sum eta_b = 1` in general";
  manual/softmax/tests assume simplex. Pick simplex for v1.
- `weight_init` vs manual weights: `weight_init` is the warm-start for
  softmax_cv only; manual weights are an entirely different strategy.
- `weight_top_k=None` means no pruning; require explicit pruning for
  default-100-op bank.
- Align `scoring="rmse_pooled"` with the existing accepted name
  `mse_pooled` in `estimators.py:201`.

### M2. Coefficient formula symbol mismatch

`MKR_IMPLEMENTATION_PLAN.md:132` uses `X^T`; should be `Xc^T` (centred).

### M3. Phase gates missing

Insert before benchmarks:
- Fold-local kernelizer CV gate (before weights phase).
- Alpha-grid boundary expansion gate.
- Softmax complexity / top-k feasibility gate before default-bank runs.
- Baseline parity against current AOM-Ridge `mkl`.
- Branch-preproc smoke only after strict-linear smoke.

### M4. Test tolerances and zero-trace handling

- Replace "≈", "close", "tags active blocks" with thresholds:
  - Row/column centring relative residual `< 1e-10`.
  - Trace relative error `< 1e-10`.
  - Primal/dual prediction `rtol=1e-8, atol=1e-8`.
  - Equivalence kernels `rtol=1e-10` after alpha/kernel scaling.
- Handle zero-trace blocks: raise or mark inactive; do not amplify noise
  with `n / eps`.
- Stability score `1 - sigma / (mean + eps)` can be negative; clamp to
  `[0, 1]` or redefine.

### M5. Benchmark stop conditions

`MKR_BENCHMARK_PROTOCOL.md:55` says "30/57"; cohort says 54 OK datasets.
Use actual denominator. Add paired uncertainty (sign test or paired
bootstrap CI). Treat median rel-RMSEP `< 0.99` as a screening signal,
not a publication criterion.

## Action Items

| # | Severity | Action | Status |
|---|----------|--------|--------|
| H1 | High | Refactor `softmax_cv_weights` API to accept kernelizer factory + raw data, OR add disclaimer for the simplified path | applied to v1 (simplified path with disclaimer) |
| H2 | High | Add explicit r_*/c_train notation to spec; add batch-invariance test | applied to spec |
| H3 | High | Fix equivalence claim: uniform trace mkR ≡ AOM-Ridge "rms" superblock | applied to spec |
| H4 | High | Update benchmark protocol description of existing mkl baseline | applied to protocol |
| M1 | Med | Clarify simplex always; rename scoring; document pruning | applied to spec |
| M2 | Med | Fix `Xc^T` in implementation plan | applied to plan |
| M3 | Med | Add intermediate phase gates | applied to plan |
| M4 | Med | Tighten test tolerances; handle zero-trace blocks | applied to test plan |
| M5 | Med | Use actual cohort size; add paired uncertainty | applied to benchmark protocol |
