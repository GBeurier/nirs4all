# mkR Implementation Log

This log is append-only. Each phase should add:

```text
date
phase
files changed
tests run
Codex review prompt used
findings fixed
findings deferred
```

## 2026-04-30: Phases 0-5 complete (Claude pilot, post-consolidation)

The mkR work originally lived in `bench/AOM_v0/Ridge/aomridge/`. After
the user's consolidation directive, all NEW mkR code was relocated to
`bench/AOM_v0/Multi-kernel/MKR/aomridge/`. The original Ridge package
is preserved as-is.

Files (under `bench/AOM_v0/Multi-kernel/MKR/`):

- `aomridge/kernelizer.py` — `AOMKernelizer` (centred + trace-normalised
  block kernels, fold-local stats, batch-invariant cross-kernel,
  `zero_trace_policy` guard).
- `aomridge/weights.py` — `uniform_weights`, `manual_weights`,
  `kta_simplex_weights`, `softmax_cv_weights` (simplex-only, KL-to-uniform
  regulariser).
- `aomridge/mkr_estimator.py` — `AOMMultiKernelRidge` sklearn estimator
  with `weight_strategy in {uniform, manual, kta, softmax_cv}` and
  `branch_preproc in {none, snv, msc, asls, osc, emsc1}`.
- `tests/synthetic_mkr.py` — R1/R2/R3 synthetic data generators.
- `tests/test_mkr_kernelizer.py`, `test_mkr_weights.py`,
  `test_mkr_estimator.py`, `test_mkr_no_leakage.py`,
  `test_ridge_mkl.py` — 48 tests, all passing.
- Plus the Ridge support files needed for mkR (kernels.py, mkl.py,
  solvers.py, selection.py, branches.py, classification.py, cv.py,
  estimators.py, preprocessing.py).

## 2026-04-30: Codex roadmap review (round 1) + applied fixes

Codex roadmap review (`/tmp/codex_mkr_roadmap.md`) returned 4 high + 5
medium severity findings. All applied to specs and code:

- H1 (softmax_cv leakage) — caveat documented; v2 fold-local kernelizer
  reserved.
- H2 (cross-kernel notation) — explicit `r_*`, `c_train`, `nu_b` in spec;
  batch-invariance test added (passes).
- H3 (uniform-equiv claim) — corrected: uniform-trace mkR ≡ AOM-Ridge
  `"rms"` superblock with rescaled alpha (not `"none"`).
- H4 (mkl baseline) — benchmark protocol updated.
- M1 (simplex always) — encoded in API.
- M2-M5 — applied in code & docs.

See `docs/MKR_PLAN_REVIEW_CORRECTIONS.md`.

## 2026-04-30: Codex round 2 (math + code review) + applied fixes

Codex math/code review (`/tmp/codex_mkr_math.md`):

- HIGH (deferred — documented as v1 caveat): inner CV in `softmax_cv`
  uses precomputed full-training kernels. The kernelizer's centring /
  trace stats are computed on the outer training set, so inner-validation
  rows do affect those moments before being held out. The outer test set
  is still held out. **Action**: documented as v1 limitation; v2 will
  refit the kernelizer per inner fold.
- HIGH (applied): Spy no-leakage tests were vacuous because operators are
  deep-copied (`clone_operator_bank`) before fitting, so the original
  `spy` instance never received any record. **Fix**: shared class-level
  log dict; cloned spies now write to the same log. Tests now assert
  the log is non-empty AND that observed centred-row signatures all match
  the centred training rows (under training mean). 4/4 no-leakage tests
  pass non-vacuously.
- MEDIUM (backlog): `AOMKernelizer.get_params` omits `zero_trace_policy`
  / `zero_trace_threshold`; `warn_keep` mode amplifies near-zero traces;
  `one_se_rule` SE not fold-wise; final Cholesky doesn't use adaptive
  jitter helper.

## 2026-04-30: Phase 5 — smoke benchmark on 3 datasets

Variants run: Ridge-raw, mkR-uniform, mkR-kta, mkR-softmax_cv, MKM-reml,
BLUP-reml. From `benchmark_runs/smoke3/summary_per_variant.csv`:

| Variant | median rel-PLS | median rel-Ridge | median fit-time |
|---------|----------------|-------------------|------------------|
| **mkR-softmax_cv** | **0.95** ✓ | 1.00 | 39 s |
| BLUP-reml | 0.99 | 1.05 | 46 s |
| MKM-reml | 0.99 | 1.05 | 54 s |
| mkR-kta | 1.17 | 1.18 | 18 s |
| mkR-uniform | 1.35 | 1.37 | 22 s |
| Ridge-raw | 2.37 | 2.40 | 0.1 s |

mkR-softmax_cv beats PLS on the smoke median by 5%. Per-dataset:
ALPINE 0.95 (mkR wins), BEER 0.62 (MKM wins), AMYLOSE 1.17 (no win
without preprocessing).

## 2026-04-30: Phase 6 — branch_preproc parameter added

`AOMMultiKernelRidge.__init__` accepts `branch_preproc` and applies the
fitted branch transformer to X **before** the kernelizer. `predict` and
`predict_dual` apply the stored branch transformer to X_test. Tests:
48 / 48 still passing (default `"none"` is a no-op).

Smoke benchmark with branch variants (`smoke3_branches/`) is running;
results CSV is written row-by-row for resumability.

Tests:

```
.venv/bin/pytest bench/AOM_v0/Multi-kernel/MKR/tests -q
# 48 passed
```
