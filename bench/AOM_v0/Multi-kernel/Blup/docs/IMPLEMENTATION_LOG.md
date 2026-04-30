# BLUP Implementation Log

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

## 2026-04-30: Planning Documents Created

Created the AOM-BLUP documentation scaffold under `bench/aom_v0/Multi-kernel/Blup`.

Key decisions:

- AOM-BLUP wraps AOM-MKM and adds **per-block prediction decomposition**.
- E-BLUP is what we ship (variances estimated by REML, not assumed known).
- `predict_components(X)` returns `{"fixed": ..., "<op_b>": ..., "total": ...}`.
- Decomposition sum identity is a primary test: `sum components == predict`.
- BLUP delegates fitting to MKM; no duplicated REML logic.
- `alpha_dual = V^-1 (y - X_f beta_hat)` is precomputed at fit time.
- Highly-aligned blocks (`align > 0.95`) flagged as non-identifiable
  individually but their sum is identifiable.

Next steps:

- Codex roadmap review.
- Phase 0 synthetic ground-truth.
- Phase 1 estimator scaffold (delegating to MKM).
- Phase 2 decomposition module.
- Phase 3 diagnostics.
- Phase 4 smoke benchmark.

## 2026-04-30: Phases 0-5 complete (Claude pilot, after consolidation)

Files added:

- `blup/__init__.py`, `estimator.py`.
- `tests/conftest.py`, `test_blup_decomposition.py` (10 tests passing).
- `docs/PLAN_REVIEW_CORRECTIONS.md` (Codex roadmap review).
- `docs/CODEX_BACKLOG_2026-04-30.md` (Codex math/code review round 2).

Phase summary:

- **Phase 1** (estimator) — `AOMMultiKernelBLUP` is a thin sklearn-style
  wrapper around `AOMMultiKernelMixedModel`. Delegates `fit`, `predict`;
  adds `predict_components` and `train_decompose`.
- **Phase 2** (decomposition) — `predict_components(X)` returns
  `{"fixed": ndarray, "random": OrderedDict[block_name, ndarray],
  "total": ndarray}`. Total accumulated INSIDE the loop independently of
  dict keys to support duplicate block names.
- **Phase 3** (diagnostics, deferred) — `contribution_table` returns a
  pandas long-format DataFrame for plotting / heatmaps. Shrinkage diag
  formula reserved for Round 2 per Codex feedback.
- **Phase 4** (smoke) — runs via the unified
  `bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py`.
  Predictions match MKM by construction, ALPINE rel-PLS=0.99, AMYLOSE
  rel-PLS=1.17, BEER rel-PLS=0.62 (same as MKM, by design).

Tests: 10 / 10 passing.

```
PYTHONPATH=bench/AOM_v0/Multi-kernel/MKR:bench/AOM_v0/Multi-kernel/MkM:bench/AOM_v0/Multi-kernel/Blup \
  pytest bench/AOM_v0/Multi-kernel/Blup/tests -q
```

## 2026-04-30: Codex review round 2 + applied fixes

Codex math/code review (`/tmp/codex_blup_math.md`):

- HIGH (applied): duplicate block names broke the decomposition identity.
  Random components stored in `OrderedDict` keyed only by `block_name`;
  duplicates collapsed and the `total` reconstructed from the dict missed
  one contribution. **Fix**: accumulate `total` inside the loop
  independently of dict keys; disambiguate keys with `__k` suffix when a
  block_name repeats.
- MEDIUM (deferred to backlog): test coverage extension, shared
  `_predict_parts(X, X_fixed_test)` helper.

Findings logged in `docs/CODEX_BACKLOG_2026-04-30.md`.

## 2026-04-30: Phase 6 — branch_preproc parameter added

`AOMMultiKernelBLUP.__init__` accepts `branch_preproc` and forwards it to
the wrapped MKM. `predict_components` applies the same branch transform
to test data using `self._mkm_._branch_` so cross kernels are consistent
with the training kernel space.

Tests: 10 / 10 still passing. Decomposition identity preserved with
branches enabled.
