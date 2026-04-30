# Codex Code Review Prompt — nicon_v2

You are reviewing the code in a research bench at `bench/nicon_v2/`. The project does NOT modify `nirs4all` itself; never propose changes outside `bench/nicon_v2/`.

## Scope

* `bench/nicon_v2/nicon_v2/` (entire package) — `datasets.py`, `metrics.py`, `preprocessing.py`, `augmentation.py`, `training.py`, `uncertainty.py`, `models/*.py`.
* `bench/nicon_v2/benchmarks/run_*.py` — runners.
* `bench/nicon_v2/tests/test_*.py` — unit tests.

## What to check

1. **Hygiene.**
   * No mutable default arguments, no global state, no cyclic imports.
   * Type hints on public API, ruff-clean (line ≤ 220), no unused imports.
   * Idiomatic PyTorch (no `.cuda()` hard-codes, device passed via context).

2. **Correctness.**
   * `nicon_v2.datasets.load_cohort` returns deterministic splits given the seed.
   * Train/val/test never overlap; physical-sample CV honoured when `Sample_ID` exists.
   * Metric computations are on the *original* y scale (not the y-processing scale).
   * The benchmark runner is *resumable* (skip rows where `(dataset, variant, seed, fold) ∈ existing CSV ∧ status = OK`) and never silently overwrites previous results.

3. **Reproducibility.**
   * Seeds propagate to: numpy, torch, CUDA (`cudnn.deterministic = True; benchmark = False`), DataLoader workers (`worker_init_fn`).
   * Each result row records python/torch/CUDA versions, git SHA, host, and full hyper-parameters JSON.

4. **Dead code / over-engineering.**
   * Per CLAUDE.md rules: no helpers/utilities/abstractions for one-time operations; no error handling for impossible scenarios; no half-finished stubs.

5. **No-leak guarantee.**
   * Augmentation never touches validation/test.
   * Preprocessing fitted on train only, applied on test (no test-set statistics in fit_transform pipelines).
   * Conformal calibration uses a separate calibration split.

## Output format

For each finding produce:
* **Severity** (High / Medium / Low / Info).
* **Location** (file:line).
* **Issue** (≤ 80 words).
* **Suggested fix** (≤ 80 words).

Plus a single one-line **verdict**: `LGTM` / `LGTM_WITH_NITS` / `NEEDS_FIXES` / `BLOCKED`.

## Reference materials

* `bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md` (analogous workflow).
* `bench/nicon_v2/docs/IMPLEMENTATION_PLAN.md` (phase plan).
* `bench/nicon_v2/docs/BENCHMARK_PROTOCOL.md` (CSV schema + acceptance gates).
