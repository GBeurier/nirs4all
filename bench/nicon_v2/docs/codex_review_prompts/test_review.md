# Codex Test Review Prompt — nicon_v2

Review the unit and integration tests under `bench/nicon_v2/tests/`. Do NOT propose changes to `nirs4all`.

## What to check

1. **Coverage of correctness contracts.**
   * Forward / backward shape tests on cohorts with diverse spectrum lengths (Beer 576, DIESEL 401, AMYLOSE 1154, ECOSIS 2151).
   * Loss decreases on a 1-batch toy fit-loop (smoke test).
   * Preprocessing parity vs scipy (SG, SNV) and statsmodels (MSC).
   * Augmentation seed-determinism and label correctness (mixup interpolates both x and y).
   * Conformal calibration achieves nominal coverage on synthetic homoscedastic data.
   * `worker_init_fn` produces reproducible DataLoader output.

2. **Anti-leak tests.**
   * Spy operator (à la `bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py::SpyOperator`) confirms validation/test never enters fit-time statistics.

3. **Benchmark schema regression test.**
   * A test reads `benchmark_runs/smoke/results.csv` and validates the column set against `BENCHMARK_PROTOCOL.md`.

4. **Hygiene.**
   * Tests run in <2 minutes on CPU (smoke).
   * No reliance on internet / GPU for unit tests.

## Output format

Same as math/code review: per-finding severity + location + issue + fix + final verdict.
