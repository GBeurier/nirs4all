# Master Driver Prompt — nicon_v2

This file is the executable contract for the iterative loop. It is consumed at the start of every iteration to remind the agent of the rules.

## Loop

```
1. Identify the next weakness (docs/WEAKNESS_ANALYSIS.md, ranked).
2. Formulate a falsifiable hypothesis (docs/HYPOTHESES.md → next H_n).
3. Implement: code under nicon_v2/, tests under tests/, benchmark under benchmarks/.
4. Run benchmark; record results in benchmark_runs/<iter>/results.csv.
5. Update docs/IMPLEMENTATION_LOG.md (append-only) with:
     - date
     - iteration name (Hn — short title)
     - files changed
     - hypothesis test result (accepted / rejected, p-value or wins)
     - bench results (median delta vs PLS / vs AOM-Ridge / vs TabPFN, win-rate)
     - findings + next step
6. Codex review (run codex:rescue with docs/codex_review_prompts/code_review.md
   or math_review.md, depending on the iteration's nature).
7. Apply codex findings (mark each one fixed in the log).
8. Re-run benchmark. If win-rate meets the target in BENCHMARK_PROTOCOL.md, move
   to publication scaffolding; otherwise loop.
```

## Hard rules

* **Never edit `nirs4all/`.** Only files under `bench/nicon_v2/` may be created or modified.
* **Append-only** for `docs/IMPLEMENTATION_LOG.md` and `docs/HYPOTHESES.md` (new sections only).
* **Codex review** is mandatory between iterations; the agent must wait for review output before the next iteration commit.
* **Tests must pass** before benchmark execution: `pytest bench/nicon_v2/tests -q`.
* **No mocked data** in benchmark runs; only real cohort datasets.
* **No silent failures** in the benchmark CSV: every row must include either a successful metric set or `status=ERROR` with the traceback message.

## Stopping criterion

The loop has **two success tiers** so that a publishable contribution does not depend on clearing the most ambitious target:

* **Leaderboard success.** Median relative rmsep of `nicon_v2-best` vs `aom_ridge_curated_best` (the per-dataset best AOM-Ridge variant) ≤ −0.02 across the 39-dataset curated cohort, paired Wilcoxon p < 0.05, ≥ 50 % wins per dataset, and no dataset regresses > 10 % vs any cohort reference. This is the primary target.
* **Scientific success.** `nicon_v2-best` beats `NICON-baseline` and `DECON-baseline` on ≥ 75 % of the curated cohort with paired Wilcoxon p < 0.05. Even if leaderboard success is not reached, this is enough for a publishable CNN-redesign contribution.

The manuscript reports both tiers honestly.

## Hard budget

Iteration ends (whether or not a tier is cleared) at the earliest of:
* 8 phases lapsed,
* 12 GPU-hours per phase on the curated cohort exceeded,
* 5-member ensemble exceeded,
* 2 consecutive phases producing non-significant improvements (futility stop).
