**Findings**

1. **Do not call Iter 2 “tuned benefit” yet.** MKM tuning is effectively no-op: BEER/TIC/MANURE are numerically identical to Iter 1, and ALPINE is slightly worse. mkR tuning has a mixed 4-row signal: BEER improves 6.1%, ALPINE/MANURE ~0.9%, TIC regresses 2.8%. That is not robust on `n=4`; treat `mkR-softmax_cv-snv-default-active15-tuned` as a candidate, not a conclusion. See Iter 2 rows [2, 6, 10, 14](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmark_runs/iter2_tuned_active/results.csv:2) vs Iter 1 rows [3, 8, 23, 28](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmark_runs/iter1_active15/results.csv:3).

2. **Active30 confirms the diminishing-returns concern.** It helps TIC for mkR, `1.197 -> 1.182`, but BEER collapses, `1.022 -> 1.115`, so the median gets worse. MKM active30 is basically flat except BEER worse, and BEER also fails REML convergence with many boundary components. That matches the Iter 1 warning that larger active sets mostly add collinear/noisy blocks once alignment is already near 1.0. See [CODEX_BACKLOG_ITER1.md](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/MKR/docs/CODEX_BACKLOG_ITER1.md:15) and Iter 2 rows [4-5, 8-9](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmark_runs/iter2_tuned_active/results.csv:4).

3. **Metric warning: focused rel-PLS medians look decent, but the plan target is rel-TabPFN-opt < 1.0.** Iter 2 medians vs TabPFN-opt are still roughly `1.25-1.30`, not converged against the stated goal. The plan’s stop target is explicit in [ITERATION_PLAN.md](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmarks/ITERATION_PLAN.md:3).

**Iter 3 Priority**

Finish the current batch, but prioritize conclusions this way:

1. Highest value: `screen_score_method` ablation on the two champions: `active15-kta` and `active15-blend` for `mkR-softmax_cv-snv` and `MKM-reml-asls`. This directly addresses the Iter 1 norm/KTA mismatch noted in [CODEX_BACKLOG_ITER1.md](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/MKR/docs/CODEX_BACKLOG_ITER1.md:5) and already queued in [ITERATION_PLAN.md](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmarks/ITERATION_PLAN.md:60).

2. Next: `mkR-softmax_cv-asls-default-active15`. BEER strongly favors ASLS under MKM, while mkR is better on TIC/MANURE; this is the most plausible untested cross-over.

3. Useful but lower: `mkR-softmax_cv-default-active15-tuned` because no-preproc mkR was the best ALPINE row in Iter 1.

4. Deprioritize more optimizer-budget runs and global active30/50. The MKM tuned result is already flat, and active30’s TIC gain is not portable.

**Convergence Rule**

Do not declare convergence from Iter 2 alone. Declare local convergence only after the current branch batch plus score-method ablation fails to improve the focused 4-dataset median by at least ~2% and fails to produce a clear hard-dataset rescue. Then run the top 2-3 variants on diverse10 before full 54. Move to full 54 only as validation of a short champion set, not as another broad search.

**Architecture**

Sparse softmax is the most justified next architectural change: it directly targets the active30 pattern, where extra capacity helps TIC but hurts BEER. POP-style per-component selection is worth considering after sparse/score ablations, but it is higher overfit risk and should use strict nested selection. Also add logging of selected block names, scores, and effective block count before deeper changes; without that, the next iteration will still be hard to diagnose.