# Codex Publication Review: mkR

Read:

```text
bench/aom_v0/Multi-kernel/MKR/docs/MKR_MATH_SPEC.md
bench/aom_v0/Multi-kernel/MKR/publication/manuscript/main.tex (if extended for mkR)
bench/aom_v0/Multi-kernel/MKR/benchmark_runs/full/results.csv (mkR variants only)
bench/aom_v0/Multi-kernel/MKR/publication/figures/   (mkR-specific figures)
bench/aom_v0/Multi-kernel/MKR/publication/tables/    (mkR-specific tables)
```

Check:

1. **Claims match results**: every numerical claim traceable to a row in
   `results.csv` (filtered to mkR variants).
2. **Comparison fair**: mkR vs existing AOM-Ridge `mkl` mode on the same
   train/test splits, the same operator banks, the same CV. Note any axis
   that changes.
3. **softmax_cv overfitting**: did the inner-CV-optimised weights overfit
   the inner CV at the cost of held-out performance? Report inner CV vs
   held-out gap.
4. **Alignment caveat**: kernels with `align > 0.95` flagged.
5. **Weight stability across folds** reported.
6. **Figures**: barplot of `eta_b` mean +/- std across folds; alignment
   heatmap; observed-vs-predicted; relative-RMSEP heatmap.

Return findings ordered by severity.
