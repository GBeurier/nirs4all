# Codex Publication Review: MKM

Read:

```text
bench/aom_v0/Multi-kernel/MkM/docs/MKM_MATH_SPEC.md
bench/aom_v0/Multi-kernel/MkM/publication/manuscript/main.tex
bench/aom_v0/Multi-kernel/MkM/publication/figures/   (list)
bench/aom_v0/Multi-kernel/MkM/publication/tables/    (list)
bench/aom_v0/Multi-kernel/MkM/benchmark_runs/full/results.csv  (if present)
```

Check honesty and completeness:

1. **Claims match results**: every numerical claim in the manuscript is
   traceable to a row in `results.csv`. No cherry-picking.
2. **Limitations stated**:
   - Variance components are not separately identifiable when
     `align(K_i, K_j) > 0.95`.
   - REML provides point estimates only; no confidence intervals shipped.
   - "Relative variance contribution" is not heritability except for
     genomic-relationship kernels.
3. **Comparisons fair**: MKM compared with the same train/test split as
   AOM-Ridge, AOM-PLS, TabPFN.
4. **Reproducibility**: commands to reproduce every figure and table.
5. **Figure correctness**: variance-contribution barplots use trace-normalised
   `sigma_b^2`, not raw scales.
6. **Equivalence with Ridge / mkR mentioned**: the manuscript should
   acknowledge that prediction-only MKM is essentially Ridge with REML
   weights and not over-claim novelty in prediction.

Return findings ordered by severity (high = misleading claim, medium =
missing limitation, low = stylistic).
