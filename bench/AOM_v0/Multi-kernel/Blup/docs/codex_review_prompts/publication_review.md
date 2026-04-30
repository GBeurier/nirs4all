# Codex Publication Review: BLUP

Read:

```text
bench/aom_v0/Multi-kernel/Blup/docs/BLUP_MATH_SPEC.md
bench/aom_v0/Multi-kernel/Blup/publication/manuscript/main.tex
bench/aom_v0/Multi-kernel/Blup/publication/figures/   (list)
bench/aom_v0/Multi-kernel/Blup/publication/tables/    (list)
bench/aom_v0/Multi-kernel/Blup/benchmark_runs/full/results.csv  (if present)
```

Check:

1. **Framing honesty**: BLUP prediction equals MKM prediction; the value
   of BLUP is **per-block decomposition**, not predictive accuracy.
   Manuscript must state this clearly.
2. **E-BLUP terminology**: clarify that variances are estimated, so
   technically E-BLUP is what we ship.
3. **Per-block claim**: contributions are interpretable only when blocks
   are reasonably independent (`align < 0.9`). Manuscript must show
   alignment matrix.
4. **Per-individual contributions**: example figure shows top deviating
   samples with stacked-bar contributions.
5. **Reproducibility**: every figure & table reproducible from
   `results.csv` via shipped scripts.

Return findings ordered by severity.
