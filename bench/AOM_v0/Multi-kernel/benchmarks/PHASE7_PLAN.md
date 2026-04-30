# Phase 7 — Full 54-Dataset Multi-Kernel Benchmark

## Cohort

Two cohorts:

1. **Iteration cohort** — `bench/AOM_v0/Multi-kernel/benchmark_runs/curated11_cohort.csv`
   (11 hand-picked diverse datasets, n from 81 to 2925, p from 125 to
   2177). Used for hypothesis-validation and rapid iteration. Ranges:

   | Dataset | n_train | p |
   |---------|---------|---|
   | An_spxyG70_30_byCultivar_MicroNIR (GRAPEVINE) | 81 | 125 |
   | DIESEL_bp50_246_b-a | 113 | 401 |
   | DIESEL_bp50_246_hla-b | 133 | 401 |
   | Malaria_Sporozoite_229_Maia | 138 | 2151 |
   | WOOD_N_402_Olale | 216 | 1038 |
   | All_manure_CaO_SPXY_strat_Manure_type | 343 | 1003 |
   | All_manure_P2O5_SPXY_strat_Manure_type | 343 | 1003 |
   | Fv_Fm_grp70_30 (FUSARIUM) | 351 | 2177 |
   | Beef_Marbling_RandomSplit (BEEFMARBLING) | 554 | 331 |
   | ta_groupSampleID_stratDateVar_balRows (BERRY) | 912 | 2101 |
   | Chla+b_spxyG_block2deg (ECOSIS) | 2925 | 196 |

2. **Final cohort** — `bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv`,
   restricted to 54 datasets with `status == "ok"` (3 dropped for size
   / NaNs). Run AT THE END once iteration is converged.

## Variants

Selected from the smoke3_branches results, 8 variants:

| family | strategy | branch | rationale |
|--------|----------|--------|-----------|
| Ridge | raw | none | sklearn baseline (no AOM, no kernel) |
| mkR | softmax_cv | none | smoke median winner (rel-PLS 0.95) |
| mkR | softmax_cv | snv | strongest on BEER (rel-PLS 0.38) |
| mkR | softmax_cv | msc | second-strongest on BEER, slightly different to SNV |
| mkR | softmax_cv | asls | asymmetric baseline correction |
| MKM | reml | none | likelihood-based reference, beats PLS on BEER |
| MKM | reml | asls | smoke best on AMYLOSE (rel-PLS 1.02) |
| BLUP | reml | asls | per-block decomposition + best-on-AMYLOSE |

## Cohort split (chunks of 12 for staged validation)

We run in three staged chunks so that we can monitor cost / accuracy
trade-offs and pivot if needed:

1. **Chunk A — extended12** (datasets 1-12 of cohort): smoke at
   moderate scale, with `n_jobs=4`. Expected wall time ~30-60 min.
2. **Chunk B — datasets 13-30**: parallel run with `n_jobs=4`.
3. **Chunk C — datasets 31-54**: parallel run with `n_jobs=4`.

Total: ~5-10 hours wall time; ~25-50 hours of CPU time.

The runner appends to a single `results.csv` per workspace and skips
already-completed `(dataset, variant)` pairs on resume.

## Stop conditions

- **Win condition**: median rel-PLS < 0.95 across the 54-dataset cohort
  for at least one variant. (Smoke median was 0.95; we hope to keep it
  on the larger cohort.)
- **Match condition**: median rel-PLS in [0.95, 1.05]. Acceptable; we
  document where each variant wins.
- **Loss condition**: median rel-PLS > 1.10. Investigate failures,
  consider alternative variants or datasets to drop.

## Outputs

- `results.csv` (raw, one row per `(dataset, variant)`).
- `summary_per_variant.csv` — median across 54 datasets per variant.
- `summary_per_dataset.csv` — best variant per dataset.
- `relative_rmsep_pivot.csv` — full pivot table (variant × dataset).
- Figures (per-variant bars, per-dataset heatmap, kernel alignment).

## Reproducibility

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_full.py \
  --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/all54 \
  --variants Ridge-raw mkR-softmax_cv mkR-softmax_cv-snv mkR-softmax_cv-msc \
             mkR-softmax_cv-asls MKM-reml MKM-reml-asls BLUP-reml-asls \
  --n-jobs 4
```

## Risk register

- **Long-running fits** (n_train > 1000): the alpha-grid CV inside
  softmax_cv scales as O(n^3) per fold. For BERRY (n=1434) this could
  take 10-15 min per variant. Mitigation: smaller alpha grid for large
  n; or skip if predict_time exceeds wall budget.
- **Convergence failures**: REML may not converge on pathological
  cohorts. The runner records `status="error"` and continues.
- **Memory**: each fit holds B kernels of size n×n. For n=2000, B=9,
  that's 9 × (2000)^2 × 8 bytes = ~290 MB per worker. With n_jobs=4,
  ~1.2 GB total. Acceptable.

## Next steps after Phase 7

- Generate critical-difference (CD) diagrams (Nemenyi post-hoc on the
  per-dataset variant rankings).
- Build per-individual contribution figure for BLUP on a representative
  dataset.
- Compare against TabPFN-opt and CNN-NICON references to assess
  competitiveness.
- Compile the final manuscript figures + tables.
