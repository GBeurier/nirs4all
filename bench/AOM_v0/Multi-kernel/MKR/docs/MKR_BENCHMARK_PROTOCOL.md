# mkR — Benchmark Protocol

## Cohorts

- **Smoke** — 3 datasets (ALPINE, AMYLOSE, BEER). Used during Phase 1-3.
- **Extended** — 12 datasets (sample of `all57_cohort.csv`). Used during Phase 4-5.
- **Full** — `all57_cohort.csv` (54 OK datasets). Used in Phase 7.

## Variants

Strict-linear (no branch preproc):
- `mkR-uniform-compact-trace` (baseline)
- `mkR-kta-compact-trace` (closed-form)
- `mkR-softmax_cv-compact-trace` (gradient on inner CV)
- `mkR-softmax_cv-default-trace` (default 100-op bank, requires top_k pruning)

Branch-preproc (Phase 6+):
- `mkR-softmax_cv-compact-asls`
- `mkR-softmax_cv-compact-snv`
- `mkR-softmax_cv-compact-msc`

## Output CSV columns

```text
dataset_group, dataset, task, variant, status, error,
weight_strategy, kernel_normalize, alpha, alpha_at_boundary, grid_expansions,
cv_min_score, top_k, branch_preproc,
eta_block_<name_1>, eta_block_<name_2>, ..., (one column per block in bank)
kernel_alignment_max, kernel_alignment_mean, weight_stability_mean,
rmsep, mae, r2,
ref_rmse_ridge, ref_rmse_pls, ref_rmse_aomridge_mkl,
relative_rmsep_vs_ridge, relative_rmsep_vs_pls,
relative_rmsep_vs_aomridge_mkl,
fit_time_s, predict_time_s,
random_state, version
```

The CSV is appended row-by-row for resumability (mirrors AOM-Ridge benchmark
runner pattern).

## Cross-validation

- `cv=5` with `cv_kind="kfold"` for smoke.
- `cv=5` with `cv_kind="spxy"` for extended/full.
- Optional `cv_kind="spxy_repeated"` (3 repeats) for stability ablation
  (Phase 7 only).

## Test split

Use the train/test split provided by `all57_cohort.csv` (Kennard-Stone or
SPXY70/30 typical). Final RMSEP / MAE / R² computed on test split.

## Stop Conditions

mkR variants are considered to "win" if the **median relative RMSEP versus
AOM-Ridge mkl baseline** is strictly less than 0.99 over the full 57
cohort, with at least 30/57 wins (per-dataset).

Comparison targets (from existing benchmarks, AOM_v0/Summary.md):

| Method | Median rel-RMSEP vs PLS | Wins (out of 57) |
|--------|--------------------------|------------------|
| AOM-PLS championship (ASLS+compact CV-5) | 0.960 | 42 |
| AOM-Ridge mkl-compact | TBD (run to fill) | TBD |
| TabPFN-opt | reference | reference |

If mkR beats one or more of these, document and proceed to publication. If
not, log it as a negative-but-informative result (still publishable).
