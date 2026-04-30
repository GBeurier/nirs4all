# MKM — Benchmark Protocol

## Cohorts

- **Smoke** — 3 datasets (ALPINE, AMYLOSE, BEER).
- **Extended** — 12 datasets sample.
- **Full** — `all57_cohort.csv` (re-using AOM-Ridge cohort file at
  `bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv`).

## Variants

- `MKM-reml-compact-none` (REML, compact bank, no branch preproc)
- `MKM-reml-compact-snv`
- `MKM-reml-compact-msc`
- `MKM-reml-compact-asls`
- `MKM-ml-compact-none` (ML for comparison; REML expected better)
- `MKM-reml-default-none` (100-op bank — needs alignment-prune step or top_k)

## Output CSV columns

```text
dataset_group, dataset, task, variant, status, error,
method, n_restarts, converged, boundary_components,
sigma2_block_<name_1>, sigma2_block_<name_2>, ...,
sigma2_residual,
relative_contribution_block_<name_1>, ...,
relative_contribution_residual,
log_likelihood, kernel_alignment_max, branch_preproc,
rmsep, mae, r2,
ref_rmse_ridge, ref_rmse_pls, ref_rmse_aomridge_mkl, ref_rmse_mkR,
relative_rmsep_vs_ridge, relative_rmsep_vs_pls,
relative_rmsep_vs_aomridge_mkl, relative_rmsep_vs_mkR,
fit_time_s, predict_time_s,
random_state, version
```

## Cross-validation / Test Split

- Same as mkR: train/test from cohort, `cv=5` SPXY for inner CV (used only
  if needed for branch_preproc selection or alpha-equivalent grid). MKM
  itself does **not** use inner CV — all hyperparameters are estimated by
  REML from training data only.

## Stop Conditions

- Median relative RMSEP versus mkR softmax_cv compact `<= 1.00` (matched
  performance) AND at least 25/57 wins.
- OR median `< 0.99` (true win).
- Otherwise: ship MKM as the **interpretation** layer — its scientific value
  is the per-block variance attribution, not necessarily prediction.

## Comparison targets

Same as mkR plus MKM is its own baseline for BLUP.
