# Benchmark Protocol

## Regression Oracle

Use `bench/tabpfn_paper/master_results.csv` as the regression reference table.
The AOM_v0 benchmark must be joinable with it by:

- `database_name`
- `dataset`
- `task`
- `model`
- `result_label`

Observed local schema:

```text
database_name, dataset, task, model, result_label, result_dir, status,
status_details, preprocessing_pipeline, RMSECV, RMSE_MF, RMSEP, MAE_test,
r2_test, search_mean_score, seed, n_splits, best_config_json,
best_model_params_json, best_fold_scores_json, trial_values_json,
search_results_path, best_config_path, final_predictions_path,
fold_predictions_path, rmse_mf_source, artifact_best_config_format,
artifact_search_results_format, artifact_final_predictions_format
```

Observed local profile:

- rows: 335
- unique regression splits: 61
- model dataset coverage:
  - TabPFN-Raw: 61
  - TabPFN-opt: 58
  - Catboost: 57
  - PLS: 54
  - Ridge: 54
  - CNN: 51

## Added AOM Columns

Add these columns after the master schema:

```text
aom_variant, backend, engine, selection, criterion, orthogonalization,
operator_bank, selected_operator_sequence_json, selected_operator_scores_json,
n_components_selected, max_components, fit_time_s, predict_time_s,
delta_rmsep_vs_master_pls, delta_rmsep_vs_tabpfn_raw,
delta_rmsep_vs_tabpfn_opt, run_seed, code_version, notes
```

## Regression Variants

Run all variants on the same cohort:

1. `PLS-standard-numpy`
2. `AOM-global-nipals-materialized-numpy`
3. `AOM-global-nipals-adjoint-numpy`
4. `POP-nipals-materialized-numpy`
5. `POP-nipals-adjoint-numpy`
6. `AOM-global-simpls-materialized-numpy`
7. `AOM-global-simpls-covariance-numpy`
8. `POP-simpls-materialized-numpy`
9. `POP-simpls-covariance-numpy`
10. `Superblock-simpls-numpy`
11. `AOM-soft-simpls-covariance-numpy` marked experimental
12. Torch equivalents for `nipals_adjoint`, `simpls_covariance`, and
    `superblock_simpls`

Smoke mode runs the same variants on 5 representative datasets. Full mode runs
all 61 available regression splits and is resumable.

## Cohort Building

`benchmarks/build_cohorts.py` must:

1. Read `master_results.csv`.
2. Extract the 61 unique regression dataset splits.
3. Resolve paths under `bench/tabpfn_paper/data/regression`.
4. Record unavailable or unparsable datasets with `status="skipped"` and a
   reason.
5. Write `bench/AOM_v0/benchmarks/cohort_regression.csv`.

Do not silently drop datasets.

## Evaluation Protocol

For every dataset and model variant:

1. Load the dataset using the same nirs4all folder parser path used by the
   TabPFN scripts when possible.
2. Preserve the original train/test split if present.
3. If no split exists, create deterministic SPXY or stratified split with
   `random_state=42`.
4. Perform operator and component selection only on calibration data.
5. Fit final model on calibration data.
6. Evaluate final untouched test set.
7. Save fold predictions and final predictions.
8. Append one result row immediately for resumability.

Metrics:

- `RMSECV`: pooled out-of-fold RMSE.
- `RMSEP`: final test RMSE.
- `MAE_test`.
- `r2_test`.
- wall-clock fit and predict times.

Statistical analysis:

- Per-database aggregation before global ranking.
- Friedman test and Nemenyi critical difference diagram.
- Wilcoxon signed-rank against PLS, TabPFN-Raw, and TabPFN-opt.
- Bootstrap confidence intervals for median delta RMSEP.

## Classification Cohort

There is no classification `master_results.csv` in the current local files.
Build a classification cohort by scanning:

```text
bench/tabpfn_paper/data/classification
```

The TabPFN paper draft states a 15-dataset classification benchmark with
balanced accuracy. Therefore:

- `cohort_classification.csv` must include every parseable classification split.
- If more than 15 candidate folders are found, keep all parseable splits and
  mark the 15 that match the paper protocol when detectable.
- Use balanced accuracy as the primary metric.
- Use macro-F1, log loss, and ECE as secondary metrics.

Classification models:

1. `PLS-DA-standard`
2. `AOM-PLS-DA-global-nipals-adjoint`
3. `POP-PLS-DA-nipals-adjoint`
4. `AOM-PLS-DA-global-simpls-covariance`
5. `POP-PLS-DA-simpls-covariance`
6. Torch equivalents where supported.

## Benchmark Commands

Smoke:

```bash
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_smoke_benchmark.py
```

Full regression:

```bash
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_aompls_benchmark.py \
  --task regression \
  --cohort bench/AOM_v0/benchmarks/cohort_regression.csv \
  --master bench/tabpfn_paper/master_results.csv \
  --workspace bench/AOM_v0/benchmark_runs/regression_full \
  --seeds 0,1,2,3,4 \
  --n-jobs -1
```

Full classification:

```bash
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_aompls_benchmark.py \
  --task classification \
  --cohort bench/AOM_v0/benchmarks/cohort_classification.csv \
  --workspace bench/AOM_v0/benchmark_runs/classification_full \
  --seeds 0,1,2,3,4 \
  --n-jobs -1
```

Summary:

```bash
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/summarize_results.py \
  --results bench/AOM_v0/benchmark_runs/regression_full/results.csv \
  --master bench/tabpfn_paper/master_results.csv \
  --out bench/AOM_v0/publication/tables
```
