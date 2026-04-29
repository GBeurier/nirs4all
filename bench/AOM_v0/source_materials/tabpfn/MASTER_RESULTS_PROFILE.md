# master_results.csv Profile

Generated from local file:

```text
bench/tabpfn_paper/master_results.csv
```

Observed on 2026-04-27:

- rows: 335
- columns: 29
- unique regression dataset splits: 61

Columns:

```text
database_name
dataset
task
model
result_label
result_dir
status
status_details
preprocessing_pipeline
RMSECV
RMSE_MF
RMSEP
MAE_test
r2_test
search_mean_score
seed
n_splits
best_config_json
best_model_params_json
best_fold_scores_json
trial_values_json
search_results_path
best_config_path
final_predictions_path
fold_predictions_path
rmse_mf_source
artifact_best_config_format
artifact_search_results_format
artifact_final_predictions_format
```

Model coverage:

```text
TabPFN-Raw    61 datasets
TabPFN-opt    58 datasets
Catboost      57 datasets
PLS           54 datasets
Ridge         54 datasets
CNN           51 datasets
```

Status profile:

```text
TabPFN-Raw: 61 ok rows
All other listed models: partial rows in this local CSV
```

Important consequence for AOM_v0:

- AOM/POP benchmark rows must retain the master schema.
- Missing reference rows for PLS/TabPFN-opt must be handled by left joins and
  explicit `NaN` deltas, not by dropping datasets.
- The full AOM/POP regression cohort is the 61 unique dataset split list.
