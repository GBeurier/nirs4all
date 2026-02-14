# Scoring Definitions Reference

Technical reference for all score types in nirs4all: definitions, formulas, computation paths, naming, and storage.

---

## 1. Score Definitions

### 1.1 Regression Scores

#### RMSECV -- Root Mean Square Error of Cross-Validation

The standard chemometrics CV score. Computed from **pooled out-of-fold (OOF) predictions**: each fold model predicts only its own held-out validation samples, and all predictions are concatenated into a single vector before computing RMSE.

```
PRESS = sum_i (y_i - y_hat_i_OOF)^2
RMSECV = sqrt(PRESS / N)
```

Where `N` is the total number of OOF predictions across all folds.

- **Source**: `_compute_oof_cv_metric_indexed()` in `nirs4all/visualization/reports.py`
- **Stored as**: `val_score` of `fold_id="avg"` entry (computed in `base_model.py:_create_fold_averages()`)
- **Report column**: `RMSECV` (NIRS mode) / `CV_Score` (ML mode)

#### RMSEP -- Root Mean Square Error of Prediction

RMSE on an independent held-out test set, evaluated after refit on the full training data.

```
RMSEP = sqrt( (1/M) * sum_j (y_j - y_hat_j)^2 )
```

Where `M` is the number of test samples.

- **Source**: `ScoreCalculator.calculate()` via `evaluator.eval()` in `nirs4all/core/metrics.py`
- **Stored as**: `test_score` of `fold_id="final"` entry
- **Report column**: `RMSEP` (NIRS mode) / `Test_Score` (ML mode)

#### RMSEC -- Root Mean Square Error of Calibration

RMSE on the training set.

```
RMSEC = sqrt( (1/P) * sum_k (y_k - y_hat_k)^2 )
```

- **Stored as**: `train_score` on individual fold entries and `fold_id="final"` refit entries
- **Report column**: Not displayed in summary tables (available in detailed per-fold reports)

#### Mean Fold Test Score

Arithmetic mean of per-fold RMSE values computed on the test partition.

```
Mean_Fold_RMSEP = (1/K) * sum_k RMSE_k^test
```

Where `K` is the number of CV folds.

- **Source**: `_compute_cv_test_averages_indexed()` in `nirs4all/visualization/reports.py`
- **Report column**: `Mean_Fold_RMSEP` (NIRS mode) / `Mean_Fold_Test` (ML mode)

#### Weighted Mean Fold Test Score

Weighted average of per-fold RMSE values, weighted by fold sample count.

```
W_Mean_Fold_RMSEP = sum_k (n_k * RMSE_k^test) / sum_k n_k
```

- **Source**: `_compute_cv_test_averages_indexed()` in `nirs4all/visualization/reports.py`
- **Report column**: `W_Mean_Fold_RMSEP` (NIRS mode) / `W_Mean_Fold_Test` (ML mode)

#### Selection Score

The score that determined why a pipeline was chosen for refit. Its value depends on the selection criterion:

| Criterion | Value used | Description |
|-----------|-----------|-------------|
| `rmsecv` (default) | `val_score` from `fold_id="avg"` | Pooled OOF RMSE |
| `mean_val` | `(1/K) * sum_k RMSE_k^val` | Arithmetic mean of per-fold validation RMSEs |

- **Source**: `RefitConfig.selection_score` set in `nirs4all/pipeline/execution/refit/config_extractor.py`
- **Stored as**: `selection_score` on `fold_id="final"` entries
- **Report column**: `Selection_Score` (both modes)

### 1.2 Classification Scores

For classification tasks, the primary metric is **balanced accuracy** (`higher_is_better=True`).

| Score | Definition |
|-------|-----------|
| **CV balanced accuracy** | Pooled OOF balanced accuracy: concatenate all OOF predictions and compute `balanced_accuracy_score(y_true, y_pred)` |
| **Test balanced accuracy** | Balanced accuracy on held-out test set after refit |
| **Selection Score** | Same concept as regression -- the criterion that selected the model |

**Supported classification metrics**: `balanced_accuracy`, `accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`, `cohen_kappa`, `matthews_corrcoef`, `jaccard`.

Classification OOF metrics are computed by concatenating OOF labels and calling `evaluator.eval()` with the appropriate metric (not using squared errors).

- **Source**: `_compute_oof_cv_metric_indexed()` with `task_type="classification"` in `nirs4all/visualization/reports.py`
- **Report columns**: `CV_BalAcc`, `Test_BalAcc`, `Mean_Fold_BalAcc` (NIRS mode)

### 1.3 Important Distinctions

**RMSECV is not the mean of per-fold RMSEs.** These are different aggregations:

```
RMSECV = sqrt( PRESS / N )           -- pooled OOF
Mean_Fold_RMSE = (1/K) * sum_k RMSE_k  -- average of fold-level scalars
```

They are generally close but not identical (Jensen's inequality applies to the square root). RMSECV is the standard chemometrics definition and the default selection criterion.

---

## 2. Naming Conventions

### 2.1 NIRS Mode (default)

Uses standard chemometrics terminology. This is the default for all reports.

| Internal key | Display name | Definition |
|-------------|-------------|-----------|
| `cv_score` | RMSECV | Pooled OOF RMSE |
| `test_score` | RMSEP | Test RMSE after refit |
| `mean_fold_test` | Mean_Fold_RMSEP | Mean of per-fold test RMSEs |
| `wmean_fold_test` | W_Mean_Fold_RMSEP | Weighted mean of per-fold test RMSEs |
| `selection_score` | Selection_Score | Score used for refit selection |

### 2.2 ML Mode

Uses machine learning / deep learning terminology.

| Internal key | Display name |
|-------------|-------------|
| `cv_score` | CV_Score |
| `test_score` | Test_Score |
| `mean_fold_test` | Mean_Fold_Test |
| `wmean_fold_test` | W_Mean_Fold_Test |
| `selection_score` | Selection_Score |

### 2.3 Configuration

Set via the `report_naming` parameter in `nirs4all.run()`:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    report_naming="nirs",  # default; or "ml", "auto"
)
```

`"auto"` currently defaults to `"nirs"`.

- **Source**: `get_metric_names()` in `nirs4all/visualization/naming.py`
- **Wired through**: `nirs4all/api/run.py` -> `PipelineRunner` -> `TabReportManager`

### 2.4 Classification Naming

Classification metrics use the same mode system with metric-specific column names:

| NIRS mode | ML mode |
|-----------|---------|
| `CV_BalAcc` | `CV_Score` |
| `Test_BalAcc` | `Test_Score` |
| `Mean_Fold_BalAcc` | `Mean_Fold_Score` |

---

## 3. Score Flow Through the Pipeline

### 3.1 CV Phase (Pass 1)

```
For each pipeline variant:
  For each CV fold k:
    1. Train model on fold k's training samples
    2. Predict on fold k's val, test, and train partitions
    3. Compute per-partition RMSE via ScoreCalculator
    4. Store entry with fold_id=str(k), val_score, test_score, train_score

  After all folds:
    5. Concatenate OOF predictions (val partition from each fold)
    6. Compute RMSECV from concatenated OOF vector
    7. Store entry with fold_id="avg", val_score=RMSECV
    8. Store entry with fold_id="w_avg" (weighted ensemble of fold models)
```

Key source: `base_model.py:_create_fold_averages()`

### 3.2 Pipeline Completion

After all variants of a pipeline are evaluated:

```
1. Retrieve fold_id="avg" entry (RMSECV)
2. Store as pipeline best_val in workspace
```

Key source: `executor.py` (line ~253)

### 3.3 Refit Phase (Pass 2)

```
1. Rank completed pipelines by selection criterion (rmsecv or mean_val)
2. Extract top-K configs per criterion
3. Deduplicate across criteria (multi-criteria mode)
4. For each selected config:
   a. Retrain on full training data
   b. Predict on test set -> RMSEP
   c. Store entry with fold_id="final", test_score=RMSEP, val_score=None
   d. Attach selection_score from RefitConfig
```

Key sources:
- `nirs4all/pipeline/execution/refit/config_extractor.py` -- config selection and `RefitConfig` construction
- `nirs4all/pipeline/execution/refit/executor.py` -- refit execution and entry enrichment

### 3.4 Multi-Criteria Refit

When multiple selection criteria are configured (e.g., top-3 by `rmsecv` + top-3 by `mean_val`):

1. Each criterion independently selects its top-K pipelines
2. Results are merged with deduplication (a pipeline selected by both criteria appears once)
3. Each `RefitConfig` records which criterion selected it and the corresponding score
4. The `selection_scores` dict on `RefitConfig` stores all applicable criterion values

Key source: `extract_top_configs()` in `config_extractor.py`

### 3.5 Report Generation

The final summary table is built from `fold_id="final"` entries:

```
For each refit entry:
  RMSEP        <- entry.test_score (direct)
  RMSECV       <- _compute_oof_cv_metric_indexed() over OOF predictions
  Mean_Fold_RMSEP <- _compute_cv_test_averages_indexed() over fold test scores
  Selection_Score <- entry.selection_score
```

Each table includes a sorting indicator line (e.g., `"Sorted by: RMSEP (ascending -- lower is better)"`).

Key source: `generate_per_model_summary()` in `nirs4all/visualization/reports.py`

---

## 4. Score Storage

### 4.1 Prediction Entry Fields

Each prediction entry stores three nullable score fields:

| Field | Type | Description |
|-------|------|-------------|
| `val_score` | `float | None` | Validation partition score. `None` for refit entries (`fold_id="final"`) |
| `test_score` | `float | None` | Test partition score. `None` when no test set exists |
| `train_score` | `float | None` | Training partition score |

These are persisted as `NULL` when `None` (not coerced to `0.0`).

### 4.2 Special fold_id Values

| `fold_id` | Meaning |
|-----------|---------|
| `"0"`, `"1"`, ... | Individual CV fold results |
| `"avg"` | OOF-aggregated entry. `val_score` = RMSECV (pooled OOF). Test/train scores computed from concatenated OOF predictions |
| `"w_avg"` | Weighted ensemble of fold models. Val partition identical to `"avg"`. Test/train use weighted-average predictions |
| `"final"` | Refit entry. `val_score=None`. `test_score` = RMSEP on held-out test. Carries `selection_score` |

### 4.3 OOF Index Keys

OOF predictions are indexed for report computation using a 5-tuple key:

```
(dataset_name, config_name, model_name, step_idx, chain_id or branch_id)
```

The `chain_id` (with `branch_id` fallback) provides branch disambiguation, preventing OOF entries from different branches being mixed when the same model class appears at the same step index.

Key source: `_build_prediction_index()` in `nirs4all/visualization/reports.py`

### 4.4 Refit Entry Metadata

Refit prediction entries (`fold_id="final"`) carry:

| Field | Description |
|-------|-------------|
| `selection_score` | Score that selected this pipeline for refit |
| `refit_context` | Refit mode (e.g., `"standalone"`) |
| `config_name` | Includes refit suffix (e.g., `"MyPipeline_refit_rmsecvt3"`) |

---

## 5. Source File Reference

| File | Role in scoring |
|------|----------------|
| `nirs4all/controllers/models/base_model.py` | Per-fold score computation, OOF concatenation, `fold_id="avg"/"w_avg"` creation |
| `nirs4all/controllers/models/utilities.py` | `get_best_score_metric()`: selects primary metric per task type |
| `nirs4all/controllers/models/components/score_calculator.py` | Score computation via `evaluator.eval()` |
| `nirs4all/core/metrics.py` | Metric implementations (RMSE, balanced accuracy, etc.) |
| `nirs4all/pipeline/execution/executor.py` | Pipeline completion: stores `best_val` (RMSECV from avg fold) |
| `nirs4all/pipeline/execution/refit/config_extractor.py` | Refit selection: `RefitConfig` with `selection_score` and `selection_scores` |
| `nirs4all/pipeline/execution/refit/executor.py` | Refit execution: attaches `selection_score` to final entries |
| `nirs4all/data/predictions.py` | Prediction storage, ranking (`top()`), persistence (`flush()`) |
| `nirs4all/visualization/reports.py` | Report generation: `generate_per_model_summary()`, OOF/test metric computation |
| `nirs4all/visualization/naming.py` | Naming convention mapping (NIRS vs ML display names) |
