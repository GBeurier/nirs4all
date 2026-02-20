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

#### MF_Val -- Mean Fold Validation Score

Arithmetic mean of per-fold validation RMSEs.

```
MF_Val = (1/K) * sum_k RMSE_k^val
```

- **Source**: `_compute_mf_val_indexed()` in `nirs4all/visualization/reports.py`
- **Report column**: `MF_Val` (NIRS mode) / `MF_CV` (ML mode)
- **Selection use**: Available as a refit selection criterion (`ranking: "mean_val"`)

#### Ens_Test -- Ensemble Test Score

Test RMSE from the simple average of K fold models' predictions on the test set. Each fold model predicts on the test set, and predictions are averaged before computing RMSE.

```
y_hat_ens = (1/K) * sum_k y_hat_k^test
Ens_Test = sqrt( (1/M) * sum_j (y_j - y_hat_ens_j)^2 )
```

This is NOT RMSEP (which is computed after refit on full training data). Ens_Test shows how well the K fold models perform as an ensemble, without retraining.

- **Source**: `test_score` of `fold_id="avg"` entry
- **Report column**: `Ens_Test` (NIRS mode) / `Ens_Test_Score` (ML mode)

#### W_Ens_Test -- Weighted Ensemble Test Score

Same as Ens_Test but using quality-weighted averaging of fold predictions (better-performing folds contribute more).

```
y_hat_wens = sum_k (w_k * y_hat_k^test)
W_Ens_Test = sqrt( (1/M) * sum_j (y_j - y_hat_wens_j)^2 )
```

Where `w_k` are fold quality weights (computed from validation scores, sum to 1).

- **Source**: `test_score` of `fold_id="w_avg"` entry
- **Report column**: `W_Ens_Test` (NIRS mode) / `W_Ens_Test_Score` (ML mode)

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

### 1.2 Classification Scores

For classification tasks, the primary metric is **balanced accuracy** (`higher_is_better=True`).

| Score | Definition |
|-------|-----------|
| **CV balanced accuracy** | Pooled OOF balanced accuracy: concatenate all OOF predictions and compute `balanced_accuracy_score(y_true, y_pred)` |
| **Test balanced accuracy** | Balanced accuracy on held-out test set after refit |

**Supported classification metrics**: `balanced_accuracy`, `accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`, `cohen_kappa`, `matthews_corrcoef`, `jaccard`.

Classification OOF metrics are computed by concatenating OOF labels and calling `evaluator.eval()` with the appropriate metric (not using squared errors).

- **Source**: `_compute_oof_cv_metric_indexed()` with `task_type="classification"` in `nirs4all/visualization/reports.py`
- **Report columns**: `CV_BalAcc`, `Test_BalAcc`, `Ens_Test_BalAcc` (NIRS mode)

### 1.3 Important Distinctions

**RMSECV is not the mean of per-fold RMSEs.** These are different aggregations:

```
RMSECV = sqrt( PRESS / N )           -- pooled OOF
MF_Val = (1/K) * sum_k RMSE_k^val    -- average of fold-level scalars
```

They are generally close but not identical (Jensen's inequality applies to the square root). RMSECV is the standard chemometrics definition and the default selection criterion.

**Ens_Test is not RMSEP.** Ens_Test uses the average of K fold models' predictions on the test set (CV-phase ensemble). RMSEP uses a single model retrained on all training data after refit. They measure fundamentally different things: ensemble performance vs. single-model refit performance.

**Three leak-free test predictions:**
1. **Ens_Test** -- average of K fold models' predictions on test
2. **W_Ens_Test** -- quality-weighted average of K fold models' predictions on test
3. **RMSEP** -- prediction of the retrained model on test

**Two selection scores (CV phase):**
1. **RMSECV** -- pooled OOF RMSE (default selection criterion)
2. **MF_Val** -- mean of per-fold validation RMSEs (alternative selection criterion)

---

## 2. Naming Conventions

### 2.1 NIRS Mode (default)

Uses standard chemometrics terminology. This is the default for all reports.

| Internal key | Display name | Definition |
|-------------|-------------|-----------|
| `cv_score` | RMSECV | Pooled OOF RMSE |
| `test_score` | RMSEP | Test RMSE after refit |
| `ens_test` | Ens_Test | Ensemble test from avg of K fold models |
| `w_ens_test` | W_Ens_Test | Weighted ensemble test from K fold models |
| `mean_fold_cv` | MF_Val | Mean of per-fold validation RMSEs |
| `selection_score` | Selection_Score | Score used for refit selection |

### 2.2 ML Mode

Uses machine learning / deep learning terminology.

| Internal key | Display name |
|-------------|-------------|
| `cv_score` | CV_Score |
| `test_score` | Test_Score |
| `ens_test` | Ens_Test_Score |
| `w_ens_test` | W_Ens_Test_Score |
| `mean_fold_cv` | MF_CV |
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
| `Ens_Test_BalAcc` | `Ens_Test_Score` |

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
    7. Store entry with fold_id="avg", val_score=RMSECV, test_score=Ens_Test
    8. Store entry with fold_id="w_avg", test_score=W_Ens_Test
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
2. Extract top-K configs per criterion (independent selection, no cross-dedup)
3. For each selected config:
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

1. Each criterion independently selects its top-K pipelines, skipping models already selected by prior criteria
2. Each criterion fills its full quota independently (guarantees `sum(top_k)` unique models)
3. Each `RefitConfig` records which criterion selected it and the corresponding score
4. The `selection_scores` dict on `RefitConfig` stores all applicable criterion values
5. In the final scores table, a star `*` marks the best model (by RMSEP) per criterion

Key source: `extract_top_configs()` in `config_extractor.py`

### 3.5 Report Generation

**Final scores table** (from `fold_id="final"` entries):

```
For each refit entry:
  RMSEP      <- entry.test_score (direct)
  Ens_Test   <- avg fold's test_score (looked up from CV phase)
  W_Ens_Test <- w_avg fold's test_score (looked up from CV phase)
  RMSECV     <- _compute_oof_cv_metric_indexed() over OOF predictions
  MF_Val     <- _compute_mf_val_indexed() over per-fold val_scores
```

The table is sorted by RMSEP (ascending for regression, descending for classification).

**Top 30 CV chains table** (from `fold_id="avg"` entries):

```
For each chain:
  RMSECV    <- avg fold val_score (pooled OOF)
  MF_Val    <- mean of per-fold val_scores
  Ens_Test  <- avg fold test_score (ensemble of K fold models)
  W_Ens_Test <- w_avg fold test_score (weighted ensemble)
  f0, f1... <- individual fold val_scores
```

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
| `"avg"` | OOF-aggregated entry. `val_score` = RMSECV (pooled OOF). `test_score` = Ens_Test (ensemble of K fold models on test) |
| `"w_avg"` | Weighted ensemble of fold models. `test_score` = W_Ens_Test (weighted ensemble on test). Val partition OOF identical to `"avg"` |
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
