# Understanding Scores, Evaluation, and Refit

This page explains how nirs4all evaluates pipeline performance across
cross-validation, ensemble scoring, and refit. It clarifies the meaning of
every score you see in reports, how refit works, and how to control it.

---

## The Two Phases of a Run

Every `nirs4all.run()` call proceeds in two phases:

1. **Cross-Validation (CV) phase** -- Each pipeline variant is evaluated using
   K-fold cross-validation. Out-of-fold (OOF) predictions are collected and
   scored. This phase ranks variants and selects winner(s).

2. **Refit phase** -- The winning variant(s) are retrained on the **full
   training set** (all folds combined) to produce a single production model.
   This model is evaluated on the held-out test set.

```
Phase 1: Cross-Validation
  For each variant:
    For each fold k (k = 0..K-1):
      Train on fold k's training samples
      Predict on val, test, train partitions
      Store per-fold scores

    After all folds:
      Pool OOF predictions -> RMSECV / CV balanced accuracy
      Average fold test predictions -> Ens_Test
      Weighted average -> W_Ens_Test

Phase 2: Refit
  Select top variant(s) by ranking criterion
  For each selected variant:
    Retrain model on ALL training samples
    Predict on test set -> RMSEP / Test balanced accuracy
    Store as fold_id="final"
```

---

## Score Definitions (Regression)

### RMSECV -- Root Mean Square Error of Cross-Validation

The standard chemometrics CV score. Computed from **pooled out-of-fold
predictions**: each fold model predicts only its held-out validation samples,
and all predictions are concatenated into a single vector before computing
RMSE.

```
PRESS = sum_i (y_i - y_hat_i_OOF)^2
RMSECV = sqrt(PRESS / N)
```

This is the **default selection criterion** for refit.

> **Important**: RMSECV is NOT the mean of per-fold RMSEs. These are
> different aggregations (Jensen's inequality applies to the square root).
> They are generally close but not identical.

### MF_Val -- Mean Fold Validation Score

Arithmetic mean of per-fold validation RMSEs:

```
MF_Val = (1/K) * sum_k RMSE_k^val
```

Available as an alternative refit selection criterion (`ranking: "mean_val"`).

### Ens_Test -- Ensemble Test Score

Test RMSE from the simple average of K fold models' predictions on the test
set:

```
y_hat_ens = (1/K) * sum_k y_hat_k^test
Ens_Test = sqrt( (1/M) * sum_j (y_j - y_hat_ens_j)^2 )
```

This is NOT RMSEP. It shows how well the K fold models perform as an ensemble
**without retraining**.

### W_Ens_Test -- Weighted Ensemble Test Score

Same as Ens_Test but using quality-weighted averaging of fold predictions
(better-performing folds contribute more).

### RMSEP -- Root Mean Square Error of Prediction

RMSE on the held-out test set, evaluated **after refit** on the full training
data:

```
RMSEP = sqrt( (1/M) * sum_j (y_j - y_hat_j)^2 )
```

This is the score of the final production model. Only available after refit.

### RMSEC -- Root Mean Square Error of Calibration

RMSE on the training set. Available on individual fold entries and on the final
refit entry.

### Summary: Three Leak-Free Test Predictions

| Score | Source | When available |
|-------|--------|----------------|
| **Ens_Test** | Average of K fold models on test | After CV |
| **W_Ens_Test** | Weighted average of K fold models on test | After CV |
| **RMSEP** | Single refit model on test | After refit |

### Summary: Two Selection Scores (CV Phase)

| Score | Definition | Use |
|-------|-----------|-----|
| **RMSECV** | Pooled OOF RMSE | Default refit selection |
| **MF_Val** | Mean of per-fold validation RMSEs | Alternative selection |

---

## Score Definitions (Classification)

For classification tasks, the primary metric is **balanced accuracy**
(`higher_is_better=True`).

| Score | Definition |
|-------|-----------|
| **CV balanced accuracy** | Pooled OOF balanced accuracy: concatenate all OOF predictions and compute `balanced_accuracy_score(y_true, y_pred)` |
| **Test balanced accuracy** | Balanced accuracy on held-out test set after refit |

Supported classification metrics: `balanced_accuracy`, `accuracy`, `f1_score`,
`precision`, `recall`, `roc_auc`, `cohen_kappa`, `matthews_corrcoef`,
`jaccard`.

---

## Naming Conventions

Scores appear in reports using two naming modes:

### NIRS Mode (default)

Standard chemometrics terminology.

| Internal key | Display name | Definition |
|-------------|-------------|-----------|
| `cv_score` | RMSECV | Pooled OOF RMSE |
| `test_score` | RMSEP | Test RMSE after refit |
| `ens_test` | Ens_Test | Ensemble test from avg of K fold models |
| `w_ens_test` | W_Ens_Test | Weighted ensemble test |
| `mean_fold_cv` | MF_Val | Mean of per-fold validation RMSEs |

### ML Mode

Machine learning terminology.

| Internal key | Display name |
|-------------|-------------|
| `cv_score` | CV_Score |
| `test_score` | Test_Score |
| `ens_test` | Ens_Test_Score |
| `w_ens_test` | W_Ens_Test_Score |
| `mean_fold_cv` | MF_CV |

Set via `report_naming`:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    report_naming="nirs",  # default; or "ml", "auto"
)
```

---

## Refit: How It Works

### What Refit Does

After cross-validation identifies the best pipeline variant(s), **refit**
retrains the winning model on the **entire training set** (all folds combined).
This produces a single model that leverages all available training data -- the
model you would deploy to production.

The refit model is then evaluated on the held-out test set to produce RMSEP
(or test balanced accuracy for classification).

### Why Refit Matters

During cross-validation, each fold model only sees a fraction of the training
data (e.g., 80% in 5-fold CV). The refit model sees 100% of the training data,
which typically improves generalization. The comparison between RMSECV (CV
performance) and RMSEP (refit test performance) tells you whether the model
benefits from more data.

### Refit Strategies

nirs4all automatically selects the appropriate refit strategy based on pipeline
topology:

| Pipeline type | Strategy | Description |
|--------------|----------|-------------|
| Simple pipeline | `simple_refit` | Retrain the single best variant on full training data |
| Stacking / merge predictions | `stacking_refit` | Retrain base models, regenerate OOF-style predictions, retrain meta-model |
| Competing branches | `competing_branches_refit` | Retrain winning branch model on full data |
| Separation branches | `separation_refit` | Retrain per-group models on full per-group data |

### Controlling Refit

The `refit` parameter in `nirs4all.run()` controls refit behavior:

```python
# Default: refit top 1 by RMSECV
result = nirs4all.run(pipeline=pipeline, dataset=dataset)

# Disable refit
result = nirs4all.run(pipeline=pipeline, dataset=dataset, refit=False)

# Refit top 3 by RMSECV
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    refit={"top_k": 3, "ranking": "rmsecv"},
)

# Refit top 3 by mean fold validation score
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    refit={"top_k": 3, "ranking": "mean_val"},
)

# Multi-criteria refit: top 3 by RMSECV + top 1 by mean_val
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    refit=[
        {"top_k": 3, "ranking": "rmsecv"},
        {"top_k": 1, "ranking": "mean_val"},
    ],
)
```

### Refit Parameters

For deep learning or custom models, refit-specific training parameters can
override CV training parameters:

```python
pipeline = [
    {"model": MyModel(), "train_params": {"epochs": 50, "lr": 0.01},
     "refit_params": {"epochs": 100, "warm_start": True}},
]
```

`refit_params` merges on top of `train_params`: unspecified keys inherit from
`train_params`, specified keys override.

---

## Understanding the Report Tables

A typical run produces two summary tables:

### Final Scores Table

Shows refit results for each selected model (`fold_id="final"`):

| Column | Source |
|--------|--------|
| RMSEP | Refit model's test score |
| RMSECV | Pooled OOF RMSE from CV phase |
| MF_Val | Mean of per-fold validation RMSEs |
| Ens_Test | Average of K fold models on test |
| W_Ens_Test | Weighted average of K fold models on test |

Sorted by RMSEP (ascending for regression, descending for classification).

### Top CV Chains Table

Shows the best cross-validation pipeline variants (`fold_id="avg"`):

| Column | Source |
|--------|--------|
| RMSECV | Pooled OOF RMSE |
| MF_Val | Mean of per-fold validation scores |
| Ens_Test | Ensemble test score |
| W_Ens_Test | Weighted ensemble test score |
| f0, f1, ... | Individual fold validation scores |

---

## Score Storage

### fold_id Values

| `fold_id` | Meaning |
|-----------|---------|
| `"0"`, `"1"`, ... | Individual CV fold results |
| `"avg"` | OOF-aggregated entry. `val_score` = RMSECV, `test_score` = Ens_Test |
| `"w_avg"` | Weighted ensemble. `test_score` = W_Ens_Test |
| `"final"` | Refit entry. `val_score=None`, `test_score` = RMSEP |

### Accessing Scores Programmatically

```python
result = nirs4all.run(pipeline=pipeline, dataset=dataset)

# Best CV score (RMSECV)
result.best_score

# Best refit test score (RMSEP)
result.best_rmse  # alias for best test score

# Top N results (CV scope)
result.top(5)

# Top N results (refit/final scope)
result.top(5, score_scope="final")

# CV-only best
result.cv_best

# Final refit result
result.final
result.final_score
```

The `score_scope` parameter controls which entries are considered:

| Scope | Behavior |
|-------|----------|
| `"cv"` | Only cross-validation entries |
| `"final"` | Only refit entries (`fold_id="final"`) |
| `"mix"` | Finals first, then CV (default) |
| `"flat"` | All entries treated equally |

---

## Practical Guidelines

### When to Disable Refit

- **Quick exploration**: When iterating on preprocessing choices and you only
  care about CV performance, `refit=False` saves time.
- **No test set**: If your dataset has no held-out test partition, refit cannot
  produce RMSEP. The refit model is still trained but only RMSEC is available.

### Interpreting Score Differences

- **RMSECV << RMSEP**: The model may be overfitting to training data patterns
  that don't generalize. Consider simpler models or more regularization.
- **RMSECV >> RMSEP**: Unusual but possible with small datasets or when the
  test set happens to be "easier" than the CV validation folds.
- **Ens_Test < RMSEP**: The ensemble of fold models outperforms the single
  refit model. This can happen when model variance is high -- consider
  increasing regularization or using ensemble-based deployment.
- **Ens_Test ≈ RMSEP**: Good sign. The refit model performs comparably to the
  ensemble, confirming stable model behavior.

### Multi-Criteria Refit

When using multiple selection criteria, each criterion independently selects
its top-K pipelines. Duplicates are skipped (if a pipeline is selected by
RMSECV, it won't be selected again by mean_val). The final scores table marks
the best model per criterion with a star (`*`).

This is useful when different ranking criteria identify genuinely different
model strengths (e.g., RMSECV favors models with consistent fold performance,
while mean_val may favor models with lower average error).
