# Cross-Validation

Cross-validation (CV) is the mechanism nirs4all uses to estimate how well a
pipeline will perform on unseen data. Instead of training once and hoping for
the best, the library trains the same pipeline multiple times on different
subsets of the data and measures performance on the held-out portions.

---

## Why Cross-Validate?

A model can memorise its training data. If you evaluate it on the same data it
was trained on, the score will be misleadingly good. Cross-validation solves
this by ensuring that every sample is predicted by a model that never saw it
during training.

```
Fold 0:  [====TRAIN====] [=VAL=]         -> predictions for VAL samples
Fold 1:  [===TRAIN===]   [=VAL=] [TRAIN] -> predictions for VAL samples
Fold 2:  [=VAL=]         [====TRAIN====] -> predictions for VAL samples
                                            -------------------------
                                            All samples predicted once
```

After all folds are done, every training sample has been predicted exactly
once -- by a model that was not trained on it. These are called **out-of-fold
(OOF)** predictions.

---

## How It Works in nirs4all

Cross-validation is driven by a **splitter** step in the pipeline. The
splitter defines how many folds to create and how to assign samples.

```python
from sklearn.model_selection import ShuffleSplit

pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.2),
    {"model": PLSRegression(n_components=10)},
]
```

When this pipeline is executed:

1. Pre-splitter steps (SNV) are applied to the entire training partition.
2. The splitter writes 5 fold definitions into the dataset.
3. For each fold (0 through 4):
   - Post-splitter transforms are fitted on the fold's train subset and
     applied to train, val, and test.
   - The model is fitted on the fold's train subset.
   - Predictions are made on val, test, and optionally train.
4. OOF predictions from all 5 folds are pooled.
5. Metrics are computed on the pooled OOF predictions.

---

## Out-of-Fold (OOF) Predictions

The pooled OOF predictions form the basis for ranking pipeline variants. Each
sample is predicted by the model that did not see it, so the resulting score
is an honest estimate of generalisation performance.

:::{note}
OOF is short for "out-of-fold". It means the sample was in the validation
portion of the fold, not in the training portion. OOF predictions are
sometimes called "cross-validated predictions" in the literature.
:::

OOF predictions are also the building block for **stacking**: when you use
`{"merge": "predictions"}`, the OOF predictions from each branch become
features for a meta-model. Because they are out-of-fold, there is no data
leakage. See {doc}`branching_and_merging` for details.

---

## The Test Partition

The **test** partition is entirely separate from cross-validation. It is never
used to select models or tune hyperparameters. After each fold, the fold's
model predicts on the test set, and after refit (see below), the final model
also predicts on the test set.

This gives you three leak-free test estimates:

| Score       | How it is computed                               |
|-------------|--------------------------------------------------|
| Ens_Test    | Average of K fold-model predictions on test      |
| W_Ens_Test  | Quality-weighted average of fold predictions      |
| RMSEP       | Single refit model's prediction on test (see below) |

---

## Scoring and Metrics

The default primary metric is **MSE** (mean squared error), but results are
reported in several forms to match chemometrics conventions:

| Metric   | Full name                              | Meaning                                |
|----------|----------------------------------------|----------------------------------------|
| RMSECV   | Root Mean Square Error of Cross-Validation | RMSE of pooled OOF predictions      |
| RMSEP    | Root Mean Square Error of Prediction   | RMSE of refit model on test set        |
| RMSEC    | Root Mean Square Error of Calibration  | RMSE of model on its own training data |
| R2       | Coefficient of determination           | How much variance is explained         |

For classification tasks, the primary metric is **balanced accuracy**, with
additional metrics like F1, precision, recall, and ROC-AUC.

:::{tip}
RMSECV is computed from the pooled OOF vector, not as the mean of per-fold
RMSEs. These two quantities are close but not identical because the square
root is a non-linear function.
:::

---

## Refit

After cross-validation has ranked all pipeline variants, the best variant is
**refitted** on the entire training partition (all folds combined). This
produces a single model that has seen all available training data -- the model
you would deploy to production.

```
Phase 1 -- Cross-Validation
  For each variant:
    Train on K folds, collect OOF predictions
    Rank by RMSECV

Phase 2 -- Refit
  Take the top variant
  Retrain on ALL training samples (no folds)
  Evaluate on the test set -> RMSEP
```

The refit model typically performs better than any individual fold model
because it has more training data. The comparison between RMSECV and RMSEP
tells you whether the model benefits from the additional data.

### Controlling refit

```python
# Default: refit top 1
result = nirs4all.run(pipeline=pipeline, dataset=dataset)

# Disable refit (faster exploration)
result = nirs4all.run(pipeline=pipeline, dataset=dataset, refit=False)

# Refit top 3 variants
result = nirs4all.run(
    pipeline=pipeline, dataset=dataset,
    refit={"top_k": 3, "ranking": "rmsecv"},
)
```

See {doc}`/user_guide/scoring_and_refit` for multi-criteria refit, refit
strategies for stacking pipelines, and detailed score interpretation.

---

## result.best vs result.final

The `RunResult` object distinguishes between CV results and refit results:

| Property          | Scope                                         |
|-------------------|-----------------------------------------------|
| `result.best_score` | Best pooled OOF score from CV (RMSECV)     |
| `result.cv_best`    | Best CV entry (all fold details)            |
| `result.final`      | Refit entry (`fold_id="final"`)             |
| `result.final_score`| Refit model's test score (RMSEP)            |
| `result.top(n)`     | Top N variants, ranked by CV or final score |

`result.best` tells you which pipeline configuration generalises best during
cross-validation. `result.final` tells you how the deployed model performs on
fresh data.

---

## Summary

```
Dataset
  |
  +-- train partition --+-- Fold 0 train --> fit model --> predict val (OOF)
  |                     +-- Fold 1 train --> fit model --> predict val (OOF)
  |                     +-- ...
  |                     +-- Pool OOF --> RMSECV --> rank variants
  |                     |
  |                     +-- Refit best on all train --> final model
  |
  +-- test partition  -----> evaluate final model --> RMSEP
```

---

## Next Steps

- {doc}`pipelines` -- understand the steps that precede and follow the
  splitter.
- {doc}`datasets` -- learn about partitions and repetition-aware splitting.
- {doc}`branching_and_merging` -- see how OOF predictions enable stacking.
- {doc}`/user_guide/scoring_and_refit` -- complete scoring and refit
  reference.
