# Understanding Predictions

This page explains the core concepts behind predictions in nirs4all. Understanding these concepts will help you work effectively with training results, model selection, and deployment.

## What Is a Prediction?

A **prediction** is a record that captures everything about one model evaluated on one fold and partition. When you run a pipeline with 3-fold cross-validation on a single model, nirs4all produces multiple prediction records -- one per fold per partition (train, val, test), plus averaged summaries.

Each prediction record contains:

- **Identity**: model name, model class, dataset name, fold ID, partition
- **Scores**: val_score, test_score, train_score (the primary metric), plus a nested `scores` dictionary with all computed metrics (RMSE, R2, MAE, etc.) per partition
- **Arrays**: y_true, y_pred (and y_proba for classification) -- the actual and predicted values for visualization and detailed analysis
- **Context**: preprocessing chain summary, branch info, exclusion stats, number of samples and features, hyperparameters
- **Chain link**: a reference to the trained chain (the complete preprocessing-to-model path) that produced this prediction

## Chains

A **chain** is the complete, ordered sequence of fitted steps that were executed during training. It captures:

- Every preprocessing transformer (fitted scaler, SNV, etc.) with its artifacts
- The model step and its fitted artifacts per fold
- The order of operations

Chains are the unit of **export** and **replay**. When you export a model or predict on new data, nirs4all loads the chain and replays each step in order -- applying fitted transformers, then running the model.

For cross-validation, the chain stores artifacts per fold. A chain with 3-fold CV has three fitted model artifacts (one per fold) plus shared preprocessing artifacts that were fitted on the full training set.

```
Chain: MinMaxScaler -> SNV -> PLSRegression(10)
       |                      |
       shared artifact        fold_0 artifact
       (fitted scaler)        fold_1 artifact
                              fold_2 artifact
```

## Partitions

Predictions are organized by **partition** -- the subset of data used for evaluation:

| Partition | Description | Purpose |
|-----------|-------------|---------|
| `train` | Samples used to fit the model in this fold | Overfitting diagnostics -- if train scores are much better than val, the model overfits |
| `val` | Held-out samples for this fold (cross-validation split) | Primary ranking metric -- models are ranked by validation score |
| `test` | Independent test set (not used during training or fold splitting) | Final performance estimate -- reported in publications and used for deployment decisions |

During cross-validation, each fold defines its own train/val split. The test partition, if present, is evaluated once per fold using the fold's model.

The **val_score** is the primary metric used for ranking models. When you call `result.best_rmse` or `result.top(5)`, the ranking is based on validation performance by default.

## Scores vs. Arrays

Predictions store both **scalar scores** and **full arrays**:

**Scalar scores** (`val_score`, `test_score`, `train_score`) are used for fast ranking and filtering. They represent the primary metric (e.g., RMSE) for each partition. The nested `scores` dictionary contains all computed metrics:

```python
{
    "val": {"rmse": 0.12, "r2": 0.95, "mae": 0.08},
    "test": {"rmse": 0.14, "r2": 0.93, "mae": 0.09},
    "train": {"rmse": 0.05, "r2": 0.99, "mae": 0.03},
}
```

**Arrays** (`y_true`, `y_pred`, `y_proba`) store the actual predicted values for each sample. These are used for:

- Actual-vs-predicted plots
- Residual analysis
- Confusion matrices (classification)
- Custom metric computation
- Aggregation by sample groups

Arrays are stored separately from scores and are loaded on demand for efficiency.

## Prediction Lifecycle

The full lifecycle of a prediction, from training to deployment:

```
1. TRAIN
   nirs4all.run(pipeline, dataset)
       |
       v
2. STORE
   For each pipeline x fold x partition:
       - Compute predictions (y_pred) and scores
       - Save prediction record to store.duckdb
       - Save arrays (y_true, y_pred) to store.duckdb
       - Save chain (fitted artifacts) to store.duckdb + artifacts/
       |
       v
3. QUERY
   result.best_rmse          # Best validation RMSE
   result.top(10)            # Top 10 by val_score
   result.filter(model_name="PLSRegression")
       |
       v
4. EXPORT
   result.export("model.n4a")   # Bundle with chain + artifacts
       |
       v
5. PREDICT
   nirs4all.predict("model.n4a", X_new)
       - Load chain from bundle
       - Replay preprocessing steps
       - Average predictions across folds
       - Return PredictResult
```

## Workspace Storage

All prediction data is stored in a DuckDB database (`store.duckdb`) inside the workspace directory:

```
workspace/
    store.duckdb          # All structured data (runs, pipelines, chains, predictions)
    artifacts/            # Flat content-addressed binaries (fitted models, transformers)
        ab/abc123.joblib
    exports/              # User-triggered exports (on demand)
```

The database contains seven tables:

| Table | Contents |
|-------|----------|
| `runs` | Top-level grouping of pipeline executions |
| `pipelines` | Individual pipeline configurations (one per generator expansion) |
| `chains` | Fitted preprocessing-to-model paths with artifact references |
| `predictions` | Scalar scores, metadata, and chain links |
| `prediction_arrays` | y_true, y_pred, y_proba arrays (stored as native DOUBLE[]) |
| `artifacts` | Metadata for binary files (path, hash, type, reference count) |
| `logs` | Structured step-level execution logs |

This architecture means:

- All predictions are queryable from a single location, across all datasets and runs
- No filesystem hierarchy to manage -- no manifests, no nested directories
- Deletion cascades cleanly (delete a run and all its predictions, chains, and orphaned artifacts are removed)
- Export is on-demand: files are only created when you explicitly call `export()`

## Next Steps

- [Making Predictions](making_predictions.md) -- Learn how to predict on new data
- [Analyzing Results](analyzing_results.md) -- Query, filter, rank, and visualize results
- [Exporting Models](exporting_models.md) -- Export models for sharing and deployment
