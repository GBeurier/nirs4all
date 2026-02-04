# Analyzing Results

After running a pipeline, you need to inspect, rank, filter, and visualize predictions to select the best model for deployment. This guide covers the `RunResult` API, store-level queries, and visualization with `PredictionAnalyzer`.

## RunResult API

`nirs4all.run()` returns a `RunResult` object that provides convenient accessors for all prediction data.

### Score Properties

```python
result = nirs4all.run(pipeline, dataset)

# Quick access to best model's scores
result.best_rmse        # Best validation RMSE (regression)
result.best_r2          # Best validation R2 (regression)
result.best_accuracy    # Best validation accuracy (classification)
result.best_score       # Best model's primary test score
```

These properties look up the best prediction (ranked by validation score) and extract the corresponding metric. They return `float('nan')` if the metric is unavailable.

### Best Prediction

```python
best = result.best  # Dict with all fields of the best prediction

print(f"Model: {best.get('model_name')}")
print(f"Preprocessing: {best.get('preprocessings')}")
print(f"Val score: {best.get('val_score'):.4f}")
print(f"Test score: {best.get('test_score'):.4f}")
print(f"Metric: {best.get('metric')}")
```

### Top N Predictions

```python
# Top 5 models ranked by validation score (default)
top5 = result.top(5)

for pred in top5:
    print(f"{pred.model_name}: val={pred.get('val_score'):.4f}")

# Top 10 ranked by a specific metric
top10 = result.top(10, rank_metric="rmse", rank_partition="val")

# Top 3 with additional display metrics computed
top3 = result.top(3, display_metrics=["rmse", "r2", "mae"])
for pred in top3:
    print(f"RMSE={pred.get('rmse'):.4f}, R2={pred.get('r2'):.4f}")
```

### Grouping

You can get the top N predictions **per group** using the `group_by` parameter:

```python
# Top 3 per dataset
top_per_ds = result.top(3, group_by="dataset_name")
for pred in top_per_ds:
    ds = pred["group_key"][0]
    print(f"{ds}: {pred.model_name} - {pred.get('val_score'):.4f}")

# Top 2 per model class (as grouped dict)
grouped = result.top(2, group_by="model_classname", return_grouped=True)
for group_key, preds in grouped.items():
    print(f"\n{group_key[0]}:")
    for p in preds:
        print(f"  {p.get('val_score'):.4f}")

# Multi-column grouping
top_combo = result.top(2, group_by=["dataset_name", "model_classname"])
```

### Filtering

```python
# Filter by dataset
wheat_preds = result.filter(dataset_name="wheat")

# Filter by model name
pls_preds = result.filter(model_name="PLSRegression")

# Filter by partition
val_preds = result.filter(partition="val")

# Filter by fold
fold0_preds = result.filter(fold_id="fold_0")

# Filter by branch
branch0_preds = result.filter(branch_id=0)

# Combine filters
filtered = result.filter(
    dataset_name="wheat",
    model_name="PLSRegression",
    partition="val",
)
```

### Dataset and Model Discovery

```python
# List all datasets in the results
datasets = result.get_datasets()
print(f"Datasets: {datasets}")

# List all model names
models = result.get_models()
print(f"Models: {models}")
```

### Metadata

```python
# Total number of predictions
print(f"Total predictions: {result.num_predictions}")

# Summary string
print(result.summary())
```

### Validation

Check for common issues in the run result:

```python
# Raises ValueError if issues found
result.validate()

# Check without raising
report = result.validate(raise_on_failure=False)
if not report["valid"]:
    for issue in report["issues"]:
        print(f"Warning: {issue}")
```

## Store-Level Queries

For cross-run analysis or workspace-wide model comparison, you can query the `WorkspaceStore` directly:

```python
from pathlib import Path
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

store = WorkspaceStore(Path("workspace"))

# Top 20 predictions across all runs
top20 = store.top_predictions(20, metric="val_score", ascending=True)

# Top 5 per model class
top_per_model = store.top_predictions(
    5, metric="val_score", group_by="model_class"
)

# Query with filters
wheat_preds = store.query_predictions(
    dataset_name="wheat",
    partition="val",
    limit=100,
)

# Query by model pattern (SQL LIKE)
pls_preds = store.query_predictions(model_class="PLS%")

# List all completed runs
runs = store.list_runs(status="completed")

# List pipelines for a specific run
pipelines = store.list_pipelines(run_id="abc123")

# Get a single prediction with arrays loaded
pred = store.get_prediction("pred_id", load_arrays=True)
y_true = pred["y_true"]  # numpy array
y_pred = pred["y_pred"]  # numpy array
```

All store query methods return `polars.DataFrame` objects, enabling efficient downstream analysis:

```python
import polars as pl

# Use Polars for advanced analysis
df = store.query_predictions(dataset_name="wheat")
summary = df.group_by("model_class").agg([
    pl.col("val_score").min().alias("best_val"),
    pl.col("val_score").mean().alias("avg_val"),
    pl.count().alias("count"),
])
print(summary)
```

## Prediction Fields Reference

The field names differ slightly between the two contexts:

- **RunResult / Predictions** (in-memory buffer): uses `model_classname`, `id`, `config_name`
- **WorkspaceStore** (database queries): uses `model_class`, `prediction_id`, `pipeline_id`

The tables below show the **store** field names. When using `result.top()` or `result.filter()`, use the buffer field names (e.g., `model_classname` instead of `model_class`).

### Identification

| Store Field | Buffer Field | Type | Description |
|-------------|-------------|------|-------------|
| `prediction_id` | `id` | str | Unique identifier |
| `pipeline_id` | `pipeline_uid` | str | Parent pipeline identifier |
| `chain_id` | -- | str | Chain that produced this prediction |
| `dataset_name` | `dataset_name` | str | Dataset name |
| `model_name` | `model_name` | str | Short model name (e.g., `"PLSRegression"`) |
| `model_class` | `model_classname` | str | Fully qualified class name |
| `fold_id` | `fold_id` | str | Fold identifier (e.g., `"fold_0"`, `"avg"`) |
| `partition` | `partition` | str | Data partition: `"train"`, `"val"`, or `"test"` |

### Scores

| Field | Type | Description |
|-------|------|-------------|
| `val_score` | float | Validation score (primary ranking metric) |
| `test_score` | float | Test score |
| `train_score` | float | Training score |
| `metric` | str | Metric name (e.g., `"rmse"`, `"r2"`) |
| `scores` | dict | Nested dict of all metrics per partition |
| `best_params` | dict | Best hyperparameters (if tuning was used) |

### Data Context

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | str | `"regression"` or `"classification"` |
| `n_samples` | int | Number of samples in this partition |
| `n_features` | int | Number of features (wavelengths) |
| `preprocessings` | str | Short preprocessing chain summary |

### Branch and Exclusion

| Field | Type | Description |
|-------|------|-------------|
| `branch_id` | int or None | Branch index (0-based) |
| `branch_name` | str or None | Human-readable branch name |
| `exclusion_count` | int | Number of excluded samples |
| `exclusion_rate` | float | Fraction of excluded samples (0.0 to 1.0) |

### Timestamps

| Field | Type | Description |
|-------|------|-------------|
| `created_at` | datetime | When the prediction was recorded |

### Arrays (loaded on demand)

| Field | Type | Description |
|-------|------|-------------|
| `y_true` | numpy array | Ground-truth values |
| `y_pred` | numpy array | Predicted values |
| `y_proba` | numpy array or None | Class probabilities (classification) |
| `sample_indices` | numpy array or None | Original dataset indices |
| `weights` | numpy array or None | Per-sample weights |

## Visualization with PredictionAnalyzer

`PredictionAnalyzer` provides a suite of chart types for visual analysis of predictions. It works directly with the `Predictions` object from a run result.

### Setup

```python
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(
    result.predictions,
    output_dir="figures",
)
```

### Top-K Comparison

Compare the top K models side by side:

```python
fig = analyzer.plot_top_k(
    k=10,
    rank_metric="rmse",
)
```

### Confusion Matrix (Classification)

```python
fig = analyzer.plot_confusion_matrix(
    rank_metric="accuracy",
)
```

### Score Histogram

Distribution of scores across all predictions:

```python
fig = analyzer.plot_histogram(
    display_metric="rmse",
)
```

### Heatmap

Compare models across two variables:

```python
fig = analyzer.plot_heatmap(
    x_var="model_classname",
    y_var="preprocessings",
    rank_metric="rmse",
)
```

### Candlestick Plot

Distribution of scores per model with quartiles:

```python
fig = analyzer.plot_candlestick(
    variable="model_name",
    display_metric="rmse",
)
```

### Branch Comparison

For branching pipelines, compare performance across branches:

```python
fig = analyzer.plot_branch_comparison(
    rank_metric="rmse",
)

fig = analyzer.plot_branch_boxplot(
    rank_metric="rmse",
)
```

### Aggregation

When your dataset has multiple measurements per sample (e.g., 4 spectra per sample ID), you can aggregate predictions before visualization:

```python
# Aggregate by a metadata column
analyzer = PredictionAnalyzer(
    result.predictions,
    default_aggregate="sample_id",
    default_aggregate_method="mean",
)

# All plots now show aggregated results
fig = analyzer.plot_top_k(k=5, rank_metric="rmse")
```

### Saving Charts

Charts are saved automatically to the `output_dir`:

```python
analyzer = PredictionAnalyzer(
    result.predictions,
    output_dir="workspace/figures",
)

# Save a chart
fig = analyzer.plot_top_k(k=10, rank_metric="rmse")
# Saved to workspace/figures/top_k_rmse.png
```

## Exporting Prediction Data

Export prediction records as a Parquet file for external analysis:

```python
from pathlib import Path
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

store = WorkspaceStore(Path("workspace"))

# Export all predictions
store.export_predictions_parquet(Path("all_predictions.parquet"))

# Export filtered predictions
store.export_predictions_parquet(
    Path("wheat_predictions.parquet"),
    dataset_name="wheat",
)

# Export only test partition
store.export_predictions_parquet(
    Path("test_results.parquet"),
    partition="test",
)
```

The exported Parquet file is readable by Polars, pandas, or any Parquet-compatible tool:

```python
import polars as pl

df = pl.read_parquet("all_predictions.parquet")
print(df.describe())
```

## See Also

- [Understanding Predictions](understanding_predictions.md) -- Core concepts (chains, partitions, scores)
- [Exporting Models](exporting_models.md) -- Export the best model for deployment
- {doc}`/reference/predictions_api` -- Full API reference for PredictionResultsList
