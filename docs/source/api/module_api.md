# Module-Level API

The module-level API provides a simplified, ergonomic interface for nirs4all. These functions serve as the primary entry points for most users.

## Overview

```python
import nirs4all

# Training
result = nirs4all.run(pipeline, dataset)

# Prediction
predictions = nirs4all.predict(model, new_data)

# Explanation
explanations = nirs4all.explain(model, data)

# Retraining
new_result = nirs4all.retrain(model, new_data)

# Session for multiple runs
with nirs4all.session(verbose=1) as s:
    result1 = nirs4all.run(pipeline1, data, session=s)
    result2 = nirs4all.run(pipeline2, data, session=s)
```

## Functions

### nirs4all.run()

Execute a training pipeline on a dataset.

```python
result = nirs4all.run(
    pipeline,           # Pipeline definition (list, dict, path, or list of pipelines)
    dataset,            # Dataset (path, arrays, config, or list of datasets)
    *,
    name="",            # Pipeline name for logging
    session=None,       # Optional Session for resource sharing
    verbose=1,          # Verbosity level (0-3)
    save_artifacts=True,  # Save model artifacts
    save_charts=True,   # Save visualization charts
    plots_visible=False,  # Show plots interactively
    random_state=None,  # Random seed for reproducibility
    **runner_kwargs     # Additional PipelineRunner options
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline` | `PipelineSpec` | Pipeline definition as list of steps, dict config, YAML path, `PipelineConfigs`, or **list of pipelines** |
| `dataset` | `DatasetSpec` | Dataset as path, numpy arrays, tuple, dict, `DatasetConfigs`, or **list of datasets** |
| `name` | `str` | Optional pipeline name for identification |
| `session` | `Session` | Optional session for resource reuse |
| `verbose` | `int` | Verbosity: 0=quiet, 1=info, 2=debug, 3=trace |
| `save_artifacts` | `bool` | Whether to save model artifacts |
| `save_charts` | `bool` | Whether to save charts |
| `plots_visible` | `bool` | Whether to display plots interactively |
| `random_state` | `int` | Random seed for reproducibility |

**Returns:** `RunResult` containing predictions and convenience accessors.

**Example - Single pipeline:**

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
    name="QuickTest",
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
```

**Example - Multiple pipelines (batch execution):**

```python
# Define different strategies
pipeline_pls = [MinMaxScaler(), PLSRegression(10)]
pipeline_rf = [StandardScaler(), RandomForestRegressor()]

# Run both independently
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf],  # List of pipelines
    dataset="sample_data/regression",
    verbose=1
)
print(f"Total configurations: {result.num_predictions}")
```

**Example - Cartesian product (pipelines × datasets):**

```python
# 2 pipelines × 2 datasets = 4 runs
result = nirs4all.run(
    pipeline=[pipeline_pls, pipeline_rf],
    dataset=["data/wheat", "data/corn"],
    verbose=1
)
# Runs: PLS×wheat, PLS×corn, RF×wheat, RF×corn
```

### nirs4all.predict()

Make predictions with a trained model.

```python
result = nirs4all.predict(
    source,             # Trained model (prediction dict, bundle path, etc.)
    dataset,            # New data for prediction
    *,
    verbose=0,          # Verbosity level
    all_predictions=False,  # Return all fold predictions
    **kwargs            # Additional options
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PredictionSource` | Trained model as prediction dict, `RunResult`, or bundle path |
| `dataset` | `DatasetSpec` | New data for prediction |
| `verbose` | `int` | Verbosity level |
| `all_predictions` | `bool` | If True, return predictions from all CV folds |

**Returns:** `PredictResult` containing prediction values.

**Example:**

```python
import nirs4all

# From training result
result = nirs4all.run(pipeline, train_data)
predictions = nirs4all.predict(result.best, test_data)
print(predictions.values)

# From exported bundle
predictions = nirs4all.predict("exports/model.n4a", new_data)
df = predictions.to_dataframe()
```

### nirs4all.explain()

Generate SHAP explanations for model predictions.

```python
result = nirs4all.explain(
    source,             # Trained model
    dataset,            # Data to explain
    *,
    n_samples=200,      # Number of samples for SHAP
    explainer_type="auto",  # SHAP explainer type
    verbose=0,          # Verbosity level
    **kwargs            # Additional SHAP options
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PredictionSource` | Trained model |
| `dataset` | `DatasetSpec` | Data to explain |
| `n_samples` | `int` | Number of background samples for SHAP |
| `explainer_type` | `str` | "auto", "tree", "kernel", "deep", or "linear" |

**Returns:** `ExplainResult` containing SHAP values and feature importance.

**Example:**

```python
import nirs4all

result = nirs4all.run(pipeline, data)
explanations = nirs4all.explain(result.best, data, n_samples=100)

print(explanations.top_features[:10])
importance = explanations.get_feature_importance(top_n=20)
```

### nirs4all.retrain()

Retrain a pipeline on new data.

```python
result = nirs4all.retrain(
    source,             # Original trained model
    dataset,            # New training data
    *,
    mode="full",        # Retrain mode
    verbose=1,          # Verbosity level
    **kwargs            # Additional options
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PredictionSource` | Original trained model |
| `dataset` | `DatasetSpec` | New training data |
| `mode` | `str` | "full", "transfer", or "finetune" |
| `verbose` | `int` | Verbosity level |

**Returns:** `RunResult` from retraining.

**Example:**

```python
import nirs4all

# Original training
result = nirs4all.run(pipeline, data_2023)

# Retrain on new data
new_result = nirs4all.retrain(
    source=result.best,
    dataset=data_2024,
    mode="transfer"
)
```

### nirs4all.session()

Create an execution session for resource reuse.

```python
with nirs4all.session(
    verbose=1,
    save_artifacts=True,
    **runner_kwargs
) as s:
    # Multiple runs share configuration
    result1 = nirs4all.run(pipeline1, data, session=s)
    result2 = nirs4all.run(pipeline2, data, session=s)
```

**Parameters:**

All parameters are passed to the underlying `PipelineRunner`. Common options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose` | `int` | Default verbosity for all runs |
| `save_artifacts` | `bool` | Default artifact saving |
| `workspace_path` | `Path` | Shared workspace directory |

**Returns:** `Session` context manager.

**Example:**

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

# Compare multiple models efficiently
with nirs4all.session(verbose=1, save_artifacts=True) as s:
    for n_components in [5, 10, 15, 20]:
        result = nirs4all.run(
            pipeline=[PLSRegression(n_components=n_components)],
            dataset="data",
            name=f"PLS_{n_components}",
            session=s
        )
        print(f"PLS({n_components}): RMSE = {result.best_rmse:.4f}")
```

## Result Classes

### RunResult

Result from `nirs4all.run()`. Wraps predictions with convenience accessors.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `best` | `dict` | Best prediction entry |
| `best_score` | `float` | Best model's primary test score |
| `best_rmse` | `float` | Best RMSE (regression) |
| `best_r2` | `float` | Best R² (regression) |
| `best_accuracy` | `float` | Best accuracy (classification) |
| `num_predictions` | `int` | Total prediction count |
| `artifacts_path` | `Path` | Path to saved artifacts |
| `predictions` | `Predictions` | Raw predictions object |
| `per_dataset` | `dict` | Per-dataset details |

**Methods:**

| Method | Description |
|--------|-------------|
| `top(n=5)` | Get top N predictions |
| `filter(**kwargs)` | Filter predictions by criteria |
| `get_models()` | List unique model names |
| `get_datasets()` | List unique dataset names |
| `export(path)` | Export best model to bundle |
| `summary()` | Get summary string |

### PredictResult

Result from `nirs4all.predict()`.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `values` | `ndarray` | Prediction values (alias for y_pred) |
| `y_pred` | `ndarray` | Raw prediction array |
| `shape` | `tuple` | Shape of predictions |
| `is_multioutput` | `bool` | True if multiple outputs |
| `model_name` | `str` | Name of model used |

**Methods:**

| Method | Description |
|--------|-------------|
| `to_numpy()` | Get as numpy array |
| `to_list()` | Get as Python list |
| `to_dataframe()` | Get as pandas DataFrame |
| `flatten()` | Get flattened 1D array |

### ExplainResult

Result from `nirs4all.explain()`.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `values` | `ndarray` | Raw SHAP values |
| `shap_values` | `Any` | SHAP Explanation object or array |
| `shape` | `tuple` | Shape of SHAP values |
| `mean_abs_shap` | `ndarray` | Mean |SHAP| per feature |
| `top_features` | `list` | Features sorted by importance |
| `feature_names` | `list` | Feature names |
| `base_value` | `float` | Baseline prediction |

**Methods:**

| Method | Description |
|--------|-------------|
| `get_feature_importance(top_n=None)` | Get importance ranking |
| `get_sample_explanation(idx)` | Get explanation for single sample |
| `to_dataframe()` | Get as pandas DataFrame |

## Type Aliases

### PipelineSpec

Pipeline definition accepts multiple formats:

```python
# List of steps (most common)
[MinMaxScaler(), PLSRegression(10)]

# Dict configuration
{"steps": [...], "name": "my_pipeline"}

# Path to YAML/JSON config
"configs/my_pipeline.yaml"

# PipelineConfigs object (backward compat)
PipelineConfigs(steps, name="...")

# List of pipelines (batch execution)
[pipeline1, pipeline2, pipeline3]  # Each runs independently
```

### DatasetSpec

Dataset accepts multiple formats:

```python
# Path to data folder
"sample_data/regression"

# Numpy arrays
(X, y)
X  # y inferred as None

# Dict with arrays
{"X": X, "y": y, "metadata": meta}

# SpectroDataset instance
SpectroDataset(...)

# DatasetConfigs object (backward compat)
DatasetConfigs("path")

# List of datasets (batch execution)
["data/wheat", "data/corn"]  # Each dataset tested
[dataset1, dataset2]  # List of SpectroDataset instances
```

### Batch Execution

When you pass **lists** for both `pipeline` and `dataset`, `nirs4all.run()` executes the **cartesian product**:

```python
# 3 pipelines × 2 datasets = 6 runs
result = nirs4all.run(
    pipeline=[pipeline_a, pipeline_b, pipeline_c],
    dataset=["data/wheat", "data/corn"]
)
```

All results are collected into a single `RunResult` for unified analysis.

## See Also

- [sklearn Integration](sklearn_integration.md) - NIRSPipeline for sklearn/SHAP
- [API Migration Guide](../../user_guide/api_migration.md) - Migrating from classic API
- [Examples](https://github.com/gbeurier/nirs4all/tree/main/examples) - Example scripts
