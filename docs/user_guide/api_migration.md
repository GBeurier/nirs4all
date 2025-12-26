# API Migration Guide: From Classic to Module-Level API

This guide helps you migrate from the classic `PipelineRunner` API to the new module-level API introduced in nirs4all v0.6+.

> **Note**: This guide covers API changes. For migration of prediction format and artifacts, see [migration_guide.md](migration_guide.md).

## Overview

nirs4all v0.6 introduces a simplified module-level API that reduces boilerplate while maintaining full functionality. The classic API remains fully supported for backward compatibility.

### What Changed

| Aspect | Classic API | New API (v0.6+) |
|--------|-------------|-----------------|
| Entry point | `PipelineRunner.run()` | `nirs4all.run()` |
| Configuration | Explicit config objects | Inline parameters |
| Result access | `predictions.top(n=1)[0]` | `result.best` |
| Sessions | N/A | `nirs4all.session()` |
| sklearn integration | Manual | `NIRSPipeline` wrapper |

## Quick Comparison

### Classic API (Still Supported)

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Create configuration objects
pipeline_config = PipelineConfigs(
    [MinMaxScaler(), PLSRegression(n_components=10)],
    name="MyPipeline"
)
dataset_config = DatasetConfigs("sample_data/regression")

# Create runner and execute
runner = PipelineRunner(
    verbose=1,
    save_artifacts=True,
    save_charts=False
)
predictions, per_dataset = runner.run(pipeline_config, dataset_config)

# Access results
best = predictions.top(n=1)[0]
print(f"Best RMSE: {best.get('rmse', 'N/A')}")
```

### New Module-Level API (Recommended)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Direct execution with inline configuration
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
    dataset="sample_data/regression",
    name="MyPipeline",
    verbose=1,
    save_artifacts=True,
    save_charts=False
)

# Convenient result access
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
```

## Detailed Migration Guide

### 1. Basic Training

**Before:**
```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1, save_artifacts=True)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "name"),
    DatasetConfigs("path/to/data")
)
best = predictions.top(n=1)[0]
```

**After:**
```python
import nirs4all

result = nirs4all.run(
    pipeline=pipeline,
    dataset="path/to/data",
    name="name",
    verbose=1,
    save_artifacts=True
)
best = result.best
```

### 2. Accessing Results

**Before:**
```python
# Get top models
top_5 = predictions.top(n=5)

# Get best score
best = predictions.top(n=1)[0]
rmse = best.get('rmse', float('nan'))
r2 = best.get('r2', float('nan'))

# Filter by model
pls_preds = predictions.filter_predictions(model_name='PLSRegression')
```

**After:**
```python
# Get top models
top_5 = result.top(n=5)

# Get best score - convenient properties
rmse = result.best_rmse
r2 = result.best_r2

# Filter by model
pls_preds = result.filter(model_name='PLSRegression')

# Additional convenience
print(result.num_predictions)  # Total count
print(result.get_models())     # List of model names
print(result.get_datasets())   # List of dataset names
```

### 3. Prediction

**Before:**
```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(verbose=0)
y_pred, metadata = runner.predict(
    source=best_prediction,
    dataset=new_data
)
```

**After:**
```python
import nirs4all

predict_result = nirs4all.predict(
    source=result.best,  # or path to bundle
    dataset=new_data,
    verbose=0
)
y_pred = predict_result.values
df = predict_result.to_dataframe()
```

### 4. SHAP Explanation

**Before:**
```python
runner = PipelineRunner()
shap_results, metadata = runner.explain(
    source=best,
    dataset=data,
    shap_params={"n_samples": 100}
)
```

**After:**
```python
import nirs4all

explain_result = nirs4all.explain(
    source=result.best,
    dataset=data,
    n_samples=100
)
print(explain_result.top_features[:10])
importance = explain_result.get_feature_importance()
```

### 5. Model Export

**Before:**
```python
runner = PipelineRunner(save_artifacts=True)
predictions, _ = runner.run(pipeline_config, dataset_config)
best = predictions.top(n=1)[0]
runner.export(source=best, output_path="exports/model.n4a")
```

**After:**
```python
import nirs4all

result = nirs4all.run(pipeline, dataset, save_artifacts=True)
result.export("exports/model.n4a")  # Exports best model automatically
```

### 6. Retraining

**Before:**
```python
runner = PipelineRunner(verbose=1)
new_preds, _ = runner.retrain(
    source=original_prediction,
    dataset=new_data,
    mode='full'
)
```

**After:**
```python
import nirs4all

new_result = nirs4all.retrain(
    source=original_result.best,  # or bundle path
    dataset=new_data,
    mode='full',
    verbose=1
)
```

## New Features (v0.6+)

### Sessions for Multiple Runs

Sessions enable resource sharing across multiple pipeline runs:

```python
import nirs4all

with nirs4all.session(verbose=1, save_artifacts=True) as s:
    # All runs share the same workspace
    result1 = nirs4all.run(pipeline1, data, name="PLS", session=s)
    result2 = nirs4all.run(pipeline2, data, name="RF", session=s)
    result3 = nirs4all.run(pipeline3, data, name="SVM", session=s)

    # Compare results
    print(f"PLS: {result1.best_rmse:.4f}")
    print(f"RF:  {result2.best_rmse:.4f}")
    print(f"SVM: {result3.best_rmse:.4f}")
```

### sklearn Integration with NIRSPipeline

The new `NIRSPipeline` wrapper provides sklearn compatibility:

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline
import shap

# Train with nirs4all
result = nirs4all.run(pipeline, dataset, save_artifacts=True)

# Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

# Standard sklearn interface
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Works with SHAP
explainer = shap.Explainer(pipe.predict, X_background)
shap_values = explainer(X_test)

# Or load from exported bundle
pipe = NIRSPipeline.from_bundle("exports/model.n4a")
```

### RunResult Convenience Methods

The `RunResult` class provides easy access to common operations:

```python
result = nirs4all.run(pipeline, dataset)

# Properties
result.best              # Best prediction dict
result.best_score        # Primary test score
result.best_rmse         # RMSE (regression)
result.best_r2           # R² (regression)
result.best_accuracy     # Accuracy (classification)
result.num_predictions   # Total predictions
result.artifacts_path    # Path to saved artifacts

# Methods
result.top(n=5)          # Top N predictions
result.filter(...)       # Filter predictions
result.get_models()      # List of model names
result.get_datasets()    # List of dataset names
result.export(path)      # Export best model
result.summary()         # String summary
```

## API Reference

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `nirs4all.run()` | Execute a training pipeline |
| `nirs4all.predict()` | Make predictions with trained model |
| `nirs4all.explain()` | Generate SHAP explanations |
| `nirs4all.retrain()` | Retrain pipeline on new data |
| `nirs4all.session()` | Create execution session |

### Result Classes

| Class | Description |
|-------|-------------|
| `RunResult` | Result from `nirs4all.run()` |
| `PredictResult` | Result from `nirs4all.predict()` |
| `ExplainResult` | Result from `nirs4all.explain()` |

### sklearn Integration

| Class | Description |
|-------|-------------|
| `NIRSPipeline` | sklearn-compatible regressor wrapper |
| `NIRSPipelineClassifier` | sklearn-compatible classifier wrapper |

## Migration Checklist

When migrating existing code:

- [ ] Replace `PipelineRunner(...)` with `nirs4all.run(...)`
- [ ] Remove explicit `PipelineConfigs` and `DatasetConfigs` wrappers
- [ ] Update result access from `predictions.top(n=1)[0]` to `result.best`
- [ ] Use `result.best_rmse`, `result.best_r2` for quick access
- [ ] Consider using `nirs4all.session()` for multiple related runs
- [ ] Use `NIRSPipeline.from_result()` for sklearn/SHAP integration
- [ ] Update exports from `runner.export(source=best, ...)` to `result.export(...)`

## Backward Compatibility

The classic API remains fully supported:

```python
# This still works exactly as before
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1)
predictions, per_dataset = runner.run(
    PipelineConfigs(pipeline, "name"),
    DatasetConfigs("path")
)
```

You can mix both APIs in the same codebase:

```python
import nirs4all
from nirs4all.pipeline import PipelineRunner

# New API for quick experiments
result = nirs4all.run(pipeline1, data)

# Classic API for more control
runner = PipelineRunner(custom_option=True)
predictions, per_dataset = runner.run(pipeline_config, dataset_config)
```

## Common Pitfalls

### 1. Accessing raw predictions

The new API returns `RunResult`, not `(predictions, per_dataset)`:

```python
# ❌ Wrong - will fail
predictions, per_dataset = nirs4all.run(pipeline, data)

# ✅ Correct
result = nirs4all.run(pipeline, data)
predictions = result.predictions  # Access raw Predictions object
per_dataset = result.per_dataset  # Access per-dataset dict
```

### 2. Session scope

Session runs share workspace but are independent:

```python
# ❌ Session is closed, runner not available
s = nirs4all.Session(verbose=1)
result = nirs4all.run(pipeline, data, session=s)
# s is not context-managed, may have issues

# ✅ Use context manager
with nirs4all.session(verbose=1) as s:
    result = nirs4all.run(pipeline, data, session=s)
```

### 3. NIRSPipeline is for prediction only

`NIRSPipeline` wraps trained models, it doesn't train:

```python
# ❌ NIRSPipeline doesn't train
from nirs4all.sklearn import NIRSPipeline
pipe = NIRSPipeline(steps=[MinMaxScaler(), PLSRegression(10)])
pipe.fit(X, y)  # Raises NotImplementedError

# ✅ Train with nirs4all, then wrap
result = nirs4all.run(pipeline, dataset)
pipe = NIRSPipeline.from_result(result)
pipe.predict(X_new)  # Works
```

## See Also

- [Q40_new_api.py](../../examples/Q40_new_api.py) - Comprehensive new API example
- [Q_sklearn_wrapper.py](../../examples/Q_sklearn_wrapper.py) - sklearn integration example
- [Q41_sklearn_shap.py](../../examples/Q41_sklearn_shap.py) - SHAP with NIRSPipeline
- [Q42_session_workflow.py](../../examples/Q42_session_workflow.py) - Session workflow example
- [API Design Specification](../specifications/api_design_v2.md) - Full API design document
- [Migration Roadmap](../specifications/api_v2_migration_roadmap.md) - Implementation roadmap
