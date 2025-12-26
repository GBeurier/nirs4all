# Module-Level API Reference

nirs4all v0.6+ introduces a simplified module-level API for common operations.

## Quick Start

```python
import nirs4all

# Training
result = nirs4all.run(pipeline, dataset, verbose=1)

# Prediction
predictions = nirs4all.predict(model, new_data)

# Explanation
explanations = nirs4all.explain(model, data)

# Session for multiple runs
with nirs4all.session(verbose=1) as s:
    r1 = nirs4all.run(pipeline1, data, session=s)
    r2 = nirs4all.run(pipeline2, data, session=s)
```

## Functions

### nirs4all.run()

Execute a training pipeline on a dataset.

```python
result = nirs4all.run(
    pipeline,           # Pipeline definition
    dataset,            # Dataset (path, arrays, config)
    *,
    name="",            # Pipeline name
    session=None,       # Session for resource sharing
    verbose=1,          # Verbosity (0-3)
    save_artifacts=True,
    save_charts=True,
    plots_visible=False,
    random_state=None,
    **runner_kwargs
) -> RunResult
```

### nirs4all.predict()

Make predictions with a trained model.

```python
result = nirs4all.predict(
    source,             # Trained model (dict, RunResult, bundle path)
    dataset,            # New data
    *,
    verbose=0,
    all_predictions=False,
    **kwargs
) -> PredictResult
```

### nirs4all.explain()

Generate SHAP explanations.

```python
result = nirs4all.explain(
    source,             # Trained model
    dataset,            # Data to explain
    *,
    n_samples=200,
    explainer_type="auto",
    verbose=0,
    **kwargs
) -> ExplainResult
```

### nirs4all.retrain()

Retrain a pipeline on new data.

```python
result = nirs4all.retrain(
    source,             # Original model
    dataset,            # New training data
    *,
    mode="full",        # "full", "transfer", or "finetune"
    verbose=1,
    **kwargs
) -> RunResult
```

### nirs4all.session()

Create execution session for resource sharing.

```python
with nirs4all.session(verbose=1, save_artifacts=True) as s:
    result1 = nirs4all.run(pipeline1, data, session=s)
    result2 = nirs4all.run(pipeline2, data, session=s)
```

## Result Classes

### RunResult

Properties:
- `best`: Best prediction entry
- `best_score`, `best_rmse`, `best_r2`, `best_accuracy`: Score shortcuts
- `num_predictions`: Total prediction count
- `predictions`: Raw Predictions object
- `per_dataset`: Per-dataset details

Methods:
- `top(n=5)`: Get top N predictions
- `filter(**kwargs)`: Filter predictions
- `export(path)`: Export best model
- `get_models()`, `get_datasets()`: List names

### PredictResult

Properties:
- `values`, `y_pred`: Prediction array
- `shape`, `is_multioutput`: Array info

Methods:
- `to_numpy()`, `to_list()`, `to_dataframe()`

### ExplainResult

Properties:
- `values`, `shap_values`: SHAP values
- `top_features`: Features sorted by importance
- `mean_abs_shap`: Mean absolute SHAP per feature

Methods:
- `get_feature_importance(top_n=None)`
- `get_sample_explanation(idx)`
- `to_dataframe()`

## See Also

- [sklearn Integration](sklearn_integration.md)
- [API Migration Guide](../user_guide/api_migration.md)
- [Q40_new_api.py](https://github.com/gbeurier/nirs4all/blob/main/examples/Q40_new_api.py)
