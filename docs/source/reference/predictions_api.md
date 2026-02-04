# Predictions API Reference

This page is the API reference for the prediction-related classes. For conceptual guidance and practical workflows, see the [Predictions User Guide](/user_guide/predictions/index.md).

## Module-Level Functions

### nirs4all.predict()

```python
nirs4all.predict(
    model=None,          # Path to .n4a bundle, prediction dict, or Path
    data=None,           # numpy array, tuple, dict, path, or SpectroDataset
    *,
    chain_id=None,       # Chain ID for store-based prediction (alternative to model)
    workspace_path=None, # Workspace root (required with chain_id outside a session)
    name="prediction_dataset",
    all_predictions=False,
    session=None,
    verbose=0,
    **runner_kwargs,
) -> PredictResult
```

Two prediction paths:

- **Store-based** (preferred): pass `chain_id` to replay a stored chain directly from the workspace.
- **Model-based**: pass `model` (bundle path, prediction dict, or config path).

`model` and `chain_id` are mutually exclusive.

See: {doc}`/user_guide/predictions/making_predictions`

### nirs4all.run()

```python
nirs4all.run(
    pipeline,            # List of steps, dict, path, or PipelineConfigs
    dataset,             # Path, arrays, dict, SpectroDataset, or DatasetConfigs
    *,
    name="",
    session=None,
    verbose=1,
    save_artifacts=True,
    save_charts=True,
    plots_visible=False,
    random_state=None,
    **runner_kwargs,
) -> RunResult
```

See: {doc}`/user_guide/predictions/analyzing_results`

### nirs4all.retrain()

```python
nirs4all.retrain(
    source,              # Prediction dict, path to .n4a bundle, or config path
    data,                # New dataset
    *,
    mode="full",         # "full", "transfer", or "finetune"
    name="retrain_dataset",
    new_model=None,
    epochs=None,
    session=None,
    verbose=1,
    save_artifacts=True,
    **kwargs,
) -> RunResult
```

See: {doc}`/user_guide/predictions/advanced_predictions`

### nirs4all.explain()

```python
nirs4all.explain(
    model,               # Prediction dict, path to .n4a bundle, or config path
    data,                # Data to explain
    *,
    name="explain_dataset",
    session=None,
    verbose=1,
    plots_visible=True,
    n_samples=None,
    explainer_type="auto",
    **shap_params,
) -> ExplainResult
```

See: {doc}`/user_guide/predictions/advanced_predictions`

---

## Result Classes

### RunResult

Returned by `nirs4all.run()` and `nirs4all.retrain()`.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `best` | dict | Best prediction entry (ranked by validation score) |
| `best_score` | float | Best model's primary test score |
| `best_rmse` | float | Best model's RMSE (NaN if unavailable) |
| `best_r2` | float | Best model's R2 (NaN if unavailable) |
| `best_accuracy` | float | Best model's accuracy (NaN if unavailable) |
| `num_predictions` | int | Total number of predictions |
| `artifacts_path` | Path or None | Path to run artifacts directory |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `top(n, **kwargs)` | PredictionResultsList | Top N predictions by ranking |
| `filter(**kwargs)` | list[dict] | Filter predictions by criteria |
| `get_datasets()` | list[str] | Unique dataset names |
| `get_models()` | list[str] | Unique model names |
| `export(output_path, format="n4a", source=None, chain_id=None)` | Path | Export model to bundle |
| `export_model(output_path, source=None, format=None, fold=None)` | Path | Export model artifact only |
| `summary()` | str | Multi-line summary string |
| `validate(...)` | dict | Check for common issues |

**top() keyword arguments:**

- `rank_metric`: Metric to rank by (default: stored metric)
- `rank_partition`: Partition to rank on (default: `"val"`)
- `display_metrics`: List of additional metrics to compute for display
- `display_partition`: Partition for display metrics (default: `"test"`)
- `ascending`: Sort order (None infers from metric)
- `group_by`: Group results by column(s) -- returns top N per group
- `return_grouped`: If True with group_by, return dict of group to results
- `aggregate`: Aggregate predictions by metadata column or `"y"`
- `aggregate_method`: `"mean"`, `"median"`, or `"vote"`

### PredictResult

Returned by `nirs4all.predict()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `y_pred` | numpy.ndarray | Predicted values |
| `metadata` | dict | Additional prediction metadata |
| `sample_indices` | numpy.ndarray or None | Indices of predicted samples |
| `model_name` | str | Name of the model used |
| `preprocessing_steps` | list[str] | Preprocessing steps applied |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `values` | numpy.ndarray | Alias for y_pred |
| `shape` | tuple | Shape of prediction array |
| `is_multioutput` | bool | True if multi-output prediction |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_numpy()` | numpy.ndarray | Predictions as numpy array |
| `to_list()` | list[float] | Predictions as Python list |
| `to_dataframe(include_indices=True)` | pandas.DataFrame | Predictions as DataFrame |
| `flatten()` | numpy.ndarray | Flattened 1D predictions |

### ExplainResult

Returned by `nirs4all.explain()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `shap_values` | Any | SHAP values (Explanation or ndarray) |
| `feature_names` | list[str] or None | Feature names |
| `base_value` | float or ndarray or None | Baseline prediction |
| `visualizations` | dict[str, Path] | Generated plot files |
| `explainer_type` | str | SHAP explainer type used |
| `model_name` | str | Explained model name |
| `n_samples` | int | Number of samples explained |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `values` | numpy.ndarray | Raw SHAP values array |
| `shape` | tuple | Shape of SHAP values |
| `mean_abs_shap` | numpy.ndarray | Mean absolute SHAP per feature |
| `top_features` | list[str] | Features sorted by importance (descending) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_feature_importance(top_n=None, normalize=False)` | dict[str, float] | Feature importance ranking |
| `get_sample_explanation(idx)` | dict[str, float] | SHAP values for one sample |
| `to_dataframe(include_feature_names=True)` | pandas.DataFrame | SHAP values as DataFrame |

---

## PredictionResultsList

Returned by `RunResult.top()` and `Predictions.top()`. Extends Python's built-in `list` with additional methods.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path="results", filename=None)` | None | Save all predictions to structured CSV |
| `get(prediction_id)` | PredictionResult or None | Retrieve prediction by ID |

Supports all standard list operations: indexing, slicing, iteration, `len()`, etc.

## PredictionResult

A dict subclass representing a single prediction. Returned as elements of `PredictionResultsList`.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `id` | str | Prediction identifier |
| `dataset_name` | str | Dataset name |
| `model_name` | str | Model name |
| `fold_id` | str | Fold identifier |
| `config_name` | str | Configuration name |
| `step_idx` | int | Pipeline step index |
| `op_counter` | int | Operation counter |

Additional fields are accessible via dict access (e.g., `pred["model_classname"]`, `pred.get("preprocessings")`).

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | str | Formatted metric table (train/val/test) |
| `save_to_csv(path_or_file, filename=None)` | None | Save to CSV file |
| `eval_score(metrics=None)` | dict | Compute metrics for this prediction |

---

## WorkspaceStore (Prediction Queries)

For store-level queries across runs. See full API in the storage reference.

**Prediction query methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_prediction(id, load_arrays=False)` | dict or None | Single prediction record |
| `query_predictions(**filters)` | polars.DataFrame | Filtered prediction records |
| `top_predictions(n, metric, ascending, partition, dataset_name, group_by)` | polars.DataFrame | Top-N ranked predictions |
| `export_predictions_parquet(output_path, **filters)` | Path | Export to Parquet file |

**Filter arguments for query_predictions:**

- `dataset_name`, `model_class`, `partition`, `fold_id`, `branch_id`, `pipeline_id`, `run_id`, `limit`, `offset`

## See Also

- {doc}`/user_guide/predictions/index` -- Predictions user guide (overview)
- {doc}`/user_guide/predictions/making_predictions` -- Practical prediction workflows
- {doc}`/user_guide/predictions/analyzing_results` -- Querying and visualization
- {doc}`/user_guide/predictions/exporting_models` -- Export formats
- {doc}`/user_guide/predictions/advanced_predictions` -- Transfer, retrain, SHAP
