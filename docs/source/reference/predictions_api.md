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
    coverage=None,      # Select already-materialized conformal intervals
    save_to_workspace=False,
    workspace_metadata=None,
    workspace_result_metadata=None,
    **runner_kwargs,
) -> PredictResult
```

Main prediction paths:

- **Store-based** (preferred): pass `chain_id` to replay a stored chain directly from the workspace.
- **Model-based**: pass `model` (bundle path, prediction dict, or config path).
- **Calibrated replayed-array**: pass a `CalibratedRunResult` or conformal result archive as `model`, plus `data={"y_pred": ..., "sample_ids": ...}`.
- **Attached conformal model bundle**: pass a model `.n4a` bundle carrying a conformal sidecar and `coverage=...`; nirs4all replays the model prediction first, then selects already-materialized intervals.

`coverage` accepts one finite coverage or a non-empty list of finite, unique coverages that were materialized during calibration. It selects existing intervals and updates `conformal_guarantee_status`; it does not fit a new calibrator. With a conformal sidecar, `all_predictions=True` remains fail-closed until every returned prediction entry can carry calibrated identity mapping. If a model bundle contains an invalid `conformal/` sidecar, prediction fails validation instead of falling back to an uncalibrated path. A structurally complete sidecar whose `calibrated_result.json` has non-empty predictions but no canonical physical `sample_ids` is also rejected before the raw model prediction runs.

`save_to_workspace=True` publishes the returned prediction through
`nirs4all.save_workspace_predict_result(...)`, adds
`workspace_prediction_id` to `result.metadata`, and stores optional
`workspace_metadata` / `workspace_result_metadata` alongside row-aligned
executable `X` when it is available from `data`. It publishes prediction
evidence only; conformal result artifacts and interval guarantees remain owned
by `save_workspace_calibrated_result(...)`.

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
    refit=True,           # Refit winning variant(s) on full train set (bool/dict/list/None)
    cache=None,           # Optional CacheConfig for step-level caching
    project=None,         # Optional project name to tag the run with
    report_naming="nirs", # Metric naming in reports: "nirs", "ml", or "auto"
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
| `intervals` | dict[float, Any] | Materialized conformal intervals keyed by coverage |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `values` | numpy.ndarray | Alias for y_pred |
| `shape` | tuple | Shape of prediction array |
| `is_multioutput` | bool | True if multi-output prediction |
| `interval_coverages` | tuple[float, ...] | Materialized interval coverages in deterministic order |
| `conformal_guarantee_status` | dict or None | Fail-loud guarantee metadata: status, engine, method, selected coverage, calibration replay source and invalidation reasons |
| `calibration_replay_source` | dict or None | Direct accessor for conformal calibration replay provenance, falling back from `conformal_guarantee_status` to top-level metadata |
| `tuning_calibration_source` | dict or None | Direct accessor for native tuning calibration provenance, including whether final conformal evidence came from `tuning.winner` instead of `tuning.score_data` |
| `robustness_evidence` | dict or None | Published robustness replay evidence metadata, read from `metadata.robustness_evidence` or `metadata.result_metadata.robustness_evidence` |
| `spectral_replay_evidence_status` | dict | Fail-closed readiness diagnostic for spectral/OOD replay evidence: reports whether published `X`/spectra markers, executable row-aligned `X`/spectra arrays, and a saved `predictor_bundle`/`model_path` are present |

Use `prediction.calibration_replay_source` when callers need the replay lane
directly without reaching into nested metadata.
Use `prediction.tuning_calibration_source` for the tuning/calibration boundary
when `run(tuning=..., calibration=...)` needs to show that `score_data` ranked
trials but did not supply final conformal evidence.
When a calibrated conformal result is loaded from a store or workspace with
`load_workspace_calibrated_result(...)`, call
`calibrated.to_predict_result()` to recover the same public prediction accessors
and interval API without re-parsing the serialized conformal payload.
Bindings and notebooks can use
`load_workspace_calibrated_predict_result(workspace, conformal_id)` for the same
conversion in one public call.
Use `prediction.spectral_replay_evidence_status` before requesting spectral/OOD
robustness scenarios from a stored or published prediction. A status of
`ready_for_spectral_replay` means the prediction metadata carries both an actual
finite 2D `X`/spectra matrix and a saved replay bundle reference; any
`needs_spectral_replay_evidence` status should be treated as a hard block rather
than an invitation to infer or synthesize missing spectra. The diagnostic keeps
`has_X_or_spectra` for publication markers and
`has_executable_X_or_spectra` for finite 2D arrays whose row count matches
`len(prediction)` and can therefore be passed to `nirs4all.robustness()`.
When `metadata["X"]` or `metadata["spectra"]` contains the actual row-aligned
matrix and `robustness_evidence` carries `predictor_bundle` or `model_path`,
`nirs4all.robustness(prediction, ...)` and `prediction.robustness(...)` use
those values as defaults for spectral/OOD scenarios. Provenance-only markers
such as `"prediction_arrays.X"` remain metadata and are not converted into
arrays.

Workspace/store prediction records can be converted explicitly when arrays were
loaded:

```python
from nirs4all.data.predictions import Predictions

predictions = Predictions.from_workspace("workspace/", load_arrays=True)
prediction = predictions.get_predict_result_by_id("pred-001")
record = predictions.get_prediction_by_id("pred-001", load_arrays=True)

# Direct public one-record shortcut:
prediction = nirs4all.load_workspace_predict_result("workspace/", "pred-001")

# Direct public bulk shortcut:
all_predictions = nirs4all.load_workspace_predict_results("workspace/")

# Direct public publisher shortcut:
prediction_id = nirs4all.save_workspace_predict_result(
    "workspace/",
    prediction,
    dataset_name="wheat",
    X=X_new,
    result_metadata={"robustness_evidence": {"predictor_bundle": "model.n4a"}},
)

# Direct prediction-time publisher shortcut:
published = nirs4all.predict(
    model="model.n4a",
    data={"X": X_new},
    workspace_path="workspace/",
    save_to_workspace=True,
    workspace_result_metadata={"robustness_evidence": {"predictor_bundle": "model.n4a"}},
)
prediction_id = published.metadata["workspace_prediction_id"]

status = prediction.spectral_replay_evidence_status
if status["status"] == "ready_for_spectral_replay":
    report = prediction.robustness(y_true=record["y_true"], scenarios=[
        {"kind": "spectral_noise", "severity": 0.01},
    ])

# Direct workspace-to-robustness shortcut:
report = nirs4all.robustness_from_workspace_prediction(
    "workspace/",
    "pred-001",
    y_true=record["y_true"],
    scenarios=[{"kind": "spectral_noise", "severity": 0.01}],
    save_to_workspace=True,
    workspace_robustness_id="pred-001-spectral-audit",
)
```

`nirs4all.save_workspace_predict_result(...)`,
`nirs4all.load_workspace_predict_result(...)`,
`nirs4all.load_workspace_predict_results(...)`,
`Predictions.get_predict_result_by_id()` and the bulk
`Predictions.to_predict_results()` call
`PredictResult.from_prediction_record()` internally. The lower-level class
method remains useful when a caller already has a store-shaped dictionary. All
three forms preserve `result_metadata`, `robustness_evidence`, actual
row-aligned `X`/`spectra` arrays, `sample_indices`, model name, preprocessing
metadata, materialized intervals, `calibration_replay_source`, and
`tuning_calibration_source`. Records loaded with `load_arrays=False` raise a
clear error because they do not carry executable `y_pred`/spectral evidence.
Treat this as the supported bridge from workspace/store predictions into the
native conformal and robustness API: callers should inspect
`prediction.tuning_calibration_source`,
`prediction.calibration_replay_source`, and
`prediction.spectral_replay_evidence_status` before launching downstream
diagnostics, rather than scraping sidecar rows or synthesizing missing replay
inputs.
Use `nirs4all.robustness_from_workspace_prediction(...)` when the desired output
is the report itself: it performs the same load/convert step, delegates to
`nirs4all.robustness()`, and can persist the report back into the same workspace
with a `prediction_id` link.

Python publishers that already hold row-aligned spectra should persist them as
first-class sidecar arrays instead of hiding them in ad-hoc metadata. The
store-backed `Predictions.add_prediction(...)` accepts executable `X` or
`spectra` arrays plus `result_metadata`:

```python
predictions.add_prediction(
    dataset_name="wheat",
    model_name="PLSRegression",
    fold_id="final",
    partition="test",
    y_pred=y_pred,
    X=X_test,
    sample_indices=sample_ids,
    result_metadata={
        "robustness_evidence": {
            "X": "prediction_arrays.X",
            "predictor_bundle": "models/pls.n4a",
        }
    },
)
predictions.flush(pipeline_id=pipeline_id, chain_id=chain_id)
```

After reload with `load_arrays=True`, `X`/`spectra` are restored both on the
record and in `PredictResult.metadata`, while `result_metadata` supplies
`PredictResult.robustness_evidence`. Store merges preserve the same sidecar
fields, so workspace consolidation does not silently drop spectral/OOD replay
evidence.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `nirs4all.save_workspace_predict_result(workspace_path, result, X=None, spectra=None, result_metadata=None, ...)` | prediction id | Public publisher that writes a `PredictResult` and optional executable spectral/OOD evidence into the workspace prediction store |
| `nirs4all.load_workspace_predict_result(workspace_path, prediction_id)` | PredictResult | Public one-record workspace loader that opens the workspace with arrays and returns a native `PredictResult` |
| `nirs4all.load_workspace_predict_results(workspace_path, dataset_name=None)` | list[PredictResult] | Public bulk workspace loader that opens the workspace with arrays and converts all matching predictions into native `PredictResult` objects |
| `from_prediction_record(record)` | PredictResult | Class method converting a workspace/store prediction record loaded with arrays into a `PredictResult` while preserving conformal/tuning/robustness replay metadata |
| `Predictions.get_predict_result_by_id(prediction_id)` | PredictResult or None | Convenience method returning one buffered prediction as a native `PredictResult` |
| `Predictions.to_predict_results()` | list[PredictResult] | Bulk conversion of buffered prediction records into native `PredictResult` objects |
| `to_numpy()` | numpy.ndarray | Predictions as numpy array |
| `to_list()` | list[float] | Predictions as Python list |
| `to_dataframe(include_indices=True)` | pandas.DataFrame | Predictions as DataFrame |
| `flatten()` | numpy.ndarray | Flattened 1D predictions |
| `interval(coverage)` | Any | Interval block for one materialized coverage |

When conformal intervals are present, `str(result)` also reports the guarantee
status, effective engine and selected coverages so interactive Python users do
not have to infer a statistical guarantee from interval arrays alone.

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
