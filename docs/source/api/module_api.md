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
    all_predictions=False,  # Return all non-conformal prediction entries
    coverage=None,      # Select materialized conformal interval coverage(s)
    save_to_workspace=False,  # Publish prediction/evidence row to workspace
    **kwargs            # Additional options
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PredictionSource` | Trained model as prediction dict, `RunResult`, or bundle path |
| `dataset` | `DatasetSpec` | New data for prediction |
| `verbose` | `int` | Verbosity level |
| `all_predictions` | `bool` | If True, return predictions from all CV folds/entries for non-conformal predictions. With a conformal sidecar and `coverage=...`, only `all_predictions=False` is currently supported. |
| `coverage` | `float \| list[float] \| None` | Select already-materialized conformal interval coverages for calibrated replayed-array results or model `.n4a` bundles carrying a conformal sidecar. It never recalibrates or creates a new guarantee; invalid sidecars fail validation instead of falling back to uncalibrated prediction, including sidecars whose non-empty calibrated prediction cohort is missing canonical physical `sample_ids`. |
| `save_to_workspace` | `bool` | If True, publish the returned `PredictResult` through the workspace prediction store and add `workspace_prediction_id` to `result.metadata`. This writes prediction/evidence rows only; conformal artifacts remain owned by `save_workspace_calibrated_result(...)`. |

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

# Publish prediction evidence directly into a workspace
published = nirs4all.predict(
    "exports/model.n4a",
    {"X": new_X},
    workspace_path="workspace/",
    save_to_workspace=True,
    workspace_result_metadata={"robustness_evidence": {"predictor_bundle": "exports/model.n4a"}},
)
prediction_id = published.metadata["workspace_prediction_id"]
```

### nirs4all.calibrate()

Fit or apply the current public split-conformal calibration surface.

```python
calibrated = nirs4all.calibrate(
    calibration_data=(y_cal, y_cal_pred, calibration_sample_ids),
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=[0.9, 0.95],
)
```

**Current scope:** replayed-array calibration evidence, explicit physical sample
ids, selected dataset-backed calibration cohorts, verified filesystem/workspace
persistence, and `.n4a` conformal result archives. Unsupported implicit splits
and broad automatic predictor discovery fail closed.

**Returns:** `CalibratedRunResult` by default, or `PredictResult` when
`as_predict_result=True`.

Related helpers:

- `nirs4all.predict_calibrated(...)` applies an existing conformal calibrator to
  new already-computed point predictions.
- `nirs4all.conformal_metrics(...)` computes empirical observed coverage,
  interval width and interval score diagnostics.
- `nirs4all.load_workspace_calibrated_predict_result(...)` loads a workspace
  conformal result directly as `PredictResult`, preserving intervals,
  `calibrated_result_fingerprint`, `calibration_replay_source`, and
  `tuning_calibration_source` for bindings and notebooks.
- `nirs4all.load_workspace_predict_result(...)` loads one workspace prediction
  row directly as `PredictResult`, preserving arrays, sample ids, model
  metadata and replay provenance when the workspace stores them.
- `nirs4all.load_workspace_predict_results(...)` performs the same conversion
  for all matching workspace prediction rows, optionally filtered by dataset.
- `nirs4all.save_workspace_predict_result(...)` persists a `PredictResult`
  back into the workspace prediction store, including row-aligned executable
  `X`/`spectra` evidence and `result_metadata` for later robustness replay.
  `nirs4all.predict(..., save_to_workspace=True, workspace_result_metadata=...)`
  is the direct prediction-time shortcut for the same publisher.
- `nirs4all.robustness_from_workspace_prediction(...)` loads one workspace
  prediction as `PredictResult`, lets `robustness()` consume any executable
  `X`/`spectra` + `predictor_bundle` evidence already stored on the row, and
  can persist the report back to the workspace linked to that prediction id.
- `nirs4all.load_calibrated_result(...)`,
  `nirs4all.export_calibrated_result(...)`, and
  `nirs4all.attach_calibrated_result_to_bundle(...)` handle local conformal
  artifacts.

### nirs4all.robustness()

Create an audit-only robustness/generalization report for already replayed
prediction cohorts.

```python
report = nirs4all.robustness(
    calibrated,
    y_true=y_observed,
    scenarios=[nirs4all.RobustnessScenarioSpec(kind="observed")],
    slice_by=["instrument"],
    metadata={"instrument": instrument_ids},
    workspace_path="workspace/",
    workspace_robustness_id="external-audit",
    workspace_name="External robustness audit",
)
report.save_json("robustness-report.json")
report.save_html("robustness-report.html")
report.save_summary("robustness-summary.json")
report.save_artifacts("robustness-artifacts")
summary_schema = nirs4all.get_robustness_summary_schema()
summary_schema_json = nirs4all.robustness_summary_schema_json()
verified_bundle = nirs4all.RobustnessReport.load_artifacts("robustness-artifacts")

# Native tuning summaries use the same lightweight publication pattern:
tuning_summary = result.tuning_result.summary_artifact()
tuning_schema = nirs4all.get_tuning_summary_schema()
tuning_schema_json = nirs4all.tuning_summary_schema_json()
tuning_space_schema = nirs4all.get_tuning_space_schema()
tuning_space_schema_json = nirs4all.tuning_space_schema_json()
result.tuning_result.save_summary("tuning-summary.json")

# Equivalent convenience syntax on public result containers:
same_report_from_result = calibrated.robustness(y_true=y_observed)
same_report_from_prediction = prediction.robustness(y_true=y_observed)
same_report_from_workspace_prediction = nirs4all.robustness_from_workspace_prediction(
    "workspace/",
    "pred-001",
    y_true=y_observed,
    scenarios=[{"kind": "spectral_offset", "severity": 0.01}],
    save_to_workspace=True,
    workspace_robustness_id="pred-001-spectral-audit",
)

robustness_id = nirs4all.save_workspace_robustness_report(
    "workspace/",
    report,
    robustness_id="external-audit-copy",
    name="External robustness audit",
)
same_report = nirs4all.load_workspace_robustness_report("workspace/", "external-audit")
```

The same verified JSON artifact can be republished from the CLI without
re-running the audit:

```bash
nirs4all robustness-report robustness-report.json --format markdown --output robustness-report.md
nirs4all robustness-report robustness-report.json --format summary --output robustness-summary.json
nirs4all robustness-report robustness-report.json --format html --output robustness-report.html
nirs4all robustness-report robustness-report.json --format parquet --output robustness-report.parquet
nirs4all robustness-report robustness-report.json --format artifacts --output robustness-artifacts/
nirs4all robustness-report robustness-artifacts/ --format markdown --output reviewed-report.md
nirs4all workspace robustness list --workspace workspace --json
nirs4all workspace robustness show external-audit --workspace workspace --json
nirs4all workspace robustness from-prediction --workspace workspace --prediction-id pred-001 --y-true "1.0,2.0,3.0" --scenarios-json '[{"kind":"spectral_offset","severity":0.01}]' --save-to-workspace --workspace-robustness-id pred-001-spectral-audit --format summary --output robustness-summary.json
nirs4all workspace robustness export external-audit --workspace workspace --format summary --output robustness-summary.json
nirs4all workspace robustness export external-audit --workspace workspace --format artifacts --output robustness-artifacts/
nirs4all workspace robustness export external-audit --workspace workspace --format html --output robustness-report.html
```

`workspace robustness from-prediction` is the CLI equivalent of
`nirs4all.robustness_from_workspace_prediction(...)`: it requires observed
targets through `--y-true` or `--y-true-json`, accepts scenarios through
`--scenarios-json`, can slice with `--metadata-json` + `--slice-by`, and can
persist the generated report with `--save-to-workspace`.

**Current scope:** frozen prediction audits, deterministic post-prediction
stress cells, optional spectral-noise replay with an explicit predictor, point
metrics, conformal diagnostics, slices, degradation rows, worst slices, and
deterministic JSON/Markdown/HTML/Parquet exports. It does not refit or renew a
conformal guarantee.

`nirs4all.ROBUSTNESS_MODES` lists the reserved mode vocabulary:
`clean_frozen`, `matched_recalibration`, and `structural_refit`.
`nirs4all.ROBUSTNESS_EXECUTABLE_MODES` currently contains only
`clean_frozen`; the other modes are reserved and fail closed until their
matched recalibration/refit protocols are implemented.

`scenarios` accepts `RobustnessScenarioSpec` helpers or equivalent mappings.
The supported scenario `kind` values are `observed`, `prediction_bias`,
`prediction_noise`, `spectral_noise`, `spectral_offset`, `spectral_scale`,
`spectral_slope`, and `spectral_shift`; `distribution` is currently
accepted only for `prediction_noise` and `spectral_noise`, and is fail-closed
to `"normal"` when provided. Spectral replay scenarios require explicit `X=`
and exactly one frozen replay source: an in-memory `predictor=` or a saved
`predictor_bundle=` path. The bundle path is replayed through public
`nirs4all.predict(model=predictor_bundle, data={"X": X, "sample_ids": ...},
all_predictions=False)` without refit or recalibration.
Spectral reports record `metadata["spectral_replay"]` with the replay source,
bundle path for saved-bundle replay, route, and sample-id forwarding status.
`RobustnessReport.summary_artifact()` carries the same `spectral_replay` block
when present so UI/CI consumers can audit replay provenance without loading the
full report.
`nirs4all.ROBUSTNESS_SCENARIO_KINDS` exposes the same ordered vocabulary for
Python tooling, bindings and Studio forms, and matches the keyword registry enum
and report metadata. `nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS` lists the
scenario kinds that accept `distribution`, and
`nirs4all.ROBUSTNESS_SCENARIO_DISTRIBUTIONS` lists the accepted distribution
tokens.

**Returns:** `RobustnessReport`.

### nirs4all.get_keyword_registry()

Return the machine-readable keyword/effect registry used by generated docs,
forms, Studio and bindings.

```python
registry = nirs4all.get_keyword_registry()
payload = nirs4all.keyword_registry_json(indent=2)
schema = nirs4all.get_keyword_registry_schema()
schema_payload = nirs4all.keyword_registry_schema_json(indent=2)

scenario_entry = next(
    entry for entry in registry["entries"] if entry["id"] == "robustness.scenarios"
)
print(scenario_entry["value_schema"])
```

The same registry can be exported from the CLI for CI, docs generation or
Studio build steps:

```bash
nirs4all keyword-registry --output keyword-registry.json
nirs4all keyword-registry --schema --output keyword-registry.schema.json
nirs4all keyword-registry --compact
nirs4all robustness-summary-schema --output robustness-summary.schema.json
```

Published HTML documentation also includes the same document as
`_static/keyword-registry.json` and the JSON Schema as
`_static/keyword-registry.schema.json`, so Studio/Web builds can consume and
validate the registry without importing Python at runtime. It also publishes
`_static/robustness-summary.schema.json` for robustness summary cards.

The registry records keyword paths, value schemas, read/change effects,
calibration invalidation behavior, engine support and UI hints. It is
descriptive only; execution still happens through the public runtime functions.

### Native tuning helpers

The native DAG-ML tuning subset can be driven through `run(tuning=...)` or the
lower-level `tune_single_estimator(...)` helper. Typed helpers are exported from
`nirs4all` for forms, bindings and scripts:

- `NativeTuning`
- `TUNING_ENGINES`
- `TUNING_DIRECTIONS`
- `TUNING_CONTRACT_KEYS`
- `TUNING_RUNTIME_KEYS`
- `TUNING_OPTIMIZER_PERSISTENCE_KEYS`
- `TUNING_SPACE_SCHEMA_ID`
- `inspect_tuning_space`
- `get_tuning_space_schema()` and `tuning_space_schema_json()`
- `OrderedSearchSpaceSpec`, `SearchSpaceParameter`, and `ParameterPatch`
- `CONFORMAL_TUNING_SCORE_METRICS`
- `FINETUNE_ENGINES`
- `FINETUNE_APPROACHES`
- `FINETUNE_EVAL_MODES`
- `FINETUNE_OPTUNA_SAMPLERS`
- `FINETUNE_OPTUNA_PRUNERS`
- `FINETUNE_N4M_SAMPLERS`
- `FINETUNE_N4M_PRUNERS`
- `FINETUNE_DAGML_DETERMINISTIC_ENGINES`
- `FINETUNE_DAGML_META_KEYS`
- `FINETUNE_DAGML_SELECTION_METRICS`
- `FINETUNE_DAGML_APPROACHES`
- `FINETUNE_DAGML_EVAL_MODES`
- `FINETUNE_ENGINE_ALIASES`
- `FINETUNE_SAMPLER_KEY_ALIASES`
- `FINETUNE_EVAL_MODE_ALIASES`
- `CONFORMAL_CALIBRATION_METHODS`
- `CONFORMAL_CALIBRATION_UNITS`
- `CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES`
- `CONFORMAL_MULTI_TARGET_POLICIES`
- `ConformalCalibrationData`
- `ConformalMethod`
- `TuningEngine`
- `TuningDirection`
- `FinetuneEngine`
- `FinetuneSampler`
- `FinetunePruner`
- `FinetuneApproach`
- `FinetuneEvalMode`
- `TuningScoreData`
- `TuningConformalScoreCalibration`
- `TuningWinner`
- `TuningCalibration`
- `TunedSingleEstimatorConformalResult`
- `TuningResult`
- `TrialResult`
- `ConformalMultiTarget`
- `ConformalUnit`
- `RobustnessScenarioSpec`
- `ROBUSTNESS_EXECUTABLE_MODES`
- `ROBUSTNESS_MODES`
- `ROBUSTNESS_SCENARIO_DISTRIBUTIONS`
- `ROBUSTNESS_SCENARIO_KINDS`
- `ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS`
- `RobustnessMode`
- `RobustnessScenarioDistribution`
- `RobustnessScenarioKind`

`NativeTuning(..., force_params={...})` is part of the deterministic tuning
contract. It enqueues an explicit first optimizer trial with public decoded
parameter values whose keys must be present in `space`; changing it can change
the selected predictor and therefore stale any previous calibration.

When calibration is requested through `run(tuning=..., calibration=...)` or
`tune_single_estimator(..., calibration=...)`, the return value is
`TunedSingleEstimatorConformalResult`. It keeps the tuned `RunResult` in
`result.run` and the calibrated `PredictResult` or `CalibratedRunResult` in
`result.calibrated`, but routine access does not require reaching into those
fields:

```python
print(result.tuning_best_params)
interval_90 = result.interval(0.9)
metrics = result.metrics(y_true_prediction)
report = result.robustness(y_true=y_true_prediction)
```

Those accessors delegate to the existing tuned run and calibrated result. They
do not refit, recalibrate, recompute intervals, or infer a new guarantee.

See {doc}`/user_guide/models/native_tuning_conformal` for the supported syntax,
keywords, guarantee metadata and current fail-closed boundaries.

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

# Public wrapper alias
{"pipeline": [...], "name": "my_pipeline"}

# Path to YAML/JSON config
"configs/my_pipeline.yaml"

# PipelineConfigs object
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

# DatasetConfigs object
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
- [Migration Guide](../user_guide/troubleshooting/migration.md) - Migrating from classic API
- [Examples](https://github.com/gbeurier/nirs4all/tree/main/examples) - Example scripts
