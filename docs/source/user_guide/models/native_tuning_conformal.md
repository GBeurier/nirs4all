# Native tuning and conformal calibration

This page documents the current native Python surface for combining model tuning
with split conformal intervals. It is intentionally narrower than the historical
`finetune_params` path: unsupported shapes fail closed instead of silently
falling back to a different engine.

Use this path when you already have explicit arrays for development, scoring,
calibration, and prediction cohorts, and you want the tuning trace plus
conformal calibration artifact persisted in a nirs4all workspace.

For a runnable smoke-tested script that chains the current public subset end to
end, see `examples/user/04_models/U09_native_tuning_conformal.py`. It runs
native tuning, final conformal calibration, calibrated prediction, robustness
reporting, workspace reload, and robustness artifact publication on small
in-memory arrays.
For the same lane with a spectroscopy-standard PLS model, see
`examples/user/04_models/U10_native_pls_conformal_robustness.py`; it tunes
`PLSRegression.n_components` through the public `model.n_components` search
space path and then runs the same conformal and robustness audit stages.

## What is native today

The supported public subset is:

- `nirs4all.run(..., engine="dag-ml", tuning={...})`
- one final sklearn-like estimator, optionally preceded by a linear chain of
  sklearn-like transformers;
- explicit array datasets: `(X, y)`, `[X, y]`,
  `(X, y, sample_ids, groups, metadata)`,
  `[X, y, sample_ids, groups, metadata]`, or `{"X": X, "y": y}`;
- explicit dataset-backed mappings with a selector for fit, scoring, winner and
  `calibration_data.dataset`, using `SpectroDataset`, `DatasetConfigs`, dataset
  config mappings, or config/path strings;
- linear pipeline wrappers `{"steps": [...]}` and the legacy public alias
  `{"pipeline": [...]}` for the same supported step sequence; do not provide
  both keys in one wrapper;
- explicit `tuning.score_data` for objective scoring, as a mapping or tuple/list;
- Optuna/n4m adapter execution through the native tuning seam;
- optional `tuning.winner` to project a terminal prediction entry;
- optional top-level `calibration` or nested `tuning.calibration` to calibrate
  conformal intervals from that projected winner;
- workspace persistence for both `TuningResult` and `CalibratedRunResult`.

The following remain intentionally closed until their gates are implemented:
bare/implicit dataset loaders for native tuning, implicit splitting,
branch/merge graphs, model aliases,
non-linear preprocessing graphs, and bit-exact resume from an interrupted
optimizer checkpoint.

## Native keyword/effect quick map

Use this table when wiring notebooks, CI, bindings, Studio forms, or generated
configuration into the native Python runtime. It separates the public syntax
from the effect of that syntax and from the evidence that downstream tools can
inspect without parsing private metadata.

| Public syntax | Runtime effect | Published evidence | Fail-closed boundary |
| --- | --- | --- | --- |
| `run(engine="dag-ml")` | Selects the DAG-ML execution backend for the current run. | `RunResult` metadata and backend traces. | It is not an optimizer selector; `run(engine="dual")` is still reserved. |
| `run(tuning={...})` / `NativeTuning(...)` | Runs the fixed-topology native tuning subset over one estimator or a linear sklearn-like chain. | `RunResult.tuning_result`, `tuning_id`, `tuning_best_params`, `tuning_best_value`, and optional workspace `TuningResult`. | Branch/merge graphs, implicit splits, aliases, non-linear graphs, and unsupported datasets fail before execution. |
| `run.tuning.space` | Defines the ordered candidate parameter contract. | `nirs4all.tuning.ordered_search_space`, `SearchSpaceParameter`, `ParameterPatch`, and `search_space_fingerprint`. | Values must stay TCV1 JSON-native and fingerprintable; Python objects, bytes, NaN/Infinity and duplicate canonical paths are rejected. |
| `run.tuning.force_params` | Enqueues a caller-provided warm-start assignment as the first optimizer trial. | First `TrialResult.params`, decoded `best_params` when it wins, and the same `search_space_fingerprint`. | Keys must exist in `run.tuning.space`; categorical values use public decoded values, not backend labels. |
| `run.tuning.score_data` | Scores candidate trials on an explicit HPO objective cohort. | Trial scores, trial diagnostics, and `TuningResult.summary_artifact()`. | It is never reused as final conformal calibration evidence. |
| `run.tuning.score_data.conformal_calibration` plus `conformal_coverage` | Fits a temporary conformal calibrator per candidate only for conformal-aware objective metrics. | Trial diagnostics include `score_family="conformal"` and `final_calibration_scope="unmodified_by_score_data"`. | The temporary calibrator is discarded after scoring and cannot renew the final guarantee. |
| `run.tuning.winner` | Supplies or selects the cohort used to project the terminal winner. | `RunResult.best`, terminal prediction metadata, and `tuning_calibration_source` when final calibration is requested. | Missing row-aligned identities or mixed dataset/array payloads fail before calibration. |
| `run(..., calibration=...)` or `run.tuning.calibration` | Fits final split-conformal intervals from the projected winner and applies them to provided prediction values. | `CalibratedRunResult`, `PredictResult.intervals`, `conformal_guarantee_status`, `calibration_replay_source`, and workspace conformal artifacts. | `calibration_data` cannot be injected in this combined tuning flow; final evidence comes from `tuning.winner`. |
| `predict(coverage=...)` | Selects already materialized conformal intervals from a calibrated result or conformal bundle sidecar. | `PredictResult.interval_coverages`, selected `conformal_guarantee_status`, and direct interval accessors. | It does not recalibrate; unsupported coverages, invalid sidecars, and conformal `all_predictions=True` fail closed. |
| `robustness(..., scenarios=[...])` | Produces an audit-only robustness/generalization report for frozen predictions. | `RobustnessReport`, `summary_artifact()`, `summary_rows()`, JSON/Markdown/HTML/Parquet/artifact exports. | It does not refit, structurally retrain, or renew conformal guarantees. |
| `robustness.X` plus `robustness.predictor` or `robustness.predictor_bundle` | Replays explicit-X spectral/OOD perturbations through an in-memory predictor or saved bundle. | `metadata["spectral_replay"]`, replay source, bundle path, sample-id forwarding status, and spectral/OOD summary rows. | `predictor` and `predictor_bundle` are mutually exclusive; stored provenance markers such as `"prediction_arrays.X"` are not executable arrays. |
| `PredictResult.spectral_replay_evidence_status` | Reports whether a stored prediction can safely run spectral/OOD replay. | `ready_for_spectral_replay` only when row-aligned finite 2D `X`/spectra and a saved replay bundle are present. | `needs_spectral_replay_evidence` is a hard block for spectral/OOD replay, not a prompt to synthesize spectra. |
| `nirs4all.save_workspace_predict_result(...)` | Publishes a `PredictResult` through the workspace prediction store. | Stored prediction id plus reloadable `y_pred`, sample ids, model metadata, `X`/`spectra` sidecar arrays and `result_metadata`. | It publishes prediction evidence only; conformal artifacts/interval guarantees remain owned by `save_workspace_calibrated_result(...)`. |
| `predict(..., save_to_workspace=True, workspace_result_metadata=...)` | Prediction-time shortcut for the same workspace prediction publisher. | Returned `PredictResult.metadata["workspace_prediction_id"]` plus reloadable `X`/`spectra` evidence when `data` carries executable arrays. | It does not persist conformal interval artifacts; use `save_workspace_calibrated_result(...)` for guarantees. |
| `nirs4all.load_workspace_predict_result(...)`; `nirs4all.load_workspace_predict_results(...)`; `Predictions.get_predict_result_by_id(...)` / `to_predict_results()` | Converts workspace/store rows loaded with arrays into native `PredictResult` objects. | Preserves intervals, `calibration_replay_source`, `tuning_calibration_source`, `robustness_evidence`, actual `X`/`spectra`, sample ids and model metadata. | Records loaded without arrays cannot become executable prediction results. |
| `nirs4all.robustness_from_workspace_prediction(...)` | Loads one workspace prediction, delegates to `robustness()`, and optionally saves the report back to the workspace. | Consumes stored executable `X`/`spectra` + `predictor_bundle` evidence as spectral/OOD defaults and links saved reports to `prediction_id`. | Does not synthesize missing spectra or bypass `spectral_replay_evidence_status`; provenance-only markers remain non-executable. |

## Minimal pattern

```python
import numpy as np
import nirs4all
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X_dev = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
y_dev = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)

X_score = np.asarray([[4.0], [5.0], [6.0]], dtype=float)
y_score = np.asarray([4.0, 5.0, 6.0], dtype=float)

X_cal = np.asarray([[7.0], [8.0], [9.0], [10.0]], dtype=float)
y_cal = np.asarray([7.2, 7.8, 9.3, 9.7], dtype=float)

result = nirs4all.run(
    pipeline=[
        {
            "name": "scale",
            "class": "sklearn.preprocessing.StandardScaler",
            "params": {"with_mean": False},
        },
        {
            "name": "ridge",
            "class": "sklearn.linear_model.Ridge",
            "params": {"fit_intercept": False},
        },
    ],
    dataset=(
        X_dev,
        y_dev,
        ["train-001", "train-002", "train-003", "train-004"],
        None,
        None,
    ),
    engine="dag-ml",
    workspace_path="workspace/",
    tuning={
        "engine": "optuna",
        "space": {"model.alpha": [0.1, 1.0]},
        "force_params": {"model.alpha": 0.1},
        "metric": "rmse",
        "direction": "minimize",
        "sampler": "grid",
        "n_trials": 2,
        "score_data": {
            "X": X_score,
            "y": y_score,
            "sample_ids": ["score-001", "score-002", "score-003"],
        },
        "workspace_tuning_id": "ridge-native-tuning",
        "winner": {
            "X": X_cal,
            "y_true": y_cal,
            "score": 0.3,
            "metric": "rmse",
            "sample_ids": ["cal-001", "cal-002", "cal-003", "cal-004"],
        },
    },
    calibration={
        "y_pred": [11.0, 12.0],
        "prediction_sample_ids": ["pred-001", "pred-002"],
        "coverage": 0.8,
        "workspace_conformal_id": "ridge-conformal",
    },
)
```

When top-level `calibration` or `tuning.calibration` is present, the return
value is a
`TunedSingleEstimatorConformalResult` with:

- `result.run`: the tuned `RunResult`;
- `result.calibrated`: the conformal `CalibratedRunResult` or `PredictResult`.

The top-level form is an alias for `tuning.calibration` in the current DAG-ML
tuning subset, so do not provide both in the same call.
Without either calibration form, the same call returns a `RunResult`.

### DAG-ML cache namespace proofs

The Python surface does not reinterpret DAG-ML D10 cache namespace proofs. When
native training, replay, bundles or prediction-cache payloads contain
`cache_namespace_fingerprints`, nirs4all treats them as signed DAG-ML control
plane data and forwards them unchanged through
`nirs4all.pipeline.dagml.native_client` to `dag_ml`. DAG-ML owns validation,
namespace-aware handle derivation, file-store payload naming and columnar
manifest exposure.

There is no nirs4all keyword for mutating this proof after signing. Change the
candidate, data identity, fold, trial or seed before creating the DAG-ML
contract.

## Inspecting tuning and conformal outputs

The tuned run keeps the optimizer evidence on `RunResult` without pretending
that development objective values are test scores:

```python
print(result.tuning_id)
print(result.tuning_best_params)
print(result.tuning_best_value)

tuning_result = result.tuning_result
if tuning_result is not None:
    print(tuning_result.optimizer)
    print(tuning_result.n_trials)
    tuning_result.save_summary("tuning-summary.json")
```

`TuningResult.summary_artifact()`, `to_summary_json()` and `save_summary()`
produce a lightweight deterministic `summary.json`-style payload for CI,
bindings and Studio cards. It includes the result fingerprint, engine, metric,
direction, optimizer, best params/value, trial count, trial state counts and
compact trial rows. Compact diagnostic fields are scalar-only; whitelisted
diagnostics carrying arrays or mappings fail closed instead of being
stringified into ambiguous card text. It also includes a safe-to-publish
`persistence` block with `resume`, `storage_configured`, `study_name`, and
`optimizer_state_resume_supported`, but deliberately omits the raw `storage`
URI so public cards do not leak local paths or credentials. The summary does not
replace the full `TuningResult.to_json()` tape.
The matching JSON Schema is public through `get_tuning_summary_schema()` and
`tuning_summary_schema_json()`. The pre-execution ordered tuning-space artifact
has its own public schema through `get_tuning_space_schema()` and
`tuning_space_schema_json()`.
When `workspace_path` is supplied, the completed `TuningResult` is written at
the optimizer-to-refit boundary, before terminal winner refit, winner
projection, or final conformal calibration. If a later refit or calibration
step fails, the workspace still preserves the optimizer evidence tape for
inspection and resume checks. If every candidate trial fails, no terminal refit
is attempted, but the all-failed `TuningResult` remains persistable with its
trial states and summary artifact.

When calibration is requested, intervals live on the calibrated prediction
container. In that case `result` is a `TunedSingleEstimatorConformalResult`:
`result.run` is the underlying `RunResult`, `result.calibrated` is the
calibrated prediction container, the `tuning_*` accessors above proxy to
`result.run`, and the common conformal accessors proxy to `result.calibrated`.
Use the accessor methods rather than reaching into metadata:

```python
prediction = result.calibrated if hasattr(result, "calibrated") else None
if prediction is not None:
    print(result.interval_coverages)
    intervals_80 = result.interval(0.8)
    status = result.conformal_guarantee_status
    print(status["status"])
    metrics = result.metrics(y_true_prediction)
    robustness_report = result.robustness(y_true=y_true_prediction)
```

These accessors are part of the public Python contract. Studio, bindings and
scripts should depend on them, not on private fields or serialized JSON layout.

## Public Python result types

The native conformal/tuning surface exposes stable top-level containers from
`nirs4all`:

- `RunResult`: returned by `run(tuning=...)` when no final conformal calibration
  is requested. Its `tuning_result`, `tuning_id`, `tuning_best_params`, and
  `tuning_best_value` accessors expose optimizer evidence.
- `TuningResult` and `TrialResult`: public optimizer-evidence containers
  returned by tuning helpers and workspace loaders. `TuningResult` provides
  deterministic JSON persistence plus `summary_artifact()`,
  `to_summary_json()` and `save_summary()` for dashboards and release bundles.
  Direct construction is fail-closed: `SearchSpaceParameter`,
  `ParameterPatch`, `OrderedSearchSpaceSpec`, `DagMLTuningSpec`, `TrialResult`
  and `TuningResult` canonicalize tuning paths, reject non-TCV1 values,
  non-finite scores, boolean integers, duplicate trial ids and params outside
  `tuning.space` before fingerprinting or summary publication. Reload through
  `TuningResult.from_dict()` or `load_json()` uses the same strict numeric
  boundary for `best_value`: booleans and numeric strings are rejected, not
  coerced.
- `TunedSingleEstimatorConformalResult`: returned by
  `run(tuning=..., calibration=...)` and `tune_single_estimator(...,
  calibration=...)`. It keeps the tuned run in `run`, the calibrated prediction
  result in `calibrated`, and proxies the common `tuning_*` accessors.
- `CalibratedRunResult`: returned by `calibrate(..., as_predict_result=False)`,
  conformal workspace/bundle loaders, and the tuning+calibration path when a
  rich calibrated artifact is requested. It carries point predictions,
  intervals, sample ids, the conformal artifact fingerprint, guarantee metadata,
  JSON persistence helpers, and `metrics(y_true)`. Reload recomputes stored
  `qhat` values from retained non-negative residual scores, then verifies that
  every materialized interval is exactly derived from the embedded artifact as
  `y_pred ± qhat`; grouped conformal results use a row-aligned `qhat` vector
  selected from the calibrated group for each prediction row. Self-consistent
  JSON with edited intervals, quantiles, group keys or vector `qhat` values is
  rejected.
- `CalibratedPredictionBlock` and `ConformalIntervalBlock`: internal/public
  substrate objects used by calibrated results and prediction helpers. Direct
  construction is fail-closed: interval coverage must match its mapping key,
  interval arrays must match `y_pred` shape, lower bounds must be `<=` upper
  bounds, `qhat` must be scalar or row-aligned and non-negative, and only the
  supported conformal method/unit vocabulary is accepted. Positive infinity
  remains valid for unbounded intervals.
- `ConformalCalibrationData`: typed input helper for
  `calibrate(calibration_data=...)`. It serializes either replayed-array
  evidence or a selected dataset-backed calibration cohort to the same mapping
  syntax used by the runtime. Dataset-backed selectors reject non-string or
  whitespace-padded keys plus non-JSON-native values, and `include_augmented`
  must be a boolean so the helper cannot publish a coerced calibration cohort.
- `ConformalMetricSet`: returned by `conformal_metrics(...)` and
  `CalibratedRunResult.metrics(y_true)`. It is empirical post-hoc evidence on an
  observed prediction cohort; it does not change the conformal guarantee. The
  container fails closed if `observed_coverage` is non-finite, outside `[0, 1]`,
  or inconsistent with `n_covered / n_samples`; if `coverage_gap` is non-finite
  or not exactly `observed_coverage - coverage`; or if width/interval-score
  diagnostics are NaN or negative. Positive infinity remains valid for
  unbounded intervals.
- `TUNING_ENGINES`, `TUNING_DIRECTIONS`, `TUNING_CONTRACT_KEYS`,
  `TUNING_OPTIMIZER_PERSISTENCE_KEYS`, `TUNING_RUNTIME_KEYS`,
  `CONFORMAL_TUNING_SCORE_METRICS`, `TuningEngine`, and `TuningDirection`:
  public discovery and typing helpers for the native
  `run(tuning=...)` contract. Use them for forms, bindings, CLI validation and
  Studio controls instead of hard-coding the currently executable optimizer
  engines, objective directions, strict contract keys, optimizer persistence
  keys, runtime wrapper keys and conformal-aware objective metrics.
- `inspect_tuning_space(...)`, `OrderedSearchSpaceSpec`,
  `SearchSpaceParameter`, and `ParameterPatch`: public inspection helpers for
  canonical ordered `tuning.space` patches. Use the JSON-native
  `nirs4all.tuning.ordered_search_space` artifact in forms, bindings and Studio
  previews when you need the exact patch order and search-space fingerprint
  before execution. Use `TUNING_SPACE_SCHEMA_ID`, `get_tuning_space_schema()`
  and `tuning_space_schema_json()` to validate this artifact without executing a
  pipeline.
- `FINETUNE_ENGINES`, `FINETUNE_APPROACHES`, `FINETUNE_EVAL_MODES`,
  `FINETUNE_OPTUNA_SAMPLERS`, `FINETUNE_OPTUNA_PRUNERS`,
  `FINETUNE_N4M_SAMPLERS`, `FINETUNE_N4M_PRUNERS`,
  `FINETUNE_DAGML_DETERMINISTIC_ENGINES`, `FINETUNE_DAGML_META_KEYS`,
  `FINETUNE_DAGML_SELECTION_METRICS`, `FINETUNE_DAGML_APPROACHES`,
  `FINETUNE_DAGML_EVAL_MODES`, `FINETUNE_ENGINE_ALIASES`,
  `FINETUNE_SAMPLER_KEY_ALIASES`, `FINETUNE_EVAL_MODE_ALIASES`,
  `FinetuneEngine`, `FinetuneSampler`, `FinetunePruner`,
  `FinetuneApproach`, and `FinetuneEvalMode`: public discovery and typing
  helpers for historical model-local `finetune_params`. These constants separate
  the registry-level vocabulary from the Optuna, n4m and deterministic DAG-ML
  executable subsets so downstream tooling does not overclaim support.
- `CONFORMAL_CALIBRATION_METHODS`, `CONFORMAL_CALIBRATION_UNITS`,
  `CONFORMAL_MULTI_TARGET_POLICIES`,
  `CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES`, `ConformalMethod`,
  `ConformalUnit`, and `ConformalMultiTarget`: public discovery and typing
  helpers for the current conformal calibration method, exchangeability unit,
  and reserved/executable multi-target policy vocabulary. The executable V1
  policies are `marginal` for one-dimensional targets and `joint_max` for
  simultaneous multi-target regions on two-dimensional replayed arrays.
- `ROBUSTNESS_MODES`, `ROBUSTNESS_EXECUTABLE_MODES`, and `RobustnessMode`:
  public discovery and typing helpers for the reserved robustness reuse
  policies and the subset executable by the current runtime.
- `RobustnessScenarioSpec`: typed input helper for
  `robustness(scenarios=...)`. It serializes the supported audit scenario
  keywords to the same mapping syntax consumed by the runtime and rejects
  unsupported kinds, non-finite severities, negative noise severities, and
  unsupported distributions before execution.
- `ROBUSTNESS_SCENARIO_KINDS` / `RobustnessScenarioKind`: public discovery and
  typing helpers for the exact scenario `kind` vocabulary exposed by the
  runtime, report metadata, and keyword registry.
- `ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS`,
  `ROBUSTNESS_SCENARIO_DISTRIBUTIONS`, and
  `RobustnessScenarioDistribution`: public discovery and typing helpers for the
  subset of scenario kinds that accepts `distribution`, and for the currently
  supported distribution vocabulary.
- `RobustnessReport`: returned by `robustness(...)`. It is an audit artifact for
  frozen prediction cohorts and provides deterministic JSON, Markdown, HTML and
  Parquet-directory exports.

These types are intended for Python users, CI automation, Studio integration and
bindings. Prefer their accessors and export helpers over private implementation
modules.

The keyword/effect registry is also exposed from the top-level package for
generated docs, forms, Studio and bindings:

```python
registry = nirs4all.get_keyword_registry()
registry_json = nirs4all.keyword_registry_json(indent=2)
registry_schema = nirs4all.get_keyword_registry_schema()
registry_schema_json = nirs4all.keyword_registry_schema_json(indent=2)
robustness_summary_schema = nirs4all.get_robustness_summary_schema()
robustness_summary_schema_json = nirs4all.robustness_summary_schema_json(indent=2)

scenario_keywords = [
    entry for entry in registry["entries"] if entry["path"].startswith("robustness.scenarios")
]
```

The registry is descriptive. It reports keyword paths, value schemas, effects,
engine support and UI hints, but it does not execute or validate a run by
itself. For static consumers, `nirs4all keyword-registry --output
keyword-registry.json` and `nirs4all keyword-registry --schema --output
keyword-registry.schema.json` emit the same artifacts as the published docs
`_static/keyword-registry*.json` files.
`get_robustness_summary_schema()` and `robustness_summary_schema_json()` expose
the JSON Schema for `RobustnessReport.summary_artifact()` / `summary.json`, so
bindings and Studio can validate card/dashboard payloads without loading the
full report schema. Docs builds also publish this contract as
`_static/robustness-summary.schema.json`; CI can emit the same schema with
`nirs4all robustness-summary-schema --output robustness-summary.schema.json`.

The mapping form above is the lowest-level public syntax. The same current
subset also has typed helpers:

```python
tuning = nirs4all.NativeTuning(
    engine="optuna",
    space={"scale": [nirs4all.TuningPassthrough()], "model.alpha": [0.1, 1.0]},
    force_params={"scale": nirs4all.TuningPassthrough(), "model.alpha": 0.1},
    metric="rmse",
    direction="minimize",
    sampler="grid",
    n_trials=2,
    score_data=nirs4all.TuningScoreData(
        X=X_score,
        y=y_score,
        sample_ids=["score-001", "score-002", "score-003"],
    ),
    winner=nirs4all.TuningWinner(
        X=X_cal,
        y_true=y_cal,
        score=0.3,
        metric="rmse",
        sample_ids=["cal-001", "cal-002", "cal-003", "cal-004"],
    ),
    workspace_tuning_id="ridge-native-tuning",
)

result = nirs4all.run(
    pipeline=[
        {"name": "scale", "transform": StandardScaler()},
        {"model": Ridge()},
    ],
    dataset=(X_dev, y_dev, ["train-001", "train-002", "train-003", "train-004"]),
    engine="dag-ml",
    workspace_path="workspace/",
    tuning=tuning,
    calibration=nirs4all.TuningCalibration(
        y_pred=[11.0, 12.0],
        prediction_sample_ids=["pred-001", "pred-002"],
        coverage=0.8,
        workspace_conformal_id="ridge-conformal",
    ),
)
```

Typed helpers normalize to the same dictionaries consumed by the runtime. They
do not expand the supported subset. `NativeTuning.to_tuning_spec()` returns the
deterministic optimizer contract; runtime-only blocks such as `score_data`,
`winner`, `calibration`, and workspace ids stay outside that fingerprint.
`force_params` is inside that deterministic contract because it changes the
optimizer trajectory: it warm-starts the first Optuna/n4m trial with explicit
public parameter values, fails closed when a key is not present in
`run.tuning.space`, and can invalidate calibration when it changes the selected
predictor.
Both optimizer adapters publish that warm-start through the same public tape:
the first `TrialResult.params`, `best_params` when the forced trial wins, and
the trial `search_space_fingerprint` are expressed in decoded nirs4all syntax.
Backend-specific labels or encoded categorical tokens are never part of the
public result.
The `tuning.space` mapping is lowered to an ordered search-space contract before
any optimizer sees it. Public dotted paths such as `ridge.alpha` and
sklearn double-underscore paths such as `ridge__alpha` are canonicalized to the
same dotted path; duplicate canonical paths are rejected instead of relying on
dictionary overwrite order.
Candidate assignments and `force_params` become ordered `ParameterPatch` records
with canonical string paths and JSON-native values, and trial diagnostics include a
`search_space_fingerprint` so a score tape can be tied back to the exact ordered
space that produced it.
In short, sklearn double-underscore paths are canonicalized to the same dotted
spelling as the public patch syntax.
The typed helper applies the same validation as mapping payloads: `storage` must
be an explicit URI such as `sqlite:///study.db`, and `study_name` is trimmed and
cannot contain NUL characters.
`TuningCalibration(...)` validates `coverage` as a scalar in `(0, 1)` or a
non-empty unique list of such values, normalizes `method` and `unit` to the
supported lower-case spellings, and refuses values outside
`CONFORMAL_CALIBRATION_METHODS` and `CONFORMAL_CALIBRATION_UNITS` before runtime
execution. Its `extra={...}` field is reserved for additional calibration
options only; it cannot override typed keys such as `coverage`, `method`, `unit`,
`prediction_sample_ids`, `y_pred`, workspace metadata, or inject
`calibration_data`. `TuningCalibration.as_predict_result` must be a boolean,
and `TuningCalibration.extra`, `TuningCalibration.workspace_metadata` and
`NativeTuning.workspace_metadata` require canonical non-empty string keys.
Their values must stay strict JSON-native: finite integers/floats, strings,
booleans, null, lists, or nested mappings with canonical string keys. Non-finite
numbers, bytes, tuples, sets and arbitrary Python objects fail before
publication. The same strict boundary is enforced again when workspace tuning
or conformal rows are persisted through `save_workspace_tuning_result(...)`,
`run(tuning=..., workspace_path=...)`, `TuningCalibration.workspace_metadata`,
or `save_workspace_calibrated_result(...)`; the workspace store does not
stringify Python objects into metadata. Prediction workspace publication applies
the same rule to `predict(..., save_to_workspace=True, workspace_metadata=...,
workspace_result_metadata=...)` and to
`save_workspace_predict_result(..., metadata=..., result_metadata=...)` before
writing the prediction sidecar.
`NativeTuning.space` and `force_params` canonicalize string patch keys before
serializing. Raw `NativeTuning(score_data={...})`, `winner={...}` and
`calibration={...}` mappings use the same fail-closed boundary before
`to_dict()` publishes them: dataset selectors require canonical string keys,
`include_augmented` and `as_predict_result` are strict booleans, winner scores
are finite numbers, and nested `calibration_data` is rejected. Raw calibration
`coverage`, `method`, `unit`, `workspace_metadata` and
`workspace_conformal_id` are validated and canonicalized like the typed
`TuningCalibration(...)` helper. Public `calibrate(..., result_metadata=...)`
and `predict_calibrated(..., result_metadata=...)` also require strict
JSON-native mappings before generated guarantee metadata is merged. Top-level
`NativeTuning` core fields are
validated through `DagMLTuningSpec` before publication: engine, metric and
direction are canonicalized, `n_trials`/`seed`/`resume` reject coercive values,
and storage, study and workspace tuning ids must be canonical strings.
`TuningScoreData.metric`/`score_metric` and raw
`NativeTuning.score_data.metric`/`score_metric` must be real non-empty strings;
non-string values, blank strings and NUL-containing strings fail closed, and the
published score metric is lowercase canonical `metric`.
`TuningWinner.metric`, `dataset_name`, `model_name` and `task_type`, plus the
same raw `NativeTuning.winner` fields and aliases, use the same no-stringify
rule; winner `metric` and `task_type` publish lowercase values, while
`dataset_name` and `model_name` publish trimmed labels.
`TuningPassthrough()` is the typed form of the structured
`{"kind": "passthrough"}` marker for optional non-final preprocessing steps;
forms, bindings and Studio should use it instead of inventing a separate
placeholder. The structured marker is exact: `kind` must be the literal string
`"passthrough"` and host objects that only stringify to that value are rejected
before native tuning can run.
For named categorical choices, `tuning.space` also accepts
`{"type": "categorical", "options": {"label": value}}`: Optuna/n4m see the
stable `label`, while `best_params`, trial rows and terminal refit receive the
public JSON-native `value`.
When `force_params` targets such a named categorical choice, callers still pass
the public decoded `value`; Optuna and n4m may encode it to a stable backend
label internally, including for `optimizer.enqueue(...)`, but both adapters
publish the decoded value back in `TrialResult.params` and `best_params`. They
reject internal labels or unknown keys instead of silently changing the
warm-start trial. The internal categorical codec is also fail-closed on direct
construction: choices must be non-empty, unique, optimizer-native without a
decoder, and TCV1 JSON-native; decoder keys must match encoded choices and
decoded public values must be unique before Optuna/n4m can use them.
The `pruner` key is transported to the selected optimizer for the supported
backend names (`none`, `median`, `successive_halving`, `hyperband`, plus native
n4m aliases such as `asha`/`racing` where available). The current shared
objective reports one final score per trial; it does not invent intermediate
fold/resource signals, so pruning is limited to what that backend can infer from
the available objective calls. If an Optuna objective raises
`optuna.exceptions.TrialPruned`, the adapter preserves that optimizer state as
`TrialResult(state="PRUNED", value=None)` with compact diagnostics
`score_extractor="pruned"` instead of collapsing it into `FAIL`; in the
reference contract, the adapter records a distinct
`TrialResult(state="PRUNED", value=None, diagnostics={...})`. The n4m adapter
uses the same portable row when the shared objective raises a prune exception,
but only if the installed native binding can terminalize the trial through
`optimizer.tell_result(..., TrialStatus.PRUNED)`. Older n4m bindings that cannot
record a pruned terminal state fail closed instead of converting the candidate
to `FAIL` or to a worst completed score.
`sampler` and `pruner` strings are normalized case-insensitively before adapter
dispatch, so forms such as `" TPE "` or `" HyperBand "` resolve to their
canonical lower-case values. The legacy Optuna and n4m managers apply the same
canonicalization to `approach`, `eval_mode`, and explicit `direction`, including
the `sample` alias for `sampler` and the `eval_mode="avg"` alias for
`eval_mode="mean"`; Optuna also canonicalizes per-phase `sampler` values before
multi-phase dispatch. Logs, traces and optimizer dispatch therefore use stable
values after validation. The shared Optuna and n4m adapters both accept
`sampler="grid"` for the portable categorical/grid subset; n4m fails closed with
an explicit enum error if the installed native optimizer bindings do not expose
their `GRID` sampler yet. Unknown sampler or pruner spellings are rejected
before optimizer construction; they do not silently fall back to TPE or to a
no-pruner study.
For Optuna persistence, `storage` must be an explicit URI string such as
`sqlite:///study.db`; a bare path like `study.db` is rejected. `study_name` is
trimmed and cannot contain NUL characters. When `resume=True`, the Optuna
adapter requires both `storage` and `study_name`; anonymous or in-memory resume
is rejected before optimizer execution. If `force_params` is also supplied while
resuming a non-empty study, it must match the already materialized warm-start
trial; the adapter does not enqueue a duplicate trial and fails closed if the
caller changes the warm-start assignment under the same `study_name`/`storage`.
Existing materialized trial params must also match the current `tuning.space`
keys exactly; changing the search-space keys under the same persisted study
fails closed before optimizer execution. Existing categorical values must also
still be present in the current choices for their key; removing or renaming a
choice under the same persisted study fails closed before optimizer execution.
Existing numeric values must also remain inside the current range for their key;
narrowing a range so that a stored trial falls outside it fails closed before
optimizer execution. If the current numeric space declares a `step`, restored
values must also lie on that grid. During Optuna storage resume, `n_trials` is
treated as the target total trial count: an already completed one-trial study
with `n_trials=1` runs no extra trial, while `n_trials=2` runs one remaining
trial. New Optuna storage-backed studies also persist nirs4all study
fingerprints in `study.user_attrs`: a format marker, schema version, optimizer
contract fingerprint and search-space fingerprint. Resume fails closed if those
fingerprints are missing from a non-empty study or no longer match.
Trials restored from Optuna storage keep compact summary diagnostics even when
they were produced by a previous process: completed rows use
`score_extractor="optuna_storage"`, failed rows use `score_extractor="failed"`,
and pruned rows use `score_extractor="pruned"`. Restored Optuna `COMPLETE`
rows must carry a finite numeric value; missing or non-finite storage values
fail closed instead of becoming completed trials with no usable objective value.
Restored non-`COMPLETE` rows must not carry a final storage value; failed,
pruned or in-flight rows with final values are rejected as corrupted optimizer
history. Restored `RUNNING` rows fail closed during resume because interrupted
active trials cannot be safely recovered into a terminal HPO tape. Restored terminal Optuna rows must keep exactly the current
`tuning.space` parameter keys when the search space is non-empty; rows whose
stored parameter table was removed fail closed instead of becoming completed
trials with empty public params. Restored queued Optuna `WAITING` rows that
already carry materialized params or `fixed_params` must also satisfy the current
`tuning.space`; incompatible values are rejected before Optuna can consume them.
Restored Optuna trial numbers must be canonical unique integers; corrupt storage
that yields non-integer or duplicate trial numbers fails closed before the HPO
tape is projected.
n4m optimizer-state persistence uses native N4MOPT checkpoints in this shared
objective seam: set `storage="file:///absolute/checkpoint-dir"` and
`study_name="filename-safe-name"`. After each terminal trial, nirs4all writes a
JSON manifest containing the native checkpoint bytes plus tuning/search-space
fingerprints. With `resume=True`, the adapter reloads that checkpoint only when
the optimizer contract still matches; changing the search space, sampler,
pruner, seed, direction, metric or warm-start assignment fails closed. The
`n_trials` value is treated as the target total trial count, so a checkpoint
with one completed trial and `n_trials=2` runs one remaining trial.
Restored n4m trial rows are decoded back to the public parameter syntax before
they enter `TuningResult.trials`, and are ordered canonically by numeric trial id
even if a native binding returns checkpoint records in another order. Duplicate
restored trial ids, or restored ids that are not canonical integers, fail closed
before `TuningResult` construction;
optimizer-only categorical labels used for JSON-native values or named `options` are never exposed as resumed
`TrialResult.params`. The restored checkpoint row params must still match the
current `tuning.space` keys and value domains; edited or incompatible checkpoint
keys, categorical choices, numeric ranges or numeric steps fail closed before
they can contribute optimizer history. Restored `COMPLETE` rows must carry a
finite numeric score; missing, boolean or non-finite scores fail closed instead
of becoming complete trials with no usable value. Restored non-success rows also keep compact diagnostics
aligned with their state: `score_extractor="failed"` for failed candidates and
`score_extractor="pruned"` for pruned candidates. If a native checkpoint
contains a terminal cancelled row, it is restored as `CANCELLED` with
`score_extractor="cancelled"`, so resumed summary cards do not collapse these
states into a generic checkpoint row. Non-terminal or unsupported checkpoint
rows such as `RUNNING` fail closed during resume instead of being counted as
completed optimizer history. Restored n4m non-`COMPLETE` rows must not carry a
final score; failed, pruned or cancelled checkpoint rows with scores are
rejected as corrupted optimizer history.
When an individual candidate fails during fit or scoring, the tuning tape keeps a
`FAIL` trial with `value=None` and diagnostic error metadata. Other candidates
can still complete; if all candidates fail, nirs4all returns the failed tuning
tape, uses a finite direction-worst `best_value` for fingerprint/summary
compatibility, and skips terminal refit because there is no winner.
nirs4all distinguishes this from Optuna- or n4m-pruned trials, which stay
visible as `PRUNED` rows in `TuningResult.trials`, `trial_states` and compact
summary rows.
n4m bindings must expose either `optimizer.tell_result(...)` with a failed trial
status or `optimizer.tell(...)` so the adapter can advance the native optimizer
after a candidate failure. Older bindings that cannot record failed candidates
are rejected fail-closed instead of silently losing the trial tape.
n4m prune-aware tapes additionally require
`optimizer.tell_result(..., TrialStatus.PRUNED)`.
Linear preprocessing steps may also use explicit `sklearn.*` class-path
mappings with `params`, or direct `sklearn.*` string steps when no constructor
parameters are needed. String imports work as direct model steps, named tuple
steps, and `transform`/`model` mapping values. When constructor parameters are
required inside a linear chain, preprocessing mappings may use
`{"transform": "sklearn.preprocessing.StandardScaler", "params": {"with_mean":
False}}` and final model mappings may use
`{"model": "sklearn.linear_model.Ridge", "params": {"fit_intercept": False}}`.
Constructor `params` must use canonical string keys and TCV1-compatible
JSON-native values; Python objects, tuples, bytes and non-finite numbers are
rejected before import or instantiation.
The `{"class": ..., "params": ...}` form remains available for preprocessing
steps and the direct single-model form.
Short aliases and arbitrary imports remain unsupported in the native tuning
compiler.
A single model can use the same declarative form directly, for example
`pipeline={"name": "ridge", "class": "sklearn.linear_model.Ridge", "params":
{"fit_intercept": False}}` or `pipeline={"name": "ridge", "model":
"sklearn.linear_model.Ridge", "params": {"fit_intercept": False}}` with
`space={"ridge.alpha": [...]}`.

The lower-level `nirs4all.tune_single_estimator()` helper also accepts
`NativeTuning(score_data=..., winner=..., calibration=...)` for explicit
array/tuple/list scoring and winner cohorts. It uses the same single-estimator
and linear compiler forms described above, including direct `sklearn.*` string
imports when no constructor parameters are needed and `{"class": ..., "params":
...}` steps when they are, plus the `{"steps": [...]}` and `{"pipeline": [...]}`
wrappers. That includes
`TuningConformalScoreCalibration` for temporary conformal-aware objective scoring,
`TuningWinner(...)` for the projected final winner entry, and
`TuningCalibration(...)` for the final integrated `tune → calibrate` call.
Dataset-backed `score_data` and `winner` still belong to `run(tuning=...)`, which
owns dataset loading and selector resolution.

## Important keywords and effects

| Keyword | Required now | Effect |
|---------|--------------|--------|
| `engine="dag-ml"` | yes | Selects the native public subset. Other engines reject `tuning` here. |
| `tuning.engine` | yes | Selects the optimizer adapter, currently `optuna` or `n4m` for the covered subset. |
| `tuning.space` | yes | Parameter search space. Linear chains use paths such as `scale.factor` or `model.alpha`; sklearn-style `scale__factor` is canonicalized to the same ordered dotted patch path. |
| `tuning.score_data` | yes | Explicit scoring cohort used by the objective. nirs4all does not derive it implicitly. Only mapping and tuple/list forms are accepted. Mapping form accepts `X`/`y` or `X_score`/`y_score`, optional `metric`/`score_metric`, sample id aliases, `groups`/`score_groups`, and `metadata`/`score_metadata`; tuple/list form is `(X_score, y_score, sample_ids, groups, metadata)` with exactly 2–5 fields and strict metadata keys. Identity fields must align with `y`; compatible estimators receive them in `predict()`. |
| `tuning.score_data.conformal_calibration` | optional | Explicit development calibration cohort for conformal-aware objective scoring. It fits a temporary calibrator per candidate and never replaces the final `run(..., calibration=...)` result. |
| `tuning.winner` | required for calibration | Explicit terminal prediction cohort used to create `RunResult.best`. |
| `tuning.calibration` | optional | Applies split conformal calibration after tuning and terminal winner projection. |
| `calibration` | optional | Top-level alias for `tuning.calibration`; requires `run(tuning=..., engine="dag-ml")` and must not be combined with nested `tuning.calibration`. |
| `workspace_path` | optional | Enables workspace persistence. The same path is inherited by `tuning.calibration` or top-level `calibration` unless overridden. |
| `workspace_tuning_id` | optional | Stable id for the persisted `TuningResult`. The raw alias `tuning_id` is accepted, but must not be combined with `workspace_tuning_id`; caller-provided ids must be canonical non-empty strings without surrounding whitespace or NULs. |
| `workspace_conformal_id` | optional | Stable id for the persisted conformal result; caller-provided ids must be canonical non-empty strings without surrounding whitespace or NULs. |
| `resume=True` | optional | Reuses an already completed workspace tuning result for this subset; it is not interrupted optimizer-state resume. |

Caller-provided conformal ids must be canonical non-empty strings without
surrounding whitespace or NULs. Caller-provided robustness ids must also be
canonical non-empty strings without surrounding whitespace or NULs. Omitting
these ids asks the workspace store to generate ids that remain outside the
scientific result fingerprints.

### Registry coverage reference

The public keyword registry is the canonical machine-readable contract for
forms, bindings, Studio and docs generation. The table below mirrors the native
tuning/conformal/robustness entries that this guide covers. `Changes` names the
state affected by the keyword; `Invalidates calibration` explains whether a
previous conformal result must be treated as stale.

| Registry path | Lifecycle | Changes | Invalidates calibration | Summary |
|---------------|-----------|---------|-------------------------|---------|
| `run.engine` | `execution` | `execution_backend` | `if_predictor_changes` | Selects the pipeline execution backend; dual is reserved but not implemented, and this does not select the HPO algorithm. |
| `run.tuning` | `search` | `candidate_fit, selection, final_predictor` | `always` | Fixed-topology full-DAG HPO argument. The public subset currently executes only `engine="dag-ml"` single-estimator array pipelines with explicit `score_data`; broader DAG shapes remain fail-closed. |
| `run.tuning.engine` | `search` | `optimizer_algorithm` | `if_predictor_changes` | Optimizer selector inside full-DAG tuning; it is distinct from `run.engine` and currently executes only the single-estimator array subset. |
| `run.tuning.space` | `search` | `parameter_patches, candidate_predictors, selection` | `always` | Fixed-topology parameter search space. The public subset supports value patches on a single estimator; structural axes and full pipeline recompilation remain planned. |
| `run.tuning.force_params` | `search` | `trial_sequence, candidate_fit, selection` | `if_predictor_changes` | Enqueues a caller-provided parameter assignment as the first optimizer trial. Keys must be a subset of `run.tuning.space`; values use the public decoded syntax, not optimizer-internal categorical labels. |
| `run.tuning.seed` | `search` | `trial_sequence, selected_predictor` | `if_predictor_changes` | Seeds the HPO trajectory. A different seed invalidates calibration only when it changes the selected predictor. |
| `run.tuning.storage` | `storage` | `optimizer_state` | `not_applicable` | Optuna storage URI for storage-backed studies, or n4m `file:///absolute/checkpoint-dir` storage for native N4MOPT checkpoints. Bare paths are rejected. |
| `run.tuning.study_name` | `storage` | `optimizer_state` | `not_applicable` | Optuna study name or n4m checkpoint filename stem. Whitespace is trimmed, NUL characters are rejected, and n4m also requires a filename-safe value. |
| `run.tuning.score_data` | `search` | `objective_scores, trial_ranking, selected_predictor` | `if_predictor_changes` | Supplies the explicit scoring cohort used by `run(tuning)` optimizer-driving. It is mandatory in the public subset and may be mapping, tuple/list, or explicit dataset-backed selector from `SpectroDataset`, `DatasetConfigs`, config mapping, or config/path string. |
| `run.tuning.score_data.conformal_calibration` | `search` | `objective_scores, trial_ranking` | `if_predictor_changes` | Provides an explicit development calibration cohort used only for conformal-aware objective scoring. It fits a temporary calibrator per candidate and never replaces the final `run(calibration=...)` result. |
| `run.tuning.score_data.conformal_coverage` | `search` | `objective_scores, trial_ranking` | `if_predictor_changes` | Selects the nominal coverage for the temporary conformal scorer inside `score_data`. It requires `score_data.conformal_calibration` and affects only trial ranking, not the final `run(calibration=...)` result. |
| `run.tuning.winner` | `refit` | `run_result_best, terminal_prediction_entry` | `replaces_existing` | Projects the refit winner to an explicit terminal prediction entry. It is required before `run(tuning).calibration` and can use arrays or an explicit dataset-backed selector from `SpectroDataset`, `DatasetConfigs`, config mapping, or config/path string. |
| `run.tuning.calibration` | `calibration` | `calibrator, calibrated_result` | `replaces_existing` | Runs conformal calibration immediately after the explicit winner projection. `calibration_data` is derived from `winner` and cannot be supplied here. |
| `run.calibration` | `calibration` | `calibrator, calibrated_result` | `replaces_existing` | Top-level alias for `run(tuning).calibration` in the public DAG-ML tuning subset. It requires `run(tuning=..., engine="dag-ml")` and derives `calibration_data` from `tuning.winner`. |
| `run.tuning.workspace_tuning_id` | `storage` | `workspace_tuning_results` | `not_applicable` | Names the workspace `TuningResult` record and is required with `resume=True` in the public `run(tuning)` subset. |
| `run.tuning.workspace_metadata` | `storage` | `workspace_tuning_results` | `not_applicable` | Persists caller-provided strict JSON-native metadata alongside the workspace `TuningResult` without changing optimizer behavior. |
| `run.tuning.resume` | `search` | `terminal_refit, terminal_prediction_entry` | `if_predictor_changes` | Reuses an already completed matching workspace `TuningResult` in the public subset. It is not interrupted optimizer checkpoint resume. |
| `run.tuning.calibration.workspace_conformal_id` | `storage` | `workspace_conformal_results` | `not_applicable` | Names the conformal workspace record produced by `run(tuning).calibration`; the top-level `run(workspace_path=...)` is inherited unless overridden in the calibration payload. |
| `calibrate.calibration_data` | `calibration` | `calibrator` | `replaces_existing` | Exchangeable calibration cohort. The public API supports replayed-array mappings and explicit dataset-backed selectors with replayed predictions, an in-memory predictor, a saved predictor bundle, a `RunResult.best`-like prediction entry, or a stored workspace chain plus physical sample ids. |
| `calibrate.calibration_data.dataset` | `calibration` | `calibration_targets, calibration_identity_source` | `replaces_existing` | Selects explicit dataset-backed calibration evidence from a `SpectroDataset`, `DatasetConfigs` object, dataset config mapping, or config/path string. |
| `calibrate.calibration_data.selector` | `calibration` | `calibration_rows` | `replaces_existing` | Filters the explicit dataset-backed calibration cohort. It is mandatory so nirs4all never guesses a calibration partition. |
| `calibrate.calibration_data.y_pred` | `calibration` | `nonconformity_scores, calibrator` | `replaces_existing` | Provides already replayed point predictions for the calibration cohort. It does not trigger model replay. |
| `calibrate.calibration_data.predictor` | `calibration` | `calibration_predictions, nonconformity_scores, calibrator` | `replaces_existing` | Replays an in-memory sklearn-like predictor on the selected dataset-backed calibration cohort. `sample_ids`/groups/metadata are forwarded when `predict()` accepts them. |
| `calibrate.calibration_data.predictor_bundle` | `calibration` | `calibration_predictions, nonconformity_scores, calibrator` | `replaces_existing` | Replays a saved predictor path through `nirs4all.predict()` on the selected dataset-backed calibration cohort. |
| `calibrate.calibration_data.predictor_result` | `calibration` | `calibration_predictions, nonconformity_scores, calibrator` | `replaces_existing` | Replays a `RunResult.best`-like prediction entry through `nirs4all.predict()` on the selected dataset-backed calibration cohort. |
| `calibrate.calibration_data.predictor_chain_id` | `calibration` | `calibration_predictions, nonconformity_scores, calibrator` | `replaces_existing` | Replays a stored workspace chain through `nirs4all.predict(chain_id=...)` on the selected dataset-backed calibration cohort. Requires `calibration_data.workspace_path`. |
| `calibrate.calibration_data.workspace_path` | `calibration` | `calibration_predictions` | `replaces_existing` | Provides the explicit workspace root used by `predictor_result` or `predictor_chain_id` replay during dataset-backed calibration. It is not the conformal result persistence workspace. |
| `calibrate.calibration_data.sample_ids` | `calibration` | `calibration_cohort_manifest, coverage_claim_unit` | `replaces_existing` | Supplies explicit row-aligned physical sample ids for calibration. `calibration_sample_ids` and `physical_sample_ids` are raw aliases; missing or duplicated ids remain fail-closed. |
| `calibrate.calibration_data.groups` | `calibration` | `calibration_cohort_manifest, coverage_claim_scope` | `replaces_existing` | Supplies optional row-aligned group labels for raw replayed-array calibration mappings. `group_by="group"` consumes these labels for grouped conformal calibration. |
| `calibrate.calibration_data.metadata` | `calibration` | `calibration_cohort_manifest, coverage_claim_scope` | `replaces_existing` | Supplies optional row-aligned metadata for raw replayed-array calibration mappings. Metadata keys used by `group_by` select grouped conformal partitions. |
| `calibrate.calibration_data.sample_id_column` | `calibration` | `calibration_cohort_manifest, coverage_claim_unit` | `replaces_existing` | Extracts physical sample ids from the selected `SpectroDataset` metadata column when `sample_ids` are not supplied directly. |
| `calibrate.calibration_data.group_column` | `calibration` | `calibration_cohort_manifest` | `replaces_existing` | Extracts optional row-aligned group labels into the calibration cohort manifest; it does not enable grouped conformal guarantees by itself. |
| `calibrate.calibration_data.metadata_columns` | `calibration` | `calibration_cohort_manifest` | `replaces_existing` | Copies selected `SpectroDataset` metadata fields into the calibration cohort manifest for audit and diagnostics. |
| `calibrate.method` | `calibration` | `calibrator` | `replaces_existing` | Conformal score and interval construction method. V1 currently supports split absolute residuals on replayed regression predictions. |
| `calibrate.coverage` | `calibration` | `quantile, prediction_bounds` | `extends_existing` | Nominal coverage used to select the finite-sample conformal quantile; retained scores allow deterministic interval materialization for replayed-array calibration. |
| `calibrate.unit` | `calibration` | `score_aggregation, coverage_claim` | `replaces_existing` | V1 exchangeability unit. The current public surface requires explicit physical sample ids for replayed-array calibration. |
| `calibrate.group_by` | `calibration` | `calibration_partitions, coverage_claim_scope` | `replaces_existing` | Grouped split conformal is executable for replayed-array calibration when calibration and prediction rows provide matching group evidence. Report slices alone never create a grouped conformal guarantee. |
| `calibrate.prediction_groups` | `prediction` | `selected_prediction_bounds, coverage_claim_scope` | `not_applicable` | Row-aligned prediction labels for `group_by="group"`; missing or unseen groups fail closed without global fallback. |
| `calibrate.prediction_metadata` | `prediction` | `selected_prediction_bounds, coverage_claim_scope` | `not_applicable` | Row-aligned prediction metadata for non-`group` `group_by` keys; every key must be present, non-null and JSON-compatible. |
| `calibrate.multi_target` | `calibration` | `nonconformity_score, prediction_region, coverage_claim_scope` | `replaces_existing` | `marginal` supports one-dimensional targets; `joint_max` supports simultaneous multi-target regions for two-dimensional replayed arrays. |
| `calibrate.result_metadata` | `calibration` | `calibrated_result_metadata, calibrated_result_fingerprint` | `not_applicable` | Stores strict JSON-native result metadata before nirs4all merges generated guarantee metadata. It changes the calibrated result fingerprint, not the fitted conformal artifact. |
| `predict_calibrated.result_metadata` | `prediction` | `calibrated_result_metadata, calibrated_result_fingerprint, source_calibrated_result_fingerprint` | `not_applicable` | Stores strict JSON-native result metadata when applying an existing calibrated result to new replayed predictions; nirs4all appends guarantee/source metadata after validation. |
| `predict.coverage` | `prediction` | `selected_prediction_bounds` | `not_applicable` | Selects already materialized calibrated coverages at prediction time for calibrated replayed-array results/stores/bundles and model `.n4a` bundles with a conformal sidecar. |
| `predict.all_predictions` | `prediction` | `prediction_entries` | `not_applicable` | Requests all model prediction entries. With a conformal sidecar and `coverage=...`, `all_predictions=True` remains fail-closed until each entry carries calibrated identity mapping. |
| `predict.save_to_workspace` | `prediction_storage` | `workspace_prediction_rows, prediction_arrays, result_metadata, workspace_prediction_id` | `not_applicable` | Publishes the returned `PredictResult` through the workspace prediction store, including executable `X`/`spectra` when available from `data`; it does not persist conformal artifacts or renew guarantees. |
| `predict.workspace_metadata` | `prediction_storage` | `prediction_sample_metadata` | `not_applicable` | Stores caller-provided strict JSON-native sample-level metadata alongside the workspace prediction sidecar when `predict.save_to_workspace` is true. |
| `predict.workspace_result_metadata` | `prediction_storage` | `result_metadata, robustness_evidence` | `not_applicable` | Stores strict JSON-native result-level metadata for the workspace prediction row, including `robustness_evidence.predictor_bundle` for later spectral/OOD replay. |

Workspace prediction ids are canonical non-empty strings without surrounding
whitespace or NULs. The public prediction publisher generates the id it returns
as `metadata["workspace_prediction_id"]`; lower-level explicit `prediction_id`
writes fail closed on invalid ids instead of silently generating or stringifying
a replacement.
Workspace link ids stored beside tuning, conformal and robustness rows follow
the same rule when supplied: `run_id`, `pipeline_id`, `chain_id`, and source
`prediction_id`/`conformal_id` must be canonical non-empty strings without
surrounding whitespace or NULs.

| `robustness.mode` | `robustness` | `robustness_report` | `mode_dependent` | Selects the robustness reuse policy. The current public API supports `clean_frozen` audit-only reports on already replayed predictions; matched recalibration and structural refit remain planned. |
| `robustness.scenarios` | `robustness` | `robustness_results` | `mode_dependent` | Defines report cells for robustness diagnostics. The current public API accepts `RobustnessScenarioSpec` helpers or equivalent mappings for observed, prediction-space and explicit-X spectral scenarios. |
| `robustness.scenarios[].kind` | `robustness` | `robustness_results` | `mode_dependent` | Selects the audit-only scenario cell. `observed` leaves predictions unchanged; prediction-space scenarios alter already materialized predictions; spectral scenarios replay an explicit frozen predictor on perturbed `X`. |
| `robustness.scenarios[].severity` | `robustness` | `robustness_results` | `mode_dependent` | Controls scenario magnitude: offset, noise standard deviation, scale/ramp amplitude or fractional feature-axis shift depending on the scenario kind. |
| `robustness.scenarios[].distribution` | `robustness` | `robustness_results` | `mode_dependent` | Selects the stochastic scenario distribution. The current executable subset accepts distribution only for `prediction_noise` and `spectral_noise`, with `normal` and centered `uniform` noise supported fail-closed. |
| `robustness.X` | `robustness` | `robustness_results` | `mode_dependent` | Supplies the explicit input matrix used by spectral scenarios. It is never fitted on; it is perturbed for evaluation-only frozen predictor replay. |
| `robustness.predictor` | `robustness` | `robustness_results` | `mode_dependent` | Frozen predictor replay hook for spectral scenarios. The current Python surface accepts `predictor.predict(X)` or a callable and does not refit or recalibrate it. |
| `robustness.predictor_bundle` | `robustness` | `robustness_results` | `mode_dependent` | Saved predictor path replay hook for spectral scenarios. The current Python surface routes perturbed `X` through `nirs4all.predict(model=predictor_bundle, data={"X": X, "sample_ids": ...})` and does not refit or recalibrate it. |
| `robustness.slice_by` | `reporting` | `reported_metrics` | `not_applicable` | Adds diagnostic report slices over supplied metadata. A sliced empirical coverage is not a conditional conformal guarantee unless a separate valid grouped calibrator supports it. |
| `robustness.seed` | `robustness` | `robustness_results` | `mode_dependent` | Controls reproducible audit-only stochastic scenarios such as `prediction_noise`. It does not change a frozen predictor or calibrator; omitted seeds use `effective_seed=0`. |
| `robustness.workspace_path` | `reporting` | `workspace_robustness_results` | `not_applicable` | Persists the verified `RobustnessReport` in the workspace after the audit is computed. This does not change metrics, fingerprints, predictions, or conformal guarantees. |
| `robustness.workspace_robustness_id` | `reporting` | `workspace_robustness_results` | `not_applicable` | Stable id for the workspace robustness row produced by `robustness(workspace_path=...)`. If omitted, the store generates an id that is not injected into the report fingerprint; if provided, it must be a canonical non-empty string without surrounding whitespace or NULs. |
| `robustness.workspace_name` | `reporting` | `workspace_robustness_results` | `not_applicable` | Human-readable workspace label for a persisted robustness report. It affects only workspace inventory and does not enter the report fingerprint. |
| `robustness.workspace_metadata` | `reporting` | `workspace_robustness_results` | `not_applicable` | Strict JSON-native workspace metadata stored alongside the robustness report row for CI/Studio indexing. It does not enter the report fingerprint. |

For tuning, `workspace_metadata` is constrained by the published keyword
registry schema to strict JSON-native values with canonical string keys; it is
display/storage metadata only and never changes optimizer behavior.
Robustness workspace metadata follows the same strict JSON-native published
schema and the workspace store rejects invalid values before inserting the
`robustness_results` row.

The calibration payload cannot provide its own `calibration_data` inside
`run(tuning=...)`, whether calibration is nested under `tuning.calibration` or
passed as top-level `run(..., calibration=...)`. Calibration evidence is derived
from `tuning.winner` so the order remains auditable: tune, terminal
refit/projection, then conformal calibration. The HPO scoring cohort from
`tuning.score_data` is never reused as final calibration evidence; if the final
intervals differ from score-data residuals, the `tuning.winner` cohort is
authoritative. In the typed helper, `TuningCalibration.extra` follows the same
rule and also refuses overrides of typed calibration keys.
When `workspace_conformal_id` is supplied, the workspace row stores the
winner-derived calibrated result, so reloading it preserves those intervals and
does not recompute them from `tuning.score_data`.
Reloaded `CalibratedRunResult` objects can be converted back to the public
prediction surface with `calibrated.to_predict_result()`, preserving intervals,
`calibrated_result_fingerprint`, `calibration_replay_source`, and
`tuning_calibration_source` for notebooks, bindings, CLI views and Studio.
The conversion and reload path validates `conformal_guarantee_status` against
the embedded conformal artifact; stale fingerprints, method/unit mismatches or
non-materialized coverage selections fail closed rather than displaying a false
guarantee.
The calibrated prediction metadata also includes
`tuning_calibration_source={"source": "tuning.winner",
"score_data_role": "hpo_objective_only", "score_data_used": false}`. Studio,
Python bindings and audit tools can use this field to show that
`tuning.score_data` influenced optimizer ranking only, not the final conformal
evidence. If the calibration payload supplies `result_metadata`, nirs4all keeps
those user fields and adds `tuning_calibration_source` only when it is absent.
Use `prediction.tuning_calibration_source` or
`calibrated.tuning_calibration_source` for the direct accessor instead of
parsing `metadata` manually.

The tuple/list dataset order is fixed: `(X, y, sample_ids, groups, metadata)`.
Use `None` for an omitted middle field. Tuple/list forms with extra fields are
rejected instead of guessed, and all identity fields must align with `y`.

The `score_data` tuple/list follows the same rule for the scoring cohort:
`(X_score, y_score, sample_ids, groups, metadata)`. Use mapping form when you
need to override the scoring metric for that cohort. `NativeTuning.to_dict()`
validates tuple/list `score_data` before publication: it must contain between
two and five fields, metadata must use canonical non-empty string keys in either
column-style mapping or row-style sequence form, metadata values must stay
strict JSON-native and finite, and tuple inputs publish as JSON-native lists.
Any other `score_data` type, including scalar strings, bytes, numbers and
arbitrary objects, is rejected before publication.

The replayed-array `calibration_data` tuple is similarly explicit:
`(y_true, y_pred, sample_ids, groups, metadata)`. `sample_ids` are still
required for conformal calibration evidence; use mapping form when you need
named fields or dataset-backed selectors. Raw replayed-array mappings accept the
same canonical payload plus the public aliases `y_pred_calibration` or
`calibration_predictions` for `y_pred`, `calibration_sample_ids` or
`physical_sample_ids` for `sample_ids`, `calibration_groups` for `groups`, and
`calibration_metadata` for `metadata`. Providing more than one alias for the same
field is rejected as ambiguous before calibration.

You can also use the typed helper when building forms or bindings:

```python
calibration_data = nirs4all.ConformalCalibrationData(
    y_true=y_cal,
    y_pred=y_cal_pred,
    sample_ids=calibration_sample_ids,
    groups=calibration_groups,
    metadata=calibration_metadata,
)

calibrated = nirs4all.calibrate(
    calibration_data=calibration_data,
    y_pred=y_pred,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Grouped split conformal, also called Mondrian conformal, is executable for the
same replayed-array substrate when both calibration and prediction rows carry
the declared grouping evidence. Use `group_by="group"` to consume
`calibration_groups` and `prediction_groups`; use metadata keys in `group_by`
when the row-aligned `calibration_metadata` and `prediction_metadata` mappings
contain those keys. Missing prediction groups, null metadata values, or groups
not seen during calibration fail closed; nirs4all never falls back to the global
calibrator for a grouped request. Group labels are strict strings too:
whitespace-padded labels, NUL-containing labels and non-string labels fail before
group quantiles are fitted or selected.

```python
calibrated = nirs4all.calibrate(
    y_true=y_cal,
    y_pred_calibration=y_cal_pred,
    y_pred=y_pred,
    calibration_sample_ids=calibration_sample_ids,
    prediction_sample_ids=prediction_sample_ids,
    calibration_groups=calibration_groups,
    prediction_groups=prediction_groups,
    group_by="group",
    coverage=0.9,
)

status = calibrated.conformal_guarantee_status
print(status["guarantee"])  # split_conformal_group_marginal_coverage
```

`calibrate.group_by` is therefore `partial`: it changes
`calibration_partitions` and `coverage_claim_scope`, and replaces any existing
calibrator. `calibrate.prediction_groups` and `calibrate.prediction_metadata`
are `partial`, `not_applicable` to calibration invalidation, and change
`selected_prediction_bounds` plus `coverage_claim_scope` because they route
prediction rows to already fitted group quantiles.

For multi-target regression, `multi_target="joint_max"` is executable on
two-dimensional replayed arrays. nirs4all stores one nonconformity score per
physical sample, `max(abs(y_true - y_pred))` across target columns, and applies
the resulting scalar `qhat` to every target column of the prediction row. The
guarantee is simultaneous for the full target vector, not a separate conditional
claim per target.

```python
calibrated = nirs4all.calibrate(
    y_true=y_cal_2d,
    y_pred_calibration=y_cal_pred_2d,
    y_pred=y_pred_2d,
    calibration_sample_ids=calibration_sample_ids,
    prediction_sample_ids=prediction_sample_ids,
    multi_target="joint_max",
    coverage=0.9,
)

print(calibrated.conformal_guarantee_status["guarantee"])
# split_conformal_joint_max_simultaneous_coverage
```

For conformal-aware development scoring, use mapping form with
`conformal_calibration` and a conformal metric:

```python
"score_data": {
    "X": X_score,
    "y": y_score,
    "metric": "conformal_mean_width",
    "conformal_coverage": 0.9,
    "conformal_calibration": {
        "X": X_dev_calibration,
        "y_true": y_dev_calibration,
        "sample_ids": dev_calibration_ids,
    },
}
```

Or use the typed helper when building public configs in Python:

```python
tuning = nirs4all.NativeTuning(
    engine="optuna",
    space={"alpha": [0.2, 0.9]},
    metric="conformal_mean_width",
    direction="minimize",
    score_data=nirs4all.TuningScoreData(
        X=X_score,
        y=y_score,
        conformal_coverage=0.9,
        conformal_calibration=nirs4all.TuningConformalScoreCalibration(
            X=X_dev_calibration,
            y_true=y_dev_calibration,
            physical_sample_ids=dev_calibration_ids,
        ),
    ),
)
```

The typed form is checked early: `conformal_coverage` is valid only with
`conformal_calibration`, must be a finite numeric scalar in `(0, 1)`, and
conformal score metrics require that temporary calibration cohort.
Numeric strings are rejected; coverage values must be real numeric scalars, not
host values that can be converted with `float(...)`.
`TuningScoreData(...)` accepts either `X`/`y` or the score-cohort aliases
`X_score`/`y_score`, plus either `metric` or its input alias `score_metric`, then
emits canonical `X`/`y` and `metric`; it also accepts `groups`/`score_groups` and
`metadata`/`score_metadata`, emitted canonically as `groups`/`metadata`.
The score metric must be a real non-empty string, not an integer, boolean,
blank string, or NUL-containing value; it is published as lowercase canonical
`metric` before objective scoring.
Metadata keys must be canonical non-empty strings for both column-style mappings
and row-style sequences of mappings; raw `NativeTuning.score_data.metadata`
follows the same rule. Metadata values must be strict JSON-native and finite:
lists and nested mappings are accepted, but non-finite numbers, bytes, tuples
and Python objects are rejected before runtime. Providing multiple aliases in
any group is rejected as ambiguous.
`TuningConformalScoreCalibration(...)` accepts exactly one feature alias among
`X`, `X_calibration`, and `features`, plus exactly one target alias among
`y_true`, `y`, `y_calibration`, `target`, and `targets`; it emits canonical
`X`/`y_true`. It also accepts `sample_ids`, `calibration_sample_ids`, or
`physical_sample_ids`, `groups`/`calibration_groups`, and
`metadata`/`calibration_metadata`, then emits canonical `sample_ids`, `groups`,
and `metadata` for the runtime. Metadata keys must be canonical non-empty
strings for both column-style mappings and row-style sequences of mappings;
raw `score_data.conformal_calibration.metadata` follows the same rule before
`NativeTuning.to_dict()` can publish it, including the same strict JSON-native
value boundary. The same one-alias rule is used by raw replayed-array
`calibrate(calibration_data={...})` mappings for `y_pred_calibration` /
`calibration_predictions`, `calibration_sample_ids` / `physical_sample_ids`,
`calibration_groups`, and `calibration_metadata`. Mapping payloads may use the
read-only alias
`coverage`, but `conformal_coverage` is the canonical spelling used by the
registry. Multi-coverage lists belong to final calibration/prediction, not to the
temporary tuning scorer. Final `TuningCalibration.coverage` accepts one numeric
coverage or a non-empty unique list of numeric coverages; list elements such as
`"0.8"` fail closed instead of being coerced. Raw mapping payloads follow the
same alias exclusivity rules as the typed helpers for both standard and
conformal-aware score data; nirs4all rejects ambiguous mappings before building
the scorer.

This calibrator is temporary and candidate-local. It is used only to rank HPO
trials; final intervals still come from `tuning.winner` plus top-level
`calibration` or nested `tuning.calibration`.

The completed `tuning_result.trials[*].diagnostics` also records this boundary
for audit: `score_family="conformal"`,
`score_extractor="conformal_temporary_calibration"`, and
`final_calibration_scope="unmodified_by_score_data"`.
The lightweight `TuningResult.summary_artifact()` mirrors a compact subset of
those diagnostics under `trials[*].diagnostics` so bindings, Studio and CI cards
can explain `FAIL` or conformal-aware trials without loading the full HPO tape.
It keeps scalar audit keys such as `error_type`, `score_family`,
`score_extractor`, `search_space_fingerprint` and `tuning_fingerprint`, but
does not include candidate params or raw exception messages.

`score_data` can also use an explicit dataset-backed selector:

```python
"score_data": {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "val"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site"],
}
```

The selector is mandatory here too. This keeps objective scoring auditable and
prevents nirs4all from silently deriving a validation cohort. If the source is a
`DatasetConfigs`, config mapping, or path, it must resolve to exactly one
dataset. The Python helper form `TuningScoreData(dataset=..., selector=...)`
executes this same selected cohort and transports `sample_id_column`,
`group_column`, and `metadata_columns` to compatible `predict()` methods.
Those dataset-backed column selectors are strict identifiers:
`sample_id_column` and `group_column` must be canonical non-empty strings
without surrounding whitespace or NULs, and `metadata_columns` must be either one
canonical string or a duplicate-free sequence of canonical strings. Raw
`NativeTuning.score_data` and `NativeTuning.winner` dataset-backed mappings use
the same column-selector rule before publication.
It also accepts `sample_ids`, `score_sample_ids`, `prediction_sample_ids`, or
`physical_sample_ids`, serializing exactly one provided alias to canonical
`sample_ids`. It rejects mixed `dataset` + explicit `X`/`y` arrays, matching the
mapping runtime boundary.

Dataset-backed inputs must be explicit mappings. nirs4all does not guess a
training partition:

```python
dataset = {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "train"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site"],
}
```

`include_augmented` defaults to `False` for this native tuning subset. Set
`"include_augmented": True` only when augmented samples are intended to be part
of the fit cohort. Dataset-backed tuning selectors reject non-string or
whitespace-padded keys plus non-JSON-native values, and `include_augmented`
must be a boolean before `TuningScoreData` or `TuningWinner` can publish a
runtime mapping. Config/path sources must resolve to exactly one dataset.

The same mapping form is available for `tuning.winner`, which is the cohort used
for `RunResult.best` and conformal calibration:

```python
"winner": {
    "dataset": spectro_dataset,  # or DatasetConfigs / config mapping / "dataset.json"
    "selector": {"partition": "test"},
    "sample_id_column": "Sample_ID",
    "group_column": "Batch",
    "metadata_columns": ["Site"],
    "score": 0.42,
    "metric": "rmse",
}
```

Do not combine this with explicit `X`/`y_true` arrays in the same `winner`
payload. The typed helper form `TuningWinner(dataset=..., selector=...)`
projects the same selected cohort into `RunResult.best`, preserving physical
sample ids, optional groups, and selected metadata columns for calibration and
audit, and rejects the same mixed `dataset` + explicit `X`/`y_true` payloads.
Use one representation per cohort. `TuningWinner.score` must be a finite number,
not a boolean or numeric string. `TuningWinner.metadata` and raw
`NativeTuning.winner.metadata` use the same strict metadata key rule as
`TuningScoreData`; their values must also stay strict JSON-native and finite.
`TuningWinner.metric`, `dataset_name`, `model_name` and `task_type` are strict
string fields: nirs4all rejects non-string values, blanks and NULs instead of
calling `str(...)`. `metric` and `task_type` are published lowercase; dataset and
model labels are published trimmed.

For the winner cohort, `sample_ids`, `prediction_sample_ids`, and
`physical_sample_ids` are accepted aliases. The typed `TuningWinner(...)` helper
also accepts these aliases plus `winner_sample_ids`, and serializes exactly one
provided alias to canonical `sample_ids`. Raw `winner` mappings follow the same
fail-closed rule for `X`/`winner_x`, `y_true`/`winner_y_true`, score, metric,
sample id, dataset name, model name, task type, and metadata aliases; provide one
spelling per field. They are stored as physical sample ids in the prediction
metadata and are required before the winner can be used as conformal calibration
evidence. When the selected `SpectroDataset` target is a single-column matrix,
the native tuning adapter normalizes it to a one-dimensional target vector before
conformal calibration; multi-target matrices remain outside
this public subset.

## Calibrating from an explicit dataset cohort

`nirs4all.calibrate()` can also use a selected dataset cohort directly when you
already have replayed point predictions for that exact cohort. The `dataset`
value may be a `SpectroDataset`, a `DatasetConfigs` object, a dataset config
mapping, or a config/path string accepted by `DatasetConfigs`. The dataset
supplies the calibration targets and identities; either `y_pred` inside the
mapping or top-level `y_pred_calibration=` supplies the replayed calibration
predictions. Alternatively, `predictor` may provide an in-memory sklearn-like
object; nirs4all extracts `X` from the same selected cohort and calls
`predict(X)`, forwarding `sample_ids`, `groups` and `metadata` when accepted by
the predictor signature. For saved predictors, use `predictor_bundle` to route
through `nirs4all.predict()` on the extracted `X`. Mapping aliases
`model_bundle`, `predictor_path` and `model_path` are accepted for this same
source; the typed helper emits canonical `predictor_bundle`. For previous
nirs4all training outputs, use `predictor_result=train_result.best`; for a
stored workspace chain, use `predictor_chain_id` together with an explicit
nested `workspace_path`. The typed helper also accepts `workspace_chain_id` as
a read alias and serializes it as canonical `predictor_chain_id`. Both spellings
must carry a canonical non-empty chain id without surrounding whitespace or NULs;
nirs4all rejects invalid values before calling `predict(chain_id=...)` or
publishing replay provenance.

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": spectro_dataset,  # or "calibration_dataset.json"
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "group_column": "Batch",
        "metadata_columns": ["Site"],
        "y_pred": y_cal_pred,
    },
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

The typed helper serializes to the same payload and is the preferred shape for
forms, bindings and Studio payload builders:

```python
calibrated = nirs4all.calibrate(
    calibration_data=nirs4all.ConformalCalibrationData(
        dataset=spectro_dataset,
        selector={"partition": "calibration"},
        sample_id_column="Sample_ID",
        group_column="Batch",
        metadata_columns=["Site"],
        y_pred=y_cal_pred,
    ),
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

For in-memory replay, replace `y_pred=` with `predictor=fitted_model`. The
selected calibration `X`, sample ids, groups and metadata are forwarded to
`predict()` when the predictor accepts them. For saved model bundles, use
`predictor_bundle="model.n4a"` in the same typed helper. For nirs4all replay
sources, use `predictor_result=train_result.best` or
`predictor_chain_id="chain-id"` with `workspace_path="workspace/"`.

If you prefer to keep all prediction arrays at the top level:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": spectro_dataset,
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
    },
    y_pred_calibration=y_cal_pred,
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

In-memory predictor replay:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "group_column": "Batch",
        "metadata_columns": ["Site"],
        "predictor": fitted_model,
    },
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Saved predictor replay:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_bundle": "model.n4a",
    },
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Previous-result replay:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_result": train_result.best,
        "workspace_path": "workspace",
    },
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Workspace-chain replay:

```python
calibrated = nirs4all.calibrate(
    calibration_data={
        "dataset": "calibration_dataset.json",
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_chain_id": "chain-abc123",
        "workspace_path": "workspace",
    },
    y_pred=y_pred_new,
    prediction_sample_ids=prediction_sample_ids,
    coverage=0.9,
)
```

Provide exactly one calibration replay source inside the mapping: `y_pred`,
`predictor`, `predictor_bundle`, `predictor_result`, or `predictor_chain_id`.
The selector is mandatory, physical sample ids must be explicit through
`sample_ids`, `calibration_sample_ids`, `physical_sample_ids`, or
`sample_id_column`, and mixed `dataset` + `y_true`/`X` payloads are rejected.
Physical sample id values for calibration and prediction cohorts must be
canonical non-empty strings without surrounding whitespace or NULs.
Supplying more than one explicit sample-id alias for a dataset-backed calibration
mapping is rejected as ambiguous; use `sample_id_column` only when the ids should
be read from dataset metadata.
Optional row-aligned groups and metadata can be supplied explicitly with
`groups`/`calibration_groups` and `metadata`/`calibration_metadata`, or derived
from `group_column` and `metadata_columns`.
Dataset-backed selectors reject non-string or whitespace-padded
keys plus non-JSON-native values, and raw mapping `include_augmented` must be a
boolean. Dataset-backed `predictor_chain_id`/`workspace_chain_id` values are
strict workspace chain ids, not labels; blanks, whitespace-padded strings, NULs
and non-strings fail before replay. Dataset-backed calibration column selectors
are canonical too:
`sample_id_column` and `group_column` must be non-empty strings without
surrounding whitespace or NULs, and `metadata_columns` must be one canonical
string or a duplicate-free sequence of canonical strings. The same rule applies
to `ConformalCalibrationData(...)` and raw `calibration_data={...}` mappings
before cohort extraction. The nested `workspace_path` is used for predictor
replay only; it is separate from the top-level workspace path used to save
conformal results.

## Applying a stored calibrator

Reload the conformal artifact from the workspace and apply it to new point
predictions:

```python
calibrated = nirs4all.load_workspace_calibrated_result(
    "workspace/",
    "ridge-conformal",
)

prediction = nirs4all.predict_calibrated(
    calibrated,
    y_pred=[13.0, 14.0],
    prediction_sample_ids=["pred-003", "pred-004"],
)

interval = prediction.interval(0.8)
print(interval.lower, interval.upper)
```

The direct `predict_calibrated()` example above does not replay a model. It
applies the stored conformal calibrator to already computed point predictions.
The `prediction_sample_ids` are required for every non-empty conformal prediction
cohort and must align with `y_pred`. They must be non-empty canonical strings,
unique physical sample identifiers, and disjoint from the calibration cohort
physical sample ids embedded in the conformal artifact; empty ids, surrounding
whitespace, NUL-containing ids, duplicating a prediction id, or reusing a
calibration sample as a prediction row fail closed instead of displaying a leaked guarantee. The
returned result preserves the original `calibration_replay_source` from the
stored calibrated result; caller-supplied `result_metadata` cannot replace that
source or the generated `conformal_guarantee_status`. The top-level
`source_calibrated_result_fingerprint` must match the same value inside
`conformal_guarantee_status`, so corrupted or mixed application metadata fails
closed on reload. If you pass
`coverage=...`, the value selects one or more already materialized coverages:
it must be finite, unique when passed as a list, and strictly between 0 and 1.
It does not recalibrate the conformal artifact. If the requested coverage was
not materialized during calibration, `predict()` fails closed and reports the
available coverages.

The same identity contract is rechecked when loading filesystem stores,
workspace conformal rows, or `.n4a` conformal sidecars. A stored
`calibrated_result.json` whose non-empty prediction cohort lacks canonical
physical `sample_ids` fails reload instead of becoming a partial
`PredictResult`. A corrupted `conformal_results` workspace row with the same
missing prediction identity is rejected by both
`load_workspace_calibrated_result(...)` and
`load_workspace_calibrated_predict_result(...)`.
Workspace conformal metadata is also validated before insertion, so invalid
metadata cannot be hidden in the row and then trusted on reload.
Result-level metadata must also be strict JSON-compatible before construction,
reload, fingerprinting or persistence: non-finite floats, Python objects,
non-string keys, whitespace-padded keys, nested non-string mapping keys and
tuple values that would otherwise be silently coerced fail closed.
Grouped conformal results keep the same contract through these persistence
paths: filesystem stores, workspace `conformal_results` rows, conformal-only
`.n4a` bundles and model `.n4a` sidecars preserve `group_keys`,
`group_calibrators` and row-aligned `qhat` vectors, then revalidate them and
strict non-boolean integer grouped `n_samples` summaries against the embedded
artifact before exposing a `CalibratedRunResult` or `PredictResult`.
Version fields on conformal cohort manifests, calibration artifacts and
calibrated results are strict integer contract tags: boolean `true`/`false` and
numeric strings are rejected on direct construction and reload instead of being
coerced to schema version `1`. Optional conformal artifact identity strings
(`target_name`, `predictor_fingerprint`, `calibration_data_fingerprint`) must be
either `None` or non-empty strings without surrounding whitespace or NULs; reload
rejects the same invalid values instead of publishing ambiguous provenance.
`ConformalCalibrationSpec` validates direct construction the same way as
`parse_conformal_calibration_spec()`: coverage values must be real numeric
scalars, not booleans or numeric strings, and method, unit, group keys and
multi-target mode are canonicalized before fingerprinting.
Reloaded cohort manifest `unit` and calibrated prediction `method`/`unit`
fields must already be strings; objects with a helpful `__str__` are rejected
instead of being stringified into valid contract values.
Conformal calibration cohort rows built directly or reloaded from JSON now
require a strict non-boolean integer `row_index`, canonical non-whitespace
`sample_id`, `role` and optional `group`, plus strict JSON-native metadata with
string keys, before the cohort manifest can be fingerprinted. The optional
serialized `n_samples` summary must also be a strict non-boolean integer matching
the row count. Row-aligned calibration metadata supplied as either column
mappings or per-row mappings uses the same strict key rule: non-string or
whitespace-padded metadata keys fail before they can be coerced into manifest
JSON.
Conformal numeric arrays (`y_true`, `y_pred`, interval bounds and `qhat`) must
contain real numeric values; boolean payloads fail closed instead of being
coerced to `0.0`/`1.0`, and numeric strings such as `"1.0"` are rejected instead
of being parsed as floats. The same bool rejection applies to Python-side
`from_dict(...)` payloads carrying NumPy boolean scalars in serialized scores or
quantiles, and serialized numeric fields reject NumPy ndarray scalars instead of
coercing them to JSON numbers. Direct `ConformalIntervalBlock` /
`CalibratedPredictionBlock` construction is also fail-closed for mismatched
coverage keys, mismatched interval shapes, inverted bounds, invalid method/unit
values, invalid group-key lengths, and negative or non-row-aligned `qhat` values.
Direct `SplitConformalCalibrator` construction validates retained residual
scores, coverage keys, recomputed quantiles, method and unit before `apply()`;
negative scores, edited `qhat` values or unsupported vocabulary fail closed
instead of materializing invalid intervals.
Empirical `ConformalMetricSet` diagnostics are checked the same way at
construction: `observed_coverage` must be finite in `[0, 1]` and match
`n_covered / n_samples`, `coverage_gap` must equal `observed_coverage -
coverage`, and widths/interval scores must be non-negative or positive infinity
for unbounded intervals.

For model `.n4a` bundles carrying a conformal sidecar, `predict(...,
coverage=...)` replays the model bundle first, then applies the attached
calibrator. This currently applies intervals only to the single selected
prediction entry (`all_predictions=False`). `all_predictions=True` remains
fail-closed until each returned prediction entry can carry its own calibrated
identity mapping. If the bundle contains a `conformal/` sidecar but that
sidecar is incomplete, duplicated, or contains unexpected members, `predict()`
fails sidecar validation instead of falling back to an uncalibrated prediction
path. The same fail-closed path applies when the sidecar looks structurally
complete but its `calibrated_result.json` has a non-empty prediction cohort
without canonical physical `sample_ids`: `predict()` rejects the sidecar before
running the raw model prediction.

## Audit-only robustness report

When observed targets are available for an external prediction cohort,
`nirs4all.robustness()` can produce the current audit-only robustness report:

```python
report = nirs4all.robustness(
    prediction,
    y_true=[13.2, 13.7],
    X=X_external,
    predictor=frozen_predictor,
    mode="clean_frozen",
    scenarios=[
        nirs4all.RobustnessScenarioSpec(kind="observed"),
        nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.2),
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.05),
        nirs4all.RobustnessScenarioSpec(kind="spectral_noise", severity=0.001),
    ],
    metadata={"instrument": ["inst-a", "inst-b"], "batch": ["b1", "b2"]},
    slice_by=["instrument"],
    seed=123,
    workspace_path="workspace/",
    workspace_robustness_id="external-audit",
    workspace_name="External robustness audit",
)

print(report.scenarios[0].metrics.rmse)
print(report.scenarios[0].conformal_metrics[0.8].observed_coverage)
print(report.scenarios[0].slices[0].metrics.mae)
print(report.summary_rows())
print(report.degradation_rows())
print(report.worst_slices(metric="rmse", top_k=3))

# Convenience syntax when you already hold a result object:
same_report_from_prediction = prediction.robustness(y_true=[13.2, 13.7])

report.save_json("robustness-report.json")
report.save_summary("robustness-summary.json")
report.save_markdown("robustness-report.md")
report.save_html("robustness-report.html")
report.save_parquet("robustness-report.parquet")
report.save_artifacts("robustness-artifacts")
same_report = nirs4all.RobustnessReport.load_json("robustness-report.json")
same_report_from_tables = nirs4all.RobustnessReport.load_parquet("robustness-report.parquet")
same_report_from_bundle = nirs4all.RobustnessReport.load_artifacts("robustness-artifacts")

robustness_id = nirs4all.save_workspace_robustness_report(
    "workspace/",
    report,
    robustness_id="external-audit-copy",
    name="External robustness audit",
)
same_report_from_workspace = nirs4all.load_workspace_robustness_report(
    "workspace/",
    "external-audit",
)
```

The same pattern composes directly with calibrated predictions produced by the
native tuning example:

```python
prediction = nirs4all.predict_calibrated(
    restored_conformal,
    y_pred=[13.0, 14.0],
    prediction_sample_ids=["pred-003", "pred-004"],
)

report = prediction.robustness(
    y_true=[13.2, 13.7],
    scenarios=[
        nirs4all.RobustnessScenarioSpec(kind="observed"),
        nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.1),
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.05),
    ],
    metadata={"batch": ["external-a", "external-b"]},
    slice_by=["batch"],
    seed=7,
    workspace_path="workspace/",
    workspace_robustness_id="u09-robustness",
    workspace_name="U09 robustness audit",
)

report.save_artifacts("workspace/u09-robustness-artifacts")
same_report = nirs4all.load_workspace_robustness_report("workspace/", "u09-robustness")
same_bundle = nirs4all.RobustnessReport.load_artifacts("workspace/u09-robustness-artifacts")
```

This is the current public Python E2E lane:
`run(tuning=...) → calibrate/predict_calibrated() → PredictResult.robustness()`
with explicit external `y_true`. It remains audit-only: robustness scenarios do
not refit, recalibrate, or renew the conformal guarantee.

For CI/release publication, an already verified JSON report can also be
republished without re-running the audit:

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

`workspace robustness from-prediction` is the CLI form of
`nirs4all.robustness_from_workspace_prediction(...)`. It keeps the same
fail-closed boundary: stored provenance markers do not become executable
spectra, but actual row-aligned `X`/`spectra` plus `predictor_bundle` evidence
are consumed as defaults for spectral/OOD scenarios.

This does not recalibrate or refit. Post-prediction scenarios compute point
metrics and conformal diagnostics on already materialized predictions:
`prediction_bias` adds `severity` to `y_pred`, and `prediction_noise` draws
seeded normal or centered uniform noise. `normal` uses `severity` as standard
deviation; `uniform` samples from `[-severity, +severity]`. In both cases existing
conformal interval bounds are shifted with the prediction delta. If `seed` is
omitted, nirs4all records and uses `effective_seed=0`, so repeated reports are
still deterministic. The explicit-X spectral cells require `X` and exactly one
frozen replay source: either an in-memory `predictor` or a saved
`predictor_bundle`. The in-memory form accepts `predictor.predict(X)` or a
callable. The saved-bundle form routes the perturbed matrix through public
`nirs4all.predict(model=predictor_bundle, data={"X": X, "sample_ids": ...},
all_predictions=False)`, preserving row identity when the prediction result
carries sample ids. Both forms replay without fit or recalibration and recenter
existing conformal intervals by the prediction delta. `predictor` and
`predictor_bundle` are mutually exclusive. Spectral reports include
`metadata["spectral_replay"]` so auditors can distinguish in-memory predictor
replay from saved-bundle replay, see the saved bundle path, confirm the
`nirs4all.predict` route, and verify whether sample ids were forwarded.
If a `PredictResult` already carries actual row-aligned `metadata["X"]` or
`metadata["spectra"]` plus published `robustness_evidence.predictor_bundle` or
`model_path`, `nirs4all.robustness(result, ...)` and
`result.robustness(...)` use those values as the spectral replay defaults.
This only consumes real arrays stored on the result; strings such as
`"prediction_arrays.X"` are provenance markers and remain fail-closed unless
the array itself is present.
For workspace-loaded predictions, convert the stored record explicitly:

```python
from nirs4all.data.predictions import Predictions

predictions = Predictions.from_workspace("workspace/", load_arrays=True)
result = predictions.get_predict_result_by_id("pred-001")
record = predictions.get_prediction_by_id("pred-001", load_arrays=True)

# Direct public one-record shortcut:
result = nirs4all.load_workspace_predict_result("workspace/", "pred-001")

# Direct public bulk shortcut:
results = nirs4all.load_workspace_predict_results("workspace/")

# Direct workspace prediction -> robustness report shortcut:
report = nirs4all.robustness_from_workspace_prediction(
    "workspace/",
    "pred-001",
    y_true=record["y_true"],
    scenarios=[{"kind": "spectral_offset", "severity": 0.01}],
    save_to_workspace=True,
    workspace_robustness_id="pred-001-spectral-audit",
)

if result.spectral_replay_evidence_status["status"] == "ready_for_spectral_replay":
    report = result.robustness(
        y_true=record["y_true"],
        scenarios=[{"kind": "spectral_noise", "severity": 0.01}],
    )
```

`nirs4all.load_workspace_predict_result()` is the direct public one-record
workspace path. `nirs4all.load_workspace_predict_results()` is the direct
public bulk workspace path, with optional `dataset_name=...` filtering.
`Predictions.get_predict_result_by_id()` is the lower-level buffer equivalent;
`Predictions.to_predict_results()` performs the same conversion for the whole
buffer. All four call
`PredictResult.from_prediction_record()` internally and
preserve `result_metadata`, `robustness_evidence`, executable `X`/`spectra`,
`sample_indices`, model name, preprocessing metadata,
`calibration_replay_source` and `tuning_calibration_source`. They intentionally
require `load_arrays=True`; a metadata-only record is not enough to replay
spectral/OOD scenarios.
`nirs4all.robustness_from_workspace_prediction()` is the direct report path for
one persisted prediction: it performs the same conversion and then calls
`robustness()`, optionally saving the report back into the workspace with a
`prediction_id` link.
Python publishers can make the evidence executable by writing it through
`Predictions.add_prediction(..., X=..., spectra=..., result_metadata=...)`
before `flush()`. `X`/`spectra` are stored in the array sidecar,
`result_metadata.robustness_evidence` records the `predictor_bundle` or
`model_path`, and `Predictions.merge_stores()` preserves those fields during
workspace consolidation. This is the supported Python publisher path for
spectral/OOD replay evidence; provenance-only strings remain non-executable.
`spectral_noise` adds seeded noise to `X`; `normal` uses `severity` as standard
deviation and `uniform` samples from `[-severity, +severity]`.
`spectral_offset` applies a deterministic additive offset; `spectral_scale` multiplies every feature by
`1.0 + severity`; `spectral_slope` adds a centered linear ramp across the
feature axis; and `spectral_shift` shifts spectra by fractional feature units
with interpolation.
These stress cells are diagnostic only and do not renew a conformal coverage
guarantee. Matched recalibration and structural refit remain planned gates.

`scenarios` accepts either the typed `RobustnessScenarioSpec` helper shown
above or the equivalent mapping form, for example
`{"kind": "prediction_noise", "severity": 0.05}`. The supported `kind` values
are currently `observed`, `prediction_bias`, `prediction_noise`,
`spectral_noise`, `spectral_offset`, `spectral_scale`, `spectral_slope`, and
`spectral_shift`. `severity` is an offset for `prediction_bias`, a standard
deviation for `normal` noise, a centered half-width for `uniform` noise, and the
deterministic magnitude for the other spectral cells. `distribution` is optional
and currently accepts `"normal"` or `"uniform"` for the two noise cells.
`extra={...}` on
`RobustnessScenarioSpec` can carry stable labels or audit metadata into the
scenario payload, but it cannot override `kind`, `severity`, or
`distribution`. Direct `RobustnessScenarioSpec` construction is fail-closed:
`kind` and `distribution` must be real strings, not host objects that stringify
to supported values;
`severity` must be a real numeric scalar, not a boolean or numeric string, and
`extra` keys must be canonical non-empty strings without NULs before
`to_dict()` can publish the scenario. Scenario mappings, including
`extra={...}` on the typed helper, use the same strict boundary: keys must be
canonical non-empty strings without NULs,
`kind` and `distribution` must be real strings, `severity` must be a real finite
numeric scalar, and the payload must stay TCV1 JSON-native and fingerprintable.
Plain strings, finite numbers, booleans,
`null`, arrays, and objects with text keys are accepted; Python objects,
NaN/Infinity, bytes, and other opaque values are rejected before report
execution.
The same no-stringify rule is applied again to published result payloads:
`RobustnessScenarioResult.scenario` and `RobustnessSliceResult.slice_key` keys
must be canonical non-empty strings without NULs, their values must stay TCV1
JSON-native, and `RobustnessScenarioResult.severity` must be a real numeric
scalar. These mappings enter report JSON, fingerprints, `summary_rows()`,
`degradation_rows()`, `worst_slices()`, `tabular_records()` and Studio/CI
summary cards, so corrupted reloads fail closed instead of silently converting
Python keys with `str(...)`.

The JSON export is deterministic and verified on reload through the report
fingerprint, which makes it suitable for CI artifacts and release audits.
`save_markdown()` writes the same report as a deterministic human-readable
summary for release notes, audit packets, or CI job summaries. The Markdown
summary includes conformal guarantee status details when present, including
requested/effective engine, method, unit, selected and calibrated coverages,
scope, invalidation reasons and limitations; this is descriptive metadata and
does not renew the guarantee. It also includes degradation versus the reference
scenario and the worst diagnostic slices when slices are available.
`summary_rows(reference=0)` returns one compact row per scenario for CI
dashboards and Studio cards: point metrics, deltas/ratios versus the reference
scenario, conformal coverage/width summary, the worst diagnostic slice, and an
execution-scope hint. `execution_scope` is `baseline`, `prediction_replay` or
`spectral_replay`; `requires_spectral_replay=true` marks explicit-X
spectral/OOD replay rows and is metadata for orchestration, not permission for
bindings or Studio to replay perturbations locally.
Markdown and HTML exports render this same compact scenario summary, including
scope and replay-evidence columns, before the detailed tables. `save_html()`
writes a standalone escaped HTML artifact with the same report tables. `save_parquet()`
writes a portable directory containing `manifest.json` and one Parquet file per
non-empty flat report table, such as compact scenario summary, scenario metrics,
degradation rows, conformal diagnostics and slices. It also stores `report.json`;
`manifest.json` records table filenames, row counts, and fingerprints, so
`RobustnessReport.load_parquet()` can reload the full report and verify that the
exported table files still match the manifest. `save_artifacts()` writes a
publication directory with `manifest.json` plus selected `report.json`,
`summary.json`, `report.md`, `report.html`, and `report.parquet/` artifacts.
Use it when CI, release automation, bindings, or Studio need one stable
directory to attach or inspect. `summary.json` is the lightweight UI/CI payload:
fingerprint, mode, report version, slice keys, optional
`conformal_guarantee_status`, optional `spectral_replay`, and compact
`summary_rows()` with execution-scope hints. The guarantee and spectral replay blocks are copied from report
metadata when present so cards and dashboards can show engine, coverage,
invalidation details, replay source, bundle path and sample-id forwarding status
without parsing the full report.
Because report metadata enters `RobustnessReport.fingerprint`, its keys must be
canonical non-empty strings without NULs and its values must stay TCV1
JSON-native; arbitrary Python objects or non-string keys fail before JSON
publication.
The same payload is available directly with `summary_artifact()`,
`to_summary_json()` and `save_summary()`, and from CLI publication with
`--format summary`. Its JSON Schema is available from
`get_robustness_summary_schema()` and `robustness_summary_schema_json()`. The
full bundle can be emitted from existing report/workspace CLI entry points with
`--format artifacts --output <directory>`.
`RobustnessReport.load_artifacts()` reloads a bundle from its manifest and
verifies the embedded JSON or Parquet report fingerprint. When Markdown, HTML or
Parquet artifacts are present, it compares them against the deterministic report
rendering or table-level Parquet manifest. When `summary.json` is present, it is
also compared against the deterministic summary payload. Missing or edited files
fail closed before downstream CI, bindings or Studio consume the directory. The
`nirs4all robustness-report` CLI accepts either a report JSON file or one of
these verified artifact directories as input.

## Workspace CLI

The workspace commands are read-only and useful in CI or release audits:

```bash
nirs4all workspace tuning list --workspace workspace/ --json
nirs4all workspace tuning show ridge-native-tuning --workspace workspace/ --json
nirs4all workspace conformal list --workspace workspace/ --json
nirs4all workspace conformal show ridge-conformal --workspace workspace/ --json
nirs4all workspace conformal predict ridge-conformal \
  --workspace workspace/ \
  --y-pred "13.0,14.0" \
  --sample-ids "pred-003,pred-004" \
  --json
```

`show` reloads and verifies the typed artifact before printing it. `conformal
predict` emits calibrated intervals and guarantee metadata without modifying the
workspace.

## Executable example

See `examples/user/04_models/U09_native_tuning_conformal.py` for a complete
smoke-tested script. It covers typed conformal-aware objective scoring with
`NativeTuning`, `TuningScoreData`, and `TuningConformalScoreCalibration`,
then keeps final conformal calibration on the projected winner, reloads the
workspace artifacts, applies the stored calibrator to new point predictions,
and includes a standalone `ConformalCalibrationData` calibration smoke path.

## Reading the guarantee metadata

Both calibrated result containers and prediction results expose
`conformal_guarantee_status`:

```python
status = prediction.conformal_guarantee_status
print(status["status"])
print(status["coverage"])
print(status["limitations"])
print(prediction.calibration_replay_source["kind"])
```

The metadata records the method, unit, requested/effective coverage, fingerprints
and known limitations. It also includes `calibration_replay_source`, a compact
provenance block that tells whether calibration predictions were provided as
arrays, a `PredictResult`, dataset-backed `y_pred`, an in-memory `predictor`, a
saved `predictor_bundle`, a `predictor_result`, or a workspace
`predictor_chain_id`; the block records the route (`provided_arrays`,
`predictor.predict`, or `nirs4all.predict`) and bundle/chain details when
applicable. Guarantee metadata string fields are not coerced: `effective_engine`,
`requested_engine`, `source_calibrated_result_fingerprint`, and each
`invalidation_reasons` entry must be non-empty strings without surrounding
whitespace, and persisted `conformal_guarantee_status.version` must be the strict
integer `1`. Status `predictor_fingerprint`, `calibration_data_fingerprint`,
`guarantee`, and `scope` are also rechecked against the embedded artifact on
construction and reload. A persisted status must include the complete generated
field set, and `status` must be `active` exactly when `invalidation_reasons` is
empty, otherwise `invalidated`. The generated `limitations` list must also match
the embedded artifact's guarantee mode exactly; edited, shortened, empty or
non-string limitation payloads fail closed before publication. Use
`prediction.calibration_replay_source` or
`calibrated.calibration_replay_source` for the direct accessor; it reads the
nested status block first and falls back to the top-level result metadata for
older or partially projected payloads. New `CalibratedRunResult` payloads fail
closed if both locations carry `calibration_replay_source` but disagree, so
notebooks, Studio and bindings cannot observe contradictory replay provenance.
Empirical diagnostics such as observed coverage and interval width can be
computed separately with `nirs4all.conformal_metrics()` when observed `y_true`
values are available for the prediction cohort. Invalid or inconsistent metric
payloads fail closed before fingerprinting, persistence or UI publication.
