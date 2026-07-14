# Public Interfaces and Runtime

NIRS4ALL has one portable workflow contract and several user surfaces around it.

The portable contract is:

```text
dataset.yaml or dataset.json
pipeline.yaml or pipeline.json
```

The public Python API runs that contract directly. The CLI currently validates and manages datasets, configs, workspaces, and artifacts; it does not expose a native training `run` subcommand in this repository.

## Python API

| Function/class | Purpose | Typical input | Typical output |
| --- | --- | --- | --- |
| `nirs4all.run(...)` | Train/evaluate one or many pipelines on one or many datasets | Pipeline spec + dataset spec | `RunResult` |
| `nirs4all.predict(...)` | Predict from a stored chain or exported bundle | `chain_id` or model bundle + data | `PredictResult` |
| `nirs4all.calibrate(...)` | Fit split-conformal intervals from explicit calibration evidence | Replayed calibration predictions or selected calibration cohort + prediction ids | `CalibratedRunResult` or `PredictResult` |
| `nirs4all.CONFORMAL_CALIBRATION_METHODS` / `CONFORMAL_CALIBRATION_UNITS` | Discover conformal method and exchangeability-unit vocabularies | None | Tuples aligned with runtime validation and registry schema |
| `nirs4all.CONFORMAL_MULTI_TARGET_POLICIES` / `CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES` | Discover reserved and currently executable multi-target policies | None | Tuples aligned with runtime validation and registry schema |
| `nirs4all.TUNING_ENGINES` / `TUNING_DIRECTIONS` | Discover optimizer engines and objective directions accepted by the strict native tuning contract | None | Tuples aligned with `DagMLTuningSpec` validation and registry schema |
| `nirs4all.TUNING_CONTRACT_KEYS` / `TUNING_OPTIMIZER_PERSISTENCE_KEYS` / `TUNING_RUNTIME_KEYS` | Separate deterministic tuning-contract keys, optimizer persistence keys, and `run(tuning=...)` runtime wrapper keys | None | Tuples for validators, bindings, CLIs and Studio forms |
| `nirs4all.NativeTuning(..., force_params=...)` | Warm-start native HPO with an explicit first trial | Public decoded parameter values keyed by `tuning.space` paths | Deterministic contract entry that can change trial order and selected predictor |
| `nirs4all.inspect_tuning_space(...)` | Inspect canonical ordered tuning-space patches for forms, bindings and Studio | `NativeTuning`, `DagMLTuningSpec`, or mapping tuning payload | JSON-native `nirs4all.tuning.ordered_search_space` artifact with `ParameterPatch` force params and fingerprints |
| `nirs4all.get_tuning_space_schema()` / `tuning_space_schema_json(...)` | Publish the ordered tuning-space JSON Schema | None or optional JSON indentation | Stable schema for bindings, CLI validators and Studio tuning forms |
| `TuningResult.summary_artifact()` / `save_summary(...)` | Publish a lightweight HPO summary for CI, release indexes, bindings and Studio cards | `TuningResult` | `nirs4all.tuning.summary` payload or `summary.json` path, including safe persistence flags without raw storage URIs |
| `nirs4all.CONFORMAL_TUNING_SCORE_METRICS` | Discover conformal-aware objective metrics that require `score_data.conformal_calibration` | None | Tuple aligned with `NativeTuning` payload validation |
| `nirs4all.FINETUNE_ENGINES` / `FINETUNE_ENGINE_ALIASES` | Discover registry-level model-local `finetune_params.engine` vocabulary and read-only aliases | None | Tuples aligned with keyword registry aliases |
| `nirs4all.FINETUNE_OPTUNA_SAMPLERS` / `FINETUNE_OPTUNA_PRUNERS` | Discover Optuna model-local HPO controls | None | Tuples aligned with `OptunaManager` validation |
| `nirs4all.FINETUNE_N4M_SAMPLERS` / `FINETUNE_N4M_PRUNERS` | Discover native n4m model-local HPO controls | None | Tuples aligned with `N4MFinetuneManager` validation |
| `nirs4all.FINETUNE_DAGML_DETERMINISTIC_ENGINES` / `FINETUNE_DAGML_SELECTION_METRICS` | Discover deterministic DAG-ML `finetune_params` lowering subset | None | Tuples aligned with native DAG-ML lowering validation |
| `nirs4all.predict_calibrated(...)` | Apply a stored conformal calibrator to new point predictions | Calibrated result/store/bundle + `y_pred` + physical ids | `PredictResult` or `CalibratedRunResult` |
| `nirs4all.load_workspace_calibrated_predict_result(...)` | Load a workspace conformal result through the public prediction surface | Workspace path + conformal id/fingerprint | `PredictResult` with intervals and provenance accessors |
| `nirs4all.save_workspace_predict_result(...)` | Publish a public prediction result through the workspace prediction store | Workspace path + `PredictResult` + optional `X`/`spectra` evidence | Stored prediction id, with arrays and result metadata reloadable by the workspace prediction helpers |
| `nirs4all.load_workspace_predict_result(...)` | Load a workspace prediction row through the public prediction surface | Workspace path + prediction id | `PredictResult` with arrays, sample ids, model metadata and replay provenance accessors |
| `nirs4all.load_workspace_predict_results(...)` | Load workspace prediction rows through the public prediction surface | Workspace path + optional dataset name | `list[PredictResult]` preserving arrays, sample ids, model metadata and replay provenance accessors |
| `nirs4all.conformal_metrics(...)` | Compute empirical conformal diagnostics | Calibrated result or conformal `PredictResult` + observed targets | `dict[coverage, ConformalMetricSet]` |
| `nirs4all.robustness(...)` | Produce an audit-only robustness/generalization report | Replayed predictions + observed targets + scenarios/slices | `RobustnessReport` |
| `nirs4all.robustness_from_workspace_prediction(...)` | Compute robustness directly from one stored prediction row | Workspace path + prediction id + observed targets | `RobustnessReport`, optionally persisted back to the same workspace |
| `nirs4all.ROBUSTNESS_MODES` / `ROBUSTNESS_EXECUTABLE_MODES` | Discover reserved robustness modes and the subset executable today | None | Tuples aligned with runtime validation and registry schema |
| `nirs4all.ROBUSTNESS_SCENARIO_KINDS` | Discover the ordered public robustness scenario `kind` vocabulary | None | Tuple aligned with report metadata and registry enum |
| `nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS` / `ROBUSTNESS_SCENARIO_DISTRIBUTIONS` | Discover which scenarios accept `distribution` and which distribution tokens are supported | None | Tuples aligned with runtime validation and registry schema |
| `PredictResult.robustness(...)` / `CalibratedRunResult.robustness(...)` | Convenience method equivalent to `nirs4all.robustness(result, ...)` | Result object + observed targets + optional scenarios/slices/workspace | `RobustnessReport` |
| `RobustnessReport.save_artifacts(...)` / `load_artifacts(...)` | Publish/reload a verified robustness artifact directory | `RobustnessReport` or artifact directory | Manifested JSON/summary/Markdown/HTML/Parquet bundle or verified `RobustnessReport` |
| `nirs4all.save_workspace_robustness_report(...)` / `load_workspace_robustness_report(...)` | Persist/reload verified robustness reports in a workspace | Workspace path + `RobustnessReport` or id/fingerprint | Robustness id or `RobustnessReport` |
| `nirs4all.tune_single_estimator(...)` | Run the native single-estimator/linear tuning lane | Estimator or linear pipeline + explicit score data | `RunResult` or `TunedSingleEstimatorConformalResult` |
| `TunedSingleEstimatorConformalResult.interval(...)` / `metrics(...)` / `robustness(...)` | Inspect the calibrated result returned by native tuning+calibration without reaching into nested fields | `TunedSingleEstimatorConformalResult` + coverage or observed targets | Materialized interval, conformal metrics, or `RobustnessReport` |
| `nirs4all.get_keyword_registry()` | Return the machine-readable keyword/effect registry | None | JSON-compatible registry dict |
| `nirs4all.keyword_registry_json(...)` | Serialize the keyword/effect registry deterministically | Optional JSON indentation | Registry JSON text |
| `nirs4all.get_keyword_registry_schema()` | Return the registry JSON Schema for static consumers | None | JSON Schema dict |
| `nirs4all.keyword_registry_schema_json(...)` | Serialize the registry JSON Schema deterministically | Optional JSON indentation | Schema JSON text |
| `nirs4all.get_robustness_summary_schema()` | Return the JSON Schema for `RobustnessReport.summary_artifact()` | None | JSON Schema dict |
| `nirs4all.robustness_summary_schema_json(...)` | Serialize the robustness summary JSON Schema deterministically | Optional JSON indentation | Schema JSON text |
| `nirs4all.get_tuning_summary_schema()` | Return the JSON Schema for `TuningResult.summary_artifact()` | None | JSON Schema dict |
| `nirs4all.tuning_summary_schema_json(...)` | Serialize the tuning summary JSON Schema deterministically | Optional JSON indentation | Schema JSON text |
| `nirs4all.get_tuning_space_schema()` | Return the JSON Schema for `inspect_tuning_space(...)` artifacts | None | JSON Schema dict |
| `nirs4all.tuning_space_schema_json(...)` | Serialize the ordered tuning-space JSON Schema deterministically | Optional JSON indentation | Schema JSON text |
| `nirs4all.explain(...)` | Generate SHAP explanations | Model/bundle + data | `ExplainResult` |
| `nirs4all.retrain(...)` | Retrain from an existing result or bundle | Source + new data | `RunResult` |
| `nirs4all.session(...)` | Share runner/workspace resources across calls | Optional pipeline and runner kwargs | `Session` |
| `nirs4all.load_session(...)` | Load an exported `.n4a` bundle for prediction | Bundle path | `Session` |
| `nirs4all.generate(...)` | Generate synthetic NIRS data | Synthetic parameters | `SpectroDataset` or arrays |
| `result.export(...)` | Export a trained pipeline bundle | Output path | `.n4a` path |

## Train from Config Files

::::{tab-set}

:::{tab-item} Python
```python
import nirs4all

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="run_from_files",
    random_state=42,
)

print(result.best_score)
result.export("exports/model.n4a")
```
:::

:::{tab-item} R
```r
library(reticulate)
nirs4all <- import("nirs4all")

result <- nirs4all$run(
  pipeline = "pipeline.yaml",
  dataset = "dataset.yaml",
  name = "run_from_files",
  random_state = 42L
)

result$export("exports/model.n4a")
```
:::

:::{tab-item} Julia
```julia
using PythonCall
nirs4all = pyimport("nirs4all")

result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    name="run_from_files",
    random_state=42,
)

result.export("exports/model.n4a")
```
:::

:::{tab-item} JavaScript
```javascript
import { spawnSync } from "node:child_process";

const code = `
import nirs4all
r = nirs4all.run("pipeline.yaml", "dataset.yaml", name="run_from_files", random_state=42)
print(r.best_score)
r.export("exports/model.n4a")
`;

const run = spawnSync("python", ["-c", code], { stdio: "inherit" });
if (run.status !== 0) process.exit(run.status);
```
:::

::::

## Predict from an Exported Bundle

::::{tab-set}

:::{tab-item} Python
```python
import nirs4all

pred = nirs4all.predict(
    model="exports/model.n4a",
    data="new_dataset.yaml",
)

df = pred.to_dataframe()
print(df.head())
```
:::

:::{tab-item} R
```r
library(reticulate)
nirs4all <- import("nirs4all")

pred <- nirs4all$predict(
  model = "exports/model.n4a",
  data = "new_dataset.yaml"
)

print(pred$to_dataframe())
```
:::

:::{tab-item} Julia
```julia
using PythonCall
nirs4all = pyimport("nirs4all")

pred = nirs4all.predict(
    model="exports/model.n4a",
    data="new_dataset.yaml",
)

println(pred.to_dataframe())
```
:::

:::{tab-item} JavaScript
```javascript
import { spawnSync } from "node:child_process";

const code = `
import nirs4all
pred = nirs4all.predict(model="exports/model.n4a", data="new_dataset.yaml")
print(pred.to_dataframe())
`;

const run = spawnSync("python", ["-c", code], { stdio: "inherit" });
if (run.status !== 0) process.exit(run.status);
```
:::

::::

## Runtime Engine

`nirs4all.run()` accepts an `engine` selector:

| Engine | Meaning |
| --- | --- |
| `None` | Use the package default resolved by `resolve_engine(...)`. |
| `"legacy"` | Use the in-process Python orchestrator. |
| `"dag-ml"` | Request the dag-ml backend for covered shapes; catchable unsupported/unavailable cases warn and fall back to legacy. |
| `"dual"` | Reserved; not implemented as a public side-by-side mode. |

The pipeline language is broader than current dag-ml native coverage. Requesting `engine="dag-ml"` is safe for user workflows because catchable unsupported shapes and unavailable dag-ml runtime dependencies warn and re-run on the legacy engine. A genuine dag-ml runtime/operator bug still propagates as an error.

```python
result = nirs4all.run(
    pipeline="pipeline.yaml",
    dataset="dataset.yaml",
    engine="dag-ml",
    random_state=42,
)
```

Environment switches:

| Variable | Effect |
| --- | --- |
| `N4A_ENGINE=dag-ml` | Request dag-ml for calls that do not pass `engine=` explicitly. |
| `N4A_NATIVE_RESULTS=/path/to/results` | For dag-ml runs, request native result output. |

## CLI Commands

Use `nirs4all <group> <command> --help` for exact options.

| Group | Command | Purpose |
| --- | --- | --- |
| root | `--test-install` | Print dependency/install diagnostics. |
| root | `--test-integration` | Run the packaged integration smoke test. |
| root | `--version` | Print installed package version. |
| root | `keyword-registry [--schema] [--output file]` | Export the keyword/effect registry or its JSON Schema for docs, CI, Studio/Web, forms, and bindings. |
| root | `robustness-summary-schema [--output file]` | Export the JSON Schema for `RobustnessReport.summary_artifact()` and bundle `summary.json` artifacts. |
| root | `tuning-summary-schema [--output file]` | Export the JSON Schema for `TuningResult.summary_artifact()` and `tuning-summary.json` artifacts. |
| root | `robustness-report <report.json-or-artifacts> [--format json\|summary\|markdown\|html\|parquet\|artifacts] [--output path]` | Republish a verified `RobustnessReport` JSON artifact or artifact directory as CI/release summary JSON, Markdown, HTML, JSON, Parquet-directory, or complete artifact-directory output. |
| workspace | `workspace robustness from-prediction --prediction-id ID --y-true CSV_OR --y-true-json JSON [--scenarios-json JSON] [--save-to-workspace]` | Compute a `RobustnessReport` from one persisted prediction row through `nirs4all.robustness_from_workspace_prediction(...)`; output formats match `workspace robustness export`. |
| `config` | `validate <file>` | Validate a pipeline or dataset config. |
| `config` | `schema pipeline|dataset` | Print the JSON schema. |
| `dataset` | `validate <file>` | Validate a dataset config/folder. |
| `dataset` | `inspect <file>` | Show sources, task type, aggregation, and optional detected file parameters. |
| `dataset` | `export <file>` | Normalize a dataset config to YAML or JSON. |
| `dataset` | `diff <a> <b>` | Compare two dataset configs. |
| `workspace` | `init <path>` | Create workspace directories and store. |
| `workspace` | `list-runs` | List recorded runs. |
| `workspace` | `tuning export <id> [--format json\|summary] [--output path]` | Republish a verified persisted `TuningResult` as full JSON or lightweight HPO summary JSON. |
| `workspace` | `query-best` | Query top predictions by metric. |
| `workspace` | `filter` | Filter prediction rows by dataset and scores. |
| `workspace` | `stats` | Show workspace score statistics. |
| `workspace` | `list-library` | List saved templates/library items. |
| `workspace` | `robustness list/show/export` | Inspect or export persisted robustness reports by id or fingerprint. |
| `artifacts` | `list-orphaned` | List artifacts not referenced by manifests. |
| `artifacts` | `cleanup` | Dry-run or delete orphaned artifacts. |
| `artifacts` | `stats` | Show artifact storage statistics. |
| `artifacts` | `purge` | Delete all artifacts for a dataset when forced. |

Example:

```bash
nirs4all dataset validate dataset.yaml
nirs4all dataset inspect dataset.yaml --detect
nirs4all config validate pipeline.yaml --type pipeline --check-imports
nirs4all workspace init workspace
nirs4all workspace query-best --workspace workspace --metric test_score -n 5
```

## Result Objects

`RunResult` exposes:

| Accessor | Purpose |
| --- | --- |
| `best`, `final`, `cv_best` | Best/final prediction entries. |
| `best_score`, `best_rmse`, `best_r2`, `best_accuracy` | Convenience score fields. |
| `top(n=...)` | Top prediction rows. |
| `filter(...)` | Filter prediction rows. |
| `models` | Per-model refit result handles when available. |
| `export(path, format="n4a")` | Export the selected model bundle. |

See also {doc}`api/session`, {doc}`predictions_api`, and {doc}`/api/module_api`.

## Conformal and Robustness Artifact Contracts

Conformal calibration artifacts and robustness reports are separate contracts.
The conformal artifact stores the fitted split-conformal calibrator, calibrated
coverage, physical sample identity, fingerprints, and guarantee-status payload.
Calibrated result metadata also exposes `calibration_replay_source`, copied into
`conformal_guarantee_status`, so downstream tools can display whether the
calibration predictions were provided arrays, a `PredictResult`, dataset-backed
`y_pred`, an in-memory `predictor`, a saved `predictor_bundle`, a
`predictor_result`, or a workspace `predictor_chain_id`. The block is
provenance only; its `route` is one of `provided_arrays`, `predictor.predict`,
or `nirs4all.predict`, and it does not authorize refit, recalibration, or replay
outside the original Python execution context. Reloaded results fail closed if
both locations carry `calibration_replay_source` but disagree, so bindings,
Studio, CLIs and notebooks cannot observe contradictory replay provenance.
Workspace conformal rows enforce the same physical prediction identity contract
as filesystem stores and `.n4a` sidecars. A corrupted `conformal_results`
workspace row with the same missing prediction identity is rejected. A stored
`calibrated_result.json` whose non-empty prediction cohort lacks
canonical physical `sample_ids` fails reload instead of becoming a partial public
`PredictResult`.
The robustness report stores audit diagnostics computed after predictions are
available. A robustness report can include conformal diagnostics, but it does not
modify or refresh the conformal calibrator.

The recommended publication layout for a robustness audit is:

```python
report = nirs4all.robustness(
    pred,
    y_true=y_observed,
    scenarios=[
        {"kind": "observed", "severity": 0.0},
        {"kind": "prediction_noise", "severity": 0.02, "distribution": "normal"},
    ],
    slice_by=["Instrument"],
)

paths = report.save_artifacts("artifacts/robustness/run-001")
```

`save_artifacts()` writes a verified bundle. By default the bundle includes:

| File | Consumer | Purpose |
| --- | --- | --- |
| `report.json` | Python/API/reload | Full verified `RobustnessReport`. |
| `summary.json` | CI, bindings, Studio/Web, dashboards | Stable compact summary rows plus optional fail-loud guarantee and replay-provenance blocks. |
| Markdown/HTML | Human review | Release notes and audit reports. |
| Parquet directory | Data analysis | Tabular report rows when Parquet support is installed. |

`summary.json` is intentionally smaller than the full report. Its schema is
available from both Python and CLI:

```python
summary = report.summary_artifact()
schema = nirs4all.get_robustness_summary_schema()
schema_json = nirs4all.robustness_summary_schema_json()
```

```bash
nirs4all robustness-summary-schema --output robustness-summary.schema.json
```

The summary payload has:

- `format="nirs4all.robustness.summary"`;
- `schema_version=1`;
- `fingerprint`, matching the full report fingerprint;
- `mode`, one of `clean_frozen`, `matched_recalibration`,
  `structural_refit`;
- `report_version`;
- `slice_by`;
- optional `conformal_guarantee_status`, carrying fail-loud guarantee metadata
  for calibrated reports;
- optional `spectral_replay`, carrying metadata-only provenance for spectral
  robustness scenarios: `source`, `route`, `sample_ids_forwarded`, and when
  applicable the saved `predictor_bundle` path and `all_predictions=False`
  replay mode;
- `summary`, an array of scenario rows.

Each row reports the scenario label, severity, sample count, point metrics,
deltas from the reference scenario, worst slice metadata, nullable conformal
diagnostics when intervals are present, and an execution-scope hint. The
`execution_scope` value is `baseline`, `prediction_replay`, or
`spectral_replay`; `requires_spectral_replay=true` marks rows that came from
explicit-X spectral/OOD replay through a frozen predictor. Static consumers
should validate `format`, `schema_version`, and `fingerprint` before rendering a
card or making a release assertion.

`spectral_replay` is publication provenance, not an execution contract for
downstream products. Bindings, Studio/Web dashboards, and CI cards may display
that a Python robustness report replayed spectra through a frozen predictor or
`predictor_bundle`, but they must not infer permission or capability to replay
spectral/OOD perturbations locally from this metadata alone.

Prediction objects expose the same boundary through
`PredictResult.robustness_evidence` and
`PredictResult.spectral_replay_evidence_status`. The first accessor returns the
published evidence metadata when present, including the Studio/store shape under
`result_metadata.robustness_evidence`; the second accessor is a fail-closed
diagnostic that reports `ready_for_spectral_replay` only when an executable
row-aligned `X`/spectra matrix and a saved `predictor_bundle`/`model_path` are
present. It keeps publication markers separate from executable arrays through
`has_X_or_spectra` and `has_executable_X_or_spectra`; the executable flag also
requires the matrix row count to match the prediction row count.
Consumers should use that diagnostic before enabling spectral/OOD robustness
actions, while still delegating the actual report to `nirs4all.robustness()`.
When a `PredictResult` carries the actual `X`/spectra matrix in metadata,
`nirs4all.robustness()` can consume that matrix and the published bundle path as
defaults; provenance-only strings remain non-executable metadata.

Workspace-loaded prediction records use the same boundary. Load them with
arrays and convert them explicitly before invoking native conformal or
robustness helpers:

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
```

For the common stored-prediction audit path, call
`nirs4all.robustness_from_workspace_prediction("workspace/", "pred-001", ...)`.
It performs the same array-preserving load/convert step, delegates to
`nirs4all.robustness()`, uses stored executable `X`/`spectra` and
`predictor_bundle` evidence as spectral/OOD defaults, and with
`save_to_workspace=True` persists the report linked to the same `prediction_id`.

`nirs4all.save_workspace_predict_result(...)`,
`nirs4all.load_workspace_predict_result(...)`,
`nirs4all.load_workspace_predict_results(...)`,
`Predictions.get_predict_result_by_id()` and
`Predictions.to_predict_results()` use `PredictResult.from_prediction_record()`
internally. The conversion preserves `result_metadata`, `robustness_evidence`,
actual `X`/`spectra` arrays, sample indices, model name, preprocessing
metadata, `calibration_replay_source` and `tuning_calibration_source`. If
records were loaded without arrays, conversion fails explicitly instead of
creating a partial result with unverifiable replay status.

Workspace persistence keeps the same separation:

```python
robustness_id = nirs4all.save_workspace_robustness_report(
    "workspace/",
    report,
    robustness_id="run-001-robustness",
    run_id="run-001",
    chain_id="chain-001",
    conformal_id="run-001-conformal",
)
restored = nirs4all.load_workspace_robustness_report("workspace/", robustness_id)
```

Python-side publishers should write executable spectral evidence through the
store-backed `Predictions.add_prediction(..., X=..., spectra=...,
result_metadata=...)` path. `X` and `spectra` are stored in the Parquet sidecar,
`result_metadata.robustness_evidence` records the replay bundle or model path,
and `Predictions.merge_stores(...)` preserves those sidecar fields when
consolidating workspaces.

Studio and other products should consume `summary.json` or
`RobustnessReport.summary_artifact()` for cards and tables, then link to the
full report for audit. They must not recompute statistical meaning from partial
metrics.
