# Native conformal finetuning release audit

This page is the public release audit for the current native tuning,
conformal-learning and robustness/generalization surface in `nirs4all`.
It describes what is implemented, which artifacts prove it, and which boundaries
remain intentionally fail-closed.

## Release status

The current release surface is production-shaped but deliberately narrow:

- `nirs4all.run(tuning=..., calibration=...)` supports the public
  single-estimator or linear sklearn-like chain subset, including the documented
  `{"steps": [...]}` wrapper, the legacy public `{"pipeline": [...]}` alias, and
  named final model steps for those linear shapes; sklearn `Pipeline` objects
  and sklearn-like `(name, step)` tuples can use the same dotted tuning paths,
  and explicit `sklearn.*` class-path mappings or direct `sklearn.*` string
  steps are accepted inside the same linear subset, including direct model
  strings, `transform`/`model` mapping values, and explicit constructor
  `params` for `sklearn.*` transform/model import paths. This includes explicit
  and searchable non-final passthrough steps,
  including the structured marker
  `{"kind": "passthrough"}` and its typed helper `TuningPassthrough()`. The
  structured marker is exact: `kind` must be the literal string
  `"passthrough"`, not a host object that stringifies to that value. The
  tuning space stays TCV1-compatible JSON-native. The strict HPO contract also
  accepts `force_params` as an explicit warm-start trial: keys must be a subset
  of `tuning.space`, values use the public decoded syntax, and the contract
  fingerprint changes because the optimizer trajectory can change.
- `nirs4all.calibrate()` and `nirs4all.predict_calibrated()` support explicit
  conformal evidence and stored conformal artifacts.
- `nirs4all.robustness()` and `PredictResult.robustness()` produce audit-only
  robustness reports for frozen prediction cohorts.
- `RobustnessReport.save_artifacts()` publishes deterministic JSON,
  `summary.json`, Markdown, HTML and Parquet-directory outputs.
- `nirs4all.get_keyword_registry()` and
  `nirs4all.get_robustness_summary_schema()` expose static contracts for docs,
  CI, Studio, forms and bindings.
- `TuningResult.summary_artifact()` / `save_summary()` and
  `nirs4all.get_tuning_summary_schema()` expose the equivalent lightweight HPO
  summary contract for optimizer cards and release indexes.
- Direct construction of `SearchSpaceParameter`, `ParameterPatch`,
  `OrderedSearchSpaceSpec`, `DagMLTuningSpec`, `TrialResult` and `TuningResult`
  is fail-closed: paths are canonicalized, values must stay TCV1 JSON-native,
  scores must be finite, integer contract fields reject booleans, trial ids are
  unique, and candidate/best params must stay inside `tuning.space` before a
  fingerprint or summary artifact can be published. Reload through
  `TuningResult.from_dict()` or `load_json()` also rejects boolean or
  numeric-string `best_value` payloads instead of coercing them with `float()`.
- Direct construction of the internal optimizer adapter seams is also
  fail-closed: `ObjectiveTuningRunResult` requires a real `TuningResult` and
  canonical optional `tuning_id`, while the categorical codec requires
  non-empty unique choices, optimizer-native choices without a decoder, matching
  decoder keys and unique TCV1 JSON-native decoded values.
- Public tuning helper payloads are fail-closed before runtime execution:
  dataset-backed `TuningScoreData` and `TuningWinner` selectors reject
  non-string or whitespace-padded keys plus non-JSON-native values,
  `include_augmented` must be a boolean, `TuningWinner.score` must be finite
  numeric evidence, `TuningCalibration`
  requires boolean `as_predict_result`, and tuning metadata/extra mappings
  require canonical non-empty string keys with strict JSON-native finite values.
  Non-finite numbers, bytes, tuples, sets and arbitrary Python objects are
  rejected before publication. `NativeTuning.space` and
  `force_params` canonicalize string patch keys before serialization while
  rejecting non-string keys. Raw `NativeTuning(score_data={...})`,
  `winner={...}` and `calibration={...}` mappings enforce the same boundary,
  including strict raw dataset selectors, boolean raw augmented/calibration
  flags, finite raw winner scores and rejection of nested `calibration_data`;
  raw calibration coverage, method, unit, workspace metadata and
  `workspace_conformal_id` are also validated/canonicalized before publication.
  Workspace persistence applies the same strict JSON-native metadata boundary
  for tuning and conformal rows, so `save_workspace_tuning_result(...)`,
  `run(tuning=..., workspace_path=...)` and
  `save_workspace_calibrated_result(...)` fail closed instead of stringifying
  Python objects into workspace metadata.
  Top-level `NativeTuning` core fields are parsed through `DagMLTuningSpec`
  before publication, so core strings are canonicalized, integer/bool fields
  reject coercive values and optimizer persistence ids remain canonical.
  Dataset-backed tuning column selectors are strict as well:
  `sample_id_column` and `group_column` must be canonical non-empty strings, and
  `metadata_columns` must be either one canonical string or a duplicate-free
  sequence of canonical strings. The same rule applies to raw
  `NativeTuning.score_data` and `NativeTuning.winner` mappings before
  publication.
  Tuple/list `NativeTuning.score_data` is also validated before publication:
  it must contain `(X_score, y_score)` and at most
  `(X_score, y_score, sample_ids, groups, metadata)`, metadata keys must be
  canonical in mapping or row-style sequence form, and valid tuple inputs publish
  as JSON-native lists. Scalar strings, bytes, numbers and arbitrary objects are
  rejected because they are neither mapping nor tuple/list score cohorts.
  Tuning score/winner metadata follows the same boundary:
  `TuningScoreData.metadata`, `TuningWinner.metadata`, raw
  `NativeTuning.score_data.metadata` and raw `NativeTuning.winner.metadata`
  reject non-string or whitespace-padded keys for column-style and row-style
  metadata mappings, and reject non-JSON-native metadata values before runtime.
  Tuning score and winner text fields also reject coercion:
  `TuningScoreData.metric`/`score_metric`, raw
  `NativeTuning.score_data.metric`/`score_metric`, `TuningWinner.metric`,
  `dataset_name`, `model_name`, `task_type` and the corresponding raw
  `NativeTuning.winner` aliases must be non-empty strings without NULs.
  Metrics and winner task types publish lowercase canonical values; winner
  dataset/model labels publish trimmed strings, never `str(...)` conversions.
  Temporary conformal objective calibration metadata follows the same boundary:
  `TuningConformalScoreCalibration.metadata` and raw
  `score_data.conformal_calibration.metadata` reject non-string or
  whitespace-padded keys for column-style and row-style metadata mappings, plus
  non-finite or non-JSON-native metadata values.
- Direct construction of `RobustnessScenarioSpec` is fail-closed: `severity`
  must be a real numeric scalar, not a boolean or numeric string, and `extra`
  keys must be canonical non-empty strings before the scenario can be published
  through `to_dict()` or executed in a report.
- Raw `robustness.scenarios` mappings use the same strict boundary: keys must
  be canonical non-empty strings, `severity` must be a real finite numeric
  scalar, and payloads must remain TCV1 JSON-native before report execution.

The broader DAG-ML graph surface remains fail-closed for native tuning. Branch
and merge graphs, implicit dataset loading, automatic spectral replay, native
predictor-package conformal replay for arbitrary bundles, and Studio spectral
OOD charts are not claimed by this audit.

## Load-bearing user syntax

The native path is explicit by design:

```python
import nirs4all

result = nirs4all.run(
    pipeline=[{"model": estimator}],
    dataset=(X_train, y_train, sample_ids),
    engine="dag-ml",
    workspace_path="workspace/",
    tuning={
        "engine": "optuna",
        "space": {"model.n_components": [2, 4, 6]},
        "force_params": {"model.n_components": 4},
        "metric": "rmse",
        "direction": "minimize",
        "score_data": {
            "X": X_score,
            "y": y_score,
            "sample_ids": score_ids,
        },
        "winner": {
            "X": X_calibration,
            "y_true": y_calibration,
            "sample_ids": calibration_ids,
        },
    },
    calibration={
        "coverage": [0.8, 0.9, 0.95],
        "workspace_conformal_id": "pls-conformal",
    },
)

interval_90 = result.interval(0.9)
status = result.conformal_guarantee_status
robustness_report = result.robustness(y_true=y_true_for_result)

prediction = nirs4all.predict_calibrated(
    result.calibrated,
    y_pred=y_pred,
    prediction_sample_ids=prediction_ids,
    coverage=0.9,
)

report = prediction.robustness(
    y_true=y_observed,
    scenarios=[
        {"kind": "observed", "severity": 0.0},
        {"kind": "prediction_noise", "severity": 0.05, "distribution": "normal"},
    ],
    workspace_path="workspace/",
    workspace_robustness_id="pls-robustness",
)
report.save_artifacts("robustness-artifacts")

workspace_report = nirs4all.robustness_from_workspace_prediction(
    "workspace/",
    "pred-001",
    y_true=y_observed,
    scenarios=[{"kind": "spectral_offset", "severity": 0.01}],
    save_to_workspace=True,
    workspace_robustness_id="pred-001-spectral-audit",
)
```

Important effects:

- `tuning.force_params` enqueues a caller-provided parameter assignment as the
  first HPO trial. Optuna uses `study.enqueue_trial(...)`; n4m requires native
  `optimizer.enqueue(...)` support and fails closed on older bindings. Changing
  it can change the selected predictor and therefore can invalidate calibration.
  On a non-empty Optuna `resume=True` study, the assignment must match the
  already materialized warm-start trial; it is not enqueued again, and changed
  warm-start values under the same `study_name`/`storage` fail closed. Existing
  materialized trial params must also match the current `tuning.space` keys
  exactly; changed search-space keys under the same persisted study fail closed
  before optimizer execution. Existing categorical values must also remain
  present in the current choices for their key; removing or renaming a choice
  under the same persisted study fails closed before optimizer execution.
  Existing numeric values must also remain inside the current range for their
  key; narrowing a range so that a stored trial falls outside it fails closed
  before optimizer execution, and a current numeric `step` also has to contain
  every restored value. During Optuna storage-backed resume, `n_trials` is the
  target total trial count; a one-trial persisted study with `n_trials=1` runs
  no extra trial, while `n_trials=2` runs one remaining trial. New Optuna
  storage-backed studies persist nirs4all `study.user_attrs` for format, schema
  version, optimizer contract fingerprint and search-space fingerprint;
  non-empty studies missing those attrs or carrying mismatched fingerprints fail
  closed during resume. Optuna storage-backed resume reconstructs compact
  diagnostics for restored rows:
  completed rows use `score_extractor="optuna_storage"`, failed rows use
  `score_extractor="failed"` and pruned rows use `score_extractor="pruned"`.
  Restored Optuna `COMPLETE` rows must carry a finite numeric value; missing or
  non-finite storage values fail closed instead of becoming completed trials
  with no usable objective value.
  Restored non-`COMPLETE` rows must not carry a final storage value; failed,
  pruned or in-flight rows with final values are rejected as corrupted optimizer
  history. Restored `RUNNING` rows fail closed during resume because interrupted
  active trials cannot be safely recovered into a terminal HPO tape. Restored terminal Optuna rows must keep exactly the current
  `tuning.space` parameter keys when the search space is non-empty; rows whose
  stored parameter table was removed fail closed instead of becoming completed
  trials with empty public params. Restored queued Optuna `WAITING` rows that
  already carry materialized params or `fixed_params` must also satisfy the current
  `tuning.space`; incompatible values are rejected before Optuna can consume
  them.
  Restored Optuna trial numbers must be canonical unique integers; corrupt
  storage that yields non-integer or duplicate trial numbers fails closed before
  the HPO tape is projected.
- n4m optimizer-state persistence is local and N4MOPT-backed in the shared
  `PipelineObjective` subset. `storage="file:///absolute/checkpoint-dir"` plus
  a filename-safe `study_name` writes a JSON checkpoint manifest after each
  terminal trial. `resume=True` reloads only when the optimizer contract and
  search-space fingerprints match, and treats `n_trials` as the target total
  trial count. Restored trial rows are decoded back to public
  `TrialResult.params`, ordered canonically by numeric trial id, and rejected
  when duplicate or non-integer restored trial ids are present, including
  named categorical `options` whose optimizer labels differ from their
  JSON-native public values. Restored checkpoint row
  params must still match the current `tuning.space` keys and value domains:
  edited or incompatible checkpoint keys, categorical choices, numeric ranges or
  numeric steps fail closed before they can contribute optimizer history.
  Restored `COMPLETE` rows must carry a finite numeric score; missing, boolean or
  non-finite scores fail closed instead of becoming complete trials with no usable
  value. Restored failed, pruned
  and cancelled rows keep compact `score_extractor="failed"`,
  `score_extractor="pruned"` or `score_extractor="cancelled"` diagnostics for
  summary cards. Non-terminal checkpoint rows such as `RUNNING` fail closed
  during resume. Restored n4m non-`COMPLETE` rows must not carry a final score;
  failed, pruned or cancelled checkpoint rows with scores are rejected as
  corrupted optimizer history.
- n4m candidate failures require native failure reporting support. Bindings must
  expose either `optimizer.tell_result(...)` with a failed trial status or
  `optimizer.tell(...)`; otherwise the n4m PipelineObjective adapter fails
  closed instead of silently losing the failed-trial tape.
- Optuna- and n4m-pruned candidates stay distinct from failed candidates. When
  the shared objective raises a prune exception, the HPO tape records
  `TrialResult(state="PRUNED", value=None)` and the compact summary counts
  `PRUNED` separately with `score_extractor="pruned"`. n4m publishes that row
  only when the installed binding can terminalize the native trial through
  `optimizer.tell_result(..., TrialStatus.PRUNED)`; older bindings fail closed
  instead of rewriting the candidate as `FAIL` or as a worst completed score.
- `tuning.score_data` affects objective scores and trial ranking only.
- `calibration.coverage` controls the conformal intervals that are materialized
  on the calibrated prediction result.
- `predict.coverage` selects already-materialized intervals; it does not fit a
  new conformal calibrator.
- `predict.all_predictions` remains a conformal sidecar boundary: model `.n4a`
  bundles with `coverage=...` support the single selected prediction entry
  (`all_predictions=False`) and reject `all_predictions=True` until each entry
  can carry calibrated identity mapping.
- Model `.n4a` bundles that contain an invalid `conformal/` sidecar fail
  validation instead of falling back to an uncalibrated prediction path.
- `predict(model="*.n4a", coverage=...)` also rejects a structurally complete
  conformal sidecar whose `calibrated_result.json` has non-empty predictions
  but no canonical physical `sample_ids`. The sidecar is validated before the
  raw model prediction runs, so invalid conformal identity cannot be hidden by a
  successful uncalibrated model replay.
- Conformal reload is artifact-derived, not only fingerprint-derived. The
  stored `qhat_by_coverage` values must recompute from the retained
  non-negative residual scores, and every materialized interval must equal
  `y_pred ± qhat` for the embedded conformal artifact. Stores, workspace rows
  and `.n4a` sidecars reject edited intervals or quantiles even when the JSON is
  otherwise self-consistent.
- `ConformalCalibrationSpec` validates direct construction the same way as
  `parse_conformal_calibration_spec()`: coverage values must be real numeric
  scalars, not booleans or numeric strings, and method, unit, group keys and
  multi-target mode are canonicalized before fingerprinting.
- Grouped split conformal is executable in the replayed-array substrate:
  `group_by="group"` consumes calibration and prediction group labels, while
  metadata keys consume `calibration_metadata` and `prediction_metadata`.
  Prediction rows with missing, null or unseen groups fail closed without a
  global-quantile fallback, and reload verifies row-aligned grouped `qhat`
  vectors plus strict non-boolean integer grouped `n_samples` summaries against
  the embedded artifact. Filesystem stores, workspace
  `conformal_results` rows, conformal-only `.n4a` bundles and model `.n4a`
  sidecars preserve and revalidate `group_keys`, `group_calibrators` and grouped
  qhat vectors.
- `multi_target="joint_max"` is executable for two-dimensional replayed-array
  regression outputs. The conformal score is one scalar per physical sample,
  `max(abs(y_true - y_pred))` across target columns, and the materialized
  interval arrays keep the same shape as `y_pred`. The published guarantee is
  simultaneous for the target vector, not separate per-target conditional
  coverage.
- Conformal reload is identity fail-closed across filesystem stores, workspace
  `conformal_results` rows and `.n4a` sidecars. A stored
  `calibrated_result.json` whose non-empty prediction cohort lacks canonical
  physical `sample_ids` fails reload, and a corrupted workspace
  `conformal_results` row with the same missing prediction identity is rejected
  by both `load_workspace_calibrated_result(...)` and
  `load_workspace_calibrated_predict_result(...)`.
  `CalibratedRunResult` metadata is also strict JSON-compatible at construction
  and reload time; non-finite floats, Python objects, non-string keys or
  whitespace-padded keys fail closed before fingerprinting or persistence.
  Nested mapping keys are checked the same way, and tuple values are rejected
  instead of being silently coerced into JSON arrays.
  Workspace conformal-row metadata is validated before insertion as well, so
  invalid application metadata cannot bypass the conformal artifact verifier via
  the database row.
  Public conformal result metadata supplied through
  `calibrate(..., result_metadata=...)` or
  `predict_calibrated(..., result_metadata=...)` is validated at the API seam
  before generated guarantee metadata is merged; non-canonical keys, Python
  objects, tuples and non-finite numbers fail closed with public labels. The
  keyword registry publishes both entries with strict JSON-native schemas and
  explicit calibrated-result fingerprint effects.
- Robustness workspace metadata uses the same strict JSON-native row boundary:
  `save_workspace_robustness_report(...)` and
  `robustness(..., workspace_path=..., workspace_metadata=...)` fail closed
  before writing `robustness_results` rows with non-canonical keys, Python
  objects, tuples or non-finite numbers.
- Prediction workspace metadata uses the same strict JSON-native sidecar
  boundary: `save_workspace_predict_result(..., metadata=...,
  result_metadata=...)` and `predict(..., save_to_workspace=True,
  workspace_metadata=..., workspace_result_metadata=...)` fail closed before
  writing prediction arrays with non-canonical keys, Python objects, tuples or
  non-finite numbers. These fields remain workspace diagnostics/replay
  provenance only and do not create or renew conformal guarantees.
  Conformal calibration cohort rows built directly or reloaded from JSON now
  require a strict non-boolean integer `row_index`, canonical non-whitespace
  `sample_id`, `role` and optional `group`, plus strict JSON-native metadata
  with string keys, before the cohort manifest can be fingerprinted. The
  optional serialized `n_samples` summary must also be a strict non-boolean
  integer matching the row count. Row-aligned calibration metadata supplied as
  either column mappings or per-row mappings uses the same strict key rule:
  non-string or whitespace-padded metadata keys fail before manifest JSON
  coercion.
  Dataset-backed selectors reject non-string or whitespace-padded keys plus
  non-JSON-native values in `ConformalCalibrationData` and raw
  `calibration_data={...}` mappings, and `include_augmented` must be a boolean
  before either surface can publish a runtime calibration payload.
  Conformal numeric arrays (`y_true`, `y_pred`, interval bounds and `qhat`)
  reject boolean payloads instead of coercing them to `0.0`/`1.0`, and reject
  numeric strings such as `"1.0"` instead of parsing them as floats. The same
  bool rejection applies to Python-side `from_dict(...)` payloads carrying NumPy
  boolean scalars in serialized scores or quantiles. Serialized method/unit
  contract fields reject arbitrary Python objects instead of accepting their
  `__str__` output as evidence.
  serialized numeric fields reject NumPy ndarray scalars instead of coercing
  them to JSON numbers.
  Empirical `ConformalMetricSet` diagnostics are validated before fingerprinting
  or publication: `observed_coverage` must be finite in `[0, 1]` and match
  `n_covered / n_samples`, `coverage_gap` must equal
  `observed_coverage - coverage`, and width/interval-score diagnostics must be
  non-negative or positive infinity for unbounded intervals.
  Direct `ConformalIntervalBlock` and `CalibratedPredictionBlock` construction is
  fail-closed for coverage-key mismatches, interval shape mismatches, inverted
  bounds, unsupported method/unit values, invalid group-key lengths, and negative
  or non-row-aligned `qhat` values.
  Direct `SplitConformalCalibrator` construction validates retained residual
  scores, coverage keys, recomputed quantiles, method and unit before `apply()`;
  negative scores, edited `qhat` values or unsupported vocabulary fail closed
  instead of materializing invalid intervals.
  Version fields on conformal cohort manifests, calibration artifacts and
  calibrated results are strict integer contract tags; boolean `true`/`false`
  and numeric strings fail closed instead of being coerced to schema version `1`.
  Optional conformal artifact identity strings (`target_name`,
  `predictor_fingerprint`, `calibration_data_fingerprint`) must be null or
  non-empty strings without surrounding whitespace; invalid direct-construction
  and reload payloads fail closed before provenance publication.
  Guarantee metadata string fields (`effective_engine`, `requested_engine`,
  `source_calibrated_result_fingerprint`, `invalidation_reasons`) are also strict
  provenance fields; booleans, objects, empty or whitespace-padded values fail
  closed instead of being stringified, and persisted
  `conformal_guarantee_status.version` must be the strict integer `1`. Status
  `predictor_fingerprint`, `calibration_data_fingerprint`, `guarantee`, and
  `scope` must also match the embedded artifact on construction and reload. A
  persisted status must include the complete generated field set, and `status`
  must be `active` exactly when `invalidation_reasons` is empty, otherwise
  `invalidated`. The generated `limitations` list must also match the embedded
  artifact's guarantee mode exactly; edited, shortened, empty or non-string
  limitation payloads fail closed.
- DAG-ML D10 `cache_namespace_fingerprints` are treated as signed native
  control-plane proofs. When present in native training, replay, bundle or
  prediction-cache payload objects, nirs4all forwards them unchanged to
  `dag_ml`; DAG-ML owns validation, namespace-aware handle derivation,
  persistent file-store payload naming and columnar manifest exposure. There is
  no nirs4all keyword that mutates this proof after signing.
- `robustness.scenarios` creates diagnostics on a frozen prediction cohort; it
  does not refit the model, recalibrate intervals or upgrade a conformal
  guarantee.
- Spectral robustness reports that replay an explicit `predictor` or
  `predictor_bundle` record `metadata["spectral_replay"]` and carry the same
  optional `spectral_replay` block in `summary.json`: source, route, saved bundle
  path when applicable, `all_predictions=False` for bundle replay, and whether
  sample ids were forwarded. This is provenance for auditors and hosts, not a
  permission for bindings or Studio to replay spectra locally.

The finite-sample split conformal quantile is
`ceil((n_calibration + 1) * coverage)`. For small calibration cohorts and high
coverages, that rank can exceed the retained score count; in that case the
materialized interval is intentionally unbounded and round-trips as
`qhat=Infinity`.

For the complete keyword/effect table, see
{doc}`pipeline_keywords` and {doc}`/user_guide/models/native_tuning_conformal`.
The guide also contains a compact `Native keyword/effect quick map` for
integrators. It maps each load-bearing syntax (`run.engine`, `run.tuning`,
`run.tuning.space`, `run.tuning.force_params`, `run.tuning.score_data`,
temporary conformal scoring, `run.tuning.winner`, final calibration,
`predict(coverage)`, `robustness`, spectral/OOD replay and the `Predictions`
bridge) to its runtime effect, published evidence and fail-closed boundary.
Use that quick map when building CI checks, bindings, Studio forms or generated
configuration.

## Public artifacts and contracts

| Artifact or API | Producer | Consumer contract |
| --- | --- | --- |
| `RunResult.tuning_result` and `tuning_*` accessors | `run(tuning=...)` | Optimizer evidence, best params, best value and trial diagnostics. |
| `CalibratedRunResult` / conformal `PredictResult` | `calibrate()`, `run(..., calibration=...)`, `predict_calibrated()` | Point predictions, materialized intervals, coverage metadata and guarantee status. |
| `TuningResult.summary_artifact()` | `run(tuning=...)`, `tune_single_estimator(...)`, workspace tuning loaders | Lightweight HPO card/index payload: fingerprint, engine, metric, optimizer, sampler, pruner, seed, best value/params, safe persistence flags, compact trial status rows and scalar trial diagnostics such as `error_type`, `score_family`, `score_extractor` and fingerprints. It excludes candidate params and raw exception messages. |
| `RobustnessReport` | `robustness()`, `PredictResult.robustness()` or `nirs4all.robustness_from_workspace_prediction()` | Audit-only metrics, slices, scenarios and deterministic exports. |
| `summary.json` | `RobustnessReport.summary_artifact()` or `save_artifacts()` | Lightweight card/dashboard payload for CI, Studio and bindings, including optional `conformal_guarantee_status` and `spectral_replay` blocks. |
| `keyword-registry.json` | `get_keyword_registry()` / docs build / CLI | Machine-readable syntax, effects, invalidations, support level and UI hints. |
| `robustness-summary.schema.json` | `get_robustness_summary_schema()` / docs build / CLI | JSON Schema for validating robustness summary cards without parsing full reports. |
| `tuning-summary.schema.json` | `get_tuning_summary_schema()` / docs build | JSON Schema for validating tuning summary cards without parsing full HPO tapes. |
| Conformal result stores, workspace `conformal_results` rows and `.n4a` sidecars | `calibrate()`, `save_workspace_calibrated_result(...)`, `attach_calibrated_result_to_bundle(...)` | Verified conformal result JSON. Reload revalidates the `CalibratedRunResult` identity contract, including required canonical prediction `sample_ids`, before exposing a `CalibratedRunResult` or conformal `PredictResult`. |
| DAG-ML `cache_namespace_fingerprints` | Native DAG-ML training/replay contracts | Candidate/data/fold/trial/seed cache identity proof owned by DAG-ML. nirs4all forwards it unchanged and does not expose a mutation keyword. |
| `nirs4all.load_workspace_predict_result()` / `nirs4all.load_workspace_predict_results()` / `Predictions.get_predict_result_by_id()` / `Predictions.to_predict_results()` | Workspace/store prediction loaders with `load_arrays=True` | Supported bridge from stored prediction rows to native `PredictResult`, preserving intervals, conformal/tuning replay provenance, robustness evidence, executable `X`/`spectra`, sample ids and model metadata. |
| `nirs4all.robustness_from_workspace_prediction()` | Workspace prediction loader plus `robustness()` | Supported bridge from one stored prediction row to a native robustness report; consumes executable `X`/`spectra` + `predictor_bundle` evidence as defaults and can persist the report back with a `prediction_id` link. |

Bindings and Studio should consume these public artifacts and schemas. They must
not infer conformal guarantees, robustness status, or invalidation causes from
raw native predictions alone. They may display `spectral_replay` provenance from
`summary.json`, but must not use it to claim local spectral replay execution.
They should likewise inspect `PredictResult.calibration_replay_source`,
`PredictResult.tuning_calibration_source` and
`PredictResult.spectral_replay_evidence_status` instead of scraping sidecar
rows or synthesizing missing spectral replay inputs.

## Cross-repository responsibility boundary

`nirs4all-methods`, `n4m` and `pls4all` provide native numerical kernels,
thin C ABI/Python bindings and the single-estimator
`n4m.model_selection.finetune_estimator(...)` selection trace. They do not emit
`CalibratedRunResult`, conformal guarantee metadata, `TuningResult` summaries,
`RobustnessReport`, workspace rows, `.n4a` bundles or Studio cards.

`nirs4all` owns the statistical lifecycle:

```text
run(tuning=...) -> calibrate()/predict_calibrated() -> robustness()
```

That lifecycle is where final winner projection, calibration evidence,
workspace persistence, artifact publication, keyword effects and invalidation
rules are defined.

## Validated local evidence

The current local implementation is backed by targeted checks rather than one
monolithic test run after every edit. Relevant gates include:

- native tuning/conformal/robustness example coverage:
  `test_native_tuning_conformal_example.py` in the integration suite;
- tuning, calibration and robustness API tests:
  `tests/unit/api/test_tuning.py`,
  `tests/unit/api/test_calibrate.py`,
  `tests/unit/api/test_robustness.py`,
  `tests/unit/api/test_result.py`;
- public API and storage regression tests:
  `tests/regression/test_public_api_contract.py`,
  `tests/regression/test_storage_schema_contract.py`;
- keyword/effect registry and docs coverage:
  `tests/unit/pipeline/test_keyword_registry.py`,
  `tests/unit/docs/test_keyword_registry_extension.py`,
  `tests/unit/docs/test_native_tuning_conformal_docs.py`;
- Studio endpoint and UI tests for conformal/robustness summaries, exports and
  stored-prediction preflight;
- `nirs4all-methods/docs/_extras/test_nirs4all_capability_boundary.py`, which
  keeps the methods/bindings responsibility boundary documented.

Before publishing a release, run a grouped final validation over the touched
repositories and include the exact command output in the release notes or PR.

## Explicit non-claims

This page does not claim:

- native tuning for arbitrary DAG branch/merge graphs;
- automatic spectral/OOD perturbation replay from a stored prediction alone;
- bindings or Studio replaying spectra merely because `summary.json` carries
  `spectral_replay` provenance;
- recalibration during robustness reporting;
- conformal guarantees inferred inside language bindings;
- post-signature mutation of DAG-ML D10 `cache_namespace_fingerprints` from a
  nirs4all keyword;
- applying a conformal sidecar to every model prediction entry with
  `all_predictions=True`;
- silently ignoring invalid `conformal/` sidecars and returning uncalibrated
  predictions;
- parity for bindings that only declare metadata-level artifact contracts;
- Studio graphical spectral/OOD campaign execution.

Those items require dedicated implementation gates, fixtures and review before
they can move from fail-closed to supported.
