# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### ⚙️ Changed

- **`dag-ml` is now a fully selectable execution backend.** Pass `engine="dag-ml"` (or set
  `$N4A_ENGINE=dag-ml`) and `nirs4all.run()` dispatches to the dag-ml backend — the pipeline
  runs natively (Rust) and returns a `RunResult` of dag-ml's native scores. This is the first
  half of the "nirs4all = nirs4all-core aggregate + Python controllers" North Star.
  **The DEFAULT engine stays `legacy`** — the public-maintained nirs4all remains pure-Python by
  default until a planned global refactoring, after which the legacy-DROP cutover makes dag-ml
  the default. (The ADR-17 flip briefly defaulted to dag-ml; it was rolled back to legacy for
  the public version — see `ADR-17_LEGACY_DROP_HANDOFF.md`.) `predict()` / `explain()` /
  `retrain()` preserve their existing public signatures and remain legacy-only for this
  transition release: passing `engine="dag-ml"` through their existing keyword passthrough, or
  setting `$N4A_ENGINE=dag-ml`, fails loudly until native replay/explain/retrain contracts exist.
  `Session.run()` continues to use the session's `PipelineRunner` directly.
- **In-process dag-ml execution is the default mechanism for `engine="dag-ml"`.** The native PyO3
  path runs without the per-call subprocess import tax. An unset `N4A_DAGML_INPROCESS` means
  in-process; set it to one of `0`/`false`/`off` (case-insensitive) to force the subprocess
  (`dag-ml-cli`) path for debugging/isolation.
- **`dag-ml` and `dag-ml-data` are now hard (core) dependencies**, no longer the optional
  `[dagml]` extra (which has been removed). The native backend ships with every install, so the
  dag-ml engine is selectable out of the box.

### ✨ Added

- **Public replayed-array conformal calibration surface.**
  `nirs4all.calibrate()` now supports the narrow V1 split-conformal path for
  already replayed calibration predictions and prediction outputs with explicit
  physical sample ids. Dataset loading and automatic predictor replay for
  calibration fitting remain fail-closed with structured missing-gate errors.
- **Grouped replayed-array conformal calibration.**
  `nirs4all.calibrate(..., group_by=...)` now executes grouped split conformal
  when calibration and prediction rows provide matching group evidence.
  `prediction_groups` and `prediction_metadata` route prediction rows to
  retained group quantiles; missing or unseen groups fail closed without global
  fallback. Filesystem stores, workspace `conformal_results` rows,
  conformal-only `.n4a` bundles and model `.n4a` sidecars preserve
  `group_keys`, `group_calibrators`, row-aligned grouped `qhat` vectors and
  strict non-boolean integer grouped `n_samples` summaries, then revalidate them
  on reload.
- **Joint-max multi-target conformal calibration.**
  `nirs4all.calibrate(..., multi_target="joint_max")` now supports
  two-dimensional replayed-array regression outputs. nirs4all retains one
  score per physical sample, `max(abs(y_true - y_pred))` across target columns,
  materializes intervals with the same shape as `y_pred`, and publishes a
  simultaneous target-vector guarantee.
- **Prediction-entry calibration input.**
  `nirs4all.calibrate(calibration_data=result.best, ...)` now accepts nirs4all
  prediction dictionaries that already contain `y_true`, `y_pred`, and explicit
  physical sample ids. Extra result metadata is ignored; missing IDs remain
  fail-closed.
- **Raw replayed-array calibration aliases.**
  `nirs4all.calibrate(calibration_data={...})` now accepts
  `y_pred_calibration` / `calibration_predictions`, `calibration_sample_ids` /
  `physical_sample_ids`, `calibration_groups`, and `calibration_metadata` in the
  replayed-array mapping form. The runtime canonicalizes exactly one alias per
  field and rejects ambiguous alias collisions before calibration.
- **Conformal calibration alias registry entries.**
  The keyword/effect registry now publishes `calibrate.calibration_data.groups`
  and `calibrate.calibration_data.metadata` with their read-only aliases for
  bindings, docs static exports and future Studio forms. It also publishes
  `calibration_sample_ids` as a read-only alias for
  `calibrate.calibration_data.sample_ids`, matching the runtime and docs.
- **Dataset-backed calibration sample-id aliases.**
  Raw dataset-backed `calibration_data={...}` mappings now accept
  `calibration_sample_ids` alongside `sample_ids`, `physical_sample_ids`, and
  `sample_id_column` for explicit calibration cohort identities. Multiple
  explicit sample-id aliases in one dataset-backed mapping are rejected as
  ambiguous before cohort extraction.
- **Dataset-backed calibration group/metadata aliases.**
  Raw dataset-backed `calibration_data={...}` mappings now accept
  `calibration_groups` and `calibration_metadata` alongside canonical `groups`
  and `metadata`, with ambiguous alias pairs rejected before cohort extraction.
- **Uniform robustness noise scenarios.**
  `prediction_noise` and `spectral_noise` robustness scenarios now accept
  `distribution="uniform"` in addition to `normal`. Uniform noise is centered and
  samples from `[-severity, +severity]`; registry/static exports and typed
  `RobustnessScenarioSpec` validation expose the same vocabulary.
- **Explicit `SpectroDataset` calibration cohorts.**
  `nirs4all.calibrate(calibration_data={"dataset": spectro_dataset,
  "selector": {...}, "y_pred": ...}, ...)` now extracts calibration targets,
  physical sample ids, optional groups and metadata from a selected
  `SpectroDataset` cohort while keeping replayed calibration predictions
  explicit. The replayed calibration predictions may also be supplied through
  the top-level `y_pred_calibration=` argument.
- **DatasetConfigs/path calibration cohorts.**
  The same `calibration_data.dataset` lane now accepts a `DatasetConfigs`
  object, dataset config mapping, or config/path string resolved by existing
  nirs4all dataset loaders. The resolved source must contain exactly one
  dataset and still requires an explicit selector.
- **In-memory predictor replay for calibration cohorts.**
  `calibration_data.predictor` can now replay a sklearn-like in-memory
  predictor on the selected dataset-backed calibration cohort when
  `calibration_data.y_pred` is not supplied. nirs4all forwards selected
  sample ids, groups and metadata to `predict()` when accepted.
- **Saved predictor path replay for calibration cohorts.**
  `calibration_data.predictor_bundle` now routes through public
  `nirs4all.predict()` on the selected dataset-backed calibration `X`, allowing
  explicit saved bundle/config paths to produce calibration predictions.
- **RunResult and workspace-chain replay for calibration cohorts.**
  `calibration_data.predictor_result` now accepts `RunResult.best`-like
  prediction entries, and `calibration_data.predictor_chain_id` replays a stored
  workspace chain when paired with nested `calibration_data.workspace_path`.
  Both routes use public `nirs4all.predict()` on the selected calibration `X`
  and preserve selected sample ids, groups and metadata. `predictor_chain_id`
  and its typed-helper read alias `workspace_chain_id` must now be canonical
  non-empty workspace chain ids without surrounding whitespace or NULs; invalid
  values fail before replay or provenance publication.
- **Physical sample ids are fail-closed before conformal publication.**
  `calibrate(..., calibration_sample_ids=..., prediction_sample_ids=...)` and
  `predict_calibrated(..., prediction_sample_ids=...)` now require canonical
  non-empty strings without surrounding whitespace or NULs. Invalid IDs are
  rejected before conformal prediction rows, intervals or replay provenance are
  published, instead of being cleaned with `strip()`.
  Conformal artifact identity/provenance strings and guarantee-status strings
  now apply the same NUL-byte rejection on direct construction and reload.
- **Grouped conformal labels no longer use implicit stripping.**
  `calibration_groups`, `prediction_groups` and persisted prediction
  `group_keys` now reject whitespace-padded, NUL-containing or non-string labels
  before group quantiles are fitted or selected.
- **Lifecycle registry entries for explicit calibration cohorts.**
  The keyword/effect registry now documents the public
  `calibration_data.dataset`, `selector`, `y_pred`, `sample_ids`,
  `sample_id_column`, `group_column`, and `metadata_columns` keys so generated
  docs and downstream UI forms can distinguish calibration evidence, replayed
  predictions, identities, groups, and audit metadata.
- **Dataset-backed calibration helper payloads are stricter.**
  `ConformalCalibrationData(dataset=..., selector=...)` now rejects selector
  keys that are not canonical non-empty strings, rejects non-JSON-native
  selector values, and requires `include_augmented` to be a boolean before
  publishing the runtime mapping. Raw `calibration_data={...}` mappings now use
  the same selector and `include_augmented` boundary before cohort extraction.
  Dataset-backed calibration column selectors are strict on both the typed
  helper and raw mapping paths: `sample_id_column` and `group_column` must be
  canonical non-empty strings, and `metadata_columns` must be one canonical
  string or a duplicate-free sequence instead of being coerced with `str(...)`.
- **Native tuning contracts are fail-closed on direct construction.**
  `SearchSpaceParameter`, `ParameterPatch`, `OrderedSearchSpaceSpec`,
  `DagMLTuningSpec`, `TrialResult` and `TuningResult` now canonicalize tuning
  paths, reject non-TCV1 values, non-finite scores, boolean integer contract
  fields, duplicate trial ids and params outside `tuning.space` before
  fingerprinting, JSON persistence or summary publication.
  `TuningResult.from_dict()` and `load_json()` now apply the same strict
  `best_value` check, rejecting booleans and numeric strings instead of
  coercing them with `float()`.
  Direct `OrderedSearchSpaceSpec.parameter_patches(...)` calls now reject
  non-string patch keys instead of converting them with `str(...)`; string keys
  still use the documented dotted/double-underscore canonicalization.
  `TuningResult.summary_artifact()` now also rejects non-scalar payloads under
  its whitelisted compact diagnostic keys instead of publishing stringified
  lists or mappings in CI/UI summary cards.
- **Structured passthrough markers are literal.**
  The native linear `run(tuning=...)` subset still accepts `None`, the plain
  `"passthrough"` string and `{"kind": "passthrough"}` for non-final
  preprocessing no-ops, but the structured marker now requires `kind` to be the
  literal string `"passthrough"` instead of accepting host objects via
  `str(...)` coercion.
- **Native optimizer adapter seams are fail-closed on direct construction.**
  `ObjectiveTuningRunResult` now requires a real `TuningResult` and canonical
  optional `tuning_id`; the internal categorical codec now rejects empty or
  duplicate choices, non-native choices without a decoder, mismatched decoder
  keys, duplicate decoded public values and non-TCV1 decoded payloads before
  Optuna/n4m can enqueue or decode categorical trials.
- **Public tuning helper payloads are stricter.**
  `TuningScoreData(dataset=..., selector=...)` and
  `TuningWinner(dataset=..., selector=...)` now reject non-string or
  whitespace-padded selector keys plus non-JSON-native selector values and
  require boolean `include_augmented`;
  `TuningWinner.score` rejects booleans and numeric strings;
  `TuningCalibration.as_predict_result` must be boolean; tuning
  metadata/extra mappings require canonical string keys and strict JSON-native
  finite values before runtime publication. Non-finite numbers, bytes, tuples,
  sets and arbitrary Python objects are rejected instead of being silently
  published. Raw `NativeTuning(score_data={...})`, `winner={...}` and
  `calibration={...}` mappings now enforce the same boundary before `to_dict()`
  can publish them, including rejection of nested `calibration_data`. Raw
  `calibration.coverage`, `method`, `unit`, `workspace_metadata` and
  `workspace_conformal_id` now use the same strict validation and
  canonicalization as the typed `TuningCalibration(...)` helper. Coverage
  values, including every element in multi-coverage lists, must be real numeric
  scalars; numeric strings are rejected instead of being coerced with
  `float(...)`. Top-level
  `NativeTuning` core fields are now validated through the `DagMLTuningSpec`
  contract before publication, so engine/metric/direction strings are
  canonicalized, integer/bool fields reject coercive strings and booleans, and
  optimizer persistence fields must be canonical strings.
  `TuningScoreData.metric`/`score_metric`, raw
  `NativeTuning.score_data.metric`/`score_metric`, `TuningWinner.metric`,
  `dataset_name`, `model_name`, `task_type` and the corresponding raw winner
  aliases now reject non-string, blank and NUL-containing values before
  publication instead of stringifying them; score/winner metrics and task types
  publish lowercase canonical values, while winner dataset/model labels publish
  trimmed strings.
  Dataset-backed tuning `sample_id_column` and `group_column` now require
  canonical non-empty strings, and `metadata_columns` must be one canonical
  string or a duplicate-free sequence of canonical strings; raw
  `NativeTuning.score_data` and `NativeTuning.winner` mappings enforce the same
  rule before publication.
  Tuple/list `NativeTuning.score_data` now validates its 2–5 field arity and
  strict metadata keys and JSON-native metadata values before publication, and
  valid tuple inputs publish as JSON-native lists instead of raw Python tuples.
  Scalar strings, bytes, numbers and arbitrary objects are rejected because
  public `score_data` must be a mapping or tuple/list cohort.
  The keyword registry now publishes the same strict JSON-native finite-value
  schema for tuning metadata, workspace metadata and calibration extras, so
  Studio, forms and bindings can reject non-canonical metadata before calling
  Python helpers.
  Workspace tuning/conformal persistence now applies the same rule when writing
  `tuning_results` and `conformal_results` rows; invalid metadata fails closed
  instead of being serialized through `default=str`. Workspace conformal-row
  metadata is validated before insertion, so invalid application metadata cannot
  bypass reload-time conformal artifact checks via the database row.
  Public conformal result metadata now uses the same strict JSON-native boundary
  in `calibrate(..., result_metadata=...)` and
  `predict_calibrated(..., result_metadata=...)` before generated guarantee
  metadata is merged. The keyword registry publishes both entries with the same
  strict schema, including their effect on calibrated result metadata and
  fingerprints.
  Robustness workspace metadata uses the same strict JSON-native row boundary
  when writing
  `robustness_results`; `save_workspace_robustness_report(...)` and
  `robustness(..., workspace_path=..., workspace_metadata=...)` fail closed on
  non-canonical keys, Python objects, tuples and non-finite numbers before
  persistence.
  `RobustnessReport.metadata`, which enters the report fingerprint and JSON
  artifacts, now applies the same no-stringify key boundary on direct
  construction and reload: non-string or NUL-containing keys and non-TCV1 values
  fail closed instead of being coerced by `to_dict()`.
  `RobustnessScenarioResult.scenario` and `RobustnessSliceResult.slice_key`,
  which feed report JSON, fingerprints, summary rows and tabular exports, now
  use the same strict mapping boundary on direct construction and reload:
  non-string or NUL-containing keys, non-TCV1 values and boolean result
  severities fail closed instead of being stringified into published evidence.
  Prediction workspace publication now applies the same strict JSON-native
  boundary to `workspace_metadata`/`workspace_result_metadata` and
  `save_workspace_predict_result(..., metadata=..., result_metadata=...)`;
  invalid keys, Python objects, tuples and non-finite numbers fail before the
  prediction sidecar is written.
- **Robustness scenario helpers are fail-closed on direct construction.**
  `RobustnessScenarioSpec` now rejects non-string `kind`/`distribution` values,
  boolean or numeric-string severities and non-canonical or NUL-containing
  `extra` keys before `to_dict()` can publish a scenario payload, instead of
  accepting host objects through `str(...)`, coercing `true` to `1.0` or
  converting non-string keys to strings.
- **Raw robustness scenario mappings reject coercive payloads.**
  `robustness.scenarios=[{...}]` now rejects non-string `kind`/`distribution`
  values, non-canonical or NUL-containing mapping keys and boolean or
  numeric-string severities before execution, matching the typed
  `RobustnessScenarioSpec` boundary.
- **Lifecycle registry correction for `predict.coverage`.**
  `predict.coverage` is now marked as partial support, matching the implemented
  calibrated replayed-array result/store/bundle lane and `.n4a` bundles with an
  attached conformal sidecar. If a bundle advertises an invalid `conformal/`
  sidecar, `predict(..., coverage=...)` now fails sidecar validation instead of
  falling back to uncalibrated prediction. A structurally complete sidecar whose
  `calibrated_result.json` has non-empty predictions but no canonical physical
  `sample_ids` is rejected before raw model prediction runs, so invalid
  conformal identity cannot be hidden by a successful uncalibrated replay. Full
  native predictor replay remains planned.
- **Conformal reload identity is fail-closed.**
  Filesystem conformal stores, workspace `conformal_results` rows and `.n4a`
  conformal sidecars now share the same reload contract: a stored
  `calibrated_result.json` whose non-empty prediction cohort lacks canonical
  physical `sample_ids` fails reload. Corrupted workspace conformal rows with
  the same missing prediction identity are rejected by both
  `load_workspace_calibrated_result(...)` and
  `load_workspace_calibrated_predict_result(...)` before a partial
  `CalibratedRunResult` or conformal `PredictResult` can be exposed. Result
  metadata must also be strict JSON-compatible at construction and reload time:
  non-finite floats, Python objects, non-string keys or whitespace-padded keys
  fail closed before fingerprinting or persistence. Nested mapping keys are
  checked the same way, and tuple values are rejected instead of being silently
  coerced into JSON arrays. Conformal calibration cohort rows built directly or
  reloaded from JSON now require a strict non-boolean integer `row_index`,
  canonical non-whitespace `sample_id`, `role` and optional `group`, plus strict
  JSON-native metadata with string keys, before the cohort manifest can be
  fingerprinted. The optional serialized `n_samples` summary must also be a
  strict non-boolean integer matching the row count. Row-aligned calibration
  metadata supplied as either column mappings or per-row mappings uses the same
  strict key rule: non-string or whitespace-padded metadata keys fail before
  manifest JSON coercion. `ConformalCalibrationSpec`
  validates direct construction the same way as
  `parse_conformal_calibration_spec()`: coverage values must be real numeric
  scalars, not booleans or numeric strings, and method, unit, group keys and
  multi-target mode are canonicalized before fingerprinting. Reloaded cohort
  manifest `unit` and calibrated prediction `method`/`unit` payloads must
  already be strings. Serialized method/unit contract fields reject arbitrary
  Python objects instead of accepting their `__str__` output as evidence.
  Conformal numeric arrays (`y_true`, `y_pred`,
  interval bounds and `qhat`) reject boolean payloads instead of coercing them
  to `0.0`/`1.0`, and reject numeric strings such as `"1.0"` instead of parsing
  them as floats. Python-side `from_dict(...)` payloads carrying NumPy boolean
  scalars in serialized scores or quantiles now fail closed the same way.
  Serialized numeric fields also reject NumPy ndarray scalars instead of
  coercing them to JSON numbers.
  Empirical `ConformalMetricSet` diagnostics now fail closed
  before fingerprinting or publication when observed coverage, coverage gap,
  width, interval score or count consistency is non-finite, out of range,
  negative or arithmetically inconsistent; positive infinity remains valid for
  unbounded interval metrics. Direct `ConformalIntervalBlock` and
  `CalibratedPredictionBlock` construction now fails closed for coverage-key
  mismatches, interval shape mismatches, inverted bounds, unsupported
  method/unit values, invalid group-key lengths, and negative or non-row-aligned
  `qhat` values. Direct `SplitConformalCalibrator` construction now validates
  retained residual scores, coverage keys, recomputed quantiles, method and unit
  before `apply()`, so negative scores, edited `qhat` values or unsupported
  vocabulary fail closed instead of materializing invalid intervals. Version
  fields on conformal cohort manifests, calibration artifacts and calibrated
  results are strict integer contract tags: boolean
  `true`/`false` and numeric strings fail closed instead of being coerced to
  schema version `1`. Optional conformal artifact identity strings
  (`target_name`, `predictor_fingerprint`, `calibration_data_fingerprint`) must
  be null or non-empty strings without surrounding whitespace; invalid
  direct-construction and reload payloads fail closed before provenance
  publication. Guarantee metadata string fields (`effective_engine`,
  `requested_engine`, `source_calibrated_result_fingerprint`,
  `invalidation_reasons`) are also strict provenance fields; booleans, objects,
  empty or whitespace-padded values fail closed instead of being stringified, and
  persisted `conformal_guarantee_status.version` must be the strict integer `1`.
  Status `predictor_fingerprint`, `calibration_data_fingerprint`, `guarantee`,
  and `scope` must also match the embedded artifact on construction and reload.
  A persisted status must include the complete generated field set, and `status`
  must be `active` exactly when `invalidation_reasons` is empty, otherwise
  `invalidated`. The generated `limitations` list must also match the embedded
  artifact's guarantee mode exactly; edited, shortened, empty or non-string
  limitation payloads fail closed.
- **Conformal reload intervals are artifact-derived.**
  Stored conformal quantiles must recompute from retained non-negative residual
  scores, and each materialized interval must equal `y_pred ± qhat` for the
  embedded artifact. Filesystem stores, workspace rows and `.n4a` sidecars reject
  edited intervals or quantiles even when the stored JSON has been made
  self-consistent.
- **DAG-ML D10 cache namespace boundary published.**
  Native training/replay objects carrying `cache_namespace_fingerprints` are
  treated as signed DAG-ML control-plane proofs. nirs4all forwards them
  unchanged through the native client; DAG-ML owns validation, namespace-aware
  handle derivation, file-store payload naming and columnar manifest exposure.
  There is no nirs4all keyword for post-signature mutation of these proofs.
- **Native HPO warm-start contract published.**
  `run(tuning=...)` and `NativeTuning` now expose `force_params` as a strict
  HPO contract key for the shared Optuna/n4m `PipelineObjective` lane. It
  enqueues a caller-provided first trial, requires keys to be a subset of
  `tuning.space`, preserves public decoded categorical values in results, and
  fails closed when n4m bindings do not expose native `optimizer.enqueue(...)`.
- **n4m failed-trial reporting is fail-closed.**
  The n4m `PipelineObjective` adapter now requires native bindings to expose
  either `optimizer.tell_result(...)` with a failed trial status or
  `optimizer.tell(...)` before it can preserve `TrialResult(state="FAIL")`
  entries after candidate fit/score errors. Older bindings that cannot report
  failed candidates fail closed instead of silently losing HPO tape evidence.
- **Optuna pruned trials remain distinct in the HPO tape.**
  The shared `PipelineObjective` adapter now preserves
  `optuna.exceptions.TrialPruned` as `TrialResult(state="PRUNED", value=None)`
  with compact diagnostics `score_extractor="pruned"`, and
  `TuningResult.summary_artifact()` counts `PRUNED` separately from `FAIL`.
- **n4m pruned trials use native `TrialStatus.PRUNED` or fail closed.**
  The n4m `PipelineObjective` adapter now preserves shared-objective prune
  exceptions as `TrialResult(state="PRUNED", value=None)` when the installed
  binding exposes `optimizer.tell_result(..., TrialStatus.PRUNED)`. Older
  bindings that cannot record a pruned terminal state fail closed instead of
  rewriting the candidate as `FAIL` or as a worst completed score.
- **Native keyword/effect quick map published.**
  The native tuning/conformal guide now includes a compact integration map that
  links each supported syntax (`run.tuning.space`, `run.tuning.force_params`,
  `run.tuning.score_data`, temporary conformal scoring, final calibration,
  `predict(coverage)`, `robustness`, spectral/OOD replay and the workspace
  `Predictions` bridge) to its runtime effect, published evidence and
  fail-closed boundary. The release audit and extended Python API index point to
  the same map, and document that integrations should use
  `PredictResult.calibration_replay_source`,
  `PredictResult.tuning_calibration_source` and
  `PredictResult.spectral_replay_evidence_status` instead of scraping sidecar
  rows or synthesizing missing spectral replay inputs.
- **Fail-loud conformal guarantee display in `PredictResult`.**
  Conformal `PredictResult` objects now expose materialized interval coverages
  and `conformal_guarantee_status` in the public prediction reference, and their
  string representation reports the guarantee status, effective engine and
  selected coverages instead of making interactive users infer a statistical
  guarantee from interval arrays alone.
- **Top-level keyword/effect registry exports.**
  `nirs4all.get_keyword_registry()` and `nirs4all.keyword_registry_json()` now
  expose the machine-readable keyword/effect registry from the primary Python
  API, so generated docs, Studio/forms and bindings can discover paths,
  schemas, effects, invalidation rules and UI hints without importing private
  pipeline modules. The same document is exportable from CI/build systems with
  `nirs4all keyword-registry --output keyword-registry.json`, and successful
  HTML documentation builds publish `_static/keyword-registry.json` for
  Studio/Web/static consumers. `get_keyword_registry_schema()`,
  `keyword_registry_schema_json()`, `nirs4all keyword-registry --schema`, and
  `_static/keyword-registry.schema.json` provide a JSON Schema for validating
  the published registry. The published JSON payload now has an explicit
  SHA-256 contract test so schema/content drift is visible before release.
- **Audit-only robustness/generalization reports.**
  `nirs4all.robustness()` now exposes the first public robustness surface for
  already replayed `PredictResult` or `CalibratedRunResult` objects. The initial
  `clean_frozen` lane computes point metrics, conformal diagnostics when
  intervals are materialized, and diagnostic metadata slices without perturbing
  spectra, replaying predictors, recalibrating, or refitting. It also supports
  deterministic `prediction_bias`, seeded `prediction_noise`, and explicit
  `X` + `predictor` or `predictor_bundle` spectral stress cells over already
  materialized predictions and intervals. The saved-bundle route replays
  perturbed spectra through public `nirs4all.predict(model=predictor_bundle,
  data={"X": X, "sample_ids": ...}, all_predictions=False)` without refit or
  recalibration, and remains mutually exclusive with in-memory `predictor`.
  Spectral reports now record `metadata["spectral_replay"]` with the replay
  source, saved bundle path when applicable, replay route, and sample-id
  forwarding status, and `RobustnessReport.summary_artifact()` carries that block
  when present for lightweight UI/CI consumers.
  Broader perturbations and modes remain fail-closed.
  Reports support deterministic JSON export/reload with fingerprint
  verification, deterministic Markdown and standalone HTML summaries, and
  reloadable Parquet-directory tabular export with table row counts and
  fingerprints for CI/release artifacts. Public
  helpers now derive degradation rows against a reference scenario and worst
  diagnostic slices for quick robustness triage. `RobustnessReport.summary_rows()`
  now exposes compact per-scenario cards for CI and Studio, Markdown/HTML
  summaries render the same compact scenario section, and the Parquet
  directory export includes the derived `summary` table. `nirs4all
  robustness-report` republishes verified robustness JSON artifacts as JSON,
  Markdown, HTML, or Parquet-directory outputs for CI/release jobs without
  re-running the audit. `RobustnessReport.save_artifacts()` now writes one
  deterministic publication directory with manifest plus JSON, lightweight
  `summary.json`, Markdown, HTML, and Parquet artifacts for CI, release
  automation, bindings, and Studio. The summary payload is also exposed through
  `summary_artifact()`, `to_summary_json()`, `save_summary(...)`, and CLI
  `--format summary`; `get_robustness_summary_schema()` and
  `robustness_summary_schema_json()` publish its JSON Schema for CI, bindings
  and Studio validation, and `nirs4all robustness-summary-schema` exports the
  same schema from CLI. The full bundle is available from `nirs4all
  robustness-report --format artifacts` and `nirs4all workspace robustness export
  --format artifacts`.
  `RobustnessReport.load_artifacts()` reloads these directories and verifies the
  manifest fingerprint, deterministic summary JSON, Markdown/HTML artifacts and
  Parquet table manifest before returning the report. `nirs4all robustness-report`
  now accepts either a verified JSON report or one of these artifact directories
  as input before republishing to any supported output format.
  `save_workspace_robustness_report()` and
  `load_workspace_robustness_report()` persist/reload verified robustness
  reports in the workspace, with `nirs4all workspace robustness list/show/export`
  exposing the same inventory and artifact publication path for CI and future Studio adapters.
  `robustness(workspace_path=..., workspace_robustness_id=...)` can now persist
  the verified report directly after audit computation, and the keyword/effect
  registry documents the workspace persistence keys separately from scientific
  robustness controls. `PredictResult.robustness(...)` and
  `CalibratedRunResult.robustness(...)` provide result-container convenience
  syntax equivalent to `nirs4all.robustness(result, ...)`. Markdown and HTML
  robustness exports now render fail-loud conformal guarantee details when
  present, including requested/effective engine, method, unit, selected and
  calibrated coverages, invalidation reasons and limitations. The lightweight
  robustness `summary.json` artifact and its JSON Schema now also carry optional
  `conformal_guarantee_status` and `spectral_replay`, so CI, bindings and
  Studio cards can display engine, coverage, invalidation state and spectral
  replay provenance without parsing the full report.
  `RobustnessScenarioSpec` is
  now a public typed helper for `robustness(scenarios=...)`, giving Python,
  Studio, forms and bindings a stable constructor plus fail-closed
  `to_dict()` serialization for observed, prediction-bias, prediction-noise
  and spectral audit cells. The machine-readable keyword registry now also
  exposes `robustness.scenarios.kind`, `.severity`, `.distribution`,
  `robustness.X`, `robustness.predictor`, and `robustness.predictor_bundle` so
  generated docs, Studio/forms and bindings can present scenario syntax and
  effects without parsing prose.
- **Lifecycle registry entries for public `run(tuning=...)` runtime keys.**
  The keyword/effect registry now documents `run.tuning.score_data`,
  `winner`, `calibration`, the top-level `run.calibration` alias, workspace ids/metadata, `resume`, and
  `calibration.workspace_conformal_id` with explicit effects and boundaries.
  The entries keep the current completed-result replay distinct from true
  interrupted optimizer checkpoint resume.
- **Compact tuning summary diagnostics.**
  `TuningResult.summary_artifact()` now includes scalar
  `trials[*].diagnostics` fields such as `error_type`, `score_family`,
  `score_extractor`, `search_space_fingerprint` and `tuning_fingerprint`.
  This lets CLI, bindings, CI cards and future Studio panels explain failed or
  conformal-aware trials from the lightweight summary artifact without parsing
  the full HPO tape. Candidate params and raw exception messages stay out of
  the compact summary.
- **Direct workspace prediction-to-`PredictResult` helper.**
  `nirs4all.save_workspace_predict_result(workspace_path, result, ...)`,
  `nirs4all.predict(..., save_to_workspace=True, workspace_result_metadata=...)`,
  `nirs4all.load_workspace_predict_result(workspace_path, prediction_id)` and
  `nirs4all.load_workspace_predict_results(workspace_path, dataset_name=None)`
  now expose the explicit publisher, prediction-time publisher shortcut,
  one-record loader and bulk workspace/store bridges as public APIs. The
  publisher writes `PredictResult` values plus optional executable
  `X`/`spectra` evidence and `result_metadata`; the loaders open the workspace
  with arrays, convert through
  `PredictResult.from_prediction_record()`, and preserve sample ids, model
  metadata, intervals and conformal/tuning/robustness replay provenance for
  notebooks, bindings, CI and Studio. Explicit lower-level workspace
  `prediction_id` values must now be canonical non-empty strings without
  surrounding whitespace or NULs; omitted ids still generate UUIDs, while invalid
  provided ids fail before insertion instead of being ignored or stringified.
- **Direct workspace prediction-to-robustness helper.**
  `nirs4all.robustness_from_workspace_prediction(workspace_path, prediction_id, ...)`
  now loads one stored prediction through the public `PredictResult` bridge,
  delegates to `nirs4all.robustness()`, consumes executable stored `X`/`spectra`
  plus `predictor_bundle` evidence as spectral/OOD defaults, and can persist the
  report back to the workspace linked to the same `prediction_id`.
  The CLI now exposes the same path as
  `nirs4all workspace robustness from-prediction --prediction-id ...`, with
  `--y-true`/`--y-true-json`, `--scenarios-json`, optional slicing metadata,
  `--save-to-workspace`, and the same JSON/summary/Markdown/HTML/Parquet/artifact
  output formats as persisted robustness exports.
- **Executable spectral/OOD evidence publishing through `Predictions`.**
  Store-backed `Predictions.add_prediction(...)` now accepts `X=...`,
  `spectra=...`, and `result_metadata=...` so Python publishers can write
  row-aligned spectral replay arrays and `robustness_evidence` into the
  workspace array sidecar directly. Reloading with `load_arrays=True` restores
  the arrays into `PredictResult.metadata`, and `Predictions.merge_stores(...)`
  preserves the same sidecar evidence during workspace consolidation.
- **Conformal-aware development objective scoring.**
  `run(tuning=...).score_data` can now include an explicit
  `conformal_calibration` cohort and a conformal metric such as
  `conformal_mean_width` or `conformal_interval_score`. Each candidate gets a
  temporary split conformal calibrator for objective scoring only; the final
  `run(..., calibration=...)` result remains derived from the projected winner.
  Trial diagnostics now expose the conformal score family, temporary-calibration
  extractor, and final-calibration boundary for audit/replay.
  The typed Python syntax now includes `TuningConformalScoreCalibration` and
  validates conformal metric/calibration/coverage consistency before execution.
  Its `metadata`/`calibration_metadata` payloads, and raw
  `score_data.conformal_calibration.metadata`, now reject non-string or
  whitespace-padded keys for both column-style and row-style metadata mappings
  before runtime publication.
  The keyword/effect registry now separately documents
  `run.tuning.score_data.conformal_coverage` and its read-only `coverage` alias.
  The U09 runnable example now exercises the typed conformal-aware objective
  scoring syntax before final winner calibration.
- **Tuning cohort metadata keys are strict before publication.**
  `TuningScoreData.metadata`, `TuningWinner.metadata` and their raw
  `NativeTuning.score_data.metadata` / `NativeTuning.winner.metadata` mapping
  forms now reject non-string or whitespace-padded keys for both column-style
  and row-style metadata before runtime publication.
  The public API regression contract now freezes the native tuning/conformal
  exports, `run(tuning=..., calibration=...)`, and `predict(coverage=...)`.
  `tune_single_estimator()` now consumes `NativeTuning(score_data=...)` directly
  for explicit array/tuple scoring cohorts, including typed conformal-aware
  temporary scoring; dataset-backed `score_data` remains routed through
  `run(tuning=...)`.
  Its public call signature is now frozen by the API regression contract.
  The same contract now also guards the result accessors consumed by downstream
  integrations: `RunResult.tuning_*` and `PredictResult` interval/guarantee
  helpers.
  The native tuning/conformal guide now shows those public accessors directly,
  so downstream code can avoid private fields and serialized JSON layouts.
  The regression contract also freezes the public conformal/robustness function
  signatures: calibration, calibrated prediction, metrics, robustness reports,
  bundle helpers and workspace save/load helpers.
  The same contract now guards the public member surface of typed tuning helpers
  and robustness report containers.
  It also snapshots typed tuning helper serialization (`to_dict()`), including
  conformal-aware scoring and dataset-backed score/winner payloads used by
  downstream forms and bindings.
  `TunedSingleEstimatorConformalResult` now proxies `tuning_best_params` and
  `tuning_best_value` in addition to `tuning_result` and `tuning_id`, so callers
  can inspect the composite tuned+calibrated result without reaching through
  `result.run` for the common tuning evidence.
  `CalibratedRunResult` and `ConformalMetricSet` are now exported from both
  `nirs4all` and `nirs4all.api`, matching the public function annotations and
  documentation for conformal calibration and diagnostics.
  Their public serialization is now covered by the API regression contract:
  `ConformalMetricSet.to_dict()` is snapshotted and `CalibratedRunResult`
  JSON round-trip plus artifact/prediction/guarantee fields are guarded.
  File-oriented conformal helpers now advertise `Path` support in their public
  signatures (`load_calibrated_result`, `export_calibrated_result`,
  `attach_calibrated_result_to_bundle`), matching the runtime behavior tested by
  the calibration suite.
  `calibrate(store_path=..., bundle_path=...)` now also advertises `Path`
  support in its public signature, and the persistence/export tests exercise
  direct `Path` inputs.
  `RobustnessReport.to_dict()` serialization is now covered by the API
  regression contract with a deterministic audit-only report snapshot and JSON
  round-trip, protecting publication artifacts consumed by CI, docs and Studio.
  Artifact container method signatures for conformal and robustness outputs are
  now frozen as public contracts, including JSON/Markdown/HTML/Parquet save/load
  methods used by release automation.
  The native tuning/conformal guide now documents the public result containers
  importable from `nirs4all`: `TunedSingleEstimatorConformalResult`,
  `CalibratedRunResult`, `ConformalMetricSet`, and `RobustnessReport`, including
  their intended accessor/export roles for Python users, CI, Studio and
  bindings.
  `calibrate(calibration_data=(y_true, y_pred, sample_ids, groups, metadata),
  ...)` now accepts a compact replayed-array tuple form with fail-closed arity,
  matching the explicit tuple conventions used by native tuning cohorts while
  still requiring physical calibration sample ids.
  `ConformalCalibrationData` is now exported as a typed helper for
  `calibrate(calibration_data=...)`, covering replayed-array evidence and
  selected dataset-backed calibration cohorts without widening the runtime
  subset. The typed helper now has execution coverage for explicit
  dataset-backed replay lanes (`y_pred`, in-memory `predictor`,
  `predictor_bundle`, `predictor_result`, and `predictor_chain_id` with
  `workspace_path`), including identity/metadata transport and fail-closed
  ambiguity checks. The public `calibrate()` signature now advertises the typed
  helper, mapping, tuple and `PredictResult` calibration evidence forms.
  The U09 executable example now smoke-tests the standalone
  `ConformalCalibrationData` path alongside the integrated
  `run(tuning=..., calibration=...)` workflow.
  The manual module-level API and public-interface reference now list the native
  conformal, robustness and tuning entry points instead of leaving them only in
  the specialized guide.
- **Prediction interval transport in `PredictResult`.**
  `PredictResult` now accepts optional materialized interval blocks, exposes
  `interval(coverage)` / `interval_coverages`, and includes lower/upper interval
  columns in `to_dataframe()`. `nirs4all.calibrate(..., as_predict_result=True)`
  returns this prediction container for the replayed-array conformal surface.
- **Fail-loud conformal guarantee metadata.**
  `CalibratedRunResult.conformal_guarantee_status` and
  `PredictResult.conformal_guarantee_status` now expose the requested/effective
  conformal engine, method, unit, calibrated and selected coverages,
  fingerprints, limitations, and invalidation reasons. `predict(...,
  coverage=...)` updates the selected coverage metadata without recalibrating.
- **Internal filesystem store for conformal calibration results.**
  `nirs4all.pipeline.dagml.conformal_store` persists calibrated replayed-array
  results as a verified directory containing manifest, artifact, and result JSON
  files. Loading checks manifest/result/artifact fingerprints and refuses
  mismatched artifacts. The public replayed-array surface can write the same
  layout with `nirs4all.calibrate(..., store_path=...)` and reload it with
  `nirs4all.load_calibrated_result(path)`.
- **Workspace persistence for conformal calibration results.**
  Workspace schema v3 adds a `conformal_results` table. `nirs4all.calibrate(...,
  workspace_path=..., workspace_conformal_id=...)`,
  `nirs4all.save_workspace_calibrated_result(...)`, and
  `nirs4all.load_workspace_calibrated_result(...)` persist and reload the same
  verified conformal artifact/result JSON from the nirs4all SQLite workspace.
- **Portable `.n4a` archives for calibrated replayed-array results.**
  `nirs4all.calibrate(..., bundle_path="result.n4a")` and
  `nirs4all.export_calibrated_result(result, "result.n4a")` package the verified
  conformal store into a reloadable archive. These bundles contain the conformal
  result/artifact, not yet the predictor/model artifact.
- **Reusing conformal calibrators for new point predictions.**
  `nirs4all.predict_calibrated(result_or_path, y_pred=..., prediction_sample_ids=...)`
  applies a saved replayed-array calibrator to already computed point
  predictions and returns either `PredictResult` with intervals or
  `CalibratedRunResult`.
- **`predict()` bridge for calibrated replayed arrays.**
  `nirs4all.predict(model=calibrated_result_or_path, data={"y_pred": ..., "sample_ids": ...}, coverage=...)`
  now routes to the same conformal replayed-array surface and selects only
  already materialized coverages.
- **Prediction coverage selector validation.**
  `predict(..., coverage=...)` now rejects empty, duplicate, non-finite, or
  out-of-range coverage selectors before selecting calibrated intervals,
  matching the keyword registry schema.
- **Conformal sidecars for existing model `.n4a` bundles.**
  `nirs4all.attach_calibrated_result_to_bundle(model_bundle, calibrated, output_path=...)`
  copies a model bundle and adds the verified `conformal/` sidecar. When such a
  bundle is passed to `nirs4all.predict(..., data={"X": ..., "sample_ids": ...},
  coverage=...)`, nirs4all replays the model bundle first and then applies the
  attached conformal intervals.
- **Empirical conformal diagnostics.**
  `nirs4all.conformal_metrics(result_or_path, y_true=...)` computes observed
  coverage, coverage gap, interval width, interval score, and miss counts for
  already materialized conformal intervals. These metrics are diagnostics on the
  supplied cohort and do not recalibrate or renew a guarantee. Metric payloads
  are validated at construction: observed coverage must match the covered count,
  coverage gap must match `observed_coverage - coverage`, and width/score
  diagnostics must be non-negative or positive infinity for unbounded intervals.
  The underlying prediction/interval blocks also validate coverage keys, shapes,
  method/unit vocabulary, group-key lengths, interval bounds and `qhat` alignment
  before metrics can be evaluated or serialized. Version fields on conformal
  cohort manifests, calibration artifacts and calibrated results reject boolean
  or numeric-string payloads instead of coercing them to schema version `1`.
  Optional artifact identity strings reject empty, non-string or
  whitespace-padded values on construction and reload. Guarantee metadata rejects
  coerced engines, source fingerprints, invalidation reasons and boolean versions
  before conversion to `PredictResult`, and stale predictor/data fingerprints,
  guarantee names or scopes no longer survive reload. Partial or internally
  incoherent guarantee-status payloads also fail closed.
- **Versioned lifecycle keyword/effect registry.**
  `nirs4all.pipeline.get_keyword_registry()` and `keyword_registry_json()` expose a
  deterministic, machine-readable description of current and planned tuning,
  training, calibration, prediction, and robustness syntax. Each entry records
  data reads, state changes, calibration invalidation, per-engine support,
  read-only aliases, documentation anchors, and Studio form hints. The pipeline
  reference renders the same registry and clearly distinguishes implemented,
  partial, and planned surfaces; the registry does not dispatch runtime behavior.
- **Workspace persistence for native tuning results.**
  Workspace schema v4 adds a `tuning_results` table. `WorkspaceStore` now
  persists verified `TuningResult` JSON with contract/result fingerprints, and
  `nirs4all.save_workspace_tuning_result(...)` /
  `nirs4all.load_workspace_tuning_result(...)` expose the round-trip for
  already-produced native tuning results. The internal
  `run_pipeline_objective_tuning(..., workspace_path=...)` seam also stores the
  completed tuning result before terminal refit, preserving the HPO evidence
  tape if refit fails later.
- **RunResult carrier for native tuning evidence.**
  `RunResult` now exposes `tuning_result`, `tuning_id`,
  `tuning_best_params`, and `tuning_best_value` when a native tuning trace is
  attached. The internal
  `project_objective_tuning_to_run_result()` seam projects an
  `ObjectiveTuningRunResult` into that carrier without fabricating prediction
  rows or test scores by default. When callers provide explicit `winner_x`,
  `winner_score`, and `winner_metric`, the same seam now projects a refit
  winner prediction row while treating the score as externally supplied
  evaluation evidence. Public `run(tuning=...)` compiler/wiring remains a
  separate gate.
- **Single-estimator and linear compiler seam for native tuning objectives.**
  `nirs4all.pipeline.dagml.pipeline_objective_compiler.compile_pipeline_objective()`
  now compiles the narrow supported shapes `estimator`, `[estimator]`,
  `{"model": estimator}`, `[{"model": estimator}]`, and linear
  transformer→estimator chains into the shared `PipelineObjective`. Broader
  pipeline syntax remains fail-closed until the public `run(tuning=...)`
  compiler is complete.
- **Prediction-based score extractor for native tuning objectives.**
  `make_prediction_score_extractor(metric, X_score, y_score)` now provides a
  standard `PipelineObjective` scorer that evaluates `predict(X_score)` against
  an explicit score cohort. This removes the need for ad hoc
  `training_result_` callbacks in the internal linear tuning lane while
  keeping data splitting explicit.
- **Linear compile→tune→refit→RunResult orchestration seam.**
  `run_single_estimator_tuning_to_run_result()` stitches the narrow compiler,
  optimizer adapters, optional workspace persistence, terminal refit, and
  `RunResult` projection into one internal path. It supports either one final
  estimator or a linear transformer→estimator chain. Winner prediction
  projection still requires explicit caller-provided
  `winner_score`/`winner_metric`; the helper does not convert development
  objective values into test scores.
- **Python API for the native single-estimator/linear tuning lane.**
  `nirs4all.tune_single_estimator(...)` exposes the implemented native
  single-estimator and linear transformer→estimator lane through the public
  Python package. It requires explicit scoring via `X_score`/`y_score` or a
  custom `score_extractor`, supports workspace tuning-result persistence, and
  returns a `RunResult` with attached tuning evidence.
- **First executable `run(tuning=...)` subset.**
  `nirs4all.run(..., engine="dag-ml", tuning={...})` now executes the same
  native lane for explicit array datasets when `tuning.score_data` is supplied.
  The subset now accepts both single-estimator pipelines and linear
  transformer→estimator chains with parameter paths such as `model.alpha` or
  `scale.factor`. String `transform` and final `model` mappings can now use
  explicit `sklearn.*` import paths with constructor `params`, e.g.
  `{"transform": "sklearn.preprocessing.StandardScaler", "params":
  {"with_mean": false}}` or `{"model": "sklearn.linear_model.Ridge", "params":
  {"fit_intercept": false}}`, while short aliases and arbitrary imports remain
  fail-closed. Constructor `params` must now use canonical string keys and
  TCV1-compatible JSON-native values before import or instantiation.
  Unsupported runtime keys, legacy-engine tuning, dataset loaders, splitters,
  branches and non-linear preprocessing graphs remain fail-closed.
- **Typed Python helpers for public native tuning syntax.**
  `NativeTuning`, `TuningScoreData`, `TuningWinner`, and `TuningCalibration`
  now provide an explicit object syntax for the same public `run(tuning=...)`
  subset. The helpers normalize to the existing mapping form, expose
  `NativeTuning.to_tuning_spec()` for the deterministic optimizer contract, and
  keep runtime-only blocks such as scoring, winner projection, calibration and
  workspace ids outside the tuning fingerprint.
- **Workspace persistence for public `run(tuning=...)` subset.**
  The single-estimator/linear array subset now honors `workspace_path` plus
  `tuning.workspace_tuning_id` / `tuning.workspace_metadata`, persisting the
  completed `TuningResult` before terminal refit and exposing the id on
  `RunResult.tuning_id`. Workspace tuning ids supplied through
  `NativeTuning.workspace_tuning_id`, the raw `tuning_id` alias,
  `save_workspace_tuning_result(..., tuning_id=...)` or the internal objective
  persistence path must now be canonical non-empty strings without surrounding
  whitespace or NULs; omit the id to request an auto-generated UUID.
  Workspace conformal ids and robustness ids now follow the same rule through
  `calibrate(..., workspace_conformal_id=...)`,
  `save_workspace_calibrated_result(..., conformal_id=...)`,
  `robustness(..., workspace_robustness_id=...)` and
  `save_workspace_robustness_report(..., robustness_id=...)`.
  Optional workspace link ids stored beside tuning, conformal and robustness
  rows, including `run_id`, `pipeline_id`, `chain_id`, source `prediction_id`
  and source `conformal_id`, are now validated with the same strict boundary
  instead of being inserted after host-side coercion.
- **Idempotent resume for public `run(tuning=...)` subset.**
  `run(..., engine="dag-ml", tuning={..., "resume": True})` can now reload a
  completed workspace `TuningResult` for the single-estimator/linear array subset when
  `workspace_path` and `tuning.workspace_tuning_id` are supplied. The resume bit
  is treated as operational metadata for contract matching, optimizer-driving is
  skipped, and the call performs only terminal refit/projection. Broader
  pipeline shapes remain fail-closed; n4m optimizer-state resume in the shared
  objective seam is now covered by the N4MOPT checkpoint path below.
- **Optuna storage resume reconstructs compact trial diagnostics.**
  Trials already present in a storage-backed Optuna study no longer fall back to
  metric/direction-only diagnostics when `resume=True`: completed rows use
  `score_extractor="optuna_storage"`, failed rows use
  `score_extractor="failed"` and pruned rows use `score_extractor="pruned"` in
  `TuningResult.trials` and summary artifacts. Resume also fails closed before
  optimizer execution if existing materialized trial params do not match the
  current `tuning.space` keys, preventing a persisted study from mixing
  incompatible search spaces under the same `study_name`/`storage`. Existing
  categorical values must also remain present in the current choices for their
  key, so removed or renamed choices fail closed before optimizer execution.
  Existing numeric values must also remain inside the current range for their
  key, so narrowed ranges that exclude stored trials fail closed before optimizer
  execution. If the current numeric space declares a `step`, stored values must
  also lie on that grid. During storage-backed resume, `n_trials` is now treated
  as the target total trial count instead of an unconditional number of
  additional trials. New storage-backed studies persist nirs4all
  `study.user_attrs` for format, schema version, optimizer contract fingerprint
  and search-space fingerprint; non-empty studies missing those attrs or
  carrying mismatched fingerprints fail closed during resume.
  Restored Optuna `COMPLETE` rows must carry a finite numeric value; missing or
  non-finite storage values fail closed instead of becoming completed trials
  with no usable objective value.
  Restored non-`COMPLETE` rows must not carry a final storage value; failed,
  pruned or in-flight rows with final values are rejected as corrupted optimizer
  history.
  Restored `RUNNING` rows fail closed during resume because interrupted active
  trials cannot be safely recovered into a terminal HPO tape.
  Restored terminal Optuna rows must keep exactly the current `tuning.space`
  parameter keys when the search space is non-empty; rows whose stored parameter
  table was removed fail closed instead of becoming completed trials with empty
  public params.
  Restored queued Optuna `WAITING` rows that already carry materialized params
  or `fixed_params` must also satisfy the current `tuning.space`; incompatible
  values are rejected before Optuna can consume them.
  Restored Optuna trial numbers must be canonical unique integers; corrupt
  storage that yields non-integer or duplicate trial numbers fails closed before
  the HPO tape is projected.
- **n4m optimizer checkpoints are resumable in the shared objective seam.**
  The n4m `PipelineObjective` adapter now accepts
  `storage="file:///absolute/checkpoint-dir"` plus a filename-safe `study_name`
  and writes a JSON manifest containing native N4MOPT checkpoint bytes after
  each terminal trial. `resume=True` reloads only a matching optimizer contract
  and treats `n_trials` as the target total trial count, so a checkpoint with one
  completed trial and `n_trials=2` runs one remaining trial. Restored trial rows
  are decoded back to public `TrialResult.params` and ordered canonically by
  numeric trial id; duplicate restored trial ids or ids that are not canonical
  integers fail closed before `TuningResult` construction. Named categorical
  `options` whose optimizer labels differ from their JSON-native public values
  are decoded before publication.
  Restored checkpoint row params must still match the current `tuning.space` keys
  and value domains: edited or incompatible checkpoint keys, categorical choices,
  numeric ranges or numeric steps fail closed before they can contribute
  optimizer history. Restored `COMPLETE` rows must carry a finite numeric score;
  missing, boolean or non-finite scores fail closed instead of becoming complete
  trials with no usable value.
  Restored failed, pruned and cancelled rows keep compact
  `score_extractor="failed"`, `score_extractor="pruned"` or
  `score_extractor="cancelled"` diagnostics for summary cards. Non-terminal
  checkpoint rows such as `RUNNING` fail closed during resume.
  Restored n4m non-`COMPLETE` rows must not carry a final score; failed, pruned
  or cancelled checkpoint rows with scores are rejected as corrupted optimizer
  history.
- **Integrated `run(tuning=...)` + conformal calibration subset.**
  `run(..., engine="dag-ml", tuning={..., "winner": {...}, "calibration": {...}})`
  now chains the projected winner entry into the replayed-array conformal
  surface and returns `TunedSingleEstimatorConformalResult(run, calibrated)`.
  The calibration payload cannot override `calibration_data`; it is derived from
  the winner projection to preserve the explicit tune→refit→calibrate order.
  When `workspace_path` is supplied to `run()`, the same workspace is used by
  default for `tuning.calibration`, so `workspace_conformal_id` can persist the
  calibrated result beside the tuning trace.
- **Explicit scoring identities for native tuning.**
  `tune_single_estimator()` and the public `run(tuning=...)` subset now accept
  row-aligned `score_sample_ids`/`score_groups`/`score_metadata` or equivalent
  `tuning.score_data` keys. Compatible estimators receive those values in
  `predict()` during objective scoring, while sklearn-style estimators without
  those predict kwargs continue to score normally. Winner ids also accept
  `prediction_sample_ids` and `physical_sample_ids` aliases. Tuple datasets in
  the public `run(tuning=...)` subset can now carry fit identities as
  `(X, y, sample_ids, groups, metadata)`, with row-alignment checks and explicit
  rejection of longer ambiguous tuples. `tuning.score_data` now supports the
  analogous tuple form `(X_score, y_score, sample_ids, groups, metadata)` for
  explicit scoring cohorts.
- **Explicit `SpectroDataset` fit-cohort extraction for native tuning.**
  The public `run(tuning=...)` subset now accepts
  `dataset={"dataset": spectro_dataset, "selector": {...}, ...}`. The selector
  is mandatory, bare `SpectroDataset` inputs remain fail-closed, and optional
  `sample_id_column`, `group_column`, and `metadata_columns` transport fit
  identities into the native tuning lane. `tuning.score_data` now supports the
  same explicit `SpectroDataset` mapping form for objective scoring cohorts.
- **DatasetConfigs/path-backed native tuning cohorts.**
  The explicit dataset-backed mapping used by `run(tuning=...)`, `score_data`,
  and `winner` now accepts `DatasetConfigs`, dataset config mappings, and
  config/path strings resolved by existing nirs4all loaders. The source must
  resolve to exactly one dataset and still requires a selector, so fit, scoring,
  and winner cohorts remain explicit.
- **Explicit `SpectroDataset` winner projection for native tuning.**
  `tuning.winner` can now select a `SpectroDataset` cohort with
  `{"dataset": spectro_dataset, "selector": {...}, ...}`. The selected rows
  supply `winner_x`, `winner_y_true`, physical sample ids and optional metadata
  for `RunResult.best` and downstream conformal calibration, while mixed
  `dataset` + explicit `X/y_true` payloads remain fail-closed. Single-target
  `(n, 1)` `SpectroDataset` targets are normalized to one-dimensional
  `winner_y_true` before the conformal contract is applied.
- **Executable example for native tuning + conformal calibration.**
  `examples/user/04_models/U09_native_tuning_conformal.py` demonstrates
  `run(tuning, calibration=...)` over a linear transformer→estimator chain, conformal
  calibration, workspace reload of both artifacts, and
  `predict_calibrated()` on new point predictions. A dedicated integration
  smoke test keeps the documented flow executable. The user guide now includes a
  dedicated native tuning + conformal page with Python syntax, keyword effects,
  workspace CLI commands, and current fail-closed limits.
- **CLI inspection for native tuning and conformal workspace artifacts.**
  `nirs4all workspace tuning list/show` and
  `nirs4all workspace conformal list/show/predict` expose read-only inspection
  and application of persisted `TuningResult` and `CalibratedRunResult`
  records, with `--json` output for CI/release audits. The `show` commands
  reload and verify the stored typed result before printing it; `conformal
  predict` applies the stored calibrator to explicit point predictions without
  modifying the workspace.
- **Single-estimator tuning output can feed conformal calibration.**
  The `RunResult.best` prediction entry produced by
  `tune_single_estimator(..., winner_x=..., winner_y_true=...,
  winner_sample_ids=...)` now has a covered public test path into
  `nirs4all.calibrate(calibration_data=result.best, ...)`, producing conformal
  `PredictResult` intervals for replayed point predictions.
- **Integrated single-estimator tuning + conformal result.**
  `tune_single_estimator(..., calibration={...})` now runs the same conformal
  replayed-array calibration immediately after winner projection and returns a
  `TunedSingleEstimatorConformalResult` carrying both the tuning `RunResult` and
  calibrated prediction result.
- **Transition workspace compatibility and offline conversion guidance.** The Python
  library keeps the transition release compatible with both legacy workspaces and the
  V1 SQLite workspace format. Legacy DuckDB, legacy filesystem-run, and legacy
  prediction-array layouts are detected before use and warn with the concrete
  `nirs4all workspace convert <workspace> --output <workspace-v2> --verify` command
  for a non-mutating migration. Install `nirs4all[transition]` to bundle the
  `nirs4all-tools` converter plus DuckDB/Parquet readers; the original workspace is
  not overwritten by the offline converter. The legacy reader/converter path is scoped
  to the Python transition line and Studio. Other V1 language/package surfaces consume
  the new workspace format only, and a later release will remove the Python legacy
  compatibility layer after the transition window.
- **Transparent legacy fallback when the dag-ml backend is unavailable.** A new narrow
  `DagMlUnavailable` error is raised by a dag-ml-backend preflight when NEITHER mechanism
  is installed (no in-process `dag_ml._dag_ml` extension AND no `dag-ml-cli` binary).
  `run()` catches it (alongside the existing `DagMlUnsupported` / `NotImplementedError`
  coverage-boundary fallback) and re-runs on the legacy engine with a warning — so a
  wheel install missing the native backend degrades gracefully instead of crashing.
  GENUINE dag-ml runtime/operator errors still propagate untouched (never swallowed).

### 🐛 Fixed

- **`RunResult.best_rmse` / `best_r2` / `best_accuracy` now describe the SELECTED model.**
  These scalar shortcuts previously re-ranked predictions independently per metric
  (`get_best(metric="rmse"/"r2"/"accuracy")`), which ranks rows by their *validation*
  score. Under cross-validation the per-metric-best row is often a different CV *fold*
  model than the one selection chose, so the shortcuts could each report a **different
  model**: e.g. `best_r2` returned a ShuffleSplit fold's test R² (0.5426) instead of the
  selected model's (0.5499), and `best_accuracy` returned a non-selected fold's plain
  accuracy instead of the balanced-accuracy-selected model's. All three now read their
  metric from `best` (the selection-metric winner that `best_score` describes), so the
  scalar shortcuts are mutually consistent (for an rmse-selected single model,
  `best_rmse == best_score`). This is a bugfix, not a contract break — but **CV runs where
  a metric's validation-rank differs from the selection rank may now report different
  `best_rmse` / `best_r2` / `best_accuracy` values** (webapp dashboards reading these
  shortcuts may see changed numbers for such runs).

## [0.10.3] - Release metadata and 0.10.2 hardening - 2026-06-29

### Highlights

Patch release used for the CILS article archive. It preserves the 0.10.x public
API while bundling the post-0.10.2 hardening commits used by the manuscript
reproducibility replay.

### Added

- `nirs4all.data.selection.sampling` with random, stratified and k-means-based
  index-selection helpers for subset and preview workflows.

### Fixed

- Raised NumPy and scientific dependency floors to the versions required by the
  current PLS stack.
- Avoid retaining workspace stores at interpreter exit.
- Updated example plotting code for the supported Matplotlib colormap API.
- Synced release-facing documentation, conda metadata and version guardrails.

---

## [0.10.0] - Heterogeneous source repetitions - 2026-06-13

### 🎯 Highlights

Minor release for the experimental relation pipeline: nirs4all can now model
heterogeneous multisource repetitions where each physical sample has different
source-specific repetition counts, for example `MIR=2`, `RAMAN=3`, `NIRS=2`.
The legacy `repetition=` path remains unchanged; relation-aware workflows opt in
explicitly through `experimental_relation_pipeline`.

### ✨ Added

- **Data relation contracts**: `RepetitionSpec`, normalized relation tables,
  `RawMultiSourceDataset`, explicit representation plans, relation fingerprints,
  and replay manifests.
- **Representation materialization**: `per_source_aggregate`,
  `per_source_observation`, `sample_aggregate`, fixed/padded stack modes, and
  bounded cartesian materializations including train-only augmentation.
- **Pipeline integration**: `rep_fusion` as the explicit boundary from ragged
  source staging to model matrices, relation-aware guards for merge/stacking,
  sample-level reducers, fit-influence policy handling, bundle replay, and
  prediction/explainability lineage accessors.
- **Examples and docs**: heterogeneous relation YAML examples, mirrored sample
  configs, fixture CSVs, an RTD user guide page, and a step-by-step RTD tutorial.

### 🛡️ Changed

- Ambiguous heterogeneous multisource inputs that would previously be silently
  treated as positionally aligned are rejected with explicit relation errors.
- Relation-aware runs persist representation, reducer, missingness,
  fit-influence, and feature-lineage contracts so exported bundles can replay
  prediction materialization safely.

### 🧪 Tests

- Added unit, integration, regression, bundle, storage, API, and example-contract
  coverage for relation materialization, replay, guardrails, fit influence,
  prediction provenance, and docs/example contracts.
- Full validation before release: `7638 passed, 13 skipped`.

---


## [0.9.4] - Slim core (optional viz/explain extras) - 2026-06-11

### 🎯 Highlights

Dependency-only release — **no API change** (the 0.9.x contracts hold). The default
`pip install nirs4all` no longer pulls matplotlib, seaborn, shap or umap-learn: shap
and umap dragged numba + llvmlite (~170 MB) into every install, yet they are only
needed for optional features. `import nirs4all` and all core (non-plotting,
non-SHAP) workflows are unaffected — the heavy imports were already lazy.

### ♻️ Changed

- `matplotlib` moved to a new `viz` extra; `shap` moved to a new `explain` extra
  (which also carries matplotlib, since the SHAP analysis module renders with it).
  Both are included in `all` / `all-gpu`. Get plotting + SHAP with
  `pip install nirs4all[viz,explain]`.

### 🗑️ Removed

- `umap-learn` and `seaborn` were unused inside the package (0 functional imports)
  and are dropped from the core dependencies (no extra). `installation_test` no
  longer reports matplotlib / seaborn / shap as required dependencies.

---

## [0.9.3] - Studio Boundary APIs - 2026-06-06

### 🎯 Highlights

Additive release — **no breaking change** (the 0.9.x stable contracts hold). Eight public APIs
requested by the nirs4all-studio tech-debt closeout, so UIs stop re-implementing library
semantics in their HTTP layers.

### ✨ Added

- **Storage**: `WorkspaceStore.count_chain_summaries()` (COUNT with the same filters as
  `query_chain_summaries`) and `offset` + list-valued filters on `query_top_chains` /
  `query_top_aggregated_predictions` — SQL-side ranking pagination for result browsers.
- **Data**: `nirs4all.data.repetition_detection` — bio-sample/replicate grouping from metadata
  columns (`auto_detect_repetition_column`) or sample-id naming conventions
  (`detect_repetition_groups`).
- **Data**: `SpectroDataset.describe()` — one-call JSON-safe structural summary (sizes, sources,
  task type, signal types, metadata columns, target presence).
- **Analysis**: `nirs4all.pipeline.analysis.model_diagnostics` — bias-variance decomposition over
  repeated CV predictions, normalized robustness axes, learning-curve aggregation.
- **Analysis**: `nirs4all.pipeline.analysis.shape_inference` — pre-fit operator output-shape
  rules (`infer_output_shape`) + the dimension-bound parameter taxonomy, for editor-time shape
  propagation.
- **Analysis**: `nirs4all.pipeline.analysis.splitter_config` — recover the CV setup
  (`extract_splitter_config`) from a stored pipeline `expanded_config`, with the step parser's
  repr-skip rule.
- **Pipeline**: cooperative run cancellation — `should_stop: Callable[[], bool]` on
  `PipelineRunner`/`PipelineOrchestrator`/`nirs4all.run()`, polled at dataset/variant/refit
  boundaries; aborts with the new `RunCancelledError` and the store run is marked failed.

### 🧪 Tests

53 new tests across the eight APIs; full unit tier green.

## [0.9.2] - Storage Crash Safety, Lighter Imports & Debt Cleanup - 2026-06-05

### 🎯 Highlights

Internal-quality release — **no public-API change** (the 0.9.x stable contracts hold). The workspace storage layer is now crash-safe across its three stores (SQLite / Parquet arrays / artifacts), `import nirs4all` is dramatically lighter, and a large technical-debt campaign removed ~17k LOC of dead code with zero behavior regression.

### 🛡️ Storage crash safety (workspace)

- **Delete ordering**: SQLite metadata is deleted before array tombstones are written — a crash mid-delete can only orphan arrays (harmless, reclaimable), never leave live metadata pointing at removable arrays
- **Validated compaction**: tombstones are checked against live SQLite ids before physical removal; a stale tombstone (crash/rollback leftover) can never delete a live prediction's arrays and is dropped instead
- **Atomic flush**: `Predictions.flush()` writes its SQLite rows and Parquet batch inside one transaction (re-entrant `WorkspaceStore.transaction()`); a crash mid-flush rolls the metadata back instead of leaving rows without arrays
- **Inter-process lock**: ArrayStore mutations serialize on an advisory file lock (POSIX `flock` / Windows `msvcrt`), fixing silent last-writer-wins between concurrent processes writing the same workspace
- **Self-healing**: pending tombstones are reconciled on workspace open (gated, validated), and threshold-gated auto-compaction reclaims space after deletes
- `Predictions.clean_dead_links()` now physically reclaims orphaned arrays (previously it only counted them)

### ⚡ Performance

- `import nirs4all` no longer eagerly loads matplotlib, shap, numba or llvmlite — chart and explainability modules import lazily on first use (pinned by a regression test)

### 🐛 Fixes

- Generator: `{"_grid_": {...}, "model": ...}` dict steps were silently dropped — now expanded correctly
- `explain()` on loaded `.n4a` bundles: fixed `step_idx` KeyError
- AOM estimators: scikit-learn ≥ 1.9 compatibility (estimator-type detection via mixin MRO)
- `reset_registry()` clears the controller registry in place (stale-reference bug)
- Same-priority controllers route deterministically (tie-break by class name instead of import order)
- Conda recipe version drift fixed; the recipe now syncs from the single-sourced package version

### 🧹 Internal

- −17,245 LOC of dead/legacy code removed (64 files) with zero behavior regression
- ~30 god-methods decomposed across orchestrator, predictions, merge/branch controllers, executor and storage
- Contract snapshot tests freeze the public API signatures, SQLite schema DDL and `.n4a` manifest format
- SQLite schema version stamped via `PRAGMA user_version` with a forward-incompatibility guard
- Layering fixed: the data layer no longer imports the pipeline layer at runtime (store backend injected at import time)
- Version single-sourced from `nirs4all.__version__` (pyproject reads it dynamically)

### 🧪 Tests

- New crash-injection, validated-compaction, flush-transactionality, process-lock and lazy-import regression suites; full suite green (7171 passed / 0 failed)

---

## [0.9.1] - Pipeline Definition Ergonomics - 2026-04-17

### ✨ Improvements

- **`steps` key alias**: Pipeline definitions accept `{"steps": [...]}` in addition to `{"pipeline": [...]}` for dict-based configs (JSON/YAML and in-code)
- **Batch pipeline detection**: Refined detection so that a single pipeline starting with a nested list step (e.g. `[[...], {...}]`) is no longer misread as a batch of pipelines
- **README**: Clarified nirs4all offerings and documented the stable API contracts introduced in 0.9.0

### 🧪 Tests

- Added coverage for `steps`-key pipeline definitions and nested-list batch detection edge cases

---

## [0.9.0] - Webapp-Ready Release: Stable Signatures & Schemas - 2026-04-16

### 🎯 Highlights

This release marks the first version of `nirs4all` that is **ready for integration with the nirs4all webapp**. Public API signatures (`run`, `predict`, `explain`, `retrain`, `session`, `generate`), result objects (`RunResult`, `PredictResult`, `ExplainResult`), and the workspace storage schemas (`WorkspaceStore` SQLite tables, `ArrayStore` Parquet layout, run manifest structure, `.n4a` bundle format) are now considered stable contracts that the webapp backend can depend on without risk of breaking changes within the 0.9.x line.

### ✅ Stable Contracts for the Webapp

- **Public API signatures**: module-level entry points (`nirs4all.run/predict/explain/retrain/session/generate`) have finalized keyword arguments and return types
- **Result schemas**: `RunResult`, `PredictResult`, and `ExplainResult` expose a locked-in surface (`best_score`, `best_rmse`, `best_r2`, `top(n)`, `export()`) for frontend consumption
- **Workspace schema**: SQLite tables (runs, pipelines, chains, logs, artifacts, predictions metadata) and Parquet array layout are frozen for the 0.9.x series
- **Run manifest**: `dataset_info`, versioning fields, and run identifiers stable for dataset-compatibility checks
- **Bundle format**: `.n4a` export/load contract stable for prediction and retraining workflows

### 🔧 Chores

- Version bumped to 0.9.0 to signal webapp-readiness and schema stability

---

## [0.8.10] - Repetition Aggregation, Grouped Splitters & Storage Refinements - 2026-04-15

### ✨ Improvements

- **Repetition-aggregated predictions**: Added support for storing and retrieving repetition-aggregated prediction scores; database schema updated with aggregation score columns
- **Stable pipeline ordering**: Pipelines are now retained in creation order in `extract_winning_config` and `extract_per_model_configs` for deterministic results
- **Grouped splitters**: Enhanced grouped splitters with explicit error handling and support for tuple groups
- **SPXYGFold capability**: Marked group capability as optional to permit non-grouped usage
- **WorkspaceStore deletion API**: Renamed deletion methods for naming consistency and added prediction-deletion helpers
- **SklearnModelController**: Improved model instance handling and error reporting; added metric normalization

### 🐛 Bug Fixes

- **D04_parallel_branches**: Refactored dataset path handling for portability

### 🧪 Tests

- Extended group-splitting and execution test coverage
- Added tests for model instantiation, prediction deletion, and tuple-group handling

---

## [0.8.9] - Refit Enhancements & Branch Improvements - 2026-04-13

### ✨ Improvements

- **Refit pipeline metadata**: Best chain entries now include `pipeline_id` and `config_name` for better traceability
- **Refit candidate selection**: Improved candidate selection logic to ensure refit pipelines are excluded from CV ranking
- **Branch controller**: Enhanced branch controller with more robust handling and improved internals

### 🧪 Tests

- Added branch controller unit tests
- Added advanced refit, refit executor, and refit infrastructure tests
- Extended stacking refit and parallel execution test coverage

---

## [0.8.8] - PCR Model, score_scope Rename & Operator Refinements - 2026-04-09

### ✨ Improvements

- **New PCR model**: Added `PCR` (Principal Component Regression) under `nirs4all.operators.models.sklearn`, exposed from `operators.models`
- **score_scope rename**: Renamed `score_scope` value from `'final'` to `'refit'` across the codebase (API, executor, orchestrator, resolver, predictions, charts) for consistency with refit semantics; cv-scope value renamed from `'cv'` to `'folds'`
- **XOutlierFilter**: Improved PCA component selection and threshold computation using a chi-squared distribution for better statistical grounding
- **Splitters**: Input-shape-aware handling in splitters to support multi-dimensional feature arrays
- **Operator defaults**: Sensible defaults added for `HighLeverageFilter`, `ResampleTransformer`, `Resampler`, and `RangeDiscretizer` to reduce boilerplate
- **Task-type filtering**: New `PredictionAnalyzer` helper to filter datasets by `task_type`

### 🐛 Bug Fixes

- **PCR**: Fixed type hint in `PCR.fit`
- **Resampler**: Removed unused `crop_mask_` declaration

### 🧪 Tests

- Updated tests to reflect `score_scope='refit'` and cv-scope `'folds'` renames across prediction, ranking, scoring, and aggregation suites

### 🔧 Chores

- **pre-publish script**: Improved error handling during execution
- **Examples & docs**: Updated visualization and cross-validation examples and user guide to use the new `score_scope` values

---

## [0.8.7] - Score Scope Defaults, Aggregation Normalization & Ranking Enhancements - 2026-04-01

### ✨ Improvements

- **Default score_scope changed to 'mix'**: `Predictions.top()` and `Predictions.ranked_scores()` now default to `score_scope='mix'`, returning both refit and CV entries ranked together instead of only refit entries
- **Chart aggregate normalization**: Added `_normalize_aggregate()` to `BaseChart`, resolving `aggregate=True` to the actual repetition column name before passing to rendering methods
- **Prediction ranking and aggregation**: Enhanced prediction ranking with improved score scope handling, better partition-aware display, and more robust aggregation workflows

### 🧪 Tests

- **Score scope and ranking tests**: Updated tests to reflect new `score_scope='mix'` default, added explicit `score_scope='final'` where refit-only behavior is required
- **Aggregation integration tests**: Expanded end-to-end aggregation tests with explicit score scope parameters

### 🔧 Chores

- **Examples updated**: Refreshed visualization and aggregation examples to document score_scope options accurately

---

## [0.8.6] - Prediction Aggregation, Plot Display Lifecycle & Metadata Loading - 2026-04-01

### ✨ Improvements

- **Prediction aggregation defaults**: `PredictionAnalyzer` now infers repetition-based aggregation from prediction context, supports explicit repetition aggregation options, and recalculates display metrics against the effective partition arrays
- **Visualization rendering flow**: Added shared figure lifecycle helpers for display/show/close behavior, opt-in chart saving, and cleaner handling of raw vs aggregated chart variants
- **Task-family-aware charts**: Visualization views now skip incompatible regression/classification chart families more cleanly while preserving task-type filtering

### 🧪 Tests

- **Aggregation and plotting coverage**: Added integration and unit tests for prediction ranking, aggregation analysis, plot visibility flags, and dual raw/aggregated chart outputs
- **SHAP plotting coverage**: Added integration coverage for SHAP visualization behavior

### 🐛 Bug Fixes

- **Metadata NA handling**: `load_XY()` now forces metadata inputs to use `na_policy='ignore'`, preserving metadata rows even when metadata columns contain missing values

### 🔧 Chores

- **Docs and examples**: Refreshed cross-validation docs, example scripts, and developer notes to match the new aggregation and plotting behavior

---

## [0.8.5] - Task Type Filtering, Visualization Improvements & Documentation - 2026-03-31

### ✨ Improvements

- **Task type filtering in visualization charts**: Added `task_type` parameter to Candlestick, Heatmap, Histogram, and TopK Comparison charts with auto-separation of mixed task types
- **Model training parameters**: Updated model training parameters, enhanced prediction handling, and improved error messaging in visualizations
- **Documentation overhaul**: Added comprehensive reference docs for pipeline keywords, models, transforms, splitters, filters, and augmentations

### 🧪 Tests

- **Task type matching tests**: Added tests for task-type matching and filtering in predictions for regression and classification tasks
- **Chart task type tests**: Added tests for task-type auto-separation in visualization charts

### 🐛 Bug Fixes

- **Prediction handling**: Enhanced prediction retrieval and error messaging in visualization components

### 🔧 Chores

- **Docs**: Added new concept guides (augmentation, branching, cross-validation, datasets, generators, pipelines, predictions & deployment)

---

## [0.8.4] - Predictions Metadata, Parquet Schema Alignment & SQLite Migration - 2026-03-27

### ✨ Improvements

- **Per-sample metadata in predictions**: Implement per-sample metadata storage and retrieval in predictions
- **Predictions grouped queries**: Refined prediction retrieval logic for grouped queries; updated chart examples
- **Parquet schema alignment**: Added schema alignment for Parquet tables to handle evolving column sets
- **Augmentation docs**: Updated augmentation module documentation

### 🧪 Tests

- **SQLite migration tests**: Refactored test suite for SQLite migration

### 🐛 Bug Fixes

- **Code quality**: Fixed ruff and mypy errors

### 🔧 Chores

- **CI**: Bump `codecov/codecov-action` from 5 to 6
- **Docs**: Archived docs

---

## [0.8.3] - API Documentation, Bug Fixes & CI Updates - 2026-03-25

### ✨ Improvements

- **Module-level API documentation**: Updated documentation and examples to use the module-level API (`nirs4all.run()`, `nirs4all.predict()`, etc.); added shorthand aliases for transforms

### 🐛 Bug Fixes

- **Repetitions & scores sorting**: Fixed repetition handling and scores sorting issues
- **Confusion matrix balanced_accuracy**: Resolved balanced_accuracy mismatch and regression model filtering in confusion matrix (closes #31, #32)
- **Folder parser file ordering**: Sort files in folder parser for consistent processing order across platforms

### 🔧 Chores

- **CI**: Bump `docker/metadata-action` from 5 to 6; bump GitHub Actions to latest versions

---

## [0.8.2] - DuckDB Stability, Branch/Merge Fixes & Parallel Tests - 2026-02-25

### 🐛 Bug Fixes

#### DuckDB / Storage Stability
- **`WorkspaceStore` atexit handler**: Registers an `atexit` callback to close the DuckDB connection on interpreter shutdown, preventing segfaults when the process exits without an explicit `close()` call
- **`_jittered_delay` return type**: Fixed return type to always be `float`, removing potential `int` return that caused type errors in retry logic

#### Branch / Merge Pipeline Correctness
- **Preserve preprocessing chains after merge** (closes #24): `MergeController.add_merged_features()` was calling `reset_features()` which wiped per-branch preprocessing history from the run summary `Preprocessing` column. Fix builds composite processing names from branch contexts before the reset; added 3 helper methods to `MergeController`, fixed 6 call sites, and removed dead code in `dataset.py`
- **Runner ownership in `RunResult`**: `RunResult` now tracks `WorkspaceStore` ownership so it is closed deterministically when the result object is garbage-collected or used in a `with` block, preventing premature closure when the store is shared across `retrain()` and `session` workflows

### 🧪 Tests

#### Merge Auto-Detection
- **Integration tests** (`test_merge_auto_detect.py`): Validate auto-detect merge strategy selection for duplication and separation branches across common configurations
- **Unit tests** (`test_merge_auto_detect.py`): Cover `MergeController` auto-detect logic for all merge modes

#### Separation Branch Generators
- **Integration tests** (`test_separation_branch_generators.py`): Extensive coverage for generator keywords (`_or_`, `_range_`, `_grid_`, etc.) inside separation branches, including parallelisation scenarios

#### Preprocessing Chain Regression
- **Unit tests** (`test_merge_preprocessing_chain.py`): 13 regression tests ensuring per-branch preprocessing chains survive merge operations

#### Pipeline Config Expansion
- **Unit tests** (`test_pipeline_config_separation_expansion.py`): Verify correct generator expansion for separation branch configs
- **Unit tests** (`test_pipeline_config_separation_gen_keys.py`): Confirm accurate detection of generator usage in separation branches

### 🔧 Improvements

#### Pre-publish Script
- **Parallel example categories** (`pre-publish.sh`): Example categories now run in parallel with per-category log files, reducing total pre-publish validation time

---

## [0.8.1] - DuckDB Resilience, PCA Projections & Sampling - 2026-02-25

### ✨ New Features

#### PCA Projection Utility
- **`nirs4all.analysis.projections`**: New PCA projection module for quick dataset visualization and dimensionality analysis

#### Sampling Functions
- **`nirs4all.data.selection.sampling`**: New sampling utilities for dataset subsampling and selection strategies

#### AOM-PLS Benchmarking Framework
- **`bench/AOM/`**: Next-generation benchmarking framework for AOM-PLS models including DARTS-PLS, MoE-PLS, zero-shot router, and enhanced AOM variants

### 🔧 Improvements

#### DuckDB Concurrency & Crash Recovery
- **Context manager support**: `WorkspaceStore` now supports `with` statement for deterministic resource management
- **Safety net in `__del__`**: Ensures connections are closed if not explicitly done by the user
- **Per-operation retry with jitter**: Replaced irreversible degraded mode with per-operation retries and jitter to avoid thundering herd
- **Transaction batching**: New transaction management to batch multiple writes, reducing lock contention
- **Orphaned file cleanup**: `ArrayStore` cleans up orphaned temporary files during initialization
- **Explicit connection closing**: CLI commands and `api/run.py` now close `WorkspaceStore` after use

#### CI/CD
- **Version consistency check**: CI workflow verifies version consistency across `pyproject.toml`, `__init__.py`, and `conda-forge/meta.yaml`
- **Conda-forge update notification**: CI alerts when conda-forge recipe needs updating

### 🧪 Testing

- **PCA projection tests**: Unit tests for the new projections module
- **Sampling function tests**: Unit tests for the new sampling utilities

---

## [0.8.0] - AOM*-PLS, Parquet Storage & Scoring Overhaul - 2026-02-20

### ✨ New Features

#### Docker Support
- **Multi-stage Dockerfile**: Lightweight production image based on `python:3.11-slim` with build-stage compilation
- **`.dockerignore`**: Optimized Docker build context

#### Conda-Forge Distribution
- **`conda-forge/meta.yaml`**: Recipe for nirs4all package on conda-forge
- **Staged recipes**: Conda-forge recipes for missing dependencies (`cvmatrix`, `ikpls`, `pyopls`, `trendfitter`)
- **Conda-forge setup guide**: Internal documentation for the submission process

### 🔧 Improvements

#### Dependency Management
- **Twinning reimplemented natively**: Replaced `twinning` external dependency with a pure NumPy implementation of the data twinning algorithm (Vakayil & Joseph 2022) in `SPlitSplitter`
- **PLS variants promoted to core dependencies**: `ikpls`, `pyopls`, `trendfitter` moved from optional `[pls]` extra to core dependencies
- **Removed `twinning` from all requirements files**

#### CI/CD
- **macOS compatibility**: Improved CI workflows to avoid deadlocks, handle test coverage, and skip problematic tests on macOS
- **Pre-publish validation**: Enhanced pre-publish script with macOS support and timeout handling
- **Reusable disk cleanup action**: Replaced manual disk cleanup with a reusable GitHub Action


#### AOM-PLS & POP-PLS Models
- **`AOMPLSRegressor`**: Adaptive Operator-selection Meta-PLS — automatic preprocessing selection using a bank of linear operators with sparsemax gating during PLS component extraction
- **`AOMPLSClassifier`**: Classification variant with probability calibration and sklearn compatibility
- **`POPPLSRegressor` / `POPPLSClassifier`**: Penalized Orthogonal Projections PLS for operator selection and validation
- **Linear operator bank**: Identity, Savitzky-Golay filter, Detrend projection, Composed operator, and additional SG/composed variants
- **PyTorch backend**: Optional Torch-based AOM-PLS implementation for GPU acceleration

#### New Preprocessing Operators
- **`NorrisWilliams`**: Norris-Williams smoothing and derivative transform with both function and transformer APIs
- **`WaveletDenoise`**: Wavelet-based denoising transform with configurable wavelet families and threshold modes
- **Orthogonalization transforms**: New orthogonalization module for spectral data

#### Prediction Storage Migration (DuckDB → Parquet)
- **Hybrid DuckDB + Parquet storage**: Structured metadata stays in DuckDB, dense prediction arrays moved to Parquet sidecar files
- **`ArrayStore`**: New module for saving, loading, and verifying prediction arrays in Parquet format
- **Migration utilities**: Automatic migration of legacy DuckDB prediction arrays to Parquet with data integrity verification
- **Tombstone-aware deletion**: Proper handling of soft-deleted data in the new storage format

#### SPXYFold Splitter
- **`SPXYFold`**: K-Fold cross-validation splitter based on the SPXY (Sample set Partitioning based on joint X-Y distances) algorithm

#### Chain Summary System
- **New `v_chain_summary` view**: Aggregate chain summaries with model metadata, CV scores, and final scores
- **Enhanced chain query methods**: `query_chain_summaries`, `top_chains` with list-based filter parameters supporting SQL `IN` clauses
- **Backfill logic**: Populate chain summary columns from existing predictions
- **`task_type` filter**: Additional filter parameter for query specificity

#### Project Management
- **Project CRUD operations**: Create, list, update, and delete projects in the database
- **`projects` table**: New SQL schema for project metadata
- **Run-project association**: Link runs to projects for experiment organization

#### Pipeline Metrics & Reporting
- **Ensemble test scores**: New `Ens_Test` and `W_Ens_Test` metrics for ensemble evaluation
- **Mean fold validation**: New `MF_Val` metric for cross-validation reporting
- **RMSEP-based sorting**: Tab report manager now sorts by RMSEP instead of RMSECV
- **Score scope filtering**: `build_aggregated_query` and `build_top_aggregated_query` support filtering by scope (`CV`, `all`, `final`)

#### Stacking & Model Helpers
- **`stack_params` helper**: New utility for fine-tuning stacking model parameters with enhanced model parameter handling

#### Data Loading
- **Gzip and tar file support**: Enhanced CSV and folder parsing for compressed file formats

### 🔧 Improvements

#### Pipeline Execution
- **Memory cleanup between datasets**: `PipelineOrchestrator.cleanup()` releases memory between dataset iterations
- **Graceful dataset failure handling**: Orchestrator logs errors and cleans up resources on dataset failures
- **Parallel execution**: Improved joblib/loky backend compatibility by removing unpicklable objects
- **Deferred artifact persistence**: `ArtifactRegistry` supports deferred persistence and enhanced generator keyword handling
- **Random state propagation**: Consistent random state throughout pipeline for reproducibility

#### Refit System
- **Multi-criteria refit**: Enhanced handling with independent model selection across multiple criteria and improved error diagnostics
- **`selection_score` rename**: `best_score` → `selection_score` in `LazyModelRefitResult` for clarity
- **Per-model config extraction**: New `extract_per_model_configs` function for extracting best configurations per model class
- **Competing branches refit**: `execute_competing_branches_refit` refits all branches with average CV scores in predictions
- **List-based refit parameters**: Support for list-based refit parameters with aggregation reporting

#### DuckDB Resilience
- **Degraded mode**: Automatic fallback when DuckDB encounters persistent lock failures
- **Retry logic**: Enhanced error handling and retry for DuckDB lock conflicts in pipeline execution and storage

#### Scoring & Validation
- **Scoring computation invariants**: Correct RMSECV calculation from pooled OOF predictions, proper None score preservation
- **NIRS/ML naming conventions**: Consistent metric naming conventions across contexts
- **`v_aggregated_predictions_all` view**: Supports querying both CV and refit entries

#### Configuration & Generators
- **Generator count limits**: `log_range_strategy`, `or_strategy`, `range_strategy`, `sample_strategy`, and `zip_strategy` now allow no limit when count ≤ 0
- **`BestChainEntry` dataclass**: Track best preprocessing chain per model during cross-validation for more efficient refit

#### Code Quality
- **2000+ mypy errors fixed**: Comprehensive type-checking cleanup across the codebase
- **Type aliases**: Added type aliases for clarity in multiple modules
- **Ruff and mypy CI integration**: Enhanced CI with ruff and mypy checks
- **Polars version**: Bumped minimum `polars` requirement to 1.0.0

### 📚 Documentation

- **Workspace architecture docs**: Updated for hybrid DuckDB + Parquet storage system
- **Operator catalog**: New spectral augmentation and advanced PLS variants documented
- **Prediction lifecycle**: Clarified scalar scores (DuckDB) vs. arrays (Parquet) storage
- **Core audit**: Pre-webapp core audit notes

### 🧪 Testing

- **AOM-PLS test suite**: Regressor, classifier, operator adjoint identity, sparsemax, sklearn compatibility, custom operator banks, Torch backend parity
- **POP-PLS test suite**: Regressor and classifier, operator selection and validation
- **New operator tests**: NorrisWilliams, FiniteDifference, WaveletProjection, FFTBandpass, wavelet denoising
- **Parquet storage tests**: ArrayStore save/load/integrity, migration from DuckDB, tombstone handling
- **Workspace store tests**: Chain replay, chain summaries, bulk update, API inventory
- **Scoring invariant tests**: RMSECV pooling, None preservation, metric naming, config deduplication
- **OptunaManager tests**: Aggregation (BUG-2), grid search (ISSUE-17), config validation, refit skip (BUG-4), single-path holdout (BUG-3), train_params sampling (ISSUE-4)
- **Parallel execution tests**: No pickling errors, result consistency between parallel and sequential runs
- **Refit tests**: Lazy refit, model selector, advanced refit, warm start, stacking refit, infrastructure
- **Prediction analyzer tests**: Comprehensive visualization and analysis coverage
- **Step cache tests**: Correctness, copy-on-write, cacheability
- **Classifier sklearn wrapper tests**: New comprehensive test suite

### 🐛 Bug Fixes

- **OptunaManager `_aggregate_scores`**: Fixed incorrect aggregation behavior (BUG-2 regression)
- **Grid search suitability**: Fixed `_is_grid_search_suitable` and `_create_grid_search_space` (ISSUE-17)
- **Single-path optimization**: Now uses holdout split to prevent overfitting (BUG-3 regression)
- **Refit phase finetuning**: Refit phase correctly skips finetuning; `finetune_params` stripped from steps (BUG-4 regression)
- **NaN checks in `RunResult`**: Refactored NaN validation and error handling
- **Polars DataFrame inference**: Set `infer_schema_length` to None for prediction DataFrames

### 🗑️ Removed

- **`csv_loader.py`**: Removed deprecated CSV loader
- **`lazy_loader.py`**: Removed deprecated lazy loading module
- **`io.py`** (data): Removed deprecated data I/O module
- **`legacy_parser.py`**: Removed deprecated legacy parser
- **Prediction component modules**: Removed `aggregator.py`, `array_registry.py`, `indexer.py`, `query.py`, `ranker.py`, `schemas.py`, `serializer.py`, `storage.py` (replaced by Parquet-based storage)
- **Storage I/O modules**: Removed `io.py`, `io_exporter.py`, `io_resolver.py`, `io_writer.py`, `manifest_manager.py` (replaced by new workspace store)
- **`reproducibility.py`**: Removed deprecated utilities; functionality integrated into runner and orchestrator
- **`branch_diagram.py`**: Removed deprecated visualization module
- **`library_manager.py`**: Removed deprecated workspace library manager
- **CI quick mode**: Removed quick mode from example verification workflows; all examples now execute fully

---

## [0.7.1] - Caching, Workspace Store & Refit Improvements - 2026-02-08

### ✨ New Features

#### Copy-on-Write Caching Mechanism
- **Step-level cache** (`StepCache`): Cache and restore dataset state between pipeline steps to avoid redundant recomputation
- **Copy-on-write `ArrayStorage`**: Block-based shared memory with automatic detach-on-write for efficient dataset cloning
- **Cache configuration** (`CacheConfig`): New centralized cache configuration dataclass
- **Memory estimation utilities** (`nirs4all.utils.memory`): Accurate byte-level memory estimation for datasets and cache entries

#### Pipeline Refit System
- **`RefitExecutor`**: Full refit pipeline for retraining best models on the entire dataset
- **`ModelSelector`**: Select best model configurations from completed runs
- **`ConfigExtractor`**: Extract pipeline configurations for refit execution
- **`StackingRefitExecutor`**: Specialized refit support for stacking/meta-model pipelines
- **`RefitParams`**: Configuration dataclass for refit behavior

#### Pipeline Topology Analysis
- **New `nirs4all.pipeline.analysis.topology` module**: Analyze pipeline structure including stacking detection, branch separation, and feature merge identification
- **Pattern detection**: Identify stacking, separation branches, and feature merges in pipeline definitions

#### WorkspaceStore Enhancements
- **Thread-safe database access**: Concurrent DuckDB access with proper locking
- **Aggregated predictions view** (`v_aggregated_predictions`): New database view aggregating prediction metrics across folds
- **Enhanced query system**: New queries for aggregated predictions, chain predictions, and top aggregated predictions with metric-aware ranking
- **Prediction array retrieval**: Direct access to prediction arrays from the store
- **Artifact query service** (`QueryService`): Centralized artifact querying with filtering and sorting

#### Hashing Utilities
- **`nirs4all.utils.hashing`**: New module for deterministic data content hashing
- **`SpectroDataset.content_hash()`**: Compute content-based hash for dataset change detection without unnecessary materialization

### 🔧 Improvements

#### Pipeline Execution
- **Enhanced `PipelineExecutor`**: Improved artifact management and prediction safety
- **Enhanced `PipelineOrchestrator`**: Explicit workspace path requirements, improved run tracking
- **Prediction resolver**: Deterministic resolution modes for consistent prediction handling

#### Model Training & Validation
- **Enhanced model training logic**: Improved validation score calculation and dataset handling
- **Refined splitter functionality**: Better cross-validation splitting behavior
- **Controller improvements**: Enhanced `TransformerMixinController` with extended step cache support

#### Storage & Artifacts
- **Enhanced `ArtifactRegistry`**: Content verification against full hashes
- **Enhanced `ArtifactLoader`**: Improved step index handling and artifact loading behavior
- **Schema improvements**: Updated store schema with new views and idempotent creation

#### CI/CD
- **Parallel pytest execution**: Enhanced CI workflows with parallel test execution support
- **Parallel CI example runner**: Job control and improved output validation in `run_ci_examples.sh`
- **Optimized example parameters**: Reduced computational load in CI examples for faster execution

### 📚 Documentation

- **Cache optimization guide**: New section in pipelines user guide
- **Session API documentation**: Detailed guide for stateful workflows
- **SpectroDataset cache investigation**: Comprehensive analysis of caching mechanisms and memory management strategy
- **Technical debt review**: Prioritized debt analysis for workspace/predictions/artifacts
- **Enhanced user guides**: Added related examples across preprocessing, visualization, merging, multi-source, and stacking docs

### 🧪 Testing

- **Pipeline executor regression tests**: Execution and prediction flushing coverage
- **Pipeline orchestrator tests**: Explicit workspace path requirement enforcement
- **Pipeline topology analysis tests**: Stacking, separation branches, and feature merge patterns
- **Execution phase and hashing tests**: Deterministic behavior verification
- **OOF prediction accumulation tests**: Correct averaging across validation folds
- **WorkspaceStore tests**: Chain replay, prediction upsert, artifact registration, method signature validation
- **Prediction resolver tests**: Determinism and resolution mode coverage
- **Step cache tests**: Cached state restoration, data integrity, statistics accuracy
- **Content hash tests**: Hash consistency on mutations
- **Memory estimation tests**: Accurate byte calculations for datasets and cache entries
- **Refit and run entity tests**: Transition validation and metric comparison

### 🗑️ Removed

- **`csv_loader.py`**: Removed deprecated CSV loader
- **`lazy_loader.py`**: Removed deprecated lazy loading module
- **`io.py`**: Removed deprecated data I/O module
- **Generator constraint/strategy files**: Removed unused generator constraint and strategy registry modules

---

## [0.7.0] - Major Architecture & Operator Overhaul - 2026-02-05

> **⚠️ Documentation Notice:** Due to the extensive scope of this release, some documentation may be temporarily incomplete or out of sync. Updates are in progress.

### ⚠ BREAKING CHANGES

#### Synthesis Module Relocated
- **`nirs4all.data.synthetic`** → **`nirs4all.synthesis`** — update all imports
- Generator now delegates to operators for path length, instrumental broadening, and noise effects

#### Augmenter Base Class Removed
- **`Augmenter` base class deleted** (`abc_augmenter.py` removed entirely)
- **`IdentityAugmenter` deleted** — remove from all configs
- All augmentation operators now inherit from `TransformerMixin + BaseEstimator` or `SpectraTransformerMixin`
- **Migration**: Replace `Augmenter` subclasses with `TransformerMixin, BaseEstimator` or `SpectraTransformerMixin`

#### Operator API Changes
- **`apply_on` parameter removed** from all augmentation operators — replaced by step-level `variation_scope`
- **`copy` parameter removed** from all augmentation operators
- **`lambda_axis` parameter removed** from all augmentation operators — wavelengths are now auto-injected by the controller
- **`augment()` method removed** — use standard `transform()` instead
- **`transform_with_wavelengths()` removed** — use `transform(X, wavelengths=wl)` or let the controller handle it

#### SpectraTransformerMixin Simplified
- `transform_with_wavelengths(X, wavelengths)` replaced by `_transform_impl(X, wavelengths)` (internal abstract method)
- Public API is now standard `transform(X, **kwargs)` — wavelengths passed via kwargs
- `_requires_wavelengths` attribute: `True`, `False`, or `"optional"`
- `_validate_wavelengths()` helper for wavelength validation

### ✨ New Features

#### NA Handling System
- **Centralized NA policy**: New `apply_na_policy` utility for consistent NA handling across all loaders
- **NA policies**: `remove_sample`, `remove_feature`, `replace`, `ignore`, `abort`
- **`NAFillConfig`**: Configurable NA replacement strategies (mean, median, constant, interpolate)
- **Enhanced error reporting**: Detailed messages for NA detection including affected rows/columns
- **Loader support**: MatlabLoader, NumpyLoader, ParquetLoader all support new NA policies

#### Controller-Managed Variation (`variation_scope`)
- New `variation_scope` parameter at the `sample_augmentation` step level: `"sample"` (default), `"batch"`
- Per-transformer override via dict spec: `{"transformer": ..., "variation_scope": "batch"}`
- Hybrid performance model: operators with `_supports_variation_scope = True` handle variation internally; others get per-sample cloning from controller

#### SpectraTransformerMixin Foundation
- **New `SpectraTransformerMixin` base class**: Enables wavelength-aware transformations with full sklearn compatibility
- **Automatic wavelength passing**: Controller detects operators that require wavelengths and extracts them from the dataset
- **`_requires_wavelengths` class flag**: Operators can declare mandatory or optional wavelength requirements

#### New Augmentation Operators
- **`PathLengthAugmenter`**: Multiplicative path length variation
- **`BatchEffectAugmenter`**: Wavelength-dependent batch effects (offset + gain)
- **`InstrumentalBroadeningAugmenter`**: Gaussian convolution broadening (FWHM-based)
- **`HeteroscedasticNoiseAugmenter`**: Signal-dependent noise
- **`DeadBandAugmenter`**: Random dead band (non-responsive region) simulation

#### Environmental Effect Operators (`nirs4all.operators.augmentation.environmental`)
- **`TemperatureAugmenter`**: Simulates temperature-induced spectral changes with region-specific effects for O-H, N-H, and C-H bands
- **`MoistureAugmenter`**: Simulates moisture/water activity effects on spectra

#### Scattering Effect Operators (`nirs4all.operators.augmentation.scattering`)
- **`ParticleSizeAugmenter`**: Simulates particle size effects on light scattering
- **`EMSCDistortionAugmenter`**: Applies EMSC-style scatter distortions

#### Spectral Components Expansion
- **111 predefined spectral components** (expanded from 48): Added petroleum/hydrocarbon components (crude oil, diesel, gasoline, kerosene, PAH)
- Enhanced metadata with synonyms and tags for better categorization

#### Run Management System
- **New `Run` module**: Manage experiment sessions with run configurations and status transitions
- **`RunConfig`, `RunSummary`, `TemplateInfo`, `DatasetInfo`**: Data classes for run-related data
- **Manifest management**: Create, update, serialize run manifests with checkpoints

#### Feature Selection Enhancement
- **`FlexiblePCA` and `FlexibleSVD` classes**: New flexible dimensionality reduction
- Enhanced feature selection module documentation

#### AutoDetector Improvements
- **Improved header detection**: Handle cases with and without headers
- **Wavelength header detection**: Based on value characteristics and spacing
- **Signal type detection**: Check both header and data values
- **Word-boundary-aware pattern matching**: For filename detection in FolderParser

### 🔧 Improvements

#### Storage & Infrastructure
- **DuckDB storage migration**: Refactored artifact storage from `binaries/` to `artifacts/` directory
- **Centralized workspace path**: `get_active_workspace()` function for consistent path management

#### RunResult Enhancements
- Simplified metrics retrieval (removed unnecessary parameters)
- Enhanced prediction metrics handling

#### Documentation Structure
- New modules: `branch_utils`, `exclude`, reconstruction submodules
- Enhanced augmentation module with new submodules: `edge_artifacts`, `random`, `spectral`, `splines`
- Pipeline documentation with branching and merging keywords

#### Tag System
- Unit tests for QueryBuilder tag filtering (boolean, numeric, range, list membership, callable, null handling)
- Tag serialization in IndexStore
- SpectroDataset tag operations (add, set, get, remove)

### 📚 Documentation

- **Pipeline samples**: 10 new pipeline samples (JSON/YAML) demonstrating branching, stacking, filtering, model tuning
- **Filtering guide**: Comprehensive documentation for non-destructive filtering system
- **Merging guide**: Detailed examples for feature and prediction merging strategies
- **ViT-NIRS roadmap**: Integration strategy for Universal Spectral Embedding project
- **Academic paper**: LaTeX document for nirs4all framework

### 🧪 Testing

- **Reconstruction module tests**: Forward model, calibration, inversion, distributions, generator, validation
- **Scattering operators tests**: ParticleSizeAugmenter, EMSCDistortionAugmenter
- **SpectraTransformerMixin tests**: 23 tests for base class behavior
- **Environmental operators tests**: With environmental parameter handling
- **Bundle export tests**: Integration tests for special operator types
- **DuckDB storage tests**: Updated assertions for `store.duckdb` persistence
- Removed obsolete tests: `test_catalog_export.py`, `test_library_manager.py`, `test_query_reporting.py`

### 🐛 Bug Fixes

- CI example scripts file permissions
- Fixture reproducibility for standard regression dataset

### Configuration Migration

**Before:**
```yaml
sample_augmentation:
  transformers:
    - GaussianAdditiveNoise(apply_on="samples", sigma=0.01)
    - LinearBaselineDrift(apply_on="global", lambda_axis=[...])
  count: 5
```

**After:**
```yaml
sample_augmentation:
  variation_scope: "sample"
  transformers:
    - GaussianAdditiveNoise(sigma=0.01)
    - transformer: LinearBaselineDrift()
      variation_scope: "batch"
  count: 5
```

---

## [0.6.3] - Wavelength-Aware Operators & Generator Migration - 2026-01-17

### New Features

#### SpectraTransformerMixin Foundation
- **New `SpectraTransformerMixin` base class**: Enables wavelength-aware transformations while maintaining full sklearn compatibility
- **Automatic wavelength passing**: Controller detects operators that require wavelengths and extracts them from the dataset
- **`_requires_wavelengths` class flag**: Operators can declare mandatory or optional wavelength requirements
- **Dual interface support**: Both `transform(X, wavelengths=...)` and `transform_with_wavelengths(X, wl)` supported

#### Environmental Effect Operators (`nirs4all.operators.augmentation.environmental`)
- **`TemperatureAugmenter`**: Simulates temperature-induced spectral changes with region-specific effects for O-H, N-H, and C-H bands
  - Configurable shift, intensity, and broadening effects
  - Literature-based parameters from Maeda et al. (1995), Segtnan et al. (2001)
- **`MoistureAugmenter`**: Simulates moisture/water activity effects on spectra
  - Models free vs. bound water state transitions
  - Affects 1st overtone (1400-1500nm) and combination (1900-2000nm) water bands

#### Scattering Effect Operators (`nirs4all.operators.augmentation.scattering`)
- **`ParticleSizeAugmenter`**: Simulates particle size effects on light scattering
  - Wavelength-dependent baseline (lambda^(-n) relationship)
  - Configurable path length effects
- **`EMSCDistortionAugmenter`**: Applies EMSC-style scatter distortions
  - Multiplicative and additive components
  - Configurable polynomial order for wavelength-dependent baseline

#### Generator Integration
- **Operators-first architecture**: Synthetic data generator now uses operators exclusively for environmental and scattering effects
- **Simplified generator API**: Removed `use_operators` flag - operators are always used when configs are provided
- **Consistent augmentation**: Same operators used in both data generation and pipeline augmentation

### Improvements

#### Controller Enhancement
- **`TransformerMixinController`**: Updated to detect `SpectraTransformerMixin` instances and pass wavelengths automatically
- **Wavelength extraction fallback**: Primary via `dataset.wavelengths_nm()`, fallback to numeric headers
- **All execution paths updated**: Main transform, batch augmentation, and sequential augmentation paths all support wavelength passing

### Code Cleanup

#### Dead Code Removal
- Removed `TemperatureEffectSimulator`, `MoistureEffectSimulator`, `EnvironmentalEffectsSimulator` classes
- Removed `ParticleSizeSimulator`, `EMSCTransformSimulator`, `ScatteringCoefficientGenerator`, `ScatteringEffectsSimulator` classes
- Removed legacy convenience functions (`apply_temperature_effects`, `apply_moisture_effects`, etc.)
- Retained configuration dataclasses used by operators

### Documentation

- **Developer guide**: New `docs/_internals/spectra_transformer_mixin.md` with implementation notes
- **User guide**: Updated augmentation guide with wavelength-aware operators section
- **API docs**: Added documentation for `operators.augmentation` and `operators.base` packages

### Testing

- **Unit tests**: 23 tests for SpectraTransformerMixin, 42 tests each for environmental and scattering operators
- **Controller tests**: 21 unit tests for wavelength passing logic
- **Integration tests**: 7 pipeline tests for spectra transformers, 11 generator parity tests
- **Configuration tests**: Updated tests for retained configuration classes

---

## [0.6.2] - Synthetic Data Enhancement & Pipeline Improvements - 2026-01-02

### ✨ New Features

#### Synthetic Data Generation (Phase 4)
- **Wavenumber utilities**: Conversions, NIR zone classification, overtone/combination band calculations, hydrogen bonding shifts
- **Procedural component generation**: Functional group types, properties dictionary, and procedural generator
- **Application domain priors**: 20 predefined domains across agriculture, food, pharmaceutical, and industrial categories
- **Instrument simulation**: Instrument archetypes (FOSS, Bruker, SCiO), detector types, multi-sensor stitching, measurement modes
- **Environmental/matrix effects**: Temperature, moisture, particle size effects simulation
- **Spectral realism validation**: Scorecard with correlation length, derivative stats, peak density, SNR metrics, adversarial validation
- **Benchmark dataset matching**: Generate synthetic data matching published dataset characteristics
- **GPU acceleration**: JAX/CuPy backends for fast generation
- **Real data fitting**: Analyze real spectra and create matching generators
- **48 predefined spectral components** (expanded from 31): Added casein, gluten, dietary fiber, glycerol, malic acid, tartaric acid, polymers, plastics

#### Pipeline Improvements
- **Run tracking in logs**: PipelineOrchestrator now logs current run and total runs for better progress visibility
- **Batch execution**: Enhanced support for multiple pipelines and datasets in `nirs4all.run()`
- **Group-by in `top()`**: Get top N results per group with `return_grouped` option

#### Optuna Integration
- **Sorted tuple parameter type**: New parameter type for OptunaManager
- **Detailed configuration support**: Enhanced parameter sampling with log-scale and advanced options

#### Deep Learning
- **FCK-PLS Torch**: End-to-end learnable Fractional Convolutional Kernel PLS prototype (in bench/)
- **PyTorch model improvements**: Better target handling and custom regularization support

### 🔧 Improvements

- **Dataset handling**: Support for lists of SpectroDataset instances, deep copies to prevent mutation
- **Non-linear target complexity**: NonLinearConfig, ConfounderConfig, MultiRegimeConfig for complex scenarios
- **CI/CD**: Aggressive disk cleanup, improved Windows test stability, logging reset for file handle issues

### 🐛 Bug Fixes

- **OptunaManager**: Support both tuple and list formats for length configuration
- **ConcentrationPrior tests**: Use unified params dictionary structure
- **Procedural generator tests**: Corrected parameter names for consistency

### 📚 Documentation

- **New examples**: D07-D09 synthetic generator examples (wavenumber, domains, instruments)
- **Reference examples**: R05-R07 for environmental effects, validation, and fitting
- **Updated developer path**: Reflects new synthetic data examples
- **pybaselines dependency**: Added to project requirements

## [0.6.1] - GitHub Actions CI/CD Update - 2025-12-31

### 🔧 Infrastructure

- **GitHub Actions**: Updated CI/CD workflows for improved build and deployment processes
- **Minor Fixes**: Removed sparse-pls and mbpls dependencies. Implemented natively the numpy versions.

## [0.6.0] - Major API Overhaul and Architecture Improvements - 2025-12-27

This release introduces a new module-level API, complete documentation overhaul, sklearn integration, branching/merging pipelines, synthetic data generation, and extensive architectural improvements.

### ✨ New Features

#### Synthetic Data Generation
- **New `nirs4all.generate()` API**: Generate realistic synthetic NIRS spectra for testing and prototyping
- **Convenience functions**: `nirs4all.generate.regression()`, `nirs4all.generate.classification()`, `nirs4all.generate.multi_source()`
- **Builder pattern**: `SyntheticDatasetBuilder` with fluent interface for full control
- **Physically-motivated generation**: Beer-Lambert law with Voigt profile peaks, realistic noise and scatter
- **Predefined components**: 8 spectral components (water, protein, lipid, starch, cellulose, chlorophyll, oil, nitrogen_compound)
- **Configurable complexity**: `"simple"` (fast tests), `"realistic"` (typical NIR), `"complex"` (challenging scenarios)
- **Classification support**: Controllable class separation and imbalanced class weights
- **Multi-source generation**: Combine NIR spectra with auxiliary data (markers, sensors)
- **Metadata generation**: Sample IDs, groups for GroupKFold, repetitions
- **Batch effects simulation**: For domain adaptation research
- **Export capabilities**: `to_folder()`, `to_csv()` compatible with DatasetConfigs
- **Real data fitting**: `from_template()` to generate data matching real dataset characteristics
- **Pytest fixtures**: Comprehensive test fixtures in `tests/conftest.py` for reproducible testing
- **CSV variation generator**: Test loaders with different formats (delimiters, headers, decimals)

#### Module-Level API (Primary Interface)
- **New `nirs4all.run()` function**: Simplified entry point for training pipelines with intuitive parameters
- **New `nirs4all.predict()` function**: Make predictions on new data using trained models
- **New `nirs4all.explain()` function**: Generate SHAP explanations for model interpretability
- **New `nirs4all.retrain()` function**: Retrain pipelines on new data
- **New `nirs4all.session()` context manager**: Resource reuse across multiple pipeline runs
- **New `nirs4all.load_session()` function**: Load saved sessions from bundle files
- **Result wrapper classes**: `RunResult`, `PredictResult`, `ExplainResult` with convenient properties like `best_rmse`, `best_r2`, `top(n)`

#### sklearn Integration
- **New `NIRSPipeline` class**: sklearn-compatible wrapper for trained pipelines
- **New `NIRSPipelineClassifier` class**: Classification-specific sklearn wrapper
- **SHAP compatibility**: Use `NIRSPipeline` with SHAP explainers directly
- **`from_result()` and `from_bundle()` factory methods**: Easy creation from training results or exported bundles

#### Branching and Merging
- **`branch` keyword**: Create parallel execution paths with different preprocessing/models
- **`merge` keyword**: Combine branch outputs (features or predictions for stacking)
- **`source_branch` keyword**: Per-source preprocessing for multi-source datasets
- **`merge_sources` keyword**: Combine features from different data sources
- **Meta-model stacking**: Train meta-models on OOF predictions from base models
- **Branch disambiguation**: Unique naming for models across branches

#### Concat-Transform Pipelines
- **`concat_transform` keyword**: Concatenate outputs from multiple transformers
- **Action modes**: `extend`, `add`, `replace` for feature augmentation
- **Cartesian product generation**: `_cartesian_` keyword for multi-stage preprocessing combinations

#### Data Handling
- **Repetition transformation**: Convert spectral repetitions to sources or preprocessings
- **`RepetitionConfig` class**: Configure handling of unequal repetition counts (error, drop, pad, truncate)
- **3D array support**: Handle preprocessing dimensions in dataset merging
- **Aggregation methods**: Mean, median, vote with outlier exclusion using Hotelling's T²
- **Signal type detection**: Automatic detection and handling of reflectance, absorbance, Kubelka-Munk

#### Filtering and Quality Control
- **`SpectralQualityFilter`**: Filter samples by NaN ratio, Inf values, zero ratio, variance, value range
- **`XOutlierFilter`**: Detect outliers using Mahalanobis distance, PCA residual, Isolation Forest
- **`YOutlierFilter`**: Detect target outliers using IQR, Z-score, percentile, MAD methods

#### Preprocessing
- **`ReflectanceToAbsorbance` transformer**: Convert reflectance spectra using Beer-Lambert law
- **`PyBaselineCorrection` class**: Integration with pybaselines library (ASLS, AirPLS, SNIP, etc.)
- **Wavelet feature extraction**: `WaveletFeatures`, `WaveletPCA`, `WaveletSVD` classes
- **CARS and MC-UVE**: Feature selection methods for wavelength selection

#### Cross-Validation
- **`force_group` parameter**: Use any splitter with group-based splitting
- **`GroupedSplitterWrapper`**: Wrap non-group splitters for group-aware splitting
- **Fold sample ID remapping**: Correct handling of sample IDs across folds

#### Model Controllers
- **AutoGluon support**: `AutoGluonModelController` with `random_state` for reproducibility
- **GPU memory management**: Automatic memory reset before/after training for CatBoost and other GPU models
- **Customizable NN architecture**: Parameters for neural network architecture in model training

### 🔧 Improvements

#### Architecture
- **Artifact ID system V3**: Chain-based artifact IDs for better traceability in complex pipelines
- **Lazy backend loading**: TensorFlow and PyTorch loaded only when needed
- **`require_backend` utility**: Clear error messages when optional backends are missing
- **Centralized logging system**: Replace print statements with structured logging throughout
- **Exception hierarchy**: Centralized error management with error codes and context
- **`PredictionCache`**: Cache expensive prediction computations for performance

#### Pipeline Execution
- **Validation score calculation**: Prevent data leakage in BaseModelController
- **Step number tracking**: `RuntimeContext` tracks execution progress
- **Substep indexing**: TraceRecorder supports substep configurations within branches
- **Branch state restoration**: PipelineExecutor restores chain state from branch snapshots

#### Configuration
- **JSON/YAML file loading**: Load dataset configurations from external files
- **Detailed error handling**: Line and column numbers for invalid JSON/YAML syntax
- **Key normalization**: Standardize configuration keys across formats
- **Generator expansion**: Always expand generator syntax during configuration processing

#### Workspace
- **Simplified directory structure**: Remove date prefixes from artifact paths
- **Lazy directory creation**: Directories created only when needed
- **Model export functionality**: `PipelineRunner.export()` method for model bundles

#### Visualization
- **Branch diagram improvements**: Branch-specific metadata and improved clarity
- **NaN value handling**: ScoreHistogramChart handles missing values gracefully
- **Best per model option**: Filter to show only best result per model type

### 📚 Documentation

- **Complete documentation refactor**: Restructured docs with Sphinx, RTD theme
- **New API reference**: Module-level API, sklearn integration, data handling, synthetic generation
- **User guides**: Preprocessing guide, API migration guide, augmentation guide, synthetic data guide
- **Specifications**: Pipeline syntax, config format, metrics, nested CV
- **40+ pipeline examples**: Comprehensive catalog for branching, merging, multi-source
- **Reorganized examples**: User examples by topic, reference examples for syntax
- **Developer guide**: Synthetic data generator internals and extension
- **Synthetic data examples**: U09, U10 (user), D10, D11 (developer)

### 🧪 Testing

- **Comprehensive unit tests**: API module, sklearn wrappers, result classes
- **Integration tests**: Source branching, merging, multi-source, stacking
- **pytest-xdist support**: Parallel test execution with GPU markers
- **Performance benchmarks**: pytest-benchmark for regression detection

### 🐛 Bug Fixes

- **Fold sample ID handling**: Correct remapping of sample IDs to positional indices
- **Aggregation sorting**: Fixed sort order for aggregated predictions
- **Missing RMSE handling**: Use `.get()` to avoid KeyErrors for missing metrics
- **DiPLS prediction handling**: Fixed prediction logic for DiPLS models
- **Whitespace cleanup**: Fixed trailing whitespace in docstrings

### 🗑️ Removed

- **Deprecated example scripts**: Removed outdated JAX and LightGBM examples
- **Emoji utility module**: Replaced by structured logging system
- **Outdated documentation**: Removed obsolete reports and proposals

### ⚠️ Breaking Changes

- **Minimum Python version**: Now requires Python 3.11+
- **Dependency versions updated**: numpy>=1.24, scikit-learn>=1.2, pandas>=2.0
- **`save_files` parameter renamed**: Now `save_artifacts` and `save_charts`
- **Metric naming**: Some metrics renamed for consistency (e.g., RMSE to MSE in certain contexts)

### 📦 Dependencies

- **Updated minimum versions**: numpy, pandas, scipy, scikit-learn, and all optional dependencies
- **New optional dependencies**: ruff for linting, mypy for type checking, sphinx for docs
- **kennard-stone**: Updated to >=2.2.0
- **flax**: Added to JAX optional dependencies

---

## [0.5.1] - Charts performance fix - 2025-11-25

### Fixed
- **Charts**: Applied Polars optimization fix to all visualization charts in the library

## [0.5.0] - Enhanced Metrics and Visualization - 2025-11-24

### Added
- **Metrics**: Added new metrics including consistency, NRMSE (Normalized Root Mean Squared Error), NMSE (Normalized Mean Squared Error), and NMAE (Normalized Mean Absolute Error).
- **Metrics Management**: Implemented full metrics calculation for all partitions in `BaseModelController`.
- **Predictions**: Added scores management in `Predictions` class with serialization and retrieval support.
- **Visualization**: Enhanced heatmap and top-k comparison charts to display scores with local scaling options.
- **Documentation**: Added initial Sphinx documentation with project overview, features, and installation instructions.
- **Examples**: Changed binary dataset for improved testing scenarios.

### Changed
- **Architecture**: Refactored chart classes to standardize signatures and support optional metrics validation.
- **Controllers**: Updated `PredictionRanker` to utilize pre-computed scores with fallback to legacy methods.
- **Controllers**: Enhanced `PredictionSerializer` to handle serialization of scores.
- **Balancing**: Updated `BalancingCalculator` methods to fix `ref_percentage` that was equivalent to `max_factor`.
- **Charts**: Improved `FoldChartController` with better debugging and documentation.
- **CSV Export**: Updated `save_to_csv` method to accept `path_or_file` and `filename` parameters for better flexibility.

### Fixed
- **Pipeline**: Plot charts layout and display during pipeline execution.
- **Tests**: Updated balancing calculator tests to reflect new method signatures.

## [0.4.2] - Torch, Jax and sklearn style models - 2025-11-21

### Added
- **Model Support**: Fixed support for XGBoost, LightGBM, CatBoost models and added support sklearn style models.
- **Deep Learning**: Added JAX and PyTorch model controllers with data preparation utilities and `JaxModelWrapper` for state management.
- **Metrics**: Added balanced accuracy metric.
- **Pipeline**: Reintroduced parallel execution in pipeline steps (ongoing development).
- **Installation**: Enhanced backend detection and installation instructions (TensorFlow, PyTorch, JAX with GPU).
- **Inference**: Automatic inference for ranking logic (ascending parameter can be None).

### Changed
- **Architecture**: Refactored file saving architecture to implement "Return, Don't Save" pattern.
- **Controllers**: Refactored `BaseModelController` for improved execution flow and parallel training.
- **Controllers**: Updated `SklearnModelController` to enhance framework detection.
- **Examples**: Updated JAX and PyTorch model examples for prediction reuse.
- **Tests**: Refactored tests to use `RuntimeContext`.

### Fixed
- **Regression**: Ensure fold averages are only created for regression tasks with multiple folds.
- **Pipeline**: Correct pipeline definition by adding missing ShuffleSplit instance for regression comparison.

### Removed
- **Examples**: Removed deprecated example scripts for JAX and LightGBM.

## [0.4.1] - Folder/File structure rc - 2025-11-20

### Major Refactoring and Architecture Improvements
This release introduces significant architectural changes, refactoring the codebase for better modularity, type safety, and maintainability.

### Added
- **Core Architecture**:
  - **Folder Structure**: Complete reorganization of `controllers`, `core`, `data`, `tests`, `examples`, and `docs`.
  - **Context Handling**: Typed `ExecutionContext` and mutable `DataSelector`.
  - **Dataset**: `SpectroDataset` refactored to use `ArrayRegistry` and split-parquet storage.
  - **Pipeline**: Refactored execution, step handling, and artifact management.
  - **Models**: Modularized `BaseModelController`.
- **Features & Tools**:
  - `run.ps1` script for unified example execution.
  - `--show-plots` CLI argument.
  - `StratifiedKFold` and `StratifiedShuffleSplit` support.
  - New storage modules for pipeline management.
- **Feature Components Architecture**:
  - New modular component-based architecture for `FeatureSource`.
  - Type-safe enums for layouts and header units.
  - Six specialized components: `ArrayStorage`, `ProcessingManager`, `HeaderManager`, `LayoutTransformer`, `UpdateStrategy`, `AugmentationHandler`.
- **Predictions Components Architecture**:
  - New modular component-based architecture for `Predictions` class.
  - Six specialized components: `PredictionStorage`, `PredictionSerializer`, `PredictionIndexer`, `PredictionRanker`, `PartitionAggregator`, `CatalogQueryEngine`.

### Changed
- **Visualization**:
  - Refactored `FoldChart`, `SpectraChart`, `ConfusionMatrix`.
  - Improved classification visualization (discrete color mapping).
  - Reorganized SHAP and PCA analyzers.
- **Internal**:
  - Migrated component imports to internal `_*` modules.
  - Centralized evaluator and serialization logic.
  - Hardened Optuna sampling and logging.
- `FeatureSource` class moved to `nirs4all/data/feature_components/feature_source.py`.

### Fixed
- **Critical**: Missing `pipeline_uid` in prediction ranker results.
- **Critical**: NumPy array weights handling in predictions.
- Evaluator import path issues.
- Header unit preservation when adding samples.

### Documentation & Tests
- Restructured tests to mirror source code.
- Added a few architecture reviews, roadmap updates, and developer guides.
- Removed obsolete or review documents.
