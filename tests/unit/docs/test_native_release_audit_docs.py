"""Coverage checks for the native conformal finetuning release audit page."""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs" / "source"
CHANGELOG = Path(__file__).resolve().parents[3] / "CHANGELOG.md"
AUDIT_PAGE = DOCS_ROOT / "reference" / "native_conformal_finetuning_release_audit.md"
PUBLIC_INTERFACES = DOCS_ROOT / "reference" / "public_interfaces.md"
REFERENCE_INDEX = DOCS_ROOT / "reference" / "index.md"


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_native_release_audit_page_is_published_in_reference_toctree() -> None:
    index = REFERENCE_INDEX.read_text(encoding="utf-8")

    assert "native_conformal_finetuning_release_audit" in index


def test_native_release_audit_page_covers_public_contracts_and_non_claims() -> None:
    text = _normalized(AUDIT_PAGE.read_text(encoding="utf-8"))

    required_phrases = (
        "nirs4all.run(tuning=..., calibration=...)",
        "named final model steps",
        "sklearn `Pipeline` objects",
        "sklearn-like `(name, step)` tuples",
        "explicit `sklearn.*` class-path mappings",
        "explicit constructor `params` for `sklearn.*` transform/model import paths",
        "searchable non-final passthrough steps",
        '{"kind": "passthrough"}',
        "TuningPassthrough()",
        'kind` must be the literal string `"passthrough"`',
        "not a host object that stringifies to that value",
        "TCV1-compatible JSON-native",
        "`force_params` as an explicit warm-start trial",
        "keys must be a subset of `tuning.space`",
        "values use the public decoded syntax",
        '"force_params": {"model.n_components": 4}',
        "`tuning.force_params` enqueues a caller-provided parameter assignment as the first HPO trial",
        "Optuna uses `study.enqueue_trial(...)`",
        "n4m requires native `optimizer.enqueue(...)` support",
        "On a non-empty Optuna `resume=True` study",
        "it is not enqueued again",
        "changed warm-start values under the same `study_name`/`storage` fail closed",
        "n4m optimizer-state persistence is local and N4MOPT-backed",
        '`storage="file:///absolute/checkpoint-dir"`',
        "writes a JSON checkpoint manifest after each terminal trial",
        "treats `n_trials` as the target total trial count",
        "Restored Optuna `COMPLETE` rows must carry a finite numeric value",
        "missing or non-finite storage values fail closed",
        "Restored non-`COMPLETE` rows must not carry a final storage value",
        "failed, pruned or in-flight rows with final values are rejected",
        "Restored `RUNNING` rows fail closed during resume",
        "interrupted active trials cannot be safely recovered into a terminal HPO tape",
        "Restored terminal Optuna rows must keep exactly the current `tuning.space` parameter keys",
        "stored parameter table was removed fail closed",
        "Restored queued Optuna `WAITING` rows that already carry materialized params or `fixed_params`",
        "incompatible values are rejected before Optuna can consume them",
        "Restored Optuna trial numbers must be canonical unique integers",
        "non-integer or duplicate trial numbers fails closed",
        "ordered canonically by numeric trial id",
        "duplicate or non-integer restored trial ids",
        "Restored `COMPLETE` rows must carry a finite numeric score",
        "missing, boolean or non-finite scores fail closed",
        "Restored n4m non-`COMPLETE` rows must not carry a final score",
        "failed, pruned or cancelled checkpoint rows with scores are rejected",
        "Restored checkpoint row params must still match the current `tuning.space` keys and value domains",
        "edited or incompatible checkpoint keys, categorical choices, numeric ranges or numeric steps fail closed",
        "n4m candidate failures require native failure reporting support",
        "expose either `optimizer.tell_result(...)` with a failed trial status or `optimizer.tell(...)`",
        "fails closed instead of silently losing the failed-trial tape",
        "Optuna- and n4m-pruned candidates stay distinct from failed candidates",
        '`TrialResult(state="PRUNED", value=None)`',
        "compact summary counts `PRUNED` separately",
        '`score_extractor="pruned"`',
        "`optimizer.tell_result(..., TrialStatus.PRUNED)`",
        "older bindings fail closed instead of rewriting the candidate as `FAIL`",
        "can invalidate calibration",
        "nirs4all.calibrate()",
        "nirs4all.predict_calibrated()",
        "nirs4all.robustness()",
        "nirs4all.robustness_from_workspace_prediction()",
        "PredictResult.robustness()",
        "RobustnessReport.save_artifacts()",
        "TuningResult.summary_artifact()",
        "Direct construction of `SearchSpaceParameter`, `ParameterPatch`,",
        "paths are canonicalized, values must stay TCV1 JSON-native",
        "scores must be finite, integer contract fields reject booleans",
        "trial ids are unique, and candidate/best params must stay inside `tuning.space`",
        "rejects boolean or numeric-string `best_value` payloads instead of coercing them with `float()`",
        "Direct construction of the internal optimizer adapter seams is also",
        "`ObjectiveTuningRunResult` requires a real `TuningResult`",
        "categorical codec requires non-empty unique choices",
        "matching decoder keys and unique TCV1 JSON-native decoded values",
        "Public tuning helper payloads are fail-closed before runtime execution",
        "dataset-backed `TuningScoreData` and `TuningWinner` selectors reject",
        "`TuningWinner.score` must be finite numeric evidence",
        "require canonical non-empty string keys with strict JSON-native finite values",
        "Non-finite numbers, bytes, tuples, sets and arbitrary Python objects are",
        "`NativeTuning.space` and `force_params` canonicalize string patch keys",
        "Raw `NativeTuning(score_data={...})`,",
        "including strict raw dataset selectors, boolean raw augmented/calibration",
        "finite raw winner scores and rejection of nested `calibration_data`",
        "raw calibration coverage, method, unit, workspace metadata and",
        "`workspace_conformal_id` are also validated/canonicalized before publication",
        "Workspace persistence applies the same strict JSON-native metadata boundary",
        "`save_workspace_tuning_result(...)`,",
        "`run(tuning=..., workspace_path=...)` and",
        "`save_workspace_calibrated_result(...)` fail closed instead of stringifying",
        "Public conformal result metadata supplied through",
        "`calibrate(..., result_metadata=...)` or",
        "`predict_calibrated(..., result_metadata=...)` is validated at the API seam",
        "keyword registry publishes both entries with strict JSON-native schemas",
        "Prediction workspace metadata uses the same strict JSON-native sidecar boundary",
        "`predict(..., save_to_workspace=True, workspace_metadata=..., workspace_result_metadata=...)` fail closed",
        "Top-level `NativeTuning` core fields are parsed through `DagMLTuningSpec`",
        "integer/bool fields",
        "reject coercive values and optimizer persistence ids remain canonical",
        "Dataset-backed tuning column selectors are strict as well",
        "`sample_id_column` and `group_column` must be canonical non-empty strings",
        "`metadata_columns` must be either one canonical string or a duplicate-free",
        "The same rule applies to raw",
        "`NativeTuning.score_data` and `NativeTuning.winner` mappings before",
        "Tuple/list `NativeTuning.score_data` is also validated before publication",
        "it must contain `(X_score, y_score)` and at most",
        "metadata keys must be canonical in mapping or row-style sequence form",
        "valid tuple inputs publish as JSON-native lists",
        "Scalar strings, bytes, numbers and arbitrary objects are",
        "neither mapping nor tuple/list score cohorts",
        "Tuning score/winner metadata follows the same boundary",
        "`TuningScoreData.metadata`, `TuningWinner.metadata`, raw",
        "`NativeTuning.score_data.metadata` and raw `NativeTuning.winner.metadata`",
        "reject non-JSON-native metadata values before runtime",
        "Tuning score and winner text fields also reject coercion",
        "`TuningScoreData.metric`/`score_metric`, raw",
        "`NativeTuning.score_data.metric`/`score_metric`, `TuningWinner.metric`,",
        "`dataset_name`, `model_name`, `task_type` and the corresponding raw",
        "must be non-empty strings without NULs",
        "publish lowercase canonical values",
        "publish trimmed strings, never `str(...)` conversions",
        "Temporary conformal objective calibration metadata follows the same boundary",
        "`TuningConformalScoreCalibration.metadata` and raw",
        "`score_data.conformal_calibration.metadata` reject non-string or",
        "non-finite or non-JSON-native metadata values",
        "Direct construction of `RobustnessScenarioSpec` is fail-closed",
        "`severity` must be a real numeric scalar, not a boolean or numeric string",
        "`extra` keys must be canonical non-empty strings",
        "Raw `robustness.scenarios` mappings use the same strict boundary",
        "keys must be canonical non-empty strings, `severity` must be a real finite numeric scalar",
        "safe persistence flags",
        "scalar trial diagnostics such as `error_type`, `score_family`, `score_extractor` and fingerprints",
        "excludes candidate params and raw exception messages",
        "nirs4all.get_tuning_summary_schema()",
        "nirs4all.get_keyword_registry()",
        "nirs4all.get_robustness_summary_schema()",
        "predict.all_predictions",
        "all_predictions=False",
        "all_predictions=True",
        "calibrated identity mapping",
        "invalid `conformal/` sidecar fail validation",
        '`predict(model="*.n4a", coverage=...)` also rejects a structurally complete',
        "but no canonical physical `sample_ids`",
        "validated before the raw model prediction runs",
        "invalid conformal identity cannot be hidden by a successful uncalibrated model replay",
        "Conformal reload is identity fail-closed across filesystem stores, workspace `conformal_results` rows and `.n4a` sidecars",
        "whose non-empty prediction cohort lacks canonical physical `sample_ids` fails reload",
        "a corrupted workspace `conformal_results` row with the same missing prediction identity is rejected",
        "by both `load_workspace_calibrated_result(...)` and `load_workspace_calibrated_predict_result(...)`",
        "`CalibratedRunResult` metadata is also strict JSON-compatible at construction and reload time",
        "non-finite floats, Python objects, non-string keys or whitespace-padded keys fail closed",
        "Nested mapping keys are checked the same way, and tuple values are rejected",
        "Conformal numeric arrays (`y_true`, `y_pred`, interval bounds and `qhat`)",
        "Dataset-backed selectors reject non-string or whitespace-padded keys plus",
        "raw `calibration_data={...}` mappings",
        "`include_augmented` must be a boolean",
        "reject boolean payloads instead of coercing them to `0.0`/`1.0`",
        'reject numeric strings such as `"1.0"` instead of parsing them as floats',
        "NumPy boolean scalars in serialized scores or quantiles",
        "serialized numeric fields reject NumPy ndarray scalars instead of coercing them to JSON numbers",
        "Empirical `ConformalMetricSet` diagnostics are validated before fingerprinting",
        "`observed_coverage` must be finite in `[0, 1]` and match `n_covered / n_samples`",
        "`coverage_gap` must equal `observed_coverage - coverage`",
        "non-negative or positive infinity for unbounded intervals",
        "Direct `ConformalIntervalBlock` and `CalibratedPredictionBlock` construction is fail-closed",
        "coverage-key mismatches, interval shape mismatches, inverted bounds",
        "negative or non-row-aligned `qhat` values",
        "Direct `SplitConformalCalibrator` construction validates retained residual scores",
        "negative scores, edited `qhat` values or unsupported vocabulary fail closed",
        "Version fields on conformal cohort manifests, calibration artifacts and calibrated results are strict integer contract tags",
        "boolean `true`/`false` and numeric strings fail closed instead of being coerced to schema version `1`",
        "Optional conformal artifact identity strings (`target_name`, `predictor_fingerprint`, `calibration_data_fingerprint`)",
        "non-empty strings without surrounding whitespace",
        "Guarantee metadata string fields (`effective_engine`, `requested_engine`,",
        "`source_calibrated_result_fingerprint`, `invalidation_reasons`) are also strict provenance fields",
        "booleans, objects, empty or whitespace-padded values fail closed instead of being stringified",
        "persisted `conformal_guarantee_status.version` must be the strict integer `1`",
        "Status `predictor_fingerprint`, `calibration_data_fingerprint`, `guarantee`, and",
        "`scope` must also match the embedded artifact on construction and reload",
        "A persisted status must include the complete generated field set",
        "`status` must be `active` exactly when `invalidation_reasons` is empty",
        "Conformal reload is artifact-derived, not only fingerprint-derived",
        "stored `qhat_by_coverage` values must recompute from the retained non-negative residual scores",
        "every materialized interval must equal `y_pred ± qhat`",
        "reject edited intervals or quantiles even when the JSON is otherwise self-consistent",
        "Grouped split conformal is executable in the replayed-array substrate",
        '`group_by="group"` consumes calibration and prediction group labels',
        "`calibration_metadata` and `prediction_metadata`",
        "fail closed without a global-quantile fallback",
        "row-aligned grouped `qhat` vectors",
        "Filesystem stores, workspace `conformal_results` rows, conformal-only `.n4a` bundles and model `.n4a` sidecars",
        "preserve and revalidate `group_keys`, `group_calibrators` and grouped qhat vectors",
        '`multi_target="joint_max"` is executable for two-dimensional replayed-array regression outputs',
        "`max(abs(y_true - y_pred))` across target columns",
        "simultaneous for the target vector",
        "DAG-ML D10 `cache_namespace_fingerprints`",
        "signed native control-plane proofs",
        "nirs4all forwards them unchanged to `dag_ml`",
        "DAG-ML owns validation, namespace-aware handle derivation",
        "persistent file-store payload naming and columnar manifest exposure",
        "no nirs4all keyword that mutates this proof after signing",
        'metadata["spectral_replay"]',
        "optional `spectral_replay` block in `summary.json`",
        "source, route, saved bundle path",
        "`all_predictions=False` for bundle replay",
        "whether sample ids were forwarded",
        "not a permission for bindings or Studio to replay spectra locally",
        "CalibratedRunResult",
        "summary.json",
        "tuning-summary.schema.json",
        "keyword-registry.json",
        "robustness-summary.schema.json",
        "Bindings and Studio should consume these public artifacts and schemas",
        "n4m.model_selection.finetune_estimator(...)",
        "They do not emit",
        "CalibratedRunResult",
        "run(tuning=...) -> calibrate()/predict_calibrated() -> robustness()",
        "Explicit non-claims",
        "native tuning for arbitrary DAG branch/merge graphs",
        "automatic spectral/OOD perturbation replay from a stored prediction alone",
        "bindings or Studio replaying spectra merely because `summary.json` carries `spectral_replay` provenance",
        "conformal guarantees inferred inside language bindings",
        "post-signature mutation of DAG-ML D10 `cache_namespace_fingerprints` from a nirs4all keyword",
        "applying a conformal sidecar to every model prediction entry with `all_predictions=True`",
        "silently ignoring invalid `conformal/` sidecars and returning uncalibrated predictions",
        "Studio graphical spectral/OOD campaign execution",
        "qhat=Infinity",
        "materialized interval is intentionally unbounded",
        "`Native keyword/effect quick map`",
        "`run.tuning.force_params`, `run.tuning.score_data`,",
        "temporary conformal scoring, `run.tuning.winner`, final calibration",
        "published evidence and fail-closed boundary",
        "CI checks, bindings, Studio forms or generated configuration",
        "`nirs4all.load_workspace_predict_result()` / `nirs4all.load_workspace_predict_results()` / `Predictions.get_predict_result_by_id()` / `Predictions.to_predict_results()`",
        "Supported bridge from stored prediction rows to native `PredictResult`",
        "preserving intervals, conformal/tuning replay provenance, robustness evidence, executable `X`/`spectra`, sample ids and model metadata",
        "Supported bridge from one stored prediction row to a native robustness report",
        "can persist the report back with a `prediction_id` link",
        "`PredictResult.calibration_replay_source`",
        "`PredictResult.tuning_calibration_source`",
        "`PredictResult.spectral_replay_evidence_status`",
        "instead of scraping sidecar rows or synthesizing missing spectral replay inputs",
    )

    missing = [phrase for phrase in required_phrases if _normalized(phrase) not in text]
    assert not missing, "native release audit page missing:\n" + "\n".join(missing)


def test_public_interfaces_publish_robustness_summary_spectral_replay_contract() -> None:
    text = _normalized(PUBLIC_INTERFACES.read_text(encoding="utf-8"))

    required_phrases = (
        "`summary.json` | CI, bindings, Studio/Web, dashboards | Stable compact summary rows plus optional fail-loud guarantee and replay-provenance blocks.",
        "optional `conformal_guarantee_status`, carrying fail-loud guarantee metadata",
        "optional `spectral_replay`, carrying metadata-only provenance for spectral robustness scenarios",
        "`source`, `route`, `sample_ids_forwarded`",
        "saved `predictor_bundle` path",
        "`all_predictions=False` replay mode",
        "`spectral_replay` is publication provenance, not an execution contract for downstream products",
        "`execution_scope` value is `baseline`, `prediction_replay`, or `spectral_replay`",
        "`requires_spectral_replay=true` marks rows that came from explicit-X spectral/OOD replay",
        "Bindings, Studio/Web dashboards, and CI cards may display",
        "must not infer permission or capability to replay spectral/OOD perturbations locally from this metadata alone",
        "`PredictResult.robustness_evidence` and",
        "`PredictResult.spectral_replay_evidence_status`",
        "`has_X_or_spectra` and `has_executable_X_or_spectra`",
        "matrix row count to match the prediction row count",
        "`nirs4all.robustness()` can consume that matrix and the published bundle path as defaults",
        "provenance-only strings remain non-executable metadata",
    )

    missing = [phrase for phrase in required_phrases if _normalized(phrase) not in text]
    assert not missing, "public interfaces page missing:\n" + "\n".join(missing)


def test_changelog_publishes_invalid_conformal_sidecar_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "invalid `conformal/` sidecar" in text
    assert "fails sidecar validation instead of falling back to uncalibrated prediction" in text
    assert "A structurally complete sidecar whose `calibrated_result.json` has non-empty predictions" in text
    assert "but no canonical physical `sample_ids` is rejected before raw model prediction runs" in text
    assert "invalid conformal identity cannot be hidden by a successful uncalibrated replay" in text


def test_changelog_publishes_conformal_reload_identity_fail_closed_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Conformal reload identity is fail-closed" in text
    assert "Filesystem conformal stores, workspace `conformal_results` rows and `.n4a` conformal sidecars" in text
    assert "whose non-empty prediction cohort lacks canonical physical `sample_ids` fails reload" in text
    assert "Corrupted workspace conformal rows with the same missing prediction identity are rejected" in text
    assert "`load_workspace_calibrated_result(...)` and `load_workspace_calibrated_predict_result(...)`" in text
    assert "before a partial `CalibratedRunResult` or conformal `PredictResult` can be exposed" in text
    assert "Empirical `ConformalMetricSet` diagnostics now fail closed" in text
    assert "observed coverage, coverage gap, width, interval score or count consistency" in text
    assert "positive infinity remains valid for unbounded interval metrics" in text
    assert "Direct `ConformalIntervalBlock` and `CalibratedPredictionBlock` construction now fails closed" in text
    assert "coverage-key mismatches, interval shape mismatches, inverted bounds" in text
    assert "negative or non-row-aligned `qhat` values" in text
    assert "Direct `SplitConformalCalibrator` construction now validates retained residual scores" in text
    assert "negative scores, edited `qhat` values or unsupported vocabulary fail closed" in text
    assert "Version fields on conformal cohort manifests, calibration artifacts and calibrated results are strict integer contract tags" in text
    assert "numeric strings fail closed instead of being coerced to schema version `1`" in text
    assert "Optional conformal artifact identity strings" in text
    assert "invalid direct-construction and reload payloads fail closed before provenance publication" in text
    assert "Guarantee metadata string fields" in text
    assert "fail closed instead of being stringified" in text
    assert "Workspace conformal-row metadata is validated before insertion" in text
    assert "Public conformal result metadata now uses the same strict JSON-native boundary" in text
    assert "`predict_calibrated(..., result_metadata=...)` before generated guarantee" in text
    assert "keyword registry publishes both entries with the same strict schema" in text
    assert "Robustness workspace metadata uses the same strict JSON-native row boundary" in text
    assert "`save_workspace_robustness_report(...)` and" in text
    assert "`robustness(..., workspace_path=..., workspace_metadata=...)` fail closed" in text
    assert "Prediction workspace publication now applies the same strict JSON-native boundary" in text
    assert "`save_workspace_predict_result(..., metadata=..., result_metadata=...)`" in text
    assert "conformal_guarantee_status.version` must be the strict integer `1`" in text
    assert "`ConformalCalibrationSpec` validates direct construction the same way as" in text
    assert "coverage values must be real numeric scalars, not booleans or numeric strings" in text
    assert "strict non-boolean integer grouped `n_samples` summaries" in text
    assert "Conformal calibration cohort rows built directly or reloaded from JSON now require" in text
    assert "strict JSON-native metadata with string keys, before the cohort manifest can be fingerprinted" in text
    assert "optional serialized `n_samples` summary must also be a strict non-boolean integer matching the row count" in text
    assert "Serialized method/unit contract fields reject arbitrary Python objects instead of accepting their" in text
    assert "`__str__` output as evidence" in text
    assert "Row-aligned calibration metadata supplied as either column mappings or per-row mappings uses the same strict key rule" in text
    assert "non-string or whitespace-padded metadata keys fail before manifest JSON coercion" in text
    assert "Status `predictor_fingerprint`, `calibration_data_fingerprint`, `guarantee`, and `scope`" in text
    assert "must also match the embedded artifact on construction and reload" in text
    assert "A persisted status must include the complete generated field set" in text
    assert "otherwise `invalidated`" in text
    assert "The generated `limitations` list must also match the embedded artifact's guarantee mode exactly" in text
    assert "edited, shortened, empty or non-string limitation payloads fail closed" in text


def test_changelog_publishes_conformal_reload_interval_derivation_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Conformal reload intervals are artifact-derived" in text
    assert "Stored conformal quantiles must recompute from retained non-negative residual scores" in text
    assert "each materialized interval must equal `y_pred ± qhat`" in text
    assert "reject edited intervals or quantiles even when the stored JSON has been made self-consistent" in text


def test_changelog_publishes_grouped_conformal_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Grouped replayed-array conformal calibration" in text
    assert "`nirs4all.calibrate(..., group_by=...)` now executes grouped split conformal" in text
    assert "`prediction_groups` and `prediction_metadata` route prediction rows to retained group quantiles" in text
    assert "missing or unseen groups fail closed without global fallback" in text
    assert "Filesystem stores, workspace `conformal_results` rows, conformal-only `.n4a` bundles and model `.n4a` sidecars" in text
    assert "preserve `group_keys`, `group_calibrators`, row-aligned grouped `qhat` vectors" in text
    assert "strict non-boolean integer grouped `n_samples` summaries" in text
    assert "revalidate them on reload" in text


def test_changelog_publishes_joint_max_conformal_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Joint-max multi-target conformal calibration" in text
    assert '`nirs4all.calibrate(..., multi_target="joint_max")` now supports' in text
    assert "two-dimensional replayed-array regression outputs" in text
    assert "`max(abs(y_true - y_pred))` across target columns" in text
    assert "simultaneous target-vector guarantee" in text


def test_changelog_publishes_dagml_cache_namespace_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "DAG-ML D10 cache namespace boundary published" in text
    assert "Native training/replay objects carrying `cache_namespace_fingerprints`" in text
    assert "nirs4all forwards them unchanged through the native client" in text
    assert "DAG-ML owns validation, namespace-aware handle derivation" in text
    assert "file-store payload naming and columnar manifest exposure" in text
    assert "no nirs4all keyword for post-signature mutation" in text


def test_changelog_publishes_native_hpo_force_params_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Native HPO warm-start contract published" in text
    assert "`force_params` as a strict HPO contract key" in text
    assert "requires keys to be a subset of `tuning.space`" in text
    assert "preserves public decoded categorical values in results" in text
    assert "fails closed when n4m bindings do not expose native `optimizer.enqueue(...)`" in text


def test_changelog_publishes_n4m_failed_trial_reporting_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "n4m failed-trial reporting is fail-closed" in text
    assert "requires native bindings to expose either `optimizer.tell_result(...)`" in text
    assert '`optimizer.tell(...)` before it can preserve `TrialResult(state="FAIL")`' in text
    assert "Older bindings that cannot report failed candidates fail closed" in text
    assert "silently losing HPO tape evidence" in text


def test_changelog_publishes_optuna_pruned_trial_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Optuna pruned trials remain distinct in the HPO tape" in text
    assert '`optuna.exceptions.TrialPruned` as `TrialResult(state="PRUNED", value=None)`' in text
    assert '`score_extractor="pruned"`' in text
    assert "`TuningResult.summary_artifact()` counts `PRUNED` separately from `FAIL`" in text


def test_changelog_publishes_n4m_pruned_trial_boundary() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "n4m pruned trials use native `TrialStatus.PRUNED` or fail closed" in text
    assert 'preserves shared-objective prune exceptions as `TrialResult(state="PRUNED", value=None)`' in text
    assert "optimizer.tell_result(..., TrialStatus.PRUNED)" in text
    assert "Older bindings that cannot record a pruned terminal state fail closed" in text


def test_changelog_publishes_compact_tuning_summary_diagnostics() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Compact tuning summary diagnostics" in text
    assert "`TuningResult.summary_artifact()` now includes scalar `trials[*].diagnostics` fields" in text
    assert "`error_type`, `score_family`, `score_extractor`, `search_space_fingerprint` and `tuning_fingerprint`" in text
    assert "Candidate params and raw exception messages stay out of the compact summary" in text


def test_changelog_publishes_optuna_storage_resume_diagnostics() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Optuna storage resume reconstructs compact trial diagnostics" in text
    assert '`score_extractor="optuna_storage"`' in text
    assert '`score_extractor="failed"` and pruned rows use `score_extractor="pruned"`' in text
    assert "`TuningResult.trials` and summary artifacts" in text
    assert "existing materialized trial params do not match the current `tuning.space` keys" in text
    assert "mixing incompatible search spaces under the same `study_name`/`storage`" in text
    assert "Existing categorical values must also remain present in the current choices for their key" in text
    assert "removed or renamed choices fail closed before optimizer execution" in text
    assert "Existing numeric values must also remain inside the current range for their key" in text
    assert "narrowed ranges that exclude stored trials fail closed before optimizer execution" in text
    assert "stored values must also lie on that grid" in text
    assert "During storage-backed resume, `n_trials` is now treated as the target total trial count" in text
    assert "instead of an unconditional number of additional trials" in text
    assert "persist nirs4all `study.user_attrs`" in text
    assert "optimizer contract fingerprint and search-space fingerprint" in text
    assert "mismatched fingerprints fail closed during resume" in text
    assert "Restored Optuna `COMPLETE` rows must carry a finite numeric value" in text
    assert "missing or non-finite storage values fail closed" in text
    assert "Restored non-`COMPLETE` rows must not carry a final storage value" in text
    assert "failed, pruned or in-flight rows with final values are rejected" in text
    assert "Restored `RUNNING` rows fail closed during resume" in text
    assert "interrupted active trials cannot be safely recovered into a terminal HPO tape" in text
    assert "Restored terminal Optuna rows must keep exactly the current `tuning.space` parameter keys" in text
    assert "stored parameter table was removed fail closed" in text
    assert "Restored queued Optuna `WAITING` rows that already carry materialized params or `fixed_params`" in text
    assert "incompatible values are rejected before Optuna can consume them" in text
    assert "Restored Optuna trial numbers must be canonical unique integers" in text
    assert "non-integer or duplicate trial numbers fails closed" in text


def test_changelog_publishes_n4m_optimizer_checkpoint_resume() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "n4m optimizer checkpoints are resumable in the shared objective seam" in text
    assert '`storage="file:///absolute/checkpoint-dir"`' in text
    assert "native N4MOPT checkpoint bytes" in text
    assert "resume=True` reloads only a matching optimizer contract" in text
    assert "checkpoint with one completed trial and `n_trials=2` runs one remaining trial" in text
    assert "Restored trial rows are decoded back to public `TrialResult.params`" in text
    assert "ordered canonically by numeric trial id" in text
    assert "duplicate restored trial ids or ids that are not canonical integers" in text
    assert "Restored `COMPLETE` rows must carry a finite numeric score" in text
    assert "missing, boolean or non-finite scores fail closed" in text
    assert "Restored n4m non-`COMPLETE` rows must not carry a final score" in text
    assert "failed, pruned or cancelled checkpoint rows with scores are rejected" in text
    assert "Named categorical `options` whose optimizer labels differ from" in text
    assert "Restored checkpoint row params must still match the current `tuning.space` keys" in text
    assert "edited or incompatible checkpoint keys, categorical choices, numeric ranges or numeric steps fail closed" in text
    assert '`score_extractor="failed"`, `score_extractor="pruned"` or' in text
    assert '`score_extractor="cancelled"` diagnostics for summary cards' in text
    assert "Non-terminal checkpoint rows such as `RUNNING` fail closed during resume" in text


def test_changelog_publishes_workspace_predict_result_helper() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Direct workspace prediction-to-`PredictResult` helper" in text
    assert "`nirs4all.save_workspace_predict_result(workspace_path, result, ...)`" in text
    assert "`nirs4all.predict(..., save_to_workspace=True, workspace_result_metadata=...)`" in text
    assert "`nirs4all.load_workspace_predict_result(workspace_path, prediction_id)`" in text
    assert "`nirs4all.load_workspace_predict_results(workspace_path, dataset_name=None)`" in text
    assert "explicit publisher, prediction-time publisher shortcut, one-record loader and bulk workspace/store bridges as public APIs" in text
    assert "publisher writes `PredictResult` values plus optional executable `X`/`spectra` evidence and `result_metadata`" in text
    assert "convert through `PredictResult.from_prediction_record()`" in text
    assert "preserve sample ids, model metadata, intervals and conformal/tuning/robustness replay provenance" in text


def test_changelog_publishes_workspace_prediction_robustness_helper() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Direct workspace prediction-to-robustness helper" in text
    assert "`nirs4all.robustness_from_workspace_prediction(workspace_path, prediction_id, ...)`" in text
    assert "loads one stored prediction through the public `PredictResult` bridge" in text
    assert "consumes executable stored `X`/`spectra` plus `predictor_bundle` evidence" in text
    assert "linked to the same `prediction_id`" in text
    assert "workspace robustness from-prediction --prediction-id" in text
    assert "`--y-true`/`--y-true-json`, `--scenarios-json`, optional slicing metadata" in text
    assert "same JSON/summary/Markdown/HTML/Parquet/artifact output formats" in text
    assert "Executable spectral/OOD evidence publishing through `Predictions`" in text
    assert "accepts `X=...`, `spectra=...`, and `result_metadata=...`" in text
    assert "preserves the same sidecar evidence during workspace consolidation" in text


def test_changelog_publishes_native_keyword_effect_quick_map() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Native keyword/effect quick map published" in text
    assert "links each supported syntax" in text
    assert "`run.tuning.force_params`, `run.tuning.score_data`, temporary conformal scoring" in text
    assert "workspace `Predictions` bridge" in text
    assert "runtime effect, published evidence and fail-closed boundary" in text
    assert "release audit and extended Python API index point to the same map" in text
    assert "`PredictResult.calibration_replay_source`" in text
    assert "`PredictResult.tuning_calibration_source`" in text
    assert "`PredictResult.spectral_replay_evidence_status`" in text
    assert "instead of scraping sidecar rows or synthesizing missing spectral replay inputs" in text


def test_changelog_publishes_fail_loud_predict_result_guarantee_display() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Fail-loud conformal guarantee display in `PredictResult`" in text
    assert "string representation reports the guarantee status, effective engine and selected coverages" in text


def test_changelog_publishes_fail_loud_robustness_guarantee_exports() -> None:
    text = _normalized(CHANGELOG.read_text(encoding="utf-8"))

    assert "Markdown and HTML robustness exports now render fail-loud conformal guarantee details" in text
    assert "requested/effective engine, method, unit, selected and calibrated coverages" in text
    assert "robustness `summary.json` artifact and its JSON Schema now also carry optional `conformal_guarantee_status` and `spectral_replay`" in text
    assert "spectral replay provenance without parsing the full report" in text
