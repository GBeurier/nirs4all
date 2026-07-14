"""Coverage checks for the native tuning/conformal guide."""

from __future__ import annotations

from pathlib import Path

import nirs4all
from nirs4all.api.predict import predict
from nirs4all.api.run import run


def _native_registry_entries() -> list[dict]:
    entries = []
    for entry in nirs4all.get_keyword_registry()["entries"]:
        path = entry["path"]
        if path in {
            "run.engine",
            "run.calibration",
            "predict.coverage",
            "predict.all_predictions",
            "predict.save_to_workspace",
            "predict.workspace_metadata",
            "predict.workspace_result_metadata",
            "predict_calibrated.result_metadata",
        }:
            entries.append(entry)
        elif path.startswith(("run.tuning", "calibrate", "robustness")):
            entries.append(entry)
    return entries


def test_native_tuning_conformal_guide_covers_registry_paths_and_effects() -> None:
    """The user guide must stay aligned with the machine-readable keyword registry."""

    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")

    missing: list[str] = []
    for entry in _native_registry_entries():
        path = entry["path"]
        expected_terms = [
            path,
            entry["lifecycle_stage"],
            entry["invalidates_calibration"],
            *entry.get("changes", []),
        ]
        for term in expected_terms:
            if str(term) not in text:
                missing.append(f"{path}: {term}")

    assert not missing, "native_tuning_conformal.md missing registry coverage:\n" + "\n".join(missing)


def test_native_tuning_conformal_guide_has_integration_quick_map() -> None:
    """The guide must expose a concise integration map beyond accessor names."""

    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "## Native keyword/effect quick map" in text
    assert "| Public syntax | Runtime effect | Published evidence | Fail-closed boundary |" in text
    assert "`run.tuning.space`" in text
    assert "`nirs4all.tuning.ordered_search_space`, `SearchSpaceParameter`, `ParameterPatch`" in normalized
    assert "`run.tuning.force_params`" in text
    assert "Keys must exist in `run.tuning.space`" in normalized
    assert "`run.tuning.score_data`" in text
    assert "It is never reused as final conformal calibration evidence" in normalized
    assert "`run.tuning.score_data.conformal_calibration` plus `conformal_coverage`" in text
    assert '`final_calibration_scope="unmodified_by_score_data"`' in normalized
    assert "`run.tuning.winner`" in text
    assert "`run(..., calibration=...)` or `run.tuning.calibration`" in text
    assert "`calibration_data` cannot be injected in this combined tuning flow" in normalized
    assert "`predict(coverage=...)`" in text
    assert "unsupported coverages, invalid sidecars, and conformal `all_predictions=True` fail closed" in normalized
    assert "`robustness(..., scenarios=[...])`" in text
    assert "It does not refit, structurally retrain, or renew conformal guarantees" in normalized
    assert "`robustness.X` plus `robustness.predictor` or `robustness.predictor_bundle`" in text
    assert 'stored provenance markers such as `"prediction_arrays.X"` are not executable arrays' in normalized
    assert "`PredictResult.spectral_replay_evidence_status`" in text
    assert "`needs_spectral_replay_evidence` is a hard block for spectral/OOD replay" in normalized
    assert "`Predictions.get_predict_result_by_id(...)` / `to_predict_results()`" in text
    assert "Records loaded without arrays cannot become executable prediction results" in normalized


def test_run_docstring_describes_current_native_tuning_subset() -> None:
    """The public run() docstring must not describe implemented tuning as rejected."""

    doc = run.__doc__ or ""

    assert "Typed native tuning specification" in doc
    assert "single estimator or linear sklearn-like" in doc
    assert "workspace persistence/resume" in doc
    assert "run(tuning=..., calibration=...)" in doc
    assert "TunedSingleEstimatorConformalResult" in doc
    assert "calibrated conformal ``calibrated`` result" in doc
    assert "intentionally rejected until" not in doc


def test_predict_docstring_describes_current_conformal_prediction_subset() -> None:
    doc = predict.__doc__ or ""
    normalized = " ".join(doc.split())

    assert "Calibrated replayed-array" in doc
    assert "Attached conformal model bundle" in doc
    assert "coverage=0.9" in doc
    assert "already materialized coverages" in normalized
    assert "only ``all_predictions=False`` is currently supported" in normalized
    assert "never recalibrates or creates a new guarantee" in normalized
    assert "invalid sidecars fail validation instead of falling back to uncalibrated prediction" in normalized


def test_native_tuning_conformal_guide_documents_typed_passthrough_marker() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "TuningPassthrough()" in text
    assert '{"kind": "passthrough"}' in text
    assert 'kind` must be the literal string `"passthrough"`' in normalized
    assert "only stringify to that value are rejected" in normalized


def test_native_tuning_conformal_guide_documents_sklearn_class_mapping_boundary() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert '"class": "sklearn.preprocessing.StandardScaler"' in text
    assert '"class": "sklearn.linear_model.Ridge"' in text
    assert '"transform": "sklearn.preprocessing.StandardScaler"' in text
    assert '"model": "sklearn.linear_model.Ridge"' in text
    assert '"params": {"with_mean": False}' in text
    assert '"params": {"fit_intercept": False}' in text
    assert 'space={"ridge.alpha": [...]}' in text
    assert 'pipeline={"name": "ridge", "model":' in text
    assert "direct `sklearn.*` string steps" in text
    assert "preprocessing mappings may use" in normalized
    assert "and final model mappings may use" in normalized
    assert "Constructor `params` must use canonical string keys and TCV1-compatible JSON-native values" in normalized
    assert "Python objects, tuples, bytes and non-finite numbers are rejected before import or instantiation" in normalized
    assert "Short aliases and arbitrary imports remain unsupported" in normalized


def test_native_tuning_conformal_guide_documents_typed_dataset_backed_tuning_helpers() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "TuningScoreData(dataset=..., selector=...)" in text
    assert "TuningWinner(dataset=..., selector=...)" in text
    assert "transports `sample_id_column`, `group_column`, and `metadata_columns`" in normalized
    assert "`sample_id_column` and `group_column` must be canonical non-empty strings" in normalized
    assert "`metadata_columns` must be either one canonical string or a duplicate-free sequence of canonical strings" in normalized
    assert "Raw `NativeTuning.score_data` and `NativeTuning.winner` dataset-backed mappings use the same column-selector rule" in normalized
    assert "Dataset-backed `score_data` and `winner` still belong to `run(tuning=...)`" in normalized
    assert "rejects mixed `dataset` + explicit `X`/`y` arrays" in normalized
    assert "rejects the same mixed `dataset` + explicit `X`/`y_true` payloads" in normalized
    assert "Raw mapping payloads follow the same alias exclusivity rules" in normalized
    assert "Raw `winner` mappings follow the same fail-closed rule" in normalized
    assert "The calibration payload cannot provide its own `calibration_data`" in normalized
    assert "passed as top-level `run(..., calibration=...)`" in normalized
    assert "The HPO scoring cohort from `tuning.score_data` is never reused as final calibration evidence" in normalized
    assert "the `tuning.winner` cohort is authoritative" in normalized
    assert "reloading it preserves those intervals and does not recompute them from `tuning.score_data`" in normalized
    assert "calibrated.to_predict_result()" in normalized
    assert "preserving intervals, `calibrated_result_fingerprint`, `calibration_replay_source`, and `tuning_calibration_source`" in normalized
    assert "validates `conformal_guarantee_status` against the embedded conformal artifact" in normalized
    assert "non-materialized coverage selections fail closed rather than displaying a false guarantee" in normalized
    assert "required for every non-empty conformal prediction cohort" in normalized
    assert "non-empty canonical strings, unique physical sample identifiers" in normalized
    assert "empty ids, surrounding whitespace, NUL-containing ids, duplicating a prediction id, or reusing a calibration sample as a prediction row" in normalized
    assert "preserves the original `calibration_replay_source` from the stored calibrated result" in normalized
    assert "caller-supplied `result_metadata` cannot replace that source or the generated `conformal_guarantee_status`" in normalized
    assert "`source_calibrated_result_fingerprint` must match the same value inside `conformal_guarantee_status`" in normalized
    assert "corrupted or mixed application metadata fails closed on reload" in normalized
    assert "Workspace conformal metadata is also validated before insertion" in normalized
    assert "Physical sample id values for calibration and prediction cohorts must be canonical non-empty strings without surrounding whitespace or NULs" in normalized
    assert 'tuning_calibration_source={"source": "tuning.winner"' in normalized
    assert '"score_data_role": "hpo_objective_only"' in normalized
    assert '"score_data_used": false' in normalized
    assert "prediction.tuning_calibration_source" in normalized
    assert "calibrated.tuning_calibration_source" in normalized
    assert "The raw alias `tuning_id` is accepted, but must not be combined with `workspace_tuning_id`" in normalized
    assert "`workspace_metadata` is constrained by the published keyword registry schema to strict JSON-native values with canonical string keys" in normalized
    assert "Robustness workspace metadata follows the same strict JSON-native published schema" in normalized
    assert "Prediction workspace publication applies the same rule to `predict(..., save_to_workspace=True, workspace_metadata=..., workspace_result_metadata=...)`" in normalized
    assert "strict JSON-native result-level metadata for the workspace prediction row" in normalized
    assert "Workspace prediction ids are canonical non-empty strings without surrounding whitespace or NULs" in normalized
    assert "lower-level explicit `prediction_id` writes fail closed on invalid ids instead of silently generating or stringifying a replacement" in normalized
    assert "Workspace link ids stored beside tuning, conformal and robustness rows follow the same rule when supplied" in normalized
    assert "`run_id`, `pipeline_id`, `chain_id`, and source `prediction_id`/`conformal_id`" in normalized
    assert "TuningResult.summary_artifact()" in text
    assert "get_tuning_summary_schema()" in text
    assert "tuning_summary_schema_json()" in text
    assert "get_tuning_space_schema()" in text
    assert "tuning_space_schema_json()" in text
    assert "`SearchSpaceParameter`, `ParameterPatch`, `OrderedSearchSpaceSpec`, `DagMLTuningSpec`, `TrialResult`" in normalized
    assert "canonicalize tuning paths, reject non-TCV1 values, non-finite scores, boolean integers" in normalized
    assert "duplicate trial ids and params outside `tuning.space` before fingerprinting or summary publication" in normalized
    assert "`TuningResult.from_dict()` or `load_json()` uses the same strict numeric boundary for `best_value`" in normalized
    assert "booleans and numeric strings are rejected, not coerced" in normalized
    assert "Dataset-backed tuning selectors reject non-string or whitespace-padded keys plus non-JSON-native values" in normalized
    assert "`TuningWinner.score` must be a finite number, not a boolean or numeric string" in normalized
    assert "`TuningCalibration.as_predict_result` must be a boolean" in normalized
    assert "Final `TuningCalibration.coverage` accepts one numeric coverage or a non-empty unique list of numeric coverages" in normalized
    assert 'list elements such as `"0.8"` fail closed instead of being coerced' in normalized
    assert "Their values must stay strict JSON-native" in normalized
    assert "Non-finite numbers, bytes, tuples, sets and arbitrary Python objects fail before publication" in normalized
    assert "workspace tuning or conformal rows are persisted" in normalized
    assert "the workspace store does not stringify Python objects into metadata" in normalized
    assert "`NativeTuning.space` and `force_params` canonicalize string patch keys before serializing" in normalized
    assert "Raw `NativeTuning(score_data={...})`, `winner={...}` and `calibration={...}` mappings use the same fail-closed boundary" in normalized
    assert "nested `calibration_data` is rejected" in normalized
    assert "Raw calibration `coverage`, `method`, `unit`, `workspace_metadata` and `workspace_conformal_id` are validated and canonicalized" in normalized
    assert "caller-provided ids must be canonical non-empty strings without surrounding whitespace or NULs" in normalized
    assert "Caller-provided conformal ids must be canonical non-empty strings without surrounding whitespace or NULs" in normalized
    assert "Caller-provided robustness ids must also be canonical non-empty strings without surrounding whitespace or NULs" in normalized
    assert "Public `calibrate(..., result_metadata=...)` and `predict_calibrated(..., result_metadata=...)` also require strict JSON-native mappings" in normalized
    assert "Top-level `NativeTuning` core fields are validated through `DagMLTuningSpec` before publication" in normalized
    assert "`n_trials`/`seed`/`resume` reject coercive values" in normalized
    assert "`TuningScoreData.metric`/`score_metric` and raw `NativeTuning.score_data.metric`/`score_metric` must be real non-empty strings" in normalized
    assert "published score metric is lowercase canonical `metric`" in normalized
    assert "tuple/list form is `(X_score, y_score, sample_ids, groups, metadata)` with exactly 2–5 fields and strict metadata keys" in normalized
    assert "Only mapping and tuple/list forms are accepted" in normalized
    assert "`NativeTuning.to_dict()` validates tuple/list `score_data` before publication" in normalized
    assert "metadata must use canonical non-empty string keys in either column-style mapping or row-style sequence form" in normalized
    assert "metadata values must stay strict JSON-native and finite" in normalized
    assert "tuple inputs publish as JSON-native lists" in normalized
    assert "Any other `score_data` type, including scalar strings, bytes, numbers and arbitrary objects, is rejected before publication" in normalized
    assert "`TuningWinner.metric`, `dataset_name`, `model_name` and `task_type` are strict string fields" in normalized
    assert "rejects non-string values, blanks and NULs instead of calling `str(...)`" in normalized
    assert "`metric` and `task_type` are published lowercase" in normalized
    assert "dataset and model labels are published trimmed" in normalized
    assert "raw `NativeTuning.score_data.metadata` follows the same rule" in normalized
    assert "`TuningWinner.metadata` and raw `NativeTuning.winner.metadata` use the same strict metadata key rule" in normalized
    assert "their values must also stay strict JSON-native and finite" in normalized
    assert "safe-to-publish `persistence` block" in normalized
    assert "`optimizer_state_resume_supported`" in normalized
    assert "deliberately omits the raw `storage` URI" in normalized
    assert "Compact diagnostic fields are scalar-only" in normalized
    assert "fail closed instead of being stringified into ambiguous card text" in normalized
    assert "the completed `TuningResult` is written at the optimizer-to-refit boundary" in normalized
    assert "the workspace still preserves the optimizer evidence tape for inspection and resume checks" in normalized
    assert "If every candidate trial fails, no terminal refit is attempted" in normalized
    assert "the all-failed `TuningResult` remains persistable with its trial states and summary artifact" in normalized
    assert "The lightweight `TuningResult.summary_artifact()` mirrors a compact subset of those diagnostics" in normalized
    assert "does not include candidate params or raw exception messages" in normalized
    assert "n4m bindings must expose either `optimizer.tell_result(...)` with a failed trial status or `optimizer.tell(...)`" in normalized
    assert "Older bindings that cannot record failed candidates are rejected fail-closed" in normalized
    assert '`TrialResult(state="PRUNED", value=None)`' in normalized
    assert '`score_extractor="pruned"`' in normalized
    assert "`optimizer.tell_result(..., TrialStatus.PRUNED)`" in normalized
    assert "Older n4m bindings that cannot record a pruned terminal state fail closed" in normalized
    assert "Optuna- or n4m-pruned trials, which stay visible as `PRUNED` rows in `TuningResult.trials`, `trial_states` and compact summary rows" in normalized
    assert "When `resume=True`, the Optuna adapter requires both `storage` and `study_name`" in normalized
    assert "anonymous or in-memory resume is rejected before optimizer execution" in normalized
    assert "If `force_params` is also supplied while resuming a non-empty study" in normalized
    assert "does not enqueue a duplicate trial and fails closed if the caller changes the warm-start assignment" in normalized
    assert "Existing materialized trial params must also match the current `tuning.space` keys exactly" in normalized
    assert "changing the search-space keys under the same persisted study fails closed before optimizer execution" in normalized
    assert "Existing categorical values must also still be present in the current choices for their key" in normalized
    assert "removing or renaming a choice under the same persisted study fails closed before optimizer execution" in normalized
    assert "Existing numeric values must also remain inside the current range for their key" in normalized
    assert "narrowing a range so that a stored trial falls outside it fails closed before optimizer execution" in normalized
    assert "If the current numeric space declares a `step`, restored values must also lie on that grid" in normalized
    assert "During Optuna storage resume, `n_trials` is treated as the target total trial count" in normalized
    assert "with `n_trials=1` runs no extra trial, while `n_trials=2` runs one remaining trial" in normalized
    assert "persist nirs4all study fingerprints in `study.user_attrs`" in normalized
    assert "optimizer contract fingerprint and search-space fingerprint" in normalized
    assert "Resume fails closed if those fingerprints are missing from a non-empty study or no longer match" in normalized
    assert "Trials restored from Optuna storage keep compact summary diagnostics" in normalized
    assert '`score_extractor="optuna_storage"`' in normalized
    assert "Restored Optuna `COMPLETE` rows must carry a finite numeric value" in normalized
    assert "missing or non-finite storage values fail closed" in normalized
    assert "Restored non-`COMPLETE` rows must not carry a final storage value" in normalized
    assert "failed, pruned or in-flight rows with final values are rejected" in normalized
    assert "Restored `RUNNING` rows fail closed during resume" in normalized
    assert "interrupted active trials cannot be safely recovered into a terminal HPO tape" in normalized
    assert "Restored terminal Optuna rows must keep exactly the current `tuning.space` parameter keys" in normalized
    assert "stored parameter table was removed fail closed" in normalized
    assert "Restored queued Optuna `WAITING` rows that already carry materialized params or `fixed_params`" in normalized
    assert "incompatible values are rejected before Optuna can consume them" in normalized
    assert "Restored Optuna trial numbers must be canonical unique integers" in normalized
    assert "non-integer or duplicate trial numbers fails closed" in normalized
    assert '`storage="file:///absolute/checkpoint-dir"`' in normalized
    assert "JSON manifest containing the native checkpoint bytes plus tuning/search-space fingerprints" in normalized
    assert "a checkpoint with one completed trial and `n_trials=2` runs one remaining trial" in normalized
    assert "Restored n4m trial rows are decoded back to the public parameter syntax" in normalized
    assert "ordered canonically by numeric trial id" in normalized
    assert "Duplicate restored trial ids, or restored ids that are not canonical integers, fail closed" in normalized
    assert "Restored `COMPLETE` rows must carry a finite numeric score" in normalized
    assert "missing, boolean or non-finite scores fail closed" in normalized
    assert "optimizer-only categorical labels used for JSON-native values or named `options` are never exposed" in normalized
    assert "The restored checkpoint row params must still match the current `tuning.space` keys and value domains" in normalized
    assert "edited or incompatible checkpoint keys, categorical choices, numeric ranges or numeric steps fail closed" in normalized
    assert '`score_extractor="failed"` for failed candidates' in normalized
    assert '`score_extractor="pruned"` for pruned candidates' in normalized
    assert 'restored as `CANCELLED` with `score_extractor="cancelled"`' in normalized
    assert "resumed summary cards do not collapse these states into a generic checkpoint row" in normalized
    assert "Non-terminal or unsupported checkpoint rows such as `RUNNING` fail closed during resume" in normalized
    assert "Restored n4m non-`COMPLETE` rows must not carry a final score" in normalized
    assert "failed, pruned or cancelled checkpoint rows with scores are rejected" in normalized
    assert "n4m rejects it until native optimizer-state persistence exists" not in normalized
    assert "Metadata keys must be canonical non-empty strings for both column-style mappings and row-style sequences of mappings" in normalized
    assert "raw `score_data.conformal_calibration.metadata` follows the same rule" in normalized


def test_native_tuning_conformal_guide_documents_conformal_replay_aliases() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "`model_bundle`, `predictor_path` and `model_path`" in normalized
    assert "canonical `predictor_bundle`" in normalized
    assert "`workspace_chain_id` as a read alias" in normalized
    assert "canonical `predictor_chain_id`" in normalized
    assert "Both spellings must carry a canonical non-empty chain id without surrounding whitespace or NULs" in normalized
    assert "nirs4all rejects invalid values before calling `predict(chain_id=...)` or publishing replay provenance" in normalized
    assert "`RunResult`/bundle replay remains planned" not in normalized
    assert "a saved predictor bundle, a `RunResult.best`-like prediction entry" in normalized
    assert "or a stored workspace chain plus physical sample ids" in normalized
    assert "Dataset-backed calibration column selectors are canonical too" in normalized
    assert "The same rule applies to `ConformalCalibrationData(...)` and raw `calibration_data={...}` mappings" in normalized
    assert "Raw replayed-array mappings accept the same canonical payload plus the public aliases" in normalized
    assert "`calibrate(calibration_data={...})` mappings for `y_pred_calibration` / `calibration_predictions`" in normalized
    assert "physical sample ids must be explicit through `sample_ids`, `calibration_sample_ids`, `physical_sample_ids`, or `sample_id_column`" in normalized
    assert "Supplying more than one explicit sample-id alias for a dataset-backed calibration mapping is rejected as ambiguous" in normalized
    assert "Optional row-aligned groups and metadata can be supplied explicitly with `groups`/`calibration_groups` and `metadata`/`calibration_metadata`" in normalized


def test_native_tuning_conformal_guide_documents_attached_bundle_prediction_boundary() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "model `.n4a` bundles carrying a conformal sidecar" in normalized
    assert "single selected prediction entry (`all_predictions=False`)" in normalized
    assert "`all_predictions=True` remains fail-closed" in normalized
    assert "`predict()` fails closed and reports the available coverages" in normalized
    assert "fails sidecar validation instead of falling back to an uncalibrated prediction path" in normalized
    assert "without canonical physical `sample_ids`: `predict()` rejects the sidecar before" in normalized


def test_native_docs_describe_dagml_cache_namespace_boundary() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    keywords = docs_root / "reference" / "pipeline_keywords.md"
    normalized = " ".join(f"{guide.read_text(encoding='utf-8')}\n{keywords.read_text(encoding='utf-8')}".split())

    assert "DAG-ML D10 cache namespace proofs" in normalized
    assert "`cache_namespace_fingerprints`" in normalized
    assert "A stale `artifact_fingerprint`, method, unit, multi-target policy, or coverage selection fails closed" in normalized
    assert "a top-level `source_calibrated_result_fingerprint` must also match the same field inside" in normalized
    assert "`conformal_guarantee_status`; mismatches fail closed on construction or reload" in normalized
    assert "When `calibration_replay_source` appears both at top level and inside `conformal_guarantee_status`" in normalized
    assert "prevents a stored result from advertising one replay lane to notebooks and a different lane to Studio or bindings" in normalized
    assert "Guarantee metadata string fields are not coerced" in normalized
    assert "`effective_engine`, `requested_engine`, `source_calibrated_result_fingerprint`, and each `invalidation_reasons` entry" in normalized
    assert "persisted `conformal_guarantee_status.version` must be the strict integer `1`" in normalized
    assert "Status `predictor_fingerprint`, `calibration_data_fingerprint`, `guarantee`, and `scope`" in normalized
    assert "rechecked against the embedded artifact on construction and reload" in normalized
    assert "A persisted status must include the complete generated field set" in normalized
    assert "`status` must be `active` exactly when `invalidation_reasons` is empty, otherwise `invalidated`" in normalized
    assert "The generated `limitations` list must also match the embedded artifact's guarantee mode exactly" in normalized
    assert "edited, shortened, empty or non-string limitation payloads fail closed" in normalized
    assert "requires prediction `sample_ids` for every non-empty prediction cohort" in normalized
    assert "refuses empty, whitespace-padded, non-string or duplicate ids" in normalized
    assert "Result-level metadata must also be strict JSON-compatible" in normalized
    assert "nested non-string mapping keys and tuple values" in normalized
    assert "`ConformalCalibrationSpec` validates direct construction the same way as" in normalized
    assert "coverage values must be real numeric scalars, not booleans or numeric strings" in normalized
    assert "Conformal calibration cohort rows built directly or reloaded from JSON now require" in normalized
    assert "strict JSON-native metadata with string keys, before the cohort manifest can be fingerprinted" in normalized
    assert "optional serialized `n_samples` summary must also be a strict non-boolean integer matching the row count" in normalized
    assert "Reloaded cohort manifest `unit` and calibrated prediction `method`/`unit`" in normalized
    assert "objects with a helpful `__str__` are rejected" in normalized
    assert "Row-aligned calibration metadata supplied as either column mappings or per-row mappings uses the same strict key rule" in normalized
    assert "non-string or whitespace-padded metadata keys fail before they can be coerced into manifest JSON" in normalized
    assert "Dataset-backed selectors reject non-string or whitespace-padded keys plus non-JSON-native values" in normalized
    assert "`include_augmented` must be a boolean" in normalized
    assert "later `predict_calibrated()` application all preserve a required, canonical, unique and disjoint physical-sample boundary" in normalized
    assert "Reload also revalidates the `CalibratedRunResult` identity contract" in normalized
    assert "including required canonical prediction `sample_ids`" in normalized
    assert "stored quantiles must recompute from the retained non-negative residual scores" in normalized
    assert "every materialized interval in `CalibratedRunResult` must equal `y_pred ± qhat`" in normalized
    assert "grouped conformal results use a row-aligned `qhat` vector" in normalized
    assert "strict non-boolean integer grouped `n_samples` summaries" in normalized
    assert "edited intervals, quantiles, group keys or vector `qhat` values is rejected" in normalized
    assert '`group_by="group"` consumes `calibration_groups`/`prediction_groups`' in normalized
    assert "`calibration_metadata`/`prediction_metadata`" in normalized
    assert "Missing, null or unseen prediction groups fail closed without global fallback" in normalized
    assert "Group labels are strict strings: whitespace-padded labels, NUL-containing labels and non-string labels fail before group quantiles are fitted or selected" in normalized
    assert "filesystem stores, workspace `conformal_results` rows, conformal-only `.n4a` bundles and model `.n4a` sidecars preserve `group_keys`" in normalized
    assert "`group_calibrators` and row-aligned grouped `qhat` vectors" in normalized
    assert "revalidate them against the embedded artifact on reload" in normalized
    assert "Conformal numeric arrays (`y_true`, `y_pred`, interval bounds and `qhat`) must contain real numeric values" in normalized
    assert "boolean payloads fail closed instead of being coerced to `0.0`/`1.0`" in normalized
    assert 'numeric strings such as `"1.0"` are rejected instead of being parsed as floats' in normalized
    assert "NumPy boolean scalars in serialized scores or quantiles" in normalized
    assert "serialized numeric fields reject NumPy ndarray scalars instead of coercing them to JSON numbers" in normalized
    assert "Direct `ConformalIntervalBlock` / `CalibratedPredictionBlock` construction is also fail-closed" in normalized
    assert "mismatched coverage keys, mismatched interval shapes, inverted bounds, invalid method/unit values" in normalized
    assert "negative or non-row-aligned `qhat` values" in normalized
    assert "`SplitConformalCalibrator` construction validates retained residual scores" in normalized
    assert "negative scores, edited `qhat` values or unsupported vocabulary fail closed" in normalized
    assert "Version fields on conformal cohort manifests, calibration artifacts and calibrated results are strict integer contract tags" in normalized
    assert "boolean `true`/`false` and numeric strings are rejected on direct construction and reload" in normalized
    assert "Optional conformal artifact identity strings (`target_name`, `predictor_fingerprint`, `calibration_data_fingerprint`)" in normalized
    assert "either `None` or non-empty strings without surrounding whitespace or NULs" in normalized
    assert "`observed_coverage` must be finite in `[0, 1]` and match `n_covered / n_samples`" in normalized
    assert "`coverage_gap` must equal `observed_coverage - coverage`" in normalized
    assert "widths/interval scores must be non-negative or positive infinity for unbounded intervals" in normalized
    assert "Applying the calibrator preserves the source result's `calibration_replay_source`" in normalized
    assert "user metadata cannot replace that provenance or the generated guarantee status" in normalized
    assert "nirs4all treats them as signed DAG-ML control plane data and forwards them unchanged" in normalized
    assert "DAG-ML owns validation, namespace-aware handle derivation" in normalized
    assert "file-store payload naming and columnar manifest exposure" in normalized
    assert "There is no nirs4all keyword for mutating this proof after signing" in normalized


def test_public_reference_docs_describe_composite_tuning_conformal_accessors() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    module_api = docs_root / "api" / "module_api.md"
    cli = docs_root / "reference" / "cli.md"
    predictions_api = docs_root / "reference" / "predictions_api.md"
    public_interfaces = docs_root / "reference" / "public_interfaces.md"
    text = "\n".join(
        [
            cli.read_text(encoding="utf-8"),
            module_api.read_text(encoding="utf-8"),
            predictions_api.read_text(encoding="utf-8"),
            public_interfaces.read_text(encoding="utf-8"),
        ]
    )
    normalized = " ".join(text.split())

    assert "TunedSingleEstimatorConformalResult" in text
    assert "result.interval(0.9)" in text
    assert "result.metrics(y_true_prediction)" in text
    assert "result.robustness(y_true=y_true_prediction)" in text
    assert "NativeTuning(..., force_params=...)" in text
    assert "public decoded parameter values" in normalized
    assert "can change trial order and selected predictor" in normalized
    assert "can change the selected predictor and therefore stale any previous calibration" in normalized
    assert "coverage=None" in text
    assert "Select materialized conformal interval coverage(s)" in text
    assert "Select already-materialized conformal intervals" in text
    assert "With a conformal sidecar and `coverage=...`, only `all_predictions=False` is currently supported" in normalized
    assert "`all_predictions=True` remains fail-closed until every returned prediction entry can carry calibrated identity mapping" in normalized
    assert "invalid `conformal/` sidecar, prediction fails validation instead of falling back to an uncalibrated path" in normalized
    assert "non-empty predictions but no canonical physical `sample_ids` is also rejected before the raw model prediction runs" in normalized
    assert "missing canonical physical `sample_ids`" in normalized
    assert "do not refit, recalibrate, recompute intervals, or infer a new guarantee" in normalized
    assert "`TunedSingleEstimatorConformalResult.interval(...)` / `metrics(...)` / `robustness(...)`" in normalized
    assert "`conformal_guarantee_status` | dict or None | Fail-loud guarantee metadata" in normalized
    assert "`calibration_replay_source` | dict or None | Direct accessor" in normalized
    assert "`tuning_calibration_source` | dict or None | Direct accessor" in normalized
    assert "`spectral_replay_evidence_status` | dict | Fail-closed readiness diagnostic" in normalized
    assert "`has_executable_X_or_spectra` for finite 2D arrays" in normalized
    assert "row count matches `len(prediction)`" in normalized
    assert "calibration_replay_source" in normalized
    assert "prediction.calibration_replay_source" in normalized
    assert "fail closed if both locations carry `calibration_replay_source` but disagree" in normalized
    assert "cannot observe contradictory replay provenance" in normalized
    assert "prediction.tuning_calibration_source" in normalized
    assert "load_workspace_calibrated_result(...)" in normalized
    assert "load_workspace_calibrated_predict_result(...)" in normalized
    assert "load_workspace_predict_result" in normalized
    assert "load_workspace_predict_results" in normalized
    assert "whose non-empty prediction cohort lacks canonical physical `sample_ids` fails reload" in normalized
    assert "A corrupted `conformal_results` workspace row with the same missing prediction identity is rejected" in normalized
    assert "save_workspace_predict_result" in normalized
    assert "predict(..., save_to_workspace=True, workspace_result_metadata=...)" in normalized
    assert "robustness_from_workspace_prediction" in normalized
    assert "workspace robustness from-prediction" in normalized
    assert "--scenarios-json" in normalized
    assert "Predictions.add_prediction(..., X=..., spectra=..., result_metadata=...)" in normalized
    assert "Predictions.merge_stores(...)" in normalized
    assert "calibrated.to_predict_result()" in normalized
    assert "`PredictResult` with intervals and provenance accessors" in normalized
    assert "workspace conformal show --as-predict-result --json" in normalized
    assert "`tuning_calibration_source`. Use this form for bindings, Studio and notebook diagnostics" in normalized
    assert "`predictor.predict`, or `nirs4all.predict`" in normalized
    assert "str(result)` also reports the guarantee status, effective engine and selected coverages" in normalized
    assert "Treat this as the supported bridge from workspace/store predictions into the native conformal and robustness API" in normalized
    assert "Direct public one-record shortcut" in normalized
    assert "Direct public bulk shortcut" in normalized
    assert "rather than scraping sidecar rows or synthesizing missing replay inputs" in normalized


def test_cli_reference_documents_keyword_registry_force_params_export() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    cli = docs_root / "reference" / "cli.md"
    text = cli.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "nirs4all keyword-registry --output artifacts/keyword-registry.json" in text
    assert "nirs4all tuning-space --input tuning.json --output artifacts/tuning-space.json" in text
    assert "nirs4all tuning-space --schema --output artifacts/tuning-space.schema.json" in text
    assert "`run.tuning.force_params`" in text
    assert "`changes` field records `trial_sequence`, `candidate_fit`, and `selection`" in normalized
    assert "strict JSON-native schemas for tuning, conformal, prediction, robustness and workspace metadata" in normalized
    assert "`nirs4all.tuning.ordered_search_space` artifact" in normalized
    assert "`nirs4all.inspect_tuning_space(...)`" in normalized
    assert "`tuning-space --schema`" in normalized
    assert "does not execute a pipeline" in normalized


def test_native_tuning_conformal_guide_documents_all_supported_robustness_scenarios() -> None:
    docs_root = Path(__file__).resolve().parents[3] / "docs" / "source"
    guide = docs_root / "user_guide" / "models" / "native_tuning_conformal.md"
    text = guide.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for kind in (
        "observed",
        "prediction_bias",
        "prediction_noise",
        "spectral_noise",
        "spectral_offset",
        "spectral_scale",
        "spectral_slope",
        "spectral_shift",
    ):
        assert f"`{kind}`" in normalized
    assert "`CONFORMAL_CALIBRATION_METHODS`, `CONFORMAL_CALIBRATION_UNITS`" in normalized
    assert "`CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES`, `ConformalMethod`" in normalized
    assert "`ConformalUnit`, and `ConformalMultiTarget`" in normalized
    assert "The executable V1 policies are `marginal` for one-dimensional targets and `joint_max`" in normalized
    assert '`multi_target="joint_max"` is executable on two-dimensional replayed arrays' in normalized
    assert "`max(abs(y_true - y_pred))` across target columns" in normalized
    assert "simultaneous for the full target vector" in normalized
    assert "`TUNING_ENGINES`, `TUNING_DIRECTIONS`, `TUNING_CONTRACT_KEYS`" in normalized
    assert "`inspect_tuning_space(...)`, `OrderedSearchSpaceSpec`" in normalized
    assert "`SearchSpaceParameter`, and `ParameterPatch`" in normalized
    assert "`TUNING_SPACE_SCHEMA_ID`, `get_tuning_space_schema()`" in normalized
    assert "`tuning_space_schema_json()`" in normalized
    assert "`nirs4all.tuning.ordered_search_space` artifact" in normalized
    assert "`force_params` is inside that deterministic contract" in normalized
    assert "`run.tuning.force_params`" in normalized
    assert "`TUNING_OPTIMIZER_PERSISTENCE_KEYS`, `TUNING_RUNTIME_KEYS`" in normalized
    assert "`CONFORMAL_TUNING_SCORE_METRICS`, `TuningEngine`" in normalized
    assert "`TuningDirection`: public discovery and typing helpers" in normalized
    assert "`FINETUNE_ENGINES`, `FINETUNE_APPROACHES`, `FINETUNE_EVAL_MODES`" in normalized
    assert "`FINETUNE_OPTUNA_SAMPLERS`, `FINETUNE_OPTUNA_PRUNERS`" in normalized
    assert "`FINETUNE_N4M_SAMPLERS`, `FINETUNE_N4M_PRUNERS`" in normalized
    assert "`FINETUNE_DAGML_DETERMINISTIC_ENGINES`, `FINETUNE_DAGML_META_KEYS`" in normalized
    assert "`FinetuneApproach`, and `FinetuneEvalMode`: public discovery and typing" in normalized
    assert "`ROBUSTNESS_MODES`, `ROBUSTNESS_EXECUTABLE_MODES`, and `RobustnessMode`" in normalized
    assert "`ROBUSTNESS_SCENARIO_KINDS` / `RobustnessScenarioKind`" in normalized
    assert "`ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS`" in normalized
    assert "`ROBUSTNESS_SCENARIO_DISTRIBUTIONS`" in normalized
    assert "`RobustnessScenarioDistribution`" in normalized
    assert "with `normal` and centered `uniform` noise supported fail-closed" in normalized
    assert "`uniform` samples from `[-severity, +severity]`" in normalized
    assert "runtime, report metadata, and keyword registry" in normalized
    assert "explicit-X spectral cells require `X` and exactly one frozen replay source" in normalized
    assert "`predictor` and `predictor_bundle` are mutually exclusive" in normalized
    assert "nirs4all.predict(model=predictor_bundle" in normalized
    assert '`metadata["spectral_replay"]`' in normalized
    assert "confirm the `nirs4all.predict` route" in normalized
    assert '`metadata["X"]` or' in normalized
    assert "`result.robustness(...)` use those values as the spectral replay defaults" in normalized
    assert '"prediction_arrays.X"` are provenance markers and remain fail-closed' in normalized
    assert "When `force_params` targets such a named categorical choice" in normalized
    assert "Optuna and n4m may encode it to a stable backend label internally" in normalized
    assert "publish the decoded value back in `TrialResult.params` and `best_params`" in normalized
    assert "reject internal labels or unknown keys instead of silently changing the warm-start trial" in normalized
    assert "The internal categorical codec is also fail-closed on direct construction" in normalized
    assert "decoder keys must match encoded choices and decoded public values must be unique" in normalized
    assert "Direct `RobustnessScenarioSpec` construction is fail-closed" in normalized
    assert "`kind` and `distribution` must be real strings" in normalized
    assert "not host objects that stringify to supported values" in normalized
    assert "`severity` must be a real numeric scalar, not a boolean or numeric string" in normalized
    assert "`extra` keys must be canonical non-empty strings without NULs before `to_dict()` can publish" in normalized
    assert "Scenario mappings, including `extra={...}` on the typed helper, use the same strict boundary" in normalized
    assert "keys must be canonical non-empty strings without NULs, `kind` and `distribution` must be real strings" in normalized
    assert "`severity` must be a real finite numeric scalar" in normalized
    assert "The same no-stringify rule is applied again to published result payloads" in normalized
    assert "`RobustnessScenarioResult.scenario` and `RobustnessSliceResult.slice_key` keys must be canonical non-empty strings without NULs" in normalized
    assert "These mappings enter report JSON, fingerprints, `summary_rows()`, `degradation_rows()`, `worst_slices()`, `tabular_records()` and Studio/CI summary cards" in normalized
    assert "lowered to an ordered search-space contract" in normalized
    assert "sklearn double-underscore paths are canonicalized to the same dotted spelling" in normalized
    assert "ordered `ParameterPatch` records with canonical string paths and JSON-native values" in normalized
    assert "`search_space_fingerprint`" in normalized
    assert "The legacy Optuna and n4m managers apply the same canonicalization to `approach`, `eval_mode`, and explicit `direction`" in normalized
    assert 'including the `sample` alias for `sampler` and the `eval_mode="avg"` alias for `eval_mode="mean"`' in normalized
    assert "Optuna also canonicalizes per-phase `sampler` values before multi-phase dispatch" in normalized
    assert "Unknown sampler or pruner spellings are rejected before optimizer construction" in normalized
    assert "they do not silently fall back to TPE or to a no-pruner study" in normalized
    assert 'adapter records a distinct `TrialResult(state="PRUNED", value=None, diagnostics={...})`' in normalized
    assert '`score_extractor="pruned"` instead of collapsing it into `FAIL`' in normalized
    assert "`optimizer.tell_result(..., TrialStatus.PRUNED)`" in normalized
    assert "Older n4m bindings that cannot record a pruned terminal state fail closed" in normalized
    assert "must stay TCV1 JSON-native and fingerprintable" in normalized
    assert "Python objects, NaN/Infinity, bytes, and other opaque values are rejected" in normalized
    assert "requested/effective engine, method, unit, selected and calibrated coverages" in normalized
    assert "does not renew the guarantee" in normalized
    assert "summary includes conformal guarantee status details" in normalized
    assert "selected and calibrated coverages, scope, invalidation reasons and limitations" in normalized
    assert "optional `conformal_guarantee_status`, optional `spectral_replay`, and compact `summary_rows()`" in normalized
    assert "report metadata enters `RobustnessReport.fingerprint`" in normalized
    assert "canonical non-empty strings without NULs and its values must stay TCV1 JSON-native" in normalized
    assert "`summary_rows(reference=0)` returns one compact row per scenario" in normalized
    assert "`execution_scope` is `baseline`, `prediction_replay` or `spectral_replay`" in normalized
    assert "`requires_spectral_replay=true` marks explicit-X spectral/OOD replay rows" in normalized
    assert "Markdown and HTML exports render this same compact scenario summary, including scope and replay-evidence columns" in normalized
    assert "replay source, bundle path and sample-id forwarding status" in normalized
    assert "engine, coverage, invalidation details, replay source, bundle path and sample-id forwarding status" in normalized
    assert "Matched recalibration and structural refit remain planned gates" in normalized
