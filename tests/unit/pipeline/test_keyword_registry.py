"""Contract tests for the descriptive lifecycle keyword registry."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

import nirs4all
from nirs4all.pipeline import get_keyword_registry as public_get_keyword_registry
from nirs4all.pipeline.keyword_registry import (
    KEYWORD_REGISTRY_SCHEMA_ID,
    KEYWORD_REGISTRY_SCHEMA_VERSION,
    KEYWORD_REGISTRY_VERSION,
    KeywordEntry,
    get_keyword_registry,
    get_keyword_registry_schema,
    keyword_registry_json,
    keyword_registry_schema_json,
)
from nirs4all.pipeline.steps.parser import StepParser

_REQUIRED_ENTRY_FIELDS = {
    "id",
    "token",
    "path",
    "surface",
    "scope",
    "canonical_term",
    "status",
    "value_schema",
    "aliases",
    "lifecycle_stage",
    "reads",
    "changes",
    "invalidates_calibration",
    "engine_support",
    "summary",
    "docs_anchor",
    "ui",
}

_VALID_STATUSES = {"supported", "partial", "planned"}
_VALID_SURFACES = {
    "run_argument",
    "pipeline_step",
    "nested_key",
    "enum_value",
    "calibrate_argument",
    "predict_argument",
    "predict_calibrated_argument",
    "robustness_argument",
}
_VALID_INVALIDATIONS = {
    "always",
    "if_predictor_changes",
    "replaces_existing",
    "extends_existing",
    "mode_dependent",
    "not_applicable",
}
_VALID_ENGINE_SUPPORT = {"supported", "partial", "planned", "unsupported", "legacy_fallback", "not_applicable"}
EXPECTED_PUBLISHED_REGISTRY_SHA256 = "38997cf10c2629259b64d7f3fcbeff589680e6585ccbbfdfbe711a3fde5eaf12"


def _entries_by_id() -> dict[str, KeywordEntry]:
    return {entry["id"]: entry for entry in get_keyword_registry()["entries"]}


def test_registry_document_has_versioned_stable_shape() -> None:
    registry = get_keyword_registry()

    assert registry["schema_id"] == KEYWORD_REGISTRY_SCHEMA_ID
    assert registry["schema_version"] == KEYWORD_REGISTRY_SCHEMA_VERSION == 1
    assert registry["registry_version"] == KEYWORD_REGISTRY_VERSION == "1.0.0"
    assert registry["scope"] == "lifecycle-v1"
    assert registry["entries"]
    assert public_get_keyword_registry() == registry


def test_registry_entries_have_unique_ids_and_valid_closed_fields() -> None:
    entries = get_keyword_registry()["entries"]
    ids = [entry["id"] for entry in entries]
    ui_orders = [entry["ui"]["order"] for entry in entries]

    assert len(ids) == len(set(ids))
    assert len(ui_orders) == len(set(ui_orders))
    assert ui_orders == sorted(ui_orders)

    for entry in entries:
        assert set(entry) == _REQUIRED_ENTRY_FIELDS
        assert entry["status"] in _VALID_STATUSES
        assert entry["surface"] in _VALID_SURFACES
        assert entry["invalidates_calibration"] in _VALID_INVALIDATIONS
        assert entry["summary"].strip()
        assert entry["docs_anchor"].strip()
        assert set(entry["engine_support"].values()) <= _VALID_ENGINE_SUPPORT
        for alias in entry["aliases"]:
            assert set(alias) == {"kind", "name", "canonical", "mode"}
            assert alias["kind"] in {"token", "value"}
            assert alias["mode"] == "read_only"


def test_registry_json_is_deterministic_round_trippable_and_isolated() -> None:
    import nirs4all

    pretty_first = keyword_registry_json()
    pretty_second = keyword_registry_json()
    compact = keyword_registry_json(indent=None)

    assert pretty_first == pretty_second
    assert pretty_first.endswith("\n")
    assert compact.endswith("\n")
    assert json.loads(pretty_first) == get_keyword_registry()
    assert json.loads(compact) == get_keyword_registry()
    assert nirs4all.get_keyword_registry() == get_keyword_registry()
    assert json.loads(nirs4all.keyword_registry_json(indent=None)) == get_keyword_registry()
    assert nirs4all.get_keyword_registry_schema() == get_keyword_registry_schema()
    assert json.loads(nirs4all.keyword_registry_schema_json(indent=None)) == get_keyword_registry_schema()

    mutated = get_keyword_registry()
    mutated["entries"][0]["summary"] = "changed by consumer"
    assert get_keyword_registry()["entries"][0]["summary"] != "changed by consumer"
    top_level = nirs4all.get_keyword_registry()
    top_level["entries"][0]["summary"] = "changed through top-level export"
    assert nirs4all.get_keyword_registry()["entries"][0]["summary"] != "changed through top-level export"


def test_published_keyword_registry_json_contract_is_explicit() -> None:
    """Published registry artifact drift must be intentional for Studio/Web."""

    payload = keyword_registry_json(indent=2).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_PUBLISHED_REGISTRY_SHA256


def test_keyword_registry_schema_validates_published_registry() -> None:
    """The exported JSON Schema validates the published registry artifact."""

    schema = get_keyword_registry_schema()
    registry = get_keyword_registry()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(registry, schema)
    assert json.loads(keyword_registry_schema_json(indent=None)) == schema


def test_supported_model_step_metadata_remains_reserved_without_driving_parser() -> None:
    entries = get_keyword_registry()["entries"]
    described_reserved_tokens = {entry["token"] for entry in entries if entry["surface"] == "pipeline_step" and entry["status"] == "supported"}

    assert described_reserved_tokens == {"finetune_params", "train_params", "refit_params"}
    assert described_reserved_tokens <= set(StepParser.RESERVED_KEYWORDS)


def test_tuning_continuation_and_training_parameter_scopes_are_unambiguous() -> None:
    entries = _entries_by_id()

    model_hpo = entries["pipeline.step.finetune_params"]
    dag_hpo = entries["run.tuning"]
    continuation = entries["retrain.mode.finetune"]
    trial_params = entries["pipeline.step.finetune_params.train_params"]
    training_params = entries["pipeline.step.train_params"]
    refit_params = entries["pipeline.step.refit_params"]

    assert model_hpo["canonical_term"] == dag_hpo["canonical_term"] == "hyperparameter_tuning"
    assert model_hpo["scope"] == "model_local"
    assert dag_hpo["scope"] == "full_dag"
    assert dag_hpo["status"] == "partial"
    assert dag_hpo["engine_support"]["dag-ml"] == "partial"
    assert continuation["canonical_term"] == "continuation_training"
    assert continuation["status"] == "partial"
    assert continuation["engine_support"] == {"legacy": "partial", "dag-ml": "unsupported"}
    assert trial_params["lifecycle_stage"] == "trial_fit"
    assert training_params["lifecycle_stage"] == "fit"
    assert refit_params["lifecycle_stage"] == "refit"


def test_execution_engine_and_optimizer_engines_are_distinct() -> None:
    entries = _entries_by_id()

    execution_engine = entries["run.engine"]
    local_driver = entries["pipeline.step.finetune_params.engine"]
    dag_optimizer = entries["run.tuning.engine"]

    assert execution_engine["canonical_term"] == "execution_backend"
    assert execution_engine["engine_support"]["dag-ml"] == "partial"
    assert local_driver["canonical_term"] == "hpo_or_generation_driver"
    assert dag_optimizer["canonical_term"] == "optimizer_engine"
    assert execution_engine["value_schema"]["enum"] == [None, "legacy", "dag-ml", "dual"]
    assert execution_engine["engine_support"]["dual"] == "planned"
    assert local_driver["value_schema"]["enum"] == ["optuna", "n4m", "dag-ml", "grid"]
    assert local_driver["status"] == "partial"
    assert local_driver["engine_support"]["n4m"] == "partial"
    assert local_driver["engine_support"]["dag-ml"] == "partial"
    assert "never selects the run() execution backend" in local_driver["summary"]
    assert dag_optimizer["status"] == "partial"
    assert dag_optimizer["engine_support"]["optuna"] == "supported"


def test_public_run_tuning_subset_runtime_keys_are_registered_without_overclaiming() -> None:
    entries = _entries_by_id()

    assert entries["run.tuning.score_data"]["status"] == "partial"
    assert entries["run.tuning.score_data"]["changes"] == ["objective_scores", "trial_ranking", "selected_predictor"]
    assert "mandatory in the public subset" in entries["run.tuning.score_data"]["summary"]
    assert "DatasetConfigs" in entries["run.tuning.score_data"]["summary"]
    score_data_schema = entries["run.tuning.score_data"]["value_schema"]
    assert score_data_schema["oneOf"][0]["anyOf"] == [
        {"required": ["X", "y"]},
        {"required": ["X_score", "y_score"]},
        {"required": ["dataset", "selector"]},
    ]
    score_selector_schema = score_data_schema["oneOf"][0]["properties"]["selector"]
    assert score_selector_schema["$ref"] == "#/$defs/json_native_mapping"
    assert score_selector_schema["x-nirs4all-json-native"] is True
    jsonschema.validate({"partition": "score", "nested": {"ok": [1, True, None]}}, score_selector_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, score_selector_schema)
    assert score_data_schema["oneOf"][0]["properties"]["conformal_coverage"] == {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
    conformal_score_schema = score_data_schema["oneOf"][0]["properties"]["conformal_calibration"]
    assert conformal_score_schema["allOf"][0]["oneOf"] == [{"required": ["X"]}, {"required": ["X_calibration"]}, {"required": ["features"]}]
    assert conformal_score_schema["allOf"][1]["oneOf"] == [
        {"required": ["y_true"]},
        {"required": ["y"]},
        {"required": ["y_calibration"]},
        {"required": ["target"]},
        {"required": ["targets"]},
    ]
    assert conformal_score_schema["properties"]["calibration_sample_ids"] == {"type": "array"}
    assert conformal_score_schema["properties"]["physical_sample_ids"] == {"type": "array"}
    assert conformal_score_schema["properties"]["metadata"] == {"$ref": "#/$defs/json_native_metadata"}
    assert score_data_schema["oneOf"][0]["properties"]["conformal_score_calibration"] == conformal_score_schema
    assert score_data_schema["oneOf"][1] == {"type": "array", "minItems": 2, "maxItems": 5}
    assert entries["run.tuning.score_data.conformal_calibration"]["changes"] == ["objective_scores", "trial_ranking"]
    assert entries["run.tuning.score_data.conformal_calibration"]["value_schema"] == conformal_score_schema
    assert "temporary calibrator per candidate" in entries["run.tuning.score_data.conformal_calibration"]["summary"]
    assert "never replaces the final run(calibration=...) result" in entries["run.tuning.score_data.conformal_calibration"]["summary"]
    assert entries["run.tuning.score_data.conformal_coverage"]["value_schema"] == {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
    assert entries["run.tuning.score_data.conformal_coverage"]["aliases"] == [{"kind": "token", "name": "coverage", "canonical": "conformal_coverage", "mode": "read_only"}]
    assert entries["run.tuning.score_data.conformal_coverage"]["changes"] == ["objective_scores", "trial_ranking"]
    assert "requires score_data.conformal_calibration" in entries["run.tuning.score_data.conformal_coverage"]["summary"]
    assert "not the final run(calibration=...) result" in entries["run.tuning.score_data.conformal_coverage"]["summary"]
    assert entries["run.tuning.storage"]["lifecycle_stage"] == "storage"
    assert entries["run.tuning.storage"]["value_schema"] == {"type": "string", "minLength": 1, "pattern": "^[A-Za-z][A-Za-z0-9+.-]*://"}
    assert entries["run.tuning.storage"]["engine_support"]["optuna"] == "supported"
    assert entries["run.tuning.storage"]["engine_support"]["n4m"] == "partial"
    assert "file:///absolute/checkpoint-dir" in entries["run.tuning.storage"]["summary"]
    assert "N4MOPT" in entries["run.tuning.storage"]["summary"]
    assert entries["run.tuning.study_name"]["lifecycle_stage"] == "storage"
    assert entries["run.tuning.study_name"]["value_schema"] == {"type": "string", "minLength": 1, "pattern": "^[^\\u0000]+$"}
    assert entries["run.tuning.study_name"]["engine_support"]["optuna"] == "supported"
    assert entries["run.tuning.study_name"]["engine_support"]["n4m"] == "partial"
    assert "NUL characters are rejected" in entries["run.tuning.study_name"]["summary"]
    assert "filename-safe" in entries["run.tuning.study_name"]["summary"]
    assert entries["run.tuning.winner"]["lifecycle_stage"] == "refit"
    assert entries["run.tuning.winner"]["invalidates_calibration"] == "replaces_existing"
    assert "required before run(tuning).calibration" in entries["run.tuning.winner"]["summary"]
    assert "config/path string" in entries["run.tuning.winner"]["summary"]
    winner_schema = entries["run.tuning.winner"]["value_schema"]
    assert winner_schema["anyOf"] == [{"required": ["X", "y_true"]}, {"required": ["dataset", "selector"]}]
    assert winner_schema["properties"]["prediction_sample_ids"] == {"type": "array"}
    assert winner_schema["properties"]["physical_sample_ids"] == {"type": "array"}
    assert winner_schema["properties"]["winner_sample_ids"] == {"type": "array"}
    assert winner_schema["properties"]["sample_id_column"] == {"type": "string", "minLength": 1}
    winner_selector_schema = winner_schema["properties"]["selector"]
    assert winner_selector_schema["$ref"] == "#/$defs/json_native_mapping"
    assert winner_selector_schema["x-nirs4all-json-native"] is True
    jsonschema.validate({"partition": "winner", "nested": {"ok": [1, True, None]}}, winner_selector_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, winner_selector_schema)
    assert winner_schema["properties"]["metadata"] == {"$ref": "#/$defs/json_native_metadata"}
    assert entries["run.tuning.calibration"]["lifecycle_stage"] == "calibration"
    assert entries["run.tuning.calibration"]["changes"] == ["calibrator", "calibrated_result"]
    assert "calibration_data is derived from winner" in entries["run.tuning.calibration"]["summary"]
    calibration_schema = entries["run.tuning.calibration"]["value_schema"]
    assert calibration_schema["required"] == ["y_pred", "prediction_sample_ids"]
    assert calibration_schema["properties"]["calibration_data"] is False
    assert calibration_schema["properties"]["coverage"]["oneOf"][1]["uniqueItems"] is True
    assert calibration_schema["properties"]["method"] == {"const": "split_absolute_residual"}
    assert calibration_schema["properties"]["unit"] == {"const": "physical_sample"}
    assert calibration_schema["properties"]["workspace_metadata"] == {"$ref": "#/$defs/json_native_mapping"}
    assert calibration_schema["additionalProperties"] == {"$ref": "#/$defs/json_native_value"}
    assert entries["run.calibration"]["surface"] == "run_argument"
    assert entries["run.calibration"]["status"] == "partial"
    assert entries["run.calibration"]["changes"] == ["calibrator", "calibrated_result"]
    assert "Top-level alias for run(tuning).calibration" in entries["run.calibration"]["summary"]
    assert entries["run.tuning.workspace_tuning_id"]["aliases"] == [{"kind": "token", "name": "tuning_id", "canonical": "workspace_tuning_id", "mode": "read_only"}]
    assert entries["run.tuning.workspace_metadata"]["invalidates_calibration"] == "not_applicable"
    assert entries["run.tuning.workspace_metadata"]["value_schema"]["$ref"] == "#/$defs/json_native_mapping"
    assert "strict JSON-native metadata" in entries["run.tuning.workspace_metadata"]["summary"]
    assert entries["run.tuning.resume"]["value_schema"] == {"type": "boolean"}
    assert "not interrupted optimizer checkpoint resume" in entries["run.tuning.resume"]["summary"]
    assert entries["run.tuning.calibration.workspace_conformal_id"]["changes"] == ["workspace_conformal_results"]


def test_tuning_metadata_registry_schemas_are_json_native_and_fail_closed() -> None:
    entries = _entries_by_id()
    valid_metadata = {"site": "north", "nested": {"ok": [1, True, None]}}

    score_data_schema = entries["run.tuning.score_data"]["value_schema"]
    winner_schema = entries["run.tuning.winner"]["value_schema"]
    conformal_score_schema = entries["run.tuning.score_data.conformal_calibration"]["value_schema"]
    calibration_schema = entries["run.tuning.calibration"]["value_schema"]
    workspace_metadata_schema = entries["run.tuning.workspace_metadata"]["value_schema"]

    assert score_data_schema["$defs"]["json_native_value"]["x-nirs4all-json-native"] is True
    assert score_data_schema["$defs"]["json_native_value"]["x-nirs4all-finite-numbers"] is True
    assert workspace_metadata_schema["x-nirs4all-json-native"] is True
    assert workspace_metadata_schema["x-nirs4all-finite-numbers"] is True

    jsonschema.validate({"X": [[1.0]], "y": [1.0], "metadata": valid_metadata}, score_data_schema)
    jsonschema.validate({"X": [[1.0]], "y": [1.0], "score_metadata": [valid_metadata]}, score_data_schema)
    jsonschema.validate({"X": [[1.0]], "y_true": [1.0], "metadata": [valid_metadata]}, winner_schema)
    jsonschema.validate({"X": [[1.0]], "y_true": [1.0], "metadata": valid_metadata}, conformal_score_schema)
    jsonschema.validate({"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "workspace_metadata": valid_metadata, "target_name": "protein"}, calibration_schema)
    jsonschema.validate(valid_metadata, workspace_metadata_schema)

    for schema, payload in (
        (score_data_schema, {"X": [[1.0]], "y": [1.0], "metadata": {" bad": 1}}),
        (score_data_schema, {"X": [[1.0]], "y": [1.0], "score_metadata": [{"bad": object()}]}),
        (winner_schema, {"X": [[1.0]], "y_true": [1.0], "metadata": {"bad": object()}}),
        (conformal_score_schema, {"X": [[1.0]], "y_true": [1.0], "metadata": {"bad": object()}}),
        (calibration_schema, {"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "workspace_metadata": {"bad": object()}}),
        (calibration_schema, {"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "bad_extra": object()}),
        (workspace_metadata_schema, {"bad": object()}),
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)


def test_legacy_hpo_aliases_are_read_only_and_emit_canonical_terms() -> None:
    entries = _entries_by_id()

    engine_aliases = entries["pipeline.step.finetune_params.engine"]["aliases"]
    sampler_alias = entries["pipeline.step.finetune_params.sampler"]["aliases"]
    eval_alias = entries["pipeline.step.finetune_params.eval_mode"]["aliases"]

    assert {"kind": "value", "name": "native", "canonical": "dag-ml", "mode": "read_only"} in engine_aliases
    assert {"kind": "value", "name": "methods", "canonical": "n4m", "mode": "read_only"} in engine_aliases
    assert sampler_alias == [{"kind": "token", "name": "sample", "canonical": "sampler", "mode": "read_only"}]
    assert eval_alias == [{"kind": "value", "name": "avg", "canonical": "mean", "mode": "read_only"}]


def test_model_local_finetune_keyword_controls_are_machine_readable() -> None:
    entries = _entries_by_id()

    assert entries["pipeline.step.finetune_params.model_params"]["value_schema"] == {"type": "object"}
    assert entries["pipeline.step.finetune_params.model_params"]["invalidates_calibration"] == "always"
    assert entries["pipeline.step.finetune_params.model_params"]["engine_support"] == {
        "optuna": "supported",
        "n4m": "supported",
        "dag-ml": "partial",
    }
    assert "_range_/_log_range_" in entries["pipeline.step.finetune_params.model_params"]["summary"]
    assert entries["pipeline.step.finetune_params.metric"]["canonical_term"] == "selection_metric"
    assert entries["pipeline.step.finetune_params.metric"]["value_schema"] == {"type": "string", "minLength": 1}
    assert "rmse, accuracy and balanced_accuracy" in entries["pipeline.step.finetune_params.metric"]["summary"]
    assert entries["pipeline.step.finetune_params.direction"]["value_schema"]["enum"] == ["minimize", "maximize"]
    assert "contradict the selected metric" in entries["pipeline.step.finetune_params.direction"]["summary"]
    assert entries["pipeline.step.finetune_params.n_trials"]["value_schema"] == {"type": "integer", "minimum": 1}
    assert entries["pipeline.step.finetune_params.n_trials"]["engine_support"]["dag-ml"] == "unsupported"
    assert entries["pipeline.step.finetune_params.approach"]["value_schema"]["enum"] == ["single", "grouped", "individual"]
    assert entries["pipeline.step.finetune_params.approach"]["engine_support"]["dag-ml"] == "partial"
    assert entries["pipeline.step.finetune_params.pruner"]["value_schema"]["enum"] == [
        "none",
        "median",
        "successive_halving",
        "hyperband",
        "asha",
        "racing",
    ]
    assert entries["pipeline.step.finetune_params.pruner"]["engine_support"]["dag-ml"] == "unsupported"


def test_partial_hpo_behaviors_and_dag_training_limits_are_not_overclaimed() -> None:
    entries = _entries_by_id()

    assert entries["pipeline.step.finetune_params"]["engine_support"]["dag-ml"] == "partial"
    assert "deterministic model_params grids/ranges" in entries["pipeline.step.finetune_params"]["summary"]
    assert entries["pipeline.step.finetune_params.n_trials"]["engine_support"]["dag-ml"] == "unsupported"
    assert entries["pipeline.step.finetune_params.pruner"]["engine_support"]["dag-ml"] == "unsupported"
    assert entries["pipeline.step.finetune_params.approach"]["engine_support"]["dag-ml"] == "partial"
    assert entries["pipeline.step.finetune_params.sampler"]["status"] == "partial"
    assert entries["pipeline.step.finetune_params.sampler"]["engine_support"]["n4m"] == "partial"
    assert entries["pipeline.step.finetune_params.eval_mode"]["status"] == "partial"
    assert entries["pipeline.step.finetune_params.eval_mode"]["engine_support"] == {
        "optuna": "partial",
        "n4m": "partial",
    }
    assert entries["pipeline.step.finetune_params.train_params"]["engine_support"]["dag-ml"] == "unsupported"
    assert "rejects it until optimizer adapters" in entries["pipeline.step.finetune_params.train_params"]["summary"]
    assert entries["pipeline.step.train_params"]["engine_support"]["dag-ml"] == "unsupported"
    assert entries["pipeline.step.refit_params"]["engine_support"]["dag-ml"] == "unsupported"
    assert "rejects it before native execution" in entries["pipeline.step.train_params"]["summary"]
    assert "rejects it before native execution" in entries["pipeline.step.refit_params"]["summary"]


def test_planned_python_surfaces_are_absent_or_fail_closed_in_current_public_api() -> None:
    run_parameters = inspect.signature(nirs4all.run).parameters

    assert "tuning" in run_parameters
    assert run_parameters["tuning"].default is None
    assert hasattr(nirs4all, "calibrate")
    calibrate_parameters = inspect.signature(nirs4all.calibrate).parameters
    assert "calibration_data" in calibrate_parameters
    assert "coverage" in calibrate_parameters


def test_predictor_changing_operations_invalidate_calibration() -> None:
    entries = _entries_by_id()
    predictor_changers = {
        "pipeline.step.finetune_params",
        "pipeline.step.finetune_params.model_params",
        "pipeline.step.train_params",
        "pipeline.step.refit_params",
        "run.tuning",
        "retrain.mode.finetune",
    }
    conditional_winner_changers = {
        "pipeline.step.finetune_params.approach",
        "pipeline.step.finetune_params.direction",
        "pipeline.step.finetune_params.engine",
        "pipeline.step.finetune_params.eval_mode",
        "pipeline.step.finetune_params.metric",
        "pipeline.step.finetune_params.n_trials",
        "pipeline.step.finetune_params.pruner",
        "pipeline.step.finetune_params.sampler",
        "pipeline.step.finetune_params.train_params",
    }

    assert {entries[entry_id]["invalidates_calibration"] for entry_id in predictor_changers} == {"always"}
    assert {entries[entry_id]["invalidates_calibration"] for entry_id in conditional_winner_changers} == {"if_predictor_changes"}


def test_planned_conformal_v1_uses_physical_samples_and_multi_coverage_sugar() -> None:
    entries = _entries_by_id()

    assert entries["calibrate.calibration_data"]["status"] == "partial"
    assert "DatasetConfigs/path sources" in entries["calibrate.calibration_data"]["summary"]
    assert "RunResult/bundle replay remains planned" not in entries["calibrate.calibration_data"]["summary"]
    assert "saved predictor bundle" in entries["calibrate.calibration_data"]["summary"]
    assert "RunResult.best-like prediction entry" in entries["calibrate.calibration_data"]["summary"]
    assert "stored workspace chain" in entries["calibrate.calibration_data"]["summary"]
    assert entries["calibrate.calibration_data.dataset"]["status"] == "partial"
    assert entries["calibrate.calibration_data.dataset"]["canonical_term"] == "calibration_dataset_source"
    assert "still planned" not in entries["calibrate.calibration_data.dataset"]["summary"]
    assert "predictor_result" in entries["calibrate.calibration_data.dataset"]["summary"]
    assert "predictor_chain_id" in entries["calibrate.calibration_data.dataset"]["summary"]
    dataset_shapes = entries["calibrate.calibration_data.dataset"]["value_schema"]["oneOf"]
    assert {shape["type"] for shape in dataset_shapes} == {"object", "string"}
    assert entries["calibrate.calibration_data.dataset"]["aliases"] == [{"kind": "token", "name": "spectro_dataset", "canonical": "dataset", "mode": "read_only"}]
    assert entries["calibrate.calibration_data.selector"]["changes"] == ["calibration_rows"]
    assert "never guesses a calibration partition" in entries["calibrate.calibration_data.selector"]["summary"]
    assert entries["calibrate.calibration_data.y_pred"]["aliases"] == [
        {"kind": "token", "name": "y_pred_calibration", "canonical": "y_pred", "mode": "read_only"},
        {"kind": "token", "name": "calibration_predictions", "canonical": "y_pred", "mode": "read_only"},
    ]
    assert entries["calibrate.calibration_data.predictor"]["aliases"] == [
        {"kind": "token", "name": "model", "canonical": "predictor", "mode": "read_only"},
        {"kind": "token", "name": "estimator", "canonical": "predictor", "mode": "read_only"},
    ]
    assert "mutually exclusive" in entries["calibrate.calibration_data.predictor"]["summary"]
    assert "predictor_result" in entries["calibrate.calibration_data.predictor"]["summary"]
    assert "predictor_chain_id" in entries["calibrate.calibration_data.predictor"]["summary"]
    assert entries["calibrate.calibration_data.predictor_bundle"]["aliases"] == [
        {"kind": "token", "name": "model_bundle", "canonical": "predictor_bundle", "mode": "read_only"},
        {"kind": "token", "name": "predictor_path", "canonical": "predictor_bundle", "mode": "read_only"},
        {"kind": "token", "name": "model_path", "canonical": "predictor_bundle", "mode": "read_only"},
    ]
    assert "predictor_result and predictor_chain_id are mutually exclusive" in entries["calibrate.calibration_data.predictor_bundle"]["summary"]
    assert entries["calibrate.calibration_data.predictor_result"]["aliases"] == [
        {"kind": "token", "name": "run_result", "canonical": "predictor_result", "mode": "read_only"},
        {"kind": "token", "name": "prediction_entry", "canonical": "predictor_result", "mode": "read_only"},
        {"kind": "token", "name": "prediction", "canonical": "predictor_result", "mode": "read_only"},
    ]
    assert "RunResult.best-like" in entries["calibrate.calibration_data.predictor_result"]["summary"]
    assert entries["calibrate.calibration_data.predictor_chain_id"]["aliases"] == [{"kind": "token", "name": "workspace_chain_id", "canonical": "predictor_chain_id", "mode": "read_only"}]
    assert "Requires calibration_data.workspace_path" in entries["calibrate.calibration_data.predictor_chain_id"]["summary"]
    assert entries["calibrate.calibration_data.workspace_path"]["canonical_term"] == "calibration_predictor_workspace_path"
    assert "not the conformal result persistence workspace" in entries["calibrate.calibration_data.workspace_path"]["summary"]
    assert entries["calibrate.calibration_data.sample_ids"]["aliases"] == [
        {"kind": "token", "name": "calibration_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
        {"kind": "token", "name": "prediction_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
        {"kind": "token", "name": "physical_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
    ]
    assert "calibration_sample_ids and physical_sample_ids are accepted raw aliases" in entries["calibrate.calibration_data.sample_ids"]["summary"]
    assert entries["calibrate.calibration_data.groups"]["aliases"] == [
        {"kind": "token", "name": "calibration_groups", "canonical": "groups", "mode": "read_only"},
    ]
    assert "group_by='group' consumes these labels" in entries["calibrate.calibration_data.groups"]["summary"]
    assert entries["calibrate.calibration_data.metadata"]["aliases"] == [
        {"kind": "token", "name": "calibration_metadata", "canonical": "metadata", "mode": "read_only"},
    ]
    assert entries["calibrate.calibration_data.metadata"]["value_schema"] == {"$ref": "#/$defs/json_native_metadata"}
    assert "Metadata keys used by group_by" in entries["calibrate.calibration_data.metadata"]["summary"]
    assert entries["calibrate.calibration_data.sample_id_column"]["aliases"] == [{"kind": "token", "name": "physical_sample_id_column", "canonical": "sample_id_column", "mode": "read_only"}]
    assert entries["calibrate.calibration_data.group_column"]["invalidates_calibration"] == "replaces_existing"
    assert "does not enable grouped conformal guarantees" in entries["calibrate.calibration_data.group_column"]["summary"]
    assert entries["calibrate.calibration_data.metadata_columns"]["ui"]["control"] == "array"
    assert entries["calibrate.unit"]["value_schema"] == {"const": "physical_sample"}
    coverage_shapes = entries["calibrate.coverage"]["value_schema"]["oneOf"]
    assert coverage_shapes[0]["type"] == "number"
    assert coverage_shapes[1]["type"] == "array"
    assert coverage_shapes[1]["uniqueItems"] is True
    assert entries["calibrate.coverage"]["invalidates_calibration"] == "extends_existing"
    assert entries["calibrate.group_by"]["status"] == "partial"
    assert entries["calibrate.group_by"]["engine_support"] == {"dag-ml": "partial", "legacy": "unsupported"}
    assert "prediction_groups or prediction_metadata" in entries["calibrate.group_by"]["summary"]
    assert "Unseen or missing prediction groups fail closed" in entries["calibrate.group_by"]["summary"]
    assert entries["calibrate.prediction_groups"]["status"] == "partial"
    assert entries["calibrate.prediction_groups"]["changes"] == ["selected_prediction_bounds", "coverage_claim_scope"]
    assert "without global fallback" in entries["calibrate.prediction_groups"]["summary"]
    assert entries["calibrate.prediction_metadata"]["status"] == "partial"
    assert "non-'group' calibrate.group_by keys" in entries["calibrate.prediction_metadata"]["summary"]
    assert entries["calibrate.multi_target"]["status"] == "partial"
    assert entries["calibrate.multi_target"]["value_schema"]["enum"] == ["marginal", "joint_max"]
    assert entries["calibrate.multi_target"]["engine_support"] == {"dag-ml": "partial", "legacy": "unsupported"}
    assert "two-dimensional y_true/y_pred arrays" in entries["calibrate.multi_target"]["summary"]
    assert "max(abs residual) across targets" in entries["calibrate.multi_target"]["summary"]
    assert entries["predict.coverage"]["status"] == "partial"
    assert entries["predict.coverage"]["engine_support"] == {"dag-ml": "partial", "legacy": "unsupported"}
    assert entries["predict.coverage"]["invalidates_calibration"] == "not_applicable"
    predict_coverage_shapes = entries["predict.coverage"]["value_schema"]["oneOf"]
    assert predict_coverage_shapes[0]["type"] == "number"
    assert predict_coverage_shapes[1]["type"] == "array"
    assert predict_coverage_shapes[1]["uniqueItems"] is True
    assert "model .n4a bundles with a conformal sidecar" in entries["predict.coverage"]["summary"]
    assert "non-materialized coverage fails closed" in entries["predict.coverage"]["summary"]
    assert "invalid conformal sidecar fails validation" in entries["predict.coverage"]["summary"]
    assert entries["predict.all_predictions"]["status"] == "partial"
    assert entries["predict.all_predictions"]["value_schema"] == {"type": "boolean"}
    assert entries["predict.all_predictions"]["changes"] == ["prediction_entries"]
    assert entries["predict.all_predictions"]["invalidates_calibration"] == "not_applicable"
    assert entries["predict.all_predictions"]["engine_support"] == {"legacy": "supported", "dag-ml": "partial"}
    assert "all_predictions=True remains fail-closed" in entries["predict.all_predictions"]["summary"]
    calibration_selector_schema = entries["calibrate.calibration_data.selector"]["value_schema"]
    assert calibration_selector_schema["$ref"] == "#/$defs/json_native_mapping"
    assert calibration_selector_schema["x-nirs4all-json-native"] is True
    assert "strict JSON-native selector mapping" in entries["calibrate.calibration_data.selector"]["summary"]
    jsonschema.validate({"partition": "calibration", "nested": {"ok": [1, True, None]}}, calibration_selector_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, calibration_selector_schema)
    assert entries["predict.save_to_workspace"]["status"] == "partial"
    assert entries["predict.save_to_workspace"]["value_schema"] == {"type": "boolean"}
    assert entries["predict.save_to_workspace"]["changes"] == [
        "workspace_prediction_rows",
        "prediction_arrays",
        "result_metadata",
        "workspace_prediction_id",
    ]
    assert entries["predict.save_to_workspace"]["invalidates_calibration"] == "not_applicable"
    assert entries["predict.save_to_workspace"]["engine_support"] == {"legacy": "supported", "dag-ml": "partial"}
    assert "does not persist conformal artifacts" in entries["predict.save_to_workspace"]["summary"]
    assert entries["predict.workspace_metadata"]["changes"] == ["prediction_sample_metadata"]
    predict_workspace_metadata_schema = entries["predict.workspace_metadata"]["value_schema"]
    assert predict_workspace_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
    assert predict_workspace_metadata_schema["x-nirs4all-json-native"] is True
    assert "strict JSON-native sample-level metadata" in entries["predict.workspace_metadata"]["summary"]
    assert entries["predict.workspace_result_metadata"]["changes"] == ["result_metadata", "robustness_evidence"]
    predict_workspace_result_metadata_schema = entries["predict.workspace_result_metadata"]["value_schema"]
    assert predict_workspace_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
    assert predict_workspace_result_metadata_schema["x-nirs4all-json-native"] is True
    assert "robustness_evidence" in entries["predict.workspace_result_metadata"]["summary"]
    jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, predict_workspace_metadata_schema)
    jsonschema.validate({"robustness_evidence": {"predictor_bundle": "model.n4a"}}, predict_workspace_result_metadata_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, predict_workspace_metadata_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, predict_workspace_result_metadata_schema)
    calibrate_result_metadata_schema = entries["calibrate.result_metadata"]["value_schema"]
    predict_calibrated_result_metadata_schema = entries["predict_calibrated.result_metadata"]["value_schema"]
    assert calibrate_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
    assert predict_calibrated_result_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
    assert calibrate_result_metadata_schema["x-nirs4all-json-native"] is True
    assert predict_calibrated_result_metadata_schema["x-nirs4all-json-native"] is True
    assert entries["calibrate.result_metadata"]["changes"] == ["calibrated_result_metadata", "calibrated_result_fingerprint"]
    assert entries["predict_calibrated.result_metadata"]["changes"] == [
        "calibrated_result_metadata",
        "calibrated_result_fingerprint",
        "source_calibrated_result_fingerprint",
    ]
    assert "not the fitted conformal artifact" in entries["calibrate.result_metadata"]["summary"]
    assert "Generated guarantee metadata" in entries["predict_calibrated.result_metadata"]["summary"]
    jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, calibrate_result_metadata_schema)
    jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, predict_calibrated_result_metadata_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, calibrate_result_metadata_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, predict_calibrated_result_metadata_schema)


def test_registry_covers_every_roadmap_keyword_and_effect_boundary() -> None:
    entries = _entries_by_id()
    required = {
        "run.tuning.engine",
        "run.tuning.force_params",
        "run.tuning.space",
        "run.tuning.seed",
        "run.tuning.score_data",
        "run.tuning.score_data.conformal_calibration",
        "run.tuning.score_data.conformal_coverage",
        "run.tuning.storage",
        "run.tuning.study_name",
        "run.tuning.winner",
        "run.tuning.calibration",
        "run.calibration",
        "run.tuning.workspace_tuning_id",
        "run.tuning.workspace_metadata",
        "run.tuning.resume",
        "run.tuning.calibration.workspace_conformal_id",
        "pipeline.step.finetune_params",
        "pipeline.step.finetune_params.approach",
        "pipeline.step.finetune_params.direction",
        "pipeline.step.finetune_params.engine",
        "pipeline.step.finetune_params.eval_mode",
        "pipeline.step.finetune_params.metric",
        "pipeline.step.finetune_params.model_params",
        "pipeline.step.finetune_params.n_trials",
        "pipeline.step.finetune_params.pruner",
        "pipeline.step.finetune_params.sampler",
        "pipeline.step.finetune_params.train_params",
        "pipeline.step.train_params",
        "pipeline.step.refit_params",
        "calibrate.calibration_data",
        "calibrate.calibration_data.dataset",
        "calibrate.calibration_data.selector",
        "calibrate.calibration_data.y_pred",
        "calibrate.calibration_data.predictor",
        "calibrate.calibration_data.predictor_bundle",
        "calibrate.calibration_data.predictor_result",
        "calibrate.calibration_data.predictor_chain_id",
        "calibrate.calibration_data.workspace_path",
        "calibrate.calibration_data.sample_ids",
        "calibrate.calibration_data.groups",
        "calibrate.calibration_data.metadata",
        "calibrate.calibration_data.sample_id_column",
        "calibrate.calibration_data.group_column",
        "calibrate.calibration_data.metadata_columns",
        "calibrate.method",
        "calibrate.coverage",
        "calibrate.unit",
        "calibrate.group_by",
        "calibrate.prediction_groups",
        "calibrate.prediction_metadata",
        "calibrate.multi_target",
        "calibrate.result_metadata",
        "predict.coverage",
        "predict.all_predictions",
        "predict_calibrated.result_metadata",
        "predict.save_to_workspace",
        "predict.workspace_metadata",
        "predict.workspace_result_metadata",
        "robustness.mode",
        "robustness.scenarios",
        "robustness.scenarios.kind",
        "robustness.scenarios.severity",
        "robustness.scenarios.distribution",
        "robustness.X",
        "robustness.predictor",
        "robustness.predictor_bundle",
        "robustness.slice_by",
        "robustness.seed",
        "robustness.workspace_path",
        "robustness.workspace_robustness_id",
        "robustness.workspace_name",
        "robustness.workspace_metadata",
    }

    assert required <= set(entries)
    assert entries["run.tuning.space"]["reads"] == ["development"]
    assert entries["run.tuning.space"]["value_schema"] == {"type": "object"}
    assert entries["run.tuning.space"]["ui"]["control"] == "object"
    assert entries["run.tuning.force_params"]["reads"] == ["development", "run.tuning.space"]
    assert entries["run.tuning.force_params"]["invalidates_calibration"] == "if_predictor_changes"
    assert "public decoded syntax" in entries["run.tuning.force_params"]["summary"]
    assert entries["calibrate.calibration_data"]["reads"] == ["calibration"]
    assert entries["robustness.mode"]["invalidates_calibration"] == "mode_dependent"
    assert entries["robustness.mode"]["status"] == "partial"
    assert entries["robustness.mode"]["value_schema"]["enum"] == ["clean_frozen", "matched_recalibration", "structural_refit"]
    assert entries["robustness.mode"]["value_schema"]["x-executable-values"] == ["clean_frozen"]
    assert "clean_frozen audit-only" in entries["robustness.mode"]["summary"]
    assert entries["robustness.scenarios"]["status"] == "partial"
    assert "prediction_bias" in entries["robustness.scenarios"]["summary"]
    assert "prediction_noise" in entries["robustness.scenarios"]["summary"]
    assert "spectral_noise" in entries["robustness.scenarios"]["summary"]
    assert "spectral_offset" in entries["robustness.scenarios"]["summary"]
    assert "spectral_scale" in entries["robustness.scenarios"]["summary"]
    assert "spectral_slope" in entries["robustness.scenarios"]["summary"]
    assert "spectral_shift" in entries["robustness.scenarios"]["summary"]
    assert "RobustnessScenarioSpec" in entries["robustness.scenarios"]["summary"]
    assert "TCV1 JSON-native mappings" in entries["robustness.scenarios"]["summary"]
    assert entries["robustness.scenarios"]["value_schema"]["items"]["required"] == ["kind"]
    assert entries["robustness.scenarios"]["value_schema"]["items"]["x-tcv1-json-native"] is True
    assert entries["robustness.scenarios"]["value_schema"]["items"]["allOf"] == [
        {
            "if": {"properties": {"kind": {"enum": ["prediction_noise", "spectral_noise"]}}, "required": ["kind"]},
            "then": {},
            "else": {"not": {"required": ["distribution"]}},
        }
    ]
    assert entries["robustness.scenarios.kind"]["value_schema"]["enum"] == [
        "observed",
        "prediction_bias",
        "prediction_noise",
        "spectral_noise",
        "spectral_offset",
        "spectral_scale",
        "spectral_slope",
        "spectral_shift",
    ]
    assert "additive offset" in entries["robustness.scenarios.severity"]["summary"]
    assert "multiplicative factor" in entries["robustness.scenarios.severity"]["summary"]
    assert "uniform-noise half-width" in entries["robustness.scenarios.severity"]["summary"]
    assert "linear ramp" in entries["robustness.scenarios.severity"]["summary"]
    assert "fractional feature-axis shift" in entries["robustness.scenarios.severity"]["summary"]
    assert entries["robustness.scenarios"]["value_schema"]["items"]["properties"]["distribution"] == {"type": "string", "enum": ["normal", "uniform"]}
    assert entries["robustness.scenarios.distribution"]["value_schema"] == {"type": "string", "enum": ["normal", "uniform"]}
    assert "only for prediction_noise and spectral_noise" in entries["robustness.scenarios.distribution"]["summary"]
    assert "centered uniform noise" in entries["robustness.scenarios.distribution"]["summary"]
    assert entries["robustness.X"]["reads"] == ["external_test_or_production"]
    assert "never fitted" in entries["robustness.X"]["summary"]
    assert entries["robustness.predictor"]["reads"] == ["frozen_predictor"]
    assert "mutually exclusive with robustness.predictor_bundle" in entries["robustness.predictor"]["summary"]
    assert "does not refit" in entries["robustness.predictor"]["summary"]
    assert entries["robustness.predictor_bundle"]["reads"] == ["frozen_predictor_bundle"]
    assert entries["robustness.predictor_bundle"]["aliases"] == [
        {"kind": "token", "name": "model_bundle", "canonical": "predictor_bundle", "mode": "read_only"},
        {"kind": "token", "name": "predictor_path", "canonical": "predictor_bundle", "mode": "read_only"},
        {"kind": "token", "name": "model_path", "canonical": "predictor_bundle", "mode": "read_only"},
    ]
    assert "nirs4all.predict(model=predictor_bundle" in entries["robustness.predictor_bundle"]["summary"]
    assert "without refit or recalibration" in entries["robustness.predictor_bundle"]["summary"]
    assert "effective_seed=0" in entries["robustness.seed"]["summary"]
    assert entries["robustness.slice_by"]["invalidates_calibration"] == "not_applicable"
    assert "not a conditional conformal guarantee" in entries["robustness.slice_by"]["summary"]
    assert entries["robustness.workspace_path"]["changes"] == ["workspace_robustness_results"]
    assert entries["robustness.workspace_path"]["invalidates_calibration"] == "not_applicable"
    assert "does not change metrics" in entries["robustness.workspace_path"]["summary"]
    assert entries["robustness.workspace_robustness_id"]["aliases"] == [{"kind": "token", "name": "robustness_id", "canonical": "workspace_robustness_id", "mode": "read_only"}]
    assert "not injected into the report fingerprint" in entries["robustness.workspace_robustness_id"]["summary"]
    robustness_workspace_metadata_schema = entries["robustness.workspace_metadata"]["value_schema"]
    assert robustness_workspace_metadata_schema["$ref"] == "#/$defs/json_native_mapping"
    assert robustness_workspace_metadata_schema["x-nirs4all-json-native"] is True
    assert "Strict JSON-native workspace metadata" in entries["robustness.workspace_metadata"]["summary"]
    jsonschema.validate({"site": "north", "nested": {"ok": [1, True, None]}}, robustness_workspace_metadata_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"bad": object()}, robustness_workspace_metadata_schema)


def test_robustness_scenarios_schema_rejects_distribution_for_deterministic_scenarios() -> None:
    schema = _entries_by_id()["robustness.scenarios"]["value_schema"]

    jsonschema.validate([{"kind": "prediction_noise", "severity": 0.1, "distribution": "normal"}], schema)
    jsonschema.validate([{"kind": "prediction_noise", "severity": 0.1, "distribution": "uniform"}], schema)
    jsonschema.validate([{"kind": "spectral_noise", "severity": 0.1, "distribution": "normal"}], schema)
    jsonschema.validate([{"kind": "spectral_noise", "severity": 0.1, "distribution": "uniform"}], schema)
    jsonschema.validate([{"kind": "spectral_shift", "severity": 0.25}], schema)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([{"kind": "spectral_shift", "severity": 0.25, "distribution": "normal"}], schema)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([{"kind": "prediction_noise", "severity": 0.1, "distribution": "laplace"}], schema)


def test_public_robustness_surface_is_present_but_partial() -> None:
    assert hasattr(nirs4all, "robustness")
    assert nirs4all.RobustnessReport.__name__ == "RobustnessReport"


def test_every_registry_docs_anchor_exists() -> None:
    reference = Path(__file__).resolve().parents[3] / "docs" / "source" / "reference" / "pipeline_keywords.md"
    markdown = reference.read_text(encoding="utf-8")

    for anchor in {entry["docs_anchor"] for entry in get_keyword_registry()["entries"]}:
        assert f"({anchor})=" in markdown
