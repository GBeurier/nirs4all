"""Tests for the keyword registry Sphinx extension."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_extension():
    pytest.importorskip("docutils")
    path = Path(__file__).resolve().parents[3] / "docs" / "source" / "_ext" / "keyword_registry.py"
    spec = importlib.util.spec_from_file_location("nirs4all_docs_keyword_registry", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyword_registry_extension_writes_static_json_artifact(tmp_path) -> None:
    """Docs builds publish the registry JSON for Studio/Web/static consumers."""

    extension = _load_extension()
    app = SimpleNamespace(outdir=str(tmp_path / "html"))

    extension.write_keyword_registry_static_artifact(app, None)

    output = tmp_path / "html" / "_static" / "keyword-registry.json"
    raw_payload = output.read_text(encoding="utf-8")
    payload = json.loads(raw_payload)
    schema = json.loads((tmp_path / "html" / "_static" / "keyword-registry.schema.json").read_text(encoding="utf-8"))
    robustness_summary_schema = json.loads((tmp_path / "html" / "_static" / "robustness-summary.schema.json").read_text(encoding="utf-8"))
    tuning_summary_schema = json.loads((tmp_path / "html" / "_static" / "tuning-summary.schema.json").read_text(encoding="utf-8"))
    assert payload["schema_id"] == "https://nirs4all.org/schemas/keyword-effects/v1"
    assert any(entry["id"] == "robustness.scenarios.kind" for entry in payload["entries"])
    assert hashlib.sha256(raw_payload.encode("utf-8")).hexdigest() == "38997cf10c2629259b64d7f3fcbeff589680e6585ccbbfdfbe711a3fde5eaf12"
    entries = {entry["id"]: entry for entry in payload["entries"]}
    assert entries["calibrate.calibration_data.sample_ids"]["aliases"] == [
        {"kind": "token", "name": "calibration_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
        {"kind": "token", "name": "prediction_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
        {"kind": "token", "name": "physical_sample_ids", "canonical": "sample_ids", "mode": "read_only"},
    ]
    assert entries["calibrate.calibration_data.groups"]["aliases"] == [
        {"kind": "token", "name": "calibration_groups", "canonical": "groups", "mode": "read_only"},
    ]
    assert entries["calibrate.calibration_data.metadata"]["aliases"] == [
        {"kind": "token", "name": "calibration_metadata", "canonical": "metadata", "mode": "read_only"},
    ]
    assert entries["robustness.scenarios.distribution"]["value_schema"] == {"type": "string", "enum": ["normal", "uniform"]}
    assert entries["run.tuning.space"]["value_schema"] == {"type": "object"}
    assert entries["run.tuning.space"]["ui"]["control"] == "object"
    assert entries["run.tuning.force_params"]["changes"] == ["trial_sequence", "candidate_fit", "selection"]
    assert entries["run.tuning.force_params"]["invalidates_calibration"] == "if_predictor_changes"
    assert "public decoded syntax" in entries["run.tuning.force_params"]["summary"]
    assert entries["robustness.predictor_bundle"]["reads"] == ["frozen_predictor_bundle"]
    assert "nirs4all.predict(model=predictor_bundle" in entries["robustness.predictor_bundle"]["summary"]
    assert "invalid conformal sidecar fails validation" in entries["predict.coverage"]["summary"]
    assert entries["predict.save_to_workspace"]["changes"] == [
        "workspace_prediction_rows",
        "prediction_arrays",
        "result_metadata",
        "workspace_prediction_id",
    ]
    assert entries["predict.workspace_metadata"]["value_schema"]["$ref"] == "#/$defs/json_native_mapping"
    assert entries["predict.workspace_metadata"]["value_schema"]["x-nirs4all-json-native"] is True
    assert entries["calibrate.result_metadata"]["value_schema"]["$ref"] == "#/$defs/json_native_mapping"
    assert entries["calibrate.result_metadata"]["value_schema"]["x-nirs4all-json-native"] is True
    assert entries["predict_calibrated.result_metadata"]["value_schema"]["$ref"] == "#/$defs/json_native_mapping"
    assert entries["predict_calibrated.result_metadata"]["value_schema"]["x-nirs4all-json-native"] is True
    assert entries["predict.workspace_result_metadata"]["changes"] == ["result_metadata", "robustness_evidence"]
    assert entries["predict.workspace_result_metadata"]["value_schema"]["$ref"] == "#/$defs/json_native_mapping"
    assert entries["predict.workspace_result_metadata"]["value_schema"]["x-nirs4all-json-native"] is True
    assert "robustness_evidence" in entries["predict.workspace_result_metadata"]["summary"]
    assert schema["$id"] == payload["schema_id"]
    assert robustness_summary_schema["$id"] == "https://nirs4all.org/schemas/robustness-summary/v1"
    assert robustness_summary_schema["properties"]["format"]["const"] == "nirs4all.robustness.summary"
    guarantee_schema = robustness_summary_schema["properties"]["conformal_guarantee_status"]
    assert guarantee_schema["type"] == ["object", "null"]
    assert guarantee_schema["properties"]["effective_engine"] == {"type": "string"}
    assert guarantee_schema["properties"]["coverage"] == {"type": "array", "items": {"type": "number"}}
    spectral_replay_schema = robustness_summary_schema["properties"]["spectral_replay"]
    assert spectral_replay_schema["required"] == ["route", "sample_ids_forwarded", "source"]
    assert spectral_replay_schema["properties"]["source"]["enum"] == ["predictor", "predictor_bundle"]
    assert tuning_summary_schema["$id"] == "https://nirs4all.org/schemas/tuning-summary/v1"
    assert tuning_summary_schema["properties"]["format"]["const"] == "nirs4all.tuning.summary"
    assert tuning_summary_schema["properties"]["sampler"] == {"type": ["string", "null"]}
    assert tuning_summary_schema["properties"]["pruner"] == {"type": ["string", "null"]}
    assert tuning_summary_schema["properties"]["seed"] == {"type": ["integer", "null"]}


def test_keyword_registry_extension_skips_static_json_after_failed_build(tmp_path) -> None:
    """A failed docs build must not publish a misleading fresh registry artifact."""

    extension = _load_extension()
    app = SimpleNamespace(outdir=str(tmp_path / "html"))

    extension.write_keyword_registry_static_artifact(app, RuntimeError("build failed"))

    assert not (tmp_path / "html" / "_static" / "keyword-registry.json").exists()
    assert not (tmp_path / "html" / "_static" / "keyword-registry.schema.json").exists()
    assert not (tmp_path / "html" / "_static" / "robustness-summary.schema.json").exists()
    assert not (tmp_path / "html" / "_static" / "tuning-summary.schema.json").exists()
