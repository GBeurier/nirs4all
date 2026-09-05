"""Closed JSON adapter for the stateless Playground library owner."""

import json
import sys

import numpy as np
import pytest

from nirs4all.api.library_playground import (
    STUDIO_PLAYGROUND_JOB_SCHEMA,
    STUDIO_PLAYGROUND_RESULT_SCHEMA,
    studio_playground_job_v1,
)
from nirs4all.api.studio_scientific import StudioScientificJobError
from nirs4all.data.dataset import SpectroDataset


def request(operation, payload):
    return {"schema": STUDIO_PLAYGROUND_JOB_SCHEMA, "operation": operation, "request_id": "preview-1", "payload": payload}


def test_inline_canonical_steps_execute_and_return_strict_json():
    x = (np.arange(60, dtype=float).reshape(12, 5) + 1).tolist()
    response = studio_playground_job_v1(request("execute", {
        "data": {"x": x, "y": list(range(12)), "wavelengths": [1000, 1001, 1002, 1003, 1004],
                 "sample_ids": [f"S{i}" for i in range(12)]},
        "steps": [{"id": "snv", "type": "preprocessing", "name": "StandardNormalVariate",
                   "params": {}, "enabled": True,
                   "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate", "params": {}}}],
        "sampling": {"method": "all", "n_samples": 100, "seed": 42},
        "options": {"compute_repetitions": False},
    }))

    assert response["schema"] == STUDIO_PLAYGROUND_RESULT_SCHEMA
    assert response["result"]["success"] is True
    assert response["result"]["processed"]["shape"] == [12, 5]
    assert response["result"]["processed"]["sample_ids"] == [f"S{i}" for i in range(12)]
    assert response["result"]["cache"] == {"used": False, "scope": "stateless_callable"}
    json.dumps(response, allow_nan=False)
    assert not ({"fastapi", "starlette", "uvicorn"} & set(sys.modules))


def test_diff_and_repetition_use_library_definitions():
    diff = studio_playground_job_v1(request("diff", {
        "reference": [[0.0, 1.0], [2.0, 4.0]], "final": [[0.0, 2.0], [1.0, 4.0]], "metric": "manhattan",
    }))
    assert diff["result"]["distances"] == [1.0, 1.0]
    repetitions = studio_playground_job_v1(request("repetition_variance", {
        "x": [[0.0, 1.0], [0.0, 2.0], [5.0, 5.0]], "group_ids": ["A", "A", "B"], "reference": "first",
    }))
    assert repetitions["result"]["sample_indices"] == [0, 1]
    assert repetitions["result"]["n_groups"] == 1


def test_dataset_selection_and_metadata_delegate_to_existing_loader(monkeypatch):
    dataset = SpectroDataset("loaded")
    dataset.add_samples(np.arange(24, dtype=float).reshape(6, 4), {"partition": "train"}, headers=["1", "2", "3", "4"])
    dataset.add_targets(np.arange(6, dtype=float))
    dataset.add_metadata(np.asarray([[f"S{i}", "A" if i < 3 else "B"] for i in range(6)]), headers=["sample_id", "batch"])
    calls = []

    def load(config, **options):
        calls.append((config, options))
        return dataset, {"backend": "qualified-test-reader"}

    monkeypatch.setattr("nirs4all.api.library_playground.load_dataset_for_analysis", load)
    execution = studio_playground_job_v1(request("execute", {
        "dataset": {"config": {"train_x": "/authorized/X.csv"}, "max_input_bytes": 1234},
        "selection": {"partition": "train", "source_index": 0, "target_index": 0},
        "options": {"compute_repetitions": False},
    }))
    assert execution["result"]["dataset_reader"] == {"backend": "qualified-test-reader"}
    assert execution["result"]["dataset_selection"]["n_train"] == 6
    metadata = studio_playground_job_v1(request("metadata_columns", {
        "dataset": {"config": {"train_x": "/authorized/X.csv"}}, "max_unique_values": 1,
    }))
    assert metadata["result"]["columns"][1]["unique_values"] == ["A"]
    assert len(calls) == 2


def test_missing_or_unapproved_operator_fails_before_execution(monkeypatch):
    import nirs4all.api.library_playground as adapter

    monkeypatch.setattr(adapter, "preview_arrays", lambda *args, **kwargs: pytest.fail("must not execute"))
    base = {"data": {"x": [[1.0, 2.0]]}, "steps": [{"id": "x", "type": "preprocessing", "name": "X"}]}
    with pytest.raises(StudioScientificJobError) as missing:
        studio_playground_job_v1(request("execute", base))
    assert missing.value.code == "missing_operator"
    base["steps"][0]["operator"] = {"class": "subprocess.Popen", "params": {}}
    with pytest.raises(StudioScientificJobError) as forbidden:
        studio_playground_job_v1(request("execute", base))
    assert forbidden.value.code == "operator_package_forbidden"


@pytest.mark.parametrize("invalid", [
    {"schema": STUDIO_PLAYGROUND_JOB_SCHEMA, "operation": "execute", "request_id": "x", "payload": []},
    request("execute", {"data": {"x": [[1.0]]}, "dataset": {"config": {}}, "unknown": True}),
    request("capabilities", {"unexpected": True}),
])
def test_closed_contract_refuses_invalid_shapes(invalid):
    with pytest.raises(StudioScientificJobError):
        studio_playground_job_v1(invalid)


def test_non_finite_scientific_outputs_are_explicit_wire_nulls():
    response = studio_playground_job_v1(request("execute", {
        "data": {"x": [[1.0, 1.0], [1.0, 1.0]]},
        "options": {"compute_repetitions": False},
    }))
    json.dumps(response, allow_nan=False)
    assert response["wire_diagnostics"]["non_finite_values_encoded_as_null"] > 0


def test_capabilities_report_stateless_limits_and_complete_descriptors():
    result = studio_playground_job_v1(request("capabilities", {}))["result"]
    assert result["stateless"] is True
    assert result["cache"] is False
    assert result["default_limits"] == {"max_samples": 10_000, "max_features": 10_000, "max_cells": 100_000_000}
    assert "hotelling_t2" in result["spectral_descriptors"]
