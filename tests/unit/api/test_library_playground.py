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


def test_library_imposes_no_request_response_or_step_count_ceiling():
    import nirs4all.api.library_playground as adapter

    # Byte budgets belong to the host transport; the library defines none.
    assert not [name for name in vars(adapter) if name.startswith("MAX_PLAYGROUND")]
    step = {"id": "snv", "type": "preprocessing", "name": "StandardNormalVariate",
            "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate", "params": {}}}
    steps = [dict(step, id=f"snv-{index}") for index in range(60)]
    assert len(studio_playground_job_v1(request("validate", {"steps": steps}))["result"]["steps"]) == 60
    # A request above the former 8 MiB request ceiling is processed.
    x = np.linspace(0.123456789, 0.987654321, 12_000 * 100).reshape(12_000, 100).tolist()
    payload = request("execute", {
        "data": {"x": x}, "sampling": {"method": "all", "n_samples": 1},
        "options": {"compute_repetitions": False, "compute_statistics": False, "compute_metrics": False},
        "limits": {"max_samples": 12_000},
    })
    assert len(json.dumps(payload).encode("utf-8")) > 8 * 1024 * 1024
    response = studio_playground_job_v1(payload)
    assert response["result"]["success"] is True
    assert response["result"]["processed"]["shape"] == [12_000, 100]


@pytest.fixture
def csv_dataset(tmp_path, monkeypatch):
    import nirs4all.api.library_playground as adapter

    monkeypatch.setattr(adapter, "_loaded_dataset", None)
    rng = np.random.default_rng(3)
    x_path, y_path = tmp_path / "Xtrain.csv", tmp_path / "Ytrain.csv"
    header = ";".join(str(1000 + 2 * index) for index in range(8))
    np.savetxt(x_path, rng.normal(1.0, 0.1, size=(10, 8)), delimiter=";", header=header, comments="")
    np.savetxt(y_path, rng.normal(size=(10, 2)), delimiter=";", header="a;b", comments="")
    loads = []
    original_load = adapter.load_dataset_for_analysis

    def counting_load(config, **options):
        loads.append(config)
        return original_load(config, **options)

    monkeypatch.setattr(adapter, "load_dataset_for_analysis", counting_load)
    return {"config": {"train_x": str(x_path), "train_y": str(y_path)}}, x_path, loads


def dataset_execution(dataset, *, target_index=0, steps=(), options=None):
    return studio_playground_job_v1(request("execute", {
        "dataset": dataset, "selection": {"partition": "train", "target_index": target_index}, "steps": list(steps),
        "options": {"compute_repetitions": False, **(options or {})},
    }))["result"]


SNV_STEP = {"id": "snv", "type": "preprocessing", "name": "StandardNormalVariate", "params": {}, "enabled": True,
            "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate", "params": {}}}


def test_warm_host_reuses_the_loaded_dataset_for_an_identical_request(csv_dataset):
    dataset, _, loads = csv_dataset
    first = dataset_execution(dataset, target_index=0)
    second = dataset_execution(dataset, target_index=1)
    metadata = studio_playground_job_v1(request("metadata_columns", {"dataset": dataset}))
    assert len(loads) == 1
    assert second["dataset_reader"] == first["dataset_reader"] == metadata["result"]["dataset_reader"]
    assert second["original"]["y"] != first["original"]["y"]


def test_dataset_reuse_is_invalidated_when_an_input_file_changes(csv_dataset):
    dataset, x_path, loads = csv_dataset
    first = dataset_execution(dataset)
    lines = x_path.read_text().splitlines()
    x_path.write_text("\n".join([lines[0], *(";".join(["2.5"] * 8) for _ in lines[1:])]) + "\n")
    second = dataset_execution(dataset)
    assert len(loads) == 2
    assert second["original"]["spectra"][0] == [2.5] * 8
    assert first["original"]["spectra"][0] != second["original"]["spectra"][0]


def test_dataset_reuse_is_invalidated_when_the_request_changes(csv_dataset):
    dataset, _, loads = csv_dataset
    dataset_execution(dataset)
    dataset_execution({**dataset, "max_input_bytes": 10**9})
    dataset_execution({"config": {**dataset["config"], "train_y_params": {"header": True}}})
    assert len(loads) == 3


def test_a_reused_dataset_is_not_changed_by_a_previous_request(csv_dataset):
    import nirs4all.api.library_playground as adapter

    dataset, _, loads = csv_dataset
    raw = dataset_execution(dataset)
    processed = dataset_execution(dataset, steps=[SNV_STEP])
    assert processed["processed"]["spectra"] != raw["processed"]["spectra"]
    again = dataset_execution(dataset)
    assert len(loads) == 1
    adapter._loaded_dataset = None
    fresh = dataset_execution(dataset)
    assert len(loads) == 2
    for result in (again, fresh):
        assert result["original"] == raw["original"]
        assert result["processed"] == raw["processed"]
        assert result["pca"] == raw["pca"]


def test_unchanged_processed_spectra_are_sent_once_only_on_request(csv_dataset):
    dataset, _, _ = csv_dataset
    both = dataset_execution(dataset)
    assert "processed_is_original" not in both
    assert both["processed"]["spectra"] == both["original"]["spectra"]
    once = dataset_execution(dataset, options={"omit_unchanged_processed_spectra": True})
    assert once["processed_is_original"] is True
    assert "spectra" not in once["processed"]
    assert once["original"] == both["original"]
    assert {key: value for key, value in both["processed"].items() if key != "spectra"} == once["processed"]
    processed = dataset_execution(dataset, steps=[SNV_STEP], options={"omit_unchanged_processed_spectra": True})
    assert "processed_is_original" not in processed
    assert processed["processed"]["spectra"] != processed["original"]["spectra"]
    with pytest.raises(StudioScientificJobError):
        dataset_execution(dataset, options={"omit_unchanged_processed_spectra": "yes"})
