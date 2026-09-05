"""Tests for the bounded Studio synthetic-dataset callable."""

import json
import sys
from pathlib import Path

import pytest

from nirs4all.api.general_synthesis import (
    STUDIO_SYNTHETIC_DATASET_JOB_SCHEMA,
    STUDIO_SYNTHETIC_DATASET_RESULT_SCHEMA,
    studio_synthetic_dataset_job_v1,
)
from nirs4all.api.studio_scientific import StudioScientificJobError


def request(output_dir: Path, **payload):
    base = {
        "task_type": "regression",
        "n_samples": 50,
        "complexity": "simple",
        "train_ratio": 0.8,
        "wavelength_range": [1000, 1010],
        "random_state": 17,
        "name": "demo",
    }
    base.update(payload)
    return {
        "schema": STUDIO_SYNTHETIC_DATASET_JOB_SCHEMA,
        "operation": "generate",
        "request_id": "synthesis-1",
        "output_dir": str(output_dir),
        "payload": base,
    }


def test_regression_exports_standard_artifacts_and_actual_summary(tmp_path):
    response = studio_synthetic_dataset_job_v1(request(tmp_path, target_range=[10, 20]))

    assert response["schema"] == STUDIO_SYNTHETIC_DATASET_RESULT_SCHEMA
    result = response["result"]
    assert result["relative_path"] == "demo"
    assert result["summary"] == {
        "samples": 50,
        "features": 6,
        "train": 40,
        "test": 10,
        "task": "regression",
        "classes": None,
    }
    assert {item["path"] for item in result["files"]} == {"Xcal.csv", "Ycal.csv", "Xval.csv", "Yval.csv"}
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in result["files"])
    assert {path.name for path in (tmp_path / "demo").iterdir()} == {"Xcal.csv", "Ycal.csv", "Xval.csv", "Yval.csv"}
    json.dumps(response, allow_nan=False)
    assert not ({"fastapi", "starlette", "uvicorn"} & set(sys.modules))


def test_seed_makes_export_byte_reproducible(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    left = studio_synthetic_dataset_job_v1(request(first))["result"]["files"]
    right = studio_synthetic_dataset_job_v1(request(second))["result"]["files"]
    assert [(item["path"], item["sha256"]) for item in left] == [(item["path"], item["sha256"]) for item in right]


@pytest.mark.parametrize(
    ("task_type", "n_classes"),
    [("binary_classification", 2), ("multiclass_classification", 3)],
)
def test_classification_reports_actual_classes(tmp_path, task_type, n_classes):
    response = studio_synthetic_dataset_job_v1(request(tmp_path, task_type=task_type, n_classes=n_classes))
    assert response["result"]["summary"]["task"] == task_type
    assert response["result"]["summary"]["classes"] == n_classes


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "demo"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")
    with pytest.raises(StudioScientificJobError) as raised:
        studio_synthetic_dataset_job_v1(request(tmp_path))
    assert raised.value.code == "output_exists"
    assert sentinel.read_text(encoding="utf-8") == "owned by caller"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"n_samples": 49}, "invalid_integer"),
        ({"train_ratio": 0.49}, "invalid_range"),
        ({"wavelength_range": [1000, 30_000]}, "resource_limit"),
        ({"name": "../escape"}, "invalid_name"),
        ({"task_type": "binary_classification", "n_classes": 3}, "invalid_integer"),
        ({"task_type": "regression", "n_classes": 2}, "unknown_field"),
    ],
)
def test_invalid_or_oversized_requests_fail_before_generation(tmp_path, monkeypatch, change, code):
    import nirs4all.api.general_synthesis as adapter

    monkeypatch.setattr(adapter, "SyntheticDatasetBuilder", lambda *args, **kwargs: pytest.fail("must not generate"))
    with pytest.raises(StudioScientificJobError) as raised:
        studio_synthetic_dataset_job_v1(request(tmp_path, **change))
    assert raised.value.code == code


def test_output_directory_must_be_absolute_and_pre_authorized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(StudioScientificJobError) as raised:
        studio_synthetic_dataset_job_v1(request(Path("relative")))
    assert raised.value.code == "output_forbidden"
