from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all import api
from nirs4all.api import studio_scientific as host


def _request() -> dict[str, Any]:
    return {
        "schema": "nirs4all.studio-scientific-job.v1",
        "operation": "run",
        "job_id": "training_0001",
        "engine": "dag-ml",
        "allow_fallback": False,
        "dataset": {
            "name": "protein",
            "task_type": "regression",
            "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0], [5.0, 11.0], [6.0, 13.0]],
            "y": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        },
        "pipeline": {
            "kind": "pls_regression",
            "n_components": 1,
            "scale": True,
            "cross_validation": {"kind": "kfold", "n_splits": 3, "shuffle": True},
        },
        "options": {"name": "protein_pls", "random_state": 17},
    }


class _Result:
    def __init__(self, *, validation_score: float = 0.25) -> None:
        self.cv_best = {
            "metric": "rmse",
            "val_score": validation_score,
            "train_score": 0.1,
        }
        self.num_predictions = 7
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _assert_code(request: object, code: str) -> None:
    with pytest.raises(host.StudioScientificJobError) as caught:
        host.studio_scientific_job_v1(request)
    assert caught.value.code == code


def test_public_exports_are_the_same_callable() -> None:
    assert nirs4all.studio_scientific_job_v1 is host.studio_scientific_job_v1
    assert api.studio_scientific_job_v1 is host.studio_scientific_job_v1
    assert nirs4all.STUDIO_SCIENTIFIC_JOB_SCHEMA == "nirs4all.studio-scientific-job.v1"
    assert nirs4all.STUDIO_SCIENTIFIC_RESULT_SCHEMA == "nirs4all.studio-scientific-job-result.v1"


def test_closed_callable_runs_only_dagml_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    result = _Result()

    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: None)

    def fake_run(**kwargs: Any) -> _Result:
        captured.update(kwargs)
        return result

    monkeypatch.setattr(host, "_run", fake_run)
    response = host.studio_scientific_job_v1(_request())

    assert response == {
        "schema": "nirs4all.studio-scientific-job-result.v1",
        "job_id": "training_0001",
        "engine": "dag-ml",
        "result": {
            "model": "pls_regression",
            "task_type": "regression",
            "metric": "rmse",
            "validation_score": 0.25,
            "training_score": 0.1,
            "prediction_count": 7,
        },
    }
    assert len(json.dumps(response, separators=(",", ":")).encode()) <= host.MAX_STUDIO_SCIENTIFIC_RESPONSE_BYTES
    assert "status" not in response
    assert "workspace" not in response
    assert captured["engine"] == "dag-ml"
    assert captured["allow_fallback"] is False
    assert captured["results_path"] is None
    assert captured["save_artifacts"] is False
    assert captured["save_charts"] is False
    assert captured["project"] is None
    assert captured["cache"] is None
    assert captured["name"] == "protein_pls"
    assert captured["random_state"] is None
    X, y = captured["dataset"]
    np.testing.assert_allclose(X, np.asarray(_request()["dataset"]["X"]))
    np.testing.assert_allclose(y, np.asarray(_request()["dataset"]["y"]))
    split, model = captured["pipeline"]
    assert isinstance(split, KFold)
    assert split.n_splits == 3 and split.shuffle and split.random_state == 17
    assert isinstance(model, PLSRegression)
    assert model.n_components == 1 and model.scale is True
    assert result.closed


@pytest.mark.parametrize("engine", ["legacy", "dual", "native", "local-python"])
def test_forbidden_engines_fail_before_runtime(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: pytest.fail("runtime preflight reached"))
    request = _request()
    request["engine"] = engine
    _assert_code(request, "engine_forbidden")


def test_fallback_and_caller_owned_paths_are_refused_before_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: pytest.fail("runtime preflight reached"))
    fallback = _request()
    fallback["allow_fallback"] = True
    _assert_code(fallback, "fallback_forbidden")

    for field, value in [("workspace_path", "/tmp/workspace"), ("path", "data.csv"), ("scheduler", "python")]:
        request = _request()
        if field == "path":
            request["dataset"][field] = value
        else:
            request[field] = value
        _assert_code(request, "unknown_field")

    identifier_path = _request()
    identifier_path["job_id"] = "../job"
    _assert_code(identifier_path, "path_forbidden")


def test_unknown_fields_objects_nonfinite_and_oversize_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: pytest.fail("runtime preflight reached"))
    unknown = _request()
    unknown["pipeline"]["class"] = "arbitrary.module.Model"
    _assert_code(unknown, "unknown_field")

    python_object = _request()
    python_object["dataset"]["X"] = np.ones((6, 2))
    _assert_code(python_object, "non_json_value")

    nonfinite = _request()
    nonfinite["dataset"]["X"][0][0] = float("nan")
    _assert_code(nonfinite, "non_finite_number")

    overflowing = _request()
    overflowing["dataset"]["X"][0][0] = 10**1000
    _assert_code(overflowing, "non_finite_number")

    oversized = _request()
    oversized["dataset"]["X"] = [[1.23456789] * 256 for _ in range(128)]
    oversized["dataset"]["y"] = [float(index) + 0.1 for index in range(128)]
    _assert_code(oversized, "request_too_large")


def test_pipeline_dataset_and_result_bounds_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: None)

    classification = _request()
    classification["dataset"]["y"] = [0, 1, 0, 1, 0, 1]
    _assert_code(classification, "unsupported_task")

    components = _request()
    components["pipeline"]["n_components"] = 3
    _assert_code(components, "invalid_components")

    malformed = _request()
    malformed["dataset"]["X"][1].pop()
    _assert_code(malformed, "invalid_dataset")

    result = _Result(validation_score=float("nan"))
    monkeypatch.setattr(host, "_run", lambda **_kwargs: result)
    _assert_code(_request(), "invalid_scientific_result")
    assert result.closed

    missing_score = _Result()
    missing_score.cv_best.pop("train_score")
    monkeypatch.setattr(host, "_run", lambda **_kwargs: missing_score)
    _assert_code(_request(), "invalid_scientific_result")
    assert missing_score.closed


def test_ambient_persistence_and_external_runtime_overrides_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_NATIVE_RESULTS", "/tmp/results")
    _assert_code(_request(), "ambient_persistence_forbidden")

    monkeypatch.delenv("N4A_NATIVE_RESULTS")
    monkeypatch.setenv("N4A_DAGML_CLI", "/tmp/dag-ml-cli")
    _assert_code(_request(), "external_runtime_forbidden")

    monkeypatch.delenv("N4A_DAGML_CLI")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "false")
    _assert_code(_request(), "external_runtime_forbidden")


@pytest.mark.parametrize(
    "variable",
    [
        "N4A_DAGML_DATASET_PATH",
        "N4A_DAGML_DATASET_PICKLE",
        "N4A_DAGML_GRAPH_PATH",
        "N4A_DAGML_METHODS_SNV",
        "N4A_DAGML_RESULT_CAPTURE",
        "N4A_DAGML_SAMPLE_META_PATH",
        "N4A_RANDOM_STATE",
    ],
)
def test_ambient_dagml_side_channels_are_refused(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, "/caller/owned/value")
    _assert_code(_request(), "ambient_runtime_forbidden")


def test_runtime_outside_active_prefix_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.delenv("N4A_DAGML_CLI", raising=False)
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    isolated_prefix = tmp_path / "isolated-prefix"
    monkeypatch.setattr(host.sys, "prefix", str(isolated_prefix))
    monkeypatch.setattr(host, "__file__", str(isolated_prefix / "lib" / "python" / "site-packages" / "nirs4all" / "api" / "studio_scientific.py"))
    _assert_code(_request(), "sibling_runtime_forbidden")


def test_source_injected_callable_outside_active_prefix_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.delenv("N4A_DAGML_CLI", raising=False)
    monkeypatch.delenv("N4A_DAGML_INPROCESS", raising=False)
    monkeypatch.setattr(host.sys, "prefix", str(tmp_path / "isolated-prefix"))
    monkeypatch.setattr(host, "__file__", str(tmp_path / "sibling-source" / "studio_scientific.py"))
    _assert_code(_request(), "sibling_runtime_forbidden")


def test_input_is_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    before = copy.deepcopy(request)
    result = _Result()
    monkeypatch.setattr(host, "_ambient_runtime_preflight", lambda: None)
    monkeypatch.setattr(host, "_run", lambda **_kwargs: result)
    host.studio_scientific_job_v1(request)
    assert request == before
