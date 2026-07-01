"""Unit tests for the runtime envelopes (B-018 / L10): ``RtResult`` / ``RtRunRequest`` / ``RtError``.

Covers the pure-projection contract — ``RtError`` classification + wire shape, ``RtResult.from_native_dir``
round-trip, ``RtResult.from_run_result`` for BOTH a dag-ml result (reports verbatim) and a legacy result
(sparse + carried diagnostics), the ``RunResult.to_rt_result()`` seam, the public ``nirs4all.runtime``
accessors, and (when the sibling ecosystem + dag-ml checkouts are present) JSON-Schema validation of every
``to_dict()`` against ``docs/contracts/runtime/*.v1.schema.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import nirs4all.runtime as runtime
from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import dagml_bridge
from nirs4all.pipeline.dagml.errors import DagMlUnavailable, DagMlUnsupported
from nirs4all.pipeline.dagml.native_results import write_native_results
from nirs4all.pipeline.dagml.rt import (
    RT_ERROR_CAUSES,
    RT_RESULT_SCHEMA_VERSION,
    RtError,
    RtResult,
    RtRunRequest,
)

# --------------------------------------------------------------------------- #
# Fixtures: a small native ScoreSet + dag-ml / legacy RunResult.
# --------------------------------------------------------------------------- #


def _score_set() -> dict:
    """A minimal but schema-shaped dag-ml ScoreSet (two reports: a CV avg + a refit final)."""
    return {
        "schema_version": 1,
        "plan_id": "plan:test",
        "reports": [
            {
                "producer_node": "model:0",
                "partition": "validation",
                "fold_id": "avg",
                "level": "observation",
                "row_count": 3,
                "target_width": 1,
                "target_names": ["y"],
                "variant_id": "variant:base",
                "prediction_id": None,
                "variant_label": None,
                "metrics": {"rmse": 0.5},
            },
            {
                "producer_node": "model:0",
                "partition": "final",
                "fold_id": None,
                "level": "observation",
                "row_count": 2,
                "target_width": 1,
                "target_names": ["y"],
                "variant_id": "variant:base",
                "prediction_id": None,
                "variant_label": None,
                "metrics": {"rmse": 0.4},
            },
        ],
    }


def _dagml_result() -> RunResult:
    """An in-memory dag-ml ``RunResult`` (predictions + stashed raw ScoreSet), as the projection produces."""
    preds = Predictions()
    preds.add_prediction(
        dataset_name="ds", config_name="config_abc", model_name="PLS", partition="val", fold_id="avg",
        sample_indices=[0, 1, 2], y_true=np.array([1.0, 2.0, 3.0]), y_pred=np.array([1.1, 2.1, 2.9]),
        val_score=0.5, metric="rmse", task_type="regression",
    )
    preds.add_prediction(
        dataset_name="ds", config_name="config_abc", model_name="PLS", partition="test", fold_id="final",
        sample_indices=[0, 1], y_true=np.array([1.0, 2.0]), y_pred=np.array([1.0, 2.0]),
        test_score=0.4, metric="rmse", task_type="regression",
    )
    preds.flush()
    result = RunResult(predictions=preds, per_dataset={"ds": {"engine": "dag-ml"}})
    result._dagml_score_set = _score_set()  # noqa: SLF001
    result._dagml_refit_artifacts = []  # noqa: SLF001
    return result


def _legacy_result() -> RunResult:
    """A legacy ``RunResult`` (no native ScoreSet) — the shape the transparent fallback returns."""
    preds = Predictions()
    preds.add_prediction(
        dataset_name="ds", config_name="config_abc", model_name="PLS", partition="test", fold_id="final",
        sample_indices=[0, 1], y_true=np.array([1.0, 2.0]), y_pred=np.array([1.0, 2.0]),
        test_score=0.4, metric="rmse", task_type="regression",
    )
    preds.flush()
    return RunResult(predictions=preds, per_dataset={"ds": {"engine": "legacy"}})


# --------------------------------------------------------------------------- #
# RtError.
# --------------------------------------------------------------------------- #


def test_rterror_classifies_unavailable_backend() -> None:
    err = RtError.from_dagml_error(DagMlUnavailable("no backend"), verb="run")
    assert err.cause == "unavailable_backend"
    assert err.verb == "run"
    assert "not available" in err.message
    assert err.mitigation  # a non-empty remedy


def test_rterror_classifies_unsupported_shape() -> None:
    for exc in (DagMlUnsupported("nope"), NotImplementedError("also nope")):
        err = RtError.from_dagml_error(exc, verb="run")
        assert err.cause == "unsupported_shape"
        assert err.mitigation


def test_rterror_is_raisable_but_not_a_fallback_signal() -> None:
    # Must NOT subclass the two fallback signals, else run()'s except would re-catch it.
    err = RtError("run", "unsupported_shape", "x")
    assert isinstance(err, Exception)
    assert not isinstance(err, (DagMlUnsupported, NotImplementedError))
    with pytest.raises(RtError):
        raise err


def test_rterror_rejects_unknown_cause() -> None:
    with pytest.raises(ValueError, match="unknown RtError cause"):
        RtError("run", "not_a_cause", "x")
    assert "unsupported_shape" in RT_ERROR_CAUSES


def test_rterror_to_dict_omits_none_optionals() -> None:
    bare = RtError("run", "runtime_error", "boom").to_dict()
    assert bare == {"verb": "run", "cause": "runtime_error", "message": "boom"}
    full = RtError(
        "inspect", "unsupported_capability", "x", mitigation="do y",
        unsupported_capability="needs_python_gil", portable_level="host_specific",
    ).to_dict()
    assert full["mitigation"] == "do y"
    assert full["unsupported_capability"] == "needs_python_gil"
    assert full["portable_level"] == "host_specific"


# --------------------------------------------------------------------------- #
# RtResult.from_native_dir (round-trip through the native triple).
# --------------------------------------------------------------------------- #


def test_from_native_dir_round_trip(tmp_path: Path) -> None:
    result = _dagml_result()
    score_set = result._dagml_score_set  # noqa: SLF001
    assert score_set is not None
    run_dir = write_native_results(result, score_set, tmp_path)

    rt = RtResult.from_native_dir(run_dir)
    assert rt.schema_version == RT_RESULT_SCHEMA_VERSION
    # reports are VERBATIM score_set.reports[] (the join key).
    assert rt.reports == score_set["reports"]
    assert rt.plan_id == "plan:test"
    assert rt.manifest is not None
    assert rt.manifest["engine"] == "dag-ml"
    assert rt.manifest["portable_level"] is None
    assert rt.manifest["fingerprints"]["score_set_hash"]  # carried from the manifest
    assert rt.manifest["files"]["score_set"] == "score_set.json"
    # predictions are projected blocks; the parquet round-trip preserves the rows.
    assert rt.predictions
    assert all("partition" in row and "sample_indices" in row for row in rt.predictions)
    # the envelope is JSON-serialisable (the wire form).
    json.dumps(rt.to_dict())


def test_runtime_from_native_dir_is_public_seam(tmp_path: Path) -> None:
    result = _dagml_result()
    score_set = result._dagml_score_set  # noqa: SLF001
    assert score_set is not None
    run_dir = write_native_results(result, score_set, tmp_path)
    rt = runtime.from_native_dir(run_dir)
    assert isinstance(rt, RtResult)
    assert rt.reports == score_set["reports"]


# --------------------------------------------------------------------------- #
# RtResult.from_run_result + RunResult.to_rt_result().
# --------------------------------------------------------------------------- #


def test_from_run_result_dagml_reports_verbatim() -> None:
    result = _dagml_result()
    score_set = result._dagml_score_set  # noqa: SLF001
    assert score_set is not None
    rt = result.to_rt_result()
    assert rt.manifest is not None
    assert rt.manifest["engine"] == "dag-ml"
    assert rt.reports == score_set["reports"]
    assert rt.plan_id == "plan:test"
    assert rt.predictions  # the in-memory predictions are projected
    json.dumps(rt.to_dict())


def test_from_run_result_legacy_is_sparse_and_carries_diagnostics() -> None:
    result = _legacy_result()
    rt_error = RtError("run", "unsupported_shape", "shape X not covered", mitigation="use legacy")
    setattr(result, "_rt_diagnostics", [rt_error])  # as run()'s fallback attaches it

    rt = result.to_rt_result()
    assert rt.manifest is not None
    assert rt.manifest["engine"] == "legacy"
    assert rt.reports == []  # no native ScoreSet
    assert rt.plan_id is None
    assert len(rt.diagnostics) == 1
    payload = rt.to_dict()
    assert payload["diagnostics"][0]["cause"] == "unsupported_shape"


def test_to_rt_result_is_pure_projection() -> None:
    # Calling the seam must not mutate the source result.
    result = _dagml_result()
    before = result.num_predictions
    result.to_rt_result()
    assert result.num_predictions == before
    assert result._dagml_score_set == _score_set()  # noqa: SLF001 — untouched


# --------------------------------------------------------------------------- #
# RtRunRequest + runtime manifest accessor.
# --------------------------------------------------------------------------- #


def test_rt_run_request_to_dict() -> None:
    req = RtRunRequest(
        pipeline_dsl={"steps": []}, dataset_ref="sample_data/regression",
        cv={"folds": 5, "seed": 42}, execution_backend="local-python",
        options={"name": "exp", "engine": "dag-ml", "allow_fallback": False},
    )
    payload = req.to_dict()
    assert set(payload) == {"pipeline_dsl", "dataset_ref", "cv", "execution_backend", "options"}
    assert payload["execution_backend"] == "local-python"
    assert payload["options"]["allow_fallback"] is False
    json.dumps(payload)


def test_runtime_list_controller_manifests() -> None:
    manifests = runtime.list_controller_manifests()
    assert isinstance(manifests, list) and manifests
    assert sorted(m["controller_id"] for m in manifests) == [
        "controller:nirs4all.merge_concat",
        "controller:nirs4all.meta_model",
        "controller:nirs4all.model",
        "controller:nirs4all.transform",
        "controller:nirs4all.y_transform",
    ]
    # JSON-ready (the inspect verb surface).
    json.dumps(manifests)


def test_runtime_controller_manifests_validate_against_dagml_schema() -> None:
    dag_ml = pytest.importorskip("dag_ml")

    manifests = runtime.list_controller_manifests()
    for manifest in manifests:
        dag_ml.ControllerManifest(manifest)
    dag_ml.ControllerManifests(manifests)


def test_runtime_controller_manifests_use_dagml_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_specs = dagml_bridge._controller_manifest_specs()  # noqa: SLF001
    expected_manifests = dagml_bridge._fallback_controller_manifests()  # noqa: SLF001
    seen: dict[str, Any] = {}

    class HostControllerSpec:
        def __init__(self, payload: dict[str, Any]) -> None:
            assert "capabilities" not in payload
            assert "supported_phases" not in payload
            self.payload = payload

        def to_dict(self) -> dict[str, Any]:
            return self.payload

    class DerivedManifest:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, Any]:
            return self.payload

    def derive_controller_manifests(specs: list[dict[str, Any]]) -> list[DerivedManifest]:
        seen["payloads"] = specs
        return [DerivedManifest(manifest) for manifest in expected_manifests]

    fake_dag_ml = ModuleType("dag_ml")
    setattr(fake_dag_ml, "HostControllerSpec", HostControllerSpec)
    setattr(fake_dag_ml, "derive_controller_manifests", derive_controller_manifests)
    monkeypatch.setitem(sys.modules, "dag_ml", fake_dag_ml)

    manifests = runtime.list_controller_manifests()

    assert seen["payloads"] == expected_specs
    assert manifests == expected_manifests


def test_runtime_controller_manifests_fallback_without_helper_has_no_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dag_ml = ModuleType("dag_ml")
    setattr(fake_dag_ml, "HostControllerSpec", object)
    monkeypatch.setitem(sys.modules, "dag_ml", fake_dag_ml)

    manifests = runtime.list_controller_manifests()

    assert manifests == dagml_bridge._fallback_controller_manifests()  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Optional: validate to_dict() against the ecosystem wire schemas when present.
# --------------------------------------------------------------------------- #


def _runtime_schema_dir() -> Path | None:
    """Locate the sibling ``nirs4all-ecosystem/docs/contracts/runtime`` dir, or ``None`` if absent."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "nirs4all-ecosystem" / "docs" / "contracts" / "runtime"
        if candidate.is_dir():
            return candidate
    return None


def _build_registry(schema_dir: Path):
    """A ``referencing`` Registry mapping every available runtime + dag-ml schema by its ``$id`` to local files."""
    from referencing import Registry, Resource

    resources = []
    for path in schema_dir.glob("*.v1.schema.json"):
        schema = json.loads(path.read_text())
        resources.append((schema["$id"], Resource.from_contents(schema)))
    dagml_contracts = schema_dir.parent.parent.parent.parent / "dag-ml" / "docs" / "contracts"
    for name in ("score_set.schema.json", "selection_decision.schema.json"):
        path = dagml_contracts / name
        if path.is_file():
            schema = json.loads(path.read_text())
            resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@pytest.mark.parametrize("envelope", ["error", "run_request", "result"])
def test_to_dict_validates_against_ecosystem_schema(tmp_path: Path, envelope: str) -> None:
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    schema_dir = _runtime_schema_dir()
    if schema_dir is None:
        pytest.skip("sibling nirs4all-ecosystem runtime schemas not checked out")
    assert schema_dir is not None

    import jsonschema

    registry = _build_registry(schema_dir)
    if envelope == "error":
        schema_id = "https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_error.v1.schema.json"
        instance = RtError("run", "unsupported_shape", "x", mitigation="use legacy").to_dict()
    elif envelope == "run_request":
        schema_id = "https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_run_request.v1.schema.json"
        instance = RtRunRequest(pipeline_dsl={"steps": []}, dataset_ref="ds", execution_backend="local-python").to_dict()
    else:
        schema_id = "https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_result.v1.schema.json"
        result = _dagml_result()
        run_dir = write_native_results(result, result._dagml_score_set, tmp_path)  # noqa: SLF001
        instance = RtResult.from_native_dir(run_dir).to_dict()

    schema = registry.get_or_retrieve(schema_id).value.contents
    validator = jsonschema.Draft202012Validator(schema, registry=registry)
    validator.validate(instance)
