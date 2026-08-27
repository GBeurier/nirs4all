"""Tests for the narrow target-free raw Methods portable replay compiler."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nirs4all.pipeline.dagml.fit_identity import normalize_predict_identity
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.raw_methods_replay import RawArrayMethodsReplayCompiler


class _Contract:
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    def to_dict(self) -> dict[str, Any]:
        return self._value


class _Context:
    def close(self) -> None:
        return None


class _Model:
    @classmethod
    def from_bytes(cls, context: _Context, payload: bytes) -> _Model:
        assert isinstance(context, _Context)
        assert payload == b"n4mm"
        return cls()

    def predict(self, context: _Context, values: np.ndarray) -> np.ndarray:
        assert isinstance(context, _Context)
        return values[:, :1] * 2.0

    def close(self) -> None:
        return None


def _package() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "fitted_artifact_mode": "portable_required",
        "execution_bundle": {"raw_artifact_payloads": {"artifact:model": [110, 52, 109, 109]}},
        "artifact_bindings": [{"artifact_id": "artifact:model", "load_mode": "native_portable"}],
        "training_outcome": {"outcome_fingerprint": "a" * 64},
        "data_identities": [{"requirement_key": "model:methods.x"}],
        "output_bindings": [{"binding_id": "binding:prediction", "node_id": "model:methods", "target_names": ["protein"]}],
    }


def _template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema_fingerprint": "b" * 64,
        "plan_fingerprint": "c" * 64,
        "relation_fingerprint": "d" * 64,
        "target_content_fingerprint": "e" * 64,
        "coordinator_relations": {"records": []},
    }


def _facade_module(calls: dict[str, Any]) -> SimpleNamespace:
    def attach(envelope: dict[str, Any], cohort: dict[str, Any]) -> _Contract:
        calls["envelope"] = envelope
        calls["cohort"] = cohort
        upgraded = dict(envelope)
        upgraded["schema_version"] = 2
        upgraded["predict_cohort"] = {"cohort_fingerprint": "f" * 64}
        return _Contract(upgraded)

    def sign(request: dict[str, Any]) -> _Contract:
        calls["request"] = request
        signed = dict(request)
        signed["request_fingerprint"] = "1" * 64
        return _Contract(signed)

    return SimpleNamespace(
        attach_predict_cohort_to_envelope=attach,
        sign_training_replay_request=sign,
    )


def test_raw_methods_replay_compiler_builds_target_free_signed_predict_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    module_name = "dag_ml_raw_methods_replay_test"
    monkeypatch.setitem(sys.modules, module_name, _facade_module(calls))
    estimator = SimpleNamespace(
        predictor_package_=_Contract(_package()),
        training_execution_=SimpleNamespace(data_envelopes={"model:methods.x": _template()}),
    )
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    frame = normalize_predict_identity(X, sample_ids=["pred-a", "pred-b"], metadata={"batch": ["a", "b"]})

    replay = RawArrayMethodsReplayCompiler(
        dagml_module=module_name,
        context_type=_Context,
        model_type=_Model,
    ).compile_replay(estimator, X, mode="predict", identity_frame=frame)

    assert calls["envelope"]["data_content_fingerprint"] == frame.data_content_fingerprint
    assert "target_content_fingerprint" not in calls["envelope"]
    assert calls["cohort"]["role"] == "inference"
    assert calls["cohort"]["target_names"] == ["protein"]
    assert calls["cohort"]["relations"]["rows"] == [
        {
            "observation_id": "pred-a",
            "sample_id": "pred-a",
            "source_id": "src0",
            "target_id": "y",
            "group_id": None,
            "origin_id": None,
            "repetition_id": None,
            "augmented": False,
            "excluded": False,
            "metadata": {"batch": "a"},
        },
        {
            "observation_id": "pred-b",
            "sample_id": "pred-b",
            "source_id": "src0",
            "target_id": "y",
            "group_id": None,
            "origin_id": None,
            "repetition_id": None,
            "augmented": False,
            "excluded": False,
            "metadata": {"batch": "b"},
        },
    ]
    assert calls["request"]["phase"] == "predict"
    assert calls["request"]["data_envelope_keys"] == ["model:methods.x"]
    assert calls["request"]["output_binding_ids"] == ["binding:prediction"]
    assert replay.artifact_handles == {}

    handle = replay.artifact_callback(
        {
            "operation": "hydrate",
            "request": {"controller_id": "controller:methods.pls", "artifact": {"kind": "n4m_model"}},
            "payload": b"n4mm",
        }
    )
    result = replay.op_callback(
        {
            "run_id": "run:nirs4all.raw_methods_predict",
            "phase": "PREDICT",
            "variant_id": None,
            "fold_id": None,
            "branch_path": [],
            "seed": None,
            "node_plan": {
                "kind": "model",
                "node_id": "model:methods",
                "controller_id": "controller:methods.pls",
                "controller_version": "1.0.0",
                "params_fingerprint": "a" * 64,
            },
            "data_views": {"x": {"partition": "predict", "sample_ids": ["pred-b", "pred-a"]}},
            "input_handles": {"model": handle},
        }
    )
    assert result["predictions"][0]["sample_ids"] == ["pred-b", "pred-a"]
    assert result["predictions"][0]["values"] == [[6.0], [2.0]]
    replay.artifact_callback({"operation": "release", "handle": handle})


def test_raw_methods_replay_compiler_refuses_host_sidecar_before_cohort_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "dag_ml_raw_methods_replay_refusal_test"
    monkeypatch.setitem(sys.modules, module_name, _facade_module({}))
    package = _package()
    package["fitted_artifact_mode"] = "allow_host_sidecar"
    estimator = SimpleNamespace(
        predictor_package_=_Contract(package),
        training_execution_=SimpleNamespace(data_envelopes={"model:methods.x": _template()}),
    )
    X = np.asarray([[1.0, 2.0]])
    frame = normalize_predict_identity(X, sample_ids=["pred-a"])

    with pytest.raises(DagMLNativeCoverageError, match="host-sidecar"):
        RawArrayMethodsReplayCompiler(dagml_module=module_name).compile_replay(
            estimator,
            X,
            mode="predict",
            identity_frame=frame,
        )
