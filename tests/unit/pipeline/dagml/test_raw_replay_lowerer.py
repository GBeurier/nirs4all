from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from nirs4all.pipeline.dagml.fit_identity import normalize_predict_identity
from nirs4all.pipeline.dagml.raw_replay_lowerer import (
    RawArrayMethodsReplayCompiler,
    RawArrayMethodsReplayError,
)


class _Context:
    def close(self) -> None:
        return None


class _Model:
    @classmethod
    def from_bytes(cls, context: _Context, payload: bytes) -> _Model:
        _ = (context, payload)
        return cls()

    def close(self) -> None:
        return None


def _package() -> dict[str, object]:
    return {
        "schema_version": 2,
        "training_outcome": {"outcome_fingerprint": "a" * 64},
        "execution_bundle": {
            "raw_artifact_payloads": {"artifact:model": [1, 2, 3]},
            "refit_artifacts": [
                {"artifact_id": "artifact:model", "kind": "n4m_model"}
            ],
            "data_requirements": [
                {
                    "node_id": "model:base",
                    "input_name": "x",
                    "schema_fingerprint": "b" * 64,
                    "plan_fingerprint": "c" * 64,
                }
            ],
        },
        "output_bindings": [
            {
                "binding_id": "binding:prediction",
                "node_id": "model:base",
                "target_names": ["y"],
            }
        ],
    }


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    runtime = types.SimpleNamespace()
    runtime.last_request = None

    def fingerprint(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()

    def sign(request: dict[str, object]) -> dict[str, object]:
        runtime.last_request = request
        signed = dict(request)
        signed["request_fingerprint"] = "d" * 64
        return signed

    runtime.sample_relation_set_fingerprint_json = fingerprint
    runtime.sign_training_replay_request = sign
    monkeypatch.setitem(sys.modules, "dag_ml_raw_replay_test", runtime)
    monkeypatch.setitem(
        sys.modules,
        "pls4all",
        types.SimpleNamespace(Context=_Context, Model=_Model),
    )
    return runtime


def test_raw_replay_compiler_builds_target_free_current_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_fake_runtime(monkeypatch)
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    identity = normalize_predict_identity(X, sample_ids=["sample.one", "sample.two"])
    compiler = RawArrayMethodsReplayCompiler(
        _package(), dagml_module="dag_ml_raw_replay_test"
    )

    replay = compiler.compile_replay(None, X, mode="predict", identity_frame=identity)  # type: ignore[arg-type]

    assert replay.request["request_fingerprint"] == "d" * 64
    assert runtime.last_request["data_envelope_keys"] == ["model:base.x"]
    envelope = replay.data_envelopes["model:base.x"]
    assert envelope["target_content_fingerprint"] is None
    assert envelope["data_content_fingerprint"] == identity.data_content_fingerprint
    assert [record["sample_id"] for record in envelope["coordinator_relations"]["records"]] == [
        "sample.one",
        "sample.two",
    ]
    assert replay.artifact_handles == {}
    assert callable(replay.artifact_callback)


def test_raw_replay_compiler_refuses_implicit_identities_and_missing_n4mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    X = np.asarray([[1.0], [2.0]])
    compiler = RawArrayMethodsReplayCompiler(
        _package(), dagml_module="dag_ml_raw_replay_test"
    )
    implicit = normalize_predict_identity(X)
    with pytest.raises(RawArrayMethodsReplayError, match="explicit current sample_ids"):
        compiler.compile_replay(None, X, mode="predict", identity_frame=implicit)  # type: ignore[arg-type]

    invalid = _package()
    invalid["execution_bundle"]["raw_artifact_payloads"] = {}  # type: ignore[index]
    explicit = normalize_predict_identity(X, sample_ids=["sample.one", "sample.two"])
    with pytest.raises(RawArrayMethodsReplayError, match="no durable raw Methods artifacts"):
        RawArrayMethodsReplayCompiler(
            invalid, dagml_module="dag_ml_raw_replay_test"
        ).compile_replay(None, X, mode="predict", identity_frame=explicit)  # type: ignore[arg-type]


def test_raw_replay_resolver_refuses_unknown_or_duplicated_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    X = np.asarray([[1.0], [2.0]])
    identity = normalize_predict_identity(X, sample_ids=["sample.one", "sample.two"])
    replay = RawArrayMethodsReplayCompiler(
        _package(), dagml_module="dag_ml_raw_replay_test"
    ).compile_replay(None, X, mode="predict", identity_frame=identity)  # type: ignore[arg-type]
    callbacks = replay.op_callback.__self__
    with pytest.raises(RawArrayMethodsReplayError, match="absent from the current cohort"):
        callbacks._resolver.resolve_features(["sample.unknown"], include_augmented=False)
    with pytest.raises(RawArrayMethodsReplayError, match="duplicate sample identities"):
        callbacks._resolver.resolve_features(
            ["sample.one", "sample.one"], include_augmented=False
        )
