"""Unit proofs for native calibration replay compilation."""

from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

from nirs4all.pipeline.dagml.native_conformal_calibration import (
    NativeConformalCalibrationError,
    compile_methods_conformal_calibration_replay,
)


def _package() -> dict[str, object]:
    return {
        "schema_version": 2,
        "training_outcome": {"outcome_fingerprint": "a" * 64},
        "execution_bundle": {
            "raw_artifact_payloads": {"artifact:model": [1, 2, 3]},
            "refit_artifacts": [{"artifact_id": "artifact:model", "kind": "n4m_model"}],
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


def _install_fake_dagml(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    runtime = types.SimpleNamespace()

    def fingerprint(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()

    def sign(request: dict[str, object]) -> dict[str, object]:
        signed = dict(request)
        signed["request_fingerprint"] = "d" * 64
        return signed

    runtime.sample_relation_set_fingerprint_json = fingerprint
    runtime.sign_training_replay_request = sign
    monkeypatch.setitem(sys.modules, "dag_ml_native_conformal_test", runtime)
    return runtime


def test_native_calibration_replay_preserves_exact_truth_and_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_dagml(monkeypatch)
    replay = compile_methods_conformal_calibration_replay(
        _package(),
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.5, 2.5]),
        sample_ids=["calibration.one", "calibration.two"],
        groups=["g1", "g2"],
        dagml_module="dag_ml_native_conformal_test",
        methods_library_path="/native/libn4m.so",
    )

    assert replay.binding_id == "binding:prediction"
    assert replay.truth == {
        "sample_ids": ["calibration.one", "calibration.two"],
        "values": [[1.5], [2.5]],
    }
    assert replay.execution.request["phase"] == "PREDICT"
    assert replay.execution.request["request_fingerprint"] == "d" * 64
    assert replay.calibration_relations["records"] == [
        {
            "observation_id": "calibration.one",
            "sample_id": "calibration.one",
            "target_id": None,
            "group_id": "g1",
            "origin_sample_id": None,
            "source_id": None,
            "is_augmented": False,
            "metadata": {},
        },
        {
            "observation_id": "calibration.two",
            "sample_id": "calibration.two",
            "target_id": None,
            "group_id": "g2",
            "origin_sample_id": None,
            "source_id": None,
            "is_augmented": False,
            "metadata": {},
        },
    ]


@pytest.mark.parametrize(
    ("sample_ids", "y", "message"),
    [
        (None, np.asarray([1.0, 2.0]), "explicit sample_ids"),
        (["one", "two"], np.asarray([1.0]), "row-aligned"),
        (["one", "two"], np.asarray([1.0, np.nan]), "non-finite"),
    ],
)
def test_native_calibration_replay_refuses_unattested_truth(
    monkeypatch: pytest.MonkeyPatch,
    sample_ids: object,
    y: np.ndarray,
    message: str,
) -> None:
    _install_fake_dagml(monkeypatch)
    with pytest.raises(NativeConformalCalibrationError, match=message):
        compile_methods_conformal_calibration_replay(
            _package(),
            np.asarray([[1.0], [2.0]]),
            y,
            sample_ids=sample_ids,  # type: ignore[arg-type]
            dagml_module="dag_ml_native_conformal_test",
        )
