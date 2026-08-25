from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from nirs4all.pipeline.dagml.fit_identity import normalize_predict_identity
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveReplayError,
    predict_methods_archive_v2_raw,
    predict_methods_archive_v2_raw_result,
    write_methods_archive_v2,
)
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


def test_raw_archive_predict_composes_core_dagml_and_methods_without_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.resolve_methods_library_path",
        lambda path: str(path),
    )
    runtime = _install_fake_runtime(monkeypatch)
    package_json = json.dumps(_package())

    class _Package:
        def __init__(self, raw: str) -> None:
            self._document = json.loads(raw)

        def to_dict(self) -> dict[str, object]:
            return self._document

    def replay(
        package: _Package,
        request: dict[str, object],
        envelopes: dict[str, object],
        methods_inputs: dict[str, object],
        *,
        methods_library_path: str,
        outcome_id: str,
        run_id: str,
    ) -> dict[str, object]:
        _ = (package, request, envelopes, methods_inputs, methods_library_path, outcome_id, run_id)
        return {
            "outputs": [
                {
                    "predictions": [
                        {
                            "sample_ids": ["sample.one", "sample.two"],
                            "values": [[1.5], [2.5]],
                        }
                    ]
                }
            ]
        }

    runtime.PortablePredictorPackage = _Package
    runtime.replay_loaded_methods_predictor_package = replay
    monkeypatch.setitem(sys.modules, "dag_ml", runtime)
    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(read_portable_predictor_package_v2=lambda _path: package_json.encode()),
    )

    values = predict_methods_archive_v2_raw(
        "portable.n4a", np.asarray([[1.0], [2.0]]), sample_ids=["sample.one", "sample.two"], methods_library_path="/native/libn4m.so"
    )

    assert values.tolist() == [[1.5], [2.5]]
    assert runtime.last_request["phase"] == "PREDICT"

    def mismatched_replay(*args: object, **kwargs: object) -> dict[str, object]:
        return {"outputs": [{"predictions": [{"sample_ids": ["sample.two"], "values": [[1.0]]}]}]}

    runtime.replay_loaded_methods_predictor_package = mismatched_replay
    with pytest.raises(NativeArchiveReplayError, match="identities do not exactly match"):
        predict_methods_archive_v2_raw(
            "portable.n4a", np.asarray([[1.0], [2.0]]), sample_ids=["sample.one", "sample.two"], methods_library_path="/native/libn4m.so"
        )


def test_raw_archive_predict_resolves_the_bundled_methods_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_fake_runtime(monkeypatch)
    package_json = json.dumps(_package())

    class _Package:
        def __init__(self, raw: str) -> None:
            self._document = json.loads(raw)

        def to_dict(self) -> dict[str, object]:
            return self._document

    runtime.PortablePredictorPackage = _Package
    runtime.replay_loaded_methods_predictor_package = lambda *_args, **_kwargs: {
        "outputs": [
            {
                "predictions": [
                    {"sample_ids": ["sample.one"], "values": [[1.5]]}
                ]
            }
        ]
    }
    monkeypatch.setitem(sys.modules, "dag_ml", runtime)
    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(
            read_portable_predictor_package_v2=lambda _path: package_json.encode()
        ),
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.resolve_methods_library_path",
        lambda _path=None: "/wheel/libn4m.so",
    )

    values = predict_methods_archive_v2_raw(
        "portable.n4a", np.asarray([[1.0]]), sample_ids=["sample.one"]
    )

    assert values.tolist() == [[1.5]]


def test_raw_archive_predict_projects_exact_native_conformal_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.resolve_methods_library_path",
        lambda path: str(path),
    )
    runtime = _install_fake_runtime(monkeypatch)
    package = _package()
    package["conformal_calibration"] = {
        "calibration_fingerprint": "f" * 64,
        "multi_target_policy": "marginal",
        "quantiles": [
            {
                "coverage": 0.9,
                "radii": [{"status": "finite", "value": 0.5}],
            }
        ],
    }

    class _Package:
        def __init__(self, raw: str) -> None:
            self._document = json.loads(raw)

        def to_dict(self) -> dict[str, object]:
            return self._document

    def replay(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return {
            "outputs": [
                {
                    "binding": {"binding_id": "binding:prediction"},
                    "predictions": [
                        {
                            "sample_ids": ["sample.one", "sample.two"],
                            "values": [[1.5], [2.5]],
                        }
                    ],
                }
            ],
            "conformal_intervals": [
                {
                    "binding_id": "binding:prediction",
                    "sample_ids": ["sample.one", "sample.two"],
                    "calibration_fingerprint": "f" * 64,
                    "point_prediction_fingerprint": "e" * 64,
                    "intervals": [
                        {
                            "coverage": 0.9,
                            "cells": [
                                [{"status": "finite", "lower": 1.0, "upper": 2.0}],
                                [{"status": "finite", "lower": 2.0, "upper": 3.0}],
                            ],
                        }
                    ],
                }
            ],
        }

    runtime.PortablePredictorPackage = _Package
    runtime.replay_loaded_methods_predictor_package = replay
    monkeypatch.setitem(sys.modules, "dag_ml", runtime)
    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(
            read_portable_predictor_package_v2=lambda _path: json.dumps(package).encode()
        ),
    )

    result = predict_methods_archive_v2_raw_result(
        "portable.n4a",
        np.asarray([[1.0], [2.0]]),
        sample_ids=["sample.one", "sample.two"],
        methods_library_path="/native/libn4m.so",
    )

    assert result.values.tolist() == [[1.5], [2.5]]
    assert result.intervals[0.9].qhat == pytest.approx(np.asarray([0.5]))
    np.testing.assert_allclose(result.intervals[0.9].lower, [[1.0], [2.0]])
    np.testing.assert_allclose(result.intervals[0.9].upper, [[2.0], [3.0]])
    assert result.conformal_guarantee_status == {
        "version": 2,
        "status": "active",
        "method": "split_absolute_residual",
        "unit": "physical_sample",
        "coverage": [0.9],
        "calibrated_coverages": [0.9],
        "multi_target": "marginal",
        "calibration_fingerprint": "f" * 64,
        "source": "dag_ml_portable_predictor_package_v2",
    }

    def unbounded(*_args, **_kwargs):  # noqa: ANN002, ANN003
        payload = replay()
        payload["conformal_intervals"][0]["intervals"][0]["cells"][0][0] = {"status": "unbounded"}
        return payload

    runtime.replay_loaded_methods_predictor_package = unbounded
    with pytest.raises(NativeArchiveReplayError, match="unbounded conformal interval"):
        predict_methods_archive_v2_raw_result(
            "portable.n4a",
            np.asarray([[1.0], [2.0]]),
            sample_ids=["sample.one", "sample.two"],
            methods_library_path="/native/libn4m.so",
        )


def test_native_archive_writer_composes_dagml_and_core_without_rebuilding_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []

    def assemble(archive_id: str, outcome: object, package: object) -> tuple[dict[str, object], dict[str, bytes]]:
        observed.extend([archive_id, outcome, package])
        return ({"schema_version": 2}, {"dagml/portable_predictor_package.json": b"package"})

    def write(path: str, manifest: dict[str, object], members: dict[str, bytes]) -> dict[str, str]:
        observed.extend([path, manifest, members])
        return {"archive_id": "archive:methods", "archive_sha256": "e" * 64}

    monkeypatch.setitem(
        sys.modules,
        "dag_ml",
        types.SimpleNamespace(build_archive_v2_native_portable_payloads=assemble),
    )
    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(write_archive_v2_from_native_payloads=write),
    )

    reference = write_methods_archive_v2(
        "portable.n4a",
        archive_id="archive:methods",
        outcome={"outcome": "native"},
        package={"package": "native"},
    )

    assert reference == {"archive_id": "archive:methods", "archive_sha256": "e" * 64}
    assert observed == [
        "archive:methods",
        {"outcome": "native"},
        {"package": "native"},
        "portable.n4a",
        {"schema_version": 2},
        {"dagml/portable_predictor_package.json": b"package"},
    ]
