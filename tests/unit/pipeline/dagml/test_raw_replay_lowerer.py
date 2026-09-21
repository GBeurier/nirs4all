from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from nirs4all.pipeline.dagml.fit_identity import normalize_predict_identity
from nirs4all.pipeline.dagml.methods_replay import (
    MethodsN4mmReplayCallbacks,
    MethodsPortableReplayError,
)
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveReplayError,
    predict_methods_archive_v2_raw,
    predict_methods_archive_v2_raw_result,
    project_methods_archive_v2_conformal_presentation,
    write_methods_archive_v2,
)
from nirs4all.pipeline.dagml.native_conformal_calibration import (
    NativeConformalCalibrationError,
    compile_methods_conformal_calibration_replay,
)
from nirs4all.pipeline.dagml.raw_replay_lowerer import (
    RawArrayMethodsReplayCompiler,
    RawArrayMethodsReplayError,
    _native_methods_refit_artifact_ids_from_bundle,
    validate_native_methods_package,
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


def _stack_package() -> dict[str, object]:
    """A portable graph with two N4MM refit nodes, as emitted by stacking."""

    package = _package()
    bundle = package["execution_bundle"]
    assert isinstance(bundle, dict)
    bundle["raw_artifact_payloads"] = {
        "artifact:pls": [1, 2, 3],
        "artifact:ridge": [4, 5, 6],
    }
    bundle["refit_artifacts"] = [
        {"artifact": {"id": "artifact:pls", "kind": "n4m_model", "backend": "raw"}},
        {"artifact": {"id": "artifact:ridge", "kind": "n4m_model", "backend": "raw"}},
    ]
    return package


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


def test_methods_replay_callbacks_hydrate_predict_and_release_exact_handle() -> None:
    closed: list[str] = []

    class Context:
        def close(self) -> None:
            closed.append("context")

    class Model:
        @classmethod
        def from_bytes(cls, context: Context, payload: bytes) -> Model:
            assert isinstance(context, Context)
            assert payload == b"model"
            return cls()

        def predict(self, context: Context, values: np.ndarray) -> np.ndarray:
            assert isinstance(context, Context)
            return np.asarray(values)[:, :1] + 0.5

        def close(self) -> None:
            closed.append("model")

    class Resolver:
        def resolve_features(self, sample_ids: list[str], *, include_augmented: bool) -> dict[str, np.ndarray]:
            assert sample_ids == ["sample.one", "sample.two"]
            assert include_augmented is False
            return {"values": np.asarray([[1.0, 2.0], [3.0, 4.0]])}

    callbacks = MethodsN4mmReplayCallbacks(
        Resolver(),
        target_names_by_node={"model:base": ["y"]},
        context_type=Context,
        model_type=Model,
    )
    handle = callbacks.artifact_callback(
        {
            "operation": "hydrate",
            "request": {"artifact": {"kind": "n4m_model"}, "controller_id": "controller:model"},
            "payload": list(b"model"),
        }
    )
    assert handle == {"handle": 1, "kind": "model", "owner_controller": "controller:model"}
    assert callbacks.active_handle_count == 1

    result = callbacks.op_callback(
        {
            "phase": "PREDICT",
            "run_id": "run:test",
            "node_plan": {
                "kind": "model",
                "node_id": "model:base",
                "controller_id": "controller:model",
                "controller_version": "1",
                "params_fingerprint": "a" * 64,
            },
            "input_handles": {"model": handle},
            "data_views": {"predict": {"partition": "predict", "sample_ids": ["sample.one", "sample.two"]}},
        }
    )
    assert result["predictions"][0]["values"] == [[1.5], [3.5]]
    assert result["predictions"][0]["target_names"] == ["y"]

    assert callbacks.artifact_callback({"operation": "release", "handle": handle}) is None
    assert callbacks.active_handle_count == 0
    assert closed == ["model", "context"]
    callbacks.close()


def test_methods_replay_callbacks_refuse_ambiguous_or_unsupported_events() -> None:
    class Resolver:
        def resolve_features(self, sample_ids: list[str], *, include_augmented: bool) -> dict[str, np.ndarray]:
            return {"values": np.ones((len(sample_ids), 1))}

    callbacks = MethodsN4mmReplayCallbacks(
        Resolver(),
        target_names_by_node={"model:base": ["y"]},
        context_type=_Context,
        model_type=_Model,
    )
    with pytest.raises(MethodsPortableReplayError, match="unknown .* callback operation"):
        callbacks.artifact_callback({"operation": "unknown"})
    with pytest.raises(MethodsPortableReplayError, match="PREDICT only"):
        callbacks.op_callback({"phase": "REFIT", "node_plan": {"kind": "model"}})
    with pytest.raises(MethodsPortableReplayError, match="non-model node"):
        callbacks.op_callback({"phase": "PREDICT", "node_plan": {"kind": "transform"}})


def test_native_conformal_replay_compiles_identity_bound_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch)
    replay = compile_methods_conformal_calibration_replay(
        _package(),
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([1.5, 2.5]),
        sample_ids=["sample.one", "sample.two"],
        groups=["g1", "g2"],
        metadata={"instrument": ["a", "b"]},
        dagml_module="dag_ml_raw_replay_test",
    )

    assert replay.binding_id == "binding:prediction"
    assert replay.truth == {"sample_ids": ["sample.one", "sample.two"], "values": [[1.5], [2.5]]}
    assert [record["sample_id"] for record in replay.calibration_relations["records"]] == ["sample.one", "sample.two"]
    envelope = replay.execution.data_envelopes["model:base.x"]
    assert envelope["target_content_fingerprint"] is not None
    assert replay.execution.request["phase"] == "PREDICT"


@pytest.mark.parametrize(
    ("X", "y", "sample_ids", "message"),
    [
        (np.ones((2, 1)), np.ones(2), None, "explicit sample_ids"),
        (np.asarray([[np.nan], [1.0]]), np.ones(2), ["a", "b"], "non-finite"),
        (np.ones((2, 1)), np.ones((2, 2)), ["a", "b"], "width does not match"),
    ],
)
def test_native_conformal_replay_rejects_unbound_or_invalid_truth(
    monkeypatch: pytest.MonkeyPatch,
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str] | None,
    message: str,
) -> None:
    _install_fake_runtime(monkeypatch)
    with pytest.raises(NativeConformalCalibrationError, match=message):
        compile_methods_conformal_calibration_replay(
            _package(), X, y, sample_ids=sample_ids, dagml_module="dag_ml_raw_replay_test"
        )


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


def test_raw_replay_accepts_complete_native_stack_and_refuses_python_callback_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stack needs every N4MM; the legacy callback cannot fake its OOF features."""

    _install_fake_runtime(monkeypatch)
    X = np.asarray([[1.0], [2.0]])
    identity = normalize_predict_identity(X, sample_ids=["sample.one", "sample.two"])
    package = _stack_package()

    assert validate_native_methods_package(package) is package
    with pytest.raises(RawArrayMethodsReplayError, match="multi-model Methods replay requires an explicit libn4m path"):
        RawArrayMethodsReplayCompiler(
            package, dagml_module="dag_ml_raw_replay_test"
        ).compile_replay(None, X, mode="predict", identity_frame=identity)  # type: ignore[arg-type]

    replay = RawArrayMethodsReplayCompiler(
        package,
        dagml_module="dag_ml_raw_replay_test",
        methods_library_path="/absolute/libn4m.so",
    ).compile_replay(None, X, mode="predict", identity_frame=identity)  # type: ignore[arg-type]
    assert replay.op_callback is None
    assert replay.methods_inputs is not None


def test_native_methods_refit_artifact_set_is_shared_by_v2_and_v3() -> None:
    package = _stack_package()
    bundle = package["execution_bundle"]
    assert isinstance(bundle, dict)

    assert _native_methods_refit_artifact_ids_from_bundle(
        bundle, package_label="Package V3"
    ) == ["artifact:pls", "artifact:ridge"]


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

    def refused_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("native provider rejected the current cohort")

    runtime.replay_loaded_methods_predictor_package = refused_replay
    with pytest.raises(NativeArchiveReplayError, match="DAG-ML Methods Archive V2 replay was refused"):
        predict_methods_archive_v2_raw(
            "portable.n4a", np.asarray([[1.0], [2.0]]), sample_ids=["sample.one", "sample.two"], methods_library_path="/native/libn4m.so"
        )

    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(
            read_portable_predictor_package_v2=lambda _path: (_ for _ in ()).throw(
                RuntimeError("corrupt archive")
            )
        ),
    )
    with pytest.raises(NativeArchiveReplayError, match="Core Archive V2 rejected"):
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


def test_raw_archive_projects_dagml_owned_scalar_conformal_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter transports the owner projection without interval arithmetic."""

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.resolve_methods_library_path",
        lambda path: str(path),
    )
    runtime = _install_fake_runtime(monkeypatch)
    package = _package()
    package["package_fingerprint"] = "a" * 64
    package["conformal_calibration"] = {
        "calibration_fingerprint": "b" * 64,
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

    outcome = {
        "outcome_fingerprint": "c" * 64,
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
                "calibration_fingerprint": "b" * 64,
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
    observed: dict[str, object] = {}

    def project(native_package, request, replay_outcome):  # noqa: ANN001
        observed.update(package=native_package, request=request, outcome=replay_outcome)
        return {
            "schema_version": 1,
            "package_fingerprint": "a" * 64,
            "replay_outcome_fingerprint": "c" * 64,
            "binding_id": "binding:prediction",
            "target_name": "y",
            "sample_ids": ["sample.one", "sample.two"],
            "point_predictions": [1.5, 2.5],
            "intervals": [
                {
                    "coverage": 0.9,
                    "lower": [1.0, 2.0],
                    "upper": [2.0, 3.0],
                    "qhat": 0.5,
                }
            ],
            "calibration_fingerprint": "b" * 64,
            "presentation_fingerprint": "d" * 64,
        }

    runtime.PortablePredictorPackage = _Package
    runtime.replay_loaded_methods_predictor_package = lambda *_args, **_kwargs: outcome
    runtime.build_conformal_presentation_v1 = project
    monkeypatch.setitem(sys.modules, "dag_ml", runtime)
    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core",
        types.SimpleNamespace(
            read_portable_predictor_package_v2=lambda _path: json.dumps(package).encode()
        ),
    )

    prediction = predict_methods_archive_v2_raw_result(
        "portable.n4a",
        np.asarray([[1.0], [2.0]]),
        sample_ids=["sample.one", "sample.two"],
        methods_library_path="/native/libn4m.so",
    )
    assert prediction.conformal_presentation is not None
    assert prediction.conformal_presentation["point_predictions"] == [1.5, 2.5]

    presentation = project_methods_archive_v2_conformal_presentation(
        "portable.n4a",
        np.asarray([[1.0], [2.0]]),
        sample_ids=["sample.one", "sample.two"],
        methods_library_path="/native/libn4m.so",
    )

    assert presentation["sample_ids"] == ["sample.one", "sample.two"]
    assert presentation["point_predictions"] == [1.5, 2.5]
    assert observed["package"].to_dict() == package
    assert observed["outcome"] is outcome
    assert observed["request"]["phase"] == "PREDICT"


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
