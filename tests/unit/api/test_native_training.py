"""Public native raw-array training entry point tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_refit_result import NativeMethodsRefitResult
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.native_training import fit_native_pipeline, refit_native_methods
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


@pytest.fixture
def native_library(tmp_path) -> str:
    """A path-shaped runtime is sufficient for client-boundary unit tests."""

    library = tmp_path / "libn4m.so"
    library.write_bytes(b"native")
    return str(library)


class _TrainingResult:
    outcome = {"native": True}
    outputs = [
        {
            "output_id": "output:prediction",
            "node_id": "model:compat.0",
            "port_name": "oof",
        }
    ]

    def export_portable_predictor_package(self, package_id: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "package_id": package_id,
            "execution_bundle": {
                "raw_artifact_payloads": {"artifact:model": [1, 2, 3]},
                "refit_artifacts": [
                    {"artifact_id": "artifact:model", "kind": "n4m_model"}
                ],
            },
        }


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def execute_training(self, *args: Any, **kwargs: Any) -> _TrainingResult:
        self.calls.append((args, kwargs))
        return _TrainingResult()

    def replay_loaded_predictor_package(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        return {
            "outputs": [
                {
                    "predictions": [
                        {
                            "sample_ids": ["predict-a"],
                            "values": [[3.5]],
                        }
                    ]
                }
            ]
        }


def test_refit_native_methods_forwards_target_contracts_to_v3_without_cv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class TargetContracts:
        def to_prepared(self) -> Any:
            return SimpleNamespace(
                request={"request_fingerprint": "a" * 64},
                data_envelopes={"model:base.x": {"envelope": True}},
                relations={"records": []},
                training_influence={"influence": True},
                methods_inputs={"model:base.x": {"x": [[1.0]], "y": [[2.0]]}},
            )

    class RefitClient:
        def execute_methods_portable_full_refit(self, *args: Any, **kwargs: Any) -> Any:
            observed["args"] = args
            observed["kwargs"] = kwargs
            return {"schema_version": 3, "package_fingerprint": "b" * 64}

    class Estimator:
        pipeline = [{"model": object()}]
        predictor_package_ = {"schema_version": 2, "package_fingerprint": "c" * 64}
        native_training_execution_ = SimpleNamespace(methods_library_path="/native/libn4m.so")
        dagml_module = "dag_ml_refit_test"

        @staticmethod
        def native_runtime_client() -> RefitClient:
            return RefitClient()

    source = object.__new__(NativeMethodsRunResult)
    source._native_estimator = Estimator()  # noqa: SLF001
    monkeypatch.setitem(
        sys.modules,
        "dag_ml_refit_test",
        SimpleNamespace(
            sign_training_request=lambda request: {
                **request,
                "request_fingerprint": "a" * 64,
            }
        ),
    )
    monkeypatch.setattr(
        "nirs4all.api.native_training.lower_raw_array_training_contracts",
        lambda *_args, **_kwargs: TargetContracts(),
    )
    result = refit_native_methods(
        source,
        {
            "X": np.asarray([[1.0], [2.0]]),
            "y": np.asarray([3.0, 4.0]),
            "sample_ids": ["target-one", "target-two"],
        },
        name="target",
    )

    assert isinstance(result, NativeMethodsRefitResult)
    assert observed["args"][0] is Estimator.predictor_package_
    assert observed["args"][1] == {"request_fingerprint": "a" * 64}
    assert observed["kwargs"] == {
        "methods_library_path": "/native/libn4m.so",
        "recipe_id": "recipe:nirs4all.full_refit.aaaaaaaaaaaaaaaa",
        "package_id": "package:nirs4all.full_refit.aaaaaaaaaaaaaaaa",
        "outcome_id": "outcome:nirs4all.full_refit.aaaaaaaaaaaaaaaa",
        "run_id": "run:nirs4all.full_refit.aaaaaaaaaaaaaaaa",
        "bundle_id": "bundle:nirs4all.full_refit.aaaaaaaaaaaaaaaa",
    }


def test_native_methods_refit_package_round_trips_without_a_legacy_exporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    class PackageV3:
        def __init__(self, payload: str) -> None:
            if payload != '{"schema_version":3,"package_fingerprint":"signed"}':
                raise ValueError("invalid strict V3 payload")
            self._payload = payload

        def json(self) -> str:
            return self._payload

    monkeypatch.setitem(
        sys.modules,
        "dag_ml_refit_package_test",
        SimpleNamespace(PortableRefitPackageV3=PackageV3),
    )
    result = NativeMethodsRefitResult(
        PackageV3('{"schema_version":3,"package_fingerprint":"signed"}'),
        methods_library_path="/native/libn4m.so",
        dagml_module="dag_ml_refit_package_test",
        decoder=lambda _outcome, _identity: np.asarray([[1.0]]),
    )
    output = result.save_package(tmp_path / "child.v3.json")
    assert output.read_text(encoding="utf-8") == result.package_json()
    restored = NativeMethodsRefitResult.load_package(
        output,
        methods_library_path="/native/libn4m.so",
        dagml_module="dag_ml_refit_package_test",
        decoder=lambda _outcome, _identity: np.asarray([[1.0]]),
    )
    assert restored.package_json() == result.package_json()
    with pytest.raises(FileExistsError):
        result.save_package(output)
    observed: dict[str, Any] = {}

    def assemble(archive_id: str, package: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
        observed["archive_id"] = archive_id
        observed["package"] = package
        return ({"schema_version": 3}, {"dagml/portable_refit_package.json": package.json().encode()})

    def write(path: Any, manifest: Any, members: Any) -> dict[str, str]:
        observed["write"] = (path, manifest, members)
        return {"archive_id": "archive:child", "archive_sha256": "a" * 64}

    def read(path: Any) -> bytes:
        observed["read"] = path
        return result.package_json().encode()

    monkeypatch.setitem(
        sys.modules,
        "nirs4all_core_refit_archive_test",
        SimpleNamespace(
            write_archive_v3_from_native_payloads=write,
            read_portable_refit_package_v3=read,
        ),
    )
    monkeypatch.setattr(
        sys.modules["dag_ml_refit_package_test"],
        "build_archive_v3_native_refit_payloads",
        assemble,
        raising=False,
    )
    archive = result.export(
        tmp_path / "child.n4a",
        core_module="nirs4all_core_refit_archive_test",
    )
    assert archive == tmp_path / "child.n4a"
    assert observed["package"] is result.package
    assert observed["archive_id"].startswith("native-refit:")
    assert observed["write"] == (
        archive,
        {"schema_version": 3},
        {"dagml/portable_refit_package.json": result.package_json().encode()},
    )
    restored_archive = NativeMethodsRefitResult.load_archive(
        archive,
        methods_library_path="/native/libn4m.so",
        dagml_module="dag_ml_refit_package_test",
        decoder=lambda _outcome, _identity: np.asarray([[1.0]]),
        core_module="nirs4all_core_refit_archive_test",
    )
    assert observed["read"] == archive
    assert restored_archive.package_json() == result.package_json()


def test_fit_native_pipeline_is_a_public_strict_native_composition(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    captured: dict[str, Any] = {}

    def compile_fit(self, estimator, X, y, **kwargs):  # noqa: ANN001
        captured.update(estimator=estimator, X=X, y=y, kwargs=kwargs)
        from nirs4all.pipeline.dagml.estimator import DagMLTrainingExecution

        return DagMLTrainingExecution(
            request={"request": "native"},
            data_envelopes={"model.x": {}},
            relations={"records": []},
            training_influence={"entries": []},
            op_callback=lambda task: task,
            outcome_id="outcome:native",
            run_id="run:native",
            bundle_id="bundle:native",
        )

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        compile_fit,
    )
    client = _Client()
    estimator = fit_native_pipeline(
        [{"split": "stub"}, {"model": "stub"}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        sample_ids=["fit-a", "fit-b"],
        native_client=client,
        methods_library_path=native_library,
    )

    assert isinstance(estimator, DagMLPipelineEstimator)
    assert captured["kwargs"]["sample_ids"] == ("fit-a", "fit-b")
    assert client.calls[0][0][:4] == (
        {"request": "native"},
        {"model.x": {}},
        {"records": []},
        {"entries": []},
    )
    assert estimator.predictor_package_ == _TrainingResult().export_portable_predictor_package(
        "outcome:native-predictor"
    )
    assert estimator.prediction_compiler is not None
    assert estimator.prediction_identity_decoder is not None
    assert nirs4all.fit_native_pipeline is fit_native_pipeline


@pytest.mark.parametrize(
    "sample_ids, message",
    [(None, "explicit sample_ids"), ([], "sample_ids length")],
)
def test_fit_native_pipeline_refuses_unportable_inputs_before_training(
    sample_ids: Any,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        fit_native_pipeline(
            [],
            np.asarray([[1.0]]),
            np.asarray([1.0]),
            sample_ids=sample_ids,
        )


def test_fit_native_pipeline_surfaces_missing_package_as_native_coverage_error(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    class NoPackage(_TrainingResult):
        def export_portable_predictor_package(self, package_id: str) -> None:
            _ = package_id
            return None

    class NoPackageClient(_Client):
        def execute_training(self, *args: Any, **kwargs: Any) -> NoPackage:
            _ = (args, kwargs)
            return NoPackage()

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        lambda *_args, **_kwargs: __import__("nirs4all.pipeline.dagml.estimator", fromlist=["DagMLTrainingExecution"]).DagMLTrainingExecution(
            request={},
            data_envelopes={},
            relations={},
            training_influence={},
            op_callback=lambda task: task,
            outcome_id="o",
            run_id="r",
            bundle_id="b",
        ),
    )
    with pytest.raises(DagMLNativeCoverageError, match="exportable Package V2"):
        fit_native_pipeline(
            [{"split": "stub"}, {"model": "stub"}],
            np.asarray([[1.0]]),
            np.asarray([1.0]),
            sample_ids=["fit-a"],
            native_client=NoPackageClient(),
            methods_library_path=native_library,
        )


def test_fit_native_pipeline_refuses_host_sidecar_package_before_returning(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    class HostSidecar(_TrainingResult):
        def export_portable_predictor_package(self, package_id: str) -> dict[str, Any]:
            return {"schema_version": 2, "package_id": package_id, "execution_bundle": {}}

    class HostSidecarClient(_Client):
        def execute_training(self, *args: Any, **kwargs: Any) -> HostSidecar:
            _ = (args, kwargs)
            return HostSidecar()

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        lambda *_args, **_kwargs: __import__("nirs4all.pipeline.dagml.estimator", fromlist=["DagMLTrainingExecution"]).DagMLTrainingExecution(
            request={},
            data_envelopes={},
            relations={},
            training_influence={},
            op_callback=lambda task: task,
            outcome_id="o",
            run_id="r",
            bundle_id="b",
        ),
    )
    with pytest.raises(DagMLNativeCoverageError, match="replayable portable Methods"):
        fit_native_pipeline(
            [{"split": "stub"}, {"model": "stub"}],
            np.asarray([[1.0]]),
            np.asarray([1.0]),
            sample_ids=["fit-a"],
            native_client=HostSidecarClient(),
            methods_library_path=native_library,
        )


def test_fit_native_pipeline_predicts_only_through_identified_native_replay(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    def compile_fit(self, estimator, X, y, **kwargs):  # noqa: ANN001
        _ = (self, estimator, X, y, kwargs)
        from nirs4all.pipeline.dagml.estimator import DagMLTrainingExecution

        return DagMLTrainingExecution(
            request={},
            data_envelopes={},
            relations={},
            training_influence={},
            op_callback=lambda task: task,
            outcome_id="o",
            run_id="r",
            bundle_id="b",
        )

    def compile_replay(self, estimator, X, *, mode, identity_frame):  # noqa: ANN001
        _ = (self, estimator, X, mode)
        from nirs4all.pipeline.dagml.estimator import DagMLReplayExecution

        assert identity_frame.sample_ids == ("predict-a",)
        return DagMLReplayExecution(
            request={"phase": "PREDICT"},
            data_envelopes={},
            artifact_handles={},
            op_callback=lambda task: task,
            outcome_id="predict:o",
            run_id="predict:r",
        )

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        compile_fit,
    )
    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayMethodsReplayCompiler.compile_replay",
        compile_replay,
    )
    estimator = fit_native_pipeline(
        [{"split": "stub"}, {"model": "stub"}],
        np.asarray([[1.0]]),
        np.asarray([1.0]),
        sample_ids=["fit-a"],
        native_client=_Client(),
        methods_library_path=native_library,
    )

    prediction = estimator.predict_with_identity(
        np.asarray([[2.0]]),
        sample_ids=["predict-a"],
    )

    assert prediction.tolist() == [[3.5]]
