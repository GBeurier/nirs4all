"""Public native raw-array training entry point tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_retrain_lineage import NativeRetrainLineage
from nirs4all.api.native_training import fit_native_pipeline
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.raw_replay_lowerer import RawArrayMethodsReplayError


@pytest.fixture
def native_library(tmp_path) -> str:
    """A path-shaped runtime is sufficient for client-boundary unit tests."""

    library = tmp_path / "libn4m.so"
    library.write_bytes(b"native")
    return str(library)


class _TrainingResult:
    def __init__(self) -> None:
        self.is_attached = True
        self.detach_calls = 0

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

    def detach(self) -> bool:
        self.detach_calls += 1
        if not self.is_attached:
            return False
        self.is_attached = False
        return True


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


def test_fit_native_pipeline_uses_the_callback_free_methods_client_when_contracts_require_it(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    """The strict lane must not silently select the generic callback client."""

    from nirs4all.pipeline.dagml.estimator import DagMLTrainingExecution

    class StrictTrainingResult(_TrainingResult):
        def export_portable_predictor_package(self, package_id: str, **_kwargs: Any) -> dict[str, Any]:
            return super().export_portable_predictor_package(package_id)

    class StrictClient:
        def __init__(self) -> None:
            self.methods_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def execute_training(self, *_args: Any, **_kwargs: Any) -> StrictTrainingResult:
            raise AssertionError("strict Methods fit selected generic execute_training")

        def execute_methods_training(self, *args: Any, **kwargs: Any) -> StrictTrainingResult:
            self.methods_calls.append((args, kwargs))
            return StrictTrainingResult()

    execution = DagMLTrainingExecution(
        request={"request": "methods"},
        data_envelopes={"data:train": {}},
        relations={"records": []},
        training_influence={"entries": []},
        op_callback=None,
        outcome_id="outcome:methods",
        run_id="run:methods",
        bundle_id="bundle:methods",
        methods_inputs={"data:train": {"sample_ids": ["fit-a", "fit-b"]}},
        methods_library_path=native_library,
    )
    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        lambda *_args, **_kwargs: execution,
    )
    client = StrictClient()

    fit_native_pipeline(
        [{"split": "stub"}, {"model": "stub"}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        sample_ids=["fit-a", "fit-b"],
        native_client=client,
        methods_library_path=native_library,
    )

    assert len(client.methods_calls) == 1
    args, kwargs = client.methods_calls[0]
    assert args[:5] == (
        {"request": "methods"},
        {"data:train": {}},
        {"records": []},
        {"entries": []},
        {"data:train": {"sample_ids": ["fit-a", "fit-b"]}},
    )
    assert kwargs["methods_library_path"] == native_library


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
    assert estimator.training_compiler.additional_diagnostics == {"nirs4all_native_seed": 12345}
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


def test_estimator_fit_detaches_an_attached_result_when_post_execution_projection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No attached Dag-ML result may escape a failed post-execution projection."""

    from nirs4all.pipeline.dagml.estimator import DagMLTrainingExecution

    class Compiler:
        def compile_fit(self, _estimator, _X, _y, **_kwargs):  # noqa: ANN001
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

    class Client:
        def __init__(self) -> None:
            self.result = _TrainingResult()

        def execute_training(self, *_args: Any, **_kwargs: Any) -> _TrainingResult:
            return self.result

    client = Client()
    estimator = DagMLPipelineEstimator(
        training_compiler=Compiler(),
        native_client=client,
    )
    monkeypatch.setattr(
        estimator,
        "_select_output_binding",
        lambda _outputs: (_ for _ in ()).throw(RuntimeError("projection failed")),
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        estimator.fit(
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
        )
    assert client.result.detach_calls == 1
    assert not client.result.is_attached


def test_fit_native_pipeline_detaches_when_post_fit_package_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    native_library: str,
) -> None:
    """Package validation errors must not retain native handles after fit returns."""

    class Client(_Client):
        def __init__(self) -> None:
            super().__init__()
            self.result = _TrainingResult()

        def execute_training(self, *args: Any, **kwargs: Any) -> _TrainingResult:
            self.calls.append((args, kwargs))
            return self.result

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
    monkeypatch.setattr(
        "nirs4all.api.native_training.validate_native_methods_package",
        lambda _package: (_ for _ in ()).throw(RawArrayMethodsReplayError("invalid package")),
    )
    client = Client()

    with pytest.raises(DagMLNativeCoverageError, match="replayable portable Methods"):
        fit_native_pipeline(
            [{"split": "stub"}, {"model": "stub"}],
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
            sample_ids=["fit-a", "fit-b"],
            native_client=client,
            methods_library_path=native_library,
        )
    assert client.result.detach_calls == 1
    assert not client.result.is_attached


def test_fit_native_pipeline_seals_internal_retrain_lineage_into_compiler_diagnostics(
    monkeypatch: pytest.MonkeyPatch, native_library: str
) -> None:
    """The signed training compiler receives parent provenance before fit."""

    def compile_fit(self, _estimator, _X, _y, **_kwargs):  # noqa: ANN001
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

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        compile_fit,
    )
    lineage = NativeRetrainLineage(
        source_outcome_fingerprint="a" * 64,
        source_training_request_fingerprint="b" * 64,
        source_effective_plan_fingerprint="c" * 64,
        source_selected_variant_id="variant:base",
        source_selected_variant_fingerprint="d" * 64,
        source_seed=17,
    )
    estimator = fit_native_pipeline(
        [{"split": "stub"}, {"model": "stub"}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        sample_ids=["fit-a", "fit-b"],
        native_client=_Client(),
        methods_library_path=native_library,
        seed=31,
        retrain_lineage=lineage,
    )

    assert estimator.training_compiler.additional_diagnostics == {
        "nirs4all_native_seed": 31,
        "nirs4all_native_retrain_lineage": lineage.to_dict(),
    }


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
