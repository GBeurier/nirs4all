"""Public native raw-array training entry point tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_training import fit_native_pipeline, run_native_methods
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


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
        return {"schema_version": 2, "package_id": package_id}


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


def test_fit_native_pipeline_is_a_public_strict_native_composition(
    monkeypatch: pytest.MonkeyPatch,
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
    )

    assert isinstance(estimator, DagMLPipelineEstimator)
    assert captured["kwargs"]["sample_ids"] == ("fit-a", "fit-b")
    assert client.calls[0][0][:4] == (
        {"request": "native"},
        {"model.x": {}},
        {"records": []},
        {"entries": []},
    )
    assert estimator.predictor_package_ == {
        "schema_version": 2,
        "package_id": "outcome:native-predictor",
    }
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
    monkeypatch: pytest.MonkeyPatch,
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
        )


def test_fit_native_pipeline_predicts_only_through_identified_native_replay(
    monkeypatch: pytest.MonkeyPatch,
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
    )

    prediction = estimator.predict_with_identity(
        np.asarray([[2.0]]),
        sample_ids=["predict-a"],
    )

    assert prediction.tolist() == [[3.5]]


def test_fit_native_pipeline_exports_the_captured_package_without_legacy_refit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
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

    captured: dict[str, Any] = {}

    def write_archive(path, *, archive_id, outcome, package):  # noqa: ANN001
        captured.update(path=path, archive_id=archive_id, outcome=outcome, package=package)
        return {"archive_id": archive_id, "archive_sha256": "f" * 64}

    monkeypatch.setattr(
        "nirs4all.api.native_training.RawArrayDagMLTrainingCompiler.compile_fit",
        compile_fit,
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.write_methods_archive_v2",
        write_archive,
    )
    estimator = fit_native_pipeline(
        [{"split": "stub"}, {"model": "stub"}],
        np.asarray([[1.0]]),
        np.asarray([1.0]),
        sample_ids=["fit-a"],
        native_client=_Client(),
    )

    reference = estimator.export_native_archive(tmp_path / "portable.n4a", archive_id="archive:native")

    assert reference == {"archive_id": "archive:native", "archive_sha256": "f" * 64}
    assert captured["archive_id"] == "archive:native"
    assert captured["outcome"] == {"native": True}
    assert captured["package"] == {"schema_version": 2, "package_id": "o-predictor"}


def test_run_native_methods_refuses_legacy_workspace_options_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nirs4all.api.native_training.fit_native_pipeline",
        lambda *_args, **_kwargs: pytest.fail("unsupported request must not train"),
    )
    with pytest.raises(NotImplementedError, match="progress verbosity"):
        run_native_methods(
            [],
            {"X": [[1.0]], "y": [1.0], "sample_ids": ["fit-a"]},
            verbose=0,
        )
    with pytest.raises(NotImplementedError, match="t.*legacy charts"):
        run_native_methods(
            [],
            {"X": [[1.0]], "y": [1.0], "sample_ids": ["fit-a"]},
            save_charts=True,
        )
