"""Unit tests for the internal DAG-ML sklearn-compatible estimator seam."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from nirs4all.pipeline.dagml.estimator import (
    DagMLPipelineEstimator,
    DagMLReplayExecution,
    DagMLTrainingExecution,
)
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


class _FakeTrainingResult:
    def __init__(self) -> None:
        self.outcome = {"outcome": True}
        self.outputs = [{"output_id": "pred", "node_id": "model", "port_name": "prediction"}]
        self.exported_package_ids: list[str] = []

    def export_portable_predictor_package(self, package_id: str) -> dict[str, str]:
        self.exported_package_ids.append(package_id)
        return {"package_id": package_id}


class _FakeNativeClient:
    def __init__(self) -> None:
        self.training_calls: list[dict[str, Any]] = []
        self.replay_calls: list[dict[str, Any]] = []
        self.training_result = _FakeTrainingResult()

    def execute_training(self, *args: Any, **kwargs: Any) -> _FakeTrainingResult:
        self.training_calls.append({"args": args, "kwargs": kwargs})
        return self.training_result

    def replay_loaded_predictor_package(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.replay_calls.append({"args": args, "kwargs": kwargs})
        return {"rows": [1.0, 2.0], "proba": [[0.25, 0.75], [0.8, 0.2]]}


def _training_execution() -> DagMLTrainingExecution:
    return DagMLTrainingExecution(
        request={"request": True},
        data_envelopes={"fit.data": {"envelope": True}},
        relations={"relations": True},
        training_influence={"entries": []},
        op_callback=lambda task: {"task": task},
        outcome_id="outcome-1",
        run_id="run-1",
        bundle_id="bundle-1",
        warnings=["warn"],
        diagnostics={"diag": True},
    )


def _replay_execution(mode: str) -> DagMLReplayExecution:
    return DagMLReplayExecution(
        request={"phase": mode},
        data_envelopes={"predict.data": {"envelope": True}},
        artifact_handles={"artifact": {"handle": 1}},
        op_callback=lambda task: {"task": task},
        outcome_id=f"{mode}-outcome",
        run_id=f"{mode}-run",
        warnings=[],
        diagnostics={"mode": mode},
    )


def test_constructor_is_sklearn_cloneable_and_does_not_fit() -> None:
    estimator = DagMLPipelineEstimator(
        pipeline=("split", "model"),
        task_type="regression",
        selection_output_id="pred",
        dagml_module="dag_ml_test",
    )

    cloned = clone(estimator)

    assert cloned is not estimator
    assert cloned.get_params()["pipeline"] == ("split", "model")
    assert cloned.get_params()["selection_output_id"] == "pred"
    assert not hasattr(estimator, "training_result_")
    assert not hasattr(cloned, "training_result_")


def test_set_params_updates_constructor_parameters_only() -> None:
    estimator = DagMLPipelineEstimator(pipeline=("old",), task_type="auto")

    returned = estimator.set_params(pipeline=("new",), task_type="classification")

    assert returned is estimator
    assert estimator.pipeline == ("new",)
    assert estimator.task_type == "classification"
    assert not hasattr(estimator, "training_result_")


def test_fit_without_compiler_is_typed_coverage_error() -> None:
    estimator = DagMLPipelineEstimator(pipeline=("model",))

    with pytest.raises(DagMLNativeCoverageError, match="training contract compiler"):
        estimator.fit(np.ones((2, 3)), np.ones(2))

    assert not hasattr(estimator, "training_result_")


def test_predict_before_fit_raises_sklearn_not_fitted() -> None:
    estimator = DagMLPipelineEstimator(pipeline=("model",))

    with pytest.raises(NotFittedError):
        estimator.predict(np.ones((2, 3)))


def test_fit_forwards_compiled_training_contracts_and_sets_fitted_attrs() -> None:
    client = _FakeNativeClient()
    compiler_calls: list[dict[str, Any]] = []

    def compiler(
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: Any = None,
    ) -> DagMLTrainingExecution:
        compiler_calls.append(
            {
                "estimator": estimator,
                "X": X,
                "y": y,
                "sample_ids": sample_ids,
                "groups": groups,
                "metadata": metadata,
                "identity_frame": identity_frame,
            }
        )
        return _training_execution()

    X = np.ones((3, 4))
    y = np.arange(3)
    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        selection_output_id="pred",
        native_client=client,
        training_compiler=compiler,
    )

    returned = estimator.fit(
        X,
        y,
        sample_ids=["s1", "s2", "s3"],
        groups=[0, 0, 1],
        metadata={"instrument": ["demo", "demo", "demo"]},
    )

    assert returned is estimator
    assert compiler_calls[0]["estimator"] is estimator
    assert compiler_calls[0]["sample_ids"] == ("s1", "s2", "s3")
    assert compiler_calls[0]["groups"] == ("0", "0", "1")
    assert compiler_calls[0]["metadata"] == {
        "s1": {"instrument": "demo"},
        "s2": {"instrument": "demo"},
        "s3": {"instrument": "demo"},
    }
    assert compiler_calls[0]["identity_frame"].sample_ids == ("s1", "s2", "s3")
    assert client.training_calls[0]["args"][:4] == (
        {"request": True},
        {"fit.data": {"envelope": True}},
        {"relations": True},
        {"entries": []},
    )
    assert client.training_calls[0]["kwargs"] == {
        "outcome_id": "outcome-1",
        "run_id": "run-1",
        "bundle_id": "bundle-1",
        "warnings": ["warn"],
        "diagnostics": {"diag": True},
    }
    assert estimator.training_result_ is client.training_result
    assert estimator.training_outcome_ == {"outcome": True}
    assert estimator.output_binding_ == {"output_id": "pred", "node_id": "model", "port_name": "prediction"}
    assert estimator.predictor_package_ == {"package_id": "outcome-1-predictor"}
    assert estimator.fit_identity_frame_ is compiler_calls[0]["identity_frame"]
    assert estimator.n_features_in_ == 4


def test_fit_can_require_explicit_sample_ids() -> None:
    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        require_explicit_sample_ids=True,
        training_compiler=lambda *args, **kwargs: _training_execution(),
    )

    with pytest.raises(ValueError, match="requires explicit sample_ids"):
        estimator.fit(np.ones((2, 2)), np.ones(2))

    assert not hasattr(estimator, "fit_identity_frame_")


def test_fit_refuses_ambiguous_outputs_without_selection_output_id() -> None:
    class MultiOutputResult(_FakeTrainingResult):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = [
                {"output_id": "pred"},
                {"output_id": "aux"},
            ]

    client = _FakeNativeClient()
    client.training_result = MultiOutputResult()
    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        native_client=client,
        training_compiler=lambda *args, **kwargs: _training_execution(),
    )

    with pytest.raises(DagMLNativeCoverageError, match="ambiguous outputs"):
        estimator.fit(np.ones((2, 2)), np.ones(2))


def test_predict_requires_replay_compiler_after_fit() -> None:
    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        selection_output_id="pred",
        native_client=_FakeNativeClient(),
        training_compiler=lambda *args, **kwargs: _training_execution(),
    ).fit(np.ones((2, 2)), np.ones(2))

    with pytest.raises(DagMLNativeCoverageError, match="loaded-package replay compiler"):
        estimator.predict(np.ones((2, 2)))


def test_predict_and_predict_proba_use_native_replay_and_explicit_decoders() -> None:
    client = _FakeNativeClient()
    replay_modes: list[str] = []

    def replay_compiler(estimator: DagMLPipelineEstimator, X: Any, *, mode: str) -> DagMLReplayExecution:
        replay_modes.append(mode)
        assert estimator.predictor_package_ == {"package_id": "outcome-1-predictor"}
        assert np.asarray(X).shape == (2, 3)
        return _replay_execution(mode)

    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        selection_output_id="pred",
        native_client=client,
        training_compiler=lambda *args, **kwargs: _training_execution(),
        prediction_compiler=replay_compiler,
        prediction_decoder=lambda outcome: outcome["rows"],
        probability_decoder=lambda outcome: outcome["proba"],
    ).fit(np.ones((3, 3)), np.ones(3))

    prediction = estimator.predict(np.ones((2, 3)))
    probability = estimator.predict_proba(np.ones((2, 3)))

    assert prediction.tolist() == [1.0, 2.0]
    assert probability.tolist() == [[0.25, 0.75], [0.8, 0.2]]
    assert replay_modes == ["predict", "predict_proba"]
    assert client.replay_calls[0]["args"][:4] == (
        {"package_id": "outcome-1-predictor"},
        {"phase": "predict"},
        {"predict.data": {"envelope": True}},
        {"artifact": {"handle": 1}},
    )
    assert client.replay_calls[1]["args"][1] == {"phase": "predict_proba"}


def test_predict_proba_never_fabricates_pseudo_probabilities() -> None:
    estimator = DagMLPipelineEstimator(
        pipeline=("model",),
        selection_output_id="pred",
        native_client=_FakeNativeClient(),
        training_compiler=lambda *args, **kwargs: _training_execution(),
        prediction_compiler=lambda *args, **kwargs: _replay_execution("predict_proba"),
        prediction_decoder=lambda outcome: outcome["rows"],
    ).fit(np.ones((2, 2)), np.ones(2))

    with pytest.raises(DagMLNativeCoverageError, match="pseudo-probabilities are forbidden"):
        estimator.predict_proba(np.ones((2, 2)))
