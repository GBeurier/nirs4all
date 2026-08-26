"""Transition-release backend selector coverage for public helper APIs."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from nirs4all.api.explain import explain
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.predict import predict
from nirs4all.api.result import PredictResult
from nirs4all.api.retrain import retrain
from nirs4all.api.run import run
from nirs4all.pipeline.engine import require_legacy_engine


def test_require_legacy_engine_accepts_legacy() -> None:
    assert require_legacy_engine("predict", "legacy") == "legacy"


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        (
            "predict",
            lambda: predict(model={"model_name": "dummy"}, data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "predict",
            lambda: predict(chain_id="chain-1", data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "explain",
            lambda: explain({"model_name": "dummy"}, np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "retrain",
            lambda: retrain({"model_name": "dummy"}, (np.zeros((2, 3)), np.zeros(2)), engine="dag-ml"),
        ),
    ],
)
def test_public_helpers_reject_dagml_until_native_paths_exist(operation: str, call) -> None:
    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation}.*dag-ml"):
        call()


def test_retrain_native_refits_the_attested_selected_methods_variant_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native retrain delegates to the V3 full-refit operation, never CV."""

    source = object.__new__(NativeMethodsRunResult)
    observed: dict[str, object] = {}
    expected = object()

    def native_refit(source_arg, dataset, **kwargs):  # noqa: ANN001
        observed.update(source=source_arg, dataset=dataset, kwargs=kwargs)
        return expected

    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(retrain_module, "refit_native_methods", native_refit)
    monkeypatch.setattr(
        retrain_module,
        "PipelineRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native retrain constructed PipelineRunner")),
    )

    dataset = {
        "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
        "y": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "sample_ids": ["next-0", "next-1", "next-2", "next-3"],
    }
    assert retrain(source, dataset, name="next") is expected
    assert observed["source"] is source
    assert observed["dataset"] is dataset
    assert observed["kwargs"] == {"name": "next"}


@pytest.mark.parametrize("engine", ["legacy", "dag-ml", "dual"])
def test_retrain_native_source_refuses_explicit_non_native_engine(engine: str) -> None:
    source = object.__new__(NativeMethodsRunResult)
    with pytest.raises(ValueError, match="explicit non-native engine"):
        retrain(source, {"X": [], "y": [], "sample_ids": []}, engine=engine)


@pytest.mark.parametrize(
    ("mode", "message"),
    [("transfer", "only mode='full'"), ("finetune", "only mode='full'")],
)
def test_retrain_native_refuses_unqualified_modes_before_execution(mode: str, message: str) -> None:
    source = object.__new__(NativeMethodsRunResult)
    with pytest.raises(NotImplementedError, match=message):
        retrain(source, {"X": [], "y": [], "sample_ids": []}, engine="native", mode=mode)

def test_predict_native_archive_is_explicit_and_never_constructs_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def native_predict(path, X, *, sample_ids, groups, metadata):  # noqa: ANN001
        observed.update(
            path=str(path), X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata
        )
        return np.asarray([[2.0], [3.0]])

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw",
        native_predict,
    )

    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        engine="native",
    )

    assert result.y_pred.tolist() == [[2.0], [3.0]]
    assert result.metadata["engine"] == "native"
    assert observed["sample_ids"] == ["p1", "p2"]


def test_run_native_dispatches_to_the_strict_methods_lane_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    sentinel = object()

    def native_run(pipeline, dataset, **kwargs):  # noqa: ANN001
        observed.update(pipeline=pipeline, dataset=dataset, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr("nirs4all.api.native_training.run_native_methods", native_run)
    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(
        run_module,
        "PipelineRunner",
        lambda *_args, **_kwargs: pytest.fail("engine='native' must not construct PipelineRunner"),
    )

    result = run(
        [{"split": "stub"}, {"model": "stub"}],
        {"X": np.asarray([[1.0]]), "y": np.asarray([2.0]), "sample_ids": ["fit-a"]},
        engine="native",
        save_charts=False,
    )

    assert result is sentinel
    dataset = observed["dataset"]
    assert isinstance(dataset, dict)
    assert np.array_equal(dataset["X"], np.asarray([[1.0]]))
    assert np.array_equal(dataset["y"], np.asarray([2.0]))
    assert dataset["sample_ids"] == ["fit-a"]
    assert observed["kwargs"] == {
        "name": "",
        "verbose": 1,
        "save_artifacts": True,
        "save_charts": False,
        "plots_visible": False,
        "random_state": None,
        "refit": True,
        "cache": None,
        "project": None,
        "report_naming": "nirs",
        "results_path": None,
        "session": None,
        "runner_kwargs": {},
    }


@pytest.mark.parametrize("keyword", ["tuning", "calibration"])
def test_run_native_refuses_tuning_and_calibration_before_native_execution(
    monkeypatch: pytest.MonkeyPatch, keyword: str
) -> None:
    monkeypatch.setattr(
        "nirs4all.api.native_training.run_native_methods",
        lambda *_args, **_kwargs: pytest.fail("native training must not start"),
    )
    kwargs = {keyword: {"placeholder": True}}
    with pytest.raises(NotImplementedError, match="strict raw Methods training subset"):
        run([], {"X": [[1.0]], "y": [1.0], "sample_ids": ["fit-a"]}, engine="native", **kwargs)


@pytest.mark.parametrize(
    ("model", "data", "kwargs", "message"),
    [
        ("portable.n4a", np.zeros((1, 2)), {}, "data={'X': matrix, 'sample_ids': explicit_ids}"),
        ("portable.joblib", {"X": [[1.0]], "sample_ids": ["p1"]}, {}, "portable .n4a"),
    ],
)
def test_predict_native_archive_fails_closed_before_execution(model, data, kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        predict(model=model, data=data, engine="native", **kwargs)


def test_predict_native_dispatches_a_v3_refit_archive_without_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def reject_v2(*_args, **_kwargs):  # noqa: ANN001
        raise ValueError("Archive V2 reader rejects V3")

    class _V3Session:
        archive_schema_version = 3

        def predict(self, X, *, sample_ids, groups, metadata):  # noqa: ANN001
            observed.update(X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata)
            return PredictResult(
                y_pred=np.asarray([[4.0], [5.0]]),
                metadata={"engine": "native"},
                model_name="MethodsN4MM",
                preprocessing_steps=[],
            )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw", reject_v2
    )
    monkeypatch.setattr(
        "nirs4all.api.native_archive_session.load_native_archive_session",
        lambda path: _V3Session(),
    )

    result = predict(
        model="refit.n4a",
        data={"X": [[1.0], [2.0]], "sample_ids": ["v3-a", "v3-b"]},
        engine="native",
    )

    assert observed["sample_ids"] == ["v3-a", "v3-b"]
    assert result.y_pred.tolist() == [[4.0], [5.0]]


def test_predict_native_archive_projects_only_materialized_conformal_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presentation = {
        "schema_version": 1,
        "package_fingerprint": "a" * 64,
        "replay_outcome_fingerprint": "b" * 64,
        "binding_id": "binding:pls",
        "target_name": "moisture",
        "sample_ids": ["p1", "p2"],
        "point_predictions": [2.0, 3.0],
        "intervals": [
            {"coverage": 0.8, "lower": [1.0, 2.0], "upper": [3.0, 4.0], "qhat": 1.0},
            {"coverage": 0.9, "lower": [0.5, 1.5], "upper": [3.5, 4.5], "qhat": 1.5},
        ],
        "calibration_fingerprint": "c" * 64,
        "presentation_fingerprint": "d" * 64,
    }
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.project_methods_archive_v2_conformal_presentation",
        lambda *_args, **_kwargs: presentation,
    )

    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        engine="native",
        coverage=0.9,
    )

    assert result.y_pred.tolist() == [[2.0], [3.0]]
    assert sorted(result.intervals) == [0.9]
    assert result.metadata["conformal_presentation"] is presentation
    assert result.metadata["selected_interval_coverages"] == [0.9]


@pytest.mark.parametrize(
    "operation",
    [
        "predict",
        "explain",
        "retrain",
    ],
)
def test_public_helpers_refuse_dagml_env_without_legacy_fallback(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    monkeypatch.setenv("N4A_ENGINE", "dag-ml")

    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation} does not have a dag-ml execution path"):
        require_legacy_engine(operation)
