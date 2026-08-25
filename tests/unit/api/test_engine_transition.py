"""Transition-release backend selector coverage for public helper APIs."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import nirs4all
from nirs4all.api.explain import explain
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.predict import predict
from nirs4all.api.retrain import retrain
from nirs4all.api.run import run
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveConformalInterval,
    NativeArchivePrediction,
)
from nirs4all.pipeline.engine import require_legacy_engine


def test_require_legacy_engine_accepts_legacy() -> None:
    assert require_legacy_engine("predict", "legacy") == "legacy"


def test_run_native_executes_the_methods_subset_without_constructing_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public native lane never constructs ``PipelineRunner``."""

    def legacy_runner(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("run(engine='native') constructed a legacy PipelineRunner")

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "PipelineRunner", legacy_runner)

    class Estimator:
        training_outcome_ = {
            "outcome_fingerprint": "a" * 64,
            "score_set": {
                "schema_version": 2,
                "selection_metric": "rmse",
                "reports": [
                    {
                        "producer_node": "model:methods",
                        "producer_port": "oof",
                        "partition": "validation",
                        "fold_id": "fold0",
                        "level": "sample",
                        "metrics": {"rmse": 0.5},
                        "row_count": 2,
                        "target_names": ["y"],
                        "target_width": 1,
                        "variant_id": "variant:base",
                    },
                    {
                        "producer_node": "model:methods",
                        "producer_port": "oof",
                        "partition": "validation",
                        "fold_id": "avg",
                        "level": "sample",
                        "metrics": {"rmse": 0.5},
                        "row_count": 2,
                        "target_names": ["y"],
                        "target_width": 1,
                        "variant_id": "variant:base",
                    },
                ],
            },
        }

        def export_native_archive(self, *_args, **_kwargs):  # noqa: ANN001
            raise AssertionError("training must not export during run")

    observed: dict[str, object] = {}

    def native_fit(pipeline, X, y, **kwargs):  # noqa: ANN001
        observed.update(pipeline=pipeline, X=np.asarray(X), y=np.asarray(y), kwargs=kwargs)
        return Estimator()

    monkeypatch.setattr("nirs4all.api.native_training.fit_native_pipeline", native_fit)

    result = run(
        [{"split": "stub"}, {"model": "stub"}],
        {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["s1", "s2"]},
        engine="native",
        save_charts=False,
    )

    assert isinstance(result, NativeMethodsRunResult)
    assert nirs4all.NativeMethodsRunResult is NativeMethodsRunResult
    assert result.cv_best_score == pytest.approx(0.5)
    assert observed["kwargs"] == {
        "sample_ids": ["s1", "s2"],
        "groups": None,
        "metadata": None,
        "seed": 12345,
    }


def test_run_native_environment_selection_is_also_fail_closed_for_unsupported_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N4A_ENGINE", "native")

    with pytest.raises(ValueError, match="missing required keys"):
        run([], {})


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


def test_predict_native_archive_is_explicit_and_never_constructs_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def native_predict(path, X, *, sample_ids, methods_library_path, groups, metadata):  # noqa: ANN001
        observed.update(
            path=str(path), X=np.asarray(X), sample_ids=list(sample_ids), methods_library_path=methods_library_path, groups=groups, metadata=metadata
        )
        return NativeArchivePrediction(
            values=np.asarray([[2.0], [3.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
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


def test_predict_native_archive_selects_dagml_materialized_conformal_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval = NativeArchiveConformalInterval(
        coverage=0.9,
        lower=np.asarray([[1.0], [2.0]]),
        upper=np.asarray([[3.0], [4.0]]),
        qhat=1.0,
        calibration_fingerprint="a" * 64,
    )

    def native_predict(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return NativeArchivePrediction(
            values=np.asarray([[2.0], [3.0]]),
            sample_ids=("p1", "p2"),
            intervals={0.9: interval},
            conformal_guarantee_status={
                "status": "active",
                "coverage": [0.9],
                "calibration_fingerprint": "a" * 64,
            },
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        native_predict,
    )
    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        coverage=0.9,
        engine="native",
    )

    assert result.interval_coverages == (0.9,)
    assert result.interval(0.9) is interval
    assert result.conformal_guarantee_status == {
        "status": "active",
        "coverage": [0.9],
        "calibration_fingerprint": "a" * 64,
    }


@pytest.mark.parametrize(
    ("model", "data", "kwargs", "message"),
    [
        ("portable.n4a", np.zeros((1, 2)), {}, "data={'X': matrix, 'sample_ids': explicit_ids}"),
        ("portable.joblib", {"X": [[1.0]], "sample_ids": ["p1"]}, {}, "Archive V2 .n4a"),
        ("portable.n4a", {"X": [[1.0]], "sample_ids": ["p1"]}, {"coverage": 0.9}, "not materialized"),
    ],
)
def test_predict_native_archive_fails_closed_before_execution(
    monkeypatch: pytest.MonkeyPatch, model, data, kwargs, message
) -> None:
    if kwargs.get("coverage") is not None:
        monkeypatch.setattr(
            "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
            lambda *_args, **_kwargs: NativeArchivePrediction(
                values=np.asarray([[1.0]]),
                sample_ids=("p1",),
                intervals={},
                conformal_guarantee_status=None,
            ),
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        predict(model=model, data=data, engine="native", **kwargs)


@pytest.mark.parametrize(
    "operation",
    [
        "predict",
        "explain",
        "retrain",
    ],
)
def test_public_helpers_ignore_dagml_env_with_warning(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    monkeypatch.setenv("N4A_ENGINE", "dag-ml")

    with pytest.warns(RuntimeWarning, match=rf"N4A_ENGINE=dag-ml.*nirs4all\.{operation}.*legacy"):
        assert require_legacy_engine(operation) == "legacy"
