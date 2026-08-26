"""Native Methods session lifecycle tests."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.native_session import NativeMethodsSession
from nirs4all.api.session import Session, session


class _Result:
    def __init__(self) -> None:
        self.native_estimator = self
        self.exported: list[object] = []

    def predict_with_identity(self, X, *, sample_ids, groups=None, metadata=None):  # noqa: ANN001
        assert sample_ids == ["p1", "p2"]
        assert groups is None
        assert metadata is None
        return np.asarray(X) + 1.0

    def export(self, path):  # noqa: ANN001
        self.exported.append(path)
        return path


def test_native_session_runs_predicts_saves_and_closes_without_legacy_runner(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    result = _Result()
    observed: dict[str, object] = {}

    def native_run(pipeline, dataset, **kwargs):  # noqa: ANN001
        observed.update(pipeline=pipeline, dataset=dataset, kwargs=kwargs)
        return result

    monkeypatch.setattr("nirs4all.api.native_session.run_native_methods", native_run)

    with session([{"split": "stub"}, {"model": "stub"}], engine="native", name="native", random_state=7) as native:
        assert isinstance(native, NativeMethodsSession)
        assert nirs4all.NativeMethodsSession is NativeMethodsSession
        trained = native.run({"X": [[1.0], [2.0]], "y": [1.0, 2.0], "sample_ids": ["s1", "s2"]})
        prediction = native.predict(np.asarray([[3.0], [4.0]]), sample_ids=["p1", "p2"])
        saved = native.save(tmp_path / "model.n4a")

        assert trained is result
        assert prediction.y_pred.tolist() == [[4.0], [5.0]]
        assert saved == tmp_path / "model.n4a"

    assert native.closed
    assert observed["kwargs"] == {
        "name": "native",
        "save_charts": False,
        "random_state": 7,
        "tuning": None,
    }
    with pytest.raises(RuntimeError, match="closed"):
        native.run({"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]})


def test_native_session_refuses_missing_pipeline_and_legacy_runner_kwargs() -> None:
    with pytest.raises(ValueError, match="explicit portable pipeline"):
        with session(engine="native"):
            pass
    with pytest.raises(TypeError, match="unexpected keyword"):
        with session([], engine="native", workspace_path="legacy"):
            pass
    with pytest.raises(TypeError, match="tuning must be a mapping"):
        NativeMethodsSession([], tuning=["methods-hpo"])  # type: ignore[arg-type]


def test_public_session_constructor_selects_native_or_refuses_before_legacy_runner() -> None:
    pipeline = [{"split": "stub"}, {"model": "stub"}]

    native = Session(pipeline, engine="native", name="native", random_state=7)
    assert isinstance(native, NativeMethodsSession)
    assert native.pipeline is pipeline
    assert native.random_state == 7

    legacy = Session(pipeline, engine="legacy")
    assert isinstance(legacy, Session)
    assert "engine" not in legacy._runner_kwargs  # noqa: SLF001

    with pytest.raises(NotImplementedError, match="does not have an execution path"):
        Session(pipeline, engine="dag-ml")
    with pytest.raises(ValueError, match="explicit portable pipeline"):
        Session(engine="native")


def test_native_session_binds_the_strict_methods_hpo_operation_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stateful native training forwards only the explicit HPO request."""

    result = _Result()
    observed: dict[str, object] = {}
    tuning = {"engine": "methods-hpo", "trials": 3}

    def native_run(_pipeline, _dataset, **kwargs):  # noqa: ANN001
        observed.update(kwargs)
        return result

    monkeypatch.setattr("nirs4all.api.native_session.run_native_methods", native_run)
    native = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}], tuning=tuning)

    assert native.tuning == tuning
    tuning["trials"] = 99
    assert native.tuning == {"engine": "methods-hpo", "trials": 3}
    assert native.run({"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]}) is result
    assert observed["tuning"] == {"engine": "methods-hpo", "trials": 3}


def test_native_run_delegates_to_the_matching_native_session(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = [{"split": "stub"}, {"model": "stub"}]
    native = NativeMethodsSession(pipeline, random_state=7)
    result = _Result()
    observed: list[object] = []

    def native_run(dataset):  # noqa: ANN001
        observed.append(dataset)
        return result

    monkeypatch.setattr(native, "run", native_run)

    assert (
        nirs4all.run(
            pipeline,
            {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]},
            engine="native",
            session=native,
            save_charts=False,
            random_state=7,
        )
        is result
    )
    assert observed == [{"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]}]

    with pytest.raises(ValueError, match="exact pipeline"):
        nirs4all.run(
            list(pipeline),
            {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]},
            engine="native",
            session=native,
            save_charts=False,
        )


def test_native_predict_delegates_to_the_trained_native_session(monkeypatch: pytest.MonkeyPatch) -> None:
    native = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}])
    result = _Result()
    native._result = result  # noqa: SLF001

    prediction = nirs4all.predict(
        data={"X": np.asarray([[3.0], [4.0]]), "sample_ids": ["p1", "p2"]},
        session=native,
        engine="native",
    )

    assert prediction.y_pred.tolist() == [[4.0], [5.0]]
    assert prediction.metadata == {"engine": "native", "sample_ids": ["p1", "p2"]}


def test_native_session_retrains_only_through_the_strict_native_full_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = object.__new__(NativeMethodsRunResult)
    retrained = object.__new__(NativeMethodsRunResult)
    native = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}], name="native")
    native._result = source  # noqa: SLF001
    observed: dict[str, object] = {}

    def native_retrain(source_arg, data, **kwargs):  # noqa: ANN001
        observed.update(source=source_arg, data=data, kwargs=kwargs)
        return retrained

    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(retrain_module, "retrain", native_retrain)
    dataset = {"X": [[3.0]], "y": [3.0], "sample_ids": ["s3"]}

    assert native.retrain(dataset) is retrained
    assert native.result is retrained
    assert observed == {
        "source": source,
        "data": dataset,
        "kwargs": {
            "mode": "full",
            "name": "native",
            "save_artifacts": True,
            "verbose": 0,
            "engine": "native",
        },
    }


def test_native_hpo_session_refits_the_result_selected_by_its_own_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session keeps its attested HPO result as the sole native refit source."""

    selected = object.__new__(NativeMethodsRunResult)
    refitted = object.__new__(NativeMethodsRunResult)
    tuning = {"engine": "methods-hpo", "trials": 2}
    native = NativeMethodsSession([{"split": "stub"}, {"model": "stub"}], tuning=tuning)
    observed: dict[str, object] = {}

    def native_run(_pipeline, _dataset, **kwargs):  # noqa: ANN001
        observed["tuning"] = kwargs["tuning"]
        return selected

    def native_retrain(source_arg, _data, **kwargs):  # noqa: ANN001
        observed["source"] = source_arg
        observed["retrain_kwargs"] = kwargs
        return refitted

    monkeypatch.setattr("nirs4all.api.native_session.run_native_methods", native_run)
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(retrain_module, "retrain", native_retrain)

    dataset = {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]}
    assert native.run(dataset) is selected
    assert native.retrain(dataset) is refitted
    assert native.result is refitted
    assert observed["tuning"] == tuning
    assert observed["source"] is selected
    assert observed["retrain_kwargs"] == {
        "mode": "full",
        "name": "",
        "save_artifacts": True,
        "verbose": 0,
        "engine": "native",
    }


@pytest.mark.parametrize("engine", ["legacy", "dag-ml", "dual"])
def test_native_session_never_falls_back_to_another_run_engine(engine: str) -> None:
    pipeline = [{"split": "stub"}, {"model": "stub"}]
    native = NativeMethodsSession(pipeline)

    with pytest.raises(ValueError, match="requires engine='native'"):
        nirs4all.run(
            pipeline,
            {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]},
            engine=engine,
            session=native,
        )
