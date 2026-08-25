"""Native Methods session lifecycle tests."""

from __future__ import annotations

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_session import NativeMethodsSession
from nirs4all.api.session import session


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


def test_native_run_delegates_to_the_matching_native_session(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = [{"split": "stub"}, {"model": "stub"}]
    native = NativeMethodsSession(pipeline, random_state=7)
    result = _Result()
    observed: list[object] = []

    def native_run(dataset):  # noqa: ANN001
        observed.append(dataset)
        return result

    monkeypatch.setattr(native, "run", native_run)

    assert nirs4all.run(
        pipeline,
        {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]},
        engine="native",
        session=native,
        save_charts=False,
        random_state=7,
    ) is result
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
