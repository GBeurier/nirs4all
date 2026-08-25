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
