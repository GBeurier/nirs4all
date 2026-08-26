"""State and boundary coverage for :class:`NativeMethodsSession`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_methods_session import NativeMethodsSession


class _Estimator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_with_identity(self, X: Any, **kwargs: Any) -> np.ndarray:
        self.calls.append({"X": X, **kwargs})
        return np.asarray([[3.0], [4.0]])


class _Result:
    def __init__(self) -> None:
        self.native_estimator = _Estimator()
        self.exports: list[Path] = []

    def export(self, path: str | Path) -> Path:
        target = Path(path)
        self.exports.append(target)
        return target


def test_native_methods_session_trains_predicts_saves_and_releases_without_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, Any] = {}
    result = _Result()

    def native_run(pipeline, dataset, **kwargs):  # noqa: ANN001
        observed.update(pipeline=pipeline, dataset=dataset, kwargs=kwargs)
        return result

    monkeypatch.setattr("nirs4all.api.native_methods_session.run_native_methods", native_run)
    session = NativeMethodsSession(["split", "model"], name="demo", random_state=17)

    trained = session.run({"X": [[1.0]], "y": [2.0], "sample_ids": ["fit-a"]})
    prediction = session.predict([[3.0], [4.0]], sample_ids=["predict-a", "predict-b"])
    output = session.save(tmp_path / "native.n4a")

    assert trained is result
    assert observed["kwargs"] == {"name": "demo", "save_charts": False, "random_state": 17}
    assert prediction.y_pred.tolist() == [[3.0], [4.0]]
    assert result.native_estimator.calls[0]["sample_ids"] == ["predict-a", "predict-b"]
    assert output == tmp_path / "native.n4a"
    assert result.exports == [output]
    assert nirs4all.NativeMethodsSession is NativeMethodsSession

    session.close()
    assert session.closed and not session.is_trained
    with pytest.raises(RuntimeError, match="closed"):
        session.predict([[1.0]], sample_ids=["after-close"])


def test_native_methods_session_refuses_invalid_state() -> None:
    with pytest.raises(TypeError, match="list pipeline"):
        NativeMethodsSession({"model": "not-a-list"})  # type: ignore[arg-type]
    session = NativeMethodsSession([])
    with pytest.raises(ValueError, match="must be trained"):
        _ = session.result
