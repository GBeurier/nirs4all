"""State and boundary coverage for :class:`NativeMethodsSession`."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_methods_session import NativeMethodsSession
from nirs4all.api.native_refit_result import NativeMethodsRefitResult


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


def test_native_methods_session_retrain_uses_the_strict_native_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session child is target-bound V3, never a legacy re-execution."""

    session = NativeMethodsSession(["split", "model"], name="portable")
    source = _Result()
    child = object.__new__(NativeMethodsRefitResult)
    observed: dict[str, Any] = {}

    def native_retrain(source_result: Any, dataset: Any, **kwargs: Any) -> NativeMethodsRefitResult:
        observed.update(source=source_result, dataset=dataset, kwargs=kwargs)
        return child

    session._result = source  # type: ignore[assignment]
    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(retrain_module, "retrain", native_retrain)

    result = session.retrain({"X": [[1.0]], "y": [2.0], "sample_ids": ["target-a"]})

    assert result is child
    assert session.result is child
    assert observed == {
        "source": source,
        "dataset": {"X": [[1.0]], "y": [2.0], "sample_ids": ["target-a"]},
        "kwargs": {
            "mode": "full",
            "name": "portable",
            "save_artifacts": True,
            "verbose": 0,
            "engine": "native",
        },
    }


def test_native_session_factory_routes_without_a_legacy_runner() -> None:
    with nirs4all.session(["split", "model"], name="portable", engine="native", random_state=9) as native:
        assert isinstance(native, NativeMethodsSession)
        assert native.pipeline == ["split", "model"]
        assert native.random_state == 9
        assert not native.closed
    assert native.closed


def test_native_session_factory_refuses_legacy_options() -> None:
    with pytest.raises(ValueError, match="requires a list pipeline"):
        with nirs4all.session(engine="native"):
            pass
    with pytest.raises(NotImplementedError, match="workspace options"):
        with nirs4all.session(["model"], engine="native", workspace_path="legacy"):
            pass


def test_native_load_session_uses_archive_loader_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    session_module = importlib.import_module("nirs4all.api.session")
    archive = tmp_path / "portable.n4a"
    archive.write_bytes(b"native archive")
    expected = object()
    monkeypatch.setattr(
        "nirs4all.api.native_archive_session.load_native_archive_session",
        lambda path: expected,
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.bundle.BundleLoader",
        lambda *_args, **_kwargs: pytest.fail("native load_session must not use BundleLoader"),
    )

    assert session_module.load_session(archive, engine="native") is expected


def test_run_routes_a_native_session_without_calling_legacy_training(monkeypatch: pytest.MonkeyPatch) -> None:
    native = NativeMethodsSession(["split", "model"], name="portable", random_state=3)
    expected = object()
    observed: list[dict[str, Any]] = []

    def native_run(dataset: Any) -> object:
        observed.append(dataset)
        return expected

    monkeypatch.setattr(native, "run", native_run)
    result = nirs4all.run(
        native.pipeline,
        {"X": [[1.0]], "y": [2.0], "sample_ids": ["fit-a"]},
        engine="native",
        session=native,
        random_state=3,
        save_charts=False,
    )

    assert result is expected
    assert observed == [{"X": [[1.0]], "y": [2.0], "sample_ids": ["fit-a"]}]


def test_predict_routes_a_trained_native_session_without_an_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    native = NativeMethodsSession(["split", "model"])
    native._result = _Result()  # type: ignore[assignment]

    result = nirs4all.predict(
        data={"X": [[1.0], [2.0]], "sample_ids": ["predict-a", "predict-b"]},
        engine="native",
        session=native,
    )

    assert result.y_pred.tolist() == [[3.0], [4.0]]
    assert result.metadata == {
        "engine": "native",
        "session_kind": "NativeMethodsSession",
        "sample_ids": ["predict-a", "predict-b"],
    }
    assert native.result.native_estimator.calls[0]["sample_ids"] == ["predict-a", "predict-b"]
