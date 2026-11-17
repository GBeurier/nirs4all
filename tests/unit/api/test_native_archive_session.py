from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from nirs4all.api.predict import predict
from nirs4all.api.session import NativeArchiveSession, load_native_archive_session, load_session
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchivePrediction,
    NativeArchiveReplayError,
)
from tests.unit.api.test_core_archive_session import _open_native, _open_native_package


def _native_archive(tmp_path, monkeypatch: pytest.MonkeyPatch, *, version: int = 2, name: str = "portable.n4a"):
    """A file that the Core double opens as a container declaring a native Package ``version``."""
    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(open_n4a_file=_open_native, open_native_package=_open_native_package, write_n4a_file=lambda *a, **k: None))
    archive = tmp_path / name
    archive.write_bytes(f"native-v{version}".encode())
    return archive


def test_native_archive_session_replays_explicit_ids_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = _native_archive(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def validate(path, *, source) -> None:  # noqa: ANN001
        assert source.kind == "native" and source.package_version == 2
        observed.update(validated_path=str(path), source=source, core_file=source.container.core_file)

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        validate,
    )

    def replay(path, X, *, sample_ids, methods_library_path, groups, metadata, source):  # noqa: ANN001
        # Replay reads through the one Core handle the session opened and validated.
        assert source is observed["source"] and not source.container.core_file.closed
        observed.update(path=str(path), X=np.asarray(X), sample_ids=list(sample_ids), methods_library_path=methods_library_path, groups=groups, metadata=metadata)
        return NativeArchivePrediction(
            values=np.asarray([[4.0], [5.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
            conformal_presentation={
                "schema_version": 1,
                "sample_ids": ["p1", "p2"],
                "point_predictions": [4.0, 5.0],
            },
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        replay,
    )
    with load_native_archive_session(archive, methods_library_path="/native/libn4m.so") as session:
        assert isinstance(session, NativeArchiveSession)
        result = session.predict(
            np.asarray([[1.0], [2.0]]),
            sample_ids=["p1", "p2"],
            groups=["g", "g"],
        )
        assert result.y_pred.tolist() == [[4.0], [5.0]]
        assert result.metadata["engine"] == "native"
        assert result.metadata["conformal_presentation"]["sample_ids"] == ["p1", "p2"]
    assert session.closed
    assert observed["sample_ids"] == ["p1", "p2"]
    assert observed["methods_library_path"] == "/native/libn4m.so"
    assert observed["validated_path"] == str(archive)
    assert observed["path"] == str(archive)
    assert observed["core_file"].closed
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(np.asarray([[1.0]]), sample_ids=["p3"])


def test_load_session_native_environment_uses_the_portable_session_without_bundle_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = _native_archive(tmp_path, monkeypatch)

    class UnexpectedBundleLoader:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("native load_session constructed a legacy BundleLoader")

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", UnexpectedBundleLoader)
    validated: list[object] = []
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda path, *, source: validated.append((path, source.kind)),
    )
    monkeypatch.setenv("N4A_ENGINE", "native")

    loaded = load_session(archive)

    assert validated == [(archive, "native")]

    assert isinstance(loaded, NativeArchiveSession)
    assert loaded.archive_path == archive
    loaded.close()


def test_predict_uses_native_archive_session_without_model_or_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = _native_archive(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def replay(path, X, *, sample_ids, methods_library_path, groups, metadata, source):  # noqa: ANN001
        assert source.kind == "native"
        observed.update(path=str(path), X=np.asarray(X), sample_ids=list(sample_ids))
        return NativeArchivePrediction(
            values=np.asarray([[7.0], [8.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        replay,
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda path, *, source: None,
    )
    native_session = load_native_archive_session(
        archive, methods_library_path="/native/libn4m.so"
    )

    result = predict(
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        session=native_session,
    )

    assert result.y_pred.tolist() == [[7.0], [8.0]]
    assert result.metadata["engine"] == "native"
    assert observed["path"] == str(archive)
    assert observed["sample_ids"] == ["p1", "p2"]
    native_session.close()


def _old_sdk_zip_bundle(tmp_path):
    """A pre-V1 SDK ZIP bundle: a real ``.n4a`` file that the installed Core refuses as not V1."""
    archive = tmp_path / "old.n4a"
    with ZipFile(archive, "w") as container:
        container.writestr("manifest.json", json.dumps({"bundle_format_version": "1.0"}))
    return archive


def _forbid_legacy_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``nirs4all.api.predict`` as an attribute is the exported function, so resolve the module explicitly.
    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(
        predict_module,
        "_predict_from_model",
        lambda **_kwargs: pytest.fail("an archive was reinterpreted by the legacy bundle lane"),
    )


@pytest.mark.parametrize("environment_value", ["native", " NATIVE "])
def test_native_environment_selects_archive_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment_value: str,
) -> None:
    """The process selector reaches native replay of a Core-declared native package without an explicit kwarg."""

    archive = _native_archive(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    def replay(path, data, *, methods_library_path=None, source):  # noqa: ANN001
        assert source.kind == "native" and source.package_version == 2
        observed.update(path=path, sample_ids=list(data["sample_ids"]), methods_library_path=methods_library_path)
        return np.asarray([[3.0]]), {"sample_ids": list(data["sample_ids"])}

    monkeypatch.setenv("N4A_ENGINE", environment_value)
    monkeypatch.setattr("nirs4all.pipeline.dagml.core_archive_replay.predict_core_methods_archive_v2", replay)
    _forbid_legacy_replay(monkeypatch)

    result = predict(
        model=str(archive),
        data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
    )

    assert result.y_pred.tolist() == [[3.0]]
    assert result.model_name == "MethodsN4MM"
    assert str(observed["path"]) == str(archive)
    assert observed["sample_ids"] == ["p1"] and observed["methods_library_path"] is None


def test_native_environment_never_reinterprets_an_invalid_archive_as_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Strict native validation failures propagate and an undeclared old archive is refused, never run as legacy."""
    from nirs4all.pipeline.dagml.core_archive_replay import CoreArchiveReplayError

    monkeypatch.setenv("N4A_ENGINE", "native")
    _forbid_legacy_replay(monkeypatch)

    declared = _native_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.core_archive_replay.predict_core_methods_archive_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CoreArchiveReplayError("not a portable Archive V2")),
    )
    with pytest.raises(CoreArchiveReplayError, match="not a portable Archive V2"):
        predict(model=str(declared), data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]})

    # The installed Core, not a double, refuses a real old SDK ZIP bundle; no reader or converter exists for it.
    monkeypatch.undo()
    monkeypatch.setenv("N4A_ENGINE", "native")
    _forbid_legacy_replay(monkeypatch)
    pytest.importorskip("nirs4all_core")
    with pytest.raises(ValueError, match="n4a unsupported format.*re-export it with the current nirs4all"):
        predict(model=str(_old_sdk_zip_bundle(tmp_path)), data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]})


def test_load_native_archive_session_refuses_a_bad_package_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Opening rejects an invalid archive before a caller supplies feature data."""
    archive = _native_archive(tmp_path, monkeypatch, name="invalid.n4a")
    handles: list[object] = []

    def reject(_path, *, source) -> None:  # noqa: ANN001
        handles.append(source.container.core_file)
        raise NativeArchiveReplayError("bad archive package")

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        reject,
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prediction was attempted")),
    )

    with pytest.raises(NativeArchiveReplayError, match="bad archive package"):
        load_native_archive_session(archive)
    # The refused open releases its own Core handle.
    assert len(handles) == 1 and handles[0].closed


def test_load_native_archive_session_dispatches_v3_without_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Core's declared Package V3 selects only the strict V3 Core/DAG-ML reader."""
    archive = _native_archive(tmp_path, monkeypatch, version=3, name="portable-v3.n4a")
    loaded_with: dict[str, object] = {}

    class V3Result:
        def predict(self, X, *, sample_ids, groups=None, metadata=None):  # noqa: ANN001
            assert np.asarray(X).tolist() == [[1.0]]
            assert sample_ids == ["p1"]
            return type(
                "Prediction",
                (),
                {
                    "y_pred": np.asarray([[9.0]]),
                    "metadata": {"engine": "native", "sample_ids": ["p1"]},
                    "intervals": {},
                },
            )()

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda *_args, **_kwargs: pytest.fail("a declared Package V3 was validated as Package V2"),
    )

    def load_archive(path, **kwargs):  # noqa: ANN001
        loaded_with.update(path=path, **kwargs)
        return V3Result()

    monkeypatch.setattr("nirs4all.api.native_refit_result.NativeMethodsRefitResult.load_archive", load_archive)

    loaded = load_native_archive_session(archive, methods_library_path="/native/libn4m.so")
    assert loaded.archive_schema_version == 3
    assert loaded_with["path"] == archive and loaded_with["methods_library_path"] == "/native/libn4m.so"
    assert loaded_with["source"].kind == "native" and loaded_with["source"].package_version == 3
    result = loaded.predict(np.asarray([[1.0]]), sample_ids=["p1"])
    assert result.y_pred.tolist() == [[9.0]]


def test_predict_native_session_refuses_explicit_non_native_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = _native_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda path, *, source: None,
    )
    with pytest.raises(ValueError, match="explicit non-native engine"):
        predict(
            data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
            session=load_native_archive_session(archive),
            engine="legacy",
        )


@pytest.mark.parametrize("engine", ["dual", "invalid"])
def test_load_session_rejects_non_native_nonlegacy_engines_before_bundle_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    engine: str,
) -> None:
    bundle = tmp_path / "portable.n4a"
    bundle.write_bytes(b"opaque-archive")

    class UnexpectedBundleLoader:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("unsupported load_session engine constructed a legacy BundleLoader")

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", UnexpectedBundleLoader)

    with pytest.raises((NotImplementedError, ValueError), match="load_session|unknown nirs4all engine"):
        load_session(bundle, engine=engine)
