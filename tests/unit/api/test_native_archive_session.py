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


def test_native_archive_session_replays_explicit_ids_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def validate(path) -> None:  # noqa: ANN001
        observed["validated_path"] = str(path)

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        validate,
    )

    def replay(path, X, *, sample_ids, methods_library_path, groups, metadata):  # noqa: ANN001
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
    with load_native_archive_session("portable.n4a", methods_library_path="/native/libn4m.so") as session:
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
    assert observed["validated_path"] == "portable.n4a"
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(np.asarray([[1.0]]), sample_ids=["p3"])


def test_load_session_native_environment_uses_the_portable_session_without_bundle_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = tmp_path / "portable.n4a"
    with ZipFile(archive, "w") as container:
        container.writestr("manifest.json", json.dumps({
            "schema_version": 2, "profile": "nirs4all.archive_workspace.v2",
            "persistence_kind": "n4a_archive", "writer": {"product_aggregate_owner": "nirs4all-core"},
        }))

    def unsupported(*args, **kwargs):
        raise ValueError("n4a unsupported format: interim archive")

    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(open_n4a_file=unsupported, write_n4a_file=lambda *a, **k: None))

    class UnexpectedBundleLoader:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("native load_session constructed a legacy BundleLoader")

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", UnexpectedBundleLoader)
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda path: None,
    )
    monkeypatch.setenv("N4A_ENGINE", "native")

    loaded = load_session(archive)

    assert isinstance(loaded, NativeArchiveSession)
    assert loaded.archive_path == archive
    loaded.close()


def test_predict_uses_native_archive_session_without_model_or_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def replay(path, X, *, sample_ids, methods_library_path, groups, metadata):  # noqa: ANN001
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
        lambda path: None,
    )
    native_session = load_native_archive_session(
        "portable.n4a", methods_library_path="/native/libn4m.so"
    )

    result = predict(
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        session=native_session,
    )

    assert result.y_pred.tolist() == [[7.0], [8.0]]
    assert result.metadata["engine"] == "native"
    assert observed["path"] == "portable.n4a"
    assert observed["sample_ids"] == ["p1", "p2"]


def _interim_core_archive(tmp_path, monkeypatch: pytest.MonkeyPatch, *, manifest: dict[str, object]):
    """A real ``.n4a`` file whose own manifest declares its profile, behind a Core that refuses it as not V1.

    The V2/V3 routing hint is only consulted after Core's "not V1" refusal, so this is the
    content-declared routing fixture; a path that does not exist routes nowhere.
    """
    archive = tmp_path / "archive.n4a"
    with ZipFile(archive, "w") as container:
        container.writestr("manifest.json", json.dumps(manifest))

    def unsupported(*_args, **_kwargs):
        raise ValueError("n4a unsupported format: interim archive")

    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(open_n4a_file=unsupported, write_n4a_file=lambda *a, **k: None))
    return archive


_CORE_V2_MANIFEST = {
    "schema_version": 2, "profile": "nirs4all.archive_workspace.v2",
    "persistence_kind": "n4a_archive", "writer": {"product_aggregate_owner": "nirs4all-core"},
}


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
    """The process selector reaches native replay of a content-declared Core archive without an explicit kwarg."""

    archive = _interim_core_archive(tmp_path, monkeypatch, manifest=_CORE_V2_MANIFEST)
    observed: dict[str, object] = {}

    def replay(path, data, *, methods_library_path=None):  # noqa: ANN001
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

    declared = _interim_core_archive(tmp_path, monkeypatch, manifest=_CORE_V2_MANIFEST)
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.core_archive_replay.predict_core_methods_archive_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CoreArchiveReplayError("not a portable Archive V2")),
    )
    with pytest.raises(CoreArchiveReplayError, match="not a portable Archive V2"):
        predict(model=str(declared), data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]})

    old_directory = tmp_path / "old"
    old_directory.mkdir()
    old = _interim_core_archive(old_directory, monkeypatch, manifest={"bundle_format_version": "1.0"})
    with pytest.raises(ValueError, match="re-export old SDK ZIP bundles with current nirs4all"):
        predict(model=str(old), data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]})


def test_load_native_archive_session_refuses_a_bad_package_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening rejects an invalid archive before a caller supplies feature data."""

    def reject(_path) -> None:  # noqa: ANN001
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
        load_native_archive_session("invalid.n4a")


def test_load_native_archive_session_dispatches_v3_without_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A V2 refusal may select only the strict V3 Core/DAG-ML reader."""

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
        lambda _path: (_ for _ in ()).throw(NativeArchiveReplayError("not V2")),
    )
    monkeypatch.setattr(
        "nirs4all.api.native_refit_result.NativeMethodsRefitResult.load_archive",
        lambda path, **kwargs: V3Result(),
    )

    loaded = load_native_archive_session("portable-v3.n4a", methods_library_path="/native/libn4m.so")
    assert loaded.archive_schema_version == 3
    result = loaded.predict(np.asarray([[1.0]]), sample_ids=["p1"])
    assert result.y_pred.tolist() == [[9.0]]


def test_predict_native_session_refuses_explicit_non_native_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2",
        lambda path: None,
    )
    with pytest.raises(ValueError, match="explicit non-native engine"):
        predict(
            data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
            session=load_native_archive_session("portable.n4a"),
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
