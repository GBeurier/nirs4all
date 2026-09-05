from __future__ import annotations

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
    archive.write_bytes(b"opaque-archive")

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


@pytest.mark.parametrize("environment_value", ["native", " NATIVE "])
def test_native_environment_selects_archive_prediction(
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str,
) -> None:
    """The process selector reaches native replay without an explicit kwarg."""

    monkeypatch.setenv("N4A_ENGINE", environment_value)
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        lambda _path, _X, *, sample_ids, **_kwargs: NativeArchivePrediction(
            values=np.asarray([[3.0]]),
            sample_ids=tuple(sample_ids),
            intervals={},
            conformal_guarantee_status=None,
        ),
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.core_archive_replay.detect_core_archive_version",
        lambda _path: None,
    )

    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
    )

    assert result.y_pred.tolist() == [[3.0]]
    assert result.metadata["engine"] == "native"


def test_native_environment_never_reinterprets_an_invalid_archive_as_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict native validation failures propagate without legacy fallback."""
    monkeypatch.setenv("N4A_ENGINE", "native")
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.core_archive_replay.detect_core_archive_version",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NativeArchiveReplayError("not a portable Archive V2")
        ),
    )

    with pytest.raises(NativeArchiveReplayError, match="not a portable Archive V2"):
        predict(
            model="legacy.n4a",
            data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
        )


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


@pytest.mark.parametrize("engine", ["dag-ml", "dual", "invalid"])
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
