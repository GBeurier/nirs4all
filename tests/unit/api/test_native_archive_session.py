from __future__ import annotations

import numpy as np
import pytest

from nirs4all.api.session import NativeArchiveSession, load_native_archive_session, load_session
from nirs4all.pipeline.dagml.native_archive_replay import NativeArchivePrediction


def test_native_archive_session_replays_explicit_ids_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def replay(path, X, *, sample_ids, groups, metadata):  # noqa: ANN001
        observed.update(path=str(path), X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata)
        return NativeArchivePrediction(
            values=np.asarray([[4.0], [5.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        replay,
    )
    with load_native_archive_session("portable.n4a") as session:
        assert isinstance(session, NativeArchiveSession)
        result = session.predict(
            np.asarray([[1.0], [2.0]]),
            sample_ids=["p1", "p2"],
            groups=["g", "g"],
        )
        assert result.y_pred.tolist() == [[4.0], [5.0]]
        assert result.metadata["engine"] == "native"
    assert session.closed
    assert observed["sample_ids"] == ["p1", "p2"]
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(np.asarray([[1.0]]), sample_ids=["p3"])


def test_load_session_native_uses_the_portable_session_without_bundle_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive = tmp_path / "portable.n4a"
    archive.write_bytes(b"opaque-archive")

    class UnexpectedBundleLoader:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("native load_session constructed a legacy BundleLoader")

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", UnexpectedBundleLoader)

    loaded = load_session(archive, engine="native")

    assert isinstance(loaded, NativeArchiveSession)
    assert loaded.archive_path == archive
    loaded.close()


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
