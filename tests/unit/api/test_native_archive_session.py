"""Public native Archive V2 session boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nirs4all.api.native_archive_session import load_native_archive_session


def test_native_archive_session_validates_on_open_replays_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "portable.n4a"
    archive.write_bytes(b"opaque archive")
    observed: dict[str, object] = {}

    def validate(path: Path) -> None:
        observed["validated"] = path

    def replay(path: Path, X, *, sample_ids, groups, metadata):  # noqa: ANN001
        observed.update(path=path, X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata)
        return np.asarray([[2.0], [3.0]])

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.validate_methods_archive_v2", validate
    )
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw", replay
    )

    session = load_native_archive_session(archive)
    result = session.predict(np.asarray([[1.0], [2.0]]), sample_ids=["p1", "p2"])

    assert observed["validated"] == archive
    assert observed["sample_ids"] == ["p1", "p2"]
    assert result.y_pred.tolist() == [[2.0], [3.0]]
    assert result.metadata["engine"] == "native"
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(np.asarray([[1.0]]), sample_ids=["p3"])


@pytest.mark.parametrize("name", ["missing.n4a", "not-an-archive.txt"])
def test_native_archive_session_refuses_invalid_path(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if path.suffix == ".txt":
        path.write_bytes(b"not archive")
    with pytest.raises((FileNotFoundError, ValueError)):
        load_native_archive_session(path)
