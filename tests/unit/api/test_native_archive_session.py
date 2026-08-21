from __future__ import annotations

import numpy as np
import pytest

from nirs4all.api.session import NativeArchiveSession, load_native_archive_session
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
