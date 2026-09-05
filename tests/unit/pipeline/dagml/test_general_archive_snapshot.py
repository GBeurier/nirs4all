"""Session identity binds the exact bytes parsed and unpickled, not a prior read."""

import hashlib
from pathlib import Path

import pytest


def test_expected_archive_fingerprint_is_checked_before_zip_or_pickle(tmp_path, monkeypatch):
    import joblib

    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    expected = "sha256:" + hashlib.sha256(b"original session archive").hexdigest()
    path = tmp_path / "replaced.n4a"
    path.write_bytes(b"replacement is not even a zip")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("changed Session artifact deserialized"))
    with pytest.raises(ValueError, match="changed after loading"):
        load_general_archive(path, expected_archive_fingerprint=expected)


def test_predict_general_archive_checks_only_its_single_byte_snapshot(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml.general_archive import predict_general_archive

    expected = "sha256:" + hashlib.sha256(b"original session archive").hexdigest()
    path = tmp_path / "changing.n4a"
    reads = []

    def read_bytes(source):
        assert source == path
        reads.append(1)
        return b"replacement"

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(ValueError, match="changed after loading"):
        predict_general_archive(path, object(), expected_archive_fingerprint=expected)
    assert reads == [1]
