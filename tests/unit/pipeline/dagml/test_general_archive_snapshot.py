"""General readers bind one verified Core source before loading trusted models."""

import hashlib
import io
import json
from pathlib import Path

import joblib
import pytest

from nirs4all.pipeline.dagml.host_container import (
    BUNDLE_CONTENT_TYPE,
    BUNDLE_MANIFEST_MEMBER,
    JOBLIB_CONTENT_TYPE,
    HostMember,
    open_n4a_source,
    write_host_container,
)


class _Predictor:
    def predict(self, X):
        return [1.0] * len(X)


def _write_general_archive(path: Path, *, metadata=None, payload=None):
    if payload is None:
        buffer = io.BytesIO()
        joblib.dump(_Predictor(), buffer)
        payload = buffer.getvalue()
    manifest = {
        "source_type": "dagml_native",
        "artifact_integrity": {"artifacts/model.joblib": "sha256:" + hashlib.sha256(payload).hexdigest()},
        **(metadata or {}),
    }
    return write_host_container(path, [
        HostMember(BUNDLE_MANIFEST_MEMBER, json.dumps(manifest).encode(), "provenance", BUNDLE_CONTENT_TYPE),
        HostMember("artifacts/model.joblib", payload, "model", JOBLIB_CONTENT_TYPE),
    ])


def test_expected_archive_fingerprint_is_checked_before_pickle(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = _write_general_archive(tmp_path / "replaced.n4a", metadata={"label": "original"})
    with open_n4a_source(path) as source:
        expected = source.require_host().sha256
    _write_general_archive(path, metadata={"label": "replacement"})
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("changed Session artifact deserialized"))
    with pytest.raises(ValueError, match="changed after loading"):
        load_general_archive(path, expected_archive_fingerprint=expected)


def test_predict_general_archive_checks_the_opened_receipt(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml.general_archive import predict_general_archive

    path = _write_general_archive(tmp_path / "changing.n4a")
    expected = "sha256:" + hashlib.sha256(b"original session archive").hexdigest()
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("changed Session artifact deserialized"))
    with pytest.raises(ValueError, match="changed after loading"):
        predict_general_archive(path, object(), expected_archive_fingerprint=expected)


@pytest.mark.parametrize("host_artifacts", ["invalid", [{"files": "invalid"}], [{"files": [{"uri": 3}]}]])
def test_malformed_sidecar_manifest_refused_before_unpickle(tmp_path, monkeypatch, host_artifacts):
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = _write_general_archive(tmp_path / "malformed.n4a", metadata={"host_artifacts": host_artifacts}, payload=b"not a pickle")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("malformed sidecar unpickled"))
    with pytest.raises(ValueError, match="invalid host sidecar manifest"):
        load_general_archive(path)


def test_manifest_larger_than_one_mib_is_accepted(tmp_path):
    from nirs4all.pipeline.dagml.general_archive import general_archive_manifest, load_general_archive

    padding = "x" * (1024 * 1024 + 1)
    path = _write_general_archive(tmp_path / "large_manifest.n4a", metadata={"padding": padding})
    with open_n4a_source(path) as source:
        assert source.require_host().size(BUNDLE_MANIFEST_MEMBER) > 1024 * 1024
        assert general_archive_manifest(path, source=source)["padding"] == padding
        loaded = load_general_archive(path, source=source)
    assert loaded["manifest"]["padding"] == padding
    assert loaded["artifact"]["estimator"].predict([0, 1]) == [1.0, 1.0]
    assert loaded["artifact_integrity_verified"] is True


def test_borrowed_source_is_shared_by_metadata_load_and_copy(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml import general_archive

    path = _write_general_archive(tmp_path / "borrowed.n4a")
    destination = tmp_path / "saved.n4a"
    with open_n4a_source(path) as source:
        container = source.require_host()
        monkeypatch.setattr(general_archive, "open_n4a_source", lambda *args, **kwargs: pytest.fail("borrowed source reopened"))
        assert general_archive.general_archive_manifest(path, source=source)["source_type"] == "dagml_native"
        loaded = general_archive.load_general_archive(path, source=source, expected_archive_fingerprint=container.sha256)
        assert loaded["artifact"]["content_fingerprint"] == container.digest("artifacts/model.joblib")
        assert general_archive.copy_general_archive(path, destination, expected_fingerprint=container.sha256, source=source) == destination
        assert container.core_file.closed is False
        assert destination.read_bytes() == path.read_bytes()
    assert container.core_file.closed is True


def test_standalone_general_reader_opens_once_and_closes(tmp_path, monkeypatch):
    import nirs4all_core

    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = _write_general_archive(tmp_path / "owned.n4a")
    original = nirs4all_core.open_n4a_file
    opened = []

    def tracked_open(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(nirs4all_core, "open_n4a_file", tracked_open)
    load_general_archive(path)
    assert len(opened) == 1
    assert opened[0].closed is True


def test_wrong_model_digest_is_refused_before_unpickle(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = _write_general_archive(tmp_path / "wrong_digest.n4a", metadata={"artifact_integrity": {"artifacts/model.joblib": "sha256:" + "0" * 64}})
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("mismatched model deserialized"))
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_general_archive(path)


def test_copy_refuses_changed_receipt_without_replacing_destination(tmp_path):
    from nirs4all.pipeline.dagml.general_archive import copy_general_archive

    path = _write_general_archive(tmp_path / "copy_source.n4a")
    destination = tmp_path / "saved.n4a"
    destination.write_bytes(b"existing destination")
    with pytest.raises(ValueError, match="changed after loading"):
        copy_general_archive(path, destination, expected_fingerprint="sha256:" + "0" * 64)
    assert destination.read_bytes() == b"existing destination"


def test_cached_general_reader_refuses_changed_source_before_replay(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml.general_archive import load_general_archive, predict_general_archive

    path = _write_general_archive(tmp_path / "cached.n4a", metadata={"label": "original"})
    loaded = load_general_archive(path)
    _write_general_archive(path, metadata={"label": "replacement"})
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("cached Session artifact deserialized again"))
    with pytest.raises(ValueError, match="changed after loading"):
        predict_general_archive(path, object(), loaded_archive=loaded, expected_archive_fingerprint=loaded["archive_fingerprint"])
