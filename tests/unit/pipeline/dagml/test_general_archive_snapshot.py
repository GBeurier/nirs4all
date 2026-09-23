"""Session identity binds the streamed snapshot parsed and unpickled."""

import hashlib
import json
import zipfile
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


def test_predict_general_archive_checks_its_streamed_snapshot(tmp_path):
    from nirs4all.pipeline.dagml.general_archive import predict_general_archive

    expected = "sha256:" + hashlib.sha256(b"original session archive").hexdigest()
    path = tmp_path / "changing.n4a"
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed after loading"):
        predict_general_archive(path, object(), expected_archive_fingerprint=expected)


@pytest.mark.parametrize("host_artifacts", ["invalid", [{"files": "invalid"}], [{"files": [{"uri": 3}]}]])
def test_malformed_sidecar_manifest_refused_before_unpickle(tmp_path, monkeypatch, host_artifacts):
    import joblib

    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = tmp_path / "malformed.n4a"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({
            "source_type": "dagml_native", "host_artifacts": host_artifacts,
            "artifact_integrity": {"artifacts/model.joblib": "sha256:" + hashlib.sha256(b"not a pickle").hexdigest()},
        }))
        archive.writestr("artifacts/model.joblib", b"not a pickle")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("malformed sidecar unpickled"))
    with pytest.raises(ValueError, match="invalid host sidecar manifest"):
        load_general_archive(path)


def test_oversized_manifest_refused_before_unpickle(tmp_path, monkeypatch):
    import joblib

    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    path = tmp_path / "oversized.n4a"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"source_type": "dagml_native", "padding": "x" * (1024 * 1024)}))
        archive.writestr("artifacts/model.joblib", b"not a pickle")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("oversized manifest unpickled"))
    with pytest.raises(ValueError, match="manifest exceeds 1 MiB"):
        load_general_archive(path)
