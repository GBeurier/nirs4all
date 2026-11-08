"""Corrupt required transforms/models must not silently disappear on replay."""

import io

import joblib
import pytest

from nirs4all.pipeline.bundle.loader import BundleArtifactLoadError, BundleArtifactProvider
from nirs4all.pipeline.dagml.host_container import JOBLIB_CONTENT_TYPE, JSON_CONTENT_TYPE, HostMember, write_host_container


@pytest.mark.parametrize("payload", [None, b"not a joblib artifact"])
def test_indexed_artifact_failure_stops_replay(tmp_path, payload):
    path = tmp_path / "broken.n4a"
    members = [HostMember("placeholder.json", b"{}", "provenance", JSON_CONTENT_TYPE)]
    if payload is not None:
        members.append(HostMember("artifacts/model.joblib", payload, "model", JOBLIB_CONTENT_TYPE))
    write_host_container(path, members, entrypoint=False)
    with BundleArtifactProvider(path, {"step_2_fold0": "model.joblib"}) as provider:
        with pytest.raises(BundleArtifactLoadError, match="step_2_fold0") as error:
            provider.get_fold_artifacts(2)
        assert error.value.__cause__ is not None


def test_unknown_optional_key_is_distinct_from_corrupt_indexed_artifact(tmp_path):
    provider = BundleArtifactProvider(tmp_path / "unused.n4a", {})
    assert provider.get_artifacts_for_step(1) == []


def test_trusted_valid_artifact_still_loads_and_caches(tmp_path, monkeypatch):
    payload = io.BytesIO()
    joblib.dump({"fitted": True}, payload)
    path = tmp_path / "valid.n4a"
    write_host_container(path, [HostMember("artifacts/model.joblib", payload.getvalue(), "model", JOBLIB_CONTENT_TYPE)], entrypoint=False)
    original = joblib.load
    loaded_paths = []

    def tracked_load(private, *args, **kwargs):
        assert private.is_file()
        assert private.read_bytes() == payload.getvalue()
        loaded_paths.append(private)
        return original(private, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", tracked_load)
    with BundleArtifactProvider(path, {"step_2_fold0": "model.joblib"}) as provider:
        first = provider.get_fold_artifacts(2)
        assert first == [(0, {"fitted": True})]
        assert provider.get_fold_artifacts(2)[0][1] is first[0][1]
        assert len(loaded_paths) == 1
        assert not loaded_paths[0].exists()
