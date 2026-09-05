"""Corrupt required transforms/models must not silently disappear on replay."""

import io
import zipfile

import joblib
import pytest

from nirs4all.pipeline.bundle.loader import BundleArtifactLoadError, BundleArtifactProvider


@pytest.mark.parametrize("payload", [None, b"not a joblib artifact"])
def test_indexed_artifact_failure_stops_replay(tmp_path, payload):
    path = tmp_path / "broken.n4a"
    with zipfile.ZipFile(path, "w") as archive:
        if payload is not None:
            archive.writestr("artifacts/model.joblib", payload)
    provider = BundleArtifactProvider(path, {"step_2_fold0": "model.joblib"})
    with pytest.raises(BundleArtifactLoadError, match="step_2_fold0") as error:
        provider.get_fold_artifacts(2)
    assert error.value.__cause__ is not None


def test_unknown_optional_key_is_distinct_from_corrupt_indexed_artifact(tmp_path):
    provider = BundleArtifactProvider(tmp_path / "unused.n4a", {})
    assert provider.get_artifacts_for_step(1) == []


def test_trusted_valid_artifact_still_loads_and_caches(tmp_path):
    payload = io.BytesIO()
    joblib.dump({"fitted": True}, payload)
    path = tmp_path / "valid.n4a"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("artifacts/model.joblib", payload.getvalue())
    provider = BundleArtifactProvider(path, {"step_2_fold0": "model.joblib"})
    first = provider.get_fold_artifacts(2)
    assert first == [(0, {"fitted": True})]
    assert provider.get_fold_artifacts(2)[0][1] is first[0][1]
