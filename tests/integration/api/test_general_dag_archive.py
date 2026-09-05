"""Public DAG host archive and Session replay never train or enter legacy."""

import json
import zipfile

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("scale_y", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
def test_public_general_archive_and_session_roundtrip(tmp_path, monkeypatch, scale_y, multi_target):
    import nirs4all

    rng = np.random.default_rng(893)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] * 3 + X[:, 2]
    if multi_target:
        y = np.column_stack([y, X[:, 3] * 6])
    pipeline = [StandardScaler(), KFold(n_splits=3)]
    if scale_y:
        pipeline.append({"y_processing": StandardScaler()})
    pipeline.append(Ridge())
    session = nirs4all.Session(pipeline=pipeline, workspace_path=tmp_path / "workspace")
    result = session.run((X, y))
    model = result._dagml_refit_artifacts[0]
    new_X = rng.normal(size=(17, 5)).astype(np.float32)
    expected = np.asarray(model["estimator"].predict(new_X), dtype=float).reshape(17, -1)
    if model["y_transform"] is not None:
        expected = model["y_transform"].inverse_transform(expected)
    if not multi_target:
        expected = expected.ravel()
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("prediction or export retrained"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("prediction or export refitted transform"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.predict", lambda *args, **kwargs: pytest.fail("legacy replay"))
    archive = session.save(tmp_path / "captured.n4a")
    live = session.predict(new_X)
    public = nirs4all.predict(archive, new_X)
    loaded = nirs4all.load_session(archive)
    replayed = loaded.predict(new_X)
    duplicate = loaded.save(tmp_path / "duplicate.n4a")
    assert duplicate.read_bytes() == archive.read_bytes()
    for prediction in (live, public, replayed):
        np.testing.assert_array_equal(prediction.y_pred, expected)
        assert prediction.metadata["phase"] == "PREDICT"
        assert prediction.metadata["scores"] is None
        assert prediction.metadata["training_performed"] is False
        assert prediction.metadata["portable"] is False
    assert public.metadata["artifact_integrity_verified"] is True
    assert loaded.execution_engine == "dag-ml"
    with pytest.raises(Exception, match="portable|host profile"):
        nirs4all.predict(archive, new_X, engine="native")
    session.close()
    loaded.close()


def test_general_archive_digest_checked_before_unpickling_and_session_binds_source(tmp_path, monkeypatch):
    import joblib

    import nirs4all

    X = np.arange(60, dtype=float).reshape(20, 3)
    result = nirs4all.run([KFold(n_splits=2), Ridge()], (X, X[:, 0]), workspace_path=tmp_path / "workspace")
    path = result.export(tmp_path / "model.n4a")
    loaded = nirs4all.load_session(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    model_member = next(iter(manifest["artifact_integrity"]))
    members[model_member] += b"tampered"
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("tampered bytes reached pickle"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        nirs4all.predict(path, X)
    with pytest.raises(ValueError, match="changed"):
        loaded.predict(X)
