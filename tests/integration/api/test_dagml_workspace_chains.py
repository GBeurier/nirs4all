"""A persisted native run is visible to ordinary workspace result readers."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def test_general_run_populates_chain_summaries(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    rng = np.random.default_rng(17)
    X = rng.normal(size=(30, 5))
    result = nirs4all.run(
        [StandardScaler(), KFold(3), Ridge()], (X, X @ np.arange(1.0, 6.0)),
        workspace_path=tmp_path, save_artifacts=True,
    )
    with WorkspaceStore(tmp_path) as store:
        chains = store.query_chain_summaries().to_dicts()
    assert chains
    cv_chains = [chain for chain in chains if chain["cv_fold_count"] == 3]
    assert len(cv_chains) == 1, chains
    assert cv_chains[0]["model_name"] == "Ridge"
    assert cv_chains[0]["model_class"].endswith(".Ridge")
    assert cv_chains[0]["preprocessings"] == "StandardScaler"
    assert cv_chains[0]["metric"] == "rmse"
    assert cv_chains[0]["dataset_name"] == "array_dataset"
    assert np.isfinite(cv_chains[0]["cv_val_score"])
    assert result.execution_engine == "dag-ml"
    result.close()


def test_public_workspace_chain_and_best_dict_replay_captured_refit(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(27)
    X = rng.normal(size=(30, 5))
    result = nirs4all.run(
        [StandardScaler(), {"y_processing": StandardScaler()}, KFold(3), Ridge()],
        (X, X @ np.arange(1.0, 6.0)), workspace_path=tmp_path, save_artifacts=True,
    )
    selected = result.best
    assert selected["chain_id"]
    artifact = result._dagml_refit_artifacts[0]
    scaled = artifact["estimator"].predict(X.astype(np.float32))
    expected = artifact["y_transform"].inverse_transform(np.asarray(scaled, dtype=float).reshape(-1, 1)).ravel()
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("workspace replay fitted a model"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.predict", lambda *args, **kwargs: pytest.fail("legacy replay"))
    for prediction in (
        nirs4all.predict(chain_id=selected["chain_id"], workspace_path=tmp_path, data=X),
        nirs4all.predict(selected, X),
    ):
        np.testing.assert_array_equal(prediction.y_pred, expected)
        assert prediction.metadata["phase"] == "PREDICT"
        assert prediction.metadata["artifact_scope"] == "full_training_refit"
        assert prediction.metadata["cv_artifacts_available"] is False
        assert prediction.metadata["training_performed"] is False
    result.close()


def test_corrupted_workspace_artifact_is_rejected_before_deserialization(tmp_path, monkeypatch):
    import sqlite3

    import joblib

    import nirs4all

    X = np.arange(120.0).reshape(30, 4)
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.12), workspace_path=tmp_path)
    connection = sqlite3.connect(f"{(tmp_path / 'store.sqlite').as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        relative = connection.execute("SELECT artifact_path FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    path = tmp_path / "artifacts" / relative
    path.write_bytes(b"corrupted fitted payload")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("unverified pickle loaded"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        nirs4all.predict(result.best, X)
    result.close()


def test_portable_native_selection_never_deserializes_general_workspace(tmp_path, monkeypatch):
    import joblib

    import nirs4all
    from nirs4all.pipeline.dagml.rt import RtError

    X = np.arange(120.0).reshape(30, 4)
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.12), workspace_path=tmp_path)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("portable profile loaded Python workspace artifact"))
    with pytest.raises(RtError):
        nirs4all.predict(result.best, X, engine="native")
    result.close()


def test_repeated_array_and_file_replay_never_mutates_workspace(tmp_path, monkeypatch):
    import hashlib

    import nirs4all
    from nirs4all.api.dataset_inspection import load_prediction_file
    from nirs4all.pipeline.storage.model_catalogue import read_model_catalogue
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 5))
    workspace = tmp_path / "workspace"
    result = nirs4all.run(
        [StandardScaler(), KFold(3), Ridge()], (X, X @ np.arange(1.0, 6.0)),
        workspace_path=workspace, save_artifacts=True,
    )
    selected = result.best
    result.close()
    csv = tmp_path / "prediction.csv"
    np.savetxt(csv, X, delimiter=",")
    dataset, _, _ = load_prediction_file(csv, params={"has_header": False, "delimiter": ","})

    def snapshot():
        return {
            path.relative_to(workspace).as_posix(): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in workspace.rglob("*") if path.is_file()
        }

    before = snapshot()
    assert not any(name.endswith(("-wal", "-shm", "-journal")) for name in before)
    catalogue = read_model_catalogue(workspace)
    monkeypatch.setattr(WorkspaceStore, "__init__", lambda *args, **kwargs: pytest.fail("prediction opened a writable store"))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("prediction fitted a model"))
    predictions = []
    for data in (X, dataset, X, dataset):
        predictions.append(nirs4all.predict(selected, data).y_pred)
        assert snapshot() == before
        assert read_model_catalogue(workspace) == catalogue
    for prediction in predictions[1:]:
        np.testing.assert_array_equal(prediction, predictions[0])


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_workspace_replay_refuses_active_journals_before_loading(tmp_path, monkeypatch, suffix):
    import joblib

    import nirs4all
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain

    X = np.arange(120.0).reshape(30, 4)
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.12), workspace_path=tmp_path)
    chain_id = result.best["chain_id"]
    result.close()
    sidecar = tmp_path / f"store.sqlite{suffix}"
    sidecar.write_bytes(b"writer-owned journal")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("active database deserialized a model"))
    with pytest.raises(RuntimeError, match="active SQLite journal"):
        load_general_workspace_chain(tmp_path, chain_id)
    assert sidecar.read_bytes() == b"writer-owned journal"
