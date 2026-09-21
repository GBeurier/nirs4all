"""A persisted native run is visible to ordinary workspace result readers."""

import json

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
        for chain in chains:
            store.update_chain_summary(chain["chain_id"])
        assert store.query_chain_summaries().to_dicts() == chains, "Single and bulk summaries must retain identical evidence"
    assert chains
    cv_chains = [chain for chain in chains if chain["cv_fold_count"] == 3]
    assert len(cv_chains) == 1, chains
    assert len(chains) == 1, "CV and REFIT of the same native variant must share one chain"
    assert cv_chains[0]["model_name"] == "Ridge"
    assert cv_chains[0]["model_class"].endswith(".Ridge")
    assert cv_chains[0]["preprocessings"] == "StandardScaler"
    assert cv_chains[0]["metric"] == "rmse"
    assert cv_chains[0]["dataset_name"] == "array_dataset"
    assert np.isfinite(cv_chains[0]["cv_val_score"])
    assert cv_chains[0]["final_train_score"] is not None
    assert cv_chains[0]["final_test_score"] is None, "No independent test cohort exists"
    assert "accuracy" not in str(cv_chains[0]["cv_scores"])
    assert result.execution_engine == "dag-ml"
    result.close()


def test_cv_and_independent_test_scores_share_the_selected_native_chain(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 5))
    y = X @ np.arange(1.0, 6.0)
    result = nirs4all.run(
        [StandardScaler(), KFold(3), Ridge()], (X, y, {"train": 30}),
        name="user_refit", workspace_path=tmp_path, save_artifacts=True,
    )
    final_rows = result.predictions.filter_predictions(fold_id="final", partition="test", load_arrays=True)
    assert len(final_rows) == 1
    with WorkspaceStore(tmp_path) as store:
        chains = store.query_chain_summaries().to_dicts()
        selected_chain = store.get_chain(result.best["chain_id"])
        store.update_chain_summary(result.best["chain_id"])
        updated = store.query_chain_summaries().to_dicts()
        for key, value in chains[0].items():
            if key.endswith("scores") and isinstance(value, str):
                assert json.loads(updated[0][key]) == json.loads(value), key
            else:
                assert updated[0][key] == value, key
    assert len(chains) == 1
    assert chains[0]["cv_fold_count"] == 3
    assert chains[0]["final_test_score"] == final_rows[0]["test_score"]
    assert chains[0]["chain_id"] == result.best["chain_id"]
    assert selected_chain is not None and selected_chain["fold_artifacts"].get("final")
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
        # Persisted replay can present the same fitted estimator with a
        # different contiguous layout to platform BLAS. Bound last-bit noise
        # while retaining the no-refit and artifact-scope invariants below.
        np.testing.assert_allclose(prediction.y_pred, expected, rtol=2e-6, atol=2e-6)
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


def test_repeated_array_and_file_replay_never_mutates_persisted_workspace_data(tmp_path, monkeypatch):
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
            for path in workspace.rglob("*") if path.is_file() and path.name not in {"store.sqlite-wal", "store.sqlite-shm"}
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


@pytest.mark.parametrize("writer_state", ["idle", "committed", "uncommitted"])
def test_workspace_replay_reads_committed_snapshot_with_active_writer(tmp_path, monkeypatch, writer_state):
    import sqlite3

    import nirs4all
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain

    X = np.arange(120.0).reshape(30, 4)
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.12), workspace_path=tmp_path)
    selected = result.best
    result.close()
    expected = nirs4all.predict(selected, X).y_pred
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("workspace replay fitted a model"))
    writer = sqlite3.connect(tmp_path / "store.sqlite")
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        original_name = writer.execute("SELECT model_name FROM chains WHERE chain_id = ?", [selected["chain_id"]]).fetchone()[0]
        if writer_state != "idle":
            writer.execute("UPDATE chains SET model_name = ? WHERE chain_id = ?", ["concurrent-name", selected["chain_id"]])
            if writer_state == "committed":
                writer.commit()
        wal = tmp_path / "store.sqlite-wal"
        assert wal.exists(), "Exercise a real SQLite WAL, not a dummy journal"
        wal_before = wal.read_bytes()
        loaded = load_general_workspace_chain(tmp_path, selected["chain_id"])
        assert loaded is not None
        assert loaded["chain"]["model_name"] == ("concurrent-name" if writer_state == "committed" else original_name)
        np.testing.assert_array_equal(nirs4all.predict(selected, X).y_pred, expected)
        assert wal.read_bytes() == wal_before, "Prediction must not commit or checkpoint the writer"
    finally:
        writer.close()


def test_workspace_replay_keeps_one_snapshot_during_concurrent_commit(tmp_path, monkeypatch):
    import sqlite3

    import nirs4all
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain
    from nirs4all.pipeline.storage.store_queries import GET_CHAIN

    X = np.arange(120.0).reshape(30, 4)
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.12), workspace_path=tmp_path)
    chain_id = result.best["chain_id"]
    result.close()
    writer = sqlite3.connect(tmp_path / "store.sqlite")
    writer.execute("PRAGMA journal_mode=WAL")
    original_connect = sqlite3.connect
    committed = []

    class ConcurrentCommitConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql == GET_CHAIN:
                # Remove the artifact record after replay has read the chain.
                # A transaction must still see the matching earlier artifact.
                writer.execute("DELETE FROM artifacts")
                writer.commit()
                committed.append(True)
            return cursor

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: original_connect(*args, factory=ConcurrentCommitConnection, **kwargs))
    try:
        loaded = load_general_workspace_chain(tmp_path, chain_id)
        assert committed == [True]
        assert loaded is not None and loaded["metadata"]["artifact_integrity_verified"]
        assert writer.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    finally:
        writer.close()
