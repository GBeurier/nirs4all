"""Real workspace readers preserve data and see consistent committed WAL state."""

import sqlite3

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def test_readonly_store_predictions_arrays_and_wal_snapshot(tmp_path):
    import nirs4all

    X = np.random.default_rng(18).normal(size=(30, 5))
    with nirs4all.run([KFold(3), Ridge()], (X, X[:, 0]), workspace_path=tmp_path, save_charts=False, verbose=0) as run:
        assert run.num_predictions > 0
    with WorkspaceStore(tmp_path) as writer:
        with WorkspaceStore.open_readonly(tmp_path) as reader:
            assert reader.query_predictions().height > 0
            assert reader.query_chain_summaries().height == 1
            prediction_id = reader.query_predictions(partition="val")["prediction_id"][0]
            prediction = reader.get_prediction(prediction_id, load_arrays=True)
            assert prediction is not None and np.isfinite(prediction["y_pred"]).all()
            original_count = reader.list_runs().height
            writer.begin_run("Concurrent", {}, [])
            assert reader.list_runs().height == original_count
        with WorkspaceStore.open_readonly(tmp_path) as reader:
            assert reader.list_runs().height == original_count + 1


def test_readonly_store_never_initializes_or_reconciles_files(tmp_path, monkeypatch):
    import nirs4all.pipeline.storage.workspace_store as module

    with WorkspaceStore(tmp_path):
        pass
    (tmp_path / "artifacts").rmdir()
    sentinel = tmp_path / "arrays" / "writer.parquet.tmp"
    sentinel.write_bytes(b"uncommitted arrays")
    tombstones = tmp_path / "arrays" / "_tombstones.json"
    tombstones.write_text('{"pending": "2026-09-19"}')
    monkeypatch.setattr(module, "create_schema", lambda *a: pytest.fail("reader migrated schema"))
    monkeypatch.setattr(WorkspaceStore, "compact_arrays", lambda *a: pytest.fail("reader reconciled arrays"))
    with WorkspaceStore.open_readonly(tmp_path) as reader:
        assert reader.query_predictions().is_empty()
        with pytest.raises(RuntimeError, match="read-only"):
            reader.begin_run("Forbidden", {}, [])
        with pytest.raises(RuntimeError, match="read-only"):
            reader.save_artifact({}, "dict", "model", "joblib")
        with pytest.raises(RuntimeError, match="read-only"):
            reader.array_store.save_batch([])
        with pytest.raises(sqlite3.OperationalError):
            reader._ensure_open().execute("DELETE FROM runs")
    assert not (tmp_path / "artifacts").exists()
    assert sentinel.read_bytes() == b"uncommitted arrays"
    assert tombstones.read_text() == '{"pending": "2026-09-19"}'


def test_readonly_missing_workspace_is_not_created(tmp_path):
    absent = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        WorkspaceStore.open_readonly(absent)
    assert not absent.exists()
