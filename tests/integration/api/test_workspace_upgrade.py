"""An explicit schema upgrade preserves the old workspace and yields a usable copy."""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.workspace.upgrade import upgrade_workspace_copy


def _source(root):
    root.mkdir()
    with sqlite3.connect(root / "store.sqlite") as connection:
        connection.executescript((Path(__file__).parents[2] / "fixtures" / "workspace_schema_v2.sql").read_text())
        connection.execute("INSERT INTO runs(run_id,name,status) VALUES ('historical', 'Original run', 'completed')")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "opaque.joblib").write_bytes(b"preserved without unpickling")
    (root / "arrays").mkdir()
    (root / "arrays" / "other.dat").write_bytes(b"preserved auxiliary content")
    return root


def _bytes(root):
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_upgrade_real_schema_two_then_train_and_read_committed_wal(tmp_path):
    import nirs4all

    source = _source(tmp_path / "source")
    original = _bytes(source)
    output = tmp_path / "upgraded"
    receipt = upgrade_workspace_copy(source, output)
    assert _bytes(source) == original
    assert receipt["source_schema_version"] == 2
    assert receipt["target_schema_version"] == 5
    assert receipt["source_preserved"] is True
    assert receipt["store_content_sha256"] == hashlib.sha256((output / "store.sqlite").read_bytes()).hexdigest()
    assert (output / "artifacts" / "opaque.joblib").read_bytes() == original["artifacts/opaque.joblib"][0]
    assert (output / "arrays" / "other.dat").read_bytes() == original["arrays/other.dat"][0]
    with WorkspaceStore.open_readonly(output) as store:
        assert store.get_run("historical")["name"] == "Original run"
        for table in ("conformal_results", "tuning_results", "robustness_results"):
            assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    X = np.random.default_rng(37).normal(size=(30, 5))
    result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0]), workspace_path=output, engine="dag-ml", verbose=0)
    result.close()
    with WorkspaceStore(output) as writer:
        run_id = writer.begin_run("After upgrade", {}, [])
        with WorkspaceStore.open_readonly(output) as reader:
            assert reader.get_run(run_id)["name"] == "After upgrade"
            assert reader.query_predictions().height > 0
    assert _bytes(source) == original
    # Receipt binds the completed conversion, not the workspace's future writes.
    assert receipt["store_content_sha256"] != hashlib.sha256((output / "store.sqlite").read_bytes()).hexdigest()


def test_upgrade_refuses_unqualified_sources_and_never_overwrites_output(tmp_path):
    source = _source(tmp_path / "source")
    original = _bytes(source)
    with pytest.raises(ValueError, match="disjoint"):
        upgrade_workspace_copy(source, source / "nested")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        upgrade_workspace_copy(source, occupied)
    with sqlite3.connect(source / "store.sqlite") as connection:
        connection.execute("CREATE TABLE prediction_arrays (prediction_id TEXT)")
    with pytest.raises(ValueError, match="legacy conversion"):
        upgrade_workspace_copy(source, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    assert (source / "artifacts" / "opaque.joblib").read_bytes() == original["artifacts/opaque.joblib"][0]


def test_upgrade_refuses_active_source_and_cleans_failed_copy(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    with sqlite3.connect(source / "store.sqlite") as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE runs SET name='Committed WAL' WHERE run_id='historical'")
        writer.commit()
        with pytest.raises(ValueError, match="checkpoint"):
            upgrade_workspace_copy(source, tmp_path / "active")
    # sqlite's transaction context does not close its connection.
    writer.close()
    original = _bytes(source)
    monkeypatch.setattr(WorkspaceStore, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("migration failed")))
    with pytest.raises(RuntimeError, match="migration failed"):
        upgrade_workspace_copy(source, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-upgrade-*"))
    assert _bytes(source) == original


def test_upgrade_after_readonly_inspection_preserves_empty_wal_and_shm(tmp_path):
    source = _source(tmp_path / "source")
    writer = sqlite3.connect(source / "store.sqlite")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.close()
    reader = sqlite3.connect(f"{(source / 'store.sqlite').as_uri()}?mode=ro", uri=True)
    try:
        assert reader.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        reader.close()
    assert (source / "store.sqlite-wal").stat().st_size == 0
    assert (source / "store.sqlite-shm").exists()
    original = _bytes(source)
    output = tmp_path / "upgraded"
    upgrade_workspace_copy(source, output)
    assert _bytes(source) == original
    with WorkspaceStore.open_readonly(output) as store:
        assert store.get_run("historical")["name"] == "Original run"
