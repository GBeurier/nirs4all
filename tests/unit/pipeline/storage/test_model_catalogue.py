"""Transactional catalogue must never invoke writable workspace startup."""

import hashlib
import sqlite3

import pytest

from nirs4all.pipeline.storage.model_catalogue import read_model_catalogue
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def test_empty_catalogue_does_not_mutate_workspace(tmp_path, monkeypatch):
    with WorkspaceStore(tmp_path):
        pass
    database = tmp_path / "store.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest(), database.stat().st_mtime_ns
    entries = sorted(path.name for path in tmp_path.iterdir())
    monkeypatch.setattr(WorkspaceStore, "__init__", lambda *a, **k: pytest.fail("writable store opened"))
    assert read_model_catalogue(tmp_path) == []
    assert (hashlib.sha256(database.read_bytes()).hexdigest(), database.stat().st_mtime_ns) == before
    assert sorted(path.name for path in tmp_path.iterdir() if path.name not in {"store.sqlite-wal", "store.sqlite-shm"}) == entries


def test_active_writer_does_not_block_catalogue_or_hide_committed_writes(tmp_path):
    with WorkspaceStore(tmp_path):
        pass
    writer = sqlite3.connect(tmp_path / "store.sqlite")
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("SELECT COUNT(*) FROM chains").fetchone()
        assert (tmp_path / "store.sqlite-wal").exists()
        assert read_model_catalogue(tmp_path) == []
        # A committed schema change exists only in WAL. An immutable reader
        # would silently accept the old main-file schema instead of refusing it.
        writer.execute("PRAGMA user_version = 999")
        writer.commit()
        with pytest.raises(RuntimeError, match="schema"):
            read_model_catalogue(tmp_path)
    finally:
        writer.close()


def test_schema_and_bounds_are_not_silently_migrated(tmp_path):
    with sqlite3.connect(tmp_path / "store.sqlite") as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError, match="schema"):
        read_model_catalogue(tmp_path)
    with pytest.raises(ValueError, match="max_models"):
        read_model_catalogue(tmp_path, max_models=True)
    with pytest.raises(FileNotFoundError):
        read_model_catalogue(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
