"""Immutable catalogue must never invoke writable workspace startup."""

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
    assert sorted(path.name for path in tmp_path.iterdir()) == entries


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_active_journal_is_refused_without_ignoring_committed_writes(tmp_path, suffix):
    with WorkspaceStore(tmp_path):
        pass
    (tmp_path / f"store.sqlite{suffix}").write_bytes(b"active")
    with pytest.raises(RuntimeError, match="active SQLite journal"):
        read_model_catalogue(tmp_path)


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
