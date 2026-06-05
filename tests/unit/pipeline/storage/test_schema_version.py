"""Schema-version stamping tests for the SQLite workspace store.

Validates that :func:`create_schema` stamps ``PRAGMA user_version`` with
``SCHEMA_VERSION`` on a fresh database, re-stamps and migrates a legacy
(pre-versioning) database without data loss, and refuses to touch a database
whose schema version is newer than this library understands.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION, create_schema
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _user_version(conn: sqlite3.Connection) -> int:
    """Return the ``PRAGMA user_version`` of a connection."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def test_fresh_store_stamps_schema_version(tmp_path: Path) -> None:
    """A fresh WorkspaceStore stamps ``user_version`` == SCHEMA_VERSION."""
    store = WorkspaceStore(tmp_path / "workspace")
    try:
        db_path = tmp_path / "workspace" / "store.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            assert _user_version(conn) == SCHEMA_VERSION
        finally:
            conn.close()
    finally:
        store.close()


def test_legacy_db_is_migrated_and_restamped() -> None:
    """A pre-versioning DB (user_version 0) migrates and re-stamps, data intact."""
    conn = sqlite3.connect(":memory:")
    try:
        # Build an initial schema and seed a row.
        create_schema(conn)
        conn.execute("INSERT INTO runs (run_id, name) VALUES ('r1', 'legacy_run')")

        # Simulate a legacy / pre-versioning database.
        conn.execute("PRAGMA user_version = 0")
        assert _user_version(conn) == 0

        # Re-running create_schema must migrate and re-stamp without error.
        create_schema(conn)

        assert _user_version(conn) == SCHEMA_VERSION
        count = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = 'r1'").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_forward_incompatible_db_is_refused() -> None:
    """A DB stamped newer than SCHEMA_VERSION raises a clear RuntimeError."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(RuntimeError, match="newer than"):
            create_schema(conn)
    finally:
        conn.close()
