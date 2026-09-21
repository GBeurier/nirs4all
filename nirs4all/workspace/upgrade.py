"""Explicit copy-and-upgrade for the existing SQLite workspace format.

This upgrades metadata, preserving stored model and array bytes. It does not
convert DuckDB, legacy inline prediction arrays, or reconstruct models.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any


def _snapshot(root: Path) -> dict[str, tuple[int, int, str] | None]:
    """Bind every source file and directory without following workspace links."""
    snapshot: dict[str, tuple[int, int, str] | None] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"workspace upgrade refuses symbolic links: {relative}")
        if path.is_dir():
            snapshot[relative] = None
        elif path.is_file():
            before = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError(f"workspace changed during upgrade: {relative}")
            snapshot[relative] = (after.st_size, after.st_mtime_ns, digest)
        else:
            raise ValueError(f"workspace upgrade requires regular files: {relative}")
    return snapshot


def upgrade_workspace_copy(source: Path | str, output: Path | str) -> dict[str, Any]:
    """Copy a closed SQLite workspace and upgrade only the new copy.

    ``output`` must not exist and must be disjoint from ``source``. SQLite
    metadata schemas 2 through the current version are accepted. Nonempty WAL or
    rollback journals require the writer to close/checkpoint before this offline
    operation. Empty WAL and shared-memory coordination files may remain after
    read-only inspection; these do not contain committed database changes. The resulting workspace can subsequently be used with normal WAL
    readers and writers; the returned digest is a conversion receipt only.

    Returns:
        JSON-compatible provenance for the completed, validated output.
    """
    from nirs4all.pipeline.storage.store_schema import SCHEMA_DDL, SCHEMA_VERSION
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path or source_path in output_path.parents or output_path in source_path.parents:
        raise ValueError("workspace upgrade requires disjoint source and output paths")
    if output_path.exists():
        raise FileExistsError(f"workspace upgrade output already exists: {output_path}")
    database = source_path / "store.sqlite"
    if not source_path.is_dir() or not database.is_file():
        raise ValueError("workspace upgrade requires an existing SQLite workspace")
    before = _snapshot(source_path)
    if any((entry := before.get(f"store.sqlite{suffix}")) is not None and entry[0] > 0 for suffix in ("-wal", "-journal")):
        raise ValueError("close and checkpoint the source workspace before upgrading its copy")
    # The source is explicitly quiescent and checked again before publishing.
    # Immutable mode here avoids creating coordination files in the source.
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not 2 <= version <= SCHEMA_VERSION:
            raise ValueError(f"unsupported workspace schema {version}; expected 2 through {SCHEMA_VERSION}")
        if "prediction_arrays" in tables or not {"runs", "pipelines", "chains", "predictions", "artifacts", "logs", "projects"} <= tables:
            raise ValueError("workspace requires legacy conversion, not a metadata schema upgrade")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}-upgrade-", dir=output_path.parent))
    try:
        shutil.copytree(source_path, staging, dirs_exist_ok=True)
        if _snapshot(staging) != before:
            raise RuntimeError("workspace copy does not match the source snapshot")
        # Reuse the library's schema migrations and checks; never stamp a new
        # user_version onto an old/incomplete database.
        with WorkspaceStore(staging):
            pass
        with WorkspaceStore.open_readonly(staging) as store, closing(sqlite3.connect(":memory:")) as expected:
            assert store._conn is not None  # noqa: SLF001
            expected.executescript(SCHEMA_DDL)
            for (table,) in expected.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                expected_columns = {row[1]: tuple(row[2:]) for row in expected.execute(f'PRAGMA table_info("{table}")')}
                actual_columns = {row[1]: tuple(row[2:]) for row in store._conn.execute(f'PRAGMA table_info("{table}")')}  # noqa: SLF001
                if actual_columns != expected_columns:
                    raise RuntimeError(f"upgraded workspace schema differs from the current owner: {table}")
            if store._conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":  # noqa: SLF001
                raise RuntimeError("upgraded workspace failed SQLite integrity validation")
            if store._conn.execute("PRAGMA foreign_key_check").fetchone() is not None:  # noqa: SLF001
                raise RuntimeError("upgraded workspace contains invalid metadata references")
        if _snapshot(source_path) != before:
            raise RuntimeError("source workspace changed during upgrade")
        with (staging / "store.sqlite").open("rb") as stream:
            output_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        staging.rename(output_path)
        return {
            "schema": "nirs4all.workspace-upgrade.v1",
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_schema_version": version,
            "target_schema_version": SCHEMA_VERSION,
            "store_content_sha256": output_digest,
            "source_preserved": True,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["upgrade_workspace_copy"]
