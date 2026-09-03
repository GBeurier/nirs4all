"""Runtime boundary tests for workspaces requiring explicit conversion."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from nirs4all.api.run import run
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.workspace.compat import ConversionRequired, inspect_workspace_format


def _snapshot(root: Path) -> tuple[dict[str, tuple[int, str]], set[str]]:
    files = {
        path.name: (
            path.stat().st_ino,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.iterdir()
        if path.is_file()
    }
    return files, {path.name for path in root.iterdir()}


def test_strict_package_has_no_runtime_migration_module() -> None:
    assert importlib.util.find_spec("nirs4all.pipeline.storage.migration") is None


def test_duckdb_open_requires_tools_and_preserves_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "store.duckdb").write_bytes(b"legacy-duckdb-placeholder")
    before = _snapshot(workspace)

    info = inspect_workspace_format(workspace)
    assert info.format == "duckdb-workspace"
    assert info.conversion_required is True
    assert info.conversion_command is not None
    assert info.conversion_command.startswith("nirs4all-tools workspace convert ")

    with pytest.raises(ConversionRequired) as caught:
        WorkspaceStore(workspace)
    with pytest.raises(ConversionRequired):
        Predictions(workspace / "store.duckdb")

    assert caught.value.unsupported_capability == "workspace_conversion_required"
    assert _snapshot(workspace) == before

    with pytest.raises(ConversionRequired):
        run([], {}, engine="legacy", workspace_path=workspace)

    # The legacy runner may initialize its ordinary log directory before it
    # opens the workspace. It must still leave the DuckDB source byte-identical
    # and must not create a converted store or backup.
    assert _snapshot(workspace)[0]["store.duckdb"] == before[0]["store.duckdb"]
    assert not (workspace / "store.sqlite").exists()
    assert not (workspace / "store.duckdb.bak").exists()


def test_sqlite_legacy_arrays_refusal_is_immutable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "store.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE prediction_arrays (prediction_id TEXT PRIMARY KEY, y_true TEXT)")
    before = _snapshot(workspace)

    info = inspect_workspace_format(workspace)
    assert info.format == "sqlite-workspace-legacy-arrays"
    assert info.conversion_required is True

    with pytest.raises(ConversionRequired):
        WorkspaceStore(workspace)

    assert _snapshot(workspace) == before
    assert not (workspace / "store.sqlite-wal").exists()
    assert not (workspace / "store.sqlite-shm").exists()
