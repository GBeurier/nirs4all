from __future__ import annotations

import sqlite3

import pytest

from nirs4all.pipeline.storage.migration import MigrationReport
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.workspace.compat import build_conversion_command, inspect_workspace_format, warn_if_legacy_workspace


def test_inspect_new_or_empty_workspace(tmp_path):
    workspace = tmp_path / "workspace"

    info = inspect_workspace_format(workspace)

    assert info.format == "new-or-empty"
    assert info.conversion_required is False
    assert info.conversion_command is None


def test_inspect_duckdb_workspace_reports_conversion_command(tmp_path):
    workspace = tmp_path / "legacy"
    workspace.mkdir()
    (workspace / "store.duckdb").touch()

    info = inspect_workspace_format(workspace)

    assert info.format == "duckdb-workspace"
    assert info.conversion_required is True
    assert "nirs4all workspace convert" in (info.conversion_command or "")
    assert "--target" not in (info.conversion_command or "")


def test_inspect_sqlite_prediction_arrays_reports_legacy(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with sqlite3.connect(workspace / "store.sqlite") as conn:
        conn.execute("CREATE TABLE prediction_arrays (prediction_id TEXT PRIMARY KEY)")

    info = inspect_workspace_format(workspace)

    assert info.format == "sqlite-workspace-legacy-arrays"
    assert info.conversion_required is True
    assert info.conversion_command == build_conversion_command(workspace)


def test_inspect_legacy_artifact_reports_conversion_command(tmp_path):
    artifact = tmp_path / "legacy-model.n4a"
    artifact.write_bytes(b"legacy artifact placeholder")

    info = inspect_workspace_format(artifact)

    assert info.format == "legacy-artifact"
    assert info.conversion_required is True
    assert info.conversion_command == build_conversion_command(artifact)


def test_warn_if_legacy_workspace_includes_command(tmp_path):
    workspace = tmp_path / "legacy"
    workspace.mkdir()
    (workspace / "store.duckdb").touch()

    with pytest.warns(RuntimeWarning, match="nirs4all workspace convert"):
        info = warn_if_legacy_workspace(workspace)

    assert info.format == "duckdb-workspace"


def test_workspace_store_refuses_filesystem_legacy_workspace(tmp_path):
    workspace = tmp_path / "old-workspace"
    manifest = workspace / "runs" / "run-1" / "pipeline-1" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("run_id: run-1\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Conversion command"):
        with pytest.raises(RuntimeError, match="nirs4all workspace convert"):
            WorkspaceStore(workspace)

    assert not (workspace / "store.sqlite").exists()


def test_workspace_store_warns_with_command_before_duckdb_migration(monkeypatch, tmp_path):
    workspace = tmp_path / "legacy-duckdb"
    workspace.mkdir()
    (workspace / "store.duckdb").touch()

    def fake_migrate(path):
        assert path == workspace
        (workspace / "store.sqlite").touch()
        return MigrationReport()

    import nirs4all.pipeline.storage.migration as migration

    monkeypatch.setattr(migration, "migrate_duckdb_to_sqlite", fake_migrate)

    with pytest.warns(RuntimeWarning, match="nirs4all workspace convert"):
        store = WorkspaceStore(workspace)

    store.close()
    assert (workspace / "store.sqlite").exists()
