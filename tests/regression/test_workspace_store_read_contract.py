"""Regression checks for the published native WorkspaceStore read contract."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nirs4all.pipeline.storage import WorkspaceStore, workspace_store_read_contract
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def test_run_summary_contract_matches_a_fresh_workspace_store_without_mutating_it(tmp_path: Path) -> None:
    """The native run projection stays executable against the exact SQLite schema."""
    contract = workspace_store_read_contract()
    projection = contract["projections"]["studio_run_summary"]

    assert contract["workspace_store_schema_version"] == SCHEMA_VERSION == 5
    assert contract["store"] == {
        "metadata_file": "store.sqlite",
        "open_mode": "sqlite_immutable_read_only",
        "compatibility": "exact_schema_version",
        "writer_lock_required": False,
        "must_not_create_wal_or_shm": True,
    }

    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    store.close()
    database = workspace / contract["store"]["metadata_file"]
    before = {path.name for path in workspace.iterdir()}

    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == contract["workspace_store_schema_version"]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        assert {field["column"] for field in projection["fields"]} <= columns
        assert connection.execute(projection["query"], (100, 0)).fetchall() == []

    assert {path.name for path in workspace.iterdir()} == before
    assert contract["excluded_surfaces"] == [
        "prediction_arrays",
        "parquet_sidecars",
        "artifacts",
        "workspace_mutation",
        "duckdb_workspaces",
        "schema_versions_other_than_5",
    ]
