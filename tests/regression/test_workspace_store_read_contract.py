"""Regression checks for the published native WorkspaceStore read contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

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
    assert contract["workspace_location"] == {
        "candidate_order": ["normalized_content_directory", "input_path"],
        "normalized_content_directory": {
            "workspace_subdirectory": "workspace",
            "direct_content_markers": ["runs", "exports"],
            "fallback": "workspace_subdirectory",
        },
        "selection": "first_existing_store_sqlite",
    }

    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    with patch(
        "nirs4all.pipeline.storage.workspace_store.uuid4",
        return_value=UUID("12345678-1234-5678-1234-567812345678"),
    ):
        run_id = store.begin_run(
            name="native scanner parity",
            config={"metric": "rmsecv"},
            datasets=[{"name": "corn", "samples": 42}],
        )
    store.complete_run(run_id, {"total_results": 3, "best_score": 0.12})
    store.close()
    database = workspace / contract["store"]["metadata_file"]
    before = {path.name for path in workspace.iterdir()}

    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == contract["workspace_store_schema_version"]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        assert {field["column"] for field in projection["fields"]} <= columns
        rows = connection.execute(projection["query"], (100, 0)).fetchall()

    assert len(rows) == 1
    (
        row_id,
        row_name,
        row_status,
        row_created_at,
        row_completed_at,
        row_datasets,
        row_summary,
        row_error,
    ) = rows[0]
    assert row_id == "12345678-1234-5678-1234-567812345678"
    assert row_name == "native scanner parity"
    assert row_status == "completed"
    assert row_created_at
    assert row_completed_at
    assert next(field for field in projection["fields"] if field["name"] == "created_at")["serialization"] == "iso8601"
    assert next(field for field in projection["fields"] if field["name"] == "completed_at")["serialization"] == "iso8601"
    assert json.loads(row_datasets) == [{"name": "corn", "samples": 42}]
    assert json.loads(row_summary) == {"total_results": 3, "best_score": 0.12}
    assert row_error is None

    assert {path.name for path in workspace.iterdir()} == before
    assert contract["excluded_surfaces"] == [
        "prediction_arrays",
        "parquet_sidecars",
        "artifacts",
        "workspace_mutation",
        "duckdb_workspaces",
        "schema_versions_other_than_5",
    ]
