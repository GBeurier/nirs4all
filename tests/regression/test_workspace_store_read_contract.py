"""Regression checks for the published native WorkspaceStore read contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from nirs4all.pipeline.storage import WorkspaceStore, workspace_store_read_contract
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def test_summary_contract_matches_a_fresh_workspace_store_without_mutating_it(tmp_path: Path) -> None:
    """The native summary projections stay executable against the exact SQLite schema."""
    contract = workspace_store_read_contract()
    run_projection = contract["projections"]["studio_run_summary"]
    pipeline_projection = contract["projections"]["studio_pipeline_summary"]

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
    with patch(
        "nirs4all.pipeline.storage.workspace_store.uuid4",
        return_value=UUID("87654321-4321-6789-4321-678943216789"),
    ):
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="0001_pls",
            expanded_config=[{"model": "PLSRegression"}],
            generator_choices=[],
            dataset_name="corn",
            dataset_hash="sha256:corn",
        )
    store.complete_pipeline(
        pipeline_id,
        best_val=0.12,
        best_test=0.15,
        metric="rmsecv",
        duration_ms=1234,
    )
    with patch(
        "nirs4all.pipeline.storage.workspace_store.uuid4",
        return_value=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ):
        pending_pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="0002_pending",
            expanded_config=[{"model": "PendingRegressor"}],
            generator_choices=[],
            dataset_name="corn",
            dataset_hash="sha256:corn",
        )
    store.close()
    database = workspace / contract["store"]["metadata_file"]
    before = {path.name for path in workspace.iterdir()}

    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == contract["workspace_store_schema_version"]
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        assert {field["column"] for field in run_projection["fields"]} <= run_columns
        rows = connection.execute(run_projection["query"], (100, 0)).fetchall()
        pipeline_columns = {row[1] for row in connection.execute("PRAGMA table_info(pipelines)")}
        assert {field["column"] for field in pipeline_projection["fields"]} <= pipeline_columns
        pipeline_rows = connection.execute(pipeline_projection["query"], (100, 0)).fetchall()
        pipeline_count = connection.execute(pipeline_projection["count_query"]).fetchone()[0]

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
    assert next(field for field in run_projection["fields"] if field["name"] == "created_at")["serialization"] == "iso8601"
    assert next(field for field in run_projection["fields"] if field["name"] == "completed_at")["serialization"] == "iso8601"
    assert json.loads(row_datasets) == [{"name": "corn", "samples": 42}]
    assert json.loads(row_summary) == {"total_results": 3, "best_score": 0.12}
    assert row_error is None

    assert pipeline_count == len(pipeline_rows) == 2
    assert pipeline_projection["response_constants"] == {"format": "store"}
    (
        row_pipeline_id,
        row_run_id,
        row_pipeline_name,
        row_pipeline_config_id,
        row_dataset_name,
        row_pipeline_created_at,
        row_best_val,
        row_best_test,
        row_metric,
        row_pipeline_status,
        row_duration_ms,
    ) = pipeline_rows[0]
    assert row_pipeline_id == "87654321-4321-6789-4321-678943216789"
    assert row_run_id == run_id
    assert row_pipeline_name == "0001_pls"
    assert row_pipeline_config_id == row_pipeline_id
    assert row_dataset_name == "corn"
    assert row_pipeline_created_at
    assert next(field for field in pipeline_projection["fields"] if field["name"] == "created_at") == {
        "name": "created_at",
        "column": "created_at",
        "type": "timestamp",
        "serialization": "iso8601",
        "default": "",
    }
    assert row_best_val == 0.12
    assert row_best_test == 0.15
    assert row_metric == "rmsecv"
    assert row_pipeline_status == "completed"
    assert row_duration_ms == 1234
    assert pipeline_rows[1][0] == pending_pipeline_id
    assert pipeline_rows[1][6] is None
    assert pipeline_rows[1][7] is None
    assert pipeline_rows[1][8] is None
    assert next(field for field in pipeline_projection["fields"] if field["name"] == "status")["nullable"] is True

    assert {path.name for path in workspace.iterdir()} == before
    assert contract["excluded_surfaces"] == [
        "prediction_arrays",
        "parquet_sidecars",
        "artifacts",
        "workspace_mutation",
        "duckdb_workspaces",
        "schema_versions_other_than_5",
    ]
