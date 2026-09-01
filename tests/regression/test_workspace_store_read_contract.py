"""Regression checks for the published native WorkspaceStore read contract."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from nirs4all.pipeline.storage import (
    WorkspaceStore,
    studio_run_detail_http_contract,
    studio_run_detail_http_inputs_v1,
    workspace_store_read_contract,
    workspace_store_results_summary_contract,
)
from nirs4all.pipeline.storage.store_schema import SCHEMA_VERSION


def _create_run_detail_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "run-detail-workspace"
    store = WorkspaceStore(workspace)
    store.close()
    del store
    database = workspace / "store.sqlite"

    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO runs
               (run_id, name, config, datasets, status, created_at,
                completed_at, summary, error, project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "run-detail-001",
                "Detail oracle",
                '{"metric":"rmsecv","drop":null,"nested":{"nan":NaN,"values":[Infinity,-Infinity,1.5]}}',
                '[{"name":"corn","n_samples":42}]',
                "completed",
                "2026-09-01T10:00:00+02:00",
                "2026-09-01T10:05:00+02:00",
                '{"total_results":2,"best_score":Infinity}',
                None,
                None,
            ),
        )
        connection.executemany(
            """INSERT INTO pipelines
               (pipeline_id, run_id, name, expanded_config, original_template,
                generator_choices, dataset_name, dataset_hash, status,
                created_at, completed_at, best_val, best_test, metric,
                duration_ms, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    "11111111-1111-1111-1111-111111111111",
                    "run-detail-001",
                    "0001_pls",
                    '[{"model":{"class":"PLSRegression","score":NaN}}]',
                    '{"name":"PLS template"}',
                    '[{"n_components":8}]',
                    "corn",
                    "sha256:corn",
                    "completed",
                    "2026-09-01T10:02:00+02:00",
                    "2026-09-01T10:03:00+02:00",
                    float("inf"),
                    float("-inf"),
                    "rmsecv",
                    321,
                    None,
                ),
                (
                    "22222222-2222-2222-2222-222222222222",
                    "run-detail-001",
                    "0002_pending",
                    None,
                    None,
                    "[]",
                    "corn",
                    "sha256:corn",
                    "running",
                    "2026-09-01T10:02:00+02:00",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )
        connection.execute(
            """INSERT INTO chains
               (chain_id, pipeline_id, steps, model_step_idx, model_class,
                final_test_score)
               VALUES ('chain-refit', ?, '[]', 1, 'PLSRegression', 0.11)""",
            ("11111111-1111-1111-1111-111111111111",),
        )
        connection.executemany(
            """INSERT INTO logs
               (log_id, pipeline_id, step_idx, operator_class, event,
                duration_ms, message, details, level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    "log-warning",
                    "11111111-1111-1111-1111-111111111111",
                    0,
                    "SNV",
                    "warning",
                    None,
                    "warning",
                    '{"value":NaN}',
                    "warning",
                    "2026-09-01T10:02:10+02:00",
                ),
                (
                    "log-end",
                    "11111111-1111-1111-1111-111111111111",
                    0,
                    "SNV",
                    "end",
                    321,
                    None,
                    None,
                    "info",
                    "2026-09-01T10:02:11+02:00",
                ),
            ],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")

    return workspace, database


def test_summary_contract_matches_a_fresh_workspace_store_without_mutating_it(tmp_path: Path) -> None:
    """The native summary projections stay executable against the exact SQLite schema."""
    contract = workspace_store_read_contract()
    run_projection = contract["projections"]["studio_run_summary"]
    run_discovery = contract["projections"]["studio_run_discovery_query_v1"]
    pipeline_projection = contract["projections"]["studio_pipeline_summary"]
    chain_projection = contract["projections"]["studio_chain_ranked_v1"]

    assert contract["workspace_store_schema_version"] == SCHEMA_VERSION == 5
    assert contract["store"] == {
        "metadata_file": "store.sqlite",
        "open_mode": "sqlite_immutable_read_only",
        "compatibility": "exact_schema_version",
        "path_support": "local_filesystem_only",
        "unsupported_paths": ["windows_unc", "windows_device_namespace"],
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
    assert run_discovery == {
        "source_projection": "studio_run_summary",
        "http": {
            "method": "GET",
            "path_suffix": "/runs",
            "query_mode": "explicit_allowlist",
            "query_absent_allowed": True,
            "unknown_parameters": "reject",
            "duplicate_parameters": "reject",
            "parameter_order": "any",
            "parameters": [
                {
                    "name": "source",
                    "type": "string",
                    "enum": ["unified", "manifests", "parquet"],
                    "default": "unified",
                },
                {
                    "name": "refresh",
                    "type": "string",
                    "enum": ["true", "false"],
                    "default": "false",
                },
            ],
        },
        "store_semantics": {
            "source": "accepted_for_store_parity_but_does_not_switch_away_from_workspace_store",
            "refresh": "every_native_request_is_an_uncached_immutable_read",
            "limit": 500,
            "offset": 0,
            "ordering": "studio_run_summary",
            "fallback_after_native_selection": "none",
        },
        "response": {
            "workspace_id": "requested_workspace_id",
            "runs": "studio_run_summary_rows",
            "total": "returned_row_count",
        },
        "incompatible_store_http_status": 409,
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
        chain_columns = {row[1] for row in connection.execute("PRAGMA table_info(chains)")}
        pipeline_columns = {row[1] for row in connection.execute("PRAGMA table_info(pipelines)")}
        contract_columns = {field["column"] for field in chain_projection["fields"]}
        assert contract_columns - {"run_id", "name"} <= chain_columns
        assert {"run_id", "name"} <= pipeline_columns
        assert connection.execute(chain_projection["ascending_query"], ("corn", "rmsecv", 5, 0)).fetchall() == []
        assert connection.execute(chain_projection["descending_query"], ("corn", "rmsecv", 5, 0)).fetchall() == []
        assert connection.execute(chain_projection["count_query"], ("corn", "rmsecv")).fetchone()[0] == 0

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
        "studio_results_summary_policy",
        "metric_direction_inference",
        "dataset_links",
        "workspace_mutation",
        "duckdb_workspaces",
        "schema_versions_other_than_5",
    ]


def test_run_detail_projection_matches_golden_without_mutating_store(tmp_path: Path) -> None:
    """The owner oracle is immutable, deterministic, finite JSON, and schema-exact."""
    contract = workspace_store_read_contract()
    projection = contract["projections"]["studio_run_detail_v1"]
    workspace, database = _create_run_detail_workspace(tmp_path)
    expected = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "workspace_store_v5_run_detail.response.json").read_text(encoding="utf-8")
    )

    before = {path.name for path in workspace.iterdir()}
    actual = WorkspaceStore.get_studio_run_detail_v1(workspace, "run-detail-001")
    assert actual == expected
    assert WorkspaceStore.get_studio_run_detail_v1(workspace, "missing-run") is None
    assert json.dumps(actual, allow_nan=False, separators=(",", ":"), sort_keys=True) == json.dumps(
        expected,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert {path.name for path in workspace.iterdir()} == before
    assert not any((workspace / f"store.sqlite{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))

    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert connection.execute(projection["queries"]["run"], ("run-detail-001",)).fetchone() is not None
        assert len(connection.execute(projection["queries"]["pipelines"], ("run-detail-001",)).fetchall()) == 2
        assert connection.execute(projection["queries"]["has_refit"], ("run-detail-001",)).fetchone()[0] == 1
        assert len(connection.execute(projection["queries"]["log_summary"], ("run-detail-001",)).fetchall()) == 2

    assert projection["ordering"] == {
        "pipelines": "created_at_desc_then_pipeline_id_asc",
        "log_summary": "pipeline_created_at_asc_then_pipeline_id_asc",
    }
    assert projection["json_policy"]["non_finite_numbers"] == "replace_with_null_recursively"
    assert projection["native_read_preconditions"] == {
        "open_mode": "sqlite_immutable_read_only",
        "pragma_user_version": 5,
        "active_sidecars": ["store.sqlite-wal", "store.sqlite-shm", "store.sqlite-journal"],
        "active_sidecar_policy": "reject_if_any_exists",
        "database_change_during_read": "reject",
        "writes_or_cache": "forbidden",
    }
    assert projection["cutover_scope"] == "store_owned_source_projection_not_complete_http_response"
    assert projection["studio_composition_required"]["route_selection"] == "forbidden_until_all_required_composition_has_exact_oracle_parity"
    assert projection["legacy_filesystem_manifest_branch"] == "not_covered"
    assert projection["fallback_after_native_selection"] == "none"


def test_run_detail_http_contract_assigns_every_composition_owner_and_forbids_cutover() -> None:
    """The HTTP contract exposes owner inputs without claiming external parity."""
    contract = studio_run_detail_http_contract()

    assert contract["request"] == {
        "method": "GET",
        "path_suffix": "/runs/{run_id}",
        "query_string": "absent",
    }
    assert contract["owner_oracle"] == {
        "callable": "nirs4all.pipeline.storage.studio_run_detail_http_inputs_v1",
        "signature": "(workspace_path: str | Path, run_id: str) -> dict[str, Any] | None",
        "inputs": ["workspace_path", "run_id"],
        "native_abi": "none_python_callable_only",
        "bounded_cpython_subprocess": "supported",
        "framework_requirements": {
            "fastapi": "none",
            "pipeline_runner_construction": "forbidden",
        },
        "scope": "store_v5_owner_inputs_only",
        "open_mode": "composed_immutable_reads_guarded_by_before_after_database_stamp",
        "writes_or_cache": "forbidden",
        "not_found": "null",
    }
    assert contract["dependencies"]["workspace_store_read"] == {
        "schema_id": "nirs4all.workspace-store-read.v1",
        "schema_version": 1,
        "projection": "studio_run_detail_v1",
    }
    assert contract["dependencies"]["splitter_config"] == {
        "callable": "nirs4all.pipeline.analysis.splitter_config.extract_splitter_config",
        "input": "pipeline.expanded_config",
        "write_boundary": "WorkspaceStore.begin_pipeline",
        "persisted_source": "pipelines.expanded_config",
        "store_v5_splitter_column": "absent_by_design",
        "historical_compatibility": "derive_or_null_from_existing_expanded_config",
        "schema_migration": "none_required_for_owner_projection",
        "consumer_expanded_config_access": "forbidden",
        "selection": "first_recognized_splitter_step",
        "output_fields": [
            "splitter_class",
            "reference",
            "n_splits",
            "shuffle",
            "random_state",
            "test_size",
            "group_by",
        ],
    }
    assert contract["dependencies"]["pipeline_runtime"] == {
        "owner_method": "WorkspaceStore.get_studio_run_detail_runtime_v1",
        "source_table": "pipelines",
        "required_columns": ["pipeline_id", "run_id", "created_at"],
        "optional_columns": [
            "engine",
            "engine_requested",
            "engine_diagnostics",
            "runtime_manifest",
            "fallback_policy",
            "native_result_refs",
        ],
        "optional_column_selection": "fixed_allowlist_present_column_or_sql_null_alias",
        "absent_optional_column": "null_with_absent_in_store_v5_provenance",
        "present_text_columns": ["engine", "engine_requested"],
        "present_json_shapes": {
            "engine_diagnostics": "array_or_null",
            "runtime_manifest": "object_or_null",
            "fallback_policy": "object_or_null",
            "native_result_refs": "array_or_null",
        },
        "malformed_or_wrong_shape": "reject",
        "non_finite_numbers": "replace_with_null_recursively",
        "ordering": "pipeline_created_at_desc_then_pipeline_id_asc",
    }
    assert contract["owner_output"]["pipeline_splitters"] == {
        "ordering": "run_detail.pipelines_order",
        "entry_fields": ["pipeline_id", "splitter"],
        "splitter": "splitter_config_output_or_null",
        "materialization": "derived_by_owner_oracle_before_consumer_boundary",
        "materialization_time": "immutable_owner_read",
        "consumer_reimplementation": "forbidden",
        "consumer_expanded_config_access": "forbidden",
    }
    assert contract["owner_output"]["pipeline_runtime"]["source"] == "pipeline_runtime_dependency"
    assert contract["owner_output"]["results"]["mapping"] == {
        "id": "pipeline.pipeline_id",
        "run_id": "pipeline.run_id",
        "dataset": "pipeline.dataset_name",
        "pipeline_config": "pipeline.name",
        "pipeline_config_id": "pipeline.pipeline_id",
        "created_at": "pipeline.created_at_or_empty_string",
        "best_score": "pipeline.best_val",
        "best_test_score": "pipeline.best_test",
        "metric": "pipeline.metric",
        "status": "pipeline.status",
        "duration_ms": "pipeline.duration_ms",
        "format": "store",
    }
    store_branch = contract["http_composition"]["store_branch"]
    assert store_branch["dataset_composition"]["owner"] == "studio_linked_dataset_configuration"
    assert store_branch["runtime_composition"]["owner"] == "studio_http_adapter"
    assert store_branch["config_splitter_inference"]["owner"] == "studio_http_adapter"
    assert contract["http_composition"]["legacy_manifest_branch"] == {
        "owner": "workspace_manifest_scanner",
        "status": "not_covered",
        "required_contract": "studio_workspace_manifest_run_detail_v1",
        "must_not_be_reconstructed_from_store_v5": True,
    }
    assert contract["cutover"] == {
        "route_selection": "forbidden",
        "blocked_on": [
            "studio_dataset_link_composition_v1",
            "studio_runtime_field_composition_v1",
            "studio_ui_splitter_strategy_vocabulary_v1",
            "studio_workspace_manifest_run_detail_v1_or_preselection_proof",
        ],
        "store_owner_inputs_complete": True,
        "complete_http_response_proven": False,
        "legacy_manifest_branch_proven": False,
        "fallback_after_native_selection": "none",
        "incompatible_store_http_status": 409,
    }


def test_run_detail_http_owner_inputs_match_golden_without_mutating_store(tmp_path: Path) -> None:
    """Splitter and optional runtime owner data form one deterministic input."""
    workspace, database = _create_run_detail_workspace(tmp_path)
    expanded_config = json.dumps(
        [
            {
                "class": "sklearn.model_selection.KFold",
                "params": {"n_splits": 5, "shuffle": True, "random_state": 17},
            },
            {"model": {"class": "PLSRegression", "score": None}},
        ],
        separators=(",", ":"),
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE pipelines SET expanded_config = ? WHERE pipeline_id = ?",
            (expanded_config, "11111111-1111-1111-1111-111111111111"),
        )
        connection.commit()
        connection.execute("PRAGMA journal_mode=DELETE")

    absent_runtime = WorkspaceStore.get_studio_run_detail_runtime_v1(workspace, "run-detail-001")
    assert absent_runtime is not None
    assert set(absent_runtime["runtime_column_provenance"].values()) == {"absent_in_store_v5"}
    assert all(
        value is None
        for row in absent_runtime["pipeline_runtime"]
        for key, value in row.items()
        if key != "pipeline_id"
    )

    with sqlite3.connect(database) as connection:
        for column in ("engine", "engine_requested", "engine_diagnostics", "runtime_manifest", "fallback_policy", "native_result_refs"):
            connection.execute(f"ALTER TABLE pipelines ADD COLUMN {column} TEXT")
        connection.execute(
            """UPDATE pipelines SET
               engine = ?, engine_requested = ?, engine_diagnostics = ?,
               runtime_manifest = ?, fallback_policy = ?, native_result_refs = ?
               WHERE pipeline_id = ?""",
            (
                "legacy",
                "dag-ml",
                '[{"cause":"unsupported_shape","score":NaN}]',
                '{"engine":"legacy","duration":Infinity}',
                '{"engine_requested":"dag-ml","allow_fallback":true}',
                '["native://result/1"]',
                "11111111-1111-1111-1111-111111111111",
            ),
        )
        connection.commit()
        connection.execute("PRAGMA journal_mode=DELETE")
    expected = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "workspace_store_v5_run_detail_http_inputs.response.json").read_text(encoding="utf-8")
    )
    before_files = {path.name for path in workspace.iterdir()}
    before_database = database.read_bytes()

    actual = studio_run_detail_http_inputs_v1(workspace, "run-detail-001")

    assert actual == expected
    assert studio_run_detail_http_inputs_v1(workspace, "missing-run") is None
    assert json.dumps(actual, allow_nan=False, separators=(",", ":"), sort_keys=True) == json.dumps(
        expected,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert {path.name for path in workspace.iterdir()} == before_files
    assert database.read_bytes() == before_database
    assert not any((workspace / f"store.sqlite{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_run_detail_http_owner_boundary_runs_in_fresh_cpython_without_studio_runtime(tmp_path: Path) -> None:
    """The sidecar-callable owner boundary needs no FastAPI or runner instance."""
    workspace = tmp_path / "fresh-owner-boundary"
    expanded_config = [
        {
            "class": "sklearn.model_selection.KFold",
            "params": {"n_splits": 4, "shuffle": True, "random_state": 23},
        },
        {"model": {"class": "PLSRegression"}},
    ]
    with WorkspaceStore(workspace) as store:
        run_id = store.begin_run("fresh owner", {"metric": "rmsecv"}, [{"name": "corn"}])
        pipeline_id = store.begin_pipeline(
            run_id,
            "0001_pls",
            expanded_config,
            [],
            "corn",
            "sha256:corn",
        )
        store.complete_pipeline(pipeline_id, 0.12, 0.15, "rmsecv", 123)
        store.complete_run(run_id, {"total_results": 1})

    database = workspace / "store.sqlite"
    before_database = database.read_bytes()
    package_root = Path(__file__).parents[2]
    dependency_roots = sorted(
        {
            str(path)
            for entry in sys.path
            if entry
            and (path := Path(entry).resolve()).is_dir()
            and path.name in {"site-packages", "dist-packages"}
        }
    )
    assert dependency_roots, "the host must expose at least one explicit dependency root"
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path[1:1] = json.loads(sys.argv[2])
from nirs4all.pipeline.storage import studio_run_detail_http_inputs_v1
from nirs4all.pipeline.runner import PipelineRunner

runner_constructions = []
def forbid_runner_construction(*args, **kwargs):
    runner_constructions.append([args, kwargs])
    raise AssertionError("owner boundary constructed PipelineRunner")

PipelineRunner.__init__ = forbid_runner_construction

payload = studio_run_detail_http_inputs_v1(Path(sys.argv[3]), sys.argv[4])
fastapi_modules = sorted(
    name for name in sys.modules
    if name == "fastapi" or name.startswith("fastapi.")
)
print(json.dumps({
    "payload": payload,
    "fastapi_modules": fastapi_modules,
    "runner_constructions": len(runner_constructions),
}, allow_nan=False))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_root), json.dumps(dependency_roots), str(workspace), run_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"fresh CPython owner boundary exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    fresh = json.loads(completed.stdout)
    payload = fresh["payload"]

    assert fresh["fastapi_modules"] == []
    assert fresh["runner_constructions"] == 0
    assert list(payload) == [
        "source_branch",
        "run_detail",
        "pipeline_splitters",
        "pipeline_runtime",
        "runtime_column_provenance",
        "results",
        "results_count",
    ]
    assert payload["source_branch"] == "store_v5"
    assert payload["results_count"] == 1
    assert payload["pipeline_splitters"] == [
        {
            "pipeline_id": pipeline_id,
            "splitter": {
                "splitter_class": "KFold",
                "reference": "sklearn.model_selection.KFold",
                "n_splits": 4,
                "shuffle": True,
                "random_state": 23,
                "test_size": None,
                "group_by": None,
            },
        }
    ]
    assert set(payload["runtime_column_provenance"].values()) == {"absent_in_store_v5"}
    assert database.read_bytes() == before_database
    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in connection.execute("PRAGMA table_info(pipelines)")}
    assert "splitter" not in columns
    assert "splitter_config" not in columns
    assert not any((workspace / f"store.sqlite{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_run_detail_projection_fails_closed_on_journal_schema_and_json(tmp_path: Path) -> None:
    """Unsafe SQLite state and malformed Store-v5 rows never produce partial output."""
    workspace, database = _create_run_detail_workspace(tmp_path)
    journal = Path(f"{database}-wal")
    journal.write_bytes(b"active writer sentinel")
    with pytest.raises(RuntimeError, match="active SQLite journal"):
        WorkspaceStore.get_studio_run_detail_v1(workspace, "run-detail-001")
    with pytest.raises(RuntimeError, match="active SQLite journal"):
        studio_run_detail_http_inputs_v1(workspace, "run-detail-001")
    journal.unlink()

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=4")
        connection.execute("PRAGMA journal_mode=DELETE")
    with pytest.raises(RuntimeError, match="requires WorkspaceStore schema 5, got 4"):
        WorkspaceStore.get_studio_run_detail_v1(workspace, "run-detail-001")

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=5")
        connection.execute("UPDATE runs SET config = '[' WHERE run_id = 'run-detail-001'")
        connection.commit()
        connection.execute("PRAGMA journal_mode=DELETE")
    with pytest.raises(ValueError, match="runs.config contains malformed JSON"):
        WorkspaceStore.get_studio_run_detail_v1(workspace, "run-detail-001")

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE runs SET config = '{}' WHERE run_id = 'run-detail-001'")
        connection.execute("ALTER TABLE pipelines ADD COLUMN engine_diagnostics TEXT")
        connection.execute(
            "UPDATE pipelines SET engine_diagnostics = '{}' WHERE pipeline_id = ?",
            ("11111111-1111-1111-1111-111111111111",),
        )
        connection.commit()
        connection.execute("PRAGMA journal_mode=DELETE")
    with pytest.raises(ValueError, match="engine_diagnostics must decode to a JSON array"):
        WorkspaceStore.get_studio_run_detail_runtime_v1(workspace, "run-detail-001")
    with pytest.raises(ValueError, match="engine_diagnostics must decode to a JSON array"):
        studio_run_detail_http_inputs_v1(workspace, "run-detail-001")


def test_ranked_chain_contract_is_filtered_deterministic_and_null_last(tmp_path: Path) -> None:
    """The native primitive has stable paging and excludes chains without predictions."""
    contract = workspace_store_read_contract()
    projection = contract["projections"]["studio_chain_ranked_v1"]
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    store.close()

    database = workspace / contract["store"]["metadata_file"]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO runs (run_id, name) VALUES (?, ?)",
            ("run-ranked", "ranked chains"),
        )
        connection.execute(
            """INSERT INTO pipelines
               (pipeline_id, run_id, name, dataset_name)
               VALUES (?, ?, ?, ?)""",
            ("pipeline-pending", "run-ranked", "pending pipeline", "corn"),
        )
        chain_rows = [
            ("chain-a", 0.5, "rmse"),
            ("chain-b", 0.1, "rmse"),
            ("chain-c", None, "rmse"),
            ("chain-no-prediction", 0.01, "rmse"),
            ("chain-z", 0.5, "rmse"),
            ("chain-other-metric", 0.99, "r2"),
        ]
        connection.executemany(
            """INSERT INTO chains
               (chain_id, pipeline_id, steps, model_step_idx, model_class,
                dataset_name, metric, cv_val_score, cv_fold_count, cv_scores)
               VALUES (?, 'pipeline-pending', '[]', 1, 'PLSRegression',
                       'corn', ?, ?, 2, '{"rmse": 0.1}')""",
            [(chain_id, metric, score) for chain_id, score, metric in chain_rows],
        )
        prediction_chain_ids = [
            "chain-a",
            "chain-b",
            "chain-c",
            "chain-z",
            "chain-other-metric",
        ]
        connection.executemany(
            """INSERT INTO predictions
               (prediction_id, pipeline_id, chain_id, dataset_name, model_name,
                model_class, fold_id, partition, metric, task_type)
               VALUES (?, 'pipeline-pending', ?, 'corn', 'PLS',
                       'PLSRegression', 'fold_0', 'val', 'rmse', 'regression')""",
            [(f"prediction-{chain_id}", chain_id) for chain_id in prediction_chain_ids],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        ascending = connection.execute(
            projection["ascending_query"],
            ("corn", "rmse", 100, 0),
        ).fetchall()
        descending = connection.execute(
            projection["descending_query"],
            ("corn", "rmse", 100, 0),
        ).fetchall()
        second_page = connection.execute(
            projection["ascending_query"],
            ("corn", "rmse", 2, 2),
        ).fetchall()
        total = connection.execute(
            projection["count_query"],
            ("corn", "rmse"),
        ).fetchone()[0]

    assert [row[0] for row in ascending] == ["chain-b", "chain-a", "chain-z", "chain-c"]
    assert [row[0] for row in descending] == ["chain-a", "chain-z", "chain-b", "chain-c"]
    assert [row[0] for row in second_page] == ["chain-z", "chain-c"]
    assert total == 4
    assert ascending[0][2:6] == ("run-ranked", "pending pipeline", "corn", "rmse")
    assert projection["parameters"][2] == {
        "name": "direction",
        "type": "string",
        "enum": ["asc", "desc"],
        "required": True,
    }
    assert projection["excluded_computed_fields"] == [
        "variant_params",
        "synthetic_refit",
        "cv_source_chain_id",
        "is_refit_only",
    ]
    assert projection["fields"][-1] == {
        "name": "best_params",
        "column": "best_params",
        "type": "json_object",
        "nullable": True,
        "empty_object": "null",
    }


def test_results_summary_contract_source_is_complete_paged_and_read_only(tmp_path: Path) -> None:
    """The summary policy can page every eligible row with pipeline metadata."""
    contract = workspace_store_results_summary_contract()
    source = contract["source_projection"]
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    store.close()
    database = workspace / "store.sqlite"

    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO runs (run_id, name) VALUES ('run-summary', 'summary')")
        connection.execute(
            """INSERT INTO pipelines
               (pipeline_id, run_id, name, expanded_config, generator_choices,
                dataset_name)
               VALUES ('pipeline-summary', 'run-summary', '0001_pls',
                       '[{"model":{"params":{"n_components":8}}}]', '[]',
                       'corn')"""
        )
        connection.executemany(
            """INSERT INTO chains
               (chain_id, pipeline_id, steps, model_step_idx, model_class,
                model_name, dataset_name, metric, task_type, cv_val_score,
                cv_fold_count, cv_scores, final_test_score, final_scores,
                best_params)
               VALUES (?, 'pipeline-summary', '[]', 1, 'PLSRegression',
                       'PLS', 'corn', 'rmsecv', 'regression', ?, ?, ?, ?, ?, ?)""",
            [
                ("chain-inserted-first", 0.3, 3, '{"val":{"rmsecv":0.3}}', 0.4, None, '{"n_components":6}'),
                ("chain-inserted-second", 0.2, 3, None, None, None, None),
                ("chain-without-prediction", 0.1, 3, None, None, None, None),
            ],
        )
        connection.executemany(
            """INSERT INTO predictions
               (prediction_id, pipeline_id, chain_id, dataset_name, model_name,
                model_class, fold_id, partition, metric, task_type)
               VALUES (?, 'pipeline-summary', ?, 'corn', 'PLS',
                       'PLSRegression', 'fold_0', 'val', 'rmsecv', 'regression')""",
            [
                ("prediction-first", "chain-inserted-first"),
                ("prediction-second", "chain-inserted-second"),
            ],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    before = {path.name for path in workspace.iterdir()}
    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        first_page = connection.execute(source["page_query"], (1, 0)).fetchall()
        second_page = connection.execute(source["page_query"], (1, 1)).fetchall()
        empty_page = connection.execute(source["page_query"], (1, 2)).fetchall()
        count = connection.execute(source["count_query"]).fetchone()[0]

    assert count == 2
    assert [first_page[0][0], second_page[0][0]] == ["chain-inserted-first", "chain-inserted-second"]
    assert empty_page == []
    assert first_page[0][1:8] == (
        "pipeline-summary",
        "run-summary",
        "0001_pls",
        '[{"model":{"params":{"n_components":8}}}]',
        1,
        "corn",
        "rmsecv",
    )
    assert first_page[0][12:17] == (0.3, None, None, 3, '{"val":{"rmsecv":0.3}}')
    assert first_page[0][17] == 0.4
    assert first_page[0][23] == '{"n_components":6}'
    assert source["parameters"] == [
        {"name": "limit", "type": "integer", "minimum": 1, "maximum": 500, "default": 500},
        {"name": "offset", "type": "integer", "minimum": 0, "default": 0},
    ]
    assert {path.name for path in workspace.iterdir()} == before


def test_results_summary_contract_freezes_one_metric_and_selection_policy() -> None:
    """Top-CV and best-final ranking share one explicit metric policy."""
    contract = workspace_store_results_summary_contract()

    assert contract["dependencies"]["workspace_store_read"] == {
        "schema_id": "nirs4all.workspace-store-read.v1",
        "schema_version": 1,
        "projection": "studio_chain_ranked_v1",
    }
    assert contract["request"] == {
        "surface": "studio_results_summary",
        "method": "GET",
        "path_suffix": "/results/summary",
        "query_string": "absent",
        "top_n": 5,
        "supported_top_n": [5],
    }
    assert contract["metric_direction"] == {
        "normalization": "trim_then_lowercase",
        "lower_is_better": ["rmse", "rmsecv", "rmsep", "mae", "mse", "mape", "bias", "sep"],
        "all_other_metrics": "higher_is_better",
        "applies_to": ["top_cv_ranking", "top_cv_fallback", "best_final_comparison"],
        "ascending_query_when_lower_is_better": True,
        "ranked_projection_nulls": "last",
    }
    assert contract["selection"]["append_order"] == [
        "top_cv",
        "all_refit_only_in_source_order",
        "best_final_if_not_already_selected",
    ]
    assert contract["selection"]["best_final_comparison"] == "strict_summary_comparison_so_first_source_row_wins_ties"
    assert contract["normalization"]["source_order"] == "chain_id_ascending"
    assert contract["synthetic_refit"]["mark_refit_only_before_synthesis"] is True
    assert contract["synthetic_refit"]["assignments"] == {
        "final_test_score": "finite_cv_test_score_or_null",
        "final_train_score": "finite_cv_train_score_or_null",
        "final_scores": "nonempty_cv_scores_else_partition_metric_object_from_finite_cv_val_test_train_scores",
        "synthetic_refit": True,
    }
    assert contract["serialization"]["schema_v5_constants"] == {"cv_source_chain_id": None}
    assert contract["serialization"]["variant_params"]["fallback"] == "merge_model_step_params_from_expanded_config_with_best_params_best_params_wins"
    assert contract["excluded_surfaces"] == [
        "dataset_link_augmentation",
        "prediction_arrays",
        "parquet_sidecars",
        "artifacts",
        "workspace_mutation",
        "duckdb_workspaces",
        "schema_versions_other_than_5",
        "top_n_values_other_than_5",
    ]
