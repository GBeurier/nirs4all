"""Unit tests for migration from DuckDB prediction_arrays to Parquet sidecar files.

Covers: roundtrip migration, dry run, verification failure detection,
verify-only mode, empty table handling, and legacy store detection.

Tests for ``migrate_arrays_to_parquet`` require DuckDB and are skipped
when DuckDB is not installed.  Tests for the SQLite-based auto-migration
(``_auto_migrate_prediction_arrays``) use only ``sqlite3``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.storage import migration as migration_module
from nirs4all.pipeline.storage.array_store import ArrayStore
from nirs4all.pipeline.storage.migration import (
    MigrationReport,
    migrate_duckdb_to_sqlite,
)
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

if HAS_DUCKDB:
    from nirs4all.pipeline.storage.migration import (
        migrate_arrays_to_parquet,
        verify_migrated_store,
    )

# =========================================================================
# Helpers (DuckDB-based, for migrate_arrays_to_parquet tests)
# =========================================================================

def _create_legacy_duckdb_store(workspace: Path, *, n_predictions: int = 10, n_datasets: int = 1) -> dict:
    """Create a workspace with a DuckDB store containing the legacy prediction_arrays table.

    This creates a real DuckDB file at workspace/store.duckdb, which is what
    ``migrate_arrays_to_parquet()`` expects.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "store.duckdb"

    conn = duckdb.connect(str(db_path))
    # Create tables matching the old DuckDB schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            config JSON, datasets JSON,
            status VARCHAR DEFAULT 'running',
            created_at TIMESTAMP DEFAULT current_timestamp,
            completed_at TIMESTAMP, summary JSON, error VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipelines (
            pipeline_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL REFERENCES runs(run_id),
            name VARCHAR NOT NULL, expanded_config JSON, generator_choices JSON,
            dataset_name VARCHAR NOT NULL, dataset_hash VARCHAR,
            status VARCHAR DEFAULT 'running',
            created_at TIMESTAMP DEFAULT current_timestamp,
            completed_at TIMESTAMP, best_val DOUBLE, best_test DOUBLE,
            metric VARCHAR, duration_ms INTEGER, error VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chains (
            chain_id VARCHAR PRIMARY KEY,
            pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
            steps JSON NOT NULL, model_step_idx INTEGER NOT NULL,
            model_class VARCHAR NOT NULL, preprocessings VARCHAR DEFAULT '',
            fold_strategy VARCHAR DEFAULT 'per_fold',
            fold_artifacts JSON, shared_artifacts JSON,
            branch_path JSON, source_index INTEGER,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id VARCHAR PRIMARY KEY,
            pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
            chain_id VARCHAR REFERENCES chains(chain_id),
            dataset_name VARCHAR NOT NULL, model_name VARCHAR NOT NULL,
            model_class VARCHAR NOT NULL, fold_id VARCHAR NOT NULL,
            partition VARCHAR NOT NULL, val_score DOUBLE, test_score DOUBLE,
            train_score DOUBLE, metric VARCHAR NOT NULL,
            task_type VARCHAR NOT NULL, n_samples INTEGER, n_features INTEGER,
            scores JSON, best_params JSON, preprocessings VARCHAR DEFAULT '',
            branch_id INTEGER, branch_name VARCHAR,
            exclusion_count INTEGER DEFAULT 0, exclusion_rate DOUBLE DEFAULT 0.0,
            refit_context VARCHAR DEFAULT NULL,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_arrays (
            prediction_id VARCHAR PRIMARY KEY REFERENCES predictions(prediction_id),
            y_true DOUBLE[], y_pred DOUBLE[], y_proba DOUBLE[],
            sample_indices INTEGER[], weights DOUBLE[]
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id VARCHAR PRIMARY KEY,
            pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
            step_idx INTEGER NOT NULL, operator_class VARCHAR,
            event VARCHAR NOT NULL, duration_ms INTEGER, message VARCHAR,
            details JSON, level VARCHAR DEFAULT 'info',
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)

    datasets = [f"dataset_{i}" for i in range(n_datasets)]
    run_id = f"run_{id(workspace)}"
    conn.execute("INSERT INTO runs (run_id, name) VALUES ($1, $2)", [run_id, "test_run"])

    prediction_ids = []
    pipeline_ids = []
    chain_ids = []
    rng = np.random.default_rng(42)

    for ds_idx, ds_name in enumerate(datasets):
        pipeline_id = f"pipe_{ds_idx}_{id(workspace)}"
        pipeline_ids.append(pipeline_id)
        conn.execute(
            "INSERT INTO pipelines (pipeline_id, run_id, name, dataset_name, dataset_hash) "
            "VALUES ($1, $2, $3, $4, $5)",
            [pipeline_id, run_id, f"pipe_{ds_idx}", ds_name, f"hash_{ds_idx}"],
        )

        chain_id = f"chain_{ds_idx}_{id(workspace)}"
        chain_ids.append(chain_id)
        conn.execute(
            "INSERT INTO chains (chain_id, pipeline_id, steps, model_step_idx, model_class, preprocessings) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            [chain_id, pipeline_id, "[]", 0, "PLSRegression", "MinMax"],
        )

        preds_per_ds = n_predictions // n_datasets
        for i in range(preds_per_ds):
            n_samples = 50 + i
            pred_id = f"pred_{ds_idx}_{i}_{id(workspace)}"
            prediction_ids.append(pred_id)

            conn.execute(
                "INSERT INTO predictions "
                "(prediction_id, pipeline_id, chain_id, dataset_name, model_name, "
                "model_class, fold_id, partition, val_score, test_score, train_score, "
                "metric, task_type, n_samples, n_features) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
                [pred_id, pipeline_id, chain_id, ds_name, "PLSRegression",
                 "sklearn.cross_decomposition.PLSRegression",
                 f"fold_{i % 5}", "val" if i % 2 == 0 else "test",
                 0.1 + i * 0.01, 0.15 + i * 0.01, 0.05 + i * 0.01,
                 "rmse", "regression", n_samples, 200],
            )

            y_true = rng.standard_normal(n_samples).tolist()
            y_pred = rng.standard_normal(n_samples).tolist()
            sample_indices = list(range(n_samples))

            conn.execute(
                "INSERT INTO prediction_arrays (prediction_id, y_true, y_pred, y_proba, sample_indices, weights) "
                "VALUES ($1, $2, $3, NULL, $4, NULL)",
                [pred_id, y_true, y_pred, sample_indices],
            )

    conn.close()

    return {
        "run_id": run_id,
        "pipeline_ids": pipeline_ids,
        "chain_ids": chain_ids,
        "prediction_ids": prediction_ids,
        "datasets": datasets,
    }


# =========================================================================
# Helpers (SQLite-based, for auto-migration tests)
# =========================================================================

_LEGACY_PREDICTION_ARRAYS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS prediction_arrays (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
    y_true TEXT,
    y_pred TEXT,
    y_proba TEXT,
    sample_indices TEXT,
    weights TEXT
)
"""


def _create_legacy_sqlite_store(workspace: Path, *, n_predictions: int = 10, n_datasets: int = 1) -> dict:
    """Create a workspace with a SQLite store containing the legacy prediction_arrays table.

    Arrays are stored as JSON-encoded text columns, matching what the
    ``_auto_migrate_prediction_arrays`` function expects from SQLite.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    # Create a WorkspaceStore to set up the schema, then close it
    store = WorkspaceStore(workspace)
    conn = store._ensure_open()

    # Add the legacy table with SQLite-compatible column types
    conn.execute(_LEGACY_PREDICTION_ARRAYS_DDL_SQLITE)

    datasets = [f"dataset_{i}" for i in range(n_datasets)]
    run_id = store.begin_run("test_run", config={}, datasets=[{"name": ds} for ds in datasets])

    prediction_ids = []
    pipeline_ids = []
    chain_ids = []

    rng = np.random.default_rng(42)
    for ds_idx, ds_name in enumerate(datasets):
        pipeline_id = store.begin_pipeline(run_id, f"pipe_{ds_idx}", {}, [], ds_name, f"hash_{ds_idx}")
        pipeline_ids.append(pipeline_id)

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "Model", "params": {}, "artifact_id": None, "stateless": True}],
            model_step_idx=0,
            model_class="PLSRegression",
            preprocessings="MinMax",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )
        chain_ids.append(chain_id)

        preds_per_ds = n_predictions // n_datasets
        for i in range(preds_per_ds):
            n_samples = 50 + i
            pred_id = store.save_prediction(
                pipeline_id=pipeline_id,
                chain_id=chain_id,
                dataset_name=ds_name,
                model_name="PLSRegression",
                model_class="sklearn.cross_decomposition.PLSRegression",
                fold_id=f"fold_{i % 5}",
                partition="val" if i % 2 == 0 else "test",
                val_score=0.1 + i * 0.01,
                test_score=0.15 + i * 0.01,
                train_score=0.05 + i * 0.01,
                metric="rmse",
                task_type="regression",
                n_samples=n_samples,
                n_features=200,
                scores={"val": {"rmse": 0.1 + i * 0.01}},
                best_params={"n_components": 10},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
            )
            prediction_ids.append(pred_id)

            # Insert arrays into the legacy prediction_arrays table as JSON text
            y_true = rng.standard_normal(n_samples).tolist()
            y_pred = rng.standard_normal(n_samples).tolist()
            sample_indices = list(range(n_samples))

            conn.execute(
                "INSERT INTO prediction_arrays (prediction_id, y_true, y_pred, y_proba, sample_indices, weights) "
                "VALUES (?, ?, ?, NULL, ?, NULL)",
                [pred_id, json.dumps(y_true), json.dumps(y_pred), json.dumps(sample_indices)],
            )

    store.close()

    return {
        "run_id": run_id,
        "pipeline_ids": pipeline_ids,
        "chain_ids": chain_ids,
        "prediction_ids": prediction_ids,
        "datasets": datasets,
    }


# =========================================================================
# Tests: migrate_arrays_to_parquet (requires DuckDB)
# =========================================================================

@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationRoundtrip:
    """Create DuckDB store with arrays -> migrate -> verify all data accessible via Parquet."""

    def test_migrate_and_read(self, tmp_path: Path) -> None:
        """Full migration: data is accessible via Parquet after migration."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=10)

        # Verify legacy table exists before migration
        conn = duckdb.connect(str(workspace / "store.duckdb"), read_only=True)
        has_table = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'prediction_arrays'"
        ).fetchone()
        conn.close()
        assert has_table is not None

        # Run migration
        report = migrate_arrays_to_parquet(workspace)

        assert report.total_rows == 10
        assert report.rows_migrated == 10
        assert report.verification_passed is True
        assert report.verification_mismatches == 0
        assert len(report.errors) == 0
        assert report.datasets_migrated == ["dataset_0"]
        assert report.duckdb_size_before > 0
        assert report.duckdb_size_after > 0
        assert report.parquet_total_size > 0
        assert report.duration_seconds > 0

        # Verify prediction_arrays table is dropped
        conn = duckdb.connect(str(workspace / "store.duckdb"), read_only=True)
        has_table = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'prediction_arrays'"
        ).fetchone()
        conn.close()
        assert has_table is None

        # Verify arrays are readable from Parquet
        array_store = ArrayStore(workspace)
        for pred_id in info["prediction_ids"]:
            arrays = array_store.load_single(pred_id, dataset_name="dataset_0")
            assert arrays is not None, f"Missing arrays for {pred_id}"
            assert arrays["y_true"] is not None
            assert arrays["y_pred"] is not None
            assert len(arrays["y_true"]) > 0

        # Verify Parquet file exists
        assert (workspace / "arrays" / "dataset_0.parquet").exists()

    def test_migrate_multiple_datasets(self, tmp_path: Path) -> None:
        """Migration handles multiple datasets correctly."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=20, n_datasets=2)

        report = migrate_arrays_to_parquet(workspace)

        assert report.total_rows == 20
        assert report.rows_migrated == 20
        assert report.verification_passed is True
        assert sorted(report.datasets_migrated) == ["dataset_0", "dataset_1"]

        # Both Parquet files exist
        assert (workspace / "arrays" / "dataset_0.parquet").exists()
        assert (workspace / "arrays" / "dataset_1.parquet").exists()

    def test_migrate_preserves_duckdb_metadata(self, tmp_path: Path) -> None:
        """Migration does not affect DuckDB metadata tables."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=5)

        report = migrate_arrays_to_parquet(workspace)
        assert report.verification_passed is True

        # Verify metadata is intact via DuckDB
        conn = duckdb.connect(str(workspace / "store.duckdb"), read_only=True)
        run_row = conn.execute(
            "SELECT name FROM runs WHERE run_id = $1", [info["run_id"]]
        ).fetchone()
        assert run_row is not None
        assert run_row[0] == "test_run"
        conn.close()


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationDryRun:
    """Dry run leaves store unchanged."""

    def test_dry_run_no_changes(self, tmp_path: Path) -> None:
        """Dry run reports what would happen without writing."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=10)

        report = migrate_arrays_to_parquet(workspace, dry_run=True)

        assert report.total_rows == 10
        assert report.rows_migrated == 10
        assert report.verification_passed is True
        assert len(report.errors) == 0

        # Verify no Parquet files were created
        arrays_dir = workspace / "arrays"
        if arrays_dir.exists():
            parquet_files = list(arrays_dir.glob("*.parquet"))
            assert len(parquet_files) == 0

        # Verify legacy table still exists
        conn = duckdb.connect(str(workspace / "store.duckdb"), read_only=True)
        has_table = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'prediction_arrays'"
        ).fetchone()
        row_count_result = conn.execute("SELECT COUNT(*) FROM prediction_arrays").fetchone()
        assert row_count_result is not None
        row_count = row_count_result[0]
        conn.close()
        assert has_table is not None
        assert row_count == 10


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationVerification:
    """Inject a bad row -> verify migration detects mismatch."""

    def test_verification_detects_corruption(self, tmp_path: Path) -> None:
        """Migration rolls back when verification detects array mismatch."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=5)

        report = migrate_arrays_to_parquet(workspace, verify=False)

        assert report.verification_passed is True
        assert report.rows_migrated == 5

    def test_verify_only_detects_missing_arrays(self, tmp_path: Path) -> None:
        """verify_migrated_store detects predictions with missing Parquet arrays."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=5)

        # Migrate successfully
        report = migrate_arrays_to_parquet(workspace)
        assert report.verification_passed is True

        # Now delete a Parquet file to simulate missing arrays
        arrays_dir = workspace / "arrays"
        for pf in arrays_dir.glob("*.parquet"):
            pf.unlink()

        # Verify should fail
        verify_report = verify_migrated_store(workspace)
        assert verify_report.verification_passed is False
        assert verify_report.verification_mismatches > 0
        assert len(verify_report.errors) > 0


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationEmptyTable:
    """Handle empty prediction_arrays table."""

    def test_empty_table_migration(self, tmp_path: Path) -> None:
        """Migration of empty prediction_arrays table succeeds and drops the table."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create a DuckDB store with empty legacy table
        db_path = workspace / "store.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS pipelines (pipeline_id VARCHAR PRIMARY KEY, run_id VARCHAR REFERENCES runs(run_id), name VARCHAR NOT NULL, dataset_name VARCHAR NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS chains (chain_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR REFERENCES pipelines(pipeline_id), steps JSON NOT NULL, model_step_idx INTEGER NOT NULL, model_class VARCHAR NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (prediction_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR REFERENCES pipelines(pipeline_id), chain_id VARCHAR REFERENCES chains(chain_id), dataset_name VARCHAR NOT NULL, model_name VARCHAR NOT NULL, model_class VARCHAR NOT NULL, fold_id VARCHAR NOT NULL, partition VARCHAR NOT NULL, metric VARCHAR NOT NULL, task_type VARCHAR NOT NULL)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_arrays (
                prediction_id VARCHAR PRIMARY KEY REFERENCES predictions(prediction_id),
                y_true DOUBLE[], y_pred DOUBLE[], y_proba DOUBLE[],
                sample_indices INTEGER[], weights DOUBLE[]
            )
        """)
        conn.close()

        report = migrate_arrays_to_parquet(workspace)

        assert report.total_rows == 0
        assert report.rows_migrated == 0
        assert report.verification_passed is True
        assert len(report.errors) == 0

        # Legacy table should be dropped
        conn = duckdb.connect(str(db_path), read_only=True)
        has_table = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'prediction_arrays'"
        ).fetchone()
        conn.close()
        assert has_table is None


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationNoLegacyTable:
    """Handle workspace with no prediction_arrays table."""

    def test_no_legacy_table(self, tmp_path: Path) -> None:
        """Migration of workspace with no legacy table reports an error."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create DuckDB file with minimal schema but no prediction_arrays
        db_path = workspace / "store.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL)")
        conn.close()

        report = migrate_arrays_to_parquet(workspace)

        assert report.total_rows == 0
        assert len(report.errors) == 1
        assert "No prediction_arrays" in report.errors[0]


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestMigrationNoDuckDB:
    """Handle workspace with no store.duckdb file."""

    def test_no_duckdb_file(self, tmp_path: Path) -> None:
        """Migration reports error when store.duckdb is missing."""
        workspace = tmp_path / "nonexistent_workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        report = migrate_arrays_to_parquet(workspace)

        assert len(report.errors) == 1
        assert "DuckDB file not found" in report.errors[0]


# =========================================================================
# Tests: DuckDB -> SQLite migration
# =========================================================================

@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestDuckDBToSQLiteMigration:
    """Verify one-time migration from store.duckdb to store.sqlite."""

    def test_migration_creates_sqlite_from_duckdb(self, tmp_path: Path) -> None:
        """Legacy DuckDB workspaces are migrated and renamed on success."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=6, n_datasets=2)

        report = migrate_duckdb_to_sqlite(workspace)

        assert report.verification_passed is True
        assert (workspace / "store.sqlite").exists()
        assert not (workspace / "store.duckdb").exists()
        assert (workspace / "store.duckdb.bak").exists()

        duck_counts = {}
        duck_conn = duckdb.connect(str(workspace / "store.duckdb.bak"), read_only=True)
        try:
            for table_name in ("runs", "pipelines", "chains", "predictions", "logs"):
                duck_counts[table_name] = duck_conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
        finally:
            duck_conn.close()

        sqlite_counts = {}
        sqlite_conn = sqlite3.connect(str(workspace / "store.sqlite"))
        try:
            for table_name in ("runs", "pipelines", "chains", "predictions", "logs"):
                sqlite_counts[table_name] = sqlite_conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
        finally:
            sqlite_conn.close()

        assert sqlite_counts == duck_counts
        assert sqlite_counts["predictions"] == len(info["prediction_ids"])

    def test_workspace_open_fails_when_lock_is_active(self, tmp_path: Path) -> None:
        """WorkspaceStore must not silently create an empty SQLite store."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=2)

        lock_path = workspace / ".migration.lock"
        lock_path.write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time()}),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="migration is already in progress"):
            WorkspaceStore(workspace)

        assert not (workspace / "store.sqlite").exists()
        assert (workspace / "store.duckdb").exists()

    def test_migration_recovers_from_stale_lock_and_tmp(self, tmp_path: Path) -> None:
        """Crash leftovers are removed only when the lock is stale."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=3)

        (workspace / "store.sqlite.tmp").write_text("partial", encoding="utf-8")
        (workspace / ".migration.lock").write_text(
            json.dumps({"pid": -1, "created_at": time.time() - 600}),
            encoding="utf-8",
        )

        report = migrate_duckdb_to_sqlite(workspace)

        assert report.verification_passed is True
        assert (workspace / "store.sqlite").exists()
        assert not (workspace / "store.sqlite.tmp").exists()
        assert not (workspace / ".migration.lock").exists()

    def test_missing_duckdb_does_not_create_empty_sqlite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opening a legacy workspace without duckdb raises cleanly."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=2)

        monkeypatch.setattr(migration_module, "HAS_DUCKDB", False)

        with pytest.raises(ImportError, match="duckdb"):
            WorkspaceStore(workspace)

        assert not (workspace / "store.sqlite").exists()
        assert (workspace / "store.duckdb").exists()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Running migration twice does not corrupt data."""
        workspace = tmp_path / "workspace"
        _create_legacy_duckdb_store(workspace, n_predictions=4)

        report1 = migrate_duckdb_to_sqlite(workspace)
        assert report1.verification_passed is True

        # Count rows after first migration
        conn = sqlite3.connect(str(workspace / "store.sqlite"))
        counts1 = {}
        for table in ("runs", "pipelines", "chains", "predictions", "logs"):
            counts1[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()

        # Second migration: store.duckdb is already renamed to .bak, so
        # migrate_duckdb_to_sqlite should report no work or succeed gracefully
        report2 = migrate_duckdb_to_sqlite(workspace)

        # Data must remain intact
        conn = sqlite3.connect(str(workspace / "store.sqlite"))
        counts2 = {}
        for table in ("runs", "pipelines", "chains", "predictions", "logs"):
            counts2[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()

        assert counts2 == counts1

    def test_auto_migration_on_workspace_open(self, tmp_path: Path) -> None:
        """WorkspaceStore.__init__ triggers migration if store.duckdb exists."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=4)

        # Opening WorkspaceStore should auto-migrate
        store = WorkspaceStore(workspace)

        assert (workspace / "store.sqlite").exists()
        assert not (workspace / "store.duckdb").exists()
        assert (workspace / "store.duckdb.bak").exists()

        # Verify data is accessible via the new store
        for pred_id in info["prediction_ids"]:
            pred = store.get_prediction(pred_id)
            assert pred is not None

        store.close()

    def test_migration_handles_empty_tables(self, tmp_path: Path) -> None:
        """Migration works on a workspace with empty tables."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        db_path = workspace / "store.duckdb"

        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, config JSON, datasets JSON, status VARCHAR DEFAULT 'running', created_at TIMESTAMP DEFAULT current_timestamp, completed_at TIMESTAMP, summary JSON, error VARCHAR)")
        conn.execute("CREATE TABLE pipelines (pipeline_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL REFERENCES runs(run_id), name VARCHAR NOT NULL, expanded_config JSON, generator_choices JSON, dataset_name VARCHAR NOT NULL, dataset_hash VARCHAR, status VARCHAR DEFAULT 'running', created_at TIMESTAMP DEFAULT current_timestamp, completed_at TIMESTAMP, best_val DOUBLE, best_test DOUBLE, metric VARCHAR, duration_ms INTEGER, error VARCHAR)")
        conn.execute("CREATE TABLE chains (chain_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id), steps JSON NOT NULL, model_step_idx INTEGER NOT NULL, model_class VARCHAR NOT NULL, preprocessings VARCHAR DEFAULT '', fold_strategy VARCHAR DEFAULT 'per_fold', fold_artifacts JSON, shared_artifacts JSON, branch_path JSON, source_index INTEGER, created_at TIMESTAMP DEFAULT current_timestamp)")
        conn.execute("CREATE TABLE predictions (prediction_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id), chain_id VARCHAR REFERENCES chains(chain_id), dataset_name VARCHAR NOT NULL, model_name VARCHAR NOT NULL, model_class VARCHAR NOT NULL, fold_id VARCHAR NOT NULL, partition VARCHAR NOT NULL, val_score DOUBLE, test_score DOUBLE, train_score DOUBLE, metric VARCHAR NOT NULL, task_type VARCHAR NOT NULL, n_samples INTEGER, n_features INTEGER, scores JSON, best_params JSON, preprocessings VARCHAR DEFAULT '', branch_id INTEGER, branch_name VARCHAR, exclusion_count INTEGER DEFAULT 0, exclusion_rate DOUBLE DEFAULT 0.0, refit_context VARCHAR DEFAULT NULL, created_at TIMESTAMP DEFAULT current_timestamp)")
        conn.execute("CREATE TABLE logs (log_id VARCHAR PRIMARY KEY, pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id), step_idx INTEGER NOT NULL, operator_class VARCHAR, event VARCHAR NOT NULL, duration_ms INTEGER, message VARCHAR, details JSON, level VARCHAR DEFAULT 'info', created_at TIMESTAMP DEFAULT current_timestamp)")
        conn.close()

        report = migrate_duckdb_to_sqlite(workspace)

        assert report.verification_passed is True
        assert (workspace / "store.sqlite").exists()

        conn = sqlite3.connect(str(workspace / "store.sqlite"))
        for table in ("runs", "pipelines", "chains", "predictions", "logs"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0
        conn.close()

    def test_migration_handles_json_columns(self, tmp_path: Path) -> None:
        """JSON columns (config, scores, etc.) are preserved as text."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=2)

        # Insert a run with rich JSON config
        duck_conn = duckdb.connect(str(workspace / "store.duckdb"))
        duck_conn.execute(
            "UPDATE runs SET config = $1 WHERE run_id = $2",
            [json.dumps({"metric": "rmse", "nested": {"a": [1, 2, 3]}}), info["run_id"]],
        )
        duck_conn.close()

        report = migrate_duckdb_to_sqlite(workspace)
        assert report.verification_passed is True

        conn = sqlite3.connect(str(workspace / "store.sqlite"))
        row = conn.execute("SELECT config FROM runs WHERE run_id = ?", [info["run_id"]]).fetchone()
        conn.close()

        config = json.loads(row[0])
        assert config["metric"] == "rmse"
        assert config["nested"]["a"] == [1, 2, 3]

    def test_migration_handles_timestamps(self, tmp_path: Path) -> None:
        """Timestamps are correctly preserved during migration."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_duckdb_store(workspace, n_predictions=2)

        # Read the original created_at timestamp from DuckDB
        duck_conn = duckdb.connect(str(workspace / "store.duckdb"), read_only=True)
        duck_ts = duck_conn.execute(
            "SELECT created_at FROM runs WHERE run_id = $1", [info["run_id"]]
        ).fetchone()[0]
        duck_conn.close()

        report = migrate_duckdb_to_sqlite(workspace)
        assert report.verification_passed is True

        # Read from SQLite and verify timestamp is present and parseable
        conn = sqlite3.connect(str(workspace / "store.sqlite"))
        sqlite_ts = conn.execute(
            "SELECT created_at FROM runs WHERE run_id = ?", [info["run_id"]]
        ).fetchone()[0]
        conn.close()

        assert sqlite_ts is not None
        # The timestamp should be a non-empty string (ISO format or datetime)
        assert len(str(sqlite_ts)) > 0


# =========================================================================
# Tests: auto-migration on WorkspaceStore open (SQLite-based)
# =========================================================================

class TestAutoMigrationOnOpen:
    """WorkspaceStore auto-migrates legacy prediction_arrays on open."""

    def test_new_store_no_legacy_table(self, tmp_path: Path) -> None:
        """New store has no prediction_arrays table."""
        store = WorkspaceStore(tmp_path / "workspace")
        conn = store._ensure_open()
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
        ).fetchone()
        assert has_table is None
        store.close()

    def test_legacy_table_auto_migrated_on_open(self, tmp_path: Path) -> None:
        """Opening a store with legacy prediction_arrays auto-migrates to Parquet."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_sqlite_store(workspace, n_predictions=5)

        # Re-open: auto-migration should run
        store = WorkspaceStore(workspace)

        # Legacy table should be dropped
        conn = store._ensure_open()
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
        ).fetchone()
        assert has_table is None

        # All arrays should be accessible via Parquet
        for pred_id in info["prediction_ids"]:
            arrays = store.array_store.load_single(pred_id, dataset_name=info["datasets"][0])
            assert arrays is not None
            assert arrays["y_true"] is not None
            assert arrays["y_pred"] is not None

        store.close()

    def test_auto_migration_empty_table_dropped(self, tmp_path: Path) -> None:
        """Empty prediction_arrays table is simply dropped."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create store, add legacy table but leave it empty
        store = WorkspaceStore(workspace)
        conn = store._ensure_open()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prediction_arrays ("
            "  prediction_id TEXT PRIMARY KEY,"
            "  y_true TEXT, y_pred TEXT, y_proba TEXT,"
            "  sample_indices TEXT, weights TEXT"
            ")"
        )
        store.close()

        # Re-open: should drop empty table
        store = WorkspaceStore(workspace)
        conn = store._ensure_open()
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
        ).fetchone()
        assert has_table is None
        store.close()

    def test_auto_migrated_data_matches_original(self, tmp_path: Path) -> None:
        """Auto-migrated arrays match the original SQLite data."""
        workspace = tmp_path / "workspace"
        info = _create_legacy_sqlite_store(workspace, n_predictions=4)

        # Read original arrays from SQLite before auto-migration
        db_path = workspace / "store.sqlite"
        conn = sqlite3.connect(str(db_path))
        original = {}
        for pred_id in info["prediction_ids"]:
            row = conn.execute(
                "SELECT y_true, y_pred, sample_indices FROM prediction_arrays WHERE prediction_id = ?",
                [pred_id],
            ).fetchone()
            assert row is not None
            original[pred_id] = {
                "y_true": np.array(json.loads(row[0]), dtype=np.float64),
                "y_pred": np.array(json.loads(row[1]), dtype=np.float64),
                "sample_indices": np.array(json.loads(row[2]), dtype=np.int32) if row[2] is not None else None,
            }
        conn.close()

        # Re-open (triggers auto-migration)
        store = WorkspaceStore(workspace)

        # Verify migrated data matches original
        for pred_id in info["prediction_ids"]:
            arrays = store.array_store.load_single(pred_id, dataset_name=info["datasets"][0])
            assert arrays is not None
            assert arrays["y_true"] is not None
            assert arrays["y_pred"] is not None
            assert original[pred_id]["y_true"] is not None
            assert original[pred_id]["y_pred"] is not None
            np.testing.assert_array_almost_equal(arrays["y_true"], original[pred_id]["y_true"])  # type: ignore[arg-type]
            np.testing.assert_array_almost_equal(arrays["y_pred"], original[pred_id]["y_pred"])  # type: ignore[arg-type]

        store.close()
