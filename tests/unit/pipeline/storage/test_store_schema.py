"""Schema tests for the SQLite workspace store.

Validates that the DDL creates all expected tables and indexes, that
the schema creation is idempotent, and that foreign key and cascade
constraints work correctly.
"""

from __future__ import annotations

import sqlite3

import pytest

from nirs4all.pipeline.storage.store_schema import (
    INDEX_DDL,
    SCHEMA_DDL,
    TABLE_NAMES,
    VIEW_DDL,
    create_schema,
)


@pytest.fixture
def conn():
    """Create an in-memory SQLite connection."""
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()

# =========================================================================
# test_schema_creation
# =========================================================================

class TestSchemaCreation:
    """Verify that create_schema produces all expected tables."""

    def test_schema_creation(self, conn):
        """Create store from scratch, verify all tables exist."""
        create_schema(conn)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        actual_tables = sorted([row[0] for row in result])
        expected_tables = sorted(TABLE_NAMES)

        assert actual_tables == expected_tables

    def test_all_tables(self, conn):
        """All expected tables are created."""
        create_schema(conn)

        result = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()
        assert result[0] == len(TABLE_NAMES)

    def test_chain_summary_view_created(self, conn):
        """The v_chain_summary VIEW is created."""
        create_schema(conn)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        view_names = [row[0] for row in result]
        assert 'v_chain_summary' in view_names

    def test_runs_table_columns(self, conn):
        """The runs table has the expected columns."""
        create_schema(conn)
        result = conn.execute("PRAGMA table_info('runs')").fetchall()
        columns = [row[1] for row in result]
        expected = [
            "run_id", "name", "config", "datasets", "status",
            "created_at", "completed_at", "summary", "error", "project_id",
        ]
        assert columns == expected

    def test_predictions_table_columns(self, conn):
        """The predictions table has the expected columns."""
        create_schema(conn)
        result = conn.execute("PRAGMA table_info('predictions')").fetchall()
        columns = [row[1] for row in result]
        assert "prediction_id" in columns
        assert "pipeline_id" in columns
        assert "chain_id" in columns
        assert "val_score" in columns
        assert "test_score" in columns
        assert "train_score" in columns
        assert "exclusion_count" in columns
        assert "exclusion_rate" in columns

    def test_prediction_arrays_table_removed(self, conn):
        """The prediction_arrays table no longer exists (arrays moved to Parquet)."""
        create_schema(conn)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
        ).fetchall()
        assert len(result) == 0

    def test_artifacts_table_columns(self, conn):
        """The artifacts table has ref_count column."""
        create_schema(conn)
        result = conn.execute("PRAGMA table_info('artifacts')").fetchall()
        columns = [row[1] for row in result]
        assert "artifact_id" in columns
        assert "content_hash" in columns
        assert "ref_count" in columns
        assert "size_bytes" in columns

# =========================================================================
# test_schema_idempotent
# =========================================================================

class TestSchemaIdempotent:
    """Verify that create_schema can be called multiple times safely."""

    def test_schema_idempotent(self, conn):
        """Create schema twice, no errors."""
        create_schema(conn)
        # Second call should succeed without error
        create_schema(conn)

        result = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()
        assert result[0] == len(TABLE_NAMES)

    def test_data_preserved_after_recreate(self, conn):
        """Data inserted before second create_schema call is preserved."""
        create_schema(conn)
        conn.execute(
            "INSERT INTO runs (run_id, name) VALUES ('r1', 'test_run')"
        )

        # Re-create schema
        create_schema(conn)

        result = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        assert result[0] == 1

# =========================================================================
# test_foreign_keys
# =========================================================================

class TestForeignKeys:
    """Verify FK constraints work."""

    def test_foreign_keys(self, conn):
        """Inserting a pipeline with a non-existent run_id fails."""
        create_schema(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pipelines "
                "(pipeline_id, run_id, name, dataset_name) "
                "VALUES ('p1', 'nonexistent_run', 'test', 'ds1')"
            )

    def test_chain_requires_valid_pipeline(self, conn):
        """Inserting a chain with a non-existent pipeline_id fails."""
        create_schema(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chains "
                "(chain_id, pipeline_id, steps, model_step_idx, model_class) "
                "VALUES ('c1', 'nonexistent', '[]', 0, 'Model')"
            )

    def test_prediction_requires_valid_pipeline(self, conn):
        """Inserting a prediction with a non-existent pipeline_id fails."""
        create_schema(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO predictions "
                "(prediction_id, pipeline_id, dataset_name, model_name, "
                "model_class, fold_id, partition, metric, task_type) "
                "VALUES ('pr1', 'nonexistent', 'ds', 'M', 'M', 'f0', 'val', 'rmse', 'regression')"
            )

    def test_log_requires_valid_pipeline(self, conn):
        """Inserting a log with a non-existent pipeline_id fails."""
        create_schema(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO logs "
                "(log_id, pipeline_id, step_idx, event) "
                "VALUES ('l1', 'nonexistent', 0, 'start')"
            )

# =========================================================================
# test_cascade_delete
# =========================================================================

class TestFKBlocksParentDeletion:
    """Verify FK constraints block deletion of parent rows with children.

    SQLite with ``PRAGMA foreign_keys=ON`` blocks deletion of parent
    rows while children exist, raising :class:`sqlite3.IntegrityError`.
    Cascade logic is handled by :class:`WorkspaceStore` at the application
    layer.
    """

    def _setup_hierarchy(self, conn):
        """Insert a full run -> pipeline -> chain -> prediction hierarchy."""
        create_schema(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("INSERT INTO runs (run_id, name) VALUES ('r1', 'run')")
        conn.execute(
            "INSERT INTO pipelines (pipeline_id, run_id, name, dataset_name) "
            "VALUES ('p1', 'r1', 'pipe', 'ds1')"
        )
        conn.execute(
            "INSERT INTO chains "
            "(chain_id, pipeline_id, steps, model_step_idx, model_class) "
            "VALUES ('c1', 'p1', '[]', 0, 'Model')"
        )
        conn.execute(
            "INSERT INTO predictions "
            "(prediction_id, pipeline_id, chain_id, dataset_name, model_name, "
            "model_class, fold_id, partition, metric, task_type) "
            "VALUES ('pr1', 'p1', 'c1', 'ds1', 'M', 'M', 'f0', 'val', 'rmse', 'regression')"
        )
        conn.execute(
            "INSERT INTO logs (log_id, pipeline_id, step_idx, event) "
            "VALUES ('l1', 'p1', 0, 'start')"
        )

    def test_run_delete_blocked_by_pipelines(self, conn):
        """Deleting a run with existing pipelines raises IntegrityError."""
        self._setup_hierarchy(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM runs WHERE run_id = 'r1'")

    def test_pipeline_delete_blocked_by_chains(self, conn):
        """Deleting a pipeline with existing chains raises IntegrityError."""
        self._setup_hierarchy(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM pipelines WHERE pipeline_id = 'p1'")

    def test_prediction_can_be_deleted(self, conn):
        """Predictions can be deleted directly (no FK from prediction_arrays)."""
        self._setup_hierarchy(conn)
        # With prediction_arrays removed, predictions can be deleted freely
        # (only logs and chains block pipeline/run deletions)
        conn.execute("DELETE FROM predictions WHERE prediction_id = 'pr1'")
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert count == 0

    def test_chain_delete_blocked_by_predictions(self, conn):
        """Deleting a chain referenced by predictions raises IntegrityError."""
        self._setup_hierarchy(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM chains WHERE chain_id = 'c1'")

# =========================================================================
# DDL string sanity checks
# =========================================================================

class TestDDLStrings:
    """Basic sanity checks on the DDL constant strings."""

    def test_schema_ddl_contains_all_tables(self):
        """SCHEMA_DDL mentions all table names."""
        for table in TABLE_NAMES:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA_DDL

    def test_index_ddl_not_empty(self):
        """INDEX_DDL contains at least one CREATE INDEX."""
        assert "CREATE INDEX IF NOT EXISTS" in INDEX_DDL

    def test_table_names_list(self):
        """TABLE_NAMES has expected number of entries."""
        assert len(TABLE_NAMES) == 7

    def test_view_ddl_contains_chain_summary(self):
        """VIEW_DDL defines the v_chain_summary view."""
        assert "CREATE VIEW IF NOT EXISTS v_chain_summary" in VIEW_DDL
