"""DuckDB schema definitions for the workspace store.

Defines the table schema used by :class:`WorkspaceStore`:
``runs``, ``pipelines``, ``chains``, ``predictions``,
``artifacts``, ``logs``, and ``projects``.

Dense prediction arrays (y_true, y_pred, etc.) are stored in Parquet
sidecar files managed by :class:`ArrayStore`, not in DuckDB.

The schema uses ``IF NOT EXISTS`` for all DDL statements, making
:func:`create_schema` safe to call repeatedly (idempotent).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# =========================================================================
# Refit context constants
# =========================================================================

REFIT_CONTEXT_STANDALONE: str = "standalone"
"""Refit context for a standalone refit (single model, no stacking)."""

REFIT_CONTEXT_STACKING: str = "stacking"
"""Refit context for a stacking-context refit (base model inside a stack)."""

# =========================================================================
# Table DDL
# =========================================================================

SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    config JSON,
    datasets JSON,
    status VARCHAR DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    summary JSON,
    error VARCHAR
);

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(run_id),
    name VARCHAR NOT NULL,
    expanded_config JSON,
    generator_choices JSON,
    dataset_name VARCHAR NOT NULL,
    dataset_hash VARCHAR,
    status VARCHAR DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    best_val DOUBLE,
    best_test DOUBLE,
    metric VARCHAR,
    duration_ms INTEGER,
    error VARCHAR
);

CREATE TABLE IF NOT EXISTS chains (
    chain_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
    steps JSON NOT NULL,
    model_step_idx INTEGER NOT NULL,
    model_class VARCHAR NOT NULL,
    preprocessings VARCHAR DEFAULT '',
    fold_strategy VARCHAR DEFAULT 'per_fold',
    fold_artifacts JSON,
    shared_artifacts JSON,
    branch_path JSON,
    source_index INTEGER,
    model_name VARCHAR,
    metric VARCHAR,
    task_type VARCHAR,
    best_params JSON,
    dataset_name VARCHAR,
    cv_val_score DOUBLE,
    cv_test_score DOUBLE,
    cv_train_score DOUBLE,
    cv_fold_count INTEGER DEFAULT 0,
    cv_scores JSON,
    final_test_score DOUBLE,
    final_train_score DOUBLE,
    final_scores JSON,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
    chain_id VARCHAR REFERENCES chains(chain_id),
    dataset_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    model_class VARCHAR NOT NULL,
    fold_id VARCHAR NOT NULL,
    partition VARCHAR NOT NULL,
    val_score DOUBLE,
    test_score DOUBLE,
    train_score DOUBLE,
    metric VARCHAR NOT NULL,
    task_type VARCHAR NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    scores JSON,
    best_params JSON,
    preprocessings VARCHAR DEFAULT '',
    branch_id INTEGER,
    branch_name VARCHAR,
    exclusion_count INTEGER DEFAULT 0,
    exclusion_rate DOUBLE DEFAULT 0.0,
    refit_context VARCHAR DEFAULT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    artifact_path VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    operator_class VARCHAR,
    artifact_type VARCHAR,
    format VARCHAR DEFAULT 'joblib',
    size_bytes BIGINT,
    ref_count INTEGER DEFAULT 1,
    chain_path_hash VARCHAR,
    input_data_hash VARCHAR,
    dataset_hash VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS logs (
    log_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
    step_idx INTEGER NOT NULL,
    operator_class VARCHAR,
    event VARCHAR NOT NULL,
    duration_ms INTEGER,
    message VARCHAR,
    details JSON,
    level VARCHAR DEFAULT 'info',
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description VARCHAR DEFAULT '',
    color VARCHAR DEFAULT '#14b8a6',
    created_at TIMESTAMP DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp
);
"""

# =========================================================================
# View DDL
# =========================================================================

VIEW_DDL: str = """
CREATE VIEW IF NOT EXISTS v_chain_summary AS
SELECT
    c.chain_id,
    c.pipeline_id,
    c.model_class,
    c.model_step_idx,
    c.model_name,
    c.preprocessings,
    c.branch_path,
    c.source_index,
    c.metric,
    c.task_type,
    c.best_params,
    c.dataset_name,
    c.cv_val_score,
    c.cv_test_score,
    c.cv_train_score,
    c.cv_fold_count,
    c.cv_scores,
    c.final_test_score,
    c.final_train_score,
    c.final_scores,
    pl.run_id,
    pl.status AS pipeline_status
FROM chains c
JOIN pipelines pl ON c.pipeline_id = pl.pipeline_id;
"""

# =========================================================================
# Index DDL
# =========================================================================

INDEX_DDL: str = """
CREATE INDEX IF NOT EXISTS idx_pipelines_run_id ON pipelines(run_id);
CREATE INDEX IF NOT EXISTS idx_pipelines_dataset ON pipelines(dataset_name);
CREATE INDEX IF NOT EXISTS idx_chains_pipeline_id ON chains(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_predictions_pipeline_id ON predictions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_predictions_chain_id ON predictions(chain_id);
CREATE INDEX IF NOT EXISTS idx_predictions_dataset ON predictions(dataset_name);
CREATE INDEX IF NOT EXISTS idx_predictions_val_score ON predictions(val_score);
CREATE INDEX IF NOT EXISTS idx_predictions_partition ON predictions(partition);
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_natural_key_v2 ON predictions(pipeline_id, chain_id, fold_id, partition, model_name, branch_id);
CREATE INDEX IF NOT EXISTS idx_logs_pipeline_id ON logs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_content_hash ON artifacts(content_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_cache_key ON artifacts(chain_path_hash, input_data_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_dataset_hash ON artifacts(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
"""

# =========================================================================
# Table names (ordered by dependency)
# =========================================================================

TABLE_NAMES: list[str] = [
    "runs",
    "pipelines",
    "chains",
    "predictions",
    "artifacts",
    "logs",
    "projects",
]

def _auto_migrate_prediction_arrays(conn: duckdb.DuckDBPyConnection, workspace_path: Path) -> None:
    """Auto-migrate legacy ``prediction_arrays`` table to Parquet sidecar files.

    Called during schema migration when a workspace still has the legacy
    DuckDB table.  Streams all rows to per-dataset Parquet files via
    :class:`ArrayStore`, then drops the table and vacuums.

    Args:
        conn: An open DuckDB connection.
        workspace_path: Workspace root directory for the ArrayStore.
    """
    import numpy as np

    from nirs4all.pipeline.storage.array_store import ArrayStore

    has_table = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'prediction_arrays' AND table_type = 'BASE TABLE'"
    ).fetchone()
    if has_table is None:
        return

    row = conn.execute("SELECT COUNT(*) FROM prediction_arrays").fetchone()
    total = row[0] if row else 0
    if total == 0:
        logger.info("Empty prediction_arrays table — dropping.")
        conn.execute("DROP TABLE prediction_arrays")
        return

    logger.info("Auto-migrating %d rows from prediction_arrays to Parquet sidecar files.", total)
    array_store = ArrayStore(workspace_path)

    # Get distinct datasets with arrays
    dataset_rows = conn.execute(
        "SELECT DISTINCT p.dataset_name "
        "FROM predictions p "
        "INNER JOIN prediction_arrays pa ON p.prediction_id = pa.prediction_id "
        "ORDER BY p.dataset_name"
    ).fetchall()

    batch_size = 10_000
    for (dataset_name,) in dataset_rows:
        offset = 0
        while True:
            rows = conn.execute(
                "SELECT pa.prediction_id, pa.y_true, pa.y_pred, pa.y_proba, "
                "       pa.sample_indices, pa.weights, "
                "       p.model_name, p.fold_id, p.partition, p.metric, "
                "       p.val_score, p.task_type "
                "FROM prediction_arrays pa "
                "INNER JOIN predictions p ON pa.prediction_id = p.prediction_id "
                "WHERE p.dataset_name = $1 "
                "ORDER BY pa.prediction_id "
                "LIMIT $2 OFFSET $3",
                [dataset_name, batch_size, offset],
            ).fetchall()

            if not rows:
                break

            records = []
            for row in rows:
                (prediction_id, y_true, y_pred, y_proba,
                 sample_indices, weights,
                 model_name, fold_id, partition, metric,
                 val_score, task_type) = row

                records.append({
                    "prediction_id": prediction_id,
                    "dataset_name": dataset_name,
                    "model_name": model_name or "",
                    "fold_id": fold_id or "",
                    "partition": partition or "",
                    "metric": metric or "",
                    "val_score": val_score,
                    "task_type": task_type or "",
                    "y_true": np.array(y_true, dtype=np.float64) if y_true is not None else None,
                    "y_pred": np.array(y_pred, dtype=np.float64) if y_pred is not None else None,
                    "y_proba": np.array(y_proba, dtype=np.float64) if y_proba is not None else None,
                    "sample_indices": np.array(sample_indices, dtype=np.int32) if sample_indices is not None else None,
                    "weights": np.array(weights, dtype=np.float64) if weights is not None else None,
                })

            array_store.save_batch(records)
            offset += batch_size

    conn.execute("DROP TABLE prediction_arrays")
    logger.info("Auto-migration complete. Dropped prediction_arrays table.")

def _backfill_chain_summaries(conn: duckdb.DuckDBPyConnection) -> None:
    """Backfill chain summary columns from existing prediction data.

    Called once when migrating an older database.  Uses SQL aggregation
    to compute CV averages and extract final/refit scores.

    Args:
        conn: An open DuckDB connection.
    """
    # Check whether there are any predictions to backfill from
    has_predictions = conn.execute(
        "SELECT 1 FROM predictions LIMIT 1"
    ).fetchone()
    if has_predictions is None:
        return

    # Backfill dataset_name from pipelines for all chains
    conn.execute("""
        UPDATE chains SET dataset_name = pl.dataset_name
        FROM pipelines pl
        WHERE chains.pipeline_id = pl.pipeline_id
          AND chains.dataset_name IS NULL
    """)

    # Backfill CV averages from cross-validation predictions
    conn.execute("""
        UPDATE chains SET
            model_name = COALESCE(chains.model_name, sub.model_name),
            metric = COALESCE(chains.metric, sub.metric),
            task_type = COALESCE(chains.task_type, sub.task_type),
            cv_val_score = sub.avg_val,
            cv_test_score = sub.avg_test,
            cv_train_score = sub.avg_train,
            cv_fold_count = sub.fold_count
        FROM (
            SELECT chain_id,
                FIRST(model_name) AS model_name,
                FIRST(metric) AS metric,
                FIRST(task_type) AS task_type,
                AVG(val_score) AS avg_val,
                AVG(test_score) AS avg_test,
                AVG(train_score) AS avg_train,
                COUNT(DISTINCT fold_id) AS fold_count
            FROM predictions
            WHERE refit_context IS NULL AND chain_id IS NOT NULL
            GROUP BY chain_id
        ) sub
        WHERE chains.chain_id = sub.chain_id
    """)

    # Backfill best_params (take first non-null per chain)
    conn.execute("""
        UPDATE chains SET best_params = sub.best_params
        FROM (
            SELECT chain_id, FIRST(best_params) AS best_params
            FROM predictions
            WHERE chain_id IS NOT NULL AND best_params IS NOT NULL
              AND best_params != '{}'
            GROUP BY chain_id
        ) sub
        WHERE chains.chain_id = sub.chain_id
          AND (chains.best_params IS NULL)
    """)

    # Backfill final scores from refit predictions
    conn.execute("""
        UPDATE chains SET
            final_test_score = sub.test_score,
            final_train_score = sub.train_score,
            final_scores = sub.scores
        FROM (
            SELECT chain_id, test_score, train_score, scores
            FROM predictions
            WHERE refit_context IS NOT NULL AND fold_id = 'final'
              AND partition = 'test' AND chain_id IS NOT NULL
        ) sub
        WHERE chains.chain_id = sub.chain_id
    """)

    # Backfill cv_scores (averaged multi-metric JSON) via Python
    # DuckDB JSON aggregation is limited, so we do this row by row
    chain_ids = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT chain_id FROM chains WHERE chain_id IS NOT NULL"
        ).fetchall()
    ]
    import json
    for cid in chain_ids:
        rows = conn.execute(
            "SELECT partition, scores FROM predictions "
            "WHERE chain_id = $1 AND refit_context IS NULL "
            "AND partition IN ('val', 'test')",
            [cid],
        ).fetchall()
        if not rows:
            continue
        partition_scores: dict[str, dict[str, list[float]]] = {}
        for partition, scores_raw in rows:
            if not scores_raw:
                continue
            scores = json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
            if not isinstance(scores, dict):
                continue
            inner = scores.get(partition, scores)
            if not isinstance(inner, dict):
                continue
            if partition not in partition_scores:
                partition_scores[partition] = {}
            for metric_name, val in inner.items():
                if isinstance(val, (int, float)):
                    partition_scores[partition].setdefault(metric_name, []).append(float(val))
        averaged: dict[str, dict[str, float]] = {}
        for part, metrics in partition_scores.items():
            averaged[part] = {m: round(sum(vs) / len(vs), 6) for m, vs in metrics.items() if vs}
        if averaged:
            conn.execute(
                "UPDATE chains SET cv_scores = $2 WHERE chain_id = $1",
                [cid, json.dumps(averaged)],
            )

def create_schema(conn: duckdb.DuckDBPyConnection, workspace_path: Path | None = None) -> None:
    """Create all tables, views, and indexes in the given DuckDB connection.

    Safe to call multiple times -- every DDL statement uses
    ``IF NOT EXISTS``.

    Args:
        conn: An open DuckDB connection.
        workspace_path: Optional workspace root directory.  When provided,
            any legacy ``prediction_arrays`` table is automatically migrated
            to Parquet sidecar files and dropped.
    """
    for statement in SCHEMA_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

    _migrate_schema(conn, workspace_path=workspace_path)

    for statement in INDEX_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

    # Drop and recreate views to pick up schema changes
    conn.execute("DROP VIEW IF EXISTS v_aggregated_predictions")
    conn.execute("DROP VIEW IF EXISTS v_aggregated_predictions_all")
    conn.execute("DROP VIEW IF EXISTS v_chain_summary")
    for statement in VIEW_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

def _migrate_schema(conn: duckdb.DuckDBPyConnection, *, workspace_path: Path | None = None) -> None:
    """Apply incremental schema migrations to existing databases.

    Adds columns that may be missing from older database versions.
    Each migration is idempotent (checks before applying).

    Args:
        conn: An open DuckDB connection with tables already created.
        workspace_path: Optional workspace root directory for auto-migrating
            legacy ``prediction_arrays`` to Parquet.
    """
    # Migration: add refit_context column to predictions table
    existing_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'predictions'"
        ).fetchall()
    }
    if "refit_context" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN refit_context VARCHAR DEFAULT NULL")

    # Migration: add cross-run cache columns to artifacts table
    artifact_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'artifacts'"
        ).fetchall()
    }
    if "chain_path_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN chain_path_hash VARCHAR")
    if "input_data_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN input_data_hash VARCHAR")
    if "dataset_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN dataset_hash VARCHAR")

    # Migration: ensure chain_id, model_name, branch_id columns exist before unique index
    if "chain_id" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN chain_id VARCHAR DEFAULT ''")
    if "model_name" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN model_name VARCHAR DEFAULT ''")
    if "branch_id" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN branch_id INTEGER")

    # Migration: replace old natural key index with v2 (includes branch_id)
    existing_indexes = {
        row[0]
        for row in conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'predictions'"
        ).fetchall()
    }
    if "idx_predictions_natural_key" in existing_indexes:
        conn.execute("DROP INDEX idx_predictions_natural_key")

    # Deduplicate predictions before unique index creation
    if "idx_predictions_natural_key_v2" not in existing_indexes:
        dup_row = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT pipeline_id, chain_id, fold_id, partition, model_name, branch_id "
            "  FROM predictions "
            "  GROUP BY pipeline_id, chain_id, fold_id, partition, model_name, branch_id "
            "  HAVING COUNT(*) > 1"
            ")"
        ).fetchone()
        dup_count = dup_row[0] if dup_row else 0
        if dup_count > 0:
            # Keep the most recent prediction per natural key, delete older duplicates
            conn.execute(
                "DELETE FROM predictions "
                "WHERE prediction_id IN ("
                "  SELECT prediction_id FROM ("
                "    SELECT prediction_id, "
                "      ROW_NUMBER() OVER ("
                "        PARTITION BY pipeline_id, chain_id, fold_id, partition, model_name, branch_id "
                "        ORDER BY created_at DESC"
                "      ) AS rn "
                "    FROM predictions"
                "  ) WHERE rn > 1"
                ")"
            )

    # Migration: add project_id column to runs table
    runs_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'runs'"
        ).fetchall()
    }
    if "project_id" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN project_id VARCHAR")

    # Migration: add chain summary columns (skip if chains table doesn't exist yet)
    chain_exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'chains' AND table_type = 'BASE TABLE'"
    ).fetchone()
    if not chain_exists:
        return
    chain_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'chains'"
        ).fetchall()
    }
    _chain_summary_cols: list[tuple[str, str]] = [
        ("model_name", "VARCHAR"),
        ("metric", "VARCHAR"),
        ("task_type", "VARCHAR"),
        ("best_params", "JSON"),
        ("dataset_name", "VARCHAR"),
        ("cv_val_score", "DOUBLE"),
        ("cv_test_score", "DOUBLE"),
        ("cv_train_score", "DOUBLE"),
        ("cv_fold_count", "INTEGER DEFAULT 0"),
        ("cv_scores", "JSON"),
        ("final_test_score", "DOUBLE"),
        ("final_train_score", "DOUBLE"),
        ("final_scores", "JSON"),
    ]
    added_chain_cols = False
    for col_name, col_type in _chain_summary_cols:
        if col_name not in chain_columns:
            conn.execute(f"ALTER TABLE chains ADD COLUMN {col_name} {col_type}")
            added_chain_cols = True

    # Backfill chain summary from existing predictions
    if added_chain_cols:
        _backfill_chain_summaries(conn)

    # Migration: auto-migrate prediction_arrays from DuckDB to Parquet sidecar files
    if workspace_path is not None:
        _auto_migrate_prediction_arrays(conn, workspace_path)
