"""SQLite schema definitions for the workspace store.

Defines the table schema used by :class:`WorkspaceStore`:
``runs``, ``pipelines``, ``chains``, ``predictions``,
``artifacts``, ``logs``, and ``projects``.

Dense prediction arrays (y_true, y_pred, etc.) are stored in Parquet
sidecar files managed by :class:`ArrayStore`, not in SQLite.

The schema uses ``IF NOT EXISTS`` for all DDL statements, making
:func:`create_schema` safe to call repeatedly (idempotent).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

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
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config TEXT,
    datasets TEXT,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    summary TEXT,
    error TEXT,
    project_id TEXT
);

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    name TEXT NOT NULL,
    expanded_config TEXT,
    generator_choices TEXT,
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    best_val REAL,
    best_test REAL,
    metric TEXT,
    duration_ms INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS chains (
    chain_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    steps TEXT NOT NULL,
    model_step_idx INTEGER NOT NULL,
    model_class TEXT NOT NULL,
    preprocessings TEXT DEFAULT '',
    fold_strategy TEXT DEFAULT 'per_fold',
    fold_artifacts TEXT,
    shared_artifacts TEXT,
    branch_path TEXT,
    source_index INTEGER,
    model_name TEXT,
    metric TEXT,
    task_type TEXT,
    best_params TEXT,
    dataset_name TEXT,
    cv_val_score REAL,
    cv_test_score REAL,
    cv_train_score REAL,
    cv_fold_count INTEGER DEFAULT 0,
    cv_scores TEXT,
    final_test_score REAL,
    final_train_score REAL,
    final_scores TEXT,
    final_agg_test_score REAL,
    final_agg_train_score REAL,
    final_agg_scores TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    chain_id TEXT REFERENCES chains(chain_id),
    dataset_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_class TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    partition TEXT NOT NULL,
    val_score REAL,
    test_score REAL,
    train_score REAL,
    metric TEXT NOT NULL,
    task_type TEXT NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    scores TEXT,
    best_params TEXT,
    preprocessings TEXT DEFAULT '',
    branch_id INTEGER,
    branch_name TEXT,
    exclusion_count INTEGER DEFAULT 0,
    exclusion_rate REAL DEFAULT 0.0,
    refit_context TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    operator_class TEXT,
    artifact_type TEXT,
    format TEXT DEFAULT 'joblib',
    size_bytes INTEGER,
    ref_count INTEGER DEFAULT 1,
    chain_path_hash TEXT,
    input_data_hash TEXT,
    dataset_hash TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    step_idx INTEGER NOT NULL,
    operator_class TEXT,
    event TEXT NOT NULL,
    duration_ms INTEGER,
    message TEXT,
    details TEXT,
    level TEXT DEFAULT 'info',
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    color TEXT DEFAULT '#14b8a6',
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
    c.final_agg_test_score,
    c.final_agg_train_score,
    c.final_agg_scores,
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

def _auto_migrate_prediction_arrays(conn: sqlite3.Connection, workspace_path: Path) -> None:
    """Auto-migrate legacy ``prediction_arrays`` table to Parquet sidecar files.

    Called during schema migration when a workspace still has the legacy
    table.  Streams all rows to per-dataset Parquet files via
    :class:`ArrayStore`, then drops the table.

    Args:
        conn: An open SQLite connection.
        workspace_path: Workspace root directory for the ArrayStore.
    """
    import json as _json

    import numpy as np

    from nirs4all.pipeline.storage.array_store import ArrayStore

    def _parse_array(val: object) -> list[float] | None:
        """Parse a value that may be a list, JSON string, or None."""
        if val is None:
            return None
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            parsed: list[float] = _json.loads(val)
            return parsed
        return list(val)  # type: ignore[call-overload,no-any-return]

    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_arrays'"
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
                "WHERE p.dataset_name = ? "
                "ORDER BY pa.prediction_id "
                "LIMIT ? OFFSET ?",
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

                _yt = _parse_array(y_true)
                _yp = _parse_array(y_pred)
                _ypr = _parse_array(y_proba)
                _si = _parse_array(sample_indices)
                _wt = _parse_array(weights)

                records.append({
                    "prediction_id": prediction_id,
                    "dataset_name": dataset_name,
                    "model_name": model_name or "",
                    "fold_id": fold_id or "",
                    "partition": partition or "",
                    "metric": metric or "",
                    "val_score": val_score,
                    "task_type": task_type or "",
                    "y_true": np.array(_yt, dtype=np.float64) if _yt is not None else None,
                    "y_pred": np.array(_yp, dtype=np.float64) if _yp is not None else None,
                    "y_proba": np.array(_ypr, dtype=np.float64) if _ypr is not None else None,
                    "sample_indices": np.array(_si, dtype=np.int32) if _si is not None else None,
                    "weights": np.array(_wt, dtype=np.float64) if _wt is not None else None,
                })

            array_store.save_batch(records)
            offset += batch_size

    conn.execute("DROP TABLE prediction_arrays")
    logger.info("Auto-migration complete. Dropped prediction_arrays table.")

def _backfill_chain_summaries(conn: sqlite3.Connection) -> None:
    """Backfill chain summary columns from existing prediction data.

    Called once when migrating an older database.  Uses SQL aggregation
    to compute CV averages and extract final/refit scores.

    Args:
        conn: An open SQLite connection.
    """
    # Check whether there are any predictions to backfill from
    has_predictions = conn.execute(
        "SELECT 1 FROM predictions LIMIT 1"
    ).fetchone()
    if has_predictions is None:
        return

    # Backfill dataset_name from pipelines for all chains
    conn.execute("""
        UPDATE chains SET dataset_name = (
            SELECT pl.dataset_name FROM pipelines pl
            WHERE pl.pipeline_id = chains.pipeline_id
        )
        WHERE chains.dataset_name IS NULL
    """)

    # Backfill CV averages from cross-validation predictions
    conn.execute("""
        UPDATE chains SET
            model_name = COALESCE(
                chains.model_name,
                (
                    SELECT p.model_name
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                      AND p.refit_context IS NULL
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                ),
                (
                    SELECT p.model_name
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                )
            ),
            metric = COALESCE(
                chains.metric,
                (
                    SELECT p.metric
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                      AND p.refit_context IS NULL
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                ),
                (
                    SELECT p.metric
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                )
            ),
            task_type = COALESCE(
                chains.task_type,
                (
                    SELECT p.task_type
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                      AND p.refit_context IS NULL
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                ),
                (
                    SELECT p.task_type
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                )
            ),
            best_params = COALESCE(
                chains.best_params,
                (
                    SELECT p.best_params
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                      AND p.refit_context IS NULL
                      AND p.best_params IS NOT NULL
                      AND p.best_params != '{}'
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                ),
                (
                    SELECT p.best_params
                    FROM predictions p
                    WHERE p.chain_id = chains.chain_id
                      AND p.best_params IS NOT NULL
                      AND p.best_params != '{}'
                    ORDER BY p.created_at ASC, p.prediction_id ASC
                    LIMIT 1
                )
            ),
            cv_val_score = (
                SELECT AVG(p.val_score)
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NULL
            ),
            cv_test_score = (
                SELECT AVG(p.test_score)
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NULL
            ),
            cv_train_score = (
                SELECT AVG(p.train_score)
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NULL
            ),
            cv_fold_count = COALESCE((
                SELECT COUNT(DISTINCT p.fold_id)
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NULL
            ), 0)
        WHERE chains.chain_id IN (SELECT DISTINCT chain_id FROM predictions WHERE chain_id IS NOT NULL)
    """)

    # Backfill final scores from refit predictions
    conn.execute("""
        UPDATE chains SET
            final_test_score = (
                SELECT p.test_score
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NOT NULL
                  AND p.fold_id = 'final'
                  AND p.partition = 'test'
                ORDER BY p.created_at ASC, p.prediction_id ASC
                LIMIT 1
            ),
            final_train_score = (
                SELECT p.train_score
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NOT NULL
                  AND p.fold_id = 'final'
                  AND p.partition = 'test'
                ORDER BY p.created_at ASC, p.prediction_id ASC
                LIMIT 1
            ),
            final_scores = (
                SELECT p.scores
                FROM predictions p
                WHERE p.chain_id = chains.chain_id
                  AND p.refit_context IS NOT NULL
                  AND p.fold_id = 'final'
                  AND p.partition = 'test'
                ORDER BY p.created_at ASC, p.prediction_id ASC
                LIMIT 1
            )
        WHERE chains.chain_id IN (SELECT DISTINCT chain_id FROM predictions WHERE chain_id IS NOT NULL AND refit_context IS NOT NULL AND fold_id = 'final' AND partition = 'test')
    """)

    # Backfill cv_scores (averaged multi-metric JSON) via Python
    chain_ids = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT chain_id FROM chains WHERE chain_id IS NOT NULL"
        ).fetchall()
    ]
    import json
    for cid in chain_ids:
        rows = conn.execute(
            "SELECT partition, scores FROM predictions "
            "WHERE chain_id = ? AND refit_context IS NULL "
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
                "UPDATE chains SET cv_scores = ? WHERE chain_id = ?",
                [json.dumps(averaged), cid],
            )

def create_schema(conn: sqlite3.Connection, workspace_path: Path | None = None) -> None:
    """Create all tables, views, and indexes in the given SQLite connection.

    Safe to call multiple times -- every DDL statement uses
    ``IF NOT EXISTS``.

    Args:
        conn: An open SQLite connection.
        workspace_path: Optional workspace root directory.  When provided,
            any legacy ``prediction_arrays`` table is automatically migrated
            to Parquet sidecar files and dropped.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

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

def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return the set of column names for a table using PRAGMA table_info.

    Args:
        conn: An open SQLite connection.
        table_name: Name of the table to inspect.

    Returns:
        Set of column name strings.
    """
    return {row[1] for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()}

def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check whether a table exists in the SQLite database.

    Args:
        conn: An open SQLite connection.
        table_name: Name of the table to check.

    Returns:
        True if the table exists.
    """
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [table_name]).fetchone() is not None

def _get_index_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return the set of index names for a table using PRAGMA index_list.

    Args:
        conn: An open SQLite connection.
        table_name: Name of the table to inspect.

    Returns:
        Set of index name strings.
    """
    return {row[1] for row in conn.execute(f"PRAGMA index_list('{table_name}')").fetchall()}

def _migrate_schema(conn: sqlite3.Connection, *, workspace_path: Path | None = None) -> None:
    """Apply incremental schema migrations to existing databases.

    Adds columns that may be missing from older database versions.
    Each migration is idempotent (checks before applying).

    Args:
        conn: An open SQLite connection with tables already created.
        workspace_path: Optional workspace root directory for auto-migrating
            legacy ``prediction_arrays`` to Parquet.
    """
    # Migration: add refit_context column to predictions table
    existing_columns = _get_table_columns(conn, "predictions")
    if "refit_context" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN refit_context TEXT DEFAULT NULL")

    # Migration: add cross-run cache columns to artifacts table
    artifact_columns = _get_table_columns(conn, "artifacts")
    if "chain_path_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN chain_path_hash TEXT")
    if "input_data_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN input_data_hash TEXT")
    if "dataset_hash" not in artifact_columns:
        conn.execute("ALTER TABLE artifacts ADD COLUMN dataset_hash TEXT")

    # Migration: ensure chain_id, model_name, branch_id columns exist before unique index
    if "chain_id" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN chain_id TEXT DEFAULT ''")
    if "model_name" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN model_name TEXT DEFAULT ''")
    if "branch_id" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN branch_id INTEGER")

    # Migration: replace old natural key index with v2 (includes branch_id)
    existing_indexes = _get_index_names(conn, "predictions")
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
    runs_columns = _get_table_columns(conn, "runs")
    if "project_id" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN project_id TEXT")

    # Migration: add chain summary columns (skip if chains table doesn't exist yet)
    if not _table_exists(conn, "chains"):
        return
    chain_columns = _get_table_columns(conn, "chains")
    _chain_summary_cols: list[tuple[str, str]] = [
        ("model_name", "TEXT"),
        ("metric", "TEXT"),
        ("task_type", "TEXT"),
        ("best_params", "TEXT"),
        ("dataset_name", "TEXT"),
        ("cv_val_score", "REAL"),
        ("cv_test_score", "REAL"),
        ("cv_train_score", "REAL"),
        ("cv_fold_count", "INTEGER DEFAULT 0"),
        ("cv_scores", "TEXT"),
        ("final_test_score", "REAL"),
        ("final_train_score", "REAL"),
        ("final_scores", "TEXT"),
        ("final_agg_test_score", "REAL"),
        ("final_agg_train_score", "REAL"),
        ("final_agg_scores", "TEXT"),
    ]
    added_chain_cols = False
    for col_name, col_type in _chain_summary_cols:
        if col_name not in chain_columns:
            conn.execute(f"ALTER TABLE chains ADD COLUMN {col_name} {col_type}")
            added_chain_cols = True

    # Backfill chain summary from existing predictions
    if added_chain_cols:
        _backfill_chain_summaries(conn)

    # Migration: auto-migrate prediction_arrays from legacy table to Parquet sidecar files
    if workspace_path is not None:
        _auto_migrate_prediction_arrays(conn, workspace_path)
