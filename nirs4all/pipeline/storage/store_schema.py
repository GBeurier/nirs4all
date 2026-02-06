"""DuckDB schema definitions for the workspace store.

Defines the seven-table schema used by :class:`WorkspaceStore`:
``runs``, ``pipelines``, ``chains``, ``predictions``,
``prediction_arrays``, ``artifacts``, and ``logs``.

The schema uses ``IF NOT EXISTS`` for all DDL statements, making
:func:`create_schema` safe to call repeatedly (idempotent).
"""

from __future__ import annotations

import duckdb

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
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS prediction_arrays (
    prediction_id VARCHAR PRIMARY KEY REFERENCES predictions(prediction_id),
    y_true DOUBLE[],
    y_pred DOUBLE[],
    y_proba DOUBLE[],
    sample_indices INTEGER[],
    weights DOUBLE[]
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
"""

# =========================================================================
# View DDL
# =========================================================================

VIEW_DDL: str = """
CREATE VIEW IF NOT EXISTS v_aggregated_predictions AS
SELECT
    pl.run_id,
    c.pipeline_id,
    c.chain_id,
    c.model_class,
    c.model_step_idx,
    c.preprocessings,
    c.branch_path,
    c.source_index,
    p.model_name,
    p.metric,
    p.dataset_name,
    COUNT(DISTINCT p.fold_id) AS fold_count,
    COUNT(DISTINCT p.partition) AS partition_count,
    LIST(DISTINCT p.partition ORDER BY p.partition) AS partitions,
    MIN(p.val_score) AS min_val_score,
    MAX(p.val_score) AS max_val_score,
    AVG(p.val_score) AS avg_val_score,
    MIN(p.test_score) AS min_test_score,
    MAX(p.test_score) AS max_test_score,
    AVG(p.test_score) AS avg_test_score,
    MIN(p.train_score) AS min_train_score,
    MAX(p.train_score) AS max_train_score,
    AVG(p.train_score) AS avg_train_score,
    LIST(p.prediction_id ORDER BY p.partition, p.fold_id) AS prediction_ids,
    LIST(p.fold_id ORDER BY p.partition, p.fold_id) AS fold_ids
FROM predictions p
JOIN chains c ON p.chain_id = c.chain_id
JOIN pipelines pl ON c.pipeline_id = pl.pipeline_id
GROUP BY
    pl.run_id, c.pipeline_id, c.chain_id, c.model_class,
    c.model_step_idx, c.preprocessings, c.branch_path,
    c.source_index, p.model_name, p.metric, p.dataset_name;
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
CREATE INDEX IF NOT EXISTS idx_logs_pipeline_id ON logs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_content_hash ON artifacts(content_hash);
"""

# =========================================================================
# Table names (ordered by dependency)
# =========================================================================

TABLE_NAMES: list[str] = [
    "runs",
    "pipelines",
    "chains",
    "predictions",
    "prediction_arrays",
    "artifacts",
    "logs",
]


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables, views, and indexes in the given DuckDB connection.

    Safe to call multiple times -- every DDL statement uses
    ``IF NOT EXISTS``.

    Args:
        conn: An open DuckDB connection.
    """
    for statement in SCHEMA_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

    for statement in INDEX_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

    for statement in VIEW_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
