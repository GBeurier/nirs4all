"""Reusable SQL query builders for :class:`WorkspaceStore`.

This module provides parameterised SQL constants and helper functions
for building dynamic ``WHERE`` clauses.  All queries use ``?``
style positional parameters for safe parameterised execution via SQLite.
"""

from __future__ import annotations

# Columns that can be used for ordering / grouping in prediction queries.
# Used to guard against SQL injection when column names are interpolated.
_PREDICTION_COLUMNS: frozenset[str] = frozenset({
    "prediction_id", "pipeline_id", "chain_id", "dataset_name",
    "model_name", "model_class", "fold_id", "partition",
    "val_score", "test_score", "train_score",
    "metric", "task_type", "n_samples", "n_features",
    "scores", "best_params", "preprocessings",
    "branch_id", "branch_name",
    "exclusion_count", "exclusion_rate", "refit_context", "created_at",
})

# =========================================================================
# Run queries
# =========================================================================

GET_RUN = "SELECT * FROM runs WHERE run_id = ?"

LIST_RUNS_BASE = "SELECT * FROM runs"

INSERT_RUN = """
INSERT INTO runs (run_id, name, config, datasets, status)
VALUES (?, ?, ?, ?, 'running')
"""

COMPLETE_RUN = """
UPDATE runs
SET status = 'completed',
    completed_at = current_timestamp,
    summary = ?
WHERE run_id = ?
"""

FAIL_RUN = """
UPDATE runs
SET status = 'failed',
    completed_at = current_timestamp,
    error = ?
WHERE run_id = ?
"""

# =========================================================================
# Pipeline queries
# =========================================================================

GET_PIPELINE = "SELECT * FROM pipelines WHERE pipeline_id = ?"

LIST_PIPELINES_BASE = "SELECT * FROM pipelines"

INSERT_PIPELINE = """
INSERT INTO pipelines
    (pipeline_id, run_id, name, expanded_config, generator_choices,
     dataset_name, dataset_hash, status)
VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
"""

COMPLETE_PIPELINE = """
UPDATE pipelines
SET status = 'completed',
    completed_at = current_timestamp,
    best_val = ?,
    best_test = ?,
    metric = ?,
    duration_ms = ?
WHERE pipeline_id = ?
"""

FAIL_PIPELINE = """
UPDATE pipelines
SET status = 'failed',
    completed_at = current_timestamp,
    error = ?
WHERE pipeline_id = ?
"""

# =========================================================================
# Chain queries
# =========================================================================

GET_CHAIN = "SELECT * FROM chains WHERE chain_id = ?"

GET_CHAINS_FOR_PIPELINE = "SELECT * FROM chains WHERE pipeline_id = ?"

INSERT_CHAIN = """
INSERT INTO chains
    (chain_id, pipeline_id, steps, model_step_idx, model_class,
     preprocessings, fold_strategy, fold_artifacts, shared_artifacts,
     branch_path, source_index, dataset_name)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_CHAIN_SUMMARY = """
UPDATE chains SET
    model_name = ?,
    metric = ?,
    task_type = ?,
    best_params = ?,
    cv_val_score = ?,
    cv_test_score = ?,
    cv_train_score = ?,
    cv_fold_count = ?,
    cv_scores = ?,
    final_test_score = ?,
    final_train_score = ?,
    final_scores = ?
WHERE chain_id = ?
"""

# =========================================================================
# Prediction queries
# =========================================================================

GET_PREDICTION = "SELECT * FROM predictions WHERE prediction_id = ?"

INSERT_PREDICTION = """
INSERT INTO predictions
    (prediction_id, pipeline_id, chain_id, dataset_name, model_name,
     model_class, fold_id, partition, val_score, test_score, train_score,
     metric, task_type, n_samples, n_features, scores, best_params,
     preprocessings, branch_id, branch_name, exclusion_count, exclusion_rate,
     refit_context)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

DELETE_PREDICTION = "DELETE FROM predictions WHERE prediction_id = ?"

QUERY_PREDICTIONS_BASE = "SELECT * FROM predictions"

# =========================================================================
# Artifact queries
# =========================================================================

GET_ARTIFACT = "SELECT * FROM artifacts WHERE artifact_id = ?"

GET_ARTIFACT_BY_HASH = "SELECT * FROM artifacts WHERE content_hash = ?"

INSERT_ARTIFACT = """
INSERT INTO artifacts
    (artifact_id, artifact_path, content_hash, operator_class,
     artifact_type, format, size_bytes, ref_count)
VALUES (?, ?, ?, ?, ?, ?, ?, 1)
"""

INSERT_ARTIFACT_WITH_CACHE_KEY = """
INSERT INTO artifacts
    (artifact_id, artifact_path, content_hash, operator_class,
     artifact_type, format, size_bytes, ref_count,
     chain_path_hash, input_data_hash, dataset_hash)
VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
"""

UPDATE_ARTIFACT_CACHE_KEY = """
UPDATE artifacts
SET chain_path_hash = ?, input_data_hash = ?, dataset_hash = ?
WHERE artifact_id = ?
"""

FIND_CACHED_ARTIFACT = """
SELECT artifact_id FROM artifacts
WHERE chain_path_hash = ? AND input_data_hash = ? AND ref_count > 0
LIMIT 1
"""

INVALIDATE_DATASET_CACHE = """
UPDATE artifacts
SET chain_path_hash = NULL, input_data_hash = NULL
WHERE dataset_hash = ?
"""

INCREMENT_ARTIFACT_REF = """
UPDATE artifacts SET ref_count = ref_count + 1 WHERE artifact_id = ?
"""

DECREMENT_ARTIFACT_REF = """
UPDATE artifacts SET ref_count = ref_count - 1 WHERE artifact_id = ?
"""

GC_ARTIFACTS = """
SELECT artifact_id, artifact_path FROM artifacts WHERE ref_count <= 0
"""

DELETE_GC_ARTIFACTS = """
DELETE FROM artifacts WHERE ref_count <= 0
"""

# =========================================================================
# Log queries
# =========================================================================

INSERT_LOG = """
INSERT INTO logs
    (log_id, pipeline_id, step_idx, operator_class, event,
     duration_ms, message, details, level)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

GET_PIPELINE_LOG = """
SELECT log_id, step_idx, operator_class, event, duration_ms,
       message, details, level, created_at
FROM logs
WHERE pipeline_id = ?
ORDER BY step_idx, created_at
"""

GET_RUN_LOG_SUMMARY = """
SELECT
    p.pipeline_id,
    p.name AS pipeline_name,
    p.status AS pipeline_status,
    COUNT(l.log_id) AS log_count,
    SUM(CASE WHEN l.event = 'end' THEN l.duration_ms ELSE 0 END) AS total_duration_ms,
    SUM(CASE WHEN l.level = 'warning' THEN 1 ELSE 0 END) AS warning_count,
    SUM(CASE WHEN l.level = 'error' THEN 1 ELSE 0 END) AS error_count
FROM pipelines p
LEFT JOIN logs l ON p.pipeline_id = l.pipeline_id
WHERE p.run_id = ?
GROUP BY p.pipeline_id, p.name, p.status, p.created_at
ORDER BY p.created_at
"""

# =========================================================================
# Project queries
# =========================================================================

GET_PROJECT = "SELECT * FROM projects WHERE project_id = ?"

GET_PROJECT_BY_NAME = "SELECT * FROM projects WHERE name = ?"

LIST_PROJECTS = "SELECT * FROM projects ORDER BY created_at DESC"

INSERT_PROJECT = """
INSERT INTO projects (project_id, name, description, color)
VALUES (?, ?, ?, ?)
"""

UPDATE_PROJECT = """
UPDATE projects
SET name = ?, description = ?, color = ?, updated_at = current_timestamp
WHERE project_id = ?
"""

DELETE_PROJECT = "DELETE FROM projects WHERE project_id = ?"

SET_RUN_PROJECT = "UPDATE runs SET project_id = ? WHERE run_id = ?"

CLEAR_RUN_PROJECT = "UPDATE runs SET project_id = NULL WHERE run_id = ?"

# =========================================================================
# Deletion queries
# =========================================================================

DELETE_RUN = "DELETE FROM runs WHERE run_id = ?"

# --- Manual cascade queries ---

# Cascade delete for a run: delete all dependents in reverse dependency order
CASCADE_DELETE_RUN_PREDICTIONS = """
DELETE FROM predictions WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = ?
)
"""

CASCADE_DELETE_RUN_LOGS = """
DELETE FROM logs WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = ?
)
"""

CASCADE_DELETE_RUN_CHAINS = """
DELETE FROM chains WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = ?
)
"""

CASCADE_DELETE_RUN_PIPELINES = "DELETE FROM pipelines WHERE run_id = ?"

# Cascade delete for a pipeline
CASCADE_DELETE_PIPELINE_PREDICTIONS = "DELETE FROM predictions WHERE pipeline_id = ?"
CASCADE_DELETE_PIPELINE_LOGS = "DELETE FROM logs WHERE pipeline_id = ?"
CASCADE_DELETE_PIPELINE_CHAINS = "DELETE FROM chains WHERE pipeline_id = ?"
DELETE_PIPELINE = "DELETE FROM pipelines WHERE pipeline_id = ?"

# Cascade delete for a chain (SET NULL on predictions.chain_id)
CASCADE_NULLIFY_CHAIN_PREDICTIONS = "UPDATE predictions SET chain_id = NULL WHERE chain_id = ?"
DELETE_CHAIN = "DELETE FROM chains WHERE chain_id = ?"

# =========================================================================
# Dynamic query builders
# =========================================================================

# ---- Chain summary (from v_chain_summary VIEW) --------------------------

QUERY_CHAIN_SUMMARY_BASE = "SELECT * FROM v_chain_summary"

def build_aggregated_query(
    *,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
    dataset_name: str | None = None,
    model_class: str | None = None,
    metric: str | None = None,
    score_scope: str = "cv",
) -> tuple[str, list[object]]:
    """Build a query against ``v_chain_summary``.

    .. deprecated::
        Use :func:`build_chain_summary_query` instead.  The *score_scope*
        parameter is ignored — the chain summary view contains both CV
        and final scores in each row.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.
    """
    return build_chain_summary_query(
        run_id=run_id,
        pipeline_id=pipeline_id,
        chain_id=chain_id,
        dataset_name=dataset_name,
        model_class=model_class,
        metric=metric,
    )

def build_chain_predictions_query(
    *,
    chain_id: str,
    partition: str | None = None,
    fold_id: str | None = None,
) -> tuple[str, list[object]]:
    """Build a drill-down query for individual predictions of a chain.

    Args:
        chain_id: Required chain identifier.
        partition: Optional partition filter (``"train"``, ``"val"``, ``"test"``).
        fold_id: Optional fold identifier filter.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.
    """
    conditions: list[str] = ["chain_id = ?"]
    params: list[object] = [chain_id]

    if partition is not None:
        conditions.append("partition = ?")
        params.append(partition)

    if fold_id is not None:
        conditions.append("fold_id = ?")
        params.append(fold_id)

    where = " WHERE " + " AND ".join(conditions)
    sql = f"SELECT * FROM predictions{where} ORDER BY partition, fold_id"
    return sql, params

def build_top_aggregated_query(
    *,
    metric: str,
    n: int = 10,
    score_column: str = "avg_val_score",
    ascending: bool = True,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    dataset_name: str | None = None,
    model_class: str | None = None,
    score_scope: str = "cv",
) -> tuple[str, list[object]]:
    """Build a ranking query on ``v_chain_summary``.

    .. deprecated::
        Use :func:`build_top_chains_query` instead.  The *score_scope*
        parameter is ignored — the chain summary view contains both CV
        and final scores in each row.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.

    Raises:
        ValueError: If *score_column* is not a valid aggregation column.
    """
    return build_top_chains_query(
        metric=metric,
        n=n,
        score_column=score_column,
        ascending=ascending,
        run_id=run_id,
        pipeline_id=pipeline_id,
        dataset_name=dataset_name,
        model_class=model_class,
    )

def build_prediction_query(
    *,
    dataset_name: str | None = None,
    model_class: str | None = None,
    partition: str | None = None,
    fold_id: str | None = None,
    branch_id: int | None = None,
    pipeline_id: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[str, list[object]]:
    """Build a full ``SELECT`` query for the predictions table.

    Supports joining through ``pipelines`` when filtering by *run_id*.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.
    """
    needs_join = run_id is not None
    base = "SELECT pr.* FROM predictions pr JOIN pipelines pl ON pr.pipeline_id = pl.pipeline_id" if needs_join else "SELECT * FROM predictions"

    conditions: list[str] = []
    params: list[object] = []
    prefix = "pr." if needs_join else ""

    for col, val in [
        ("dataset_name", dataset_name),
        ("model_class", model_class),
        ("partition", partition),
        ("fold_id", fold_id),
        ("branch_id", branch_id),
        ("pipeline_id", pipeline_id),
    ]:
        if val is None:
            continue
        full_col = f"{prefix}{col}"
        if isinstance(val, str) and "%" in val:
            conditions.append(f"{full_col} LIKE ?")
        else:
            conditions.append(f"{full_col} = ?")
        params.append(val)

    if run_id is not None:
        conditions.append("pl.run_id = ?")
        params.append(run_id)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    order = f" ORDER BY {prefix}created_at DESC"
    pagination = ""
    if limit is not None:
        pagination += " LIMIT ?"
        params.append(limit)
    if offset:
        pagination += " OFFSET ?"
        params.append(offset)

    return base + where + order + pagination, params

def build_top_predictions_query(
    *,
    n: int,
    metric: str = "val_score",
    ascending: bool = True,
    partition: str = "val",
    dataset_name: str | None = None,
    group_by: str | None = None,
) -> tuple[str, list[object]]:
    """Build a ranking query for top-N predictions.

    When *group_by* is set, returns top *n* per group using a
    window function.

    Args:
        n: Number of top predictions to return.
        metric: Column name to rank by.  Must be a valid prediction
            column (validated against ``_PREDICTION_COLUMNS``).
        ascending: Sort direction.
        partition: Only consider this partition.
        dataset_name: Optional dataset filter.
        group_by: Optional grouping column.  Must be a valid prediction
            column if provided.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.

    Raises:
        ValueError: If *metric* or *group_by* is not a valid column name.
    """
    if metric not in _PREDICTION_COLUMNS:
        raise ValueError(f"Invalid metric column: {metric!r}")
    if group_by is not None and group_by not in _PREDICTION_COLUMNS:
        raise ValueError(f"Invalid group_by column: {group_by!r}")

    direction = "ASC" if ascending else "DESC"

    conditions: list[str] = []
    params: list[object] = []

    conditions.append("partition = ?")
    params.append(partition)

    if dataset_name is not None:
        conditions.append("dataset_name = ?")
        params.append(dataset_name)

    where = " WHERE " + " AND ".join(conditions)

    if group_by is not None:
        sql = (
            f"SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {group_by} ORDER BY ({metric} IS NULL), {metric} {direction}) AS _rn "
            f"FROM predictions{where}"
            f") sub WHERE _rn <= ?"
        )
        params.append(n)
    else:
        sql = (
            f"SELECT * FROM predictions{where} "
            f"ORDER BY ({metric} IS NULL), {metric} {direction} "
            f"LIMIT ?"
        )
        params.append(n)

    return sql, params

# =========================================================================
# Chain summary query builders (v_chain_summary VIEW)
# =========================================================================

_CHAIN_SUMMARY_COLUMNS: frozenset[str] = frozenset({
    "chain_id", "pipeline_id", "model_class", "model_step_idx",
    "model_name", "preprocessings", "branch_path", "source_index",
    "metric", "task_type", "best_params", "dataset_name",
    "cv_val_score", "cv_test_score", "cv_train_score", "cv_fold_count",
    "cv_scores", "final_test_score", "final_train_score", "final_scores",
    "run_id", "pipeline_status",
})

def build_chain_summary_query(
    *,
    run_id: str | list[str] | None = None,
    pipeline_id: str | list[str] | None = None,
    chain_id: str | list[str] | None = None,
    dataset_name: str | list[str] | None = None,
    model_class: str | list[str] | None = None,
    metric: str | None = None,
    task_type: str | None = None,
) -> tuple[str, list[object]]:
    """Build a query against the ``v_chain_summary`` VIEW.

    Returns one row per chain with CV averages, final scores,
    multi-metric JSON, and chain metadata.

    All filters are optional and combined with ``AND``.
    String filters accept a single value or a list of values (``IN``).

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.
    """
    conditions: list[str] = []
    params: list[object] = []

    for col, val in [
        ("run_id", run_id),
        ("pipeline_id", pipeline_id),
        ("chain_id", chain_id),
        ("dataset_name", dataset_name),
        ("model_class", model_class),
        ("metric", metric),
        ("task_type", task_type),
    ]:
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                continue
            if len(val) == 1:
                conditions.append(f"{col} = ?")
                params.append(val[0])
            else:
                placeholders = ", ".join("?" for _ in val)
                conditions.append(f"{col} IN ({placeholders})")
                params.extend(val)
        elif isinstance(val, str) and "%" in val:
            conditions.append(f"{col} LIKE ?")
            params.append(val)
        else:
            conditions.append(f"{col} = ?")
            params.append(val)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    return QUERY_CHAIN_SUMMARY_BASE + where, params

def build_top_chains_query(
    *,
    metric: str | None = None,
    n: int = 10,
    score_column: str = "cv_val_score",
    ascending: bool = True,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    dataset_name: str | None = None,
    model_class: str | None = None,
) -> tuple[str, list[object]]:
    """Build a ranking query on ``v_chain_summary``.

    Args:
        metric: Optional metric name filter.
        n: Number of top results to return.
        score_column: Column to sort by (e.g. ``"cv_val_score"``,
            ``"final_test_score"``).
        ascending: Sort direction.  ``True`` for lower-is-better metrics.
        run_id: Optional run filter.
        pipeline_id: Optional pipeline filter.
        dataset_name: Optional dataset filter.
        model_class: Optional model class filter.

    Returns:
        ``(sql, params)`` ready for ``conn.execute(sql, params)``.

    Raises:
        ValueError: If *score_column* is not a valid column.
    """
    valid_score_columns = frozenset({
        "cv_val_score", "cv_test_score", "cv_train_score",
        "final_test_score", "final_train_score",
        # Deprecated aliases for backward compatibility
        "min_val_score", "max_val_score", "avg_val_score",
        "min_test_score", "max_test_score", "avg_test_score",
        "min_train_score", "max_train_score", "avg_train_score",
    })
    # Map old aggregated column names to new chain summary columns
    _column_aliases: dict[str, str] = {
        "avg_val_score": "cv_val_score",
        "avg_test_score": "cv_test_score",
        "avg_train_score": "cv_train_score",
        "min_val_score": "cv_val_score",
        "max_val_score": "cv_val_score",
        "min_test_score": "cv_test_score",
        "max_test_score": "cv_test_score",
        "min_train_score": "cv_train_score",
        "max_train_score": "cv_train_score",
    }
    resolved_column = _column_aliases.get(score_column, score_column)
    if score_column not in valid_score_columns:
        raise ValueError(f"Invalid score column: {score_column!r}")

    conditions: list[str] = []
    params: list[object] = []

    if metric is not None:
        conditions.append("metric = ?")
        params.append(metric)

    for col, val in [
        ("run_id", run_id),
        ("pipeline_id", pipeline_id),
        ("dataset_name", dataset_name),
        ("model_class", model_class),
    ]:
        if val is None:
            continue
        conditions.append(f"{col} = ?")
        params.append(val)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    direction = "ASC" if ascending else "DESC"

    sql = (
        f"SELECT * FROM v_chain_summary{where} "
        f"ORDER BY ({resolved_column} IS NULL), {resolved_column} {direction} "
        f"LIMIT ?"
    )
    params.append(n)

    return sql, params
