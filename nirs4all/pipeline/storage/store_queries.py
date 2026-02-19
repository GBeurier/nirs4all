"""Reusable SQL query builders for :class:`WorkspaceStore`.

This module provides parameterised SQL constants and helper functions
for building dynamic ``WHERE`` clauses.  All queries use ``$1``, ``$2``
style positional parameters for safe parameterised execution via DuckDB.
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

GET_RUN = "SELECT * FROM runs WHERE run_id = $1"

LIST_RUNS_BASE = "SELECT * FROM runs"

INSERT_RUN = """
INSERT INTO runs (run_id, name, config, datasets, status)
VALUES ($1, $2, $3, $4, 'running')
"""

COMPLETE_RUN = """
UPDATE runs
SET status = 'completed',
    completed_at = current_timestamp,
    summary = $2
WHERE run_id = $1
"""

FAIL_RUN = """
UPDATE runs
SET status = 'failed',
    completed_at = current_timestamp,
    error = $2
WHERE run_id = $1
"""

# =========================================================================
# Pipeline queries
# =========================================================================

GET_PIPELINE = "SELECT * FROM pipelines WHERE pipeline_id = $1"

LIST_PIPELINES_BASE = "SELECT * FROM pipelines"

INSERT_PIPELINE = """
INSERT INTO pipelines
    (pipeline_id, run_id, name, expanded_config, generator_choices,
     dataset_name, dataset_hash, status)
VALUES ($1, $2, $3, $4, $5, $6, $7, 'running')
"""

COMPLETE_PIPELINE = """
UPDATE pipelines
SET status = 'completed',
    completed_at = current_timestamp,
    best_val = $2,
    best_test = $3,
    metric = $4,
    duration_ms = $5
WHERE pipeline_id = $1
"""

FAIL_PIPELINE = """
UPDATE pipelines
SET status = 'failed',
    completed_at = current_timestamp,
    error = $2
WHERE pipeline_id = $1
"""

# =========================================================================
# Chain queries
# =========================================================================

GET_CHAIN = "SELECT * FROM chains WHERE chain_id = $1"

GET_CHAINS_FOR_PIPELINE = "SELECT * FROM chains WHERE pipeline_id = $1"

INSERT_CHAIN = """
INSERT INTO chains
    (chain_id, pipeline_id, steps, model_step_idx, model_class,
     preprocessings, fold_strategy, fold_artifacts, shared_artifacts,
     branch_path, source_index, dataset_name)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
"""

UPDATE_CHAIN_SUMMARY = """
UPDATE chains SET
    model_name = $2,
    metric = $3,
    task_type = $4,
    best_params = $5,
    cv_val_score = $6,
    cv_test_score = $7,
    cv_train_score = $8,
    cv_fold_count = $9,
    cv_scores = $10,
    final_test_score = $11,
    final_train_score = $12,
    final_scores = $13
WHERE chain_id = $1
"""

# =========================================================================
# Prediction queries
# =========================================================================

GET_PREDICTION = "SELECT * FROM predictions WHERE prediction_id = $1"

INSERT_PREDICTION = """
INSERT INTO predictions
    (prediction_id, pipeline_id, chain_id, dataset_name, model_name,
     model_class, fold_id, partition, val_score, test_score, train_score,
     metric, task_type, n_samples, n_features, scores, best_params,
     preprocessings, branch_id, branch_name, exclusion_count, exclusion_rate,
     refit_context)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
        $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
"""

DELETE_PREDICTION = "DELETE FROM predictions WHERE prediction_id = $1"

QUERY_PREDICTIONS_BASE = "SELECT * FROM predictions"

# =========================================================================
# Artifact queries
# =========================================================================

GET_ARTIFACT = "SELECT * FROM artifacts WHERE artifact_id = $1"

GET_ARTIFACT_BY_HASH = "SELECT * FROM artifacts WHERE content_hash = $1"

INSERT_ARTIFACT = """
INSERT INTO artifacts
    (artifact_id, artifact_path, content_hash, operator_class,
     artifact_type, format, size_bytes, ref_count)
VALUES ($1, $2, $3, $4, $5, $6, $7, 1)
"""

INSERT_ARTIFACT_WITH_CACHE_KEY = """
INSERT INTO artifacts
    (artifact_id, artifact_path, content_hash, operator_class,
     artifact_type, format, size_bytes, ref_count,
     chain_path_hash, input_data_hash, dataset_hash)
VALUES ($1, $2, $3, $4, $5, $6, $7, 1, $8, $9, $10)
"""

UPDATE_ARTIFACT_CACHE_KEY = """
UPDATE artifacts
SET chain_path_hash = $2, input_data_hash = $3, dataset_hash = $4
WHERE artifact_id = $1
"""

FIND_CACHED_ARTIFACT = """
SELECT artifact_id FROM artifacts
WHERE chain_path_hash = $1 AND input_data_hash = $2 AND ref_count > 0
LIMIT 1
"""

INVALIDATE_DATASET_CACHE = """
UPDATE artifacts
SET chain_path_hash = NULL, input_data_hash = NULL
WHERE dataset_hash = $1
"""

INCREMENT_ARTIFACT_REF = """
UPDATE artifacts SET ref_count = ref_count + 1 WHERE artifact_id = $1
"""

DECREMENT_ARTIFACT_REF = """
UPDATE artifacts SET ref_count = ref_count - 1 WHERE artifact_id = $1
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
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

GET_PIPELINE_LOG = """
SELECT log_id, step_idx, operator_class, event, duration_ms,
       message, details, level, created_at
FROM logs
WHERE pipeline_id = $1
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
WHERE p.run_id = $1
GROUP BY p.pipeline_id, p.name, p.status, p.created_at
ORDER BY p.created_at
"""

# =========================================================================
# Project queries
# =========================================================================

GET_PROJECT = "SELECT * FROM projects WHERE project_id = $1"

GET_PROJECT_BY_NAME = "SELECT * FROM projects WHERE name = $1"

LIST_PROJECTS = "SELECT * FROM projects ORDER BY created_at DESC"

INSERT_PROJECT = """
INSERT INTO projects (project_id, name, description, color)
VALUES ($1, $2, $3, $4)
"""

UPDATE_PROJECT = """
UPDATE projects
SET name = $2, description = $3, color = $4, updated_at = current_timestamp
WHERE project_id = $1
"""

DELETE_PROJECT = "DELETE FROM projects WHERE project_id = $1"

SET_RUN_PROJECT = "UPDATE runs SET project_id = $2 WHERE run_id = $1"

CLEAR_RUN_PROJECT = "UPDATE runs SET project_id = NULL WHERE run_id = $1"

# =========================================================================
# Deletion queries
# =========================================================================

DELETE_RUN = "DELETE FROM runs WHERE run_id = $1"

# --- Manual cascade queries (DuckDB does not support ON DELETE CASCADE) ---

# Cascade delete for a run: delete all dependents in reverse dependency order
CASCADE_DELETE_RUN_PREDICTIONS = """
DELETE FROM predictions WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = $1
)
"""

CASCADE_DELETE_RUN_LOGS = """
DELETE FROM logs WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = $1
)
"""

CASCADE_DELETE_RUN_CHAINS = """
DELETE FROM chains WHERE pipeline_id IN (
    SELECT pipeline_id FROM pipelines WHERE run_id = $1
)
"""

CASCADE_DELETE_RUN_PIPELINES = "DELETE FROM pipelines WHERE run_id = $1"

# Cascade delete for a pipeline
CASCADE_DELETE_PIPELINE_PREDICTIONS = "DELETE FROM predictions WHERE pipeline_id = $1"
CASCADE_DELETE_PIPELINE_LOGS = "DELETE FROM logs WHERE pipeline_id = $1"
CASCADE_DELETE_PIPELINE_CHAINS = "DELETE FROM chains WHERE pipeline_id = $1"
DELETE_PIPELINE = "DELETE FROM pipelines WHERE pipeline_id = $1"

# Cascade delete for a chain (SET NULL on predictions.chain_id)
CASCADE_NULLIFY_CHAIN_PREDICTIONS = "UPDATE predictions SET chain_id = NULL WHERE chain_id = $1"
DELETE_CHAIN = "DELETE FROM chains WHERE chain_id = $1"

# Artifact ref-count queries for cascade cleanup
GET_CHAIN_ARTIFACT_IDS = """
SELECT DISTINCT unnest(
    list_concat(
        COALESCE(json_keys(fold_artifacts)::VARCHAR[], ARRAY[]::VARCHAR[]),
        COALESCE(json_keys(shared_artifacts)::VARCHAR[], ARRAY[]::VARCHAR[])
    )
) AS key_name,
fold_artifacts,
shared_artifacts
FROM chains
WHERE pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = $1)
"""

# =========================================================================
# Dynamic query builders
# =========================================================================

# ---- Chain summary (from v_chain_summary VIEW) --------------------------

QUERY_CHAIN_SUMMARY_BASE = "SELECT * FROM v_chain_summary"

# Deprecated aliases -- kept for backward compatibility
QUERY_AGGREGATED_PREDICTIONS_BASE = QUERY_CHAIN_SUMMARY_BASE
QUERY_AGGREGATED_PREDICTIONS_ALL_BASE = QUERY_CHAIN_SUMMARY_BASE

GET_CHAIN_PREDICTIONS = """
SELECT * FROM predictions WHERE chain_id = $1
"""


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
    conditions: list[str] = ["chain_id = $1"]
    params: list[object] = [chain_id]
    idx = 2

    if partition is not None:
        conditions.append(f"partition = ${idx}")
        params.append(partition)
        idx += 1

    if fold_id is not None:
        conditions.append(f"fold_id = ${idx}")
        params.append(fold_id)
        idx += 1

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
    idx = 1
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
            conditions.append(f"{full_col} LIKE ${idx}")
        else:
            conditions.append(f"{full_col} = ${idx}")
        params.append(val)
        idx += 1

    if run_id is not None:
        conditions.append(f"pl.run_id = ${idx}")
        params.append(run_id)
        idx += 1

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    order = f" ORDER BY {prefix}created_at DESC"
    pagination = ""
    if limit is not None:
        pagination += f" LIMIT ${idx}"
        params.append(limit)
        idx += 1
    if offset:
        pagination += f" OFFSET ${idx}"
        params.append(offset)
        idx += 1

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
    nulls = "NULLS LAST"

    conditions: list[str] = []
    params: list[object] = []
    idx = 1

    conditions.append(f"partition = ${idx}")
    params.append(partition)
    idx += 1

    if dataset_name is not None:
        conditions.append(f"dataset_name = ${idx}")
        params.append(dataset_name)
        idx += 1

    where = " WHERE " + " AND ".join(conditions)

    if group_by is not None:
        sql = (
            f"SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {group_by} ORDER BY {metric} {direction} {nulls}) AS _rn "
            f"FROM predictions{where}"
            f") sub WHERE _rn <= ${idx}"
        )
        params.append(n)
    else:
        sql = (
            f"SELECT * FROM predictions{where} "
            f"ORDER BY {metric} {direction} {nulls} "
            f"LIMIT ${idx}"
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
    idx = 1

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
                conditions.append(f"{col} = ${idx}")
                params.append(val[0])
                idx += 1
            else:
                placeholders = ", ".join(f"${idx + i}" for i in range(len(val)))
                conditions.append(f"{col} IN ({placeholders})")
                params.extend(val)
                idx += len(val)
        elif isinstance(val, str) and "%" in val:
            conditions.append(f"{col} LIKE ${idx}")
            params.append(val)
            idx += 1
        else:
            conditions.append(f"{col} = ${idx}")
            params.append(val)
            idx += 1

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
    idx = 1

    if metric is not None:
        conditions.append(f"metric = ${idx}")
        params.append(metric)
        idx += 1

    for col, val in [
        ("run_id", run_id),
        ("pipeline_id", pipeline_id),
        ("dataset_name", dataset_name),
        ("model_class", model_class),
    ]:
        if val is None:
            continue
        conditions.append(f"{col} = ${idx}")
        params.append(val)
        idx += 1

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    direction = "ASC" if ascending else "DESC"
    nulls = "NULLS LAST"

    sql = (
        f"SELECT * FROM v_chain_summary{where} "
        f"ORDER BY {resolved_column} {direction} {nulls} "
        f"LIMIT ${idx}"
    )
    params.append(n)

    return sql, params
