"""Contract snapshot: the WorkspaceStore on-disk SQLite schema.

The workspace SQLite/Parquet schema is a stable 0.9.x contract (read by the
nirs4all-studio backend and by every workspace migration/inspection path).
This file freezes the relational schema that ``WorkspaceStore`` creates on a
fresh workspace: the set of tables, each table's ordered ``(column, type)``
list, and the full DDL of every named index.

A fresh ``WorkspaceStore`` is instantiated in a ``tmp_path`` and its
``store.sqlite`` is introspected via ``sqlite_master`` (table/index names and
the index ``sql`` text) and ``PRAGMA table_info`` (columns). The normalized
snapshot is compared exactly against the frozen ``EXPECTED_SCHEMA`` /
``EXPECTED_INDEXES`` captured from the current code. A diff means the schema
drifted and a migration/version bump is warranted.

Indexes are snapshotted by their full ``CREATE INDEX`` text (the ``sql`` column
of ``sqlite_master``), not just by name. That DDL string includes the
``UNIQUE`` keyword and the exact indexed-column list, so a change that keeps an
index's name but flips its uniqueness or alters its columns — e.g. dropping
``UNIQUE`` from ``idx_predictions_natural_key_v2`` or changing the columns it
spans — is caught. Every index created by ``WorkspaceStore`` is an explicit
named ``CREATE INDEX`` / ``CREATE UNIQUE INDEX`` (uniqueness is enforced via
indexes, not table-level ``UNIQUE`` constraints), so every row in
``sqlite_master`` carries a non-NULL ``sql`` value and no SQLite auto-index
needs special handling.

``PRAGMA user_version`` is currently 0 (no explicit SCHEMA_VERSION is stamped
into the database yet). That fact is itself snapshotted in
``test_user_version_is_zero``: when a real ``SCHEMA_VERSION`` is later
introduced, this test will fail and must be updated to the new value — making
the version bump a deliberate, reviewed event.

Snapshot captured from nirs4all 0.9.1.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# ---------------------------------------------------------------------------
# Frozen schema snapshot (captured from nirs4all 0.9.1).
#
# {table_name: [(column_name, declared_type), ...]} in table_info order.
# ---------------------------------------------------------------------------

EXPECTED_SCHEMA: dict[str, list[tuple[str, str]]] = {
    "artifacts": [
        ("artifact_id", "TEXT"),
        ("artifact_path", "TEXT"),
        ("content_hash", "TEXT"),
        ("operator_class", "TEXT"),
        ("artifact_type", "TEXT"),
        ("format", "TEXT"),
        ("size_bytes", "INTEGER"),
        ("ref_count", "INTEGER"),
        ("chain_path_hash", "TEXT"),
        ("input_data_hash", "TEXT"),
        ("dataset_hash", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ],
    "chains": [
        ("chain_id", "TEXT"),
        ("pipeline_id", "TEXT"),
        ("steps", "TEXT"),
        ("model_step_idx", "INTEGER"),
        ("model_class", "TEXT"),
        ("preprocessings", "TEXT"),
        ("fold_strategy", "TEXT"),
        ("fold_artifacts", "TEXT"),
        ("shared_artifacts", "TEXT"),
        ("branch_path", "TEXT"),
        ("source_index", "INTEGER"),
        ("model_name", "TEXT"),
        ("metric", "TEXT"),
        ("task_type", "TEXT"),
        ("best_params", "TEXT"),
        ("dataset_name", "TEXT"),
        ("cv_val_score", "REAL"),
        ("cv_test_score", "REAL"),
        ("cv_train_score", "REAL"),
        ("cv_fold_count", "INTEGER"),
        ("cv_scores", "TEXT"),
        ("final_test_score", "REAL"),
        ("final_train_score", "REAL"),
        ("final_scores", "TEXT"),
        ("final_agg_test_score", "REAL"),
        ("final_agg_train_score", "REAL"),
        ("final_agg_scores", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ],
    "logs": [
        ("log_id", "TEXT"),
        ("pipeline_id", "TEXT"),
        ("step_idx", "INTEGER"),
        ("operator_class", "TEXT"),
        ("event", "TEXT"),
        ("duration_ms", "INTEGER"),
        ("message", "TEXT"),
        ("details", "TEXT"),
        ("level", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ],
    "pipelines": [
        ("pipeline_id", "TEXT"),
        ("run_id", "TEXT"),
        ("name", "TEXT"),
        ("expanded_config", "TEXT"),
        ("original_template", "TEXT"),
        ("generator_choices", "TEXT"),
        ("dataset_name", "TEXT"),
        ("dataset_hash", "TEXT"),
        ("status", "TEXT"),
        ("created_at", "TIMESTAMP"),
        ("completed_at", "TIMESTAMP"),
        ("best_val", "REAL"),
        ("best_test", "REAL"),
        ("metric", "TEXT"),
        ("duration_ms", "INTEGER"),
        ("error", "TEXT"),
    ],
    "predictions": [
        ("prediction_id", "TEXT"),
        ("pipeline_id", "TEXT"),
        ("chain_id", "TEXT"),
        ("dataset_name", "TEXT"),
        ("model_name", "TEXT"),
        ("model_class", "TEXT"),
        ("fold_id", "TEXT"),
        ("partition", "TEXT"),
        ("val_score", "REAL"),
        ("test_score", "REAL"),
        ("train_score", "REAL"),
        ("metric", "TEXT"),
        ("task_type", "TEXT"),
        ("n_samples", "INTEGER"),
        ("n_features", "INTEGER"),
        ("scores", "TEXT"),
        ("best_params", "TEXT"),
        ("preprocessings", "TEXT"),
        ("branch_id", "INTEGER"),
        ("branch_name", "TEXT"),
        ("exclusion_count", "INTEGER"),
        ("exclusion_rate", "REAL"),
        ("refit_context", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ],
    "projects": [
        ("project_id", "TEXT"),
        ("name", "TEXT"),
        ("description", "TEXT"),
        ("color", "TEXT"),
        ("created_at", "TIMESTAMP"),
        ("updated_at", "TIMESTAMP"),
    ],
    "runs": [
        ("run_id", "TEXT"),
        ("name", "TEXT"),
        ("config", "TEXT"),
        ("datasets", "TEXT"),
        ("status", "TEXT"),
        ("created_at", "TIMESTAMP"),
        ("completed_at", "TIMESTAMP"),
        ("summary", "TEXT"),
        ("error", "TEXT"),
        ("project_id", "TEXT"),
    ],
}

# Frozen index DDL set (captured from nirs4all 0.9.1).
#
# {index_name: create_index_sql}. The CREATE INDEX text is the ``sql`` column
# of ``sqlite_master`` and encodes uniqueness (UNIQUE) plus the exact indexed
# columns, so freezing it catches a uniqueness/columns change to an index that
# keeps its name (e.g. idx_predictions_natural_key_v2).
EXPECTED_INDEXES: dict[str, str] = {
    "idx_artifacts_cache_key": "CREATE INDEX idx_artifacts_cache_key ON artifacts(chain_path_hash, input_data_hash)",
    "idx_artifacts_content_hash": "CREATE INDEX idx_artifacts_content_hash ON artifacts(content_hash)",
    "idx_artifacts_dataset_hash": "CREATE INDEX idx_artifacts_dataset_hash ON artifacts(dataset_hash)",
    "idx_chains_pipeline_id": "CREATE INDEX idx_chains_pipeline_id ON chains(pipeline_id)",
    "idx_logs_pipeline_id": "CREATE INDEX idx_logs_pipeline_id ON logs(pipeline_id)",
    "idx_pipelines_dataset": "CREATE INDEX idx_pipelines_dataset ON pipelines(dataset_name)",
    "idx_pipelines_run_id": "CREATE INDEX idx_pipelines_run_id ON pipelines(run_id)",
    "idx_predictions_chain_id": "CREATE INDEX idx_predictions_chain_id ON predictions(chain_id)",
    "idx_predictions_dataset": "CREATE INDEX idx_predictions_dataset ON predictions(dataset_name)",
    "idx_predictions_natural_key_v2": (
        "CREATE UNIQUE INDEX idx_predictions_natural_key_v2 ON predictions"
        "(pipeline_id, chain_id, fold_id, partition, model_name, branch_id)"
    ),
    "idx_predictions_partition": "CREATE INDEX idx_predictions_partition ON predictions(partition)",
    "idx_predictions_pipeline_id": "CREATE INDEX idx_predictions_pipeline_id ON predictions(pipeline_id)",
    "idx_predictions_val_score": "CREATE INDEX idx_predictions_val_score ON predictions(val_score)",
    "idx_runs_project_id": "CREATE INDEX idx_runs_project_id ON runs(project_id)",
}

# ``PRAGMA user_version`` value stamped into a fresh workspace. Currently 0:
# no explicit SCHEMA_VERSION is written yet. A later commit may set a real
# version — at which point this snapshot (and the test below) must be updated.
EXPECTED_USER_VERSION = 0


def _introspect_schema(db_path: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str], int]:
    """Return (schema, index_ddl, user_version) for a SQLite database.

    ``index_ddl`` maps each non-internal index name to its full ``CREATE INDEX``
    text (``sqlite_master.sql``), which encodes uniqueness and indexed columns.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        index_ddl = dict(cursor.fetchall())

        schema: dict[str, list[tuple[str, str]]] = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = [(row[1], row[2]) for row in cursor.fetchall()]

        cursor.execute("PRAGMA user_version")
        user_version = int(cursor.fetchone()[0])
    finally:
        conn.close()

    return schema, index_ddl, user_version


def test_fresh_store_schema_frozen(tmp_path: Path) -> None:
    """A fresh WorkspaceStore creates exactly the frozen table/column schema."""
    store = WorkspaceStore(tmp_path / "workspace")
    try:
        db_path = tmp_path / "workspace" / "store.sqlite"
        assert db_path.exists(), "WorkspaceStore did not create store.sqlite"
        schema, _, _ = _introspect_schema(db_path)
    finally:
        store.close()

    assert schema == EXPECTED_SCHEMA


def test_fresh_store_indexes_frozen(tmp_path: Path) -> None:
    """A fresh WorkspaceStore creates exactly the frozen indexes, DDL included.

    Comparing the full ``CREATE INDEX`` text (not just names) freezes each
    index's uniqueness and indexed columns. Dropping ``UNIQUE`` from
    ``idx_predictions_natural_key_v2`` or changing the columns of any index
    fails this assertion even though the index name is unchanged.
    """
    store = WorkspaceStore(tmp_path / "workspace")
    try:
        _, index_ddl, _ = _introspect_schema(tmp_path / "workspace" / "store.sqlite")
    finally:
        store.close()

    assert index_ddl == EXPECTED_INDEXES


def test_user_version_is_zero(tmp_path: Path) -> None:
    """A fresh workspace stamps ``PRAGMA user_version`` = 0 (no SCHEMA_VERSION yet).

    If a real schema version is introduced later, update EXPECTED_USER_VERSION;
    the failure is intentional and flags the version bump for review.
    """
    store = WorkspaceStore(tmp_path / "workspace")
    try:
        _, _, user_version = _introspect_schema(tmp_path / "workspace" / "store.sqlite")
    finally:
        store.close()

    assert user_version == EXPECTED_USER_VERSION
