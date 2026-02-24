"""Database-backed workspace storage.

Replaces: ManifestManager, SimulationSaver, PipelineWriter,
          PredictionStorage, ArrayRegistry, WorkspaceExporter,
          PredictionResolver/TargetResolver.

All metadata, configs, logs, chains, and predictions are stored in a DuckDB
database. Binary artifacts (fitted models, transformers) are stored in a flat
content-addressed directory alongside the database file. Human-readable files
(exported bundles, CSVs, charts) are produced only by explicit export operations.

Workspace layout after migration::

    workspace/
        store.duckdb                    # All structured data
        artifacts/                      # Flat content-addressed binaries
            ab/abc123def456.joblib
        exports/                        # User-triggered exports (on demand)
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import logging
import random
import threading
import time
import zipfile
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np
import polars as pl
import yaml

logger = logging.getLogger(__name__)

from nirs4all.pipeline.storage.array_store import ArrayStore
from nirs4all.pipeline.storage.store_queries import (
    CASCADE_DELETE_PIPELINE_CHAINS,
    CASCADE_DELETE_PIPELINE_LOGS,
    CASCADE_DELETE_PIPELINE_PREDICTIONS,
    CASCADE_DELETE_RUN_CHAINS,
    CASCADE_DELETE_RUN_LOGS,
    CASCADE_DELETE_RUN_PIPELINES,
    CASCADE_DELETE_RUN_PREDICTIONS,
    CASCADE_NULLIFY_CHAIN_PREDICTIONS,
    CLEAR_RUN_PROJECT,
    COMPLETE_PIPELINE,
    COMPLETE_RUN,
    DECREMENT_ARTIFACT_REF,
    DELETE_CHAIN,
    DELETE_GC_ARTIFACTS,
    DELETE_PIPELINE,
    DELETE_PREDICTION,
    DELETE_PROJECT,
    DELETE_RUN,
    FAIL_PIPELINE,
    FAIL_RUN,
    FIND_CACHED_ARTIFACT,
    GC_ARTIFACTS,
    GET_ARTIFACT,
    GET_ARTIFACT_BY_HASH,
    GET_CHAIN,
    GET_CHAINS_FOR_PIPELINE,
    GET_PIPELINE,
    GET_PIPELINE_LOG,
    GET_PREDICTION,
    GET_PROJECT,
    GET_PROJECT_BY_NAME,
    GET_RUN,
    GET_RUN_LOG_SUMMARY,
    INCREMENT_ARTIFACT_REF,
    INSERT_ARTIFACT,
    INSERT_ARTIFACT_WITH_CACHE_KEY,
    INSERT_CHAIN,
    INSERT_LOG,
    INSERT_PIPELINE,
    INSERT_PREDICTION,
    INSERT_PROJECT,
    INSERT_RUN,
    INVALIDATE_DATASET_CACHE,
    LIST_PROJECTS,
    SET_RUN_PROJECT,
    UPDATE_ARTIFACT_CACHE_KEY,
    UPDATE_CHAIN_SUMMARY,
    UPDATE_PROJECT,
    build_aggregated_query,
    build_chain_predictions_query,
    build_chain_summary_query,
    build_prediction_query,
    build_top_aggregated_query,
    build_top_chains_query,
    build_top_predictions_query,
)
from nirs4all.pipeline.storage.store_schema import create_schema

_MAX_RETRIES = 8
_BASE_DELAY = 0.15


def _jittered_delay(base: float, attempt: int) -> float:
    """Exponential backoff with random jitter to avoid thundering herd."""
    return base * (2 ** attempt) * (0.5 + random.random() * 0.5)


def _retry_on_lock(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that retries the wrapped method on DuckDB lock errors.

    DuckDB uses process-level file locks that can conflict when another
    process holds the WAL lock (e.g. a lingering child process from
    parallel execution).  Retries with exponential backoff + jitter,
    then raises the last error.

    The first argument (``self``) must be a :class:`WorkspaceStore`.
    """
    import functools

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception = Exception("DuckDB retry exhausted")
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return func(self, *args, **kwargs)
            except duckdb.TransactionException as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = _jittered_delay(_BASE_DELAY, attempt)
                    logger.warning(
                        "DuckDB lock conflict in %s (attempt %d/%d), retrying in %.2fs: %s",
                        func.__name__, attempt + 1, _MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
        logger.error(
            "DuckDB lock conflict persisted after %d retries in %s",
            _MAX_RETRIES, func.__name__,
        )
        raise last_error
    return wrapper

def _to_json(obj: Any) -> str | None:
    """Serialize *obj* to a JSON string, or return ``None``."""
    if obj is None:
        return None
    return json.dumps(obj, default=str)

def _from_json(val: str | None) -> Any:
    """Deserialize a JSON string back to a Python object."""
    if val is None:
        return None
    return json.loads(val)

def _serialize_artifact(obj: Any, fmt: str) -> bytes:
    """Serialize a Python object to bytes using *fmt* as the strategy."""
    if fmt == "joblib":
        import joblib

        buf = io.BytesIO()
        joblib.dump(obj, buf, compress=3)
        return buf.getvalue()
    import pickle

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

def _deserialize_artifact(data: bytes, fmt: str) -> Any:
    """Deserialize bytes produced by :func:`_serialize_artifact`."""
    if fmt == "joblib":
        import joblib

        return joblib.load(io.BytesIO(data))
    import pickle

    return pickle.loads(data)  # noqa: S301

def _format_to_ext(fmt: str) -> str:
    """Map serialisation format to file extension."""
    return {"joblib": "joblib", "pickle": "pkl", "cloudpickle": "pkl"}.get(fmt, "bin")

# ---- Metric direction heuristics ----------------------------------------

_HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({
    "r2", "accuracy", "f1", "precision", "recall",
    "auc", "roc_auc", "balanced_accuracy", "kappa",
    "rpd", "rpiq",
})

def _infer_metric_ascending(metric: str) -> bool:
    """Infer sort direction from a metric name.

    Args:
        metric: Metric name (e.g. ``"rmse"``, ``"r2"``).

    Returns:
        ``True`` if lower is better (ascending sort),
        ``False`` if higher is better (descending sort).
    """
    return metric.lower() not in _HIGHER_IS_BETTER_METRICS

class WorkspaceStore:
    """Database-backed workspace storage.

    Central storage facade for all workspace data: runs, pipelines, chains,
    predictions, prediction arrays, artifacts, and structured execution logs.

    Replaces the file-hierarchy storage composed of ManifestManager,
    SimulationSaver, PipelineWriter, PredictionStorage, ArrayRegistry,
    WorkspaceExporter, and PredictionResolver.

    The store manages three on-disk resources:

    * ``store.duckdb`` -- a single DuckDB database for relational metadata
      (runs, pipelines, chains, predictions, artifacts, logs).
    * ``arrays/`` -- one Parquet file per dataset containing dense
      prediction arrays (y_true, y_pred, y_proba, etc.) with Zstd
      compression.
    * ``artifacts/`` -- a flat, content-addressed directory for binary
      artifacts (fitted models, transformers, etc.).

    All query methods that return tabular data return ``polars.DataFrame``
    via DuckDB's zero-copy Arrow transfer, enabling efficient downstream
    analysis with Polars or conversion to pandas/numpy.

    Args:
        workspace_path: Root directory of the workspace.  The database file
            ``store.duckdb`` and the ``artifacts/`` directory are created
            inside this path.

    Example:
        >>> from pathlib import Path
        >>> store = WorkspaceStore(Path("./workspace"))
        >>> run_id = store.begin_run("experiment_1", config={}, datasets=[])
        >>> store.complete_run(run_id, summary={"total_pipelines": 5})
    """

    def __init__(self, workspace_path: Path) -> None:
        self._workspace_path = Path(workspace_path)
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        db_path = self._workspace_path / "store.duckdb"
        self._conn: duckdb.DuckDBPyConnection | None = duckdb.connect(str(db_path))

        # Disable progress bar for non-interactive usage
        self._conn.execute("PRAGMA enable_progress_bar=false")

        # DuckDB connection tuning
        self._conn.execute("SET memory_limit = '2GB'")
        self._conn.execute("SET threads = 4")
        self._conn.execute("SET checkpoint_threshold = '256MB'")

        # Create schema (auto-migrates legacy prediction_arrays if present)
        create_schema(self._conn, workspace_path=self._workspace_path)

        # Ensure artifacts directory exists
        self._artifacts_dir = self._workspace_path / "artifacts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Parquet-backed array storage
        self._array_store = ArrayStore(self._workspace_path)

    @property
    def workspace_path(self) -> Path:
        """Root directory of the workspace."""
        return self._workspace_path

    @property
    def array_store(self) -> ArrayStore:
        """Parquet-backed array storage."""
        return self._array_store

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> duckdb.DuckDBPyConnection:
        """Return the connection or raise if closed."""
        if self._conn is None:
            raise RuntimeError("WorkspaceStore is closed")
        return self._conn

    def _execute_with_retry(
        self,
        sql: str,
        params: list[object] | None = None,
        *,
        max_retries: int = _MAX_RETRIES,
        base_delay: float = _BASE_DELAY,
    ) -> None:
        """Execute a write statement with retry on transient DuckDB lock errors.

        Raises the last ``TransactionException`` after all retries are
        exhausted.

        Args:
            sql: SQL statement to execute.
            params: Query parameters.
            max_retries: Maximum number of retry attempts.
            base_delay: Initial delay in seconds (doubles each retry with jitter).
        """
        with self._lock:
            conn = self._ensure_open()
            last_error: Exception = Exception("DuckDB retry exhausted")
            for attempt in range(max_retries + 1):
                try:
                    conn.execute(sql, params or [])
                    return
                except duckdb.TransactionException as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = _jittered_delay(base_delay, attempt)
                        logger.warning(
                            "DuckDB lock conflict (attempt %d/%d), retrying in %.2fs: %s",
                            attempt + 1, max_retries, delay, e,
                        )
                        time.sleep(delay)
            raise last_error

    def _safe_execute(self, sql: str, params: list[object] | None = None) -> None:
        """Execute a write statement, suppressing DuckDB lock errors after retries.

        Used for non-critical operations (logging, error recording) that
        must never crash the pipeline.
        """
        with contextlib.suppress(duckdb.TransactionException):
            self._execute_with_retry(sql, params)

    def _fetch_one(self, sql: str, params: list[object] | None = None) -> dict | None:
        """Execute *sql* and return the first row as a dict, or ``None``."""
        with self._lock:
            conn = self._ensure_open()
            result = conn.execute(sql, params or [])
            row = result.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in result.description]
            return dict(zip(columns, row, strict=False))

    def _fetch_pl(self, sql: str, params: list[object] | None = None) -> pl.DataFrame:
        """Execute *sql* and return results as a Polars DataFrame."""
        with self._lock:
            conn = self._ensure_open()
            result = conn.execute(sql, params or [])
            try:
                return result.pl()
            except Exception:
                # Empty result sets may fail .pl(); return empty frame
                return pl.DataFrame()

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Execute a block inside a single DuckDB transaction.

        Batches multiple writes into one ``BEGIN … COMMIT`` to reduce
        lock-contention windows.  Rolls back on any exception.
        """
        with self._lock:
            conn = self._ensure_open()
            conn.execute("BEGIN TRANSACTION")
            try:
                yield
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Close the database connection and release resources.

        Flushes the WAL via ``CHECKPOINT`` before closing so that no
        orphaned ``.wal`` file remains on disk.  Safe to call multiple
        times.  After closing, all other methods will raise
        ``RuntimeError``.
        """
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.execute("CHECKPOINT")
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        """Safety net: close connection if caller forgot to call :meth:`close`."""
        with contextlib.suppress(Exception):
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> WorkspaceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # =====================================================================
    # Run lifecycle
    # =====================================================================

    def begin_run(
        self,
        name: str,
        config: Any,
        datasets: list[dict],
    ) -> str:
        """Create a new run and return its unique identifier.

        A *run* groups one or more pipeline executions that share a common
        configuration and set of target datasets.  The run is created in
        ``"running"`` status.

        Args:
            name: Human-readable name for the run (e.g. ``"protein_sweep"``).
            config: Serialisable run-level configuration (stored as JSON).
                Typically contains cross-validation strategy, metric, and
                global flags.
            datasets: List of dataset metadata dictionaries.  Each entry
                should contain at minimum ``{"name": str, "path": str}``.
                Additional keys (``hash``, ``n_samples``, ``n_features``,
                ``y_stats``) are preserved for provenance tracking.

        Returns:
            A unique run identifier (UUID-based string).
        """
        run_id = str(uuid4())
        self._execute_with_retry(INSERT_RUN, [run_id, name, _to_json(config), _to_json(datasets)])
        return run_id

    def complete_run(self, run_id: str, summary: dict) -> None:
        """Mark a run as successfully completed.

        Updates the run status to ``"completed"`` and stores the provided
        summary (e.g. total pipelines completed, best overall score).

        Args:
            run_id: Identifier returned by :meth:`begin_run`.
            summary: Free-form summary dictionary persisted alongside the
                run record.
        """
        self._execute_with_retry(COMPLETE_RUN, [run_id, _to_json(summary)])

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed.

        Updates the run status to ``"failed"`` and records the error
        message.  Any pipelines still in ``"running"`` status under this
        run should be considered implicitly failed.

        Uses :meth:`_safe_execute` so that a DuckDB lock conflict in the
        error-handling path never masks the original pipeline error.

        Args:
            run_id: Identifier returned by :meth:`begin_run`.
            error: Human-readable error description or traceback excerpt.
        """
        self._safe_execute(FAIL_RUN, [run_id, error])

    # =====================================================================
    # Pipeline lifecycle
    # =====================================================================

    def begin_pipeline(
        self,
        run_id: str,
        name: str,
        expanded_config: Any,
        generator_choices: list,
        dataset_name: str,
        dataset_hash: str,
    ) -> str:
        """Register a new pipeline execution under a run.

        A *pipeline* represents a single expanded configuration (after
        generator expansion) executed on a single dataset.

        Args:
            run_id: Parent run identifier.
            name: Pipeline name (e.g. ``"0001_pls_abc123"``).
            expanded_config: The fully expanded pipeline configuration
                (serialisable to JSON).  This is the resolved configuration
                after all ``_or_``, ``_range_``, ``_log_range_``, and
                ``_cartesian_`` generators have been applied.
            generator_choices: List of generator choices that produced this
                pipeline.  Each entry is a dict like ``{"_or_": value}``
                or ``{"_range_": 18}``.
            dataset_name: Name of the dataset being processed.
            dataset_hash: Content hash of the dataset at execution time,
                enabling later run-compatibility checks.

        Returns:
            A unique pipeline identifier (UUID-based string).
        """
        pipeline_id = str(uuid4())
        self._execute_with_retry(INSERT_PIPELINE, [
            pipeline_id, run_id, name,
            _to_json(expanded_config), _to_json(generator_choices),
            dataset_name, dataset_hash,
        ])
        return pipeline_id

    def complete_pipeline(
        self,
        pipeline_id: str,
        best_val: float,
        best_test: float,
        metric: str,
        duration_ms: int,
    ) -> None:
        """Mark a pipeline execution as successfully completed.

        Args:
            pipeline_id: Identifier returned by :meth:`begin_pipeline`.
            best_val: Best validation score achieved by this pipeline.
            best_test: Corresponding test score for the best validation
                model.
            metric: Name of the metric used for ranking (e.g. ``"rmse"``).
            duration_ms: Total execution time in milliseconds.
        """
        self._execute_with_retry(COMPLETE_PIPELINE, [pipeline_id, best_val, best_test, metric, duration_ms])

    def fail_pipeline(self, pipeline_id: str, error: str) -> None:
        """Mark a pipeline execution as failed and roll back its data.

        Predictions, chains, and logs associated with this pipeline are
        removed.  Artifacts whose reference count drops to zero become
        candidates for garbage collection.

        Uses :meth:`_safe_execute` so that a DuckDB lock conflict in the
        error-handling path never masks the original pipeline error.

        Args:
            pipeline_id: Identifier returned by :meth:`begin_pipeline`.
            error: Human-readable error description.
        """
        try:
            with self._lock:
                self._ensure_open()
                self._decrement_artifact_refs_for_pipeline(pipeline_id)
        except Exception:
            logger.warning("Could not decrement artifact refs for pipeline %s", pipeline_id)
        self._safe_execute(FAIL_PIPELINE, [pipeline_id, error])

    # =====================================================================
    # Chain management
    # =====================================================================

    def save_chain(
        self,
        pipeline_id: str,
        steps: list[dict],
        model_step_idx: int,
        model_class: str,
        preprocessings: str,
        fold_strategy: str,
        fold_artifacts: dict,
        shared_artifacts: dict,
        branch_path: list[int] | None = None,
        source_index: int | None = None,
        dataset_name: str | None = None,
    ) -> str:
        """Store a preprocessing-to-model chain.

        A *chain* captures the complete, ordered sequence of steps
        (transformers and model) that were executed during training,
        together with references to the fitted artifacts for each fold.
        Chains are the unit of export and replay.

        Args:
            pipeline_id: Parent pipeline identifier.
            steps: Ordered list of step descriptors.  Each dict should
                contain at minimum::

                    {
                        "step_idx": int,
                        "operator_class": str,
                        "params": dict,
                        "artifact_id": str | None,
                        "stateless": bool,
                    }

            model_step_idx: Index (within *steps*) of the model step.
            model_class: Fully qualified class name of the model
                (e.g. ``"sklearn.cross_decomposition.PLSRegression"``).
            preprocessings: Short display string summarising the
                preprocessing chain (e.g. ``"SNV>Detr>MinMax"``).
            fold_strategy: Cross-validation fold strategy identifier
                (e.g. ``"per_fold"`` or ``"shared"``).
            fold_artifacts: Mapping from fold identifier (string) to the
                artifact ID of the model trained on that fold.
                Example: ``{"fold_0": "art_abc123", "fold_1": "art_def456"}``.
            shared_artifacts: Mapping from step index (string) to a list
                of artifact IDs of shared (non-fold-specific) fitted objects.
                Example: ``{"0": ["art_scaler_abc"], "3": ["art_snv", "art_sg"]}``.
            branch_path: For branching pipelines, the path of branch
                indices leading to this chain.  ``None`` for non-branching
                pipelines.
            source_index: For multi-source pipelines, the source index
                this chain belongs to.  ``None`` for single-source
                pipelines.
            dataset_name: Dataset name for this chain.  Resolved from
                the parent pipeline if not provided.

        Returns:
            A unique chain identifier (UUID-based string).
        """
        with self._lock:
            conn = self._ensure_open()
            chain_id = str(uuid4())
            # Resolve dataset_name from parent pipeline if not provided
            if dataset_name is None:
                row = conn.execute(
                    "SELECT dataset_name FROM pipelines WHERE pipeline_id = $1",
                    [pipeline_id],
                ).fetchone()
                if row is not None:
                    dataset_name = row[0]
        self._execute_with_retry(INSERT_CHAIN, [
            chain_id, pipeline_id, _to_json(steps), model_step_idx,
            model_class, preprocessings, fold_strategy,
            _to_json(fold_artifacts), _to_json(shared_artifacts),
            _to_json(branch_path), source_index, dataset_name,
        ])
        return chain_id

    def get_chain(self, chain_id: str) -> dict | None:
        """Retrieve a chain by its identifier.

        Returns:
            A dictionary containing all chain fields (steps, model_step_idx,
            fold_artifacts, shared_artifacts, etc.), or ``None`` if the
            chain does not exist.
        """
        row = self._fetch_one(GET_CHAIN, [chain_id])
        if row is None:
            return None
        # Deserialize JSON fields
        for field in ("steps", "fold_artifacts", "shared_artifacts", "branch_path"):
            row[field] = _from_json(row[field])
        return row

    def get_chains_for_pipeline(self, pipeline_id: str) -> pl.DataFrame:
        """List all chains belonging to a pipeline.

        Returns:
            A :class:`polars.DataFrame` with one row per chain, including
            columns for ``chain_id``, ``model_class``, ``preprocessings``,
            ``branch_path``, and ``source_index``.
        """
        return self._fetch_pl(GET_CHAINS_FOR_PIPELINE, [pipeline_id])

    @_retry_on_lock
    def update_chain_summary(self, chain_id: str) -> None:
        """Recompute and store CV/final summary on the chain record.

        Reads all predictions for the given chain, computes averaged
        CV scores and multi-metric averages, extracts final/refit
        scores, and persists the summary columns on the ``chains`` table.

        This method is called automatically after predictions are flushed
        to ensure chain summary data is always up to date.

        Args:
            chain_id: The chain identifier to update.
        """
        with self._lock:
            conn = self._ensure_open()

            # --- CV averages ---
            cv_row = conn.execute(
                "SELECT "
                "  FIRST(model_name) AS model_name, "
                "  FIRST(metric) AS metric, "
                "  FIRST(task_type) AS task_type, "
                "  FIRST(best_params) AS best_params, "
                "  AVG(val_score) AS avg_val, "
                "  AVG(test_score) AS avg_test, "
                "  AVG(train_score) AS avg_train, "
                "  COUNT(DISTINCT fold_id) AS fold_count "
                "FROM predictions "
                "WHERE chain_id = $1 AND refit_context IS NULL",
                [chain_id],
            ).fetchone()

            model_name = cv_row[0] if cv_row else None
            metric = cv_row[1] if cv_row else None
            task_type = cv_row[2] if cv_row else None
            best_params = cv_row[3] if cv_row else None
            avg_val = cv_row[4] if cv_row else None
            avg_test = cv_row[5] if cv_row else None
            avg_train = cv_row[6] if cv_row else None
            fold_count = cv_row[7] if cv_row else 0

            # If no CV predictions exist, try to get model_name etc. from any prediction
            if model_name is None:
                any_row = conn.execute(
                    "SELECT FIRST(model_name), FIRST(metric), FIRST(task_type), FIRST(best_params) "
                    "FROM predictions WHERE chain_id = $1",
                    [chain_id],
                ).fetchone()
                if any_row:
                    model_name = any_row[0]
                    metric = any_row[1]
                    task_type = any_row[2]
                    best_params = any_row[3]

            # --- CV multi-metric averages (cv_scores JSON) ---
            cv_scores_json: str | None = None
            cv_metrics_rows = conn.execute(
                "SELECT partition, scores FROM predictions "
                "WHERE chain_id = $1 AND refit_context IS NULL "
                "AND partition IN ('val', 'test')",
                [chain_id],
            ).fetchall()
            if cv_metrics_rows:
                import json as _json

                partition_scores: dict[str, dict[str, list[float]]] = {}
                for partition, scores_raw in cv_metrics_rows:
                    if not scores_raw:
                        continue
                    scores = _json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
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
                    cv_scores_json = _json.dumps(averaged)

            # --- Final/refit scores ---
            final_row = conn.execute(
                "SELECT test_score, train_score, scores "
                "FROM predictions "
                "WHERE chain_id = $1 AND refit_context IS NOT NULL "
                "AND fold_id = 'final' AND partition = 'test' "
                "LIMIT 1",
                [chain_id],
            ).fetchone()

            final_test = final_row[0] if final_row else None
            final_train = final_row[1] if final_row else None
            final_scores_json = final_row[2] if final_row else None

            conn.execute(UPDATE_CHAIN_SUMMARY, [
                chain_id,
                model_name,
                metric,
                task_type,
                best_params,
                avg_val,
                avg_test,
                avg_train,
                fold_count or 0,
                cv_scores_json,
                final_test,
                final_train,
                final_scores_json,
            ])

    @_retry_on_lock
    def bulk_update_chain_summaries(self, chain_ids: list[str]) -> None:
        """Recompute CV/final summary for multiple chains in bulk SQL.

        Uses set-based aggregation instead of per-chain queries,
        reducing DuckDB round-trips from ``4 × N`` to ``4`` total.

        Args:
            chain_ids: List of chain identifiers to update.
        """
        if not chain_ids:
            return

        with self._lock:
            conn = self._ensure_open()

            # Create a temp table with the chain IDs to update
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _bulk_chain_ids (chain_id VARCHAR)")
            conn.execute("DELETE FROM _bulk_chain_ids")
            conn.executemany(
                "INSERT INTO _bulk_chain_ids VALUES ($1)",
                [(cid,) for cid in chain_ids],
            )

            # --- Bulk CV averages ---
            conn.execute("""
                UPDATE chains SET
                    model_name = COALESCE(chains.model_name, sub.model_name),
                    metric = COALESCE(chains.metric, sub.metric),
                    task_type = COALESCE(chains.task_type, sub.task_type),
                    best_params = COALESCE(chains.best_params, sub.best_params),
                    cv_val_score = sub.avg_val,
                    cv_test_score = sub.avg_test,
                    cv_train_score = sub.avg_train,
                    cv_fold_count = sub.fold_count
                FROM (
                    SELECT chain_id,
                        FIRST(model_name) AS model_name,
                        FIRST(metric) AS metric,
                        FIRST(task_type) AS task_type,
                        FIRST(best_params) AS best_params,
                        AVG(val_score) AS avg_val,
                        AVG(test_score) AS avg_test,
                        AVG(train_score) AS avg_train,
                        COUNT(DISTINCT fold_id) AS fold_count
                    FROM predictions
                    WHERE refit_context IS NULL
                      AND chain_id IN (SELECT chain_id FROM _bulk_chain_ids)
                    GROUP BY chain_id
                ) sub
                WHERE chains.chain_id = sub.chain_id
            """)

            # --- Fallback: get model_name etc. from any prediction for
            #     chains that had no CV predictions ---
            conn.execute("""
                UPDATE chains SET
                    model_name = COALESCE(chains.model_name, sub.model_name),
                    metric = COALESCE(chains.metric, sub.metric),
                    task_type = COALESCE(chains.task_type, sub.task_type),
                    best_params = COALESCE(chains.best_params, sub.best_params)
                FROM (
                    SELECT chain_id,
                        FIRST(model_name) AS model_name,
                        FIRST(metric) AS metric,
                        FIRST(task_type) AS task_type,
                        FIRST(best_params) AS best_params
                    FROM predictions
                    WHERE chain_id IN (
                        SELECT chain_id FROM _bulk_chain_ids
                        WHERE chain_id NOT IN (
                            SELECT chain_id FROM predictions
                            WHERE refit_context IS NULL AND chain_id IS NOT NULL
                            GROUP BY chain_id
                        )
                    )
                    GROUP BY chain_id
                ) sub
                WHERE chains.chain_id = sub.chain_id
            """)

            # --- Bulk final/refit scores ---
            conn.execute("""
                UPDATE chains SET
                    final_test_score = sub.test_score,
                    final_train_score = sub.train_score,
                    final_scores = sub.scores
                FROM (
                    SELECT chain_id, test_score, train_score, scores
                    FROM predictions
                    WHERE refit_context IS NOT NULL AND fold_id = 'final'
                      AND partition = 'test'
                      AND chain_id IN (SELECT chain_id FROM _bulk_chain_ids)
                ) sub
                WHERE chains.chain_id = sub.chain_id
            """)

            # --- Bulk cv_scores (averaged multi-metric JSON) ---
            # DuckDB JSON aggregation is limited, so we use a single
            # query to fetch all rows and compute in Python once.
            import json as _json

            rows = conn.execute(
                "SELECT chain_id, partition, scores FROM predictions "
                "WHERE refit_context IS NULL "
                "AND partition IN ('val', 'test') "
                "AND chain_id IN (SELECT chain_id FROM _bulk_chain_ids)",
            ).fetchall()

            if rows:
                per_chain: dict[str, dict[str, dict[str, list[float]]]] = {}
                for chain_id, partition, scores_raw in rows:
                    if not scores_raw:
                        continue
                    scores = _json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
                    if not isinstance(scores, dict):
                        continue
                    inner = scores.get(partition, scores)
                    if not isinstance(inner, dict):
                        continue
                    chain_data = per_chain.setdefault(chain_id, {})
                    part_data = chain_data.setdefault(partition, {})
                    for metric_name, val in inner.items():
                        if isinstance(val, (int, float)):
                            part_data.setdefault(metric_name, []).append(float(val))

                for chain_id, partition_scores in per_chain.items():
                    averaged: dict[str, dict[str, float]] = {}
                    for part, metrics in partition_scores.items():
                        averaged[part] = {m: round(sum(vs) / len(vs), 6) for m, vs in metrics.items() if vs}
                    if averaged:
                        conn.execute(
                            "UPDATE chains SET cv_scores = $2 WHERE chain_id = $1",
                            [chain_id, _json.dumps(averaged)],
                        )

            conn.execute("DROP TABLE IF EXISTS _bulk_chain_ids")

    # =====================================================================
    # Prediction storage
    # =====================================================================

    @_retry_on_lock
    def save_prediction(
        self,
        pipeline_id: str,
        chain_id: str,
        dataset_name: str,
        model_name: str,
        model_class: str,
        fold_id: str,
        partition: str,
        val_score: float | None,
        test_score: float | None,
        train_score: float | None,
        metric: str,
        task_type: str,
        n_samples: int,
        n_features: int,
        scores: dict,
        best_params: dict,
        branch_id: int | None,
        branch_name: str | None,
        exclusion_count: int,
        exclusion_rate: float,
        preprocessings: str = "",
        prediction_id: str | None = None,
        refit_context: str | None = None,
    ) -> str:
        """Store a single prediction record.

        A prediction captures the result of evaluating one model on one
        fold/partition combination: the scalar scores, the model identity,
        the preprocessing summary, and links back to the pipeline and chain
        that produced it.

        Arrays (y_true, y_pred, etc.) are stored separately via
        :attr:`array_store`.

        Args:
            pipeline_id: Parent pipeline identifier.
            chain_id: Chain that produced this prediction.
            dataset_name: Name of the dataset.
            model_name: Short model name (e.g. ``"PLSRegression"``).
            model_class: Fully qualified model class name.
            fold_id: Fold identifier (e.g. ``"fold_0"``, ``"avg"``,
                ``"final"`` for refit).
            partition: Data partition (``"train"``, ``"val"``, ``"test"``).
            val_score: Validation score (primary ranking metric). None for refit entries.
            test_score: Test score (for reporting). None if not available.
            train_score: Training score (for overfitting diagnostics). None if not available.
            metric: Name of the metric (e.g. ``"rmse"``, ``"r2"``).
            task_type: ``"regression"`` or ``"classification"``.
            n_samples: Number of samples in this partition.
            n_features: Number of features (wavelengths).
            scores: Nested dict of all computed scores per partition.
                Example: ``{"val": {"rmse": 0.12, "r2": 0.95}, ...}``.
            best_params: Best hyperparameters found (if tuning was used).
            branch_id: Branch index (0-based) or ``None``.
            branch_name: Human-readable branch name or ``None``.
            exclusion_count: Number of samples excluded by outlier filters.
            exclusion_rate: Fraction of samples excluded (0.0 -- 1.0).
            preprocessings: Short display string for the preprocessing
                chain applied before this model.
            prediction_id: Optional explicit prediction ID.
            refit_context: Refit context label. ``None`` for CV entries,
                ``"standalone"`` for standalone refit,
                ``"stacking"`` for stacking-context refit.

        Returns:
            A unique prediction identifier (UUID-based string).
        """
        with self._lock:
            conn = self._ensure_open()

            # Natural-key lookup includes branch_id to distinguish
            # predictions from different branches with the same model name.
            if branch_id is not None:
                existing = conn.execute(
                    "SELECT prediction_id FROM predictions "
                    "WHERE pipeline_id = $1 AND chain_id = $2 AND fold_id = $3 AND partition = $4 "
                    "AND model_name = $5 AND branch_id = $6 "
                    "LIMIT 1",
                    [pipeline_id, chain_id, fold_id, partition, model_name, branch_id],
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT prediction_id FROM predictions "
                    "WHERE pipeline_id = $1 AND chain_id = $2 AND fold_id = $3 AND partition = $4 "
                    "AND model_name = $5 AND branch_id IS NULL "
                    "LIMIT 1",
                    [pipeline_id, chain_id, fold_id, partition, model_name],
                ).fetchone()

            if existing is not None:
                # Upsert: delete old row and re-insert.
                prediction_id = str(existing[0])
                self._array_store.delete_batch({prediction_id})
                conn.execute(DELETE_PREDICTION, [prediction_id])
            elif prediction_id is not None:
                # Explicit prediction_id provided — guard against PK collision
                # from a different natural key (e.g. different chain/branch).
                pk_existing = conn.execute(
                    "SELECT 1 FROM predictions WHERE prediction_id = $1",
                    [prediction_id],
                ).fetchone()
                if pk_existing is not None:
                    self._array_store.delete_batch({prediction_id})
                    conn.execute(DELETE_PREDICTION, [prediction_id])

            if prediction_id is None:
                prediction_id = str(uuid4())

            conn.execute(INSERT_PREDICTION, [
                prediction_id, pipeline_id, chain_id, dataset_name, model_name,
                model_class, fold_id, partition, val_score, test_score, train_score,
                metric, task_type, n_samples, n_features,
                _to_json(scores), _to_json(best_params),
                preprocessings, branch_id, branch_name,
                exclusion_count, exclusion_rate, refit_context,
            ])
            return prediction_id

    # =====================================================================
    # Artifact storage
    # =====================================================================

    @_retry_on_lock
    def save_artifact(
        self,
        obj: Any,
        operator_class: str,
        artifact_type: str,
        format: str,
    ) -> str:
        """Persist a binary artifact (fitted model or transformer).

        The object is serialised to disk in the ``artifacts/`` directory
        using content-addressed storage.  If an identical artifact already
        exists (same content hash), the existing entry is reused and its
        reference count incremented.

        Args:
            obj: The Python object to persist (e.g. a fitted
                ``sklearn.preprocessing.StandardScaler``).
            operator_class: Fully qualified class name of the operator
                that produced this artifact.
            artifact_type: Category label (``"model"``, ``"transformer"``,
                ``"scaler"``, etc.).
            format: Serialisation format hint (``"joblib"``,
                ``"cloudpickle"``, ``"keras_h5"``, etc.).

        Returns:
            A unique artifact identifier.  If the content already existed,
            the *same* identifier is returned (content-addressed
            deduplication).
        """
        # Serialize outside the lock (CPU-bound, no DB access)
        data = _serialize_artifact(obj, format)
        content_hash = hashlib.sha256(data).hexdigest()

        with self._lock:
            conn = self._ensure_open()

            # Check for existing artifact with same hash
            existing = self._fetch_one(GET_ARTIFACT_BY_HASH, [content_hash])
            if existing is not None:
                conn.execute(INCREMENT_ARTIFACT_REF, [existing["artifact_id"]])
                return str(existing["artifact_id"])

            # New artifact
            artifact_id = str(uuid4())
            ext = _format_to_ext(format)
            shard = content_hash[:2]
            relative_path = f"{shard}/{content_hash}.{ext}"
            absolute_path = self._artifacts_dir / shard / f"{content_hash}.{ext}"
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_bytes(data)

            conn.execute(INSERT_ARTIFACT, [
                artifact_id, relative_path, content_hash, operator_class,
                artifact_type, format, len(data),
            ])
            return artifact_id

    def register_existing_artifact(
        self,
        artifact_id: str,
        path: str,
        content_hash: str,
        operator_class: str,
        artifact_type: str,
        format: str,
        size_bytes: int,
    ) -> str:
        """Register an artifact that was already saved to disk.

        This is used to bridge the ArtifactRegistry (which persists files
        during pipeline execution) with the DuckDB ``artifacts`` table so
        that :meth:`load_artifact` and chain replay can find them.

        If an artifact with the same *artifact_id* already exists the call
        is silently ignored (idempotent).

        Args:
            artifact_id: Artifact identifier (may be V3 format).
            path: Relative path within the ``artifacts/`` directory.
            content_hash: SHA-256 hex digest of the serialised content.
            operator_class: Class name of the operator that produced it.
            artifact_type: Category label (``"model"``, ``"transformer"``, …).
            format: Serialisation format (``"joblib"``, ``"cloudpickle"``, …).
            size_bytes: Size of the serialised content in bytes.

        Returns:
            The *artifact_id* that was registered.
        """
        with self._lock:
            conn = self._ensure_open()

            # Skip if already registered
            existing = self._fetch_one(GET_ARTIFACT, [artifact_id])
            if existing is not None:
                return artifact_id

            conn.execute(INSERT_ARTIFACT, [
                artifact_id, path, content_hash, operator_class,
                artifact_type, format, size_bytes,
            ])
            return artifact_id

    def load_artifact(self, artifact_id: str) -> Any:
        """Load a binary artifact from disk.

        Args:
            artifact_id: Identifier returned by :meth:`save_artifact`.

        Returns:
            The deserialised Python object.

        Raises:
            FileNotFoundError: If the artifact file is missing from disk.
            KeyError: If the artifact identifier is unknown.
        """
        row = self._fetch_one(GET_ARTIFACT, [artifact_id])
        if row is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        path = self._artifacts_dir / row["artifact_path"]
        if not path.exists():
            raise FileNotFoundError(f"Artifact file missing: {path}")
        data = path.read_bytes()
        return _deserialize_artifact(data, row["format"])

    def get_artifact_path(self, artifact_id: str) -> Path:
        """Return the filesystem path of a stored artifact.

        Useful when external tools need direct file access (e.g. for
        building a ``.n4a`` bundle ZIP).

        Args:
            artifact_id: Identifier returned by :meth:`save_artifact`.

        Returns:
            Absolute path to the artifact file.

        Raises:
            KeyError: If the artifact identifier is unknown.
        """
        row = self._fetch_one(GET_ARTIFACT, [artifact_id])
        if row is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return self._artifacts_dir / str(row["artifact_path"])

    # =====================================================================
    # Cross-run artifact caching
    # =====================================================================

    @_retry_on_lock
    def save_artifact_with_cache_key(
        self,
        obj: Any,
        operator_class: str,
        artifact_type: str,
        format: str,
        chain_path_hash: str,
        input_data_hash: str,
        dataset_hash: str,
    ) -> str:
        """Persist a binary artifact with cross-run cache key metadata.

        Behaves like :meth:`save_artifact` but also stores a
        ``(chain_path_hash, input_data_hash)`` cache key alongside the
        artifact record.  This enables :meth:`find_cached_artifact` to
        locate previously computed artifacts across runs.

        Args:
            obj: The Python object to persist.
            operator_class: Fully qualified class name of the operator.
            artifact_type: Category label (``"model"``, ``"transformer"``,
                etc.).
            format: Serialisation format hint.
            chain_path_hash: Hash identifying the chain of preprocessing
                steps up to (and including) this step.
            input_data_hash: Hash of the input data fed to this step.
            dataset_hash: Content hash of the source dataset, used for
                cache invalidation when the dataset changes.

        Returns:
            A unique artifact identifier.
        """
        data = _serialize_artifact(obj, format)
        content_hash = hashlib.sha256(data).hexdigest()

        with self._lock:
            conn = self._ensure_open()

            existing = self._fetch_one(GET_ARTIFACT_BY_HASH, [content_hash])
            if existing is not None:
                conn.execute(INCREMENT_ARTIFACT_REF, [existing["artifact_id"]])
                # Update cache key on the existing record
                conn.execute(UPDATE_ARTIFACT_CACHE_KEY, [
                    existing["artifact_id"],
                    chain_path_hash,
                    input_data_hash,
                    dataset_hash,
                ])
                return str(existing["artifact_id"])

            artifact_id = str(uuid4())
            ext = _format_to_ext(format)
            shard = content_hash[:2]
            relative_path = f"{shard}/{content_hash}.{ext}"
            absolute_path = self._artifacts_dir / shard / f"{content_hash}.{ext}"
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_bytes(data)

            conn.execute(INSERT_ARTIFACT_WITH_CACHE_KEY, [
                artifact_id, relative_path, content_hash, operator_class,
                artifact_type, format, len(data),
                chain_path_hash, input_data_hash, dataset_hash,
            ])
            return artifact_id

    def update_artifact_cache_key(
        self,
        artifact_id: str,
        chain_path_hash: str,
        input_data_hash: str,
        dataset_hash: str,
    ) -> None:
        """Attach cross-run cache key metadata to an existing artifact.

        This is used to retrofit cache keys onto artifacts that were
        saved with :meth:`save_artifact` (without cache keys) during
        pipeline execution.

        Args:
            artifact_id: Identifier of the artifact to update.
            chain_path_hash: Hash identifying the chain of steps.
            input_data_hash: Hash of the input data.
            dataset_hash: Content hash of the source dataset.
        """
        with self._lock:
            conn = self._ensure_open()
            conn.execute(UPDATE_ARTIFACT_CACHE_KEY, [
                artifact_id, chain_path_hash, input_data_hash, dataset_hash,
            ])

    def find_cached_artifact(
        self,
        chain_path_hash: str,
        input_data_hash: str,
    ) -> str | None:
        """Look up a previously cached artifact by its cache key.

        Searches the ``artifacts`` table for a record matching the
        composite key ``(chain_path_hash, input_data_hash)`` with a
        positive reference count.

        Args:
            chain_path_hash: Hash identifying the chain of steps.
            input_data_hash: Hash of the input data.

        Returns:
            The artifact identifier if a cached entry exists, or
            ``None`` on cache miss.
        """
        row = self._fetch_one(FIND_CACHED_ARTIFACT, [chain_path_hash, input_data_hash])
        if row is None:
            return None
        return str(row["artifact_id"])

    def invalidate_dataset_cache(self, dataset_hash: str) -> int:
        """Invalidate all cached artifacts for a dataset.

        Clears the ``chain_path_hash`` and ``input_data_hash`` fields on
        every artifact whose ``dataset_hash`` matches the given value.
        The artifact files and records remain intact -- only the cache
        keys are removed, preventing future cache hits.

        This should be called when the dataset content has changed (e.g.
        the source file was modified) to ensure stale cached results are
        not reused.

        Args:
            dataset_hash: Content hash of the dataset whose cached
                artifacts should be invalidated.

        Returns:
            Number of artifact records that were invalidated.
        """
        with self._lock:
            conn = self._ensure_open()
            # Count matching rows before invalidation
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM artifacts "
                "WHERE dataset_hash = $1 AND chain_path_hash IS NOT NULL",
                [dataset_hash],
            ).fetchone()
            count = count_row[0] if count_row else 0
            if count > 0:
                conn.execute(INVALIDATE_DATASET_CACHE, [dataset_hash])
            return count

    # =====================================================================
    # Structured logging
    # =====================================================================

    def log_step(
        self,
        pipeline_id: str,
        step_idx: int,
        operator_class: str,
        event: str,
        duration_ms: int | None = None,
        message: str | None = None,
        details: dict | None = None,
        level: str = "info",
    ) -> None:
        """Record a structured log entry for a pipeline step.

        Step logs enable post-hoc analysis of execution timelines,
        per-step durations, warnings, and errors without parsing text
        log files.

        Args:
            pipeline_id: Pipeline the step belongs to.
            step_idx: Zero-based step index in the pipeline.
            operator_class: Fully qualified class name of the operator
                executed in this step.
            event: Event name (e.g. ``"start"``, ``"end"``, ``"skip"``,
                ``"warning"``, ``"error"``).
            duration_ms: Step execution time in milliseconds (typically
                set on ``"end"`` events).
            message: Optional human-readable message.
            details: Optional structured details (stored as JSON).
            level: Log level (``"debug"``, ``"info"``, ``"warning"``,
                ``"error"``).
        """
        log_id = str(uuid4())
        self._safe_execute(INSERT_LOG, [
            log_id, pipeline_id, step_idx, operator_class, event,
            duration_ms, message, _to_json(details), level,
        ])

    # =====================================================================
    # Queries -- Runs
    # =====================================================================

    def get_run(self, run_id: str) -> dict | None:
        """Retrieve a single run record.

        Returns:
            A dictionary with all run fields (``run_id``, ``name``,
            ``status``, ``config``, ``datasets``, ``summary``,
            ``created_at``, ``completed_at``, ``error``), or ``None``
            if the run does not exist.
        """
        row = self._fetch_one(GET_RUN, [run_id])
        if row is None:
            return None
        for field in ("config", "datasets", "summary"):
            row[field] = _from_json(row[field])
        return row

    def list_runs(
        self,
        status: str | None = None,
        dataset: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> pl.DataFrame:
        """List runs with optional filtering and pagination.

        Args:
            status: Filter by run status (``"running"``, ``"completed"``,
                ``"failed"``).  ``None`` returns all statuses.
            dataset: Filter to runs that reference this dataset name in
                their ``datasets`` list.
            project_id: Filter by project.  ``None`` returns all projects.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip (for pagination).

        Returns:
            A :class:`polars.DataFrame` with one row per matching run,
            ordered by ``created_at`` descending (newest first).
        """
        conditions: list[str] = []
        params: list[object] = []
        idx = 1

        if status is not None:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        if dataset is not None:
            # JSON contains check -- search for dataset name in the JSON array
            conditions.append(f"datasets::VARCHAR LIKE ${idx}")
            params.append(f"%{dataset}%")
            idx += 1

        if project_id is not None:
            conditions.append(f"project_id = ${idx}")
            params.append(project_id)
            idx += 1

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        sql = f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1}"
        params.extend([limit, offset])

        return self._fetch_pl(sql, params)

    # =====================================================================
    # Queries -- Pipelines
    # =====================================================================

    def get_pipeline(self, pipeline_id: str) -> dict | None:
        """Retrieve a single pipeline record.

        Returns:
            A dictionary with all pipeline fields (``pipeline_id``,
            ``run_id``, ``name``, ``status``, ``expanded_config``,
            ``generator_choices``, ``dataset_name``, ``dataset_hash``,
            ``best_val``, ``best_test``, ``metric``, ``duration_ms``,
            ``created_at``, ``completed_at``, ``error``), or ``None``
            if the pipeline does not exist.
        """
        row = self._fetch_one(GET_PIPELINE, [pipeline_id])
        if row is None:
            return None
        for field in ("expanded_config", "generator_choices"):
            row[field] = _from_json(row[field])
        return row

    def list_pipelines(
        self,
        run_id: str | None = None,
        dataset_name: str | None = None,
    ) -> pl.DataFrame:
        """List pipelines with optional filtering.

        Args:
            run_id: Filter by parent run.  ``None`` returns pipelines
                from all runs.
            dataset_name: Filter by dataset name.  ``None`` returns
                pipelines for all datasets.

        Returns:
            A :class:`polars.DataFrame` with one row per matching
            pipeline, ordered by ``created_at`` descending.
        """
        conditions: list[str] = []
        params: list[object] = []
        idx = 1

        if run_id is not None:
            conditions.append(f"run_id = ${idx}")
            params.append(run_id)
            idx += 1
        if dataset_name is not None:
            conditions.append(f"dataset_name = ${idx}")
            params.append(dataset_name)
            idx += 1

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        sql = f"SELECT * FROM pipelines{where} ORDER BY created_at DESC"
        return self._fetch_pl(sql, params)

    # =====================================================================
    # Queries -- Predictions
    # =====================================================================

    def get_prediction(
        self,
        prediction_id: str,
        load_arrays: bool = False,
    ) -> dict | None:
        """Retrieve a single prediction record.

        Args:
            prediction_id: Unique prediction identifier.
            load_arrays: If ``True``, the returned dictionary includes
                ``y_true``, ``y_pred``, ``y_proba``, ``sample_indices``,
                and ``weights`` as :class:`numpy.ndarray` objects
                (loaded from the Parquet array store).
                If ``False`` (default), arrays are omitted for speed.

        Returns:
            Prediction dictionary or ``None`` if not found.
        """
        row = self._fetch_one(GET_PREDICTION, [prediction_id])

        if row is None:
            return None

        for field in ("scores", "best_params"):
            row[field] = _from_json(row[field])

        if load_arrays:
            arrays = self._array_store.load_single(prediction_id, dataset_name=row.get("dataset_name"))
            if arrays:
                for field in ("y_true", "y_pred", "y_proba", "sample_indices", "weights"):
                    row[field] = arrays.get(field)

        return row

    def query_predictions(
        self,
        dataset_name: str | None = None,
        model_class: str | None = None,
        partition: str | None = None,
        fold_id: str | None = None,
        branch_id: int | None = None,
        pipeline_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> pl.DataFrame:
        """Query predictions with flexible filtering.

        All filter arguments are optional.  When multiple filters are
        specified, they are combined with ``AND`` semantics.

        Args:
            dataset_name: Filter by dataset name.
            model_class: Filter by model class name (supports SQL ``LIKE``
                patterns, e.g. ``"PLS%"``).
            partition: Filter by partition (``"train"``, ``"val"``,
                ``"test"``).
            fold_id: Filter by fold identifier.
            branch_id: Filter by branch index.
            pipeline_id: Filter by parent pipeline.
            run_id: Filter by parent run (joins through pipelines).
            limit: Maximum number of rows.  ``None`` for unlimited.
            offset: Number of rows to skip.

        Returns:
            A :class:`polars.DataFrame` with one row per matching
            prediction (arrays are *not* included; use
            :meth:`get_prediction` with ``load_arrays=True`` for that).
        """
        sql, params = build_prediction_query(
            dataset_name=dataset_name,
            model_class=model_class,
            partition=partition,
            fold_id=fold_id,
            branch_id=branch_id,
            pipeline_id=pipeline_id,
            run_id=run_id,
            limit=limit,
            offset=offset,
        )
        return self._fetch_pl(sql, params)

    def top_predictions(
        self,
        n: int,
        metric: str = "val_score",
        ascending: bool = True,
        partition: str = "val",
        dataset_name: str | None = None,
        group_by: str | None = None,
    ) -> pl.DataFrame:
        """Return the top-N predictions ranked by a score column.

        This is the primary ranking interface for finding the best
        models across a workspace.

        Args:
            n: Number of top predictions to return.  When ``group_by``
                is set, returns top *n* per group.
            metric: Column to rank by (``"val_score"``, ``"test_score"``,
                ``"train_score"``).
            ascending: If ``True`` (default), lower values rank higher
                (appropriate for error metrics like RMSE).  Set to
                ``False`` for higher-is-better metrics like R2.
            partition: Only consider predictions from this partition.
            dataset_name: Optional dataset filter.
            group_by: Optional grouping column (e.g. ``"model_class"``,
                ``"dataset_name"``).  When set, the result contains
                top *n* rows per distinct value of this column.

        Returns:
            A :class:`polars.DataFrame` with the top predictions.
        """
        sql, params = build_top_predictions_query(
            n=n,
            metric=metric,
            ascending=ascending,
            partition=partition,
            dataset_name=dataset_name,
            group_by=group_by,
        )
        df = self._fetch_pl(sql, params)
        # Drop internal ranking column if present
        if "_rn" in df.columns:
            df = df.drop("_rn")
        return df

    # =====================================================================
    # Queries -- Aggregated Predictions
    # =====================================================================

    def query_aggregated_predictions(
        self,
        run_id: str | None = None,
        pipeline_id: str | None = None,
        chain_id: str | None = None,
        dataset_name: str | None = None,
        model_class: str | None = None,
        metric: str | None = None,
        score_scope: str = "cv",
    ) -> pl.DataFrame:
        """Query the aggregated predictions VIEW with optional filters.

        Returns one row per ``(chain_id, metric, dataset_name)``
        combination, with fold/partition counts, score aggregates
        (min/max/avg per partition type), and lists of prediction IDs
        and fold IDs.

        All filter arguments are optional and combined with ``AND``.

        Args:
            run_id: Filter by parent run.
            pipeline_id: Filter by parent pipeline.
            chain_id: Filter by chain.
            dataset_name: Filter by dataset name.
            model_class: Filter by model class (supports SQL ``LIKE``).
            metric: Filter by metric name.
            score_scope: Which predictions to include.
                ``'cv'`` (default) uses CV-only entries,
                ``'all'`` includes both CV and refit entries,
                ``'final'`` includes only refit entries.

        Returns:
            A :class:`polars.DataFrame` with one row per aggregated
            prediction entry.
        """
        sql, params = build_aggregated_query(
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name=dataset_name,
            model_class=model_class,
            metric=metric,
            score_scope=score_scope,
        )
        return self._fetch_pl(sql, params)

    def get_chain_predictions(
        self,
        chain_id: str,
        partition: str | None = None,
        fold_id: str | None = None,
    ) -> pl.DataFrame:
        """Get individual prediction rows for a chain.

        Used for drill-down from the aggregated view to partition-level
        and fold-level prediction details.

        Args:
            chain_id: Chain identifier (required).
            partition: Optional partition filter (``"train"``, ``"val"``,
                ``"test"``).
            fold_id: Optional fold identifier filter.

        Returns:
            A :class:`polars.DataFrame` with one row per matching
            prediction, ordered by ``(partition, fold_id)``.
        """
        sql, params = build_chain_predictions_query(
            chain_id=chain_id,
            partition=partition,
            fold_id=fold_id,
        )
        return self._fetch_pl(sql, params)

    def query_top_aggregated_predictions(
        self,
        metric: str,
        n: int = 10,
        score_column: str = "avg_val_score",
        ascending: bool | None = None,
        score_scope: str = "cv",
        **filters: Any,
    ) -> pl.DataFrame:
        """Query aggregated predictions ranked by best score for a metric.

        Uses metric-direction heuristics (lower-is-better for error
        metrics like RMSE, higher-is-better for score metrics like R²)
        when *ascending* is not explicitly provided.

        Args:
            metric: Metric name (e.g. ``"rmse"``, ``"r2"``).
            n: Number of top results to return.
            score_column: Aggregation column to sort by (default
                ``"avg_val_score"``).
            ascending: Sort direction.  If ``None`` (default), inferred
                from *metric* using standard heuristics.
            score_scope: Which predictions to include.
                ``'cv'`` (default) uses CV-only entries,
                ``'all'`` includes both CV and refit entries,
                ``'final'`` includes only refit entries.
            **filters: Additional filters passed to the query builder
                (``run_id``, ``pipeline_id``, ``dataset_name``,
                ``model_class``).

        Returns:
            A :class:`polars.DataFrame` with the top *n* aggregated
            predictions.
        """
        if ascending is None:
            ascending = _infer_metric_ascending(metric)

        sql, params = build_top_aggregated_query(
            metric=metric,
            n=n,
            score_column=score_column,
            ascending=ascending,
            run_id=filters.get("run_id"),
            pipeline_id=filters.get("pipeline_id"),
            dataset_name=filters.get("dataset_name"),
            model_class=filters.get("model_class"),
            score_scope=score_scope,
        )
        return self._fetch_pl(sql, params)

    # =====================================================================
    # Queries -- Chain Summaries (v_chain_summary VIEW)
    # =====================================================================

    def query_chain_summaries(
        self,
        run_id: str | list[str] | None = None,
        pipeline_id: str | list[str] | None = None,
        chain_id: str | list[str] | None = None,
        dataset_name: str | list[str] | None = None,
        model_class: str | list[str] | None = None,
        metric: str | None = None,
        task_type: str | None = None,
    ) -> pl.DataFrame:
        """Query chain summaries with optional filters.

        Returns one row per chain with CV averages, final/refit scores,
        multi-metric JSON, and chain metadata from ``v_chain_summary``.

        All filter arguments are optional and combined with ``AND``.
        String filters accept a single value or a list of values (``IN``).

        Args:
            run_id: Filter by parent run(s).
            pipeline_id: Filter by parent pipeline(s).
            chain_id: Filter by chain(s).
            dataset_name: Filter by dataset name(s).
            model_class: Filter by model class(es) (supports SQL ``LIKE``).
            metric: Filter by metric name.
            task_type: Filter by task type (regression/classification).

        Returns:
            A :class:`polars.DataFrame` with one row per chain.
        """
        sql, params = build_chain_summary_query(
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name=dataset_name,
            model_class=model_class,
            metric=metric,
            task_type=task_type,
        )
        return self._fetch_pl(sql, params)

    def query_top_chains(
        self,
        metric: str | None = None,
        n: int = 10,
        score_column: str = "cv_val_score",
        ascending: bool | None = None,
        **filters: Any,
    ) -> pl.DataFrame:
        """Query chain summaries ranked by score.

        Uses metric-direction heuristics (lower-is-better for error
        metrics like RMSE, higher-is-better for score metrics like R2)
        when *ascending* is not explicitly provided.

        Args:
            metric: Optional metric name filter.
            n: Number of top results.
            score_column: Column to sort by (default ``"cv_val_score"``).
            ascending: Sort direction.  Inferred from *metric* if ``None``.
            **filters: Additional filters (``run_id``, ``pipeline_id``,
                ``dataset_name``, ``model_class``).

        Returns:
            A :class:`polars.DataFrame` with the top *n* chain summaries.
        """
        if ascending is None:
            ascending = _infer_metric_ascending(metric) if metric else True

        sql, params = build_top_chains_query(
            metric=metric,
            n=n,
            score_column=score_column,
            ascending=ascending,
            run_id=filters.get("run_id"),
            pipeline_id=filters.get("pipeline_id"),
            dataset_name=filters.get("dataset_name"),
            model_class=filters.get("model_class"),
        )
        return self._fetch_pl(sql, params)

    # =====================================================================
    # Queries -- Logs
    # =====================================================================

    def get_pipeline_log(self, pipeline_id: str) -> pl.DataFrame:
        """Retrieve all log entries for a pipeline.

        Returns:
            A :class:`polars.DataFrame` ordered by ``(step_idx, timestamp)``
            with columns ``step_idx``, ``operator_class``, ``event``,
            ``duration_ms``, ``message``, ``details``, ``level``,
            ``timestamp``.
        """
        return self._fetch_pl(GET_PIPELINE_LOG, [pipeline_id])

    def get_run_log_summary(self, run_id: str) -> pl.DataFrame:
        """Aggregate log entries across all pipelines of a run.

        Produces a summary suitable for dashboard display: per-pipeline
        total duration, step counts, warning/error counts.

        Returns:
            A :class:`polars.DataFrame` with one row per pipeline in the
            run.
        """
        return self._fetch_pl(GET_RUN_LOG_SUMMARY, [run_id])

    # =====================================================================
    # Projects
    # =====================================================================

    def create_project(
        self, name: str, description: str = "", color: str = "#14b8a6"
    ) -> str:
        """Create a new project and return its ID."""
        with self._lock:
            conn = self._ensure_open()
            project_id = str(uuid4())
            conn.execute(INSERT_PROJECT, [project_id, name, description, color])
            return project_id

    def list_projects(self) -> pl.DataFrame:
        """List all projects ordered by creation date (newest first)."""
        return self._fetch_pl(LIST_PROJECTS, [])

    def get_project(self, project_id: str) -> dict | None:
        """Retrieve a single project by ID."""
        return self._fetch_one(GET_PROJECT, [project_id])

    def get_project_by_name(self, name: str) -> dict | None:
        """Retrieve a project by its unique name."""
        return self._fetch_one(GET_PROJECT_BY_NAME, [name])

    def update_project(
        self, project_id: str, name: str, description: str = "", color: str = "#14b8a6"
    ) -> None:
        """Update project attributes."""
        with self._lock:
            conn = self._ensure_open()
            conn.execute(UPDATE_PROJECT, [project_id, name, description, color])

    def delete_project(self, project_id: str) -> None:
        """Delete a project.  Runs referencing it will have ``project_id`` set to NULL."""
        with self._lock:
            conn = self._ensure_open()
            conn.execute("UPDATE runs SET project_id = NULL WHERE project_id = $1", [project_id])
            conn.execute(DELETE_PROJECT, [project_id])

    def set_run_project(self, run_id: str, project_id: str) -> None:
        """Assign a run to a project."""
        with self._lock:
            conn = self._ensure_open()
            conn.execute(SET_RUN_PROJECT, [run_id, project_id])

    def get_or_create_project(self, name: str) -> str:
        """Get an existing project by name, or create one.

        Returns:
            The ``project_id`` of the existing or newly-created project.
        """
        existing = self.get_project_by_name(name)
        if existing is not None:
            return str(existing["project_id"])
        return self.create_project(name=name)

    # =====================================================================
    # Export operations (produce files on demand)
    # =====================================================================

    def export_chain(
        self,
        chain_id: str,
        output_path: Path,
        format: str = "n4a",
    ) -> Path:
        """Export a chain as a standalone prediction bundle.

        Builds a self-contained archive from the chain's steps and
        artifacts, suitable for deployment or sharing without the
        workspace.

        Supported formats:

        * ``"n4a"`` -- ZIP archive containing ``manifest.json``,
          ``chain.json``, and all referenced artifact files.
        * ``"n4a.py"`` -- Portable Python script with embedded
          (base64-encoded) artifacts.

        Args:
            chain_id: Chain to export.
            output_path: Destination file path.
            format: Export format (``"n4a"`` or ``"n4a.py"``).

        Returns:
            The resolved output path (may have an extension appended if
            not already present).

        Raises:
            KeyError: If the chain does not exist.
            FileNotFoundError: If any referenced artifact file is missing.
        """
        chain = self.get_chain(chain_id)
        if chain is None:
            raise KeyError(f"Chain not found: {chain_id}")

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".n4a")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect artifact IDs referenced by the chain
        artifact_ids: set[str] = set()
        fold_artifacts = chain.get("fold_artifacts") or {}
        shared_artifacts = chain.get("shared_artifacts") or {}

        # Detect refit model using canonical fold key; keep legacy key fallback.
        refit_key = "fold_final" if "fold_final" in fold_artifacts else ("final" if "final" in fold_artifacts else None)
        has_refit = refit_key is not None
        if has_refit:
            # Only include the single refit model artifact.
            refit_artifact_id = fold_artifacts.get(refit_key, "")
            export_fold_artifacts = {"fold_final": refit_artifact_id}
            if refit_artifact_id:
                artifact_ids.add(refit_artifact_id)
        else:
            export_fold_artifacts = fold_artifacts
            for aid in fold_artifacts.values():
                if aid:
                    artifact_ids.add(aid)

        for v in shared_artifacts.values():
            if isinstance(v, list):
                for aid in v:
                    if aid:
                        artifact_ids.add(aid)
            elif v:
                artifact_ids.add(v)

        # Build manifest
        fold_strategy = "single_refit" if has_refit else chain["fold_strategy"]
        manifest = {
            "chain_id": chain_id,
            "model_class": chain["model_class"],
            "model_step_index": chain["model_step_idx"],
            "preprocessings": chain["preprocessings"],
            "fold_strategy": fold_strategy,
            "has_refit": has_refit,
            "exported_at": datetime.now(UTC).isoformat(),
        }

        # Write ZIP bundle
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("chain.json", json.dumps({
                "steps": chain["steps"],
                "model_step_idx": chain["model_step_idx"],
                "fold_artifacts": export_fold_artifacts,
                "shared_artifacts": shared_artifacts,
            }, indent=2))

            for aid in artifact_ids:
                path = self.get_artifact_path(aid)
                if not path.exists():
                    raise FileNotFoundError(f"Artifact file missing: {path}")
                zf.write(path, f"artifacts/{path.name}")

        return output_path

    def export_pipeline_config(
        self,
        pipeline_id: str,
        output_path: Path,
    ) -> Path:
        """Export a pipeline's expanded configuration as JSON.

        The exported file contains the fully resolved pipeline
        configuration (after generator expansion) and can be re-used
        with ``nirs4all.run()`` or ``nirs4all.session()``.

        Args:
            pipeline_id: Pipeline to export.
            output_path: Destination ``.json`` file path.

        Returns:
            The resolved output path.

        Raises:
            KeyError: If the pipeline does not exist.
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            raise KeyError(f"Pipeline not found: {pipeline_id}")

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(pipeline["expanded_config"], indent=2, default=str))
        return output_path

    def export_run(
        self,
        run_id: str,
        output_path: Path,
    ) -> Path:
        """Export full run metadata (run + pipelines + chains) as YAML.

        Produces an archival snapshot of the entire run, including all
        pipeline configs, chain descriptions, and summary metrics.
        Does *not* include binary artifacts or arrays.

        Args:
            run_id: Run to export.
            output_path: Destination ``.yaml`` file path.

        Returns:
            The resolved output path.

        Raises:
            KeyError: If the run does not exist.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".yaml")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect pipelines
        pipelines_df = self.list_pipelines(run_id=run_id)
        pipelines_list = []
        if len(pipelines_df) > 0:
            for row in pipelines_df.iter_rows(named=True):
                pid = row["pipeline_id"]
                p = self.get_pipeline(pid)
                chains_df = self.get_chains_for_pipeline(pid)
                chains_list = chains_df.to_dicts() if len(chains_df) > 0 else []
                pipelines_list.append({
                    "pipeline_id": pid,
                    "name": p["name"] if p else row.get("name"),
                    "status": p["status"] if p else row.get("status"),
                    "dataset_name": p["dataset_name"] if p else row.get("dataset_name"),
                    "best_val": p["best_val"] if p else None,
                    "best_test": p["best_test"] if p else None,
                    "metric": p["metric"] if p else None,
                    "duration_ms": p["duration_ms"] if p else None,
                    "chains": chains_list,
                })

        # Convert timestamps to strings for YAML
        created_at = run.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        completed_at = run.get("completed_at")
        if isinstance(completed_at, datetime):
            completed_at = completed_at.isoformat()

        export_data = {
            "run_id": run_id,
            "name": run["name"],
            "status": run["status"],
            "created_at": str(created_at) if created_at else None,
            "completed_at": str(completed_at) if completed_at else None,
            "config": run["config"],
            "datasets": run["datasets"],
            "summary": run["summary"],
            "pipelines": pipelines_list,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        return output_path

    def export_predictions_parquet(
        self,
        output_path: Path,
        **filters: Any,
    ) -> Path:
        """Export prediction records to a Parquet file.

        Writes a Polars-compatible Parquet file containing prediction
        metadata (no arrays).  Filters use the same keyword arguments
        as :meth:`query_predictions`.

        Args:
            output_path: Destination ``.parquet`` file path.
            **filters: Optional filters (``dataset_name``,
                ``model_class``, ``partition``, ``fold_id``,
                ``branch_id``, ``pipeline_id``, ``run_id``).

        Returns:
            The resolved output path.
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".parquet")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.query_predictions(**filters)
        df.write_parquet(output_path)
        return output_path

    # =====================================================================
    # Deletion and cleanup
    # =====================================================================

    def delete_run(self, run_id: str, delete_artifacts: bool = True) -> int:
        """Delete a run and all its descendant data.

        Cascades to pipelines, chains, predictions, prediction arrays,
        and log entries.  If ``delete_artifacts`` is ``True``, artifact
        reference counts are decremented and files with zero references
        are removed from disk.

        Args:
            run_id: Run to delete.
            delete_artifacts: Whether to remove orphaned artifact files.

        Returns:
            Total number of database rows deleted across all tables.
        """
        with self._lock:
            conn = self._ensure_open()

            # Collect prediction_ids for Parquet array deletion
            pred_ids_result = conn.execute(
                "SELECT prediction_id FROM predictions WHERE pipeline_id IN "
                "(SELECT pipeline_id FROM pipelines WHERE run_id = $1)",
                [run_id],
            ).fetchall()
            pred_ids = {row[0] for row in pred_ids_result}

            # Count rows that will be deleted
            total = 0
            for table, column in [
                ("logs", "pipeline_id"),
                ("predictions", "pipeline_id"),
                ("chains", "pipeline_id"),
                ("pipelines", "run_id"),
            ]:
                if table in ("logs", "predictions", "chains"):
                    sql = f"SELECT COUNT(*) AS cnt FROM {table} WHERE pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = $1)"
                else:
                    sql = f"SELECT COUNT(*) AS cnt FROM {table} WHERE {column} = $1"
                result = conn.execute(sql, [run_id]).fetchone()
                total += result[0] if result else 0

            # Count the run itself
            run_exists = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = $1", [run_id]).fetchone()
            if run_exists and run_exists[0] > 0:
                total += 1

            if delete_artifacts:
                self._decrement_artifact_refs_for_run(run_id)

            # Delete arrays from Parquet store
            if pred_ids:
                self._array_store.delete_batch(pred_ids)

            # Manual cascade (DuckDB does not support ON DELETE CASCADE).
            # Delete in reverse dependency order.
            conn.execute(CASCADE_DELETE_RUN_LOGS, [run_id])
            conn.execute(CASCADE_DELETE_RUN_PREDICTIONS, [run_id])
            conn.execute(CASCADE_DELETE_RUN_CHAINS, [run_id])
            conn.execute(CASCADE_DELETE_RUN_PIPELINES, [run_id])
            conn.execute(DELETE_RUN, [run_id])

            if delete_artifacts:
                self.gc_artifacts()

            return total

    def delete_prediction(self, prediction_id: str) -> bool:
        """Delete a single prediction and its associated arrays.

        Args:
            prediction_id: Prediction to delete.

        Returns:
            ``True`` if the prediction existed and was deleted,
            ``False`` otherwise.
        """
        with self._lock:
            conn = self._ensure_open()
            existing = self._fetch_one(GET_PREDICTION, [prediction_id])
            if existing is None:
                return False
            self._array_store.delete_batch({prediction_id})
            conn.execute(DELETE_PREDICTION, [prediction_id])
            return True

    def delete_dataset_predictions(self, dataset_name: str) -> int:
        """Delete all predictions for a dataset and their arrays.

        Args:
            dataset_name: Dataset whose predictions should be removed.

        Returns:
            Number of predictions deleted.
        """
        with self._lock:
            conn = self._ensure_open()
            rows = conn.execute(
                "SELECT prediction_id FROM predictions WHERE dataset_name = $1",
                [dataset_name],
            ).fetchall()
            pred_ids = {row[0] for row in rows}
            if not pred_ids:
                return 0

            # Delete arrays (Parquet file + tombstones)
            self._array_store.delete_dataset(dataset_name)

            conn.execute("DELETE FROM predictions WHERE dataset_name = $1", [dataset_name])
            return len(pred_ids)

    def cleanup_transient_artifacts(
        self,
        run_id: str,
        dataset_name: str,
        winning_pipeline_ids: list[str],
    ) -> int:
        """Clean up transient fold artifacts after successful refit.

        After a refit pass produces ``fold_id="final"`` artifacts, the
        per-fold model artifacts from the CV pass become redundant.  This
        method decrements reference counts on those transient artifacts
        and garbage-collects any that become orphaned.

        Specifically, for every pipeline in the run that targets the
        given dataset:

        1. **Losing pipelines** (not in *winning_pipeline_ids*): all
           fold *and* shared artifacts have their ref counts decremented.
        2. **Winning pipeline**: only fold-specific model artifacts
           (fold keys that are NOT ``"final"``/``"avg"``/``"w_avg"``)
           have their ref counts decremented.  Shared (preprocessing)
           artifacts are kept because they are needed by the refit chain.
        3. **Prediction safety**: artifacts still needed to replay an
           existing prediction entry are preserved, even if they are
           otherwise transient.

        Prediction records in DuckDB are **never** deleted -- they are
        lightweight metadata useful for analysis.

        Chain records are **never** deleted -- only the binary files
        behind the decremented artifact IDs may be garbage-collected.

        Content-addressed deduplication is respected: if the same binary
        file backs both a transient and a permanent artifact record,
        its ``ref_count`` will still be positive after cleanup.

        Args:
            run_id: Run identifier.
            dataset_name: Dataset name to scope the cleanup to.
            winning_pipeline_ids: Pipeline IDs that produced the refit
                model(s).  Artifacts from these pipelines' shared
                preprocessing chains are preserved.

        Returns:
            Number of artifact files removed from disk.
        """
        def _iter_shared_artifact_refs(shared_artifacts: dict | None):
            """Yield concrete shared artifact IDs (skip metadata keys)."""
            if not isinstance(shared_artifacts, dict):
                return
            for key, value in shared_artifacts.items():
                if str(key).startswith("_"):
                    # Metadata key, e.g. "_source_map"
                    continue
                if isinstance(value, list):
                    for artifact_id in value:
                        if artifact_id:
                            yield artifact_id
                elif isinstance(value, str) and value:
                    yield value

        def _fold_is_protected(
            fold_key: str,
            protected_fold_ids: set[str],
        ) -> bool:
            """Check whether a fold key is required for prediction replay."""
            if "__all_folds__" in protected_fold_ids:
                return True

            if fold_key in protected_fold_ids:
                return True

            # Accept both "0"/"final" and "fold_0"/"fold_final" styles.
            if fold_key.startswith("fold_"):
                return fold_key[5:] in protected_fold_ids
            return f"fold_{fold_key}" in protected_fold_ids

        with self._lock:
            conn = self._ensure_open()

            # Permanent fold IDs whose model artifacts should never be cleaned
            permanent_fold_ids = frozenset({
                "final", "avg", "w_avg",
                "fold_final", "fold_avg", "fold_w_avg",
            })

            # Get all pipelines for this run and dataset
            all_pipelines = conn.execute(
                "SELECT pipeline_id FROM pipelines WHERE run_id = $1 AND dataset_name = $2",
                [run_id, dataset_name],
            ).fetchall()

            winning_set = set(winning_pipeline_ids)

            # Bulk-load ALL chains for these pipelines in a single query
            # instead of one query per pipeline.
            all_pipeline_ids = [pid for (pid,) in all_pipelines]
            chains_by_pipeline: dict[str, list[tuple[str, Any, Any]]] = {pid: [] for pid in all_pipeline_ids}
            chain_lookup: dict[str, tuple[dict, dict]] = {}

            if all_pipeline_ids:
                all_chain_rows = conn.execute(
                    "SELECT pipeline_id, chain_id, fold_artifacts, shared_artifacts "
                    "FROM chains WHERE pipeline_id IN ("
                    "  SELECT pipeline_id FROM pipelines WHERE run_id = $1 AND dataset_name = $2"
                    ")",
                    [run_id, dataset_name],
                ).fetchall()

                for pipeline_id, chain_id, fold_artifacts_raw, shared_artifacts_raw in all_chain_rows:
                    row_tuple = (chain_id, fold_artifacts_raw, shared_artifacts_raw)
                    chains_by_pipeline.setdefault(pipeline_id, []).append(row_tuple)

                    fold_artifacts = (
                        _from_json(fold_artifacts_raw)
                        if isinstance(fold_artifacts_raw, str)
                        else fold_artifacts_raw
                    ) or {}
                    shared_artifacts = (
                        _from_json(shared_artifacts_raw)
                        if isinstance(shared_artifacts_raw, str)
                        else shared_artifacts_raw
                    ) or {}
                    chain_lookup[chain_id] = (fold_artifacts, shared_artifacts)

            # Build replay-protection map from persisted predictions:
            # chain_id -> set of fold identifiers required for replay.
            protected_folds_by_chain: dict[str, set[str]] = {}
            prediction_refs = conn.execute(
                """
                SELECT p.chain_id, p.fold_id
                FROM predictions p
                JOIN pipelines pl ON p.pipeline_id = pl.pipeline_id
                WHERE pl.run_id = $1
                  AND p.dataset_name = $2
                  AND p.chain_id IS NOT NULL
                """,
                [run_id, dataset_name],
            ).fetchall()

            for chain_id, fold_id in prediction_refs:
                if chain_id not in chain_lookup:
                    continue

                requested = protected_folds_by_chain.setdefault(chain_id, set())
                fold_value = "" if fold_id is None else str(fold_id)

                # Averaged/empty fold IDs depend on CV fold artifacts.
                if fold_value in {"", "None", "avg", "w_avg"}:
                    requested.add("__all_folds__")
                else:
                    requested.add(fold_value)

            # Collect all artifact IDs to decrement, then batch-update once.
            artifacts_to_decrement: list[str] = []

            for (pipeline_id,) in all_pipelines:
                chains = chains_by_pipeline.get(pipeline_id, [])

                is_winner = pipeline_id in winning_set

                for chain_id, fold_artifacts_raw, shared_artifacts_raw in chains:
                    fold_artifacts = (
                        _from_json(fold_artifacts_raw)
                        if isinstance(fold_artifacts_raw, str)
                        else fold_artifacts_raw
                    ) or {}
                    shared_artifacts = (
                        _from_json(shared_artifacts_raw)
                        if isinstance(shared_artifacts_raw, str)
                        else shared_artifacts_raw
                    ) or {}

                    protected_fold_ids = protected_folds_by_chain.get(chain_id, set())
                    protect_shared = chain_id in protected_folds_by_chain

                    if is_winner:
                        # Winning pipeline: only decrement CV fold model artifacts
                        if fold_artifacts:
                            for fold_key, artifact_id in fold_artifacts.items():
                                if not artifact_id:
                                    continue
                                if fold_key in permanent_fold_ids:
                                    continue
                                if _fold_is_protected(str(fold_key), protected_fold_ids):
                                    continue
                                artifacts_to_decrement.append(artifact_id)
                        # Keep shared artifacts (preprocessing) -- needed by refit chain
                    else:
                        # Losing pipeline: decrement artifacts unless replay-protected
                        if fold_artifacts:
                            for fold_key, artifact_id in fold_artifacts.items():
                                if not artifact_id:
                                    continue
                                if _fold_is_protected(str(fold_key), protected_fold_ids):
                                    continue
                                artifacts_to_decrement.append(artifact_id)

                        if shared_artifacts:
                            for artifact_id in _iter_shared_artifact_refs(shared_artifacts):
                                if protect_shared:
                                    continue
                                artifacts_to_decrement.append(artifact_id)

            # Batch decrement: count how many times each artifact should be
            # decremented, then issue one UPDATE per distinct decrement count.
            if artifacts_to_decrement:
                from collections import Counter
                decrement_counts: dict[int, list[str]] = {}
                for aid, cnt in Counter(artifacts_to_decrement).items():
                    decrement_counts.setdefault(cnt, []).append(aid)
                for delta, ids in decrement_counts.items():
                    placeholders = ", ".join(f"${i+2}" for i in range(len(ids)))
                    conn.execute(
                        f"UPDATE artifacts SET ref_count = ref_count - $1 "
                        f"WHERE artifact_id IN ({placeholders})",
                        [delta] + ids,
                    )

        # Run garbage collection to remove orphaned files
        return self.gc_artifacts()

    def gc_artifacts(self) -> int:
        """Garbage-collect unreferenced artifacts.

        Removes artifact files from disk whose reference count has
        dropped to zero (i.e. no chain references them).  Also removes
        the corresponding rows from the ``artifacts`` table.

        Returns:
            Number of artifact files removed.
        """
        with self._lock:
            conn = self._ensure_open()
            orphans = conn.execute(GC_ARTIFACTS).fetchall()

            # Keep files that are still referenced by at least one live row.
            live_paths = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT artifact_path FROM artifacts WHERE ref_count > 0"
                ).fetchall()
            }

            removed_paths: set[str] = set()
            count = 0
            for _artifact_id, artifact_path in orphans:
                if artifact_path in live_paths:
                    # Same content-addressed file still used by another artifact row.
                    continue
                if artifact_path in removed_paths:
                    continue

                path = self._artifacts_dir / artifact_path
                if path.exists():
                    path.unlink()
                    # Remove empty parent directory
                    with contextlib.suppress(OSError):
                        path.parent.rmdir()
                removed_paths.add(artifact_path)
                count += 1

            conn.execute(DELETE_GC_ARTIFACTS)
            return count

    def vacuum(self) -> None:
        """Reclaim unused space in the DuckDB database file.

        Equivalent to ``VACUUM`` in SQL.  Call after large deletions
        to reduce the on-disk size of ``store.duckdb``.
        """
        with self._lock:
            conn = self._ensure_open()
            conn.execute("VACUUM")

    # =====================================================================
    # Chain replay
    # =====================================================================

    def replay_chain(
        self,
        chain_id: str,
        X: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> np.ndarray:
        """Replay a stored chain on new data to produce predictions.

        Loads each step's artifact, applies the transformation in order,
        and for the model step loads all fold models and returns the
        averaged prediction.

        This is the primary in-workspace prediction path.  For
        out-of-workspace prediction, export to ``.n4a`` first.

        Args:
            chain_id: Chain to replay.
            X: Input feature matrix (``n_samples x n_features``).
            wavelengths: Optional wavelength array for wavelength-aware
                operators.  Required if any step in the chain uses
                ``SpectraTransformerMixin``.

        Returns:
            Predicted values as a 1-D :class:`numpy.ndarray` of shape
            ``(n_samples,)``.

        Raises:
            KeyError: If the chain does not exist.
            RuntimeError: If the chain has no model step.
        """
        chain = self.get_chain(chain_id)
        if chain is None:
            raise KeyError(f"Chain not found: {chain_id}")

        steps = chain["steps"]
        model_step_idx = chain["model_step_idx"]
        fold_artifacts = chain.get("fold_artifacts") or {}
        shared_artifacts = chain.get("shared_artifacts") or {}

        X_current = X.copy()

        def _transform_with_optional_wavelengths(transformer: Any, X_in: np.ndarray) -> np.ndarray:
            """Call transformer.transform with wavelengths when supported."""
            if wavelengths is None:
                return np.asarray(transformer.transform(X_in))

            supports_wavelengths = False
            try:
                params = inspect.signature(transformer.transform).parameters
                supports_wavelengths = (
                    "wavelengths" in params
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                )
            except (TypeError, ValueError):
                supports_wavelengths = False

            if supports_wavelengths or hasattr(transformer, "_requires_wavelengths"):
                return np.asarray(transformer.transform(X_in, wavelengths=wavelengths))

            return np.asarray(transformer.transform(X_in))

        for step in steps:
            idx = step["step_idx"]

            if idx == model_step_idx:
                # Refit model: single canonical refit artifact, direct prediction
                refit_artifact_id = fold_artifacts.get("fold_final") or fold_artifacts.get("final")
                if refit_artifact_id:
                    model = self.load_artifact(refit_artifact_id)
                    return np.asarray(model.predict(X_current))

                # Legacy: load all fold models, predict, average
                fold_preds = []
                for _fold_id, artifact_id in fold_artifacts.items():
                    model = self.load_artifact(artifact_id)
                    fold_preds.append(model.predict(X_current))
                if not fold_preds:
                    raise RuntimeError("Chain has no fold model artifacts")
                return np.asarray(np.mean(fold_preds, axis=0))

            str_idx = str(idx)
            if str_idx in shared_artifacts:
                artifact_ids = shared_artifacts[str_idx]
                # shared_artifacts values are lists of artifact IDs
                if isinstance(artifact_ids, str):
                    artifact_ids = [artifact_ids]
                for artifact_id in artifact_ids:
                    transformer = self.load_artifact(artifact_id)
                    X_current = _transform_with_optional_wavelengths(transformer, X_current)
            elif step.get("stateless", False):
                # Stateless step -- skip (no artifact needed)
                pass

        raise RuntimeError("Chain has no model step")

    # =====================================================================
    # Private helpers for artifact reference management
    # =====================================================================

    def _collect_artifact_ids_from_chain_row(self, row: dict) -> set[str]:
        """Extract all artifact IDs referenced by a chain row."""
        ids: set[str] = set()
        fold_artifacts = _from_json(row.get("fold_artifacts")) if isinstance(row.get("fold_artifacts"), str) else row.get("fold_artifacts")
        shared_artifacts = _from_json(row.get("shared_artifacts")) if isinstance(row.get("shared_artifacts"), str) else row.get("shared_artifacts")
        if fold_artifacts:
            for v in fold_artifacts.values():
                if v:
                    ids.add(v)
        if shared_artifacts:
            for v in shared_artifacts.values():
                if isinstance(v, list):
                    for aid in v:
                        if aid:
                            ids.add(aid)
                elif v:
                    ids.add(v)
        return ids

    def _decrement_artifact_refs_for_pipeline(self, pipeline_id: str) -> None:
        """Decrement ref counts for all artifacts referenced by chains in a pipeline."""
        with self._lock:
            conn = self._ensure_open()
            result = conn.execute("SELECT fold_artifacts, shared_artifacts FROM chains WHERE pipeline_id = $1", [pipeline_id])
            desc = result.description
            chains = result.fetchall()
            if not chains or not desc:
                return
            columns = [d[0] for d in desc]
            for chain_row in chains:
                row_dict = dict(zip(columns, chain_row, strict=False))
                for aid in self._collect_artifact_ids_from_chain_row(row_dict):
                    conn.execute(DECREMENT_ARTIFACT_REF, [aid])

    def _decrement_artifact_refs_for_run(self, run_id: str) -> None:
        """Decrement ref counts for all artifacts referenced by chains in a run."""
        with self._lock:
            conn = self._ensure_open()
            sql = "SELECT fold_artifacts, shared_artifacts FROM chains WHERE pipeline_id IN (SELECT pipeline_id FROM pipelines WHERE run_id = $1)"
            result = conn.execute(sql, [run_id])
            desc = result.description
            chains = result.fetchall()
            if not chains or not desc:
                return
            columns = [d[0] for d in desc]
            for chain_row in chains:
                row_dict = dict(zip(columns, chain_row, strict=False))
                for aid in self._collect_artifact_ids_from_chain_row(row_dict):
                    conn.execute(DECREMENT_ARTIFACT_REF, [aid])
