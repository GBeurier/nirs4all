"""Store-backed predictions management.

This module provides the ``Predictions`` facade for accumulating, ranking,
filtering, and exporting prediction records.  In Phase 3 of the DuckDB
storage migration the class was rewritten to work exclusively with
:class:`~nirs4all.pipeline.storage.workspace_store.WorkspaceStore`.

When a *store* is supplied (normal execution path), every call to
:meth:`add_prediction` buffers the record in memory and
:meth:`flush` writes the buffer to the database.  Query methods
(``top``, ``filter_predictions``, ``get_best``, ...) delegate to the
store's columnar query engine.

When *no store* is supplied (lightweight / test usage), the class
operates in a purely in-memory mode backed by a Polars DataFrame,
preserving backward compatibility for code that creates
``Predictions()`` without a workspace.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

import numpy as np
import polars as pl

from nirs4all.core import metrics as evaluator
from nirs4all.core.logging import get_logger

from ._predictions.result import PredictionResult, PredictionResultsList

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

logger = get_logger(__name__)

__all__ = ["Predictions", "PredictionResult", "PredictionResultsList"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Handle numpy scalars for json.dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _infer_ascending(metric: str) -> bool:
    """Infer sort direction from metric name.

    Args:
        metric: Metric name (e.g. ``"rmse"``, ``"r2"``, ``"accuracy"``).

    Returns:
        ``True`` if lower is better, ``False`` if higher is better.
    """
    higher_is_better = {"r2", "accuracy", "f1", "precision", "recall", "auc", "roc_auc", "balanced_accuracy", "kappa", "rpd", "rpiq"}
    return metric.lower() not in higher_is_better


_CASE_INSENSITIVE_COLS = {"model_name", "model_classname", "preprocessings", "dataset_name", "config_name"}


def _make_group_key(row: dict[str, Any], group_by: list[str]) -> tuple:
    """Create a hashable group key from row values.

    String columns in :data:`_CASE_INSENSITIVE_COLS` are lowered for
    case-insensitive grouping.  List values are converted to tuples.
    """
    parts: list[Any] = []
    for col in group_by:
        val = row.get(col)
        if val is None:
            parts.append(None)
        elif isinstance(val, str) and col in _CASE_INSENSITIVE_COLS:
            parts.append(val.lower())
        elif isinstance(val, list):
            parts.append(tuple(val))
        else:
            parts.append(val)
    return tuple(parts)


def _build_prediction_row(
    *,
    dataset_name: str,
    dataset_path: str = "",
    config_name: str = "",
    config_path: str = "",
    pipeline_uid: str | None = None,
    step_idx: int = 0,
    op_counter: int = 0,
    model_name: str = "",
    model_classname: str = "",
    model_path: str = "",
    fold_id: str | int | None = None,
    sample_indices: list[int] | np.ndarray | None = None,
    weights: list[float] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    partition: str = "",
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    val_score: float | None = None,
    test_score: float | None = None,
    train_score: float | None = None,
    metric: str = "mse",
    task_type: str = "regression",
    n_samples: int = 0,
    n_features: int = 0,
    preprocessings: str = "",
    best_params: dict[str, Any] | None = None,
    scores: dict[str, dict[str, float]] | None = None,
    branch_id: int | None = None,
    branch_path: list[int] | None = None,
    branch_name: str | None = None,
    exclusion_count: int | None = None,
    exclusion_rate: float | None = None,
    model_artifact_id: str | None = None,
    trace_id: str | None = None,
    refit_context: str | None = None,
    target_processing: str = "",
) -> dict[str, Any]:
    """Normalise add_prediction kwargs into a flat dict with a generated id."""
    pred_id = str(uuid4())[:16]
    return {
        "id": pred_id,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "config_name": config_name,
        "config_path": config_path,
        "pipeline_uid": pipeline_uid or "",
        "step_idx": step_idx,
        "op_counter": op_counter,
        "model_name": model_name,
        "model_classname": model_classname,
        "model_path": model_path,
        "fold_id": str(fold_id) if fold_id is not None else "",
        "sample_indices": (sample_indices.tolist() if isinstance(sample_indices, np.ndarray) else sample_indices) if sample_indices is not None else [],
        "weights": (weights.tolist() if isinstance(weights, np.ndarray) else weights) if weights is not None else [],
        "metadata": metadata if metadata is not None else {},
        "partition": partition,
        "y_true": y_true if y_true is not None else np.array([]),
        "y_pred": y_pred if y_pred is not None else np.array([]),
        "y_proba": y_proba if y_proba is not None else np.array([]),
        "val_score": val_score,
        "test_score": test_score,
        "train_score": train_score,
        "metric": metric,
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": n_features,
        "preprocessings": preprocessings,
        "best_params": best_params if best_params is not None else {},
        "scores": scores if scores is not None else {},
        "branch_id": branch_id,
        "branch_path": branch_path,
        "branch_name": branch_name or "",
        "exclusion_count": exclusion_count,
        "exclusion_rate": exclusion_rate,
        "model_artifact_id": model_artifact_id or "",
        "trace_id": trace_id or "",
        "refit_context": refit_context,
        "target_processing": target_processing or "",
        "created_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Predictions facade
# ---------------------------------------------------------------------------

class Predictions:
    """Facade for prediction management.

    When constructed with a :class:`WorkspaceStore`, prediction records are
    buffered in memory during pipeline execution and flushed to DuckDB at
    the end of each pipeline via :meth:`flush`.  Queries delegate to the
    store's columnar engine.

    When constructed *without* a store (``Predictions()``), the class
    operates in a lightweight in-memory mode backed by a list of dicts.
    This preserves backward compatibility for code that creates bare
    ``Predictions`` instances (tests, visualisation adapters, the webapp).

    Args:
        store: Optional :class:`WorkspaceStore` for database-backed mode.

    Examples:
        >>> # In-memory mode (lightweight)
        >>> pred = Predictions()
        >>> pred.add_prediction(dataset_name="wheat", model_name="PLS", ...)

        >>> # Store-backed mode (pipeline execution)
        >>> from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
        >>> store = WorkspaceStore(Path("workspace"))
        >>> pred = Predictions(store=store)
        >>> pred.add_prediction(...)
        >>> pred.flush(pipeline_id="abc")
    """

    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self._store = store
        # In-memory buffer for predictions accumulated during a pipeline run.
        self._buffer: list[dict[str, Any]] = []
        # Dataset repetition column for by_repetition=True resolution
        self._dataset_repetition: str | None = None

    # =========================================================================
    # LOADING FROM WORKSPACE
    # =========================================================================

    @classmethod
    def from_workspace(
        cls,
        workspace_path: str | Path,
        *,
        dataset_name: str | None = None,
        load_arrays: bool = True,
    ) -> Predictions:
        """Load predictions from a workspace DuckDB store.

        Opens the workspace store, queries all predictions (with optional
        dataset filter), loads associated arrays (y_true, y_pred), and
        returns a fully populated in-memory ``Predictions`` instance
        ready for use with ``PredictionAnalyzer``.

        Args:
            workspace_path: Path to the workspace directory containing
                ``store.duckdb``.
            dataset_name: Optional dataset name filter.
            load_arrays: If ``True`` (default), load y_true/y_pred arrays
                for each prediction.  Set to ``False`` for metadata-only
                queries (faster, but no scatter/residual plots).

        Returns:
            A ``Predictions`` instance with the buffer populated from
            the store.

        Example:
            >>> predictions = Predictions.from_workspace("workspace")
            >>> predictions.top(5)
            >>> analyzer = PredictionAnalyzer(predictions)
            >>> analyzer.plot_histogram()
        """
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        workspace_path = Path(workspace_path)
        store = WorkspaceStore(workspace_path)
        try:
            return cls._load_from_store(store, dataset_name=dataset_name, load_arrays=load_arrays)
        finally:
            store.close()

    @classmethod
    def _load_from_store(
        cls,
        store: WorkspaceStore,
        *,
        dataset_name: str | None = None,
        load_arrays: bool = True,
    ) -> Predictions:
        """Load predictions from an open WorkspaceStore.

        Args:
            store: An open WorkspaceStore instance.
            dataset_name: Optional dataset name filter.
            load_arrays: If True, load y_true/y_pred arrays.

        Returns:
            A Predictions instance with buffer populated.
        """
        predictions = cls()

        df = store.query_predictions(dataset_name=dataset_name)
        if df.is_empty():
            return predictions

        for row in df.iter_rows(named=True):
            pred_id = row["prediction_id"]

            y_true = None
            y_pred = None
            y_proba = None
            sample_indices = None

            if load_arrays:
                arrays = store.get_prediction_arrays(pred_id)
                if arrays:
                    y_true = arrays.get("y_true")
                    y_pred = arrays.get("y_pred")
                    y_proba = arrays.get("y_proba")
                    sample_indices = arrays.get("sample_indices")

            # Parse JSON fields from DuckDB
            scores = row.get("scores")
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except (json.JSONDecodeError, TypeError):
                    scores = None

            best_params = row.get("best_params")
            if isinstance(best_params, str):
                try:
                    best_params = json.loads(best_params)
                except (json.JSONDecodeError, TypeError):
                    best_params = None

            predictions.add_prediction(
                dataset_name=row.get("dataset_name", ""),
                model_name=row.get("model_name", ""),
                model_classname=row.get("model_class", ""),
                fold_id=row.get("fold_id", ""),
                partition=row.get("partition", ""),
                val_score=row.get("val_score"),
                test_score=row.get("test_score"),
                train_score=row.get("train_score"),
                metric=row.get("metric", "rmse"),
                task_type=row.get("task_type", "regression"),
                n_samples=row.get("n_samples") or 0,
                n_features=row.get("n_features") or 0,
                preprocessings=row.get("preprocessings", ""),
                scores=scores,
                best_params=best_params,
                branch_id=row.get("branch_id"),
                branch_name=row.get("branch_name"),
                refit_context=row.get("refit_context"),
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                sample_indices=sample_indices,
                pipeline_uid=row.get("pipeline_id", ""),
            )

        return predictions

    # =========================================================================
    # DATASET CONTEXT
    # =========================================================================

    def set_repetition_column(self, column: str | None) -> None:
        """Set the dataset repetition column for by_repetition=True resolution.

        This is typically called automatically when a SpectroDataset with a
        defined repetition column is used in pipeline execution.

        Args:
            column: Metadata column name identifying repetitions, or None.

        Example:
            >>> predictions.set_repetition_column("Sample_ID")
            >>> # Now by_repetition=True will use "Sample_ID" column
            >>> predictions.top(5, by_repetition=True)
        """
        self._dataset_repetition = column

    @property
    def repetition_column(self) -> str | None:
        """Get the dataset repetition column if set.

        Returns:
            The repetition column name from dataset context, or None.
        """
        return self._dataset_repetition

    # =========================================================================
    # CORE CRUD OPERATIONS
    # =========================================================================

    def add_prediction(
        self,
        dataset_name: str,
        dataset_path: str = "",
        config_name: str = "",
        config_path: str = "",
        pipeline_uid: str | None = None,
        step_idx: int = 0,
        op_counter: int = 0,
        model_name: str = "",
        model_classname: str = "",
        model_path: str = "",
        fold_id: str | int | None = None,
        sample_indices: list[int] | np.ndarray | None = None,
        weights: list[float] | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        partition: str = "",
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
        y_proba: np.ndarray | None = None,
        val_score: float | None = None,
        test_score: float | None = None,
        train_score: float | None = None,
        metric: str = "mse",
        task_type: str = "regression",
        n_samples: int = 0,
        n_features: int = 0,
        preprocessings: str = "",
        best_params: dict[str, Any] | None = None,
        scores: dict[str, dict[str, float]] | None = None,
        branch_id: int | None = None,
        branch_path: list[int] | None = None,
        branch_name: str | None = None,
        exclusion_count: int | None = None,
        exclusion_rate: float | None = None,
        model_artifact_id: str | None = None,
        trace_id: str | None = None,
        refit_context: str | None = None,
        target_processing: str = "",
    ) -> str:
        """Add a single prediction to the in-memory buffer.

        The record is buffered locally.  Call :meth:`flush` to persist
        buffered predictions to the workspace store.

        Args:
            dataset_name: Dataset name.
            dataset_path: Path to dataset file.
            config_name: Configuration name.
            config_path: Path to config file.
            pipeline_uid: Unique pipeline identifier.
            step_idx: Pipeline step index.
            op_counter: Operation counter.
            model_name: Model name.
            model_classname: Model class name.
            model_path: Path to saved model.
            fold_id: Cross-validation fold ID (e.g. ``0``, ``"avg"``,
                ``"final"`` for refit).
            sample_indices: Indices of samples used.
            weights: Sample weights.
            metadata: Additional metadata.
            partition: Data partition (train/val/test).
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Class probabilities (n_samples x n_classes).
            val_score: Validation score.
            test_score: Test score.
            train_score: Training score.
            metric: Metric name.
            task_type: Task type (classification/regression).
            n_samples: Number of samples.
            n_features: Number of features.
            preprocessings: Preprocessing steps applied.
            best_params: Best hyperparameters.
            scores: Dictionary of pre-computed scores per partition.
            branch_id: Branch identifier for pipeline branching.
            branch_path: List of branch indices for nested branching.
            branch_name: Human-readable branch name.
            exclusion_count: Number of excluded samples.
            exclusion_rate: Rate of excluded samples (0.0-1.0).
            model_artifact_id: Deterministic artifact ID.
            trace_id: Execution trace ID.
            refit_context: Refit context label. ``None`` for CV entries,
                ``"standalone"`` for standalone refit,
                ``"stacking"`` for stacking-context refit.
            target_processing: Target (y) processing chain name.

        Returns:
            Prediction ID (short UUID string).
        """
        row = _build_prediction_row(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            config_name=config_name,
            config_path=config_path,
            pipeline_uid=pipeline_uid,
            step_idx=step_idx,
            op_counter=op_counter,
            model_name=model_name,
            model_classname=model_classname,
            model_path=model_path,
            fold_id=fold_id,
            sample_indices=sample_indices,
            weights=weights,
            metadata=metadata,
            partition=partition,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            val_score=val_score,
            test_score=test_score,
            train_score=train_score,
            metric=metric,
            task_type=task_type,
            n_samples=n_samples,
            n_features=n_features,
            preprocessings=preprocessings,
            best_params=best_params,
            scores=scores,
            branch_id=branch_id,
            branch_path=branch_path,
            branch_name=branch_name,
            exclusion_count=exclusion_count,
            exclusion_rate=exclusion_rate,
            model_artifact_id=model_artifact_id,
            trace_id=trace_id,
            refit_context=refit_context,
            target_processing=target_processing,
        )
        self._buffer.append(row)
        return row["id"]

    # =========================================================================
    # FLUSH (store-backed mode)
    # =========================================================================

    def flush(
        self,
        pipeline_id: str | None = None,
        chain_id: str | None = None,
        *,
        store: WorkspaceStore | None = None,
        chain_id_resolver: Callable[[dict[str, Any]], str] | None = None,
        fold_id_override: str | None = None,
        refit_context_override: str | None = None,
    ) -> None:
        """Persist buffered predictions to the workspace store.

        Each buffered record is written via
        :meth:`WorkspaceStore.save_prediction` followed by
        :meth:`WorkspaceStore.save_prediction_arrays`.

        Args:
            pipeline_id: Pipeline ID to associate predictions with.
            chain_id: Chain ID to associate predictions with.
            store: Optional store override. If omitted, uses the instance
                store passed at construction.
            chain_id_resolver: Optional callback that maps each buffered row
                to a chain ID. Used when predictions from multiple chains are
                persisted in one flush.
            fold_id_override: Optional fold-id override applied to all rows
                (used by refit persistence).
            refit_context_override: Optional refit-context override applied to
                all rows.
        """
        target_store = store or self._store
        if target_store is None or not self._buffer:
            return

        for row in self._buffer:
            resolved_chain_id = chain_id or ""
            if chain_id_resolver is not None:
                resolved = chain_id_resolver(row)
                resolved_chain_id = "" if resolved is None else str(resolved)

            # Write chain_id back into the buffer row for downstream lookups
            row["chain_id"] = resolved_chain_id

            fold_id = fold_id_override if fold_id_override is not None else row["fold_id"]
            refit_context = (
                refit_context_override
                if refit_context_override is not None
                else row.get("refit_context")
            )

            pred_id = target_store.save_prediction(
                pipeline_id=pipeline_id or "",
                chain_id=resolved_chain_id,
                dataset_name=row["dataset_name"],
                model_name=row["model_name"],
                model_class=row["model_classname"],
                fold_id=str(fold_id),
                partition=row["partition"],
                val_score=row.get("val_score"),
                test_score=row.get("test_score"),
                train_score=row.get("train_score"),
                metric=row["metric"],
                task_type=row["task_type"],
                n_samples=row["n_samples"],
                n_features=row["n_features"],
                scores=row.get("scores", {}),
                best_params=row.get("best_params", {}),
                branch_id=row.get("branch_id"),
                branch_name=row.get("branch_name") or None,
                exclusion_count=row.get("exclusion_count") or 0,
                exclusion_rate=row.get("exclusion_rate") or 0.0,
                preprocessings=row.get("preprocessings", ""),
                prediction_id=row.get("id"),
                refit_context=refit_context,
            )

            # Save arrays
            y_true = row.get("y_true")
            y_pred = row.get("y_pred")
            y_proba = row.get("y_proba")
            sample_indices = row.get("sample_indices")
            weights = row.get("weights")

            has_y_true = y_true is not None and (isinstance(y_true, np.ndarray) and y_true.size > 0)
            has_y_pred = y_pred is not None and (isinstance(y_pred, np.ndarray) and y_pred.size > 0)

            if has_y_true or has_y_pred:
                target_store.save_prediction_arrays(
                    prediction_id=pred_id,
                    y_true=y_true if has_y_true else None,
                    y_pred=y_pred if has_y_pred else None,
                    y_proba=y_proba if (y_proba is not None and isinstance(y_proba, np.ndarray) and y_proba.size > 0) else None,
                    sample_indices=np.array(sample_indices, dtype=np.int64) if sample_indices and len(sample_indices) > 0 else None,
                    weights=np.array(weights, dtype=np.float64) if weights and len(weights) > 0 else None,
                )

    # =========================================================================
    # RANKING OPERATIONS
    # =========================================================================

    def top(
        self,
        n: int,
        rank_metric: str = "",
        rank_partition: str = "val",
        score_scope: str = "mix",
        display_metrics: list[str] | None = None,
        display_partition: str = "test",
        aggregate_partitions: bool = False,
        ascending: bool | None = None,
        group_by_fold: bool = False,
        by_repetition: bool | str | None = None,
        repetition_method: str | None = None,
        repetition_exclude_outliers: bool = False,
        group_by: str | list[str] | None = None,
        return_grouped: bool = False,
        **filters: Any,
    ) -> PredictionResultsList | dict[tuple, PredictionResultsList]:
        """Get top *n* predictions ranked by a metric.

        Operates on the in-memory buffer.  If a store is available *and*
        the buffer has been flushed, store results are merged, but the
        primary source is always the buffer (for in-pipeline queries the
        data has not been flushed yet).

        Args:
            n: Number of top models to return.  When ``group_by`` is set,
               this means top *n* **per group**.
            rank_metric: Metric to rank by.  Empty string uses the stored
               metric from the prediction record.
            rank_partition: Partition to rank on (default ``"val"``).
            score_scope: Controls how refit (final) entries interact with
               CV entries in ranking.  One of:

               - ``"final"``: Only refit entries (``fold_id="final"``),
                 ranked by their selection score (``selection_score``).
               - ``"cv"``: Only CV entries (exclude refit entries).
               - ``"mix"``: Refit entries ranked first, then CV entries
                 below.  Each group ranked independently.
               - ``"flat"``: All entries ranked equally, no special
                 treatment for refit entries.

               Default is ``"mix"``.  ``"auto"`` is an alias for ``"mix"``.
            display_metrics: Metrics to compute for display.
            display_partition: Partition to display results from.
            aggregate_partitions: If ``True``, add train/val/test dicts.
            ascending: Sort order.  ``None`` infers from metric.
            group_by_fold: If ``True``, include fold_id in identity.
            by_repetition: Aggregate predictions by repetition column.
                - ``True``: Uses ``dataset.repetition`` from context
                  (requires :meth:`set_repetition_column` to be called).
                - ``str``: Explicit column name or ``"y"`` for target grouping.
                - ``False``/``None`` (default): No aggregation.
            repetition_method: Aggregation method for repetitions.
                ``"mean"`` (default), ``"median"``, or ``"vote"``.
            repetition_exclude_outliers: If ``True``, exclude outlier
                measurements before aggregating within each group.
            group_by: Group predictions by column(s) for ranking.
            return_grouped: Return dict of group->results.
            **filters: Additional filter criteria.

        Returns:
            ``PredictionResultsList`` or grouped dict.
        """
        # Resolve by_repetition=True from dataset context
        effective_by_repetition = by_repetition
        effective_repetition_method = repetition_method
        effective_repetition_exclude_outliers = repetition_exclude_outliers

        if effective_by_repetition is True:
            if self._dataset_repetition is None:
                warnings.warn(
                    "by_repetition=True specified but no repetition column available from dataset context. "
                    "Use set_repetition_column() or pass an explicit column name. "
                    "Skipping aggregation.",
                    UserWarning,
                    stacklevel=2,
                )
                effective_by_repetition = None
            else:
                effective_by_repetition = self._dataset_repetition
        # Strip non-filter kwargs that callers may pass
        _ = filters.pop("partition", None)
        _ = filters.pop("load_arrays", None)
        _ = filters.pop("higher_is_better", None)
        _ = filters.pop("rank_metric", None)
        _ = filters.pop("rank_partition", None)
        _ = filters.pop("aggregate_partitions", None)
        _ = filters.pop("ascending", None)
        _ = filters.pop("group_by_fold", None)
        _ = filters.pop("score_scope", None)
        _ = filters.pop("aggregate", None)
        _ = filters.pop("aggregate_method", None)
        _ = filters.pop("aggregate_exclude_outliers", None)

        # Normalise score_scope alias
        effective_scope = score_scope if score_scope != "auto" else "mix"

        # Filter the in-memory buffer first, then copy only matching entries.
        # This avoids creating ~N shallow dict copies when only a fraction
        # of the buffer matches the filters.
        filter_items = list(filters.items())
        candidates: list[dict[str, Any]] = []
        for r in self._buffer:
            fold_id_str = str(r.get("fold_id", ""))
            is_final = fold_id_str == "final"
            refit_context = r.get("refit_context")

            # Apply score_scope filtering
            if effective_scope == "final" and not is_final:
                continue
            if effective_scope == "cv" and refit_context is not None:
                continue

            # Apply rank_partition filtering
            if rank_partition and not is_final and r.get("partition") != rank_partition:
                continue

            # Apply custom filters
            if filter_items and not all(r.get(k) == v for k, v in filter_items):
                continue

            # Only copy entries that pass all filters
            c = dict(r)
            c["is_final"] = is_final
            candidates.append(c)

        if not candidates:
            if return_grouped:
                return {}
            return PredictionResultsList([])

        # Resolve effective metric
        effective_metric = rank_metric or candidates[0].get("metric", "mse")

        # Determine sort direction
        if ascending is None:
            ascending = _infer_ascending(effective_metric)

        # Compute rank_score for each candidate
        partition_key = f"{rank_partition}_score" if rank_partition in ("val", "test", "train") else "val_score"
        for r in candidates:
            if r["is_final"]:
                # Final entries rank by their selection score
                r["rank_score"] = r.get("selection_score")
            else:
                score = self._get_rank_score(r, effective_metric, rank_partition, partition_key, effective_by_repetition, effective_repetition_method, effective_repetition_exclude_outliers)
                r["rank_score"] = score

        # Filter out None / NaN scores
        def _is_valid(score: Any) -> bool:
            if score is None:
                return False
            try:
                return not np.isnan(score)
            except (TypeError, ValueError):
                return True

        candidates = [r for r in candidates if _is_valid(r["rank_score"])]

        # Sort by rank_score
        if effective_scope == "mix":
            # Two-level sort: final entries first, then CV entries
            # Within each group, sort by rank_score
            finals = [r for r in candidates if r["is_final"]]
            cvs = [r for r in candidates if r.get("refit_context") is None]
            finals.sort(key=lambda r: r["rank_score"], reverse=not ascending)
            cvs.sort(key=lambda r: r["rank_score"], reverse=not ascending)
            candidates = finals + cvs
        else:
            candidates.sort(key=lambda r: r["rank_score"], reverse=not ascending)

        effective_group_by: list[str] | None = None
        if group_by is not None:
            effective_group_by = [group_by] if isinstance(group_by, str) else list(group_by)

        # Apply group-by filtering (top N per group)
        if effective_group_by:
            group_counts: dict[tuple, int] = {}
            filtered_candidates: list[dict[str, Any]] = []
            for r in candidates:
                gk = _make_group_key(r, effective_group_by)
                cnt = group_counts.get(gk, 0)
                if cnt < n:
                    r["group_key"] = gk
                    filtered_candidates.append(r)
                    group_counts[gk] = cnt + 1
            candidates = filtered_candidates
        else:
            candidates = candidates[:n]

        # Enrich results
        enriched_results = [
            PredictionResult(self._enrich_result(r, display_metrics, display_partition, aggregate_partitions, effective_by_repetition))
            for r in candidates
        ]

        if return_grouped and effective_group_by:
            grouped_out: dict[tuple, PredictionResultsList] = {}
            for res in enriched_results:
                gk = res.get("group_key")
                if gk is not None:
                    if gk not in grouped_out:
                        grouped_out[gk] = PredictionResultsList([])
                    grouped_out[gk].append(res)
            return grouped_out

        return PredictionResultsList(enriched_results)

    def _get_rank_score(
        self,
        row: dict[str, Any],
        metric: str,
        partition: str,
        partition_key: str,
        by_repetition: str | None = None,
        repetition_method: str | None = None,
        repetition_exclude_outliers: bool = False,
    ) -> float | None:
        """Compute the ranking score for a single buffer row.

        Priority:
        1. If *by_repetition* is set, apply aggregation and recompute.
        2. Pre-computed scores dict (``scores[partition][metric]``).
        3. Legacy ``{partition}_score`` if metric matches the stored metric.
        4. Fall back to computing from arrays.
        """
        # Aggregated path: must compute from arrays after aggregation
        if by_repetition:
            y_true, y_pred = row.get("y_true"), row.get("y_pred")
            if y_true is not None and isinstance(y_true, np.ndarray) and y_true.size > 0 and y_pred is not None and isinstance(y_pred, np.ndarray) and y_pred.size > 0:
                metadata = row.get("metadata", {})
                agg_y_true, agg_y_pred, _, was_agg = self._apply_aggregation(
                    y_true, y_pred, row.get("y_proba"), metadata, by_repetition,
                    row.get("model_name", ""), repetition_method, repetition_exclude_outliers,
                )
                if was_agg and agg_y_true is not None and agg_y_pred is not None:
                    try:
                        return evaluator.eval(agg_y_true, agg_y_pred, metric)
                    except Exception:
                        return None
            # Fall through to non-aggregated path if aggregation fails

        # Pre-computed scores dict
        scores_dict = row.get("scores")
        if isinstance(scores_dict, dict) and partition in scores_dict and metric in scores_dict[partition]:
            return scores_dict[partition][metric]
        if isinstance(scores_dict, str):
            try:
                parsed = json.loads(scores_dict)
                if partition in parsed and metric in parsed[partition]:
                    return parsed[partition][metric]
            except (json.JSONDecodeError, TypeError):
                pass

        # Legacy field
        if metric == row.get("metric") or metric == "":
            val = row.get(partition_key)
            if val is not None:
                return float(val)

        # Compute from arrays
        y_true, y_pred = row.get("y_true"), row.get("y_pred")
        if y_true is not None and isinstance(y_true, np.ndarray) and y_true.size > 0 and y_pred is not None and isinstance(y_pred, np.ndarray) and y_pred.size > 0:
            try:
                return evaluator.eval(y_true, y_pred, metric)
            except Exception:
                return None

        return row.get(partition_key)

    @staticmethod
    def _apply_aggregation(
        y_true: np.ndarray | None,
        y_pred: np.ndarray | None,
        y_proba: np.ndarray | None,
        metadata: dict[str, Any],
        by_repetition: str,
        model_name: str = "",
        repetition_method: str | None = None,
        repetition_exclude_outliers: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
        """Apply aggregation to predictions by a group column.

        Args:
            y_true: True values array.
            y_pred: Predicted values array.
            y_proba: Optional class probabilities.
            metadata: Metadata dict containing group column.
            by_repetition: Group column name or ``"y"``.
            model_name: Model name for warning messages.
            repetition_method: ``"mean"``, ``"median"``, or ``"vote"``.
            repetition_exclude_outliers: Exclude outliers before aggregation.

        Returns:
            Tuple ``(y_true, y_pred, y_proba, was_aggregated)``.
        """
        if y_pred is None:
            return y_true, y_pred, y_proba, False

        if by_repetition == "y":
            if y_true is None:
                return y_true, y_pred, y_proba, False
            group_ids = y_true
        else:
            if by_repetition not in metadata:
                if model_name:
                    warnings.warn(
                        f"Aggregation column '{by_repetition}' not found in metadata for model '{model_name}'. "
                        f"Available columns: {list(metadata.keys())}. Skipping aggregation.",
                        UserWarning,
                        stacklevel=2,
                    )
                return y_true, y_pred, y_proba, False
            group_ids = np.asarray(metadata[by_repetition])

        if len(group_ids) != len(y_pred):
            if model_name:
                warnings.warn(
                    f"Aggregation column '{by_repetition}' length ({len(group_ids)}) doesn't match "
                    f"predictions length ({len(y_pred)}) for model '{model_name}'. Skipping aggregation.",
                    UserWarning,
                    stacklevel=2,
                )
            return y_true, y_pred, y_proba, False

        result = Predictions.aggregate(
            y_pred=y_pred,
            group_ids=group_ids,
            y_proba=y_proba,
            y_true=y_true,
            method=repetition_method or "mean",
            exclude_outliers=repetition_exclude_outliers,
        )
        return result.get("y_true"), result.get("y_pred"), result.get("y_proba"), True

    def get_best(
        self,
        metric: str = "",
        ascending: bool | None = None,
        score_scope: str = "mix",
        aggregate_partitions: bool = False,
        by_repetition: bool | str | None = None,
        repetition_method: str | None = None,
        repetition_exclude_outliers: bool = False,
        **filters: Any,
    ) -> PredictionResult | None:
        """Get the best prediction for a specific metric.

        Tries ranking by ``"val"`` partition first; falls back to
        ``"test"`` if no val-partition data exists.

        Args:
            metric: Metric to optimise.
            ascending: Sort order.  ``None`` infers from metric.
            score_scope: Controls how refit (final) entries interact with
               CV entries.  See :meth:`top` for details.  Default ``"mix"``.
            aggregate_partitions: If ``True``, add partition data.
            by_repetition: Aggregate predictions by repetition column.
                - ``True``: Uses ``dataset.repetition`` from context.
                - ``str``: Explicit column name or ``"y"`` for target grouping.
                - ``False``/``None`` (default): No aggregation.
            repetition_method: Aggregation method (``"mean"``, ``"median"``, ``"vote"``).
            repetition_exclude_outliers: Exclude outliers before aggregation.
            **filters: Additional filter criteria.

        Returns:
            Best :class:`PredictionResult` or ``None``.
        """
        results = self.top(
            n=1,
            rank_metric=metric,
            rank_partition="val",
            score_scope=score_scope,
            ascending=ascending,
            aggregate_partitions=aggregate_partitions,
            by_repetition=by_repetition,
            repetition_method=repetition_method,
            repetition_exclude_outliers=repetition_exclude_outliers,
            **filters,
        )
        # Fallback to test partition
        if not results:
            results = self.top(
                n=1,
                rank_metric=metric,
                rank_partition="test",
                score_scope=score_scope,
                ascending=ascending,
                aggregate_partitions=aggregate_partitions,
                by_repetition=by_repetition,
                repetition_method=repetition_method,
                repetition_exclude_outliers=repetition_exclude_outliers,
                **filters,
            )
        return results[0] if results else None

    # =========================================================================
    # FILTERING OPERATIONS
    # =========================================================================

    def filter_predictions(
        self,
        dataset_name: str | None = None,
        partition: str | None = None,
        config_name: str | None = None,
        model_name: str | None = None,
        fold_id: str | None = None,
        step_idx: int | None = None,
        branch_id: int | None = None,
        branch_name: str | None = None,
        load_arrays: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Filter predictions and return as list of dicts.

        Args:
            dataset_name: Filter by dataset name.
            partition: Filter by partition.
            config_name: Filter by config name.
            model_name: Filter by model name.
            fold_id: Filter by fold ID.
            step_idx: Filter by step index.
            branch_id: Filter by branch ID.
            branch_name: Filter by branch name.
            load_arrays: If ``True``, include arrays (y_true, y_pred).
            **kwargs: Additional filter criteria.

        Returns:
            List of matching prediction dicts.
        """
        filter_map: dict[str, Any] = {}
        if dataset_name is not None:
            filter_map["dataset_name"] = dataset_name
        if partition is not None:
            filter_map["partition"] = partition
        if config_name is not None:
            filter_map["config_name"] = config_name
        if model_name is not None:
            filter_map["model_name"] = model_name
        if fold_id is not None:
            filter_map["fold_id"] = str(fold_id)
        if step_idx is not None:
            filter_map["step_idx"] = step_idx
        if branch_id is not None:
            # Accept both int and string since get_unique_values returns strings
            filter_map["branch_id"] = int(branch_id) if isinstance(branch_id, str) and branch_id.isdigit() else branch_id
        if branch_name is not None:
            filter_map["branch_name"] = branch_name
        filter_map.update(kwargs)

        # Single-pass filter instead of sequential list comprehensions
        filter_items = list(filter_map.items())
        _ARRAY_KEYS = {"y_true", "y_pred", "y_proba"}
        results: list[dict[str, Any]] = []
        for row in self._buffer:
            if all(row.get(k) == v for k, v in filter_items):
                if not load_arrays:
                    results.append({k: v for k, v in row.items() if k not in _ARRAY_KEYS})
                else:
                    results.append(row)

        return results

    def get_similar(self, **filter_kwargs: Any) -> dict[str, Any] | None:
        """Get the first prediction matching filter criteria.

        Args:
            **filter_kwargs: Filter criteria.

        Returns:
            First matching prediction or ``None``.
        """
        results = self.filter_predictions(**filter_kwargs)
        return results[0] if results else None

    def get_prediction_by_id(self, prediction_id: str, load_arrays: bool = True) -> dict[str, Any] | None:
        """Get a single prediction by its ID.

        Args:
            prediction_id: Unique prediction identifier.
            load_arrays: If ``True``, include arrays.

        Returns:
            Prediction dict or ``None``.
        """
        for row in self._buffer:
            if row.get("id") == prediction_id:
                if not load_arrays:
                    return {k: v for k, v in row.items() if k not in ("y_true", "y_pred", "y_proba")}
                return dict(row)
        return None

    # =========================================================================
    # METADATA / UTILITY
    # =========================================================================

    @property
    def num_predictions(self) -> int:
        """Number of predictions currently buffered."""
        return len(self._buffer)

    def get_unique_values(self, column: str) -> list[str]:
        """Get unique values for a column across buffered predictions.

        Args:
            column: Column name.

        Returns:
            List of unique values (as strings).
        """
        seen: set[str] = set()
        for row in self._buffer:
            val = row.get(column)
            if val is not None and str(val):
                seen.add(str(val))
        return sorted(seen)

    def get_datasets(self) -> list[str]:
        """Get list of unique dataset names."""
        return self.get_unique_values("dataset_name")

    def get_partitions(self) -> list[str]:
        """Get list of unique partitions."""
        return self.get_unique_values("partition")

    def get_configs(self) -> list[str]:
        """Get list of unique config names."""
        return self.get_unique_values("config_name")

    def get_models(self) -> list[str]:
        """Get list of unique model names."""
        return self.get_unique_values("model_name")

    def get_folds(self) -> list[str]:
        """Get list of unique fold IDs."""
        return self.get_unique_values("fold_id")

    # =========================================================================
    # MERGE / CLEAR
    # =========================================================================

    def merge_predictions(self, other: Predictions) -> None:
        """Merge predictions from another instance into this one.

        Also inherits the repetition column from the other instance if
        this instance doesn't have one set.

        Args:
            other: Another :class:`Predictions` instance.
        """
        self._buffer.extend(other._buffer)
        # Inherit repetition column if not already set
        if self._dataset_repetition is None and other._dataset_repetition is not None:
            self._dataset_repetition = other._dataset_repetition

    def clear(self) -> None:
        """Clear all buffered predictions."""
        self._buffer.clear()

    # =========================================================================
    # PARTITION HELPERS
    # =========================================================================

    def get_entry_partitions(self, entry: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
        """Get all partition data for an entry.

        Finds train, val, and test partitions in a single pass over
        the buffer instead of three separate ``filter_predictions`` calls.

        Args:
            entry: Prediction entry dict.

        Returns:
            Dict with ``train``, ``val``, ``test`` keys.
        """
        target_dataset = entry["dataset_name"]
        target_config = entry.get("config_name", "")
        target_model = entry["model_name"]
        target_fold = entry.get("fold_id", "")
        target_step = entry.get("step_idx", 0)

        res: dict[str, dict[str, Any] | None] = {"train": None, "val": None, "test": None}
        found = 0
        for row in self._buffer:
            part = row.get("partition")
            if part not in res or res[part] is not None:
                continue
            if (
                row.get("dataset_name") == target_dataset
                and row.get("config_name", "") == target_config
                and row.get("model_name") == target_model
                and row.get("fold_id", "") == target_fold
                and row.get("step_idx", 0) == target_step
            ):
                res[part] = row
                found += 1
                if found == 3:
                    break

        return res

    # =========================================================================
    # STACKING HELPERS
    # =========================================================================

    def get_predictions_by_step(self, step_idx: int, partition: str | None = None, branch_id: int | None = None, load_arrays: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
        """Get predictions from a specific pipeline step.

        Args:
            step_idx: Pipeline step index.
            partition: Optional partition filter.
            branch_id: Optional branch ID filter.
            load_arrays: If ``True``, load arrays.
            **kwargs: Additional filters.

        Returns:
            List of prediction dicts from the specified step.
        """
        return self.filter_predictions(step_idx=step_idx, partition=partition, branch_id=branch_id, load_arrays=load_arrays, **kwargs)

    def get_oof_predictions(
        self,
        model_name: str | None = None,
        step_idx: int | None = None,
        branch_id: int | None = None,
        exclude_averaged: bool = True,
        load_arrays: bool = True,
    ) -> list[dict[str, Any]]:
        """Get out-of-fold (validation) predictions.

        Args:
            model_name: Optional model name filter.
            step_idx: Optional step index filter.
            branch_id: Optional branch ID filter.
            exclude_averaged: If ``True``, exclude avg/w_avg folds.
            load_arrays: If ``True``, load arrays.

        Returns:
            List of validation partition predictions.
        """
        preds = self.filter_predictions(model_name=model_name, step_idx=step_idx, branch_id=branch_id, partition="val", load_arrays=load_arrays)
        if exclude_averaged:
            preds = [p for p in preds if p.get("fold_id") not in ("avg", "w_avg")]
        return preds

    def filter_by_branch(self, branch_id: int | None = None, branch_name: str | None = None, include_no_branch: bool = False, load_arrays: bool = True) -> list[dict[str, Any]]:
        """Filter predictions by branch context.

        Args:
            branch_id: Branch ID to filter by.
            branch_name: Branch name to filter by.
            include_no_branch: If ``True``, include branchless predictions.
            load_arrays: If ``True``, load arrays.

        Returns:
            List of branch-filtered predictions.
        """
        if branch_id is not None:
            preds = self.filter_predictions(branch_id=branch_id, load_arrays=load_arrays)
        elif branch_name is not None:
            preds = self.filter_predictions(branch_name=branch_name, load_arrays=load_arrays)
        else:
            preds = self.filter_predictions(load_arrays=load_arrays)

        if not include_no_branch:
            preds = [p for p in preds if p.get("branch_id") is not None or p.get("branch_name")]
        return preds

    def get_models_before_step(self, step_idx: int, branch_id: int | None = None, unique_names: bool = True) -> list[str]:
        """Get model names from steps before a given step index.

        Args:
            step_idx: Current step index.
            branch_id: Optional branch ID filter.
            unique_names: If ``True``, return unique names only.

        Returns:
            List of model names.
        """
        candidates = [r for r in self._buffer if r.get("step_idx", 0) < step_idx]
        if branch_id is not None:
            candidates = [r for r in candidates if r.get("branch_id") == branch_id]
        names = [r.get("model_name", "") for r in candidates]
        if unique_names:
            return sorted(set(names))
        return names

    # =========================================================================
    # CONVERSION
    # =========================================================================

    def to_dataframe(self) -> pl.DataFrame:
        """Get predictions as Polars DataFrame (metadata only, no arrays)."""
        if not self._buffer:
            return pl.DataFrame()
        # Filter out array columns for DataFrame representation
        safe_rows = []
        for row in self._buffer:
            safe = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    continue  # Skip arrays
                if isinstance(v, (dict, list)):
                    safe[k] = json.dumps(v, default=_json_default)
                else:
                    safe[k] = v
            safe_rows.append(safe)
        return pl.DataFrame(safe_rows)

    def to_dicts(self, load_arrays: bool = True) -> list[dict[str, Any]]:
        """Get predictions as list of dicts.

        Args:
            load_arrays: If ``True``, include arrays.

        Returns:
            List of prediction dicts.
        """
        if load_arrays:
            return [dict(r) for r in self._buffer]
        return [{k: v for k, v in r.items() if k not in ("y_true", "y_pred", "y_proba")} for r in self._buffer]

    def to_pandas(self) -> Any:
        """Get predictions as pandas DataFrame (metadata only)."""
        return self.to_dataframe().to_pandas()

    # =========================================================================
    # STATIC UTILITY METHODS
    # =========================================================================

    @staticmethod
    def save_predictions_to_csv(
        y_true: np.ndarray | list[float] | None = None,
        y_pred: np.ndarray | list[float] | None = None,
        filepath: str = "",
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """Save y_true and y_pred arrays to a CSV file.

        Args:
            y_true: True values array.
            y_pred: Predicted values array.
            filepath: Output CSV file path.
            prefix: Optional prefix for column names.
            suffix: Optional suffix for column names.
        """
        if y_pred is None:
            raise ValueError("y_pred is required")

        y_pred_arr = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        y_pred_flat = y_pred_arr.flatten()
        data_dict: dict[str, list[float]] = {f"{prefix}y_pred{suffix}": y_pred_flat.tolist()}

        if y_true is not None:
            y_true_arr = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
            y_true_flat = y_true_arr.flatten()
            if len(y_true_flat) != len(y_pred_flat):
                raise ValueError(f"Length mismatch: y_true ({len(y_true_flat)}) != y_pred ({len(y_pred_flat)})")
            data_dict[f"{prefix}y_true{suffix}"] = y_true_flat.tolist()

        df_csv = pl.DataFrame(data_dict)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df_csv.write_csv(filepath)
        logger.info(f"Saved predictions to {filepath}")

    @classmethod
    def pred_short_string(cls, entry: dict, metrics: list[str] | None = None, partition: str | list[str] = "test") -> str:
        """Generate short string representation of a prediction.

        Args:
            entry: Prediction dict.
            metrics: Optional list of metrics to display.
            partition: Partition(s) for score computation.

        Returns:
            Short description string.
        """
        scores_str = ""
        if metrics:
            metrics = metrics.copy()
            if isinstance(partition, str):
                partition = [partition]
            for p in partition:
                if "partitions" in entry and p in entry["partitions"]:
                    y_true = entry["partitions"][p].get("y_true")
                    y_pred = entry["partitions"][p].get("y_pred")
                else:
                    y_true = entry.get("y_true")
                    y_pred = entry.get("y_pred")
                if y_true is not None and y_pred is not None:
                    computed = evaluator.eval_list(y_true, y_pred, metrics=metrics)
                    scores_str += f" [{p}]: "
                    scores_str += ", ".join(f"[{k}:{v:.4f}]" for k, v in zip(metrics, computed, strict=False))

        metric_str = entry.get("metric", "N/A")
        test_score = entry.get("test_score")
        val_score = entry.get("val_score")
        fold_id = entry.get("fold_id", "")
        op_counter = entry.get("op_counter", entry.get("id", "unknown"))
        step_idx = entry.get("step_idx", 0)
        entry_id = entry.get("id", "unknown")

        desc = f"{entry.get('model_name', 'unknown')}"
        if metric_str != "N/A":
            desc += f" - {metric_str} "
            if test_score is not None and val_score is not None:
                desc += f"[test: {test_score:.4f}], [val: {val_score:.4f}]"
            elif test_score is not None:
                desc += f"[test: {test_score:.4f}]"
        if scores_str:
            desc += f", {scores_str}"
        desc += f", (fold: {fold_id}, id: {op_counter}, step: {step_idx}) - [{entry_id}]"
        return desc

    @classmethod
    def pred_long_string(cls, entry: dict, metrics: list[str] | None = None) -> str:
        """Generate long string representation of a prediction.

        Args:
            entry: Prediction dict.
            metrics: Optional list of metrics to display.

        Returns:
            Long description string with config.
        """
        return cls.pred_short_string(entry, metrics=metrics) + f" | [{entry.get('config_name', '')}]"

    @staticmethod
    def aggregate(
        y_pred: np.ndarray,
        group_ids: np.ndarray,
        y_proba: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        method: str = "mean",
        exclude_outliers: bool = False,
        outlier_threshold: float = 0.95,
    ) -> dict[str, Any]:
        """Aggregate predictions by group.

        For datasets with multiple samples per target (e.g., 4 measurements
        per sample ID), averages predictions within each group.

        Args:
            y_pred: Predicted values (n_samples,).
            group_ids: Group identifiers (n_samples,).
            y_proba: Optional class probabilities (n_samples, n_classes).
            y_true: Optional true values (n_samples,).
            method: ``"mean"``, ``"median"``, or ``"vote"``.
            exclude_outliers: If ``True``, exclude outliers per group.
            outlier_threshold: Confidence level for outlier detection.

        Returns:
            Dict with ``y_pred``, ``group_ids``, ``group_sizes`` and
            optional ``y_proba``, ``y_true``, ``outliers_excluded``.
        """
        y_pred = np.asarray(y_pred).flatten()
        group_ids = np.asarray(group_ids).flatten()
        if len(y_pred) != len(group_ids):
            raise ValueError(f"Length mismatch: y_pred ({len(y_pred)}) != group_ids ({len(group_ids)})")

        unique_groups, inverse_indices = np.unique(group_ids, return_inverse=True)
        n_groups = len(unique_groups)

        aggregated_pred = np.zeros(n_groups)
        group_sizes = np.zeros(n_groups, dtype=int)
        outliers_excluded = np.zeros(n_groups, dtype=int) if exclude_outliers else None

        if exclude_outliers:
            valid_mask = Predictions._compute_outlier_mask(y_pred, inverse_indices, n_groups, outlier_threshold)
            for g in range(n_groups):
                group_mask = inverse_indices == g
                outliers_excluded[g] = int(np.sum(group_mask)) - int(np.sum(group_mask & valid_mask))
        else:
            valid_mask = np.ones(len(y_pred), dtype=bool)

        for _i, (idx, valid) in enumerate(zip(inverse_indices, valid_mask, strict=False)):
            if valid:
                group_sizes[idx] += 1

        is_classification = y_proba is not None and y_proba.size > 0

        if is_classification:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                y_proba = np.column_stack([1 - y_proba, y_proba])
            n_classes = y_proba.shape[1]
            aggregated_proba = np.zeros((n_groups, n_classes))
            for _i, (group_idx, proba, valid) in enumerate(zip(inverse_indices, y_proba, valid_mask, strict=False)):
                if valid:
                    aggregated_proba[group_idx] += proba
            for g in range(n_groups):
                if group_sizes[g] > 0:
                    aggregated_proba[g] /= group_sizes[g]
            aggregated_pred = np.argmax(aggregated_proba, axis=1).astype(float)
        else:
            unique_preds = np.unique(y_pred[valid_mask])
            is_likely_classification = len(unique_preds) <= 20 and np.allclose(y_pred[valid_mask], np.round(y_pred[valid_mask]))
            effective_method = method
            if is_likely_classification and method == "mean":
                effective_method = "vote"

            if effective_method == "vote":
                from scipy import stats
                for g in range(n_groups):
                    mask = (inverse_indices == g) & valid_mask
                    if np.any(mask):
                        mode_result = stats.mode(y_pred[mask], keepdims=True)
                        aggregated_pred[g] = mode_result.mode[0]
            elif effective_method == "median":
                for g in range(n_groups):
                    mask = (inverse_indices == g) & valid_mask
                    if np.any(mask):
                        aggregated_pred[g] = np.median(y_pred[mask])
            else:
                for _i, (group_idx, pred, valid) in enumerate(zip(inverse_indices, y_pred, valid_mask, strict=False)):
                    if valid:
                        aggregated_pred[group_idx] += pred
                for g in range(n_groups):
                    if group_sizes[g] > 0:
                        aggregated_pred[g] /= group_sizes[g]

            is_classification = is_likely_classification
            aggregated_proba = None

        aggregated_true = None
        if y_true is not None:
            y_true = np.asarray(y_true).flatten()
            aggregated_true = np.zeros(n_groups)
            if is_classification:
                from scipy import stats
                for g in range(n_groups):
                    mask = (inverse_indices == g) & valid_mask
                    if np.any(mask):
                        mode_result = stats.mode(y_true[mask], keepdims=True)
                        aggregated_true[g] = mode_result.mode[0]
            else:
                if method == "median":
                    for g in range(n_groups):
                        mask = (inverse_indices == g) & valid_mask
                        if np.any(mask):
                            aggregated_true[g] = np.median(y_true[mask])
                else:
                    for _i, (group_idx, true_val, valid) in enumerate(zip(inverse_indices, y_true, valid_mask, strict=False)):
                        if valid:
                            aggregated_true[group_idx] += true_val
                    for g in range(n_groups):
                        if group_sizes[g] > 0:
                            aggregated_true[g] /= group_sizes[g]

        result: dict[str, Any] = {"y_pred": aggregated_pred, "group_ids": unique_groups, "group_sizes": group_sizes}
        if aggregated_proba is not None:
            result["y_proba"] = aggregated_proba
        if aggregated_true is not None:
            result["y_true"] = aggregated_true
        if outliers_excluded is not None:
            result["outliers_excluded"] = outliers_excluded
        return result

    @staticmethod
    def _compute_outlier_mask(y_pred: np.ndarray, inverse_indices: np.ndarray, n_groups: int, threshold: float = 0.95) -> np.ndarray:
        """Compute outlier mask using robust modified Z-score within each group.

        Args:
            y_pred: Predictions array.
            inverse_indices: Group assignment per sample.
            n_groups: Number of unique groups.
            threshold: Confidence level.

        Returns:
            Boolean mask where ``True`` = valid (non-outlier).
        """
        valid_mask = np.ones(len(y_pred), dtype=bool)
        if threshold >= 0.99:
            z_threshold = 4.5
        elif threshold >= 0.95:
            z_threshold = 3.5
        elif threshold >= 0.90:
            z_threshold = 3.0
        else:
            z_threshold = 2.5

        for g in range(n_groups):
            group_mask = inverse_indices == g
            group_preds = y_pred[group_mask]
            if len(group_preds) <= 2:
                continue
            median = np.median(group_preds)
            mad = np.median(np.abs(group_preds - median))
            if mad < 1e-10:
                continue
            modified_z = 0.6745 * (group_preds - median) / mad
            group_indices = np.where(group_mask)[0]
            for idx, mz in zip(group_indices, modified_z, strict=False):
                if abs(mz) > z_threshold:
                    valid_mask[idx] = False
        return valid_mask

    # =========================================================================
    # INTERNAL ENRICHMENT
    # =========================================================================

    def _enrich_result(
        self,
        row: dict[str, Any],
        display_metrics: list[str] | None = None,
        display_partition: str = "test",
        aggregate_partitions: bool = False,
        by_repetition: str | None = None,
    ) -> dict[str, Any]:
        """Enrich a raw buffer row into a user-facing result dict.

        Computes display metrics on-the-fly, applies aggregation when
        requested, and adds partition sub-dicts when
        ``aggregate_partitions`` is ``True``.
        """
        enriched = dict(row)

        # Apply aggregation to display arrays if requested
        y_true = row.get("y_true")
        y_pred = row.get("y_pred")
        y_proba = row.get("y_proba")
        was_aggregated = False

        if by_repetition and y_pred is not None and isinstance(y_pred, np.ndarray) and y_pred.size > 0:
            metadata = row.get("metadata", {})
            agg_y_true, agg_y_pred, agg_y_proba, was_aggregated = self._apply_aggregation(
                y_true, y_pred, y_proba, metadata, by_repetition, row.get("model_name", ""),
            )
            if was_aggregated:
                enriched["y_true"] = agg_y_true
                enriched["y_pred"] = agg_y_pred
                enriched["y_proba"] = agg_y_proba
                enriched["aggregated"] = True
                y_true, y_pred = agg_y_true, agg_y_pred

        # Compute display metrics on the fly
        if display_metrics:
            scores_dict = row.get("scores", {})
            if isinstance(scores_dict, str):
                try:
                    scores_dict = json.loads(scores_dict)
                except (json.JSONDecodeError, TypeError):
                    scores_dict = {}

            for m in display_metrics:
                # If aggregated, always recalculate
                if was_aggregated and y_true is not None and isinstance(y_true, np.ndarray) and y_true.size > 0 and y_pred is not None and isinstance(y_pred, np.ndarray) and y_pred.size > 0:
                    try:
                        enriched[m] = evaluator.eval(y_true, y_pred, m)
                    except Exception:
                        enriched[m] = None
                # Try pre-computed scores
                elif isinstance(scores_dict, dict) and display_partition in scores_dict and m in scores_dict[display_partition]:
                    enriched[m] = scores_dict[display_partition][m]
                # Compute from arrays
                elif y_true is not None and isinstance(y_true, np.ndarray) and y_true.size > 0 and y_pred is not None and isinstance(y_pred, np.ndarray) and y_pred.size > 0:
                    try:
                        enriched[m] = evaluator.eval(y_true, y_pred, m)
                    except Exception:
                        enriched[m] = None
                else:
                    enriched[m] = None

        # Aggregate partitions if requested
        if aggregate_partitions:
            partitions_data = self.get_entry_partitions(row)
            enriched["partitions"] = {}
            for part_name, part_data in partitions_data.items():
                if part_data is not None:
                    enriched["partitions"][part_name] = part_data

        return enriched

    # =========================================================================
    # DUNDER METHODS
    # =========================================================================

    def __len__(self) -> int:
        """Return number of buffered predictions."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"Predictions({len(self._buffer)} entries)" if self._buffer else "Predictions(empty)"

    def __str__(self) -> str:
        if not self._buffer:
            return "Predictions: No predictions stored"
        datasets = self.get_datasets()
        models = self.get_models()
        return f"Predictions: {len(self._buffer)} entries\n   Datasets: {datasets}\n   Models: {models}"
