"""
Result classes for nirs4all API.

These dataclasses wrap the outputs from pipeline execution, prediction,
and explanation operations, providing convenient accessor methods.

Classes:
    RunResult: Result from nirs4all.run()
    PredictResult: Result from nirs4all.predict()
    ExplainResult: Result from nirs4all.explain()

Phase 1 Implementation (v0.6.0):
    - RunResult: Full implementation with best, best_score, top(), export()
    - PredictResult: Full implementation with values, to_dataframe()
    - ExplainResult: Full implementation with values, feature attributions
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from nirs4all.core.logging import get_logger

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline import PipelineRunner
    from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
    from nirs4all.pipeline.execution.refit.model_selector import PerModelSelection

logger = get_logger(__name__)

@dataclass
class ModelRefitResult:
    """Per-model refit result metadata.

    Captures the refit outcome for a single model node in the pipeline.
    For Phase 2 (non-stacking), there is only one model node, so
    ``RunResult.models`` will contain exactly one entry.

    Attributes:
        model_name: Name of the model (e.g. ``"PLSRegression"``).
        final_entry: The refit prediction entry (``fold_id="final"``).
        cv_entry: The best CV prediction entry for this model.
        final_score: Test score from the refit model.
        cv_score: Best validation score from CV.
        metric: Metric name used for evaluation.
    """

    model_name: str = ""
    final_entry: dict[str, Any] = field(default_factory=dict)
    cv_entry: dict[str, Any] = field(default_factory=dict)
    final_score: float | None = None
    cv_score: float | None = None
    metric: str = ""

class LazyModelRefitResult:
    """Lazy per-model refit result that triggers refit on first access.

    Wraps a :class:`PerModelSelection` and defers the actual refit
    execution until a property requiring the result (e.g. ``.score``,
    ``.final_entry``, ``.export()``) is accessed.

    After the first access, the result is cached so subsequent accesses
    return instantly.

    If the underlying resources (artifact registry, step cache) have
    been destroyed by the time of access, the lazy refit re-executes
    from scratch.

    Attributes:
        model_name: Name of the model.
        selection: The PerModelSelection metadata from the CV phase.
    """

    def __init__(
        self,
        model_name: str,
        selection: PerModelSelection,
        refit_config: RefitConfig,
        dataset: Any,
        context: Any,
        runtime_context: Any,
        artifact_registry: Any,
        executor: Any,
        prediction_store: Any,
    ) -> None:
        self.model_name = model_name
        self.selection = selection
        self._refit_config = refit_config
        self._dataset = dataset
        self._context = context
        self._runtime_context = runtime_context
        self._artifact_registry = artifact_registry
        self._executor = executor
        self._prediction_store = prediction_store
        self._result: ModelRefitResult | None = None
        self._lock = threading.Lock()

    def _execute_refit(self) -> ModelRefitResult:
        """Execute the refit and return a ModelRefitResult.

        Thread-safe: uses a lock to prevent concurrent refits.
        """
        with self._lock:
            if self._result is not None:
                return self._result

            # Build a RefitConfig from the selection metadata
            from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
            from nirs4all.pipeline.execution.refit.executor import execute_simple_refit

            refit_config = RefitConfig(
                expanded_steps=self.selection.expanded_steps,
                best_params=self.selection.best_params,
                variant_index=self.selection.variant_index,
                metric=self._refit_config.metric,
                selection_score=self.selection.best_score,
            )

            # Create a fresh prediction store for capturing refit predictions
            from nirs4all.data.predictions import Predictions

            refit_predictions = Predictions()

            try:
                refit_result = execute_simple_refit(
                    refit_config=refit_config,
                    dataset=self._dataset,
                    context=self._context,
                    runtime_context=self._runtime_context,
                    artifact_registry=self._artifact_registry,
                    executor=self._executor,
                    prediction_store=refit_predictions,
                )
            except Exception:
                logger.warning(
                    f"Lazy refit for model '{self.model_name}' failed. "
                    f"Resources may have been destroyed."
                )
                # Return a minimal result with just the CV info
                self._result = ModelRefitResult(
                    model_name=self.model_name,
                    cv_score=self.selection.best_score,
                    metric=self._refit_config.metric,
                )
                return self._result

            # Build ModelRefitResult from the refit output
            final_entries = refit_predictions.iter_entries(fold_id="final")
            final_entry = final_entries[0] if final_entries else {}

            self._result = ModelRefitResult(
                model_name=self.model_name,
                final_entry=final_entry,
                cv_entry={},
                final_score=refit_result.test_score,
                cv_score=self.selection.best_score,
                metric=refit_result.metric,
            )
            return self._result

    def _ensure_result(self) -> ModelRefitResult:
        """Ensure the refit has been executed and return the result."""
        if self._result is None:
            return self._execute_refit()
        return self._result

    @property
    def score(self) -> float | None:
        """Get the refit model's test score (triggers refit on first access)."""
        return self._ensure_result().final_score

    @property
    def final_score(self) -> float | None:
        """Get the refit model's test score (triggers refit on first access)."""
        return self._ensure_result().final_score

    @property
    def final_entry(self) -> dict[str, Any]:
        """Get the refit prediction entry (triggers refit on first access)."""
        return self._ensure_result().final_entry

    @property
    def cv_entry(self) -> dict[str, Any]:
        """Get the best CV prediction entry."""
        return self._ensure_result().cv_entry

    @property
    def cv_score(self) -> float | None:
        """Get the best CV validation score (does not trigger refit)."""
        return self.selection.best_score

    @property
    def metric(self) -> str:
        """Get the metric name (does not trigger refit)."""
        return self._refit_config.metric

    @property
    def is_resolved(self) -> bool:
        """Check if the refit has already been executed."""
        return self._result is not None

    def __repr__(self) -> str:
        status = "resolved" if self.is_resolved else "pending"
        return f"LazyModelRefitResult(model='{self.model_name}', status={status})"

@dataclass
class RunResult:
    """Result from nirs4all.run().

    Provides convenient access to predictions, best model, and artifacts.
    Wraps the raw (predictions, per_dataset) tuple returned by PipelineRunner.run().

    Attributes:
        predictions: Predictions object containing all pipeline results.
        per_dataset: Dictionary with per-dataset execution details.

    Properties:
        best: Best prediction entry by default ranking.
        best_score: Best model's primary test score (CV selection metric).
        best_rmse: Best model's RMSE (regression).
        best_r2: Best model's R² (regression).
        best_accuracy: Best model's accuracy (classification).
        final: Refit model prediction entry (``fold_id="final"``), or ``None``.
        final_score: Refit model test score, or ``None`` if no refit.
        cv_best: Best CV prediction entry.
        cv_best_score: Best CV validation score.
        models: Per-model refit results (for Phase 2, one entry).
        artifacts_path: Path to run artifacts directory.
        num_predictions: Total number of predictions stored.

    Key Operations:
        top(n): Get top N predictions by ranking.
        export(path): Export best model to .n4a bundle.
        filter(**kwargs): Filter predictions by criteria.
        get_datasets(): Get list of unique dataset names.
        get_models(): Get list of unique model names.

    Example:
        >>> result = nirs4all.run(pipeline, dataset)
        >>> print(f"Best RMSE: {result.best_rmse:.4f}")
        >>> print(f"Best R²: {result.best_r2:.4f}")
        >>> result.export("exports/best_model.n4a")
    """

    predictions: Predictions
    per_dataset: dict[str, Any]
    _runner: PipelineRunner | None = field(default=None, repr=False)

    # Lazy refit dependencies (set by the orchestrator when per-model
    # selections are available so that ``models`` returns lazy results)
    _per_model_selections: dict[str, PerModelSelection] | None = field(default=None, repr=False)
    _refit_config: RefitConfig | None = field(default=None, repr=False)
    _refit_dataset: Any = field(default=None, repr=False)
    _refit_context: Any = field(default=None, repr=False)
    _refit_runtime_context: Any = field(default=None, repr=False)
    _refit_artifact_registry: Any = field(default=None, repr=False)
    _refit_executor: Any = field(default=None, repr=False)

    # --- Primary accessors ---

    @property
    def best(self) -> dict[str, Any]:
        """Get the best prediction entry, preferring refit (final) models.

        When refit entries exist, returns the best final entry.
        Otherwise falls back to the best CV entry.

        Returns:
            Dictionary containing best model's metrics, name, and configuration.
            Empty dict if no predictions available.
        """
        final = self.best_final
        if final:
            return final
        return self.cv_best

    @property
    def best_score(self) -> float:
        """Get best model's primary test score.

        Returns:
            The test_score value from best prediction, or NaN if unavailable.
        """
        return float(self.best.get('test_score', float('nan')))

    @property
    def best_rmse(self) -> float:
        """Get best model's RMSE score.

        Looks for 'rmse' as a flat key (from display_metrics), then in scores dict,
        then falls back to test_score if metric is rmse-like.

        Returns:
            RMSE value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        # Try flat 'rmse' key first (from display_metrics)
        if 'rmse' in best and best['rmse'] is not None:
            return float(best['rmse'])

        # Try nested scores dict (legacy format)
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'rmse' in test_scores and test_scores['rmse'] is not None:
                return float(test_scores['rmse'])

        # Fall back to test_score if metric is rmse-like
        metric = best.get('metric', '')
        if metric in ('rmse', 'mse'):
            test_score = best.get('test_score')
            if test_score is not None:
                return float(test_score)

        return float('nan')

    @property
    def best_r2(self) -> float:
        """Get best model's R² score.

        Looks for 'r2' as a flat key (from display_metrics), then in scores dict.

        Returns:
            R² value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        # Try flat 'r2' key first (from display_metrics)
        if 'r2' in best and best['r2'] is not None:
            return float(best['r2'])

        # Try nested scores dict (legacy format)
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'r2' in test_scores and test_scores['r2'] is not None:
                return float(test_scores['r2'])

        return float('nan')

    @property
    def best_accuracy(self) -> float:
        """Get best model's accuracy score (for classification).

        Looks for 'accuracy' as a flat key (from display_metrics), then in scores dict.

        Returns:
            Accuracy value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        # Try flat 'accuracy' key first (from display_metrics)
        if 'accuracy' in best and best['accuracy'] is not None:
            return float(best['accuracy'])

        # Try nested scores dict (legacy format)
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'accuracy' in test_scores and test_scores['accuracy'] is not None:
                return float(test_scores['accuracy'])

        # Fall back to test_score if metric is accuracy
        metric = best.get('metric', '')
        if metric == 'accuracy':
            test_score = best.get('test_score')
            if test_score is not None:
                return float(test_score)

        return float('nan')

    # --- Refit accessors ---

    @property
    def best_final(self) -> dict[str, Any]:
        """Get the best refit entry across all models.

        Filters predictions to ``fold_id="final"`` entries and ranks them
        by their selection score (``selection_score``).

        Returns:
            Best refit prediction dict, or empty dict if no refit entries.
        """
        results = self.predictions.top(n=1, score_scope="final")
        top = cast(list, results)
        return top[0] if top else {}

    @property
    def final(self) -> dict[str, Any] | None:
        """Get the refit model prediction entry (``fold_id="final"``).

        Searches the per-dataset prediction stores where refit entries
        are stored (they are not merged into the global predictions
        buffer to avoid polluting CV-centric ranking).

        Returns:
            Prediction dict for the refit model, or ``None`` if refit
            was not performed or no refit entries exist.
        """
        # Search per-dataset prediction stores (refit entries live here)
        for ds_info in self.per_dataset.values():
            ds_preds = ds_info.get("run_predictions")
            if ds_preds is None:
                continue
            entries = ds_preds.filter_predictions(fold_id="final")
            for entry in entries:
                if str(entry.get("fold_id")) == "final":
                    return dict(entry)
        # Fallback: check global predictions (for backward compatibility)
        entries = self.predictions.filter_predictions(fold_id="final")
        for entry in entries:
            if str(entry.get("fold_id")) == "final":
                return dict(entry)
        return None

    @property
    def final_score(self) -> float | None:
        """Get the refit model's test score.

        Returns:
            Test score from the refit entry, or ``None`` if refit was
            not performed.
        """
        entry = self.final
        if entry is None:
            return None
        score = entry.get("test_score")
        if score is not None:
            return float(score)
        return None

    @property
    def cv_best(self) -> dict[str, Any]:
        """Get the best CV prediction entry (excludes refit entries).

        This is the prediction entry that won the cross-validation
        selection phase.  Refit entries (``fold_id="final"``) are
        excluded from ranking.

        Returns:
            Best CV prediction dict, or empty dict if no CV predictions.
        """
        results = self.predictions.top(n=1, score_scope="cv")
        top = cast(list, results)
        return top[0] if top else {}

    @property
    def cv_best_score(self) -> float:
        """Get the best CV validation score.

        Returns:
            The val_score from the best CV entry, or NaN if unavailable.
        """
        entry = self.cv_best
        if not entry:
            return float("nan")
        score = entry.get("val_score")
        if score is not None:
            return float(score)
        return float("nan")

    @property
    def models(self) -> dict[str, ModelRefitResult | LazyModelRefitResult]:
        """Get per-model refit results.

        When per-model selections are available (set by the orchestrator),
        returns :class:`LazyModelRefitResult` instances that defer the
        actual refit until a property requiring the result is accessed.

        When per-model selections are not available, falls back to
        the eager approach using already-executed refit entries.

        Returns:
            Dictionary mapping model name to :class:`ModelRefitResult`
            or :class:`LazyModelRefitResult`.
        """
        # Lazy path: per-model selections were stored by the orchestrator
        if self._per_model_selections is not None and self._refit_config is not None:
            result: dict[str, ModelRefitResult | LazyModelRefitResult] = {}
            for model_name, selection in self._per_model_selections.items():
                result[model_name] = LazyModelRefitResult(
                    model_name=model_name,
                    selection=selection,
                    refit_config=self._refit_config,
                    dataset=self._refit_dataset,
                    context=self._refit_context,
                    runtime_context=self._refit_runtime_context,
                    artifact_registry=self._refit_artifact_registry,
                    executor=self._refit_executor,
                    prediction_store=self.predictions,
                )
            return result

        # Eager fallback: use already-executed refit entries
        final_entry = self.final
        if final_entry is None:
            return {}

        model_name = final_entry.get("model_name", "unknown")
        cv_entry = self.cv_best
        metric = final_entry.get("metric", "")

        final_score_val = final_entry.get("test_score")
        cv_score_val = cv_entry.get("val_score") if cv_entry else None

        return {
            model_name: ModelRefitResult(
                model_name=model_name,
                final_entry=final_entry,
                cv_entry=cv_entry,
                final_score=float(final_score_val) if final_score_val is not None else None,
                cv_score=float(cv_score_val) if cv_score_val is not None else None,
                metric=metric,
            )
        }

    # --- Metadata accessors ---

    @property
    def artifacts_path(self) -> Path | None:
        """Get path to workspace artifacts directory.

        Returns:
            Path to the workspace directory, or None if not available.
        """
        if self._runner and hasattr(self._runner, 'workspace_path'):
            return self._runner.workspace_path
        return None

    @property
    def num_predictions(self) -> int:
        """Get total number of predictions stored.

        Returns:
            Number of prediction entries.
        """
        return self.predictions.num_predictions

    # --- Query methods ---

    def top(self, n: int = 5, **kwargs) -> Any:
        """Get top N predictions by ranking.

        Args:
            n: Number of top predictions to return. When group_by is used,
               this means top N **per group** (e.g., top 3 per dataset).
            **kwargs: Additional arguments passed to predictions.top().
                Supported kwargs include:
                - rank_metric: Metric to rank by (default: uses record's metric)
                - rank_partition: Partition to rank on (default: "val")
                - display_partition: Partition for display metrics (default: "test")
                - aggregate_partitions: If True, include train/val/test data
                - ascending: Sort order (None = infer from metric)
                - group_by: Group predictions by column(s). Returns top N per group.
                  Each result includes 'group_key' for easy filtering.
                - return_grouped: If True with group_by, return dict of group->results
                  instead of flat list. Default: False.

        Returns:
            - If return_grouped=False (default): List of prediction dicts,
              ranked by score. With group_by, returns top N per group as flat list.
            - If return_grouped=True: Dict mapping group keys to lists of predictions.

        Examples:
            >>> # Top 5 overall
            >>> result.top(5)
            >>>
            >>> # Top 3 per dataset (flat list)
            >>> top_per_ds = result.top(3, group_by='dataset_name')
            >>> ds1 = [r for r in top_per_ds if r['group_key'] == ('my_dataset',)]
            >>>
            >>> # Top 3 per dataset (grouped dict)
            >>> grouped = result.top(3, group_by='dataset_name', return_grouped=True)
            >>> for key, results in grouped.items():
            ...     print(f"{key}: {len(results)} results")
            >>>
            >>> # Multi-column grouping: top 2 per (dataset, model) combination
            >>> top_per_combo = result.top(2, group_by=['dataset_name', 'model_name'])
            >>> # Group keys are tuples: ('wheat', 'PLSRegression'), ('corn', 'RandomForest')
            >>> for r in top_per_combo:
            ...     dataset, model = r['group_key']
            ...     print(f"{dataset}/{model}: {r['test_score']:.4f}")
        """
        return self.predictions.top(n=n, **kwargs)

    def filter(self, **kwargs) -> list[dict[str, Any]]:
        """Filter predictions by criteria.

        Args:
            **kwargs: Filter criteria passed to predictions.filter_predictions().
                Supported kwargs include:
                - dataset_name: Filter by dataset name
                - model_name: Filter by model name
                - partition: Filter by partition ('train', 'val', 'test')
                - fold_id: Filter by fold ID
                - step_idx: Filter by pipeline step index
                - branch_id: Filter by branch ID
                - load_arrays: If True, load actual arrays (default: True)

        Returns:
            List of matching prediction dictionaries.
        """
        return self.predictions.filter_predictions(**kwargs)

    def get_datasets(self) -> list[str]:
        """Get list of unique dataset names.

        Returns:
            List of dataset names in predictions.
        """
        return self.predictions.get_datasets()

    def get_models(self) -> list[str]:
        """Get list of unique model names.

        Returns:
            List of model names in predictions.
        """
        return self.predictions.get_models()

    # --- Export methods ---

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
    ) -> Path:
        """Export a model to bundle.

        Two export paths are supported:

        **Store-based** (preferred) -- pass ``chain_id`` to export
        directly from the DuckDB workspace:

        >>> result.export("model.n4a", chain_id="abc123")

        **Resolver-based** (legacy) -- exports via ``PipelineRunner.export``:

        >>> result.export("model.n4a")  # uses best prediction

        Args:
            output_path: Path for the exported bundle file.
            format: Export format ('n4a' or 'n4a.py').
            source: Prediction dict to export. If None, exports best model.
            chain_id: Chain identifier for store-based export.
                When provided, ``source`` is ignored and the chain is
                exported directly from the DuckDB store.

        Returns:
            Path to the exported bundle file.

        Raises:
            RuntimeError: If runner reference is not available.
            ValueError: If no predictions available and source not provided.
        """
        if self._runner is None:
            raise RuntimeError("Cannot export: runner reference not available")

        # Store-based export path
        if chain_id is not None:
            store = self._runner.store
            if store is None:
                raise RuntimeError(
                    "Cannot export from chain_id: no WorkspaceStore available on runner"
                )
            return Path(store.export_chain(chain_id, Path(output_path), format=format))

        # Legacy resolver-based path
        if source is None:
            # Prefer the refit entry when available (single model)
            source = self.final or self.best
            if not source:
                raise ValueError("No predictions available to export")

        return Path(self._runner.export(
            source=source,
            output_path=output_path,
            format=format
        ))

    def export_model(
        self,
        output_path: str | Path,
        source: dict[str, Any] | None = None,
        format: str | None = None,
        fold: int | None = None
    ) -> Path:
        """Export only the model artifact (lightweight).

        Unlike export() which creates a full bundle, this exports just the model.

        Args:
            output_path: Path for the output model file.
            source: Prediction dict to export. If None, exports best model.
            format: Model format (inferred from extension if None).
            fold: Fold index to export (default: fold 0).

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If runner reference is not available.
        """
        if self._runner is None:
            raise RuntimeError("Cannot export: runner reference not available")

        if source is None:
            source = self.best
            if not source:
                raise ValueError("No predictions available to export")

        return self._runner.export_model(
            source=source,
            output_path=output_path,
            format=format,
            fold=fold
        )

    # --- Utility methods ---

    def summary(self) -> str:
        """Get a summary string of the run result.

        Returns:
            Multi-line summary string with key metrics.
        """
        lines = []
        lines.append(f"RunResult: {self.num_predictions} predictions")

        if self.artifacts_path:
            lines.append(f"  Artifacts: {self.artifacts_path}")

        datasets = self.get_datasets()
        if datasets:
            lines.append(f"  Datasets: {', '.join(datasets)}")

        models = self.get_models()
        if models:
            lines.append(f"  Models: {', '.join(models[:5])}" +
                        (f" (+{len(models)-5} more)" if len(models) > 5 else ""))

        best = self.best
        if best:
            lines.append(f"  Best: {best.get('model_name', 'unknown')}")
            lines.append(f"    test_score: {self.best_score:.4f}")
            if not np.isnan(self.best_rmse):
                lines.append(f"    rmse: {self.best_rmse:.4f}")
            if not np.isnan(self.best_r2):
                lines.append(f"    r2: {self.best_r2:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"RunResult(predictions={self.num_predictions}, best_score={self.best_score:.4f})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.summary()

    def validate(
        self,
        check_nan_metrics: bool = True,
        check_empty: bool = True,
        raise_on_failure: bool = True,
        nan_threshold: float = 0.0
    ) -> dict[str, Any]:
        """Validate the run result for common issues.

        Checks for NaN values in metrics, empty predictions, and other issues
        that might indicate problems with the pipeline execution.

        Args:
            check_nan_metrics: If True, check for NaN values in metrics.
            check_empty: If True, check for empty predictions.
            raise_on_failure: If True, raise ValueError on validation failure.
            nan_threshold: Maximum allowed ratio of predictions with NaN metrics (0.0 = none allowed).

        Returns:
            Dictionary with validation results:
                - valid: True if all checks passed.
                - issues: List of issue descriptions.
                - nan_count: Number of predictions with NaN metrics.
                - total_count: Total number of predictions.

        Raises:
            ValueError: If raise_on_failure=True and validation fails.

        Example:
            >>> result = nirs4all.run(pipeline, dataset)
            >>> result.validate()  # Raises if issues found
            >>> # Or check without raising
            >>> report = result.validate(raise_on_failure=False)
            >>> if not report['valid']:
            ...     print(f"Issues: {report['issues']}")
        """
        issues = []
        nan_count = 0
        total_count = self.num_predictions

        # Check for empty predictions
        if check_empty and total_count == 0:
            issues.append("No predictions found")

        # Check for NaN metrics
        if check_nan_metrics and total_count > 0:
            all_preds = self.predictions.filter_predictions(load_arrays=False)
            for pred in all_preds:
                has_nan = False
                # Check common metrics
                for metric in ['rmse', 'r2', 'accuracy', 'mse', 'mae']:
                    value = pred.get(metric)
                    if value is not None and np.isnan(value):
                        has_nan = True
                        break

                # Check scores dict
                if not has_nan:
                    scores = pred.get('scores', {})
                    if isinstance(scores, dict):
                        for partition_scores in scores.values():
                            if isinstance(partition_scores, dict):
                                for val in partition_scores.values():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        has_nan = True
                                        break

                # Check score fields
                if not has_nan:
                    for score_key in ('test_score', 'val_score', 'train_score'):
                        score_val = pred.get(score_key)
                        if score_val is not None and isinstance(score_val, float) and np.isnan(score_val):
                            has_nan = True
                            break

                if has_nan:
                    nan_count += 1

            # Check threshold
            if nan_count > 0:
                nan_ratio = nan_count / total_count if total_count > 0 else 0
                if nan_ratio > nan_threshold:
                    issues.append(
                        f"NaN ratio ({nan_ratio:.1%}) exceeds threshold ({nan_threshold:.1%}): "
                        f"{nan_count} of {total_count} predictions have NaN metrics"
                    )

        valid = len(issues) == 0

        report = {
            'valid': valid,
            'issues': issues,
            'nan_count': nan_count,
            'total_count': total_count,
        }

        if raise_on_failure and not valid:
            raise ValueError(
                "RunResult validation failed:\n" +
                "\n".join(f"  - {issue}" for issue in issues)
            )

        return report

@dataclass
class PredictResult:
    """Result from nirs4all.predict().

    Wraps prediction outputs with convenient accessors and conversion methods.

    Attributes:
        y_pred: Predicted values array (n_samples,) or (n_samples, n_outputs).
        metadata: Additional prediction metadata (uncertainty, timing, etc.).
        sample_indices: Optional indices of predicted samples.
        model_name: Name of the model used for prediction.
        preprocessing_steps: List of preprocessing steps applied.

    Properties:
        values: Alias for y_pred (for consistency).
        shape: Shape of prediction array.
        is_multioutput: True if predictions have multiple outputs.

    Key Operations:
        to_numpy(): Get predictions as numpy array.
        to_list(): Get predictions as Python list.
        to_dataframe(): Get predictions as pandas DataFrame.
        flatten(): Get flattened 1D predictions.

    Example:
        >>> result = nirs4all.predict(model, X_new)
        >>> print(f"Predictions shape: {result.shape}")
        >>> df = result.to_dataframe()
    """

    y_pred: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_indices: np.ndarray | None = None
    model_name: str = ""
    preprocessing_steps: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure y_pred is a numpy array."""
        if self.y_pred is not None and not isinstance(self.y_pred, np.ndarray):
            self.y_pred = np.asarray(self.y_pred)

    @property
    def values(self) -> np.ndarray:
        """Get prediction values (alias for y_pred)."""
        return self.y_pred

    @property
    def shape(self) -> tuple:
        """Get shape of prediction array."""
        if self.y_pred is None:
            return (0,)
        return self.y_pred.shape

    @property
    def is_multioutput(self) -> bool:
        """Check if predictions have multiple outputs."""
        return len(self.shape) > 1 and self.shape[1] > 1

    def __len__(self) -> int:
        """Return number of predictions."""
        if self.y_pred is None:
            return 0
        return len(self.y_pred)

    def to_numpy(self) -> np.ndarray:
        """Get predictions as numpy array.

        Returns:
            Numpy array of predictions.
        """
        return self.y_pred

    def to_list(self) -> list[float]:
        """Get predictions as Python list.

        Returns:
            List of prediction values (flattened if 2D).
        """
        if self.y_pred is None:
            return []
        return self.y_pred.flatten().tolist()

    def to_dataframe(self, include_indices: bool = True):
        """Get predictions as pandas DataFrame.

        Args:
            include_indices: If True and sample_indices available, include as column.

        Returns:
            pandas DataFrame with predictions.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe()") from err

        data = {}

        if include_indices and self.sample_indices is not None:
            data['sample_index'] = self.sample_indices

        if self.is_multioutput:
            for i in range(self.shape[1]):
                data[f'y_pred_{i}'] = self.y_pred[:, i]
        else:
            data['y_pred'] = self.y_pred.flatten()

        return pd.DataFrame(data)

    def flatten(self) -> np.ndarray:
        """Get flattened 1D predictions.

        Returns:
            1D numpy array of predictions.
        """
        if self.y_pred is None:
            return np.array([])
        return self.y_pred.flatten()

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictResult(shape={self.shape}, model='{self.model_name}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"PredictResult: {len(self)} predictions"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        if self.preprocessing_steps:
            lines.append(f"  Preprocessing: {' -> '.join(self.preprocessing_steps)}")
        lines.append(f"  Shape: {self.shape}")
        return "\n".join(lines)

@dataclass
class ExplainResult:
    """Result from nirs4all.explain().

    Wraps SHAP explanation outputs with visualization helpers and accessors.

    Attributes:
        shap_values: SHAP values array or Explanation object.
        feature_names: Names/labels of features explained.
        base_value: Expected value (baseline prediction).
        visualizations: Paths to generated visualization files.
        explainer_type: Type of SHAP explainer used.
        model_name: Name of the explained model.
        n_samples: Number of samples explained.

    Properties:
        values: Raw SHAP values array.
        shape: Shape of SHAP values array.
        mean_abs_shap: Mean absolute SHAP values per feature.
        top_features: Feature names sorted by importance.

    Key Operations:
        get_feature_importance(): Get feature importance ranking.
        get_sample_explanation(idx): Get explanation for a single sample.
        to_dataframe(): Get SHAP values as DataFrame.

    Example:
        >>> result = nirs4all.explain(model, X_test)
        >>> print(f"Top features: {result.top_features[:5]}")
        >>> importance = result.get_feature_importance()
    """

    shap_values: Any  # shap.Explanation or np.ndarray
    feature_names: list[str] | None = None
    base_value: float | np.ndarray | None = None
    visualizations: dict[str, Path] = field(default_factory=dict)
    explainer_type: str = "auto"
    model_name: str = ""
    n_samples: int = 0

    def __post_init__(self):
        """Extract metadata from shap_values if available."""
        if hasattr(self.shap_values, 'values'):
            # It's a shap.Explanation object
            if self.feature_names is None and hasattr(self.shap_values, 'feature_names'):
                self.feature_names = list(self.shap_values.feature_names)
            if self.base_value is None and hasattr(self.shap_values, 'base_values'):
                self.base_value = self.shap_values.base_values
            if self.n_samples == 0:
                self.n_samples = len(self.shap_values.values)

    @property
    def values(self) -> np.ndarray:
        """Get raw SHAP values array.

        Returns:
            Numpy array of SHAP values (n_samples, n_features).
        """
        if hasattr(self.shap_values, 'values'):
            return np.asarray(self.shap_values.values)
        return np.asarray(self.shap_values)

    @property
    def shape(self) -> tuple:
        """Get shape of SHAP values array."""
        return self.values.shape

    @property
    def mean_abs_shap(self) -> np.ndarray:
        """Get mean absolute SHAP values per feature.

        Returns:
            1D array of mean |SHAP| values, one per feature.
        """
        vals = self.values
        if vals.ndim == 1:
            return np.asarray(np.abs(vals))
        return np.asarray(np.mean(np.abs(vals), axis=0))

    @property
    def top_features(self) -> list[str]:
        """Get feature names sorted by importance (descending).

        Returns:
            List of feature names, most important first.
            Returns indices as strings if feature_names not available.
        """
        importance = self.mean_abs_shap
        sorted_indices = np.argsort(importance)[::-1]

        if self.feature_names:
            return [self.feature_names[i] for i in sorted_indices]
        return [str(i) for i in sorted_indices]

    def get_feature_importance(
        self,
        top_n: int | None = None,
        normalize: bool = False
    ) -> dict[str, float]:
        """Get feature importance ranking.

        Args:
            top_n: If provided, return only top N features.
            normalize: If True, normalize values to sum to 1.

        Returns:
            Dictionary mapping feature names to importance values.
        """
        importance = self.mean_abs_shap

        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()

        sorted_indices = np.argsort(importance)[::-1]

        if top_n:
            sorted_indices = sorted_indices[:top_n]

        result = {}
        for idx in sorted_indices:
            name = self.feature_names[idx] if self.feature_names else str(idx)
            result[name] = float(importance[idx])

        return result

    def get_sample_explanation(
        self,
        idx: int
    ) -> dict[str, float]:
        """Get SHAP explanation for a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary mapping feature names to SHAP values for that sample.
        """
        vals = self.values
        if idx >= len(vals):
            raise IndexError(f"Sample index {idx} out of range (n_samples={len(vals)})")

        sample_shap = vals[idx] if vals.ndim > 1 else vals

        result = {}
        for i, val in enumerate(sample_shap):
            name = self.feature_names[i] if self.feature_names else str(i)
            result[name] = float(val)

        return result

    def to_dataframe(self, include_feature_names: bool = True):
        """Get SHAP values as pandas DataFrame.

        Args:
            include_feature_names: If True, use feature names as columns.

        Returns:
            pandas DataFrame with SHAP values.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe()") from err

        vals = self.values

        columns = self.feature_names if include_feature_names and self.feature_names else [f"feature_{i}" for i in range(vals.shape[-1])]

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        return pd.DataFrame(vals, columns=columns)

    def __repr__(self) -> str:
        """String representation."""
        return f"ExplainResult(shape={self.shape}, explainer='{self.explainer_type}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"ExplainResult: {self.n_samples} samples explained"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        lines.append(f"  Explainer: {self.explainer_type}")
        lines.append(f"  Shape: {self.shape}")
        if self.feature_names:
            lines.append(f"  Features: {len(self.feature_names)}")

        # Show top 5 features
        top = self.top_features[:5]
        if top:
            lines.append(f"  Top features: {', '.join(top)}")

        if self.visualizations:
            lines.append(f"  Visualizations: {list(self.visualizations.keys())}")

        return "\n".join(lines)
