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

import contextlib
import threading
from collections.abc import Mapping, Sequence
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


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    """Return a shallow dict copy when *value* is mapping-like."""
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _materialization_manifest(value: Any) -> dict[str, Any] | None:
    """Extract the materialization manifest from a relation replay payload."""
    manifest = _plain_mapping(value)
    if manifest is None:
        return None
    inner = manifest.get("materialization_manifest")
    if isinstance(inner, Mapping):
        return dict(inner)
    if manifest.get("representation") is not None and any(key in manifest for key in ("headers", "model_headers", "shape", "fingerprint")):
        return manifest
    return None


def _relation_manifest_from_metadata(metadata: Any) -> dict[str, Any] | None:
    meta = _metadata_mapping(metadata)
    for key in ("relation_replay_manifest", "relation_materialization_manifest"):
        manifest = _plain_mapping(meta.get(key))
        if manifest is not None:
            return manifest
    return None


def _derive_relation_lineage(
    manifest: Any,
    *,
    feature_names: Sequence[str] | None = None,
    n_features: int | None = None,
) -> Any | None:
    payload = _plain_mapping(manifest)
    if payload is None:
        return None
    from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage

    return derive_relation_explain_lineage(
        payload,
        feature_names=feature_names,
        n_features=n_features,
    )


def _lineage_by_feature(feature_lineage: Mapping[str, Any], feature: str | int) -> dict[str, Any]:
    if isinstance(feature, int):
        for lineage in feature_lineage.values():
            if isinstance(lineage, Mapping) and lineage.get("feature_index") == feature:
                return dict(lineage)
        names = list(feature_lineage)
        if 0 <= feature < len(names):
            lineage = feature_lineage.get(names[feature])
            return dict(lineage) if isinstance(lineage, Mapping) else {}
        return {}

    lineage = feature_lineage.get(feature)
    return dict(lineage) if isinstance(lineage, Mapping) else {}


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
    _owns_runner: bool = field(default=True, repr=False)
    _workspace_path: Path | None = field(default=None, repr=False)

    # Lazy refit dependencies (set by the orchestrator when per-model
    # selections are available so that ``models`` returns lazy results)
    _per_model_selections: dict[str, PerModelSelection] | None = field(default=None, repr=False)
    _refit_config: RefitConfig | None = field(default=None, repr=False)
    _refit_dataset: Any = field(default=None, repr=False)
    _refit_context: Any = field(default=None, repr=False)
    _refit_runtime_context: Any = field(default=None, repr=False)
    _refit_artifact_registry: Any = field(default=None, repr=False)
    _refit_executor: Any = field(default=None, repr=False)

    # dag-ml export bridge (P1c, TRANSITIONAL): the dag-ml backend returns scores in memory with NO
    # workspace / artifacts, so .n4a export has nothing to bundle. This spec FREEZES the run inputs (a
    # deepcopy of the pipeline + the dataset deepcopied for in-memory forms / kept as a stable path/config
    # ref otherwise, plus name/random_state) at dag-ml run time, so an export request re-runs the SAME
    # frozen pipeline through the LEGACY engine (save_artifacts=True) ON DEMAND — producing the workspace +
    # chain + artifacts the existing export path needs. ``_dagml_legacy_result`` caches that materialized
    # legacy RunResult (kept alive so its workspace store stays open for the export, closed on close()).
    #
    # PARITY SCOPE (honest, transitional): for a FULLY-SEEDED deterministic run the legacy refit reproduces
    # the dag-ml-scored model EXACTLY (engine numerical parity); otherwise the export is BEST-EFFORT. A
    # per-run WARNING fires on the two ``_dagml_export_stochastic`` signals — (a) CERTAIN: a
    # sample_augmentation step (re-augmentation is non-reproducible across processes and the augmenter's
    # own RNG is not covered by run(random_state)); (b) CONSERVATIVE: run(random_state) is None (nothing
    # globally seeded — this may over-warn a fully-deterministic pipeline, the safe direction for a "may
    # differ" caveat). A per-estimator "unseeded-stochastic?" probe is NOT attempted: random_state use is
    # solver/config-conditional (Ridge() / PCA(svd_solver="full") carry a DORMANT random_state=None yet are
    # deterministic → false alarm; MLPRegressor(shuffle=False) is stochastic via weight init; wrapped
    # estimators hide theirs), so any static heuristic both over- and under-warns. The uncertain middle (a
    # seeded run whose individual component left random_state=None) is documented in the export()/
    # export_model() docstrings (general caveat), not warned. P3 (native fitted-model capture) removes the
    # limitation by exporting the actual scored artifacts. Reloadable on-disk-PATH datasets must be
    # unchanged at export time (only non-reloadable / in-memory dataset forms are snapshotted; a path/file
    # config is replayed from disk).
    _dagml_export_spec: dict[str, Any] | None = field(default=None, repr=False)
    _dagml_legacy_result: RunResult | None = field(default=None, repr=False)
    _dagml_export_stochastic: bool = field(default=False, repr=False)

    # --- Lifecycle ---

    def detach(self) -> None:
        """Detach from the runner by closing its store and dropping the reference.

        After detaching, the RunResult operates in detached mode: export
        operations re-open the store on demand.  This releases the DB
        connection so other processes can access the workspace.

        Called automatically by :func:`nirs4all.run` for non-session runs.
        """
        if self._runner is not None and self._owns_runner:
            if self._workspace_path is None:
                self._workspace_path = getattr(self._runner, 'workspace_path', None)
            self._runner.close()
            self._runner = None

    def close(self) -> None:
        """Close the underlying WorkspaceStore to release DB resources.

        Safe to call multiple times.  For detached results this is a no-op.
        Session-owned runners are closed by the session. A dag-ml export bridge's
        materialized legacy result (if any) is closed too, releasing its workspace store.
        """
        if self._runner is not None and self._owns_runner:
            self._runner.close()
        if self._dagml_legacy_result is not None:
            self._dagml_legacy_result.close()

    def __enter__(self) -> RunResult:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Safety net: close store if caller forgot."""
        with contextlib.suppress(Exception):
            self.close()

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

    def _selected_metric(self, metric: str, *test_score_aliases: str) -> float:
        """Read ``metric`` from the SELECTED model — the one ``best``/``best_score`` describe.

        Shared anchor for ``best_rmse`` / ``best_r2`` / ``best_accuracy`` so every scalar
        shortcut reports the SAME model (the selection-metric winner: the refit/``best_final``
        entry, or the selected CV entry for a no-refit run), never a per-metric-reranked row.

        Lookup tiers on that one entry: a flat ``metric`` key (from ``display_metrics``), then the
        per-partition ``scores['test'][metric]`` block, then ``test_score`` when the entry's own
        selection ``metric`` is one of ``test_score_aliases`` (so e.g. ``best_rmse`` still returns
        the test score of an rmse-selected entry that carries no expanded ``scores`` dict).

        Re-ranking each shortcut independently via ``get_best(metric=X)`` (the prior behaviour) is
        a BUG: it ranks rows by their VALIDATION ``X`` and, under cross-validation, the X-best row
        is a different *fold* model than the selection winner — so ``best_r2`` returned a CV fold's
        test R² (e.g. a ShuffleSplit fold's 0.5426 instead of the selected model's 0.5499) and
        ``best_accuracy`` returned a different fold's plain accuracy than the balanced-accuracy-
        selected model's. Anchoring all of them on ``best`` makes the trio self-consistent.
        """
        best = self.best
        if not best:
            return float('nan')

        # Flat key first (from display_metrics)
        if metric in best and best[metric] is not None:
            return float(best[metric])

        # Nested per-partition scores dict
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if metric in test_scores and test_scores[metric] is not None:
                return float(test_scores[metric])

        # Fall back to test_score when the selection metric IS this metric (or an alias)
        if test_score_aliases and best.get('metric', '') in test_score_aliases:
            test_score = best.get('test_score')
            if test_score is not None:
                return float(test_score)

        return float('nan')

    @property
    def best_score(self) -> float:
        """Get the selected model's primary test score (the selection-metric value).

        Returns:
            The test_score value from best prediction, or NaN if unavailable.
        """
        score = self.best.get('test_score')
        return float(score) if score is not None else float('nan')

    @property
    def best_rmse(self) -> float:
        """Get the SELECTED model's RMSE (the same model ``best_score``/``best_r2`` describe).

        Reads RMSE from :attr:`best` — the selection-metric winner — so the scalar shortcuts are
        mutually consistent (for an rmse-selected single model this equals ``best_score``). See
        :meth:`_selected_metric` for why per-shortcut ``get_best(metric=...)`` re-ranking was wrong.

        Returns:
            RMSE value or NaN if unavailable.
        """
        return self._selected_metric('rmse', 'rmse', 'mse')

    @property
    def best_r2(self) -> float:
        """Get the SELECTED model's R² (the same model ``best_score``/``best_rmse`` describe).

        Reads R² from :attr:`best` — the selection-metric winner — instead of re-ranking by R²,
        which under CV surfaced a different fold model's test R². See :meth:`_selected_metric`.

        Returns:
            R² value or NaN if unavailable.
        """
        return self._selected_metric('r2', 'r2')

    @property
    def best_accuracy(self) -> float:
        """Get the SELECTED model's accuracy (the same model ``best_score`` describes).

        Reads plain ``accuracy`` from :attr:`best` — the selection-metric winner (selection uses
        ``balanced_accuracy``) — instead of re-ranking by accuracy, which surfaced a different
        fold model's accuracy than the selected model's. See :meth:`_selected_metric`.

        Returns:
            Accuracy value or NaN if unavailable.
        """
        return self._selected_metric('accuracy', 'accuracy')

    # --- Refit accessors ---

    @property
    def best_final(self) -> dict[str, Any]:
        """Get the best refit entry across all models.

        Filters predictions to ``fold_id="final"`` entries and ranks them
        by their selection score (``selection_score``).

        Returns:
            Best refit prediction dict, or empty dict if no refit entries.
        """
        results = self.predictions.top(n=1, score_scope="refit")
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
        # Fallback: check global predictions (when refit entries were merged there)
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
        results = self.predictions.top(n=1, score_scope="folds", fold_id="avg")
        top = cast(list[dict[str, Any]], results)
        if top:
            return top[0]

        results = self.predictions.top(n=1, score_scope="folds")
        top = cast(list[dict[str, Any]], results)
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
        if self._workspace_path is not None:
            return self._workspace_path
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

    @property
    def relation_replay_manifests(self) -> dict[str, dict[str, Any]]:
        """Return relation replay manifests keyed by chain or prediction id."""
        manifests: dict[str, dict[str, Any]] = {}
        chain_ids: list[str] = []

        for row in self.predictions.filter_predictions(load_arrays=False):
            row_key = str(row.get("chain_id") or row.get("prediction_id") or row.get("id") or len(manifests))
            row_manifest = _relation_manifest_from_metadata(row.get("metadata"))
            if row_manifest is not None:
                manifests.setdefault(row_key, row_manifest)

            chain_id = row.get("chain_id")
            if chain_id is not None and str(chain_id):
                chain_ids.append(str(chain_id))

        missing_chain_ids = [chain_id for chain_id in dict.fromkeys(chain_ids) if chain_id not in manifests]
        if not missing_chain_ids:
            return manifests

        try:
            with self._open_store_for_export() as store:
                if store is None:
                    return manifests
                for chain_id in missing_chain_ids:
                    chain = store.get_chain(chain_id)
                    if not isinstance(chain, Mapping):
                        continue
                    manifest = _plain_mapping(chain.get("relation_replay_manifest"))
                    if manifest is not None:
                        manifests[chain_id] = manifest
        except Exception as exc:
            logger.debug("Could not load relation replay manifests for RunResult: %s", exc)

        return manifests

    @property
    def relation_replay_manifest(self) -> dict[str, Any] | None:
        """Return the relation replay manifest for the best/only chain, if any."""
        manifests = self.relation_replay_manifests
        if not manifests:
            return None

        best = self.best
        best_chain_id = best.get("chain_id") if isinstance(best, Mapping) else None
        if best_chain_id is not None and str(best_chain_id) in manifests:
            return dict(manifests[str(best_chain_id)])

        return dict(next(iter(manifests.values())))

    @property
    def relation_materialization_manifest(self) -> dict[str, Any] | None:
        """Return materialization provenance for the best relation replay manifest."""
        return _materialization_manifest(self.relation_replay_manifest)

    @property
    def feature_lineage(self) -> dict[str, Any]:
        """Feature provenance derived from the relation manifest, when available."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None or not getattr(lineage, "feature_lineage", None):
            return {}
        return {str(name): dict(payload) for name, payload in lineage.feature_lineage.items()}

    @property
    def lineage_warning(self) -> str | None:
        """Warning describing derived relation features, when applicable."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None:
            return None
        warning = lineage.lineage_warning
        return str(warning) if warning is not None else None

    @property
    def explanation_level(self) -> str | None:
        """Feature explanation level inferred from relation provenance."""
        lineage = _derive_relation_lineage(
            self.relation_replay_manifest,
            n_features=self._best_n_features(),
        )
        if lineage is None:
            return None
        level = lineage.explanation_level
        return str(level) if level is not None else None

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for a feature name or zero-based feature index."""
        return _lineage_by_feature(self.feature_lineage, feature)

    def _best_n_features(self) -> int | None:
        best = self.best
        if not isinstance(best, Mapping):
            return None
        n_features = best.get("n_features")
        try:
            return int(n_features) if n_features is not None else None
        except (TypeError, ValueError):
            return None

    # --- Export methods ---

    def _is_dagml_engine(self) -> bool:
        """True when this result came from the dag-ml backend (``run(engine="dag-ml")``).

        The dag-ml backend builds an in-memory :class:`Predictions` with native scores and NO workspace
        (no SQLite store, no artifacts dir), tagging ``per_dataset[name]["engine"] == "dag-ml"``. Export
        (.n4a bundle) needs the workspace artifacts, so it is not yet available on a dag-ml result — the
        export entry points use this to fail loud CATCHABLY (a ``NotImplementedError`` the planned
        try-dag-ml/except-NotImplementedError→legacy cutover catches) instead of a bare ``RuntimeError``.
        """
        return any(isinstance(info, dict) and info.get("engine") == "dag-ml" for info in self.per_dataset.values())

    def _no_workspace_export_error(self) -> Exception:
        """The right fail-loud error when an export has no workspace path.

        A dag-ml result WITHOUT an export spec → a CATCHABLE :class:`NotImplementedError` (so the cutover
        fallback redirects export to the legacy engine); a genuinely detached/misused legacy result → the
        original :class:`RuntimeError` (a real misuse, not a fall-back-able dag-ml gap). A dag-ml result
        WITH a spec never reaches here — :meth:`_dagml_export_delegate` materializes a legacy workspace first.
        """
        if self._is_dagml_engine():
            return NotImplementedError(
                "engine='dag-ml' does not support .n4a export for this run: the dag-ml backend returns native "
                "scores with no workspace artifacts to bundle, and no export spec was captured to re-run the "
                "pipeline on the legacy engine; run the export-bound pipeline on the legacy engine."
            )
        return RuntimeError("Cannot export: no workspace path available (result was not created from a workspace run)")

    def _dagml_export_delegate(self) -> RunResult | None:
        """Materialize (once) a LEGACY RunResult to back .n4a export of a dag-ml run; ``None`` if N/A.

        The dag-ml backend produces no workspace/artifacts, so export of a dag-ml result re-runs the SAME
        FROZEN pipeline (the deepcopy captured at run time) through the legacy engine with
        ``save_artifacts=True`` — that legacy run owns a real workspace + chain + artifacts the existing
        export path consumes. Returns ``None`` for a non-dag-ml result or a dag-ml result with no captured
        spec (the caller then falls back to the original no-workspace error). The legacy result is cached
        and kept alive (its store stays open for the export and is closed by :meth:`close`), so repeated
        exports do not refit.

        PARITY (transitional): this re-fit is EXACT for a fully-seeded deterministic run (the dag-ml and
        legacy engines are at numerical parity, so the exported model's ``predict()`` matches the dag-ml run
        within tolerance), but only BEST-EFFORT otherwise — see the export()/export_model() docstrings for
        the general caveat. A per-run WARNING fires on the two ``_dagml_export_stochastic`` signals — a
        ``sample_augmentation`` step (CERTAIN), or ``run(random_state) is None`` (CONSERVATIVE — may
        over-warn a fully-deterministic pipeline, the safe direction); the uncertain middle (a seeded run
        whose individual component left ``random_state=None``) is documented, not warned. P3 captures the
        real fitted artifacts natively.
        """
        if self._dagml_export_spec is None:
            return None
        if self._dagml_legacy_result is None:
            from nirs4all.api.run import run as _run

            spec = self._dagml_export_spec
            if self._dagml_export_stochastic:
                import warnings

                warnings.warn(
                    "engine='dag-ml' export re-fits this pipeline on the legacy engine, and this run may be "
                    "nondeterministic (sample_augmentation, or run(random_state) is None), so the dag-ml-scored "
                    "model and the on-export legacy refit may differ. For an EXACT export set "
                    "run(random_state=...) AND seed every stochastic component's random_state; P3 will "
                    "capture the fitted model natively.",
                    stacklevel=3,
                )
            # Re-run the export-bound pipeline on the legacy engine: save_artifacts=True persists the
            # workspace + chain the export path needs; the same name/random_state keep the refit aligned
            # with the dag-ml run. verbose=0 keeps the on-demand refit quiet (export is not a training call).
            self._dagml_legacy_result = _run(
                spec["pipeline"],
                spec["dataset"],
                name=spec.get("name", ""),
                random_state=spec.get("random_state"),
                save_artifacts=True,
                save_charts=False,
                verbose=0,
                engine="legacy",
            )
        return self._dagml_legacy_result

    def _open_store_for_export(self):
        """Open a temporary WorkspaceStore for export operations.

        Returns a context-managed store that is closed after use.
        Works in both attached (runner alive) and detached modes.
        """
        # Attached mode: use runner's store directly
        if self._runner is not None:
            return contextlib.nullcontext(self._runner.store)

        # Detached mode: open a fresh store
        if self._workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        class _TempStore:
            def __init__(self, path: Path):
                self._store = WorkspaceStore(path)

            def __enter__(self):
                return self._store

            def __exit__(self, *exc: object):
                self._store.close()

        return _TempStore(self._workspace_path)

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
        directly from the workspace store:

        >>> result.export("model.n4a", chain_id="abc123")

        **Resolver-based** -- exports via ``BundleGenerator``:

        >>> result.export("model.n4a")  # uses best prediction

        Works in both attached (runner alive) and detached modes.
        In detached mode, a temporary store is opened for the export
        and closed immediately after.

        **dag-ml runs (P1c, transitional):** the dag-ml backend keeps no workspace/artifacts, so this
        re-fits the pipeline via the legacy engine to produce the bundle. For an EXACT export, set
        ``run(random_state=...)`` AND seed every stochastic component's ``random_state``; otherwise
        (``sample_augmentation``, an unseeded run, or any unseeded-stochastic component such as
        ``RandomForest`` / ``MLP`` / ``CARS``) the exported model may differ from the dag-ml-scored model.
        ``source`` / ``chain_id`` are not supported for a dag-ml run (they reference its non-existent
        workspace); the run's best model is exported. P3 will capture the fitted model natively.

        Args:
            output_path: Path for the exported bundle file.
            format: Export format ('n4a' or 'n4a.py').
            source: Prediction dict to export. If None, exports best model.
            chain_id: Chain identifier for store-based export.
                When provided, ``source`` is ignored and the chain is
                exported directly from the workspace store.

        Returns:
            Path to the exported bundle file.

        Raises:
            RuntimeError: If no workspace path available.
            ValueError: If no predictions available and source not provided.
            NotImplementedError: For a dag-ml run, if ``source``/``chain_id`` is given.
        """
        # dag-ml export bridge (P1c): the dag-ml backend has no workspace, so delegate to a legacy re-run of
        # the same pipeline (materialized on demand, cached). FAIL-FAST FIRST: an EXPLICIT non-default
        # ``source`` / ``chain_id`` references the (non-existent) dag-ml workspace, so it cannot be honored
        # — raise the CATCHABLE NotImplementedError BEFORE materializing the delegate (no wasted refit, no
        # spurious stochastic warning). The predicate is the cheap spec flag, NOT the delegate (which would
        # trigger the refit). Otherwise materialize the delegate and export its OWN best/final (the
        # single-winner the dag-ml run scored).
        if self._dagml_export_spec is not None:
            if source is not None or chain_id is not None:
                raise NotImplementedError(
                    "engine='dag-ml' export does not support an explicit source=/chain_id= (they reference "
                    "the dag-ml run's non-existent workspace); export the run's best model with "
                    "result.export(path) (no source/chain_id)."
                )
            delegate = self._dagml_export_delegate()
            assert delegate is not None  # spec present ⇒ delegate materializes
            return delegate.export(output_path, format=format)

        # Store-based export path
        if chain_id is not None:
            with self._open_store_for_export() as store:
                if store is None:
                    raise RuntimeError("Cannot export from chain_id: no WorkspaceStore available")
                return Path(store.export_chain(chain_id, Path(output_path), format=format))

        # Resolver-based path
        if source is None:
            source = self.final or self.best
            if not source:
                raise ValueError("No predictions available to export")

        # Attached mode: delegate to runner
        if self._runner is not None:
            return Path(self._runner.export(source=source, output_path=output_path, format=format))

        # Detached mode: create BundleGenerator with temporary store
        workspace_path = self._workspace_path
        if workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.bundle import BundleGenerator

        with self._open_store_for_export() as store:
            generator = BundleGenerator(workspace_path=workspace_path, verbose=0, store=store)
            return generator.export(source=source, output_path=output_path, format=format)

    def export_model(
        self,
        output_path: str | Path,
        source: dict[str, Any] | None = None,
        format: str | None = None,
        fold: int | None = None
    ) -> Path:
        """Export only the model artifact (lightweight).

        Unlike export() which creates a full bundle, this exports just the model.
        Works in both attached (runner alive) and detached modes.

        **dag-ml runs (P1c, transitional):** same as :meth:`export` — the model is produced by a legacy
        re-fit. For an EXACT export, set ``run(random_state=...)`` AND seed every stochastic component's
        ``random_state``; otherwise (``sample_augmentation``, an unseeded run, or any unseeded-stochastic
        component such as ``RandomForest`` / ``MLP`` / ``CARS``) the exported model may differ from the
        dag-ml-scored model. ``source`` is not supported for a dag-ml run (it references its non-existent
        workspace); ``fold`` is honored (it selects a fold's model from the legacy re-fit's workspace).

        Args:
            output_path: Path for the output model file.
            source: Prediction dict to export. If None, exports best model.
            format: Model format (inferred from extension if None).
            fold: Fold index to export (default: fold 0).

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If no workspace path available.
        """
        # dag-ml export bridge (P1c): same as export() — FAIL-FAST on an explicit ``source`` BEFORE
        # materializing the delegate (no wasted refit / no spurious stochastic warning), using the cheap
        # spec flag (not the delegate, which would trigger the refit). ``fold`` is FORWARDED (it selects a
        # fold's model from the delegate's REAL legacy workspace, which does have per-fold artifacts).
        if self._dagml_export_spec is not None:
            if source is not None:
                raise NotImplementedError(
                    "engine='dag-ml' export_model does not support an explicit source= (it references the "
                    "dag-ml run's non-existent workspace); export the run's best model with "
                    "result.export_model(path[, fold=...]) (no source)."
                )
            delegate = self._dagml_export_delegate()
            assert delegate is not None  # spec present ⇒ delegate materializes
            return delegate.export_model(output_path, format=format, fold=fold)

        if source is None:
            source = self.best
            if not source:
                raise ValueError("No predictions available to export")

        # Attached mode: delegate to runner
        if self._runner is not None:
            return self._runner.export_model(source=source, output_path=output_path, format=format, fold=fold)

        # Detached mode: replicate runner.export_model logic with temporary store
        workspace_path = self._workspace_path
        if workspace_path is None:
            raise self._no_workspace_export_error()

        from nirs4all.pipeline.resolver import PredictionResolver
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._open_store_for_export() as store:
            resolver = PredictionResolver(workspace_path=workspace_path, runs_dir=workspace_path, store=store)
            resolved = resolver.resolve(source, verbose=0)

            if resolved.model_step_index is None:
                raise ValueError("No model step found in the resolved prediction")
            if resolved.artifact_provider is None:
                raise ValueError("No artifact provider available for this source")

            artifacts = resolved.artifact_provider.get_artifacts_for_step(resolved.model_step_index)
            if not artifacts:
                raise ValueError(f"No model artifacts found at step {resolved.model_step_index}")

            if fold is not None:
                model = None
                for artifact_id, artifact in artifacts:
                    if f":{fold}" in str(artifact_id) or artifact_id.endswith(f"_{fold}"):
                        model = artifact
                        break
                if model is None:
                    raise ValueError(f"No artifact found for fold {fold}")
            else:
                _, model = artifacts[0]

        if format is None:
            ext = output_path.suffix.lower()
            format_map = {'.joblib': 'joblib', '.pkl': 'cloudpickle', '.pickle': 'cloudpickle', '.h5': 'keras_h5', '.hdf5': 'keras_h5', '.keras': 'tensorflow_keras', '.pt': 'pytorch_state_dict', '.pth': 'pytorch_state_dict'}
            format = format_map.get(ext, 'joblib')

        data, _actual_format = to_bytes(model, format)
        with open(output_path, 'wb') as f:
            f.write(data)

        return output_path

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
        if self.metadata is None:
            self.metadata = {}

    @property
    def values(self) -> np.ndarray:
        """Get prediction values (alias for y_pred)."""
        return self.y_pred

    @property
    def relation_replay_manifest(self) -> dict[str, Any] | None:
        """Relation replay manifest used to materialize heterogeneous inputs."""
        return _plain_mapping(_metadata_mapping(self.metadata).get("relation_replay_manifest"))

    @property
    def relation_materialization_manifest(self) -> dict[str, Any] | None:
        """Materialization provenance embedded in the relation replay manifest."""
        metadata = _metadata_mapping(self.metadata)
        explicit = _plain_mapping(metadata.get("relation_materialization_manifest"))
        if explicit is not None:
            return explicit
        return _materialization_manifest(self.relation_replay_manifest)

    @property
    def feature_lineage(self) -> dict[str, Any]:
        """Feature provenance derived from prediction relation metadata."""
        metadata = _metadata_mapping(self.metadata)
        explicit = _plain_mapping(metadata.get("feature_lineage"))
        if explicit is not None:
            return explicit
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None or not getattr(lineage, "feature_lineage", None):
            return {}
        return {str(name): dict(payload) for name, payload in lineage.feature_lineage.items()}

    @property
    def lineage_warning(self) -> str | None:
        """Warning describing derived relation features, when applicable."""
        metadata = _metadata_mapping(self.metadata)
        warning = metadata.get("lineage_warning")
        if warning is not None:
            return str(warning)
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None:
            return None
        warning = lineage.lineage_warning
        return str(warning) if warning is not None else None

    @property
    def explanation_level(self) -> str | None:
        """Feature explanation level inferred from relation provenance."""
        metadata = _metadata_mapping(self.metadata)
        level = metadata.get("explanation_level")
        if level is not None:
            return str(level)
        lineage = _derive_relation_lineage(self.relation_replay_manifest or self.relation_materialization_manifest)
        if lineage is None:
            return None
        level = lineage.explanation_level
        return str(level) if level is not None else None

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for a feature name or zero-based feature index."""
        return _lineage_by_feature(self.feature_lineage, feature)

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
        if self.relation_replay_manifest is not None or self.relation_materialization_manifest is not None:
            lines.append("  Relation provenance: available")
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
        explanation_level: Unit level explained, such as raw_observation,
            source_aggregate, sample_aggregate, combo, or stack.
        feature_lineage: Mapping from explained feature names to relation
            lineage/provenance payloads.
        lineage_warning: Optional warning when explanations are for derived or
            aggregated features rather than raw observations.

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
    explanation_level: str | None = None
    feature_lineage: dict[str, Any] = field(default_factory=dict)
    lineage_warning: str | None = None

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

    def get_feature_lineage(self, feature: str | int) -> dict[str, Any]:
        """Get relation lineage for an explained feature.

        Args:
            feature: Feature name or positional feature index.

        Returns:
            Lineage payload for the feature, or an empty dictionary when absent.
        """
        if isinstance(feature, int):
            if self.feature_names and 0 <= feature < len(self.feature_names):
                feature_name = self.feature_names[feature]
            else:
                feature_name = str(feature)
        else:
            feature_name = feature
        lineage = self.feature_lineage.get(feature_name, {})
        return dict(lineage) if isinstance(lineage, dict) else {}

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
        if self.explanation_level:
            lines.append(f"  Explanation level: {self.explanation_level}")
        if self.feature_lineage:
            lines.append("  Feature lineage: available")
        if self.lineage_warning:
            lines.append(f"  Lineage warning: {self.lineage_warning}")

        # Show top 5 features
        top = self.top_features[:5]
        if top:
            lines.append(f"  Top features: {', '.join(top)}")

        if self.visualizations:
            lines.append(f"  Visualizations: {list(self.visualizations.keys())}")

        return "\n".join(lines)
