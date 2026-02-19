"""Winning configuration extraction for the refit phase.

After cross-validation (Pass 1) completes, this module extracts the
winning pipeline configuration -- the expanded steps, finetuned params,
and generator choices -- so that the refit phase can replay the exact
configuration that produced the best CV score.

Also provides per-model configuration extraction for refitting ALL
unique model classes independently, and multi-criteria top-K selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from nirs4all.core.logging import get_logger

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.execution.refit.model_selector import PerModelSelection

logger = get_logger(__name__)

@dataclass
class RefitConfig:
    """Configuration for a refit execution.

    Captures all information needed to replay the winning pipeline
    variant on the full training set.

    Attributes:
        expanded_steps: The fully expanded step list for the winning
            variant (after generator expansion).
        best_params: Best hyperparameters from finetuning (Optuna),
            or empty dict if no finetuning was used.
        variant_index: Index of the winning variant in the original
            ``PipelineConfigs.steps`` list.
        generator_choices: Generator choices that produced the winning
            variant (e.g. ``[{"_or_": "SNV"}, {"_range_": 10}]``).
        pipeline_id: Store pipeline ID of the winning variant.
        metric: Metric used for ranking.
        selection_score: Selection score that determined this config was chosen for refit.
            Matches the primary criterion used (rmsecv or mean_val).
        selection_scores: Per-criterion scores for all applicable criteria.
            Example: ``{"rmsecv": 3.21, "mean_val": 3.25}``.
            Used in multi-criteria refit to display all scores.
        primary_selection_criterion: Name of the primary criterion used for selection.
            One of: ``"rmsecv"``, ``"mean_val"``. Used for sorting and primary display.
        config_name: Original CV pipeline config name (for linking
            refit entries back to their CV fold data).
        selected_by_criteria: List of criterion labels that selected this
            config in multi-criteria refit (e.g. ["rmsecv(top3)", "mean_val(top1)"]).
            Empty for single-criterion refit.
    """

    expanded_steps: list[Any]
    best_params: dict[str, Any] = field(default_factory=dict)
    variant_index: int = 0
    generator_choices: list[dict[str, Any]] = field(default_factory=list)
    pipeline_id: str = ""
    metric: str = ""
    selection_score: float = 0.0
    selection_scores: dict[str, float] = field(default_factory=dict)
    primary_selection_criterion: str = "rmsecv"
    config_name: str = ""
    selected_by_criteria: list[str] = field(default_factory=list)

@dataclass
class RefitCriterion:
    """A single refit selection criterion.

    Attributes:
        top_k: Number of top pipeline variants to select.
        ranking: Ranking method.  ``"rmsecv"`` uses the avg fold's
            validation score (RMSECV from concatenated OOF predictions).
            ``"mean_val"`` uses the arithmetic mean of individual fold
            validation scores.
        metric: Metric name for ranking.  Empty string uses the metric
            recorded in the pipeline records.
    """

    top_k: int = 1
    ranking: str = "rmsecv"
    metric: str = ""

def parse_refit_param(
    refit: bool | dict[str, Any] | list[dict[str, Any]] | None,
) -> list[RefitCriterion]:
    """Normalize the user-facing ``refit`` parameter to a list of criteria.

    Args:
        refit: Refit configuration from :func:`nirs4all.run`.
            - ``True``: Default criterion (top 1 by RMSECV).
            - ``False`` / ``None``: Empty list (refit disabled).
            - ``dict``: Single criterion.
            - ``list[dict]``: Multiple criteria.

    Returns:
        List of :class:`RefitCriterion` instances.

    Examples:
        >>> parse_refit_param(True)
        [RefitCriterion(top_k=1, ranking='rmsecv', metric='')]
        >>> parse_refit_param({"top_k": 3, "ranking": "mean_val"})
        [RefitCriterion(top_k=3, ranking='mean_val', metric='')]
        >>> parse_refit_param([{"top_k": 3}, {"top_k": 1, "ranking": "mean_val"}])
        [RefitCriterion(top_k=3, ...), RefitCriterion(top_k=1, ranking='mean_val', ...)]
    """
    if refit is True:
        return [RefitCriterion()]
    if not refit:
        return []
    if isinstance(refit, dict):
        return [RefitCriterion(**{k: v for k, v in refit.items() if k in ("top_k", "ranking", "metric")})]
    if isinstance(refit, list):
        return [
            RefitCriterion(**{k: v for k, v in c.items() if k in ("top_k", "ranking", "metric")})
            for c in refit
        ]
    return []

def extract_top_configs(
    store: Any,
    run_id: str,
    criteria: list[RefitCriterion],
    predictions: Predictions | None = None,
    dataset_name: str | None = None,
) -> list[RefitConfig]:
    """Extract top pipeline configurations for refit based on multiple criteria.

    For each criterion, ranks all completed pipelines by the specified
    method and selects the top K.  Results are unioned and deduplicated
    (preserving first-seen order).

    Args:
        store: :class:`WorkspaceStore` instance with completed run data.
        run_id: Run identifier.
        criteria: List of :class:`RefitCriterion` specifying ranking
            methods and top-K counts.
        predictions: In-memory :class:`Predictions` object containing
            all CV prediction entries.  Required for ``"mean_val"``
            ranking (to access individual fold scores).
        dataset_name: Filter pipelines to this dataset.

    Returns:
        Deduplicated list of :class:`RefitConfig` objects, one per
        selected pipeline variant.

    Raises:
        ValueError: If the run has no completed pipelines.
    """
    from nirs4all.pipeline.storage.workspace_store import _infer_metric_ascending

    if not criteria:
        return []

    # Get all completed pipelines
    pipelines_df = store.list_pipelines(run_id=run_id, dataset_name=dataset_name)
    if pipelines_df.is_empty():
        raise ValueError(f"Run {run_id} has no pipelines")

    completed = pipelines_df.filter(pipelines_df["status"] == "completed")
    if completed.is_empty():
        raise ValueError(f"Run {run_id} has no completed pipelines")

    # Build lookup: pipeline_id → row index
    pipeline_ids = completed["pipeline_id"].to_list()

    # Determine effective metric from first criterion or pipeline records
    first_metric = criteria[0].metric
    if not first_metric:
        metric_col = completed["metric"]
        non_null = [v for v in metric_col.to_list() if v is not None and v != ""]
        first_metric = non_null[0] if non_null else "rmse"

    # Collect selected pipeline_ids (preserving order, deduplicating)
    # Also track which criteria selected each pipeline
    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    pid_to_criteria: dict[str, list[str]] = {}  # pipeline_id -> list of criterion labels

    for _crit_idx, criterion in enumerate(criteria):
        effective_metric = criterion.metric or first_metric
        ascending = _infer_metric_ascending(effective_metric)
        crit_label = f"{criterion.ranking}(top{criterion.top_k})"

        if criterion.ranking == "rmsecv":
            # Rank by best_val (which is RMSECV from the avg fold after the fix)
            scored = list(zip(pipeline_ids, completed["best_val"].to_list(), strict=False))
            scored = [(pid, s) for pid, s in scored if s is not None]
            scored.sort(key=lambda x: x[1], reverse=not ascending)

        elif criterion.ranking == "mean_val":
            # Rank by mean of individual fold validation scores
            if predictions is None:
                logger.warning(
                    "Cannot rank by 'mean_val': predictions not available. "
                    "Falling back to 'rmsecv'."
                )
                scored = list(zip(pipeline_ids, completed["best_val"].to_list(), strict=False))
                scored = [(pid, s) for pid, s in scored if s is not None]
                scored.sort(key=lambda x: x[1], reverse=not ascending)
            else:
                # Compute mean of individual fold val_scores for each pipeline
                scored = _compute_mean_val_scores(
                    predictions, pipeline_ids, completed, effective_metric,
                )
                scored.sort(key=lambda x: x[1], reverse=not ascending)

        else:
            logger.warning(f"Unknown ranking method '{criterion.ranking}', using 'rmsecv'")
            scored = list(zip(pipeline_ids, completed["best_val"].to_list(), strict=False))
            scored = [(pid, s) for pid, s in scored if s is not None]
            scored.sort(key=lambda x: x[1], reverse=not ascending)

        # Select top_k unique pipeline IDs not already globally selected.
        # Each criterion independently fills its quota by skipping models
        # already taken by prior criteria, ensuring sum(top_k) unique models.
        top_ids: list[str] = []
        for pid, _ in scored:
            if pid not in seen_ids:
                top_ids.append(pid)
                if len(top_ids) >= criterion.top_k:
                    break

        # Log this criterion's selections
        pid_to_name = dict(zip(pipeline_ids, completed["name"].to_list(), strict=False))
        top_names = [pid_to_name.get(pid, pid) for pid in top_ids]
        logger.info(f"  Criterion '{crit_label}' selected: {', '.join(top_names)}")

        for pid in top_ids:
            pid_to_criteria.setdefault(pid, []).append(crit_label)
            selected_ids.append(pid)
            seen_ids.add(pid)

    if not selected_ids:
        raise ValueError(f"No pipelines selected for refit in run {run_id}")

    # Build RefitConfig for each selected pipeline
    all_pipeline_ids = pipelines_df["pipeline_id"].to_list()
    configs: list[RefitConfig] = []

    # Collect all criterion labels for this selection
    all_criteria_labels = list(pid_to_criteria.values())
    all_criteria_labels_flat = [item for sublist in all_criteria_labels for item in sublist]

    # Pre-compute mean_val scores if needed (for any mean_val criterion)
    mean_val_scores_map: dict[str, float] = {}
    has_mean_val_criterion = any("mean_val" in c for c in all_criteria_labels_flat)
    if has_mean_val_criterion and predictions is not None:
        mean_val_scores = _compute_mean_val_scores(
            predictions, pipeline_ids, completed, effective_metric,
        )
        mean_val_scores_map = dict(mean_val_scores)

    for pipeline_id in selected_ids:
        pipeline_record = store.get_pipeline(pipeline_id)
        if pipeline_record is None:
            logger.warning(f"Pipeline {pipeline_id} not found, skipping")
            continue

        expanded_steps = pipeline_record.get("expanded_config", [])
        generator_choices = pipeline_record.get("generator_choices", [])

        # Determine scores for all applicable criteria
        pid_idx = pipeline_ids.index(pipeline_id) if pipeline_id in pipeline_ids else None

        # Get the criteria that selected this pipeline
        selected_by = pid_to_criteria.get(pipeline_id, [])

        # Compute scores for all criteria
        rmsecv_score = completed.row(pid_idx, named=True).get("best_val", 0.0) if pid_idx is not None else 0.0
        mean_val_score = mean_val_scores_map.get(pipeline_id, rmsecv_score)  # Fallback to rmsecv if not computed

        # Build selection_scores dict with all applicable scores
        selection_scores = {
            "rmsecv": rmsecv_score,
            "mean_val": mean_val_score,
        }

        # Determine primary criterion based on which criteria selected this pipeline
        # Priority: If selected by mean_val at all, use mean_val; otherwise use rmsecv
        use_mean_val = any("mean_val" in c for c in selected_by)
        primary_criterion = "mean_val" if use_mean_val else "rmsecv"
        best_score = selection_scores[primary_criterion]

        cv_config_name = pipeline_record.get("name", "")
        variant_index = all_pipeline_ids.index(pipeline_id) if pipeline_id in all_pipeline_ids else 0
        best_params = _extract_best_params(store, pipeline_id, first_metric, _infer_metric_ascending(first_metric))

        configs.append(RefitConfig(
            expanded_steps=expanded_steps,
            best_params=best_params,
            variant_index=variant_index,
            generator_choices=generator_choices,
            pipeline_id=pipeline_id,
            metric=first_metric,
            selection_score=best_score or 0.0,
            selection_scores=selection_scores,
            primary_selection_criterion=primary_criterion,
            config_name=cv_config_name,
            selected_by_criteria=pid_to_criteria.get(pipeline_id, []),
        ))

    return configs

def _compute_mean_val_scores(
    predictions: Predictions,
    pipeline_ids: list[str],
    completed_df: Any,
    metric: str,
) -> list[tuple[str, float]]:
    """Compute mean of individual fold validation scores for each pipeline.

    Filters predictions to individual folds (excluding avg, w_avg, final),
    groups by config_name (matched to pipeline via the completed DataFrame),
    and computes the arithmetic mean of fold val_scores.

    Args:
        predictions: In-memory Predictions with all CV entries.
        pipeline_ids: List of pipeline IDs to score.
        completed_df: Polars DataFrame of completed pipelines.
        metric: Metric name for extracting scores.

    Returns:
        List of (pipeline_id, mean_val_score) tuples.
    """
    # Map config_name → pipeline_id from store records
    names = completed_df["name"].to_list() if "name" in completed_df.columns else []
    name_to_pid = dict(zip(names, pipeline_ids, strict=False))

    # Get all val-partition predictions from buffer
    val_preds = predictions.filter_predictions(partition="val", load_arrays=False)

    # Group fold val_scores by config_name (excluding virtual folds)
    config_scores: dict[str, list[float]] = {}
    for entry in val_preds:
        fold_id = str(entry.get("fold_id", ""))
        if fold_id in ("avg", "w_avg", "final"):
            continue

        config_name = entry.get("config_name", "")
        val_score = entry.get("val_score")
        if val_score is None:
            # Try to get from scores dict
            scores_dict = entry.get("scores", {})
            if isinstance(scores_dict, dict) and "val" in scores_dict:
                val_scores_part = scores_dict["val"]
                if isinstance(val_scores_part, dict):
                    val_score = val_scores_part.get(metric)
        if val_score is not None:
            if config_name not in config_scores:
                config_scores[config_name] = []
            config_scores[config_name].append(float(val_score))

    # Compute mean for each pipeline
    result: list[tuple[str, float]] = []
    for config_name, fold_scores in config_scores.items():
        pid = name_to_pid.get(config_name)
        if pid is not None and fold_scores:
            result.append((pid, float(np.mean(fold_scores))))

    # Include pipelines not found in predictions (with no score)
    scored_pids = {pid for pid, _ in result}
    for pid in pipeline_ids:
        if pid not in scored_pids:
            # Fallback: use best_val from store
            pid_idx = pipeline_ids.index(pid)
            best_val = completed_df.row(pid_idx, named=True).get("best_val")
            if best_val is not None:
                result.append((pid, float(best_val)))

    return result

def extract_winning_config(
    store: Any,
    run_id: str,
    metric: str | None = None,
    ascending: bool | None = None,
    dataset_name: str | None = None,
) -> RefitConfig:
    """Extract the winning pipeline configuration from a completed CV run.

    Queries the store for all pipelines in the run, finds the one with
    the best validation score, and reconstructs a :class:`RefitConfig`
    containing the expanded steps and best hyperparameters.

    For Phase 2 (non-stacking), there is only one model node per
    pipeline, so global selection equals per-model selection.

    Args:
        store: :class:`WorkspaceStore` instance with completed run data.
        run_id: Run identifier returned by ``store.begin_run()``.
        metric: Metric name for ranking.  If ``None``, uses the metric
            recorded in the pipeline records.
        ascending: Sort direction for the metric.  If ``None``, inferred
            from the metric name (lower-is-better for error metrics).
        dataset_name: Filter pipelines to this dataset.  Required for
            multi-dataset runs to avoid cross-dataset contamination.

    Returns:
        A :class:`RefitConfig` containing the winning variant's steps,
        best hyperparameters, and metadata.

    Raises:
        ValueError: If the run has no completed pipelines or no
            predictions to rank.
    """
    from nirs4all.pipeline.storage.workspace_store import _infer_metric_ascending

    # Get pipelines for this run, filtered by dataset if specified
    pipelines_df = store.list_pipelines(run_id=run_id, dataset_name=dataset_name)
    if pipelines_df.is_empty():
        raise ValueError(f"Run {run_id} has no pipelines")

    # Filter to completed pipelines
    completed = pipelines_df.filter(pipelines_df["status"] == "completed")
    if completed.is_empty():
        raise ValueError(f"Run {run_id} has no completed pipelines")

    # Determine metric and sort direction
    effective_metric = metric
    if effective_metric is None:
        # Use metric from the first completed pipeline
        metric_col = completed["metric"]
        non_null = [v for v in metric_col.to_list() if v is not None and v != ""]
        effective_metric = non_null[0] if non_null else "rmse"

    if ascending is None:
        ascending = _infer_metric_ascending(effective_metric)

    # Find the best pipeline by best_val score
    best_val_col = completed["best_val"]
    best_idx = best_val_col.arg_min() if ascending else best_val_col.arg_max()

    best_pipeline = completed.row(best_idx, named=True)
    best_pipeline_id = best_pipeline["pipeline_id"]

    # Get full pipeline record with deserialized JSON fields
    pipeline_record = store.get_pipeline(best_pipeline_id)
    if pipeline_record is None:
        raise ValueError(f"Pipeline {best_pipeline_id} not found in store")

    expanded_steps = pipeline_record.get("expanded_config", [])
    generator_choices = pipeline_record.get("generator_choices", [])
    best_score = best_pipeline.get("best_val") or 0.0
    cv_config_name = pipeline_record.get("name", "")

    # Determine variant_index: match pipeline_id against all pipelines in order
    all_pipeline_ids = pipelines_df["pipeline_id"].to_list()
    variant_index = all_pipeline_ids.index(best_pipeline_id) if best_pipeline_id in all_pipeline_ids else 0

    # Extract best_params from the best prediction for this pipeline
    best_params = _extract_best_params(store, best_pipeline_id, effective_metric, ascending)

    # For single-criterion selection, only rmsecv is available
    selection_scores = {"rmsecv": best_score}

    return RefitConfig(
        expanded_steps=expanded_steps,
        best_params=best_params,
        variant_index=variant_index,
        generator_choices=generator_choices,
        pipeline_id=best_pipeline_id,
        metric=effective_metric,
        selection_score=best_score,
        selection_scores=selection_scores,
        primary_selection_criterion="rmsecv",
        config_name=cv_config_name,
    )

def _extract_best_params(
    store: Any,
    pipeline_id: str,
    metric: str,
    ascending: bool,
) -> dict[str, Any]:
    """Extract the best hyperparameters from predictions for a pipeline.

    For ``individual`` Optuna mode, selects params from the
    best-performing fold.  For ``unified`` mode (default), all folds
    share the same params, so any prediction's ``best_params`` suffices.

    Args:
        store: WorkspaceStore instance.
        pipeline_id: Pipeline to query predictions for.
        metric: Metric name for ranking.
        ascending: Sort direction for the metric.

    Returns:
        Best hyperparameters dict, or empty dict if none found.
    """
    import json

    predictions_df = store.query_predictions(
        pipeline_id=pipeline_id,
        partition="val",
    )

    if predictions_df.is_empty():
        return {}

    # Sort predictions by val_score to find the best fold
    sort_col = "val_score"
    sorted_df = predictions_df.sort(sort_col, descending=not ascending, nulls_last=True)

    # Get the best prediction's best_params
    best_row = sorted_df.row(0, named=True)
    best_params_raw = best_row.get("best_params")

    if best_params_raw is None:
        return {}

    # Handle JSON string or dict
    if isinstance(best_params_raw, str):
        try:
            return json.loads(best_params_raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    if isinstance(best_params_raw, dict):
        return best_params_raw

    return {}

def extract_per_model_configs(
    store: Any,
    run_id: str,
    metric: str | None = None,
    ascending: bool | None = None,
    dataset_name: str | None = None,
) -> dict[str, tuple[PerModelSelection, RefitConfig]]:
    """Extract the best RefitConfig for each unique model class.

    Queries all completed pipeline variants from the store, identifies the
    model class used in each, and for each unique model picks the variant
    with the best validation score.

    Args:
        store: :class:`WorkspaceStore` instance.
        run_id: Run identifier.
        metric: Metric for ranking.  Inferred if ``None``.
        ascending: Sort direction.  Inferred if ``None``.
        dataset_name: Filter pipelines to this dataset.  Required for
            multi-dataset runs to avoid cross-dataset contamination.

    Returns:
        Mapping from model class name to ``(PerModelSelection, RefitConfig)``
        tuples.  Empty dict if only one model class exists (standard refit
        suffices).
    """
    from nirs4all.pipeline.analysis.topology import analyze_topology
    from nirs4all.pipeline.execution.refit.model_selector import PerModelSelection
    from nirs4all.pipeline.storage.workspace_store import _infer_metric_ascending

    pipelines_df = store.list_pipelines(run_id=run_id, dataset_name=dataset_name)
    if pipelines_df.is_empty():
        return {}

    completed = pipelines_df.filter(pipelines_df["status"] == "completed")
    if completed.is_empty() or len(completed) <= 1:
        return {}

    # Determine metric and ascending
    if metric is None:
        metric_col = completed["metric"]
        non_null = [v for v in metric_col.to_list() if v is not None and v != ""]
        metric = non_null[0] if non_null else "rmse"
    if ascending is None:
        ascending = _infer_metric_ascending(metric)

    pipeline_ids = completed["pipeline_id"].to_list()
    best_vals = completed["best_val"].to_list()

    # For each pipeline variant, identify its model class via topology
    model_to_variants: dict[str, list[tuple[str, float, int]]] = {}

    for idx, (pid, bv) in enumerate(zip(pipeline_ids, best_vals, strict=False)):
        record = store.get_pipeline(pid)
        if record is None:
            continue
        expanded_steps = record.get("expanded_config", [])
        topo = analyze_topology(expanded_steps)

        for model_node in topo.model_nodes:
            model_class = model_node.model_class
            if model_class not in model_to_variants:
                model_to_variants[model_class] = []
            model_to_variants[model_class].append((pid, bv if bv is not None else 0.0, idx))

    if len(model_to_variants) <= 1:
        return {}

    # For each model, pick the variant with best val_score
    result: dict[str, tuple[PerModelSelection, RefitConfig]] = {}

    for model_class, variants in model_to_variants.items():
        if ascending:
            best_pid, best_score, best_idx = min(variants, key=lambda x: x[1])
        else:
            best_pid, best_score, best_idx = max(variants, key=lambda x: x[1])

        record = store.get_pipeline(best_pid)
        if record is None:
            continue

        expanded_steps = record.get("expanded_config", [])
        generator_choices = record.get("generator_choices", [])
        best_params = _extract_best_params(store, best_pid, metric, ascending)
        cv_config_name = record.get("name", "")

        selection = PerModelSelection(
            variant_index=best_idx,
            best_score=best_score,
            best_params=best_params,
            expanded_steps=expanded_steps,
        )

        # For per-model selection, only rmsecv is available
        selection_scores = {"rmsecv": best_score}

        config = RefitConfig(
            expanded_steps=expanded_steps,
            best_params=best_params,
            variant_index=best_idx,
            generator_choices=generator_choices,
            pipeline_id=best_pid,
            metric=metric,
            selection_score=best_score,
            selection_scores=selection_scores,
            primary_selection_criterion="rmsecv",
            config_name=cv_config_name,
        )

        result[model_class] = (selection, config)

    return result
