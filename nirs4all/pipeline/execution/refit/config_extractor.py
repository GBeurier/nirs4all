"""Winning configuration extraction for the refit phase.

After cross-validation (Pass 1) completes, this module extracts the
winning pipeline configuration -- the expanded steps, finetuned params,
and generator choices -- so that the refit phase can replay the exact
configuration that produced the best CV score.

Also provides per-model configuration extraction for refitting ALL
unique model classes independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nirs4all.core.logging import get_logger

if TYPE_CHECKING:
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
        best_score: Best validation score achieved.
    """

    expanded_steps: list[Any]
    best_params: dict[str, Any] = field(default_factory=dict)
    variant_index: int = 0
    generator_choices: list[dict[str, Any]] = field(default_factory=list)
    pipeline_id: str = ""
    metric: str = ""
    best_score: float = 0.0


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

    # Determine variant_index: match pipeline_id against all pipelines in order
    all_pipeline_ids = pipelines_df["pipeline_id"].to_list()
    variant_index = all_pipeline_ids.index(best_pipeline_id) if best_pipeline_id in all_pipeline_ids else 0

    # Extract best_params from the best prediction for this pipeline
    best_params = _extract_best_params(store, best_pipeline_id, effective_metric, ascending)

    return RefitConfig(
        expanded_steps=expanded_steps,
        best_params=best_params,
        variant_index=variant_index,
        generator_choices=generator_choices,
        pipeline_id=best_pipeline_id,
        metric=effective_metric,
        best_score=best_score,
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

    for idx, (pid, bv) in enumerate(zip(pipeline_ids, best_vals)):
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

        selection = PerModelSelection(
            variant_index=best_idx,
            best_score=best_score,
            best_params=best_params,
            expanded_steps=expanded_steps,
        )

        config = RefitConfig(
            expanded_steps=expanded_steps,
            best_params=best_params,
            variant_index=best_idx,
            generator_choices=generator_choices,
            pipeline_id=best_pid,
            metric=metric,
            best_score=best_score,
        )

        result[model_class] = (selection, config)

    return result
