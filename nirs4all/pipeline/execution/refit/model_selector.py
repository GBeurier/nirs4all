"""Per-model variant selection for the refit phase.

After Pass 1 (cross-validation), for each model node in the pipeline,
identifies the variant where that model achieved the best validation
score -- independently of other models.

In stacking pipelines with generators, different base models may have
their best variant differ from the meta-model's best variant.  This
module selects the optimal variant per model so that the refit phase
can replay each model's best configuration.

Example:
    >>> from nirs4all.pipeline.execution.refit.model_selector import (
    ...     select_best_per_model,
    ... )
    >>> selections = select_best_per_model(predictions, topology, variant_configs)
    >>> for model_name, sel in selections.items():
    ...     print(f"{model_name}: best variant {sel.variant_index}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nirs4all.core.logging import get_logger
from nirs4all.pipeline.analysis.topology import PipelineTopology

logger = get_logger(__name__)

@dataclass
class PerModelSelection:
    """Selection result for a single model node.

    Attributes:
        variant_index: Index of the variant (in ``PipelineConfigs.steps``)
            where this model achieved its best validation score.
        best_score: Best validation score achieved by this model.
        best_params: Best hyperparameters for this model (from finetuning
            or the variant's expanded config).
        expanded_steps: The fully expanded step list for the winning
            variant of this model.
        branch_path: Branch path leading to this model (empty for
            top-level models).
    """

    variant_index: int = 0
    best_score: float = 0.0
    best_params: dict[str, Any] = field(default_factory=dict)
    expanded_steps: list[Any] = field(default_factory=list)
    branch_path: list[int] = field(default_factory=list)

def select_best_per_model(
    predictions: list[dict[str, Any]],
    topology: PipelineTopology,
    variant_configs: list[dict[str, Any]],
    metric: str = "",
    ascending: bool | None = None,
) -> dict[str, PerModelSelection]:
    """Select the best variant for each model node independently.

    Enumerates all model nodes from the topology, filters prediction
    entries by model name/branch path, and for each model finds the
    variant with the best validation score.

    Args:
        predictions: List of prediction dicts from all variants.  Each
            dict must contain at least ``model_name``, ``val_score``,
            and ``branch_name`` (or ``branch_id``) fields.
        topology: Pipeline topology descriptor with ``model_nodes``.
        variant_configs: List of variant configuration dicts, one per
            variant.  Each dict should contain ``variant_index``,
            ``expanded_steps``, and optionally ``best_params``.
        metric: Metric name for ranking.  If empty, uses ``val_score``
            directly.
        ascending: Sort direction.  ``True`` means lower is better
            (e.g. RMSE).  ``None`` infers from the metric name.

    Returns:
        Mapping from model class name to its ``PerModelSelection``.
        For single-variant pipelines, all models share the same
        variant index.

    Edge cases:
        - **Single variant**: all models map to variant 0.
        - **Branches without merge**: models in different branches are
          alternatives, not cooperating.  Each gets independent
          selection.
        - **Branches with merge (stacking)**: all branch models are
          cooperating; the meta-model's winning variant determines
          the stacking context.  Branch base models are still
          independently selected within that context.
    """
    if ascending is None:
        ascending = _infer_ascending(metric)

    # Single variant shortcut
    if len(variant_configs) <= 1:
        return _single_variant_selection(topology, variant_configs)

    # Build a lookup: model_class -> list of (variant_index, val_score, config)
    model_scores: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}

    for pred in predictions:
        model_name = pred.get("model_name", "") or pred.get("model_classname", "")
        if not model_name:
            continue

        val_score = pred.get("val_score")
        if val_score is None:
            continue

        # Determine which variant this prediction belongs to
        variant_index = _resolve_variant_index(pred, variant_configs)

        config = variant_configs[variant_index] if variant_index < len(variant_configs) else {}

        if model_name not in model_scores:
            model_scores[model_name] = []
        model_scores[model_name].append((variant_index, float(val_score), config))

    # For each model, select the variant with the best score
    result: dict[str, PerModelSelection] = {}

    for model_node in topology.model_nodes:
        model_class = model_node.model_class
        # Try matching by full class path, then by short class name
        scores = model_scores.get(model_class)
        if scores is None:
            # Try matching by short name (last component of dotted path)
            short_name = model_class.rsplit(".", 1)[-1] if "." in model_class else model_class
            scores = model_scores.get(short_name)

        if not scores:
            # No predictions for this model -- use first variant as fallback
            result[model_class] = PerModelSelection(
                variant_index=0,
                expanded_steps=variant_configs[0].get("expanded_steps", []) if variant_configs else [],
                branch_path=list(model_node.branch_path),
            )
            continue

        # Aggregate scores per variant (average val_score across folds)
        variant_aggregated = _aggregate_scores_per_variant(scores)

        # Select best variant
        if ascending:
            best_variant_idx, best_score = min(variant_aggregated.items(), key=lambda x: x[1])
        else:
            best_variant_idx, best_score = max(variant_aggregated.items(), key=lambda x: x[1])

        best_config = variant_configs[best_variant_idx] if best_variant_idx < len(variant_configs) else {}

        result[model_class] = PerModelSelection(
            variant_index=best_variant_idx,
            best_score=best_score,
            best_params=best_config.get("best_params", {}),
            expanded_steps=best_config.get("expanded_steps", []),
            branch_path=list(model_node.branch_path),
        )

    # Handle stacking: if there's a meta-model, log the stacking context
    if topology.has_stacking and len(result) > 1:
        meta_models = [
            (name, sel) for name, sel in result.items()
            if any(
                mn.model_class == name and mn.merge_type == "predictions"
                for mn in topology.model_nodes
            )
        ]
        if meta_models:
            meta_name, meta_sel = meta_models[0]
            logger.info(
                f"Stacking context: meta-model '{meta_name}' best at "
                f"variant {meta_sel.variant_index}"
            )

    return result

def _single_variant_selection(
    topology: PipelineTopology,
    variant_configs: list[dict[str, Any]],
) -> dict[str, PerModelSelection]:
    """Handle the trivial case of a single variant.

    All models share variant 0.
    """
    config = variant_configs[0] if variant_configs else {}
    result: dict[str, PerModelSelection] = {}
    for model_node in topology.model_nodes:
        result[model_node.model_class] = PerModelSelection(
            variant_index=0,
            expanded_steps=config.get("expanded_steps", []),
            best_params=config.get("best_params", {}),
            branch_path=list(model_node.branch_path),
        )
    return result

def _resolve_variant_index(
    pred: dict[str, Any],
    variant_configs: list[dict[str, Any]],
) -> int:
    """Determine which variant index a prediction belongs to.

    Checks for explicit ``variant_index`` or ``pipeline_uid`` fields,
    falling back to positional matching via ``config_name``.

    Args:
        pred: Prediction dict.
        variant_configs: List of variant config dicts.

    Returns:
        Variant index (0-based).
    """
    # Explicit variant_index
    vi = pred.get("variant_index")
    if vi is not None:
        return int(vi)

    # Match by pipeline_uid
    pipeline_uid = pred.get("pipeline_uid")
    if pipeline_uid:
        for i, cfg in enumerate(variant_configs):
            if cfg.get("pipeline_uid") == pipeline_uid:
                return i

    # Match by config_name
    config_name = pred.get("config_name", "")
    if config_name:
        for i, cfg in enumerate(variant_configs):
            if cfg.get("name", "") == config_name:
                return i

    return 0

def _aggregate_scores_per_variant(
    scores: list[tuple[int, float, dict[str, Any]]],
) -> dict[int, float]:
    """Average val_score per variant across folds.

    Args:
        scores: List of (variant_index, val_score, config) tuples.

    Returns:
        Mapping from variant_index to mean val_score.
    """
    from collections import defaultdict

    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)

    for variant_idx, val_score, _config in scores:
        sums[variant_idx] += val_score
        counts[variant_idx] += 1

    return {vi: sums[vi] / counts[vi] for vi in sums}

def _infer_ascending(metric: str) -> bool:
    """Infer whether lower-is-better from the metric name.

    Args:
        metric: Metric name string.

    Returns:
        ``True`` if lower is better (error metrics), ``False`` otherwise.
    """
    if not metric:
        return True  # Default: lower is better (RMSE, MSE, MAE)

    metric_lower = metric.lower()
    higher_is_better = {"r2", "accuracy", "f1", "precision", "recall", "auc", "roc_auc", "balanced_accuracy"}
    return metric_lower not in higher_is_better
