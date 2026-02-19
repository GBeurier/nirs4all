"""Pipeline topology analyzer.

Inspects an expanded pipeline configuration and identifies its structural
properties (branching, merging, stacking, separation, model placement, etc.).

The topology descriptor is used by:
- The refit mechanism to dispatch to the appropriate strategy
  (simple, stacking, nested, separation).
- The caching layer for shared-prefix detection across pipeline variants.

Example:
    >>> from nirs4all.pipeline.analysis.topology import analyze_topology
    >>> steps = [
    ...     {"class": "sklearn.preprocessing.MinMaxScaler"},
    ...     {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
    ...     {"model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}}},
    ... ]
    >>> topo = analyze_topology(steps)
    >>> topo.has_stacking
    False
    >>> len(topo.model_nodes)
    1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Step keywords that indicate a model step
_MODEL_KEYWORDS = {"model", "meta_model"}

# Step keywords that indicate a merge step
_MERGE_KEYWORDS = {"merge", "merge_sources", "merge_predictions"}

# Separation branch indicators (within a branch dict value)
_SEPARATION_KEYWORDS = {"by_tag", "by_metadata", "by_filter", "by_source"}

# Module path fragments that indicate a model (for serialized class detection)
_MODEL_MODULE_FRAGMENTS = frozenset({
    "cross_decomposition",
    "linear_model",
    "ensemble",
    "svm",
    "neighbors",
    "tree",
    "naive_bayes",
    "neural_network",
    "gaussian_process",
    "discriminant_analysis",
    "xgboost",
    "lightgbm",
    "catboost",
})

# Module path fragments that indicate a splitter
_SPLITTER_MODULE_FRAGMENTS = frozenset({
    "model_selection",
})

# Class name fragments that indicate a splitter
_SPLITTER_CLASS_FRAGMENTS = frozenset({
    "Fold",
    "Split",
    "Splitter",
    "LeaveOne",
    "LeaveP",
})

@dataclass
class ModelNodeInfo:
    """Describes a single model node found in the pipeline.

    Attributes:
        model_class: Fully qualified or short class name of the model.
        branch_path: Branch indices leading to this model (empty for top-level).
        step_index: Position of this model in the step list at its nesting level.
        merge_type: Merge type that follows this model's branch (if any).
        branch_depth: How many branch levels deep this model sits.
    """

    model_class: str
    branch_path: list[int]
    step_index: int
    merge_type: str | None
    branch_depth: int

@dataclass
class PipelineTopology:
    """Structural descriptor of an expanded pipeline configuration.

    Attributes:
        has_stacking: True if ``merge: "predictions"`` is present.
        has_feature_merge: True if ``merge: "features"`` is present.
        has_mixed_merge: True if a merge step combines both features and
            predictions (e.g. ``merge: {"features": ..., "predictions": ...}``).
        has_concat_merge: True if ``merge: "concat"`` is present.
        has_separation_branch: True if any branch uses ``by_metadata``,
            ``by_tag``, ``by_source``, or ``by_filter``.
        has_branches_without_merge: True if a duplication branch is not
            followed by a merge step at the same nesting level.
        max_stacking_depth: Deepest nesting of ``merge: "predictions"``
            (1 for flat stacking, 2+ for nested stacking, 0 if no stacking).
        model_nodes: All model nodes discovered, with location metadata.
        splitter_step_index: Index of the first CV splitter step (0-based),
            or ``None`` if no splitter is found.
        has_sequential_models: True if multiple model steps exist at the
            same nesting level without intervening branch/merge steps.
        has_multi_source: True if a ``by_source`` branch is present.
    """

    has_stacking: bool = False
    has_feature_merge: bool = False
    has_mixed_merge: bool = False
    has_concat_merge: bool = False
    has_separation_branch: bool = False
    has_branches_without_merge: bool = False
    max_stacking_depth: int = 0
    model_nodes: list[ModelNodeInfo] = field(default_factory=list)
    splitter_step_index: int | None = None
    has_sequential_models: bool = False
    has_multi_source: bool = False

def analyze_topology(steps: list[Any]) -> PipelineTopology:
    """Walk an expanded pipeline step list and produce a topology descriptor.

    The function handles nested branch/merge structures recursively, tracking
    branch paths, merge types, and model positions.

    Args:
        steps: List of expanded pipeline steps (dicts, serialized components,
            or direct operator instances).

    Returns:
        A ``PipelineTopology`` describing the structural properties of the
        pipeline.
    """
    topo = PipelineTopology()
    _walk_steps(steps, topo, branch_path=[], stacking_depth=0, pending_merge_type=None)
    return topo

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _walk_steps(
    steps: list[Any],
    topo: PipelineTopology,
    branch_path: list[int],
    stacking_depth: int,
    pending_merge_type: str | None,
) -> None:
    """Recursively walk a list of steps, updating the topology in place.

    Args:
        steps: Steps at the current nesting level.
        topo: Topology accumulator (mutated in place).
        branch_path: Current branch index path (empty at top level).
        stacking_depth: Current nesting depth of ``merge: "predictions"``.
        pending_merge_type: Merge type from the enclosing branch, if any.
    """
    # Track whether consecutive model steps occur without branch/merge
    consecutive_models = 0
    last_had_branch = False  # True after a branch step (before merge)

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            # Non-dict step: check if it looks like a model or splitter
            if _is_model_instance(step):
                model_class = _extract_class_name(step)
                topo.model_nodes.append(ModelNodeInfo(
                    model_class=model_class,
                    branch_path=list(branch_path),
                    step_index=idx,
                    merge_type=pending_merge_type,
                    branch_depth=len(branch_path),
                ))
                consecutive_models += 1
                if consecutive_models >= 2:
                    topo.has_sequential_models = True
            elif _is_splitter_instance(step):
                if topo.splitter_step_index is None:
                    topo.splitter_step_index = idx
                consecutive_models = 0
            else:
                # Other non-dict step (transformer, etc.) -- reset consecutive counter
                consecutive_models = 0
            continue

        # --- Dict step processing ---
        keyword = _primary_keyword(step)

        # Model step
        if keyword in _MODEL_KEYWORDS:
            model_class = _extract_model_class(step, keyword)
            topo.model_nodes.append(ModelNodeInfo(
                model_class=model_class,
                branch_path=list(branch_path),
                step_index=idx,
                merge_type=pending_merge_type,
                branch_depth=len(branch_path),
            ))
            consecutive_models += 1
            if consecutive_models >= 2:
                topo.has_sequential_models = True
            continue

        # Merge step
        if keyword in _MERGE_KEYWORDS:
            merge_value = step[keyword]
            _record_merge(topo, merge_value, stacking_depth)
            last_had_branch = False
            consecutive_models = 0
            continue

        # Branch step
        if keyword == "branch":
            branch_value = step["branch"]
            merge_type_for_branch = _find_next_merge_type(steps, idx)

            if isinstance(branch_value, list):
                # Duplication branches: list of sub-pipelines
                if merge_type_for_branch is None:
                    topo.has_branches_without_merge = True

                new_stacking_depth = stacking_depth
                if merge_type_for_branch == "predictions":
                    new_stacking_depth = stacking_depth + 1
                    if new_stacking_depth > topo.max_stacking_depth:
                        topo.max_stacking_depth = new_stacking_depth

                for branch_idx, sub_steps in enumerate(branch_value):
                    sub_list = sub_steps if isinstance(sub_steps, list) else [sub_steps]
                    _walk_steps(
                        sub_list,
                        topo,
                        branch_path=branch_path + [branch_idx],
                        stacking_depth=new_stacking_depth,
                        pending_merge_type=merge_type_for_branch,
                    )

            elif isinstance(branch_value, dict):
                # Separation or named branches
                if branch_value.keys() & _SEPARATION_KEYWORDS:
                    topo.has_separation_branch = True
                    if "by_source" in branch_value:
                        topo.has_multi_source = True

                    # Walk sub-steps inside the branch dict if present
                    inner_steps = branch_value.get("steps")
                    if inner_steps is not None:
                        if isinstance(inner_steps, list):
                            _walk_steps(
                                inner_steps,
                                topo,
                                branch_path=branch_path + [0],
                                stacking_depth=stacking_depth,
                                pending_merge_type=merge_type_for_branch,
                            )
                        elif isinstance(inner_steps, dict):
                            # Per-source or per-group steps: {"NIR": [...], "markers": [...]}
                            for source_idx, (_source_name, source_steps) in enumerate(inner_steps.items()):
                                if isinstance(source_steps, list):
                                    _walk_steps(
                                        source_steps,
                                        topo,
                                        branch_path=branch_path + [source_idx],
                                        stacking_depth=stacking_depth,
                                        pending_merge_type=merge_type_for_branch,
                                    )
                else:
                    # Named duplication branches: {"name1": [steps], "name2": [steps]}
                    named_entries = [
                        (k, v) for k, v in branch_value.items()
                        if isinstance(v, list) and not k.startswith("_")
                    ]
                    if named_entries:
                        if merge_type_for_branch is None:
                            topo.has_branches_without_merge = True

                        new_stacking_depth = stacking_depth
                        if merge_type_for_branch == "predictions":
                            new_stacking_depth = stacking_depth + 1
                            if new_stacking_depth > topo.max_stacking_depth:
                                topo.max_stacking_depth = new_stacking_depth

                        for branch_idx, (_name, sub_steps) in enumerate(named_entries):
                            _walk_steps(
                                sub_steps,
                                topo,
                                branch_path=branch_path + [branch_idx],
                                stacking_depth=new_stacking_depth,
                                pending_merge_type=merge_type_for_branch,
                            )

            last_had_branch = True
            consecutive_models = 0
            continue

        # Splitter step (serialized or keyword-based)
        if keyword == "split" or _is_serialized_splitter(step):
            if topo.splitter_step_index is None:
                topo.splitter_step_index = idx
            consecutive_models = 0
            continue

        # Serialized model (no explicit "model" keyword)
        if _is_serialized_model(step):
            model_class = _extract_serialized_class_name(step)
            topo.model_nodes.append(ModelNodeInfo(
                model_class=model_class,
                branch_path=list(branch_path),
                step_index=idx,
                merge_type=pending_merge_type,
                branch_depth=len(branch_path),
            ))
            consecutive_models += 1
            if consecutive_models >= 2:
                topo.has_sequential_models = True
            continue

        # Default: other dict step (transform, y_processing, etc.)
        consecutive_models = 0

def _primary_keyword(step: dict[str, Any]) -> str | None:
    """Return the primary actionable keyword of a dict step.

    Checks for model, merge, branch, and split keywords in priority order.
    Falls back to the first non-reserved key.
    """
    # Priority keywords
    for kw in ("model", "meta_model", "merge", "merge_sources", "merge_predictions", "branch", "split"):
        if kw in step:
            return kw
    return None

def _find_next_merge_type(steps: list[Any], branch_idx: int) -> str | None:
    """Look ahead in the step list for the merge step following a branch.

    Only checks the immediate next steps at the same nesting level.

    Returns:
        The merge type string (e.g. ``"predictions"``, ``"features"``),
        ``"mixed"`` for dict merge configs, or ``None`` if no merge found.
    """
    for step in steps[branch_idx + 1:]:
        if not isinstance(step, dict):
            continue
        for kw in ("merge", "merge_sources", "merge_predictions"):
            if kw in step:
                return _classify_merge_value(step[kw])
        # Stop searching after encountering another branch or model
        if isinstance(step, dict):
            pk = _primary_keyword(step)
            if pk in ("branch", "model", "meta_model"):
                break
    return None

def _classify_merge_value(merge_value: Any) -> str:
    """Classify a merge step's value into a canonical type string."""
    if isinstance(merge_value, str):
        return merge_value  # "predictions", "features", "concat", etc.
    if isinstance(merge_value, dict):
        has_preds = "predictions" in merge_value
        has_feats = "features" in merge_value
        if has_preds and has_feats:
            return "mixed"
        if has_preds:
            return "predictions"
        if has_feats:
            return "features"
        # Other dict merge configs (e.g. {"sources": "concat"})
        return "dict"
    return "unknown"

def _record_merge(topo: PipelineTopology, merge_value: Any, stacking_depth: int) -> None:
    """Update topology flags based on a merge step's value."""
    merge_type = _classify_merge_value(merge_value)

    if merge_type == "predictions":
        topo.has_stacking = True
        depth = stacking_depth + 1
        if depth > topo.max_stacking_depth:
            topo.max_stacking_depth = depth
    elif merge_type == "features":
        topo.has_feature_merge = True
    elif merge_type == "mixed":
        topo.has_mixed_merge = True
    elif merge_type == "concat":
        topo.has_concat_merge = True

# ---------------------------------------------------------------------------
# Model / Splitter detection helpers
# ---------------------------------------------------------------------------

def _is_model_instance(obj: Any) -> bool:
    """Check if a non-dict object is a model instance (has fit + predict)."""
    return hasattr(obj, "fit") and hasattr(obj, "predict") and not _is_splitter_instance(obj)

def _is_splitter_instance(obj: Any) -> bool:
    """Check if a non-dict object is a CV splitter (has split(X, ...))."""
    import inspect as _inspect

    split_fn = getattr(obj, "split", None)
    if not callable(split_fn):
        return False
    try:
        sig = _inspect.signature(split_fn)
    except (TypeError, ValueError):
        return False
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return bool(params) and params[0].name == "X"

def _is_serialized_model(step: dict[str, Any]) -> bool:
    """Check if a serialized dict step represents a model.

    Looks for ``class`` key with a module path containing known model
    module fragments.
    """
    class_path = step.get("class", "")
    if not isinstance(class_path, str):
        return False
    class_path_lower = class_path.lower()
    return any(frag in class_path_lower for frag in _MODEL_MODULE_FRAGMENTS)

def _is_serialized_splitter(step: dict[str, Any]) -> bool:
    """Check if a serialized dict step represents a splitter.

    Looks for ``class`` key with a module path containing ``model_selection``
    or a class name containing splitter fragments.
    """
    class_path = step.get("class", "")
    if not isinstance(class_path, str) or not class_path:
        return False
    class_name = class_path.rsplit(".", 1)[-1]
    class_path_lower = class_path.lower()

    if any(frag in class_path_lower for frag in _SPLITTER_MODULE_FRAGMENTS):
        return True
    return any(frag in class_name for frag in _SPLITTER_CLASS_FRAGMENTS)

def _extract_class_name(obj: Any) -> str:
    """Get the class name string from a live object."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"

def _extract_model_class(step: dict[str, Any], keyword: str) -> str:
    """Extract the model class name from a model step dict."""
    model_value = step.get(keyword)
    if model_value is None:
        return "unknown"
    if isinstance(model_value, dict):
        # Serialized: {"class": "module.Class", "params": {...}}
        return _extract_serialized_class_name(model_value)
    if isinstance(model_value, str):
        return model_value
    # Live instance
    return _extract_class_name(model_value)

def _extract_serialized_class_name(step: dict[str, Any]) -> str:
    """Extract a class name from a serialized step dict."""
    for key in ("class", "function", "module"):
        val = step.get(key)
        if isinstance(val, str):
            return val
    return "unknown"
