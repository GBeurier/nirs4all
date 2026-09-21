"""Bind public whole-stack search paths to native nodes and the selected recipe."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .detect import _detect_by_source_stacking_branch
from .source_stacking import lower_source_stacking
from .steps import _is_split_step
from .tuning_contracts import normalize_parameter_path


@dataclass
class LateFusionTuningRecipe:
    """One unfitted recipe with shared parameter addresses for search and refit."""

    pipeline: list[Any]
    branches: list[list[Any]]
    meta: Any
    layout: dict[str, Any]
    bindings: dict[str, dict[str, str]]
    operators: dict[str, tuple[Any, str]]

    def selected_pipeline(self, params: dict[str, Any]) -> list[Any]:
        """Patch a private copy while preserving references within its branches."""
        selected = copy.deepcopy(self)
        for path, value in params.items():
            operator, parameter = selected.operators[path]
            operator.set_params(**{parameter: value})
        return selected.pipeline


def prepare_late_tuning(pipeline: list[Any], dataset: Any, paths: Any) -> LateFusionTuningRecipe:
    """Validate the full recipe before opening a study or fitting any operator."""
    pipeline = copy.deepcopy(pipeline)
    detected = _detect_by_source_stacking_branch(pipeline, dataset.n_sources)
    if detected is None:
        raise ValueError("multimodal tuning requires one multimodal model or by_source branches followed by merge='predictions' and a meta-model")
    if not dataset.cohort.target_mask.all() or any(not mask.all() for mask in dataset.cohort.source_presence().values()):
        raise ValueError("late-fusion tuning requires complete targets and complete sources")
    body, meta = detected
    _lowered, branches, layout = lower_source_stacking(
        pipeline, body, source_widths=dataset.num_features, source_names=list(dataset.source_names),
        source_descriptors=dataset.cohort.schema_descriptors(),
    )
    # Store independently addressable source bodies even for a shared public list.
    branch_step = next(step for step in pipeline if isinstance(step, dict) and "branch" in step)
    branch_step["branch"]["steps"] = dict(zip(dataset.source_names, branches, strict=True))
    meta_step = next(step for step in pipeline if isinstance(step, dict) and "model" in step)
    meta_step["model"] = meta
    addresses: dict[str, tuple[Any, str, str]] = {}

    def register(step: Any, prefix: str, node_id: str) -> None:
        if isinstance(step, dict):
            if set(step) != {"model"}:
                raise ValueError("whole-stack tuning requires plain model steps; put all searched controls in tuning.space")
            operator = step["model"]
        else:
            operator = step
        if isinstance(operator, type) or not callable(getattr(operator, "get_params", None)):
            raise ValueError("whole-stack tuning requires instantiated sklearn-compatible operators")
        for parameter in operator.get_params(deep=True):
            path, _segments = normalize_parameter_path(f"{prefix}.{parameter}")
            if path in addresses:
                raise ValueError(f"ambiguous whole-stack parameter path {path!r}")
            addresses[path] = (operator, parameter, node_id)

    register(meta_step, "meta", "merge:stack")
    for index, (source, branch) in enumerate(zip(dataset.source_names, branches, strict=True)):
        node_index = 0
        for step_index, step in enumerate(branch):
            if step is None:
                continue
            if _is_split_step(step):
                raise ValueError("whole-stack tuning requires only the outer pipeline splitter")
            register(step, f"branches.{source}.{step_index}", f"branch:{index}.node:{node_index}")
            node_index += 1
    bindings = {}
    operators = {}
    for path in paths:
        if path not in addresses:
            raise ValueError(f"unknown whole-stack tuning path {path!r}; use branches.<source>.<step_index>.<parameter> or meta.<parameter>")
        operator, parameter, node_id = addresses[path]
        bindings[path] = {"node_id": node_id, "param_path": normalize_parameter_path(parameter)[0]}
        operators[path] = (operator, parameter)
    return LateFusionTuningRecipe(pipeline, branches, meta, layout, bindings, operators)
