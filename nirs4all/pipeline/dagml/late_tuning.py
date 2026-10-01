"""Bind public whole-stack search paths to native nodes and the selected recipe."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .detect import _detect_by_source_stacking_branch
from .host_finetune import is_host_finetune, validate_host_finetune
from .source_stacking import lower_source_stacking
from .steps import _is_split_step
from .tuning_contracts import normalize_parameter_path


def validate_nested_local_finetune(step: dict[str, Any], paths: Any, *, prefix: str = "") -> dict[str, Any] | None:
    """Normalize local host search and refuse competing parameter owners before fitting."""
    if "finetune_params" not in step:
        return None
    config = step["finetune_params"]
    if not isinstance(config, dict):
        raise TypeError("nested finetune_params must be a mapping")
    if not is_host_finetune(config):
        raise ValueError("nested finetune_params requires the Optuna or n4m host search profile; use tuning.space for deterministic generation")
    params = validate_host_finetune(config)
    manager: Any
    if params["engine"] == "n4m":
        from nirs4all.optimization.n4m_engine import N4MFinetuneManager

        manager = N4MFinetuneManager()
        model_keys = manager._flatten(params["model_params"]).keys()  # noqa: SLF001 -- optimizer owns nested parameter grammar
    else:
        from nirs4all.optimization.optuna import OptunaManager

        manager = OptunaManager()
        model_keys = manager._flatten_nested_params(params["model_params"]).keys()  # noqa: SLF001 -- optimizer owns nested parameter grammar
    forced = params.get("force_params", {})
    if forced is None:
        forced = {}
    if not isinstance(forced, dict):
        raise TypeError("nested finetune_params.force_params must be a mapping")
    local_keys = [*model_keys, *(params.get("train_params") or {}), *forced]
    local_paths = [normalize_parameter_path(f"{prefix}.{key}" if prefix else key)[0] for key in local_keys]
    global_paths = [normalize_parameter_path(path)[0] for path in paths]
    collisions = sorted({
        path
        for path in global_paths
        for local in local_paths
        if path == local or path.startswith(local + ".") or local.startswith(path + ".")
    })
    if collisions:
        raise ValueError(f"global tuning.space and local finetune_params overlap: {collisions}; each parameter must have one search owner")
    return params


@dataclass
class LateFusionTuningRecipe:
    """One unfitted recipe with shared parameter addresses for search and refit."""

    pipeline: list[Any]
    branches: list[list[Any]]
    meta: Any
    layout: dict[str, Any]
    bindings: dict[str, dict[str, str]]
    operators: dict[str, tuple[Any, str]]
    has_local_finetune: bool = False

    def selected_pipeline(self, params: dict[str, Any]) -> list[Any]:
        """Patch a private copy while preserving references within its branches."""
        selected = copy.deepcopy(self)
        for path, value in params.items():
            operator, parameter = selected.operators[path]
            operator.set_params(**{parameter: value})
        for branch in selected.branches:
            for step in branch:
                if isinstance(step, dict) and "finetune_params" in step:
                    # Re-enter the public runner with public declarations; the
                    # real splitter is reattached by source stacking lowering.
                    step["finetune_params"].pop("__dagml_inner_splitter", None)
        return selected.pipeline


def prepare_late_tuning(pipeline: list[Any], dataset: Any, paths: Any) -> LateFusionTuningRecipe:
    """Validate the full recipe before opening a study or fitting any operator."""
    pipeline = copy.deepcopy(pipeline)
    detected = _detect_by_source_stacking_branch(pipeline, dataset.n_sources)
    if detected is None:
        raise ValueError("multimodal tuning requires one multimodal model or by_source branches followed by merge='predictions' and a meta-model")
    if not dataset.cohort.target_mask.all():
        raise ValueError("late-fusion tuning requires complete targets")
    body, meta = detected
    meta_step = next((step for step in pipeline if isinstance(step, dict) and "model" in step), None)
    if meta_step is not None and "finetune_params" in meta_step:
        raise ValueError("meta-model HPO requires a native whole-stack nested search; reusing a precomputed OOF matrix would leak its inner selection targets")
    has_local_finetune = False
    bodies = body.items() if isinstance(body, dict) else ((source, body) for source in dataset.source_names)
    for source, branch in bodies:
        for index, step in enumerate(branch):
            if isinstance(step, dict) and "finetune_params" in step:
                params = validate_nested_local_finetune(step, paths, prefix=f"branches.{source}.{index}")
                step["finetune_params"] = params
                has_local_finetune = True
    _lowered, branches, layout = lower_source_stacking(
        pipeline, body, source_widths=dataset.num_features, source_names=list(dataset.source_names),
        source_descriptors=dataset.cohort.schema_descriptors(),
    )
    missing_policy = layout.get("missing_source_policy", "error")
    if has_local_finetune and missing_policy != "error":
        raise ValueError("nested local HPO requires complete sources and missing_source_policy='error'")
    if missing_policy == "zero_with_indicator" and dataset.is_classification:
        raise ValueError("late-fusion missing_source_policy='zero_with_indicator' currently requires regression")
    if missing_policy == "error" and any(not mask.all() for mask in dataset.cohort.source_presence().values()):
        raise ValueError("late-fusion tuning requires complete sources unless missing_source_policy='zero_with_indicator' is explicit")
    # Store independently addressable source bodies even for a shared public list.
    branch_step = next(step for step in pipeline if isinstance(step, dict) and "branch" in step)
    branch_step["branch"]["steps"] = dict(zip(dataset.source_names, branches, strict=True))
    meta_step = next(step for step in pipeline if isinstance(step, dict) and "model" in step)
    meta_step["model"] = meta
    addresses: dict[str, tuple[Any, str, str]] = {}

    def register(step: Any, prefix: str, node_id: str) -> None:
        if isinstance(step, dict):
            if "model" not in step or set(step) - {"model", "train_params", "refit_params", "finetune_params", "name"}:
                raise ValueError("whole-stack tuning model steps accept only model, train_params, refit_params, finetune_params and name")
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
    return LateFusionTuningRecipe(pipeline, branches, meta, layout, bindings, operators, has_local_finetune)
