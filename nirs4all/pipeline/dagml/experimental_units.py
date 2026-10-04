"""Explicit experimental-unit contracts and native supplied fit influence.

This adapter never counts repetitions, computes weights, resamples rows or
scores predictions. DAG-ML owns those operations and the split authority.
"""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Mapping
from typing import Any

import numpy as np


def experimental_unit_metadata(dataset: Any) -> dict[str, dict[int, str]]:
    """Project only explicitly declared IO identities into relation metadata."""
    units = getattr(dataset, "independent_unit_ids", None)
    if units is None:
        return {}
    result = {"independent_unit_id": dict(enumerate(units))}
    repetitions = getattr(dataset, "repetition_ids", None)
    if repetitions is not None:
        result["repetition_id"] = dict(enumerate(repetitions))
    return result


def experimental_unit_contract(dataset: Any, identity: Any, sample_ints: list[int], *,
                               target_values: Any, target_names: list[str]) -> dict[str, Any] | None:
    """Serialize the declared Train design and native numeric target vocabulary."""
    units = getattr(dataset, "independent_unit_ids", None)
    if units is None:
        return None
    cohort = getattr(dataset, "cohort", None)
    if cohort is None or not sample_ints or len(sample_ints) != len(set(sample_ints)):
        raise ValueError("experimental-unit influence requires an explicit IO cohort and unique Train rows")
    if (len(units) != len(cohort) or any(isinstance(row, bool) or not isinstance(row, (int, np.integer))
                                        or row < 0 or row >= len(cohort) for row in sample_ints)):
        raise ValueError("experimental-unit influence requires exact IO row identities")
    if any(cohort.partitions[row] != "train" for row in sample_ints):
        raise ValueError("experimental-unit descriptor may contain only Train rows")
    if any(not isinstance(unit, str) or re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", unit) is None for unit in units):
        raise ValueError("experimental-unit Group scoring requires ASCII unit identifiers of at most 128 bytes (letters, digits, '_-.:'), without aliases")
    values = np.asarray(target_values, dtype=float).reshape(len(sample_ints), -1)
    if values.shape[1] != len(target_names) or not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("experimental-unit target values must align to explicit target names")
    if cohort.target_mask is None:
        raise ValueError("experimental-unit influence requires explicit Train target availability")
    masks = np.asarray(cohort.target_mask, dtype=bool)[sample_ints].reshape(values.shape)
    if not np.isfinite(values[masks]).all():
        raise ValueError("experimental-unit observed target values must be finite")
    task_type = cohort.task_type
    if task_type not in {"regression", "classification"}:
        raise ValueError("experimental-unit influence requires explicit IO task_type regression or classification")
    if task_type == "classification" and (len(target_names) != 1 or not masks.all()):
        raise ValueError("experimental-unit classification requires complete mono-y labels")
    return {"schema_version": 1, "sample_ids": [identity.to_wire(row) for row in sample_ints],
            "independent_unit_ids": [units[row] for row in sample_ints],
            "fit_influence_policy": "equal_sample_influence", "task_type": task_type,
            "target_names": list(target_names),
            "target_values": [[float(value) if observed else None for value, observed in zip(row, mask, strict=True)]
                              for row, mask in zip(values, masks, strict=True)]}


def apply_experimental_unit_contract(dsl: dict[str, Any], dataset: Any, identity: Any,
                                     sample_ints: list[int], *, target_values: Any,
                                     target_names: list[str]) -> dict[str, Any] | None:
    """Bind unit policy before native compilation/HPO and campaign scoring."""
    contract = experimental_unit_contract(dataset, identity, sample_ints,
                                          target_values=target_values, target_names=target_names)
    if contract is None:
        return None
    metadata = dsl.setdefault("metadata", {})
    if "experimental_unit" in metadata and metadata["experimental_unit"] != contract:
        raise ValueError("experimental-unit declaration changed before native compilation")
    metadata["experimental_unit"] = copy.deepcopy(contract)
    policy = {"aggregation_level": "group", "selection_metric_level": "group",
              "method": "vote" if contract["task_type"] == "classification" else "mean",
              "grouping_key": {"kind": "relation_metadata", "key": "independent_unit_id"}}
    dsl["aggregation_policy"] = {**(dsl.get("aggregation_policy") or {}), **policy}
    selected = {dataset.cohort.sample_ids[row] for row in sample_ints}
    missing_sources = any(not present for source in dataset.cohort.sources.values()
                          for sample, present in zip(source.sample_ids, source.presence_mask, strict=True)
                          if sample in selected)
    require_weighted_context(dsl, missing_sources=missing_sources)
    return contract


def require_sample_weight_support(estimator: Any) -> None:
    """Admit explicit weighted fit signatures across every learned component."""
    from sklearn.pipeline import Pipeline

    from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor

    if estimator is None or isinstance(estimator, str) and estimator == "passthrough":
        return
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            require_sample_weight_support(step)
        return
    if isinstance(estimator, (MultimodalClassifier, MultimodalRegressor)):
        if estimator.backend != "sklearn":
            raise ValueError("experimental-unit influence is unavailable for the Methods backend")
        from .named_torch import is_named_torch_model

        if is_named_torch_model(estimator):
            raise ValueError("named Torch is outside the experimental-unit influence profile")
        for transformer in estimator.transformers.values():
            require_sample_weight_support(transformer)
        if estimator.model is None or isinstance(estimator.model, str):
            raise ValueError("experimental-unit influence requires a learned weighted prediction head")
        require_sample_weight_support(estimator.model)
        return
    if type(estimator).__module__ == "nirs4all.pipeline.dagml.node_runner" and type(estimator).__name__ == "_PerTargetLateEstimator":
        for transformer in estimator.chain_template or ():
            require_sample_weight_support(transformer)
        require_sample_weight_support(estimator.model)
        return
    try:
        parameter = inspect.signature(estimator.fit).parameters.get("sample_weight")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("experimental-unit influence requires an explicit sample_weight fit parameter") from exc
    if parameter is None or parameter.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
        raise ValueError(f"{type(estimator).__name__} does not explicitly support sample_weight; refusing before FIT")
    if any(callable(getattr(estimator, name, None)) for name in ("fit_with_views", "fit_selected", "select_with_views")):
        raise ValueError("experimental-unit influence does not support partition-aware transfer fit APIs")


def weighted_fit_kwargs(estimator: Any, weights: Any) -> dict[str, Any]:
    """Route the exact native weight vector through sklearn's public fit API."""
    from sklearn.pipeline import Pipeline

    if weights is None:
        return {}
    require_sample_weight_support(estimator)
    if isinstance(estimator, Pipeline):
        return {f"{name}__{key}": value for name, step in estimator.steps
                for key, value in weighted_fit_kwargs(step, weights).items()
                if step is not None and not isinstance(step, str)}
    if estimator is None or isinstance(estimator, str):
        return {}
    return {"sample_weight": weights}


def native_fit_weights(task: dict[str, Any], dataset: Any, identity: Any, sample_ids: list[str], *,
                       target_names: list[str] | None = None, per_target: bool = False) -> np.ndarray | None:
    """Validate and consume native weights verbatim; never recompute influence."""
    units = getattr(dataset, "independent_unit_ids", None)
    influence = task.get("fit_influence")
    if units is None:
        if isinstance(influence, dict) and influence.get("independent_unit_ids"):
            raise ValueError("native unit weights require explicitly declared IO independent_unit_ids")
        return None
    if (task.get("phase") not in {"FIT_CV", "REFIT"} or not isinstance(influence, dict)
            or influence.get("fit_sample_ids") != sample_ids
            or influence.get("independent_unit_ids") != [units[identity.to_int(sample)] for sample in sample_ids]):
        raise ValueError("native fit influence does not match the actual ordered unit/row scope")
    raw = influence.get("target_row_weights") if per_target else influence.get("row_weights")
    weights = np.array(raw, dtype=float, copy=True)
    expected = (len(sample_ids), len(target_names or ())) if per_target else (len(sample_ids),)
    if (weights.shape != expected or not np.isfinite(weights).all() or np.any(weights < 0)
            or not per_target and np.any(weights <= 0)
            or per_target and influence.get("target_names") != target_names):
        raise ValueError("native fit influence has invalid row/target weight shape, values or target names")
    weights.setflags(write=False)
    return weights


def require_weighted_context(context: Any, *, missing_sources: bool = False) -> bool:
    """Check all declared operators before advertising native weighted support."""
    from .operator_routing import route_graph_node

    if not isinstance(context, dict) or not context.get("metadata", {}).get("experimental_unit"):
        return False

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            metadata = value.get("metadata") or {}
            if metadata.get("nirs4all_finetune_params") or value.get("finetune_params"):
                raise ValueError("experimental-unit influence requires native whole-pipeline tuning; branch-local finetune is unsupported")
            node = value if value.get("kind") in {"model", "transform", "y_transform"} and "operator" in value else None
            if node is None and isinstance(value.get("model"), str):
                node = {"kind": "model", "operator": value["model"], "params": value.get("params") or {}}
            if node is None and "y_processing" in value:
                raise ValueError("experimental-unit influence does not support separate target transforms")
            if node is None and isinstance(value.get("preprocessing"), dict):
                node = {"kind": "transform", "operator": value["preprocessing"]}
            if node is None and isinstance(value.get("class"), str) and "." in value["class"]:
                node = {"kind": "transform", "operator": value}
            if node is not None:
                if node["kind"] == "y_transform":
                    raise ValueError("experimental-unit influence does not support separate target transforms")
                estimator = route_graph_node(node)
                if callable(getattr(estimator, "split", None)) and not callable(getattr(estimator, "fit", None)):
                    return
                from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor

                availability = context.get("metadata", {}).get("prediction_availability") or {}
                if (isinstance(estimator, (MultimodalClassifier, MultimodalRegressor))
                        and (missing_sources or any(not all(rows) for rows in availability.get("source_presence", {}).values()))):
                    raise ValueError("experimental-unit early/intermediate fusion requires complete feature sources")
                if metadata.get("source_concat_preprocessing"):
                    raise ValueError("experimental-unit influence does not support learned source concatenation")
                require_sample_weight_support(estimator)
            for key, child in value.items():
                if key not in {"params", "operator", "model", "preprocessing", "class"}:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(context.get("pipeline", context.get("steps", context.get("nodes", []))))
    return True
