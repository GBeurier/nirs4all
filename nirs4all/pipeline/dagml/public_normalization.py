"""Structural normalization of public pipeline shorthand before DAG lowering.

These helpers neither instantiate nor fit operators, and never select an engine.
Explicit keyword steps remain authoritative: an estimator under preprocessing
must not become a model merely because it also exposes ``predict``.
"""

from __future__ import annotations

from typing import Any

from nirs4all.operators.models.base import BaseModelOperator
from nirs4all.operators.models.residual import ResidualModel


def normalize_model_steps(steps: list[Any]) -> list[Any]:
    """Make bare fit/predict operators explicit without changing caller objects.

    The historical model controller recognizes this same duck-typed model
    contract before checking the transformer contract (notably for PLS).
    Apply it only to executable steps, never to estimator parameters or to
    explicit transform/model payloads. Branch order and metadata are preserved.
    """
    normalized: list[Any] = []
    for step in steps:
        if isinstance(step, list):
            substeps = normalize_model_steps(step)
            if substeps and _is_linear_subpipeline(substeps):
                # Legacy's StepRunner executes these transforms and model
                # choices in order. DAG-ML's linear chain and model selection
                # retain their behavior after removing the grouping list.
                normalized.extend(substeps)
            else:
                # Other subpipelines can carry branches or post-model
                # transforms whose scope needs separate lowering.
                normalized.append(substeps)
        elif isinstance(step, dict) and "branch" in step:
            normalized.append({**step, "branch": _normalize_branch(step["branch"])})
        elif isinstance(step, dict) and step.get("framework") == "autogluon" and "model" not in step:
            normalized.append({**step, "model": {"framework": "autogluon"}})
        elif isinstance(step, BaseModelOperator) or (
            not isinstance(step, dict)
            and callable(getattr(step, "fit", None))
            and callable(getattr(step, "predict", None))
        ):
            normalized.append({"model": step})
        else:
            normalized.append(step)
    return normalized


def _is_linear_subpipeline(steps: list[Any]) -> bool:
    from .steps import _is_split_step

    if len(steps) > 1 and _is_split_step(steps[0]) and all(_is_bare_transform(step) for step in steps[1:]):
        return True
    seen_model = False
    for step in steps:
        if isinstance(step, dict) and "model" in step and set(step) <= {"model", "train_params", "refit_params"}:
            model = step["model"]
            if not isinstance(model, BaseModelOperator) and (
                not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None))
            ):
                return False
            seen_model = True
        elif isinstance(step, dict) and set(step) == {"residual"}:
            operator = step["residual"]
            if not isinstance(operator, ResidualModel) and not (
                isinstance(operator, dict) and {"base", "learner"} <= set(operator)
            ):
                return False
            seen_model = True
        elif seen_model or not _is_bare_transform(step):
            return False
    return True


def _is_bare_transform(step: Any) -> bool:
    if isinstance(step, dict):
        return set(step) == {"preprocessing"} and _is_bare_transform(step["preprocessing"])
    return (
        callable(getattr(step, "fit", None))
        and callable(getattr(step, "transform", None))
        and not callable(getattr(step, "predict", None))
    )


def _normalize_branch(branch: Any) -> Any:
    if isinstance(branch, list):
        return [normalize_model_steps(body) if isinstance(body, list) else body for body in branch]
    if not isinstance(branch, dict):
        return branch
    if any(key in branch for key in ("by_source", "by_metadata", "by_tag", "by_filter")):
        body = branch.get("steps")
        if isinstance(body, list):
            return {**branch, "steps": normalize_model_steps(body)}
        if isinstance(body, dict):
            return {**branch, "steps": {key: normalize_model_steps(value) if isinstance(value, list) else value for key, value in body.items()}}
        return branch
    return {
        key: value if key in {"parallel", "n_jobs"} or not isinstance(key, str) or key.startswith("_")
        else normalize_model_steps(value if isinstance(value, list) else [value])
        for key, value in branch.items()
    }
