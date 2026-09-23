"""Structural normalization of public pipeline shorthand before DAG lowering.

These helpers neither instantiate nor fit operators, and never select an engine.
Explicit keyword steps remain authoritative: an estimator under preprocessing
must not become a model merely because it also exposes ``predict``.
"""

from __future__ import annotations

from typing import Any


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
            if substeps and all(_is_bare_transform(substep) for substep in substeps):
                # Legacy's StepRunner executes a transform-only subpipeline in
                # order against the same dataset. DAG-ML's linear X chain has
                # the same structure once the grouping list is removed.
                normalized.extend(substeps)
            else:
                # Model subpipelines have special winner selection on replay;
                # keep their grouping until that contract is lowered explicitly.
                normalized.append(substeps)
        elif isinstance(step, dict) and "branch" in step:
            normalized.append({**step, "branch": _normalize_branch(step["branch"])})
        elif isinstance(step, dict) and step.get("framework") == "autogluon" and "model" not in step:
            normalized.append({**step, "model": {"framework": "autogluon"}})
        elif not isinstance(step, dict) and callable(getattr(step, "fit", None)) and callable(getattr(step, "predict", None)):
            normalized.append({"model": step})
        else:
            normalized.append(step)
    return normalized


def _is_bare_transform(step: Any) -> bool:
    return (
        not isinstance(step, dict)
        and callable(getattr(step, "fit", None))
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
