"""Lower the deterministic subset of model-local ``finetune_params``.

This module deliberately does not implement an optimizer. It only translates
fixed model-parameter variant generation into the generator syntax already
owned by the DAG-ML pipeline bridge. Adaptive controls stay fail-closed until
the n4m/Optuna adapter lane is wired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nirs4all.pipeline.dagml_bridge import is_grid_param_generator_spec, is_param_generator_spec

SUPPORTED_FINETUNE_META_KEYS = frozenset({"model_params", "metric", "direction", "eval_mode", "approach", "engine"})
DETERMINISTIC_FINETUNE_ENGINES = frozenset({"", "dag-ml", "dagml", "native", "grid"})
CORE_DAGML_SELECTION_METRICS = frozenset({"mse", "rmse", "mae", "r2", "accuracy", "balanced_accuracy"})
PUBLIC_DAGML_SELECTION_METRICS = frozenset({"rmse", "accuracy", "balanced_accuracy"})
UNSUPPORTED_NATIVE_TRAINING_PARAM_KEYS = frozenset({"train_params", "refit_params"})


def reject_native_training_param_overrides(
    steps: list[Any],
    *,
    context: str = "native DAG-ML",
    allowed_keys: frozenset[str] = frozenset(),
) -> None:
    """Reject fit/refit kwargs that native DAG-ML would otherwise ignore."""

    rejected_keys = UNSUPPORTED_NATIVE_TRAINING_PARAM_KEYS - allowed_keys
    hits: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            hits.extend(sorted(rejected_keys & set(step)))
    if hits:
        raise NotImplementedError(f"{context} does not yet support step-level {sorted(set(hits))}; running natively would ignore fit/refit arguments instead of preserving legacy parity.")


def lower_deterministic_finetune_params_to_generators(
    steps: list[Any],
    *,
    context: str = "native DAG-ML",
    supported_selection_metrics: frozenset[str] | None = CORE_DAGML_SELECTION_METRICS,
) -> tuple[list[Any], dict[str, str]]:
    """Lower deterministic ``finetune_params.model_params`` to DAG-ML generators.

    Supported:
        - plain JSON grids, lowered to a step-level ``_grid_``;
        - native ``_range_`` / ``_log_range_`` list-form per-parameter specs;
        - optional ``metric`` and ``direction`` selection metadata.

    Refused:
        - adaptive engines such as n4m/Optuna;
        - trial counts, samplers, pruners and phases;
        - train/refit fit-argument sampling.
    """

    lowered: list[Any] = []
    overrides: dict[str, str] = {}
    seen = False
    for step in steps:
        if not (isinstance(step, dict) and "model" in step and "finetune_params" in step):
            lowered.append(step)
            continue
        if seen:
            raise ValueError(f"{context} finetune_params lowering supports exactly one model step")
        seen = True
        finetune_params = step["finetune_params"]
        if not isinstance(finetune_params, Mapping):
            raise TypeError("finetune_params must be a mapping for native DAG-ML lowering")
        unknown = sorted(set(finetune_params) - SUPPORTED_FINETUNE_META_KEYS)
        if unknown:
            raise ValueError(f"{context} finetune_params does not support keys {unknown}; supported deterministic keys are {sorted(SUPPORTED_FINETUNE_META_KEYS)}")
        engine = str(finetune_params.get("engine", "dag-ml")).strip().lower()
        if engine not in DETERMINISTIC_FINETUNE_ENGINES:
            raise ValueError(f"{context} finetune_params currently supports only deterministic DAG-ML generation; n4m/Optuna engines remain a follow-up adapter lane")
        if finetune_params.get("approach", "grouped") != "grouped":
            raise ValueError(f"{context} finetune_params currently supports only approach='grouped'")
        eval_mode = finetune_params.get("eval_mode", "mean")
        if eval_mode == "avg":
            eval_mode = "mean"
        if eval_mode not in {"mean", "best"}:
            raise ValueError(f"{context} finetune_params currently supports only eval_mode='mean' or 'best'")
        if "metric" in finetune_params:
            metric = str(finetune_params["metric"]).strip().lower() if isinstance(finetune_params["metric"], str) else finetune_params["metric"]
            if not isinstance(metric, str) or not metric:
                raise ValueError("finetune_params.metric must be a non-empty string")
            if supported_selection_metrics is not None and metric not in supported_selection_metrics:
                raise ValueError(f"{context} finetune_params.metric={metric!r} is not supported; supported metrics are {sorted(supported_selection_metrics)}")
            overrides["selection_metric"] = metric
        if "direction" in finetune_params:
            direction = str(finetune_params["direction"]).strip().lower()
            if direction not in {"minimize", "maximize"}:
                raise ValueError("finetune_params.direction must be 'minimize' or 'maximize'")
            overrides["selection_objective"] = direction
        model_params = finetune_params.get("model_params")
        if not isinstance(model_params, Mapping) or not model_params:
            raise ValueError(f"{context} finetune_params requires a non-empty model_params mapping")
        lowered_step = {key: value for key, value in step.items() if key != "finetune_params"}
        if any(key in lowered_step for key in model_params):
            collision = sorted(key for key in model_params if key in lowered_step)
            raise ValueError(f"finetune_params.model_params collide with explicit model step keys {collision}")
        if "_grid_" in lowered_step:
            raise ValueError("native DAG-ML finetune_params cannot be combined with an explicit _grid_ model generator")
        if is_grid_param_generator_spec(dict(model_params)):
            lowered_step["_grid_"] = dict(model_params)
        else:
            unsupported: list[str] = []
            for key, value in model_params.items():
                if not isinstance(key, str) or not is_param_generator_spec(value):
                    unsupported.append(str(key))
                    continue
                lowered_step[key] = value
            if unsupported:
                raise ValueError(f"native DAG-ML finetune_params.model_params currently supports only plain JSON grids or _range_/_log_range_ list forms; unsupported params: {sorted(unsupported)}")
        lowered.append(lowered_step)
    return lowered, overrides


__all__ = [
    "CORE_DAGML_SELECTION_METRICS",
    "PUBLIC_DAGML_SELECTION_METRICS",
    "lower_deterministic_finetune_params_to_generators",
    "reject_native_training_param_overrides",
]
