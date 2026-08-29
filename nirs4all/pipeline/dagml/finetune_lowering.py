"""Lower the deterministic subset of model-local ``finetune_params``.

This module deliberately does not implement an optimizer. It only translates
fixed model-parameter variant generation into the generator syntax already
owned by the DAG-ML pipeline bridge. Adaptive controls stay fail-closed until
the n4m/Optuna adapter lane is wired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_FINETUNE_META_KEYS = frozenset({"model_params", "metric", "direction", "eval_mode", "approach", "engine"})
DETERMINISTIC_FINETUNE_ENGINES = frozenset({"", "dag-ml", "dagml", "native", "grid"})
CORE_DAGML_SELECTION_METRICS = frozenset({"mse", "rmse", "mae", "r2", "accuracy", "balanced_accuracy"})
PUBLIC_DAGML_SELECTION_METRICS = frozenset({"rmse", "accuracy", "balanced_accuracy"})
UNSUPPORTED_NATIVE_TRAINING_PARAM_KEYS = frozenset({"train_params", "refit_params"})
_REFIT_NOOP_MODEL_STEP_KEYS = frozenset({"name", "model", "refit_params"})
_BRANCH_CONFIG_KEYS = frozenset({"parallel", "n_jobs"})
_SEPARATION_BRANCH_KEYS = frozenset({"by_source", "by_tag", "by_metadata", "by_filter"})


def _is_safe_refit_noop_direct_step(value: Any) -> bool:
    """Whether a direct item is unambiguously non-model for the no-op proof."""

    if value is None:
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    is_predictor = callable(getattr(value, "predict", None))
    return (
        callable(getattr(value, "split", None)) and not is_predictor
    ) or (
        callable(getattr(value, "transform", None)) and not is_predictor
    )


def _has_nested_structural_refit_params(steps: list[Any]) -> bool:
    """Find refit keys only inside executable branch/wrapper/generator grammar.

    Arbitrary estimator, preprocessing, metadata, and label payload mappings
    remain opaque. The active identity stack makes malformed cyclic branch
    payloads fail safely without treating aliases as a single occurrence.
    """

    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS

    active_container_ids: set[int] = set()

    def enter(value: Any) -> bool:
        value_id = id(value)
        if value_id in active_container_ids:
            return False
        active_container_ids.add(value_id)
        return True

    def visit_sequence(value: Any, *, branch_body: bool = False) -> bool:
        if not isinstance(value, (list, tuple)):
            return visit_step(value, branch_body=branch_body)
        if not enter(value):
            return False
        try:
            return any(visit_sequence(item, branch_body=branch_body) for item in value)
        finally:
            active_container_ids.remove(id(value))

    def visit_step(value: Any, *, branch_body: bool = False, branch_list_wrapper: bool = False) -> bool:
        if not isinstance(value, dict):
            return False
        if not enter(value):
            return False
        try:
            if branch_list_wrapper and "steps" in value:
                return visit_sequence(value["steps"], branch_body=True)
            if branch_body and "refit_params" in value:
                return True
            if branch_body and any(key in GENERATION_KEYWORDS for key in value):
                return visit_generator(value)
            if "branch" in value:
                return visit_branch(value["branch"])
            if not branch_body and set(value) <= {"name", "pipeline"} and "pipeline" in value:
                return visit_sequence(value["pipeline"], branch_body=True)
            if not branch_body and set(value) <= {"name", "steps"} and "steps" in value:
                return visit_sequence(value["steps"], branch_body=True)
            return False
        finally:
            active_container_ids.remove(id(value))

    def visit_generator(value: dict[Any, Any]) -> bool:
        """Inspect generator choices, but never arbitrary sibling payload maps."""

        return any(
            visit_sequence(payload, branch_body=True)
            for key, payload in value.items()
            if key in GENERATION_KEYWORDS
        )

    def visit_separation_body(value: Any) -> bool:
        if not isinstance(value, dict) or {"model", "branch", "refit_params"} & set(value):
            return visit_sequence(value, branch_body=True)
        if not enter(value):
            return False
        try:
            return any(visit_sequence(body, branch_body=True) for body in value.values())
        finally:
            active_container_ids.remove(id(value))

    def visit_branch(value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            if not enter(value):
                return False
            try:
                return any(
                    visit_sequence(body, branch_body=True)
                    if isinstance(body, (list, tuple))
                    else visit_step(body, branch_body=True, branch_list_wrapper=True)
                    for body in value
                )
            finally:
                active_container_ids.remove(id(value))
        if not isinstance(value, dict):
            return visit_sequence(value, branch_body=True)
        if not enter(value):
            return False
        try:
            if any(key in GENERATION_KEYWORDS for key in value):
                return visit_generator(value)
            if _SEPARATION_BRANCH_KEYS & set(value):
                return visit_separation_body(value.get("steps"))
            return any(
                visit_sequence(body, branch_body=True)
                for name, body in value.items()
                if isinstance(name, str) and not name.startswith("_") and name not in _BRANCH_CONFIG_KEYS
            )
        finally:
            active_container_ids.remove(id(value))

    return visit_sequence(steps)


def _is_supported_native_refit_params_noop(steps: list[Any]) -> bool:
    """Whether the sole legacy refit option is safe for native PLS to ignore.

    ``use_all_partitions`` is a legacy controller no-op for exactly one plain,
    top-level :class:`~sklearn.cross_decomposition.PLSRegression` model step.
    Keep the proof deliberately narrow: cycles, aliases, bare estimators, a
    subclass, another model, another occurrence, or any payload variation
    remains on the fail-closed boundary.
    """

    from sklearn.cross_decomposition import PLSRegression

    model_step: dict[Any, Any] | None = None
    for step in steps:
        if isinstance(step, dict):
            if type(step) is not dict or model_step is not None:
                return False
            if set(step) - _REFIT_NOOP_MODEL_STEP_KEYS or {"model", "refit_params"} - set(step):
                return False
            if "name" in step and type(step["name"]) is not str:
                return False
            model_step = step
        elif not _is_safe_refit_noop_direct_step(step):
            return False

    if model_step is None:
        return False

    model = model_step["model"]
    if type(model) is not PLSRegression or "use_all_partitions" in model.get_params(deep=False):
        return False

    refit_params = model_step["refit_params"]
    if type(refit_params) is not dict or len(refit_params) != 1:
        return False
    key, value = next(iter(refit_params.items()))
    return type(key) is str and key == "use_all_partitions" and value is True


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

    from nirs4all.pipeline.dagml_bridge import is_grid_param_generator_spec, is_param_generator_spec

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
