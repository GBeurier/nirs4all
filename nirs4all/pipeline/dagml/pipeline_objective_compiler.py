"""Narrow nirs4all pipeline compiler for the native tuning objective seam.

This is not the full public ``run(tuning=...)`` compiler.  It accepts the
single-estimator and linear transformer→estimator shapes that the existing
``PipelineObjective`` can execute honestly today, and rejects broader nirs4all
pipeline syntax fail-closed.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult

from .pipeline_objective import PipelineObjective, PipelineObjectiveResult
from .training_contracts import tcv1_sha256
from .tuning_contracts import DagMLTuningSpec, OrderedSearchSpaceSpec, ParameterPatch, parse_tuning_spec


@dataclass(frozen=True)
class CompiledPipelineObjective:
    """A compiled objective plus its aligned fit data."""

    objective: PipelineObjective
    search_space: OrderedSearchSpaceSpec
    X: Any
    y: Any
    sample_ids: Any = None
    groups: Any = None
    metadata: Any = None

    def evaluate(self, params: Mapping[str, Any], *, trial_index: int = 0) -> PipelineObjectiveResult:
        """Evaluate one candidate against the compiled data payload."""

        return self.objective.evaluate(
            params,
            self.X,
            self.y,
            trial_index=trial_index,
            sample_ids=self.sample_ids,
            groups=self.groups,
            metadata=self.metadata,
        )

    def parameter_patches(self, params: Mapping[str, Any]) -> tuple[ParameterPatch, ...]:
        """Return canonical ordered patches for one candidate mapping."""

        return self.search_space.patches_from_mapping(params)

    def run_tuning(self, **kwargs: Any) -> Any:
        """Run optimizer-driving over the compiled objective.

        Keyword arguments are forwarded to
        :func:`nirs4all.pipeline.dagml.tuning_adapters.run_pipeline_objective_tuning`
        for internal orchestration options such as ``refit`` or
        ``workspace_path``.
        """

        from .tuning_adapters import run_pipeline_objective_tuning

        return run_pipeline_objective_tuning(
            self.objective,
            self.X,
            self.y,
            sample_ids=self.sample_ids,
            groups=self.groups,
            metadata=self.metadata,
            **kwargs,
        )


def compile_pipeline_objective(
    pipeline: Any,
    X: Any,
    y: Any,
    tuning: DagMLTuningSpec | Mapping[str, Any],
    *,
    score_extractor: Callable[[Any], float],
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    clone_estimator: bool = True,
) -> CompiledPipelineObjective:
    """Compile the currently supported nirs4all shape into ``PipelineObjective``.

    Supported input forms are:

    - an estimator instance with ``fit``;
    - ``[estimator]``;
    - ``[{"model": estimator}]`` or ``{"model": estimator}``.
    - ``{"steps": [...]}`` or ``{"pipeline": [...]}`` for the same supported
      step sequence;
    - a linear sequence of transformer instances followed by one estimator;
    - transformer mapping steps ``{"transform": transformer, "name": "..."}``
      followed by ``{"model": estimator}``;
    - explicit ``"sklearn.module.Class"`` string steps without constructor
      parameters.

    Splitters, branches, string model aliases, ``finetune_params`` and arbitrary
    step keywords remain outside this seam and raise explicit errors.
    """

    tuning_spec = tuning if isinstance(tuning, DagMLTuningSpec) else parse_tuning_spec(tuning)
    search_space = tuning_spec.ordered_search_space
    estimator = _compile_linear_estimator(pipeline)
    objective = PipelineObjective(
        estimator,
        tuning_spec,
        score_extractor=score_extractor,
        clone_estimator=clone_estimator,
    )
    return CompiledPipelineObjective(
        objective=objective,
        search_space=search_space,
        X=X,
        y=y,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
    )


def run_single_estimator_tuning_to_run_result(
    pipeline: Any,
    X: Any,
    y: Any,
    tuning: DagMLTuningSpec | Mapping[str, Any],
    *,
    score_extractor: Callable[[Any], float],
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    clone_estimator: bool = True,
    refit: bool = True,
    workspace_path: str | Path | None = None,
    workspace_name: str = "",
    workspace_tuning_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
    resume_tuning_result: Any | None = None,
    resume_tuning_id: str | None = None,
    per_dataset: Mapping[str, Any] | None = None,
    winner_x: Any | None = None,
    winner_y_true: Any | None = None,
    winner_score: float | None = None,
    winner_metric: str | None = None,
    winner_sample_ids: Sequence[Any] | None = None,
    winner_dataset_name: str = "tuning_winner",
    winner_model_name: str | None = None,
    winner_task_type: str = "regression",
    winner_metadata: Mapping[str, Any] | None = None,
) -> RunResult:
    """Run the narrow single-estimator tuning lane and return a ``RunResult``.

    This helper stitches together the current internal seams:
    ``compile_pipeline_objective()`` → optimizer/refit orchestration →
    ``project_objective_tuning_to_run_result()``.  It remains deliberately
    narrower than public ``run(tuning=...)`` and inherits the projection rule
    that winner predictions require explicit caller-provided score evidence.
    """

    compiled = compile_pipeline_objective(
        pipeline,
        X,
        y,
        tuning,
        score_extractor=score_extractor,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        clone_estimator=clone_estimator,
    )
    if resume_tuning_result is None:
        objective_result = compiled.run_tuning(
            refit=refit,
            workspace_path=workspace_path,
            workspace_name=workspace_name,
            workspace_tuning_id=workspace_tuning_id,
            workspace_metadata=workspace_metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
        )
    else:
        from .tuning_adapters import ObjectiveTuningRunResult

        refit_estimator = None
        if refit:
            refit_estimator = compiled.objective.refit_best(
                resume_tuning_result,
                compiled.X,
                compiled.y,
                sample_ids=compiled.sample_ids,
                groups=compiled.groups,
                metadata=compiled.metadata,
            )
        objective_result = ObjectiveTuningRunResult(
            tuning_result=resume_tuning_result,
            refit_estimator=refit_estimator,
            tuning_id=resume_tuning_id or workspace_tuning_id,
        )

    from .tuning_projection import project_objective_tuning_to_run_result

    return project_objective_tuning_to_run_result(
        objective_result,
        per_dataset=per_dataset,
        winner_x=winner_x,
        winner_y_true=winner_y_true,
        winner_score=winner_score,
        winner_metric=winner_metric,
        winner_sample_ids=winner_sample_ids,
        winner_dataset_name=winner_dataset_name,
        winner_model_name=winner_model_name,
        winner_task_type=winner_task_type,
        winner_metadata=winner_metadata,
    )


class _LinearPipelineEstimator:
    """Minimal sklearn-cloneable linear transformer→estimator wrapper."""

    def __init__(self, steps: Sequence[tuple[str, Any]]) -> None:
        self.steps = tuple(steps)
        self._assign_step_attributes()

    def _assign_step_attributes(self) -> None:
        for name, step in self.steps:
            setattr(self, name, step)
        self.model = self.steps[-1][1]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"steps": self.steps}
        if deep:
            for name, step in self.steps:
                params[name] = step
                if hasattr(step, "get_params"):
                    for key, value in step.get_params(deep=True).items():
                        params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> _LinearPipelineEstimator:
        if "steps" in params:
            self.steps = tuple(params.pop("steps"))
            self._assign_step_attributes()
        step_names = [name for name, _ in self.steps]
        for key, value in params.items():
            if "__" in key:
                name, nested_key = key.split("__", 1)
                step = getattr(self, name)
                if not hasattr(step, "set_params"):
                    raise ValueError(f"step {name!r} does not support set_params")
                step.set_params(**{nested_key: value})
            elif key in step_names:
                index = step_names.index(key)
                if index == len(self.steps) - 1:
                    raise ValueError("linear run(tuning=...) does not support replacing the final model step")
                replacement = _normalize_preprocessing_replacement(value, name=key)
                updated_steps = list(self.steps)
                updated_steps[index] = (key, replacement)
                self.steps = tuple(updated_steps)
            else:
                setattr(self, key, value)
            step_names = [name for name, _ in self.steps]
        self._assign_step_attributes()
        return self

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> _LinearPipelineEstimator:
        X_current = X
        fitted_steps: list[tuple[str, Any]] = []
        for name, transformer in self.steps[:-1]:
            fitted_transformer = _fit_transformer(transformer, X_current, y)
            X_current = fitted_transformer.transform(X_current)
            fitted_steps.append((name, fitted_transformer))
        model_name, estimator = self.steps[-1]
        fit_kwargs = {}
        if sample_ids is not None:
            fit_kwargs["sample_ids"] = sample_ids
        if groups is not None:
            fit_kwargs["groups"] = groups
        if metadata is not None:
            fit_kwargs["metadata"] = metadata
        fitted_estimator = estimator.fit(X_current, y, **fit_kwargs) if fit_kwargs else estimator.fit(X_current, y)
        fitted_steps.append((model_name, fitted_estimator))
        self.steps = tuple(fitted_steps)
        self._assign_step_attributes()
        return self

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> Any:
        X_current = X
        for _, transformer in self.steps[:-1]:
            X_current = transformer.transform(X_current)
        kwargs: dict[str, Any] = {}
        accepted = _accepted_kwargs(self.model.predict)
        if sample_ids is not None and ("sample_ids" in accepted or "**" in accepted):
            kwargs["sample_ids"] = sample_ids
        if groups is not None and ("groups" in accepted or "**" in accepted):
            kwargs["groups"] = groups
        if metadata is not None and ("metadata" in accepted or "**" in accepted):
            kwargs["metadata"] = metadata
        if kwargs:
            return self.model.predict(X_current, **kwargs)
        return self.model.predict(X_current)


class _PassthroughTransformer:
    """Cloneable no-op transformer for explicit linear tuning passthrough steps."""

    def fit(self, X: Any, y: Any | None = None) -> _PassthroughTransformer:
        return self

    def transform(self, X: Any) -> Any:
        return X

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> _PassthroughTransformer:
        if params:
            raise ValueError(f"passthrough step does not support parameters: {sorted(params)}")
        return self


def _normalize_preprocessing_replacement(value: Any, *, name: str) -> Any:
    if _is_passthrough_step(value):
        return _PassthroughTransformer()
    if not hasattr(value, "transform"):
        raise TypeError(f"linear run(tuning=...) replacement for preprocessing step {name!r} must expose transform() or be passthrough")
    return value


def _fit_transformer(transformer: Any, X: Any, y: Any) -> Any:
    if not hasattr(transformer, "transform"):
        raise TypeError("linear run(tuning=...) preprocessing steps must expose transform()")
    if hasattr(transformer, "fit"):
        try:
            return transformer.fit(X, y)
        except TypeError:
            return transformer.fit(X)
    return transformer


def _accepted_kwargs(callable_obj: Any) -> set[str]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set()
    accepted: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepted.add("**")
        elif parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            accepted.add(parameter.name)
    return accepted


def _compile_linear_estimator(pipeline: Any) -> Any:
    pipeline = _unwrap_steps_mapping(pipeline)
    if _is_step_sequence(pipeline) and not _is_named_step_tuple(pipeline):
        if len(pipeline) < 1:
            raise ValueError("pipeline must contain at least one estimator step")
        steps = [_normalize_step(raw_step, index=index, is_final=index == len(pipeline) - 1) for index, raw_step in enumerate(pipeline)]
        _reject_duplicate_step_names(steps)
        if len(steps) == 1:
            if steps[0][0] == "model":
                return steps[0][1]
            return _LinearPipelineEstimator(steps)
        return _LinearPipelineEstimator(steps)
    name, estimator = _normalize_step(pipeline, index=0, is_final=True)
    if name == "model":
        return estimator
    return _LinearPipelineEstimator([(name, estimator)])


def _unwrap_steps_mapping(pipeline: Any) -> Any:
    if not isinstance(pipeline, Mapping) or ("steps" not in pipeline and "pipeline" not in pipeline):
        return pipeline
    if "steps" in pipeline and "pipeline" in pipeline:
        raise ValueError("compile_pipeline_objective pipeline mappings must use either 'steps' or 'pipeline', not both")
    unsupported = sorted(set(pipeline) - {"name", "pipeline", "steps"})
    if unsupported:
        raise NotImplementedError(f"compile_pipeline_objective pipeline mappings support only 'steps'/'pipeline' and 'name'; unsupported keys: {unsupported}")
    return pipeline["steps"] if "steps" in pipeline else pipeline["pipeline"]


def _reject_duplicate_step_names(steps: Sequence[tuple[str, Any]]) -> None:
    names = [name for name, _ in steps]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"linear run(tuning=...) step names must be unique; duplicates: {duplicates}")


def _normalize_step(step: Any, *, index: int, is_final: bool) -> tuple[str, Any]:
    name = "model" if is_final else f"step_{index}"

    if _is_named_step_tuple(step):
        raw_name, step = step
        name = _validate_step_name(str(raw_name), is_final=is_final)

    if not is_final and _is_passthrough_step(step):
        step = _PassthroughTransformer()
    elif isinstance(step, Mapping):
        if "class" in step:
            unsupported = sorted(set(step) - {"class", "name", "params"})
            if unsupported:
                raise NotImplementedError(f"linear run(tuning=...) class steps support only 'class', 'params' and 'name'; unsupported step keys: {unsupported}")
            raw_name = step.get("name")
            if raw_name is not None:
                if not isinstance(raw_name, str):
                    raise ValueError("linear run(tuning=...) step name must be a non-empty string")
                name = _validate_step_name(raw_name, is_final=is_final)
            step = _instantiate_class_step(step["class"], step.get("params"), is_final=is_final)
        if is_final:
            if isinstance(step, Mapping):
                unsupported = sorted(set(step) - {"model", "name", "params"})
                if unsupported:
                    raise NotImplementedError(
                        f"compile_pipeline_objective final step currently supports only {{'model': estimator_or_sklearn_path, 'params': optional_mapping, 'name': optional_name}}; unsupported step keys: {unsupported}"
                    )
                if "model" not in step:
                    raise ValueError("final pipeline mapping must contain a 'model' key")
                raw_name = step.get("name")
                if raw_name is not None:
                    if not isinstance(raw_name, str):
                        raise ValueError("linear run(tuning=...) model step name must be a non-empty string")
                    name = _validate_step_name(raw_name, is_final=True)
                params = step.get("params")
                step = step["model"]
                if params is not None:
                    if not isinstance(step, str):
                        raise NotImplementedError("linear run(tuning=...) final step params are supported only with explicit sklearn.* model import paths")
                    step = _instantiate_class_step(step, params, is_final=True)
        else:
            if isinstance(step, Mapping):
                unsupported = sorted(set(step) - {"name", "params", "transform"})
                if unsupported:
                    raise NotImplementedError(f"linear run(tuning=...) preprocessing step supports only 'transform', 'params' and 'name'; unsupported step keys: {unsupported}")
                if "transform" not in step:
                    raise NotImplementedError("linear run(tuning=...) preprocessing mappings must use {'transform': transformer}")
                raw_name = step.get("name")
                if raw_name is not None:
                    if not isinstance(raw_name, str):
                        raise ValueError("linear run(tuning=...) preprocessing step name must be a non-empty string other than 'model'")
                    name = _validate_step_name(raw_name, is_final=False)
                params = step.get("params")
                step = step["transform"]
                if params is not None:
                    if not isinstance(step, str):
                        raise NotImplementedError("linear run(tuning=...) preprocessing step params are supported only with explicit sklearn.* transform import paths")
                    step = _instantiate_class_step(step, params, is_final=False)

    if not is_final and _is_passthrough_step(step):
        step = _PassthroughTransformer()
    if isinstance(step, str):
        if step.strip().startswith("sklearn."):
            step = _instantiate_class_step(step, None, is_final=is_final)
        else:
            raise NotImplementedError("compile_pipeline_objective requires an estimator instance; string model aliases are not compiled yet")
    if is_final:
        if not hasattr(step, "fit") or not hasattr(step, "predict"):
            raise TypeError("compile_pipeline_objective requires a final estimator instance with fit() and predict()")
    elif not hasattr(step, "transform"):
        raise TypeError("linear run(tuning=...) preprocessing steps must expose transform()")
    return name, step


def _is_named_step_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


def _is_passthrough_step(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == "passthrough"
    if isinstance(value, Mapping):
        return set(value) == {"kind"} and value.get("kind") == "passthrough"
    return False


def _instantiate_class_step(class_path: Any, params: Any, *, is_final: bool) -> Any:
    if not isinstance(class_path, str) or not class_path.strip():
        raise ValueError("linear run(tuning=...) class step requires a non-empty import path")
    normalized = class_path.strip()
    if not normalized.startswith("sklearn.") or "." not in normalized:
        raise NotImplementedError("linear run(tuning=...) class steps currently support only explicit sklearn.* import paths")
    if params is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        kwargs = _normalize_constructor_params(params, f"linear run(tuning=...) {normalized}.params")
    else:
        raise TypeError("linear run(tuning=...) class step params must be a mapping")
    module_name, class_name = normalized.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"linear run(tuning=...) cannot import class step {normalized!r}") from exc
    try:
        return cls(**kwargs)
    except Exception as exc:
        step_role = "final model" if is_final else "preprocessing"
        raise ValueError(f"linear run(tuning=...) could not instantiate {step_role} class step {normalized!r}") from exc


def _normalize_constructor_params(params: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError(f"{label} keys must be canonical non-empty strings")
        if key in normalized:
            raise ValueError(f"{label} contains duplicate keys")
        try:
            tcv1_sha256(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{key}] must contain TCV1-compatible JSON-native values") from exc
        normalized[key] = value
    return normalized


def _validate_step_name(raw_name: str, *, is_final: bool) -> str:
    name = raw_name.strip()
    if not name or (not is_final and name == "model"):
        if is_final:
            raise ValueError("linear run(tuning=...) model step name must be a non-empty string")
        raise ValueError("linear run(tuning=...) preprocessing step name must be a non-empty string other than 'model'")
    return name


def _is_step_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["CompiledPipelineObjective", "compile_pipeline_objective", "run_single_estimator_tuning_to_run_result"]
