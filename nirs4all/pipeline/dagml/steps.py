"""Pipeline-step utilities for the dag-ml host backend.

Splitting the cross-validator out, parsing handled ``tag`` steps, applying sibling model
hyperparameters, flattening nested sub-pipelines, and expanding operator-level generators into
concrete pipelines — the step-shaping helpers the run paths share.
"""

from __future__ import annotations

from typing import Any


def _split_pipeline(pipeline: list[Any]) -> tuple[list[Any], Any]:
    """Separate the cross-validator step (the object exposing ``.split``) from the operator steps."""
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    steps = [step for step in pipeline if step is not splitter]
    return steps, splitter


def _taggers_from_step(step: Any) -> list[tuple[str, Any]] | None:
    """Parse a handled ``{"tag": SampleFilter}`` step, else return ``None`` for bridge fail-loud."""
    if not isinstance(step, dict) or "tag" not in step:
        return None

    from nirs4all.controllers.data.tag import TagController

    try:
        taggers = TagController()._parse_taggers(step.get("tag", {}))  # noqa: SLF001 - reuse production parsing/name rules
    except (TypeError, ValueError):
        return None
    return taggers or None


# Keys on a step dict that are NOT model hyperparameters (mirrors StepParser.RESERVED_KEYWORDS).
_RESERVED_STEP_KEYS = frozenset({"model", "params", "metadata", "steps", "name", "finetune_params", "train_params", "refit_params", "fit_on_all", "force_layout", "na_policy", "fill_value", "y_processing"})


def _apply_model_params(steps: list[Any]) -> list[Any]:
    """Apply sibling hyperparameters to the model (e.g. `{"model": PLS(), "n_components": 9}`).

    Generators expand a param sweep into `{"model": M, "<param>": value}` steps; nirs4all applies the
    non-reserved siblings to the model via set_params. We do the same on a clone so concurrent
    variants do not share mutated state.
    """
    from sklearn.base import clone

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            params = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS}
            if params:
                model = step["model"]
                # A class-model (e.g. ``PLSRegression`` rather than ``PLSRegression()``) must be
                # instantiated before clone — ``clone`` rejects a class. The expansion path normally
                # instantiates it (:func:`_expand_operator_generators`); this guards the remaining
                # bare-class shape rather than crashing on ``clone(<class>)``.
                if isinstance(model, type):
                    model = model()
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**params)
                step = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS}
                step["model"] = model
        out.append(step)
    return out


def _flatten_steps(steps: list[Any]) -> list[Any]:
    """Flatten nested-list pipeline steps into a single top-level step list.

    Operator-level generators expand a stage into a SUB-PIPELINE of steps held inside a list
    (``_cartesian_`` builds ``[[A, B], splitter, model]`` and ``_or_``+``pick`` builds
    ``[[A, B], …]``). The legacy executor flattens such nested lists into consecutive steps; the
    dag-ml bridge lowers one step at a time and would otherwise lower the inner list to a
    ``builtins.list`` node (→ ``make_pipeline([], model)`` crash), so flatten here first.
    """
    out: list[Any] = []
    for step in steps:
        if isinstance(step, list):
            out.extend(_flatten_steps(step))
        else:
            out.append(step)
    return out


# Step keywords whose value is the operator a top-level param-keyed sweep targets (mirrors the
# generator core's ``_OPERATOR_WRAPPER_KEYS``). When the step carries a top-level generator keyword
# (``_range_``/``_grid_``/…) over such an operator, the operator must be a ``{"class": …}`` dict for
# ``expand_spec``'s ``_normalize_param_sweep``/``_normalize_param_grid`` to expand it.
_OPERATOR_WRAPPER_KEYS = ("model", "y_processing")


def _wrap_param_keyed_operator(step: Any) -> Any:
    """Wrap a bare-string operator into a ``{"class": …}`` dict for a top-level param-keyed sweep.

    The canonical generator dialect places the generator keyword at the TOP level of the step dict
    beside ``param`` (the attribute) and ``model`` (the operator), e.g.
    ``{"_range_": [5, 25, 5], "param": "n_components", "model": PLSRegression}``. After
    ``serialize_component`` the operator is a bare string (``"sklearn…PLSRegression"``), but
    ``expand_spec``'s sweep/grid normalizers only fire on a ``{"class": …}`` dict (or a top-level
    ``class`` key) — a bare string is left unexpanded (1 variant), then ``clone`` crashes on the
    class. Wrapping the string in ``{"class": …}`` routes the sweep through the same nested-form
    expansion the list-form already uses. Steps without a top-level generator are returned unchanged.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS

    if not isinstance(step, dict) or not (GENERATION_KEYWORDS & set(step)):
        return step
    wrapped = dict(step)
    for opkw in _OPERATOR_WRAPPER_KEYS:
        operator = wrapped.get(opkw)
        if isinstance(operator, str):
            wrapped[opkw] = {"class": operator}
    return wrapped


def _expand_operator_generators(pipeline: list[Any]) -> list[list[Any]]:
    """Expand operator-level generators into concrete, flat pipelines of live operator instances.

    Mirrors nirs4all's production expansion (``PipelineConfigs``): ``serialize_component`` the
    pipeline to its canonical form FIRST so a bare-class/instance operator and the param-keyed sweep
    dialect (``_range_``/``_log_range_``/``_grid_``/``_zip_``/``_sample_`` beside ``param``/``model``)
    normalize and expand identically to legacy; then ``deserialize_component`` each variant back to
    live instances and flatten any nested sub-pipeline lists (``_cartesian_``/``_or_``+``pick``). The
    result is what ``_run_concrete`` expects — a flat ``[transform…, splitter, model]`` list with no
    nested lists and no bare classes — so the bridge lowers it cleanly instead of crashing on a
    ``clone(<class>)`` or a ``builtins.list`` intermediate step.

    A generator-free pipeline is returned unchanged (one variant of the original live operators) —
    the serialize/deserialize round-trip is reserved for the generator path so the common
    transform+model shape keeps its exact operator instances.
    """
    from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords
    from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
    from nirs4all.pipeline.config.generator import expand_spec

    if not has_nested_generator_keywords(pipeline):
        return [pipeline]

    serialized = serialize_component(pipeline)
    normalized = [_wrap_param_keyed_operator(step) for step in serialized]
    return [_flatten_steps(deserialize_component(variant)) for variant in expand_spec(normalized)]


def _model_name(steps: list[Any]) -> str:
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            return type(step["model"]).__name__
    return "model"


def _apply_plain_model_params(steps: list[Any]) -> list[Any]:
    """Apply only the PLAIN (non-generator) sibling hyperparameters to the model, keeping sweeps.

    The native path lowers param-level sweeps (``_range_``/``_log_range_``/``_grid_``) to dag-ml
    ``generators``, so they must stay on the step dict; plain siblings (e.g. ``scale=False``) are
    set on a model clone, exactly like ``_apply_model_params`` but leaving the generator dicts in
    place for the bridge to lower.
    """
    from sklearn.base import clone

    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            plain = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS and not is_param_generator_spec(value)}
            if plain:
                model = step["model"]
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**plain)
                kept = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS or is_param_generator_spec(value)}
                kept["model"] = model
                step = kept
        out.append(step)
    return out
