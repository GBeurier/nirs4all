"""Pipeline-step utilities for the dag-ml host backend.

Splitting the cross-validator out, parsing handled ``tag`` steps, applying sibling model
hyperparameters, flattening nested sub-pipelines, and expanding operator-level generators into
concrete pipelines — the step-shaping helpers the run paths share.
"""

from __future__ import annotations

from typing import Any

from .errors import DagMlUnsupported


def _needs_wavelength_injection(operator: Any) -> bool:
    """True when ``operator`` *requires* a ``wavelengths=`` injection the dag-ml X-chain cannot provide.

    The dag-ml node runner fits an X-transform with only ``(X, y)`` (a plain sklearn ``make_pipeline``),
    whereas the legacy ``TransformerMixinController`` extracts wavelengths from ``dataset.headers()`` and
    passes them to ``fit(..., wavelengths=...)``. Only operators that *hard-require* wavelengths — i.e.
    ``fit(X, y)`` *raises* without them — are unsupported and converted to a catchable fallback:

    * a :class:`SpectraTransformerMixin` whose ``_requires_wavelengths is True`` (strict); the ``"optional"``
      family (and feature-selection ops like CARS/MC-UVE, which merely *accept* a ``wavelengths`` kwarg and
      fall back to index space when it is absent) run natively at parity, so they are NOT flagged; and
    * a configured :class:`~nirs4all.operators.transforms.Resampler` (``target_wavelengths`` set), which
      raises ``Wavelengths must be provided to fit()``; an identity Resampler (no target grid) is a
      pass-through that fits without wavelengths, so it stays supported.

    The signature is *not* used as the trigger (CARS/MC-UVE declare a ``wavelengths`` param but do not
    require it) — only the explicit strict flag and the Resampler's configured-state contract are.
    """
    if getattr(operator, "_requires_wavelengths", False) is True:
        return True
    from nirs4all.operators.transforms.resampler import Resampler

    return isinstance(operator, Resampler) and operator.target_wavelengths is not None


def _is_fqn_importable(operator: Any) -> bool:
    """True when ``operator``'s class round-trips through the routing's FQN import (reconstructible class).

    The dag-ml runtime reconstructs every X-transform from its serialized ``module.QualName`` —
    :func:`~nirs4all.pipeline.dagml_bridge._step_to_dsl` emits the FQN and
    :func:`~nirs4all.pipeline.dagml.operator_routing._import_class` re-imports it at fit time. A
    locally-defined class (``__qualname__`` carries ``<locals>``) or one whose module is not importable
    cannot be re-imported, so routing would crash mid-run with an uncaught error. Mirror exactly what the
    runtime does (import the FQN, confirm it resolves to the SAME class) so a non-reconstructible operator
    is caught UP FRONT instead.
    """
    from nirs4all.pipeline.dagml.operator_routing import _import_class
    from nirs4all.pipeline.dagml_bridge import _qualname

    # Mirror ``_qualname``: a step may be a bare CLASS (e.g. ``StandardScaler``) or an INSTANCE; the
    # reconstructible class is the object itself when it is a class, else its type.
    cls = operator if isinstance(operator, type) else type(operator)
    if "<locals>" in cls.__qualname__:
        return False
    try:
        return _import_class(_qualname(operator)) is cls
    except (ImportError, AttributeError, ValueError, TypeError):
        return False


def _params_losslessly_serializable(operator: Any) -> bool:
    """True when ``operator``'s ``get_params()`` survive the routing's JSON round-trip with NO information loss.

    The runtime reconstructs an X-transform with ``cls(**json_params)`` where ``json_params`` come from
    :func:`~nirs4all.pipeline.dagml_bridge._json_safe_params` — ``json.dumps(get_params(), default=repr)``.
    The ``default=repr`` fallback stringifies any non-JSON value (a callable, a fitted object, an
    ``np.ufunc`` …) into its ``repr`` — e.g. ``FunctionTransformer(func=lambda x: x)`` becomes
    ``func="<function <lambda> at 0x…>"``, which the constructor then receives as a STRING, crashing
    uncaught in ``fit``. So a param set is reconstructible only when it serializes WITHOUT hitting that
    fallback: ``json.dumps(get_params())`` (no ``default``) must succeed. A bare CLASS step carries no
    instance params (the bridge emits ``{}`` for it), so it is trivially serializable.
    """
    import json

    if isinstance(operator, type):
        return True
    try:
        json.dumps(operator.get_params())
    except (TypeError, ValueError):
        return False
    return True


def _is_routable_transform(operator: Any) -> bool:
    """True when ``operator`` is a sklearn-contract transform the dag-ml X-chain can fit, apply, AND rebuild.

    The X-chain calls ``fit(X, y)`` then ``transform(X)`` on an instance the runtime RECONSTRUCTS from the
    serialized class + ``get_params()``. An operator is routable only when it satisfies all four:

    * has ``fit`` and ``transform`` (sklearn transform contract); and
    * exposes ``get_params`` (sklearn param contract — the routing serializes/re-applies params through it);
      a transform without it serializes empty params and the runtime would re-instantiate it with the bare
      constructor, silently dropping state — refuse it; and
    * its class is FQN-importable (:func:`_is_fqn_importable`) so the runtime can re-import it; and
    * its params are losslessly JSON-serializable (:func:`_params_losslessly_serializable`) so the runtime
      rebuilds it with the SAME values — a non-JSON param (a ``lambda``, a fitted object) would be passed
      as its ``repr`` string and crash in ``fit``.

    An operator missing any of these is handled only by a dedicated (non-sklearn) legacy controller the
    dag-ml path does not implement, or cannot be faithfully reconstructed, so it is unsupported.
    """
    return (
        callable(getattr(operator, "fit", None))
        and callable(getattr(operator, "transform", None))
        and callable(getattr(operator, "get_params", None))
        and _is_fqn_importable(operator)
        and _params_losslessly_serializable(operator)
    )


def _check_x_operator(operator: Any) -> None:
    """Raise a catchable :class:`DagMlUnsupported` for one X-side transform the runtime cannot run/rebuild.

    The single per-operator gate the top-level steps AND the nested ``concat_transform`` /
    ``feature_augmentation`` sub-transforms both pass through, so the wavelength + routability +
    reconstructibility checks are identical wherever a transform is fit/reconstructed.
    """
    if operator is None:
        return
    if _needs_wavelength_injection(operator):
        raise DagMlUnsupported(
            f"engine='dag-ml' does not inject wavelengths into fit(), but {type(operator).__name__} "
            "requires them (the dag-ml X-chain fits transforms with (X, y) only). Use the legacy engine."
        )
    if not _is_routable_transform(operator):
        raise DagMlUnsupported(
            f"engine='dag-ml' cannot route {type(operator).__name__} — it is not a reconstructible "
            "sklearn-contract transform (needs fit/transform/get_params, an importable class, and "
            "JSON-serializable params) and requires a custom controller the dag-ml path does not "
            "implement. Use the legacy engine."
        )


def _flatten_nested_operations(operations: Any) -> list[Any]:
    """Every transform instance inside a ``concat_transform``/``feature_augmentation`` ops spec, at ANY depth.

    Mirrors :func:`~nirs4all.operators.transforms.concat._build_operation` exactly: a ``list`` is a CHAIN
    whose members are each themselves an operation (recursed), so a transform nested inside a chain inside a
    chain (``[[A, [B]]]``) is reconstructed + fit by ``FeatureConcat`` just like a top-level one — and must
    be checked. ``None`` (the raw pass-through layer) lowers to sklearn ``"passthrough"`` and is skipped; a
    ``dict`` is a nested ``concat_transform`` (a 3D concat-of-concat) the bridge rejects on its own, so it is
    not a flat fittable transform here and is skipped.
    """
    flat: list[Any] = []
    for operation in operations:
        if isinstance(operation, list):
            flat.extend(_flatten_nested_operations(operation))
        elif operation is not None and not isinstance(operation, dict):
            flat.append(operation)
    return flat


def _nested_x_operators(step: dict[str, Any]) -> list[Any]:
    """The X-side transform instances NESTED inside a ``concat_transform`` / ``feature_augmentation`` step.

    ``FeatureConcat`` reconstructs and FITS each of these (the same serialize → import → ``cls(**params)``
    round-trip as a bare transform), so they need the SAME checks — but they ride inside a dict step the
    top-level loop skips. Recursively walks the supported list / chain (nested-list) forms here
    (:func:`_flatten_nested_operations`, matching :func:`~nirs4all.operators.transforms.concat._build_operation`):

    * ``{"concat_transform": [op, [chain...], ...]}`` or the ``{"concat_transform": {"operations": [...]}}``
      dict form; and
    * ``{"feature_augmentation": op}`` / ``{"feature_augmentation": [op, ...]}``.

    Unhandled shapes (a generator dict, a ``name``/``source_processing`` selector, an empty/None config)
    are left to the bridge's own fail-loud lowering; this only extracts the transform instances (at every
    nesting level) so a wavelength-requiring or non-reconstructible op nested among them is caught up front.
    """
    if "concat_transform" in step:
        config = step["concat_transform"]
        operations = config.get("operations") if isinstance(config, dict) else config
    else:  # feature_augmentation
        operations = step["feature_augmentation"]
    if operations is None:
        return []
    if not isinstance(operations, list):
        operations = [operations]
    return _flatten_nested_operations(operations)


def _assert_supported_operators(steps: list[Any]) -> None:
    """Reject UP FRONT the transform-side operators the dag-ml X-chain cannot run (catchable fallback).

    Inspects the X-side transform operators (never the model — a structurally valid model that fails
    numerically at fit, e.g. ``PLSRegression`` with too many components, is a REAL bug that must propagate,
    not a coverage gap). For each it converts the recognizable unsupported shapes to
    :class:`DagMlUnsupported` so :func:`run.run`'s fallback redirects them to the legacy engine:

    * a wavelength-requiring operator (:func:`_needs_wavelength_injection`); and
    * a NON-sklearn / non-reconstructible custom operator (:func:`_is_routable_transform` is false) — the
      dag-ml X-chain has no controller for it, or the runtime cannot rebuild it faithfully.

    Both the BARE transform steps AND the transforms NESTED inside ``concat_transform`` /
    ``feature_augmentation`` (which ``FeatureConcat`` reconstructs + fits, :func:`_nested_x_operators`) are
    checked, so an unsupported op anywhere — top-level or nested — raises a catchable error before any
    ``estimator.fit`` reaches the runtime. Applied to EVERY leaf/body lowerer (the simple/native/repetition
    paths AND each branch body, augmentation rest, and rep-fusion body).
    """
    for operator in steps:
        if isinstance(operator, dict):
            # A keyword dict the bridge handles structurally (model / y_processing / …) is validated by its
            # own lowering path; but concat_transform / feature_augmentation CARRY X-side transforms that
            # FeatureConcat fits — recurse into those.
            if "concat_transform" in operator or "feature_augmentation" in operator:
                for nested in _nested_x_operators(operator):
                    _check_x_operator(nested)
            continue
        _check_x_operator(operator)


def _supported_body_steps(steps: list[Any]) -> list[Any]:
    """Drop ``None`` no-ops and assert the remaining X-side operators are supported — for any body/leaf.

    The single chokepoint every branch body / sub-pipeline / leaf lowerer routes its operator steps
    through, so the top-level guarantees hold uniformly: a ``None`` (identity no-op) is dropped everywhere
    legacy skips it (NOT lowered to a ``builtins.NoneType`` node), and a wavelength-requiring or
    non-reconstructible transform anywhere raises a catchable :class:`DagMlUnsupported`. Returns the
    cleaned (``None``-free) step list the caller lowers.
    """
    cleaned = [step for step in steps if step is not None]
    _assert_supported_operators(cleaned)
    return cleaned


def _split_pipeline(pipeline: list[Any]) -> tuple[list[Any], Any]:
    """Separate the cross-validator step (the object exposing ``.split``) from the operator steps.

    A ``None`` step is a no-op / identity (the legacy executor treats it as a pass-through — the
    common ``{"_or_": [None, Scaler()]}`` sweep idiom uses ``None`` for the "no preprocessing"
    variant). The dag-ml bridge would lower it to a ``builtins.NoneType`` node that the runtime
    cannot instantiate, so drop it here: a dropped identity transform yields a numerically identical
    pipeline, exactly the legacy no-op semantic.
    """
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    steps = [step for step in pipeline if step is not splitter and step is not None]
    return steps, splitter


def _legacy_skips_refit(splitter: Any) -> bool:
    """Whether LEGACY would SKIP the standalone refit pass for this splitter (→ no ``(final, *)`` rows).

    The legacy refit gate is NOT "shuffle=False" — it is a SERIALIZATION artifact of
    ``execution.refit.executor.execute_simple_refit``: it reloads the winning config's ``expanded_steps``
    and calls ``_step_is_splitter`` on each, which recognizes ONLY a live splitter instance or a
    ``{"class": ...}`` dict — NOT a bare class-name STRING. ``serialize_component`` collapses a splitter
    with NO non-default params (``KFold()`` / ``KFold(n_splits=5)`` / ``KFold(shuffle=False)`` /
    ``ShuffleSplit()`` / ``ShuffleSplit(n_splits=10)`` / …) to that bare string, so legacy finds no
    splitter, logs "No cross-validation detected … Skipping refit", and emits NO refit ``(final, train)`` /
    ``(final, test)`` rows. Any non-default param (``KFold(n_splits=3)``, ``ShuffleSplit(n_splits=3,
    random_state=42)``) serializes to a dict → legacy refits.

    We reproduce the EXACT gate by reusing the SAME ``serialize_component`` (no hand-rolled heuristic):
    the refit is skipped iff the splitter serializes to a bare string. ``None`` (no splitter) is handled
    upstream — every CV+refit path requires a splitter — so it conservatively does NOT skip.
    """
    if splitter is None:
        return False
    from nirs4all.pipeline.config.component_serialization import serialize_component

    return isinstance(serialize_component(splitter), str)


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
    """The legacy display model_name for the model step (matches ``ModelIdentifierGenerator``).

    Legacy ``extract_core_name`` prefers a user-provided ``name`` on the model step over the class
    name (``name`` > ``function`` > ``class``), so a ``{"name": "RF_Finetuned", "model": RF()}`` step
    surfaces as ``"RF_Finetuned"`` — not ``"RandomForestRegressor"``. We mirror that exact priority
    here so a NAMED model (finetune cases, any explicit ``name``) carries the same model_name on the
    dag-ml engine; an unnamed step falls back to the model class name, byte-identical to legacy. A
    step-level ``_grid_`` sweep over a BARE model CLASS (``{"_grid_": …, "model": PLSRegression}``)
    keeps the class on the step, so the class name is read from the class itself — ``type(<class>)`` is
    the metaclass (``ABCMeta``), which would mis-name the model.
    """
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            name = step.get("name")
            if name:
                return str(name)
            model = step["model"]
            return model.__name__ if isinstance(model, type) else type(model).__name__
    return "model"


def _apply_plain_model_params(steps: list[Any]) -> list[Any]:
    """Apply only the PLAIN (non-generator) sibling hyperparameters to the model, keeping sweeps.

    The native path lowers param-level sweeps (``_range_``/``_log_range_`` siblings, a step-level
    ``_grid_``) to dag-ml ``generators``, so they must stay on the step dict; plain siblings (e.g.
    ``scale=False``) are set on a model clone, exactly like ``_apply_model_params`` but leaving the
    generator specs in place for the bridge to lower. A native step-level ``_grid_`` keyword is kept on
    the step (NOT a per-param sibling, so it is recognized by KEY, not :func:`is_param_generator_spec`).
    """
    from sklearn.base import clone

    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec, step_has_native_grid

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            # The step's native `_grid_` key (if any) is kept on the step for the bridge to lower; a
            # per-param `_range_`/`_log_range_` sibling is kept too. Everything else non-reserved is a
            # plain hyperparameter set on the model clone.
            native_grid = step_has_native_grid(step)

            def _is_native_generator_sibling(key: str, value: Any, _native_grid: bool = native_grid) -> bool:
                return (key == "_grid_" and _native_grid) or is_param_generator_spec(value)

            plain = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS and not _is_native_generator_sibling(key, value)}
            if plain:
                model = step["model"]
                # A class-model (e.g. ``PLSRegression`` not ``PLSRegression()``) — common with a step-level
                # ``_grid_`` over a bare class — must be instantiated before ``clone`` (``clone`` rejects a
                # class), mirroring :func:`_apply_model_params`. Without this a bare-class model carrying a
                # plain sibling param (``{"_grid_": …, "model": PLSRegression, "scale": False}``) crashes.
                if isinstance(model, type):
                    model = model()
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**plain)
                kept = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS or _is_native_generator_sibling(key, value)}
                kept["model"] = model
                step = kept
        out.append(step)
    return out
