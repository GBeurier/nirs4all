"""Optional nirs4all-methods attribution backend for general ``explain``.

Selected with ``explainer_backend="methods"``. Methods owns every attribution
numeric (``n4m.explain``); this host adapter only chooses the explained
function, rows, background and feature groups, exports model-specific inputs
and records the result in the ``nirs4all.explanation.v1`` envelope.

Model-agnostic methods (exact, permutation, kernel, LIME, occlusion) call the
same complete captured predictor as the default ``shap`` backend: raw input
columns in, original target units or class probabilities out. Linear and tree
SHAP explain the terminal estimator itself, so they apply in raw input space
only when the captured predictor has no preprocessing; otherwise
``attribution_space="transformed_features"`` explains the terminal estimator
over the outputs of the fitted preprocessing chain. Every model-specific export
must reproduce the captured predictor before its attributions are returned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .explain_export import affine_coefficients, export_tree_ensemble, is_tree_ensemble
from .explanation_envelope import build_explanation_v1, describe_groups, feature_index_axis

#: Options that only the Methods backend accepts.
METHODS_OPTIONS = frozenset({"baseline", "group_width", "group_ranges", "n_permutations", "n_coalitions", "seed", "attribution_space"})
#: ``explainer_type`` values accepted with ``explainer_backend="methods"``.
METHODS_EXPLAINER_TYPES = ("auto", "permutation", "exact", "kernel", "lime", "occlusion", "linear", "tree")
#: Explainer types whose attributions do not sum to the prediction.
NON_ADDITIVE_EXPLAINER_TYPES = frozenset({"lime", "occlusion"})
#: Default contiguous group width of model-agnostic methods on inputs wider than this width.
MODEL_AGNOSTIC_GROUP_WIDTH = 20

_SESSION_METHODS = {"permutation": "permutation_shap", "exact": "exact_shap", "kernel": "kernel_shap", "lime": "lime", "occlusion": "occlusion"}
_BASELINES = ("background", "mean", "interpolate")
_ATTRIBUTION_SPACES = ("raw_input", "transformed_features")
_REQUIRED_METHODS = "nirs4all-methods >= 1.3.17 (explain ABI 2.23)"
_MAX_SEED = 2**53


@dataclass(frozen=True)
class MethodsExplainSettings:
    """Validated Methods-backend options.

    Attributes:
        explainer_type: Requested explainer (one of ``METHODS_EXPLAINER_TYPES``).
        baseline: Explicit perturbation baseline, or ``None`` for the default.
        group_width: Explicit fixed group width, or ``None``.
        group_ranges: Explicit half-open feature-index ranges, or ``None``.
        n_permutations: Explicit antithetic permutation pairs, or ``None``.
        n_coalitions: Explicit sampled coalitions (``0`` = native auto), or ``None``.
        seed: Seed of background subsampling and sampled explainers.
        attribution_space: ``raw_input`` or ``transformed_features``.
    """

    explainer_type: str
    baseline: str | None
    group_width: int | None
    group_ranges: tuple[tuple[int, int], ...] | None
    n_permutations: int | None
    n_coalitions: int | None
    seed: int
    attribution_space: str


@dataclass(frozen=True)
class MethodsExplanation:
    """Attributions of one selected output computed by the Methods backend.

    Attributes:
        kind: Attribution kind actually computed (for example ``tree_shap``).
        values: ``(rows, groups)`` attributions of the selected output.
        base_value: Expected value of the selected output; a per-row array
            when the baseline depends on the explained row.
        labels: Unique group labels.
        explained_names: Names of the explained features.
        explained_rows: Explained rows in the explained feature space.
        group_rows: Per-group mean of ``explained_rows`` (for figures).
        attribution_space: ``raw_input`` or ``transformed_features``.
        preprocessing: Class names of the fitted preprocessing steps applied
            before explanation in ``transformed_features`` space.
        envelope: The ``nirs4all.explanation.v1`` envelope.
    """

    kind: str
    values: np.ndarray
    base_value: float | np.ndarray
    labels: list[str]
    explained_names: list[str]
    explained_rows: np.ndarray
    group_rows: np.ndarray
    attribution_space: str
    preprocessing: list[str]
    envelope: dict[str, Any]


@dataclass(frozen=True)
class _CaptureParts:
    separable: bool
    is_pipeline: bool
    chain: tuple[Any, ...]
    terminal: Any
    inverses: tuple[Any, ...]
    affine_inverses: tuple[tuple[np.ndarray, np.ndarray], ...] | None


def _integer(name: str, value: Any, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def parse_methods_options(explainer_type: str, options: dict[str, Any]) -> MethodsExplainSettings:
    """Validate the Methods-backend options of one explain request.

    Args:
        explainer_type: Requested ``explainer_type``.
        options: General explain options.

    Returns:
        The validated settings.

    Raises:
        ValueError: If an option is invalid or does not apply to the requested explainer.
    """
    if explainer_type not in METHODS_EXPLAINER_TYPES:
        raise ValueError(f"explainer_type={explainer_type!r} is not available with explainer_backend='methods'; choose one of {list(METHODS_EXPLAINER_TYPES)}")
    baseline = options.get("baseline")
    if baseline is not None and baseline not in _BASELINES:
        raise ValueError(f"baseline must be one of {list(_BASELINES)}")
    if baseline is not None and explainer_type == "tree":
        raise ValueError("Path-dependent TreeSHAP uses no baseline; remove baseline or choose a model-agnostic explainer_type")
    if baseline == "interpolate" and explainer_type == "linear":
        raise ValueError("Linear SHAP is interventional over the background mean; baseline='interpolate' requires a model-agnostic explainer_type")
    n_permutations = options.get("n_permutations")
    if n_permutations is not None:
        _integer("n_permutations", n_permutations, 1)
        if explainer_type not in ("auto", "permutation"):
            raise ValueError("n_permutations applies to explainer_type='permutation' only")
    n_coalitions = options.get("n_coalitions")
    if n_coalitions is not None:
        _integer("n_coalitions", n_coalitions, 0)
        if explainer_type not in ("kernel", "lime"):
            raise ValueError("n_coalitions applies to explainer_type='kernel' or 'lime' only")
    seed = _integer("seed", options.get("seed", 0), 0)
    if seed >= _MAX_SEED:
        raise ValueError("seed must be smaller than 2**53 to reproduce across Python and JavaScript hosts")
    width = options.get("group_width")
    ranges = options.get("group_ranges")
    if width is not None and ranges is not None:
        raise ValueError("group_width and group_ranges are mutually exclusive")
    if width is not None:
        _integer("group_width", width, 1)
    if ranges is not None:
        if (isinstance(ranges, (str, bytes)) or not isinstance(ranges, Sequence) or not ranges
                or any(isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2 or any(type(bound) is not int for bound in item) for item in ranges)):
            raise ValueError("group_ranges must be a nonempty list of (start, end) half-open feature-index pairs")
        ranges = tuple((item[0], item[1]) for item in ranges)
    space = options.get("attribution_space", "raw_input")
    if space not in _ATTRIBUTION_SPACES:
        raise ValueError(f"attribution_space must be one of {list(_ATTRIBUTION_SPACES)}")
    return MethodsExplainSettings(explainer_type, baseline, width, ranges, n_permutations, n_coalitions, seed, space)


def _load_binding() -> Any:
    try:
        from n4m import explain as n4m_explain
    except ImportError as error:
        raise ImportError(f"explainer_backend='methods' requires the n4m.explain binding of {_REQUIRED_METHODS}; the installed nirs4all-methods does not provide it") from error
    return n4m_explain


def _affine_inverse(inverse: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(scale, offset)`` of a target inverse known to be elementwise affine.

    ``inverse.inverse_transform(P) == P * scale + offset`` for the recognized
    fitted transformers; ``None`` for anything else (including label decoders
    and nonlinear transforms), which linear and tree SHAP then refuse.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler

    from nirs4all.pipeline.dagml.target_capture import CategoricalTargetTransform

    one, zero = np.ones(1), np.zeros(1)
    if isinstance(inverse, CategoricalTargetTransform) or (isinstance(inverse, FunctionTransformer) and inverse.inverse_func is None):
        return one, zero
    if isinstance(inverse, StandardScaler):
        scale = np.asarray(inverse.scale_, dtype=np.float64) if inverse.with_std else one
        return scale, np.asarray(inverse.mean_, dtype=np.float64) if inverse.with_mean else zero
    if isinstance(inverse, MinMaxScaler):
        scale = np.asarray(inverse.scale_, dtype=np.float64)
        return 1.0 / scale, -np.asarray(inverse.min_, dtype=np.float64) / scale
    if isinstance(inverse, MaxAbsScaler):
        return np.asarray(inverse.scale_, dtype=np.float64), zero
    if isinstance(inverse, Pipeline):
        # A pipeline inverts its steps in reverse order.
        total_scale, total_offset = one, zero
        for _, step in reversed(inverse.steps):
            if step is None or (isinstance(step, str) and step == "passthrough"):
                continue
            affine = _affine_inverse(step)
            if affine is None:
                return None
            total_scale, total_offset = total_scale * affine[0], total_offset * affine[0] + affine[1]
        return total_scale, total_offset
    return None


def _decompose(captured: Any, target_inverse: Any) -> _CaptureParts:
    from sklearn.pipeline import Pipeline

    from .result import _DagmlExportedModel

    inverses = []
    inner = captured
    separable = True
    if isinstance(captured, _DagmlExportedModel):
        # Named Torch and late-partial captures validate their REFIT anchor
        # around every prediction, so their estimator is never called directly.
        separable = not (hasattr(captured, "named_refit_artifact") or hasattr(captured, "late_partial_refit_artifact"))
        inner = captured.estimator
        if captured.y_transform is not None:
            inverses.append(captured.y_transform)
    if target_inverse is not None:
        inverses.append(target_inverse)
    affine = [_affine_inverse(inverse) for inverse in inverses]
    affine_inverses = None if any(item is None for item in affine) else tuple(item for item in affine if item is not None)
    if isinstance(inner, Pipeline):
        steps = [step for _, step in inner.steps]
        chain = tuple(step for step in steps[:-1] if step is not None and not (isinstance(step, str) and step == "passthrough"))
        return _CaptureParts(separable, True, chain, steps[-1], tuple(inverses), affine_inverses)
    return _CaptureParts(separable, False, (), inner, tuple(inverses), affine_inverses)


def _apply_inverses(inverses: Sequence[Any], values: Any) -> np.ndarray:
    rows = len(values)
    output = np.asarray(values, dtype=np.float64).reshape(rows, -1)
    for inverse in inverses:
        output = np.asarray(inverse.inverse_transform(output), dtype=np.float64).reshape(rows, -1)
    return output


def _transform(chain: Sequence[Any], values: np.ndarray) -> np.ndarray:
    for step in chain:
        values = step.transform(values)
    return np.asarray(values, dtype=np.float64)


def _require_reproduces(reconstructed: np.ndarray, reference: np.ndarray, what: str) -> None:
    deviation = float(np.max(np.abs(reconstructed - reference)))
    if deviation > 1e-6 * max(1.0, float(np.max(np.abs(reference)))):
        raise ValueError(f"{what} does not reproduce the captured predictor (max deviation {deviation:.3g}); its attributions would not explain this model")


def _model_specific_refusal(kind: str, parts: _CaptureParts, probabilities: bool, transformed: bool) -> str | None:
    if not parts.separable:
        return "the captured estimator is validated against its REFIT anchor and cannot be called separately"
    if parts.chain and not transformed:
        return "the captured predictor applies preprocessing before its estimator; use attribution_space='transformed_features' or a model-agnostic explainer_type"
    if parts.affine_inverses is None:
        return "the captured target inverse is not a recognized affine transform (StandardScaler, MinMaxScaler, MaxAbsScaler or identity)"
    if kind == "tree" and not is_tree_ensemble(parts.terminal, probabilities=probabilities):
        return f"{type(parts.terminal).__name__} is not a supported {'class-probability' if probabilities else 'regression'} tree ensemble"
    if kind == "linear" and probabilities:
        return "class probabilities are not an affine function of the inputs"
    if kind == "linear" and getattr(parts.terminal, "coef_", None) is None:
        return f"{type(parts.terminal).__name__} exposes no linear coefficients"
    return None


def _select_kind(settings: MethodsExplainSettings, parts: _CaptureParts, probabilities: bool, transformed: bool) -> str:
    requested = settings.explainer_type
    if requested in ("tree", "linear"):
        refusal = _model_specific_refusal(requested, parts, probabilities, transformed)
        if refusal is not None:
            raise ValueError(f"explainer_type={requested!r} cannot explain this captured predictor: {refusal}")
        return requested
    if requested != "auto":
        return requested
    # Auto considers only kinds compatible with the options set explicitly.
    candidates = []
    if settings.baseline is None and settings.n_permutations is None:
        candidates.append("tree")
    if settings.baseline in (None, "background", "mean") and settings.n_permutations is None:
        candidates.append("linear")
    return next((kind for kind in candidates if _model_specific_refusal(kind, parts, probabilities, transformed) is None), "permutation")


def _grouping(binding: Any, n_features: int, settings: MethodsExplainSettings, model_agnostic: bool) -> tuple[np.ndarray | None, int, list[str], str, int | None]:
    if settings.group_ranges is not None:
        feature_group, n_groups, range_of_group = binding.groups_from_ranges(n_features, list(settings.group_ranges))
        n_groups = int(n_groups)
        sources = ["ranges" if int(index) >= 0 else "remainder" for index in np.asarray(range_of_group)[:n_groups]]
        return np.asarray(feature_group), n_groups, sources, "ranges", None
    width = settings.group_width
    if width is None and model_agnostic and n_features > MODEL_AGNOSTIC_GROUP_WIDTH:
        width = MODEL_AGNOSTIC_GROUP_WIDTH
    if width is None or width == 1:
        return None, n_features, ["per_feature"] * n_features, "per_feature", width
    feature_group, n_groups = binding.groups_fixed_width(n_features, width)
    return np.asarray(feature_group), int(n_groups), ["fixed_width"] * int(n_groups), "fixed_width", width


def explain_with_methods(
    *,
    function: Callable[[np.ndarray], Any],
    probabilities: bool,
    captured: Any,
    target_inverse: Any,
    X: np.ndarray,
    rows: Sequence[int],
    names: Sequence[str],
    axis: dict[str, Any],
    output_index: int,
    output_label: str | None,
    background_samples: int,
    settings: MethodsExplainSettings,
) -> MethodsExplanation:
    """Explain one captured predictor with the nirs4all-methods attribution kernels.

    Args:
        function: Complete captured predictor (raw input rows to every output column).
        probabilities: Whether ``function`` returns class probabilities.
        captured: Captured model object (used for model-specific exports and
            ``transformed_features`` space).
        target_inverse: Artifact-level target inverse applied by ``function``.
        X: Every supplied raw input row; background rows are drawn from it.
        rows: Indices in ``X`` of the explained rows.
        names: Raw input feature names.
        axis: Recorded axis of the raw input features.
        output_index: Selected output column.
        output_label: Optional label of the selected output.
        background_samples: Maximum background rows, subsampled without
            replacement with ``settings.seed``.
        settings: Validated Methods options.

    Returns:
        The attributions of the selected output and the complete envelope.

    Raises:
        ImportError: If the installed nirs4all-methods has no ``n4m.explain`` binding.
        ValueError: If the request cannot be explained faithfully.
    """
    binding = _load_binding()
    X = np.asarray(X, dtype=np.float64)
    raw_rows = X[np.asarray(rows, dtype=np.int64)]
    background_index = np.sort(np.random.default_rng(settings.seed).choice(len(X), size=min(background_samples, len(X)), replace=False))
    parts = _decompose(captured, target_inverse)

    def raw_output(values: np.ndarray) -> np.ndarray:
        return np.asarray(function(values), dtype=np.float64).reshape(len(values), -1)

    def terminal_output(values: np.ndarray) -> np.ndarray:
        return _apply_inverses(parts.inverses, parts.terminal.predict_proba(values) if probabilities else parts.terminal.predict(values))

    reference = raw_output(raw_rows)
    transformed = settings.attribution_space == "transformed_features"
    explained_output: Callable[[np.ndarray], np.ndarray] = terminal_output if transformed else raw_output
    if transformed:
        if not parts.separable or not parts.is_pipeline:
            raise ValueError(f"attribution_space='transformed_features' requires a captured scikit-learn Pipeline with a separable terminal estimator; this capture is {type(parts.terminal).__name__}")
        explained_rows = _transform(parts.chain, raw_rows)
        background = _transform(parts.chain, X[background_index])
        _require_reproduces(explained_output(explained_rows), reference, "The fitted preprocessing chain followed by the terminal estimator")
        explained_names = [f"transformed_{index}" for index in range(explained_rows.shape[1])]
        explained_axis = feature_index_axis(explained_rows.shape[1])
    else:
        explained_rows, background = raw_rows, X[background_index]
        explained_names, explained_axis = list(names), axis

    kind = _select_kind(settings, parts, probabilities, transformed)
    model_agnostic = kind in _SESSION_METHODS
    feature_group, n_groups, sources, grouping, width = _grouping(binding, len(explained_names), settings, model_agnostic)
    group_arguments = {"feature_group": feature_group, "n_groups": n_groups if feature_group is not None else None}
    if model_agnostic:
        baseline: str | None = settings.baseline or "background"
        result = binding.explain(
            lambda values: explained_output(np.asarray(values, dtype=np.float64)), explained_rows,
            background=None if baseline == "interpolate" else background, method=_SESSION_METHODS[kind], baseline=baseline,
            n_outputs=reference.shape[1], n_permutations=settings.n_permutations or 10, n_coalitions=settings.n_coalitions or 0,
            seed=settings.seed, **group_arguments,
        )
        attributions, expected, predictions, std_error = result.attributions, result.expected, result.predictions, result.std_error
        additivity_error = result.additivity_max_abs_error
        n_background = 0 if baseline == "interpolate" else len(background)
    else:
        if kind == "linear":
            baseline = settings.baseline or "background"
            coefficients, intercept = affine_coefficients(parts.terminal, np.vstack([explained_rows, background]))
            result = binding.linear_shap(explained_rows, coefficients, reference=background.mean(axis=0), intercept=intercept, **group_arguments)
            n_background = len(background)
        else:
            baseline = None
            ensemble = export_tree_ensemble(parts.terminal, probabilities=probabilities)
            # Trees compare inputs in their own precision; explain exactly those rows.
            rounded = explained_rows.astype(ensemble.input_dtype).astype(np.float64)
            result = binding.tree_shap(rounded, binding.TreeEnsemble(**ensemble.ensemble_fields()), **group_arguments)
            n_background = 0
        # Fold the recognized elementwise-affine target inverses exactly: an
        # affine map of an additive explanation scales each attribution.
        assert parts.affine_inverses is not None  # refused by _model_specific_refusal otherwise
        scale, offset = np.ones(1), np.zeros(1)
        for step_scale, step_offset in parts.affine_inverses:
            scale, offset = scale * step_scale, offset * step_scale + step_offset
        attributions = result.attributions * scale
        expected = result.expected * scale + offset
        predictions = result.predictions * scale + offset
        _require_reproduces(predictions, reference, f"The exported {kind} model of {type(parts.terminal).__name__}")
        additivity_error = float(np.max(np.abs(attributions.sum(axis=1) + expected - predictions)))
        std_error = None

    groups = describe_groups(explained_names, explained_axis, feature_group, n_groups, sources)
    labels = [group["label"] for group in groups]
    group_rows = explained_rows if feature_group is None else np.column_stack(
        [explained_rows[:, group["feature_start"]:group["feature_end"]].mean(axis=1) for group in groups])
    selected_expected = np.asarray(expected, dtype=np.float64)[:, output_index]
    constant = float(np.ptp(selected_expected)) <= 1e-12 * max(1.0, float(np.max(np.abs(selected_expected))))
    preprocessing = [type(step).__name__ for step in parts.chain] if transformed else []
    envelope_settings = {
        **result.settings,
        "explainer_type_requested": settings.explainer_type, "baseline": baseline, "n_background": n_background,
        "background_selection": "seeded_without_replacement", "grouping": grouping, "group_width": width, "n_groups": n_groups,
        "seed": settings.seed, "evaluations": int(result.evaluations), "preprocessing": " > ".join(preprocessing) or None,
    }
    envelope = build_explanation_v1(
        kind=result.kind, engine=result.engine, exactness=result.exactness, additive=result.additive, group_aggregation=result.group_aggregation,
        attribution_space=settings.attribution_space,
        output={"index": output_index, "label": output_label, "space": "class_probability" if probabilities else "target"},
        axis=explained_axis, feature_names=explained_names, groups=groups, sample_indices=rows, attributions=attributions,
        std_error=std_error, expected=expected, prediction=predictions, additivity_max_abs_error=additivity_error, settings=envelope_settings,
    )
    return MethodsExplanation(
        kind=result.kind, values=np.asarray(attributions, dtype=np.float64)[:, :, output_index],
        base_value=float(selected_expected[0]) if constant else selected_expected.copy(), labels=labels, explained_names=explained_names,
        explained_rows=explained_rows, group_rows=group_rows, attribution_space=settings.attribution_space, preprocessing=preprocessing, envelope=envelope,
    )
