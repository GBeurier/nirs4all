"""Pre-fit operator output-shape inference.

Editor-time answer to "given this operator, these params and an input shape,
what shape comes out?" — without fitting anything. Fitted estimators expose
``n_features_out_`` only after ``fit``; UIs (e.g. the nirs4all-studio pipeline
editor) need the static answer while the user is still composing the pipeline.

The semantics live HERE, next to the operators, so adding or changing an
operator keeps its shape rule in the same repo (previously hand-encoded in the
studio HTTP layer — boundary violation PIPE-01 in its 2026-06-05 closeout).

:func:`infer_output_shape` returns ``None`` for unknown operators — callers
should treat that as "shape-preserving, not guaranteed".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "DIMENSION_BOUND_PARAMS",
    "infer_output_shape",
]

#: Parameters whose value is bounded by an input dimension. Maps parameter
#: name to the bounding dimension (``"features"`` or ``"samples"``).
DIMENSION_BOUND_PARAMS: dict[str, str] = {
    "n_components": "features",
    "n_splits": "samples",
    "window_length": "features",
    "start": "features",
    "end": "features",
    "n_features": "features",
    "target_points": "features",
}


def _identity(samples: int, features: int, params: Mapping[str, Any]) -> tuple[int, int]:
    return samples, features


def _components(default: int) -> _ShapeRule:
    def rule(samples: int, features: int, params: Mapping[str, Any]) -> tuple[int, int]:
        return samples, min(int(params.get("n_components", default)), features, samples)

    return rule


def _resample(samples: int, features: int, params: Mapping[str, Any]) -> tuple[int, int]:
    return samples, int(params.get("n_features", params.get("target_points", features)))


def _crop(samples: int, features: int, params: Mapping[str, Any]) -> tuple[int, int]:
    return samples, max(1, int(params.get("end", features)) - int(params.get("start", 0)))


def _wavelet(samples: int, features: int, params: Mapping[str, Any]) -> tuple[int, int]:
    return samples, features // (2 ** int(params.get("level", 1)))


_ShapeRule = Callable[[int, int, Mapping[str, Any]], tuple[int, int]]

_SHAPE_RULES: dict[str, _ShapeRule] = {
    # Shape-preserving preprocessing
    "StandardNormalVariate": _identity,
    "SNV": _identity,
    "MultiplicativeScatterCorrection": _identity,
    "MSC": _identity,
    "StandardScaler": _identity,
    "MinMaxScaler": _identity,
    "RobustScaler": _identity,
    "Normalize": _identity,
    "LogTransform": _identity,
    "Detrend": _identity,
    "Baseline": _identity,
    "ASLSBaseline": _identity,
    "AirPLS": _identity,
    "ArPLS": _identity,
    "SNIP": _identity,
    "Gaussian": _identity,
    "ReflectanceToAbsorbance": _identity,
    "ToAbsorbance": _identity,
    "FromAbsorbance": _identity,
    "FirstDerivative": _identity,
    "SecondDerivative": _identity,
    "SavitzkyGolay": _identity,
    # Component/latent-space reduction
    "PLSRegression": _components(10),
    "PCA": _components(0),  # default resolved below: full features
    "IKPLS": _components(10),
    "OPLS": _components(10),
    # Resampling / cropping
    "ResampleTransformer": _resample,
    "Resampler": _resample,
    "CropTransformer": _crop,
    # Wavelets
    "Wavelet": _wavelet,
    "Haar": _wavelet,
}


def infer_output_shape(
    operator_name: str,
    params: Mapping[str, Any] | None,
    samples: int,
    features: int,
) -> tuple[int, int] | None:
    """Infer an operator's output shape before fitting.

    Args:
        operator_name: Display/class name of the operator (e.g. ``"PCA"``,
            ``"SNV"``, ``"CropTransformer"``).
        params: Operator parameters as authored in the pipeline config.
        samples: Input sample count.
        features: Input feature count.

    Returns:
        ``(samples, features)`` after the operator, or ``None`` when the
        operator has no registered shape rule (callers should treat unknown
        operators as shape-preserving but unverified).
    """
    rule = _SHAPE_RULES.get(operator_name)
    if rule is None:
        return None
    params = params or {}
    if operator_name == "PCA" and "n_components" not in params:
        # PCA without n_components keeps min(features, samples) components.
        return samples, min(features, samples)
    return rule(samples, features, params)
