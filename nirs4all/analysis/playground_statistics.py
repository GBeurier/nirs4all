"""Descriptive statistics for spectral exploration, independent of any UI.

The population standard deviation and linear NumPy percentiles preserve the
Studio 0.9.1 exploration contract. They are not model evaluation metrics.
Non-finite observations are reported, never silently replaced with zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def spectral_matrix(values: ArrayLike, *, name: str = "X", finite: bool = False) -> NDArray[np.float64]:
    """Validate a nonempty rectangular spectral matrix without altering rows.

    No application-specific cardinality limit is imposed here. Hosts must admit
    their input size before decoding; this validation does not bound allocation.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if finite and not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite observations")
    return matrix


def spectral_statistics(values: ArrayLike) -> dict[str, Any]:
    """Return full-axis statistics and explicit missing/non-finite diagnostics.

    Percentiles use the historical linear interpolation and ``std`` uses
    ``ddof=0``. NaN/Inf propagation is retained in numerical values; a transport
    may encode these as null, but must retain the accompanying diagnostics.
    Statistics are computed before any display-only wavelength decimation.
    """
    matrix = spectral_matrix(values)
    with np.errstate(invalid="ignore", over="ignore"):
        percentiles = np.percentile(matrix, [5, 25, 50, 75, 95], axis=0)
        result: dict[str, Any] = {
            "mean": np.mean(matrix, axis=0).tolist(),
            "std": np.std(matrix, axis=0).tolist(),
            "min": np.min(matrix, axis=0).tolist(),
            "max": np.max(matrix, axis=0).tolist(),
            **{key: row.tolist() for key, row in zip(
                ("p5", "p25", "p50", "p75", "p95"), percentiles, strict=True,
            )},
            "global": {
                "mean": float(np.mean(matrix)),
                "std": float(np.std(matrix)),
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "n_samples": matrix.shape[0],
                "n_features": matrix.shape[1],
            },
        }
    result["diagnostics"] = {
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "non_finite_feature_indices": np.flatnonzero(~np.isfinite(matrix).all(axis=0)).tolist(),
        "non_finite_policy": "propagate",
        "std_ddof": 0,
        "percentile_method": "linear",
    }
    return result


def distance_statistics(values: ArrayLike) -> dict[str, Any]:
    """Summarize distances, distinguishing empty/undefined from real zero.

    Empty arrays retain historical zero summaries and explicitly report count
    zero. Undefined distances propagate; they are not discarded from a mean.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("distances must be one-dimensional")
    quantiles = np.percentile(array, [50, 75, 90, 95]) if array.size else np.zeros(4)
    return {
        "mean": float(np.mean(array)) if array.size else 0.0,
        "std": float(np.std(array)) if array.size else 0.0,
        "min": float(np.min(array)) if array.size else 0.0,
        "max": float(np.max(array)) if array.size else 0.0,
        "quantiles": dict(zip(("50", "75", "90", "95"), quantiles.tolist(), strict=True)),
        "count": int(array.size),
        "undefined_count": int((~np.isfinite(array)).sum()),
    }
