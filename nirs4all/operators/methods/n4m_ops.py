"""Thin nirs4all operators dispatching to the ``nirs4all-methods`` engine.

The numerics live entirely in the portable C++ core (``libn4m``) and are
reached through the ``n4m`` ctypes binding. These wrappers own no numerical
logic: they only translate sklearn ``fit`` / ``transform`` / ``predict`` calls
into the binding's own sklearn-contract objects and surface JSON-serializable
parameters via ``get_params`` so both nirs4all engines can round-trip them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

try:
    from n4m.transform.scatter import SNV as _N4MSNV
except ImportError:  # pragma: no cover - older wheel layout fallback
    try:
        from n4m.sklearn.preprocessing import SNV as _N4MSNV
    except ImportError:
        _N4MSNV = None

try:
    from n4m.estimators.regression.latent import PLS as _N4MPLS
except ImportError:  # pragma: no cover - older wheel layout fallback
    try:
        from n4m.sklearn.native_sweeps import NativePLSRegressor as _N4MPLS
    except ImportError:
        _N4MPLS = None

METHODS_AVAILABLE = _N4MSNV is not None and _N4MPLS is not None

_MISSING_MSG = (
    "nirs4all-methods (the `n4m` binding) is not installed. Install the "
    "`nirs4all-methods` wheel to use the MethodsSNV / MethodsPLS operators."
)


class MethodsSNV(TransformerMixin, BaseEstimator):
    """Standard Normal Variate backed by the ``nirs4all-methods`` engine.

    Row-wise scatter correction computed by the portable ``libn4m`` core.
    Drop-in replacement for the pure-Python
    :class:`nirs4all.operators.transforms.StandardNormalVariate`, numerically
    matching it (and scikit-learn) to floating-point precision.

    Parameters
    ----------
    with_mean : bool, default=True
        Center each sample before scaling.
    with_std : bool, default=True
        Scale each sample to unit variance.
    ddof : int, default=0
        Delta degrees of freedom for the per-sample standard deviation.
    """

    _webapp_meta = {
        "category": "scatter-correction",
        "tier": "methods",
        "tags": ["scatter-correction", "snv", "normalization", "row-wise", "n4m"],
    }
    _stateless = True

    def __init__(self, with_mean: bool = True, with_std: bool = True, ddof: int = 0) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def _backend(self) -> Any:
        if _N4MSNV is None:
            raise ImportError(_MISSING_MSG)
        return _N4MSNV(
            with_mean=bool(self.with_mean),
            with_std=bool(self.with_std),
            ddof=int(self.ddof),
        )

    def fit(self, X: np.ndarray, y: Any = None) -> MethodsSNV:
        """No-op fit; SNV is stateless (row-wise)."""
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the SNV-normalised spectra from the native engine."""
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._backend().fit_transform(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"with_mean": self.with_mean, "with_std": self.with_std, "ddof": self.ddof}


class MethodsPLS(RegressorMixin, BaseEstimator):
    """PLS regression backed by the ``nirs4all-methods`` engine.

    Single-target moment-based PLS computed by the portable ``libn4m`` core.
    Numerically matches scikit-learn's
    :class:`sklearn.cross_decomposition.PLSRegression` to floating-point
    precision for a fixed component count.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent variables to fit.
    cv : int, default=5
        Cross-validation folds used by the native component-selection sweep.
        Use ``cv >= 2``; ``cv=1`` is rejected by the engine.
    scale_x : bool, default=True
        Scale features to unit variance (matches sklearn ``scale=True``).
    """

    _webapp_meta = {
        "category": "regression",
        "tier": "methods",
        "tags": ["pls", "regression", "latent-variable", "n4m"],
    }

    def __init__(self, n_components: int = 2, cv: int = 5, scale_x: bool = True) -> None:
        self.n_components = n_components
        self.cv = cv
        self.scale_x = scale_x

    def _backend(self) -> Any:
        if _N4MPLS is None:
            raise ImportError(_MISSING_MSG)
        # Pin a single candidate so the native CV sweep always selects the
        # requested component count (mirrors a fixed-component sklearn PLS).
        return _N4MPLS(
            pls_components=[int(self.n_components)],
            cv=int(self.cv),
            scale_x=bool(self.scale_x),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MethodsPLS:
        """Fit the native PLS model on ``(X, y)``.

        The native engine is single-target (PLS1); ``y`` must be 1D or a
        single-column 2D array.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(
                f"MethodsPLS is single-target (PLS1); got y with {y.shape[1]} columns."
            )
        y = y.reshape(-1)
        self.n_features_in_ = X.shape[1]
        self._model_ = self._backend().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target from fitted native PLS coefficients."""
        if not hasattr(self, "_model_"):
            raise RuntimeError("MethodsPLS instance is not fitted yet; call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._model_.predict(X)).reshape(-1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_components": self.n_components, "cv": self.cv, "scale_x": self.scale_x}
