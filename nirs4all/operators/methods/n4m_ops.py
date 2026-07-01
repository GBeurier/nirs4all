"""Thin nirs4all operators dispatching to the ``nirs4all-methods`` engine.

The numerics live entirely in the portable C++ core (``libn4m``) and are
reached through the ``n4m`` ctypes binding. These wrappers own no numerical
logic: they only translate sklearn ``fit`` / ``transform`` / ``predict`` calls
into the binding's own sklearn-contract objects and surface JSON-serializable
parameters via ``get_params`` so both nirs4all engines can round-trip them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


def _load_optional_class(primary_fqn: str, fallback_fqn: str) -> tuple[Any | None, BaseException | None]:
    last_error: BaseException | None = None
    for fqn in (primary_fqn, fallback_fqn):
        module_name, _, attr = fqn.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), attr), None
        except Exception as exc:  # noqa: BLE001 - optional binding boundary
            last_error = exc
    return None, last_error


_N4MSNV, _N4MSNV_IMPORT_ERROR = _load_optional_class(
    "n4m.transform.scatter.SNV",
    "n4m.sklearn.preprocessing.SNV",
)
_N4MPLS, _N4MPLS_IMPORT_ERROR = _load_optional_class(
    "n4m.estimators.regression.latent.PLS",
    "n4m.sklearn.native_sweeps.NativePLSRegressor",
)

METHODS_AVAILABLE = _N4MSNV is not None and _N4MPLS is not None

_MISSING_MSG = "nirs4all-methods (the `n4m` binding) is not installed or not loadable."
_MISSING_MITIGATION = (
    "Install a compatible `nirs4all-methods` wheel exposing `n4m.transform.scatter.SNV` "
    "and `n4m.estimators.regression.latent.PLS`; until then, use the existing Python/sklearn "
    "operators and keep `N4A_DAGML_METHODS_SNV` unset."
)


def _error_text(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    return f"{type(exc).__name__}: {exc}"


def methods_binding_status() -> dict[str, Any]:
    """Return the installed ``n4m`` binding status consumed by these operators.

    The result is intentionally JSON-serializable so tests, CLIs, and runtime
    diagnostics can report a useful blocker without importing or reimplementing
    any ``nirs4all-methods`` numerical logic.
    """
    status: dict[str, Any] = {
        "available": False,
        "snv_available": _N4MSNV is not None,
        "pls_available": _N4MPLS is not None,
        "abi_version": None,
        "library_path": None,
        "module_path": None,
        "message": "",
        "mitigation": _MISSING_MITIGATION,
        "snv_import_error": _error_text(_N4MSNV_IMPORT_ERROR),
        "pls_import_error": _error_text(_N4MPLS_IMPORT_ERROR),
    }

    try:
        n4m = importlib.import_module("n4m")
    except Exception as exc:  # noqa: BLE001 - optional binding boundary
        status["message"] = f"{_MISSING_MSG} import n4m failed: {type(exc).__name__}: {exc}. MethodsSNV and MethodsPLS require the binding."
        return status

    status["module_path"] = getattr(n4m, "__file__", None)

    try:
        abi_version = n4m.abi_version()
    except Exception as exc:  # noqa: BLE001 - ABI probe is part of the diagnostic
        status["message"] = f"{_MISSING_MSG} n4m.abi_version() failed: {type(exc).__name__}: {exc}. MethodsSNV and MethodsPLS require a loadable libn4m."
        return status
    if not abi_version:
        status["message"] = f"{_MISSING_MSG} n4m.abi_version() returned an empty value. MethodsSNV and MethodsPLS require a compatible libn4m."
        return status
    status["abi_version"] = list(abi_version) if isinstance(abi_version, tuple) else abi_version

    try:
        library_path = Path(n4m.library_path())
    except Exception as exc:  # noqa: BLE001 - library probe is part of the diagnostic
        status["message"] = f"{_MISSING_MSG} n4m.library_path() failed: {type(exc).__name__}: {exc}. MethodsSNV and MethodsPLS require a loadable libn4m."
        return status
    status["library_path"] = str(library_path)
    if not library_path.exists():
        status["message"] = f"{_MISSING_MSG} n4m.library_path() points to a missing file: {library_path}. MethodsSNV and MethodsPLS require a loadable libn4m."
        return status

    missing = []
    if _N4MSNV is None:
        missing.append("n4m.transform.scatter.SNV")
    if _N4MPLS is None:
        missing.append("n4m.estimators.regression.latent.PLS")
    if missing:
        status["message"] = f"{_MISSING_MSG} Missing binding surface(s): {', '.join(missing)}. nirs4all consumes these through MethodsSNV and MethodsPLS."
        return status

    status["available"] = True
    status["message"] = "nirs4all-methods (n4m binding) is loadable; nirs4all can consume MethodsSNV and MethodsPLS."
    status["mitigation"] = ""
    return status


def _missing_methods_message() -> str:
    status = methods_binding_status()
    if status["message"]:
        message = status["message"]
    else:
        message = _MISSING_MSG
    mitigation = status.get("mitigation")
    return f"{message} {mitigation}".strip()


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
            raise ImportError(_missing_methods_message())
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
            raise ImportError(_missing_methods_message())
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
