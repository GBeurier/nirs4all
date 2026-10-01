"""Run-scoped CV weight initialization for the closed sklearn SGD profile.

Only public coefficient/intercept arrays cross from a completed, explicitly
selected CV fold to a fresh REFIT estimator. sklearn owns fitting and resets
its optimization counter on ``fit``. This module retains no training rows,
estimator objects, optimizer state, or cross-fold ranking implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from threading import RLock
from typing import Any

import numpy as np
from sklearn.linear_model import SGDRegressor

_FOLD = re.compile(r"fold(?:0|[1-9][0-9]*)\Z")
_BUDGET_PARAMS = frozenset({"max_iter", "tol", "warm_start"})
_SCHEMA = "nirs4all.cv-weight-transfer.v1"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"CV weight transfer {name} must be a nonempty string without NUL")
    return value


def _fold(value: Any) -> str:
    if not isinstance(value, str) or not _FOLD.fullmatch(value):
        raise ValueError("CV weight transfer requires an explicit native foldN selector; best, last and implicit selectors are unsupported")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WarmStartRequest:
    """An opt-in request naming the one completed native CV fold to retain."""

    fold_id: str

    def __post_init__(self) -> None:
        _fold(self.fold_id)


def parse_warm_start_request(
    refit_params: Mapping[str, Any], *, train_params: Mapping[str, Any] | None = None,
) -> WarmStartRequest | None:
    """Validate decoded public controls without changing cold-start behavior.

    Args:
        refit_params: Decoded per-model REFIT controls.
        train_params: Decoded CV controls. Ordinary constructor warm_start is
            preserved without transfer; transferred CV folds must stay cold.

    Returns:
        An explicit fold request, or None when weight transfer is disabled.
    """
    if not isinstance(refit_params, Mapping) or (train_params is not None and not isinstance(train_params, Mapping)):
        raise TypeError("CV weight transfer train/refit parameters must be mappings")
    train = train_params or {}
    if "warm_start_fold" in train:
        raise ValueError("CV weight transfer warm_start_fold belongs in refit_params only")
    enabled = refit_params.get("warm_start", False)
    if type(enabled) is not bool:
        raise TypeError("refit_params.warm_start must be a boolean")
    if not enabled:
        if "warm_start_fold" in refit_params:
            raise ValueError("refit_params.warm_start_fold requires warm_start=True")
        return None
    if "warm_start_fold" not in refit_params:
        raise ValueError("refit_params.warm_start=True requires an explicit warm_start_fold=foldN")
    if type(train.get("warm_start", False)) is not bool or train.get("warm_start", False):
        raise ValueError("CV weight transfer requires train_params.warm_start=False; CV folds must start independently")
    return WarmStartRequest(_fold(refit_params["warm_start_fold"]))


@dataclass(frozen=True)
class TransferIdentity:
    """Exact native owner identity; variant_label carries native variant_id."""

    run_id: str
    node_id: str
    controller_id: str
    variant_label: str

    def __post_init__(self) -> None:
        for name in ("run_id", "node_id", "controller_id", "variant_label"):
            _text(getattr(self, name), name)

    def as_dict(self) -> dict[str, str]:
        """Return JSON-native identity for archive provenance."""
        return {name: getattr(self, name) for name in ("run_id", "node_id", "controller_id", "variant_label")}


@dataclass(frozen=True)
class TransferInputContract:
    """Ordered, unchanged single-source and single-target representation."""

    source_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    x_dtype: str
    y_dtype: str

    def __post_init__(self) -> None:
        for name in ("source_names", "feature_names", "target_names"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"CV weight transfer {name} must be a nonempty ordered tuple")
            for value in values:
                _text(value, name)
            if len(set(values)) != len(values):
                raise ValueError(f"CV weight transfer {name} must contain distinct names")
        if len(self.source_names) != 1 or len(self.target_names) != 1:
            raise ValueError("CV weight transfer supports one numeric source and one complete target only")
        for name in ("x_dtype", "y_dtype"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"CV weight transfer {name} must be a dtype string")
            dtype = np.dtype(value)
            if dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("CV weight transfer requires unchanged native float32 or float64 inputs")
            object.__setattr__(self, name, dtype.str)

    def as_dict(self) -> dict[str, Any]:
        """Return representation metadata without rows or target values."""
        return {"source_names": list(self.source_names), "feature_names": list(self.feature_names),
                "target_names": list(self.target_names), "x_dtype": self.x_dtype, "y_dtype": self.y_dtype}


def validate_transfer_input(X: Any, y: Any, contract: TransferInputContract) -> None:
    """Check actual dense inputs against ordered metadata without coercing them."""
    if not isinstance(contract, TransferInputContract):
        raise TypeError("CV weight transfer requires a typed input contract")
    if np.ma.isMaskedArray(X) or np.ma.isMaskedArray(y):
        raise ValueError("CV weight transfer refuses masked features or partial targets")
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(contract.feature_names):
        raise ValueError("CV weight transfer requires a nonempty dense 2D input matching the feature width")
    if not isinstance(y, np.ndarray) or (y.ndim != 1 and not (y.ndim == 2 and y.shape[1] == 1)) or y.shape[0] != X.shape[0]:
        raise ValueError("CV weight transfer requires one complete target value per training row")
    if X.dtype.str != contract.x_dtype or y.dtype.str != contract.y_dtype:
        raise ValueError("CV weight transfer input dtype differs from its representation contract")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("CV weight transfer requires finite features and complete finite targets")


def _real(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"CV weight transfer SGD parameter {name} must be finite numeric data")
    result = float(value)
    if result < minimum or (strict and result == minimum):
        raise ValueError(f"CV weight transfer SGD parameter {name} is outside the supported range")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"CV weight transfer SGD parameter {name} must be an integer >= {minimum}")
    return int(value)


def validate_sgd_profile(model: Any, *, phase: str | None = None) -> dict[str, Any]:
    """Resolve the closed deterministic SGD capability without fitting.

    The exact public estimator type is required. Budgets max_iter/tol may
    change at REFIT; all other effective constructor options are identity.
    """
    if type(model) is not SGDRegressor:
        raise NotImplementedError("CV weight transfer supports only a plain sklearn SGDRegressor; Pipelines, subclasses, PLS and Ridge are unsupported")
    params = dict(model.get_params(deep=False))
    if params["loss"] != "squared_error" or params["penalty"] != "l2":
        raise NotImplementedError("CV weight transfer requires SGD loss='squared_error' and penalty='l2'")
    if params["average"] is not False or params["early_stopping"] is not False:
        raise NotImplementedError("CV weight transfer requires average=False and early_stopping=False; optimizer averaging state is not transferred")
    if params["learning_rate"] not in {"constant", "invscaling"}:
        raise NotImplementedError("CV weight transfer supports constant or invscaling SGD schedules only; fit resets the optimization counter")
    for name in ("fit_intercept", "shuffle", "warm_start"):
        if type(params[name]) is not bool:
            raise ValueError(f"CV weight transfer SGD parameter {name} must be a boolean")
    params["random_state"] = _integer(params["random_state"], "random_state")
    if params["random_state"] > np.iinfo(np.uint32).max:
        raise ValueError("CV weight transfer requires a fixed random_state in the uint32 seed range")
    for name in ("alpha", "power_t", "epsilon", "l1_ratio"):
        params[name] = _real(params[name], name)
    if params["l1_ratio"] > 1:
        raise ValueError("CV weight transfer SGD l1_ratio must be in [0, 1]")
    params["eta0"] = _real(params["eta0"], "eta0", strict=True)
    params["validation_fraction"] = _real(params["validation_fraction"], "validation_fraction", strict=True)
    if params["validation_fraction"] >= 1:
        raise ValueError("CV weight transfer SGD validation_fraction must be in (0, 1)")
    for name in ("max_iter", "n_iter_no_change"):
        params[name] = _integer(params[name], name, minimum=1)
    params["verbose"] = _integer(params["verbose"], "verbose")
    if params["tol"] is not None:
        params["tol"] = _real(params["tol"], "tol")
    if phase == "FIT_CV" and params["warm_start"]:
        raise ValueError("CV weight transfer CV estimators must use warm_start=False and start independently")
    if phase == "REFIT" and not params["warm_start"]:
        raise ValueError("CV weight transfer REFIT estimator must declare warm_start=True")
    if phase not in (None, "FIT_CV", "REFIT"):
        raise ValueError("CV weight transfer supports FIT_CV and REFIT phases only")
    return params


def _recipe(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in params.items() if name not in _BUDGET_PARAMS}


def _budget(params: dict[str, Any]) -> dict[str, Any]:
    return {name: params[name] for name in ("max_iter", "tol")}


def _weights(model: SGDRegressor, contract: TransferInputContract) -> tuple[np.ndarray, np.ndarray]:
    coefficients = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if not isinstance(coefficients, np.ndarray) or not isinstance(intercept, np.ndarray):
        raise ValueError("CV weight transfer requires a successfully fitted source SGDRegressor")
    if coefficients.shape != (len(contract.feature_names),) or intercept.shape != (1,) or getattr(model, "n_features_in_", None) != len(contract.feature_names):
        raise ValueError("CV weight transfer fitted coefficient/intercept dimensions differ from the feature contract")
    if coefficients.dtype.str != contract.x_dtype or intercept.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("CV weight transfer fitted weight dtype differs from the feature representation")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError("CV weight transfer refuses nonfinite fitted weights")
    return coefficients, intercept


@dataclass(frozen=True)
class _Snapshot:
    coefficients: np.ndarray
    intercept: np.ndarray
    recipe_json: str
    input_json: str
    budget_json: str
    byte_count: int


class CVWeightTransferStore:
    """Bounded run-local snapshots, consumed once and cleared by their owner.

    Args:
        run_id: Native execution scope; another candidate/run cannot use it.
        max_entries: Maximum simultaneously retained requested folds.
        max_bytes: Bound on retained weights and encoded identity metadata.
    """

    def __init__(self, run_id: str, *, max_entries: int = 64, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._run_id = _text(run_id, "run_id")
        self.max_entries = _integer(max_entries, "max_entries", minimum=1)
        self.max_bytes = _integer(max_bytes, "max_bytes", minimum=1)
        self._snapshots: dict[tuple[TransferIdentity, str], _Snapshot] = {}
        self._pending_bytes = 0
        self._closed = False
        self._lock = RLock()

    @property
    def run_id(self) -> str:
        """Immutable native run/candidate owner."""
        return self._run_id

    @property
    def pending_count(self) -> int:
        """Number of actual retained snapshots, for lifecycle evidence."""
        with self._lock:
            return len(self._snapshots)

    @property
    def pending_bytes(self) -> int:
        """Retained array/encoded metadata bytes, excluding interpreter overhead."""
        with self._lock:
            return self._pending_bytes

    def _owner(self, identity: TransferIdentity) -> None:
        if self._closed:
            raise RuntimeError("CV weight transfer store is closed")
        if not isinstance(identity, TransferIdentity) or identity.run_id != self.run_id:
            raise ValueError("CV weight transfer snapshot belongs to another run or candidate")

    def capture(
        self, request: WarmStartRequest, identity: TransferIdentity, fold_id: str,
        model: Any, input_contract: TransferInputContract,
    ) -> bool:
        """Capture only the requested completed CV fold; retain no estimator."""
        with self._lock:
            self._owner(identity)
            if not isinstance(request, WarmStartRequest) or not isinstance(input_contract, TransferInputContract):
                raise TypeError("CV weight transfer requires typed request and input contracts")
            _fold(fold_id)
            if fold_id != request.fold_id:
                return False
            params = validate_sgd_profile(model, phase="FIT_CV")
            key = (identity, request.fold_id)
            if key in self._snapshots:
                raise ValueError("CV weight transfer source fold was already captured for this node/variant")
            if len(self._snapshots) >= self.max_entries:
                raise ValueError("CV weight transfer snapshot entry limit exceeded")
            coefficients, intercept = _weights(model, input_contract)
            recipe_json = _json(_recipe(params))
            input_json = _json(input_contract.as_dict())
            budget_json = _json(_budget(params))
            byte_count = coefficients.nbytes + intercept.nbytes + len((recipe_json + input_json + budget_json + _json(identity.as_dict()) + request.fold_id).encode("utf-8"))
            if self._pending_bytes + byte_count > self.max_bytes:
                raise ValueError("CV weight transfer snapshot byte limit exceeded")
            coefficients = coefficients.copy(order="C")
            intercept = intercept.copy(order="C")
            coefficients.setflags(write=False)
            intercept.setflags(write=False)
            self._snapshots[key] = _Snapshot(coefficients, intercept, recipe_json, input_json, budget_json, byte_count)
            self._pending_bytes += byte_count
            return True

    def prepare_refit(
        self, request: WarmStartRequest, identity: TransferIdentity,
        model: Any, input_contract: TransferInputContract,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Consume an exact snapshot and return detached public fit initializers.

        The caller validates actual REFIT inputs, then calls sklearn fit once.
        The snapshot is released before fit, including when that fit fails.
        """
        with self._lock:
            self._owner(identity)
            if not isinstance(request, WarmStartRequest) or not isinstance(input_contract, TransferInputContract):
                raise TypeError("CV weight transfer requires typed request and input contracts")
            params = validate_sgd_profile(model, phase="REFIT")
            if hasattr(model, "coef_") or hasattr(model, "intercept_"):
                raise ValueError("CV weight transfer requires a fresh REFIT estimator, not an already fitted object")
            key = (identity, request.fold_id)
            snapshot = self._snapshots.get(key)
            if snapshot is None:
                raise ValueError("CV weight transfer requested fold has no captured state for this run/node/controller/variant; cold-start fallback is forbidden")
            recipe_json = _json(_recipe(params))
            if snapshot.recipe_json != recipe_json:
                raise ValueError("CV weight transfer effective SGD recipe differs from the captured fold; only max_iter/tol budgets may change")
            if snapshot.input_json != _json(input_contract.as_dict()):
                raise ValueError("CV weight transfer input source/features/dtype/target representation differs from the captured fold")
            fit_options = {"coef_init": snapshot.coefficients.copy(order="C"), "intercept_init": snapshot.intercept.copy(order="C")}
            provenance = {"schema": _SCHEMA, **identity.as_dict(), "source_fold_id": request.fold_id,
                          "estimator": "sklearn.linear_model.SGDRegressor", "recipe_fingerprint": _fingerprint(_recipe(params)),
                          "input_contract_fingerprint": _fingerprint(input_contract.as_dict()),
                          "coef_sha256": hashlib.sha256(snapshot.coefficients.tobytes(order="C")).hexdigest(),
                          "intercept_sha256": hashlib.sha256(snapshot.intercept.tobytes(order="C")).hexdigest(),
                          "weight_dtype": snapshot.coefficients.dtype.str, "coef_shape": list(snapshot.coefficients.shape),
                          "coef_dtype": snapshot.coefficients.dtype.str, "intercept_dtype": snapshot.intercept.dtype.str,
                          "intercept_shape": list(snapshot.intercept.shape), "cv_fit_budget": json.loads(snapshot.budget_json),
                          "refit_fit_budget": _budget(params), "optimization_counter_policy": "reset_by_fit",
                          "training_rows_retained": False, "initializers_detached": True}
            del self._snapshots[key]
            self._pending_bytes -= snapshot.byte_count
            return fit_options, provenance

    def clear(self) -> None:
        """Release every snapshot after success, cancellation or error."""
        with self._lock:
            self._snapshots.clear()
            self._pending_bytes = 0

    def close(self) -> None:
        """Idempotently release snapshots and refuse further use."""
        with self._lock:
            self.clear()
            self._closed = True

    def __enter__(self) -> CVWeightTransferStore:
        with self._lock:
            if self._closed:
                raise RuntimeError("CV weight transfer store is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
