"""Operator-Adaptive PLS-DA classifiers.

Class encoding:

    Y_ic = 1 / sqrt(pi_c) if y_i = c, else 0,

where `pi_c` is the empirical class prior on the training set. This balances
the rare classes' contribution to the cross-covariance, mirroring
class-balanced regression.

Pipeline:

1. One-hot class-balanced coding `Y_bal` of `y`.
2. Center `Y_bal` and apply the unified Operator-Adaptive PLS engine.
3. Transform training samples to latent scores `T = X Z`.
4. Fit `LogisticRegression(class_weight="balanced", max_iter=2000)` on `T`.
5. `predict_proba(X)` transforms `X` to latent scores and calls the logistic
   calibrator.
6. Fallback: temperature-scaled softmax of raw PLS scores fitted on training
   data only.

Selection criterion for classification defaults to inner-CV balanced log loss;
covariance is available for smoke-mode selection.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .banks import bank_by_name, fit_bank
from .diagnostics import RunDiagnostics
from .metrics import balanced_accuracy
from .nipals import (
    nipals_adjoint,
    nipals_materialized_fixed,
    nipals_materialized_per_component,
)
from .operators import LinearSpectralOperator
from .scorers import CriterionConfig
from .selection import _resolve_engine, select


def _class_balanced_encode(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return the class-balanced one-hot encoding of `y`.

    `Y_ic = 1 / sqrt(pi_c)` if `y_i == classes[c]`, else 0.
    """
    n = y.shape[0]
    K = classes.shape[0]
    Y = np.zeros((n, K))
    for c, cls in enumerate(classes):
        mask = y == cls
        prior = float(mask.sum()) / n
        if prior <= 0:
            continue
        Y[mask, c] = 1.0 / np.sqrt(prior)
    return Y


class _AOMPLSDABase(BaseEstimator, ClassifierMixin):
    """Common backbone for AOM/POP PLS-DA classifiers."""

    def __init__(
        self,
        n_components: Union[int, str] = "auto",
        max_components: int = 25,
        engine: str = "simpls_covariance",
        selection: str = "global",
        criterion: str = "cv",
        operator_bank: Union[str, Sequence[LinearSpectralOperator]] = "compact",
        orthogonalization: str = "auto",
        center: bool = True,
        scale: bool = False,
        cv: int = 5,
        random_state: int = 0,
        backend: str = "numpy",
    ) -> None:
        self.n_components = n_components
        self.max_components = max_components
        self.engine = engine
        self.selection = selection
        self.criterion = criterion
        self.operator_bank = operator_bank
        self.orthogonalization = orthogonalization
        self.center = center
        self.scale = scale
        self.cv = cv
        self.random_state = random_state
        self.backend = backend

    # ---------------------------------------------------------- helpers

    def _resolve_bank(self, p: int) -> List[LinearSpectralOperator]:
        if isinstance(self.operator_bank, str):
            return bank_by_name(self.operator_bank, p=p)
        return list(self.operator_bank)

    # ---------------------------------------------------------- fitting

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_AOMPLSDABase":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        Y_bal = _class_balanced_encode(y, self.classes_)
        n, p = X.shape
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = Y_bal.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(p)
            self.y_mean_ = np.zeros(Y_bal.shape[1])
        Xc = X - self.x_mean_
        Yc = Y_bal - self.y_mean_
        # Bank
        bank = self._resolve_bank(p)
        fit_bank(bank, X, y)
        # Component limit
        max_components = min(self.max_components, n - 1, p)
        if isinstance(self.n_components, int):
            n_request = min(self.n_components, max_components)
            auto_prefix = False
        else:
            n_request = max_components
            auto_prefix = True
        # Criterion: classification -> covariance proxy or per-component CV using
        # balanced log loss. We use the regression criterion with covariance
        # default for simplicity; "cv" runs an inner CV with classification fold scoring.
        criterion = CriterionConfig(
            kind=self.criterion,
            cv=self.cv,
            random_state=self.random_state,
            task="classification",
        )
        # Orthogonalisation resolution
        orth = self.orthogonalization
        if orth == "auto":
            if self.selection in ("none", "global"):
                orth = "transformed"
            else:
                orth = "original"
        # If criterion is CV we run a *classification-aware* selection by
        # delegating to a custom inner CV using balanced log loss after
        # logistic calibration on the latent scores. For simplicity in this
        # implementation, we use the regression-style CV on the encoded Y_bal,
        # which is a sound monotone surrogate.
        sel = select(
            Xc=Xc,
            yc=Yc,
            operators=bank,
            engine=self.engine,
            selection=self.selection,
            n_components_max=n_request,
            criterion=criterion,
            orthogonalization=orth,
            auto_prefix=auto_prefix,
        )
        res = sel.result
        self.x_weights_ = res.Z.copy()
        self.x_effective_weights_ = res.Z.copy()
        self.x_loadings_ = res.P.copy()
        self.y_loadings_ = res.Q.copy()
        self.x_scores_ = res.T.copy()
        self.selected_operators_ = sel.operator_names
        self.selected_operator_indices_ = sel.operator_indices
        self.operator_scores_ = sel.operator_scores
        self.n_components_ = res.n_components
        self.engine_ = self.engine
        self.selection_ = self.selection
        self.criterion_ = self.criterion
        self.orthogonalization_ = orth
        # Calibration: fit logistic regression on training latent scores.
        # Train logistic on (T, y).
        T_train = res.T
        self._calibrator_kind = "logistic"
        self._calibrator = None
        try:
            calib = LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=self.random_state
            )
            calib.fit(T_train, y)
            self._calibrator = calib
        except Exception:
            self._calibrator_kind = "temperature"
        if self._calibrator is None:
            # Fallback: temperature-scaled softmax of regression scores `Yhat`.
            coef = res.coef()
            if coef.ndim == 1:
                coef = coef.reshape(-1, 1)
            scores_train = Xc @ coef
            self._fallback_temperature = _fit_temperature(scores_train, y, self.classes_)
            self._fallback_scale = coef
            self._calibrator_kind = "temperature"
        # Coefficient/intercept for the Y_bal prediction (used by transform).
        coef = res.coef()
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        self.coef_ = coef
        self.intercept_ = self.y_mean_ - self.x_mean_ @ coef
        self.diagnostics_ = RunDiagnostics(
            engine=self.engine,
            selection=self.selection,
            criterion=self.criterion,
            orthogonalization=orth,
            operator_bank=self.operator_bank if isinstance(self.operator_bank, str) else "custom",
            selected_operator_indices=list(sel.operator_indices),
            selected_operator_names=list(sel.operator_names),
            operator_scores={k: _to_jsonable(v) for k, v in sel.operator_scores.items()},
            n_components_selected=res.n_components,
            max_components=max_components,
            fit_time_s=time.perf_counter() - start,
            predict_time_s=0.0,
            backend=self.backend,
            extras={"task": "classification", **_to_jsonable(sel.diagnostics)},
        )
        self._bank = bank
        return self

    # ---------------------------------------------------------- inference

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_effective_weights_"):
            raise RuntimeError("Estimator not fitted")
        Xc = np.asarray(X, dtype=float) - self.x_mean_
        return Xc @ self.x_effective_weights_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_calibrator_kind"):
            raise RuntimeError("Estimator not fitted")
        T = self.transform(X)
        if self._calibrator_kind == "logistic" and self._calibrator is not None:
            # LogisticRegression returns columns aligned to classes_.
            proba = self._calibrator.predict_proba(T)
            return _align_proba(proba, self._calibrator.classes_, self.classes_)
        # Temperature-scaled softmax fallback
        Xc = np.asarray(X, dtype=float) - self.x_mean_
        scores = Xc @ self._fallback_scale  # shape (n, n_classes)
        scaled = scores / max(self._fallback_temperature, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        proba = e / e.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return balanced_accuracy(np.asarray(y), self.predict(X))

    # ---------------------------------------------------------- diagnostics

    def get_selected_operators(self) -> List[str]:
        return list(self.selected_operators_)

    def get_diagnostics(self) -> dict:
        return self.diagnostics_.to_dict()

    def selected_operator_sequence_json(self) -> str:
        return json.dumps(self.selected_operator_indices_)


def _align_proba(proba: np.ndarray, calib_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    """Reorder columns of `proba` to match `target_classes` ordering."""
    out = np.zeros((proba.shape[0], target_classes.shape[0]))
    cls_to_idx = {c: i for i, c in enumerate(calib_classes.tolist())}
    for j, cls in enumerate(target_classes.tolist()):
        if cls in cls_to_idx:
            out[:, j] = proba[:, cls_to_idx[cls]]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def _fit_temperature(scores: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Fit a single-temperature scalar by golden-section search on log loss."""
    eps = 1e-12

    def loss(T: float) -> float:
        scaled = scores / max(T, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        proba = e / e.sum(axis=1, keepdims=True)
        out = []
        for i, yi in enumerate(y):
            idx = int(np.where(classes == yi)[0][0])
            out.append(-np.log(max(proba[i, idx], eps)))
        return float(np.mean(out))

    a, b = 1e-2, 5.0
    phi = (1 + 5**0.5) / 2
    res_phi = 1.0 / phi
    c = b - res_phi * (b - a)
    d = a + res_phi * (b - a)
    for _ in range(40):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c = b - res_phi * (b - a)
        d = a + res_phi * (b - a)
    return 0.5 * (a + b)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------


class AOMPLSDAClassifier(_AOMPLSDABase):
    """Adaptive Operator-Mixture PLS-DA with global selection and probability calibration."""

    def __init__(
        self,
        n_components: Union[int, str] = "auto",
        max_components: int = 25,
        engine: str = "simpls_covariance",
        selection: str = "global",
        criterion: str = "cv",
        operator_bank: Union[str, Sequence[LinearSpectralOperator]] = "compact",
        orthogonalization: str = "auto",
        center: bool = True,
        scale: bool = False,
        cv: int = 5,
        random_state: int = 0,
        backend: str = "numpy",
    ) -> None:
        super().__init__(
            n_components=n_components,
            max_components=max_components,
            engine=engine,
            selection=selection,
            criterion=criterion,
            operator_bank=operator_bank,
            orthogonalization=orthogonalization,
            center=center,
            scale=scale,
            cv=cv,
            random_state=random_state,
            backend=backend,
        )


class POPPLSDAClassifier(_AOMPLSDABase):
    """Per-Operator-Per-Component PLS-DA with logistic calibration."""

    def __init__(
        self,
        n_components: Union[int, str] = "auto",
        max_components: int = 25,
        engine: str = "simpls_covariance",
        selection: str = "per_component",
        criterion: str = "cv",
        operator_bank: Union[str, Sequence[LinearSpectralOperator]] = "compact",
        orthogonalization: str = "auto",
        center: bool = True,
        scale: bool = False,
        cv: int = 5,
        random_state: int = 0,
        backend: str = "numpy",
    ) -> None:
        super().__init__(
            n_components=n_components,
            max_components=max_components,
            engine=engine,
            selection=selection,
            criterion=criterion,
            operator_bank=operator_bank,
            orthogonalization=orthogonalization,
            center=center,
            scale=scale,
            cv=cv,
            random_state=random_state,
            backend=backend,
        )
