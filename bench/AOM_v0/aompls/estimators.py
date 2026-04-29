"""sklearn-compatible regressors for Operator-Adaptive PLS.

`AOMPLSRegressor` and `POPPLSRegressor` share a common backbone; the only
differences are the default `selection` policy and the default operator bank
preset. Both provide `fit`, `predict`, `transform`, `score`, `get_params`,
`set_params`, plus diagnostic helpers `get_selected_operators` and
`get_diagnostics`.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .banks import bank_by_name, fit_bank
from .diagnostics import RunDiagnostics
from .metrics import r2, rmse
from .operators import LinearSpectralOperator
from .scorers import CriterionConfig
from .selection import SelectionResult, select


class _AOMPLSBase(BaseEstimator, RegressorMixin):
    """Base class shared by AOMPLSRegressor and POPPLSRegressor."""

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
        repeats: int = 1,
        one_se_rule: bool = False,
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
        self.repeats = repeats
        self.one_se_rule = one_se_rule

    # ---------------------------------------------------------- fitting

    def _resolve_bank(self, p: int) -> List[LinearSpectralOperator]:
        if isinstance(self.operator_bank, str):
            bank = bank_by_name(self.operator_bank, p=p)
        else:
            bank = list(self.operator_bank)
        # Ensure identity is in the bank, matching the production AOM-PLS
        # behaviour: with identity available, AOM_v0 cannot do worse than
        # standard PLS on the selection criterion.
        from .operators import IdentityOperator
        if not any(isinstance(op, IdentityOperator) for op in bank):
            bank = [IdentityOperator(p=p)] + bank
        return bank

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_AOMPLSBase":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n, p = X.shape
        # Centering
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(p)
            self.y_mean_ = np.zeros(y.shape[1])
        Xc = X - self.x_mean_
        yc = y - self.y_mean_
        # Bank
        bank = self._resolve_bank(p)
        fit_bank(bank, X)
        # Components limit
        max_components = min(self.max_components, n - 1, p)
        if isinstance(self.n_components, int):
            n_request = min(self.n_components, max_components)
            auto_prefix = False
        else:
            n_request = max_components
            auto_prefix = True
        # Criterion config
        criterion = CriterionConfig(
            kind=self.criterion,
            cv=self.cv,
            random_state=self.random_state,
            task="regression",
            repeats=getattr(self, "repeats", 1),
            one_se_rule=getattr(self, "one_se_rule", False),
        )
        # Orthogonalization resolution
        orth = self.orthogonalization
        if orth == "auto":
            orth = "transformed" if self.selection in ("none", "global") else "original"
        # Run selection
        sel = select(
            Xc=Xc,
            yc=yc,
            operators=bank,
            engine=self.engine,
            selection=self.selection,
            n_components_max=n_request,
            criterion=criterion,
            orthogonalization=orth,
            auto_prefix=auto_prefix,
        )
        res = sel.result
        # Pull final coefficients and store sklearn-style attributes.
        self.x_weights_ = res.Z.copy()
        self.x_effective_weights_ = res.Z.copy()
        self.x_loadings_ = res.P.copy()
        self.y_loadings_ = res.Q.copy()
        self.x_scores_ = res.T.copy()
        # Build coefficient and intercept.
        coef = res.coef()
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        self.coef_ = coef
        self.intercept_ = self.y_mean_ - self.x_mean_ @ coef
        # Persist selection results / diagnostics
        self.selected_operators_ = sel.operator_names
        self.selected_operator_indices_ = sel.operator_indices
        self.operator_scores_ = sel.operator_scores
        self.n_components_ = res.n_components
        self.engine_ = self.engine
        self.selection_ = self.selection
        self.criterion_ = self.criterion
        self.orthogonalization_ = orth
        self.rotations_ = res.Z.copy()
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
            extras=_to_jsonable(sel.diagnostics),
        )
        self._bank = bank
        return self

    # ---------------------------------------------------------- inference

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "coef_"):
            raise RuntimeError("Estimator not fitted")
        start = time.perf_counter()
        pred = (X - self.x_mean_) @ self.coef_ + self.y_mean_
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        self.diagnostics_.predict_time_s = time.perf_counter() - start
        return pred

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_effective_weights_"):
            raise RuntimeError("Estimator not fitted")
        Xc = np.asarray(X, dtype=float) - self.x_mean_
        return Xc @ self.x_effective_weights_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return r2(np.asarray(y).ravel(), np.asarray(pred).ravel())

    # ---------------------------------------------------------- diagnostics

    def get_selected_operators(self) -> List[str]:
        return list(self.selected_operators_)

    def get_diagnostics(self) -> dict:
        return self.diagnostics_.to_dict()

    def selected_operator_sequence_json(self) -> str:
        return json.dumps(self.selected_operator_indices_)


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


class AOMPLSRegressor(_AOMPLSBase):
    """AOM-PLS regressor: one operator selected for the whole model.

    Default engine is `simpls_covariance`; default criterion is `cv`. The
    default bank is the `default` preset (~77 operators), matching the
    production nirs4all AOM-PLS bank for fair comparison.
    """

    def __init__(
        self,
        n_components: Union[int, str] = "auto",
        max_components: int = 25,
        engine: str = "simpls_covariance",
        selection: str = "global",
        criterion: str = "cv",
        operator_bank: Union[str, Sequence[LinearSpectralOperator]] = "default",
        orthogonalization: str = "auto",
        center: bool = True,
        scale: bool = False,
        cv: int = 5,
        random_state: int = 0,
        backend: str = "numpy",
        repeats: int = 1,
        one_se_rule: bool = False,
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
            repeats=repeats,
            one_se_rule=one_se_rule,
        )


class POPPLSRegressor(_AOMPLSBase):
    """POP-PLS regressor: per-component operator selection.

    Default engine is `simpls_covariance`; default selection is
    `per_component`; default orthogonalization is `original` (resolved via
    `auto`).
    """

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
        repeats: int = 1,
        one_se_rule: bool = False,
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
            repeats=repeats,
            one_se_rule=one_se_rule,
        )
