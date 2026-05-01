"""Classification wrappers for multi-view AOM regressors.

`AOMMoEClassifier` and `BlockSparseAOMMBPLSClassifier` wrap their regressor
counterparts using the class-balanced one-hot encoding from
`aompls.classification` (one-vs-rest scoring + LogisticRegression calibration
on combined latent scores).

Pattern: train one regressor per class on the class-balanced 0/1 indicator,
combine their predictions into a class-score matrix, fit a softmax /
LogisticRegression head for proba calibration. argmax = predict.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .estimators_mbpls import BlockSparseAOMMBPLSRegressor
from .moe import AOMMoERegressor


def _class_balanced_encode(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot with `1/sqrt(pi_c)` magnitude for class c (mirror existing AOM-PLS-DA)."""
    n = y.shape[0]
    K = classes.shape[0]
    Y = np.zeros((n, K))
    for c, cls in enumerate(classes):
        mask = y == cls
        prior = float(mask.sum()) / n
        if prior > 0:
            Y[mask, c] = 1.0 / np.sqrt(prior)
    return Y


class _OneVsRestMultiViewBase(BaseEstimator, ClassifierMixin):
    """Common backbone: fit one regressor per class with balanced y."""

    _estimator_type = "classifier"

    def _build_regressor(self):  # pragma: no cover
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OneVsRestMultiViewBase":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        Y_bal = _class_balanced_encode(y, self.classes_)
        regressors = []
        for c in range(len(self.classes_)):
            reg = self._build_regressor()
            reg.fit(X, Y_bal[:, c])
            regressors.append(reg)
        self.regressors_ = regressors

        # LogisticRegression calibration on combined latent scores.
        scores = np.column_stack([reg.predict(X).ravel() for reg in regressors])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._logreg = LogisticRegression(
                class_weight="balanced", max_iter=2000,
            )
            self._logreg.fit(scores, y)
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "regressors_"):
            raise RuntimeError("Estimator not fitted")
        scores = np.column_stack([reg.predict(X).ravel() for reg in self.regressors_])
        return self._logreg.predict(scores)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "regressors_"):
            raise RuntimeError("Estimator not fitted")
        scores = np.column_stack([reg.predict(X).ravel() for reg in self.regressors_])
        return self._logreg.predict_proba(scores)


class AOMMoEClassifier(_OneVsRestMultiViewBase):
    """One-vs-rest AOM-MoE classifier.

    See `AOMMoERegressor` for parameter semantics. Each class indicator is
    regressed by an independent MoE; predictions are combined via
    LogisticRegression on the latent score matrix.
    """

    def __init__(
        self,
        expert_layout: str = "per_view",
        routing: str = "soft",
        K: int = 3,
        bank_name: str = "compact",
        per_expert_components: int = 10,
        n_oof_folds: int = 3,
        random_state: int = 0,
    ) -> None:
        self.expert_layout = expert_layout
        self.routing = routing
        self.K = K
        self.bank_name = bank_name
        self.per_expert_components = per_expert_components
        self.n_oof_folds = n_oof_folds
        self.random_state = random_state

    def _build_regressor(self) -> AOMMoERegressor:
        return AOMMoERegressor(
            expert_layout=self.expert_layout,
            routing=self.routing,
            K=self.K,
            bank_name=self.bank_name,
            per_expert_components=self.per_expert_components,
            n_oof_folds=self.n_oof_folds,
            random_state=self.random_state,
        )


class BlockSparseAOMMBPLSClassifier(_OneVsRestMultiViewBase):
    """One-vs-rest block-sparse AOM-MBPLS classifier.

    Parameters mirror `BlockSparseAOMMBPLSRegressor`.
    """

    def __init__(
        self,
        n_components: str = "auto",
        max_components: int = 15,
        K: int = 3,
        strategy: str = "equal_width",
        preproc_bank_name: Optional[str] = None,
        criterion: str = "holdout",
        cv: int = 3,
        cv_splitter=None,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_components = max_components
        self.K = K
        self.strategy = strategy
        self.preproc_bank_name = preproc_bank_name
        self.criterion = criterion
        self.cv = cv
        self.cv_splitter = cv_splitter
        self.random_state = random_state

    def _build_regressor(self) -> BlockSparseAOMMBPLSRegressor:
        return BlockSparseAOMMBPLSRegressor(
            n_components=self.n_components,
            max_components=self.max_components,
            K=self.K,
            strategy=self.strategy,
            preproc_bank_name=self.preproc_bank_name,
            criterion=self.criterion,
            cv=self.cv,
            cv_splitter=self.cv_splitter,
            random_state=self.random_state,
        )
