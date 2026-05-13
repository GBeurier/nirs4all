"""sklearn-compatible AOMPLSCompact estimator (PLS1, compact bank, CV)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from . import _binding  # type: ignore[attr-defined]


class AOMPLSCompact(BaseEstimator, RegressorMixin):
    """Adaptive Operator Mixture PLS (compact bank, PLS1).

    Parameters
    ----------
    max_components : int, default=15
        Maximum number of PLS components to extract during CV scoring and refit.
    n_folds : int, default=5
        Number of CV folds. Ignored when ``cv_mode='external'``.
    cv_mode : {'kfold', 'spxy', 'holdout', 'external'}, default='kfold'
        Cross-validation strategy. Use 'external' with ``external_folds``
        for bit-exact parity tests against fold indices computed in another
        environment (e.g. sklearn's ``KFold``).
    one_se_rule : bool, default=False
        Apply the one-standard-error parsimony rule to the operator/k selection.
    random_state : int, default=0
        Seed for the internal CV shuffler. Not bit-compatible with numpy's
        ``MT19937``.
    preproc : {'none','snv','msc','osc','asls','snv+osc','asls+osc'}, default='none'
        One-shot preprocessing applied before AOM.
    osc_n_components : int, default=1
        Number of OSC components (only used when preproc includes ``osc``).
    asls_lam, asls_p, asls_n_iter
        Asymmetric Least Squares hyperparameters (used when preproc includes ``asls``).
    center : bool, default=True
        Whether to mean-center X and y before AOM. Strongly recommended.
    external_folds : sequence of int sequences, optional
        Required when ``cv_mode='external'``. Test indices per fold.
    """

    def __init__(
        self,
        max_components: int = 15,
        n_folds: int = 5,
        cv_mode: str = "kfold",
        one_se_rule: bool = False,
        random_state: int = 0,
        preproc: str = "none",
        osc_n_components: int = 1,
        asls_lam: float = 1e5,
        asls_p: float = 0.01,
        asls_n_iter: int = 10,
        center: bool = True,
        external_folds: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.max_components = max_components
        self.n_folds = n_folds
        self.cv_mode = cv_mode
        self.one_se_rule = one_se_rule
        self.random_state = random_state
        self.preproc = preproc
        self.osc_n_components = osc_n_components
        self.asls_lam = asls_lam
        self.asls_p = asls_p
        self.asls_n_iter = asls_n_iter
        self.center = center
        self.external_folds = external_folds

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        ext = None
        if self.cv_mode == "external":
            if self.external_folds is None:
                raise ValueError("cv_mode='external' requires external_folds")
            ext = [list(map(int, f)) for f in self.external_folds]
        model = _binding.fit(
            X,
            y,
            max_components=int(self.max_components),
            n_folds=int(self.n_folds),
            cv_mode=str(self.cv_mode),
            one_se_rule=bool(self.one_se_rule),
            center=bool(self.center),
            random_state=int(self.random_state),
            preproc=str(self.preproc),
            osc_n_components=int(self.osc_n_components),
            asls_lam=float(self.asls_lam),
            asls_p=float(self.asls_p),
            asls_n_iter=int(self.asls_n_iter),
            external_folds=ext,
        )
        self._model = model
        self.coef_ = np.asarray(model["coef"], dtype=np.float64)
        self.intercept_ = float(model["intercept"])
        self.x_mean_ = np.asarray(model["x_mean"], dtype=np.float64)
        self.y_mean_ = float(model["y_mean"])
        self.selected_operator_index_ = int(model["selected_operator_index"])
        self.selected_operator_name_ = str(model["selected_operator_name"])
        self.bank_names_ = list(model["bank_names"])
        self.n_components_ = int(model["n_components_selected"])
        self.rmse_curves_ = np.asarray(model["rmse_curves"], dtype=np.float64)
        self.fold_indices_ = [list(map(int, f)) for f in model["fold_indices"]]
        self.one_se_applied_ = bool(model["one_se_applied"])
        self.fit_time_s_ = float(model["fit_time_s"])
        return self

    def predict(self, X):
        if not hasattr(self, "_model"):
            raise RuntimeError("AOMPLSCompact is not fitted yet")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return _binding.predict(self._model, X)

    def score(self, X, y) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(np.asarray(y, dtype=np.float64).ravel(), self.predict(X)))
