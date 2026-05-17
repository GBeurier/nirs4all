"""V5b — pure residual stacking of AOMRidge-Blender + TabPFN-2.5.

The estimator decomposes the target additively into a linear part learnt by
the deployable :class:`AOMRidgeBlender` (or any base estimator passed in)
and a nonlinear residual part learnt by TabPFN-2.5:

.. math::

    \\hat{y}(x_*) \\;=\\; \\hat{y}_{\\mathrm{base}}(x_*) +
                       \\alpha \\cdot \\sigma_r \\cdot
                       \\widehat{r}_{\\mathrm{TabPFN}}(x_*)

where:

* :math:`\\hat{y}_{\\mathrm{base}}` is the prediction of the base AOM-Ridge
  estimator (an ``AOMRidgeBlender`` over the 8 HEADLINE candidates by
  default).
* :math:`\\widehat{r}_{\\mathrm{TabPFN}}` is the prediction of TabPFN-2.5
  fitted on standardised out-of-fold residuals
  :math:`r_i = y_i - \\hat{y}_{\\mathrm{base}}^{\\mathrm{OOF}}(x_i)`.
* :math:`\\alpha \\in [0, 1]` is a non-negative scalar tuned on the same OOF
  predictions and capped at zero by a circuit-breaker if TabPFN does not
  improve OOF RMSE by at least ``min_improvement`` (default 1%).
* :math:`\\sigma_r` is the standard deviation of OOF residuals over training
  rows, used to invert the standardisation that TabPFN sees.

Anti-leakage protocol (mirrors AOM-PLS / AOM-Ridge fold-local invariants):

1. Outer CV (default SPXY, 3 folds) splits the training set.
2. For each fold ``f``:

   a. Re-fit the base estimator on ``train \\ f``.
   b. Predict ``y_hat_f`` on ``f``.
   c. Compute ``r_f = y_f - y_hat_f``.

3. The OOF residual vector ``r`` is concatenated across folds.
4. ``r`` is standardised fold-locally before being passed to TabPFN as the
   target. ``\\sigma_r`` is the in-fold residual std for that fold;
   prediction time uses the global training-set residual std for
   un-standardisation (this is the only "global" statistic and it is
   computed once after step 3).
5. TabPFN is fitted on the *full* training set with ``y = r``, downsampling
   the spectrum if ``p > max_features`` (uniform stride, no PCA).
6. For each fold ``f`` we also predict ``r_hat_f = TabPFN_f(X_f)`` so that
   ``alpha`` can be tuned on OOF.
7. ``alpha`` is the non-negative scalar in :math:`[0, 1]` minimising the
   OOF squared error of ``y_pred_f = y_hat_f + alpha * sigma_r * r_hat_f``.
8. If the resulting OOF improvement vs. the base alone is < ``min_improvement``,
   ``alpha`` is forced to zero (circuit-breaker).
9. Final fit: re-fit the base on the full training set, re-fit TabPFN with
   ``y = r_OOF``, store ``alpha``, ``sigma_r``, and the trained estimators.
   At test time, the prediction is the additive sum.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array, check_is_fitted

from .blender import AOMRidgeBlender
from .cv import _try_import_spxyfold
from .tabpfn_candidate import TabPFNCandidate

__all__ = ["AOMRidgeResidualTabPFN"]


@dataclass
class _OOFOutputs:
    y_oof: np.ndarray
    r_oof: np.ndarray
    r_hat_oof: np.ndarray
    sigma_r: float


class AOMRidgeResidualTabPFN(BaseEstimator, RegressorMixin):
    """Residual stacking estimator AOMRidge + TabPFN-2.5.

    Parameters
    ----------
    base_estimator : sklearn estimator, optional
        Base regressor that learns the linear part. Defaults to
        :class:`AOMRidgeBlender` over the 8 HEADLINE candidates.
    tabpfn_estimator : sklearn estimator, optional
        Residual learner. Defaults to :class:`TabPFNCandidate` with
        ``standardise_y=False`` since we standardise the residual target
        ourselves before passing it in.
    outer_cv : int or sklearn splitter, default=3
        Outer CV used to compute OOF residuals. ``int`` is interpreted as
        a 3-fold SPXY (the default convention in the rest of the package).
    min_improvement : float, default=0.01
        Minimum OOF RMSE relative drop required before TabPFN is given
        a non-zero alpha. ``0.01`` means "at least 1% RMSE drop"; below
        that threshold ``alpha`` is forced to ``0``.
    standardise_residual : bool, default=True
        Whether to z-score residuals fold-locally before fitting TabPFN.
        Recommended; TabPFN-2.5 is trained on standardised targets.
    random_state : int, default=0
        Forwarded to splitter, base, and TabPFN. ``0`` keeps everything
        reproducible.

    Attributes
    ----------
    base_estimator_ : fitted sklearn estimator
    tabpfn_estimator_ : fitted sklearn estimator (the TabPFN candidate
        fitted on the full residual target).
    alpha_ : float
        The selected non-negative scalar in :math:`[0, 1]`.
    sigma_r_ : float
        Standard deviation of the OOF residuals over the training set.
    oof_improvement_ : float
        Relative RMSE drop of the (base + alpha * TabPFN) over the base
        alone, on the OOF predictions. Negative or zero values mean
        TabPFN did not help.
    diagnostics_ : dict
        Run record (alpha, sigma_r, oof_improvement, fold-by-fold RMSE).
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        tabpfn_estimator: BaseEstimator | None = None,
        *,
        outer_cv: int | object = 3,
        outer_cv_kind: str = "spxy",
        min_improvement: float = 0.01,
        standardise_residual: bool = True,
        random_state: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.tabpfn_estimator = tabpfn_estimator
        self.outer_cv = outer_cv
        self.outer_cv_kind = outer_cv_kind
        self.min_improvement = min_improvement
        self.standardise_residual = standardise_residual
        self.random_state = random_state

    # ----- sklearn API ----------------------------------------------------

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim > 1 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        if y_arr.ndim != 1:
            raise ValueError(
                "AOMRidgeResidualTabPFN requires 1D targets; got shape "
                f"{y_arr.shape}."
            )
        n = X.shape[0]

        base = self._make_base()
        tabpfn = self._make_tabpfn()

        # ---- Stage 1: base OOF predictions ------------------------------
        splitter = self._make_splitter()
        y_hat_oof = np.zeros_like(y_arr)
        fold_records: list[dict] = []
        for f, (tr_idx, va_idx) in enumerate(self._split(splitter, X, y_arr)):
            base_f = clone(base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_f.fit(X[tr_idx], y_arr[tr_idx])
            y_hat_oof[va_idx] = np.asarray(base_f.predict(X[va_idx]),
                                            dtype=np.float64).ravel()
            fold_records.append({"fold": f, "n_train": int(len(tr_idx)),
                                 "n_val": int(len(va_idx))})

        r_oof = y_arr - y_hat_oof
        sigma_r = float(np.std(r_oof) or 1.0)

        if self.standardise_residual:
            r_train_target = (r_oof - float(np.mean(r_oof))) / sigma_r
        else:
            r_train_target = r_oof.copy()

        # ---- Stage 2: TabPFN OOF residual predictions -------------------
        # We need r_hat_OOF (TabPFN OOF predictions of the residual).
        # We re-use the same outer fold partition: for each fold f, fit
        # TabPFN on (X[tr], standardised r[tr]) and predict on X[va].
        # The standardisation uses the *training-fold* residual mean and
        # std so TabPFN does not see validation-fold information.
        r_hat_oof = np.zeros_like(y_arr)
        for f, (tr_idx, va_idx) in enumerate(self._split(splitter, X, y_arr)):
            r_tr = r_oof[tr_idx]
            mu_f = float(np.mean(r_tr))
            sigma_f = float(np.std(r_tr) or 1.0)
            if self.standardise_residual:
                r_tr_target = (r_tr - mu_f) / sigma_f
            else:
                r_tr_target = r_tr.copy()
                mu_f, sigma_f = 0.0, 1.0
            tab_f = clone(tabpfn)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tab_f.fit(X[tr_idx], r_tr_target)
                r_hat_z = np.asarray(tab_f.predict(X[va_idx]),
                                     dtype=np.float64).ravel()
            r_hat_oof[va_idx] = mu_f + sigma_f * r_hat_z

        # ---- Stage 3: alpha selection ----------------------------------
        alpha_star, oof_improvement = self._select_alpha(y_arr, y_hat_oof, r_hat_oof)
        if oof_improvement < self.min_improvement:
            alpha_star = 0.0

        # ---- Stage 4: refit on full training set -----------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base.fit(X, y_arr)

        if alpha_star > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.standardise_residual:
                    tabpfn.fit(X, (r_oof - float(np.mean(r_oof))) / sigma_r)
                else:
                    tabpfn.fit(X, r_oof)
        # Otherwise we don't bother fitting tabpfn on the full set; the
        # circuit-breaker means we won't use it at predict time.

        self.base_estimator_ = base
        self.tabpfn_estimator_ = tabpfn if alpha_star > 0 else None
        self.alpha_ = float(alpha_star)
        self.sigma_r_ = sigma_r
        self.oof_improvement_ = float(oof_improvement)
        self.diagnostics_ = {
            "alpha": float(alpha_star),
            "alpha_pre_breaker": float(alpha_star) if oof_improvement >= self.min_improvement
                                else float(self._select_alpha(y_arr, y_hat_oof, r_hat_oof)[0]),
            "sigma_r": sigma_r,
            "oof_improvement": float(oof_improvement),
            "circuit_breaker_active": alpha_star == 0
                                       and oof_improvement < self.min_improvement,
            "folds": fold_records,
            "n_train": int(n),
        }
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["base_estimator_"])
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_hat_base = np.asarray(self.base_estimator_.predict(X),
                                     dtype=np.float64).ravel()
        if self.alpha_ > 0 and self.tabpfn_estimator_ is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_hat_z = np.asarray(self.tabpfn_estimator_.predict(X),
                                     dtype=np.float64).ravel()
            scale = self.sigma_r_ if self.standardise_residual else 1.0
            return y_hat_base + self.alpha_ * scale * r_hat_z
        return y_hat_base

    # ----- internals ------------------------------------------------------

    def _make_base(self) -> BaseEstimator:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return AOMRidgeBlender(
            outer_cv=3,
            outer_cv_kind="spxy",
            random_state=self.random_state,
            regularizer=0.01,
        )

    def _make_tabpfn(self) -> BaseEstimator:
        if self.tabpfn_estimator is not None:
            return clone(self.tabpfn_estimator)
        return TabPFNCandidate(
            n_estimators=4,
            max_features=2000,
            max_samples=9500,
            standardise_y=False,  # we standardise ourselves
            device="auto",
            random_state=self.random_state,
        )

    def _make_splitter(self):
        n_splits = self.outer_cv if isinstance(self.outer_cv, int) else 3
        if self.outer_cv_kind == "spxy":
            spxy = _try_import_spxyfold()
            if spxy is not None:
                return spxy(n_splits=n_splits, random_state=self.random_state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def _split(self, splitter, X, y) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        try:
            return list(splitter.split(X, y))
        except TypeError:
            return list(splitter.split(X))

    def _select_alpha(
        self, y: np.ndarray, y_hat: np.ndarray, r_hat: np.ndarray,
    ) -> tuple[float, float]:
        """Closed-form alpha for the bounded scalar regression.

        Minimises ||y - y_hat - alpha * r_hat||^2 over alpha and clips to
        [0, 1]. Returns (alpha, relative_RMSE_improvement).
        """
        residual = y - y_hat
        denom = float(np.dot(r_hat, r_hat))
        if denom < 1e-12:
            alpha = 0.0
        else:
            alpha = float(np.dot(residual, r_hat) / denom)
            alpha = max(0.0, min(1.0, alpha))
        rmse_base = float(np.sqrt(np.mean(residual ** 2)))
        rmse_blend = float(np.sqrt(np.mean((residual - alpha * r_hat) ** 2)))
        improvement = (rmse_base - rmse_blend) / rmse_base if rmse_base > 0 else 0.0
        return alpha, improvement
