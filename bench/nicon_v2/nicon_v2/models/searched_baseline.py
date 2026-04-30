"""SearchedRidge / SearchedPLS — Ridge / PLS with REDUCED cartesian preprocessing search.

**Important scope note (Codex round 4 finding F5).**

This module implements a *reduced approximation* of the paper Ridge / PLS
baseline used in ``bench/tabpfn_paper/run_reg_pls.py``. The reference paper's
recipe set includes EMSC(degree=1, 2), SG(31, 2, 1) and SG(15, 3, 2) and SG(21, 3, 2),
``Gaussian(order=0, sigma={1,2})``, ``ASLSBaseline``, ``OSC(1, 2, 3)``, and a
60-trial α grid. Our recipe set is **deliberately smaller** (3 scatter × 5
SG variants × 2 detrend × 11 α = 330 candidates) because (a) we want the inner
CV to fit under 30 s per dataset on smoke and (b) the missing operators
(EMSC, OSC, ASLSBaseline) require careful implementation that is out of scope
for the headline contribution.

Therefore: **`SearchedRidge` is NOT a fair drop-in replacement for paper
Ridge.** It is an *upper-bound surrogate* that gives a strong cartesian-search
Ridge baseline on equal-platform footing with `nicon_v2`. Treat its numbers
as approximate evidence that the paper-Ridge gap is at least partially due to
preprocessing search, not as an apples-to-apples replacement.

Search space (SNV / MSC / None) × (SG / no SG with five windows) × (None /
Detrend) × (α grid). Inner KFold CV picks the combination with the lowest
mean rmsep. Refit on full train.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-12
    return (X - mu) / sd


def _msc(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float).ravel()
    rc = ref - ref.mean()
    denom = (rc * rc).sum() + 1e-12
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        xi = X[i]
        xc = xi - xi.mean()
        b = (xc * rc).sum() / denom
        a = xi.mean() - b * ref.mean()
        b = max(b, 1e-6)
        out[i] = (xi - a) / b
    return out


def _detrend(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    t = np.arange(p, dtype=float)
    t_centered = t - t.mean()
    denom = (t_centered * t_centered).sum() + 1e-12
    out = np.empty_like(X)
    for i in range(n):
        xi = X[i]
        a = xi.mean()
        b = ((xi - a) * t_centered).sum() / denom
        out[i] = xi - (a + b * t_centered)
    return out


def _sg(X: np.ndarray, window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv,
                         axis=-1, mode="interp")


def _apply_preproc(X_train: np.ndarray, X_test: np.ndarray, recipe: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Fit preproc on X_train, apply to X_train and X_test, return (X_train_proc, X_test_proc).

    `recipe = (scatter_kind, sg_kind, detrend_kind)`.
    """
    scatter, sg_kind, detrend_kind = recipe
    Xtr, Xte = X_train, X_test
    if scatter == "snv":
        Xtr = _snv(Xtr)
        Xte = _snv(Xte)
    elif scatter == "msc":
        ref = X_train.mean(axis=0)
        Xtr = _msc(X_train, ref)
        Xte = _msc(X_test, ref)
    if sg_kind:
        w, p, d = sg_kind
        Xtr = _sg(Xtr, window_length=w, polyorder=p, deriv=d)
        Xte = _sg(Xte, window_length=w, polyorder=p, deriv=d)
    if detrend_kind == "detrend":
        Xtr = _detrend(Xtr)
        Xte = _detrend(Xte)
    return Xtr, Xte


def _build_recipes() -> list[tuple]:
    scatters = (None, "snv", "msc")
    sg_kinds = (None, (11, 2, 1), (15, 2, 1), (21, 2, 1), (15, 3, 2))
    detrends = (None, "detrend")
    return [(s, g, d) for s in scatters for g in sg_kinds for d in detrends]


@dataclass
class SearchedRidge:
    """Cartesian preprocessing search around sklearn Ridge.

    ~30 preprocessing recipes × 11 α values = 330 candidates; default budget keeps
    the smoke runtime under 30 s per dataset.
    """

    alphas: tuple[float, ...] = field(default_factory=lambda: tuple(10.0 ** k for k in np.linspace(-3.0, 3.0, 11)))
    cv_splits: int = 5
    seed: int = 0
    selected_recipe_: tuple | None = None
    selected_alpha_: float = float("nan")
    pipeline_: tuple | None = None  # (recipe, scaler_mean, scaler_std, ridge)
    n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SearchedRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]

        recipes = _build_recipes()
        kf = KFold(n_splits=min(self.cv_splits, max(2, X.shape[0] // 4)), shuffle=True, random_state=self.seed)

        best_rmse = float("inf")
        best_recipe: tuple | None = None
        best_alpha = float("nan")
        for recipe in recipes:
            for alpha in self.alphas:
                errs = []
                for tr, va in kf.split(X):
                    Xtr_p, Xva_p = _apply_preproc(X[tr], X[va], recipe)
                    mu = Xtr_p.mean(axis=0)
                    sd = Xtr_p.std(axis=0) + 1e-12
                    model = Ridge(alpha=alpha).fit((Xtr_p - mu) / sd, y[tr])
                    pred = model.predict((Xva_p - mu) / sd)
                    errs.append(float(np.sqrt(np.mean((pred - y[va]) ** 2))))
                m = float(np.mean(errs))
                if m < best_rmse:
                    best_rmse = m
                    best_recipe = recipe
                    best_alpha = alpha

        # Refit on full training set with the best (recipe, alpha).
        Xtr_p, _ = _apply_preproc(X, X, best_recipe)  # apply on train only; test will be re-applied at predict
        mu = Xtr_p.mean(axis=0)
        sd = Xtr_p.std(axis=0) + 1e-12
        ridge = Ridge(alpha=best_alpha).fit((Xtr_p - mu) / sd, y)
        self.selected_recipe_ = best_recipe
        self.selected_alpha_ = best_alpha
        self.pipeline_ = (best_recipe, mu, sd, ridge, X)  # keep X for MSC ref handling
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("SearchedRidge not fitted")
        recipe, mu, sd, ridge, X_train_kept = self.pipeline_
        _, X_test_p = _apply_preproc(X_train_kept, X_test, recipe)
        return ridge.predict((X_test_p - mu) / sd)

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {
            "model": "SearchedRidge",
            "selected_recipe": str(self.selected_recipe_),
            "selected_alpha": self.selected_alpha_,
            "alpha_grid_size": len(self.alphas),
            "n_recipes": len(_build_recipes()),
        }


@dataclass
class SearchedPLS:
    """Cartesian preprocessing search around sklearn PLSRegression."""

    n_components_grid: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 15, 20, 25)
    cv_splits: int = 5
    seed: int = 0
    selected_recipe_: tuple | None = None
    selected_n_components_: int = 0
    pipeline_: tuple | None = None
    n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SearchedPLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        max_components = min(X.shape[0] - 1, X.shape[1])
        candidates = tuple(c for c in self.n_components_grid if c <= max_components) or (1,)
        recipes = _build_recipes()
        kf = KFold(n_splits=min(self.cv_splits, max(2, X.shape[0] // 4)), shuffle=True, random_state=self.seed)

        best_rmse = float("inf")
        best_recipe: tuple | None = None
        best_n = 1
        for recipe in recipes:
            for n_comp in candidates:
                errs = []
                for tr, va in kf.split(X):
                    Xtr_p, Xva_p = _apply_preproc(X[tr], X[va], recipe)
                    mu = Xtr_p.mean(axis=0)
                    sd = Xtr_p.std(axis=0) + 1e-12
                    pls = PLSRegression(n_components=n_comp).fit((Xtr_p - mu) / sd, y[tr])
                    pred = pls.predict((Xva_p - mu) / sd).ravel()
                    errs.append(float(np.sqrt(np.mean((pred - y[va]) ** 2))))
                m = float(np.mean(errs))
                if m < best_rmse:
                    best_rmse = m
                    best_recipe = recipe
                    best_n = n_comp

        Xtr_p, _ = _apply_preproc(X, X, best_recipe)
        mu = Xtr_p.mean(axis=0)
        sd = Xtr_p.std(axis=0) + 1e-12
        pls = PLSRegression(n_components=best_n).fit((Xtr_p - mu) / sd, y)
        self.selected_recipe_ = best_recipe
        self.selected_n_components_ = best_n
        self.pipeline_ = (best_recipe, mu, sd, pls, X)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("SearchedPLS not fitted")
        recipe, mu, sd, pls, X_train_kept = self.pipeline_
        _, X_test_p = _apply_preproc(X_train_kept, X_test, recipe)
        return pls.predict((X_test_p - mu) / sd).ravel()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {
            "model": "SearchedPLS",
            "selected_recipe": str(self.selected_recipe_),
            "selected_n_components": self.selected_n_components_,
        }
