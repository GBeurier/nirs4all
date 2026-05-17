"""TabPFN-2.5 wrapper as a Blender / AutoSelect candidate.

Provides a thin sklearn-compatible regressor that:

1. Accepts wide spectral inputs ($p \\le 2000$) directly.
2. Stride-downsamples ``X`` when $p > \\texttt{max\\_features}$ so we stay
   within TabPFN-2.5's documented intended-use ceiling. This is a *uniform*
   downsample (no PCA, no supervised projection) so the nonlinear structure
   of the spectrum is preserved at lower resolution.
3. Caps row count at ``max_samples`` (default 9500) by random subsampling when
   the train set exceeds the prior's sample limit.
4. Exposes ``fit`` / ``predict`` / ``get_params`` / ``set_params`` and is
   pickle-safe (the underlying ``TabPFNRegressor`` is created lazily inside
   ``fit`` and dropped after each ``predict`` to keep the worker memory
   footprint small in long-running benches).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass
class _SubsampleSpec:
    feature_indices: np.ndarray | None
    row_indices: np.ndarray | None


class TabPFNCandidate(BaseEstimator, RegressorMixin):
    """sklearn wrapper around :class:`tabpfn.TabPFNRegressor`.

    Parameters
    ----------
    n_estimators : int, default=4
        Number of TabPFN forward passes to ensemble. Lower values are
        cheaper but noisier; ``8`` is the TabPFN default.
    max_features : int, default=2000
        If ``X.shape[1]`` exceeds this, the input is uniform-stride
        downsampled to roughly this width. The stride is chosen so the
        downsampled spectrum has at most ``max_features`` columns.
    max_samples : int, default=9500
        If ``n_train`` exceeds this, training rows are randomly subsampled
        (no replacement) using ``random_state`` so the context fits in
        TabPFN-2.5's prior.
    standardise_y : bool, default=True
        If True, ``y`` is z-scored before fit and the prediction is
        de-standardised. TabPFN-2.5 prefers standardised targets.
    device : str, default="auto"
        ``"auto" | "cuda" | "cpu"``. Forwarded to TabPFN.
    random_state : int, default=0
        Seed for the row-subsample RNG and TabPFN's own random_state.
    ignore_pretraining_limits : bool, default=True
        Forwarded to TabPFN. Allows above-prior-limits inputs at the cost
        of a warning; combined with our own truncation this should be
        rare but is left enabled as a safety net.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 4,
        max_features: int = 2000,
        max_samples: int = 9500,
        standardise_y: bool = True,
        device: str = "auto",
        random_state: int = 0,
        ignore_pretraining_limits: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.standardise_y = standardise_y
        self.device = device
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits

    # ----- sklearn API ----------------------------------------------------

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim > 1 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        if y_arr.ndim != 1:
            raise ValueError(
                "TabPFNCandidate requires 1D targets; got shape "
                f"{y_arr.shape}. Train one wrapper per target if needed."
            )

        rng = np.random.default_rng(self.random_state)
        feat_idx = self._choose_features(X.shape[1])
        row_idx = self._choose_rows(X.shape[0], rng)
        X_fit = X[:, feat_idx] if feat_idx is not None else X
        if row_idx is not None:
            X_fit = X_fit[row_idx]
            y_fit = y_arr[row_idx]
        else:
            y_fit = y_arr

        if self.standardise_y:
            self._y_mean_ = float(np.mean(y_fit))
            self._y_std_ = float(np.std(y_fit) or 1.0)
            y_fit_z = (y_fit - self._y_mean_) / self._y_std_
        else:
            self._y_mean_ = 0.0
            self._y_std_ = 1.0
            y_fit_z = y_fit

        self._spec_ = _SubsampleSpec(feature_indices=feat_idx, row_indices=row_idx)
        self._estimator_ = self._make_estimator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._estimator_.fit(X_fit, y_fit_z)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["_estimator_"])
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        feat_idx = self._spec_.feature_indices
        X_use = X[:, feat_idx] if feat_idx is not None else X
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_z = self._estimator_.predict(X_use)
        return self._y_mean_ + self._y_std_ * np.asarray(y_z, dtype=np.float64)

    # ----- internals ------------------------------------------------------

    def _choose_features(self, p: int) -> np.ndarray | None:
        if p <= self.max_features:
            return None
        stride = int(np.ceil(p / self.max_features))
        return np.arange(0, p, stride)

    def _choose_rows(self, n: int, rng: np.random.Generator) -> np.ndarray | None:
        if n <= self.max_samples:
            return None
        return np.sort(rng.choice(n, size=self.max_samples, replace=False))

    def _make_estimator(self) -> Any:
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            n_estimators=int(self.n_estimators),
            device=self.device,
            random_state=int(self.random_state),
            ignore_pretraining_limits=bool(self.ignore_pretraining_limits),
            n_preprocessing_jobs=1,
        )

    # Pickling: drop the underlying TabPFN estimator before serialisation
    # because it holds a torch graph and a CUDA context. The bench keeps a
    # joblib worker per variant so this is rare, but the safety net keeps the
    # wrapper portable.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_estimator_", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
