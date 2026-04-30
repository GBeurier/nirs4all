"""Centering and scaling helpers shared by all engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StandardScaler:
    """Lightweight centering / optional scaling helper.

    Stores per-feature means and (optionally) standard deviations on the
    training set, to be applied to subsequent matrices including test sets.
    Centering only is the default; per-feature scaling is opt-in because it
    destroys the spectral shape that operators exploit.
    """

    center: bool = True
    scale: bool = False
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("StandardScaler.fit expects a 2D array")
        if self.center:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        if self.scale:
            scale = X.std(axis=0, ddof=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        return X * self.scale_ + self.mean_


def center_xy(X: np.ndarray, Y: np.ndarray, center: bool = True, scale: bool = False):
    """Center (and optionally scale) X and Y on the training set.

    Returns (Xc, Yc, x_mean, y_mean, x_scale, y_scale).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if center:
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = np.zeros(Y.shape[1])
    if scale:
        x_scale = X.std(axis=0, ddof=0)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        y_scale = Y.std(axis=0, ddof=0)
        y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
    else:
        x_scale = np.ones(X.shape[1])
        y_scale = np.ones(Y.shape[1])
    Xc = (X - x_mean) / x_scale
    Yc = (Y - y_mean) / y_scale
    return Xc, Yc, x_mean, y_mean, x_scale, y_scale
