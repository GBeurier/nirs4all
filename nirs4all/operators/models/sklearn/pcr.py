"""Principal Component Regression (PCR) for nirs4all.

A sklearn-compatible PCR estimator: PCA-based dimensionality reduction
followed by ordinary least-squares linear regression on the principal
components. Classic chemometrics baseline alongside PLS.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class PCR(BaseEstimator, RegressorMixin):
    """Principal Component Regression.

    Fits a PCA on X then regresses Y on the retained scores using
    ordinary least squares.

    Parameters
    ----------
    n_components : int, default=10
        Number of principal components to retain.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X: ArrayLike, y: ArrayLike) -> PCR:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.pca_ = PCA(n_components=self.n_components)
        scores = self.pca_.fit_transform(X)
        self.regressor_ = LinearRegression()
        self.regressor_.fit(scores, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        check_is_fitted(self, ["pca_", "regressor_"])
        X = np.asarray(X, dtype=float)
        scores = self.pca_.transform(X)
        return cast(NDArray[np.floating], np.asarray(self.regressor_.predict(scores), dtype=float))
