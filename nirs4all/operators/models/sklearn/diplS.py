"""Dynamic PLS (DiPLS) regressor for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_trendfitter_available():
    try:
        from trendfitter.models import DiPLS as _DiPLS
        return True
    except ImportError:
        return False

class DiPLS(BaseEstimator, RegressorMixin):
    """Dynamic PLS (DiPLS) regressor.
    (See pls.py for full docstring)
    """
    def __init__(self, n_components: int = 5, lags: int = 1, cv_splits: int = 7, tol: float = 1e-8, max_iter: int = 1000):
        self.n_components = n_components
        self.lags = lags
        self.cv_splits = cv_splits
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        if not _check_trendfitter_available():
            raise ImportError("trendfitter package is required for DiPLS. Install with 'pip install trendfitter'.")
        from trendfitter.models import DiPLS as _DiPLS
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.model_ = _DiPLS(n_components=n_comp, lags=self.lags, cv_splits=self.cv_splits, tol=self.tol, max_iter=self.max_iter)
        self.model_.fit(X, y)
        self.n_components_ = self.model_.n_components_
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.model_.predict(X)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "lags": self.lags, "cv_splits": self.cv_splits, "tol": self.tol, "max_iter": self.max_iter}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
