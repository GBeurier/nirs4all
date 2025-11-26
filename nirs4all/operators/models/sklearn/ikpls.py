"""Improved Kernel PLS (IKPLS) regressor for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_ikpls_available():
    try:
        import ikpls
        return True
    except ImportError:
        return False

class IKPLS(BaseEstimator, RegressorMixin):
    """Improved Kernel PLS (IKPLS) regressor.
    (See pls.py for full docstring)
    """
    def __init__(self, n_components: int = 10, algorithm: int = 1, center: bool = True, scale: bool = True, backend: str = 'numpy'):
        self.n_components = n_components
        self.algorithm = algorithm
        self.center = center
        self.scale = scale
        self.backend = backend

    def fit(self, X, y):
        if not _check_ikpls_available():
            raise ImportError("ikpls package is required for IKPLS. Install with 'pip install ikpls'.")
        import ikpls
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.model_ = ikpls.IKPLS(n_components=n_comp, algorithm=self.algorithm, center=self.center, scale=self.scale, backend=self.backend)
        self.model_.fit(X, y)
        self.n_components_ = self.model_.n_components_
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X, n_components=None):
        X = np.asarray(X)
        return self.model_.predict(X, n_components=n_components)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "algorithm": self.algorithm, "center": self.center, "scale": self.scale, "backend": self.backend}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
