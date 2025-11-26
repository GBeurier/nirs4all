"""Orthogonal PLS Discriminant Analysis (OPLS-DA) classifier for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def _check_pyopls_available():
    try:
        import pyopls
        return True
    except ImportError:
        return False

from .plsda import PLSDA
from .opls import OPLS

class OPLSDA(BaseEstimator, ClassifierMixin):
    """Orthogonal PLS Discriminant Analysis (OPLS-DA) classifier.
    (See pls.py for full docstring)
    """
    def __init__(self, n_components: int = 1, pls_components: int = 5, scale: bool = True):
        self.n_components = n_components
        self.pls_components = pls_components
        self.scale = scale

    def fit(self, X, y):
        if not _check_pyopls_available():
            raise ImportError("pyopls package is required for OPLSDA. Install with 'pip install pyopls'.")
        self.opls_ = OPLS(n_components=self.n_components, pls_components=1, scale=self.scale, backend='numpy')
        X_filtered = self.opls_.fit_transform(X, y)
        self.plsda_ = PLSDA(n_components=self.pls_components)
        self.plsda_.fit(X_filtered, y)
        self.classes_ = self.plsda_.classes_
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X_filtered = self.opls_.transform(X)
        return self.plsda_.predict(X_filtered)

    def predict_proba(self, X):
        X_filtered = self.opls_.transform(X)
        return self.plsda_.predict_proba(X_filtered)

    def transform(self, X):
        return self.opls_.transform(X)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "pls_components": self.pls_components, "scale": self.scale}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
