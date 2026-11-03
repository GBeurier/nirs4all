"""PLS Discriminant Analysis (PLS-DA) classifier for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


class PLSDA(ClassifierMixin, BaseEstimator):
    """PLS Discriminant Analysis (PLS-DA) classifier.
    (See pls.py for full docstring)
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "standard",
        "tags": ["pls", "discriminant-analysis", "classification"],
    }

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingClassifier)
    _estimator_type = "classifier"

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes == 2:
            self.encoder_ = LabelEncoder()
            y_encoded = self.encoder_.fit_transform(y)
        else:
            self.encoder_ = OneHotEncoder(sparse_output=False, dtype=float)
            y_encoded = self.encoder_.fit_transform(y.reshape(-1, 1))
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=n_comp)
        self.pls_.fit(X, y_encoded)
        return self

    def predict(self, X):
        check_is_fitted(self, ['pls_', 'classes_'])
        X = np.asarray(X)
        y_pred_raw = self.pls_.predict(X)
        if len(self.classes_) == 2:
            y_pred = (y_pred_raw > 0.5).astype(int).ravel()
            return self.encoder_.inverse_transform(y_pred)
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)
            return self.encoder_.categories_[0][y_pred]

    def predict_proba(self, X):
        """Class membership scores as probabilities consistent with ``predict``.

        PLS responses are unbounded. A binary response is clipped to [0, 1];
        multiclass responses are clipped at zero and each row is normalized to
        sum to one, which keeps the ``predict`` class most probable. A row with
        no positive response puts its whole mass on that class.
        """
        check_is_fitted(self, ['pls_', 'classes_'])
        X = np.asarray(X)
        y_pred_raw = self.pls_.predict(X)
        if len(self.classes_) == 2:
            positive = np.clip(np.asarray(y_pred_raw, dtype=float).ravel(), 0.0, 1.0)
            return np.column_stack([1.0 - positive, positive])
        scores = np.clip(np.asarray(y_pred_raw, dtype=float), 0.0, None)
        totals = scores.sum(axis=1, keepdims=True)
        winners = np.eye(scores.shape[1])[np.argmax(y_pred_raw, axis=1)]
        return np.where(totals > 0, scores / np.where(totals > 0, totals, 1.0), winners)

    def get_params(self, deep=True):
        return {"n_components": self.n_components}

    def set_params(self, **params):
        super().set_params(**params)
        return self
