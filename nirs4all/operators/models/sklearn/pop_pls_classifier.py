"""POP-PLS Discriminant Analysis classifier for nirs4all.

Wraps POPPLSRegressor for classification tasks using the PLS-DA approach:
one-hot encode targets, fit PLS regression on encoded targets, predict
class by argmax of regression outputs, and provide calibrated probabilities
via softmax normalization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .aom_pls import LinearOperator
from .pop_pls import POPPLSRegressor


class POPPLSClassifier(BaseEstimator, ClassifierMixin):
    """POP-PLS Discriminant Analysis classifier.

    Combines per-component operator selection (POP-PLS) with PLS-DA
    classification. Each PLS component independently selects the best
    preprocessing operator, while classification is handled via one-hot
    encoded targets and argmax prediction.

    Probabilities are computed via softmax normalization of raw PLS predictions,
    providing better-calibrated outputs than raw regression values.

    Parameters
    ----------
    n_components : int, default=15
        Maximum number of PLS components.
    operator_bank : list of LinearOperator or None, default=None
        Operator bank. If None, uses default_operator_bank().
    n_orth : int, default=0
        Number of OPLS orthogonal components to pre-filter.
    center : bool, default=True
        Whether to center X.
    scale : bool, default=False
        Whether to scale X per column.
    auto_select : bool, default=True
        If True, automatically select the optimal number of components
        via GCV or validation data.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    pop_ : POPPLSRegressor
        Fitted POP-PLS regressor on encoded targets.
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "advanced",
        "tags": ["pls", "pop-pls", "classification", "discriminant-analysis", "preprocessing"],
    }

    _estimator_type = "classifier"

    def __init__(
        self,
        n_components: int = 15,
        operator_bank: list[LinearOperator] | None = None,
        n_orth: int = 0,
        center: bool = True,
        scale: bool = False,
        auto_select: bool = True,
    ):
        self.n_components = n_components
        self.operator_bank = operator_bank
        self.n_orth = n_orth
        self.center = center
        self.scale = scale
        self.auto_select = auto_select

    def fit(self, X, y, X_val=None, y_val=None) -> "POPPLSClassifier":
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.
        X_val : array-like or None
            Validation data for operator/prefix selection.
        y_val : array-like or None
            Validation labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            self.encoder_ = LabelEncoder()
            y_encoded = self.encoder_.fit_transform(y).astype(np.float64)
        else:
            self.encoder_ = OneHotEncoder(sparse_output=False, dtype=np.float64)
            y_encoded = self.encoder_.fit_transform(y.reshape(-1, 1))

        # Encode validation labels if provided
        y_val_encoded = None
        if y_val is not None:
            y_v = np.asarray(y_val).ravel()
            if n_classes == 2:
                y_val_encoded = self.encoder_.transform(y_v).astype(np.float64)
            else:
                y_val_encoded = self.encoder_.transform(y_v.reshape(-1, 1))

        self.pop_ = POPPLSRegressor(
            n_components=self.n_components,
            operator_bank=self.operator_bank,
            n_orth=self.n_orth,
            center=self.center,
            scale=self.scale,
            auto_select=self.auto_select,
        )
        self.pop_.fit(X, y_encoded, X_val=X_val, y_val=y_val_encoded)
        return self

    def predict(self, X) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y_raw = self.pop_.predict(X)
        if len(self.classes_) == 2:
            y_pred = (y_raw > 0.5).astype(int).ravel()
            return self.encoder_.inverse_transform(y_pred)
        else:
            y_pred = np.argmax(y_raw, axis=1)
            return self.encoder_.categories_[0][y_pred]

    def predict_proba(self, X) -> NDArray:
        """Predict class probabilities using softmax normalization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X = np.asarray(X, dtype=np.float64)
        y_raw = self.pop_.predict(X)
        if len(self.classes_) == 2:
            if y_raw.ndim == 1:
                p1 = np.clip(y_raw, 0, 1)
            else:
                p1 = np.clip(y_raw.ravel(), 0, 1)
            return np.column_stack([1.0 - p1, p1])
        else:
            # Softmax normalization for calibrated probabilities
            exp_y = np.exp(y_raw - np.max(y_raw, axis=1, keepdims=True))
            return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def get_block_weights(self) -> NDArray:
        """Get per-component block gating weights from underlying POP-PLS."""
        return self.pop_.get_block_weights()

    def get_preprocessing_report(self) -> list[dict]:
        """Get preprocessing selection report from underlying POP-PLS."""
        return self.pop_.get_preprocessing_report()

    def get_component_operators(self) -> list[str]:
        """Get operator name selected for each component."""
        return self.pop_.get_component_operators()

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_components": self.n_components,
            "operator_bank": self.operator_bank,
            "n_orth": self.n_orth,
            "center": self.center,
            "scale": self.scale,
            "auto_select": self.auto_select,
        }

    def set_params(self, **params) -> "POPPLSClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"POPPLSClassifier(n_components={self.n_components}, auto_select={self.auto_select})"
