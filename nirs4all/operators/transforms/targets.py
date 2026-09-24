from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class IntegerKBinsDiscretizer(TransformerMixin, BaseEstimator):
    """KBinsDiscretizer qui retourne des entiers au lieu de floats"""

    _webapp_meta = {
        "category": "scaling",
        "tier": "standard",
        "tags": ["discretization", "binning", "target-processing", "classification"],
    }

    def __init__(self, n_bins=5, encode='ordinal', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)

    def fit(self, X, y=None):
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)
        self.discretizer.fit(X)
        return self

    def transform(self, X):
        result = self.discretizer.transform(X)
        return result.astype(np.int32)

    def inverse_transform(self, X):
        return self.discretizer.inverse_transform(X)

class RangeDiscretizer(TransformerMixin, BaseEstimator):

    _webapp_meta = {
        "category": "scaling",
        "tier": "standard",
        "tags": ["discretization", "binning", "target-processing", "range-based"],
    }

    def __init__(self, bins=5):
        # Store the original bins as received (could be int, list, array, etc.)
        self.bins = bins
        # Convert to numpy array for internal use. An int default simply means
        # "no edges configured yet" and produces an empty edge array so that
        # instantiation with defaults succeeds (edges would normally be supplied
        # explicitly at pipeline-construction time).
        if isinstance(bins, int):
            self._bins_array = np.array([], dtype=float)
            self.n_bins = bins
        else:
            self._bins_array = np.array(bins)
            self.n_bins = len(bins) + 1

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Return the original bins (not the numpy array) for proper cloning
        return {'bins': self.bins}

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if key == 'bins':
                self.bins = value
                if isinstance(value, int):
                    self._bins_array = np.array([], dtype=float)
                    self.n_bins = value
                else:
                    self._bins_array = np.array(value)
                    self.n_bins = len(value) + 1
            else:
                setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("RangeDiscretizer requires non-empty finite targets")
        if isinstance(self.bins, Integral) and not isinstance(self.bins, bool):
            if self.bins < 1:
                raise ValueError("bins must be a positive integer or increasing finite edges")
            lower, upper = float(values.min()), float(values.max())
            if lower == upper:
                self._bins_array = np.array([], dtype=float)
                self._centers_ = np.array([lower])
            else:
                edges = np.linspace(lower, upper, self.bins + 1)
                self._bins_array = edges[1:-1]
                self._centers_ = (edges[:-1] + edges[1:]) / 2
        else:
            self._bins_array = np.asarray(self.bins, dtype=float)
            if self._bins_array.ndim != 1 or not np.isfinite(self._bins_array).all() or np.any(np.diff(self._bins_array) <= 0):
                raise ValueError("bins must be a positive integer or increasing finite edges")
            if self._bins_array.size == 0:
                self._centers_ = np.array([values.mean()])
            else:
                # Preserve the established representatives for explicit edges.
                self._centers_ = np.concatenate([
                    self._bins_array[:1] - 1,
                    (self._bins_array[:-1] + self._bins_array[1:]) / 2,
                    self._bins_array[-1:] + 1,
                ])
        self.n_bins = len(self._centers_)
        return self

    def transform(self, X):
        X = np.asarray(X).flatten()
        result = np.digitize(X, self._bins_array, right=False)
        return result.reshape(-1, 1).astype(np.int32)

    def inverse_transform(self, X):
        X = np.asarray(X).flatten()
        from sklearn.utils.validation import check_is_fitted
        # Existing archives store explicit edges without learned centers.
        if not hasattr(self, "_centers_") and self._bins_array.size:
            self._centers_ = np.concatenate([
                self._bins_array[:1] - 1,
                (self._bins_array[:-1] + self._bins_array[1:]) / 2,
                self._bins_array[-1:] + 1,
            ])
        check_is_fitted(self, "_centers_")
        if not np.isfinite(X).all() or np.any(np.floor(X) != X) or np.any(X < 0) or np.any(self.n_bins <= X):
            raise ValueError("RangeDiscretizer inverse_transform requires valid integer class labels; use a classifier after discretizing targets")
        result = self._centers_[X.astype(int)]
        return result.reshape(-1, 1)

    def __sklearn_clone__(self):
        """Custom cloning method for sklearn compatibility."""
        return RangeDiscretizer(bins=self.bins)
