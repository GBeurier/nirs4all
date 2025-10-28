import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

class IntegerKBinsDiscretizer(BaseEstimator, TransformerMixin):
    """KBinsDiscretizer qui retourne des entiers au lieu de floats"""

    def __init__(self, n_bins=5, encode='ordinal', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)

    def fit(self, X, y=None):
        self.discretizer.fit(X)
        return self

    def transform(self, X):
        result = self.discretizer.transform(X)
        return result.astype(np.int32)

    def inverse_transform(self, X):
        return self.discretizer.inverse_transform(X)


class RangeDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, bins):
        # Store the original bins as received (could be list, array, etc.)
        self.bins = bins
        # Convert to numpy array for internal use
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
                self._bins_array = np.array(value)
                self.n_bins = len(value) + 1
            else:
                setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        # Ensure _bins_array is properly initialized
        if not hasattr(self, '_bins_array'):
            self._bins_array = np.array(self.bins)
        return self

    def transform(self, X):
        X = np.asarray(X).flatten()
        result = np.digitize(X, self._bins_array, right=False)
        return result.reshape(-1, 1).astype(np.int32)

    def inverse_transform(self, X):
        X = np.asarray(X).flatten()

        # Créer les centres d'intervalles
        extended_bins = np.concatenate([[-np.inf], self._bins_array, [np.inf]])
        centers = []

        for i in range(len(extended_bins) - 1):
            left = extended_bins[i]
            right = extended_bins[i + 1]

            if left == -np.inf:
                center = right - 1.0  # Arbitraire pour la première classe
            elif right == np.inf:
                center = left + 1.0   # Arbitraire pour la dernière classe
            else:
                center = (left + right) / 2

            centers.append(center)

        # Mapper les classes vers leurs centres
        result = np.array([centers[int(cls)] for cls in X])
        return result.reshape(-1, 1)

    def __sklearn_clone__(self):
        """Custom cloning method for sklearn compatibility."""
        return RangeDiscretizer(bins=self.bins)
