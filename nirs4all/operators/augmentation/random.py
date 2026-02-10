import numpy as np
import operator

from sklearn.base import TransformerMixin, BaseEstimator


class Rotate_Translate(TransformerMixin, BaseEstimator):
    """
    Class for rotating and translating data augmentation.

    Vectorized implementation that processes all samples in batch.

    Parameters
    ----------
    p_range : int, optional
        Range for generating random slope values. Default is 2.
    y_factor : int, optional
        Scaling factor for the initial value. Default is 3.
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    """

    _webapp_meta = {
        "category": "random",
        "tier": "standard",
        "tags": ["random", "rotate", "translate", "augmentation"],
    }

    def __init__(self, p_range=2, y_factor=3, random_state=None):
        self.p_range = p_range
        self.y_factor = y_factor
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Transform the data by rotating and translating the signal.

        Vectorized implementation using NumPy broadcasting.

        Parameters
        ----------
        X : ndarray
            Input data to be transformed, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Transformed data.
        """
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        n_samples, n_features = X.shape

        # Pre-compute x_range once for all samples
        x_range = np.linspace(0, 1, n_features)  # (n_features,)

        # Generate all random parameters at once for all samples
        p1 = rng.uniform(-self.p_range / 5, self.p_range / 5, n_samples)  # (n_samples,)
        p2 = rng.uniform(-self.p_range / 5, self.p_range / 5, n_samples)  # (n_samples,)
        xI = rng.uniform(0, 1, n_samples)  # (n_samples,)

        # Compute yI for each sample based on max values
        max_vals = np.max(X, axis=1)  # (n_samples,)
        yI_upper = np.maximum(0, max_vals / self.y_factor)
        yI = rng.uniform(0, 1, n_samples) * yI_upper  # (n_samples,)

        # Vectorized computation using broadcasting
        # x_range: (n_features,), xI: (n_samples,) -> broadcast to (n_samples, n_features)
        x_expanded = x_range[np.newaxis, :]  # (1, n_features)
        xI_expanded = xI[:, np.newaxis]  # (n_samples, 1)

        # Compute mask for each sample
        mask = x_expanded <= xI_expanded  # (n_samples, n_features)

        # Compute slopes for both branches using broadcasting
        p1_expanded = p1[:, np.newaxis]  # (n_samples, 1)
        p2_expanded = p2[:, np.newaxis]  # (n_samples, 1)
        yI_expanded = yI[:, np.newaxis]  # (n_samples, 1)

        # Vectorized angle computation for all samples at once
        distor = np.where(
            mask,
            p1_expanded * (x_expanded - xI_expanded) + yI_expanded,
            p2_expanded * (x_expanded - xI_expanded) + yI_expanded
        )  # (n_samples, n_features)

        # Multiply by per-sample std
        stds = np.std(X, axis=1, keepdims=True)  # (n_samples, 1)
        increment = distor * stds  # (n_samples, n_features)

        return X + increment


class Random_X_Operation(TransformerMixin, BaseEstimator):
    """
    Class for applying random operation on data augmentation.

    Parameters
    ----------
    operator_func : function, optional
        Operator function to be applied. Default is operator.mul.
    operator_range : tuple, optional
        Range for generating random values for the operator. Default is (0.97, 1.03).
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    """

    _webapp_meta = {
        "category": "random",
        "tier": "standard",
        "tags": ["random", "operation", "multiplicative", "augmentation"],
    }

    def __init__(self, operator_func=operator.mul, operator_range=(0.97, 1.03), random_state=None):
        self.operator_func = operator_func
        self.operator_range = operator_range
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Transform the data by applying random operation.

        Parameters
        ----------
        X : ndarray
            Input data to be transformed.

        Returns
        -------
        ndarray
            Transformed data.
        """
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        min_val = self.operator_range[0]
        interval = self.operator_range[1] - self.operator_range[0]

        increment = rng.random(X.shape) * interval + min_val

        new_X = self.operator_func(X, increment)
        # Clip the augmented data within the float32 range
        new_X = np.clip(new_X, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        return new_X
