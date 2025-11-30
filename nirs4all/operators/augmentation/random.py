import numpy as np
import operator

from .abc_augmenter import Augmenter


class Rotate_Translate(Augmenter):
    """
    Class for rotating and translating data augmentation.

    Vectorized implementation that processes all samples in batch.

    Parameters
    ----------
    apply_on : str, optional
        Apply augmentation on "samples" or "global" data. Default is "samples".
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    copy : bool, optional
        If True, creates a copy of the input data. Default is True.
    p_range : int, optional
        Range for generating random slope values. Default is 2.
    y_factor : int, optional
        Scaling factor for the initial value. Default is 3.
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, p_range=2, y_factor=3):
        self.p_range = p_range
        self.y_factor = y_factor
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data by rotating and translating the signal.

        Vectorized implementation using NumPy broadcasting.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented, shape (n_samples, n_features).
        apply_on : str, optional
            Apply augmentation on "samples" or "global" data. Default is "samples".

        Returns
        -------
        ndarray
            Augmented data.
        """
        n_samples, n_features = X.shape

        # Pre-compute x_range once for all samples
        x_range = np.linspace(0, 1, n_features)  # (n_features,)

        if apply_on == "global":
            # Single deformation for all samples
            p1 = self.random_gen.uniform(-self.p_range / 5, self.p_range / 5)
            p2 = self.random_gen.uniform(-self.p_range / 5, self.p_range / 5)
            xI = self.random_gen.uniform(0, 1)
            yI = self.random_gen.uniform(0, max(0, np.max(X) / self.y_factor))

            # Vectorized angle computation
            mask = x_range <= xI  # (n_features,)
            distor = np.where(mask, p1 * (x_range - xI) + yI, p2 * (x_range - xI) + yI)
            increment = distor * np.std(X)  # (n_features,)
        else:
            # Generate all random parameters at once for all samples
            p1 = self.random_gen.uniform(-self.p_range / 5, self.p_range / 5, n_samples)  # (n_samples,)
            p2 = self.random_gen.uniform(-self.p_range / 5, self.p_range / 5, n_samples)  # (n_samples,)
            xI = self.random_gen.uniform(0, 1, n_samples)  # (n_samples,)

            # Compute yI for each sample based on max values
            max_vals = np.max(X, axis=1)  # (n_samples,)
            yI_upper = np.maximum(0, max_vals / self.y_factor)
            yI = self.random_gen.uniform(0, 1, n_samples) * yI_upper  # (n_samples,)

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


class Random_X_Operation(Augmenter):
    """
    Class for applying random operation on data augmentation.

    Parameters
    ----------
    apply_on : str, optional
        Apply augmentation on "features" or "samples" data. Default is "features".
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    copy : bool, optional
        If True, creates a copy of the input data. Default is True.
    operator_func : function, optional
        Operator function to be applied. Default is operator.mul.
    operator_range : tuple, optional
        Range for generating random values for the operator. Default is (0.97, 1.03).
    """

    def __init__(self, apply_on="global", random_state=None, *, copy=True, operator_func=operator.mul, operator_range=(0.97, 1.03)):
        self.operator_func = operator_func
        self.operator_range = operator_range
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="global"):
        """
        Augment the data by applying random operation.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented.
        apply_on : str, optional
            Apply augmentation on "features" or "samples" data. Default is "features".

        Returns
        -------
        ndarray
            Augmented data.
        """
        min_val = self.operator_range[0]
        interval = self.operator_range[1] - self.operator_range[0]

        if apply_on == "global":
            increment = self.random_gen.random(X.shape[-1]) * interval + min_val
        else:
            increment = self.random_gen.random(X.shape) * interval + min_val

        new_X = self.operator_func(X, increment)
        # Clip the augmented data within the float32 range
        new_X = np.clip(new_X, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        # Log the min and max values to help debug any potential overflow
        # print("Augmented Data Range:", new_X.min(), new_X.max())

        return new_X
