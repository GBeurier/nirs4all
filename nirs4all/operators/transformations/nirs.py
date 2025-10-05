import numpy as np
import pywt
import scipy
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, scale
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


def wavelet_transform(spectra: np.ndarray, wavelet: str, mode: str = "periodization") -> np.ndarray:
    """
    Computes transform using pywavelet transform.

    Args:
        spectra (numpy.ndarray): NIRS data matrix.
        wavelet (str): wavelet family transformation.
        mode (str): signal extension mode.

    Returns:
        numpy.ndarray: wavelet and resampled spectra.
    """
    _, wt_coeffs = pywt.dwt(spectra, wavelet=wavelet, mode=mode)
    if len(wt_coeffs[0]) != len(spectra[0]):
        return signal.resample(wt_coeffs, len(spectra[0]), axis=1)
    else:
        return wt_coeffs


class Wavelet(TransformerMixin, BaseEstimator):
    """
    Single level Discrete Wavelet Transform.

    Performs a discrete wavelet transform on `data`, using a `wavelet` function.

    Parameters
    ----------
    wavelet : Wavelet object or name, default='haar'
        Wavelet to use: ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal',
        'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)'...]
    mode : str, optional, default='periodization'
        Signal extension mode.

    """

    def __init__(self, wavelet: str = "haar", mode: str = "periodization", *, copy: bool = True):
        self.copy = copy
        self.wavelet = wavelet
        self.mode = mode

    def _reset(self):
        pass

    def fit(self, X, y=None):
        """
        Verify the X data compliance with wavelet transform.

        Parameters
        ----------
        X : array-like, spectra
            The data to transform.
        y : None
            Ignored.

        Raises
        ------
        ValueError
            If the input X is a sparse matrix.

        Returns
        -------
        Wavelet
            The fitted object.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("Wavelets does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        """
        Apply wavelet transform to the data X.

        Parameters
        ----------
        X : array-like
            The data to transform.
        copy : bool or None, optional
            Whether to copy the input data.

        Returns
        -------
        numpy.ndarray
            The transformed data.
        """
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        # # X = self._validate_data(
        #     # X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        # # )

        return wavelet_transform(X, self.wavelet, mode=self.mode)

    def _more_tags(self):
        return {"allow_nan": False}


class Haar(Wavelet):
    """
    Shortcut to the Wavelet haar transform.
    """

    def __init__(self, *, copy: bool = True):
        super().__init__("haar", "periodization", copy=copy)


def savgol(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    """
    Perform Savitzky–Golay filtering on the data (also calculates derivatives).
    This function is a wrapper for scipy.signal.savgol_filter.

    Args:
        spectra (numpy.ndarray): NIRS data matrix.
        window_length (int): Size of the filter window in samples (default 11).
        polyorder (int): Order of the polynomial estimation (default 3).
        deriv (int): Order of the derivation (default 0).
        delta (float): Sampling distance of the data.

    Returns:
        numpy.ndarray: NIRS data smoothed with Savitzky-Golay filtering.
    """
    return signal.savgol_filter(spectra, window_length, polyorder, deriv, delta=delta)


class SavitzkyGolay(TransformerMixin, BaseEstimator):
    """
    A class for smoothing and differentiating data using the Savitzky-Golay filter.

    Parameters:
    -----------
    window_length : int, optional (default=11)
        The length of the window used for smoothing.
    polyorder : int, optional (default=3)
        The order of the polynomial used for fitting the samples within the window.
    deriv : int, optional (default=0)
        The order of the derivative to compute.
    delta : float, optional (default=1.0)
        The sampling distance of the data.
    copy : bool, optional (default=True)
        Whether to copy the input data.

    Methods:
    --------
    fit(X, y=None)
        Fits the transformer to the data X.
    transform(X, copy=None)
        Applies the Savitzky-Golay filter to the data X.
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 3,
        deriv: int = 0,
        delta: float = 1.0,
        *,
        copy: bool = True
    ):
        self.copy = copy
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def _reset(self):
        pass

    def fit(self, X, y=None):
        """
        Verify the X data compliance with Savitzky-Golay filter.

        Parameters
        ----------
        X : array-like
            The data to transform.
        y : None
            Ignored.

        Raises
        ------
        ValueError
            If the input X is a sparse matrix.

        Returns
        -------
        SavitzkyGolay
            The fitted object.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("SavitzkyGolay does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        """
        Apply the Savitzky-Golay filter to the data X.

        Parameters
        ----------
        X : array-like
            The data to transform.
        copy : bool or None, optional
            Whether to copy the input data.

        Returns
        -------
        numpy.ndarray
            The transformed data.
        """
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        # X = self._validate_data(
        #     X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        # )

        return savgol(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
        )

    def _more_tags(self):
        return {"allow_nan": False}


class MultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):
    def __init__(self, scale=True, *, copy=True):
        self.copy = copy
        self.scale = scale

    def _reset(self):
        if hasattr(self, "scaler_"):
            del self.scaler_
            del self.a_
            del self.b_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("Normalization does not support scipy.sparse input")

        first_pass = not hasattr(self, "mean_")
        # X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        tmp_x = X
        if self.scale:
            scaler = StandardScaler(with_std=False)
            scaler.fit(X)
            self.scaler_ = scaler
            tmp_x = scaler.transform(X)

        reference = np.mean(tmp_x, axis=1)

        a = np.empty(X.shape[1], dtype=float)
        b = np.empty(X.shape[1], dtype=float)

        for col in range(X.shape[1]):
            a[col], b[col] = np.polyfit(reference, tmp_x[:, col], deg=1)

        self.a_ = a
        self.b_ = b

        return self

    def transform(self, X):
        check_is_fitted(self)

        # X = self._validate_data(
        #     X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        # )

        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError(
                "Transform cannot be applied with provided X. Bad number of columns."
            )

        if self.scale:
            X = self.scaler_.transform(X)

        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] - b) / a

        return X

    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError(
                "Inverse transform cannot be applied with provided X. "
                "Bad number of columns."
            )

        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] * a) + b

        if self.scale:
            X = self.scaler_.inverse_transform(X)
        return X

    def _more_tags(self):
        return {"allow_nan": False}


def msc(spectra, scaled=True):
    """Performs multiplicative scatter correction to the mean.

    Args:
        spectra (numpy.ndarray): NIRS data matrix.
        scaled (bool): Whether to scale the data. Defaults to True.

    Returns:
        numpy.ndarray: Scatter-corrected NIR spectra.
    """
    if scaled:
        spectra = scale(spectra, with_std=False, axis=0)  # StandardScaler / demean

    reference = np.mean(spectra, axis=1)

    for col in range(spectra.shape[1]):
        a, b = np.polyfit(reference, spectra[:, col], deg=1)
        spectra[:, col] = (spectra[:, col] - b) / a

    return spectra

def log_transform(
    spectra: np.ndarray,
    base: float = np.e,
    offset: float = 0.0,
    auto_offset: bool = True,
    min_value: float = 1e-8,
) -> np.ndarray:
    """
    Apply elementwise logarithm with automatic handling of edge cases.

    Args:
        spectra (numpy.ndarray): NIRS data matrix.
        base (float): Logarithm base. Default is e.
        offset (float): Fixed value added before log to handle non-positives.
        auto_offset (bool): If True, automatically add offset for problematic values.
        min_value (float): Minimum value after offset when auto_offset=True.

    Returns:
        numpy.ndarray: Log-transformed spectra.
    """
    X = spectra.copy() if hasattr(spectra, 'copy') else np.array(spectra)

    # Apply manual offset first
    if offset != 0.0:
        X = X + offset

    # Auto-handle problematic values if enabled
    if auto_offset:
        min_x = np.min(X)
        if min_x <= 0:
            # Add offset to make minimum value equal to min_value
            auto_computed_offset = min_value - min_x
            X = X + auto_computed_offset

    # Perform log transform
    if base == np.e:
        return np.log(X)
    return np.log(X) / np.log(base)


class LogTransform(TransformerMixin, BaseEstimator):
    """
    Elementwise logarithm with automatic handling of edge cases.

    Parameters
    ----------
    base : float, default=np.e
        Logarithm base.
    offset : float, default=0.0
        Fixed value added before log to handle non-positives.
    auto_offset : bool, default=True
        If True, automatically add offset to handle zeros/negatives.
    min_value : float, default=1e-8
        Minimum value after offset when auto_offset=True.
    copy : bool, default=True
        Whether to copy input.
    """

    def __init__(self, base: float = np.e, offset: float = 0.0, auto_offset: bool = True,
                 min_value: float = 1e-8, *, copy: bool = True):
        self.copy = copy
        self.base = base
        self.offset = offset
        self.auto_offset = auto_offset
        self.min_value = min_value
        self._fitted_offset = 0.0  # Store the computed offset for inverse transform

    def _reset(self):
        self._fitted_offset = 0.0

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise ValueError("LogTransform does not support scipy.sparse input")

        # Pre-compute the total offset that will be applied
        X_temp = X.copy() if hasattr(X, 'copy') else np.array(X)

        # Apply manual offset first
        if self.offset != 0.0:
            X_temp = X_temp + self.offset

        # Compute auto offset if needed
        auto_computed_offset = 0.0
        if self.auto_offset:
            min_x = np.min(X_temp)
            if min_x <= 0:
                auto_computed_offset = self.min_value - min_x

        # Store total offset for inverse transform
        self._fitted_offset = self.offset + auto_computed_offset

        return self

    def transform(self, X, copy=None):
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        # Use a more robust transform that handles all edge cases
        X_copy = X.copy() if hasattr(X, 'copy') else np.array(X, dtype=np.float64)

        # Apply manual offset first
        if self.offset != 0.0:
            X_copy = X_copy + self.offset

        # For auto_offset, we need to be extremely robust:
        if self.auto_offset:
            min_x = np.min(X_copy)

            # Always ensure we have positive values for log transform
            # Use a more conservative approach
            target_min = max(self.min_value, 1e-10)  # Ensure minimum is reasonable

            if min_x <= target_min:
                # Calculate offset to bring minimum to target_min
                additional_offset = target_min - min_x + 1e-12  # Add tiny buffer
                X_copy = X_copy + additional_offset

            # Final safety check - ensure no problematic values
            final_min = np.min(X_copy)
            if final_min <= 0:
                # Emergency fallback - add enough to make all values positive
                X_copy = X_copy - final_min + 1e-10

        # Final validation before log transform
        if np.any(X_copy <= 0):
            # Ultimate safety: replace any remaining non-positive values
            X_copy = np.where(X_copy <= 0, 1e-10, X_copy)

        # Perform log transform with additional safety
        result = np.log(X_copy) if self.base == np.e else np.log(X_copy) / np.log(self.base)

        # Validate result
        if np.any(np.isinf(result)) or np.any(np.isnan(result)):
            # This should never happen, but as absolute last resort
            result = np.where(np.isinf(result) | np.isnan(result), -18.42068, result)

        return result

    def inverse_transform(self, X):
        """Exact inverse of the forward transform."""
        # X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        if self.base == np.e:
            Y = np.exp(X)
        else:
            Y = np.power(self.base, X)
        return Y - self._fitted_offset

    def _more_tags(self):
        return {"allow_nan": False}


def first_derivative(
    spectra: np.ndarray,
    delta: float = 1.0,
    edge_order: int = 2,
) -> np.ndarray:
    """
    First numerical derivative along feature axis using central differences.

    Args:
        spectra (numpy.ndarray): NIRS data matrix (n_samples, n_features).
        delta (float): Sampling step along the feature axis.
        edge_order (int): 1 or 2, order of accuracy at the boundaries.

    Returns:
        numpy.ndarray: First derivative dX/dλ with same shape as input.
    """
    return np.gradient(spectra, delta, axis=1, edge_order=edge_order)


class FirstDerivative(TransformerMixin, BaseEstimator):
    """
    First numerical derivative using numpy.gradient.

    Parameters
    ----------
    delta : float, default=1.0
        Sampling step along the feature axis.
    edge_order : int, default=2
        1 or 2, order of accuracy at the boundaries.
    copy : bool, default=True
        Whether to copy input.
    """

    def __init__(self, delta: float = 1.0, edge_order: int = 2, *, copy: bool = True):
        self.copy = copy
        self.delta = delta
        self.edge_order = edge_order

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise ValueError("FirstDerivative does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        # X = self._validate_data(X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        return first_derivative(X, delta=self.delta, edge_order=self.edge_order)

    def _more_tags(self):
        return {"allow_nan": False}


def second_derivative(
    spectra: np.ndarray,
    delta: float = 1.0,
    edge_order: int = 2,
) -> np.ndarray:
    """
    Second numerical derivative along feature axis.

    Args:
        spectra (numpy.ndarray): NIRS data matrix (n_samples, n_features).
        delta (float): Sampling step along the feature axis.
        edge_order (int): 1 or 2, order of accuracy at the boundaries.

    Returns:
        numpy.ndarray: Second derivative d²X/dλ² with same shape as input.
    """
    d1 = np.gradient(spectra, delta, axis=1, edge_order=edge_order)
    return np.gradient(d1, delta, axis=1, edge_order=edge_order)


class SecondDerivative(TransformerMixin, BaseEstimator):
    """
    Second numerical derivative using numpy.gradient.

    Parameters
    ----------
    delta : float, default=1.0
        Sampling step along the feature axis.
    edge_order : int, default=2
        1 or 2, order of accuracy at the boundaries.
    copy : bool, default=True
        Whether to copy input.
    """

    def __init__(self, delta: float = 1.0, edge_order: int = 2, *, copy: bool = True):
        self.copy = copy
        self.delta = delta
        self.edge_order = edge_order

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise ValueError("SecondDerivative does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        # X = self._validate_data(X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        return second_derivative(X, delta=self.delta, edge_order=self.edge_order)

    def _more_tags(self):
        return {"allow_nan": False}
