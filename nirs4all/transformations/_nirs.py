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

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

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

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

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
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

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

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

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
