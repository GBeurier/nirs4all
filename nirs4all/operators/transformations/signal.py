import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class Baseline(TransformerMixin, BaseEstimator):
    """
    Removes baseline (mean) from each spectrum.

    Parameters
    ----------
    copy : bool, optional
        Flag to indicate whether to make a copy of the object, by default True.
    """

    def __init__(self, *, copy=True):
        """
        Constructor for the Baseline class.

        Parameters
        ----------
        copy : bool, optional
            Flag to indicate whether to make a copy of the object, by default True.
        """

        self.copy = copy

    def _reset(self):
        if hasattr(self, "mean_"):
            del self.mean_

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted Baseline object.
        """

        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("Baseline does not support scipy.sparse input")

        first_pass = not hasattr(self, "mean_")
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        X = X - self.mean_
        return X

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X = X + self.mean_
        return X

    def _more_tags(self):
        return {"allow_nan": False}


def baseline(spectra):
    """
    Removes baseline (mean) from each spectrum.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.

    Returns
    -------
    numpy.ndarray
        Mean-centered NIRS data matrix.
    """

    return spectra - np.mean(spectra, axis=0)


def detrend(spectra, bp=0):
    """
    Perform spectral detrending to remove linear trend from data.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.
    bp : list, optional
        A sequence of break points. If given, an individual linear fit is performed for each part of data between two break points.
        Break points are specified as indices into data. Default is 0.

    Returns
    -------
    numpy.ndarray
        Detrended NIR spectra.
    """

    return signal.detrend(spectra, bp=bp)


class Detrend(TransformerMixin, BaseEstimator):
    """
    Perform spectral detrending to remove linear trend from data.

    Parameters
    ----------
    bp : int, optional
        Breakpoints for piecewise linear detrending. Default is 0.
    copy : bool, optional
        Whether to make a copy of the input data. Default is True.
    """

    def __init__(self, bp=0, *, copy=True):
        self.copy = copy
        self.bp = bp

    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        """

        pass

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """

        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        """
        Transform the data by removing linear trend.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        copy : bool or None, optional
            Whether to make a copy of the input data. If None, `self.copy` is used. Default is None.

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

        X = detrend(X, bp=self.bp)

        return X

    def _more_tags(self):
        """
        Get tags for the estimator.

        Returns
        -------
        dict
            Dictionary of tags for the estimator.
        """

        return {"allow_nan": False}


def gaussian(spectra, order=2, sigma=1):
    """
    Computes 1D gaussian filter using scipy.ndimage gaussian 1d filter.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.
    order : float, optional
        Order of the derivation.
    sigma : int, optional
        Sigma of the gaussian.

    Returns
    -------
    numpy.ndarray
        Gaussian NIR spectra.
    """

    return gaussian_filter1d(spectra, order=order, sigma=sigma)


class Gaussian(TransformerMixin, BaseEstimator):
    def __init__(self, order=2, sigma=1, *, copy=True):
        """
        Initialize Gaussian filter.

        Parameters
        ----------
        order : float, optional
            Order of the derivation.
        sigma : int, optional
            Sigma of the gaussian.
        copy : bool, default=True
            Whether to make a copy of the input data.
        """

        self.copy = copy
        self.order = order
        self.sigma = sigma

    def _reset(self):
        """
        Reset internal data-dependent state of the Gaussian filter.
        """

        pass

    def fit(self, X, y=None):
        """
        Fit the Gaussian filter.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if scipy.sparse.issparse(X):
            raise ValueError("SavitzkyGolay does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        """
        Transform the input data using the Gaussian filter.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        copy : bool, default=None
            Whether to make a copy of the input data.

        Returns
        -------
        numpy.ndarray
            Transformed data.
        """

        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=np.float64, estimator=self
        )

        X = gaussian(X, order=self.order, sigma=self.sigma)

        return X

    def _more_tags(self):
        """
        Provide additional tags for the Gaussian filter.
        """

        return {"allow_nan": False}
