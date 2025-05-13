import warnings

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES


IdentityTransformer = FunctionTransformer
RobustNormalVariate = RobustScaler
StandardNormalVariate = StandardScaler


class Normalize(TransformerMixin, BaseEstimator):
    """Normalize spectrum using either custom range of linalg normalization

    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, -1)
        Desired range of transformed data. If range min and max equals -1, linalg
        normalization is applied, otherwise user defined normalization
        is applied

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    """

    def __init__(self, feature_range=(-1, 1), *, copy=True):
        self.copy = copy
        self.feature_range = feature_range
        self.user_defined = feature_range[0] != -1 or feature_range[1] != 1

    def _reset(self):
        if hasattr(self, "min_"):
            del self.min_
            del self.max_
            del self.f_

        if hasattr(self, "linalg_norm_"):
            del self.linalg_norm_

    def fit(self, X, y=None):
        """Fit the Normalize transformer on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Perform incremental fit on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        feature_range = self.feature_range
        if self.user_defined and feature_range[0] > feature_range[1]:
            warnings.warn(
                f"Minimum of desired feature range should be smaller than maximum. Got {feature_range}",
                SyntaxWarning,
            )

        if self.user_defined and feature_range[0] == feature_range[1]:
            raise ValueError(
                "Feature range is not correctly defined. Got %s." % str(feature_range)
            )

        if scipy.sparse.issparse(X):
            raise TypeError("Normalization does not support scipy.sparse input")

        first_pass = not hasattr(self, "min_")
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        if self.user_defined:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            imin = self.feature_range[0]
            imax = self.feature_range[1]
            self.f_ = (imax - imin) / (self.max_ - self.min_)
        else:
            self.linalg_norm_ = np.linalg.norm(X, axis=0)
        return self

    def transform(self, X):
        """Transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)

        if self.user_defined:
            imin = self.feature_range[0]
            f = self.f_
            X = imin + f * (X - self.min_)
        else:
            X = X / self.linalg_norm_

        return X

    def inverse_transform(self, X):
        """Transform the normalized data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The normalized data to be transformed back.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The inverse transformed data.
        """
        check_is_fitted(self)
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        if self.user_defined:
            imin = self.feature_range[0]
            f = self.f_
            X = (X - imin) / f + self.min_
        else:
            X = X * self.linalg_norm_

        return X

    def _more_tags(self):
        return {"allow_nan": False}


def norml(spectra, feature_range=(-1, 1)):
    """
    Perform spectral normalization with user-defined limits.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.
    feature_range : tuple (min, max), default=(-1, 1)
        Desired range of transformed data. If range min and max equals -1, linalg
        normalization is applied; otherwise, user bounds-defined normalization
        is applied.

    Returns
    -------
    spectra : numpy.ndarray
        Normalized NIR spectra.
    """
    if feature_range[0] != -1 and feature_range[1] != 1:
        imin = feature_range[0]
        imax = feature_range[1]
        if imin > imax:
            warnings.warn(
                "Minimum of desired feature range should be smaller than maximum. "
                f"Got {feature_range}.",
                SyntaxWarning,
            )
        if imin == imax:
            raise ValueError(
                f"Feature range is not correctly defined. Got {feature_range}."
            )

        f = (imax - imin) / (np.max(spectra) - np.min(spectra))
        n = spectra.shape
        arr = np.empty((0, n[0]), dtype=float)  # create empty array for spectra
        for i in range(0, n[1]):
            d = spectra[:, i]
            dnorm = imin + f * d
            arr = np.append(arr, [dnorm], axis=0)
        return np.transpose(arr)
    else:
        return spectra / np.linalg.norm(spectra, axis=0)


class Derivate(TransformerMixin, BaseEstimator):
    def __init__(self, order=1, delta=1, copy=True):
        self.copy = copy
        self.order = order
        self.delta = delta

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise ValueError("SavitzkyGolay does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        for n in range(self.order):
            X = np.gradient(X, self.delta, axis=0)

        return X

    def _more_tags(self):
        return {"allow_nan": False}


def derivate(spectra, order=1, delta=1):
    """
    Computes Nth order derivatives with the desired spacing using numpy.gradient.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.
    order : float, optional
        Order of the derivation, by default 1.
    delta : int, optional
        Delta of the derivative (in samples), by default 1.

    Returns
    -------
    spectra : numpy.ndarray
        Derived NIR spectra.
    """
    for n in range(order):
        spectra = np.gradient(spectra, delta, axis=0)
    return spectra


class SimpleScale(TransformerMixin, BaseEstimator):
    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        if hasattr(self, "min_"):
            del self.min_
            del self.max_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("Normalization does not support scipy.sparse input")

        first_pass = not hasattr(self, "min_")
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        X = (X - self.min_) / (self.max_ - self.min_)

        return X

    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        f = self.max_ - self.min_
        X = (X * f) + self.min_

        return X

    def _more_tags(self):
        return {"allow_nan": False}


def spl_norml(spectra):
    """
    Perform simple spectral normalization.

    Parameters
    ----------
    spectra : numpy.ndarray
        NIRS data matrix.

    Returns
    -------
    spectra : numpy.ndarray
        Normalized NIR spectra.
    """
    min_ = np.min(spectra, axis=0)
    max_ = np.max(spectra, axis=0)
    return (spectra - min_) / (max_ - min_)
