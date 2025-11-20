import warnings

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES


IdentityTransformer = FunctionTransformer


class StandardNormalVariate(TransformerMixin, BaseEstimator):
    """Standard Normal Variate (SNV) transformation.

    SNV is a row-wise normalization technique commonly used in spectroscopy
    to remove scatter effects. Each sample (row) is centered and scaled
    independently.

    For each sample: SNV = (X - mean(X)) / std(X)

    Parameters
    ----------
    axis : int, default=1
        Axis along which to compute mean and standard deviation.
        - axis=1: Row-wise (default, standard SNV behavior for spectroscopy)
        - axis=0: Column-wise (equivalent to StandardScaler)

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance.

    ddof : int, default=0
        Delta Degrees of Freedom for standard deviation calculation.

    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    Examples
    --------
    >>> from nirs4all.operators.transforms import StandardNormalVariate
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> snv = StandardNormalVariate()
    >>> X_transformed = snv.fit_transform(X)
    """

    def __init__(self, axis=1, with_mean=True, with_std=True, ddof=0, copy=True):
        self.axis = axis
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the StandardNormalVariate transformer.

        For SNV, this is a no-op as the transformation is computed
        independently for each sample.

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
        if scipy.sparse.issparse(X):
            raise TypeError("StandardNormalVariate does not support scipy.sparse input")

        # Validate input
        X = check_array(X, dtype=FLOAT_DTYPES, copy=False)

        # SNV is computed per sample, so no fitting is needed
        # But we validate the axis parameter
        if self.axis not in [0, 1]:
            raise ValueError(f"axis must be 0 or 1, got {self.axis}")

        return self

    def transform(self, X):
        """Perform SNV transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        if scipy.sparse.issparse(X):
            raise TypeError("StandardNormalVariate does not support scipy.sparse input")

        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)

        if self.with_mean:
            mean = np.mean(X, axis=self.axis, keepdims=True)
            X = X - mean

        if self.with_std:
            std = np.std(X, axis=self.axis, ddof=self.ddof, keepdims=True)
            # Avoid division by zero
            std[std == 0] = 1.0
            X = X / std

        return X

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : None
            Ignored variable.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"allow_nan": False, "stateless": True}


class LocalStandardNormalVariate(TransformerMixin, BaseEstimator):
    """Local Standard Normal Variate (LSNV).

    Per-sample local normalization with a sliding window along features.
    For each sample and feature j:
        mean_w = mean(X[..., j-w//2 : j+w//2+1])
        std_w  = std (X[..., j-w//2 : j+w//2+1])
        X'[j]  = (X[j] - mean_w) / std_w

    Parameters
    ----------
    window : int, default=11
        Odd positive window size along features.
    pad_mode : {'reflect','edge','constant'}, default='reflect'
        Padding mode at boundaries.
    constant_values : float, default=0.0
        Used only if pad_mode='constant'.
    copy : bool, default=True
        If False, try in-place.

    Notes
    -----
    - Operates row-wise (axis=1). Input must be (n_samples, n_features).
    - std_w==0 → divide by 1 to avoid NaN.
    """

    def __init__(self, window=11, pad_mode="reflect", constant_values=0.0, copy=True):
        self.window = window
        self.pad_mode = pad_mode
        self.constant_values = constant_values
        self.copy = copy

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("LSNV does not support scipy.sparse input")
        X = check_array(X, dtype=FLOAT_DTYPES, copy=False)
        if X.ndim != 2:
            raise ValueError("LSNV expects 2D array (n_samples, n_features)")
        if not isinstance(self.window, int) or self.window <= 1 or self.window % 2 == 0:
            raise ValueError("window must be an odd integer > 1")
        if self.pad_mode not in {"reflect", "edge", "constant"}:
            raise ValueError("pad_mode must be 'reflect', 'edge', or 'constant'")
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X):
            raise TypeError("LSNV does not support scipy.sparse input")
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        n, m = X.shape
        w = self.window
        half = w // 2

        if self.pad_mode == "constant":
            pad_kwargs = dict(mode="constant", constant_values=self.constant_values)
        else:
            pad_kwargs = dict(mode=self.pad_mode)

        # pad along feature axis
        Xp = np.pad(X, ((0, 0), (half, half)), **pad_kwargs)

        # moving mean via cumsum
        csum = np.cumsum(Xp, axis=1, dtype=float)
        csum = np.pad(csum, ((0, 0), (1, 0)), mode="constant")  # align for window subtraction
        mov_mean = (csum[:, w:] - csum[:, :-w]) / w

        # moving variance via mean of squares
        Xp2 = Xp * Xp
        csum2 = np.cumsum(Xp2, axis=1, dtype=float)
        csum2 = np.pad(csum2, ((0, 0), (1, 0)), mode="constant")
        mov_mean2 = (csum2[:, w:] - csum2[:, :-w]) / w
        mov_var = np.maximum(mov_mean2 - mov_mean * mov_mean, 0.0)
        mov_std = np.sqrt(mov_var, dtype=float)
        mov_std[mov_std == 0] = 1.0

        # normalize relative to local stats
        X_norm = (X - mov_mean) / mov_std
        return X_norm

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"allow_nan": False, "stateless": True}


class RobustStandardNormalVariate(TransformerMixin, BaseEstimator):
    """Robust Standard Normal Variate (RSNV).

    Per-sample robust centering and scaling using median and MAD:
        med = median(X, axis=1, keepdims=True)
        mad = median(|X - med|, axis=1, keepdims=True)
        X'  = (X - med) / (k * mad)

    Parameters
    ----------
    axis : int, default=1
        1 for row-wise (spectroscopy default). 0 for column-wise.
    with_center : bool, default=True
        If True, subtract median.
    with_scale : bool, default=True
        If True, divide by k * MAD.
    k : float, default=1.4826
        Consistency constant to make MAD a robust estimator of std
        for Gaussian data.
    copy : bool, default=True
        If False, try in-place.

    Notes
    -----
    - MAD==0 → divide by 1 to avoid NaN.
    """

    def __init__(self, axis=1, with_center=True, with_scale=True, k=1.4826, copy=True):
        self.axis = axis
        self.with_center = with_center
        self.with_scale = with_scale
        self.k = k
        self.copy = copy

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("RSNV does not support scipy.sparse input")

        X = check_array(X, dtype=FLOAT_DTYPES, copy=False)
        if self.axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X):
            raise TypeError("RSNV does not support scipy.sparse input")

        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)

        # choose axis and keepdims for broadcasting
        keep = dict(axis=self.axis, keepdims=True)

        if self.with_center:
            med = np.median(X, **keep)
            X = X - med

        if self.with_scale:
            mad = np.median(np.abs(X), **keep)
            scale = self.k * mad
            scale[scale == 0] = 1.0
            X = X / scale

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"allow_nan": False, "stateless": True}





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
        # # X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

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
        # X = self._validate_data(X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        # X = self._validate_data(X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)

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

        # X = self._validate_data(
        #     X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        # )

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
        # X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)
        # X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)

        # X = self._validate_data(
        #     X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        # )

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
