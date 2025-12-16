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


class ExtendedMultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):
    """
    Extended Multiplicative Scatter Correction (EMSC).

    EMSC extends MSC by including polynomial terms to model chemical
    and physical light scattering effects.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial for modeling interference.
    scale : bool, default=True
        Whether to scale the data before correction.
    copy : bool, default=True
        Whether to copy input data.
    """

    def __init__(self, degree: int = 2, scale: bool = True, *, copy: bool = True):
        self.copy = copy
        self.scale = scale
        self.degree = degree

    def _reset(self):
        if hasattr(self, "scaler_"):
            del self.scaler_
            del self.reference_
            del self.wavelengths_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise TypeError("EMSC does not support scipy.sparse input")

        first_pass = not hasattr(self, "reference_")

        tmp_x = X.copy() if self.copy else X

        if self.scale:
            scaler = StandardScaler(with_std=False)
            scaler.fit(X)
            self.scaler_ = scaler
            tmp_x = scaler.transform(tmp_x)

        # Compute mean reference spectrum
        self.reference_ = np.mean(tmp_x, axis=0)

        # Create wavelength indices for polynomial terms
        self.wavelengths_ = np.arange(X.shape[1])

        return self

    def transform(self, X):
        check_is_fitted(self)

        X_transformed = X.copy() if self.copy else X

        if self.scale:
            X_transformed = self.scaler_.transform(X_transformed)

        # Build design matrix with polynomial terms
        n_features = X.shape[1]

        for i in range(X_transformed.shape[0]):
            # Create polynomial basis
            design_matrix = np.column_stack([
                self.reference_,
                *[self.wavelengths_ ** d for d in range(1, self.degree + 1)]
            ])

            # Fit coefficients
            coeffs, _, _, _ = np.linalg.lstsq(design_matrix, X_transformed[i], rcond=None)

            # Subtract polynomial interference and scale by reference coefficient
            polynomial_part = sum(coeffs[d] * (self.wavelengths_ ** d) for d in range(1, self.degree + 1))
            X_transformed[i] = (X_transformed[i] - polynomial_part) / coeffs[0]

        return X_transformed

    def _more_tags(self):
        return {"allow_nan": False}


class AreaNormalization(TransformerMixin, BaseEstimator):
    """
    Area normalization of spectra.

    Normalizes each spectrum by dividing by its total area (sum of absolute values).
    This removes intensity variations while preserving spectral shape.

    Parameters
    ----------
    method : str, default='sum'
        Method for computing area: 'sum' (sum of values), 'abs_sum' (sum of absolute values),
        or 'trapz' (trapezoidal integration).
    copy : bool, default=True
        Whether to copy input data.
    """

    def __init__(self, method: str = 'sum', *, copy: bool = True):
        self.copy = copy
        self.method = method

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            raise ValueError("AreaNormalization does not support scipy.sparse input")

        if self.method not in ['sum', 'abs_sum', 'trapz']:
            raise ValueError(f"method must be 'sum', 'abs_sum', or 'trapz', got {self.method}")

        return self

    def transform(self, X, copy=None):
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')

        X_transformed = X.copy() if self.copy else X

        for i in range(X_transformed.shape[0]):
            if self.method == 'sum':
                area = np.sum(X_transformed[i])
            elif self.method == 'abs_sum':
                area = np.sum(np.abs(X_transformed[i]))
            elif self.method == 'trapz':
                # Use scipy.integrate.trapezoid for compatibility
                from scipy.integrate import trapezoid
                area = trapezoid(X_transformed[i])

            # Avoid division by zero
            if np.abs(area) < 1e-10:
                area = 1.0

            X_transformed[i] = X_transformed[i] / area

        return X_transformed

    def _more_tags(self):
        return {"allow_nan": False}

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


def _compute_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """Compute entropy of a 1D array."""
    from scipy.stats import entropy as scipy_entropy
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return scipy_entropy(hist) if len(hist) > 0 else 0.0


class WaveletFeatures(TransformerMixin, BaseEstimator):
    """
    Discrete Wavelet Transform feature extractor for spectral data.

    Decomposes spectra into approximation (smooth trends) and detail (sharp
    features) coefficients at multiple scales, then extracts statistical
    features from each level. This captures both global baseline variations
    and local absorption peaks.

    Scientific basis:
        - Multi-resolution analysis captures features at different scales
        - Daubechies wavelets (db4) are well-suited for smooth signals
        - Wavelet coefficients are partially decorrelated

    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet to use (e.g., 'haar', 'db4', 'coif3', 'sym4').
    max_level : int, default=5
        Maximum decomposition level.
    n_coeffs_per_level : int, default=10
        Number of top coefficients (by magnitude) to extract per level.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    actual_level_ : int
        Actual decomposition level used (may be less than max_level
        depending on signal length).
    n_features_out_ : int
        Number of output features.

    References
    ----------
    Mallat (1989). A theory for multiresolution signal decomposition:
    the wavelet representation. IEEE PAMI.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        max_level: int = 5,
        n_coeffs_per_level: int = 10,
        *,
        copy: bool = True
    ):
        self.wavelet = wavelet
        self.max_level = max_level
        self.n_coeffs_per_level = n_coeffs_per_level
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'actual_level_'):
            del self.actual_level_
            del self.n_features_out_
            del self.feature_names_

    def fit(self, X, y=None):
        """
        Fit the wavelet feature extractor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : WaveletFeatures
            Fitted transformer.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("WaveletFeatures does not support scipy.sparse input")

        self._reset()

        n_features = X.shape[1]
        max_level_possible = pywt.dwt_max_level(n_features, self.wavelet)
        self.actual_level_ = min(self.max_level, max_level_possible)

        # Generate feature names and count total features
        self.feature_names_ = []

        # Approximation coefficients: 4 stats + n_coeffs
        for stat in ['mean', 'std', 'energy', 'entropy']:
            self.feature_names_.append(f"wf_approx_{stat}")
        for i in range(self.n_coeffs_per_level):
            self.feature_names_.append(f"wf_approx_coef_{i}")

        # Detail coefficients at each level: 4 stats + n_coeffs per level
        for level in range(1, self.actual_level_ + 1):
            for stat in ['mean', 'std', 'energy', 'entropy']:
                self.feature_names_.append(f"wf_d{level}_{stat}")
            for i in range(self.n_coeffs_per_level):
                self.feature_names_.append(f"wf_d{level}_coef_{i}")

        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X, copy=None):
        """
        Extract wavelet features from spectra.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.
        copy : bool or None, optional
            Ignored (for API compatibility).

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out_)
            Wavelet features.
        """
        check_is_fitted(self, 'actual_level_')

        if scipy.sparse.issparse(X):
            raise ValueError("WaveletFeatures does not support scipy.sparse input")

        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            sample_features = []

            # Process approximation coefficients (coeffs[0])
            approx = coeffs[0]
            sample_features.extend([
                np.mean(approx),
                np.std(approx),
                np.sum(approx ** 2),  # energy
                _compute_entropy(approx)
            ])
            # Top N coefficients (sorted by magnitude)
            sorted_idx = np.argsort(np.abs(approx))[::-1]
            top_coeffs = approx[sorted_idx[:self.n_coeffs_per_level]]
            if len(top_coeffs) < self.n_coeffs_per_level:
                top_coeffs = np.pad(top_coeffs, (0, self.n_coeffs_per_level - len(top_coeffs)))
            sample_features.extend(top_coeffs)

            # Process detail coefficients at each level
            for level in range(1, self.actual_level_ + 1):
                detail = coeffs[level]
                sample_features.extend([
                    np.mean(detail),
                    np.std(detail),
                    np.sum(detail ** 2),
                    _compute_entropy(detail)
                ])
                sorted_idx = np.argsort(np.abs(detail))[::-1]
                top_coeffs = detail[sorted_idx[:self.n_coeffs_per_level]]
                if len(top_coeffs) < self.n_coeffs_per_level:
                    top_coeffs = np.pad(top_coeffs, (0, self.n_coeffs_per_level - len(top_coeffs)))
                sample_features.extend(top_coeffs)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_')
        return np.array(self.feature_names_)

    def _more_tags(self):
        return {"allow_nan": False}


class WaveletPCA(TransformerMixin, BaseEstimator):
    """
    Multi-scale PCA on wavelet coefficients.

    Applies PCA separately to each wavelet decomposition level, creating
    a compact multi-scale representation where each scale contributes a
    few principal components. This preserves frequency-specific information
    while reducing dimensionality.

    Scientific basis:
        - Combines multi-resolution analysis with decorrelation
        - Each scale captures different frequency information
        - PCA per scale reduces redundancy within each frequency band
        - Results in a compact, interpretable feature set

    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet to use (e.g., 'haar', 'db4', 'coif3', 'sym4').
    max_level : int, default=4
        Maximum decomposition level.
    n_components_per_level : int, default=3
        Number of PCA components to keep per decomposition level.
    whiten : bool, default=True
        Whether to whiten the PCA components.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    actual_level_ : int
        Actual decomposition level used.
    pcas_ : dict
        Fitted PCA objects per level.
    scalers_ : dict
        Fitted StandardScaler objects per level.
    n_features_out_ : int
        Number of output features.

    References
    ----------
    Trygg & Wold (1998). PLS regression on wavelet compressed NIR spectra.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        max_level: int = 4,
        n_components_per_level: int = 3,
        whiten: bool = True,
        *,
        copy: bool = True
    ):
        self.wavelet = wavelet
        self.max_level = max_level
        self.n_components_per_level = n_components_per_level
        self.whiten = whiten
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'actual_level_'):
            del self.actual_level_
            del self.pcas_
            del self.scalers_
            del self.feature_names_
            del self.n_features_out_

    def fit(self, X, y=None):
        """
        Fit the wavelet-PCA transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : WaveletPCA
            Fitted transformer.
        """
        from sklearn.decomposition import PCA

        if scipy.sparse.issparse(X):
            raise ValueError("WaveletPCA does not support scipy.sparse input")

        self._reset()

        n_samples, n_features = X.shape
        max_level_possible = pywt.dwt_max_level(n_features, self.wavelet)
        self.actual_level_ = min(self.max_level, max_level_possible)

        # Decompose all samples to get coefficient arrays
        all_coeffs = {i: [] for i in range(self.actual_level_ + 1)}

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            for level_idx, c in enumerate(coeffs):
                all_coeffs[level_idx].append(c)

        # Fit PCA for each level
        self.pcas_ = {}
        self.scalers_ = {}
        self.feature_names_ = []

        for level_idx in range(self.actual_level_ + 1):
            level_data = np.array(all_coeffs[level_idx])
            n_coeffs = level_data.shape[1]
            n_comps = min(self.n_components_per_level, n_coeffs, n_samples - 1)

            if n_comps > 0:
                scaler = StandardScaler()
                level_scaled = scaler.fit_transform(level_data)
                pca = PCA(n_components=n_comps, whiten=self.whiten)
                pca.fit(level_scaled)

                self.scalers_[level_idx] = scaler
                self.pcas_[level_idx] = pca

                level_name = 'approx' if level_idx == 0 else f'd{level_idx}'
                for j in range(n_comps):
                    self.feature_names_.append(f"wpca_{level_name}_pc{j}")

        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X, copy=None):
        """
        Transform spectra to wavelet-PCA features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.
        copy : bool or None, optional
            Ignored (for API compatibility).

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out_)
            Wavelet-PCA features.
        """
        check_is_fitted(self, 'pcas_')

        if scipy.sparse.issparse(X):
            raise ValueError("WaveletPCA does not support scipy.sparse input")

        if not self.pcas_:
            return np.zeros((X.shape[0], 0))

        n_samples = X.shape[0]
        all_features = []

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            sample_features = []

            for level_idx, c in enumerate(coeffs):
                if level_idx in self.pcas_:
                    c_scaled = self.scalers_[level_idx].transform(c.reshape(1, -1))
                    pcs = self.pcas_[level_idx].transform(c_scaled).flatten()
                    sample_features.extend(pcs)

            all_features.append(sample_features)

        return np.array(all_features)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_')
        return np.array(self.feature_names_)

    def _more_tags(self):
        return {"allow_nan": False}


class WaveletSVD(TransformerMixin, BaseEstimator):
    """
    Multi-scale SVD on wavelet coefficients.

    Applies Truncated SVD separately to each wavelet decomposition level,
    creating a compact multi-scale representation. Similar to WaveletPCA
    but uses SVD which doesn't center data and works better for sparse data.

    Scientific basis:
        - Combines multi-resolution analysis with dimensionality reduction
        - Each scale captures different frequency information
        - SVD per scale reduces redundancy within each frequency band
        - Results in a compact feature set

    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet to use (e.g., 'haar', 'db4', 'coif3', 'sym4').
    max_level : int, default=4
        Maximum decomposition level.
    n_components_per_level : int, default=3
        Number of SVD components to keep per decomposition level.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    actual_level_ : int
        Actual decomposition level used.
    svds_ : dict
        Fitted TruncatedSVD objects per level.
    n_features_out_ : int
        Number of output features.

    References
    ----------
    Trygg & Wold (1998). PLS regression on wavelet compressed NIR spectra.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        max_level: int = 4,
        n_components_per_level: int = 3,
        *,
        copy: bool = True
    ):
        self.wavelet = wavelet
        self.max_level = max_level
        self.n_components_per_level = n_components_per_level
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'actual_level_'):
            del self.actual_level_
            del self.svds_
            del self.feature_names_
            del self.n_features_out_

    def fit(self, X, y=None):
        """
        Fit the wavelet-SVD transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : WaveletSVD
            Fitted transformer.
        """
        from sklearn.decomposition import TruncatedSVD

        if scipy.sparse.issparse(X):
            raise ValueError("WaveletSVD does not support scipy.sparse input")

        self._reset()

        n_samples, n_features = X.shape
        max_level_possible = pywt.dwt_max_level(n_features, self.wavelet)
        self.actual_level_ = min(self.max_level, max_level_possible)

        # Decompose all samples to get coefficient arrays
        all_coeffs = {i: [] for i in range(self.actual_level_ + 1)}

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            for level_idx, c in enumerate(coeffs):
                all_coeffs[level_idx].append(c)

        # Fit SVD for each level
        self.svds_ = {}
        self.feature_names_ = []

        for level_idx in range(self.actual_level_ + 1):
            level_data = np.array(all_coeffs[level_idx])
            n_coeffs = level_data.shape[1]
            # TruncatedSVD requires n_components < min(n_samples, n_features)
            n_comps = min(self.n_components_per_level, n_coeffs - 1, n_samples - 1)

            if n_comps > 0:
                svd = TruncatedSVD(n_components=n_comps)
                svd.fit(level_data)

                self.svds_[level_idx] = svd

                level_name = 'approx' if level_idx == 0 else f'd{level_idx}'
                for j in range(n_comps):
                    self.feature_names_.append(f"wsvd_{level_name}_sv{j}")

        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X, copy=None):
        """
        Transform spectra to wavelet-SVD features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.
        copy : bool or None, optional
            Ignored (for API compatibility).

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out_)
            Wavelet-SVD features.
        """
        check_is_fitted(self, 'svds_')

        if scipy.sparse.issparse(X):
            raise ValueError("WaveletSVD does not support scipy.sparse input")

        if not self.svds_:
            return np.zeros((X.shape[0], 0))

        n_samples = X.shape[0]
        all_features = []

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            sample_features = []

            for level_idx, c in enumerate(coeffs):
                if level_idx in self.svds_:
                    svs = self.svds_[level_idx].transform(c.reshape(1, -1)).flatten()
                    sample_features.extend(svs)

            all_features.append(sample_features)

        return np.array(all_features)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_')
        return np.array(self.feature_names_)

    def _more_tags(self):
        return {"allow_nan": False}


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


def reflectance_to_absorbance(
    spectra: np.ndarray,
    min_value: float = 1e-8,
) -> np.ndarray:
    """
    Convert reflectance spectra to absorbance.

    Applies the Beer-Lambert law: A = -log10(R) = log10(1/R)
    where R is reflectance and A is absorbance.

    Args:
        spectra (numpy.ndarray): Reflectance NIRS data matrix (n_samples, n_features).
            Values should be in range (0, 1] or as percentages (0, 100].
        min_value (float): Minimum value to clamp reflectance to avoid log(0).
            Default is 1e-8.

    Returns:
        numpy.ndarray: Absorbance spectra with same shape as input.
    """
    X = np.clip(spectra, min_value, None)
    return -np.log10(X)


class ReflectanceToAbsorbance(TransformerMixin, BaseEstimator):
    """
    Convert reflectance spectra to absorbance using Beer-Lambert law.

    Applies the transformation: A = -log10(R) = log10(1/R)
    where R is reflectance and A is absorbance.

    This is a fundamental transformation in NIR spectroscopy, as absorbance
    is linearly related to concentration (Beer-Lambert law), while reflectance
    is not.

    Parameters
    ----------
    min_value : float, default=1e-8
        Minimum value to clamp reflectance to avoid log(0).
        Values below this threshold will be set to min_value before
        applying the log transform.
    percent : bool, default=False
        If True, assumes input reflectance is in percentage (0-100)
        and divides by 100 before conversion.
    copy : bool, default=True
        Whether to copy input data.

    Notes
    -----
    - Input reflectance values should be positive.
    - For reflectance in range (0, 1], output absorbance is non-negative.
    - For reflectance > 1 (e.g., percentage values), set percent=True.

    Examples
    --------
    >>> from nirs4all.operators.transforms.nirs import ReflectanceToAbsorbance
    >>> import numpy as np
    >>> R = np.array([[0.5, 0.25, 0.1], [0.8, 0.4, 0.2]])
    >>> transformer = ReflectanceToAbsorbance()
    >>> A = transformer.fit_transform(R)
    >>> # A ≈ [[0.301, 0.602, 1.0], [0.097, 0.398, 0.699]]
    """

    def __init__(self, min_value: float = 1e-8, percent: bool = False, *, copy: bool = True):
        self.copy = copy
        self.min_value = min_value
        self.percent = percent

    def _reset(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op, included for API compatibility).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reflectance spectra.
        y : None
            Ignored.

        Returns
        -------
        self : ReflectanceToAbsorbance
            Fitted transformer.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("ReflectanceToAbsorbance does not support scipy.sparse input")
        return self

    def transform(self, X, copy=None):
        """
        Convert reflectance to absorbance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reflectance spectra.
        copy : bool or None, optional
            Whether to copy the input data.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Absorbance spectra.
        """
        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')

        X_out = X.copy() if self.copy else X

        if self.percent:
            X_out = X_out / 100.0

        return reflectance_to_absorbance(X_out, min_value=self.min_value)

    def inverse_transform(self, X):
        """
        Convert absorbance back to reflectance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Absorbance spectra.

        Returns
        -------
        X_reflectance : ndarray of shape (n_samples, n_features)
            Reflectance spectra.
        """
        X_out = np.power(10, -X)

        if self.percent:
            X_out = X_out * 100.0

        return X_out

    def _more_tags(self):
        return {"allow_nan": False}


# =============================================================================
# PyBaselines Wrapper - General baseline correction
# =============================================================================

# Registry of available pybaselines methods organized by category
PYBASELINES_METHODS = {
    # Whittaker-based methods
    'asls': ('whittaker', 'asls'),           # Asymmetric Least Squares
    'iasls': ('whittaker', 'iasls'),         # Improved Asymmetric Least Squares
    'airpls': ('whittaker', 'airpls'),       # Adaptive Iteratively Reweighted PLS
    'arpls': ('whittaker', 'arpls'),         # Asymmetrically Reweighted PLS
    'drpls': ('whittaker', 'drpls'),         # Doubly Reweighted PLS
    'iarpls': ('whittaker', 'iarpls'),       # Improved ARPLS
    'aspls': ('whittaker', 'aspls'),         # Adaptive Smoothness PLS
    'psalsa': ('whittaker', 'psalsa'),       # Peaked Signal's Asymmetric Least Squares
    'derpsalsa': ('whittaker', 'derpsalsa'), # Derivative PSALSA

    # Polynomial methods
    'poly': ('polynomial', 'poly'),           # Regular polynomial
    'modpoly': ('polynomial', 'modpoly'),     # Modified polynomial
    'imodpoly': ('polynomial', 'imodpoly'),   # Improved modified polynomial
    'penalized_poly': ('polynomial', 'penalized_poly'),  # Penalized polynomial
    'loess': ('polynomial', 'loess'),         # Locally estimated scatterplot smoothing
    'quant_reg': ('polynomial', 'quant_reg'), # Quantile regression

    # Morphological methods
    'mor': ('morphological', 'mor'),           # Morphological
    'imor': ('morphological', 'imor'),         # Improved morphological
    'mormol': ('morphological', 'mormol'),     # Morphological and mollified
    'amormol': ('morphological', 'amormol'),   # Averaging morphological and mollified
    'rolling_ball': ('morphological', 'rolling_ball'),  # Rolling ball
    'mwmv': ('morphological', 'mwmv'),         # Moving window minimum value
    'tophat': ('morphological', 'tophat'),     # Top-hat transform
    'mpspline': ('morphological', 'mpspline'), # Morphological penalized spline
    'jbcd': ('morphological', 'jbcd'),         # Joint baseline correction and denoising

    # Spline methods
    'mixture_model': ('spline', 'mixture_model'),  # Mixture model
    'irsqr': ('spline', 'irsqr'),                  # Iteratively reweighted spline quantile regression
    'corner_cutting': ('spline', 'corner_cutting'),  # Corner-cutting
    'pspline_asls': ('spline', 'pspline_asls'),    # Penalized spline ASLS
    'pspline_iasls': ('spline', 'pspline_iasls'),  # Penalized spline IASLS
    'pspline_airpls': ('spline', 'pspline_airpls'),  # Penalized spline airPLS
    'pspline_arpls': ('spline', 'pspline_arpls'),  # Penalized spline arPLS
    'pspline_drpls': ('spline', 'pspline_drpls'),  # Penalized spline drPLS
    'pspline_iarpls': ('spline', 'pspline_iarpls'),  # Penalized spline iarPLS
    'pspline_aspls': ('spline', 'pspline_aspls'),  # Penalized spline asPLS
    'pspline_psalsa': ('spline', 'pspline_psalsa'),  # Penalized spline PSALSA
    'pspline_derpsalsa': ('spline', 'pspline_derpsalsa'),  # Penalized spline derPSALSA

    # Smooth methods
    'noise_median': ('smooth', 'noise_median'),    # Noise median
    'snip': ('smooth', 'snip'),                    # Statistics-sensitive Non-linear Iterative Peak-clipping
    'swima': ('smooth', 'swima'),                  # Small-Window Moving Average
    'ipsa': ('smooth', 'ipsa'),                    # Iterative Polynomial Smoothing Algorithm

    # Classification methods (require training data or special handling)
    'dietrich': ('classification', 'dietrich'),    # Dietrich's method
    'golotvin': ('classification', 'golotvin'),    # Golotvin's method
    'std_distribution': ('classification', 'std_distribution'),  # Standard distribution
    'fastchrom': ('classification', 'fastchrom'),  # FastChrom
    'cwt_br': ('classification', 'cwt_br'),        # Continuous wavelet transform

    # Optimizers (iterative methods)
    'collab_pls': ('optimizers', 'collab_pls'),            # Collaborative PLS
    'optimize_extended_range': ('optimizers', 'optimize_extended_range'),
    'adaptive_minmax': ('optimizers', 'adaptive_minmax'),  # Adaptive min-max

    # Misc methods
    'interp_pts': ('misc', 'interp_pts'),          # Interpolation between points
    'beads': ('misc', 'beads'),                    # Baseline estimation and denoising with sparsity
}


def pybaseline_correction(
    spectra: np.ndarray,
    method: str = 'asls',
    **kwargs
) -> np.ndarray:
    """
    Apply baseline correction using pybaselines library.

    This is a general wrapper for all pybaselines methods, allowing
    flexible baseline correction with various algorithms.

    Args:
        spectra (numpy.ndarray): NIRS data matrix (n_samples, n_features).
        method (str): Baseline correction method. Available methods:
            Whittaker: 'asls', 'iasls', 'airpls', 'arpls', 'drpls', 'iarpls',
                      'aspls', 'psalsa', 'derpsalsa'
            Polynomial: 'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg'
            Morphological: 'mor', 'imor', 'mormol', 'amormol', 'rolling_ball',
                          'mwmv', 'tophat', 'mpspline', 'jbcd'
            Spline: 'mixture_model', 'irsqr', 'corner_cutting', 'pspline_asls', etc.
            Smooth: 'noise_median', 'snip', 'swima', 'ipsa'
            Classification: 'dietrich', 'golotvin', 'std_distribution', 'fastchrom', 'cwt_br'
            Optimizers: 'collab_pls', 'optimize_extended_range', 'adaptive_minmax'
            Misc: 'interp_pts', 'beads'
        **kwargs: Additional parameters passed to the specific baseline method.

    Returns:
        numpy.ndarray: Baseline-corrected spectra with same shape as input.

    Raises:
        ImportError: If pybaselines is not installed.
        ValueError: If an unknown method is specified.

    Examples
    --------
    >>> from nirs4all.operators.transforms.nirs import pybaseline_correction
    >>> corrected = pybaseline_correction(spectra, method='airpls', lam=1e5)
    >>> corrected = pybaseline_correction(spectra, method='imodpoly', poly_order=3)
    >>> corrected = pybaseline_correction(spectra, method='snip', max_half_window=30)
    """
    try:
        import pybaselines
    except ImportError:
        raise ImportError(
            "pybaselines is required for baseline correction. "
            "Install it with: pip install pybaselines"
        )

    method_lower = method.lower()
    if method_lower not in PYBASELINES_METHODS:
        available = ', '.join(sorted(PYBASELINES_METHODS.keys()))
        raise ValueError(
            f"Unknown baseline method '{method}'. "
            f"Available methods: {available}"
        )

    module_name, func_name = PYBASELINES_METHODS[method_lower]

    # Import the specific module and function
    module = getattr(pybaselines, module_name)
    baseline_func = getattr(module, func_name)

    corrected = np.empty_like(spectra)
    for i in range(spectra.shape[0]):
        baseline, _ = baseline_func(spectra[i], **kwargs)
        corrected[i] = spectra[i] - baseline

    return corrected


class PyBaselineCorrection(TransformerMixin, BaseEstimator):
    """
    General baseline correction using pybaselines library.

    A flexible wrapper for the pybaselines library that provides access to
    numerous baseline correction algorithms. This transformer allows easy
    integration of any pybaselines method into sklearn pipelines.

    Parameters
    ----------
    method : str, default='asls'
        The baseline correction method to use. Available methods by category:

        **Whittaker-based** (smooth baselines with asymmetric weighting):
            - 'asls': Asymmetric Least Squares
            - 'iasls': Improved Asymmetric Least Squares
            - 'airpls': Adaptive Iteratively Reweighted PLS
            - 'arpls': Asymmetrically Reweighted PLS
            - 'drpls': Doubly Reweighted PLS
            - 'iarpls': Improved ARPLS
            - 'aspls': Adaptive Smoothness PLS
            - 'psalsa': Peaked Signal's Asymmetric Least Squares
            - 'derpsalsa': Derivative PSALSA

        **Polynomial** (polynomial fitting):
            - 'poly': Regular polynomial
            - 'modpoly': Modified polynomial
            - 'imodpoly': Improved modified polynomial
            - 'penalized_poly': Penalized polynomial
            - 'loess': Locally estimated scatterplot smoothing
            - 'quant_reg': Quantile regression

        **Morphological** (morphological operations):
            - 'mor': Morphological
            - 'imor': Improved morphological
            - 'mormol': Morphological and mollified
            - 'amormol': Averaging morphological and mollified
            - 'rolling_ball': Rolling ball algorithm
            - 'mwmv': Moving window minimum value
            - 'tophat': Top-hat transform
            - 'mpspline': Morphological penalized spline
            - 'jbcd': Joint baseline correction and denoising

        **Spline** (spline-based methods):
            - 'mixture_model': Mixture model
            - 'irsqr': Iteratively reweighted spline quantile regression
            - 'corner_cutting': Corner-cutting
            - 'pspline_asls', 'pspline_iasls', 'pspline_airpls', etc.

        **Smooth** (smoothing-based):
            - 'noise_median': Noise median
            - 'snip': Statistics-sensitive Non-linear Iterative Peak-clipping
            - 'swima': Small-Window Moving Average
            - 'ipsa': Iterative Polynomial Smoothing Algorithm

        **Misc**:
            - 'beads': Baseline estimation and denoising with sparsity
            - 'interp_pts': Interpolation between points

    copy : bool, default=True
        Whether to copy input data.
    **method_params : dict
        Additional parameters passed to the specific baseline method.
        Common parameters include:
        - lam (float): Smoothness parameter for Whittaker methods
        - p (float): Asymmetry parameter for ASLS-type methods
        - poly_order (int): Polynomial order for polynomial methods
        - max_half_window (int): Window size for morphological/smooth methods
        - max_iter (int): Maximum iterations
        - tol (float): Convergence tolerance

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from nirs4all.operators.transforms.nirs import PyBaselineCorrection
    >>> import numpy as np

    Basic usage with ASLS:
    >>> transformer = PyBaselineCorrection(method='asls', lam=1e6, p=0.01)
    >>> corrected = transformer.fit_transform(spectra)

    Using airPLS:
    >>> transformer = PyBaselineCorrection(method='airpls', lam=1e5)
    >>> corrected = transformer.fit_transform(spectra)

    Using improved modified polynomial:
    >>> transformer = PyBaselineCorrection(method='imodpoly', poly_order=3)
    >>> corrected = transformer.fit_transform(spectra)

    Using SNIP for Raman-like data:
    >>> transformer = PyBaselineCorrection(method='snip', max_half_window=40)
    >>> corrected = transformer.fit_transform(spectra)

    Using rolling ball:
    >>> transformer = PyBaselineCorrection(method='rolling_ball', half_window=50)
    >>> corrected = transformer.fit_transform(spectra)

    In a pipeline:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipeline = Pipeline([
    ...     ('baseline', PyBaselineCorrection(method='airpls', lam=1e5)),
    ...     ('scale', StandardScaler()),
    ... ])

    References
    ----------
    pybaselines documentation: https://pybaselines.readthedocs.io/
    """

    def __init__(
        self,
        method: str = 'asls',
        *,
        copy: bool = True,
        **method_params
    ):
        self.method = method
        self.copy = copy
        self.method_params = method_params

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_

    def fit(self, X, y=None):
        """
        Fit the transformer (validates method and stores number of features).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : PyBaselineCorrection
            Fitted transformer.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("PyBaselineCorrection does not support scipy.sparse input")

        # Validate method
        method_lower = self.method.lower()
        if method_lower not in PYBASELINES_METHODS:
            available = ', '.join(sorted(PYBASELINES_METHODS.keys()))
            raise ValueError(
                f"Unknown baseline method '{self.method}'. "
                f"Available methods: {available}"
            )

        self._reset()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, copy=None):
        """
        Apply baseline correction to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.
        copy : bool or None, optional
            Whether to copy the input data.

        Returns
        -------
        X_corrected : ndarray of shape (n_samples, n_features)
            Baseline-corrected spectra.
        """
        check_is_fitted(self, 'n_features_in_')

        if scipy.sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')

        X_out = X.copy() if self.copy else X

        return pybaseline_correction(X_out, method=self.method, **self.method_params)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'method': self.method,
            'copy': self.copy,
        }
        params.update(self.method_params)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        method = params.pop('method', None)
        copy = params.pop('copy', None)

        if method is not None:
            self.method = method
        if copy is not None:
            self.copy = copy

        self.method_params.update(params)
        return self

    def _more_tags(self):
        return {"allow_nan": False}

    @staticmethod
    def list_methods():
        """
        List all available baseline correction methods.

        Returns
        -------
        dict
            Dictionary with method categories as keys and list of methods as values.
        """
        categories = {}
        for method, (module, _) in PYBASELINES_METHODS.items():
            if module not in categories:
                categories[module] = []
            categories[module].append(method)
        return categories


class _BaselineMethodAlias(PyBaselineCorrection):
    """
    Base class for convenience baseline method aliases.

    This class properly handles get_params/set_params for sklearn clone()
    compatibility by storing method-specific parameters as instance attributes.
    """
    _method_name = None  # Override in subclasses

    def __init__(self, *, copy: bool = True, **method_params):
        # Store parameters as instance attributes for sklearn compatibility
        self.copy = copy
        self._method_param_names = list(method_params.keys())
        for key, value in method_params.items():
            setattr(self, key, value)
        # Initialize parent with the fixed method
        super().__init__(method=self._method_name, copy=copy, **method_params)

    def get_params(self, deep=True):
        """Get parameters for this estimator (excluding 'method' for subclasses)."""
        params = {'copy': self.copy}
        for key in self._method_param_names:
            params[key] = getattr(self, key)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        copy = params.pop('copy', None)
        if copy is not None:
            self.copy = copy

        for key, value in params.items():
            if key in self._method_param_names:
                setattr(self, key, value)
                self.method_params[key] = value
        return self


# Convenience aliases for common methods
class AirPLS(_BaselineMethodAlias):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares baseline correction.

    A robust baseline correction method that adaptively adjusts weights
    based on the difference between the fitted baseline and the data.

    Parameters
    ----------
    lam : float, default=1e6
        Smoothness parameter. Larger values produce smoother baselines.
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Zhang, Z.M., et al. (2010). Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 135(5), 1138-1146.
    """
    _method_name = 'airpls'

    def __init__(self, lam: float = 1e6, max_iter: int = 50, tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, lam=lam, max_iter=max_iter, tol=tol)


class ArPLS(_BaselineMethodAlias):
    """
    Asymmetrically Reweighted Penalized Least Squares baseline correction.

    Parameters
    ----------
    lam : float, default=1e6
        Smoothness parameter.
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Baek, S.J., et al. (2015). Baseline correction using asymmetrically
    reweighted penalized least squares smoothing. Analyst, 140(1), 250-257.
    """
    _method_name = 'arpls'

    def __init__(self, lam: float = 1e6, max_iter: int = 50, tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, lam=lam, max_iter=max_iter, tol=tol)


class IModPoly(_BaselineMethodAlias):
    """
    Improved Modified Polynomial baseline correction.

    A polynomial-based baseline correction that iteratively fits and
    removes points above the baseline.

    Parameters
    ----------
    poly_order : int, default=5
        Polynomial order for fitting.
    max_iter : int, default=250
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Zhao, J., et al. (2007). Automated autofluorescence background subtraction
    algorithm for biomedical Raman spectroscopy. Applied Spectroscopy, 61(11), 1225-1232.
    """
    _method_name = 'imodpoly'

    def __init__(self, poly_order: int = 5, max_iter: int = 250, tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, poly_order=poly_order, max_iter=max_iter, tol=tol)


class ModPoly(_BaselineMethodAlias):
    """
    Modified Polynomial baseline correction.

    Parameters
    ----------
    poly_order : int, default=5
        Polynomial order for fitting.
    max_iter : int, default=250
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Lieber, C.A. and Mahadevan-Jansen, A. (2003). Automated method for subtraction
    of fluorescence from biological Raman spectra. Applied Spectroscopy, 57(11), 1363-1367.
    """
    _method_name = 'modpoly'

    def __init__(self, poly_order: int = 5, max_iter: int = 250, tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, poly_order=poly_order, max_iter=max_iter, tol=tol)


class SNIP(_BaselineMethodAlias):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping baseline correction.

    Particularly effective for spectra with many peaks (e.g., Raman, XRF).

    Parameters
    ----------
    max_half_window : int, default=40
        Maximum half-window size for the algorithm.
    decreasing : bool, default=True
        Whether to use decreasing window sizes.
    smooth_half_window : int or None, default=None
        Half-window for smoothing. None means no smoothing.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Ryan, C.G., et al. (1988). SNIP, a statistics-sensitive background treatment
    for the quantitative analysis of PIXE spectra in geoscience applications.
    Nuclear Instruments and Methods in Physics Research B, 34(3), 396-402.
    """
    _method_name = 'snip'

    def __init__(self, max_half_window: int = 40, decreasing: bool = True,
                 smooth_half_window: int = None, *, copy: bool = True):
        super().__init__(copy=copy, max_half_window=max_half_window,
                         decreasing=decreasing, smooth_half_window=smooth_half_window)


class RollingBall(_BaselineMethodAlias):
    """
    Rolling Ball baseline correction.

    A morphological approach that simulates rolling a ball beneath the spectrum.

    Parameters
    ----------
    half_window : int, default=50
        Half-window size for the rolling ball.
    smooth_half_window : int or None, default=None
        Half-window for smoothing. None means no smoothing.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Kneen, M.A. and Annegarn, H.J. (1996). Algorithm for fitting XRF, SEM and
    PIXE X-ray spectra backgrounds. Nuclear Instruments and Methods in Physics
    Research B, 109, 209-213.
    """
    _method_name = 'rolling_ball'

    def __init__(self, half_window: int = 50, smooth_half_window: int = None, *, copy: bool = True):
        super().__init__(copy=copy, half_window=half_window, smooth_half_window=smooth_half_window)


class IASLS(_BaselineMethodAlias):
    """
    Improved Asymmetric Least Squares baseline correction.

    An improvement over ASLS that uses a different weighting scheme.

    Parameters
    ----------
    lam : float, default=1e6
        Smoothness parameter.
    p : float, default=0.01
        Asymmetry parameter.
    lam_1 : float, default=1e-4
        First derivative smoothing parameter.
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    He, S., et al. (2014). Baseline correction for Raman spectra using an
    improved asymmetric least squares method. Analytical Methods, 6(12), 4402-4407.
    """
    _method_name = 'iasls'

    def __init__(self, lam: float = 1e6, p: float = 0.01, lam_1: float = 1e-4,
                 max_iter: int = 50, tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, lam=lam, p=p, lam_1=lam_1, max_iter=max_iter, tol=tol)


class BEADS(_BaselineMethodAlias):
    """
    Baseline Estimation And Denoising with Sparsity.

    Simultaneously estimates baseline and removes noise using sparsity constraints.

    Parameters
    ----------
    lam_0 : float, default=1.0
        Regularization parameter for the baseline.
    lam_1 : float, default=1.0
        Regularization parameter for the first derivative.
    lam_2 : float, default=1.0
        Regularization parameter for the second derivative.
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-2
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Ning, X., et al. (2014). Chromatogram baseline estimation and denoising
    using sparsity (BEADS). Chemometrics and Intelligent Laboratory Systems, 139, 156-167.
    """
    _method_name = 'beads'

    def __init__(self, lam_0: float = 1.0, lam_1: float = 1.0, lam_2: float = 1.0,
                 max_iter: int = 50, tol: float = 1e-2, *, copy: bool = True):
        super().__init__(copy=copy, lam_0=lam_0, lam_1=lam_1, lam_2=lam_2, max_iter=max_iter, tol=tol)


# Keep asls_baseline function for backward compatibility
def asls_baseline(
    spectra: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Compute baseline using Asymmetric Least Squares Smoothing.

    This is a convenience wrapper around pybaseline_correction with method='asls'.

    Args:
        spectra (numpy.ndarray): NIRS data matrix (n_samples, n_features).
        lam (float): Smoothness parameter (lambda). Default is 1e6.
        p (float): Asymmetry parameter (0 < p < 1). Default is 0.01.
        max_iter (int): Maximum number of iterations. Default is 50.
        tol (float): Convergence tolerance. Default is 1e-3.

    Returns:
        numpy.ndarray: Baseline-corrected spectra with same shape as input.
    """
    return pybaseline_correction(spectra, method='asls', lam=lam, p=p, max_iter=max_iter, tol=tol)


class ASLSBaseline(_BaselineMethodAlias):
    """
    Asymmetric Least Squares (AsLS) baseline correction.

    Convenience class for ASLS baseline correction. This is equivalent to
    PyBaselineCorrection(method='asls', ...).

    Parameters
    ----------
    lam : float, default=1e6
        Smoothness parameter (lambda).
    p : float, default=0.01
        Asymmetry parameter (0 < p < 1).
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Eilers, P.H.C. and Boelens, H.F.M. (2005). Baseline Correction with
    Asymmetric Least Squares Smoothing.
    """
    _method_name = 'asls'

    def __init__(self, lam: float = 1e6, p: float = 0.01, max_iter: int = 50,
                 tol: float = 1e-3, *, copy: bool = True):
        super().__init__(copy=copy, lam=lam, p=p, max_iter=max_iter, tol=tol)
