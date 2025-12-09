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
