"""Wavelet denoising transforms for spectral data."""

import numpy as np
import pywt
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def wavelet_denoise(
    spectra: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
    mode: str = "periodization",
    threshold_mode: str = "soft",
    noise_estimator: str = "median",
) -> np.ndarray:
    """
    Apply wavelet denoising to spectral data while preserving original length.

    Uses multi-level discrete wavelet decomposition with thresholding on detail
    coefficients, followed by reconstruction to maintain the original signal length.
    This is ideal for NIRS preprocessing before PLS or other regression methods.

    Parameters
    ----------
    spectra : np.ndarray of shape (n_samples, n_features)
        Input spectral data.
    wavelet : str, default="db4"
        Wavelet family to use (e.g., 'db4', 'sym8', 'coif3', 'haar').
    level : int, default=5
        Decomposition level. Higher levels capture lower frequency components.
    mode : str, default="periodization"
        Signal extension mode for boundary handling.
        Options: 'zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization'.
    threshold_mode : str, default="soft"
        Thresholding method: 'soft' (continuous) or 'hard' (discontinuous).
    noise_estimator : str, default="median"
        Method to estimate noise level:
        - 'median': sigma = median(|cD_finest|) / 0.6745 (robust to outliers)
        - 'std': sigma = std(cD_finest) (assumes Gaussian noise)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Denoised spectra with original length preserved.

    Notes
    -----
    The universal threshold is computed as: uthresh = sigma * sqrt(2 * log(L))
    where L is the signal length and sigma is estimated from the finest detail coefficients.

    References
    ----------
    Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet
    shrinkage. Biometrika, 81(3), 425-455.
    """
    L = spectra.shape[-1]

    # Decompose signal into wavelet coefficients
    coeffs = pywt.wavedec(spectra, wavelet=wavelet, level=level, mode=mode)

    # Estimate noise from finest detail coefficients (last element)
    cD_finest = coeffs[-1]
    if noise_estimator == "median":
        sigma = np.median(np.abs(cD_finest), axis=-1, keepdims=True) / 0.6745
    elif noise_estimator == "std":
        sigma = np.std(cD_finest, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown noise_estimator: {noise_estimator}. Use 'median' or 'std'.")

    # Universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(L))

    # Apply thresholding to detail coefficients only (keep approximation untouched)
    coeffs_filtered = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=threshold_mode) for c in coeffs[1:]
    ]

    # Reconstruct signal
    x_rec = pywt.waverec(coeffs_filtered, wavelet=wavelet, mode=mode)

    # Truncate to original length (waverec may add padding)
    return x_rec[..., :L]


class WaveletDenoise(TransformerMixin, BaseEstimator):
    """
    Wavelet denoising transformer for spectral data.

    Applies multi-level discrete wavelet decomposition with soft or hard thresholding
    on detail coefficients, then reconstructs the signal to the original length.
    This preprocessing step reduces noise while preserving spectral features,
    making it ideal for NIRS analysis with PLS regression or other methods.

    Parameters
    ----------
    wavelet : str, default="db4"
        Wavelet family to use. Common choices:
        - 'db4', 'db8': Daubechies wavelets (good for general signals)
        - 'sym8': Symlets (nearly symmetric, smooth reconstruction)
        - 'coif3': Coiflets (good for edge preservation)
        - 'haar': Haar wavelet (simple, fast)
    level : int, default=5
        Decomposition level. Higher values capture lower frequency components.
        Typical range: 3-7 for NIRS data (~1000 features).
    mode : str, default="periodization"
        Signal extension mode for boundary handling:
        - 'periodization': Assumes signal is periodic (recommended)
        - 'symmetric': Mirror extension (good for edge preservation)
        - 'smooth': Smooth extension (minimal boundary artifacts)
    threshold_mode : str, default="soft"
        Thresholding method:
        - 'soft': Shrinks coefficients continuously (smoother, more denoising)
        - 'hard': Zeroes coefficients below threshold (preserves peaks)
    noise_estimator : str, default="median"
        Method to estimate noise level from finest detail coefficients:
        - 'median': Robust to outliers (recommended for real data)
        - 'std': Standard deviation (assumes Gaussian noise)
    copy : bool, default=True
        Whether to copy input data before transformation.

    Attributes
    ----------
    None (stateless transformer)

    Examples
    --------
    >>> from nirs4all.operators.transforms import WaveletDenoise
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pipeline = [
    ...     WaveletDenoise(wavelet='db8', level=5),
    ...     {"model": PLSRegression(n_components=10)}
    ... ]

    Notes
    -----
    - The transformer is stateless (does not learn from training data)
    - Output shape matches input shape exactly
    - The universal threshold is data-dependent: sigma * sqrt(2 * log(L))

    References
    ----------
    Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet
    shrinkage. Biometrika, 81(3), 425-455.
    """

    _webapp_meta = {
        "category": "smoothing",
        "tier": "advanced",
        "tags": ["wavelet", "denoising", "noise-reduction", "multi-resolution"],
    }

    _stateless = True

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 5,
        mode: str = "periodization",
        threshold_mode: str = "soft",
        noise_estimator: str = "median",
        *,
        copy: bool = True
    ):
        """
        Initialize WaveletDenoise transformer.

        Parameters
        ----------
        wavelet : str, default="db4"
            Wavelet family to use.
        level : int, default=5
            Decomposition level.
        mode : str, default="periodization"
            Signal extension mode.
        threshold_mode : str, default="soft"
            Thresholding method ('soft' or 'hard').
        noise_estimator : str, default="median"
            Noise estimation method ('median' or 'std').
        copy : bool, default=True
            Whether to copy input data.
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.threshold_mode = threshold_mode
        self.noise_estimator = noise_estimator
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state (no-op for stateless transformer)."""
        pass

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for stateless wavelet denoising).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectral data.
        y : None
            Ignored.

        Returns
        -------
        self : WaveletDenoise
            Returns self.

        Raises
        ------
        ValueError
            If input is a sparse matrix.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("WaveletDenoise does not support sparse matrices")
        return self

    def transform(self, X, y=None):
        """
        Apply wavelet denoising to input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectral data.
        y : None
            Ignored.

        Returns
        -------
        X_denoised : np.ndarray of shape (n_samples, n_features)
            Denoised spectral data with original length preserved.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("Sparse matrices not supported")

        return wavelet_denoise(
            X,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
            threshold_mode=self.threshold_mode,
            noise_estimator=self.noise_estimator,
        )

    def _more_tags(self):
        """Provide additional tags for sklearn compatibility."""
        return {"allow_nan": False, "stateless": True}
