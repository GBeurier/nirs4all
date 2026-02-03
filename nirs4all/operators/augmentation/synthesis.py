"""
Synthesis-derived augmentation operators for spectral data.

This module provides augmentation operators extracted from the synthetic NIRS
spectra generator's effect chain. These operators simulate realistic instrumental
and physical effects that occur during NIR measurements.

Operators:
    - PathLengthAugmenter: Optical path length variation (multiplicative scaling)
    - BatchEffectAugmenter: Batch/session measurement effects (offset + gain)
    - InstrumentalBroadeningAugmenter: Instrumental spectral broadening (Gaussian convolution)
    - HeteroscedasticNoiseAugmenter: Signal-dependent detector noise
    - DeadBandAugmenter: Dead spectral bands (detector saturation/failure)

References:
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
      CRC Press.
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy. CRC Press.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import SpectraTransformerMixin


class PathLengthAugmenter(TransformerMixin, BaseEstimator):
    """Simulates optical path length variation.

    Multiplicatively scales spectra to simulate variations in optical
    path length due to sample positioning, particle size effects, etc.

    The path length factor L is drawn from a normal distribution centered
    at 1.0, then clipped to a minimum value to prevent sign inversion.

    Parameters
    ----------
    path_length_std : float, default=0.05
        Standard deviation of the path length factor distribution.
    min_path_length : float, default=0.5
        Minimum allowed path length factor.
    random_state : int or None, default=None
        Random seed for reproducibility.
    variation_scope : str, default="sample"
        Scope of variation: "sample" for per-sample variation,
        "batch" for a single factor applied to all samples.

    Examples
    --------
    >>> from nirs4all.operators.augmentation.synthesis import PathLengthAugmenter
    >>> aug = PathLengthAugmenter(path_length_std=0.1)
    >>> X_aug = aug.fit_transform(X)
    """

    _supports_variation_scope = True

    def __init__(self, path_length_std=0.05, min_path_length=0.5,
                 random_state=None, variation_scope="sample"):
        self.path_length_std = path_length_std
        self.min_path_length = min_path_length
        self.random_state = random_state
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        n_samples = X.shape[0]

        if self.variation_scope == "batch":
            L = np.full(n_samples, rng.normal(1.0, self.path_length_std))
        else:  # "sample"
            L = rng.normal(1.0, self.path_length_std, size=n_samples)

        L = np.maximum(L, self.min_path_length)
        return X * L[:, np.newaxis]


class BatchEffectAugmenter(SpectraTransformerMixin):
    """Simulates batch/session effects in spectroscopic measurements.

    Applies wavelength-dependent additive offset and multiplicative gain
    to simulate variations between measurement sessions or instruments.

    The offset consists of a constant term plus a wavelength-dependent
    slope. The gain is a uniform multiplicative factor.

    Parameters
    ----------
    offset_std : float, default=0.02
        Standard deviation of the constant offset term.
    slope_std : float, default=0.01
        Standard deviation of the wavelength-dependent slope.
    gain_std : float, default=0.03
        Standard deviation of the multiplicative gain (centered at 1.0).
    random_state : int or None, default=None
        Random seed for reproducibility.
    variation_scope : str, default="sample"
        Scope of variation: "sample" for per-sample effects,
        "batch" for a single effect applied to all samples.

    Examples
    --------
    >>> from nirs4all.operators.augmentation.synthesis import BatchEffectAugmenter
    >>> aug = BatchEffectAugmenter(offset_std=0.03, gain_std=0.05)
    >>> X_aug = aug.fit_transform(X, wavelengths=wavelengths)
    """

    _requires_wavelengths = "optional"
    _supports_variation_scope = True

    def __init__(self, offset_std=0.02, slope_std=0.01, gain_std=0.03,
                 random_state=None, variation_scope="sample"):
        self.offset_std = offset_std
        self.slope_std = slope_std
        self.gain_std = gain_std
        self.random_state = random_state
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        super().fit(X, y, **kwargs)
        return self

    def _transform_impl(self, X, wavelengths):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        n_samples = X.shape[0]
        wl = wavelengths if wavelengths is not None else np.arange(X.shape[1])
        x = (wl - wl.mean()) / (wl.max() - wl.min() + 1e-10)  # Normalized

        if self.variation_scope == "batch":
            offset = rng.normal(0, self.offset_std) + rng.normal(0, self.slope_std) * x
            gain = rng.normal(1.0, self.gain_std)
            return X * gain + offset[np.newaxis, :]
        else:  # "sample"
            offsets = rng.normal(0, self.offset_std, size=n_samples)
            slopes = rng.normal(0, self.slope_std, size=n_samples)
            gains = rng.normal(1.0, self.gain_std, size=n_samples)

            offset_2d = offsets[:, np.newaxis] + slopes[:, np.newaxis] * x[np.newaxis, :]
            return X * gains[:, np.newaxis] + offset_2d


class InstrumentalBroadeningAugmenter(SpectraTransformerMixin):
    """Simulates instrumental spectral broadening.

    Applies Gaussian convolution to simulate the finite spectral resolution
    of the instrument. FWHM is converted to sigma for the Gaussian kernel.

    The relationship between FWHM and Gaussian sigma is:
        sigma = FWHM / (2 * sqrt(2 * ln(2)))

    Parameters
    ----------
    fwhm : float, default=3.0
        Full Width at Half Maximum in wavelength units (e.g., nm).
    fwhm_range : tuple of (float, float) or None, default=None
        If provided, randomly sample FWHM from this range instead
        of using the fixed ``fwhm`` value.
    random_state : int or None, default=None
        Random seed for reproducibility.
    variation_scope : str, default="sample"
        Scope of variation when ``fwhm_range`` is used: "sample" for
        per-sample FWHM, "batch" for a single FWHM for all samples.

    Examples
    --------
    >>> from nirs4all.operators.augmentation.synthesis import InstrumentalBroadeningAugmenter
    >>> aug = InstrumentalBroadeningAugmenter(fwhm=5.0)
    >>> X_aug = aug.fit_transform(X, wavelengths=wavelengths)

    >>> # Variable broadening
    >>> aug = InstrumentalBroadeningAugmenter(fwhm_range=(2.0, 6.0))
    >>> X_aug = aug.fit_transform(X, wavelengths=wavelengths)
    """

    _requires_wavelengths = "optional"
    _supports_variation_scope = True

    def __init__(self, fwhm=3.0, fwhm_range=None, random_state=None,
                 variation_scope="sample"):
        self.fwhm = fwhm
        self.fwhm_range = fwhm_range
        self.random_state = random_state
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        super().fit(X, y, **kwargs)
        return self

    def _transform_impl(self, X, wavelengths):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        n_samples = X.shape[0]

        wl = wavelengths if wavelengths is not None else np.arange(X.shape[1])
        wl_step = np.median(np.diff(wl)) if len(wl) > 1 else 1.0

        if self.fwhm_range is not None:
            if self.variation_scope == "batch":
                fwhm = rng.uniform(*self.fwhm_range)
                sigma_pts = fwhm / (2 * np.sqrt(2 * np.log(2))) / wl_step
                result = np.empty_like(X)
                for i in range(n_samples):
                    result[i] = gaussian_filter1d(X[i], sigma_pts)
                return result
            else:  # "sample"
                fwhms = rng.uniform(*self.fwhm_range, size=n_samples)
                result = np.empty_like(X)
                for i in range(n_samples):
                    sigma_pts = fwhms[i] / (2 * np.sqrt(2 * np.log(2))) / wl_step
                    result[i] = gaussian_filter1d(X[i], sigma_pts)
                return result
        else:
            # Fixed FWHM for all samples
            sigma_pts = self.fwhm / (2 * np.sqrt(2 * np.log(2))) / wl_step
            result = np.empty_like(X)
            for i in range(n_samples):
                result[i] = gaussian_filter1d(X[i], sigma_pts)
            return result


class HeteroscedasticNoiseAugmenter(TransformerMixin, BaseEstimator):
    """Simulates signal-dependent (heteroscedastic) detector noise.

    Noise variance is proportional to signal magnitude, modeling
    shot noise and detector-limited measurements.

    The noise standard deviation at each point is computed as:
        sigma = noise_base + noise_signal_dep * |X|

    Parameters
    ----------
    noise_base : float, default=0.001
        Base noise standard deviation (signal-independent component).
    noise_signal_dep : float, default=0.005
        Signal-dependent noise coefficient.
    random_state : int or None, default=None
        Random seed for reproducibility.
    variation_scope : str, default="sample"
        Scope of variation: "sample" for independent noise per sample,
        "batch" for a shared noise pattern across all samples.

    Examples
    --------
    >>> from nirs4all.operators.augmentation.synthesis import HeteroscedasticNoiseAugmenter
    >>> aug = HeteroscedasticNoiseAugmenter(noise_base=0.002, noise_signal_dep=0.01)
    >>> X_aug = aug.fit_transform(X)
    """

    _supports_variation_scope = True

    def __init__(self, noise_base=0.001, noise_signal_dep=0.005,
                 random_state=None, variation_scope="sample"):
        self.noise_base = noise_base
        self.noise_signal_dep = noise_signal_dep
        self.random_state = random_state
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        sigma = self.noise_base + self.noise_signal_dep * np.abs(X)

        if self.variation_scope == "batch":
            # Same noise pattern for all samples (based on mean sigma)
            mean_sigma = sigma.mean(axis=0, keepdims=True)
            noise_pattern = rng.normal(0, 1, size=(1, X.shape[1])) * mean_sigma
            noise = np.tile(noise_pattern, (X.shape[0], 1))
        else:  # "sample"
            noise = rng.normal(0, 1, size=X.shape) * sigma

        return X + noise


class DeadBandAugmenter(TransformerMixin, BaseEstimator):
    """Simulates dead spectral bands (detector saturation/failure regions).

    Zeroes out random wavelength regions and adds noise, simulating
    detector dead bands or saturation artifacts.

    Parameters
    ----------
    n_bands : int, default=1
        Number of dead bands to introduce per sample.
    width_range : tuple of (int, int), default=(10, 30)
        Range for the width (in wavelength points) of each dead band.
    noise_std : float, default=0.05
        Standard deviation of the noise injected into dead band regions.
    probability : float, default=1.0
        Probability that a dead band is applied to a given sample.
    random_state : int or None, default=None
        Random seed for reproducibility.
    variation_scope : str, default="sample"
        Scope of variation: "sample" for per-sample dead bands,
        "batch" for the same dead bands across all samples.

    Examples
    --------
    >>> from nirs4all.operators.augmentation.synthesis import DeadBandAugmenter
    >>> aug = DeadBandAugmenter(n_bands=2, width_range=(15, 40))
    >>> X_aug = aug.fit_transform(X)
    """

    _supports_variation_scope = True

    def __init__(self, n_bands=1, width_range=(10, 30), noise_std=0.05,
                 probability=1.0, random_state=None, variation_scope="sample"):
        self.n_bands = n_bands
        self.width_range = width_range
        self.noise_std = noise_std
        self.probability = probability
        self.random_state = random_state
        self.variation_scope = variation_scope

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))
        result = X.copy()
        n_samples, n_features = X.shape

        if self.variation_scope == "batch":
            # Same dead bands for all samples
            if rng.random() < self.probability:
                for _ in range(self.n_bands):
                    width = rng.integers(self.width_range[0], self.width_range[1] + 1)
                    start = rng.integers(0, max(1, n_features - width))
                    end = min(start + width, n_features)
                    result[:, start:end] = rng.normal(0, self.noise_std,
                                                       size=(n_samples, end - start))
        else:  # "sample"
            for i in range(n_samples):
                if rng.random() < self.probability:
                    for _ in range(self.n_bands):
                        width = rng.integers(self.width_range[0], self.width_range[1] + 1)
                        start = rng.integers(0, max(1, n_features - width))
                        end = min(start + width, n_features)
                        result[i, start:end] = rng.normal(0, self.noise_std,
                                                           size=end - start)

        return result


__all__ = [
    "PathLengthAugmenter",
    "BatchEffectAugmenter",
    "InstrumentalBroadeningAugmenter",
    "HeteroscedasticNoiseAugmenter",
    "DeadBandAugmenter",
]
