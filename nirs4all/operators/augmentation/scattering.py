"""
Scattering effects augmentation operators for spectral data.

This module provides wavelength-aware augmentation operators that simulate
light scattering effects on NIR spectra, including particle size effects
and EMSC-style distortions.

These operators inherit from SpectraTransformerMixin and automatically receive
wavelength information from the dataset when used in nirs4all pipelines.

References:
    - Martens, H., Nielsen, J. P., & Engelsen, S. B. (2003). Light scattering
      and light absorbance separated by extended multiplicative signal correction.
      Analytical Chemistry, 75(3), 394-404.
    - Dahm, D. J., & Dahm, K. D. (2007). Interpreting Diffuse Reflectance and
      Transmittance. NIR Publications.
    - Burger, J., & Geladi, P. (2005). Hyperspectral NIR image regression.
      Journal of Chemometrics, 19(5‐7), 355-363.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..base import SpectraTransformerMixin


class ParticleSizeAugmenter(SpectraTransformerMixin):
    """
    Simulate particle size effects on scattering for data augmentation.

    Particle size affects NIR spectra through wavelength-dependent baseline
    scattering, typically following a λ^(-n) relationship where n depends
    on the particle size regime (Rayleigh vs Mie).

    Smaller particles cause:
    - Increased scattering baseline (especially at shorter wavelengths)
    - Reduced effective optical path length
    - Additional sample-to-sample variation

    Parameters
    ----------
    mean_size_um : float, default=50.0
        Mean particle size in micrometers.
    size_variation_um : float, default=15.0
        Standard deviation of particle size.
    size_range_um : tuple of (float, float), optional
        If provided, randomly sample particle sizes from this range.
        Overrides mean_size_um and size_variation_um.
    reference_size_um : float, default=50.0
        Reference particle size for baseline calculations.
    wavelength_exponent : float, default=1.5
        Exponent for wavelength dependence (higher = finer particles).
        - 4.0 = Rayleigh regime (particles << wavelength)
        - 1.0-2.0 = Typical for NIR powder samples
        - 0.0 = No wavelength dependence
    size_effect_strength : float, default=0.1
        Overall strength of the scattering effect (0-1).
    include_path_length : bool, default=True
        Whether to include path length effects (multiplicative).
    path_length_sensitivity : float, default=0.5
        How strongly particle size affects effective path length.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import ParticleSizeAugmenter
    >>> aug = ParticleSizeAugmenter(mean_size_um=30.0)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Random particle size in pipeline
    >>> aug = ParticleSizeAugmenter(size_range_um=(20, 100))
    >>> pipeline = [aug, PLSRegression(10)]

    References
    ----------
    - Dahm & Dahm (2007). Interpreting Diffuse Reflectance and Transmittance.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        mean_size_um: float = 50.0,
        size_variation_um: float = 15.0,
        size_range_um: Optional[Tuple[float, float]] = None,
        reference_size_um: float = 50.0,
        wavelength_exponent: float = 1.5,
        size_effect_strength: float = 0.1,
        include_path_length: bool = True,
        path_length_sensitivity: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.mean_size_um = mean_size_um
        self.size_variation_um = size_variation_um
        self.size_range_um = size_range_um
        self.reference_size_um = reference_size_um
        self.wavelength_exponent = wavelength_exponent
        self.size_effect_strength = size_effect_strength
        self.include_path_length = include_path_length
        self.path_length_sensitivity = path_length_sensitivity
        self.random_state = random_state

    def transform_with_wavelengths(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply particle size effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with particle size effects applied.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        result = X.copy()

        # Generate particle sizes for each sample
        if self.size_range_um is not None:
            particle_sizes = rng.uniform(
                self.size_range_um[0],
                self.size_range_um[1],
                n_samples
            )
        else:
            particle_sizes = rng.normal(
                self.mean_size_um,
                self.size_variation_um,
                n_samples
            )
            # Clip to reasonable range
            particle_sizes = np.clip(particle_sizes, 5.0, 500.0)

        # Compute size ratios relative to reference
        size_ratios = particle_sizes / self.reference_size_um

        for i in range(n_samples):
            result[i] = self._apply_size_effects(
                result[i], wavelengths, size_ratios[i], rng
            )

        return result

    def _apply_size_effects(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        size_ratio: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Apply particle size effects to a single spectrum."""
        result = spectrum.copy()

        # 1. Wavelength-dependent scattering baseline
        scatter_baseline = self._compute_scatter_baseline(wavelengths, size_ratio)
        result = result + scatter_baseline

        # 2. Multiplicative path length effect
        if self.include_path_length:
            path_factor = self._compute_path_length_factor(size_ratio)
            result = result * path_factor

        # 3. Additional scatter noise
        scatter_noise = self._compute_scatter_noise(len(wavelengths), rng)
        result = result + scatter_noise

        return result

    def _compute_scatter_baseline(
        self,
        wavelengths: np.ndarray,
        size_ratio: float
    ) -> np.ndarray:
        """Compute scattering-induced baseline offset."""
        # Normalize wavelengths to reference (1500 nm)
        # Ensure wavelengths are positive to avoid division issues
        wl_safe = np.maximum(wavelengths, 1.0)
        wl_norm = wl_safe / 1500.0

        # Wavelength-dependent scattering (λ^(-n) relationship)
        # Clip to avoid extreme values with very small wavelengths
        wl_norm_clipped = np.clip(wl_norm, 0.1, 10.0)
        wl_factor = wl_norm_clipped ** (-self.wavelength_exponent)

        # Size effect: smaller particles scatter more
        # Clip size_ratio to avoid extreme values
        size_ratio_clipped = np.clip(size_ratio, 0.1, 10.0)
        size_factor = size_ratio_clipped ** (-0.5)

        # Scale by strength parameter
        baseline = self.size_effect_strength * (size_factor - 1.0) * wl_factor

        # Center so mean offset is controlled
        baseline = baseline - baseline.mean()

        return baseline

    def _compute_path_length_factor(self, size_ratio: float) -> float:
        """Compute effective path length factor."""
        # Smaller particles reduce mean free path
        path_factor = 1.0 + self.path_length_sensitivity * np.log(size_ratio)
        return np.clip(path_factor, 0.7, 1.5)

    def _compute_scatter_noise(
        self,
        n_wavelengths: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Compute scatter-related noise/variation."""
        noise_std = 0.005 * self.size_effect_strength
        noise = rng.normal(0, noise_std, n_wavelengths)
        # Slight wavelength correlation
        noise = gaussian_filter1d(noise, sigma=3)
        return noise


class EMSCDistortionAugmenter(SpectraTransformerMixin):
    """
    Apply EMSC-style scatter distortions for data augmentation.

    Simulates the spectral distortions that Extended Multiplicative Scatter
    Correction (EMSC) is designed to correct:

        x_distorted = a + b*x + c1*λ + c2*λ² + c3*λ³ + ...

    where:
    - a is additive offset
    - b is multiplicative gain
    - c1, c2, ... are polynomial scattering coefficients

    Parameters
    ----------
    multiplicative_range : tuple of (float, float), default=(0.9, 1.1)
        Range for multiplicative gain factor (b term).
    additive_range : tuple of (float, float), default=(-0.05, 0.05)
        Range for additive offset (a term).
    polynomial_order : int, default=2
        Order of wavelength polynomial (0 = no polynomial term).
    polynomial_strength : float, default=0.02
        Base strength of polynomial scattering terms.
    correlation : float, default=0.3
        Correlation between multiplicative and additive terms.
        Higher values create more realistic scatter patterns.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import EMSCDistortionAugmenter
    >>> aug = EMSCDistortionAugmenter(multiplicative_range=(0.85, 1.15))
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Use in pipeline for data augmentation
    >>> aug = EMSCDistortionAugmenter(polynomial_order=3)
    >>> pipeline = [aug, SNV(), PLSRegression(10)]

    Notes
    -----
    This augmenter is particularly useful when:
    - Training models that need to be robust to scatter variations
    - Simulating data from different instruments or sample presentation
    - Creating training data for transfer learning

    References
    ----------
    - Martens et al. (2003). Light scattering and light absorbance separated
      by extended multiplicative signal correction. Analytical Chemistry.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        multiplicative_range: Tuple[float, float] = (0.9, 1.1),
        additive_range: Tuple[float, float] = (-0.05, 0.05),
        polynomial_order: int = 2,
        polynomial_strength: float = 0.02,
        correlation: float = 0.3,
        random_state: Optional[int] = None,
    ):
        self.multiplicative_range = multiplicative_range
        self.additive_range = additive_range
        self.polynomial_order = polynomial_order
        self.polynomial_strength = polynomial_strength
        self.correlation = correlation
        self.random_state = random_state

    def transform_with_wavelengths(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply EMSC-style distortions to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with EMSC-style distortions applied.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        result = np.zeros_like(X)

        # Normalize wavelengths to [-1, 1] for polynomial stability
        wl_norm = self._normalize_wavelengths(wavelengths)

        for i in range(n_samples):
            params = self._generate_emsc_params(rng)
            result[i] = self._apply_emsc_transform(X[i], wl_norm, params)

        return result

    def _normalize_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Normalize wavelengths to [-1, 1] for polynomial stability."""
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        return 2.0 * (wavelengths - wl_min) / (wl_max - wl_min) - 1.0

    def _generate_emsc_params(self, rng: np.random.Generator) -> dict:
        """Generate EMSC parameters for one sample."""
        params = {}

        # Multiplicative scatter (b term)
        b_mean = (self.multiplicative_range[0] + self.multiplicative_range[1]) / 2
        b_std = (self.multiplicative_range[1] - self.multiplicative_range[0]) / 4
        params['b'] = rng.normal(b_mean, b_std)
        params['b'] = np.clip(params['b'], self.multiplicative_range[0], self.multiplicative_range[1])

        # Additive offset (a term) - potentially correlated with b
        a_mean = (self.additive_range[0] + self.additive_range[1]) / 2
        a_std = (self.additive_range[1] - self.additive_range[0]) / 4

        # Add correlation: when b > 1, tend to have negative a (and vice versa)
        b_deviation = (params['b'] - 1.0) / b_std if b_std > 0 else 0
        correlated_mean = a_mean - self.correlation * a_std * b_deviation
        params['a'] = rng.normal(correlated_mean, a_std * (1 - self.correlation**2)**0.5)
        params['a'] = np.clip(params['a'], self.additive_range[0], self.additive_range[1])

        # Wavelength polynomial coefficients
        if self.polynomial_order > 0:
            for order in range(1, self.polynomial_order + 1):
                # Decreasing strength for higher orders
                coef_std = self.polynomial_strength / (order ** 0.5)
                params[f'c{order}'] = rng.normal(0.0, coef_std)

        return params

    def _apply_emsc_transform(
        self,
        spectrum: np.ndarray,
        wl_norm: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """Apply EMSC transformation to a single spectrum."""
        # Base transformation: a + b * x
        result = params['a'] + params['b'] * spectrum

        # Add wavelength-dependent polynomial terms
        if self.polynomial_order > 0:
            for order in range(1, self.polynomial_order + 1):
                coef_name = f'c{order}'
                if coef_name in params:
                    result = result + params[coef_name] * (wl_norm ** order)

        return result


__all__ = [
    "ParticleSizeAugmenter",
    "EMSCDistortionAugmenter",
]
