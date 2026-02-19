"""
Environmental effects augmentation operators for spectral data.

This module provides wavelength-aware augmentation operators that simulate
environmental effects on NIR spectra, including temperature-induced changes
and moisture/water activity effects.

These operators inherit from SpectraTransformerMixin and automatically receive
wavelength information from the dataset when used in nirs4all pipelines.

References:
    - Maeda, H., Ozaki, Y., et al. (1995). Near infrared spectroscopy and
      chemometrics studies of temperature-dependent spectral variations of water.
      Journal of Near Infrared Spectroscopy, 3(4), 191-201.
    - Segtnan, V. H., et al. (2001). Studies on the structure of water using
      two-dimensional near-infrared correlation spectroscopy. Analytical Chemistry,
      73(13), 3153-3161.
    - Luck, W. A. P. (1998). The importance of cooperativity for the properties
      of liquid water. Journal of Molecular Structure, 448(2-3), 131-142.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..base import SpectraTransformerMixin

# =============================================================================
# Temperature Effect Parameters by Spectral Region
# =============================================================================

# Literature-based temperature effect parameters (nm, shift/°C, intensity/°C, broadening/°C)
TEMPERATURE_REGION_PARAMS: dict[str, dict] = {
    "oh_1st_overtone": {
        "range": (1400, 1520),
        "shift_per_degree": -0.30,
        "intensity_per_degree": -0.002,
        "broadening_per_degree": 0.001,
    },
    "oh_combination": {
        "range": (1900, 2000),
        "shift_per_degree": -0.40,
        "intensity_per_degree": -0.003,
        "broadening_per_degree": 0.0012,
    },
    "ch_1st_overtone": {
        "range": (1650, 1780),
        "shift_per_degree": -0.05,
        "intensity_per_degree": -0.0005,
        "broadening_per_degree": 0.0008,
    },
    "nh_1st_overtone": {
        "range": (1490, 1560),
        "shift_per_degree": -0.20,
        "intensity_per_degree": -0.0015,
        "broadening_per_degree": 0.001,
    },
    "water_free": {
        "range": (1380, 1420),
        "shift_per_degree": -0.10,
        "intensity_per_degree": 0.003,
        "broadening_per_degree": 0.0008,
    },
    "water_bound": {
        "range": (1440, 1500),
        "shift_per_degree": -0.35,
        "intensity_per_degree": -0.004,
        "broadening_per_degree": 0.0015,
    },
}

class TemperatureAugmenter(SpectraTransformerMixin):
    """
    Simulate temperature-induced spectral changes for data augmentation.

    Temperature affects NIR spectra through:
    - Peak position shifts (especially O-H, N-H bands)
    - Intensity changes (hydrogen bonding disruption)
    - Band broadening (thermal motion)

    This operator applies region-specific temperature effects based on
    literature values for NIR spectroscopy.

    Parameters
    ----------
    temperature_delta : float, default=5.0
        Temperature change from reference (°C). Positive = heating.
    temperature_range : tuple of (float, float), optional
        If provided, randomly sample temperature_delta from this range
        for each sample. Overrides temperature_delta parameter.
    reference_temperature : float, default=25.0
        Reference temperature for the input spectra (°C).
    enable_shift : bool, default=True
        Apply peak position shifts.
    enable_intensity : bool, default=True
        Apply intensity changes.
    enable_broadening : bool, default=True
        Apply band broadening.
    region_specific : bool, default=True
        Apply region-specific effects (recommended). If False, applies
        uniform average effects across all wavelengths.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import TemperatureAugmenter
    >>> aug = TemperatureAugmenter(temperature_delta=10.0)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Random temperature variation in pipeline
    >>> aug = TemperatureAugmenter(temperature_range=(-5, 10))
    >>> pipeline = [aug, PLSRegression(10)]

    References
    ----------
    - Maeda et al. (1995). JNIR Spectroscopy, 3(4), 191-201.
    - Segtnan et al. (2001). Analytical Chemistry, 73(13), 3153-3161.
    """

    _webapp_meta = {
        "category": "environmental",
        "tier": "advanced",
        "tags": ["temperature", "environmental", "physical", "augmentation"],
    }

    _requires_wavelengths: bool = True

    def __init__(
        self,
        temperature_delta: float = 5.0,
        temperature_range: tuple[float, float] | None = None,
        reference_temperature: float = 25.0,
        enable_shift: bool = True,
        enable_intensity: bool = True,
        enable_broadening: bool = True,
        region_specific: bool = True,
        random_state: int | None = None,
    ):
        self.temperature_delta = temperature_delta
        self.temperature_range = temperature_range
        self.reference_temperature = reference_temperature
        self.enable_shift = enable_shift
        self.enable_intensity = enable_intensity
        self.enable_broadening = enable_broadening
        self.region_specific = region_specific
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply temperature effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with temperature effects applied.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        result = X.copy()

        # Determine temperature delta for each sample
        delta_temps = rng.uniform(self.temperature_range[0], self.temperature_range[1], n_samples) if self.temperature_range is not None else np.full(n_samples, self.temperature_delta)

        # Apply effects to each sample
        for i in range(n_samples):
            delta_t = delta_temps[i]
            if abs(delta_t) < 0.01:
                continue

            if self.region_specific:
                result[i] = self._apply_region_specific(
                    result[i], wavelengths, delta_t
                )
            else:
                result[i] = self._apply_uniform(result[i], wavelengths, delta_t)

        return result

    def _apply_region_specific(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        delta_temp: float
    ) -> np.ndarray:
        """Apply region-specific temperature effects."""
        result = spectrum.copy()

        for _region_name, params in TEMPERATURE_REGION_PARAMS.items():
            wl_min, wl_max = params["range"]
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)

            if not np.any(mask):
                continue

            weights = self._create_region_weights(wavelengths, wl_min, wl_max)

            # Apply wavelength shift
            if self.enable_shift:
                shift = params["shift_per_degree"] * delta_temp
                result = self._apply_wavelength_shift(result, wavelengths, shift, weights)

            # Apply intensity change
            if self.enable_intensity:
                intensity_factor = 1.0 + params["intensity_per_degree"] * delta_temp
                result = result * (1 + (intensity_factor - 1) * weights)

            # Apply broadening
            if self.enable_broadening and params["broadening_per_degree"] != 0:
                broadening_factor = 1.0 + params["broadening_per_degree"] * delta_temp
                if broadening_factor > 1.0:
                    result = self._apply_broadening(result, broadening_factor, weights)

        return result

    def _apply_uniform(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        delta_temp: float
    ) -> np.ndarray:
        """Apply uniform temperature effect across all wavelengths."""
        result = spectrum.copy()

        # Average parameters across all regions
        avg_shift = np.mean([p["shift_per_degree"] for p in TEMPERATURE_REGION_PARAMS.values()])
        avg_intensity = np.mean([p["intensity_per_degree"] for p in TEMPERATURE_REGION_PARAMS.values()])
        avg_broadening = np.mean([p["broadening_per_degree"] for p in TEMPERATURE_REGION_PARAMS.values()])

        # Apply shift
        if self.enable_shift:
            shift = avg_shift * delta_temp
            result = np.interp(wavelengths, wavelengths + shift, result)

        # Apply intensity change
        if self.enable_intensity:
            intensity_factor = 1.0 + avg_intensity * delta_temp
            result = result * intensity_factor

        # Apply broadening
        if self.enable_broadening:
            broadening_factor = 1.0 + avg_broadening * delta_temp
            if broadening_factor > 1.0:
                sigma = (broadening_factor - 1.0) * 5
                if sigma > 0.1:
                    result = gaussian_filter1d(result, sigma)

        return result

    def _create_region_weights(
        self,
        wavelengths: np.ndarray,
        wl_min: float,
        wl_max: float,
        edge_width: float = 20.0
    ) -> np.ndarray:
        """Create smooth weights for a spectral region using sigmoid transitions."""
        k = 10.0 / edge_width
        rising = 1.0 / (1.0 + np.exp(-k * (wavelengths - wl_min)))
        falling = 1.0 / (1.0 + np.exp(k * (wavelengths - wl_max)))
        return rising * falling

    def _apply_wavelength_shift(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        shift: float,
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply weighted wavelength shift."""
        shifted_wl = wavelengths + shift * weights
        return np.interp(wavelengths, shifted_wl, spectrum)

    def _apply_broadening(
        self,
        spectrum: np.ndarray,
        factor: float,
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply weighted spectral broadening."""
        sigma = (factor - 1.0) * 3.0
        if sigma < 0.1:
            return spectrum
        broadened = gaussian_filter1d(spectrum, sigma)
        return spectrum * (1 - weights) + broadened * weights

class MoistureAugmenter(SpectraTransformerMixin):
    """
    Simulate moisture-induced spectral changes for data augmentation.

    Water activity and moisture content affect NIR spectra through shifts
    in water bands between free and bound states. Higher water activity
    leads to more free water, while lower water activity means more
    water is hydrogen-bonded to the sample matrix.

    Parameters
    ----------
    water_activity_delta : float, default=0.1
        Change in water activity from reference (0-1 scale).
    water_activity_range : tuple of (float, float), optional
        If provided, randomly sample water_activity_delta from this range
        for each sample.
    reference_water_activity : float, default=0.5
        Reference water activity for the input spectra.
    free_water_fraction : float, default=0.3
        Base fraction of water that is "free" vs. bound (0-1).
    bound_water_shift : float, default=25.0
        Wavelength shift (nm) for bound water relative to free water.
    moisture_content : float, default=0.10
        Base moisture content as fraction (affects intensity).
    enable_shift : bool, default=True
        Apply water band position shifts.
    enable_intensity : bool, default=True
        Apply water band intensity changes based on moisture content.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import MoistureAugmenter
    >>> aug = MoistureAugmenter(water_activity_delta=0.2)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Random moisture variation in pipeline
    >>> aug = MoistureAugmenter(water_activity_range=(-0.2, 0.2))
    >>> pipeline = [aug, PLSRegression(10)]

    References
    ----------
    - Büning-Pfaue, H. (2003). Analysis of water in food by near infrared
      spectroscopy. Food Chemistry, 82(1), 107-115.
    - Luck, W. A. P. (1998). The importance of cooperativity for the
      properties of liquid water. Journal of Molecular Structure.
    """

    _webapp_meta = {
        "category": "environmental",
        "tier": "advanced",
        "tags": ["moisture", "water", "environmental", "physical", "augmentation"],
    }

    _requires_wavelengths: bool = True

    # Water band positions (nm)
    FREE_WATER_PEAK_1ST = 1410
    BOUND_WATER_PEAK_1ST = 1460
    FREE_WATER_PEAK_COMB = 1920
    BOUND_WATER_PEAK_COMB = 1940

    def __init__(
        self,
        water_activity_delta: float = 0.1,
        water_activity_range: tuple[float, float] | None = None,
        reference_water_activity: float = 0.5,
        free_water_fraction: float = 0.3,
        bound_water_shift: float = 25.0,
        moisture_content: float = 0.10,
        enable_shift: bool = True,
        enable_intensity: bool = True,
        random_state: int | None = None,
    ):
        self.water_activity_delta = water_activity_delta
        self.water_activity_range = water_activity_range
        self.reference_water_activity = reference_water_activity
        self.free_water_fraction = free_water_fraction
        self.bound_water_shift = bound_water_shift
        self.moisture_content = moisture_content
        self.enable_shift = enable_shift
        self.enable_intensity = enable_intensity
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply moisture effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with moisture effects applied.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        result = X.copy()

        # Determine water activity for each sample
        aw_deltas = rng.uniform(self.water_activity_range[0], self.water_activity_range[1], n_samples) if self.water_activity_range is not None else np.full(n_samples, self.water_activity_delta)

        # Apply effects to each sample
        for i in range(n_samples):
            aw = np.clip(
                self.reference_water_activity + aw_deltas[i],
                0.0, 1.0
            )
            effective_fraction = self._compute_free_water_fraction(aw)
            result[i] = self._apply_water_state_effects(
                result[i], wavelengths, effective_fraction
            )

        return result

    def _compute_free_water_fraction(self, water_activity: float) -> float:
        """Compute effective free water fraction based on water activity."""
        # Sigmoid centered around a_w = 0.5
        sigmoid_factor = 1.0 / (1.0 + np.exp(-8 * (water_activity - 0.5)))
        return self.free_water_fraction * sigmoid_factor

    def _apply_water_state_effects(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        free_fraction: float
    ) -> np.ndarray:
        """Apply water state (free vs bound) effects to spectrum."""
        result = spectrum.copy()
        bound_fraction = 1.0 - free_fraction

        if self.enable_shift:
            # Apply to 1st overtone region (1400-1500 nm)
            result = self._shift_water_band(
                result, wavelengths,
                center=1435,
                width=50,
                shift=self.bound_water_shift * bound_fraction
            )

            # Apply to combination region (1900-2000 nm)
            result = self._shift_water_band(
                result, wavelengths,
                center=1930,
                width=40,
                shift=self.bound_water_shift * 0.8 * bound_fraction
            )

        if self.enable_intensity:
            # Enhance water bands based on moisture content
            water_region_1 = self._create_gaussian_region(wavelengths, 1435, 40)
            water_region_2 = self._create_gaussian_region(wavelengths, 1930, 35)

            enhancement = self.moisture_content / 0.10 - 1.0
            result += enhancement * 0.1 * (water_region_1 + water_region_2) * np.abs(result.mean())

        return result

    def _shift_water_band(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        center: float,
        width: float,
        shift: float
    ) -> np.ndarray:
        """Apply a localized wavelength shift to a water band."""
        if abs(shift) < 0.01:
            return spectrum
        weights = self._create_gaussian_region(wavelengths, center, width)
        shifted_wl = wavelengths + shift * weights
        return np.interp(wavelengths, shifted_wl, spectrum)

    def _create_gaussian_region(
        self,
        wavelengths: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """Create Gaussian weighting for a spectral region."""
        return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

__all__ = [
    "TemperatureAugmenter",
    "MoistureAugmenter",
    "TEMPERATURE_REGION_PARAMS",
]
