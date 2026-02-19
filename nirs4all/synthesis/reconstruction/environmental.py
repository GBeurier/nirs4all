"""
Environmental effects model for physical signal-chain reconstruction.

Implements temperature, moisture, and scattering effects as fittable parameters.
These effects are applied to the absorption spectrum in canonical space,
before domain transform and instrument effects.

Uses literature-based parameters from the augmentation module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# Temperature Region Parameters (from literature)
# =============================================================================

# Import from augmentation module for consistency
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

# Water band positions for moisture effects
FREE_WATER_PEAK_1ST = 1410  # nm
BOUND_WATER_PEAK_1ST = 1460  # nm
FREE_WATER_PEAK_COMB = 1920  # nm
BOUND_WATER_PEAK_COMB = 1940  # nm

# =============================================================================
# Environmental Effects Model
# =============================================================================

@dataclass
class EnvironmentalEffectsModel:
    """
    Environmental effects on the canonical absorption spectrum.

    Applied to absorption in canonical space before domain transform and
    instrument effects. Implements region-specific temperature and moisture
    effects based on literature parameters.

    Attributes:
        temperature_delta: Temperature deviation from reference (25°C).
        water_activity: Effective water activity (0-1 scale).
        scattering_power: Wavelength-dependent scattering exponent (λ^-n).
        scattering_amplitude: Amplitude of scattering baseline.
        enabled: Whether to apply environmental effects.
        reference_wavelength: Reference wavelength for scattering normalization (nm).
    """

    temperature_delta: float = 0.0
    water_activity: float = 0.5
    scattering_power: float = 1.5
    scattering_amplitude: float = 0.0
    enabled: bool = True
    reference_wavelength: float = 1500.0

    # Cached region masks for efficiency
    _region_masks: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _cached_wavelengths: np.ndarray | None = field(default=None, repr=False)

    def apply(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply environmental effects to absorption spectrum.

        Effects are applied in order:
        1. Temperature effects (region-specific shifts, intensity changes)
        2. Moisture effects (water band shifts based on water activity)
        3. Scattering baseline (wavelength-dependent λ^-n)

        Args:
            absorption: Absorption coefficient on canonical grid.
            wavelengths: Wavelength grid (nm).

        Returns:
            Modified absorption spectrum with environmental effects.
        """
        if not self.enabled:
            return absorption

        result = absorption.copy()

        # Update region masks if wavelengths changed
        self._update_region_masks(wavelengths)

        # Apply temperature effects
        if abs(self.temperature_delta) > 0.01:
            result = self._apply_temperature_effects(result, wavelengths)

        # Apply moisture effects
        if abs(self.water_activity - 0.5) > 0.01:
            result = self._apply_moisture_effects(result, wavelengths)

        # Apply scattering baseline
        if self.scattering_amplitude > 1e-6:
            result = self._apply_scattering_baseline(result, wavelengths)

        return result

    def _update_region_masks(self, wavelengths: np.ndarray) -> None:
        """Update cached region masks if wavelengths changed."""
        if (
            self._cached_wavelengths is not None
            and len(self._cached_wavelengths) == len(wavelengths)
            and np.allclose(self._cached_wavelengths, wavelengths)
        ):
            return

        self._cached_wavelengths = wavelengths.copy()
        self._region_masks = {}

        for region_name, params in TEMPERATURE_REGION_PARAMS.items():
            wl_min, wl_max = params["range"]
            # Smooth Gaussian-weighted region mask
            self._region_masks[region_name] = self._create_region_weights(
                wavelengths, wl_min, wl_max
            )

    def _create_region_weights(
        self,
        wavelengths: np.ndarray,
        wl_min: float,
        wl_max: float,
        edge_width: float = 20.0,
    ) -> np.ndarray:
        """Create smooth Gaussian-weighted region mask."""
        center = (wl_min + wl_max) / 2
        width = (wl_max - wl_min) / 2 + edge_width
        return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    def _apply_temperature_effects(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply temperature effects to absorption spectrum.

        Temperature affects:
        - Peak positions (wavelength shifts, negative for most bands)
        - Intensities (generally decrease with temperature)
        - Band broadening (thermal motion)
        """
        result = absorption.copy()
        delta_t = self.temperature_delta

        for region_name, params in TEMPERATURE_REGION_PARAMS.items():
            weights = self._region_masks[region_name]

            if np.max(weights) < 0.01:
                continue

            # Wavelength shift effect
            shift = params["shift_per_degree"] * delta_t
            if abs(shift) > 0.01:
                shifted_wl = wavelengths + shift * weights
                result = np.interp(wavelengths, shifted_wl, result)

            # Intensity change effect
            intensity_factor = 1.0 + params["intensity_per_degree"] * delta_t
            result = result * (1 + (intensity_factor - 1) * weights)

            # Broadening effect (apply if heating)
            if params["broadening_per_degree"] > 0 and delta_t > 0:
                broadening_factor = 1.0 + params["broadening_per_degree"] * delta_t
                if broadening_factor > 1.0:
                    sigma = (broadening_factor - 1.0) * 3.0
                    if sigma > 0.1:
                        broadened = gaussian_filter1d(result, sigma)
                        result = result * (1 - weights) + broadened * weights

        return result

    def _apply_moisture_effects(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply moisture/water activity effects.

        Water activity controls the free/bound water ratio:
        - Higher water_activity → more free water (peak at ~1410nm)
        - Lower water_activity → more bound water (peak at ~1460nm)

        This shifts water bands between these two positions.
        """
        result = absorption.copy()

        # Compute effective free water fraction based on water activity
        # Sigmoid centered at water_activity = 0.5
        free_fraction = 1.0 / (1.0 + np.exp(-8 * (self.water_activity - 0.5)))

        # Shift from bound to free (positive shift for high water activity)
        # Bound water peak is at higher wavelength than free water
        bound_fraction = 1.0 - free_fraction
        reference_fraction = 0.5  # At water_activity = 0.5

        # Net shift relative to reference
        shift_1st = (FREE_WATER_PEAK_1ST - BOUND_WATER_PEAK_1ST) * (
            free_fraction - reference_fraction
        )
        shift_comb = (FREE_WATER_PEAK_COMB - BOUND_WATER_PEAK_COMB) * (
            free_fraction - reference_fraction
        )

        # Apply to 1st overtone region (1400-1500 nm)
        if abs(shift_1st) > 0.1:
            result = self._shift_water_band(
                result, wavelengths, center=1435, width=50, shift=shift_1st
            )

        # Apply to combination region (1900-2000 nm)
        if abs(shift_comb) > 0.1:
            result = self._shift_water_band(
                result, wavelengths, center=1930, width=40, shift=shift_comb
            )

        return result

    def _shift_water_band(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        center: float,
        width: float,
        shift: float,
    ) -> np.ndarray:
        """Apply localized wavelength shift to a water band."""
        if abs(shift) < 0.01:
            return spectrum
        weights = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        shifted_wl = wavelengths + shift * weights
        return np.interp(wavelengths, shifted_wl, spectrum)

    def _apply_scattering_baseline(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """
        Apply wavelength-dependent scattering baseline.

        Scattering typically follows λ^-n relationship where:
        - n ≈ 4 for Rayleigh scattering (very small particles)
        - n ≈ 1-2 for Mie scattering (larger particles typical in NIR)
        """
        # Normalize wavelengths to reference
        wl_norm = wavelengths / self.reference_wavelength

        # Compute scattering baseline
        scattering = self.scattering_amplitude * (wl_norm ** (-self.scattering_power))

        return absorption + scattering

    def get_jacobian_wrt_temperature(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        eps: float = 0.1,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. temperature_delta."""
        orig = self.temperature_delta

        self.temperature_delta = orig + eps
        spec_plus = self.apply(absorption, wavelengths)

        self.temperature_delta = orig - eps
        spec_minus = self.apply(absorption, wavelengths)

        self.temperature_delta = orig
        return (spec_plus - spec_minus) / (2 * eps)

    def get_jacobian_wrt_water_activity(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        eps: float = 0.01,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. water_activity."""
        orig = self.water_activity

        self.water_activity = min(0.99, orig + eps)
        spec_plus = self.apply(absorption, wavelengths)

        self.water_activity = max(0.01, orig - eps)
        spec_minus = self.apply(absorption, wavelengths)

        self.water_activity = orig
        return (spec_plus - spec_minus) / (2 * eps)

    def get_jacobian_wrt_scattering_power(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        eps: float = 0.05,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. scattering_power."""
        orig = self.scattering_power

        self.scattering_power = orig + eps
        spec_plus = self.apply(absorption, wavelengths)

        self.scattering_power = max(0.1, orig - eps)
        spec_minus = self.apply(absorption, wavelengths)

        self.scattering_power = orig
        return (spec_plus - spec_minus) / (2 * eps)

    def get_jacobian_wrt_scattering_amplitude(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        eps: float = 0.001,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. scattering_amplitude."""
        orig = self.scattering_amplitude

        self.scattering_amplitude = orig + eps
        spec_plus = self.apply(absorption, wavelengths)

        self.scattering_amplitude = max(0, orig - eps)
        spec_minus = self.apply(absorption, wavelengths)

        self.scattering_amplitude = orig
        return (spec_plus - spec_minus) / (2 * eps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature_delta": self.temperature_delta,
            "water_activity": self.water_activity,
            "scattering_power": self.scattering_power,
            "scattering_amplitude": self.scattering_amplitude,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentalEffectsModel:
        """Create from dictionary."""
        return cls(
            temperature_delta=d.get("temperature_delta", 0.0),
            water_activity=d.get("water_activity", 0.5),
            scattering_power=d.get("scattering_power", 1.5),
            scattering_amplitude=d.get("scattering_amplitude", 0.0),
            enabled=d.get("enabled", True),
        )

    def copy(self) -> EnvironmentalEffectsModel:
        """Create a copy of this model."""
        return EnvironmentalEffectsModel(
            temperature_delta=self.temperature_delta,
            water_activity=self.water_activity,
            scattering_power=self.scattering_power,
            scattering_amplitude=self.scattering_amplitude,
            enabled=self.enabled,
            reference_wavelength=self.reference_wavelength,
        )

# =============================================================================
# Environmental Parameter Bounds and Priors
# =============================================================================

@dataclass
class EnvironmentalParameterConfig:
    """
    Configuration for environmental parameter fitting.

    Defines bounds and prior distributions for each parameter.
    """

    # Temperature bounds and prior
    temperature_bounds: tuple[float, float] = (-15.0, 15.0)
    temperature_prior_mean: float = 0.0
    temperature_prior_std: float = 5.0

    # Water activity bounds and prior (Beta distribution)
    water_activity_bounds: tuple[float, float] = (0.1, 0.9)
    water_activity_prior_alpha: float = 2.0
    water_activity_prior_beta: float = 2.0

    # Scattering power bounds and prior
    scattering_power_bounds: tuple[float, float] = (0.5, 3.0)
    scattering_power_prior_mean: float = 1.5
    scattering_power_prior_std: float = 0.5

    # Scattering amplitude bounds and prior
    scattering_amplitude_bounds: tuple[float, float] = (0.0, 0.2)
    scattering_amplitude_prior_scale: float = 0.02

    def get_bounds_list(self) -> list[tuple[float, float]]:
        """Get list of bounds for all 4 environmental parameters."""
        return [
            self.temperature_bounds,
            self.water_activity_bounds,
            self.scattering_power_bounds,
            self.scattering_amplitude_bounds,
        ]

    def compute_prior_penalty(
        self,
        temperature_delta: float,
        water_activity: float,
        scattering_power: float,
        scattering_amplitude: float,
    ) -> float:
        """
        Compute prior penalty for regularization.

        Returns negative log-prior (to be added to objective function).
        """
        penalty = 0.0

        # Gaussian prior on temperature
        penalty += 0.5 * (
            (temperature_delta - self.temperature_prior_mean)
            / self.temperature_prior_std
        ) ** 2

        # Beta prior on water activity (transformed to [0, 1])
        aw_norm = (water_activity - self.water_activity_bounds[0]) / (
            self.water_activity_bounds[1] - self.water_activity_bounds[0]
        )
        aw_norm = np.clip(aw_norm, 1e-6, 1 - 1e-6)
        # Beta log-likelihood: (alpha-1)*log(x) + (beta-1)*log(1-x)
        alpha, beta = self.water_activity_prior_alpha, self.water_activity_prior_beta
        penalty += -(alpha - 1) * np.log(aw_norm) - (beta - 1) * np.log(1 - aw_norm)

        # Gaussian prior on scattering power
        penalty += 0.5 * (
            (scattering_power - self.scattering_power_prior_mean)
            / self.scattering_power_prior_std
        ) ** 2

        # Exponential prior on scattering amplitude (encourage small values)
        penalty += scattering_amplitude / self.scattering_amplitude_prior_scale

        return penalty

__all__ = [
    "EnvironmentalEffectsModel",
    "EnvironmentalParameterConfig",
    "TEMPERATURE_REGION_PARAMS",
    "FREE_WATER_PEAK_1ST",
    "BOUND_WATER_PEAK_1ST",
    "FREE_WATER_PEAK_COMB",
    "BOUND_WATER_PEAK_COMB",
]
