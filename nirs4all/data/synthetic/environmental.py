"""
Environmental effects simulation for synthetic NIRS data generation.

This module provides simulation of environmental and matrix effects on NIR spectra,
including temperature-induced changes and moisture/water activity effects.

Key Features:
    - Temperature-dependent peak shifts (especially O-H bands)
    - Temperature-dependent intensity changes (hydrogen bonding effects)
    - Temperature-dependent band broadening (thermal motion)
    - Region-specific temperature effects (water vs. C-H bands)
    - Moisture/water activity effects on hydrogen bonding
    - Free water vs. bound water band differentiation

References:
    - Maeda, H., Ozaki, Y., Tanaka, M., Hayashi, N., & Kojima, T. (1995).
      Near infrared spectroscopy and chemometrics studies of temperature-dependent
      spectral variations of water. Journal of Near Infrared Spectroscopy, 3(4), 191-201.
    - Segtnan, V. H., Šašić, Š., Isaksson, T., & Ozaki, Y. (2001). Studies on the
      structure of water using two-dimensional near-infrared correlation spectroscopy
      and principal component analysis. Analytical Chemistry, 73(13), 3153-3161.
    - Büning-Pfaue, H. (2003). Analysis of water in food by near infrared spectroscopy.
      Food Chemistry, 82(1), 107-115.
    - Luck, W. A. P. (1998). The importance of cooperativity for the properties of
      liquid water. Journal of Molecular Structure, 448(2-3), 131-142.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .wavenumber import (
    wavelength_to_wavenumber,
    wavenumber_to_wavelength,
    convert_bandwidth_to_wavelength,
)


# ============================================================================
# Temperature Effect Parameters by Spectral Region
# ============================================================================

class SpectralRegion(str, Enum):
    """NIR spectral regions with distinct temperature responses."""
    OH_FIRST_OVERTONE = "oh_1st_overtone"      # ~1400-1500 nm (O-H stretch 1st overtone)
    OH_COMBINATION = "oh_combination"          # ~1900-2000 nm (O-H stretch + bend)
    CH_FIRST_OVERTONE = "ch_1st_overtone"      # ~1650-1750 nm (C-H stretch 1st overtone)
    CH_COMBINATION = "ch_combination"          # ~2200-2400 nm (C-H combinations)
    NH_FIRST_OVERTONE = "nh_1st_overtone"      # ~1500-1550 nm (N-H stretch 1st overtone)
    NH_COMBINATION = "nh_combination"          # ~2000-2100 nm (N-H combinations)
    WATER_FREE = "water_free"                  # Free water O-H
    WATER_BOUND = "water_bound"                # Hydrogen-bonded water O-H


@dataclass
class TemperatureEffectParams:
    """
    Temperature effect parameters for a spectral region.

    Based on literature values for temperature-induced spectral changes in NIR.

    Attributes:
        wavelength_range: Affected wavelength range (nm).
        shift_per_degree: Peak position shift per °C (nm). Negative = blue shift.
        intensity_change_per_degree: Fractional intensity change per °C.
        broadening_per_degree: Fractional bandwidth increase per °C.
        reference: Literature reference for values.
    """
    wavelength_range: Tuple[float, float]
    shift_per_degree: float          # nm/°C
    intensity_change_per_degree: float  # fraction/°C (e.g., -0.002 = -0.2%/°C)
    broadening_per_degree: float     # fraction/°C
    reference: str = ""


# Literature-based temperature effect parameters
TEMPERATURE_EFFECT_PARAMS: Dict[SpectralRegion, TemperatureEffectParams] = {
    SpectralRegion.OH_FIRST_OVERTONE: TemperatureEffectParams(
        wavelength_range=(1400, 1520),
        shift_per_degree=-0.30,      # Blue shift with increasing temperature
        intensity_change_per_degree=-0.002,  # Decreased H-bonding = lower intensity
        broadening_per_degree=0.001,
        reference="Maeda et al. (1995), Segtnan et al. (2001)"
    ),
    SpectralRegion.OH_COMBINATION: TemperatureEffectParams(
        wavelength_range=(1900, 2000),
        shift_per_degree=-0.40,      # Stronger shift in combination region
        intensity_change_per_degree=-0.003,
        broadening_per_degree=0.0012,
        reference="Segtnan et al. (2001)"
    ),
    SpectralRegion.CH_FIRST_OVERTONE: TemperatureEffectParams(
        wavelength_range=(1650, 1780),
        shift_per_degree=-0.05,      # Small shift for non-H-bonding groups
        intensity_change_per_degree=-0.0005,
        broadening_per_degree=0.0008,
        reference="Workman & Weyer (2012)"
    ),
    SpectralRegion.CH_COMBINATION: TemperatureEffectParams(
        wavelength_range=(2200, 2400),
        shift_per_degree=-0.08,
        intensity_change_per_degree=-0.0006,
        broadening_per_degree=0.0006,
        reference="Workman & Weyer (2012)"
    ),
    SpectralRegion.NH_FIRST_OVERTONE: TemperatureEffectParams(
        wavelength_range=(1490, 1560),
        shift_per_degree=-0.20,      # Moderate shift for N-H
        intensity_change_per_degree=-0.0015,
        broadening_per_degree=0.001,
        reference="Burns & Ciurczak (2007)"
    ),
    SpectralRegion.NH_COMBINATION: TemperatureEffectParams(
        wavelength_range=(2000, 2150),
        shift_per_degree=-0.25,
        intensity_change_per_degree=-0.002,
        broadening_per_degree=0.001,
        reference="Burns & Ciurczak (2007)"
    ),
    SpectralRegion.WATER_FREE: TemperatureEffectParams(
        wavelength_range=(1380, 1420),  # Free O-H peak
        shift_per_degree=-0.10,      # Small shift for free water
        intensity_change_per_degree=0.003,  # Increases with temperature
        broadening_per_degree=0.0008,
        reference="Luck (1998)"
    ),
    SpectralRegion.WATER_BOUND: TemperatureEffectParams(
        wavelength_range=(1440, 1500),  # Bound O-H peak
        shift_per_degree=-0.35,      # Larger shift for H-bonded water
        intensity_change_per_degree=-0.004,  # Decreases with temperature
        broadening_per_degree=0.0015,
        reference="Luck (1998), Segtnan et al. (2001)"
    ),
}


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class TemperatureConfig:
    """
    Configuration for temperature effect simulation.

    Attributes:
        reference_temperature: Reference temperature in °C (typically 25°C).
        sample_temperature: Actual sample temperature in °C.
        temperature_variation: Sample-to-sample temperature variation (std dev in °C).
        enable_shift: Whether to apply wavelength shifts.
        enable_intensity: Whether to apply intensity changes.
        enable_broadening: Whether to apply band broadening.
        region_specific: Whether to use region-specific parameters.
        custom_regions: Optional custom region parameters to override defaults.
    """
    reference_temperature: float = 25.0
    sample_temperature: float = 25.0
    temperature_variation: float = 0.0  # Sample-to-sample variation
    enable_shift: bool = True
    enable_intensity: bool = True
    enable_broadening: bool = True
    region_specific: bool = True
    custom_regions: Optional[Dict[SpectralRegion, TemperatureEffectParams]] = None

    @property
    def delta_temperature(self) -> float:
        """Temperature difference from reference."""
        return self.sample_temperature - self.reference_temperature


@dataclass
class MoistureConfig:
    """
    Configuration for moisture/water activity effect simulation.

    Moisture affects NIR spectra through:
    - Direct water absorption bands
    - Hydrogen bonding with sample matrix
    - Free vs. bound water ratio

    Attributes:
        water_activity: Water activity (a_w) value (0.0 to 1.0).
        moisture_content: Moisture content as fraction (optional, for intensity).
        free_water_fraction: Fraction of water that is "free" vs. bound (0-1).
        bound_water_shift: Wavelength shift for bound water relative to free (nm).
        temperature_interaction: Whether moisture effects interact with temperature.
        reference_aw: Reference water activity for baseline.
    """
    water_activity: float = 0.5
    moisture_content: float = 0.10  # 10% moisture by default
    free_water_fraction: float = 0.3
    bound_water_shift: float = 25.0  # nm shift for bound vs free water
    temperature_interaction: bool = True
    reference_aw: float = 0.5

    def __post_init__(self):
        """Validate water activity range."""
        if not 0.0 <= self.water_activity <= 1.0:
            raise ValueError(f"water_activity must be 0-1, got {self.water_activity}")
        if not 0.0 <= self.free_water_fraction <= 1.0:
            raise ValueError(f"free_water_fraction must be 0-1, got {self.free_water_fraction}")


@dataclass
class EnvironmentalEffectsConfig:
    """
    Combined configuration for all environmental effects.

    Attributes:
        temperature: Temperature effect configuration.
        moisture: Moisture effect configuration.
        enable_temperature: Whether to apply temperature effects.
        enable_moisture: Whether to apply moisture effects.
    """
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    moisture: MoistureConfig = field(default_factory=MoistureConfig)
    enable_temperature: bool = True
    enable_moisture: bool = True


# ============================================================================
# Temperature Effect Simulator
# ============================================================================

class TemperatureEffectSimulator:
    """
    Simulate temperature-dependent spectral changes.

    Temperature affects NIR spectra through:
    - Peak position shifts (especially hydrogen-bonded groups)
    - Intensity changes (hydrogen bond population changes)
    - Band broadening (thermal motion)

    The effects are strongest for O-H and N-H groups due to their
    involvement in hydrogen bonding. C-H groups show smaller effects.

    Attributes:
        config: Temperature effect configuration.
        rng: Random number generator for reproducibility.

    Example:
        >>> config = TemperatureConfig(
        ...     sample_temperature=40.0,
        ...     reference_temperature=25.0
        ... )
        >>> simulator = TemperatureEffectSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[TemperatureConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the temperature effect simulator.

        Args:
            config: Temperature effect configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else TemperatureConfig()
        self.rng = np.random.default_rng(random_state)

        # Merge default and custom region parameters
        self.region_params = TEMPERATURE_EFFECT_PARAMS.copy()
        if self.config.custom_regions:
            self.region_params.update(self.config.custom_regions)

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        sample_temperatures: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply temperature effects to spectra.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            sample_temperatures: Optional per-sample temperatures. If None,
                uses config.sample_temperature with variation.

        Returns:
            Modified spectra with temperature effects applied.
        """
        n_samples = spectra.shape[0]
        result = spectra.copy()

        # Generate sample temperatures if not provided
        if sample_temperatures is None:
            base_temp = self.config.sample_temperature
            variation = self.config.temperature_variation
            if variation > 0:
                sample_temperatures = self.rng.normal(base_temp, variation, n_samples)
            else:
                sample_temperatures = np.full(n_samples, base_temp)

        # Compute temperature deltas from reference
        delta_temps = sample_temperatures - self.config.reference_temperature

        # Apply effects for each sample
        for i in range(n_samples):
            delta_t = delta_temps[i]
            if abs(delta_t) < 0.01:  # Skip if temperature is at reference
                continue

            result[i] = self._apply_temperature_to_spectrum(
                result[i], wavelengths, delta_t
            )

        return result

    def _apply_temperature_to_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        delta_temp: float
    ) -> np.ndarray:
        """Apply temperature effects to a single spectrum."""
        result = spectrum.copy()

        if self.config.region_specific:
            # Apply region-specific effects
            for region, params in self.region_params.items():
                result = self._apply_region_effect(
                    result, wavelengths, params, delta_temp
                )
        else:
            # Apply uniform effect across all wavelengths
            result = self._apply_uniform_effect(result, wavelengths, delta_temp)

        return result

    def _apply_region_effect(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        params: TemperatureEffectParams,
        delta_temp: float
    ) -> np.ndarray:
        """Apply temperature effect to a specific spectral region."""
        result = spectrum.copy()

        # Find wavelengths in this region
        wl_min, wl_max = params.wavelength_range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)

        if not np.any(mask):
            return result

        # Create smooth transition weights (avoid sharp edges)
        region_indices = np.where(mask)[0]
        weights = self._create_region_weights(wavelengths, wl_min, wl_max)

        # Apply wavelength shift
        if self.config.enable_shift:
            shift = params.shift_per_degree * delta_temp
            result = self._apply_wavelength_shift(result, wavelengths, shift, weights)

        # Apply intensity change
        if self.config.enable_intensity:
            intensity_factor = 1.0 + params.intensity_change_per_degree * delta_temp
            result = result * (1 + (intensity_factor - 1) * weights)

        # Apply broadening
        if self.config.enable_broadening and params.broadening_per_degree != 0:
            broadening_factor = 1.0 + params.broadening_per_degree * delta_temp
            if broadening_factor > 1.0:
                result = self._apply_broadening(result, wavelengths, broadening_factor, weights)

        return result

    def _apply_uniform_effect(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        delta_temp: float
    ) -> np.ndarray:
        """Apply uniform temperature effect across all wavelengths."""
        result = spectrum.copy()

        # Average parameters across all regions
        avg_shift = np.mean([p.shift_per_degree for p in self.region_params.values()])
        avg_intensity = np.mean([p.intensity_change_per_degree for p in self.region_params.values()])
        avg_broadening = np.mean([p.broadening_per_degree for p in self.region_params.values()])

        # Apply shift
        if self.config.enable_shift:
            shift = avg_shift * delta_temp
            result = np.interp(wavelengths, wavelengths + shift, result)

        # Apply intensity change
        if self.config.enable_intensity:
            intensity_factor = 1.0 + avg_intensity * delta_temp
            result = result * intensity_factor

        # Apply broadening
        if self.config.enable_broadening:
            broadening_factor = 1.0 + avg_broadening * delta_temp
            if broadening_factor > 1.0:
                sigma = (broadening_factor - 1.0) * 5  # Convert to smoothing sigma
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
        """
        Create smooth weights for a spectral region.

        Uses sigmoid transitions at edges to avoid discontinuities.
        """
        weights = np.zeros_like(wavelengths, dtype=float)

        # Sigmoid parameters
        k = 10.0 / edge_width  # Steepness

        # Rising edge at wl_min
        rising = 1.0 / (1.0 + np.exp(-k * (wavelengths - wl_min)))
        # Falling edge at wl_max
        falling = 1.0 / (1.0 + np.exp(k * (wavelengths - wl_max)))

        # Combine
        weights = rising * falling

        return weights

    def _apply_wavelength_shift(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        shift: float,
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply weighted wavelength shift."""
        # Create shifted wavelength grid
        shifted_wl = wavelengths + shift * weights

        # Interpolate spectrum to original grid
        return np.interp(wavelengths, shifted_wl, spectrum)

    def _apply_broadening(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        factor: float,
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply weighted spectral broadening."""
        # Convert factor to sigma
        sigma = (factor - 1.0) * 3.0

        if sigma < 0.1:
            return spectrum

        # Apply Gaussian smoothing
        broadened = gaussian_filter1d(spectrum, sigma)

        # Blend original and broadened based on weights
        return spectrum * (1 - weights) + broadened * weights


# ============================================================================
# Moisture/Water Activity Effect Simulator
# ============================================================================

class MoistureEffectSimulator:
    """
    Simulate moisture and water activity effects on NIR spectra.

    Water in samples exists in different states:
    - Free water: bulk water with normal O-H bands
    - Bound water: hydrogen-bonded to matrix, shifted peaks

    The ratio of free to bound water depends on water activity (a_w),
    temperature, and sample matrix composition.

    Attributes:
        config: Moisture effect configuration.
        rng: Random number generator.

    Example:
        >>> config = MoistureConfig(
        ...     water_activity=0.7,
        ...     moisture_content=0.15,
        ...     free_water_fraction=0.4
        ... )
        >>> simulator = MoistureEffectSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[MoistureConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the moisture effect simulator.

        Args:
            config: Moisture effect configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else MoistureConfig()
        self.rng = np.random.default_rng(random_state)

        # Water band positions (nm)
        self.free_water_peak_1st = 1410  # Free O-H 1st overtone
        self.bound_water_peak_1st = 1460  # Bound O-H 1st overtone
        self.free_water_peak_comb = 1920  # Free O-H combination
        self.bound_water_peak_comb = 1940  # Bound O-H combination

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        water_activities: Optional[np.ndarray] = None,
        temperature_offset: float = 0.0
    ) -> np.ndarray:
        """
        Apply moisture effects to spectra.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            water_activities: Optional per-sample water activity values.
            temperature_offset: Temperature offset from reference (affects water state).

        Returns:
            Modified spectra with moisture effects applied.
        """
        n_samples = spectra.shape[0]
        result = spectra.copy()

        # Get effective free water fraction (temperature-dependent)
        base_fraction = self.config.free_water_fraction
        if self.config.temperature_interaction and temperature_offset != 0:
            # Higher temperature = more free water (less H-bonding)
            temp_factor = 1.0 + 0.01 * temperature_offset
            base_fraction = np.clip(base_fraction * temp_factor, 0.0, 1.0)

        # Apply effects
        for i in range(n_samples):
            # Per-sample water activity variation
            if water_activities is not None:
                aw = water_activities[i]
            else:
                aw = self.config.water_activity

            # Compute effective free water fraction based on water activity
            # Higher a_w = more free water
            effective_fraction = self._compute_free_water_fraction(aw, base_fraction)

            result[i] = self._apply_water_state_effects(
                result[i], wavelengths, effective_fraction
            )

        return result

    def _compute_free_water_fraction(
        self,
        water_activity: float,
        base_fraction: float
    ) -> float:
        """
        Compute effective free water fraction based on water activity.

        At low a_w, most water is bound. At high a_w, more is free.
        Uses a sigmoid relationship.
        """
        # Sigmoid centered around a_w = 0.5
        # Low a_w (0.2) -> ~10% of base free fraction
        # High a_w (0.9) -> ~90% of base free fraction
        sigmoid_factor = 1.0 / (1.0 + np.exp(-8 * (water_activity - 0.5)))
        return base_fraction * sigmoid_factor

    def _apply_water_state_effects(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        free_fraction: float
    ) -> np.ndarray:
        """Apply water state (free vs bound) effects to spectrum."""
        result = spectrum.copy()
        bound_fraction = 1.0 - free_fraction

        # Apply to 1st overtone region (1400-1500 nm)
        result = self._shift_water_band(
            result, wavelengths,
            center=1435,  # Center of water band
            width=50,     # Band width
            shift=self.config.bound_water_shift * bound_fraction
        )

        # Apply to combination region (1900-2000 nm)
        result = self._shift_water_band(
            result, wavelengths,
            center=1930,  # Center of combination band
            width=40,     # Narrower band
            shift=self.config.bound_water_shift * 0.8 * bound_fraction  # Smaller shift
        )

        # Add intensity modification based on moisture content
        moisture_intensity = self.config.moisture_content
        water_region_1 = self._create_gaussian_region(wavelengths, 1435, 40)
        water_region_2 = self._create_gaussian_region(wavelengths, 1930, 35)

        # Enhance water bands based on moisture content
        # (relative to a nominal 10% baseline)
        enhancement = moisture_intensity / 0.10 - 1.0
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

        # Create Gaussian weighting for the band
        weights = self._create_gaussian_region(wavelengths, center, width)

        # Weighted shift
        shifted_wl = wavelengths + shift * weights

        # Interpolate
        return np.interp(wavelengths, shifted_wl, spectrum)

    def _create_gaussian_region(
        self,
        wavelengths: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """Create Gaussian weighting for a spectral region."""
        return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    def get_water_band_positions(
        self,
        free_fraction: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get expected water band positions for given free/bound ratio.

        Args:
            free_fraction: Fraction of water that is free (0-1).

        Returns:
            Dictionary with band names and (free_position, effective_position).
        """
        bound_fraction = 1.0 - free_fraction
        shift = self.config.bound_water_shift * bound_fraction

        return {
            "1st_overtone": (self.free_water_peak_1st, self.free_water_peak_1st + shift),
            "combination": (self.free_water_peak_comb, self.free_water_peak_comb + shift * 0.8),
        }


# ============================================================================
# Combined Environmental Effects Simulator
# ============================================================================

class EnvironmentalEffectsSimulator:
    """
    Combined simulator for all environmental effects.

    Applies temperature and moisture effects in the correct order
    with proper interactions.

    Attributes:
        config: Environmental effects configuration.
        temperature_sim: Temperature effect simulator.
        moisture_sim: Moisture effect simulator.
        rng: Random number generator.

    Example:
        >>> config = EnvironmentalEffectsConfig(
        ...     temperature=TemperatureConfig(sample_temperature=40.0),
        ...     moisture=MoistureConfig(water_activity=0.7)
        ... )
        >>> simulator = EnvironmentalEffectsSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[EnvironmentalEffectsConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the combined environmental effects simulator.

        Args:
            config: Environmental effects configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else EnvironmentalEffectsConfig()
        self.rng = np.random.default_rng(random_state)

        # Initialize component simulators
        self.temperature_sim = TemperatureEffectSimulator(
            self.config.temperature, random_state
        )
        self.moisture_sim = MoistureEffectSimulator(
            self.config.moisture, random_state
        )

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        sample_temperatures: Optional[np.ndarray] = None,
        water_activities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply all environmental effects to spectra.

        Effects are applied in order:
        1. Moisture effects (with temperature interaction)
        2. Temperature effects

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            sample_temperatures: Optional per-sample temperatures.
            water_activities: Optional per-sample water activities.

        Returns:
            Modified spectra with all environmental effects applied.
        """
        result = spectra.copy()
        n_samples = spectra.shape[0]

        # Compute temperature offset for moisture interaction
        if sample_temperatures is None:
            temp_offset = self.config.temperature.delta_temperature
        else:
            temp_offset = np.mean(sample_temperatures) - self.config.temperature.reference_temperature

        # Apply moisture effects first (temperature can affect water state)
        if self.config.enable_moisture:
            result = self.moisture_sim.apply(
                result, wavelengths,
                water_activities=water_activities,
                temperature_offset=temp_offset
            )

        # Apply temperature effects
        if self.config.enable_temperature:
            result = self.temperature_sim.apply(
                result, wavelengths,
                sample_temperatures=sample_temperatures
            )

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def apply_temperature_effects(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    temperature: float = 25.0,
    reference_temperature: float = 25.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply temperature effects to spectra with simple API.

    Args:
        spectra: Input spectra (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        temperature: Sample temperature in °C.
        reference_temperature: Reference temperature in °C.
        random_state: Random seed.

    Returns:
        Spectra with temperature effects applied.

    Example:
        >>> # Simulate spectra measured at 40°C
        >>> spectra_40c = apply_temperature_effects(spectra, wavelengths, temperature=40.0)
    """
    config = TemperatureConfig(
        sample_temperature=temperature,
        reference_temperature=reference_temperature
    )
    simulator = TemperatureEffectSimulator(config, random_state)
    return simulator.apply(spectra, wavelengths)


def apply_moisture_effects(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    water_activity: float = 0.5,
    moisture_content: float = 0.10,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply moisture effects to spectra with simple API.

    Args:
        spectra: Input spectra (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        water_activity: Water activity (0-1).
        moisture_content: Moisture content fraction.
        random_state: Random seed.

    Returns:
        Spectra with moisture effects applied.

    Example:
        >>> # Simulate wet sample (high water activity)
        >>> spectra_wet = apply_moisture_effects(spectra, wavelengths, water_activity=0.9)
    """
    config = MoistureConfig(
        water_activity=water_activity,
        moisture_content=moisture_content
    )
    simulator = MoistureEffectSimulator(config, random_state)
    return simulator.apply(spectra, wavelengths)


def simulate_temperature_series(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    temperatures: List[float],
    reference_temperature: float = 25.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate a series of spectra at different temperatures.

    Useful for simulating temperature studies or generating
    training data for temperature-robust models.

    Args:
        spectrum: Single reference spectrum (n_wavelengths,).
        wavelengths: Wavelength array (nm).
        temperatures: List of temperatures to simulate.
        reference_temperature: Reference temperature for the input spectrum.
        random_state: Random seed.

    Returns:
        Array of spectra (n_temperatures, n_wavelengths).

    Example:
        >>> temps = [20, 25, 30, 35, 40]
        >>> temp_series = simulate_temperature_series(spectrum, wavelengths, temps)
    """
    n_temps = len(temperatures)
    result = np.zeros((n_temps, len(wavelengths)))

    for i, temp in enumerate(temperatures):
        config = TemperatureConfig(
            sample_temperature=temp,
            reference_temperature=reference_temperature
        )
        simulator = TemperatureEffectSimulator(config, random_state)
        result[i:i+1] = simulator.apply(spectrum[np.newaxis, :], wavelengths)

    return result


def get_temperature_effect_regions() -> Dict[str, Tuple[float, float]]:
    """
    Get the wavelength regions with significant temperature effects.

    Returns:
        Dictionary mapping region names to (start, end) wavelength tuples.
    """
    return {
        region.value: params.wavelength_range
        for region, params in TEMPERATURE_EFFECT_PARAMS.items()
    }


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Enums
    "SpectralRegion",
    # Dataclasses
    "TemperatureEffectParams",
    "TemperatureConfig",
    "MoistureConfig",
    "EnvironmentalEffectsConfig",
    # Constants
    "TEMPERATURE_EFFECT_PARAMS",
    # Simulators
    "TemperatureEffectSimulator",
    "MoistureEffectSimulator",
    "EnvironmentalEffectsSimulator",
    # Convenience functions
    "apply_temperature_effects",
    "apply_moisture_effects",
    "simulate_temperature_series",
    "get_temperature_effect_regions",
]
