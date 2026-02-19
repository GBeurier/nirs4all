"""
Environmental effects configuration for synthetic NIRS data generation.

This module provides configuration classes for environmental and matrix effects
on NIR spectra, including temperature-induced changes and moisture/water activity
effects.

Note:
    For applying environmental effects to spectra, use the operators in
    `nirs4all.operators.augmentation.environmental`:
    - TemperatureAugmenter: Temperature-induced spectral changes
    - MoistureAugmenter: Moisture/water activity effects

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
from enum import Enum, StrEnum
from typing import Optional

# ============================================================================
# Temperature Effect Parameters by Spectral Region
# ============================================================================

class SpectralRegion(StrEnum):
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
    wavelength_range: tuple[float, float]
    shift_per_degree: float          # nm/°C
    intensity_change_per_degree: float  # fraction/°C (e.g., -0.002 = -0.2%/°C)
    broadening_per_degree: float     # fraction/°C
    reference: str = ""

# Literature-based temperature effect parameters
TEMPERATURE_EFFECT_PARAMS: dict[SpectralRegion, TemperatureEffectParams] = {
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
    custom_regions: dict[SpectralRegion, TemperatureEffectParams] | None = None

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
# Utility Functions
# ============================================================================

def get_temperature_effect_regions() -> dict[str, tuple[float, float]]:
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
    # Utility functions
    "get_temperature_effect_regions",
]
