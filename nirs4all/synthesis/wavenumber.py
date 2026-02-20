"""
Wavenumber conversion utilities for NIR spectroscopy.

This module provides utilities for converting between wavenumber (cm⁻¹) and
wavelength (nm) units, which is essential for physically-correct band placement.

Harmonic relationships (overtones, combinations) are LINEAR in wavenumber space,
not in wavelength space. This is a critical distinction for generating realistic
synthetic spectra with proper overtone and combination band placement.

Mathematical relationships:
    - λ (nm) = 10^7 / ν̃ (cm⁻¹)
    - ν̃ (cm⁻¹) = 10^7 / λ (nm)
    - Δλ ≈ Δν̃ × λ² / 10^7 (bandwidth conversion)

References:
    - Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (2002). Near-Infrared
      Spectroscopy: Principles, Instruments, Applications. Wiley-VCH.
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# ============================================================================
# Spectral Zones in Wavenumber Space (Vis-NIR: 350-2500 nm)
# ============================================================================

# Phase 2 Extension: Added visible-region zones for electronic transitions
# These cover the full Vis-NIR range commonly used in spectroscopy

# Extended spectral zones defined in wavenumber (cm⁻¹) for physically-correct band placement
# Includes both visible (electronic transitions) and NIR (vibrational overtones/combinations)
EXTENDED_SPECTRAL_ZONES: list[tuple[float, float, str, str]] = [
    # Visible region - electronic transitions (350-700 nm)
    (14285, 28571, "visible_electronic", "Electronic transitions, pigments"),  # 350-700 nm

    # Red-edge / Short-wave NIR - electronic tail (700-800 nm)
    (12500, 14285, "red_edge", "Chlorophyll red edge, electronic tail"),       # 700-800 nm

    # Short-wave NIR: 3rd overtones (800-1111 nm)
    (9000, 12500, "3rd_overtones", "3rd overtones C-H, O-H, N-H"),             # 800-1111 nm

    # 2nd overtone region (1111-1333 nm)
    (7500, 9000, "2nd_overtones", "2nd overtones C-H"),                         # 1111-1333 nm

    # 1st overtone region - O-H, N-H (1333-1667 nm)
    (6000, 7500, "1st_overtones_OH_NH", "1st overtones O-H, N-H"),             # 1333-1667 nm

    # 1st overtone region - C-H (1600-1818 nm)
    (5500, 6250, "1st_overtones_CH", "1st overtones C-H"),                      # 1600-1818 nm

    # Combination band region - O-H (1818-2000 nm)
    (5000, 5500, "combination_OH", "O-H combination bands"),                    # 1818-2000 nm

    # Combination band region - N-H (1923-2222 nm)
    (4500, 5200, "combination_NH", "N-H combination bands"),                    # 1923-2222 nm

    # Combination band region - C-H, C-O (2200-2500 nm)
    (4000, 4545, "combination_CH", "C-H combination bands"),                    # 2200-2500 nm
]

# Backward-compatible NIR zones (3-tuple format for existing code)
# These zones correspond to specific molecular vibration types
NIR_ZONES_WAVENUMBER: list[tuple[float, float, str]] = [
    # Short-wave NIR: Electronic transitions and 3rd overtones
    (9000, 12500, "3rd_overtones"),  # ~800-1111 nm

    # 2nd overtone region
    (7500, 9000, "2nd_overtones"),   # ~1111-1333 nm

    # 1st overtone region - O-H, N-H
    (6000, 7500, "1st_overtones_OH_NH"),  # ~1333-1667 nm

    # 1st overtone region - C-H
    (5500, 6250, "1st_overtones_CH"),  # ~1600-1818 nm

    # Combination band region - O-H
    (5000, 5500, "combination_OH"),  # ~1818-2000 nm

    # Combination band region - N-H, C-H
    (4500, 5200, "combination_NH"),  # ~1923-2222 nm

    # Combination band region - C-H, C-O
    (4000, 4545, "combination_CH"),  # ~2200-2500 nm
]

# Visible region zones (electronic transitions, pigments)
# These are separate from vibrational NIR zones
VISIBLE_ZONES_WAVENUMBER: list[tuple[float, float, str, str]] = [
    # UV-Vis transition region
    (20000, 28571, "uv_vis_transition", "UV-visible transition, aromatic absorptions"),  # 350-500 nm

    # Blue region - Soret bands, carotenoid absorptions
    (20000, 25000, "blue_absorption", "Soret bands, carotenoid peak absorptions"),       # 400-500 nm

    # Green region - anthocyanins, flavonoids
    (16667, 20000, "green_absorption", "Anthocyanins, flavonoids"),                       # 500-600 nm

    # Red region - chlorophyll Q bands, hemoglobin
    (14285, 16667, "red_absorption", "Chlorophyll Q bands, hemoglobin bands"),           # 600-700 nm

    # Red-edge - chlorophyll fluorescence, electronic tail
    (12500, 14285, "red_edge", "Chlorophyll red edge, electronic transition tail"),      # 700-800 nm
]

# Fundamental vibration wavenumbers for common functional groups (cm⁻¹)
# Used for calculating overtone and combination band positions
FUNDAMENTAL_VIBRATIONS = {
    # O-H vibrations
    "O-H_stretch_free": 3650,      # Free hydroxyl
    "O-H_stretch_hbond": 3400,     # Hydrogen-bonded hydroxyl
    "O-H_bend": 1640,              # O-H bending (water)

    # N-H vibrations
    "N-H_stretch_primary": 3400,   # Primary amine
    "N-H_stretch_secondary": 3300, # Secondary amine / amide
    "N-H_bend": 1550,              # N-H bending

    # C-H vibrations
    "C-H_stretch_CH3_asym": 2960,  # CH3 asymmetric stretch
    "C-H_stretch_CH3_sym": 2870,   # CH3 symmetric stretch
    "C-H_stretch_CH2_asym": 2930,  # CH2 asymmetric stretch
    "C-H_stretch_CH2_sym": 2850,   # CH2 symmetric stretch
    "C-H_stretch_aromatic": 3060,  # Aromatic C-H
    "C-H_stretch_vinyl": 3100,     # =C-H vinyl
    "C-H_bend": 1450,              # C-H bending (scissors)

    # C=O vibrations
    "C=O_stretch_ester": 1740,     # Ester carbonyl
    "C=O_stretch_ketone": 1715,    # Ketone carbonyl
    "C=O_stretch_amide": 1650,     # Amide I (C=O stretch)
    "C=O_stretch_carboxylic": 1710, # Carboxylic acid

    # Other vibrations
    "C=C_stretch": 1650,           # C=C double bond
    "C-O_stretch": 1100,           # C-O single bond
    "C-N_stretch": 1200,           # C-N single bond
    "S-H_stretch": 2550,           # Thiol
    "P-H_stretch": 2400,           # Phosphine
}

# ============================================================================
# Conversion Functions
# ============================================================================

def wavenumber_to_wavelength(nu_cm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavenumber (cm⁻¹) to wavelength (nm).

    The conversion follows the relationship: λ = 10^7 / ν̃

    Args:
        nu_cm: Wavenumber in cm⁻¹. Can be a scalar or numpy array.

    Returns:
        Wavelength in nm (same shape as input).

    Raises:
        ValueError: If wavenumber is zero or negative.

    Example:
        >>> wavenumber_to_wavelength(6896)  # 1st overtone O-H
        1450.26...
        >>> wavenumber_to_wavelength(np.array([6896, 5155]))
        array([1450.26..., 1939.88...])
    """
    nu_cm = np.asarray(nu_cm)
    if np.any(nu_cm <= 0):
        raise ValueError("Wavenumber must be positive")
    return 1e7 / nu_cm

def wavelength_to_wavenumber(lambda_nm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength (nm) to wavenumber (cm⁻¹).

    The conversion follows the relationship: ν̃ = 10^7 / λ

    Args:
        lambda_nm: Wavelength in nm. Can be a scalar or numpy array.

    Returns:
        Wavenumber in cm⁻¹ (same shape as input).

    Raises:
        ValueError: If wavelength is zero or negative.

    Example:
        >>> wavelength_to_wavenumber(1450)  # O-H 1st overtone region
        6896.55...
        >>> wavelength_to_wavenumber(np.array([1450, 1940]))
        array([6896.55..., 5154.64...])
    """
    lambda_nm = np.asarray(lambda_nm)
    if np.any(lambda_nm <= 0):
        raise ValueError("Wavelength must be positive")
    return 1e7 / lambda_nm

def convert_bandwidth_to_wavelength(
    bandwidth_cm: float,
    center_nm: float
) -> float:
    """
    Convert bandwidth from wavenumber to wavelength units.

    Since the relationship between wavenumber and wavelength is non-linear,
    the bandwidth conversion depends on the center wavelength/wavenumber.

    The approximation is: Δλ ≈ Δν̃ × λ² / 10^7

    This is derived from the differential: dλ = -dν̃ × (10^7 / ν̃²) = -dν̃ × λ² / 10^7
    (taking absolute value for bandwidth)

    Args:
        bandwidth_cm: Bandwidth in cm⁻¹ (e.g., FWHM).
        center_nm: Center wavelength in nm.

    Returns:
        Bandwidth in nm.

    Example:
        >>> convert_bandwidth_to_wavelength(100, 1450)  # 100 cm⁻¹ at 1450 nm
        21.025  # approximately
        >>> convert_bandwidth_to_wavelength(100, 2200)  # Same bandwidth at 2200 nm
        48.4    # Broader in nm due to non-linear relationship
    """
    return bandwidth_cm * (center_nm ** 2) / 1e7

def convert_bandwidth_to_wavenumber(
    bandwidth_nm: float,
    center_nm: float
) -> float:
    """
    Convert bandwidth from wavelength to wavenumber units.

    The inverse of convert_bandwidth_to_wavelength:
    Δν̃ ≈ Δλ × 10^7 / λ²

    Args:
        bandwidth_nm: Bandwidth in nm (e.g., FWHM).
        center_nm: Center wavelength in nm.

    Returns:
        Bandwidth in cm⁻¹.

    Example:
        >>> convert_bandwidth_to_wavenumber(25, 1450)  # 25 nm at 1450 nm
        118.9   # approximately
    """
    return bandwidth_nm * 1e7 / (center_nm ** 2)

# ============================================================================
# Zone and Region Functions
# ============================================================================

def get_zone_wavelength_range(zone_name: str) -> tuple[float, float] | None:
    """
    Get the wavelength range (nm) for a named NIR zone.

    Args:
        zone_name: Name of the NIR zone (e.g., '1st_overtones_OH_NH').

    Returns:
        Tuple of (min_wavelength, max_wavelength) in nm, or None if not found.

    Example:
        >>> get_zone_wavelength_range('1st_overtones_CH')
        (1600.0, 1818.18...)
    """
    for nu_max, nu_min, name in NIR_ZONES_WAVENUMBER:
        if name == zone_name:
            # Note: higher wavenumber = lower wavelength
            return (float(wavenumber_to_wavelength(nu_max)), float(wavenumber_to_wavelength(nu_min)))
    return None

def get_all_zones_wavelength() -> list[tuple[float, float, str]]:
    """
    Get all NIR zones converted to wavelength space.

    Returns:
        List of (min_wavelength, max_wavelength, zone_name) tuples in nm.

    Example:
        >>> zones = get_all_zones_wavelength()
        >>> for min_wl, max_wl, name in zones:
        ...     print(f"{name}: {min_wl:.0f}-{max_wl:.0f} nm")
    """
    zones = []
    for nu_max, nu_min, name in NIR_ZONES_WAVENUMBER:
        wl_min = float(wavenumber_to_wavelength(nu_max))
        wl_max = float(wavenumber_to_wavelength(nu_min))
        zones.append((wl_min, wl_max, name))
    return zones

def classify_wavelength_zone(wavelength_nm: float) -> str | None:
    """
    Classify a wavelength into its corresponding NIR zone.

    Args:
        wavelength_nm: Wavelength in nm.

    Returns:
        Zone name string, or None if outside defined zones.

    Example:
        >>> classify_wavelength_zone(1450)
        '1st_overtones_OH_NH'
        >>> classify_wavelength_zone(2300)
        'combination_CH'
    """
    wavenumber = wavelength_to_wavenumber(wavelength_nm)
    for nu_min, nu_max, name in NIR_ZONES_WAVENUMBER:
        if nu_min <= wavenumber <= nu_max:
            return name
    return None

def classify_wavelength_extended(wavelength_nm: float) -> tuple[str, str] | None:
    """
    Classify a wavelength into extended spectral zones (Vis-NIR: 350-2500 nm).

    This function covers both visible (electronic transitions) and NIR
    (vibrational overtones/combinations) regions.

    Args:
        wavelength_nm: Wavelength in nm.

    Returns:
        Tuple of (zone_name, description), or None if outside defined zones.

    Example:
        >>> classify_wavelength_extended(450)
        ('blue_absorption', 'Soret bands, carotenoid peak absorptions')
        >>> classify_wavelength_extended(660)
        ('red_absorption', 'Chlorophyll Q bands, hemoglobin bands')
        >>> classify_wavelength_extended(1450)
        ('1st_overtones_OH_NH', '1st overtones O-H, N-H')
    """
    wavenumber = wavelength_to_wavenumber(wavelength_nm)

    # Check extended zones first (includes visible region)
    for nu_min, nu_max, name, description in EXTENDED_SPECTRAL_ZONES:
        if nu_min <= wavenumber <= nu_max:
            return (name, description)

    # Also check visible-specific zones
    for nu_min, nu_max, name, description in VISIBLE_ZONES_WAVENUMBER:
        if nu_min <= wavenumber <= nu_max:
            return (name, description)

    return None

def get_all_zones_extended() -> list[tuple[float, float, str, str]]:
    """
    Get all extended spectral zones (Vis-NIR) converted to wavelength space.

    Returns:
        List of (min_wavelength, max_wavelength, zone_name, description) tuples in nm.

    Example:
        >>> zones = get_all_zones_extended()
        >>> for min_wl, max_wl, name, desc in zones:
        ...     print(f"{name}: {min_wl:.0f}-{max_wl:.0f} nm - {desc}")
    """
    zones = []
    for nu_max, nu_min, name, description in EXTENDED_SPECTRAL_ZONES:
        wl_min = float(wavenumber_to_wavelength(nu_max))
        wl_max = float(wavenumber_to_wavelength(nu_min))
        zones.append((wl_min, wl_max, name, description))
    return zones

def is_visible_region(wavelength_nm: float) -> bool:
    """
    Check if a wavelength is in the visible region (350-700 nm).

    Args:
        wavelength_nm: Wavelength in nm.

    Returns:
        True if wavelength is in visible region.

    Example:
        >>> is_visible_region(500)
        True
        >>> is_visible_region(1450)
        False
    """
    return 350 <= wavelength_nm <= 700

def is_nir_region(wavelength_nm: float) -> bool:
    """
    Check if a wavelength is in the NIR region (700-2500 nm).

    Args:
        wavelength_nm: Wavelength in nm.

    Returns:
        True if wavelength is in NIR region.

    Example:
        >>> is_nir_region(1450)
        True
        >>> is_nir_region(500)
        False
    """
    return 700 <= wavelength_nm <= 2500

# ============================================================================
# Overtone and Combination Band Calculations
# ============================================================================

@dataclass
class OvertoneResult:
    """Result of overtone calculation."""
    order: int                  # Overtone order (1 = fundamental, 2 = 1st overtone, etc.)
    wavenumber_cm: float        # Position in cm⁻¹
    wavelength_nm: float        # Position in nm
    amplitude_factor: float     # Relative amplitude (decreases with order)
    bandwidth_factor: float     # Relative bandwidth (increases with order)

    def __repr__(self) -> str:
        order_name = {1: "fundamental", 2: "1st overtone", 3: "2nd overtone",
                      4: "3rd overtone"}.get(self.order, f"{self.order-1}th overtone")
        return (f"OvertoneResult({order_name}: {self.wavelength_nm:.1f} nm / "
                f"{self.wavenumber_cm:.0f} cm⁻¹, amp={self.amplitude_factor:.3f})")

def calculate_overtone_position(
    vibration_type_or_frequency: str | float,
    overtone_order: int,
    anharmonicity: float | None = None
) -> OvertoneResult:
    """
    Calculate overtone band position with anharmonicity correction.

    For a harmonic oscillator, overtones would be exactly at n × ν̃₀. However,
    real molecular vibrations are anharmonic, causing overtones to appear at
    slightly lower wavenumbers than the harmonic prediction.

    The anharmonic wavenumber is: ν̃ₙ = n × ν̃₀ × (1 - n × χ)
    where χ is the anharmonicity constant (typically 0.01-0.03).

    Args:
        vibration_type_or_frequency: Either a vibration type string (e.g., 'O-H_stretch')
            from FUNDAMENTAL_VIBRATIONS, or a numeric fundamental frequency in cm⁻¹.
        overtone_order: Order (1 = fundamental, 2 = 1st overtone, 3 = 2nd overtone).
        anharmonicity: Anharmonicity constant χ. If None and vibration_type is a string,
            uses the default for that vibration type. Otherwise defaults to 0.02.

    Returns:
        OvertoneResult with position and intensity information.

    Example:
        >>> result = calculate_overtone_position("O-H_stretch", 2)  # O-H 1st overtone
        >>> print(f"{result.wavelength_nm:.0f} nm")
        1442 nm  # (with anharmonicity)

        >>> result = calculate_overtone_position(3400, 2)  # Numeric frequency
        >>> print(f"{result.wavelength_nm:.0f} nm")
        1503 nm
    """
    if overtone_order < 1:
        raise ValueError("Overtone order must be >= 1")

    # Resolve vibration type to frequency and anharmonicity
    if isinstance(vibration_type_or_frequency, str):
        vib_type = vibration_type_or_frequency
        if vib_type not in FUNDAMENTAL_VIBRATIONS:
            raise KeyError(f"Unknown vibration type: {vib_type}. "
                          f"Available types: {list(FUNDAMENTAL_VIBRATIONS.keys())}")
        fundamental_cm = float(FUNDAMENTAL_VIBRATIONS[vib_type])
        if anharmonicity is None:
            # Use typical anharmonicity values based on bond type
            if "O-H" in vib_type or "N-H" in vib_type:
                anharmonicity = 0.022  # X-H bonds with heavier X
            elif "C-H" in vib_type:
                anharmonicity = 0.020  # C-H bonds
            elif "S-H" in vib_type:
                anharmonicity = 0.025  # S-H bonds
            else:
                anharmonicity = 0.015  # Other bonds (lower anharmonicity)
    else:
        fundamental_cm = float(vibration_type_or_frequency)
        if anharmonicity is None:
            anharmonicity = 0.02  # Default anharmonicity

    # Apply anharmonicity correction
    # ν̃ₙ = n × ν̃₀ × (1 - n × χ)
    nu_harmonic = overtone_order * fundamental_cm
    nu_anharmonic = nu_harmonic * (1 - overtone_order * anharmonicity)

    # Amplitude decreases exponentially with overtone order
    # Roughly follows: A_n ≈ A_1 × 10^(-n/2)
    amplitude_factor = 10 ** (-(overtone_order - 1) / 2)

    # Bandwidth increases with overtone order (roughly linearly)
    bandwidth_factor = 1.0 + 0.3 * (overtone_order - 1)

    return OvertoneResult(
        order=overtone_order,
        wavenumber_cm=nu_anharmonic,
        wavelength_nm=float(wavenumber_to_wavelength(nu_anharmonic)),
        amplitude_factor=amplitude_factor,
        bandwidth_factor=bandwidth_factor,
    )

@dataclass
class CombinationBandResult:
    """Result of combination band calculation."""
    mode1_cm: float             # First fundamental wavenumber
    mode2_cm: float             # Second fundamental wavenumber
    wavenumber_cm: float        # Combination band position in cm⁻¹
    wavelength_nm: float        # Combination band position in nm
    amplitude_factor: float     # Relative amplitude
    band_type: str              # 'sum' or 'difference'

    def __repr__(self) -> str:
        return (f"CombinationBandResult({self.band_type}: "
                f"{self.wavelength_nm:.1f} nm / {self.wavenumber_cm:.0f} cm⁻¹)")

def _resolve_vibration_to_frequency(vibration: str | float) -> float:
    """Resolve vibration type or frequency to numeric frequency."""
    if isinstance(vibration, str):
        if vibration not in FUNDAMENTAL_VIBRATIONS:
            raise KeyError(f"Unknown vibration type: {vibration}. "
                          f"Available types: {list(FUNDAMENTAL_VIBRATIONS.keys())}")
        return float(FUNDAMENTAL_VIBRATIONS[vibration])
    return float(vibration)

def calculate_combination_band(
    mode1: str | float | list[str | float],
    mode2: str | float | None = None,
    band_type: str = "sum",
    coupling_factor: float = 1.0
) -> CombinationBandResult:
    """
    Calculate combination band position.

    Combination bands arise from simultaneous excitation of two vibrational modes.
    - Sum bands: ν̃_comb = ν̃₁ + ν̃₂ (most common in NIR)
    - Difference bands: ν̃_comb = |ν̃₁ - ν̃₂| (less common)

    Args:
        mode1: First vibration - either a vibration type string (e.g., 'O-H_stretch'),
            a numeric wavenumber in cm⁻¹, or a list of two modes.
        mode2: Second vibration (same format as mode1). If mode1 is a list,
            this should be None.
        band_type: 'sum' or 'difference'.
        coupling_factor: Mechanical coupling between modes (0-1, affects amplitude).

    Returns:
        CombinationBandResult with position and intensity information.

    Example:
        >>> # O-H stretch + O-H bend combination (water) using strings
        >>> result = calculate_combination_band("O-H_stretch", "O-H_bend")
        >>> print(f"{result.wavelength_nm:.0f} nm")
        1984 nm

        >>> # Using a list of modes
        >>> result = calculate_combination_band(["O-H_stretch", "O-H_bend"])

        >>> # Using numeric values
        >>> result = calculate_combination_band(3400, 1640)
    """
    # Handle list input
    if isinstance(mode1, list):
        if len(mode1) != 2:
            raise ValueError("List mode must contain exactly 2 elements")
        mode1_val = _resolve_vibration_to_frequency(mode1[0])
        mode2_val = _resolve_vibration_to_frequency(mode1[1])
    else:
        if mode2 is None:
            raise ValueError("mode2 is required unless mode1 is a list")
        mode1_val = _resolve_vibration_to_frequency(mode1)
        mode2_val = _resolve_vibration_to_frequency(mode2)

    if band_type == "sum":
        nu_comb = mode1_val + mode2_val
    elif band_type == "difference":
        nu_comb = abs(mode1_val - mode2_val)
    else:
        raise ValueError("band_type must be 'sum' or 'difference'")

    # Combination bands are typically weaker than overtones
    # Amplitude depends on mechanical/electrical coupling
    amplitude_factor = 0.2 * coupling_factor

    return CombinationBandResult(
        mode1_cm=mode1_val,
        mode2_cm=mode2_val,
        wavenumber_cm=nu_comb,
        wavelength_nm=float(wavenumber_to_wavelength(nu_comb)),
        amplitude_factor=amplitude_factor,
        band_type=band_type,
    )

def get_nir_overtones_for_fundamental(
    fundamental_cm: float,
    max_order: int = 4,
    wavelength_range: tuple[float, float] = (800, 2500),
    anharmonicity: float = 0.02
) -> list[OvertoneResult]:
    """
    Get all overtones of a fundamental that fall within the NIR range.

    Args:
        fundamental_cm: Fundamental vibration wavenumber in cm⁻¹.
        max_order: Maximum overtone order to consider.
        wavelength_range: NIR wavelength range in nm (min, max).
        anharmonicity: Anharmonicity constant.

    Returns:
        List of OvertoneResult objects for bands within the NIR range.

    Example:
        >>> # Get NIR overtones of O-H stretch
        >>> overtones = get_nir_overtones_for_fundamental(3400)
        >>> for ot in overtones:
        ...     print(ot)
    """
    results = []
    for order in range(1, max_order + 1):
        ot = calculate_overtone_position(fundamental_cm, order, anharmonicity)
        if wavelength_range[0] <= ot.wavelength_nm <= wavelength_range[1]:
            results.append(ot)
    return results

# ============================================================================
# Hydrogen Bonding Effects
# ============================================================================

def apply_hydrogen_bonding_shift(
    wavenumber_cm: float,
    h_bond_strength: float = 0.5,
    is_donor: bool = True
) -> float:
    """
    Apply hydrogen bonding shift to a wavenumber.

    Hydrogen bonding weakens X-H bonds, shifting stretching frequencies
    to lower wavenumbers (red shift). The shift magnitude depends on
    the hydrogen bond strength.

    Typical shifts for O-H:
    - Free O-H: ~3650 cm⁻¹
    - Weak H-bond: ~3500 cm⁻¹
    - Strong H-bond: ~3200 cm⁻¹

    Args:
        wavenumber_cm: Original wavenumber in cm⁻¹.
        h_bond_strength: Hydrogen bond strength (0 = none, 1 = very strong).
        is_donor: Whether the group is a hydrogen bond donor.

    Returns:
        Shifted wavenumber in cm⁻¹.

    Example:
        >>> apply_hydrogen_bonding_shift(3650, h_bond_strength=0.5)
        3467.5  # Red-shifted by hydrogen bonding
    """
    if not is_donor:
        # Acceptors show smaller shifts
        h_bond_strength *= 0.3

    # Maximum shift is about 10-15% of the wavenumber
    max_shift_fraction = 0.12
    shift = wavenumber_cm * max_shift_fraction * h_bond_strength

    return wavenumber_cm - shift

def estimate_bandwidth_broadening(
    baseline_bandwidth_cm: float,
    h_bond_strength: float = 0.0,
    temperature_k: float = 298.0
) -> float:
    """
    Estimate bandwidth broadening due to environmental effects.

    Hydrogen bonding and temperature increase band widths through:
    - Distribution of H-bond geometries (inhomogeneous broadening)
    - Thermal population of vibrational levels

    Args:
        baseline_bandwidth_cm: Intrinsic bandwidth in cm⁻¹.
        h_bond_strength: Hydrogen bond strength (0-1).
        temperature_k: Temperature in Kelvin.

    Returns:
        Broadened bandwidth in cm⁻¹.

    Example:
        >>> estimate_bandwidth_broadening(50, h_bond_strength=0.7)
        85.0  # Broadened by hydrogen bonding
    """
    # H-bonding broadening: can double or triple bandwidth
    h_bond_broadening = 1.0 + 1.5 * h_bond_strength

    # Temperature broadening: roughly linear with sqrt(T)
    # Reference at 298 K
    temp_factor = np.sqrt(temperature_k / 298.0)

    return float(baseline_bandwidth_cm * h_bond_broadening * temp_factor)

# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Constants
    "NIR_ZONES_WAVENUMBER",
    "EXTENDED_SPECTRAL_ZONES",
    "VISIBLE_ZONES_WAVENUMBER",
    "FUNDAMENTAL_VIBRATIONS",
    # Basic conversions
    "wavenumber_to_wavelength",
    "wavelength_to_wavenumber",
    "convert_bandwidth_to_wavelength",
    "convert_bandwidth_to_wavenumber",
    # Zone functions
    "get_zone_wavelength_range",
    "get_all_zones_wavelength",
    "classify_wavelength_zone",
    # Extended zone functions (Phase 2)
    "classify_wavelength_extended",
    "get_all_zones_extended",
    "is_visible_region",
    "is_nir_region",
    # Overtone/combination calculations
    "OvertoneResult",
    "CombinationBandResult",
    "calculate_overtone_position",
    "calculate_combination_band",
    "get_nir_overtones_for_fundamental",
    # Environmental effects
    "apply_hydrogen_bonding_shift",
    "estimate_bandwidth_broadening",
]
