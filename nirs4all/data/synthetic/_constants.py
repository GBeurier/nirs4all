"""
Predefined constants for synthetic NIRS spectra generation.

This module contains predefined spectral components and band assignments
based on established NIR spectroscopy literature.

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
      CRC Press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict
    from .components import SpectralComponent

# Lazy imports to avoid circular dependencies
_PREDEFINED_COMPONENTS: "Dict[str, SpectralComponent] | None" = None


def get_predefined_components() -> "Dict[str, SpectralComponent]":
    """
    Get predefined spectral components based on NIR band assignments.

    Returns:
        Dictionary mapping component names to SpectralComponent objects.

    Note:
        This function uses lazy initialization to avoid circular imports.
        The components are created once and cached for subsequent calls.
    """
    global _PREDEFINED_COMPONENTS

    if _PREDEFINED_COMPONENTS is None:
        from .components import NIRBand, SpectralComponent

        _PREDEFINED_COMPONENTS = {
            "water": SpectralComponent(
                name="water",
                bands=[
                    NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8, name="O-H 1st overtone"),
                    NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0, name="O-H combination"),
                    NIRBand(center=2500, sigma=50, gamma=5, amplitude=0.3, name="O-H stretch + bend"),
                ],
                correlation_group=1,
            ),
            "protein": SpectralComponent(
                name="protein",
                bands=[
                    NIRBand(center=1510, sigma=20, gamma=2, amplitude=0.5, name="N-H 1st overtone"),
                    NIRBand(center=1680, sigma=25, gamma=3, amplitude=0.4, name="C-H aromatic"),
                    NIRBand(center=2050, sigma=30, gamma=3, amplitude=0.6, name="N-H combination"),
                    NIRBand(center=2180, sigma=25, gamma=2, amplitude=0.5, name="Protein C-H"),
                    NIRBand(center=2300, sigma=20, gamma=2, amplitude=0.3, name="N-H+Amide III"),
                ],
                correlation_group=2,
            ),
            "lipid": SpectralComponent(
                name="lipid",
                bands=[
                    NIRBand(center=1210, sigma=20, gamma=2, amplitude=0.4, name="C-H 2nd overtone"),
                    NIRBand(center=1390, sigma=15, gamma=1, amplitude=0.3, name="C-H combination"),
                    NIRBand(center=1720, sigma=25, gamma=2, amplitude=0.6, name="C-H 1st overtone"),
                    NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.5, name="CH2 combination"),
                    NIRBand(center=2350, sigma=18, gamma=2, amplitude=0.4, name="CH3 combination"),
                ],
                correlation_group=3,
            ),
            "starch": SpectralComponent(
                name="starch",
                bands=[
                    NIRBand(center=1460, sigma=25, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
                    NIRBand(center=1580, sigma=20, gamma=2, amplitude=0.3, name="Starch combination"),
                    NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.6, name="O-H+C-O combination"),
                    NIRBand(center=2270, sigma=25, gamma=2, amplitude=0.4, name="C-O+C-C stretch"),
                ],
                correlation_group=4,
            ),
            "cellulose": SpectralComponent(
                name="cellulose",
                bands=[
                    NIRBand(center=1490, sigma=22, gamma=2, amplitude=0.4, name="O-H 1st overtone"),
                    NIRBand(center=1780, sigma=18, gamma=2, amplitude=0.3, name="Cellulose C-H"),
                    NIRBand(center=2090, sigma=28, gamma=3, amplitude=0.5, name="O-H combination"),
                    NIRBand(center=2280, sigma=22, gamma=2, amplitude=0.4, name="Cellulose C-O"),
                    NIRBand(center=2340, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=4,
            ),
            "chlorophyll": SpectralComponent(
                name="chlorophyll",
                bands=[
                    NIRBand(center=1070, sigma=15, gamma=1, amplitude=0.3, name="Chl absorption"),
                    NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.4, name="C-H 1st overtone"),
                    NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
                ],
                correlation_group=5,
            ),
            "oil": SpectralComponent(
                name="oil",
                bands=[
                    NIRBand(center=1165, sigma=18, gamma=2, amplitude=0.35, name="C-H 2nd overtone"),
                    NIRBand(center=1215, sigma=16, gamma=1.5, amplitude=0.3, name="CH2 2nd overtone"),
                    NIRBand(center=1410, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
                    NIRBand(center=1725, sigma=22, gamma=2, amplitude=0.7, name="C-H 1st overtone"),
                    NIRBand(center=2140, sigma=25, gamma=2, amplitude=0.4, name="C=C unsaturation"),
                    NIRBand(center=2305, sigma=18, gamma=2, amplitude=0.5, name="CH2 combination"),
                ],
                correlation_group=3,
            ),
            "nitrogen_compound": SpectralComponent(
                name="nitrogen_compound",
                bands=[
                    NIRBand(center=1500, sigma=18, gamma=2, amplitude=0.45, name="N-H 1st overtone"),
                    NIRBand(center=2060, sigma=25, gamma=2, amplitude=0.5, name="N-H combination"),
                    NIRBand(center=2150, sigma=22, gamma=2, amplitude=0.4, name="N-H+C-N"),
                ],
                correlation_group=2,
            ),
        }

    return _PREDEFINED_COMPONENTS


# Default wavelength parameters
DEFAULT_WAVELENGTH_START: float = 1000.0
DEFAULT_WAVELENGTH_END: float = 2500.0
DEFAULT_WAVELENGTH_STEP: float = 2.0

# Default NIR-relevant zones for random band placement
DEFAULT_NIR_ZONES = [
    (1100, 1300),  # 2nd overtones
    (1400, 1550),  # 1st overtones O-H, N-H
    (1650, 1800),  # 1st overtones C-H
    (1850, 2000),  # Combination O-H
    (2000, 2200),  # Combination N-H
    (2200, 2400),  # Combination C-H
]

# Complexity presets for generator parameters
COMPLEXITY_PARAMS = {
    "simple": {
        "path_length_std": 0.02,
        "baseline_amplitude": 0.01,
        "scatter_alpha_std": 0.02,
        "scatter_beta_std": 0.005,
        "tilt_std": 0.005,
        "global_slope_mean": 0.0,
        "global_slope_std": 0.02,
        "shift_std": 0.2,
        "stretch_std": 0.0005,
        "instrumental_fwhm": 4,
        "noise_base": 0.002,
        "noise_signal_dep": 0.005,
        "artifact_prob": 0.0,
    },
    "realistic": {
        "path_length_std": 0.05,
        "baseline_amplitude": 0.02,
        "scatter_alpha_std": 0.05,
        "scatter_beta_std": 0.01,
        "tilt_std": 0.01,
        "global_slope_mean": 0.05,
        "global_slope_std": 0.03,
        "shift_std": 0.5,
        "stretch_std": 0.001,
        "instrumental_fwhm": 8,
        "noise_base": 0.005,
        "noise_signal_dep": 0.01,
        "artifact_prob": 0.02,
    },
    "complex": {
        "path_length_std": 0.08,
        "baseline_amplitude": 0.05,
        "scatter_alpha_std": 0.08,
        "scatter_beta_std": 0.02,
        "tilt_std": 0.02,
        "global_slope_mean": 0.08,
        "global_slope_std": 0.05,
        "shift_std": 1.0,
        "stretch_std": 0.002,
        "instrumental_fwhm": 12,
        "noise_base": 0.008,
        "noise_signal_dep": 0.015,
        "artifact_prob": 0.05,
    },
}

# Default predefined component names for realistic/complex modes
DEFAULT_REALISTIC_COMPONENTS = ["water", "protein", "lipid", "starch", "cellulose"]
