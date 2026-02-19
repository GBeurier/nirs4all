"""
Scattering effects configuration for synthetic NIRS data generation.

This module provides configuration classes for light scattering effects in NIR
spectra, including particle size effects and scattering coefficient generation.

Note:
    For applying scattering effects to spectra, use the operators in
    `nirs4all.operators.augmentation.scattering`:
    - ParticleSizeAugmenter: Particle size-dependent scattering
    - EMSCDistortionAugmenter: EMSC-style scatter distortions

Key Features:
    - EMSC-style (Extended Multiplicative Scatter Correction) transformations
    - Particle size-dependent scattering simulation
    - Scattering coefficient generation for Kubelka-Munk
    - Sample-to-sample scatter variation
    - Wavelength-dependent scattering (Rayleigh-like)

Physics Background:
    Light scattering in particulate samples is complex and depends on:
    - Particle size relative to wavelength (Mie vs Rayleigh regimes)
    - Particle shape and surface roughness
    - Refractive index differences
    - Packing density

    Rather than implementing full Mie theory (computationally expensive and
    may not match real data), this module uses empirical EMSC-style models
    that approximate the distortions that chemometric preprocessing corrects.

References:
    - Martens, H., Nielsen, J. P., & Engelsen, S. B. (2003). Light scattering
      and light absorbance separated by extended multiplicative signal
      correction. Application to near-infrared transmission analysis of
      powder mixtures. Analytical Chemistry, 75(3), 394-404.
    - Kubelka, P. (1948). New contributions to the optics of intensely
      light-scattering materials. Part I. JOSA, 38(5), 448-457.
    - Dahm, D. J., & Dahm, K. D. (2007). Interpreting Diffuse Reflectance
      and Transmittance. NIR Publications.
    - Burger, J., & Geladi, P. (2005). Hyperspectral NIR image regression
      part I: calibration and correction. Journal of Chemometrics, 19(5‐7),
      355-363.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Optional

import numpy as np


class ScatteringModel(StrEnum):
    """Available scattering models."""
    EMSC = "emsc"                    # Extended Multiplicative Scatter Correction style
    RAYLEIGH = "rayleigh"            # Rayleigh-like (λ⁻⁴ dependence)
    MIE_APPROX = "mie_approx"        # Simplified Mie approximation
    KUBELKA_MUNK = "kubelka_munk"    # K-M scattering coefficient model
    POLYNOMIAL = "polynomial"        # Polynomial baseline scattering

# ============================================================================
# Scattering Model Parameters
# ============================================================================

@dataclass
class ParticleSizeDistribution:
    """
    Particle size distribution parameters.

    Models particle size as a log-normal distribution, which is common
    for ground/milled samples in NIR analysis.

    Attributes:
        mean_size_um: Mean particle size in micrometers.
        std_size_um: Standard deviation of particle size in micrometers.
        min_size_um: Minimum particle size (lower truncation).
        max_size_um: Maximum particle size (upper truncation).
        distribution: Type of distribution ('lognormal', 'normal', 'uniform').
    """
    mean_size_um: float = 50.0
    std_size_um: float = 15.0
    min_size_um: float = 5.0
    max_size_um: float = 200.0
    distribution: str = "lognormal"

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample particle sizes from the distribution."""
        if self.distribution == "lognormal":
            # Convert to log-space parameters
            mu = np.log(self.mean_size_um)
            sigma = self.std_size_um / self.mean_size_um
            sizes = rng.lognormal(mu, sigma, n_samples)
        elif self.distribution == "normal":
            sizes = rng.normal(self.mean_size_um, self.std_size_um, n_samples)
        elif self.distribution == "uniform":
            sizes = rng.uniform(self.min_size_um, self.max_size_um, n_samples)
        else:
            sizes = rng.normal(self.mean_size_um, self.std_size_um, n_samples)

        # Clip to valid range
        return np.clip(sizes, self.min_size_um, self.max_size_um)

@dataclass
class ParticleSizeConfig:
    """
    Configuration for particle size effects.

    Attributes:
        distribution: Particle size distribution parameters.
        reference_size_um: Reference particle size for baseline scattering.
        size_effect_strength: How strongly size affects scattering (0-1).
        wavelength_exponent: Exponent for wavelength dependence of scattering.
            - 4.0 = Rayleigh (particles << wavelength)
            - 0.0 = No wavelength dependence
            - 1.0-2.0 = Typical for NIR powder samples
        include_path_length_effect: Whether particle size affects optical path.
        path_length_sensitivity: How strongly size affects path length.
    """
    distribution: ParticleSizeDistribution = field(
        default_factory=ParticleSizeDistribution
    )
    reference_size_um: float = 50.0
    size_effect_strength: float = 1.0
    wavelength_exponent: float = 1.5  # Empirical value for powder samples
    include_path_length_effect: bool = True
    path_length_sensitivity: float = 0.5

@dataclass
class EMSCConfig:
    """
    Configuration for EMSC-style scattering transformation.

    EMSC models scattering distortion as:
    x = a + b*x_ref + d*λ + e*λ² + ...

    where a, b are multiplicative/additive scatter, and higher terms
    model baseline curvature due to scattering.

    Attributes:
        polynomial_order: Order of polynomial for wavelength-dependent scatter.
        multiplicative_scatter_std: Std dev of multiplicative scatter factor b.
        additive_scatter_std: Std dev of additive scatter offset a.
        include_wavelength_terms: Whether to include λ, λ² terms.
        wavelength_coef_std: Std dev of wavelength coefficient.
        reference_spectrum: Optional reference spectrum for EMSC.
    """
    polynomial_order: int = 2
    multiplicative_scatter_std: float = 0.15
    additive_scatter_std: float = 0.05
    include_wavelength_terms: bool = True
    wavelength_coef_std: float = 0.02
    reference_spectrum: np.ndarray | None = None

@dataclass
class ScatteringCoefficientConfig:
    """
    Configuration for scattering coefficient (S) generation.

    For Kubelka-Munk reflectance, we need both absorption (K) and
    scattering (S) coefficients. This config controls S(λ) generation.

    Attributes:
        baseline_scattering: Base scattering coefficient value.
        wavelength_exponent: Exponent for wavelength dependence.
            S(λ) ∝ λ^(-exponent)
        particle_size_factor: How strongly particle size affects S.
        sample_variation: Sample-to-sample variation in S.
        wavelength_reference_nm: Reference wavelength for normalization.
    """
    baseline_scattering: float = 1.0
    wavelength_exponent: float = 1.0
    particle_size_factor: float = 0.5
    sample_variation: float = 0.15
    wavelength_reference_nm: float = 1500.0

@dataclass
class ScatteringEffectsConfig:
    """
    Combined configuration for all scattering effects.

    Attributes:
        model: Which scattering model to use.
        particle_size: Particle size effect configuration.
        emsc: EMSC-style transformation configuration.
        scattering_coefficient: Scattering coefficient generation config.
        enable_particle_size: Whether to apply particle size effects.
        enable_emsc: Whether to apply EMSC-style transformation.
    """
    model: ScatteringModel = ScatteringModel.EMSC
    particle_size: ParticleSizeConfig = field(default_factory=ParticleSizeConfig)
    emsc: EMSCConfig = field(default_factory=EMSCConfig)
    scattering_coefficient: ScatteringCoefficientConfig = field(
        default_factory=ScatteringCoefficientConfig
    )
    enable_particle_size: bool = True
    enable_emsc: bool = True

# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Enums
    "ScatteringModel",
    # Dataclasses
    "ParticleSizeDistribution",
    "ParticleSizeConfig",
    "EMSCConfig",
    "ScatteringCoefficientConfig",
    "ScatteringEffectsConfig",
]
