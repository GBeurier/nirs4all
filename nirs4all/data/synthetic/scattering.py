"""
Scattering effects simulation for synthetic NIRS data generation.

This module provides simulation of light scattering effects in NIR spectra,
including particle size effects and scattering coefficient generation.

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
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d


class ScatteringModel(str, Enum):
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
    reference_spectrum: Optional[np.ndarray] = None


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
# Particle Size Effect Simulator
# ============================================================================

class ParticleSizeSimulator:
    """
    Simulate particle size effects on NIR spectra.

    Particle size affects NIR spectra through:
    - Scattering baseline (smaller particles = more scattering)
    - Path length through sample (affects Beer-Lambert)
    - Wavelength dependence of scattering

    Uses EMSC-style approach: applies distortions that chemometric
    preprocessing (SNV, MSC) would correct.

    Attributes:
        config: Particle size configuration.
        rng: Random number generator.

    Example:
        >>> config = ParticleSizeConfig(
        ...     distribution=ParticleSizeDistribution(mean_size_um=30.0)
        ... )
        >>> simulator = ParticleSizeSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[ParticleSizeConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the particle size simulator.

        Args:
            config: Particle size effect configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else ParticleSizeConfig()
        self.rng = np.random.default_rng(random_state)

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        particle_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply particle size effects to spectra.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            particle_sizes: Optional per-sample particle sizes (μm).
                If None, samples from configured distribution.

        Returns:
            Modified spectra with particle size effects applied.
        """
        n_samples = spectra.shape[0]
        result = spectra.copy()

        # Sample particle sizes if not provided
        if particle_sizes is None:
            particle_sizes = self.config.distribution.sample(n_samples, self.rng)

        # Compute size-relative scattering factors
        size_ratios = particle_sizes / self.config.reference_size_um

        for i in range(n_samples):
            result[i] = self._apply_size_effects_to_spectrum(
                result[i], wavelengths, size_ratios[i], particle_sizes[i]
            )

        return result

    def _apply_size_effects_to_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        size_ratio: float,
        particle_size: float
    ) -> np.ndarray:
        """Apply particle size effects to a single spectrum."""
        result = spectrum.copy()

        # 1. Wavelength-dependent scattering baseline
        # Smaller particles = more scattering, especially at shorter wavelengths
        scatter_baseline = self._compute_scatter_baseline(
            wavelengths, size_ratio
        )
        result = result + scatter_baseline

        # 2. Multiplicative scatter (path length effect)
        if self.config.include_path_length_effect:
            # Smaller particles = shorter mean free path = less absorption
            path_factor = self._compute_path_length_factor(size_ratio)
            result = result * path_factor

        # 3. Additional scatter variance
        scatter_noise = self._compute_scatter_noise(
            wavelengths, size_ratio
        )
        result = result + scatter_noise

        return result

    def _compute_scatter_baseline(
        self,
        wavelengths: np.ndarray,
        size_ratio: float
    ) -> np.ndarray:
        """
        Compute scattering-induced baseline.

        Smaller particles (size_ratio < 1) increase scattering baseline.
        Baseline follows λ^(-exponent) dependence.
        """
        # Normalize wavelengths
        wl_norm = wavelengths / 1500.0  # Reference at 1500 nm

        # Wavelength-dependent scattering
        exponent = self.config.wavelength_exponent
        wl_factor = wl_norm ** (-exponent)

        # Size effect: smaller particles scatter more
        # Use inverse relationship (size_ratio^(-0.5) empirical)
        size_factor = size_ratio ** (-0.5)

        # Scale by strength parameter
        strength = self.config.size_effect_strength

        # Baseline offset (normalized to reasonable values)
        baseline = 0.1 * strength * (size_factor - 1.0) * wl_factor

        # Center so mean offset is controlled
        baseline = baseline - baseline.mean()

        return baseline

    def _compute_path_length_factor(
        self,
        size_ratio: float
    ) -> float:
        """
        Compute effective path length factor.

        Smaller particles reduce mean free path, affecting total absorption.
        """
        sensitivity = self.config.path_length_sensitivity

        # Empirical relationship: path_factor ≈ 1 + sens * log(size_ratio)
        path_factor = 1.0 + sensitivity * np.log(size_ratio)

        # Clip to reasonable range
        return np.clip(path_factor, 0.7, 1.5)

    def _compute_scatter_noise(
        self,
        wavelengths: np.ndarray,
        size_ratio: float
    ) -> np.ndarray:
        """
        Compute scatter-related noise/variation.

        Adds sample-specific scatter variation.
        """
        n_wl = len(wavelengths)

        # Random scatter variation
        noise_std = 0.005 * self.config.size_effect_strength
        noise = self.rng.normal(0, noise_std, n_wl)

        # Slight wavelength correlation
        noise = gaussian_filter1d(noise, sigma=3)

        return noise

    def generate_particle_sizes(self, n_samples: int) -> np.ndarray:
        """
        Generate particle sizes for a set of samples.

        Args:
            n_samples: Number of samples.

        Returns:
            Array of particle sizes in μm.
        """
        return self.config.distribution.sample(n_samples, self.rng)


# ============================================================================
# EMSC Transformation Simulator
# ============================================================================

class EMSCTransformSimulator:
    """
    Simulate EMSC-style scattering distortions.

    Applies the inverse of Extended Multiplicative Scatter Correction,
    generating realistic scatter distortions that EMSC would correct.

    EMSC models spectra as: x = a + b*m + d*λ + e*λ² + ...
    where m is a reference spectrum.

    This simulator generates a, b, d, e, ... to create scatter distortions.

    Attributes:
        config: EMSC configuration.
        rng: Random number generator.

    Example:
        >>> config = EMSCConfig(polynomial_order=2)
        >>> simulator = EMSCTransformSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[EMSCConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the EMSC transformation simulator.

        Args:
            config: EMSC configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else EMSCConfig()
        self.rng = np.random.default_rng(random_state)

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        reference_spectrum: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply EMSC-style scattering distortions.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            reference_spectrum: Optional reference spectrum. If None,
                uses mean of input spectra or config reference.

        Returns:
            Modified spectra with scatter distortions applied.
        """
        n_samples, n_wl = spectra.shape
        result = spectra.copy()

        # Get reference spectrum
        if reference_spectrum is None:
            if self.config.reference_spectrum is not None:
                reference = self.config.reference_spectrum
            else:
                reference = np.mean(spectra, axis=0)
        else:
            reference = reference_spectrum

        # Normalize wavelengths for polynomial basis
        wl_norm = self._normalize_wavelengths(wavelengths)

        # Generate EMSC parameters for each sample
        for i in range(n_samples):
            params = self._generate_emsc_params()
            result[i] = self._apply_emsc_transform(
                result[i], reference, wl_norm, params
            )

        return result

    def _normalize_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Normalize wavelengths to [-1, 1] for polynomial stability."""
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        return 2.0 * (wavelengths - wl_min) / (wl_max - wl_min) - 1.0

    def _generate_emsc_params(self) -> Dict[str, float]:
        """Generate EMSC parameters for one sample."""
        params = {}

        # Multiplicative scatter (b term)
        params['b'] = self.rng.normal(1.0, self.config.multiplicative_scatter_std)

        # Additive offset (a term)
        params['a'] = self.rng.normal(0.0, self.config.additive_scatter_std)

        # Wavelength polynomial terms
        if self.config.include_wavelength_terms:
            for order in range(1, self.config.polynomial_order + 1):
                coef_name = f'c{order}'
                params[coef_name] = self.rng.normal(
                    0.0,
                    self.config.wavelength_coef_std / (order ** 0.5)
                )

        return params

    def _apply_emsc_transform(
        self,
        spectrum: np.ndarray,
        reference: np.ndarray,
        wl_norm: np.ndarray,
        params: Dict[str, float]
    ) -> np.ndarray:
        """Apply EMSC transformation to a single spectrum."""
        result = params['a'] + params['b'] * spectrum

        # Add wavelength-dependent terms
        if self.config.include_wavelength_terms:
            for order in range(1, self.config.polynomial_order + 1):
                coef_name = f'c{order}'
                if coef_name in params:
                    result = result + params[coef_name] * (wl_norm ** order)

        return result

    def get_emsc_basis(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Get EMSC polynomial basis functions.

        Args:
            wavelengths: Wavelength array.

        Returns:
            Basis matrix (n_wavelengths, n_terms).
        """
        wl_norm = self._normalize_wavelengths(wavelengths)
        n_wl = len(wavelengths)
        n_terms = self.config.polynomial_order + 1  # Including constant

        basis = np.zeros((n_wl, n_terms))
        basis[:, 0] = 1.0  # Constant term

        for order in range(1, n_terms):
            basis[:, order] = wl_norm ** order

        return basis


# ============================================================================
# Scattering Coefficient Generator
# ============================================================================

class ScatteringCoefficientGenerator:
    """
    Generate scattering coefficients S(λ) for Kubelka-Munk simulation.

    The Kubelka-Munk equation relates reflectance R to absorption K and
    scattering S: f(R) = (1-R)²/(2R) = K/S

    This generator produces realistic S(λ) values for different sample types.

    Attributes:
        config: Scattering coefficient configuration.
        rng: Random number generator.

    Example:
        >>> config = ScatteringCoefficientConfig(
        ...     baseline_scattering=1.5,
        ...     wavelength_exponent=1.2
        ... )
        >>> generator = ScatteringCoefficientGenerator(config, random_state=42)
        >>> S = generator.generate(n_samples=100, wavelengths=wavelengths)
    """

    def __init__(
        self,
        config: Optional[ScatteringCoefficientConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the scattering coefficient generator.

        Args:
            config: Scattering coefficient configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else ScatteringCoefficientConfig()
        self.rng = np.random.default_rng(random_state)

    def generate(
        self,
        n_samples: int,
        wavelengths: np.ndarray,
        particle_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate scattering coefficients for samples.

        Args:
            n_samples: Number of samples.
            wavelengths: Wavelength array in nm.
            particle_sizes: Optional per-sample particle sizes (μm).

        Returns:
            Scattering coefficient array (n_samples, n_wavelengths).
        """
        n_wl = len(wavelengths)

        # Wavelength-dependent base scattering
        wl_ref = self.config.wavelength_reference_nm
        wl_factor = (wl_ref / wavelengths) ** self.config.wavelength_exponent

        # Sample-to-sample variation
        sample_factors = self.rng.normal(
            1.0,
            self.config.sample_variation,
            n_samples
        )
        sample_factors = np.maximum(sample_factors, 0.3)

        # Particle size effect
        if particle_sizes is not None:
            # Smaller particles = more scattering
            ref_size = 50.0  # Reference particle size
            size_factors = (ref_size / particle_sizes) ** self.config.particle_size_factor
        else:
            size_factors = np.ones(n_samples)

        # Generate scattering coefficients
        S = np.zeros((n_samples, n_wl))
        for i in range(n_samples):
            S[i] = (
                self.config.baseline_scattering *
                sample_factors[i] *
                size_factors[i] *
                wl_factor
            )

            # Add small random fluctuation
            fluctuation = self.rng.normal(0, 0.05 * S[i])
            S[i] = S[i] + fluctuation
            S[i] = np.maximum(S[i], 0.1)  # Ensure positive

        return S

    def generate_for_particle_sizes(
        self,
        particle_sizes: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Generate scattering coefficients based on particle sizes.

        Args:
            particle_sizes: Array of particle sizes in μm.
            wavelengths: Wavelength array in nm.

        Returns:
            Scattering coefficient array.
        """
        return self.generate(
            n_samples=len(particle_sizes),
            wavelengths=wavelengths,
            particle_sizes=particle_sizes
        )


# ============================================================================
# Combined Scattering Effects Simulator
# ============================================================================

class ScatteringEffectsSimulator:
    """
    Combined simulator for all scattering effects.

    Applies particle size effects and EMSC-style transformations
    in the correct order.

    Attributes:
        config: Scattering effects configuration.
        particle_sim: Particle size simulator.
        emsc_sim: EMSC transformation simulator.
        scatter_gen: Scattering coefficient generator.
        rng: Random number generator.

    Example:
        >>> config = ScatteringEffectsConfig(
        ...     model=ScatteringModel.EMSC,
        ...     particle_size=ParticleSizeConfig(
        ...         distribution=ParticleSizeDistribution(mean_size_um=30.0)
        ...     )
        ... )
        >>> simulator = ScatteringEffectsSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: Optional[ScatteringEffectsConfig] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the combined scattering effects simulator.

        Args:
            config: Scattering effects configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else ScatteringEffectsConfig()
        self.rng = np.random.default_rng(random_state)

        # Initialize component simulators
        self.particle_sim = ParticleSizeSimulator(
            self.config.particle_size, random_state
        )
        self.emsc_sim = EMSCTransformSimulator(
            self.config.emsc, random_state
        )
        self.scatter_gen = ScatteringCoefficientGenerator(
            self.config.scattering_coefficient, random_state
        )

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        particle_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply all scattering effects to spectra.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            particle_sizes: Optional per-sample particle sizes.

        Returns:
            Modified spectra with scattering effects applied.
        """
        result = spectra.copy()

        # Apply particle size effects
        if self.config.enable_particle_size:
            result = self.particle_sim.apply(
                result, wavelengths, particle_sizes
            )

        # Apply EMSC-style transformation
        if self.config.enable_emsc:
            result = self.emsc_sim.apply(result, wavelengths)

        return result

    def generate_scattering_coefficients(
        self,
        n_samples: int,
        wavelengths: np.ndarray,
        particle_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate scattering coefficients for Kubelka-Munk.

        Args:
            n_samples: Number of samples.
            wavelengths: Wavelength array.
            particle_sizes: Optional particle sizes.

        Returns:
            Scattering coefficient array (n_samples, n_wavelengths).
        """
        return self.scatter_gen.generate(n_samples, wavelengths, particle_sizes)


# ============================================================================
# Convenience Functions
# ============================================================================

def apply_particle_size_effects(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    mean_particle_size_um: float = 50.0,
    size_variation: float = 15.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply particle size effects to spectra with simple API.

    Args:
        spectra: Input spectra (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        mean_particle_size_um: Mean particle size in micrometers.
        size_variation: Standard deviation of particle size.
        random_state: Random seed.

    Returns:
        Spectra with particle size effects applied.

    Example:
        >>> # Simulate fine powder sample
        >>> spectra_fine = apply_particle_size_effects(
        ...     spectra, wavelengths,
        ...     mean_particle_size_um=20.0
        ... )
    """
    config = ParticleSizeConfig(
        distribution=ParticleSizeDistribution(
            mean_size_um=mean_particle_size_um,
            std_size_um=size_variation
        )
    )
    simulator = ParticleSizeSimulator(config, random_state)
    return simulator.apply(spectra, wavelengths)


def apply_emsc_distortion(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    multiplicative_std: float = 0.15,
    additive_std: float = 0.05,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply EMSC-style scatter distortions with simple API.

    Args:
        spectra: Input spectra (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        multiplicative_std: Std dev of multiplicative scatter.
        additive_std: Std dev of additive scatter.
        random_state: Random seed.

    Returns:
        Spectra with EMSC-style distortions applied.

    Example:
        >>> # Add realistic scatter distortions
        >>> spectra_scattered = apply_emsc_distortion(spectra, wavelengths)
    """
    config = EMSCConfig(
        multiplicative_scatter_std=multiplicative_std,
        additive_scatter_std=additive_std
    )
    simulator = EMSCTransformSimulator(config, random_state)
    return simulator.apply(spectra, wavelengths)


def generate_scattering_coefficients(
    n_samples: int,
    wavelengths: np.ndarray,
    baseline_scattering: float = 1.0,
    wavelength_exponent: float = 1.0,
    particle_sizes: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate scattering coefficients with simple API.

    Args:
        n_samples: Number of samples.
        wavelengths: Wavelength array (nm).
        baseline_scattering: Base scattering coefficient.
        wavelength_exponent: Wavelength dependence exponent.
        particle_sizes: Optional particle sizes (μm).
        random_state: Random seed.

    Returns:
        Scattering coefficient array (n_samples, n_wavelengths).

    Example:
        >>> S = generate_scattering_coefficients(100, wavelengths)
    """
    config = ScatteringCoefficientConfig(
        baseline_scattering=baseline_scattering,
        wavelength_exponent=wavelength_exponent
    )
    generator = ScatteringCoefficientGenerator(config, random_state)
    return generator.generate(n_samples, wavelengths, particle_sizes)


def simulate_snv_correctable_scatter(
    spectra: np.ndarray,
    intensity: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply scatter effects that SNV (Standard Normal Variate) would correct.

    SNV corrects multiplicative and additive scatter. This function applies
    such effects so that SNV preprocessing would restore the original spectra.

    Args:
        spectra: Input spectra.
        intensity: Intensity of scatter effects (0-2, default 1).
        random_state: Random seed.

    Returns:
        Spectra with SNV-correctable scatter.

    Example:
        >>> # Add scatter that SNV will correct
        >>> scattered = simulate_snv_correctable_scatter(spectra, intensity=1.5)
    """
    rng = np.random.default_rng(random_state)
    n_samples = spectra.shape[0]

    # Multiplicative factor
    mult = rng.normal(1.0, 0.1 * intensity, n_samples)
    mult = np.maximum(mult, 0.5)

    # Additive offset
    add = rng.normal(0.0, 0.05 * intensity, n_samples)

    result = spectra * mult[:, np.newaxis] + add[:, np.newaxis]

    return result


def simulate_msc_correctable_scatter(
    spectra: np.ndarray,
    reference: Optional[np.ndarray] = None,
    intensity: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply scatter effects that MSC (Multiplicative Scatter Correction) would correct.

    MSC regresses each spectrum against a reference to remove multiplicative
    and baseline scatter. This function applies such effects.

    Args:
        spectra: Input spectra.
        reference: Reference spectrum (mean if None).
        intensity: Intensity of scatter effects.
        random_state: Random seed.

    Returns:
        Spectra with MSC-correctable scatter.

    Example:
        >>> # Add scatter that MSC will correct
        >>> scattered = simulate_msc_correctable_scatter(spectra)
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_wl = spectra.shape

    if reference is None:
        reference = np.mean(spectra, axis=0)

    # MSC model: x_i = a_i + b_i * x_ref + e_i
    # We generate a_i, b_i to distort spectra

    result = np.zeros_like(spectra)
    for i in range(n_samples):
        b = rng.normal(1.0, 0.15 * intensity)
        a = rng.normal(0.0, 0.05 * intensity)
        result[i] = a + b * spectra[i]

    return result


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
    # Simulators
    "ParticleSizeSimulator",
    "EMSCTransformSimulator",
    "ScatteringCoefficientGenerator",
    "ScatteringEffectsSimulator",
    # Convenience functions
    "apply_particle_size_effects",
    "apply_emsc_distortion",
    "generate_scattering_coefficients",
    "simulate_snv_correctable_scatter",
    "simulate_msc_correctable_scatter",
]
