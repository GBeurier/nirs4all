"""
Measurement mode simulation for synthetic NIRS data generation.

This module provides simulation of different NIR measurement geometries
and their associated physics. The relationship between absorption coefficients
and observed signal varies significantly with measurement mode.

Supported Measurement Modes:
    - Transmittance: Beer-Lambert law, direct transmission
    - Diffuse Reflectance: Kubelka-Munk theory for scattering samples
    - Transflectance: Double-pass transmission with mirror backing
    - ATR: Attenuated Total Reflectance with wavelength-dependent penetration

References:
    - Kubelka, P. (1948). New contributions to the optics of intensely
      light-scattering materials. Part I. JOSA, 38(5), 448-457.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared
      Analysis. CRC Press.
    - Harrick, N. J. (1967). Internal Reflection Spectroscopy. Wiley.
    - Dahm, D. J., & Dahm, K. D. (2007). Interpreting Diffuse Reflectance
      and Transmittance. NIR Publications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Optional, Union

import numpy as np


class MeasurementMode(StrEnum):
    """Types of NIR measurement geometries."""
    TRANSMITTANCE = "transmittance"       # Direct transmission (Beer-Lambert)
    REFLECTANCE = "reflectance"           # Diffuse reflectance (Kubelka-Munk)
    TRANSFLECTANCE = "transflectance"     # Double-pass with reflector
    ATR = "atr"                           # Attenuated Total Reflectance
    INTERACTANCE = "interactance"         # Partial transmission/reflection
    FIBER_OPTIC = "fiber_optic"           # Fiber-coupled reflectance probe

@dataclass
class TransmittanceConfig:
    """
    Configuration for transmittance measurement mode.

    Implements Beer-Lambert law: A = εcl
    where A is absorbance, ε is molar absorptivity, c is concentration,
    and l is path length.

    Attributes:
        path_length_mm: Optical path length in mm.
        path_length_variation: Sample-to-sample variation in path length.
        cuvette_material: Material of sample holder (affects NIR absorption).
        reference_type: Type of reference measurement.
    """
    path_length_mm: float = 1.0
    path_length_variation: float = 0.02  # Coefficient of variation
    cuvette_material: str = "quartz"     # quartz, sapphire, glass
    reference_type: str = "air"          # air, solvent, empty_cuvette

@dataclass
class ReflectanceConfig:
    """
    Configuration for diffuse reflectance measurement mode.

    Implements Kubelka-Munk theory: f(R) = (1-R)² / 2R = K/S
    where R is reflectance, K is absorption coefficient, S is scattering.

    Attributes:
        geometry: Measurement geometry (integrating sphere, fiber probe, etc.).
        reference_material: Reference standard material.
        reference_reflectance: Reflectance of reference standard.
        illumination_angle: Angle of illumination (degrees from normal).
        collection_angle: Angle of collection (degrees from normal).
        sample_presentation: How sample is presented (powder, solid, slurry).
    """
    geometry: str = "integrating_sphere"  # integrating_sphere, 0_45, fiber_probe
    reference_material: str = "spectralon"  # spectralon, ptfe, baso4
    reference_reflectance: float = 0.99
    illumination_angle: float = 0.0
    collection_angle: float = 45.0
    sample_presentation: str = "powder"  # powder, solid, slurry, liquid

@dataclass
class TransflectanceConfig:
    """
    Configuration for transflectance measurement mode.

    Light passes through sample, reflects off a mirror/diffuser,
    and passes through sample again (double-pass).

    Attributes:
        path_length_mm: Single-pass path length in mm.
        reflector_type: Type of backing reflector.
        reflector_reflectance: Reflectance of backing material.
        spacer_thickness_mm: Spacer thickness controlling path length.
    """
    path_length_mm: float = 0.5
    reflector_type: str = "gold"  # gold, aluminum, diffuser
    reflector_reflectance: float = 0.95
    spacer_thickness_mm: float = 0.5

@dataclass
class ATRConfig:
    """
    Configuration for Attenuated Total Reflectance mode.

    ATR uses internal reflection within a high-refractive-index crystal.
    The evanescent wave penetrates into the sample, with penetration depth
    depending on wavelength.

    Attributes:
        crystal_material: ATR crystal material.
        crystal_refractive_index: Refractive index of crystal.
        incidence_angle: Angle of incidence (degrees).
        n_reflections: Number of internal reflections.
        sample_refractive_index: Approximate refractive index of sample.
    """
    crystal_material: str = "diamond"  # diamond, znse, ge, si
    crystal_refractive_index: float = 2.4  # Diamond
    incidence_angle: float = 45.0
    n_reflections: int = 1
    sample_refractive_index: float = 1.5  # Typical organic

@dataclass
class ScatteringConfig:
    """
    Configuration for scattering coefficient generation.

    Controls how scattering coefficients are generated for samples,
    which is essential for Kubelka-Munk reflectance simulation.

    Attributes:
        baseline_scattering: Base scattering coefficient (arbitrary units).
        wavelength_exponent: Exponent for wavelength dependence (Rayleigh-like).
            S(λ) ∝ λ^(-exponent), typically 0.5-2.0
        particle_size_um: Mean particle size in micrometers.
        particle_size_variation: Coefficient of variation in particle size.
        sample_to_sample_variation: How much scattering varies between samples.
    """
    baseline_scattering: float = 1.0
    wavelength_exponent: float = 1.0  # 0 = no wavelength dependence
    particle_size_um: float = 50.0
    particle_size_variation: float = 0.2
    sample_to_sample_variation: float = 0.15

@dataclass
class MeasurementModeConfig:
    """
    Complete configuration for measurement mode simulation.

    Combines all mode-specific configurations into a single object.

    Attributes:
        mode: The measurement mode to simulate.
        transmittance: Config for transmittance mode.
        reflectance: Config for reflectance mode.
        transflectance: Config for transflectance mode.
        atr: Config for ATR mode.
        scattering: Scattering coefficient configuration.
        add_specular: Whether to add specular reflection component.
        specular_fraction: Fraction of specular vs diffuse reflection.
    """
    mode: MeasurementMode = MeasurementMode.TRANSMITTANCE
    transmittance: TransmittanceConfig = field(default_factory=TransmittanceConfig)
    reflectance: ReflectanceConfig = field(default_factory=ReflectanceConfig)
    transflectance: TransflectanceConfig = field(default_factory=TransflectanceConfig)
    atr: ATRConfig = field(default_factory=ATRConfig)
    scattering: ScatteringConfig = field(default_factory=ScatteringConfig)
    add_specular: bool = False
    specular_fraction: float = 0.04  # Fresnel reflection at normal incidence

# ============================================================================
# Crystal refractive indices for ATR
# ============================================================================

ATR_CRYSTAL_PROPERTIES = {
    "diamond": {"refractive_index": 2.4, "critical_angle": 24.6, "range": (2500, 25000)},
    "znse": {"refractive_index": 2.4, "critical_angle": 24.6, "range": (650, 20000)},
    "ge": {"refractive_index": 4.0, "critical_angle": 14.5, "range": (2000, 12000)},
    "si": {"refractive_index": 3.4, "critical_angle": 17.1, "range": (1500, 8000)},
    "thallium_bromide": {"refractive_index": 2.37, "critical_angle": 25.0, "range": (550, 35000)},
}

# ============================================================================
# Measurement Mode Simulator
# ============================================================================

class MeasurementModeSimulator:
    """
    Simulate different NIR measurement modes.

    Converts absorption coefficients to measured signal (absorbance, reflectance, etc.)
    based on the physics of different measurement geometries.

    Attributes:
        config: Measurement mode configuration.
        rng: Random number generator for reproducibility.

    Example:
        >>> config = MeasurementModeConfig(mode=MeasurementMode.REFLECTANCE)
        >>> simulator = MeasurementModeSimulator(config, random_state=42)
        >>> reflectance = simulator.apply(absorption_coefficients, wavelengths)
    """

    def __init__(
        self,
        config: MeasurementModeConfig | None = None,
        random_state: int | None = None
    ) -> None:
        """
        Initialize the measurement mode simulator.

        Args:
            config: Measurement mode configuration. If None, uses default
                transmittance configuration.
            random_state: Random seed for reproducibility.
        """
        self.config = config if config is not None else MeasurementModeConfig()
        self.rng = np.random.default_rng(random_state)

    def apply(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        scattering: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply measurement mode transformation.

        Converts absorption coefficients (K) to measured signal based on
        the configured measurement mode.

        Args:
            absorption: Absorption coefficient array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            scattering: Optional scattering coefficient array (n_samples, n_wavelengths).
                If None and needed, will be generated automatically.

        Returns:
            Measured signal (absorbance, reflectance, etc.) depending on mode.
        """
        mode = self.config.mode

        if mode == MeasurementMode.TRANSMITTANCE:
            return self._apply_transmittance(absorption, wavelengths)
        elif mode == MeasurementMode.REFLECTANCE:
            if scattering is None:
                scattering = self.generate_scattering_coefficients(
                    absorption.shape, wavelengths
                )
            return self._apply_reflectance(absorption, wavelengths, scattering)
        elif mode == MeasurementMode.TRANSFLECTANCE:
            return self._apply_transflectance(absorption, wavelengths)
        elif mode == MeasurementMode.ATR:
            return self._apply_atr(absorption, wavelengths)
        elif mode == MeasurementMode.INTERACTANCE:
            if scattering is None:
                scattering = self.generate_scattering_coefficients(
                    absorption.shape, wavelengths
                )
            return self._apply_interactance(absorption, wavelengths, scattering)
        else:
            # Default to transmittance
            return self._apply_transmittance(absorption, wavelengths)

    def _apply_transmittance(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply Beer-Lambert transmittance model.

        A = εcl = K * l

        where K is the absorption coefficient and l is path length.
        """
        config = self.config.transmittance
        n_samples = absorption.shape[0]

        # Generate path lengths with variation
        base_path = config.path_length_mm
        path_variation = config.path_length_variation
        path_lengths = self.rng.normal(
            base_path,
            base_path * path_variation,
            n_samples
        )
        path_lengths = np.maximum(path_lengths, base_path * 0.5)

        # Apply Beer-Lambert: A = K * l
        absorbance = absorption * path_lengths[:, np.newaxis]

        return absorbance

    def _apply_reflectance(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        scattering: np.ndarray
    ) -> np.ndarray:
        """
        Apply Kubelka-Munk diffuse reflectance model.

        The Kubelka-Munk function: f(R∞) = (1 - R∞)² / (2 * R∞) = K / S

        Solving for R∞: R∞ = 1 + K/S - sqrt((K/S)² + 2*K/S)
        """
        config = self.config.reflectance

        # Compute K/S ratio
        # Avoid division by zero
        K_over_S = absorption / (scattering + 1e-10)

        # Calculate reflectance from Kubelka-Munk
        # R∞ = 1 + K/S - sqrt((K/S)² + 2*K/S)
        reflectance = 1 + K_over_S - np.sqrt(K_over_S**2 + 2 * K_over_S)

        # Ensure physically meaningful values
        reflectance = np.clip(reflectance, 0.001, 0.999)

        # Apply reference correction
        reflectance = reflectance / config.reference_reflectance

        # Add specular component if configured
        if self.config.add_specular:
            reflectance += self.config.specular_fraction
            reflectance = np.clip(reflectance, 0, 1)

        # Convert to apparent absorbance (log 1/R)
        apparent_absorbance = -np.log10(reflectance)

        return apparent_absorbance

    def _apply_transflectance(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply transflectance (double-pass) model.

        Light passes through sample twice, so effective path length is doubled.
        Some light is lost at the reflector.
        """
        config = self.config.transflectance
        n_samples = absorption.shape[0]

        # Effective path length is approximately 2x single pass
        effective_path = 2 * config.path_length_mm

        # Path length variation
        path_lengths = self.rng.normal(
            effective_path,
            effective_path * 0.03,  # Less variation than single-pass
            n_samples
        )

        # Apply Beer-Lambert with double path
        absorbance = absorption * path_lengths[:, np.newaxis]

        # Add reflector losses (appears as baseline offset)
        reflector_loss = -np.log10(config.reflector_reflectance)
        absorbance += reflector_loss

        return absorbance

    def _apply_atr(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply ATR (Attenuated Total Reflectance) model.

        The penetration depth (dp) is wavelength-dependent:
        dp = λ / (2π * n1 * sqrt(sin²θ - (n2/n1)²))

        where n1 is crystal refractive index, n2 is sample refractive index,
        and θ is incidence angle.
        """
        config = self.config.atr
        n_samples = absorption.shape[0]
        n_wl = len(wavelengths)

        n1 = config.crystal_refractive_index
        n2 = config.sample_refractive_index
        theta = np.radians(config.incidence_angle)

        # Calculate penetration depth as function of wavelength
        # dp = λ / (2π * n1 * sqrt(sin²θ - (n2/n1)²))
        sin_theta_sq = np.sin(theta) ** 2
        n_ratio_sq = (n2 / n1) ** 2

        # Check for total internal reflection condition
        if sin_theta_sq <= n_ratio_sq:
            raise ValueError(
                f"No total internal reflection: sin²θ ({sin_theta_sq:.3f}) <= (n2/n1)² ({n_ratio_sq:.3f})"
            )

        denominator = 2 * np.pi * n1 * np.sqrt(sin_theta_sq - n_ratio_sq)

        # Penetration depth in nm (wavelengths in nm)
        dp = wavelengths / denominator

        # Effective path length = dp * n_reflections
        effective_path = dp * config.n_reflections

        # Convert to mm (wavelengths in nm, so dp in nm)
        effective_path_mm = effective_path / 1e6  # nm to mm

        # ATR absorbance proportional to absorption * effective path
        absorbance = absorption * effective_path_mm

        # Add sample-to-sample contact variation
        contact_variation = self.rng.normal(1.0, 0.05, n_samples)
        absorbance = absorbance * contact_variation[:, np.newaxis]

        return absorbance

    def _apply_interactance(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        scattering: np.ndarray
    ) -> np.ndarray:
        """
        Apply interactance mode model.

        Interactance is a hybrid between transmittance and reflectance,
        where light enters at one point and exits at another on the
        same sample surface. Path length depends on scattering.
        """
        n_samples = absorption.shape[0]

        # Effective path length depends on scattering
        # Higher scattering = shorter mean free path = less penetration
        mean_scattering = np.mean(scattering, axis=1)
        effective_path = 5.0 / (mean_scattering + 0.5)  # Empirical relationship
        effective_path = np.clip(effective_path, 1.0, 10.0)

        # Absorbance with scattering-dependent path
        absorbance = absorption * effective_path[:, np.newaxis]

        # Add some scattering-induced baseline
        baseline = 0.1 * mean_scattering[:, np.newaxis] * np.ones((1, len(wavelengths)))
        absorbance = absorbance + baseline

        return absorbance

    def generate_scattering_coefficients(
        self,
        shape: tuple[int, int],
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Generate realistic scattering coefficients.

        Scattering coefficient follows approximate relationship:
        S(λ) ∝ λ^(-α) * (particle_size)^β

        Args:
            shape: Output shape (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.

        Returns:
            Scattering coefficient array.
        """
        config = self.config.scattering
        n_samples, n_wl = shape

        # Wavelength dependence (normalized to 1500 nm)
        wl_factor = (1500 / wavelengths) ** config.wavelength_exponent

        # Sample-to-sample variation in baseline scattering
        sample_scatter = self.rng.normal(
            config.baseline_scattering,
            config.baseline_scattering * config.sample_to_sample_variation,
            n_samples
        )
        sample_scatter = np.maximum(sample_scatter, 0.1)

        # Particle size effect on scattering (rough empirical relationship)
        particle_sizes = self.rng.normal(
            config.particle_size_um,
            config.particle_size_um * config.particle_size_variation,
            n_samples
        )
        particle_sizes = np.maximum(particle_sizes, 5.0)

        # Smaller particles scatter more (Rayleigh-like)
        size_factor = (100 / particle_sizes) ** 0.5

        # Combine factors
        scattering = np.zeros(shape)
        for i in range(n_samples):
            scattering[i] = (
                sample_scatter[i] *
                size_factor[i] *
                wl_factor
            )

        return scattering

    def absorbance_to_reflectance(self, absorbance: np.ndarray) -> np.ndarray:
        """
        Convert apparent absorbance to reflectance.

        R = 10^(-A)

        Args:
            absorbance: Apparent absorbance values.

        Returns:
            Reflectance values (0-1).
        """
        reflectance = 10 ** (-absorbance)
        return np.clip(reflectance, 0, 1)

    def reflectance_to_absorbance(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Convert reflectance to apparent absorbance.

        A = log10(1/R) = -log10(R)

        Args:
            reflectance: Reflectance values (0-1).

        Returns:
            Apparent absorbance values.
        """
        # Avoid log of zero
        reflectance = np.clip(reflectance, 1e-10, 1.0)
        return -np.log10(reflectance)

    def kubelka_munk(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Apply Kubelka-Munk transformation.

        f(R) = (1 - R)² / (2R) = K/S

        Args:
            reflectance: Reflectance values (0-1).

        Returns:
            Kubelka-Munk function values (K/S ratio).
        """
        # Avoid division by zero
        reflectance = np.clip(reflectance, 1e-10, 0.999)
        return (1 - reflectance) ** 2 / (2 * reflectance)

    def inverse_kubelka_munk(
        self,
        ks_ratio: np.ndarray
    ) -> np.ndarray:
        """
        Inverse Kubelka-Munk transformation.

        Given K/S, solve for R∞:
        R∞ = 1 + K/S - sqrt((K/S)² + 2*K/S)

        Args:
            ks_ratio: K/S ratio values.

        Returns:
            Reflectance values.
        """
        reflectance = 1 + ks_ratio - np.sqrt(ks_ratio**2 + 2 * ks_ratio)
        return np.clip(reflectance, 0, 1)

# ============================================================================
# Convenience functions
# ============================================================================

def create_transmittance_simulator(
    path_length_mm: float = 1.0,
    random_state: int | None = None
) -> MeasurementModeSimulator:
    """
    Create a transmittance mode simulator.

    Args:
        path_length_mm: Optical path length in mm.
        random_state: Random seed.

    Returns:
        Configured MeasurementModeSimulator.
    """
    config = MeasurementModeConfig(
        mode=MeasurementMode.TRANSMITTANCE,
        transmittance=TransmittanceConfig(path_length_mm=path_length_mm)
    )
    return MeasurementModeSimulator(config, random_state)

def create_reflectance_simulator(
    geometry: str = "integrating_sphere",
    particle_size_um: float = 50.0,
    random_state: int | None = None
) -> MeasurementModeSimulator:
    """
    Create a diffuse reflectance mode simulator.

    Args:
        geometry: Measurement geometry.
        particle_size_um: Mean particle size.
        random_state: Random seed.

    Returns:
        Configured MeasurementModeSimulator.
    """
    config = MeasurementModeConfig(
        mode=MeasurementMode.REFLECTANCE,
        reflectance=ReflectanceConfig(geometry=geometry),
        scattering=ScatteringConfig(particle_size_um=particle_size_um)
    )
    return MeasurementModeSimulator(config, random_state)

def create_atr_simulator(
    crystal_material: str = "diamond",
    incidence_angle: float = 45.0,
    n_reflections: int = 1,
    random_state: int | None = None
) -> MeasurementModeSimulator:
    """
    Create an ATR mode simulator.

    Args:
        crystal_material: ATR crystal material.
        incidence_angle: Incidence angle in degrees.
        n_reflections: Number of internal reflections.
        random_state: Random seed.

    Returns:
        Configured MeasurementModeSimulator.
    """
    # Get crystal properties
    n_crystal = ATR_CRYSTAL_PROPERTIES[crystal_material]["refractive_index"] if crystal_material in ATR_CRYSTAL_PROPERTIES else 2.4  # Default to diamond-like

    config = MeasurementModeConfig(
        mode=MeasurementMode.ATR,
        atr=ATRConfig(
            crystal_material=crystal_material,
            crystal_refractive_index=n_crystal,
            incidence_angle=incidence_angle,
            n_reflections=n_reflections
        )
    )
    return MeasurementModeSimulator(config, random_state)

# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Enums
    "MeasurementMode",
    # Configuration dataclasses
    "TransmittanceConfig",
    "ReflectanceConfig",
    "TransflectanceConfig",
    "ATRConfig",
    "ScatteringConfig",
    "MeasurementModeConfig",
    # Crystal properties
    "ATR_CRYSTAL_PROPERTIES",
    # Simulator
    "MeasurementModeSimulator",
    # Convenience functions
    "create_transmittance_simulator",
    "create_reflectance_simulator",
    "create_atr_simulator",
]
