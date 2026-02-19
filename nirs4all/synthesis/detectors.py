"""
Detector simulation for synthetic NIRS data generation.

This module provides detailed simulation of NIR detector characteristics
including spectral response curves, noise models, and nonlinearity effects.

Key Features:
    - Detector spectral response curves (Si, InGaAs, PbS, etc.)
    - Shot noise, thermal noise, read noise, and 1/f noise models
    - Detector nonlinearity simulation
    - Temperature-dependent behavior

References:
    - Rogalski, A. (2002). Infrared Detectors: An Overview.
      Infrared Physics & Technology, 43(3-5), 187-210.
    - Vincent, J. D., Hodges, S., Vampola, J., Stegall, M., & Pierce, G. (2015).
      Fundamentals of Infrared and Visible Detector Operation and Testing.
      Wiley.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared
      Analysis. CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from scipy.interpolate import interp1d

from .instruments import DetectorType


@dataclass
class DetectorSpectralResponse:
    """
    Spectral response curve for a detector.

    Defines the wavelength-dependent sensitivity (quantum efficiency)
    of the detector.

    Attributes:
        detector_type: Type of detector.
        wavelengths: Wavelength grid for response curve (nm).
        response: Relative response at each wavelength (0-1).
        peak_wavelength: Wavelength of peak response (nm).
        cutoff_wavelength: Long-wavelength cutoff (nm).
        short_cutoff: Short-wavelength cutoff (nm).
        peak_qe: Peak quantum efficiency (0-1).
    """
    detector_type: DetectorType
    wavelengths: np.ndarray
    response: np.ndarray
    peak_wavelength: float
    cutoff_wavelength: float
    short_cutoff: float
    peak_qe: float = 0.7

    def get_response_at(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Get detector response at specified wavelengths.

        Args:
            wavelengths: Wavelengths to evaluate (nm).

        Returns:
            Detector response at each wavelength.
        """
        # Interpolate response curve
        interp_func = interp1d(
            self.wavelengths,
            self.response,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        return interp_func(wavelengths)

def _create_silicon_response() -> DetectorSpectralResponse:
    """Create silicon detector spectral response."""
    wl = np.array([350, 400, 500, 600, 700, 800, 900, 1000, 1050, 1100, 1150])
    resp = np.array([0.1, 0.3, 0.55, 0.7, 0.8, 0.85, 0.75, 0.5, 0.3, 0.1, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.SI,
        wavelengths=wl,
        response=resp,
        peak_wavelength=850,
        cutoff_wavelength=1100,
        short_cutoff=350,
        peak_qe=0.85
    )

def _create_ingaas_response() -> DetectorSpectralResponse:
    """Create InGaAs detector spectral response."""
    wl = np.array([850, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1650, 1700, 1750])
    resp = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.92, 0.9, 0.85, 0.75, 0.6, 0.3, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.INGAAS,
        wavelengths=wl,
        response=resp,
        peak_wavelength=1300,
        cutoff_wavelength=1700,
        short_cutoff=850,
        peak_qe=0.92
    )

def _create_ingaas_extended_response() -> DetectorSpectralResponse:
    """Create extended InGaAs detector spectral response."""
    wl = np.array([850, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 2600])
    resp = np.array([0.1, 0.4, 0.65, 0.75, 0.8, 0.75, 0.65, 0.5, 0.3, 0.15, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.INGAAS_EXTENDED,
        wavelengths=wl,
        response=resp,
        peak_wavelength=1600,
        cutoff_wavelength=2500,
        short_cutoff=850,
        peak_qe=0.80
    )

def _create_pbs_response() -> DetectorSpectralResponse:
    """Create PbS detector spectral response."""
    wl = np.array([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200])
    resp = np.array([0.1, 0.3, 0.55, 0.7, 0.8, 0.85, 0.75, 0.5, 0.25, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.PBS,
        wavelengths=wl,
        response=resp,
        peak_wavelength=2200,
        cutoff_wavelength=3000,
        short_cutoff=1000,
        peak_qe=0.85
    )

def _create_pbse_response() -> DetectorSpectralResponse:
    """Create PbSe detector spectral response."""
    wl = np.array([1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5200])
    resp = np.array([0.1, 0.35, 0.55, 0.7, 0.8, 0.75, 0.55, 0.25, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.PBSE,
        wavelengths=wl,
        response=resp,
        peak_wavelength=4000,
        cutoff_wavelength=5000,
        short_cutoff=1500,
        peak_qe=0.80
    )

def _create_mems_response() -> DetectorSpectralResponse:
    """Create MEMS-based detector spectral response (typically InGaAs-based)."""
    wl = np.array([900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500])
    resp = np.array([0.15, 0.35, 0.55, 0.7, 0.75, 0.7, 0.55, 0.35, 0.15, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.MEMS,
        wavelengths=wl,
        response=resp,
        peak_wavelength=1600,
        cutoff_wavelength=2400,
        short_cutoff=900,
        peak_qe=0.75
    )

def _create_mct_response() -> DetectorSpectralResponse:
    """Create MCT (Mercury Cadmium Telluride) detector spectral response."""
    wl = np.array([2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 14000])
    resp = np.array([0.1, 0.4, 0.65, 0.8, 0.85, 0.8, 0.65, 0.4, 0.0])

    return DetectorSpectralResponse(
        detector_type=DetectorType.MCT,
        wavelengths=wl,
        response=resp,
        peak_wavelength=6000,
        cutoff_wavelength=12000,
        short_cutoff=2000,
        peak_qe=0.85
    )

# Registry of detector spectral responses
DETECTOR_RESPONSES: dict[DetectorType, DetectorSpectralResponse] = {}

def _register_detector_responses() -> None:
    """Register all detector spectral responses."""
    global DETECTOR_RESPONSES

    DETECTOR_RESPONSES = {
        DetectorType.SI: _create_silicon_response(),
        DetectorType.INGAAS: _create_ingaas_response(),
        DetectorType.INGAAS_EXTENDED: _create_ingaas_extended_response(),
        DetectorType.PBS: _create_pbs_response(),
        DetectorType.PBSE: _create_pbse_response(),
        DetectorType.MEMS: _create_mems_response(),
        DetectorType.MCT: _create_mct_response(),
    }

_register_detector_responses()

def get_detector_response(detector_type: DetectorType) -> DetectorSpectralResponse:
    """
    Get spectral response curve for a detector type.

    Args:
        detector_type: Type of detector.

    Returns:
        DetectorSpectralResponse object.
    """
    return DETECTOR_RESPONSES[detector_type]

# ============================================================================
# Noise Models
# ============================================================================

@dataclass
class NoiseModelConfig:
    """
    Configuration for detector noise model.

    Attributes:
        shot_noise_enabled: Enable shot (photon) noise.
        thermal_noise_enabled: Enable thermal (Johnson) noise.
        read_noise_enabled: Enable readout noise.
        flicker_noise_enabled: Enable 1/f (flicker) noise.
        quantization_noise_enabled: Enable ADC quantization noise.
        shot_noise_factor: Scaling factor for shot noise.
        thermal_noise_factor: Scaling factor for thermal noise.
        read_noise_electrons: Read noise in electrons.
        flicker_corner_freq: 1/f noise corner frequency (Hz).
        adc_bits: ADC resolution in bits.
        full_scale: Full-scale signal level.
    """
    shot_noise_enabled: bool = True
    thermal_noise_enabled: bool = True
    read_noise_enabled: bool = True
    flicker_noise_enabled: bool = False
    quantization_noise_enabled: bool = False
    shot_noise_factor: float = 1.0
    thermal_noise_factor: float = 1.0
    read_noise_electrons: float = 50.0
    flicker_corner_freq: float = 100.0  # Hz
    adc_bits: int = 16
    full_scale: float = 3.0  # AU

@dataclass
class DetectorConfig:
    """
    Complete detector configuration.

    Attributes:
        detector_type: Type of detector.
        temperature_k: Operating temperature in Kelvin.
        integration_time_ms: Integration time in milliseconds.
        gain: Amplifier gain.
        noise_model: Noise model configuration.
        apply_response_curve: Whether to apply spectral response.
        apply_nonlinearity: Whether to apply detector nonlinearity.
        nonlinearity_coefficient: Quadratic nonlinearity coefficient.
    """
    detector_type: DetectorType = DetectorType.INGAAS
    temperature_k: float = 293.0  # 20°C room temperature
    integration_time_ms: float = 100.0
    gain: float = 1.0
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    apply_response_curve: bool = True
    apply_nonlinearity: bool = False
    nonlinearity_coefficient: float = 0.02  # Quadratic term

# Detector-specific default noise parameters
DETECTOR_NOISE_DEFAULTS = {
    DetectorType.SI: {
        "shot_noise_factor": 0.8,
        "thermal_noise_factor": 0.5,
        "read_noise_electrons": 30.0,
        "flicker_noise_enabled": False,
    },
    DetectorType.INGAAS: {
        "shot_noise_factor": 1.0,
        "thermal_noise_factor": 0.8,
        "read_noise_electrons": 50.0,
        "flicker_noise_enabled": False,
    },
    DetectorType.INGAAS_EXTENDED: {
        "shot_noise_factor": 1.2,
        "thermal_noise_factor": 1.2,
        "read_noise_electrons": 80.0,
        "flicker_noise_enabled": False,
    },
    DetectorType.PBS: {
        "shot_noise_factor": 1.5,
        "thermal_noise_factor": 1.8,
        "read_noise_electrons": 150.0,
        "flicker_noise_enabled": True,  # PbS has significant 1/f noise
    },
    DetectorType.PBSE: {
        "shot_noise_factor": 1.4,
        "thermal_noise_factor": 1.5,
        "read_noise_electrons": 120.0,
        "flicker_noise_enabled": True,
    },
    DetectorType.MEMS: {
        "shot_noise_factor": 1.5,
        "thermal_noise_factor": 1.0,
        "read_noise_electrons": 100.0,
        "flicker_noise_enabled": False,
    },
    DetectorType.MCT: {
        "shot_noise_factor": 0.7,  # Cooled, low noise
        "thermal_noise_factor": 0.3,
        "read_noise_electrons": 20.0,
        "flicker_noise_enabled": False,
    },
}

def get_default_noise_config(detector_type: DetectorType) -> NoiseModelConfig:
    """
    Get default noise model configuration for a detector type.

    Args:
        detector_type: Type of detector.

    Returns:
        NoiseModelConfig with appropriate defaults.
    """
    defaults = DETECTOR_NOISE_DEFAULTS.get(detector_type, {})
    return NoiseModelConfig(**defaults)

# ============================================================================
# Detector Simulator
# ============================================================================

class DetectorSimulator:
    """
    Simulate detector effects on NIR spectra.

    Applies detector spectral response, noise models, and nonlinearity
    to synthetic spectra.

    Attributes:
        config: Detector configuration.
        rng: Random number generator.

    Example:
        >>> config = DetectorConfig(detector_type=DetectorType.INGAAS)
        >>> simulator = DetectorSimulator(config, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        random_state: int | None = None
    ) -> None:
        """
        Initialize the detector simulator.

        Args:
            config: Detector configuration. If None, uses defaults.
            random_state: Random seed for reproducibility.
        """
        if config is None:
            config = DetectorConfig()
        self.config = config
        self.rng = np.random.default_rng(random_state)

        # Get spectral response curve
        self.response = get_detector_response(config.detector_type)

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        base_signal_level: float = 1.0
    ) -> np.ndarray:
        """
        Apply detector effects to spectra.

        Args:
            spectra: Input spectra (n_samples, n_wavelengths).
            wavelengths: Wavelength array (nm).
            base_signal_level: Reference signal level for noise scaling.

        Returns:
            Spectra with detector effects applied.
        """
        result = spectra.copy()

        # 1. Apply spectral response (wavelength-dependent sensitivity)
        if self.config.apply_response_curve:
            result = self._apply_spectral_response(result, wavelengths)

        # 2. Apply gain
        result = result * self.config.gain

        # 3. Apply detector nonlinearity
        if self.config.apply_nonlinearity:
            result = self._apply_nonlinearity(result)

        # 4. Apply noise model
        result = self._apply_noise(result, wavelengths, base_signal_level)

        return result

    def _apply_spectral_response(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Apply detector spectral response curve."""
        response = self.response.get_response_at(wavelengths)

        # Avoid amplifying noise where response is very low
        response = np.maximum(response, 0.01)

        # Response affects signal (higher response = more signal collected)
        # but spectroscopically we see absorbance, which is inverse
        # For realism, we add noise proportional to 1/response
        noise_scaling = 1.0 / response
        noise_scaling = np.minimum(noise_scaling, 10.0)  # Cap scaling

        # Store for use in noise application
        self._response_noise_scaling = noise_scaling

        return spectra

    def _apply_nonlinearity(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply detector nonlinearity.

        Models: signal_measured = signal_true * (1 + a * signal_true)
        where a is the nonlinearity coefficient.
        """
        coef = self.config.nonlinearity_coefficient

        # Quadratic nonlinearity
        result = spectra * (1 + coef * spectra)

        return result

    def _apply_noise(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        base_signal_level: float
    ) -> np.ndarray:
        """Apply noise model to spectra."""
        n_samples, n_wl = spectra.shape
        noise_config = self.config.noise_model
        result = spectra.copy()

        # Get wavelength-dependent noise scaling from response
        wl_noise_scale = self._response_noise_scaling if hasattr(self, '_response_noise_scaling') else np.ones(n_wl)

        # Noise level based on integration time
        # Longer integration = lower noise (sqrt relationship)
        time_factor = np.sqrt(100.0 / self.config.integration_time_ms)

        # Base noise level
        base_noise = 0.001 * time_factor  # ~0.001 AU typical

        for i in range(n_samples):
            total_noise = np.zeros(n_wl)

            # Shot noise (signal-dependent, Poisson statistics)
            if noise_config.shot_noise_enabled:
                shot_sigma = (
                    noise_config.shot_noise_factor *
                    base_noise *
                    np.sqrt(np.abs(spectra[i]) + 0.01)
                )
                total_noise += self.rng.normal(0, shot_sigma)

            # Thermal noise (constant, wavelength-independent)
            if noise_config.thermal_noise_enabled:
                thermal_sigma = (
                    noise_config.thermal_noise_factor *
                    base_noise *
                    self._temperature_factor()
                )
                total_noise += self.rng.normal(0, thermal_sigma, n_wl)

            # Read noise (constant per pixel)
            if noise_config.read_noise_enabled:
                # Convert electrons to AU (rough conversion)
                read_au = noise_config.read_noise_electrons * 1e-5
                total_noise += self.rng.normal(0, read_au, n_wl)

            # 1/f noise (correlated, low frequency)
            if noise_config.flicker_noise_enabled:
                flicker = self._generate_flicker_noise(
                    n_wl,
                    base_noise * 0.5
                )
                total_noise += flicker

            # Apply wavelength-dependent scaling
            total_noise = total_noise * wl_noise_scale

            result[i] = result[i] + total_noise

        # Quantization noise
        if noise_config.quantization_noise_enabled:
            result = self._apply_quantization(result, noise_config)

        return result

    def _temperature_factor(self) -> float:
        """
        Calculate temperature-dependent noise factor.

        Thermal noise increases with temperature.
        """
        reference_temp = 293.0  # 20°C
        actual_temp = self.config.temperature_k

        # Thermal noise proportional to sqrt(T)
        return np.sqrt(actual_temp / reference_temp)

    def _generate_flicker_noise(
        self,
        n_points: int,
        amplitude: float
    ) -> np.ndarray:
        """
        Generate 1/f (flicker) noise.

        Flicker noise has power spectral density proportional to 1/f.
        """
        # Generate white noise
        white = self.rng.normal(0, 1, n_points)

        # Create 1/f filter
        freqs = np.fft.fftfreq(n_points)
        freqs[0] = 1e-10  # Avoid division by zero
        fft_filter = 1.0 / np.sqrt(np.abs(freqs))
        fft_filter[0] = 0  # Remove DC

        # Apply filter in frequency domain
        fft_white = np.fft.fft(white)
        fft_pink = fft_white * fft_filter
        pink = np.real(np.fft.ifft(fft_pink))

        # Normalize and scale
        pink = pink / np.std(pink) * amplitude

        return pink

    def _apply_quantization(
        self,
        spectra: np.ndarray,
        config: NoiseModelConfig
    ) -> np.ndarray:
        """Apply ADC quantization effects."""
        # Calculate step size
        n_levels = 2 ** config.adc_bits
        step = config.full_scale / n_levels

        # Quantize
        quantized = np.round(spectra / step) * step

        # Add small uniform noise (quantization noise approximation)
        q_noise = self.rng.uniform(-step / 2, step / 2, spectra.shape)

        return quantized + q_noise

# ============================================================================
# Convenience Functions
# ============================================================================

def simulate_detector_effects(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    detector_type: DetectorType = DetectorType.INGAAS,
    include_response: bool = True,
    include_noise: bool = True,
    random_state: int | None = None
) -> np.ndarray:
    """
    Apply detector effects to spectra with simple API.

    Args:
        spectra: Input spectra (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        detector_type: Type of detector to simulate.
        include_response: Whether to apply spectral response.
        include_noise: Whether to apply noise.
        random_state: Random seed.

    Returns:
        Spectra with detector effects applied.

    Example:
        >>> spectra_out = simulate_detector_effects(
        ...     spectra, wavelengths,
        ...     detector_type=DetectorType.PBS
        ... )
    """
    # Create noise config
    noise_config = get_default_noise_config(detector_type)
    if not include_noise:
        noise_config.shot_noise_enabled = False
        noise_config.thermal_noise_enabled = False
        noise_config.read_noise_enabled = False
        noise_config.flicker_noise_enabled = False

    config = DetectorConfig(
        detector_type=detector_type,
        noise_model=noise_config,
        apply_response_curve=include_response
    )

    simulator = DetectorSimulator(config, random_state)
    return simulator.apply(spectra, wavelengths)

def get_detector_wavelength_range(detector_type: DetectorType) -> tuple[float, float]:
    """
    Get the effective wavelength range for a detector type.

    Args:
        detector_type: Type of detector.

    Returns:
        Tuple of (min_wavelength, max_wavelength) in nm.
    """
    response = get_detector_response(detector_type)
    return (response.short_cutoff, response.cutoff_wavelength)

def list_detector_types() -> list[str]:
    """
    List available detector types.

    Returns:
        List of detector type names.
    """
    return [dt.value for dt in DetectorType]

# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Data classes
    "DetectorSpectralResponse",
    "NoiseModelConfig",
    "DetectorConfig",
    # Registry
    "DETECTOR_RESPONSES",
    "DETECTOR_NOISE_DEFAULTS",
    "get_detector_response",
    "get_default_noise_config",
    # Simulator
    "DetectorSimulator",
    # Convenience functions
    "simulate_detector_effects",
    "get_detector_wavelength_range",
    "list_detector_types",
]
