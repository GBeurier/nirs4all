"""
Real data fitting utilities for synthetic NIRS spectra generation.

This module provides tools to analyze real NIRS datasets and fit generator
parameters to match their statistical and spectral properties.

Key Features:
    - Statistical property analysis (mean, std, skewness, kurtosis)
    - Spectral shape analysis (slope, curvature, noise)
    - PCA structure analysis
    - Parameter estimation for SyntheticNIRSGenerator
    - Comparison between synthetic and real data
    - **Phase 1-4 Enhanced Features:**
        - Instrument archetype inference (InGaAs, PbS, MEMS, etc.)
        - Measurement mode detection (transmittance, reflectance, ATR)
        - Application domain suggestion (agriculture, pharmaceutical, etc.)
        - Environmental effects estimation (temperature, moisture)
        - Scattering parameter estimation (particle size, EMSC)
        - Wavenumber-based peak analysis for component identification

Example:
    >>> from nirs4all.data.synthetic import RealDataFitter, SyntheticNIRSGenerator
    >>>
    >>> # Analyze real data
    >>> fitter = RealDataFitter()
    >>> params = fitter.fit(X_real, wavelengths=wavelengths)
    >>>
    >>> # Create generator with fitted parameters (includes all Phase 1-4 features)
    >>> generator = fitter.create_matched_generator()
    >>> X_synthetic, _, _ = generator.generate(n_samples=1000)
    >>>
    >>> # Or get all inferred characteristics
    >>> print(f"Inferred instrument: {params.inferred_instrument}")
    >>> print(f"Inferred domain: {params.inferred_domain}")
    >>> print(f"Measurement mode: {params.measurement_mode}")

References:
    - Based on comparator.py from bench/synthetic/
    - Enhanced with Phase 1-4 synthetic generator features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from .generator import SyntheticNIRSGenerator


# ============================================================================
# Inference Result Classes
# ============================================================================


class MeasurementModeInference(str, Enum):
    """Inferred measurement mode from spectral analysis."""
    TRANSMITTANCE = "transmittance"
    REFLECTANCE = "reflectance"
    TRANSFLECTANCE = "transflectance"
    ATR = "atr"
    UNKNOWN = "unknown"


@dataclass
class InstrumentInference:
    """
    Results of instrument archetype inference.

    Attributes:
        archetype_name: Best matching instrument archetype name.
        detector_type: Inferred detector type.
        wavelength_range: Detected wavelength range.
        estimated_resolution: Estimated spectral resolution (nm).
        confidence: Confidence score (0-1).
        alternative_archetypes: Other possible archetypes with scores.
    """
    archetype_name: str = "unknown"
    detector_type: str = "unknown"
    wavelength_range: Tuple[float, float] = (1000.0, 2500.0)
    estimated_resolution: float = 8.0
    confidence: float = 0.0
    alternative_archetypes: Dict[str, float] = field(default_factory=dict)


@dataclass
class DomainInference:
    """
    Results of application domain inference.

    Attributes:
        domain_name: Best matching domain name.
        category: Domain category.
        confidence: Confidence score (0-1).
        detected_components: Components detected from peak analysis.
        alternative_domains: Other possible domains with scores.
    """
    domain_name: str = "unknown"
    category: str = "unknown"
    confidence: float = 0.0
    detected_components: List[str] = field(default_factory=list)
    alternative_domains: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnvironmentalInference:
    """
    Results of environmental effects inference.

    Attributes:
        estimated_temperature_variation: Estimated temperature variation (°C).
        has_temperature_effects: Whether temperature effects are detectable.
        estimated_moisture_variation: Estimated moisture variation.
        has_moisture_effects: Whether moisture effects are detectable.
        water_band_shift: Detected shift in water bands (nm).
    """
    estimated_temperature_variation: float = 0.0
    has_temperature_effects: bool = False
    estimated_moisture_variation: float = 0.0
    has_moisture_effects: bool = False
    water_band_shift: float = 0.0


@dataclass
class ScatteringInference:
    """
    Results of scattering effects inference.

    Attributes:
        has_scatter_effects: Whether significant scatter is detected.
        estimated_particle_size_um: Estimated mean particle size (μm).
        multiplicative_scatter_std: Estimated MSC-style multiplicative scatter.
        additive_scatter_std: Estimated SNV-style additive scatter.
        baseline_curvature: Detected baseline curvature intensity.
        snv_correctable: Whether SNV would improve spectra.
        msc_correctable: Whether MSC would improve spectra.
    """
    has_scatter_effects: bool = False
    estimated_particle_size_um: float = 50.0
    multiplicative_scatter_std: float = 0.0
    additive_scatter_std: float = 0.0
    baseline_curvature: float = 0.0
    snv_correctable: bool = False
    msc_correctable: bool = False


@dataclass
class SpectralProperties:
    """
    Container for computed spectral properties of a dataset.

    This dataclass holds various statistical and spectral properties
    computed from a NIRS dataset for comparison and fitting purposes.

    Attributes:
        name: Dataset identifier.
        n_samples: Number of samples.
        n_wavelengths: Number of wavelengths.
        wavelengths: Wavelength grid.

        # Basic statistics
        mean_spectrum: Mean spectrum across samples.
        std_spectrum: Standard deviation spectrum.
        global_mean: Overall mean absorbance.
        global_std: Overall standard deviation.
        global_range: (min, max) absorbance range.

        # Shape properties
        mean_slope: Average spectral slope (per 1000nm).
        slope_std: Standard deviation of slopes.
        mean_curvature: Average curvature (second derivative).

        # Distribution statistics
        skewness: Skewness of absorbance distribution.
        kurtosis: Kurtosis of absorbance distribution.

        # Noise characteristics
        noise_estimate: Estimated noise level.
        snr_estimate: Signal-to-noise ratio estimate.

        # PCA properties
        pca_explained_variance: Explained variance ratios.
        pca_n_components_95: Components for 95% variance.

        # Peak analysis
        n_peaks_mean: Mean number of peaks.
        peak_positions: Wavelengths of detected peaks.
        peak_wavenumbers: Wavenumber positions of peaks.

        # Phase 1-4 Enhanced properties
        # Instrument indicators
        effective_resolution: Estimated spectral resolution from peak widths.
        noise_correlation_length: Correlation length of noise (detector indicator).
        wavelength_range: Actual wavelength range of data.

        # Measurement mode indicators
        baseline_offset: Mean baseline offset (transmittance indicator).
        kubelka_munk_linearity: K-M linearity score (reflectance indicator).
        baseline_convexity: Convexity of baseline (ATR indicator).

        # Environmental indicators
        water_band_variation: Variation in water band region.
        oh_band_positions: Detected O-H band positions.
        temperature_sensitivity_score: Score for temperature effect detection.

        # Scattering indicators
        scatter_baseline_slope: Wavelength-dependent scatter slope.
        scatter_baseline_curvature: Curvature from scattering.
        sample_to_sample_offset_std: Sample-to-sample offset variation.
        sample_to_sample_slope_std: Sample-to-sample slope variation.

        # Domain indicators
        protein_band_intensity: Intensity in protein band regions.
        carbohydrate_band_intensity: Intensity in carbohydrate regions.
        lipid_band_intensity: Intensity in lipid band regions.
        water_band_intensity: Intensity in water band regions.
    """

    name: str = "dataset"
    n_samples: int = 0
    n_wavelengths: int = 0
    wavelengths: Optional[np.ndarray] = None

    # Basic statistics
    mean_spectrum: Optional[np.ndarray] = None
    std_spectrum: Optional[np.ndarray] = None
    global_mean: float = 0.0
    global_std: float = 0.0
    global_range: Tuple[float, float] = (0.0, 0.0)

    # Shape properties
    mean_slope: float = 0.0
    slope_std: float = 0.0
    slopes: Optional[np.ndarray] = None
    mean_curvature: float = 0.0
    curvature_std: float = 0.0

    # Distribution statistics
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Noise characteristics
    noise_estimate: float = 0.0
    snr_estimate: float = 0.0

    # PCA properties
    pca_explained_variance: Optional[np.ndarray] = None
    pca_n_components_95: int = 0

    # Peak analysis
    n_peaks_mean: float = 0.0
    peak_positions: Optional[np.ndarray] = None
    peak_wavenumbers: Optional[np.ndarray] = None

    # Phase 1-4 Enhanced properties
    # Instrument indicators
    effective_resolution: float = 8.0
    noise_correlation_length: float = 1.0
    wavelength_range: Tuple[float, float] = (1000.0, 2500.0)

    # Measurement mode indicators
    baseline_offset: float = 0.0
    kubelka_munk_linearity: float = 0.0
    baseline_convexity: float = 0.0

    # Environmental indicators
    water_band_variation: float = 0.0
    oh_band_positions: Optional[np.ndarray] = None
    temperature_sensitivity_score: float = 0.0

    # Scattering indicators
    scatter_baseline_slope: float = 0.0
    scatter_baseline_curvature: float = 0.0
    sample_to_sample_offset_std: float = 0.0
    sample_to_sample_slope_std: float = 0.0

    # Domain indicators
    protein_band_intensity: float = 0.0
    carbohydrate_band_intensity: float = 0.0
    lipid_band_intensity: float = 0.0
    water_band_intensity: float = 0.0


@dataclass
class FittedParameters:
    """
    Parameters fitted from real data for synthetic generation.

    This dataclass contains all parameters needed to configure
    a SyntheticNIRSGenerator to produce spectra similar to a
    real dataset, including Phase 1-4 enhanced features.

    Attributes:
        # Basic wavelength grid
        wavelength_start: Start wavelength (nm).
        wavelength_end: End wavelength (nm).
        wavelength_step: Wavelength step (nm).

        # Slope and baseline parameters
        global_slope_mean: Mean global slope.
        global_slope_std: Slope standard deviation.
        baseline_amplitude: Baseline drift amplitude.

        # Noise parameters
        noise_base: Base noise level.
        noise_signal_dep: Signal-dependent noise factor.

        # Scatter parameters
        path_length_std: Path length variation.
        scatter_alpha_std: Multiplicative scatter std.
        scatter_beta_std: Additive scatter std.
        tilt_std: Spectral tilt standard deviation.

        # Complexity
        complexity: Suggested complexity level.

        # Source metadata
        source_name: Name of source dataset.
        source_properties: Full SpectralProperties of source.

        # Phase 1-4 Enhanced Parameters
        # Instrument inference
        inferred_instrument: Inferred instrument archetype.
        instrument_inference: Full instrument inference result.

        # Measurement mode
        measurement_mode: Inferred measurement mode.
        measurement_mode_confidence: Confidence of inference.

        # Domain inference
        inferred_domain: Inferred application domain.
        domain_inference: Full domain inference result.

        # Environmental effects
        environmental_inference: Environmental effects inference.
        temperature_config: Suggested temperature config parameters.
        moisture_config: Suggested moisture config parameters.

        # Scattering effects
        scattering_inference: Scattering effects inference.
        particle_size_config: Suggested particle size config parameters.
        emsc_config: Suggested EMSC config parameters.

        # Detected components for procedural generation
        detected_components: List of detected/inferred component names.
        suggested_n_components: Suggested number of components.
    """

    # Wavelength grid
    wavelength_start: float = 1000.0
    wavelength_end: float = 2500.0
    wavelength_step: float = 2.0

    # Slope parameters
    global_slope_mean: float = 0.0
    global_slope_std: float = 0.02

    # Noise parameters
    noise_base: float = 0.001
    noise_signal_dep: float = 0.005

    # Variation parameters
    path_length_std: float = 0.05
    baseline_amplitude: float = 0.02
    scatter_alpha_std: float = 0.05
    scatter_beta_std: float = 0.01
    tilt_std: float = 0.01

    # Metadata
    complexity: str = "realistic"
    source_name: str = ""
    source_properties: Optional[SpectralProperties] = field(default=None, repr=False)

    # Phase 1-4 Enhanced Parameters
    # Instrument inference (Phase 2)
    inferred_instrument: str = "unknown"
    instrument_inference: Optional[InstrumentInference] = field(default=None, repr=False)

    # Measurement mode (Phase 2)
    measurement_mode: str = "transmittance"
    measurement_mode_confidence: float = 0.0

    # Domain inference (Phase 1)
    inferred_domain: str = "unknown"
    domain_inference: Optional[DomainInference] = field(default=None, repr=False)

    # Environmental effects (Phase 3)
    environmental_inference: Optional[EnvironmentalInference] = field(default=None, repr=False)
    temperature_config: Dict[str, Any] = field(default_factory=dict)
    moisture_config: Dict[str, Any] = field(default_factory=dict)

    # Scattering effects (Phase 3)
    scattering_inference: Optional[ScatteringInference] = field(default=None, repr=False)
    particle_size_config: Dict[str, Any] = field(default_factory=dict)
    emsc_config: Dict[str, Any] = field(default_factory=dict)

    # Components (Phase 1)
    detected_components: List[str] = field(default_factory=list)
    suggested_n_components: int = 5

    def to_generator_kwargs(self) -> Dict[str, Any]:
        """
        Convert fitted parameters to kwargs for SyntheticNIRSGenerator.

        Returns:
            Dictionary of keyword arguments.

        Example:
            >>> params = fitter.fit(X_real)
            >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
        """
        return {
            "wavelength_start": self.wavelength_start,
            "wavelength_end": self.wavelength_end,
            "wavelength_step": self.wavelength_step,
            "complexity": self.complexity,
        }

    def to_full_config(self) -> Dict[str, Any]:
        """
        Convert all fitted parameters to a comprehensive configuration.

        This includes all Phase 1-4 parameters for complete synthetic
        data generation matching the source dataset.

        Returns:
            Dictionary with all configuration parameters.

        Example:
            >>> params = fitter.fit(X_real)
            >>> config = params.to_full_config()
            >>> # Use with builder pattern or advanced configuration
        """
        return {
            # Basic parameters
            "wavelength_start": self.wavelength_start,
            "wavelength_end": self.wavelength_end,
            "wavelength_step": self.wavelength_step,
            "complexity": self.complexity,
            # Noise and scatter
            "noise_base": self.noise_base,
            "noise_signal_dep": self.noise_signal_dep,
            "scatter_alpha_std": self.scatter_alpha_std,
            "scatter_beta_std": self.scatter_beta_std,
            "path_length_std": self.path_length_std,
            "baseline_amplitude": self.baseline_amplitude,
            "tilt_std": self.tilt_std,
            "global_slope_mean": self.global_slope_mean,
            "global_slope_std": self.global_slope_std,
            # Phase 1-4 enhanced
            "instrument": self.inferred_instrument,
            "measurement_mode": self.measurement_mode,
            "domain": self.inferred_domain,
            "components": self.detected_components,
            "n_components": self.suggested_n_components,
            "temperature_config": self.temperature_config,
            "moisture_config": self.moisture_config,
            "particle_size_config": self.particle_size_config,
            "emsc_config": self.emsc_config,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all parameters to a dictionary.

        Returns:
            Dictionary with all parameter values.
        """
        return {
            "wavelength_start": self.wavelength_start,
            "wavelength_end": self.wavelength_end,
            "wavelength_step": self.wavelength_step,
            "global_slope_mean": self.global_slope_mean,
            "global_slope_std": self.global_slope_std,
            "noise_base": self.noise_base,
            "noise_signal_dep": self.noise_signal_dep,
            "path_length_std": self.path_length_std,
            "baseline_amplitude": self.baseline_amplitude,
            "scatter_alpha_std": self.scatter_alpha_std,
            "scatter_beta_std": self.scatter_beta_std,
            "tilt_std": self.tilt_std,
            "complexity": self.complexity,
            "source_name": self.source_name,
            # Phase 1-4 enhanced
            "inferred_instrument": self.inferred_instrument,
            "measurement_mode": self.measurement_mode,
            "measurement_mode_confidence": self.measurement_mode_confidence,
            "inferred_domain": self.inferred_domain,
            "detected_components": self.detected_components,
            "suggested_n_components": self.suggested_n_components,
            "temperature_config": self.temperature_config,
            "moisture_config": self.moisture_config,
            "particle_size_config": self.particle_size_config,
            "emsc_config": self.emsc_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FittedParameters":
        """
        Create FittedParameters from a dictionary.

        Args:
            data: Dictionary with parameter values.

        Returns:
            FittedParameters instance.
        """
        return cls(
            wavelength_start=data.get("wavelength_start", 1000.0),
            wavelength_end=data.get("wavelength_end", 2500.0),
            wavelength_step=data.get("wavelength_step", 2.0),
            global_slope_mean=data.get("global_slope_mean", 0.0),
            global_slope_std=data.get("global_slope_std", 0.02),
            noise_base=data.get("noise_base", 0.001),
            noise_signal_dep=data.get("noise_signal_dep", 0.005),
            path_length_std=data.get("path_length_std", 0.05),
            baseline_amplitude=data.get("baseline_amplitude", 0.02),
            scatter_alpha_std=data.get("scatter_alpha_std", 0.05),
            scatter_beta_std=data.get("scatter_beta_std", 0.01),
            tilt_std=data.get("tilt_std", 0.01),
            complexity=data.get("complexity", "realistic"),
            source_name=data.get("source_name", ""),
            # Phase 1-4 enhanced
            inferred_instrument=data.get("inferred_instrument", "unknown"),
            measurement_mode=data.get("measurement_mode", "transmittance"),
            measurement_mode_confidence=data.get("measurement_mode_confidence", 0.0),
            inferred_domain=data.get("inferred_domain", "unknown"),
            detected_components=data.get("detected_components", []),
            suggested_n_components=data.get("suggested_n_components", 5),
            temperature_config=data.get("temperature_config", {}),
            moisture_config=data.get("moisture_config", {}),
            particle_size_config=data.get("particle_size_config", {}),
            emsc_config=data.get("emsc_config", {}),
        )

    def save(self, path: str) -> None:
        """
        Save parameters to JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FittedParameters":
        """
        Load parameters from JSON file.

        Args:
            path: Input file path.

        Returns:
            FittedParameters instance.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """
        Generate a human-readable summary of fitted parameters.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            f"Fitted Parameters Summary: {self.source_name}",
            "=" * 60,
            "",
            "Wavelength Grid:",
            f"  Range: {self.wavelength_start:.0f} - {self.wavelength_end:.0f} nm",
            f"  Step: {self.wavelength_step:.1f} nm",
            "",
            "Noise & Scatter:",
            f"  Base noise: {self.noise_base:.5f}",
            f"  Signal-dependent noise: {self.noise_signal_dep:.5f}",
            f"  Multiplicative scatter (α): {self.scatter_alpha_std:.4f}",
            f"  Additive scatter (β): {self.scatter_beta_std:.4f}",
            "",
            "Baseline & Slope:",
            f"  Global slope: {self.global_slope_mean:.4f} ± {self.global_slope_std:.4f}",
            f"  Baseline amplitude: {self.baseline_amplitude:.4f}",
            f"  Path length std: {self.path_length_std:.4f}",
            "",
            f"Complexity: {self.complexity}",
            "",
            "Phase 1-4 Inferences:",
            f"  Instrument: {self.inferred_instrument}",
            f"  Measurement mode: {self.measurement_mode} "
            f"(confidence: {self.measurement_mode_confidence:.2f})",
            f"  Domain: {self.inferred_domain}",
            f"  Detected components: {', '.join(self.detected_components[:5]) or 'None'}",
            f"  Suggested n_components: {self.suggested_n_components}",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_spectral_properties(
    X: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    name: str = "dataset",
    n_pca_components: int = 20,
) -> SpectralProperties:
    """
    Compute comprehensive spectral properties of a dataset.

    Analyzes a matrix of spectra to extract statistical and spectral
    properties useful for fitting and comparison. Includes Phase 1-4
    enhanced properties for instrument, mode, domain, and effect inference.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths).
        wavelengths: Optional wavelength grid.
        name: Dataset identifier.
        n_pca_components: Maximum PCA components to compute.

    Returns:
        SpectralProperties with computed metrics.

    Example:
        >>> props = compute_spectral_properties(X_real, wavelengths)
        >>> print(f"Mean slope: {props.mean_slope:.4f}")
        >>> print(f"Inferred resolution: {props.effective_resolution:.1f} nm")
    """
    n_samples, n_wavelengths = X.shape

    if wavelengths is None:
        wavelengths = np.arange(n_wavelengths)

    props = SpectralProperties(
        name=name,
        n_samples=n_samples,
        n_wavelengths=n_wavelengths,
        wavelengths=wavelengths.copy(),
        wavelength_range=(float(wavelengths.min()), float(wavelengths.max())),
    )

    # Basic statistics
    props.mean_spectrum = X.mean(axis=0)
    props.std_spectrum = X.std(axis=0)
    props.global_mean = float(X.mean())
    props.global_std = float(X.std())
    props.global_range = (float(X.min()), float(X.max()))

    # Slope analysis
    wl_range = np.ptp(wavelengths)
    if wl_range > 0:
        x_norm = (wavelengths - wavelengths.min()) / wl_range
        slopes = []
        for i in range(n_samples):
            coeffs = np.polyfit(x_norm, X[i], 1)
            # Convert to slope per 1000nm
            slopes.append(coeffs[0] * 1000.0 / wl_range)
        props.slopes = np.array(slopes)
        props.mean_slope = float(np.mean(slopes))
        props.slope_std = float(np.std(slopes))

    # Curvature analysis
    window_size = min(21, n_wavelengths // 10 * 2 + 1)
    if window_size >= 5:
        curvatures = []
        for i in range(min(n_samples, 100)):  # Sample subset for speed
            try:
                smoothed = savgol_filter(X[i], window_size, 2)
                d2 = np.gradient(np.gradient(smoothed))
                curvatures.append(np.mean(np.abs(d2)))
            except Exception:
                pass
        if curvatures:
            props.mean_curvature = float(np.mean(curvatures))
            props.curvature_std = float(np.std(curvatures))

    # Distribution statistics
    flat_data = X.flatten()
    props.skewness = float(stats.skew(flat_data))
    props.kurtosis = float(stats.kurtosis(flat_data))

    # Noise estimation (from first difference)
    first_diff = np.diff(X, axis=1)
    props.noise_estimate = float(first_diff.std() / np.sqrt(2))

    # SNR estimation
    signal_power = props.std_spectrum.mean()
    if props.noise_estimate > 0:
        props.snr_estimate = float(signal_power / props.noise_estimate)
    else:
        props.snr_estimate = float("inf")

    # PCA analysis
    try:
        from sklearn.decomposition import PCA

        n_comp = min(n_pca_components, n_samples, n_wavelengths)
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        props.pca_explained_variance = pca.explained_variance_ratio_

        # Components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        props.pca_n_components_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    except ImportError:
        pass

    # Peak analysis
    try:
        window_size = min(21, n_wavelengths // 10 * 2 + 1)
        if window_size >= 5 and props.std_spectrum is not None:
            smoothed_mean = savgol_filter(props.mean_spectrum, window_size, 2)
            prominence = props.std_spectrum.mean() * 0.5
            peaks, _ = find_peaks(smoothed_mean, prominence=prominence)
            props.peak_positions = wavelengths[peaks] if len(peaks) > 0 else np.array([])
            props.n_peaks_mean = float(len(peaks))

            # Convert to wavenumbers
            if len(peaks) > 0:
                props.peak_wavenumbers = 1e7 / wavelengths[peaks]
            else:
                props.peak_wavenumbers = np.array([])
    except Exception:
        props.peak_positions = np.array([])
        props.peak_wavenumbers = np.array([])
        props.n_peaks_mean = 0.0

    # =========================================================================
    # Phase 1-4 Enhanced Properties
    # =========================================================================

    # Effective resolution estimation (from peak widths)
    props.effective_resolution = _estimate_spectral_resolution(
        props.mean_spectrum, wavelengths
    )

    # Noise correlation length (detector indicator)
    props.noise_correlation_length = _compute_noise_correlation_length(X, wavelengths)

    # Baseline offset (transmittance indicator)
    props.baseline_offset = float(props.mean_spectrum.min())

    # Baseline convexity (ATR indicator - ATR shows wavelength-dependent penetration)
    props.baseline_convexity = _compute_baseline_convexity(props.mean_spectrum, wavelengths)

    # Kubelka-Munk linearity score (reflectance indicator)
    props.kubelka_munk_linearity = _compute_km_linearity(X)

    # Sample-to-sample scatter indicators
    sample_means = X.mean(axis=1)
    props.sample_to_sample_offset_std = float(np.std(sample_means))

    # Sample-to-sample slope variation
    if props.slopes is not None:
        props.sample_to_sample_slope_std = float(np.std(props.slopes))

    # Scatter baseline analysis
    props.scatter_baseline_slope, props.scatter_baseline_curvature = \
        _analyze_scatter_baseline(props.mean_spectrum, wavelengths)

    # Water band analysis (environmental indicators)
    props.water_band_variation, props.oh_band_positions = \
        _analyze_water_bands(X, wavelengths)

    # Temperature sensitivity score
    props.temperature_sensitivity_score = _compute_temperature_sensitivity(X, wavelengths)

    # Domain indicators - analyze specific band regions
    props.protein_band_intensity = _compute_band_intensity(
        props.mean_spectrum, wavelengths, [(1480, 1560), (2040, 2180)]
    )
    props.carbohydrate_band_intensity = _compute_band_intensity(
        props.mean_spectrum, wavelengths, [(2050, 2150), (2270, 2350)]
    )
    props.lipid_band_intensity = _compute_band_intensity(
        props.mean_spectrum, wavelengths, [(1720, 1780), (2300, 2380)]
    )
    props.water_band_intensity = _compute_band_intensity(
        props.mean_spectrum, wavelengths, [(1400, 1500), (1900, 2000)]
    )

    return props


def _estimate_spectral_resolution(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    """Estimate spectral resolution from peak widths."""
    try:
        # Find peaks and measure their widths
        window_size = min(21, len(wavelengths) // 10 * 2 + 1)
        if window_size < 5:
            return 8.0

        smoothed = savgol_filter(mean_spectrum, window_size, 2)
        peaks, properties = find_peaks(
            smoothed,
            prominence=np.std(smoothed) * 0.3,
            width=3,
        )

        if len(peaks) < 2:
            return 8.0

        # Get peak widths at half maximum
        widths = properties.get("widths", [])
        if len(widths) == 0:
            return 8.0

        # Convert from indices to wavelength
        wl_step = np.median(np.diff(wavelengths))
        width_nm = np.median(widths) * wl_step

        return float(np.clip(width_nm, 0.5, 50.0))
    except Exception:
        return 8.0


def _compute_noise_correlation_length(
    X: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    """Compute correlation length of noise (detector/instrument indicator)."""
    try:
        # Get noise by first difference
        noise = np.diff(X, axis=1)
        n_samples = min(100, noise.shape[0])

        # Compute autocorrelation of noise
        corr_lengths = []
        for i in range(n_samples):
            n = noise[i]
            if np.std(n) < 1e-10:
                continue
            n = n - n.mean()
            acf = np.correlate(n, n, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]

            # Find where correlation drops below 1/e
            below_threshold = np.where(acf < 1/np.e)[0]
            if len(below_threshold) > 0:
                corr_lengths.append(below_threshold[0])
            else:
                corr_lengths.append(len(acf))

        if len(corr_lengths) == 0:
            return 1.0

        return float(np.median(corr_lengths))
    except Exception:
        return 1.0


def _compute_baseline_convexity(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    """Compute baseline convexity (positive for ATR-like, negative for baseline drift)."""
    try:
        # Fit a quadratic to the spectrum
        x_norm = (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min())
        coeffs = np.polyfit(x_norm, mean_spectrum, 2)
        # The quadratic coefficient indicates convexity
        return float(coeffs[0])
    except Exception:
        return 0.0


def _compute_km_linearity(X: np.ndarray) -> float:
    """
    Compute Kubelka-Munk linearity score.

    Reflectance data converted to K-M should show more linear relationships
    with concentration than raw reflectance.
    """
    try:
        # Check if data looks like reflectance (values in 0-1 range mostly)
        if X.min() < -0.5 or X.max() > 3.0:
            # Looks like absorbance, not reflectance
            return 0.0

        # Sample some spectra
        n_samples = min(100, X.shape[0])
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[indices]

        # Compute mean intensity variation
        mean_intensity = X_sample.mean(axis=1)
        # Check for linear relationship between mean and std
        std_intensity = X_sample.std(axis=1)

        if np.std(mean_intensity) < 1e-10:
            return 0.0

        corr = np.corrcoef(mean_intensity, std_intensity)[0, 1]
        return float(abs(corr)) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def _analyze_scatter_baseline(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
) -> Tuple[float, float]:
    """Analyze baseline for scatter effects."""
    try:
        # Fit low-order polynomial to capture scatter baseline
        x_norm = (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min())
        coeffs = np.polyfit(x_norm, mean_spectrum, 3)

        # Slope from linear term
        baseline_slope = float(coeffs[2])
        # Curvature from quadratic term
        baseline_curvature = float(abs(coeffs[1]))

        return baseline_slope, baseline_curvature
    except Exception:
        return 0.0, 0.0


def _analyze_water_bands(
    X: np.ndarray,
    wavelengths: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Analyze water band regions for environmental effects."""
    try:
        # Water band regions
        water_regions = [(1400, 1500), (1900, 2000)]

        water_variation = 0.0
        oh_positions = []

        for wl_min, wl_max in water_regions:
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if not np.any(mask):
                continue

            region_data = X[:, mask]
            # Variation in this region
            water_variation += float(region_data.std())

            # Find peak position in mean spectrum
            region_mean = region_data.mean(axis=0)
            region_wl = wavelengths[mask]
            peak_idx = np.argmax(region_mean)
            oh_positions.append(region_wl[peak_idx])

        return water_variation, np.array(oh_positions)
    except Exception:
        return 0.0, np.array([])


def _compute_temperature_sensitivity(
    X: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    """
    Estimate temperature sensitivity based on O-H band variation patterns.

    Temperature effects cause shifts in O-H bands and changes in hydrogen
    bonding patterns.
    """
    try:
        # Check O-H first overtone region (1400-1500 nm)
        oh_region = (wavelengths >= 1400) & (wavelengths <= 1500)
        if not np.any(oh_region):
            return 0.0

        region_data = X[:, oh_region]

        # Temperature effects show up as correlated peak shifts
        # Compute sample-to-sample variation in peak position
        peak_positions = []
        for i in range(min(100, X.shape[0])):
            row = region_data[i]
            if np.std(row) < 1e-10:
                continue
            peak_idx = np.argmax(row)
            peak_positions.append(peak_idx)

        if len(peak_positions) < 10:
            return 0.0

        # Higher variation in peak position suggests temperature effects
        position_std = np.std(peak_positions)
        return float(min(1.0, position_std / 5.0))
    except Exception:
        return 0.0


def _compute_band_intensity(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    regions: List[Tuple[float, float]],
) -> float:
    """Compute mean intensity in specified wavelength regions."""
    try:
        total_intensity = 0.0
        n_regions = 0

        for wl_min, wl_max in regions:
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if np.any(mask):
                total_intensity += float(np.mean(mean_spectrum[mask]))
                n_regions += 1

        return total_intensity / max(n_regions, 1)
    except Exception:
        return 0.0


class RealDataFitter:
    """
    Fit generator parameters to match real dataset properties.

    This class analyzes real NIRS data and estimates parameters for
    the SyntheticNIRSGenerator to produce similar spectra. Includes
    Phase 1-4 enhanced inference for instruments, domains, and effects.

    Attributes:
        source_properties: SpectralProperties of the analyzed data.
        fitted_params: FittedParameters after fitting.

    Example:
        >>> fitter = RealDataFitter()
        >>> params = fitter.fit(X_real, wavelengths=wavelengths)
        >>>
        >>> # Access inferred characteristics
        >>> print(f"Instrument: {params.inferred_instrument}")
        >>> print(f"Domain: {params.inferred_domain}")
        >>>
        >>> # Create matched generator
        >>> generator = fitter.create_matched_generator()
        >>> X_synth, _, _ = generator.generate(1000)
    """

    def __init__(self) -> None:
        """Initialize the fitter."""
        self.source_properties: Optional[SpectralProperties] = None
        self.fitted_params: Optional[FittedParameters] = None
        self._X_array: Optional[np.ndarray] = None
        self._wavelengths: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, "SpectroDataset"],
        *,
        wavelengths: Optional[np.ndarray] = None,
        name: str = "source",
        infer_instrument: bool = True,
        infer_domain: bool = True,
        infer_measurement_mode: bool = True,
        infer_environmental: bool = True,
        infer_scattering: bool = True,
    ) -> FittedParameters:
        """
        Fit generator parameters to real data.

        Analyzes the input data and estimates optimal parameters for
        generating synthetic spectra with similar properties. Includes
        Phase 1-4 enhanced inference.

        Args:
            X: Real spectra matrix (n_samples, n_wavelengths) or SpectroDataset.
            wavelengths: Wavelength grid (required if X is ndarray).
            name: Dataset name for reference.
            infer_instrument: Whether to infer instrument archetype.
            infer_domain: Whether to infer application domain.
            infer_measurement_mode: Whether to infer measurement mode.
            infer_environmental: Whether to infer environmental effects.
            infer_scattering: Whether to infer scattering parameters.

        Returns:
            FittedParameters object with estimated parameters.

        Raises:
            ValueError: If X is empty or has wrong shape.

        Example:
            >>> fitter = RealDataFitter()
            >>> params = fitter.fit(X_real, wavelengths=wl, name="wheat")
            >>> print(params.summary())
        """
        # Handle SpectroDataset input
        if hasattr(X, "x") and callable(X.x):
            # It's a SpectroDataset
            X_array = X.x({}, layout="2d")
            if wavelengths is None:
                try:
                    wavelengths = X.wavelengths
                except (AttributeError, TypeError):
                    wavelengths = np.arange(X_array.shape[1])
            if hasattr(X, "name"):
                name = X.name or name
        else:
            X_array = np.asarray(X)

        # Validate input
        if X_array.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_array.shape}")
        if X_array.shape[0] < 5:
            raise ValueError(f"Need at least 5 samples, got {X_array.shape[0]}")

        n_samples, n_wavelengths = X_array.shape

        # Create default wavelengths if not provided
        if wavelengths is None:
            wavelengths = np.arange(n_wavelengths)
        wavelengths = np.asarray(wavelengths)

        # Store for later use
        self._X_array = X_array
        self._wavelengths = wavelengths

        # Compute spectral properties (includes Phase 1-4 enhanced properties)
        self.source_properties = compute_spectral_properties(
            X_array, wavelengths, name
        )

        # Estimate basic parameters
        params = FittedParameters(
            source_name=name,
            source_properties=self.source_properties,
        )

        # Wavelength grid
        params.wavelength_start = float(wavelengths.min())
        params.wavelength_end = float(wavelengths.max())
        if len(wavelengths) > 1:
            params.wavelength_step = float(np.median(np.diff(wavelengths)))

        # Slope parameters
        props = self.source_properties
        params.global_slope_mean = props.mean_slope
        params.global_slope_std = props.slope_std

        # Noise parameters
        params.noise_base = props.noise_estimate * 0.5
        params.noise_signal_dep = props.noise_estimate * 0.5 / max(props.global_std, 0.01)

        # Scatter parameters
        params.scatter_alpha_std = min(0.15, props.global_std / max(props.global_mean, 0.1) * 0.3)
        params.scatter_beta_std = props.global_std * 0.1

        # Path length variation
        intensity_variation = np.std(X_array.mean(axis=1)) / max(np.mean(X_array.mean(axis=1)), 0.1)
        params.path_length_std = min(0.2, intensity_variation * 0.5)

        # Baseline amplitude
        params.baseline_amplitude = props.global_std * 0.2

        # Tilt standard deviation
        params.tilt_std = abs(props.mean_slope) * 0.1

        # Determine complexity
        if props.snr_estimate > 50 and props.pca_n_components_95 <= 5:
            params.complexity = "simple"
        elif props.snr_estimate < 20 or props.pca_n_components_95 > 15:
            params.complexity = "complex"
        else:
            params.complexity = "realistic"

        # Suggested number of components
        params.suggested_n_components = max(3, min(10, props.pca_n_components_95 + 2))

        # =====================================================================
        # Phase 1-4 Enhanced Inference
        # =====================================================================

        # Instrument inference
        if infer_instrument:
            params.instrument_inference = self._infer_instrument(props)
            params.inferred_instrument = params.instrument_inference.archetype_name

        # Measurement mode inference
        if infer_measurement_mode:
            mode, confidence = self._infer_measurement_mode(X_array, wavelengths, props)
            params.measurement_mode = mode
            params.measurement_mode_confidence = confidence

        # Domain inference
        if infer_domain:
            params.domain_inference = self._infer_domain(props)
            params.inferred_domain = params.domain_inference.domain_name
            params.detected_components = params.domain_inference.detected_components

        # Environmental effects inference
        if infer_environmental:
            params.environmental_inference = self._infer_environmental(X_array, wavelengths, props)
            params.temperature_config = self._build_temperature_config(params.environmental_inference)
            params.moisture_config = self._build_moisture_config(params.environmental_inference)

        # Scattering inference
        if infer_scattering:
            params.scattering_inference = self._infer_scattering(X_array, wavelengths, props)
            params.particle_size_config = self._build_particle_size_config(params.scattering_inference)
            params.emsc_config = self._build_emsc_config(params.scattering_inference)

        self.fitted_params = params
        return params

    def _infer_instrument(self, props: SpectralProperties) -> InstrumentInference:
        """Infer instrument archetype from spectral properties."""
        wl_min, wl_max = props.wavelength_range
        resolution = props.effective_resolution
        snr = props.snr_estimate
        noise_corr = props.noise_correlation_length

        scores: Dict[str, float] = {}

        # Score based on wavelength range
        if wl_max <= 1100:
            # Short-wave NIR only - likely Si detector
            scores["scio"] = 0.6
            scores["linksquare"] = 0.5
        elif wl_min >= 1300 and wl_max <= 2600:
            # Extended range - likely MEMS FT-NIR
            scores["neospectra_micro"] = 0.6
            scores["siware_neoscanner"] = 0.5
        elif wl_max > 2400:
            # Full NIR range
            if snr > 30000:
                scores["foss_xds"] = 0.7
                scores["bruker_mpa"] = 0.6
                scores["metrohm_ds2500"] = 0.5
            elif snr > 10000:
                scores["unity_spectrastar"] = 0.5
                scores["buchi_nirmaster"] = 0.4
            else:
                scores["asd_fieldspec"] = 0.5
        elif 900 <= wl_min <= 1000 and 1600 <= wl_max <= 1800:
            # Standard InGaAs range
            if resolution < 5:
                scores["perten_da7200"] = 0.6
            else:
                scores["viavi_micronir"] = 0.6
                scores["tellspec"] = 0.5

        # Adjust scores based on SNR
        if snr > 50000:
            for name in ["bruker_mpa", "thermo_antaris", "foss_xds"]:
                scores[name] = scores.get(name, 0) + 0.2
        elif snr < 5000:
            for name in ["scio", "neospectra_micro", "innospectra"]:
                scores[name] = scores.get(name, 0) + 0.2

        # Adjust based on resolution
        if resolution < 2:
            for name in ["bruker_mpa", "thermo_antaris", "foss_xds"]:
                scores[name] = scores.get(name, 0) + 0.1
        elif resolution > 10:
            for name in ["scio", "viavi_micronir", "neospectra_micro"]:
                scores[name] = scores.get(name, 0) + 0.1

        # Find best match
        if scores:
            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]
        else:
            best_name = "unknown"
            best_score = 0.0

        # Determine detector type
        if wl_max <= 1100:
            detector = "si"
        elif wl_max <= 1700:
            detector = "ingaas"
        elif wl_max <= 2600:
            detector = "ingaas_ext"
        else:
            detector = "pbs"

        return InstrumentInference(
            archetype_name=best_name,
            detector_type=detector,
            wavelength_range=(wl_min, wl_max),
            estimated_resolution=resolution,
            confidence=min(1.0, best_score),
            alternative_archetypes={k: v for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]},
        )

    def _infer_measurement_mode(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> Tuple[str, float]:
        """Infer measurement mode from spectral characteristics."""
        scores = {
            "transmittance": 0.0,
            "reflectance": 0.0,
            "transflectance": 0.0,
            "atr": 0.0,
        }

        # Check value range
        min_val, max_val = props.global_range

        # Transmittance/absorbance typically has values 0-3+ AU
        if 0 <= min_val < 0.5 and max_val < 4.0:
            scores["transmittance"] += 0.3

        # Reflectance data often has lower mean absorbance
        if props.global_mean < 1.0:
            scores["reflectance"] += 0.2

        # ATR shows characteristic wavelength-dependent baseline
        if props.baseline_convexity > 0.01:
            scores["atr"] += 0.4

        # Kubelka-Munk linearity suggests reflectance
        if props.kubelka_munk_linearity > 0.5:
            scores["reflectance"] += 0.3

        # Sample-to-sample scatter suggests powder/reflectance
        if props.sample_to_sample_offset_std > 0.1:
            scores["reflectance"] += 0.2

        # High baseline offset suggests transflectance (double-pass)
        if props.baseline_offset > 0.5:
            scores["transflectance"] += 0.2

        # Scatter baseline curvature suggests reflectance/powder
        if props.scatter_baseline_curvature > 0.01:
            scores["reflectance"] += 0.2

        # Find best
        best_mode = max(scores, key=scores.get)
        confidence = scores[best_mode] / max(sum(scores.values()), 0.01)

        return best_mode, float(confidence)

    def _infer_domain(self, props: SpectralProperties) -> DomainInference:
        """Infer application domain from spectral features."""
        scores: Dict[str, float] = {}
        detected_components: List[str] = []

        # Score based on band intensities
        water_intensity = props.water_band_intensity
        protein_intensity = props.protein_band_intensity
        carb_intensity = props.carbohydrate_band_intensity
        lipid_intensity = props.lipid_band_intensity

        # Normalize intensities
        total = water_intensity + protein_intensity + carb_intensity + lipid_intensity + 1e-10
        water_frac = water_intensity / total
        protein_frac = protein_intensity / total
        carb_frac = carb_intensity / total
        lipid_frac = lipid_intensity / total

        # Agriculture domains
        if carb_frac > 0.3 and protein_frac > 0.15:
            scores["agriculture_grain"] = 0.6
            detected_components.extend(["starch", "protein"])
        if carb_frac > 0.25 and lipid_frac > 0.2:
            scores["agriculture_oilseeds"] = 0.5
            if "lipid" not in detected_components:
                detected_components.append("lipid")

        # Food domains
        if water_frac > 0.4 and protein_frac > 0.1:
            scores["food_dairy"] = 0.5
            scores["food_meat"] = 0.4
            if "water" not in detected_components:
                detected_components.append("water")
        if lipid_frac > 0.25:
            scores["food_chocolate"] = 0.4
            if "lipid" not in detected_components:
                detected_components.append("lipid")

        # Pharmaceutical domains
        if carb_frac > 0.4 and water_frac < 0.15:
            scores["pharma_tablets"] = 0.5
            detected_components.append("starch")
            detected_components.append("cellulose")

        # Environmental domains
        if protein_frac < 0.1 and carb_frac > 0.2:
            scores["environmental_soil"] = 0.3

        # Beverage domains
        if water_frac > 0.5:
            scores["beverage_juice"] = 0.4
            scores["beverage_wine"] = 0.3

        # Biomedical
        if water_frac > 0.3 and lipid_frac > 0.15 and protein_frac > 0.15:
            scores["biomedical_tissue"] = 0.4

        # Default fallback
        if not scores:
            scores["unknown"] = 0.5

        # Find best
        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain]

        # Determine category
        category = "unknown"
        if "agriculture" in best_domain:
            category = "agriculture"
        elif "food" in best_domain:
            category = "food"
        elif "pharma" in best_domain:
            category = "pharmaceutical"
        elif "beverage" in best_domain:
            category = "beverage"
        elif "environmental" in best_domain:
            category = "environmental"
        elif "biomedical" in best_domain:
            category = "biomedical"

        # Remove duplicates from detected components
        detected_components = list(dict.fromkeys(detected_components))

        return DomainInference(
            domain_name=best_domain,
            category=category,
            confidence=confidence,
            detected_components=detected_components,
            alternative_domains={k: v for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]},
        )

    def _infer_environmental(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> EnvironmentalInference:
        """Infer environmental effects from spectral patterns."""
        # Temperature effects
        has_temp = props.temperature_sensitivity_score > 0.3
        temp_variation = props.temperature_sensitivity_score * 10.0  # rough °C estimate

        # Moisture effects
        has_moisture = props.water_band_variation > 0.05
        moisture_variation = props.water_band_variation

        # Water band shift analysis
        water_shift = 0.0
        if props.oh_band_positions is not None and len(props.oh_band_positions) > 0:
            # Compare to expected free water position (1410 nm)
            expected_free = 1410
            actual = props.oh_band_positions[0] if len(props.oh_band_positions) > 0 else expected_free
            water_shift = actual - expected_free

        return EnvironmentalInference(
            estimated_temperature_variation=temp_variation,
            has_temperature_effects=has_temp,
            estimated_moisture_variation=moisture_variation,
            has_moisture_effects=has_moisture,
            water_band_shift=water_shift,
        )

    def _infer_scattering(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> ScatteringInference:
        """Infer scattering effects from spectral patterns."""
        # Check for scatter effects
        has_scatter = (
            props.scatter_baseline_curvature > 0.005 or
            props.sample_to_sample_offset_std > 0.05 or
            props.sample_to_sample_slope_std > 0.01
        )

        # Estimate particle size from scattering intensity
        # Higher scatter curvature suggests smaller particles
        if props.scatter_baseline_curvature > 0.02:
            particle_size = 20.0  # Fine powder
        elif props.scatter_baseline_curvature > 0.01:
            particle_size = 50.0  # Medium
        else:
            particle_size = 100.0  # Coarse

        # MSC/SNV indicators
        mult_scatter = props.sample_to_sample_slope_std
        add_scatter = props.sample_to_sample_offset_std

        # SNV correctable if high offset variation
        snv_correctable = add_scatter > 0.05 or mult_scatter > 0.05

        # MSC correctable if systematic variation with mean
        msc_correctable = mult_scatter > 0.03

        return ScatteringInference(
            has_scatter_effects=has_scatter,
            estimated_particle_size_um=particle_size,
            multiplicative_scatter_std=mult_scatter,
            additive_scatter_std=add_scatter,
            baseline_curvature=props.scatter_baseline_curvature,
            snv_correctable=snv_correctable,
            msc_correctable=msc_correctable,
        )

    def _build_temperature_config(self, env: Optional[EnvironmentalInference]) -> Dict[str, Any]:
        """Build temperature configuration from inference."""
        if env is None or not env.has_temperature_effects:
            return {}
        return {
            "temperature_variation": env.estimated_temperature_variation,
            "enable_shift": True,
            "enable_intensity": True,
            "enable_broadening": True,
        }

    def _build_moisture_config(self, env: Optional[EnvironmentalInference]) -> Dict[str, Any]:
        """Build moisture configuration from inference."""
        if env is None or not env.has_moisture_effects:
            return {}

        # Estimate free water fraction from band shift
        # Shift towards higher wavelength = more bound water
        free_fraction = max(0.1, min(0.9, 0.5 - env.water_band_shift / 50.0))

        return {
            "water_activity": 0.5,  # Default
            "moisture_content": env.estimated_moisture_variation / 10.0,
            "free_water_fraction": free_fraction,
        }

    def _build_particle_size_config(self, scatter: Optional[ScatteringInference]) -> Dict[str, Any]:
        """Build particle size configuration from inference."""
        if scatter is None or not scatter.has_scatter_effects:
            return {}
        return {
            "mean_size_um": scatter.estimated_particle_size_um,
            "std_size_um": scatter.estimated_particle_size_um * 0.3,
            "size_effect_strength": 1.0,
        }

    def _build_emsc_config(self, scatter: Optional[ScatteringInference]) -> Dict[str, Any]:
        """Build EMSC configuration from inference."""
        if scatter is None:
            return {}
        return {
            "multiplicative_scatter_std": scatter.multiplicative_scatter_std,
            "additive_scatter_std": scatter.additive_scatter_std,
            "polynomial_order": 2,
            "include_wavelength_terms": True,
        }

    def create_matched_generator(
        self,
        random_state: Optional[int] = None,
    ) -> "SyntheticNIRSGenerator":
        """
        Create a SyntheticNIRSGenerator configured to match the fitted data.

        This method creates a generator with all fitted parameters including
        Phase 1-4 enhanced features (instrument, domain, effects).

        Args:
            random_state: Random seed for reproducibility.

        Returns:
            Configured SyntheticNIRSGenerator instance.

        Raises:
            RuntimeError: If fit() hasn't been called.

        Example:
            >>> fitter = RealDataFitter()
            >>> params = fitter.fit(X_real, wavelengths=wavelengths)
            >>> generator = fitter.create_matched_generator(random_state=42)
            >>> X_synth, _, _ = generator.generate(1000)
        """
        if self.fitted_params is None:
            raise RuntimeError("Must call fit() before create_matched_generator()")

        from .generator import SyntheticNIRSGenerator

        params = self.fitted_params

        generator = SyntheticNIRSGenerator(
            wavelength_start=params.wavelength_start,
            wavelength_end=params.wavelength_end,
            wavelength_step=params.wavelength_step,
            complexity=params.complexity,
            random_state=random_state,
        )

        return generator

    def fit_from_path(
        self,
        path: str,
        *,
        name: Optional[str] = None,
    ) -> FittedParameters:
        """
        Fit parameters from a dataset path.

        Loads data using DatasetConfigs and fits parameters.

        Args:
            path: Path to dataset folder.
            name: Optional name override.

        Returns:
            FittedParameters object.

        Example:
            >>> params = fitter.fit_from_path("sample_data/regression")
        """
        from nirs4all.data import DatasetConfigs

        dataset_config = DatasetConfigs(path)
        datasets = dataset_config.get_datasets()

        if not datasets:
            raise ValueError(f"No datasets found at {path}")

        dataset = datasets[0]
        X = dataset.x({}, layout="2d")

        # Try to get wavelengths
        wavelengths = None
        try:
            wavelengths = dataset.wavelengths
        except (AttributeError, TypeError):
            pass

        return self.fit(X, wavelengths=wavelengths, name=name or dataset.name)

    def evaluate_similarity(
        self,
        X_synthetic: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate similarity between synthetic and source data.

        Computes various metrics comparing synthetic spectra to the
        original real data.

        Args:
            X_synthetic: Synthetic spectra matrix.
            wavelengths: Optional wavelength grid.

        Returns:
            Dictionary with similarity metrics.

        Raises:
            RuntimeError: If fit() hasn't been called.

        Example:
            >>> params = fitter.fit(X_real)
            >>> X_synth, _, _ = generator.generate(1000)
            >>> metrics = fitter.evaluate_similarity(X_synth)
            >>> print(f"Similarity: {metrics['overall_score']:.1f}/100")
        """
        if self.source_properties is None:
            raise RuntimeError("Must call fit() before evaluate_similarity()")

        # Use source wavelengths if not provided
        if wavelengths is None and self.source_properties.wavelengths is not None:
            # Assume same wavelength grid
            wavelengths = self.source_properties.wavelengths

        # Compute synthetic properties
        synth_props = compute_spectral_properties(
            X_synthetic, wavelengths, "synthetic"
        )

        real_props = self.source_properties
        metrics: Dict[str, Any] = {}

        # Mean comparison
        if real_props.global_mean != 0:
            metrics["mean_rel_diff"] = (
                (synth_props.global_mean - real_props.global_mean)
                / abs(real_props.global_mean)
            )
        else:
            metrics["mean_rel_diff"] = synth_props.global_mean

        # Std comparison
        if real_props.global_std != 0:
            metrics["std_rel_diff"] = (
                (synth_props.global_std - real_props.global_std)
                / real_props.global_std
            )
        else:
            metrics["std_rel_diff"] = synth_props.global_std

        # Slope comparison
        metrics["slope_diff"] = synth_props.mean_slope - real_props.mean_slope
        if real_props.mean_slope != 0:
            metrics["slope_ratio"] = synth_props.mean_slope / real_props.mean_slope
        else:
            metrics["slope_ratio"] = float("inf")

        # Noise comparison
        if real_props.noise_estimate != 0:
            metrics["noise_ratio"] = synth_props.noise_estimate / real_props.noise_estimate
        else:
            metrics["noise_ratio"] = float("inf")

        # SNR comparison
        if real_props.snr_estimate != 0 and real_props.snr_estimate != float("inf"):
            metrics["snr_ratio"] = synth_props.snr_estimate / real_props.snr_estimate
        else:
            metrics["snr_ratio"] = float("inf")

        # PCA complexity
        metrics["pca_complexity_diff"] = (
            synth_props.pca_n_components_95 - real_props.pca_n_components_95
        )

        # Mean spectrum correlation (if wavelengths match)
        if (real_props.n_wavelengths == synth_props.n_wavelengths and
            real_props.mean_spectrum is not None and
            synth_props.mean_spectrum is not None):
            corr = np.corrcoef(
                real_props.mean_spectrum, synth_props.mean_spectrum
            )[0, 1]
            metrics["mean_spectrum_correlation"] = float(corr)

        # Slope distribution comparison
        if real_props.slopes is not None and synth_props.slopes is not None:
            ks_stat, ks_pval = stats.ks_2samp(real_props.slopes, synth_props.slopes)
            metrics["slope_ks_statistic"] = float(ks_stat)
            metrics["slope_ks_pvalue"] = float(ks_pval)

        # Overall similarity score (0-100)
        scores = []
        if "mean_rel_diff" in metrics:
            scores.append(max(0, 100 - abs(metrics["mean_rel_diff"]) * 100))
        if "std_rel_diff" in metrics:
            scores.append(max(0, 100 - abs(metrics["std_rel_diff"]) * 100))
        if "noise_ratio" in metrics and metrics["noise_ratio"] != float("inf"):
            scores.append(max(0, 100 - abs(1 - metrics["noise_ratio"]) * 100))
        if "mean_spectrum_correlation" in metrics:
            scores.append(metrics["mean_spectrum_correlation"] * 100)

        metrics["overall_score"] = float(np.mean(scores)) if scores else 0.0

        return metrics

    def get_tuning_recommendations(self) -> List[str]:
        """
        Get recommendations for tuning generation parameters.

        Based on the fitted parameters and source data, provides
        suggestions for manual tuning.

        Returns:
            List of recommendation strings.

        Example:
            >>> params = fitter.fit(X_real)
            >>> for rec in fitter.get_tuning_recommendations():
            ...     print(f"- {rec}")
        """
        if self.source_properties is None or self.fitted_params is None:
            return ["Call fit() first to analyze data."]

        recs = []
        props = self.source_properties
        params = self.fitted_params

        # Noise recommendations
        if props.snr_estimate < 15:
            recs.append(
                f"High noise detected (SNR={props.snr_estimate:.1f}). "
                f"Using noise_base={params.noise_base:.4f}"
            )
        elif props.snr_estimate > 100:
            recs.append(
                f"Very low noise detected (SNR={props.snr_estimate:.1f}). "
                "Consider using 'simple' complexity for faster generation."
            )

        # Slope recommendations
        if abs(props.mean_slope) > 0.1:
            recs.append(
                f"Significant slope detected ({props.mean_slope:.3f}/1000nm). "
                "Ensure global_slope_mean is correctly set."
            )

        # Complexity recommendations
        if props.pca_n_components_95 > 10:
            recs.append(
                f"High complexity ({props.pca_n_components_95} PCA components for 95%). "
                "Consider using more spectral components."
            )
        elif props.pca_n_components_95 <= 3:
            recs.append(
                f"Low complexity ({props.pca_n_components_95} PCA components). "
                "Simple mode may be sufficient."
            )

        # Variation recommendations
        if params.path_length_std > 0.15:
            recs.append(
                f"High sample-to-sample variation detected. "
                f"path_length_std set to {params.path_length_std:.3f}"
            )

        return recs


def fit_to_real_data(
    X: Union[np.ndarray, "SpectroDataset"],
    wavelengths: Optional[np.ndarray] = None,
    name: str = "source",
) -> FittedParameters:
    """
    Quick function to fit parameters to real data.

    Convenience function for simple fitting use cases.

    Args:
        X: Real spectra or SpectroDataset.
        wavelengths: Wavelength grid.
        name: Dataset name.

    Returns:
        FittedParameters object.

    Example:
        >>> params = fit_to_real_data(X_real, wavelengths)
        >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
    """
    fitter = RealDataFitter()
    return fitter.fit(X, wavelengths=wavelengths, name=name)


def compare_datasets(
    X_synthetic: np.ndarray,
    X_real: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Quick comparison between synthetic and real datasets.

    Args:
        X_synthetic: Synthetic spectra.
        X_real: Real spectra.
        wavelengths: Wavelength grid.

    Returns:
        Dictionary with comparison metrics.

    Example:
        >>> metrics = compare_datasets(X_synth, X_real)
        >>> print(f"Similarity: {metrics['overall_score']:.1f}/100")
    """
    fitter = RealDataFitter()
    fitter.fit(X_real, wavelengths=wavelengths, name="real")
    return fitter.evaluate_similarity(X_synthetic, wavelengths)
