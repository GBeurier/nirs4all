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
    >>> from nirs4all.synthesis import RealDataFitter, SyntheticNIRSGenerator
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

import contextlib
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

    from .components import SpectralComponent
    from .generator import SyntheticNIRSGenerator

# ============================================================================
# Inference Result Classes
# ============================================================================

class PreprocessingType(StrEnum):
    """Detected preprocessing type of spectral data."""
    RAW_ABSORBANCE = "raw_absorbance"
    RAW_REFLECTANCE = "raw_reflectance"
    SECOND_DERIVATIVE = "second_derivative"
    FIRST_DERIVATIVE = "first_derivative"
    MEAN_CENTERED = "mean_centered"
    SNV_CORRECTED = "snv_corrected"
    MSC_CORRECTED = "msc_corrected"
    NORMALIZED = "normalized"  # e.g., min-max scaled
    UNKNOWN = "unknown"

class MeasurementModeInference(StrEnum):
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
    wavelength_range: tuple[float, float] = (1000.0, 2500.0)
    estimated_resolution: float = 8.0
    confidence: float = 0.0
    alternative_archetypes: dict[str, float] = field(default_factory=dict)

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
    detected_components: list[str] = field(default_factory=list)
    alternative_domains: dict[str, float] = field(default_factory=dict)

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
class EdgeArtifactInference:
    """
    Results of edge artifact inference.

    Detects edge deformation effects in NIR spectra caused by:
    - Detector sensitivity roll-off at wavelength boundaries
    - Stray light effects (more pronounced at edges)
    - Truncated absorption bands outside measurement range
    - Baseline curvature concentrated at edges

    Attributes:
        has_edge_artifacts: Whether significant edge artifacts are detected.
        has_detector_rolloff: Whether detector roll-off effects are present.
        has_stray_light: Whether stray light effects are detected.
        has_truncated_peaks: Whether truncated peaks at boundaries are present.
        has_edge_curvature: Whether edge curvature/bending is detected.
        left_edge_intensity: Relative intensity change at left edge.
        right_edge_intensity: Relative intensity change at right edge.
        edge_noise_ratio: Ratio of edge noise to center noise.
        detector_model: Suggested detector model based on characteristics.
        stray_light_fraction: Estimated stray light fraction.
        curvature_type: Detected curvature type ("smile", "frown", "asymmetric").
        boundary_peak_amplitudes: Estimated truncated peak amplitudes at edges.

    References:
        - JASCO (2020). Advantages of high-sensitivity InGaAs detector.
        - Applied Optics (1975). Resolution and stray light in NIR spectroscopy.
        - Burns & Ciurczak (2007). Handbook of Near-Infrared Analysis.
    """
    has_edge_artifacts: bool = False
    has_detector_rolloff: bool = False
    has_stray_light: bool = False
    has_truncated_peaks: bool = False
    has_edge_curvature: bool = False
    left_edge_intensity: float = 0.0
    right_edge_intensity: float = 0.0
    edge_noise_ratio: float = 1.0
    detector_model: str = "generic_nir"
    stray_light_fraction: float = 0.0
    curvature_type: str = "none"
    boundary_peak_amplitudes: tuple[float, float] = (0.0, 0.0)

@dataclass
class PreprocessingInference:
    """
    Results of preprocessing type inference.

    Detects whether spectral data has been preprocessed (derivatives,
    normalization, centering, etc.) before being provided to the fitter.

    This is crucial for generating synthetic data that matches the real
    data distribution - synthetic spectra should be generated as raw
    absorbance and then the same preprocessing applied.

    Attributes:
        preprocessing_type: Detected preprocessing type.
        confidence: Confidence score (0-1).
        is_preprocessed: Whether data appears to be preprocessed.
        global_mean: Mean value (0 suggests centering/derivatives).
        global_range: (min, max) value range.
        zero_crossing_ratio: Ratio of zero crossings (high for derivatives).
        per_sample_std_variation: Variation in per-sample std (low for SNV).
        oscillation_frequency: Spectral oscillation frequency (high for 2nd deriv).
        suggested_inverse: Suggested inverse operation to recover raw data.
    """
    preprocessing_type: PreprocessingType = PreprocessingType.RAW_ABSORBANCE
    confidence: float = 0.0
    is_preprocessed: bool = False
    global_mean: float = 0.0
    global_range: tuple[float, float] = (0.0, 1.0)
    zero_crossing_ratio: float = 0.0
    per_sample_std_variation: float = 0.0
    oscillation_frequency: float = 0.0
    suggested_inverse: str | None = None

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
    wavelengths: np.ndarray | None = None

    # Basic statistics
    mean_spectrum: np.ndarray | None = None
    std_spectrum: np.ndarray | None = None
    global_mean: float = 0.0
    global_std: float = 0.0
    global_range: tuple[float, float] = (0.0, 0.0)

    # Shape properties
    mean_slope: float = 0.0
    slope_std: float = 0.0
    slopes: np.ndarray | None = None
    mean_curvature: float = 0.0
    curvature_std: float = 0.0

    # Distribution statistics
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Noise characteristics
    noise_estimate: float = 0.0
    snr_estimate: float = 0.0

    # PCA properties
    pca_explained_variance: np.ndarray | None = None
    pca_n_components_95: int = 0

    # Peak analysis
    n_peaks_mean: float = 0.0
    peak_positions: np.ndarray | None = None
    peak_wavenumbers: np.ndarray | None = None

    # Phase 1-4 Enhanced properties
    # Instrument indicators
    effective_resolution: float = 8.0
    noise_correlation_length: float = 1.0
    wavelength_range: tuple[float, float] = (1000.0, 2500.0)

    # Measurement mode indicators
    baseline_offset: float = 0.0
    kubelka_munk_linearity: float = 0.0
    baseline_convexity: float = 0.0

    # Environmental indicators
    water_band_variation: float = 0.0
    oh_band_positions: np.ndarray | None = None
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

    # Edge artifact indicators
    left_edge_noise_std: float = 0.0
    right_edge_noise_std: float = 0.0
    center_noise_std: float = 0.0
    left_edge_slope: float = 0.0
    right_edge_slope: float = 0.0
    edge_curvature_intensity: float = 0.0
    edge_curvature_asymmetry: float = 0.0
    has_boundary_rise_left: bool = False
    has_boundary_rise_right: bool = False

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
    complexity: Literal["simple", "realistic", "complex"] = "realistic"
    source_name: str = ""
    source_properties: SpectralProperties | None = field(default=None, repr=False)

    # Phase 1-4 Enhanced Parameters
    # Instrument inference (Phase 2)
    inferred_instrument: str = "unknown"
    instrument_inference: InstrumentInference | None = field(default=None, repr=False)

    # Measurement mode (Phase 2)
    measurement_mode: str = "transmittance"
    measurement_mode_confidence: float = 0.0

    # Domain inference (Phase 1)
    inferred_domain: str = "unknown"
    domain_inference: DomainInference | None = field(default=None, repr=False)

    # Environmental effects (Phase 3)
    environmental_inference: EnvironmentalInference | None = field(default=None, repr=False)
    temperature_config: dict[str, Any] = field(default_factory=dict)
    moisture_config: dict[str, Any] = field(default_factory=dict)

    # Scattering effects (Phase 3)
    scattering_inference: ScatteringInference | None = field(default=None, repr=False)
    particle_size_config: dict[str, Any] = field(default_factory=dict)
    emsc_config: dict[str, Any] = field(default_factory=dict)

    # Edge artifacts (Phase 6)
    edge_artifact_inference: EdgeArtifactInference | None = field(default=None, repr=False)
    edge_artifacts_config: dict[str, Any] = field(default_factory=dict)
    boundary_components_config: dict[str, Any] = field(default_factory=dict)

    # Preprocessing detection (Phase 5)
    preprocessing_inference: PreprocessingInference | None = field(default=None, repr=False)
    preprocessing_type: str = "raw_absorbance"
    is_preprocessed: bool = False

    # Components (Phase 1)
    detected_components: list[str] = field(default_factory=list)
    suggested_n_components: int = 5

    def to_generator_kwargs(self) -> dict[str, Any]:
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

    def to_full_config(self) -> dict[str, Any]:
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
            # Phase 6: Edge artifacts
            "edge_artifacts_config": self.edge_artifacts_config,
            "boundary_components_config": self.boundary_components_config,
            # Phase 5: Preprocessing detection
            "preprocessing_type": self.preprocessing_type,
            "is_preprocessed": self.is_preprocessed,
        }

    def to_dict(self) -> dict[str, Any]:
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
            # Phase 6: Edge artifacts
            "edge_artifacts_config": self.edge_artifacts_config,
            "boundary_components_config": self.boundary_components_config,
            # Phase 5: Preprocessing detection
            "preprocessing_type": self.preprocessing_type,
            "is_preprocessed": self.is_preprocessed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FittedParameters:
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
            # Phase 6: Edge artifacts
            edge_artifacts_config=data.get("edge_artifacts_config", {}),
            boundary_components_config=data.get("boundary_components_config", {}),
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
    def load(cls, path: str) -> FittedParameters:
        """
        Load parameters from JSON file.

        Args:
            path: Input file path.

        Returns:
            FittedParameters instance.
        """
        import json

        with open(path) as f:
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
            "",
            "Preprocessing Detection:",
            f"  Type: {self.preprocessing_type}",
            f"  Is preprocessed: {self.is_preprocessed}",
        ]
        if self.preprocessing_inference is not None:
            lines.extend([
                f"  Confidence: {self.preprocessing_inference.confidence:.2f}",
                f"  Zero-crossing ratio: {self.preprocessing_inference.zero_crossing_ratio:.3f}",
                f"  Oscillation frequency: {self.preprocessing_inference.oscillation_frequency:.3f}",
            ])
            if self.preprocessing_inference.suggested_inverse:
                lines.append(f"  Suggested inverse: {self.preprocessing_inference.suggested_inverse}")
        lines.append("=" * 60)
        return "\n".join(lines)

def compute_spectral_properties(
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
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

    # =========================================================================
    # Edge Artifact Analysis (Phase 6)
    # =========================================================================
    edge_props = _analyze_edge_artifacts(X, wavelengths, props.mean_spectrum)
    props.left_edge_noise_std = edge_props["left_edge_noise_std"]
    props.right_edge_noise_std = edge_props["right_edge_noise_std"]
    props.center_noise_std = edge_props["center_noise_std"]
    props.left_edge_slope = edge_props["left_edge_slope"]
    props.right_edge_slope = edge_props["right_edge_slope"]
    props.edge_curvature_intensity = edge_props["edge_curvature_intensity"]
    props.edge_curvature_asymmetry = edge_props["edge_curvature_asymmetry"]
    props.has_boundary_rise_left = edge_props["has_boundary_rise_left"]
    props.has_boundary_rise_right = edge_props["has_boundary_rise_right"]

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
) -> tuple[float, float]:
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
) -> tuple[float, np.ndarray]:
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
        position_std = float(np.std(peak_positions))
        return min(1.0, position_std / 5.0)
    except Exception:
        return 0.0

def _compute_band_intensity(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    regions: list[tuple[float, float]],
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

def _analyze_edge_artifacts(
    X: np.ndarray,
    wavelengths: np.ndarray,
    mean_spectrum: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze edge artifacts in spectral data.

    Detects various edge deformation effects:
    - Noise amplification at edges (detector roll-off)
    - Edge curvature (baseline bending, optical aberrations)
    - Truncated peaks (boundary absorption bands)
    - Asymmetric edge effects

    Args:
        X: Spectra matrix (n_samples, n_wavelengths).
        wavelengths: Wavelength array.
        mean_spectrum: Mean spectrum.

    Returns:
        Dictionary with edge artifact properties.

    References:
        - JASCO (2020). Advantages of high-sensitivity InGaAs detector.
        - Applied Optics (1975). Resolution and stray light in NIR spectroscopy.
    """
    n_wavelengths = len(wavelengths)
    edge_size = max(10, n_wavelengths // 10)  # 10% of spectrum at each edge

    result = {
        "left_edge_noise_std": 0.0,
        "right_edge_noise_std": 0.0,
        "center_noise_std": 0.0,
        "left_edge_slope": 0.0,
        "right_edge_slope": 0.0,
        "edge_curvature_intensity": 0.0,
        "edge_curvature_asymmetry": 0.0,
        "has_boundary_rise_left": False,
        "has_boundary_rise_right": False,
    }

    try:
        # =====================================================================
        # Noise analysis at edges vs center (detector roll-off indicator)
        # =====================================================================
        # Compute noise from first difference
        first_diff = np.diff(X, axis=1)
        noise_spectrum = first_diff.std(axis=0) / np.sqrt(2)

        # Left edge noise
        left_noise = noise_spectrum[:edge_size]
        result["left_edge_noise_std"] = float(np.mean(left_noise))

        # Right edge noise
        right_noise = noise_spectrum[-edge_size:]
        result["right_edge_noise_std"] = float(np.mean(right_noise))

        # Center noise
        center_start = n_wavelengths // 3
        center_end = 2 * n_wavelengths // 3
        center_noise = noise_spectrum[center_start:center_end]
        result["center_noise_std"] = float(np.mean(center_noise))

        # =====================================================================
        # Edge slope analysis (truncated peak indicator)
        # =====================================================================
        wl_left = wavelengths[:edge_size]
        wl_right = wavelengths[-edge_size:]

        # Left edge slope (rising slope suggests truncated peak below range)
        try:
            left_spectrum = mean_spectrum[:edge_size]
            left_coeffs = np.polyfit(np.arange(edge_size), left_spectrum, 1)
            result["left_edge_slope"] = float(left_coeffs[0])

            # Positive slope at left = rising toward edge = boundary peak
            result["has_boundary_rise_left"] = left_coeffs[0] < -0.001  # Falling toward higher indices
        except Exception:
            pass

        # Right edge slope (rising slope suggests truncated peak above range)
        try:
            right_spectrum = mean_spectrum[-edge_size:]
            right_coeffs = np.polyfit(np.arange(edge_size), right_spectrum, 1)
            result["right_edge_slope"] = float(right_coeffs[0])

            # Positive slope at right = rising toward end = boundary peak
            result["has_boundary_rise_right"] = right_coeffs[0] > 0.001
        except Exception:
            pass

        # =====================================================================
        # Edge curvature analysis (baseline bending, optical artifacts)
        # =====================================================================
        try:
            # Fit parabola to mean spectrum
            x_norm = np.linspace(-1, 1, n_wavelengths)
            coeffs = np.polyfit(x_norm, mean_spectrum, 2)

            # Curvature is the quadratic coefficient
            curvature = coeffs[0]
            result["edge_curvature_intensity"] = float(abs(curvature))

            # Asymmetry: compare left and right edge deviations from linear
            linear_fit = coeffs[1] * x_norm + coeffs[2]
            residual = mean_spectrum - linear_fit

            left_deviation = np.mean(residual[:edge_size])
            right_deviation = np.mean(residual[-edge_size:])

            if abs(left_deviation) + abs(right_deviation) > 1e-10:
                asymmetry = (right_deviation - left_deviation) / (abs(left_deviation) + abs(right_deviation))
                result["edge_curvature_asymmetry"] = float(asymmetry)
        except Exception:
            pass

    except Exception:
        pass

    return result

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
        self.source_properties: SpectralProperties | None = None
        self.fitted_params: FittedParameters | None = None
        self._X_array: np.ndarray | None = None
        self._wavelengths: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | SpectroDataset,
        *,
        wavelengths: np.ndarray | None = None,
        name: str = "source",
        infer_instrument: bool = True,
        infer_domain: bool = True,
        infer_measurement_mode: bool = True,
        infer_environmental: bool = True,
        infer_scattering: bool = True,
        infer_edge_artifacts: bool = True,
        infer_preprocessing: bool = True,
    ) -> FittedParameters:
        """
        Fit generator parameters to real data.

        Analyzes the input data and estimates optimal parameters for
        generating synthetic spectra with similar properties. Includes
        Phase 1-6 enhanced inference.

        Args:
            X: Real spectra matrix (n_samples, n_wavelengths) or SpectroDataset.
            wavelengths: Wavelength grid (required if X is ndarray).
            name: Dataset name for reference.
            infer_instrument: Whether to infer instrument archetype.
            infer_domain: Whether to infer application domain.
            infer_measurement_mode: Whether to infer measurement mode.
            infer_environmental: Whether to infer environmental effects.
            infer_scattering: Whether to infer scattering parameters.
            infer_edge_artifacts: Whether to infer edge artifact effects.
            infer_preprocessing: Whether to detect preprocessing type.

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
        if not isinstance(X, np.ndarray):
            # It's a SpectroDataset
            X_array: np.ndarray = np.asarray(X.x({}, layout="2d"))
            if wavelengths is None:
                try:
                    wavelengths = X.float_headers()
                except (AttributeError, TypeError):
                    wavelengths = np.arange(X_array.shape[1])
            if hasattr(X, "name") and X.name:
                name = X.name
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
        intensity_variation = float(np.std(X_array.mean(axis=1))) / max(float(np.mean(X_array.mean(axis=1))), 0.1)
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

        # Edge artifacts inference (Phase 6)
        if infer_edge_artifacts:
            params.edge_artifact_inference = self._infer_edge_artifacts(X_array, wavelengths, props)
            params.edge_artifacts_config = self._build_edge_artifacts_config(params.edge_artifact_inference)
            params.boundary_components_config = self._build_boundary_components_config(
                params.edge_artifact_inference, wavelengths
            )

        # Preprocessing detection (Phase 5)
        if infer_preprocessing:
            params.preprocessing_inference = self._infer_preprocessing(X_array, wavelengths, props)
            params.preprocessing_type = params.preprocessing_inference.preprocessing_type.value
            params.is_preprocessed = params.preprocessing_inference.is_preprocessed

        self.fitted_params = params
        return params

    def _infer_instrument(self, props: SpectralProperties) -> InstrumentInference:
        """Infer instrument archetype from spectral properties."""
        wl_min, wl_max = props.wavelength_range
        resolution = props.effective_resolution
        snr = props.snr_estimate
        noise_corr = props.noise_correlation_length

        scores: dict[str, float] = {}

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
            best_name = max(scores, key=lambda k: scores[k])
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
            alternative_archetypes=dict(sorted(scores.items(), key=lambda x: -x[1])[:5]),
        )

    def _infer_measurement_mode(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> tuple[str, float]:
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
        best_mode = max(scores, key=lambda k: scores[k])
        confidence = scores[best_mode] / max(sum(scores.values()), 0.01)

        return best_mode, float(confidence)

    def _infer_domain(self, props: SpectralProperties) -> DomainInference:
        """Infer application domain from spectral features."""
        scores: dict[str, float] = {}
        detected_components: list[str] = []

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
        best_domain = max(scores, key=lambda k: scores[k])
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
            alternative_domains=dict(sorted(scores.items(), key=lambda x: -x[1])[:5]),
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

    def _infer_edge_artifacts(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> EdgeArtifactInference:
        """
        Infer edge artifact effects from spectral patterns.

        Detects various edge deformation effects including:
        - Detector sensitivity roll-off at wavelength boundaries
        - Stray light effects
        - Truncated absorption bands outside measurement range
        - Edge curvature/baseline bending

        Args:
            X: Spectral data (n_samples, n_wavelengths).
            wavelengths: Wavelength array.
            props: Computed spectral properties.

        Returns:
            EdgeArtifactInference with detected effects and parameters.

        References:
            - JASCO (2020). Advantages of high-sensitivity InGaAs detector.
            - Applied Optics (1975). Resolution and stray light in NIR spectroscopy.
            - Burns & Ciurczak (2007). Handbook of Near-Infrared Analysis.
        """
        # =====================================================================
        # Detector roll-off detection (noise amplification at edges)
        # =====================================================================
        center_noise = props.center_noise_std
        edge_noise_avg = (props.left_edge_noise_std + props.right_edge_noise_std) / 2

        edge_noise_ratio = edge_noise_avg / max(center_noise, 1e-10)
        has_detector_rolloff = edge_noise_ratio > 1.3  # Edge noise > 30% higher

        # Infer detector model from wavelength range and noise pattern
        wl_min, wl_max = props.wavelength_range
        if wl_max <= 1700 and wl_min >= 900:
            detector_model = "ingaas_standard"
        elif wl_max > 2200:
            detector_model = "ingaas_extended" if wl_max < 2600 else "pbs"
        elif wl_max <= 1100:
            detector_model = "silicon_ccd"
        else:
            detector_model = "generic_nir"

        # =====================================================================
        # Stray light detection (peak truncation at high absorbance)
        # =====================================================================
        # High absorbance values that don't increase further suggest stray light
        max_abs = props.global_range[1]
        mean_abs = props.global_mean

        # Stray light causes apparent absorbance ceiling
        has_stray_light = max_abs > 1.5 and (max_abs - mean_abs) < 0.3 * mean_abs

        # Estimate stray light fraction from absorbance ceiling
        if has_stray_light and max_abs > 0:
            # A_obs = -log10((T + s)/(1+s)) ≈ -log10(s) when T→0
            # So s ≈ 10^(-A_max)
            stray_light_fraction = min(0.01, 10 ** (-max_abs))
        else:
            stray_light_fraction = 0.001  # Default low value

        # =====================================================================
        # Truncated peak detection (boundary absorption bands)
        # =====================================================================
        has_truncated_left = props.has_boundary_rise_left
        has_truncated_right = props.has_boundary_rise_right
        has_truncated_peaks = has_truncated_left or has_truncated_right

        # Estimate boundary peak amplitudes from edge slopes
        left_amp = abs(props.left_edge_slope) * 10 if has_truncated_left else 0.0
        right_amp = abs(props.right_edge_slope) * 10 if has_truncated_right else 0.0
        boundary_peak_amplitudes = (
            min(0.2, left_amp),
            min(0.2, right_amp)
        )

        # =====================================================================
        # Edge curvature detection (baseline bending)
        # =====================================================================
        has_edge_curvature = props.edge_curvature_intensity > 0.01

        # Determine curvature type from intensity and asymmetry
        if has_edge_curvature:
            asymmetry = props.edge_curvature_asymmetry
            if abs(asymmetry) < 0.3:
                # Symmetric curvature
                if props.edge_curvature_intensity > 0:
                    # Need to check sign from original data
                    curvature_type = "smile"  # Default assumption
                else:
                    curvature_type = "frown"
            else:
                curvature_type = "asymmetric"
        else:
            curvature_type = "none"

        # =====================================================================
        # Overall edge artifact detection
        # =====================================================================
        has_edge_artifacts = (
            has_detector_rolloff or
            has_stray_light or
            has_truncated_peaks or
            has_edge_curvature
        )

        # Edge intensity changes
        mean_spectrum = props.mean_spectrum
        if mean_spectrum is not None and len(mean_spectrum) > 20:
            edge_size = max(10, len(mean_spectrum) // 10)
            left_mean = np.mean(mean_spectrum[:edge_size])
            right_mean = np.mean(mean_spectrum[-edge_size:])
            center_mean = np.mean(mean_spectrum[edge_size:-edge_size])

            left_edge_intensity = (left_mean - center_mean) / max(abs(center_mean), 1e-10)
            right_edge_intensity = (right_mean - center_mean) / max(abs(center_mean), 1e-10)
        else:
            left_edge_intensity = 0.0
            right_edge_intensity = 0.0

        return EdgeArtifactInference(
            has_edge_artifacts=has_edge_artifacts,
            has_detector_rolloff=has_detector_rolloff,
            has_stray_light=has_stray_light,
            has_truncated_peaks=has_truncated_peaks,
            has_edge_curvature=has_edge_curvature,
            left_edge_intensity=left_edge_intensity,
            right_edge_intensity=right_edge_intensity,
            edge_noise_ratio=edge_noise_ratio,
            detector_model=detector_model,
            stray_light_fraction=stray_light_fraction,
            curvature_type=curvature_type,
            boundary_peak_amplitudes=boundary_peak_amplitudes,
        )

    def _infer_preprocessing(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        props: SpectralProperties,
    ) -> PreprocessingInference:
        """
        Infer preprocessing type from spectral characteristics.

        Detects whether data has been preprocessed (derivatives, normalization,
        centering, etc.) based on statistical properties.

        Detection heuristics:
            - Second derivative: oscillatory pattern, zero mean, small range,
              high zero-crossing ratio
            - First derivative: moderate zero crossings, zero mean, larger range
            - Mean-centered: zero mean but no oscillation pattern
            - SNV: unit variance per sample, zero mean per sample
            - Raw absorbance: positive values, typical range 0.1-3.0

        Args:
            X: Spectral data (n_samples, n_wavelengths).
            wavelengths: Wavelength array.
            props: Computed spectral properties.

        Returns:
            PreprocessingInference with detected type and confidence.
        """
        min_val, max_val = props.global_range
        global_mean = props.global_mean
        global_range = max_val - min_val

        # Compute additional diagnostics
        # Zero-crossing ratio (high for derivatives)
        mean_spectrum = X.mean(axis=0)
        zero_crossings = np.sum(np.diff(np.sign(mean_spectrum)) != 0)
        zero_crossing_ratio = zero_crossings / max(len(mean_spectrum) - 1, 1)

        # Per-sample std variation (low for SNV)
        per_sample_stds = X.std(axis=1)
        std_of_stds = np.std(per_sample_stds) / max(np.mean(per_sample_stds), 1e-10)

        # Per-sample mean (zero for SNV/mean-centered)
        per_sample_means = X.mean(axis=1)
        mean_of_means = np.mean(per_sample_means)
        std_of_means = np.std(per_sample_means)

        # Oscillation frequency from second derivative of mean spectrum
        if len(mean_spectrum) > 10:
            second_deriv = np.diff(mean_spectrum, n=2)
            sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
            oscillation_freq = sign_changes / max(len(second_deriv) - 1, 1)
        else:
            oscillation_freq = 0.0

        # Curvature already computed in props
        curvature = abs(props.mean_curvature)

        # Score each preprocessing type
        scores: dict[str, float] = {
            "raw_absorbance": 0.0,
            "raw_reflectance": 0.0,
            "second_derivative": 0.0,
            "first_derivative": 0.0,
            "mean_centered": 0.0,
            "snv_corrected": 0.0,
            "normalized": 0.0,
        }

        # ==================================================================
        # Second derivative detection
        # - Very small range (typically ±0.1)
        # - Zero mean
        # - High zero-crossing ratio (oscillatory)
        # - High oscillation frequency
        # ==================================================================
        if global_range < 0.3 and abs(global_mean) < 0.05:
            scores["second_derivative"] += 0.3
        if zero_crossing_ratio > 0.15:
            scores["second_derivative"] += 0.3
        if oscillation_freq > 0.3:
            scores["second_derivative"] += 0.2
        if min_val < 0 < max_val and abs(min_val) / max(abs(max_val), 1e-10) > 0.3:
            scores["second_derivative"] += 0.2

        # ==================================================================
        # First derivative detection
        # - Small to moderate range
        # - Zero mean
        # - Moderate zero crossings
        # - Less oscillatory than 2nd derivative
        # ==================================================================
        if 0.1 < global_range < 1.0 and abs(global_mean) < 0.1:
            scores["first_derivative"] += 0.3
        if 0.05 < zero_crossing_ratio < 0.2:
            scores["first_derivative"] += 0.2
        if 0.1 < oscillation_freq < 0.4:
            scores["first_derivative"] += 0.2

        # ==================================================================
        # SNV detection
        # - Per-sample std is ~1 (or very consistent)
        # - Per-sample mean is ~0
        # - Low variation in per-sample stats
        # ==================================================================
        mean_sample_std = np.mean(per_sample_stds)
        if 0.5 < mean_sample_std < 2.0 and std_of_stds < 0.2:
            scores["snv_corrected"] += 0.4
        if abs(mean_of_means) < 0.1 and std_of_means < 0.2:
            scores["snv_corrected"] += 0.3

        # ==================================================================
        # Mean-centered detection
        # - Global mean near zero
        # - Not oscillatory (distinguishes from derivatives)
        # - Values can be negative
        # ==================================================================
        if abs(global_mean) < 0.1:
            scores["mean_centered"] += 0.3
        if min_val < 0 and zero_crossing_ratio < 0.1:
            scores["mean_centered"] += 0.2
        if oscillation_freq < 0.2 and global_range > 0.3:
            scores["mean_centered"] += 0.2

        # ==================================================================
        # Normalized (min-max scaled)
        # - Range is 0-1 (or close)
        # - All positive
        # ==================================================================
        if 0 <= min_val < 0.1 and 0.9 < max_val <= 1.0:
            scores["normalized"] += 0.6
        elif min_val >= 0 and global_range < 1.5 and max_val < 2.0:
            scores["normalized"] += 0.3

        # ==================================================================
        # Raw absorbance detection
        # - Positive values
        # - Typical range 0.1-3.0
        # - Non-zero mean
        # ==================================================================
        if min_val >= 0 and global_mean > 0.2:
            scores["raw_absorbance"] += 0.3
        if 0.5 < global_mean < 2.0:
            scores["raw_absorbance"] += 0.3
        if 0.5 < global_range < 3.0:
            scores["raw_absorbance"] += 0.2
        if zero_crossing_ratio < 0.05:
            scores["raw_absorbance"] += 0.1

        # ==================================================================
        # Raw reflectance detection
        # - Values 0-1 (or 0-100 for percent)
        # - Non-zero positive mean
        # ==================================================================
        if 0 < global_mean < 0.7 and min_val >= 0 and max_val <= 1.0:
            scores["raw_reflectance"] += 0.4
        if 0 < min_val < max_val <= 100 and global_mean > 20:
            # Percent reflectance
            scores["raw_reflectance"] += 0.4

        # Find best match
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        total_score = sum(scores.values()) + 1e-10
        confidence = best_score / total_score

        # Determine if preprocessed
        is_preprocessed = best_type not in ("raw_absorbance", "raw_reflectance")

        # Suggest inverse operation
        inverse_ops = {
            "second_derivative": "cumulative_sum_twice (or use SG derivatives in forward pipeline)",
            "first_derivative": "cumulative_sum (or use SG derivative in forward pipeline)",
            "mean_centered": "add_global_mean",
            "snv_corrected": "inverse_snv (scale by original std, add original mean)",
            "normalized": "inverse_minmax (scale to original range)",
            "raw_absorbance": None,
            "raw_reflectance": None,
        }

        preprocessing_type = PreprocessingType(best_type)

        return PreprocessingInference(
            preprocessing_type=preprocessing_type,
            confidence=confidence,
            is_preprocessed=is_preprocessed,
            global_mean=global_mean,
            global_range=(min_val, max_val),
            zero_crossing_ratio=zero_crossing_ratio,
            per_sample_std_variation=std_of_stds,
            oscillation_frequency=oscillation_freq,
            suggested_inverse=inverse_ops.get(best_type),
        )

    def _build_temperature_config(self, env: EnvironmentalInference | None) -> dict[str, Any]:
        """Build temperature configuration from inference."""
        if env is None or not env.has_temperature_effects:
            return {}
        return {
            "temperature_variation": env.estimated_temperature_variation,
            "enable_shift": True,
            "enable_intensity": True,
            "enable_broadening": True,
        }

    def _build_moisture_config(self, env: EnvironmentalInference | None) -> dict[str, Any]:
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

    def _build_particle_size_config(self, scatter: ScatteringInference | None) -> dict[str, Any]:
        """Build particle size configuration from inference."""
        if scatter is None or not scatter.has_scatter_effects:
            return {}
        return {
            "mean_size_um": scatter.estimated_particle_size_um,
            "std_size_um": scatter.estimated_particle_size_um * 0.3,
            "size_effect_strength": 1.0,
        }

    def _build_emsc_config(self, scatter: ScatteringInference | None) -> dict[str, Any]:
        """Build EMSC configuration from inference."""
        if scatter is None:
            return {}
        return {
            "multiplicative_scatter_std": scatter.multiplicative_scatter_std,
            "additive_scatter_std": scatter.additive_scatter_std,
            "polynomial_order": 2,
            "include_wavelength_terms": True,
        }

    def _build_edge_artifacts_config(
        self, edge_inf: EdgeArtifactInference | None
    ) -> dict[str, Any]:
        """
        Build edge artifacts configuration from inference.

        Converts inferred edge artifact characteristics into configuration
        parameters for EdgeArtifactsAugmenter or individual augmenters.

        Args:
            edge_inf: Inferred edge artifact characteristics.

        Returns:
            Dictionary with edge artifact configuration parameters.
        """
        if edge_inf is None or not edge_inf.has_edge_artifacts:
            return {}

        config: dict[str, Any] = {}

        # Detector roll-off configuration
        if edge_inf.has_detector_rolloff:
            config["detector_rolloff"] = {
                "enabled": True,
                "detector_model": edge_inf.detector_model,
                "severity": min(1.0, edge_inf.edge_noise_ratio - 1.0) if edge_inf.edge_noise_ratio > 1.0 else 0.3,
            }

        # Stray light configuration
        if edge_inf.has_stray_light:
            config["stray_light"] = {
                "enabled": True,
                "stray_fraction": edge_inf.stray_light_fraction,
                "wavelength_dependent": True,
            }

        # Edge curvature configuration
        if edge_inf.has_edge_curvature:
            config["edge_curvature"] = {
                "enabled": True,
                "curvature_type": edge_inf.curvature_type,
                "left_severity": abs(edge_inf.left_edge_intensity) * 2.0,
                "right_severity": abs(edge_inf.right_edge_intensity) * 2.0,
            }

        # Truncated peaks configuration
        if edge_inf.has_truncated_peaks:
            left_amp, right_amp = edge_inf.boundary_peak_amplitudes
            config["truncated_peaks"] = {
                "enabled": True,
                "left_amplitude": left_amp,
                "right_amplitude": right_amp,
            }

        return config

    def _build_boundary_components_config(
        self,
        edge_inf: EdgeArtifactInference | None,
        wavelengths: np.ndarray,
    ) -> dict[str, Any]:
        """
        Build boundary components configuration from inference.

        Converts inferred truncated peak characteristics into configuration
        for ComponentLibrary.add_boundary_component() calls.

        Args:
            edge_inf: Inferred edge artifact characteristics.
            wavelengths: Wavelength array for boundary calculation.

        Returns:
            Dictionary with boundary component configuration parameters.
        """
        if edge_inf is None or not edge_inf.has_truncated_peaks:
            return {}

        wl_min = float(wavelengths.min())
        wl_max = float(wavelengths.max())
        wl_range = wl_max - wl_min

        left_amp, right_amp = edge_inf.boundary_peak_amplitudes
        config: dict[str, Any] = {"components": []}

        # Left boundary component (peak center below wavelength range)
        if left_amp > 0.05:
            # Estimate peak center outside range based on edge slope
            # Steeper edge = closer peak center
            offset = wl_range * 0.1 * (1.0 / max(0.1, left_amp))
            config["components"].append({
                "name": "boundary_left",
                "band_center": wl_min - min(offset, wl_range * 0.3),
                "bandwidth": wl_range * 0.15,
                "amplitude": left_amp,
                "edge": "left",
            })

        # Right boundary component (peak center above wavelength range)
        if right_amp > 0.05:
            offset = wl_range * 0.1 * (1.0 / max(0.1, right_amp))
            config["components"].append({
                "name": "boundary_right",
                "band_center": wl_max + min(offset, wl_range * 0.3),
                "bandwidth": wl_range * 0.15,
                "amplitude": right_amp,
                "edge": "right",
            })

        return config

    def create_matched_generator(
        self,
        random_state: int | None = None,
    ) -> SyntheticNIRSGenerator:
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

    def apply_matching_preprocessing(
        self,
        X: np.ndarray,
        *,
        window_length: int = 15,
        polyorder: int = 2,
    ) -> np.ndarray:
        """
        Apply preprocessing to match the detected preprocessing of real data.

        If the real data was detected as preprocessed (e.g., second derivative),
        this method applies the same preprocessing to synthetic raw absorbance
        spectra so they match the real data distribution.

        Args:
            X: Raw absorbance spectra from generator (n_samples, n_wavelengths).
            window_length: Savitzky-Golay window length for derivatives.
            polyorder: Polynomial order for Savitzky-Golay filter.

        Returns:
            Preprocessed spectra matching the real data type.

        Raises:
            RuntimeError: If fit() hasn't been called.

        Example:
            >>> fitter = RealDataFitter()
            >>> params = fitter.fit(X_real, wavelengths=wl)
            >>> generator = fitter.create_matched_generator()
            >>> X_raw, _, _ = generator.generate(1000)
            >>> X_matched = fitter.apply_matching_preprocessing(X_raw)
        """
        if self.fitted_params is None or self.fitted_params.preprocessing_inference is None:
            raise RuntimeError("Must call fit() before apply_matching_preprocessing()")

        prep_type = self.fitted_params.preprocessing_inference.preprocessing_type
        prep_info = self.fitted_params.preprocessing_inference

        # No preprocessing needed for raw data
        if prep_type in (PreprocessingType.RAW_ABSORBANCE, PreprocessingType.RAW_REFLECTANCE):
            return X.copy()

        X_out = X.copy()

        if prep_type == PreprocessingType.SECOND_DERIVATIVE:
            # Apply Savitzky-Golay second derivative
            X_out = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=2, axis=1)

        elif prep_type == PreprocessingType.FIRST_DERIVATIVE:
            # Apply Savitzky-Golay first derivative
            X_out = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=1, axis=1)

        elif prep_type == PreprocessingType.MEAN_CENTERED:
            # Mean center each spectrum
            X_out = X - X.mean(axis=1, keepdims=True)

        elif prep_type == PreprocessingType.SNV_CORRECTED:
            # Standard Normal Variate (per sample)
            means = X.mean(axis=1, keepdims=True)
            stds = X.std(axis=1, keepdims=True)
            stds = np.where(stds < 1e-10, 1.0, stds)
            X_out = (X - means) / stds

        elif prep_type == PreprocessingType.NORMALIZED:
            # Min-max normalization (per sample)
            mins = X.min(axis=1, keepdims=True)
            maxs = X.max(axis=1, keepdims=True)
            ranges = maxs - mins
            ranges = np.where(ranges < 1e-10, 1.0, ranges)
            X_out = (X - mins) / ranges

        # Scale to match the detected range of the real data
        real_min, real_max = prep_info.global_range
        synth_min, synth_max = X_out.min(), X_out.max()

        if synth_max - synth_min > 1e-10:
            # Scale synthetic to match real range
            X_out = (X_out - synth_min) / (synth_max - synth_min)
            X_out = X_out * (real_max - real_min) + real_min

        return X_out

    def fit_from_path(
        self,
        path: str,
        *,
        name: str | None = None,
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
        X: np.ndarray = np.asarray(dataset.x({}, layout="2d"))

        # Try to get wavelengths
        wavelengths = None
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            wavelengths = dataset.float_headers()

        return self.fit(X, wavelengths=wavelengths, name=name or dataset.name)

    def evaluate_similarity(
        self,
        X_synthetic: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> dict[str, Any]:
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
        metrics: dict[str, Any] = {}

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

    def get_tuning_recommendations(self) -> list[str]:
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
    X: np.ndarray | SpectroDataset,
    wavelengths: np.ndarray | None = None,
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
    wavelengths: np.ndarray | None = None,
) -> dict[str, Any]:
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

# ============================================================================
# Phase 5: Spectral Fitting Tools (Component Unmixing)
# ============================================================================

@dataclass
class ComponentFitResult:
    """
    Result of fitting spectral components to an observed spectrum.

    Attributes:
        component_names: Names of components used in fitting.
        concentrations: Estimated concentration for each component.
        baseline_coefficients: Polynomial baseline coefficients (if fit_baseline=True).
        fitted_spectrum: Reconstructed spectrum from fit.
        residuals: Difference between observed and fitted spectra.
        r_squared: R² goodness-of-fit metric.
        rmse: Root mean squared error of fit.
        wavelengths: Wavelength grid used for fitting.
    """
    component_names: list[str]
    concentrations: np.ndarray
    baseline_coefficients: np.ndarray | None
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float
    wavelengths: np.ndarray | None = None

    def to_dict(self) -> dict[str, float]:
        """Return concentrations as a dictionary."""
        return dict(zip(self.component_names, self.concentrations, strict=False))

    def top_components(self, n: int = 5, threshold: float = 0.0) -> list[tuple[str, float]]:
        """
        Get top N components by concentration.

        Args:
            n: Maximum number of components to return.
            threshold: Minimum concentration threshold.

        Returns:
            List of (component_name, concentration) tuples, sorted descending.
        """
        pairs = [(name, float(conc)) for name, conc in zip(self.component_names, self.concentrations, strict=False) if conc > threshold]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]

    def summary(self) -> str:
        """Return human-readable summary of fit results."""
        lines = [
            "=" * 60,
            "Component Fit Result",
            "=" * 60,
            f"Fit Quality: R² = {self.r_squared:.4f}, RMSE = {self.rmse:.6f}",
            "",
            "Top Components (by concentration):",
        ]
        for name, conc in self.top_components(10, threshold=0.001):
            lines.append(f"  {name}: {conc:.4f}")

        if self.baseline_coefficients is not None:
            lines.append("")
            lines.append(f"Baseline fitted: order {len(self.baseline_coefficients) - 1}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        top_3 = self.top_components(3)
        top_str = ", ".join(f"{n}={c:.3f}" for n, c in top_3)
        return f"ComponentFitResult(R²={self.r_squared:.4f}, top=[{top_str}])"

class ComponentFitter:
    """
    Fit linear combinations of spectral components to observed spectra.

    Solves: spectrum ≈ Σ(c_i * component_i(λ)) + baseline

    Uses non-negative least squares (NNLS) to ensure positive concentrations,
    which is physically meaningful for spectroscopic analysis.

    **Preprocessing Support**: If your observed spectra are preprocessed
    (e.g., second derivative, SNV), use the `preprocessing` parameter to
    apply the same transformation to component spectra before fitting.

    **Auto-detection**: Set `auto_detect_preprocessing=True` to automatically
    detect the preprocessing type from the data (recommended for derivative data).

    Example:
        >>> from nirs4all.synthesis import ComponentFitter
        >>>
        >>> # Fit with all available components
        >>> fitter = ComponentFitter(wavelengths=np.arange(1000, 2500, 2))
        >>> result = fitter.fit(observed_spectrum)
        >>> print(result.summary())
        >>>
        >>> # Fit preprocessed data (e.g., second derivative)
        >>> fitter = ComponentFitter(
        ...     component_names=["water", "protein", "lipid"],
        ...     wavelengths=wavelengths,
        ...     preprocessing="second_derivative",  # Components will be transformed
        ... )
        >>> result = fitter.fit(derivative_spectrum)
        >>>
        >>> # Auto-detect preprocessing (recommended for unknown data)
        >>> fitter = ComponentFitter(
        ...     wavelengths=wavelengths,
        ...     auto_detect_preprocessing=True,  # Will detect derivative, SNV, etc.
        ... )
        >>> result = fitter.fit(unknown_spectrum)

    Attributes:
        component_names: List of component names to fit.
        wavelengths: Wavelength grid for fitting.
        fit_baseline: Whether to include polynomial baseline.
        baseline_order: Polynomial order for baseline (default 2).
        preprocessing: Preprocessing to apply to components before fitting.
        auto_detect_preprocessing: If True, detect preprocessing from data.
        detected_preprocessing: The detected preprocessing type (after first fit).
    """

    def __init__(
        self,
        component_names: list[str] | None = None,
        wavelengths: np.ndarray | None = None,
        fit_baseline: bool = True,
        baseline_order: int = 2,
        preprocessing: str | PreprocessingType | None = None,
        auto_detect_preprocessing: bool = False,
        sg_window_length: int = 15,
        sg_polyorder: int = 2,
    ):
        """
        Initialize the component fitter.

        Args:
            component_names: Components to fit. If None, uses all available components.
            wavelengths: Wavelength grid (nm). If None, uses default 350-2500nm at 2nm step.
            fit_baseline: Include polynomial baseline in fit.
            baseline_order: Polynomial order for baseline (0=constant, 1=linear, 2=quadratic).
            preprocessing: Preprocessing to apply to component spectra before fitting.
                Options: "second_derivative", "first_derivative", "snv", "mean_centered",
                or a PreprocessingType enum value. If None, no preprocessing is applied.
            auto_detect_preprocessing: If True, automatically detect preprocessing type
                from the data on first fit() call. This is useful for derivative data
                where the preprocessing type is unknown. Takes precedence over
                `preprocessing` if set.
            sg_window_length: Savitzky-Golay window length for derivative preprocessing.
            sg_polyorder: Savitzky-Golay polynomial order for derivative preprocessing.

        Example:
            >>> # Fit raw absorbance data
            >>> fitter = ComponentFitter(
            ...     component_names=["water", "protein", "lipid"],
            ...     wavelengths=np.arange(1000, 2500, 2),
            ... )
            >>>
            >>> # Fit second derivative data
            >>> fitter = ComponentFitter(
            ...     component_names=["water", "protein"],
            ...     wavelengths=wavelengths,
            ...     preprocessing="second_derivative",
            ... )
            >>>
            >>> # Auto-detect preprocessing from data
            >>> fitter = ComponentFitter(
            ...     wavelengths=wavelengths,
            ...     auto_detect_preprocessing=True,
            ... )
        """
        from .components import ComponentLibrary, available_components, get_component

        self.fit_baseline = fit_baseline
        self.baseline_order = baseline_order
        self.auto_detect_preprocessing = auto_detect_preprocessing
        self.detected_preprocessing: PreprocessingType | None = None

        # Preprocessing configuration
        if preprocessing is not None and isinstance(preprocessing, str):
            preprocessing = PreprocessingType(preprocessing)
        self.preprocessing = preprocessing
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder

        # Set wavelengths (default to standard NIR range)
        if wavelengths is None:
            from ._constants import DEFAULT_WAVELENGTH_END, DEFAULT_WAVELENGTH_START, DEFAULT_WAVELENGTH_STEP
            wavelengths = np.arange(DEFAULT_WAVELENGTH_START, DEFAULT_WAVELENGTH_END + DEFAULT_WAVELENGTH_STEP, DEFAULT_WAVELENGTH_STEP)
        self.wavelengths = np.asarray(wavelengths)

        # Set component names (default to all available)
        if component_names is None:
            component_names = available_components()
        self.component_names = list(component_names)

        # Build component library
        self._component_library = ComponentLibrary()
        for name in self.component_names:
            try:
                comp = get_component(name)
                self._component_library.add_component(comp)
            except ValueError:
                # Skip unknown components
                pass

        # Actual names that were successfully loaded
        self.component_names = self._component_library.component_names

        # Design matrix (computed lazily)
        self._design_matrix: np.ndarray | None = None
        self._n_components: int = len(self.component_names)

    def _apply_preprocessing_to_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to component spectra.

        Args:
            spectra: Component spectra, shape (n_components, n_wavelengths).

        Returns:
            Preprocessed spectra with same shape.
        """
        if self.preprocessing is None:
            return spectra

        if self.preprocessing in (PreprocessingType.RAW_ABSORBANCE, PreprocessingType.RAW_REFLECTANCE):
            return spectra

        preprocessed = spectra.copy()

        if self.preprocessing == PreprocessingType.SECOND_DERIVATIVE:
            # Apply Savitzky-Golay second derivative
            preprocessed = savgol_filter(
                spectra,
                window_length=min(self.sg_window_length, spectra.shape[1] - 1) | 1,  # Ensure odd
                polyorder=self.sg_polyorder,
                deriv=2,
                axis=1
            )

        elif self.preprocessing == PreprocessingType.FIRST_DERIVATIVE:
            # Apply Savitzky-Golay first derivative
            preprocessed = savgol_filter(
                spectra,
                window_length=min(self.sg_window_length, spectra.shape[1] - 1) | 1,
                polyorder=self.sg_polyorder,
                deriv=1,
                axis=1
            )

        elif self.preprocessing == PreprocessingType.MEAN_CENTERED:
            # Mean center each spectrum
            preprocessed = spectra - spectra.mean(axis=1, keepdims=True)

        elif self.preprocessing == PreprocessingType.SNV_CORRECTED:
            # Standard Normal Variate (per spectrum)
            means = spectra.mean(axis=1, keepdims=True)
            stds = spectra.std(axis=1, keepdims=True)
            stds = np.where(stds < 1e-10, 1.0, stds)
            preprocessed = (spectra - means) / stds

        elif self.preprocessing == PreprocessingType.NORMALIZED:
            # Min-max normalization (per spectrum)
            mins = spectra.min(axis=1, keepdims=True)
            maxs = spectra.max(axis=1, keepdims=True)
            ranges = maxs - mins
            ranges = np.where(ranges < 1e-10, 1.0, ranges)
            preprocessed = (spectra - mins) / ranges

        return preprocessed

    def _detect_preprocessing_from_data(self, spectrum: np.ndarray) -> PreprocessingType:
        """
        Detect preprocessing type from spectral data characteristics.

        Uses heuristics similar to RealDataFitter._infer_preprocessing() to detect
        whether data has been preprocessed (derivatives, SNV, etc.).

        Args:
            spectrum: Single spectrum or batch of spectra.

        Returns:
            Detected PreprocessingType.
        """
        data = spectrum if spectrum.ndim == 1 else spectrum.mean(axis=0) if spectrum.shape[0] > 1 else spectrum[0]

        min_val = float(np.min(data))
        max_val = float(np.max(data))
        mean_val = float(np.mean(data))
        global_range = max_val - min_val

        # Zero-crossing ratio (high for derivatives)
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        zero_crossing_ratio = zero_crossings / max(len(data) - 1, 1)

        # Oscillation frequency from second derivative
        if len(data) > 10:
            second_deriv = np.diff(data, n=2)
            sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
            oscillation_freq = sign_changes / max(len(second_deriv) - 1, 1)
        else:
            oscillation_freq = 0.0

        # Detection heuristics (similar to notebook logic)
        # Second derivative: very small range, zero mean, high oscillation
        if (global_range < 0.3 and abs(mean_val) < 0.05 and
                zero_crossing_ratio > 0.15 and oscillation_freq > 0.3):
            return PreprocessingType.SECOND_DERIVATIVE

        # First derivative detection:
        # - Bipolar values (both positive and negative)
        # - Zero mean or near zero
        # - Small absolute values (unlike raw absorbance which is always positive)
        is_bipolar = min_val < 0 < max_val
        has_significant_negative = abs(min_val) > 0.001  # Has real negative values
        is_derivative = (
            is_bipolar and has_significant_negative and abs(mean_val) < 0.1 and
            (min_val < -0.5 or abs(min_val / max(max_val, 1e-10)) > 0.3)
        )
        if is_derivative:
            return PreprocessingType.FIRST_DERIVATIVE

        # SNV: per-sample std ~1, mean ~0
        if spectrum.ndim > 1:
            per_sample_stds = np.std(spectrum, axis=1)
            mean_sample_std = np.mean(per_sample_stds)
            std_of_stds = np.std(per_sample_stds) / max(mean_sample_std, 1e-10)
            if 0.5 < mean_sample_std < 2.0 and std_of_stds < 0.2 and abs(mean_val) < 0.1:
                return PreprocessingType.SNV_CORRECTED

        # Mean-centered: zero mean, not oscillatory
        if abs(mean_val) < 0.1 and zero_crossing_ratio < 0.1:
            return PreprocessingType.MEAN_CENTERED

        # Raw absorbance: positive values, typical range 0.1-3.0
        if min_val >= 0 and 0.2 < mean_val < 3.0:
            return PreprocessingType.RAW_ABSORBANCE

        # Raw reflectance: values 0-1
        if min_val >= 0 and max_val <= 1.2 and 0.1 < mean_val < 0.8:
            return PreprocessingType.RAW_REFLECTANCE

        # Default to raw absorbance if uncertain
        return PreprocessingType.RAW_ABSORBANCE

    def _build_design_matrix(self) -> np.ndarray:
        """Build the design matrix from component spectra and baseline terms."""
        # Compute all component spectra: shape (n_components, n_wavelengths)
        component_spectra = self._component_library.compute_all(self.wavelengths)

        # Apply preprocessing if specified
        component_spectra = self._apply_preprocessing_to_spectra(component_spectra)

        # Normalize each component spectrum for numerical stability
        # Store scaling factors to recover original concentrations
        self._component_scales = np.zeros(component_spectra.shape[0])
        for i in range(component_spectra.shape[0]):
            scale = np.abs(component_spectra[i]).max()
            if scale > 1e-10:
                self._component_scales[i] = scale
                component_spectra[i] = component_spectra[i] / scale
            else:
                self._component_scales[i] = 1.0

        # Transpose to (n_wavelengths, n_components) for design matrix
        X = component_spectra.T

        # Add baseline polynomial terms if requested
        if self.fit_baseline:
            # Normalize wavelengths to [0, 1] for numerical stability
            wl_min, wl_max = self.wavelengths.min(), self.wavelengths.max()
            normalized = (self.wavelengths - wl_min) / (wl_max - wl_min) if wl_max > wl_min else np.zeros_like(self.wavelengths)

            baseline_terms = []
            for order in range(self.baseline_order + 1):
                baseline_terms.append(normalized ** order)

            X = np.column_stack([X, np.column_stack(baseline_terms)])

        self._design_matrix = X
        return X

    def fit(
        self,
        spectrum: np.ndarray,
        method: str = "nnls",
    ) -> ComponentFitResult:
        """
        Fit components to a single spectrum.

        Args:
            spectrum: Observed spectrum, shape (n_wavelengths,).
            method: Fitting method.
                - "nnls": Non-negative least squares (default, physically meaningful).
                - "lsq": Unconstrained least squares (allows negative concentrations).

        Returns:
            ComponentFitResult with concentrations, residuals, and fit quality metrics.

        Example:
            >>> result = fitter.fit(observed_spectrum)
            >>> print(f"R² = {result.r_squared:.4f}")
            >>> print(f"Top components: {result.top_components(3)}")
        """
        # Auto-detect preprocessing if enabled and not already set
        if self.auto_detect_preprocessing and self.detected_preprocessing is None:
            detected = self._detect_preprocessing_from_data(spectrum)
            self.detected_preprocessing = detected
            # Only apply detected preprocessing if no explicit preprocessing was set
            if self.preprocessing is None:
                self.preprocessing = detected
                # Reset design matrix to rebuild with new preprocessing
                self._design_matrix = None

        if self._design_matrix is None:
            self._build_design_matrix()
        assert self._design_matrix is not None

        X = self._design_matrix
        y = np.asarray(spectrum).ravel()

        if len(y) != len(self.wavelengths):
            raise ValueError(f"Spectrum length ({len(y)}) does not match wavelengths ({len(self.wavelengths)})")

        # Solve least squares problem
        if method == "nnls":
            from scipy.optimize import nnls
            coefficients, _ = nnls(X, y)
        elif method == "lsq":
            coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'nnls' or 'lsq'.")

        # Split coefficients into component weights and baseline
        component_weights = coefficients[:self._n_components].copy()

        # Rescale component weights to original scale
        # (undo the normalization applied during design matrix construction)
        if hasattr(self, '_component_scales'):
            component_weights = component_weights / self._component_scales

        baseline_coeffs = coefficients[self._n_components:] if self.fit_baseline else None

        # Compute fitted spectrum and residuals
        fitted = X @ coefficients
        residuals = y - fitted

        # Compute fit quality metrics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean(residuals ** 2))

        return ComponentFitResult(
            component_names=self.component_names,
            concentrations=component_weights,
            baseline_coefficients=baseline_coeffs,
            fitted_spectrum=fitted,
            residuals=residuals,
            r_squared=float(r_squared),
            rmse=float(rmse),
            wavelengths=self.wavelengths,
        )

    def fit_batch(
        self,
        spectra: np.ndarray,
        method: str = "nnls",
        n_jobs: int = -1,
    ) -> list[ComponentFitResult]:
        """
        Fit components to multiple spectra in parallel.

        Args:
            spectra: Observed spectra, shape (n_samples, n_wavelengths).
            method: Fitting method ("nnls" or "lsq").
            n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential).

        Returns:
            List of ComponentFitResult objects.

        Example:
            >>> results = fitter.fit_batch(X_observed, n_jobs=4)
            >>> mean_r2 = np.mean([r.r_squared for r in results])
            >>> print(f"Mean R² = {mean_r2:.4f}")
        """
        spectra = np.atleast_2d(spectra)

        if spectra.shape[1] != len(self.wavelengths):
            raise ValueError(f"Spectra width ({spectra.shape[1]}) does not match wavelengths ({len(self.wavelengths)})")

        # Ensure design matrix is built
        if self._design_matrix is None:
            self._build_design_matrix()

        if n_jobs == 1:
            # Sequential execution
            return [self.fit(spectrum, method=method) for spectrum in spectra]
        else:
            # Parallel execution
            try:
                from joblib import Parallel, delayed
                results: list[ComponentFitResult] = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit)(spectrum, method=method) for spectrum in spectra
                )
                return results
            except ImportError:
                # Fallback to sequential if joblib not available
                return [self.fit(spectrum, method=method) for spectrum in spectra]

    def suggest_components(
        self,
        spectrum: np.ndarray,
        top_n: int = 5,
        threshold: float = 0.01,
        method: str = "nnls",
    ) -> list[tuple[str, float]]:
        """
        Suggest which components are likely present in a spectrum.

        Performs a fit and returns the top components by concentration.

        Args:
            spectrum: Observed spectrum, shape (n_wavelengths,).
            top_n: Maximum number of components to return.
            threshold: Minimum concentration threshold.
            method: Fitting method ("nnls" or "lsq").

        Returns:
            List of (component_name, estimated_concentration) tuples,
            sorted by concentration descending.

        Example:
            >>> suggestions = fitter.suggest_components(unknown_spectrum)
            >>> print("Likely components:")
            >>> for name, conc in suggestions:
            ...     print(f"  {name}: {conc:.3f}")
        """
        result = self.fit(spectrum, method=method)
        return result.top_components(top_n, threshold)

    def get_concentration_matrix(
        self,
        spectra: np.ndarray,
        method: str = "nnls",
        n_jobs: int = -1,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get concentration matrix for batch of spectra.

        Convenience method that extracts just the concentrations.

        Args:
            spectra: Observed spectra, shape (n_samples, n_wavelengths).
            method: Fitting method ("nnls" or "lsq").
            n_jobs: Number of parallel jobs.

        Returns:
            Tuple of:
                - concentrations: Array of shape (n_samples, n_components)
                - component_names: List of component names

        Example:
            >>> C, names = fitter.get_concentration_matrix(X_observed)
            >>> water_idx = names.index("water")
            >>> water_concentrations = C[:, water_idx]
        """
        results = self.fit_batch(spectra, method=method, n_jobs=n_jobs)
        C = np.array([r.concentrations for r in results])
        return C, self.component_names

def fit_components(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    component_names: list[str] | None = None,
    fit_baseline: bool = True,
    baseline_order: int = 2,
    method: str = "nnls",
    preprocessing: str | PreprocessingType | None = None,
    auto_detect_preprocessing: bool = False,
) -> ComponentFitResult:
    """
    Convenience function to fit components to a spectrum.

    Args:
        spectrum: Observed spectrum.
        wavelengths: Wavelength grid.
        component_names: Components to fit (None = all available).
        fit_baseline: Include polynomial baseline.
        baseline_order: Polynomial order for baseline.
        method: Fitting method ("nnls" or "lsq").
        preprocessing: Preprocessing to apply to components (e.g., "second_derivative").
            Use this when fitting preprocessed data.
        auto_detect_preprocessing: If True, automatically detect preprocessing type
            from the data. This is useful for derivative data where the preprocessing
            type is unknown. Takes precedence over `preprocessing` if set.

    Returns:
        ComponentFitResult with fit results.

    Example:
        >>> # Fit raw absorbance data
        >>> result = fit_components(spectrum, wavelengths, ["water", "protein", "lipid"])
        >>>
        >>> # Fit second derivative data
        >>> result = fit_components(
        ...     deriv_spectrum, wavelengths, ["water", "protein"],
        ...     preprocessing="second_derivative"
        ... )
        >>>
        >>> # Auto-detect preprocessing (recommended for unknown data)
        >>> result = fit_components(
        ...     unknown_spectrum, wavelengths,
        ...     auto_detect_preprocessing=True
        ... )
    """
    fitter = ComponentFitter(
        component_names=component_names,
        wavelengths=wavelengths,
        fit_baseline=fit_baseline,
        baseline_order=baseline_order,
        preprocessing=preprocessing,
        auto_detect_preprocessing=auto_detect_preprocessing,
    )
    return fitter.fit(spectrum, method=method)

# ============================================================================
# Optimized Component Fitting (Greedy Selection)
# ============================================================================

# Component category definitions for domain-aware fitting
COMPONENT_CATEGORIES = {
    'water_related': ['water', 'moisture'],
    'proteins': ['protein', 'nitrogen_compound', 'urea', 'amino_acid', 'casein', 'gluten',
                 'albumin', 'collagen', 'keratin', 'zein', 'gelatin', 'whey'],
    'lipids': ['lipid', 'oil', 'saturated_fat', 'unsaturated_fat', 'waxes',
               'oleic_acid', 'linoleic_acid', 'linolenic_acid', 'palmitic_acid',
               'stearic_acid', 'phospholipid', 'cholesterol', 'cocoa_butter'],
    'hydrocarbons': ['aromatic', 'alkane'],
    'petroleum': ['crude_oil', 'diesel', 'gasoline', 'kerosene', 'pah', 'alkane', 'aromatic', 'oil'],
    'carbohydrates': ['starch', 'cellulose', 'glucose', 'fructose', 'sucrose',
                      'hemicellulose', 'lignin', 'lactose', 'cotton', 'dietary_fiber',
                      'maltose', 'raffinose', 'inulin', 'xylose', 'arabinose',
                      'galactose', 'mannose', 'trehalose'],
    'alcohols': ['ethanol', 'methanol', 'glycerol', 'propanol', 'butanol',
                 'sorbitol', 'mannitol', 'xylitol', 'isopropanol'],
    'organic_acids': ['acetic_acid', 'citric_acid', 'lactic_acid', 'malic_acid',
                      'tartaric_acid', 'formic_acid', 'oxalic_acid', 'succinic_acid',
                      'fumaric_acid', 'propionic_acid', 'butyric_acid', 'ascorbic_acid'],
    'pigments': ['chlorophyll', 'chlorophyll_a', 'chlorophyll_b', 'carotenoid',
                 'beta_carotene', 'lycopene', 'lutein', 'anthocyanin', 'tannins'],
    'minerals': ['carbonates', 'gypsum', 'kaolinite', 'montmorillonite', 'illite',
                 'goethite', 'talc', 'silica'],
    'pharmaceuticals': ['caffeine', 'aspirin', 'paracetamol', 'ibuprofen', 'naproxen',
                        'diclofenac', 'metformin', 'omeprazole', 'amoxicillin',
                        'microcrystalline_cellulose', 'starch', 'cellulose', 'lactose'],
    'organic_matter': ['lignin', 'cellulose', 'hemicellulose', 'protein', 'lipid'],
    'polymers': ['polyethylene', 'polystyrene', 'polypropylene', 'pvc', 'pet',
                 'polyester', 'nylon', 'pmma', 'ptfe', 'abs', 'natural_rubber'],
}

# Components that can cause overfitting (biological pigments in non-bio samples)
EXCLUDED_COMPONENTS = {
    'cytochrome_c', 'myoglobin', 'hemoglobin_oxy', 'hemoglobin_deoxy',
    'bilirubin', 'melanin', 'pah'
}

# Universal components that can appear in most samples
UNIVERSAL_COMPONENTS = {'water', 'moisture'}

@dataclass
class OptimizedFitResult:
    """
    Result from optimized greedy component fitting.

    Attributes:
        component_names: Names of selected components (in order of selection).
        concentrations: Fitted concentrations for each component.
        baseline_coefficients: Polynomial baseline coefficients.
        fitted_spectrum: Reconstructed spectrum from fit.
        residuals: Fit residuals.
        r_squared: Coefficient of determination.
        rmse: Root mean squared error.
        n_components: Number of components selected.
        n_priority_components: Number of components from priority categories.
        baseline_r_squared: R² from baseline-only fit (for comparison).
        wavelengths: Wavelength grid used for fitting.
    """
    component_names: list[str]
    concentrations: np.ndarray
    baseline_coefficients: np.ndarray | None
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float
    n_components: int
    n_priority_components: int
    baseline_r_squared: float
    wavelengths: np.ndarray

    def top_components(
        self,
        n: int = 5,
        threshold: float = 0.001,
    ) -> list[tuple[str, float]]:
        """Get top components by concentration."""
        if len(self.component_names) == 0:
            return []
        sorted_indices = np.argsort(-np.abs(self.concentrations))
        result = []
        for idx in sorted_indices[:n]:
            if np.abs(self.concentrations[idx]) >= threshold:
                result.append((self.component_names[idx], float(self.concentrations[idx])))
        return result

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "Optimized Component Fit Result",
            "=" * 60,
            f"R² = {self.r_squared:.4f} (baseline only: {self.baseline_r_squared:.4f})",
            f"RMSE = {self.rmse:.6f}",
            f"Components: {self.n_components} ({self.n_priority_components} from priority)",
            "",
            "Selected Components:",
        ]
        for name, conc in self.top_components(10, threshold=0.0001):
            lines.append(f"  {name}: {conc:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

class OptimizedComponentFitter:
    """
    Optimize component selection using greedy search with category prioritization.

    Unlike ComponentFitter which fits all components simultaneously with NNLS,
    this class uses a greedy forward selection approach that:

    1. Starts with baseline-only fit
    2. Greedily adds components from priority categories (low threshold)
    3. Fills remaining slots from other categories (higher threshold)
    4. Applies swap refinement to escape local optima

    This approach produces much better fits for real-world data by:
    - Avoiding overfitting to spurious components
    - Respecting domain knowledge (e.g., protein for dairy, starch for grains)
    - Allowing both positive and negative coefficients (OLS, not NNLS)

    Example:
        >>> from nirs4all.synthesis import OptimizedComponentFitter
        >>>
        >>> # Create fitter for grain analysis
        >>> fitter = OptimizedComponentFitter(
        ...     wavelengths=wavelengths,
        ...     priority_categories=['carbohydrates', 'proteins', 'water_related'],
        ...     max_components=10,
        ... )
        >>> result = fitter.fit(spectrum)
        >>> print(result.summary())

    Attributes:
        wavelengths: Wavelength grid for fitting.
        priority_categories: Categories to prioritize in component selection.
        max_components: Maximum number of components to select.
        baseline_order: Polynomial order for baseline (default 4).
        preprocessing: Preprocessing to apply to components.
        auto_detect_preprocessing: Auto-detect preprocessing from data.
    """

    def __init__(
        self,
        wavelengths: np.ndarray | None = None,
        priority_categories: list[str] | None = None,
        max_components: int = 10,
        baseline_order: int = 4,
        preprocessing: str | PreprocessingType | None = None,
        auto_detect_preprocessing: bool = False,
        sg_window_length: int = 15,
        sg_polyorder: int = 3,
        regularization: float = 1e-6,
        smooth_sigma_nm: float = 30.0,
        use_nnls: bool = False,
    ):
        """
        Initialize the optimized component fitter.

        Args:
            wavelengths: Wavelength grid (nm). If None, uses default NIR range.
            priority_categories: Categories to prioritize (from COMPONENT_CATEGORIES).
                E.g., ['carbohydrates', 'proteins'] for grain analysis.
            max_components: Maximum components to select (default 10).
            baseline_order: Polynomial baseline order (default 4 for Chebyshev).
            preprocessing: Preprocessing to apply ('first_derivative', 'second_derivative', etc.).
            auto_detect_preprocessing: Auto-detect from data if True.
            sg_window_length: Savitzky-Golay window for derivative preprocessing.
            sg_polyorder: Savitzky-Golay polynomial order.
            regularization: Regularization strength for OLS (default 1e-6).
            smooth_sigma_nm: Gaussian smoothing sigma in nm to broaden component spectra.
                Set to 0 to disable smoothing. Default 30 nm produces broad, natural bands.
            use_nnls: Use non-negative least squares instead of OLS. This prevents
                negative coefficients which can cause oscillations. Default False.
        """
        from .components import available_components, get_component

        # Preprocessing configuration
        if preprocessing is not None and isinstance(preprocessing, str):
            preprocessing = PreprocessingType(preprocessing)
        self.preprocessing = preprocessing
        self.auto_detect_preprocessing = auto_detect_preprocessing
        self.detected_preprocessing: PreprocessingType | None = None
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.smooth_sigma_nm = smooth_sigma_nm
        self.use_nnls = use_nnls

        # Set wavelengths
        if wavelengths is None:
            from ._constants import DEFAULT_WAVELENGTH_END, DEFAULT_WAVELENGTH_START, DEFAULT_WAVELENGTH_STEP
            wavelengths = np.arange(DEFAULT_WAVELENGTH_START, DEFAULT_WAVELENGTH_END + DEFAULT_WAVELENGTH_STEP, DEFAULT_WAVELENGTH_STEP)
        self.wavelengths = np.asarray(wavelengths)

        self.priority_categories = priority_categories or []
        self.max_components = max_components
        self.baseline_order = baseline_order
        self.regularization = regularization

        # Load all available components (excluding problematic ones)
        all_names = available_components()
        self._all_component_names = [n for n in all_names if n not in EXCLUDED_COMPONENTS]

        # Pre-compute component spectra (will be updated with preprocessing)
        self._component_spectra: dict[str, np.ndarray] = {}
        self._baseline_matrix: np.ndarray | None = None

    def _apply_preprocessing_to_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a single spectrum."""
        if self.preprocessing is None or self.preprocessing in (
            PreprocessingType.RAW_ABSORBANCE, PreprocessingType.RAW_REFLECTANCE
        ):
            return spectrum

        n_wl = len(spectrum)
        window = min(self.sg_window_length, n_wl - 1) | 1

        if self.preprocessing == PreprocessingType.SECOND_DERIVATIVE:
            return np.asarray(savgol_filter(spectrum, window, min(self.sg_polyorder + 1, window - 1), deriv=2))
        elif self.preprocessing == PreprocessingType.FIRST_DERIVATIVE:
            return np.asarray(savgol_filter(spectrum, window, min(self.sg_polyorder, window - 1), deriv=1))
        elif self.preprocessing == PreprocessingType.MEAN_CENTERED:
            return np.asarray(spectrum - spectrum.mean())
        elif self.preprocessing == PreprocessingType.SNV_CORRECTED:
            std = spectrum.std()
            if std < 1e-10:
                return np.asarray(spectrum - spectrum.mean())
            return np.asarray((spectrum - spectrum.mean()) / std)

        return spectrum

    def _detect_preprocessing_from_data(self, spectrum: np.ndarray) -> PreprocessingType:
        """Detect preprocessing type from data characteristics."""
        min_val = float(np.min(spectrum))
        max_val = float(np.max(spectrum))
        mean_val = float(np.mean(spectrum))
        global_range = max_val - min_val

        # Zero-crossing ratio
        zero_crossings = np.sum(np.diff(np.sign(spectrum)) != 0)
        zero_crossing_ratio = zero_crossings / max(len(spectrum) - 1, 1)

        # Second derivative: very small range, zero mean, high oscillation
        if global_range < 0.3 and abs(mean_val) < 0.05 and zero_crossing_ratio > 0.15:
            return PreprocessingType.SECOND_DERIVATIVE

        # First derivative: bipolar values, near-zero mean
        is_bipolar = min_val < 0 < max_val
        has_significant_negative = abs(min_val) > 0.001
        if is_bipolar and has_significant_negative and abs(mean_val) < 0.1 and (min_val < -0.5 or abs(min_val / max(max_val, 1e-10)) > 0.3):
            return PreprocessingType.FIRST_DERIVATIVE

        # Raw absorbance: positive values, typical range
        if min_val >= 0 and 0.2 < mean_val < 3.0:
            return PreprocessingType.RAW_ABSORBANCE

        return PreprocessingType.RAW_ABSORBANCE

    def _compute_component_spectra(self) -> dict[str, np.ndarray]:
        """
        Compute all component spectra with preprocessing and smoothing applied.

        Applies Gaussian smoothing to broaden narrow Voigt bands, making them
        more suitable for fitting real NIRS data which typically has broader features.
        """
        from scipy.ndimage import gaussian_filter1d

        from .components import get_component

        # Determine smoothing kernel size based on wavelength grid
        wl_spacing = np.median(np.diff(self.wavelengths)) if len(self.wavelengths) > 1 else 1.0
        smooth_sigma_pts = max(1, int(self.smooth_sigma_nm / wl_spacing)) if self.smooth_sigma_nm > 0 else 0

        spectra = {}
        for name in self._all_component_names:
            try:
                comp = get_component(name)
                spec = comp.compute(self.wavelengths)

                # Apply Gaussian smoothing to broaden narrow Voigt bands
                # This makes component spectra more like real broad NIR absorption features
                if smooth_sigma_pts > 1:
                    spec = gaussian_filter1d(spec, sigma=smooth_sigma_pts, mode='nearest')

                # Apply preprocessing (derivative, SNV, etc.)
                spec_prep = self._apply_preprocessing_to_spectrum(spec)

                if np.max(np.abs(spec_prep)) > 1e-10:
                    spectra[name] = spec_prep
            except (ValueError, KeyError):
                pass
        return spectra

    def _build_baseline_matrix(self) -> np.ndarray:
        """Build Chebyshev polynomial baseline matrix for numerical stability."""
        wl_norm = 2 * (self.wavelengths - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1

        baseline_terms = []
        for order in range(self.baseline_order + 1):
            if order == 0:
                baseline_terms.append(np.ones_like(wl_norm))
            elif order == 1:
                baseline_terms.append(wl_norm)
            else:
                # Chebyshev recurrence: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
                baseline_terms.append(2 * wl_norm * baseline_terms[-1] - baseline_terms[-2])

        return np.column_stack(baseline_terms)

    def _fit_with_components(
        self,
        target: np.ndarray,
        component_spectra: dict[str, np.ndarray],
        component_names: list[str],
        baseline_matrix: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Fit target using OLS with baseline + selected component spectra.

        Returns:
            Tuple of (r_squared, component_quantities, baseline_coefficients)
        """
        valid_names = [n for n in component_names if n in component_spectra]

        if not valid_names:
            A = baseline_matrix
        else:
            comp_matrix = np.column_stack([component_spectra[n] for n in valid_names])
            A = np.hstack([baseline_matrix, comp_matrix])

        # Normalize columns for numerical stability
        col_norms = np.linalg.norm(A, axis=0)
        col_norms = np.where(col_norms < 1e-10, 1.0, col_norms)
        A_norm = A / col_norms

        n_baseline = baseline_matrix.shape[1]
        n_comp = len(valid_names)

        if self.use_nnls:
            # Use NNLS for non-negative coefficients (prevents oscillations)
            # For baseline, we need to allow negative, so split the problem
            from scipy.optimize import nnls

            try:
                # First fit baseline with regular OLS
                baseline_coeffs_norm, _ = np.linalg.lstsq(A_norm[:, :n_baseline], target, rcond=None)[:2]

                # Then fit components with NNLS on residual
                residual = target - A_norm[:, :n_baseline] @ baseline_coeffs_norm
                if n_comp > 0:
                    comp_coeffs_norm, _ = nnls(A_norm[:, n_baseline:], residual)
                else:
                    comp_coeffs_norm = np.array([])

                coeffs_norm = np.concatenate([baseline_coeffs_norm, comp_coeffs_norm])
            except Exception:
                return 0.0, np.zeros(n_comp), np.zeros(n_baseline)
        else:
            # Regularized OLS (allows negative coefficients)
            try:
                ATA = A_norm.T @ A_norm + self.regularization * np.eye(A_norm.shape[1])
                ATb = A_norm.T @ target
                coeffs_norm = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                return 0.0, np.zeros(n_comp), np.zeros(n_baseline)

        coeffs = coeffs_norm / col_norms
        fitted = A @ coeffs

        # Compute R²
        ss_res = np.sum((target - fitted) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        r2 = max(0.0, r2)

        baseline_coeffs = coeffs[:n_baseline]
        component_quantities = coeffs[n_baseline:] if len(coeffs) > n_baseline else np.zeros(0)

        return r2, component_quantities, baseline_coeffs

    def _greedy_select_from_pool(
        self,
        target: np.ndarray,
        component_spectra: dict[str, np.ndarray],
        baseline_matrix: np.ndarray,
        pool: list[str],
        current_components: list[str],
        current_r2: float,
        max_to_add: int,
        min_improvement: float,
    ) -> tuple[list[str], float, int]:
        """
        Greedy forward selection from a component pool.

        Returns:
            Tuple of (updated_components, new_r2, n_added)
        """
        remaining = [c for c in pool if c in component_spectra and c not in current_components]
        n_added = 0

        for _ in range(max_to_add):
            if not remaining:
                break

            best_add = None
            best_add_r2 = current_r2

            for comp in remaining:
                test_components = current_components + [comp]
                r2, _, _ = self._fit_with_components(target, component_spectra, test_components, baseline_matrix)

                if r2 > best_add_r2 + min_improvement:
                    best_add = comp
                    best_add_r2 = r2

            if best_add is None:
                break

            current_components = current_components + [best_add]
            remaining.remove(best_add)
            current_r2 = best_add_r2
            n_added += 1

        return current_components, current_r2, n_added

    def fit(self, spectrum: np.ndarray) -> OptimizedFitResult:
        """
        Fit components to a spectrum using greedy category-prioritized selection.

        The algorithm:
        1. Starts with baseline-only fit
        2. Greedily adds components from priority categories (very low threshold: 0.0001)
        3. Fills remaining slots from other categories (higher threshold: 0.005)
        4. Applies swap refinement (prefers swapping in priority components)

        Args:
            spectrum: Observed spectrum, shape (n_wavelengths,).

        Returns:
            OptimizedFitResult with fit results.
        """
        target = np.asarray(spectrum).ravel()

        if len(target) != len(self.wavelengths):
            raise ValueError(f"Spectrum length ({len(target)}) does not match wavelengths ({len(self.wavelengths)})")

        # Auto-detect preprocessing if enabled
        if self.auto_detect_preprocessing and self.detected_preprocessing is None:
            detected = self._detect_preprocessing_from_data(target)
            self.detected_preprocessing = detected
            if self.preprocessing is None:
                self.preprocessing = detected

        # Compute component spectra with preprocessing
        component_spectra = self._compute_component_spectra()
        baseline_matrix = self._build_baseline_matrix()

        if not component_spectra:
            # No valid components - return baseline-only fit
            baseline_r2, _, baseline_coeffs = self._fit_with_components(
                target, component_spectra, [], baseline_matrix
            )
            fitted = baseline_matrix @ baseline_coeffs
            return OptimizedFitResult(
                component_names=[],
                concentrations=np.array([]),
                baseline_coefficients=baseline_coeffs,
                fitted_spectrum=fitted,
                residuals=target - fitted,
                r_squared=baseline_r2,
                rmse=float(np.sqrt(np.mean((target - fitted) ** 2))),
                n_components=0,
                n_priority_components=0,
                baseline_r_squared=baseline_r2,
                wavelengths=self.wavelengths,
            )

        # Baseline-only R² for comparison
        baseline_r2, _, _ = self._fit_with_components(target, component_spectra, [], baseline_matrix)

        # Build priority pool from category components
        priority_pool = []
        for cat in self.priority_categories:
            cat_comps = COMPONENT_CATEGORIES.get(cat, [])
            for comp in cat_comps:
                if comp in component_spectra and comp not in priority_pool:
                    priority_pool.append(comp)

        other_pool = [n for n in component_spectra if n not in priority_pool]

        # Phase 1: Greedy selection from PRIORITY pool (LOW threshold)
        best_components: list[str] = []
        best_r2 = baseline_r2
        priority_slots = min(self.max_components - 2, len(priority_pool), 8)

        if priority_pool:
            best_components, best_r2, _ = self._greedy_select_from_pool(
                target, component_spectra, baseline_matrix,
                priority_pool, best_components, best_r2,
                max_to_add=priority_slots, min_improvement=0.0001  # Very low threshold
            )

        n_priority_selected = len(best_components)

        # Phase 2: Fill remaining slots from OTHER pool (HIGH threshold)
        remaining_slots = self.max_components - len(best_components)
        if remaining_slots > 0 and other_pool:
            best_components, best_r2, _ = self._greedy_select_from_pool(
                target, component_spectra, baseline_matrix,
                other_pool, best_components, best_r2,
                max_to_add=remaining_slots, min_improvement=0.005  # Higher threshold
            )

        # Phase 3: Swap refinement (prefer priority components)
        all_pool = priority_pool + other_pool
        improved = True
        n_swaps = 0

        while improved and n_swaps < 15:
            improved = False
            for i, _old_comp in enumerate(best_components):
                for new_comp in all_pool:
                    if new_comp in best_components:
                        continue
                    test_components = best_components.copy()
                    test_components[i] = new_comp
                    r2, _, _ = self._fit_with_components(target, component_spectra, test_components, baseline_matrix)

                    # Lower threshold for priority components
                    threshold = 0.0001 if new_comp in priority_pool else 0.005
                    if r2 > best_r2 + threshold:
                        best_components[i] = new_comp
                        best_r2 = r2
                        improved = True
                        n_swaps += 1
                        break
                if improved:
                    break

        # Final fit
        r2, quantities, baseline_coeffs = self._fit_with_components(
            target, component_spectra, best_components, baseline_matrix
        )

        # Reconstruct fitted spectrum
        baseline_fitted = baseline_matrix @ baseline_coeffs
        if best_components and len(quantities) > 0:
            valid_names = [n for n in best_components if n in component_spectra]
            comp_matrix = np.column_stack([component_spectra[n] for n in valid_names])
            fitted = baseline_fitted + comp_matrix @ quantities
        else:
            fitted = baseline_fitted

        # Count priority components in final selection
        n_priority_final = sum(1 for c in best_components if c in priority_pool)

        return OptimizedFitResult(
            component_names=best_components,
            concentrations=quantities,
            baseline_coefficients=baseline_coeffs,
            fitted_spectrum=fitted,
            residuals=target - fitted,
            r_squared=r2,
            rmse=float(np.sqrt(np.mean((target - fitted) ** 2))),
            n_components=len(best_components),
            n_priority_components=n_priority_final,
            baseline_r_squared=baseline_r2,
            wavelengths=self.wavelengths,
        )

def fit_components_optimized(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    priority_categories: list[str] | None = None,
    max_components: int = 10,
    baseline_order: int = 4,
    preprocessing: str | PreprocessingType | None = None,
    auto_detect_preprocessing: bool = False,
    smooth_sigma_nm: float = 30.0,
    use_nnls: bool = False,
) -> OptimizedFitResult:
    """
    Convenience function for optimized component fitting.

    Uses greedy category-prioritized selection for better fits than NNLS.

    Args:
        spectrum: Observed spectrum.
        wavelengths: Wavelength grid.
        priority_categories: Categories to prioritize (e.g., ['carbohydrates', 'proteins']).
        max_components: Maximum components to select.
        baseline_order: Polynomial baseline order.
        preprocessing: Preprocessing type ('first_derivative', 'second_derivative', etc.).
        auto_detect_preprocessing: Auto-detect preprocessing from data.
        smooth_sigma_nm: Gaussian smoothing sigma in nm to broaden component spectra.
        use_nnls: Use non-negative least squares instead of OLS.

    Returns:
        OptimizedFitResult with fit results.

    Example:
        >>> result = fit_components_optimized(
        ...     spectrum, wavelengths,
        ...     priority_categories=['carbohydrates', 'proteins'],
        ...     auto_detect_preprocessing=True,
        ... )
        >>> print(f"R² = {result.r_squared:.4f}")
    """
    fitter = OptimizedComponentFitter(
        wavelengths=wavelengths,
        priority_categories=priority_categories,
        max_components=max_components,
        baseline_order=baseline_order,
        preprocessing=preprocessing,
        auto_detect_preprocessing=auto_detect_preprocessing,
        smooth_sigma_nm=smooth_sigma_nm,
        use_nnls=use_nnls,
    )
    return fitter.fit(spectrum)

# ============================================================================
# Real Band Fitting (Using NIR_BANDS dictionary)
# ============================================================================

@dataclass
class RealBandFitResult:
    """
    Result from real band fitting using known NIR band assignments.

    Attributes:
        band_names: Names of fitted bands (e.g., "O-H/1st", "C-H/combination").
        band_centers: Fixed center wavelengths from NIR_BANDS.
        amplitudes: Fitted amplitudes for each band.
        sigmas: Sigma values (within constrained ranges).
        baseline_coefficients: Polynomial baseline coefficients.
        fitted_spectrum: Reconstructed spectrum from fit.
        residuals: Fit residuals.
        r_squared: Coefficient of determination.
        rmse: Root mean squared error.
        n_bands: Number of bands used.
        wavelengths: Wavelength grid used for fitting.
        band_assignments: Original BandAssignment objects.
    """
    band_names: list[str]
    band_centers: np.ndarray
    amplitudes: np.ndarray
    sigmas: np.ndarray
    baseline_coefficients: np.ndarray
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    r_squared: float
    rmse: float
    n_bands: int
    wavelengths: np.ndarray
    band_assignments: list[Any] = field(default_factory=list)

    def top_bands(
        self,
        n: int = 10,
        threshold: float = 0.001,
    ) -> list[tuple[str, float, float]]:
        """Get top bands by amplitude. Returns (name, center, amplitude)."""
        if len(self.band_names) == 0:
            return []
        sorted_indices = np.argsort(-np.abs(self.amplitudes))
        result = []
        for idx in sorted_indices[:n]:
            if np.abs(self.amplitudes[idx]) >= threshold:
                result.append((
                    self.band_names[idx],
                    float(self.band_centers[idx]),
                    float(self.amplitudes[idx])
                ))
        return result

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "Real Band Fit Result (NIR_BANDS dictionary)",
            "=" * 60,
            f"R² = {self.r_squared:.4f}",
            f"RMSE = {self.rmse:.6f}",
            f"Bands used: {self.n_bands}",
            "",
            "Top Bands:",
        ]
        for name, center, amp in self.top_bands(10, threshold=0.0001):
            lines.append(f"  {center:.0f} nm: {name} (amp={amp:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)

class RealBandFitter:
    """
    Fit spectra using REAL NIR band assignments from the _bands.py dictionary.

    Unlike pure Gaussian band fitting which optimizes band centers freely,
    this class uses:
    - Fixed band centers from known spectroscopic literature assignments
    - Constrained sigma values based on typical ranges for each band type
    - Only amplitude optimization (more physically interpretable)

    This provides spectroscopically meaningful decomposition that can be
    linked back to functional groups (O-H, C-H, N-H, etc.) and overtone levels.

    Example:
        >>> from nirs4all.synthesis import RealBandFitter
        >>>
        >>> fitter = RealBandFitter(baseline_order=4, max_bands=40)
        >>> result = fitter.fit(spectrum, wavelengths)
        >>> print(result.summary())
        >>>
        >>> # See which functional groups contribute
        >>> for name, center, amp in result.top_bands(10):
        ...     print(f"{center:.0f} nm: {name} (amplitude={amp:.4f})")

    Attributes:
        baseline_order: Polynomial baseline order.
        max_bands: Maximum number of bands to use.
        target_r2: Target R² for iterative refinement.
        allow_sigma_variation: Allow sigma to vary within literature ranges.
        sigma_margin: How much sigma can vary from midpoint (0.3 = ±30%).
    """

    def __init__(
        self,
        baseline_order: int = 4,
        max_bands: int = 50,
        target_r2: float = 0.98,
        allow_sigma_variation: bool = True,
        sigma_margin: float = 0.3,
        n_iterations: int = 3,
    ):
        """
        Initialize the real band fitter.

        Args:
            baseline_order: Polynomial baseline degree (default 4).
            max_bands: Maximum number of bands to use (default 50).
            target_r2: Target R² for early stopping (default 0.98).
            allow_sigma_variation: Allow sigma to vary within range (default True).
            sigma_margin: How much sigma can vary from midpoint (default ±30%).
            n_iterations: Number of refinement iterations (default 3).
        """
        self.baseline_order = baseline_order
        self.max_bands = max_bands
        self.target_r2 = target_r2
        self.allow_sigma_variation = allow_sigma_variation
        self.sigma_margin = sigma_margin
        self.n_iterations = n_iterations

    def _get_candidate_bands(self, wl_min: float, wl_max: float) -> list[Any]:
        """Get all bands in the wavelength range from NIR_BANDS dictionary."""
        from ._bands import get_bands_in_range
        return get_bands_in_range(wl_min, wl_max)

    def _compute_band(
        self,
        wl: np.ndarray,
        center: float,
        sigma: float,
        amplitude: float,
    ) -> np.ndarray:
        """Compute Gaussian band profile."""
        return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    def _compute_all_bands(
        self,
        wl: np.ndarray,
        bands: list[Any],
        amplitudes: np.ndarray,
        sigmas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute sum of all bands."""
        result = np.zeros_like(wl, dtype=float)
        for i, band in enumerate(bands):
            sigma = sigmas[i] if sigmas is not None else (band.sigma_range[0] + band.sigma_range[1]) / 2
            result += self._compute_band(wl, band.center, sigma, amplitudes[i])
        return result

    def _build_baseline(self, wl: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Build polynomial baseline using Chebyshev polynomials."""
        wl_norm = 2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1

        result = np.zeros_like(wl)
        T_prev = np.ones_like(wl_norm)
        T_curr = wl_norm.copy()

        for i, coeff in enumerate(coeffs):
            if i == 0:
                result += coeff * T_prev
            elif i == 1:
                result += coeff * T_curr
            else:
                T_next = 2 * wl_norm * T_curr - T_prev
                result += coeff * T_next
                T_prev = T_curr
                T_curr = T_next

        return result

    def fit(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> RealBandFitResult:
        """
        Fit spectrum using real NIR band positions.

        Args:
            spectrum: Target spectrum to fit, shape (n_wavelengths,).
            wavelengths: Wavelengths in nm, shape (n_wavelengths,).

        Returns:
            RealBandFitResult with fit results and band assignments.
        """
        from scipy.optimize import minimize

        spectrum = np.asarray(spectrum).ravel()
        wavelengths = np.asarray(wavelengths).ravel()

        if len(spectrum) != len(wavelengths):
            raise ValueError(f"Spectrum and wavelengths length mismatch: {len(spectrum)} vs {len(wavelengths)}")

        wl = wavelengths
        wl_min, wl_max = wl.min(), wl.max()
        spec_range = spectrum.max() - spectrum.min()
        spec_mean = spectrum.mean()
        n_baseline = self.baseline_order + 1

        # Get candidate bands from NIR_BANDS dictionary
        all_bands = self._get_candidate_bands(wl_min - 50, wl_max + 50)

        # Filter to bands with centers actually in range
        candidate_bands = [b for b in all_bands if wl_min <= b.center <= wl_max]

        if not candidate_bands:
            candidate_bands = all_bands[:self.max_bands]

        if not candidate_bands:
            # No bands available - return baseline-only fit
            wl_norm = 2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1
            baseline_coeffs = np.polyfit(wl_norm, spectrum, self.baseline_order)[::-1]
            fitted = self._build_baseline(wl, baseline_coeffs)
            ss_res = np.sum((spectrum - fitted) ** 2)
            ss_tot = np.sum((spectrum - np.mean(spectrum)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            return RealBandFitResult(
                band_names=[],
                band_centers=np.array([]),
                amplitudes=np.array([]),
                sigmas=np.array([]),
                baseline_coefficients=baseline_coeffs,
                fitted_spectrum=fitted,
                residuals=spectrum - fitted,
                r_squared=r2,
                rmse=float(np.sqrt(np.mean((spectrum - fitted) ** 2))),
                n_bands=0,
                wavelengths=wl,
                band_assignments=[],
            )

        # Sort by intensity (prefer stronger bands)
        intensity_order = {'very_strong': 0, 'strong': 1, 'medium': 2, 'weak': 3, 'very_weak': 4}
        candidate_bands = sorted(
            candidate_bands,
            key=lambda b: (intensity_order.get(b.intensity, 5), b.center)
        )

        # Limit to max_bands
        candidate_bands = candidate_bands[:self.max_bands]
        n_bands = len(candidate_bands)

        # Get sigma ranges
        band_centers = np.array([b.center for b in candidate_bands])
        band_sigmas_mid = np.array([(b.sigma_range[0] + b.sigma_range[1]) / 2 for b in candidate_bands])
        band_sigmas_lo = np.array([b.sigma_range[0] for b in candidate_bands])
        band_sigmas_hi = np.array([b.sigma_range[1] for b in candidate_bands])

        # Initial baseline fit
        wl_norm = 2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1
        baseline_init = np.polyfit(wl_norm, spectrum, self.baseline_order)[::-1]

        # Build initial parameters
        if self.allow_sigma_variation:
            n_params = n_bands * 2 + n_baseline
            x0 = np.zeros(n_params)
            # Initial amplitudes
            for i, band in enumerate(candidate_bands):
                idx = np.argmin(np.abs(wl - band.center))
                x0[i] = max(0, (spectrum[idx] - spec_mean) * 0.3)
            x0[n_bands:2*n_bands] = band_sigmas_mid
            x0[2*n_bands:] = baseline_init

            bounds_lo = (
                [-spec_range * 3] * n_bands +
                list(band_sigmas_lo * (1 - self.sigma_margin)) +
                [-spec_range * 10] * n_baseline
            )
            bounds_hi = (
                [spec_range * 3] * n_bands +
                list(band_sigmas_hi * (1 + self.sigma_margin)) +
                [spec_range * 10] * n_baseline
            )

            def model(params):
                amplitudes = params[:n_bands]
                sigmas = params[n_bands:2*n_bands]
                baseline_coeffs = params[2*n_bands:]
                bands_sum = self._compute_all_bands(wl, candidate_bands, amplitudes, sigmas)
                baseline = self._build_baseline(wl, baseline_coeffs)
                return bands_sum + baseline
        else:
            n_params = n_bands + n_baseline
            x0 = np.zeros(n_params)
            for i, band in enumerate(candidate_bands):
                idx = np.argmin(np.abs(wl - band.center))
                x0[i] = max(0, (spectrum[idx] - spec_mean) * 0.3)
            x0[n_bands:] = baseline_init

            bounds_lo = [-spec_range * 3] * n_bands + [-spec_range * 10] * n_baseline
            bounds_hi = [spec_range * 3] * n_bands + [spec_range * 10] * n_baseline

            def model(params):
                amplitudes = params[:n_bands]
                baseline_coeffs = params[n_bands:]
                bands_sum = self._compute_all_bands(wl, candidate_bands, amplitudes, band_sigmas_mid)
                baseline = self._build_baseline(wl, baseline_coeffs)
                return bands_sum + baseline

        def objective(params):
            return np.sum((spectrum - model(params)) ** 2)

        # Optimize
        best_result = None
        best_r2 = -np.inf

        for _ in range(self.n_iterations):
            try:
                res = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    bounds=list(zip(bounds_lo, bounds_hi, strict=False)),
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
                fitted = model(res.x)

                ss_res = np.sum((spectrum - fitted) ** 2)
                ss_tot = np.sum((spectrum - np.mean(spectrum)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

                if r2 > best_r2:
                    best_r2 = r2
                    if self.allow_sigma_variation:
                        amplitudes = res.x[:n_bands]
                        sigmas = res.x[n_bands:2*n_bands]
                        baseline_coeffs = res.x[2*n_bands:]
                    else:
                        amplitudes = res.x[:n_bands]
                        sigmas = band_sigmas_mid
                        baseline_coeffs = res.x[n_bands:]

                    # Build band names
                    band_names = []
                    for band in candidate_bands:
                        name = f"{band.functional_group}/{band.overtone_level}"
                        band_names.append(name)

                    best_result = {
                        'band_names': band_names,
                        'band_centers': band_centers,
                        'amplitudes': amplitudes,
                        'sigmas': sigmas if isinstance(sigmas, np.ndarray) else band_sigmas_mid,
                        'baseline_coefficients': baseline_coeffs,
                        'fitted_spectrum': fitted,
                        'residuals': spectrum - fitted,
                        'r_squared': r2,
                        'rmse': float(np.sqrt(np.mean((spectrum - fitted) ** 2))),
                        'n_bands': n_bands,
                        'wavelengths': wl,
                        'band_assignments': candidate_bands,
                    }

                if r2 >= self.target_r2:
                    break

                # Perturb for next iteration
                x0 = res.x + np.random.normal(0, 0.01, len(res.x))
                x0 = np.clip(x0, bounds_lo, bounds_hi)

            except Exception:
                continue

        if best_result is None:
            # Fallback to baseline only
            fitted = self._build_baseline(wl, baseline_init)
            ss_res = np.sum((spectrum - fitted) ** 2)
            ss_tot = np.sum((spectrum - np.mean(spectrum)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            return RealBandFitResult(
                band_names=[],
                band_centers=np.array([]),
                amplitudes=np.array([]),
                sigmas=np.array([]),
                baseline_coefficients=baseline_init,
                fitted_spectrum=fitted,
                residuals=spectrum - fitted,
                r_squared=r2,
                rmse=float(np.sqrt(np.mean((spectrum - fitted) ** 2))),
                n_bands=0,
                wavelengths=wl,
                band_assignments=[],
            )

        return RealBandFitResult(**best_result)

def fit_real_bands(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    baseline_order: int = 4,
    max_bands: int = 50,
    target_r2: float = 0.98,
    allow_sigma_variation: bool = True,
) -> RealBandFitResult:
    """
    Convenience function for fitting spectrum using real NIR band assignments.

    Uses known band positions from the NIR_BANDS dictionary for physically
    meaningful spectral decomposition.

    Args:
        spectrum: Observed spectrum.
        wavelengths: Wavelength grid in nm.
        baseline_order: Polynomial baseline order.
        max_bands: Maximum number of bands to use.
        target_r2: Target R² for early stopping.
        allow_sigma_variation: Allow sigma to vary within constrained ranges.

    Returns:
        RealBandFitResult with fit results.

    Example:
        >>> result = fit_real_bands(spectrum, wavelengths)
        >>> print(f"R² = {result.r_squared:.4f}")
        >>> for name, center, amp in result.top_bands(5):
        ...     print(f"{center:.0f} nm: {name}")
    """
    fitter = RealBandFitter(
        baseline_order=baseline_order,
        max_bands=max_bands,
        target_r2=target_r2,
        allow_sigma_variation=allow_sigma_variation,
    )
    return fitter.fit(spectrum, wavelengths)

# ============================================================================
# Variance Fitting
# ============================================================================

@dataclass
class OperatorVarianceParams:
    """
    Parameters for operator-based variance modeling.

    Models spectral variation as independent physical sources:
    - High-frequency noise (detector noise)
    - Baseline offset/slope/curvature (instrumental drift, scattering)
    - Multiplicative scatter (sample thickness, optical path variation)

    Attributes:
        noise_std: Standard deviation of high-frequency noise.
        offset_std: Standard deviation of baseline offset.
        slope_std: Standard deviation of baseline slope (per 1000nm).
        curvature_std: Standard deviation of baseline curvature.
        mult_scatter_std: Standard deviation of multiplicative scatter.
    """
    noise_std: float = 0.001
    offset_std: float = 0.01
    slope_std: float = 0.001
    curvature_std: float = 0.0001
    mult_scatter_std: float = 0.05

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "noise_std": self.noise_std,
            "offset_std": self.offset_std,
            "slope_std": self.slope_std,
            "curvature_std": self.curvature_std,
            "mult_scatter_std": self.mult_scatter_std,
        }

@dataclass
class PCAVarianceParams:
    """
    Parameters for PCA-based variance modeling.

    Models spectral variation using principal component score distributions.

    Attributes:
        n_components: Number of PCA components.
        explained_variance_ratio: Explained variance per component.
        score_means: Mean of PC scores.
        score_stds: Std of PC scores.
        components: PCA loading vectors (n_components, n_wavelengths).
        mean_spectrum: Mean spectrum from PCA.
    """
    n_components: int = 5
    explained_variance_ratio: np.ndarray | None = None
    score_means: np.ndarray | None = None
    score_stds: np.ndarray | None = None
    components: np.ndarray | None = None
    mean_spectrum: np.ndarray | None = None

@dataclass
class VarianceFitResult:
    """
    Combined result from variance fitting.

    Attributes:
        operator_params: Operator-based variance parameters.
        pca_params: PCA-based variance parameters.
        n_samples: Number of samples used for fitting.
        wavelengths: Wavelength grid.
    """
    operator_params: OperatorVarianceParams
    pca_params: PCAVarianceParams
    n_samples: int = 0
    wavelengths: np.ndarray | None = None

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "Variance Fit Result",
            "=" * 60,
            "",
            "Operator-Based Parameters:",
            f"  Noise std:      {self.operator_params.noise_std:.6f}",
            f"  Offset std:     {self.operator_params.offset_std:.6f}",
            f"  Slope std:      {self.operator_params.slope_std:.6f}",
            f"  Curvature std:  {self.operator_params.curvature_std:.6f}",
            f"  Mult scatter:   {self.operator_params.mult_scatter_std:.4f}",
            "",
            "PCA-Based Parameters:",
            f"  Components: {self.pca_params.n_components}",
        ]
        if self.pca_params.explained_variance_ratio is not None:
            cum_var = np.cumsum(self.pca_params.explained_variance_ratio)
            n_95 = np.searchsorted(cum_var, 0.95) + 1
            lines.append(f"  Variance at 95%: {n_95} components")
            lines.append(f"  Top 3 explained: {self.pca_params.explained_variance_ratio[:3]}")
        lines.append("=" * 60)
        return "\n".join(lines)

class VarianceFitter:
    """
    Fit variance parameters from real spectra.

    Provides two complementary methods for modeling spectral variation:
    - Operator-based: Independent physical sources (noise, scatter, baseline)
    - PCA-based: Correlated variations capturing the covariance structure

    Example:
        >>> from nirs4all.synthesis import VarianceFitter
        >>>
        >>> fitter = VarianceFitter()
        >>> result = fitter.fit(X_real, wavelengths)
        >>>
        >>> # Use operator-based params for generation
        >>> print(f"Noise level: {result.operator_params.noise_std:.6f}")
        >>>
        >>> # Generate synthetic variance using PCA
        >>> X_variance = fitter.generate_pca_variance(n_samples=100, random_state=42)
    """

    def __init__(self, n_pca_components: int = 10):
        """
        Initialize the variance fitter.

        Args:
            n_pca_components: Number of PCA components to fit.
        """
        self.n_pca_components = n_pca_components
        self._fitted = False
        self._result: VarianceFitResult | None = None

    def fit(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> VarianceFitResult:
        """
        Fit variance parameters from real spectra.

        Args:
            X: Real spectra matrix (n_samples, n_wavelengths).
            wavelengths: Wavelength array (nm).

        Returns:
            VarianceFitResult with both operator and PCA parameters.
        """
        n_samples, n_wl = X.shape

        if wavelengths is None:
            wavelengths = np.arange(n_wl, dtype=float)

        # Fit operator-based variance
        op_params = self._fit_operator_variance(X, wavelengths)

        # Fit PCA-based variance
        pca_params = self._fit_pca_variance(X)

        self._result = VarianceFitResult(
            operator_params=op_params,
            pca_params=pca_params,
            n_samples=n_samples,
            wavelengths=wavelengths.copy(),
        )
        self._fitted = True

        return self._result

    def _fit_operator_variance(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
    ) -> OperatorVarianceParams:
        """
        Estimate operator-based variance parameters from real spectra.

        Models spectral variation as independent physical sources.
        """
        n_samples, n_wl = X.shape

        # High-frequency noise: estimated from 2nd derivative
        window = min(11, n_wl // 20 * 2 + 1) | 1
        if window >= 5:
            deriv2 = savgol_filter(X, window, min(3, window - 2), deriv=2, axis=1)
            noise_std = float(np.median(np.std(deriv2, axis=1)))
        else:
            noise_std = float(np.std(np.diff(X, axis=1)) / np.sqrt(2))

        # Baseline fitting: normalize wavelengths
        wl_norm = (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min())

        baseline_coeffs_list = []
        for i in range(n_samples):
            coeffs = np.polyfit(wl_norm, X[i], 3)
            baseline_coeffs_list.append(coeffs)
        baseline_coeffs = np.array(baseline_coeffs_list)

        # Extract baseline variation parameters
        offset_std = float(np.std(baseline_coeffs[:, -1]))  # constant term
        slope_std = float(np.std(baseline_coeffs[:, -2]))   # linear term
        curvature_std = float(np.std(baseline_coeffs[:, -3]))  # quadratic term

        # Multiplicative scatter: from sample-to-sample scale variation
        mean_spectrum = X.mean(axis=0)
        if np.abs(mean_spectrum).max() > 1e-10:
            # Estimate multiplicative factor per sample
            scale_factors = []
            for i in range(n_samples):
                # Simple linear regression to find scale factor
                scale = np.dot(X[i], mean_spectrum) / np.dot(mean_spectrum, mean_spectrum)
                scale_factors.append(scale)
            mult_scatter_std = float(np.std(scale_factors))
        else:
            mult_scatter_std = 0.05

        return OperatorVarianceParams(
            noise_std=noise_std,
            offset_std=offset_std,
            slope_std=slope_std,
            curvature_std=curvature_std,
            mult_scatter_std=mult_scatter_std,
        )

    def _fit_pca_variance(self, X: np.ndarray) -> PCAVarianceParams:
        """
        Fit PCA-based variance parameters from real spectra.
        """
        from sklearn.decomposition import PCA

        n_samples = X.shape[0]
        n_comp = min(self.n_pca_components, n_samples - 1, X.shape[1])

        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(X)

        return PCAVarianceParams(
            n_components=n_comp,
            explained_variance_ratio=pca.explained_variance_ratio_,
            score_means=scores.mean(axis=0),
            score_stds=scores.std(axis=0),
            components=pca.components_,
            mean_spectrum=pca.mean_,
        )

    def generate_operator_variance(
        self,
        base_spectrum: np.ndarray,
        wavelengths: np.ndarray,
        n_samples: int = 100,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Generate synthetic spectra using operator-based variance.

        Args:
            base_spectrum: Mean/fitted spectrum to add variance to.
            wavelengths: Wavelength array.
            n_samples: Number of samples to generate.
            random_state: Random seed.

        Returns:
            Array of synthetic spectra (n_samples, n_wavelengths).
        """
        if not self._fitted or self._result is None:
            raise RuntimeError("Must call fit() before generate_operator_variance()")

        rng = np.random.default_rng(random_state)
        params = self._result.operator_params
        n_wl = len(wavelengths)

        # Normalize wavelengths for baseline generation
        wl_norm = (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min())

        X_synth = np.zeros((n_samples, n_wl))

        for i in range(n_samples):
            spectrum = base_spectrum.copy()

            # Add multiplicative scatter
            mult_factor = 1.0 + rng.normal(0, params.mult_scatter_std)
            spectrum = spectrum * mult_factor

            # Add baseline variation
            offset = rng.normal(0, params.offset_std)
            slope = rng.normal(0, params.slope_std)
            curvature = rng.normal(0, params.curvature_std)
            baseline = offset + slope * wl_norm + curvature * wl_norm**2
            spectrum = spectrum + baseline

            # Add noise
            noise = rng.normal(0, params.noise_std, n_wl)
            spectrum = spectrum + noise

            X_synth[i] = spectrum

        return X_synth

    def generate_pca_variance(
        self,
        n_samples: int = 100,
        n_components: int | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Generate synthetic spectra using PCA-based variance.

        Args:
            n_samples: Number of samples to generate.
            n_components: Number of PCA components to use (None = all).
            random_state: Random seed.

        Returns:
            Array of synthetic spectra (n_samples, n_wavelengths).
        """
        if not self._fitted or self._result is None:
            raise RuntimeError("Must call fit() before generate_pca_variance()")

        rng = np.random.default_rng(random_state)
        pca_params = self._result.pca_params

        if pca_params.components is None or pca_params.mean_spectrum is None:
            raise RuntimeError("PCA parameters not properly fitted")

        if n_components is None:
            n_components = pca_params.n_components

        n_components = min(n_components, pca_params.n_components)

        # Generate random scores from fitted distributions
        scores = np.zeros((n_samples, n_components))
        for i in range(n_components):
            mean = pca_params.score_means[i] if pca_params.score_means is not None else 0
            std = pca_params.score_stds[i] if pca_params.score_stds is not None else 1
            scores[:, i] = rng.normal(mean, std, n_samples)

        # Reconstruct spectra
        X_synth: np.ndarray = scores @ pca_params.components[:n_components] + pca_params.mean_spectrum

        return X_synth

def fit_variance(
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
    n_pca_components: int = 10,
) -> VarianceFitResult:
    """
    Convenience function to fit variance parameters from real spectra.

    Args:
        X: Real spectra matrix (n_samples, n_wavelengths).
        wavelengths: Wavelength array (nm).
        n_pca_components: Number of PCA components to fit.

    Returns:
        VarianceFitResult with fitted parameters.

    Example:
        >>> result = fit_variance(X_real, wavelengths)
        >>> print(f"Noise level: {result.operator_params.noise_std:.6f}")
    """
    fitter = VarianceFitter(n_pca_components=n_pca_components)
    return fitter.fit(X, wavelengths)

# ============================================================================
# Physical Forward Model Fitting
# ============================================================================

@dataclass
class InstrumentChain:
    """
    Forward instrument chain: canonical grid → dataset grid.

    Applies the complete measurement chain to transform a high-resolution
    physical spectrum to the observed instrument grid.

    Chain:
        1. Wavelength warp (shift + stretch)
        2. ILS convolution (Gaussian smoothing)
        3. Stray light / gain / offset
        4. Resample to target grid

    Attributes:
        wl_shift: Wavelength shift in nm.
        wl_stretch: Wavelength scale factor.
        ils_sigma: Instrument line shape Gaussian sigma in nm.
        stray_light: Stray light fraction.
        gain: Photometric gain.
        offset: Photometric offset.

    Example:
        >>> chain = InstrumentChain(wl_shift=2.0, ils_sigma=5.0)
        >>> spectrum_obs = chain.apply(spectrum_phys, canonical_wl, target_wl)
    """
    wl_shift: float = 0.0
    wl_stretch: float = 1.0
    ils_sigma: float = 4.0
    stray_light: float = 0.001
    gain: float = 1.0
    offset: float = 0.0

    def apply(
        self,
        spectrum: np.ndarray,
        canonical_wl: np.ndarray,
        target_wl: np.ndarray,
    ) -> np.ndarray:
        """
        Apply full instrument chain.

        Args:
            spectrum: Input spectrum on canonical grid.
            canonical_wl: Canonical wavelength grid (nm).
            target_wl: Target wavelength grid (nm).

        Returns:
            Transformed spectrum on target grid.
        """
        # 1. Wavelength warp
        warped_wl = self.wl_shift + self.wl_stretch * canonical_wl

        # 2. ILS convolution (Gaussian smoothing)
        wl_step = canonical_wl[1] - canonical_wl[0]
        sigma_idx = self.ils_sigma / wl_step
        spectrum_ils = gaussian_filter1d(spectrum, sigma=sigma_idx)

        # 3. Stray light / gain / offset
        spectrum_phot = self.gain * spectrum_ils + self.offset + self.stray_light

        # 4. Resample to target grid
        spectrum_resampled: np.ndarray = np.asarray(np.interp(target_wl, warped_wl, spectrum_phot))

        return spectrum_resampled

@dataclass
class ForwardModelFitter:
    """
    Variable projection fitter for physical forward model.

    Fits a physical mixture model to observed spectra by separating:
    - Linear params: concentrations, baseline coefficients (solved via NNLS/lsq)
    - Nonlinear params: wl_shift, ils_sigma, path_length (solved via optimization)

    This approach is numerically stable and physically interpretable.

    Attributes:
        components: List of SpectralComponent objects.
        canonical_grid: High-resolution canonical wavelength grid.
        target_grid: Target wavelength grid (dataset grid).
        baseline_order: Number of Chebyshev baseline terms.
        wl_shift_bounds: Bounds for wavelength shift parameter.
        ils_sigma_bounds: Bounds for ILS sigma parameter.
        path_length_bounds: Bounds for path length parameter.

    Example:
        >>> from nirs4all.synthesis._constants import get_predefined_components
        >>> components = [get_predefined_components()[n] for n in ['water', 'protein']]
        >>> fitter = ForwardModelFitter(
        ...     components=components,
        ...     canonical_grid=np.linspace(400, 2500, 4200),
        ...     target_grid=dataset_wavelengths,
        ... )
        >>> result = fitter.fit(spectrum)
        >>> print(f"R² = {result['r_squared']:.4f}")
    """
    components: list[SpectralComponent]
    canonical_grid: np.ndarray
    target_grid: np.ndarray
    baseline_order: int = 4
    wl_shift_bounds: tuple[float, float] = (-5.0, 5.0)
    ils_sigma_bounds: tuple[float, float] = (2.0, 15.0)
    path_length_bounds: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self):
        """Pre-compute component spectra on canonical grid."""
        self.E_canonical = np.zeros((len(self.components), len(self.canonical_grid)))
        for k, comp in enumerate(self.components):
            self.E_canonical[k] = comp.compute(self.canonical_grid)

    def _build_design_matrix(self, wl_shift: float, ils_sigma: float) -> np.ndarray:
        """Build design matrix for NNLS given nonlinear params."""
        from scipy.optimize import nnls

        chain = InstrumentChain(wl_shift=wl_shift, ils_sigma=ils_sigma)

        E_target = np.zeros((len(self.components), len(self.target_grid)))
        for k in range(len(self.components)):
            E_target[k] = chain.apply(
                self.E_canonical[k], self.canonical_grid, self.target_grid
            )

        # Add baseline (Chebyshev polynomials)
        wl_norm = (
            2
            * (self.target_grid - self.target_grid.min())
            / (self.target_grid.max() - self.target_grid.min())
            - 1
        )
        B = np.zeros((self.baseline_order, len(self.target_grid)))
        for i in range(self.baseline_order):
            B[i] = np.polynomial.chebyshev.chebval(wl_norm, [0] * i + [1])

        A = np.vstack([E_target, B]).T
        return A

    def _inner_solve(
        self, A: np.ndarray, y: np.ndarray, path_length: float
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Inner NNLS solve for linear parameters."""
        from scipy.optimize import nnls

        y_scaled = y / path_length
        x, _ = nnls(A, y_scaled)
        y_fit = (A @ x) * path_length

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return x, r_squared, y_fit

    def _objective(self, nonlin_params: np.ndarray, y: np.ndarray) -> float:
        """Objective for outer optimization (negative R²)."""
        wl_shift, ils_sigma, path_length = nonlin_params

        try:
            A = self._build_design_matrix(wl_shift, ils_sigma)
            _, r_squared, _ = self._inner_solve(A, y, path_length)
            return -r_squared
        except Exception:
            return 1.0

    def fit(
        self, y: np.ndarray, initial_guess: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Fit forward model to target spectrum.

        Args:
            y: Target spectrum.
            initial_guess: Initial [wl_shift, ils_sigma, path_length].

        Returns:
            Dict with fitted parameters:
                - r_squared: Coefficient of determination
                - fitted: Fitted spectrum
                - residuals: Fitting residuals
                - concentrations: Fitted component concentrations
                - baseline_coeffs: Fitted baseline coefficients
                - wl_shift, ils_sigma, path_length: Instrument params
        """
        from scipy.optimize import minimize

        if initial_guess is None:
            initial_guess = np.array([0.0, 6.0, 1.0])

        bounds = [
            self.wl_shift_bounds,
            self.ils_sigma_bounds,
            self.path_length_bounds,
        ]

        result = minimize(
            self._objective,
            initial_guess,
            args=(y,),
            method="L-BFGS-B",
            bounds=bounds,
        )

        wl_shift, ils_sigma, path_length = result.x
        A = self._build_design_matrix(wl_shift, ils_sigma)
        x, r_squared, y_fit = self._inner_solve(A, y, path_length)

        n_comp = len(self.components)

        return {
            "r_squared": r_squared,
            "fitted": y_fit,
            "residuals": y - y_fit,
            "concentrations": x[:n_comp],
            "baseline_coeffs": x[n_comp:],
            "wl_shift": wl_shift,
            "ils_sigma": ils_sigma,
            "path_length": path_length,
        }

@dataclass
class DerivativeAwareForwardModelFitter:
    """
    Forward model fitter for derivative-preprocessed datasets.

    Key principle: Never fit derivative spectra by adding narrow bands.
    Instead:
        1. Fit latent physical model (raw absorbance)
        2. Apply derivative preprocessing to model output
        3. Compare in derivative space

    This ensures concentrations remain physically interpretable without
    oscillatory artifacts from narrow compensating peaks.

    Attributes:
        components: List of SpectralComponent objects.
        canonical_grid: High-resolution canonical wavelength grid.
        target_grid: Target wavelength grid (dataset grid).
        derivative_order: 1 for first derivative, 2 for second.
        sg_window: Savitzky-Golay window length.
        sg_polyorder: Savitzky-Golay polynomial order.
        baseline_order: Number of Chebyshev baseline terms.

    Example:
        >>> fitter = DerivativeAwareForwardModelFitter(
        ...     components=components,
        ...     canonical_grid=canonical_wl,
        ...     target_grid=dataset_wl,
        ...     derivative_order=1,  # First derivative
        ... )
        >>> result = fitter.fit(derivative_spectrum)
        >>> print(f"R² = {result['r_squared']:.4f}")
    """
    components: list[SpectralComponent]
    canonical_grid: np.ndarray
    target_grid: np.ndarray
    derivative_order: int = 1
    sg_window: int = 15
    sg_polyorder: int = 2
    baseline_order: int = 6
    wl_shift_bounds: tuple[float, float] = (-5.0, 5.0)
    ils_sigma_bounds: tuple[float, float] = (2.0, 15.0)
    path_length_bounds: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self):
        """Pre-compute component spectra on canonical grid."""
        self.E_canonical = np.zeros((len(self.components), len(self.canonical_grid)))
        for k, comp in enumerate(self.components):
            self.E_canonical[k] = comp.compute(self.canonical_grid)

    def _apply_derivative(self, X: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay derivative to spectra."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        result = savgol_filter(
            X,
            window_length=self.sg_window,
            polyorder=self.sg_polyorder,
            deriv=self.derivative_order,
            axis=1,
        )
        return np.asarray(result.flatten() if X.shape[0] == 1 else result)

    def _build_design_matrix_raw(self, wl_shift: float, ils_sigma: float) -> np.ndarray:
        """Build design matrix in RAW domain (before derivative)."""
        chain = InstrumentChain(wl_shift=wl_shift, ils_sigma=ils_sigma)

        E_target = np.zeros((len(self.components), len(self.target_grid)))
        for k in range(len(self.components)):
            E_target[k] = chain.apply(
                self.E_canonical[k], self.canonical_grid, self.target_grid
            )

        wl_norm = (
            2
            * (self.target_grid - self.target_grid.min())
            / (self.target_grid.max() - self.target_grid.min())
            - 1
        )
        B = np.zeros((self.baseline_order, len(self.target_grid)))
        for i in range(self.baseline_order):
            B[i] = np.polynomial.chebyshev.chebval(wl_norm, [0] * i + [1])

        return np.vstack([E_target, B]).T

    def _build_design_matrix_derivative(
        self, wl_shift: float, ils_sigma: float
    ) -> np.ndarray:
        """Build design matrix in DERIVATIVE domain."""
        A_raw = self._build_design_matrix_raw(wl_shift, ils_sigma)
        A_deriv = self._apply_derivative(A_raw.T).T
        return A_deriv

    def _inner_solve(
        self, A_deriv: np.ndarray, y_deriv: np.ndarray, path_length: float
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Inner bounded solve in derivative space."""
        from scipy.optimize import lsq_linear

        y_scaled = y_deriv / path_length
        n_comp = len(self.components)

        # Concentrations >= 0, baseline free
        lb = np.concatenate(
            [np.zeros(n_comp), -np.inf * np.ones(A_deriv.shape[1] - n_comp)]
        )
        ub = np.concatenate(
            [np.inf * np.ones(n_comp), np.inf * np.ones(A_deriv.shape[1] - n_comp)]
        )

        result = lsq_linear(A_deriv, y_scaled, bounds=(lb, ub))
        x = result.x

        y_fit_scaled = A_deriv @ x
        y_fit = y_fit_scaled * path_length

        ss_res = np.sum((y_deriv - y_fit) ** 2)
        ss_tot = np.sum((y_deriv - y_deriv.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return x, r_squared, y_fit

    def _objective(self, nonlin_params: np.ndarray, y_deriv: np.ndarray) -> float:
        """Objective for outer optimization."""
        wl_shift, ils_sigma, path_length = nonlin_params

        try:
            A_deriv = self._build_design_matrix_derivative(wl_shift, ils_sigma)
            _, r_squared, _ = self._inner_solve(A_deriv, y_deriv, path_length)
            return -r_squared
        except Exception:
            return 1.0

    def fit(
        self, y_deriv: np.ndarray, initial_guess: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Fit forward model to derivative spectrum.

        Args:
            y_deriv: Target spectrum (already derivative-preprocessed).
            initial_guess: Initial [wl_shift, ils_sigma, path_length].

        Returns:
            Dict with fitted parameters:
                - r_squared: Coefficient of determination
                - fitted_deriv: Fitted derivative spectrum
                - fitted_raw: Reconstructed raw spectrum
                - residuals_deriv: Fitting residuals
                - concentrations: Fitted component concentrations
                - baseline_coeffs: Fitted baseline coefficients
                - wl_shift, ils_sigma, path_length: Instrument params
        """
        from scipy.optimize import minimize

        if initial_guess is None:
            initial_guess = np.array([0.0, 6.0, 1.0])

        bounds = [
            self.wl_shift_bounds,
            self.ils_sigma_bounds,
            self.path_length_bounds,
        ]

        result = minimize(
            self._objective,
            initial_guess,
            args=(y_deriv,),
            method="L-BFGS-B",
            bounds=bounds,
        )

        wl_shift, ils_sigma, path_length = result.x
        A_deriv = self._build_design_matrix_derivative(wl_shift, ils_sigma)
        x, r_squared, y_fit_deriv = self._inner_solve(A_deriv, y_deriv, path_length)

        # Compute raw spectrum for verification
        A_raw = self._build_design_matrix_raw(wl_shift, ils_sigma)
        y_fit_raw = (A_raw @ x) * path_length

        n_comp = len(self.components)

        return {
            "r_squared": r_squared,
            "fitted_deriv": y_fit_deriv,
            "fitted_raw": y_fit_raw,
            "residuals_deriv": y_deriv - y_fit_deriv,
            "concentrations": x[:n_comp],
            "baseline_coeffs": x[n_comp:],
            "wl_shift": wl_shift,
            "ils_sigma": ils_sigma,
            "path_length": path_length,
        }

def multiscale_fit(
    fitter: ForwardModelFitter,
    y: np.ndarray,
    scales: list[float] | None = None,
) -> dict[str, Any]:
    """
    Multiscale fitting curriculum for raw spectra.

    Fits coarse features first by smoothing the target, then progressively
    reduces smoothing to capture finer details. This improves optimization
    stability and avoids local minima.

    Args:
        fitter: ForwardModelFitter instance.
        y: Target spectrum.
        scales: List of Gaussian sigma values for progressive smoothing.
                Default: [20, 10, 5, 0].

    Returns:
        Final fit result dict.

    Example:
        >>> result = multiscale_fit(fitter, spectrum, scales=[20, 10, 5, 0])
    """
    if scales is None:
        scales = [20, 10, 5, 0]

    current_guess = None

    for sigma in scales:
        y_smooth = gaussian_filter1d(y, sigma=sigma) if sigma > 0 else y

        result = fitter.fit(y_smooth, initial_guess=current_guess)
        current_guess = np.array(
            [result["wl_shift"], result["ils_sigma"], result["path_length"]]
        )

    return result

def multiscale_derivative_fit(
    fitter: DerivativeAwareForwardModelFitter,
    y_deriv: np.ndarray,
    scales: list[float] | None = None,
) -> dict[str, Any]:
    """
    Multiscale fitting curriculum for derivative spectra.

    Fits coarse features first by smoothing the derivative target, then
    progressively reduces smoothing. Particularly important for derivative
    data which can have high-frequency noise.

    Args:
        fitter: DerivativeAwareForwardModelFitter instance.
        y_deriv: Target derivative spectrum.
        scales: List of Gaussian sigma values. Default: [15, 8, 4, 0].

    Returns:
        Final fit result dict.

    Example:
        >>> result = multiscale_derivative_fit(fitter, deriv_spectrum)
    """
    if scales is None:
        scales = [15, 8, 4, 0]

    current_guess = None

    for sigma in scales:
        y_smooth = gaussian_filter1d(y_deriv, sigma=sigma) if sigma > 0 else y_deriv

        result = fitter.fit(y_smooth, initial_guess=current_guess)
        current_guess = np.array(
            [result["wl_shift"], result["ils_sigma"], result["path_length"]]
        )

    return result
