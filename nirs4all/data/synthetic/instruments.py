"""
Instrument archetype simulation for synthetic NIRS data generation.

This module provides realistic simulation of different NIR instrument types,
including their optical characteristics, noise models, and measurement
configurations. It also supports multi-sensor systems that stitch together
signal chunks from different wavelength ranges, and multi-scan averaging.

Key Features:
    - 20+ instrument archetypes covering benchtop, handheld, process, and embedded
    - Multi-sensor stitching simulation (combining multiple detector ranges)
    - Multi-scan averaging with realistic noise reduction
    - Detector-specific noise models (shot, thermal, 1/f)
    - Wavelength calibration effects
    - Stray light and etalon interference

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (2002). Near-Infrared
      Spectroscopy: Principles, Instruments, Applications. Wiley-VCH.
    - ASTM E1944-98(2017): Standard Practice for Describing and Measuring
      Performance of NIR Instruments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .wavenumber import wavenumber_to_wavelength, wavelength_to_wavenumber


class InstrumentCategory(str, Enum):
    """Categories of NIR instruments."""
    BENCHTOP = "benchtop"           # High-end laboratory instruments
    HANDHELD = "handheld"           # Portable/mobile instruments
    PROCESS = "process"             # Industrial inline/atline
    EMBEDDED = "embedded"           # MEMS-based compact modules
    FT_NIR = "ft_nir"              # Fourier-Transform NIR
    FILTER = "filter"              # Discrete filter instruments
    DIODE_ARRAY = "diode_array"    # Diode array detectors


class DetectorType(str, Enum):
    """Types of NIR detectors."""
    SI = "si"                       # Silicon (400-1100 nm)
    INGAAS = "ingaas"              # InGaAs (900-1700 nm)
    INGAAS_EXTENDED = "ingaas_ext" # Extended InGaAs (900-2500 nm)
    PBS = "pbs"                     # Lead sulfide (1000-3000 nm)
    PBSE = "pbse"                   # Lead selenide (1500-5000 nm)
    MEMS = "mems"                   # MEMS-based spectrometers
    MCT = "mct"                     # Mercury cadmium telluride (cooled)


class MonochromatorType(str, Enum):
    """Types of wavelength selection mechanisms."""
    GRATING = "grating"             # Diffraction grating
    FT = "fourier_transform"        # Interferometer (FTIR)
    FILTER_WHEEL = "filter_wheel"   # Discrete filters
    AOTF = "aotf"                   # Acousto-optic tunable filter
    LVF = "lvf"                     # Linear variable filter
    DMD = "dmd"                     # Digital micromirror device
    FABRY_PEROT = "fabry_perot"    # MEMS Fabry-Perot


@dataclass
class SensorConfig:
    """
    Configuration for a single sensor/detector in a multi-sensor system.

    Multi-sensor instruments use multiple detectors with different wavelength
    ranges, then stitch the signals together. This is common in extended-range
    instruments (e.g., 400-2500 nm coverage using Si + InGaAs detectors).

    Attributes:
        detector_type: Type of detector for this sensor.
        wavelength_range: (start, end) wavelength range in nm.
        spectral_resolution: Resolution in nm (FWHM).
        noise_level: Relative noise level (1.0 = standard).
        gain: Detector gain multiplier.
        overlap_range: Wavelength overlap with adjacent sensor for stitching (nm).
    """
    detector_type: DetectorType
    wavelength_range: Tuple[float, float]
    spectral_resolution: float = 8.0
    noise_level: float = 1.0
    gain: float = 1.0
    overlap_range: float = 20.0  # nm of overlap for smooth stitching


@dataclass
class MultiSensorConfig:
    """
    Configuration for multi-sensor spectral stitching.

    Modern NIR instruments often use multiple sensors/detectors to cover
    wide wavelength ranges. This config controls how the signals are combined.

    Attributes:
        enabled: Whether multi-sensor mode is enabled.
        sensors: List of SensorConfig for each sensor.
        stitch_method: Method for combining overlapping regions.
            Options: 'weighted', 'average', 'first', 'last', 'optimal'
        stitch_smoothing: Smoothing window (nm) at stitch boundaries.
        add_stitch_artifacts: Whether to simulate stitching artifacts.
        artifact_intensity: Intensity of stitching artifacts (0-1).
    """
    enabled: bool = False
    sensors: List[SensorConfig] = field(default_factory=list)
    stitch_method: str = "weighted"  # weighted, average, first, last, optimal
    stitch_smoothing: float = 10.0   # nm
    add_stitch_artifacts: bool = True
    artifact_intensity: float = 0.02


@dataclass
class MultiScanConfig:
    """
    Configuration for multi-scan averaging/accumulation.

    Real instruments often acquire multiple scans per sample and average
    them to improve signal-to-noise ratio. This config simulates that process.

    Attributes:
        enabled: Whether multi-scan mode is enabled.
        n_scans: Number of scans to simulate and average.
        averaging_method: How to combine scans.
            Options: 'mean', 'median', 'weighted', 'savgol'
        scan_to_scan_noise: Additional noise between scans (simulates drift).
        wavelength_jitter: Random wavelength shift between scans (nm).
        discard_outliers: Whether to discard outlier scans.
        outlier_threshold: Z-score threshold for outlier detection.
    """
    enabled: bool = False
    n_scans: int = 16
    averaging_method: str = "mean"  # mean, median, weighted, savgol
    scan_to_scan_noise: float = 0.001  # Additional noise between scans
    wavelength_jitter: float = 0.05    # nm of wavelength shift between scans
    discard_outliers: bool = False
    outlier_threshold: float = 3.0     # Z-score threshold


@dataclass
class EdgeArtifactsConfig:
    """
    Configuration for edge artifact effects in synthetic NIRS spectra.

    Edge artifacts are common in NIR spectra and arise from various sources:
    - Detector sensitivity roll-off at spectral extremes
    - Stray light contamination
    - Truncated absorption peaks at measurement boundaries
    - Baseline curvature/bending at spectrum edges

    These artifacts are well-documented in the literature:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press. Chapters 4-5.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared
      Analysis. CRC Press. Chapters on instrumentation.
    - ASTM E1944-98(2017): Standard Practice for Describing and Measuring
      Performance of NIR Instruments.

    Attributes:
        enable_detector_rolloff: Enable detector sensitivity roll-off.
        enable_stray_light: Enable stray light effects.
        enable_truncated_peaks: Enable truncated absorption peaks.
        enable_edge_curvature: Enable baseline curvature at edges.
        detector_model: Detector model for roll-off ('generic_nir', 'ingaas',
            'pbs', 'silicon_ccd'). Defaults to 'generic_nir'.
        rolloff_severity: Severity of detector roll-off (0.0-1.0).
        stray_fraction: Stray light fraction (0.0-0.02 typical).
        stray_wavelength_dependent: Whether stray light varies with wavelength.
        left_peak_amplitude: Amplitude of truncated peak at low wavelength edge.
        right_peak_amplitude: Amplitude of truncated peak at high wavelength edge.
        curvature_type: Type of edge curvature ('concave', 'convex', 'asymmetric').
        left_curvature_severity: Severity of left edge curvature (0.0-1.0).
        right_curvature_severity: Severity of right edge curvature (0.0-1.0).
    """
    # Master switches
    enable_detector_rolloff: bool = False
    enable_stray_light: bool = False
    enable_truncated_peaks: bool = False
    enable_edge_curvature: bool = False

    # Detector roll-off parameters
    detector_model: str = "generic_nir"
    rolloff_severity: float = 0.3

    # Stray light parameters
    stray_fraction: float = 0.001
    stray_wavelength_dependent: bool = True

    # Truncated peaks parameters
    left_peak_amplitude: float = 0.0
    right_peak_amplitude: float = 0.0

    # Edge curvature parameters
    curvature_type: str = "concave"  # concave, convex, asymmetric
    left_curvature_severity: float = 0.0
    right_curvature_severity: float = 0.0


@dataclass
class InstrumentArchetype:
    """
    Parameterized NIR instrument simulation.

    Represents a complete instrument model with optical, electronic, and
    measurement characteristics. Can be used to generate realistic synthetic
    spectra that match specific instrument types.

    Attributes:
        name: Instrument archetype name.
        category: Instrument category (benchtop, handheld, etc.).
        detector_type: Primary detector type.
        monochromator_type: Wavelength selection mechanism.
        wavelength_range: Nominal wavelength range (nm).
        spectral_resolution: Spectral resolution (FWHM in nm).
        wavelength_accuracy: Wavelength accuracy (nm).
        photometric_noise: Photometric noise level (AU).
        photometric_range: Photometric range (min, max AU).
        snr: Signal-to-noise ratio at 1 AU.
        stray_light: Stray light level (fraction).
        warm_up_drift: Intensity drift during warm-up (%/hour).
        temperature_sensitivity: Wavelength shift per °C.
        scan_speed: Scans per second.
        integration_time_ms: Integration time in milliseconds.
        optical_path: Optical path type ('transmission', 'reflection', etc.).
        multi_sensor: Multi-sensor configuration.
        multi_scan: Multi-scan averaging configuration.
        description: Human-readable description.
    """
    name: str
    category: InstrumentCategory
    detector_type: DetectorType
    monochromator_type: MonochromatorType
    wavelength_range: Tuple[float, float]
    spectral_resolution: float = 8.0
    wavelength_accuracy: float = 0.5
    photometric_noise: float = 0.0001  # AU
    photometric_range: Tuple[float, float] = (0.0, 3.0)
    snr: float = 10000.0
    stray_light: float = 0.0001
    warm_up_drift: float = 0.1
    temperature_sensitivity: float = 0.01  # nm/°C
    scan_speed: float = 1.0
    integration_time_ms: float = 100.0
    optical_path: str = "transmission"
    multi_sensor: MultiSensorConfig = field(default_factory=MultiSensorConfig)
    multi_scan: MultiScanConfig = field(default_factory=MultiScanConfig)
    description: str = ""

    def get_noise_model_params(self) -> Dict[str, float]:
        """Get noise model parameters based on detector type."""
        params = {
            "shot_noise_factor": 1.0,
            "thermal_noise_factor": 1.0,
            "read_noise_factor": 1.0,
            "flicker_noise_factor": 0.0,  # 1/f noise
        }

        if self.detector_type == DetectorType.SI:
            params["shot_noise_factor"] = 0.8
            params["thermal_noise_factor"] = 0.5
        elif self.detector_type == DetectorType.INGAAS:
            params["shot_noise_factor"] = 1.0
            params["thermal_noise_factor"] = 0.8
        elif self.detector_type == DetectorType.INGAAS_EXTENDED:
            params["shot_noise_factor"] = 1.2
            params["thermal_noise_factor"] = 1.2
        elif self.detector_type == DetectorType.PBS:
            params["shot_noise_factor"] = 1.5
            params["thermal_noise_factor"] = 1.8
            params["flicker_noise_factor"] = 0.3  # PbS has significant 1/f noise
        elif self.detector_type == DetectorType.MEMS:
            params["shot_noise_factor"] = 1.5
            params["thermal_noise_factor"] = 1.0
            params["read_noise_factor"] = 1.5

        return params


# ============================================================================
# Predefined Instrument Archetypes
# ============================================================================

def _create_benchtop_foss_xds() -> InstrumentArchetype:
    """FOSS XDS-style benchtop dispersive NIR."""
    return InstrumentArchetype(
        name="foss_xds",
        category=InstrumentCategory.BENCHTOP,
        detector_type=DetectorType.SI,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(400, 2500),
        spectral_resolution=0.5,
        wavelength_accuracy=0.05,
        photometric_noise=0.00005,
        snr=50000,
        stray_light=0.00005,
        scan_speed=2.0,
        multi_sensor=MultiSensorConfig(
            enabled=True,
            sensors=[
                SensorConfig(DetectorType.SI, (400, 1100), 0.5, 0.8),
                SensorConfig(DetectorType.PBS, (1100, 2500), 0.5, 1.2),
            ],
            stitch_method="weighted",
            add_stitch_artifacts=True,
            artifact_intensity=0.01,
        ),
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=32,
            averaging_method="mean",
        ),
        description="High-end benchtop dispersive NIR with Si+PbS dual detector",
    )


def _create_benchtop_bruker_mpa() -> InstrumentArchetype:
    """Bruker MPA-style FT-NIR benchtop."""
    return InstrumentArchetype(
        name="bruker_mpa",
        category=InstrumentCategory.FT_NIR,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(800, 2778),  # 12500-3600 cm⁻¹
        spectral_resolution=2.0,  # cm⁻¹ resolution
        wavelength_accuracy=0.01,
        photometric_noise=0.00003,
        snr=80000,
        stray_light=0.00001,
        scan_speed=10.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=64,
            averaging_method="mean",
        ),
        description="Research-grade FT-NIR with extended InGaAs detector",
    )


def _create_benchtop_perkin_spectrum() -> InstrumentArchetype:
    """PerkinElmer Spectrum Two-style FTIR/NIR."""
    return InstrumentArchetype(
        name="perkin_spectrum_two",
        category=InstrumentCategory.FT_NIR,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(780, 2500),
        spectral_resolution=4.0,
        wavelength_accuracy=0.02,
        photometric_noise=0.00005,
        snr=40000,
        stray_light=0.0001,
        scan_speed=4.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=32),
        description="General-purpose benchtop FT-NIR",
    )


def _create_handheld_viavi_micronir() -> InstrumentArchetype:
    """VIAVI MicroNIR-style handheld dispersive."""
    return InstrumentArchetype(
        name="viavi_micronir",
        category=InstrumentCategory.HANDHELD,
        detector_type=DetectorType.INGAAS,
        monochromator_type=MonochromatorType.LVF,
        wavelength_range=(908, 1676),
        spectral_resolution=12.0,
        wavelength_accuracy=1.0,
        photometric_noise=0.0005,
        snr=5000,
        stray_light=0.001,
        scan_speed=100.0,  # Very fast
        integration_time_ms=10.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=50,
            averaging_method="mean",
            scan_to_scan_noise=0.003,
        ),
        description="Compact handheld NIR with LVF technology",
    )


def _create_handheld_scio() -> InstrumentArchetype:
    """SCiO-style consumer handheld."""
    return InstrumentArchetype(
        name="scio",
        category=InstrumentCategory.HANDHELD,
        detector_type=DetectorType.MEMS,
        monochromator_type=MonochromatorType.FABRY_PEROT,
        wavelength_range=(740, 1070),
        spectral_resolution=15.0,
        wavelength_accuracy=2.0,
        photometric_noise=0.002,
        snr=1000,
        stray_light=0.005,
        scan_speed=200.0,
        integration_time_ms=5.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=100,
            averaging_method="median",
            scan_to_scan_noise=0.005,
        ),
        description="Consumer-grade MEMS-based miniature NIR",
    )


def _create_handheld_tellspec() -> InstrumentArchetype:
    """TellSpec-style food scanner."""
    return InstrumentArchetype(
        name="tellspec",
        category=InstrumentCategory.HANDHELD,
        detector_type=DetectorType.INGAAS,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(900, 1700),
        spectral_resolution=10.0,
        wavelength_accuracy=1.5,
        photometric_noise=0.001,
        snr=3000,
        stray_light=0.002,
        scan_speed=50.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=30),
        description="Handheld food analysis NIR scanner",
    )


def _create_handheld_linkam() -> InstrumentArchetype:
    """LinkSquare-style portable NIR."""
    return InstrumentArchetype(
        name="linksquare",
        category=InstrumentCategory.HANDHELD,
        detector_type=DetectorType.INGAAS,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(750, 1050),
        spectral_resolution=10.0,
        wavelength_accuracy=1.0,
        photometric_noise=0.0015,
        snr=2000,
        stray_light=0.003,
        scan_speed=100.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=50),
        description="Compact portable NIR for material identification",
    )


def _create_process_niro() -> InstrumentArchetype:
    """NIR-O-style process NIR probe."""
    return InstrumentArchetype(
        name="nir_o_process",
        category=InstrumentCategory.PROCESS,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(1000, 2200),
        spectral_resolution=6.0,
        wavelength_accuracy=0.5,
        photometric_noise=0.0002,
        snr=15000,
        stray_light=0.0005,
        temperature_sensitivity=0.02,
        scan_speed=5.0,
        optical_path="reflection",
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=16,
            averaging_method="mean",
            wavelength_jitter=0.1,
        ),
        description="Robust process NIR with fiber-coupled probe",
    )


def _create_process_asd_fieldspec() -> InstrumentArchetype:
    """ASD FieldSpec-style portable/process spectrometer."""
    return InstrumentArchetype(
        name="asd_fieldspec",
        category=InstrumentCategory.PROCESS,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(350, 2500),
        spectral_resolution=3.0,
        wavelength_accuracy=0.5,
        photometric_noise=0.0001,
        snr=25000,
        stray_light=0.0002,
        scan_speed=10.0,
        multi_sensor=MultiSensorConfig(
            enabled=True,
            sensors=[
                SensorConfig(DetectorType.SI, (350, 1000), 3.0, 0.6),
                SensorConfig(DetectorType.INGAAS, (1000, 1830), 8.0, 1.0),
                SensorConfig(DetectorType.INGAAS, (1830, 2500), 8.0, 1.2),
            ],
            stitch_method="weighted",
            add_stitch_artifacts=True,
            artifact_intensity=0.015,
        ),
        description="Field portable full-range spectrometer with 3 detectors",
    )


def _create_embedded_neospectra() -> InstrumentArchetype:
    """NeoSpectra Micro-style MEMS FT-NIR module."""
    return InstrumentArchetype(
        name="neospectra_micro",
        category=InstrumentCategory.EMBEDDED,
        detector_type=DetectorType.MEMS,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(1350, 2500),
        spectral_resolution=16.0,
        wavelength_accuracy=1.0,
        photometric_noise=0.001,
        snr=5000,
        stray_light=0.002,
        scan_speed=20.0,
        integration_time_ms=50.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=20,
            averaging_method="mean",
        ),
        description="Ultra-compact MEMS FT-NIR chip module",
    )


def _create_embedded_innospectra() -> InstrumentArchetype:
    """InnoSpectra-style compact MEMS spectrometer."""
    return InstrumentArchetype(
        name="innospectra",
        category=InstrumentCategory.EMBEDDED,
        detector_type=DetectorType.MEMS,
        monochromator_type=MonochromatorType.FABRY_PEROT,
        wavelength_range=(900, 1700),
        spectral_resolution=10.0,
        wavelength_accuracy=1.5,
        photometric_noise=0.0008,
        snr=4000,
        stray_light=0.003,
        scan_speed=50.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=25),
        description="Compact MEMS NIR for embedded applications",
    )


def _create_ft_thermo_antaris() -> InstrumentArchetype:
    """Thermo Antaris-style research FT-NIR."""
    return InstrumentArchetype(
        name="thermo_antaris",
        category=InstrumentCategory.FT_NIR,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(800, 2500),
        spectral_resolution=1.0,  # Very high resolution
        wavelength_accuracy=0.01,
        photometric_noise=0.00002,
        snr=100000,
        stray_light=0.00001,
        scan_speed=30.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=64,
            averaging_method="mean",
        ),
        description="High-resolution research-grade FT-NIR",
    )


def _create_ft_abb_mb3600() -> InstrumentArchetype:
    """ABB MB3600-style FT-NIR analyzer."""
    return InstrumentArchetype(
        name="abb_mb3600",
        category=InstrumentCategory.FT_NIR,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(833, 2632),  # 12000-3800 cm⁻¹
        spectral_resolution=4.0,
        wavelength_accuracy=0.02,
        photometric_noise=0.00004,
        snr=70000,
        stray_light=0.00002,
        scan_speed=20.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=32),
        description="QC/QA laboratory FT-NIR analyzer",
    )


def _create_filter_foss_infratec() -> InstrumentArchetype:
    """FOSS Infratec-style discrete filter instrument."""
    return InstrumentArchetype(
        name="foss_infratec",
        category=InstrumentCategory.FILTER,
        detector_type=DetectorType.PBS,
        monochromator_type=MonochromatorType.FILTER_WHEEL,
        wavelength_range=(850, 1050),  # Limited by filter selection
        spectral_resolution=15.0,  # Broad filter bandwidth
        wavelength_accuracy=2.0,
        photometric_noise=0.0005,
        snr=10000,
        stray_light=0.001,
        scan_speed=3.0,  # Time per filter
        multi_scan=MultiScanConfig(enabled=True, n_scans=5),
        description="Discrete filter grain analyzer",
    )


def _create_filter_perten_da7200() -> InstrumentArchetype:
    """Perten DA7200-style diode array."""
    return InstrumentArchetype(
        name="perten_da7200",
        category=InstrumentCategory.DIODE_ARRAY,
        detector_type=DetectorType.SI,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(950, 1650),
        spectral_resolution=5.0,
        wavelength_accuracy=0.3,
        photometric_noise=0.0002,
        snr=20000,
        stray_light=0.0003,
        scan_speed=40.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=10,
            averaging_method="mean",
        ),
        description="Diode array NIR for grain/food analysis",
    )


def _create_benchtop_unity() -> InstrumentArchetype:
    """Unity Scientific SpectraStar-style benchtop."""
    return InstrumentArchetype(
        name="unity_spectrastar",
        category=InstrumentCategory.BENCHTOP,
        detector_type=DetectorType.INGAAS_EXTENDED,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(680, 2500),
        spectral_resolution=5.0,
        wavelength_accuracy=0.2,
        photometric_noise=0.0001,
        snr=35000,
        stray_light=0.0002,
        scan_speed=3.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=16),
        description="Post-dispersive benchtop NIR analyzer",
    )


def _create_handheld_si_ware() -> InstrumentArchetype:
    """Si-Ware NeoSpectra Scanner-style handheld."""
    return InstrumentArchetype(
        name="siware_neoscanner",
        category=InstrumentCategory.HANDHELD,
        detector_type=DetectorType.MEMS,
        monochromator_type=MonochromatorType.FT,
        wavelength_range=(1350, 2500),
        spectral_resolution=16.0,
        wavelength_accuracy=1.5,
        photometric_noise=0.001,
        snr=4000,
        stray_light=0.003,
        scan_speed=10.0,
        multi_scan=MultiScanConfig(
            enabled=True,
            n_scans=20,
            averaging_method="mean",
            scan_to_scan_noise=0.004,
        ),
        description="MEMS FT-NIR handheld scanner",
    )


def _create_process_buchi() -> InstrumentArchetype:
    """BUCHI NIRMaster-style process NIR."""
    return InstrumentArchetype(
        name="buchi_nirmaster",
        category=InstrumentCategory.PROCESS,
        detector_type=DetectorType.INGAAS,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(1000, 2500),
        spectral_resolution=4.0,
        wavelength_accuracy=0.3,
        photometric_noise=0.0002,
        snr=25000,
        stray_light=0.0003,
        temperature_sensitivity=0.01,
        scan_speed=5.0,
        multi_scan=MultiScanConfig(enabled=True, n_scans=16),
        description="Industrial process NIR analyzer",
    )


def _create_benchtop_metrohm() -> InstrumentArchetype:
    """Metrohm NIRS DS2500-style benchtop."""
    return InstrumentArchetype(
        name="metrohm_ds2500",
        category=InstrumentCategory.BENCHTOP,
        detector_type=DetectorType.PBS,
        monochromator_type=MonochromatorType.GRATING,
        wavelength_range=(400, 2500),
        spectral_resolution=0.5,
        wavelength_accuracy=0.05,
        photometric_noise=0.00005,
        snr=50000,
        stray_light=0.00005,
        scan_speed=2.0,
        multi_sensor=MultiSensorConfig(
            enabled=True,
            sensors=[
                SensorConfig(DetectorType.SI, (400, 1100), 0.5, 0.7),
                SensorConfig(DetectorType.PBS, (1100, 2500), 0.5, 1.0),
            ],
            stitch_method="weighted",
        ),
        multi_scan=MultiScanConfig(enabled=True, n_scans=32),
        description="Vis-NIR benchtop with dual detector",
    )


# Registry of all predefined instrument archetypes
INSTRUMENT_ARCHETYPES: Dict[str, InstrumentArchetype] = {}


def _register_archetypes() -> None:
    """Register all predefined instrument archetypes."""
    global INSTRUMENT_ARCHETYPES

    creators = [
        _create_benchtop_foss_xds,
        _create_benchtop_bruker_mpa,
        _create_benchtop_perkin_spectrum,
        _create_handheld_viavi_micronir,
        _create_handheld_scio,
        _create_handheld_tellspec,
        _create_handheld_linkam,
        _create_process_niro,
        _create_process_asd_fieldspec,
        _create_embedded_neospectra,
        _create_embedded_innospectra,
        _create_ft_thermo_antaris,
        _create_ft_abb_mb3600,
        _create_filter_foss_infratec,
        _create_filter_perten_da7200,
        _create_benchtop_unity,
        _create_handheld_si_ware,
        _create_process_buchi,
        _create_benchtop_metrohm,
    ]

    for creator in creators:
        archetype = creator()
        INSTRUMENT_ARCHETYPES[archetype.name] = archetype


# Register archetypes on module load
_register_archetypes()


def get_instrument_archetype(name: str) -> InstrumentArchetype:
    """
    Get a predefined instrument archetype by name.

    Args:
        name: Instrument archetype name.

    Returns:
        InstrumentArchetype instance.

    Raises:
        KeyError: If archetype name not found.

    Example:
        >>> archetype = get_instrument_archetype("foss_xds")
        >>> print(archetype.wavelength_range)
        (400, 2500)
    """
    if name not in INSTRUMENT_ARCHETYPES:
        available = list(INSTRUMENT_ARCHETYPES.keys())
        raise KeyError(
            f"Unknown instrument archetype: '{name}'. "
            f"Available: {available}"
        )
    return INSTRUMENT_ARCHETYPES[name]


def list_instrument_archetypes(
    category: Optional[InstrumentCategory] = None
) -> List[str]:
    """
    List available instrument archetype names.

    Args:
        category: Optional filter by category.

    Returns:
        List of archetype names.

    Example:
        >>> list_instrument_archetypes(InstrumentCategory.HANDHELD)
        ['viavi_micronir', 'scio', 'tellspec', 'linksquare', 'siware_neoscanner']
    """
    if category is None:
        return list(INSTRUMENT_ARCHETYPES.keys())
    return [
        name for name, arch in INSTRUMENT_ARCHETYPES.items()
        if arch.category == category
    ]


def get_instruments_by_category() -> Dict[str, List[str]]:
    """
    Get all instruments organized by category.

    Returns:
        Dictionary mapping category name to list of instrument names.
    """
    result: Dict[str, List[str]] = {}
    for name, arch in INSTRUMENT_ARCHETYPES.items():
        cat_name = arch.category.value
        if cat_name not in result:
            result[cat_name] = []
        result[cat_name].append(name)
    return result


# ============================================================================
# Phase 6: Instrument Wavelength Grids
# ============================================================================

# Predefined wavelength grids for common NIR instruments.
# These allow generating synthetic data that exactly matches real instrument wavelengths.
INSTRUMENT_WAVELENGTHS: Dict[str, np.ndarray] = {
    # Handheld/portable instruments
    "micronir_onsite": np.linspace(908, 1676, 125),          # VIAVI MicroNIR OnSite
    "scio": np.linspace(740, 1070, 331),                     # Consumer Scio scanner
    "neospectra_micro": np.linspace(1350, 2500, 228),        # Si-Ware NeoSpectra Micro
    "linksquare": np.linspace(750, 1050, 301),               # LinkSquare portable

    # Benchtop dispersive
    "foss_xds": np.arange(400, 2498, 2),                     # FOSS XDS II (2nm step)
    "foss_nirs_ds2500": np.arange(400, 2500, 0.5),           # FOSS NIRS DS2500 (0.5nm step)

    # FT-NIR benchtop (wavenumber-based, converted to wavelength)
    "bruker_mpa": np.arange(800, 2778, 4),                   # Bruker MPA FT-NIR

    # High-resolution field portable
    "asd_fieldspec": np.arange(350, 2500, 1),                # ASD FieldSpec (1nm step)

    # Process NIR
    "abb_ftpa2000": np.arange(1000, 2500, 1),                # ABB FT-NIR process analyzer

    # Embedded/MEMS
    "texas_dlp_nirscan": np.linspace(900, 1700, 228),        # TI DLP NIRscan Nano
    "hamamatsu_c14384ma": np.linspace(1350, 2150, 256),      # Hamamatsu micro spectrometer

    # Specialty instruments
    "buchi_nirflex": np.arange(1000, 2500, 4),               # BUCHI NIRFlex FT-NIR
    "thermo_antaris": np.arange(833, 2500, 1),               # Thermo Antaris II
}


def get_instrument_wavelengths(instrument: str) -> np.ndarray:
    """
    Get the wavelength grid for a known instrument.

    Returns a copy of the predefined wavelength array for the specified
    instrument, enabling generation of synthetic data that matches real
    instrument wavelength grids exactly.

    Args:
        instrument: Instrument identifier (e.g., "micronir_onsite", "foss_xds").

    Returns:
        NumPy array of wavelengths in nm.

    Raises:
        ValueError: If the instrument is not recognized.

    Example:
        >>> wl = get_instrument_wavelengths("micronir_onsite")
        >>> print(f"MicroNIR: {len(wl)} wavelengths from {wl[0]:.0f} to {wl[-1]:.0f} nm")
        MicroNIR: 125 wavelengths from 908 to 1676 nm

        >>> # Use with SyntheticNIRSGenerator
        >>> from nirs4all.data.synthetic import SyntheticNIRSGenerator
        >>> gen = SyntheticNIRSGenerator(wavelengths=wl)
    """
    instrument = instrument.lower().replace("-", "_").replace(" ", "_")

    if instrument not in INSTRUMENT_WAVELENGTHS:
        available = list(INSTRUMENT_WAVELENGTHS.keys())
        raise ValueError(
            f"Unknown instrument: '{instrument}'. "
            f"Available instruments: {available}"
        )
    return INSTRUMENT_WAVELENGTHS[instrument].copy()


def list_instrument_wavelength_grids() -> List[str]:
    """
    List all available predefined instrument wavelength grids.

    Returns:
        List of instrument identifiers.

    Example:
        >>> grids = list_instrument_wavelength_grids()
        >>> print(grids[:3])
        ['micronir_onsite', 'scio', 'neospectra_micro']
    """
    return list(INSTRUMENT_WAVELENGTHS.keys())


def get_instrument_wavelength_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all instrument wavelength grids.

    Returns:
        Dictionary mapping instrument names to info dicts containing:
            - n_wavelengths: Number of wavelength points
            - wavelength_start: Start wavelength (nm)
            - wavelength_end: End wavelength (nm)
            - mean_step: Mean wavelength step (nm)

    Example:
        >>> info = get_instrument_wavelength_info()
        >>> print(info["micronir_onsite"])
        {'n_wavelengths': 125, 'wavelength_start': 908.0, ...}
    """
    result = {}
    for name, wl in INSTRUMENT_WAVELENGTHS.items():
        wl_arr = np.asarray(wl)
        result[name] = {
            "n_wavelengths": len(wl_arr),
            "wavelength_start": float(wl_arr[0]),
            "wavelength_end": float(wl_arr[-1]),
            "mean_step": float(np.mean(np.diff(wl_arr))) if len(wl_arr) > 1 else 0.0,
        }
    return result


# ============================================================================
# Instrument Simulation
# ============================================================================

class InstrumentSimulator:
    """
    Apply instrument-specific effects to synthetic spectra.

    Simulates the complete instrument response including:
    - Spectral resolution (instrumental broadening)
    - Multi-sensor stitching
    - Multi-scan averaging
    - Detector noise (shot, thermal, 1/f)
    - Wavelength calibration errors
    - Stray light effects
    - Etalon/fringing interference

    Attributes:
        archetype: The instrument archetype being simulated.
        rng: Random number generator for reproducibility.

    Example:
        >>> archetype = get_instrument_archetype("viavi_micronir")
        >>> simulator = InstrumentSimulator(archetype, random_state=42)
        >>> spectra_out = simulator.apply(spectra, wavelengths)
    """

    def __init__(
        self,
        archetype: InstrumentArchetype,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the instrument simulator.

        Args:
            archetype: Instrument archetype to simulate.
            random_state: Random seed for reproducibility.
        """
        self.archetype = archetype
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def apply(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        temperature_offset: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all instrument effects to spectra.

        Args:
            spectra: Input spectra array (n_samples, n_wavelengths).
            wavelengths: Wavelength array in nm.
            temperature_offset: Temperature deviation from calibration (°C).

        Returns:
            Tuple of (modified_spectra, output_wavelengths).
            Output wavelengths may differ if resampled to instrument grid.
        """
        # Start with input spectra
        result = spectra.copy()
        output_wl = wavelengths.copy()

        # 1. Resample to instrument wavelength range if needed
        result, output_wl = self._resample_to_instrument_range(result, wavelengths)

        # 2. Apply multi-sensor stitching effects
        if self.archetype.multi_sensor.enabled:
            result = self._apply_multi_sensor_effects(result, output_wl)

        # 3. Apply instrumental broadening (spectral resolution)
        result = self._apply_instrumental_broadening(result, output_wl)

        # 4. Apply wavelength calibration effects
        result = self._apply_wavelength_effects(result, output_wl, temperature_offset)

        # 5. Apply stray light
        result = self._apply_stray_light(result)

        # 6. Apply multi-scan simulation
        if self.archetype.multi_scan.enabled:
            result = self._apply_multi_scan_averaging(result, output_wl)
        else:
            # Apply detector noise (single scan)
            result = self._apply_detector_noise(result, output_wl)

        # 7. Apply photometric range limiting
        result = self._apply_photometric_range(result)

        return result, output_wl

    def _resample_to_instrument_range(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample spectra to instrument wavelength range."""
        wl_min, wl_max = self.archetype.wavelength_range

        # Determine output wavelength grid
        step = np.median(np.diff(wavelengths))
        grid_start = max(wl_min, wavelengths.min())
        grid_end = min(wl_max, wavelengths.max())
        output_wl = np.arange(grid_start, grid_end + step / 2, step)

        # Ensure we don't exceed instrument range
        output_wl = output_wl[output_wl <= wl_max]

        # Interpolate if dimensions don't match or wavelengths differ
        if len(wavelengths) != len(output_wl) or not np.allclose(wavelengths, output_wl):
            result = np.zeros((spectra.shape[0], len(output_wl)))
            for i in range(spectra.shape[0]):
                result[i] = np.interp(output_wl, wavelengths, spectra[i])
            return result, output_wl

        # Mask to instrument range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        return spectra[:, mask], wavelengths[mask]

    def _apply_instrumental_broadening(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Apply spectral resolution broadening via Gaussian convolution."""
        fwhm = self.archetype.spectral_resolution
        step = np.median(np.diff(wavelengths))

        # Convert FWHM to sigma in pixel units
        sigma_pts = (fwhm / 2.355) / step

        if sigma_pts < 0.5:
            return spectra  # No significant broadening needed

        result = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            result[i] = gaussian_filter1d(spectra[i], sigma_pts)

        return result

    def _apply_multi_sensor_effects(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Apply multi-sensor stitching simulation."""
        config = self.archetype.multi_sensor
        n_samples, n_wl = spectra.shape
        result = spectra.copy()

        for sensor in config.sensors:
            # Get mask for this sensor's range
            wl_min, wl_max = sensor.wavelength_range
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)

            if not np.any(mask):
                continue

            # Apply sensor-specific gain variation
            gain_variation = self.rng.normal(sensor.gain, sensor.gain * 0.01, n_samples)
            result[:, mask] *= gain_variation[:, np.newaxis]

            # Apply sensor-specific noise level
            noise_scale = sensor.noise_level * self.archetype.photometric_noise
            result[:, mask] += self.rng.normal(0, noise_scale, (n_samples, mask.sum()))

        # Add stitching artifacts at sensor boundaries
        if config.add_stitch_artifacts:
            result = self._add_stitch_artifacts(
                result, wavelengths, config
            )

        return result

    def _add_stitch_artifacts(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        config: MultiSensorConfig
    ) -> np.ndarray:
        """Add artifacts at sensor stitch boundaries."""
        result = spectra.copy()
        step = np.median(np.diff(wavelengths))

        for i, sensor in enumerate(config.sensors[:-1]):
            next_sensor = config.sensors[i + 1]

            # Find stitch point (end of current sensor range)
            stitch_wl = sensor.wavelength_range[1]
            stitch_idx = np.argmin(np.abs(wavelengths - stitch_wl))

            # Define transition region
            half_width = int(config.stitch_smoothing / step / 2)
            start_idx = max(0, stitch_idx - half_width)
            end_idx = min(len(wavelengths), stitch_idx + half_width)

            # Add small offset artifact
            artifact_offset = self.rng.normal(
                0, config.artifact_intensity, spectra.shape[0]
            )

            # Create smooth transition for artifact
            x = np.linspace(0, 1, end_idx - start_idx)
            transition = 0.5 * (1 - np.cos(np.pi * x))  # Smooth S-curve

            for sample_idx in range(spectra.shape[0]):
                result[sample_idx, start_idx:end_idx] += (
                    artifact_offset[sample_idx] * transition
                )

        return result

    def _apply_wavelength_effects(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        temperature_offset: float
    ) -> np.ndarray:
        """Apply wavelength calibration effects."""
        n_samples = spectra.shape[0]
        result = np.zeros_like(spectra)

        for i in range(n_samples):
            # Random wavelength shift within accuracy specification
            shift = self.rng.normal(0, self.archetype.wavelength_accuracy)

            # Add temperature-induced shift
            shift += temperature_offset * self.archetype.temperature_sensitivity

            # Random stretch (ppm-level)
            stretch = self.rng.normal(1.0, 0.0001)

            # Apply shift and stretch via interpolation
            wl_shifted = stretch * wavelengths + shift
            result[i] = np.interp(wavelengths, wl_shifted, spectra[i])

        return result

    def _apply_stray_light(self, spectra: np.ndarray) -> np.ndarray:
        """Apply stray light offset."""
        stray_offset = self.archetype.stray_light

        # Stray light appears as a constant offset that varies slightly
        offset = self.rng.normal(
            stray_offset,
            stray_offset * 0.2,
            spectra.shape[0]
        )

        return spectra + offset[:, np.newaxis]

    def _apply_multi_scan_averaging(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Simulate multi-scan acquisition and averaging."""
        config = self.archetype.multi_scan
        n_samples, n_wl = spectra.shape
        result = np.zeros_like(spectra)

        for sample_idx in range(n_samples):
            # Generate multiple scans
            scans = np.zeros((config.n_scans, n_wl))

            for scan_idx in range(config.n_scans):
                # Start with base spectrum
                scan = spectra[sample_idx].copy()

                # Add scan-to-scan noise
                scan += self.rng.normal(0, config.scan_to_scan_noise, n_wl)

                # Add wavelength jitter
                if config.wavelength_jitter > 0:
                    jitter = self.rng.normal(0, config.wavelength_jitter)
                    scan = np.interp(
                        wavelengths,
                        wavelengths + jitter,
                        scan
                    )

                # Add detector noise for this scan
                scan = self._apply_detector_noise_single(scan, wavelengths)

                scans[scan_idx] = scan

            # Discard outlier scans if configured
            if config.discard_outliers:
                # Z-score based outlier detection
                mean_scan = np.mean(scans, axis=0)
                std_scan = np.std(scans, axis=0)

                valid_mask = np.ones(config.n_scans, dtype=bool)
                for scan_idx in range(config.n_scans):
                    z_scores = np.abs((scans[scan_idx] - mean_scan) / (std_scan + 1e-10))
                    if np.mean(z_scores) > config.outlier_threshold:
                        valid_mask[scan_idx] = False

                scans = scans[valid_mask]

            # Average scans
            if config.averaging_method == "mean":
                result[sample_idx] = np.mean(scans, axis=0)
            elif config.averaging_method == "median":
                result[sample_idx] = np.median(scans, axis=0)
            elif config.averaging_method == "weighted":
                # Weight by inverse variance
                weights = 1.0 / (np.var(scans, axis=1) + 1e-10)
                weights /= weights.sum()
                result[sample_idx] = np.average(scans, axis=0, weights=weights)
            else:
                result[sample_idx] = np.mean(scans, axis=0)

        return result

    def _apply_detector_noise(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Apply detector noise to all spectra."""
        result = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            result[i] = self._apply_detector_noise_single(spectra[i], wavelengths)
        return result

    def _apply_detector_noise_single(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray
    ) -> np.ndarray:
        """Apply detector noise to a single spectrum."""
        noise_params = self.archetype.get_noise_model_params()
        base_noise = self.archetype.photometric_noise

        n_wl = len(wavelengths)
        total_noise = np.zeros(n_wl)

        # Shot noise (signal-dependent)
        shot = noise_params["shot_noise_factor"] * base_noise
        total_noise += self.rng.normal(0, shot * np.sqrt(np.abs(spectrum) + 0.01))

        # Thermal noise (constant)
        thermal = noise_params["thermal_noise_factor"] * base_noise
        total_noise += self.rng.normal(0, thermal, n_wl)

        # Read noise (constant)
        read = noise_params["read_noise_factor"] * base_noise * 0.5
        total_noise += self.rng.normal(0, read, n_wl)

        # 1/f (flicker) noise - correlated
        if noise_params["flicker_noise_factor"] > 0:
            flicker = noise_params["flicker_noise_factor"] * base_noise
            # Generate correlated noise with 1/f spectrum
            pink_noise = self._generate_pink_noise(n_wl, flicker)
            total_noise += pink_noise

        return spectrum + total_noise

    def _generate_pink_noise(self, n_points: int, amplitude: float) -> np.ndarray:
        """Generate 1/f (pink) noise."""
        white = self.rng.normal(0, 1, n_points)

        # Create 1/f filter in frequency domain
        freqs = np.fft.fftfreq(n_points)
        freqs[0] = 1e-10  # Avoid division by zero
        fft_filter = 1.0 / np.sqrt(np.abs(freqs))
        fft_filter[0] = 0  # Remove DC component

        # Apply filter
        pink_fft = np.fft.fft(white) * fft_filter
        pink = np.real(np.fft.ifft(pink_fft))

        return pink * amplitude

    def _apply_photometric_range(self, spectra: np.ndarray) -> np.ndarray:
        """Clip spectra to instrument photometric range."""
        pmin, pmax = self.archetype.photometric_range
        return np.clip(spectra, pmin, pmax)


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Enums
    "InstrumentCategory",
    "DetectorType",
    "MonochromatorType",
    # Configuration dataclasses
    "SensorConfig",
    "MultiSensorConfig",
    "MultiScanConfig",
    "EdgeArtifactsConfig",
    "InstrumentArchetype",
    # Registry
    "INSTRUMENT_ARCHETYPES",
    "get_instrument_archetype",
    "list_instrument_archetypes",
    "get_instruments_by_category",
    # Phase 6: Instrument wavelength grids
    "INSTRUMENT_WAVELENGTHS",
    "get_instrument_wavelengths",
    "list_instrument_wavelength_grids",
    "get_instrument_wavelength_info",
    # Simulator
    "InstrumentSimulator",
]
