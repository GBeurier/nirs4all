"""
D09 - Synthetic Generator: Instrument Simulation
=================================================

Learn to simulate realistic NIR instrument effects in synthetic spectra.

This tutorial covers:

* Selecting instrument archetypes (FOSS, Bruker, SCiO, etc.)
* Understanding detector types and their characteristics
* Multi-sensor stitching for extended wavelength range instruments
* Multi-scan averaging for improved SNR
* Measurement mode effects (transmittance, reflectance, ATR)

Prerequisites
-------------
Complete D05 and D06 first.

Duration: ~8 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np

# NIRS4All imports
from nirs4all.synthesis import (
    ATRConfig,
    DetectorConfig,
    # Detectors
    DetectorSimulator,
    DetectorType,
    InstrumentArchetype,
    # Instrument simulation (Phase 2)
    InstrumentCategory,
    InstrumentSimulator,
    # Measurement modes
    MeasurementMode,
    MeasurementModeSimulator,
    MonochromatorType,
    MultiScanConfig,
    MultiSensorConfig,
    NoiseModelConfig,
    ReflectanceConfig,
    SensorConfig,
    SyntheticNIRSGenerator,
    TransmittanceConfig,
    get_detector_response,
    get_instrument_archetype,
    get_instruments_by_category,
    list_detector_types,
    list_instrument_archetypes,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D09 Instrument Simulation Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "D09_synthetic_instruments"

# =============================================================================
# Section 1: Exploring Instrument Archetypes
# =============================================================================
print("\n" + "=" * 60)
print("D09 - Instrument Simulation for Synthetic Spectra")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Available Instrument Archetypes")
print("-" * 60)

# List all available instruments
all_instruments = list_instrument_archetypes()
print(f"\n📋 Available instrument archetypes: {len(all_instruments)}")
for name in sorted(all_instruments)[:10]:
    arch = get_instrument_archetype(name)
    print(f"   • {name}: {arch.description[:50]}...")

# Group instruments by category
by_category = get_instruments_by_category()
print("\n📊 Instruments by category:")
for cat, instruments in by_category.items():
    print(f"   {cat}: {', '.join(instruments)}")

# =============================================================================
# Section 2: Understanding Instrument Properties
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Instrument Archetype Properties")
print("-" * 60)

# Get a high-end benchtop instrument
foss_xds = get_instrument_archetype("foss_xds")

print("\n🔬 FOSS XDS Properties:")
print(f"   Category: {foss_xds.category.value}")
print(f"   Detector: {foss_xds.detector_type.value}")
print(f"   Monochromator: {foss_xds.monochromator_type.value}")
print(f"   Wavelength range: {foss_xds.wavelength_range[0]}-{foss_xds.wavelength_range[1]} nm")
print(f"   Spectral resolution: {foss_xds.spectral_resolution} nm")
print(f"   SNR: {foss_xds.snr}")
print(f"   Multi-sensor: {foss_xds.multi_sensor.enabled}")
print(f"   Multi-scan: {foss_xds.multi_scan.enabled} ({foss_xds.multi_scan.n_scans} scans)")

# Compare with a handheld device
scio = get_instrument_archetype("scio")
print("\n📱 SCiO (Consumer Handheld) Properties:")
print(f"   Category: {scio.category.value}")
print(f"   Detector: {scio.detector_type.value}")
print(f"   Monochromator: {scio.monochromator_type.value}")
print(f"   Wavelength range: {scio.wavelength_range[0]}-{scio.wavelength_range[1]} nm")
print(f"   Spectral resolution: {scio.spectral_resolution} nm")
print(f"   SNR: {scio.snr}")

# =============================================================================
# Section 3: Multi-Sensor Configuration (Spectral Stitching)
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Multi-Sensor Configuration")
print("-" * 60)

print("""
Multi-sensor instruments use multiple detectors to cover wide wavelength ranges.
For example, FOSS XDS uses:
  - Silicon (Si) detector: 400-1100 nm
  - Lead Sulfide (PbS) detector: 1100-2500 nm

The signals are stitched together in the overlap region.
""")

# Examine FOSS XDS multi-sensor config
ms_config = foss_xds.multi_sensor
print("🔗 FOSS XDS Multi-Sensor Configuration:")
print(f"   Enabled: {ms_config.enabled}")
print(f"   Number of sensors: {len(ms_config.sensors)}")
print(f"   Stitch method: {ms_config.stitch_method}")
print(f"   Stitch smoothing: {ms_config.stitch_smoothing} nm")
print(f"   Add stitch artifacts: {ms_config.add_stitch_artifacts}")

for i, sensor in enumerate(ms_config.sensors):
    print(f"\n   Sensor {i+1}:")
    print(f"      Detector type: {sensor.detector_type.value}")
    print(f"      Range: {sensor.wavelength_range[0]}-{sensor.wavelength_range[1]} nm")
    print(f"      Resolution: {sensor.spectral_resolution} nm")

# Create custom multi-sensor config
custom_multi_sensor = MultiSensorConfig(
    enabled=True,
    sensors=[
        SensorConfig(DetectorType.SI, (400, 1000), spectral_resolution=2.0),
        SensorConfig(DetectorType.INGAAS, (950, 1700), spectral_resolution=4.0),
        SensorConfig(DetectorType.INGAAS_EXTENDED, (1600, 2500), spectral_resolution=6.0),
    ],
    stitch_method="weighted",  # Options: 'weighted', 'average', 'first', 'last', 'optimal'
    stitch_smoothing=15.0,
    add_stitch_artifacts=True,
    artifact_intensity=0.02,
)
print("\n✨ Created custom 3-sensor configuration (400-2500 nm)")

# =============================================================================
# Section 4: Multi-Scan Averaging
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Multi-Scan Averaging")
print("-" * 60)

print("""
Real instruments take multiple scans and average them to:
  - Reduce random noise (SNR improves by √n)
  - Detect and discard outliers
  - Handle wavelength drift between scans
""")

# Examine multi-scan config
scan_config = foss_xds.multi_scan
print("📈 FOSS XDS Multi-Scan Configuration:")
print(f"   Enabled: {scan_config.enabled}")
print(f"   Number of scans: {scan_config.n_scans}")
print(f"   Averaging method: {scan_config.averaging_method}")
print(f"   Scan-to-scan noise: {scan_config.scan_to_scan_noise}")
print(f"   Wavelength jitter: {scan_config.wavelength_jitter} nm")
print(f"   Discard outliers: {scan_config.discard_outliers}")

# Create custom multi-scan config
custom_multi_scan = MultiScanConfig(
    enabled=True,
    n_scans=64,
    averaging_method="median",  # Options: 'mean', 'median', 'weighted', 'savgol'
    scan_to_scan_noise=0.002,
    wavelength_jitter=0.1,
    discard_outliers=True,
    outlier_threshold=2.5,  # Z-score threshold
)
print("\n✨ Created custom multi-scan config (64 scans with outlier removal)")

# =============================================================================
# Section 5: Using Instruments with the Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Generating Spectra with Instrument Effects")
print("-" * 60)

# Generate baseline spectra (no instrument effects)
gen_ideal = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2000,
    wavelength_step=4,
    random_state=42,
)
X_ideal, conc_ideal, pure_ideal, *_rest_ideal = gen_ideal.generate(
    n_samples=50,
    include_instrument_effects=False,
    return_metadata=True
)
meta_ideal: dict[str, Any] = _rest_ideal[0] if _rest_ideal else {}
print(f"\n🎯 Generated ideal spectra: {X_ideal.shape}")

# Generate with high-end instrument
gen_foss = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2000,
    wavelength_step=4,
    instrument="foss_xds",  # Use FOSS XDS instrument model
    random_state=42,
)
X_foss, _, _, *_rest_foss = gen_foss.generate(n_samples=50, return_metadata=True)
meta_foss: dict[str, Any] = _rest_foss[0] if _rest_foss else {}
print(f"🔬 Generated FOSS XDS spectra: {X_foss.shape}")
print(f"   Multi-sensor applied: {meta_foss.get('multi_sensor', 'N/A')}")
print(f"   Multi-scan applied: {meta_foss.get('multi_scan', 'N/A')}")

# Generate with handheld device (more noise)
gen_scio = SyntheticNIRSGenerator(
    wavelength_start=740,
    wavelength_end=1070,
    wavelength_step=2,
    instrument="scio",
    random_state=42,
)
X_scio, _, _, *_rest_scio = gen_scio.generate(n_samples=50, return_metadata=True)
meta_scio: dict[str, Any] = _rest_scio[0] if _rest_scio else {}
print(f"📱 Generated SCiO spectra: {X_scio.shape}")

# =============================================================================
# Section 6: Detector Response Curves
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Detector Response Curves")
print("-" * 60)

print(f"\n🔍 Available detector types: {list_detector_types()}")

# Get detector responses
wavelengths = np.linspace(400, 2500, 1000)

detector_types = [DetectorType.SI, DetectorType.INGAAS, DetectorType.PBS]
for det_type in detector_types:
    response = get_detector_response(det_type)
    responsivity = response.get_response_at(wavelengths)
    print(f"\n   {det_type.value.upper()}:")
    print(f"      Sensitivity range: {response.short_cutoff:.0f}-{response.cutoff_wavelength:.0f} nm")
    print(f"      Peak wavelength: {response.peak_wavelength:.0f} nm")
    print(f"      Peak QE: {response.peak_qe:.2f}")

# =============================================================================
# Section 7: Measurement Modes
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Measurement Mode Effects")
print("-" * 60)

print(f"\n📐 Available measurement modes: {[m.value for m in MeasurementMode]}")

# Create sample spectra for testing
sample_wl = np.linspace(1000, 2000, 500)
sample_spectra = np.random.default_rng(42).normal(0.5, 0.1, (10, 500))

# Import the full config class
from nirs4all.synthesis.measurement_modes import MeasurementModeConfig

# Transmittance mode
trans_config = MeasurementModeConfig(
    mode=MeasurementMode.TRANSMITTANCE,
    transmittance=TransmittanceConfig(path_length_mm=10.0)
)
trans_sim = MeasurementModeSimulator(config=trans_config, random_state=42)
trans_spectra = trans_sim.apply(sample_spectra.copy(), sample_wl)
print("\n📊 Transmittance simulation:")
print(f"   Path length: {trans_config.transmittance.path_length_mm} mm")
print(f"   Output range: {trans_spectra.min():.3f} to {trans_spectra.max():.3f}")

# Reflectance mode
refl_config = MeasurementModeConfig(
    mode=MeasurementMode.REFLECTANCE,
    reflectance=ReflectanceConfig(geometry="integrating_sphere", sample_presentation="powder")
)
refl_sim = MeasurementModeSimulator(config=refl_config, random_state=42)
refl_spectra = refl_sim.apply(sample_spectra.copy(), sample_wl)
print("\n📊 Reflectance simulation:")
print(f"   Geometry: {refl_config.reflectance.geometry}")
print(f"   Output range: {refl_spectra.min():.3f} to {refl_spectra.max():.3f}")

# ATR mode (common for solids/pastes)
atr_config = MeasurementModeConfig(
    mode=MeasurementMode.ATR,
    atr=ATRConfig(crystal_material="diamond", incidence_angle=45.0)
)
atr_sim = MeasurementModeSimulator(config=atr_config, random_state=42)
atr_spectra = atr_sim.apply(sample_spectra.copy(), sample_wl)
print("\n📊 ATR simulation:")
print(f"   Crystal: {atr_config.atr.crystal_material}")
print(f"   Angle: {atr_config.atr.incidence_angle}°")
print(f"   Output range: {atr_spectra.min():.3f} to {atr_spectra.max():.3f}")

# =============================================================================
# Section 8: Creating Custom Instruments
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Creating Custom Instrument Archetypes")
print("-" * 60)

# Create a custom instrument archetype
custom_instrument = InstrumentArchetype(
    name="custom_benchtop",
    category=InstrumentCategory.BENCHTOP,
    detector_type=DetectorType.INGAAS_EXTENDED,
    monochromator_type=MonochromatorType.GRATING,
    wavelength_range=(900, 2500),
    spectral_resolution=4.0,
    wavelength_accuracy=0.1,
    photometric_noise=0.0001,
    snr=30000,
    stray_light=0.0001,
    multi_sensor=custom_multi_sensor,
    multi_scan=custom_multi_scan,
    description="Custom high-performance extended-range benchtop",
)

print("\n🛠️ Created custom instrument archetype:")
print(f"   Name: {custom_instrument.name}")
print(f"   Range: {custom_instrument.wavelength_range}")
print(f"   Resolution: {custom_instrument.spectral_resolution} nm")
print(f"   SNR: {custom_instrument.snr}")
print(f"   Sensors: {len(custom_instrument.multi_sensor.sensors)}")
print(f"   Scans: {custom_instrument.multi_scan.n_scans}")

# Use the custom instrument
custom_sim = InstrumentSimulator(custom_instrument, random_state=42)
custom_spectra, custom_wl = custom_sim.apply(sample_spectra.copy(), sample_wl)
print("\n📊 Applied custom instrument effects:")
print(f"   Input shape: {sample_spectra.shape}")
print(f"   Output shape: {custom_spectra.shape}")
print(f"   Output wavelength range: {custom_wl.min():.0f}-{custom_wl.max():.0f} nm")

# =============================================================================
# Section 9: Comparing Instrument Quality Levels
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Comparing Instrument Quality")
print("-" * 60)

# Generate spectra with different instrument qualities
generators = {
    "Ideal (no noise)": SyntheticNIRSGenerator(
        wavelength_start=1000, wavelength_end=1700, wavelength_step=2,
        random_state=42
    ),
    "High-end (Bruker)": SyntheticNIRSGenerator(
        wavelength_start=1000, wavelength_end=1700, wavelength_step=2,
        instrument="bruker_mpa", random_state=42
    ),
    "Mid-range (Viavi)": SyntheticNIRSGenerator(
        wavelength_start=1000, wavelength_end=1700, wavelength_step=2,
        instrument="viavi_micronir", random_state=42
    ),
}

print("\n📊 Spectral noise comparison (std of differences from mean):")
quality_results = {}
for name, gen in generators.items():
    if name == "Ideal (no noise)":
        _gen_out = gen.generate(n_samples=20, include_instrument_effects=False, return_metadata=True)
    else:
        _gen_out = gen.generate(n_samples=20, return_metadata=True)
    X = _gen_out[0]
    # Compute residual noise as deviation from sample mean
    mean_spectrum = X.mean(axis=0)
    residuals = X - mean_spectrum
    noise_std = residuals.std()
    print(f"   {name}: {noise_std:.6f}")
    quality_results[name] = X

# =============================================================================
# Section 10: Plotting (optional)
# =============================================================================
if args.plots:
    from example_utils import get_examples_output_dir
    output_path = get_examples_output_dir() / EXAMPLE_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Multi-sensor detector responses
    fig, ax = plt.subplots(figsize=(10, 6))
    wavelengths_det = np.linspace(400, 2800, 1000)

    for det_type in [DetectorType.SI, DetectorType.INGAAS, DetectorType.PBS]:
        response = get_detector_response(det_type)
        responsivity = response.get_response_at(wavelengths_det)
        ax.plot(wavelengths_det, responsivity, label=det_type.value.upper(), linewidth=2)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative Responsivity")
    ax.set_title("Detector Response Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_path / "detector_responses.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved detector response plot: {plot_path}")

    # Plot 2: Instrument quality comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    wavelengths_plot = np.linspace(1000, 1700, quality_results["Ideal (no noise)"].shape[1])

    for ax, (name, X) in zip(axes, quality_results.items(), strict=False):
        for i in range(min(5, len(X))):
            ax.plot(wavelengths_plot, X[i], alpha=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "instrument_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved instrument comparison plot: {plot_path}")

    if args.show:
        plt.show()

    print_output_location(output_path)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Key concepts covered:

1. 🔬 Instrument Archetypes: Pre-defined models of real instruments
   - 19 archetypes covering benchtop, handheld, process, etc.
   - Each with realistic optical and electronic characteristics

2. 🔗 Multi-Sensor Stitching: For extended wavelength range
   - Multiple detectors with different sensitivities
   - Smooth stitching in overlap regions
   - Optional artifact simulation

3. 📈 Multi-Scan Averaging: For improved SNR
   - Multiple scans averaged together
   - Outlier detection and removal
   - Wavelength jitter simulation

4. 📐 Measurement Modes: Physical sampling effects
   - Transmittance, reflectance, ATR, transflectance
   - Proper physics-based transformations

5. 🎛️ Detectors: Response curves and noise models
   - Si, InGaAs, PbS, MCT, MEMS detectors
   - Shot noise, thermal noise, dark current

Next steps:
- R05: Environmental and matrix effects simulation
- R06: Validation and quality assessment
- R07: Fitting generators to real data
""")

print("\n✅ Example completed successfully!")
