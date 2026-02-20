"""
R05 - Synthetic Generator: Environmental and Matrix Effects
============================================================

Comprehensive reference for environmental and matrix effects simulation
in synthetic NIR spectra (Phase 3 features).

This reference covers:

* Temperature effects on spectral bands (O-H, N-H, C-H shifts)
* Moisture and water activity effects
* Particle size effects (EMSC-style scattering)
* Combined environmental and matrix effect simulation

Prerequisites
-------------
Complete D05-D09 synthetic generator developer examples first.

Duration: ~10 minutes
Difficulty: *****

Scientific Background
---------------------
Temperature Effects:
    O-H stretching bands shift ~0.11 nm/C to lower wavelengths with increasing
    temperature (hydrogen bond weakening). N-H bands show similar but smaller
    shifts. C-H bands are relatively temperature-insensitive.

Moisture Effects:
    Water activity affects the intensity and position of water bands at 1450 nm,
    1940 nm, and 2500 nm. Free vs. bound water shows different spectral signatures.

Particle Size Effects:
    Smaller particles cause more scattering (higher baseline, wavelength-dependent).
    This is corrected by EMSC, SNV, or MSC preprocessing in real applications.

References:
    - Maeda et al. (1995): Temperature effects on NIR water bands
    - Segtnan et al. (2001): Water structure and temperature
    - Martens & Naes (1989): Multivariate calibration and scattering
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np

# Operators for applying effects
from nirs4all.operators.augmentation import (
    EMSCDistortionAugmenter,
    MoistureAugmenter,
    ParticleSizeAugmenter,
    TemperatureAugmenter,
)

# NIRS4All imports
from nirs4all.synthesis import (
    TEMPERATURE_EFFECT_PARAMS,
    EMSCConfig,
    EnvironmentalEffectsConfig,
    MoistureConfig,
    ParticleSizeConfig,
    ParticleSizeDistribution,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
    # Phase 3: Scattering effects configuration
    ScatteringModel,
    # Phase 3: Environmental effects configuration
    SpectralRegion,
    SyntheticNIRSGenerator,
    TemperatureConfig,
    TemperatureEffectParams,
    get_temperature_effect_regions,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from example_utils import get_examples_output_dir, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='R05 Environmental Effects Reference')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "R05_synthetic_environmental"

# =============================================================================
# Section 1: Temperature Effects Overview
# =============================================================================
print("\n" + "=" * 60)
print("R05 - Environmental and Matrix Effects for Synthetic Spectra")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Temperature Effect Parameters")
print("-" * 60)

print("""
Temperature affects NIR spectra through:
  1. Peak position shifts (hydrogen bond weakening)
  2. Peak width changes (bandwidth broadening)
  3. Intensity variations (absorbance changes)

Different functional groups respond differently:
""")

# Examine temperature effect parameters
for region in SpectralRegion:
    params = TEMPERATURE_EFFECT_PARAMS[region]
    print(f"\n  {region.value.upper()} Region ({params.wavelength_range[0]}-{params.wavelength_range[1]} nm):")
    print(f"   Position shift: {params.shift_per_degree:.4f} nm/C")
    print(f"   Bandwidth change: {params.broadening_per_degree:.4f} per C")
    print(f"   Intensity change: {params.intensity_change_per_degree:.4f} per C")

# Get regions sorted by sensitivity
regions = get_temperature_effect_regions()
print(f"\nAll affected wavelength regions: {len(regions)} regions")

# =============================================================================
# Section 2: Temperature Effect Simulation with Operators
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Temperature Effect Simulation with Operators")
print("-" * 60)

# Create sample spectra with water bands
wavelengths = np.arange(900, 2501, 2)
n_wl = len(wavelengths)

# Create synthetic spectra with realistic features
rng = np.random.default_rng(42)
n_samples = 20
sample_spectra = np.zeros((n_samples, n_wl))

for i in range(n_samples):
    # Baseline
    sample_spectra[i] = 0.3 + 0.0002 * (wavelengths - 1500)

    # Add water bands (affected by temperature)
    for center, width, height in [(1450, 35, 0.4), (1940, 50, 0.6), (2200, 40, 0.3)]:
        band = height * rng.uniform(0.8, 1.2) * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        sample_spectra[i] += band

print(f"\nCreated sample spectra: {sample_spectra.shape}")
print(f"   Wavelength range: {wavelengths.min()}-{wavelengths.max()} nm")

# Create temperature operator for different temperature deltas
print("\nApplying temperature effects with TemperatureAugmenter:")

temps_test = [15.0, 25.0, 35.0, 45.0]
temp_results = {}
reference_temp = 25.0

for temp in temps_test:
    delta = temp - reference_temp
    temp_op = TemperatureAugmenter(
        temperature_delta=delta,
        reference_temperature=reference_temp,
        enable_shift=True,
        enable_intensity=True,
        enable_broadening=True,
        random_state=42,
    )
    result = temp_op.transform(sample_spectra.copy(), wavelengths=wavelengths)
    temp_results[temp] = result
    print(f"   Applied T={temp}C (delta={delta:+.1f}C): shape {result.shape}")

# =============================================================================
# Section 3: Moisture and Water Activity Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Moisture and Water Activity Effects")
print("-" * 60)

print("""
Water activity (aw) affects NIR spectra through:
  1. Water band intensity (1450, 1940, 2500 nm)
  2. Hydrogen bonding state (bound vs free water)
  3. Matrix interactions (protein-water, starch-water)
""")

# Create moisture operator
print("\nApplying moisture effects with MoistureAugmenter:")

aw_levels = [0.3, 0.5, 0.7, 0.9]
aw_results = {}

for aw in aw_levels:
    # Create operator with water activity delta from reference (0.5)
    moisture_op = MoistureAugmenter(
        water_activity_delta=aw - 0.5,
        reference_water_activity=0.5,
        free_water_fraction=0.3,
        moisture_content=0.12,
        random_state=42,
    )
    result = moisture_op.transform(sample_spectra.copy(), wavelengths=wavelengths)
    aw_results[aw] = result

    # Find difference at 1940 nm water band
    idx_1940 = np.abs(wavelengths - 1940).argmin()
    print(f"   aw={aw:.1f}: mean absorbance at 1940nm = {result[:, idx_1940].mean():.4f}")

# =============================================================================
# Section 4: Particle Size Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Particle Size Effects")
print("-" * 60)

print("""
Particle size affects NIR spectra through scattering:
  - Smaller particles -> more scattering -> higher baseline
  - Wavelength-dependent: shorter wavelengths scatter more
  - Corrected by SNV, MSC, or EMSC preprocessing
""")

# Configure particle size distribution
ps_distribution = ParticleSizeDistribution(
    mean_size_um=50.0,  # Mean particle diameter
    std_size_um=15.0,   # Standard deviation
    min_size_um=10.0,
    max_size_um=200.0,
    distribution="lognormal",  # Realistic for most materials
)

print("\nParticle Size Distribution:")
print(f"   Mean: {ps_distribution.mean_size_um} um")
print(f"   Std: {ps_distribution.std_size_um} um")
print(f"   Range: {ps_distribution.min_size_um}-{ps_distribution.max_size_um} um")
print(f"   Distribution type: {ps_distribution.distribution}")

# Apply particle size effects with operator
print("\nApplying particle size effects with ParticleSizeAugmenter:")

size_levels = [20, 50, 100, 200]  # um
size_results = {}

for size in size_levels:
    particle_op = ParticleSizeAugmenter(
        mean_size_um=float(size),
        size_variation_um=0.0,  # Fixed size for comparison
        reference_size_um=50.0,
        wavelength_exponent=1.5,
        random_state=42,
    )
    result = particle_op.transform(sample_spectra.copy(), wavelengths=wavelengths)
    size_results[size] = result
    print(f"   {size} um: baseline increase = {(result - sample_spectra).mean():.4f}")

# =============================================================================
# Section 5: EMSC-Style Distortion
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: EMSC-Style Scattering Distortion")
print("-" * 60)

print("""
EMSC (Extended Multiplicative Scatter Correction) models scattering as:
  X_observed = a + b*X_reference + c*wavelength + d*wavelength^2 + ...

We can simulate the INVERSE problem: adding realistic scatter distortions
that EMSC can later correct.
""")

# Apply EMSC distortion with operator
emsc_op = EMSCDistortionAugmenter(
    multiplicative_range=(0.85, 1.15),  # ~15% multiplicative variation
    additive_range=(-0.08, 0.08),       # Additive offset
    polynomial_order=2,
    polynomial_strength=0.02,
    random_state=42,
)

emsc_result = emsc_op.transform(sample_spectra.copy(), wavelengths=wavelengths)

print("\nApplied EMSC distortion:")
print(f"   Original mean: {sample_spectra.mean():.4f}")
print(f"   After EMSC distortion: {emsc_result.mean():.4f}")
print(f"   Std change: {sample_spectra.std():.4f} -> {emsc_result.std():.4f}")

# =============================================================================
# Section 6: Combined Environmental Effects in Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Integration with SyntheticNIRSGenerator")
print("-" * 60)

print("""
Phase 3 effects can be directly integrated with the SyntheticNIRSGenerator
by passing environmental_config and scattering_effects_config at initialization.
The generator uses nirs4all operators internally for applying these effects.
""")

# Configure combined environmental effects
env_config = EnvironmentalEffectsConfig(
    temperature=TemperatureConfig(
        reference_temperature=25.0,
        sample_temperature=30.0,
        temperature_variation=5.0,
    ),
    moisture=MoistureConfig(
        water_activity=0.6,
        moisture_content=0.12,
    ),
    enable_temperature=True,
    enable_moisture=True,
)

# Configure combined scattering effects
scatter_effects_config = ScatteringEffectsConfig(
    model=ScatteringModel.EMSC,
    particle_size=ParticleSizeConfig(
        distribution=ParticleSizeDistribution(mean_size_um=40.0, std_size_um=12.0)
    ),
    emsc=EMSCConfig(polynomial_order=2, multiplicative_scatter_std=0.1),
    scattering_coefficient=ScatteringCoefficientConfig(wavelength_exponent=1.2),
    enable_particle_size=True,
    enable_emsc=True,
)

print("\nCombined Environmental Effects:")
print(f"   Temperature enabled: {env_config.enable_temperature}")
print(f"   Moisture enabled: {env_config.enable_moisture}")

print("\nCombined Scattering Effects:")
print(f"   Model: {scatter_effects_config.model.value}")
print(f"   Particle size enabled: {scatter_effects_config.enable_particle_size}")
print(f"   EMSC enabled: {scatter_effects_config.enable_emsc}")

# Create generator with all Phase 3 effects
generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2200,
    wavelength_step=4,
    complexity="realistic",
    environmental_config=env_config,
    scattering_effects_config=scatter_effects_config,
    random_state=42,
)

print("\nCreated generator with Phase 3 effects:")
print(f"   {generator}")

# Generate spectra with Phase 3 effects
_result_p3 = generator.generate(
    n_samples=100,
    include_environmental_effects=True,
    include_scattering_effects=True,
    return_metadata=True,
)
assert len(_result_p3) == 4
X_p3, Y, E, meta = _result_p3

print("\nGenerated spectra with Phase 3 effects:")
print(f"   Spectra shape: {X_p3.shape}")
print(f"   Targets shape: {Y.shape}")
print(f"   Environmental effects: {meta.get('environmental_effects', 'N/A')}")
print(f"   Scattering effects: {meta.get('scattering_effects', 'N/A')}")

# Generate without Phase 3 for comparison
_result_no_p3 = generator.generate(
    n_samples=100,
    include_environmental_effects=False,
    include_scattering_effects=False,
)
X_no_p3 = _result_no_p3[0]

print("\nComparison:")
print(f"   Without Phase 3: mean={X_no_p3.mean():.4f}, std={X_no_p3.std():.4f}")
print(f"   With Phase 3: mean={X_p3.mean():.4f}, std={X_p3.std():.4f}")

# Generate with specific temperatures
specific_temps = np.linspace(15, 40, 100)
_result_temps = generator.generate(
    n_samples=100,
    include_environmental_effects=True,
    include_scattering_effects=True,
    temperatures=specific_temps,
    return_metadata=True,
)
assert len(_result_temps) == 4
X_temps, _, _, meta_temps = _result_temps
print("\nGenerated with specific temperatures:")
print(f"   Temperature range: {specific_temps.min():.1f}C to {specific_temps.max():.1f}C")

# =============================================================================
# Section 7: Plotting (optional)
# =============================================================================
if args.plots:
    output_path = get_examples_output_dir() / EXAMPLE_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Temperature effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a: Spectra at different temperatures
    ax = axes[0, 0]
    wl_plot = wavelengths
    for temp in [15, 25, 35, 45]:
        ax.plot(wl_plot, temp_results[temp].mean(axis=0),
                label=f'{temp}C', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Temperature Effects on NIR Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: Difference from reference temperature
    ax = axes[0, 1]
    for temp in [15, 35, 45]:
        diff = temp_results[temp].mean(axis=0) - temp_results[25].mean(axis=0)
        ax.plot(wl_plot, diff, label=f'{temp}C - 25C', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance Difference")
    ax.set_title("Temperature-Induced Spectral Changes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 1c: Water activity effects
    ax = axes[1, 0]
    idx_1940 = np.abs(wavelengths - 1940).argmin()
    for aw in [0.3, 0.5, 0.7, 0.9]:
        ax.plot(wl_plot, aw_results[aw].mean(axis=0),
                label=f'aw={aw}', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Water Activity Effects on NIR Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(1940, color='gray', linestyle='--', alpha=0.5)

    # 1d: Water activity at 1940 nm
    ax = axes[1, 1]
    aw_values = list(aw_results.keys())
    abs_at_1940 = [aw_results[aw][:, idx_1940].mean() for aw in aw_values]
    ax.plot(aw_values, abs_at_1940, 'o-', markersize=10)
    ax.set_xlabel("Water Activity (aw)")
    ax.set_ylabel("Mean Absorbance at 1940 nm")
    ax.set_title("Water Band Response to Water Activity")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "temperature_moisture_effects.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved temperature/moisture plot: {plot_path}")

    # Plot 2: Particle size and scattering effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Particle size effects
    ax = axes[0, 0]
    for size in [20, 50, 100, 200]:
        ax.plot(wl_plot, size_results[size].mean(axis=0),
                label=f'{size} um', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Particle Size Effects on NIR Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b: Baseline shift from particle size
    ax = axes[0, 1]
    sizes = list(size_results.keys())
    baseline_shifts = [(size_results[s] - sample_spectra).mean() for s in sizes]
    ax.plot(sizes, baseline_shifts, 'o-', markersize=10)
    ax.set_xlabel("Particle Size (um)")
    ax.set_ylabel("Mean Baseline Shift")
    ax.set_title("Particle Size vs Baseline Shift")
    ax.grid(True, alpha=0.3)

    # 2c: EMSC distortion
    ax = axes[1, 0]
    ax.plot(wl_plot, sample_spectra[0], 'b-', label='Original', linewidth=2)
    ax.plot(wl_plot, emsc_result[0], 'r--', label='After EMSC distortion', alpha=0.7)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("EMSC-Style Scatter Distortion (Single Spectrum)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2d: Combined effects comparison
    gen_wl = np.arange(1000, 2204, 4)
    ax = axes[1, 1]
    ax.plot(gen_wl, X_no_p3.std(axis=0), 'b-', label='Without Phase 3', linewidth=2)
    ax.plot(gen_wl, X_p3.std(axis=0), 'r-', label='With Phase 3', linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Spectral Variability Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "scattering_effects.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved scattering effects plot: {plot_path}")

    # Plot 3: Combined effects spectra
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i in range(min(5, 100)):
        ax.plot(gen_wl, X_no_p3[i], alpha=0.3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Without Phase 3 Effects")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(min(5, 100)):
        ax.plot(gen_wl, X_p3[i], alpha=0.3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("With Phase 3 Effects")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "combined_effects.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined effects plot: {plot_path}")

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

1. Temperature Effects:
   - O-H, N-H, C-H band shifts with temperature
   - Literature-based shift coefficients (~0.11 nm/C for O-H)
   - TemperatureAugmenter operator for applying effects

2. Moisture/Water Activity Effects:
   - Water band intensity modulation (1450, 1940, 2500 nm)
   - Free vs bound water simulation
   - MoistureAugmenter operator for applying effects

3. Particle Size Effects:
   - Log-normal size distributions (realistic for powders)
   - Wavelength-dependent scattering
   - ParticleSizeAugmenter operator for applying effects

4. EMSC-Style Distortion:
   - Multiplicative and additive scatter
   - Polynomial baseline components
   - EMSCDistortionAugmenter operator for applying effects

5. Integration with Generator:
   - environmental_config for temperature/moisture
   - scattering_effects_config for particle/scatter effects
   - include_environmental_effects and include_scattering_effects flags
   - Generator uses operators internally for consistency

Use cases:
- Calibration transfer studies (temperature/instrument variation)
- Data augmentation for robust models
- Simulation of real-world measurement conditions
- Testing preprocessing effectiveness (EMSC, SNV, MSC)

Next steps:
- Combine with instrument effects (D09) for full realism
- Use synthetic data for model robustness testing
- Apply to domain adaptation research
""")

print("\nExample completed successfully!")
