"""
D12 - Synthetic Generator: Environmental and Matrix Effects
============================================================

Learn to simulate realistic environmental and matrix effects in synthetic NIR spectra.

This tutorial covers Phase 3 of the synthetic generator:

* Temperature effects on spectral bands (O-H, N-H, C-H shifts)
* Moisture and water activity effects
* Particle size effects (EMSC-style scattering)
* Scattering coefficient generation (Kubelka-Munk)
* Combined environmental and matrix effect simulation

Prerequisites
-------------
Complete D05-D09 first.

Duration: ~10 minutes
Difficulty: ★★★★☆

Scientific Background
---------------------
Temperature Effects:
    O-H stretching bands shift ~0.11 nm/°C to lower wavelengths with increasing
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
    - Martens & Næs (1989): Multivariate calibration and scattering
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# NIRS4All imports
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    # Phase 3: Environmental effects
    SpectralRegion,
    TemperatureEffectParams,
    TemperatureConfig,
    MoistureConfig,
    EnvironmentalEffectsConfig,
    TemperatureEffectSimulator,
    MoistureEffectSimulator,
    EnvironmentalEffectsSimulator,
    TEMPERATURE_EFFECT_PARAMS,
    # Convenience functions
    apply_temperature_effects,
    apply_moisture_effects,
    simulate_temperature_series,
    get_temperature_effect_regions,
    # Phase 3: Scattering effects
    ScatteringModel,
    ParticleSizeDistribution,
    ParticleSizeConfig,
    EMSCConfig,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
    ParticleSizeSimulator,
    EMSCTransformSimulator,
    ScatteringCoefficientGenerator,
    ScatteringEffectsSimulator,
    # Scattering convenience functions
    apply_particle_size_effects,
    apply_emsc_distortion,
    generate_scattering_coefficients,
    simulate_snv_correctable_scatter,
    simulate_msc_correctable_scatter,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D12 Environmental Effects Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "D12_synthetic_environmental"


# =============================================================================
# Section 1: Temperature Effects Overview
# =============================================================================
print("\n" + "=" * 60)
print("D12 - Environmental and Matrix Effects for Synthetic Spectra")
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
    print(f"\n🌡️  {region.value.upper()} Region ({params.wavelength_range[0]}-{params.wavelength_range[1]} nm):")
    print(f"   Position shift: {params.shift_per_degree:.4f} nm/°C")
    print(f"   Bandwidth change: {params.broadening_per_degree:.4f} per °C")
    print(f"   Intensity change: {params.intensity_change_per_degree:.4f} per °C")

# Get regions sorted by sensitivity
regions = get_temperature_effect_regions()
print(f"\n📊 All affected wavelength regions: {len(regions)} regions")


# =============================================================================
# Section 2: Temperature Effect Simulation
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Temperature Effect Simulation")
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

print(f"\n📊 Created sample spectra: {sample_spectra.shape}")
print(f"   Wavelength range: {wavelengths.min()}-{wavelengths.max()} nm")

# Configure temperature simulation
temp_config = TemperatureConfig(
    reference_temperature=25.0,  # Reference at 25°C
    sample_temperature=35.0,  # Nominal sample temperature
    temperature_variation=5.0,  # Sample-to-sample variation (±5°C)
    region_specific=True,  # Use region-specific parameters
    enable_shift=True,
    enable_intensity=True,
    enable_broadening=True,
)

print(f"\n🌡️  Temperature Configuration:")
print(f"   Reference: {temp_config.reference_temperature}°C")
print(f"   Sample temp: {temp_config.sample_temperature}°C")
print(f"   Variation: ±{temp_config.temperature_variation}°C")
print(f"   Region-specific: {temp_config.region_specific}")

# Create simulator and apply
temp_simulator = TemperatureEffectSimulator(temp_config, random_state=42)

# Simulate at different temperatures
temps_test = [15.0, 25.0, 35.0, 45.0]
temp_results = {}

for temp in temps_test:
    temps_array = np.full(n_samples, temp)
    result = temp_simulator.apply(sample_spectra.copy(), wavelengths, temps_array)
    temp_results[temp] = result
    print(f"   Applied T={temp}°C: shape {result.shape}")


# =============================================================================
# Section 3: Temperature Series Generation
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Temperature Series Generation")
print("-" * 60)

# Generate a temperature series for one sample
single_spectrum = sample_spectra[0]  # 1D spectrum
temp_series = np.linspace(10, 50, 21)  # 10°C to 50°C in 2°C steps

series_result = simulate_temperature_series(
    single_spectrum,
    wavelengths,
    temperatures=list(temp_series),
    reference_temperature=25.0,
    random_state=42
)

print(f"\n📈 Generated temperature series:")
print(f"   Input spectrum shape: {single_spectrum.shape}")
print(f"   Temperature range: {temp_series.min():.1f}°C to {temp_series.max():.1f}°C")
print(f"   Output series shape: {series_result.shape}")
print(f"   Temperature steps: {len(temp_series)}")


# =============================================================================
# Section 4: Moisture and Water Activity Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Moisture and Water Activity Effects")
print("-" * 60)

print("""
Water activity (aw) affects NIR spectra through:
  1. Water band intensity (1450, 1940, 2500 nm)
  2. Hydrogen bonding state (bound vs free water)
  3. Matrix interactions (protein-water, starch-water)
""")

# Configure moisture effects
moisture_config = MoistureConfig(
    water_activity=0.7,  # Typical food product
    moisture_content=0.12,  # 12% moisture
    free_water_fraction=0.3,
    bound_water_shift=25.0,  # nm shift for bound water
    temperature_interaction=True,
)

print(f"\n💧 Moisture Configuration:")
print(f"   Water activity: {moisture_config.water_activity}")
print(f"   Moisture content: {moisture_config.moisture_content*100:.0f}%")
print(f"   Free water fraction: {moisture_config.free_water_fraction}")
print(f"   Bound water shift: {moisture_config.bound_water_shift} nm")

# Apply moisture effects
moisture_simulator = MoistureEffectSimulator(moisture_config, random_state=42)
moisture_result = moisture_simulator.apply(sample_spectra.copy(), wavelengths)

print(f"\n📊 Applied moisture effects:")
print(f"   Original mean absorbance: {sample_spectra.mean():.4f}")
print(f"   After moisture effects: {moisture_result.mean():.4f}")

# Compare different water activities
aw_levels = [0.3, 0.5, 0.7, 0.9]
aw_results = {}

for aw in aw_levels:
    result = apply_moisture_effects(
        sample_spectra.copy(), wavelengths,
        water_activity=aw,
        random_state=42
    )
    aw_results[aw] = result

    # Find difference at 1940 nm water band
    idx_1940 = np.abs(wavelengths - 1940).argmin()
    print(f"   aw={aw:.1f}: mean absorbance at 1940nm = {result[:, idx_1940].mean():.4f}")


# =============================================================================
# Section 5: Particle Size Effects (EMSC-Style)
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Particle Size Effects")
print("-" * 60)

print("""
Particle size affects NIR spectra through scattering:
  - Smaller particles → more scattering → higher baseline
  - Wavelength-dependent: shorter λ scatter more
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

ps_config = ParticleSizeConfig(
    distribution=ps_distribution,
    reference_size_um=50.0,  # Reference particle size
    size_effect_strength=1.0,
    wavelength_exponent=1.5,  # λ^(-1.5) dependence
)

print(f"\n🔬 Particle Size Distribution:")
print(f"   Mean: {ps_distribution.mean_size_um} µm")
print(f"   Std: {ps_distribution.std_size_um} µm")
print(f"   Range: {ps_distribution.min_size_um}-{ps_distribution.max_size_um} µm")
print(f"   Distribution type: {ps_distribution.distribution}")

# Apply particle size effects
ps_simulator = ParticleSizeSimulator(ps_config, random_state=42)
ps_result = ps_simulator.apply(sample_spectra.copy(), wavelengths)

print(f"\n📊 Applied particle size effects:")
print(f"   Baseline change (mean): {(ps_result - sample_spectra).mean():.4f}")

# Compare different particle sizes
size_levels = [20, 50, 100, 200]  # µm
size_results = {}

for size in size_levels:
    sizes = np.full(n_samples, float(size))
    result = ps_simulator.apply(sample_spectra.copy(), wavelengths, sizes)
    size_results[size] = result
    print(f"   {size} µm: baseline increase = {(result - sample_spectra).mean():.4f}")


# =============================================================================
# Section 6: EMSC-Style Distortion
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: EMSC-Style Scattering Distortion")
print("-" * 60)

print("""
EMSC (Extended Multiplicative Scatter Correction) models scattering as:
  X_observed = a + b·X_reference + c·λ + d·λ² + ...

We can simulate the INVERSE problem: adding realistic scatter distortions
that EMSC can later correct.
""")

# Configure EMSC distortion
emsc_config = EMSCConfig(
    polynomial_order=2,  # Up to λ² terms
    multiplicative_scatter_std=0.15,  # Multiplicative term variation
    additive_scatter_std=0.08,  # Additive offset variation
    include_wavelength_terms=True,
)

print(f"\n📐 EMSC Configuration:")
print(f"   Polynomial order: {emsc_config.polynomial_order}")
print(f"   Multiplicative std: {emsc_config.multiplicative_scatter_std}")
print(f"   Additive std: {emsc_config.additive_scatter_std}")

# Apply EMSC distortion
emsc_simulator = EMSCTransformSimulator(emsc_config, random_state=42)
emsc_result = emsc_simulator.apply(sample_spectra.copy(), wavelengths)

# Examine basis functions
basis = emsc_simulator.get_emsc_basis(wavelengths)
print(f"\n📊 EMSC basis functions: {basis.shape}")
print(f"   Constant term: {basis[:, 0].mean():.1f}")
print(f"   Linear term range: {basis[:, 1].min():.2f} to {basis[:, 1].max():.2f}")
print(f"   Quadratic term range: {basis[:, 2].min():.2f} to {basis[:, 2].max():.2f}")


# =============================================================================
# Section 7: SNV/MSC-Correctable Scatter
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: SNV/MSC-Correctable Scatter Effects")
print("-" * 60)

print("""
These convenience functions add scatter that can be corrected by standard
preprocessing methods:
  - SNV (Standard Normal Variate): Removes multiplicative/additive scatter
  - MSC (Multiplicative Scatter Correction): Linear regression to reference
""")

# Apply SNV-correctable scatter
snv_scattered = simulate_snv_correctable_scatter(
    sample_spectra.copy(),
    intensity=1.0,
    random_state=42
)
print(f"\n📊 SNV-correctable scatter applied:")
print(f"   Original std (per spectrum): {np.std(sample_spectra, axis=1).mean():.4f}")
print(f"   Scattered std (per spectrum): {np.std(snv_scattered, axis=1).mean():.4f}")

# Apply MSC-correctable scatter
msc_scattered = simulate_msc_correctable_scatter(
    sample_spectra.copy(),
    intensity=1.0,
    random_state=42
)
print(f"\n📊 MSC-correctable scatter applied:")
print(f"   Original mean: {sample_spectra.mean():.4f}")
print(f"   Scattered mean: {msc_scattered.mean():.4f}")


# =============================================================================
# Section 8: Scattering Coefficient Generation
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Scattering Coefficient Generation")
print("-" * 60)

print("""
Generate wavelength-dependent scattering coefficients based on:
  - Kubelka-Munk theory: S(λ) ∝ λ^(-n)
  - Particle size distribution effects
  - Sample-to-sample variation
""")

# Configure scattering coefficient generation
scatter_config = ScatteringCoefficientConfig(
    baseline_scattering=1.0,
    wavelength_exponent=1.0,  # λ^(-1) dependence
    wavelength_reference_nm=1500.0,
    sample_variation=0.2,  # 20% variation between samples
)

print(f"\n📐 Scattering Coefficient Configuration:")
print(f"   Baseline: {scatter_config.baseline_scattering}")
print(f"   Wavelength exponent: {scatter_config.wavelength_exponent}")
print(f"   Reference wavelength: {scatter_config.wavelength_reference_nm} nm")
print(f"   Sample variation: {scatter_config.sample_variation}")

# Generate scattering coefficients
scatter_gen = ScatteringCoefficientGenerator(scatter_config, random_state=42)
S = scatter_gen.generate(n_samples, wavelengths)

print(f"\n📊 Generated scattering coefficients:")
print(f"   Shape: {S.shape}")
print(f"   Mean S at 1000 nm: {S[:, np.abs(wavelengths-1000).argmin()].mean():.4f}")
print(f"   Mean S at 2000 nm: {S[:, np.abs(wavelengths-2000).argmin()].mean():.4f}")
print(f"   Short/long wavelength ratio: {S[:, 0].mean() / S[:, -1].mean():.2f}")

# Generate with particle sizes
particle_sizes = ps_simulator.generate_particle_sizes(n_samples)
S_sized = scatter_gen.generate(n_samples, wavelengths, particle_sizes)
print(f"\n📊 Scattering with particle sizes:")
print(f"   Mean particle size: {particle_sizes.mean():.1f} µm")
print(f"   Mean S at 1500 nm: {S_sized[:, np.abs(wavelengths-1500).argmin()].mean():.4f}")


# =============================================================================
# Section 9: Combined Environmental Effects
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Combined Environmental and Scattering Effects")
print("-" * 60)

print("""
The EnvironmentalEffectsSimulator and ScatteringEffectsSimulator combine
multiple effects in a physically realistic way.
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

print(f"\n🌡️💧 Combined Environmental Effects:")
print(f"   Temperature enabled: {env_config.enable_temperature}")
print(f"   Moisture enabled: {env_config.enable_moisture}")

print(f"\n🔬 Combined Scattering Effects:")
print(f"   Model: {scatter_effects_config.model.value}")
print(f"   Particle size enabled: {scatter_effects_config.enable_particle_size}")
print(f"   EMSC enabled: {scatter_effects_config.enable_emsc}")

# Apply combined effects
env_simulator = EnvironmentalEffectsSimulator(env_config, random_state=42)
scatter_simulator = ScatteringEffectsSimulator(scatter_effects_config, random_state=42)

# Generate random temperatures
temperatures = np.random.default_rng(42).uniform(15, 40, n_samples)

# Apply in sequence
spectra_env = env_simulator.apply(sample_spectra.copy(), wavelengths, sample_temperatures=temperatures)
spectra_full = scatter_simulator.apply(spectra_env.copy(), wavelengths)

print(f"\n📊 Effect application results:")
print(f"   Original: mean={sample_spectra.mean():.4f}, std={sample_spectra.std():.4f}")
print(f"   After environmental: mean={spectra_env.mean():.4f}, std={spectra_env.std():.4f}")
print(f"   After scattering: mean={spectra_full.mean():.4f}, std={spectra_full.std():.4f}")


# =============================================================================
# Section 10: Integration with Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 10: Integration with SyntheticNIRSGenerator")
print("-" * 60)

print("""
Phase 3 effects can be directly integrated with the SyntheticNIRSGenerator
by passing environmental_config and scattering_effects_config at initialization.
""")

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

print(f"\n🔧 Created generator with Phase 3 effects:")
print(f"   {generator}")

# Generate spectra with Phase 3 effects
X_p3, Y, E, meta = generator.generate(
    n_samples=100,
    include_environmental_effects=True,
    include_scattering_effects=True,
    return_metadata=True,
)

print(f"\n📊 Generated spectra with Phase 3 effects:")
print(f"   Spectra shape: {X_p3.shape}")
print(f"   Targets shape: {Y.shape}")
print(f"   Environmental effects: {meta.get('environmental_effects', 'N/A')}")
print(f"   Scattering effects: {meta.get('scattering_effects', 'N/A')}")

# Generate without Phase 3 for comparison
X_no_p3, _, _, _ = generator.generate(
    n_samples=100,
    include_environmental_effects=False,
    include_scattering_effects=False,
    return_metadata=True,
)

print(f"\n📊 Comparison:")
print(f"   Without Phase 3: mean={X_no_p3.mean():.4f}, std={X_no_p3.std():.4f}")
print(f"   With Phase 3: mean={X_p3.mean():.4f}, std={X_p3.std():.4f}")

# Generate with specific temperatures
specific_temps = np.linspace(15, 40, 100)
X_temps, _, _, meta_temps = generator.generate(
    n_samples=100,
    include_environmental_effects=True,
    include_scattering_effects=True,
    temperatures=specific_temps,
    return_metadata=True,
)
print(f"\n📊 Generated with specific temperatures:")
print(f"   Temperature range: {specific_temps.min():.1f}°C to {specific_temps.max():.1f}°C")


# =============================================================================
# Section 11: Plotting (optional)
# =============================================================================
if args.plots:
    output_path = get_example_output_path(EXAMPLE_NAME)

    # Plot 1: Temperature effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a: Spectra at different temperatures
    ax = axes[0, 0]
    wl_plot = wavelengths
    for temp in [15, 25, 35, 45]:
        ax.plot(wl_plot, temp_results[temp].mean(axis=0),
                label=f'{temp}°C', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Temperature Effects on NIR Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: Temperature series
    ax = axes[0, 1]
    im = ax.imshow(series_result.squeeze().T, aspect='auto',
                   extent=[temp_series.min(), temp_series.max(), wavelengths.max(), wavelengths.min()],
                   cmap='viridis')
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Temperature Series Heatmap")
    plt.colorbar(im, ax=ax, label='Absorbance')

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
    print(f"\n📊 Saved temperature/moisture plot: {plot_path}")

    # Plot 2: Particle size and scattering effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Particle size effects
    ax = axes[0, 0]
    for size in [20, 50, 100, 200]:
        ax.plot(wl_plot, size_results[size].mean(axis=0),
                label=f'{size} µm', alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Particle Size Effects on NIR Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b: Scattering coefficients
    ax = axes[0, 1]
    for i in range(min(5, n_samples)):
        ax.plot(wl_plot, S[i], alpha=0.5)
    ax.plot(wl_plot, S.mean(axis=0), 'k-', linewidth=2, label='Mean')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Scattering Coefficient")
    ax.set_title("Wavelength-Dependent Scattering Coefficients")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2c: SNV vs MSC scatter
    ax = axes[1, 0]
    ax.plot(wl_plot, sample_spectra[0], 'b-', label='Original', linewidth=2)
    ax.plot(wl_plot, snv_scattered[0], 'r--', label='SNV-correctable', alpha=0.7)
    ax.plot(wl_plot, msc_scattered[0], 'g--', label='MSC-correctable', alpha=0.7)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Scatter Types (Single Spectrum)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2d: EMSC basis functions
    ax = axes[1, 1]
    ax.plot(wl_plot, basis[:, 0], label='Constant', linewidth=2)
    ax.plot(wl_plot, basis[:, 1], label='Linear (λ)', linewidth=2)
    ax.plot(wl_plot, basis[:, 2], label='Quadratic (λ²)', linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Basis Value")
    ax.set_title("EMSC Polynomial Basis Functions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "scattering_effects.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved scattering effects plot: {plot_path}")

    # Plot 3: Combined effects comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gen_wl = np.arange(1000, 2204, 4)  # Generator wavelengths

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

    ax = axes[2]
    ax.plot(gen_wl, X_no_p3.std(axis=0), 'b-', label='Without Phase 3', linewidth=2)
    ax.plot(gen_wl, X_p3.std(axis=0), 'r-', label='With Phase 3', linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Spectral Variability Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "combined_effects.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved combined effects plot: {plot_path}")

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

1. 🌡️ Temperature Effects:
   - O-H, N-H, C-H band shifts with temperature
   - Literature-based shift coefficients (~0.11 nm/°C for O-H)
   - Temperature series generation for calibration transfer studies

2. 💧 Moisture/Water Activity Effects:
   - Water band intensity modulation (1450, 1940, 2500 nm)
   - Free vs bound water simulation
   - Baseline shifts with moisture content

3. 🔬 Particle Size Effects:
   - Log-normal size distributions (realistic for powders)
   - Wavelength-dependent scattering (λ^(-n) relationship)
   - Reference-size normalized effects

4. 📐 EMSC-Style Distortion:
   - Multiplicative and additive scatter
   - Polynomial baseline components
   - Inverse of EMSC correction for data augmentation

5. 📊 Scattering Coefficients:
   - Kubelka-Munk theory-based generation
   - Sample-to-sample variation
   - Particle-size-dependent coefficients

6. 🔧 Integration with Generator:
   - environmental_config for temperature/moisture
   - scattering_effects_config for particle/scatter effects
   - include_environmental_effects and include_scattering_effects flags

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

print("\n✅ Example completed successfully!")
