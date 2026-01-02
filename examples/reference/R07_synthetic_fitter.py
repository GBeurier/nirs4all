"""
R07 - Synthetic Generator: Fitting to Real Data
================================================

Comprehensive reference for analyzing real NIRS datasets and creating
matching synthetic spectra (Phase 4 features).

Topics covered:

* Analyzing spectral properties of real data
* Inferring instrument archetypes from spectra
* Detecting measurement modes (transmittance, reflectance, ATR)
* Identifying application domains from band intensities
* Detecting environmental and scattering effects
* Creating generators that match real data characteristics
* Evaluating synthetic-to-real similarity

Prerequisites
-------------
Complete D05-D09 synthetic generator developer examples first.

Duration: ~10 minutes
Difficulty: ★★★★★
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
    # Real data fitting (Phase 4)
    RealDataFitter,
    FittedParameters,
    SpectralProperties,
    compute_spectral_properties,
    fit_to_real_data,
    compare_datasets,
    # Phase 1-4 inference classes
    InstrumentInference,
    DomainInference,
    EnvironmentalInference,
    ScatteringInference,
)

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from example_utils import get_example_output_path, print_output_location

# Parse command-line arguments
parser = argparse.ArgumentParser(description='R07 Fitting to Real Data Reference')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "R07_synthetic_fitter"


# =============================================================================
# Section 1: Loading Real Data
# =============================================================================
print("\n" + "=" * 60)
print("R07 - Fitting Synthetic Generators to Real Data")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Loading Real NIR Data")
print("-" * 60)

# Load the sample regression dataset
data_path = Path(__file__).parent.parent / "sample_data" / "regression"

# Load calibration data (semicolon-separated)
X_cal = np.loadtxt(data_path / "Xcal.csv.gz", delimiter=";")
Y_cal = np.loadtxt(data_path / "Ycal.csv.gz", delimiter=";")

# Load validation data
X_val = np.loadtxt(data_path / "Xval.csv.gz", delimiter=";")
Y_val = np.loadtxt(data_path / "Yval.csv.gz", delimiter=";")

# Combine for analysis
X_real = np.vstack([X_cal, X_val])
Y_real = np.concatenate([Y_cal, Y_val])

# Create wavelength grid (assuming standard NIR range)
# The dataset has 700 wavelengths, likely 1100-2500 nm
n_wavelengths = X_real.shape[1]
wavelengths = np.linspace(1100, 2500, n_wavelengths)

print(f"\n📊 Loaded sample_data/regression dataset:")
print(f"   Calibration samples: {X_cal.shape[0]}")
print(f"   Validation samples: {X_val.shape[0]}")
print(f"   Total samples: {X_real.shape[0]}")
print(f"   Wavelengths: {n_wavelengths}")
print(f"   Target range: {Y_real.min():.2f} to {Y_real.max():.2f}")
print(f"   Assumed wavelength range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")


# =============================================================================
# Section 2: Computing Spectral Properties
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Analyzing Spectral Properties")
print("-" * 60)

# Compute spectral properties
props = compute_spectral_properties(X_real, wavelengths, name="sample_regression")

print(f"\n📈 Spectral Properties of the Dataset:")
print(f"\n   Basic Statistics:")
print(f"      Global mean: {props.global_mean:.4f}")
print(f"      Global std: {props.global_std:.4f}")
print(f"      Range: {props.global_range[0]:.4f} to {props.global_range[1]:.4f}")

print(f"\n   Spectral Shape:")
print(f"      Mean slope: {props.mean_slope:.4f} (per 1000nm)")
print(f"      Slope std: {props.slope_std:.4f}")
print(f"      Mean curvature: {props.mean_curvature:.6f}")

print(f"\n   Noise Characteristics:")
print(f"      Noise estimate: {props.noise_estimate:.6f}")
print(f"      SNR estimate: {props.snr_estimate:.1f}")

print(f"\n   Complexity (PCA):")
print(f"      Components for 95% variance: {props.pca_n_components_95}")

print(f"\n   Peak Analysis:")
print(f"      Number of peaks detected: {props.n_peaks_mean:.0f}")
if props.peak_positions is not None and len(props.peak_positions) > 0:
    print(f"      Peak positions: {[f'{p:.0f}' for p in props.peak_positions[:5]]} nm")

print(f"\n   Phase 1-4 Enhanced Properties:")
print(f"      Effective resolution: {props.effective_resolution:.1f} nm")
print(f"      Noise correlation length: {props.noise_correlation_length:.1f}")
print(f"      Baseline offset: {props.baseline_offset:.4f}")
print(f"      Baseline convexity: {props.baseline_convexity:.4f}")


# =============================================================================
# Section 3: Using the RealDataFitter
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Fitting Generator Parameters")
print("-" * 60)

# Create and fit the RealDataFitter
fitter = RealDataFitter()
params = fitter.fit(X_real, wavelengths=wavelengths, name="sample_regression")

print(f"\n🔧 Fitted Generator Parameters:")
print(f"\n   Wavelength Grid:")
print(f"      Range: {params.wavelength_start:.0f}-{params.wavelength_end:.0f} nm")
print(f"      Step: {params.wavelength_step:.2f} nm")

print(f"\n   Noise Parameters:")
print(f"      Base noise: {params.noise_base:.6f}")
print(f"      Signal-dependent noise: {params.noise_signal_dep:.6f}")

print(f"\n   Scatter Parameters:")
print(f"      Multiplicative (α): {params.scatter_alpha_std:.4f}")
print(f"      Additive (β): {params.scatter_beta_std:.4f}")
print(f"      Path length std: {params.path_length_std:.4f}")

print(f"\n   Baseline Parameters:")
print(f"      Amplitude: {params.baseline_amplitude:.4f}")
print(f"      Tilt std: {params.tilt_std:.4f}")
print(f"      Global slope: {params.global_slope_mean:.4f} ± {params.global_slope_std:.4f}")

print(f"\n   Suggested complexity: {params.complexity}")
print(f"   Suggested n_components: {params.suggested_n_components}")


# =============================================================================
# Section 4: Phase 1-4 Enhanced Inference
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Phase 1-4 Enhanced Inference")
print("-" * 60)

# Instrument inference
print(f"\n🔬 Instrument Inference:")
print(f"   Inferred archetype: {params.inferred_instrument}")
if params.instrument_inference:
    inst = params.instrument_inference
    print(f"   Detector type: {inst.detector_type}")
    print(f"   Estimated resolution: {inst.estimated_resolution:.1f} nm")
    print(f"   Confidence: {inst.confidence:.2f}")
    if inst.alternative_archetypes:
        alts = list(inst.alternative_archetypes.items())[:3]
        print(f"   Alternatives: {', '.join([f'{n}({s:.2f})' for n, s in alts])}")

# Measurement mode inference
print(f"\n📐 Measurement Mode Inference:")
print(f"   Detected mode: {params.measurement_mode}")
print(f"   Confidence: {params.measurement_mode_confidence:.2f}")

# Domain inference
print(f"\n🏭 Application Domain Inference:")
print(f"   Inferred domain: {params.inferred_domain}")
if params.domain_inference:
    dom = params.domain_inference
    print(f"   Category: {dom.category}")
    print(f"   Confidence: {dom.confidence:.2f}")
    print(f"   Detected components: {', '.join(dom.detected_components) or 'None'}")
    if dom.alternative_domains:
        alts = list(dom.alternative_domains.items())[:3]
        print(f"   Alternatives: {', '.join([f'{n}({s:.2f})' for n, s in alts])}")

# Environmental effects inference
print(f"\n🌡️ Environmental Effects Inference:")
if params.environmental_inference:
    env = params.environmental_inference
    print(f"   Temperature effects: {env.has_temperature_effects}")
    if env.has_temperature_effects:
        print(f"      Estimated variation: ±{env.estimated_temperature_variation:.1f}°C")
    print(f"   Moisture effects: {env.has_moisture_effects}")
    if env.has_moisture_effects:
        print(f"      Estimated variation: {env.estimated_moisture_variation:.4f}")
    print(f"   Water band shift: {env.water_band_shift:.1f} nm")

# Scattering effects inference
print(f"\n💨 Scattering Effects Inference:")
if params.scattering_inference:
    scat = params.scattering_inference
    print(f"   Scatter effects detected: {scat.has_scatter_effects}")
    print(f"   Estimated particle size: {scat.estimated_particle_size_um:.0f} μm")
    print(f"   Multiplicative scatter std: {scat.multiplicative_scatter_std:.4f}")
    print(f"   Additive scatter std: {scat.additive_scatter_std:.4f}")
    print(f"   SNV correctable: {scat.snv_correctable}")
    print(f"   MSC correctable: {scat.msc_correctable}")


# =============================================================================
# Section 5: Creating a Matched Generator
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Creating a Matched Generator")
print("-" * 60)

# Create a generator that matches the real data
generator = fitter.create_matched_generator(random_state=42)

print(f"\n🏭 Created matched generator:")
print(f"   Type: {type(generator).__name__}")
print(f"   Wavelength range: {generator.wavelength_start}-{generator.wavelength_end} nm")
print(f"   Wavelength step: {generator.wavelength_step} nm")
print(f"   Complexity: {generator.complexity}")

# Generate synthetic data
n_synthetic = X_real.shape[0]
X_synth, concentrations, pure_spectra = generator.generate(n_synthetic)

print(f"\n📊 Generated synthetic data:")
print(f"   Shape: {X_synth.shape}")
print(f"   Mean: {X_synth.mean():.4f} (real: {X_real.mean():.4f})")
print(f"   Std: {X_synth.std():.4f} (real: {X_real.std():.4f})")
print(f"   Range: {X_synth.min():.4f} to {X_synth.max():.4f}")


# =============================================================================
# Section 6: Evaluating Similarity
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Evaluating Synthetic-to-Real Similarity")
print("-" * 60)

# Evaluate similarity metrics
metrics = fitter.evaluate_similarity(X_synth, wavelengths)

print(f"\n📊 Similarity Metrics:")
print(f"\n   Statistical Comparison:")
print(f"      Mean relative difference: {metrics['mean_rel_diff']:.4f}")
print(f"      Std relative difference: {metrics['std_rel_diff']:.4f}")

print(f"\n   Noise Comparison:")
print(f"      Noise ratio (synth/real): {metrics['noise_ratio']:.2f}")
if metrics['snr_ratio'] != float('inf'):
    print(f"      SNR ratio (synth/real): {metrics['snr_ratio']:.2f}")

print(f"\n   Spectral Shape:")
print(f"      Slope difference: {metrics['slope_diff']:.4f}")
if 'mean_spectrum_correlation' in metrics:
    print(f"      Mean spectrum correlation: {metrics['mean_spectrum_correlation']:.4f}")

print(f"\n   Complexity:")
print(f"      PCA components difference: {metrics['pca_complexity_diff']}")

print(f"\n   Overall Similarity Score: {metrics['overall_score']:.1f}/100")


# =============================================================================
# Section 7: Tuning Recommendations
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Tuning Recommendations")
print("-" * 60)

recommendations = fitter.get_tuning_recommendations()
print("\n💡 Recommendations for improving synthetic data:")
for rec in recommendations:
    print(f"   • {rec}")


# =============================================================================
# Section 8: Using the Quick Function
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Quick Fitting Function")
print("-" * 60)

# Use the convenience function
quick_params = fit_to_real_data(X_real, wavelengths, name="quick_fit")

print(f"\n🚀 Quick fit result:")
print(f"   Complexity: {quick_params.complexity}")
print(f"   Instrument: {quick_params.inferred_instrument}")
print(f"   Domain: {quick_params.inferred_domain}")
print(f"   Suggested components: {quick_params.suggested_n_components}")


# =============================================================================
# Section 9: Comparing Datasets Directly
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Direct Dataset Comparison")
print("-" * 60)

# Use the compare_datasets convenience function
comparison = compare_datasets(X_synth, X_real, wavelengths)

print(f"\n📊 Direct comparison (synthetic vs real):")
print(f"   Mean relative diff: {comparison['mean_rel_diff']:.4f}")
print(f"   Std relative diff: {comparison['std_rel_diff']:.4f}")
print(f"   Overall score: {comparison['overall_score']:.1f}/100")


# =============================================================================
# Section 10: Exporting and Loading Parameters
# =============================================================================
print("\n" + "-" * 60)
print("Section 10: Saving and Loading Parameters")
print("-" * 60)

# Save parameters to JSON
params_file = get_example_output_path(EXAMPLE_NAME, "fitted_params.json")
params.save(str(params_file))
print(f"\n💾 Saved parameters to: {params_file}")

# Load parameters back
loaded_params = FittedParameters.load(str(params_file))
print(f"   Loaded back: {loaded_params.source_name}")
print(f"   Complexity: {loaded_params.complexity}")
print(f"   Instrument: {loaded_params.inferred_instrument}")

# Export full configuration
full_config = params.to_full_config()
print(f"\n📋 Full configuration keys: {list(full_config.keys())}")


# =============================================================================
# Section 11: Parameter Summary
# =============================================================================
print("\n" + "-" * 60)
print("Section 11: Human-Readable Summary")
print("-" * 60)

print(params.summary())


# =============================================================================
# Section 12: Plotting (optional)
# =============================================================================
if args.plots:
    print("\n" + "-" * 60)
    print("Section 12: Visualization")
    print("-" * 60)

    # Plot 1: Mean spectra comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean spectra
    ax = axes[0, 0]
    ax.plot(wavelengths, X_real.mean(axis=0), 'b-', label='Real', linewidth=2)
    ax.plot(wavelengths, X_synth.mean(axis=0), 'r--', label='Synthetic', linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Mean Spectra Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Std spectra
    ax = axes[0, 1]
    ax.plot(wavelengths, X_real.std(axis=0), 'b-', label='Real', linewidth=2)
    ax.plot(wavelengths, X_synth.std(axis=0), 'r--', label='Synthetic', linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Std Dev")
    ax.set_title("Standard Deviation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample spectra - Real
    ax = axes[1, 0]
    for i in range(min(10, len(X_real))):
        ax.plot(wavelengths, X_real[i], alpha=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Real Spectra (10 samples)")
    ax.grid(True, alpha=0.3)

    # Sample spectra - Synthetic
    ax = axes[1, 1]
    for i in range(min(10, len(X_synth))):
        ax.plot(wavelengths, X_synth[i], alpha=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Synthetic Spectra (10 samples)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = get_example_output_path(EXAMPLE_NAME, "spectra_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved spectra comparison: {plot_path}")

    # Plot 2: Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Global distribution
    ax = axes[0]
    ax.hist(X_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(X_synth.flatten(), bins=50, alpha=0.5, label='Synthetic', density=True)
    ax.set_xlabel("Absorbance")
    ax.set_ylabel("Density")
    ax.set_title("Absorbance Distribution")
    ax.legend()

    # Slope distribution
    ax = axes[1]
    real_slopes = fitter.source_properties.slopes
    synth_props = compute_spectral_properties(X_synth, wavelengths, "synth")
    synth_slopes = synth_props.slopes
    ax.hist(real_slopes, bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(synth_slopes, bins=30, alpha=0.5, label='Synthetic', density=True)
    ax.set_xlabel("Slope (per 1000nm)")
    ax.set_ylabel("Density")
    ax.set_title("Spectral Slope Distribution")
    ax.legend()

    # Sample mean distribution
    ax = axes[2]
    ax.hist(X_real.mean(axis=1), bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(X_synth.mean(axis=1), bins=30, alpha=0.5, label='Synthetic', density=True)
    ax.set_xlabel("Sample Mean Absorbance")
    ax.set_ylabel("Density")
    ax.set_title("Sample Mean Distribution")
    ax.legend()

    plt.tight_layout()
    plot_path = get_example_output_path(EXAMPLE_NAME, "distribution_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved distribution comparison: {plot_path}")

    # Plot 3: PCA comparison
    try:
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Real data PCA
        pca_real = PCA(n_components=10).fit(X_real)
        pca_synth = PCA(n_components=10).fit(X_synth)

        # Explained variance
        ax = axes[0]
        ax.bar(range(1, 11), pca_real.explained_variance_ratio_ * 100, alpha=0.5, label='Real')
        ax.bar(range(1, 11), pca_synth.explained_variance_ratio_ * 100, alpha=0.5, label='Synthetic')
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title("PCA Explained Variance")
        ax.legend()

        # PCA score plot
        ax = axes[1]
        scores_real = pca_real.transform(X_real)
        scores_synth = pca_synth.transform(X_synth)
        ax.scatter(scores_real[:, 0], scores_real[:, 1], alpha=0.5, label='Real', s=20)
        ax.scatter(scores_synth[:, 0], scores_synth[:, 1], alpha=0.5, label='Synthetic', s=20)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Score Plot (PC1 vs PC2)")
        ax.legend()

        plt.tight_layout()
        plot_path = get_example_output_path(EXAMPLE_NAME, "pca_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved PCA comparison: {plot_path}")

    except ImportError:
        print("   (sklearn not available for PCA plot)")

    if args.show:
        plt.show()

    print(f"\n📁 Output directory: {plot_path.parent}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Key concepts covered:

1. 📊 Spectral Properties Analysis:
   - Statistical moments (mean, std, skewness, kurtosis)
   - Spectral shape (slope, curvature, peaks)
   - Noise characteristics (SNR estimation)
   - PCA complexity analysis

2. 🔬 Instrument Inference (Phase 2):
   - Automatic detection of instrument archetype
   - Detector type identification
   - Resolution estimation
   - SNR-based quality assessment

3. 📐 Measurement Mode Detection (Phase 2):
   - Transmittance vs reflectance vs ATR
   - Baseline analysis for mode identification
   - Kubelka-Munk linearity scoring

4. 🏭 Domain Inference (Phase 1):
   - Application domain detection (agriculture, food, pharma, etc.)
   - Component identification from band intensities
   - Protein, carbohydrate, lipid, water band analysis

5. 🌡️ Environmental Effects (Phase 3):
   - Temperature effect detection
   - Moisture effect detection
   - Water band shift analysis

6. 💨 Scattering Effects (Phase 3):
   - Particle size estimation
   - MSC/SNV correctability assessment
   - Multiplicative and additive scatter quantification

7. 🔧 Generator Creation:
   - create_matched_generator() for automatic configuration
   - to_full_config() for comprehensive parameter export
   - save()/load() for parameter persistence

8. 📈 Similarity Evaluation:
   - Statistical comparison metrics
   - Spectral shape comparison
   - Overall similarity scoring

Next steps:
- Use fitted parameters for data augmentation
- Apply preprocessing recommendations (SNV, MSC) based on inference
- Create domain-specific synthetic datasets for model training
""")

print("\n✅ Example completed successfully!")
