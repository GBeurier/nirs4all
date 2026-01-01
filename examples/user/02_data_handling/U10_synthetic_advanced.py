"""
U10 - Advanced Synthetic Data: Builder API and Customization
=============================================================

Master the full synthetic data generation API for complex scenarios.

This tutorial covers:

* Using ``SyntheticDatasetBuilder`` for full control
* Metadata generation (groups, repetitions)
* Multi-source datasets
* Batch effects simulation
* **NEW**: Non-linear target complexity for realistic benchmarks
* Exporting to files
* Matching real data characteristics

Prerequisites
-------------
Complete :ref:`U09_synthetic_data` first.

Next Steps
----------
See developer examples for extending the generator.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import sys
import tempfile
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.data.synthetic import SyntheticDatasetBuilder
from nirs4all.data import DatasetConfigs

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U10 Advanced Synthetic Data Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "U10_synthetic_advanced"


# =============================================================================
# Section 1: Builder Pattern Basics
# =============================================================================
print("\n" + "=" * 60)
print("U10 - Advanced Synthetic Data Generation")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Builder Pattern Basics")
print("-" * 60)

# Create a dataset using the builder
dataset = (
    SyntheticDatasetBuilder(n_samples=500, random_state=42)
    .with_features(
        wavelength_range=(1000, 2500),
        complexity="realistic",
        components=["water", "protein", "lipid"]
    )
    .with_targets(
        distribution="lognormal",
        range=(5, 50),
        component="protein"          # Use protein as primary target
    )
    .with_partitions(train_ratio=0.8)
    .build()
)

print(f"\n📊 Builder-created dataset:")
print(f"   Samples: {dataset.num_samples}")

y = dataset.y({})
print(f"   Target range: [{y.min():.1f}, {y.max():.1f}]")


# =============================================================================
# Section 2: Metadata Generation
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Metadata Generation (Groups & Repetitions)")
print("-" * 60)

# Generate dataset with sample groupings
# Useful for GroupKFold cross-validation
dataset_meta = (
    SyntheticDatasetBuilder(n_samples=300, random_state=42)
    .with_features(complexity="realistic")
    .with_metadata(
        n_groups=5,                  # 5 sample groups
        n_repetitions=(2, 4),        # 2-4 measurements per biological sample
        sample_id_prefix="sample"
    )
    .with_partitions(train_ratio=0.8)
    .build()
)

print(f"\n📊 Dataset with metadata:")
print(f"   Samples: {dataset_meta.num_samples}")
print(f"   Groups configured: 5")
print(f"   Repetitions: 2-4 per biological sample")

# This would enable group-based cross-validation
print("\n   Use with GroupKFold for proper validation!")


# =============================================================================
# Section 3: Multi-Source Datasets
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Multi-Source Datasets")
print("-" * 60)

# Generate dataset with multiple data sources
# Common scenario: NIR spectra + chemical markers
dataset_multi = nirs4all.generate.multi_source(
    n_samples=400,
    sources=[
        {
            "name": "NIR",
            "type": "nir",
            "wavelength_range": (1000, 2500),
            "complexity": "realistic"
        },
        {
            "name": "markers",
            "type": "aux",
            "n_features": 15
        }
    ],
    random_state=42
)

print(f"\n📊 Multi-source dataset:")
print(f"   Samples: {dataset_multi.num_samples}")
print(f"   Sources: NIR (spectra) + markers (15 features)")


# =============================================================================
# Section 4: Batch Effects (Domain Adaptation)
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Batch Effects Simulation")
print("-" * 60)

# Simulate measurement batch variations
# Useful for domain adaptation research
dataset_batch = (
    SyntheticDatasetBuilder(n_samples=600, random_state=42)
    .with_features(complexity="realistic")
    .with_batch_effects(
        enabled=True,
        n_batches=3                  # 3 measurement sessions
    )
    .with_partitions(train_ratio=0.8)
    .build()
)

print(f"\n📊 Dataset with batch effects:")
print(f"   Samples: {dataset_batch.num_samples}")
print(f"   Batches: 3 simulated measurement sessions")
print("\n   Batch effects add systematic variations between sessions.")
print("   Useful for testing domain adaptation algorithms.")


# =============================================================================
# Section 5: Non-Linear Target Complexity (NEW!)
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Non-Linear Target Complexity (NEW!)")
print("-" * 60)

print("\n⚠️  Default synthetic targets are too easy to predict!")
print("   Use these options to create more realistic challenges.\n")

# 5a: Polynomial Interactions
print("5a) Polynomial interactions (C₁², C₁×C₂, etc.):")
dataset_poly = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_nonlinear_targets(
        interactions="polynomial",      # polynomial, synergistic, antagonistic
        interaction_strength=0.6,        # 0=linear, 1=fully non-linear
        polynomial_degree=2              # Quadratic terms
    )
    .with_partitions(train_ratio=0.8)
    .build()
)
print(f"   Created dataset with polynomial target relationships")

# 5b: Hidden Factors (unexplainable variance)
print("\n5b) Hidden factors (latent variables not in spectra):")
dataset_hidden = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_nonlinear_targets(
        hidden_factors=3                # 3 latent variables affect target
    )
    .with_partitions(train_ratio=0.8)
    .build()
)
print(f"   Created dataset with 3 hidden factors (irreducible error)")

# 5c: Confounders and Partial Predictability
print("\n5c) Confounders (partial predictability):")
dataset_confound = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_target_complexity(
        signal_to_confound_ratio=0.7,   # Only 70% of target is predictable
        n_confounders=2,                 # 2 confounding variables
        temporal_drift=True              # Relationship changes over samples
    )
    .with_partitions(train_ratio=0.8)
    .build()
)
print(f"   Created dataset with 70% predictable target + temporal drift")

# 5d: Multi-Regime Landscapes
print("\n5d) Multi-regime landscapes (subpopulations):")
dataset_regime = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    .with_complex_target_landscape(
        n_regimes=3,                     # 3 different relationship regimes
        regime_method="concentration",   # Partition by concentration space
        regime_overlap=0.2,              # Smooth transitions
        noise_heteroscedasticity=0.5     # Noise varies by region
    )
    .with_partitions(train_ratio=0.8)
    .build()
)
print(f"   Created dataset with 3 regimes + heteroscedastic noise")

# 5e: Combining All Complexity Features
print("\n5e) Combining all complexity features (realistic benchmark):")
dataset_hard = (
    SyntheticDatasetBuilder(n_samples=500, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(component=0, range=(0, 100))
    # Non-linear interactions
    .with_nonlinear_targets(
        interactions="polynomial",
        interaction_strength=0.4,
        hidden_factors=2
    )
    # Confounders
    .with_target_complexity(
        signal_to_confound_ratio=0.75,
        n_confounders=2
    )
    # Multi-regime
    .with_complex_target_landscape(
        n_regimes=3,
        noise_heteroscedasticity=0.3
    )
    .with_partitions(train_ratio=0.8)
    .build()
)
print(f"   Created challenging benchmark dataset")

# Quick comparison of prediction difficulty
print("\n📊 Difficulty comparison (lower R² = harder):")
from sklearn.metrics import r2_score

for name, ds in [("Simple linear", dataset), ("All complexity", dataset_hard)]:
    X_tr = ds.x({"partition": "train"}, layout="2d")
    y_tr = ds.y({"partition": "train"})
    X_te = ds.x({"partition": "test"}, layout="2d")
    y_te = ds.y({"partition": "test"})

    pls = PLSRegression(n_components=10)
    pls.fit(X_tr, y_tr)
    y_pred = pls.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    print(f"   {name:20s}: R² = {r2:.3f}")


# =============================================================================
# Section 6: Export to Files
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Export to Files")
print("-" * 60)

# Export synthetic data to files (DatasetConfigs compatible)
with tempfile.TemporaryDirectory() as tmpdir:
    # Export to folder
    export_path = Path(tmpdir) / "synthetic_dataset"

    path = nirs4all.generate.to_folder(
        export_path,
        n_samples=200,
        train_ratio=0.8,
        complexity="simple",
        random_state=42,
        format="standard"           # Creates Xcal.csv, Ycal.csv, etc.
    )

    print(f"\n📁 Exported to folder: {path.name}/")
    print("   Files created:")
    for f in sorted(path.iterdir()):
        print(f"   - {f.name}")

    # Load back with DatasetConfigs
    loaded_dataset = DatasetConfigs(str(path)).get_datasets()[0]
    print(f"\n   Loaded back: {loaded_dataset.num_samples} samples")


# =============================================================================
# Section 7: Single CSV Export
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Single CSV Export")
print("-" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "synthetic_data.csv"

    path = nirs4all.generate.to_csv(
        csv_path,
        n_samples=100,
        complexity="simple",
        random_state=42
    )

    print(f"\n📄 Exported to CSV: {path.name}")

    # Show file size
    import os
    size_kb = os.path.getsize(path) / 1024
    print(f"   File size: {size_kb:.1f} KB")


# =============================================================================
# Section 8: Matching Real Data (Template Fitting)
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Matching Real Data Characteristics")
print("-" * 60)

# First, create some "real" data to mimic
real_like = nirs4all.generate(
    n_samples=200,
    complexity="realistic",
    random_state=99
)
X_real = real_like.x({}, layout="2d")

# Now generate synthetic data that matches its characteristics
dataset_fitted = nirs4all.generate.from_template(
    X_real,
    n_samples=500,
    random_state=42
)

print(f"\n📊 Template-fitted dataset:")
print(f"   Template samples: {X_real.shape[0]}")
print(f"   Generated samples: {dataset_fitted.num_samples}")
print("\n   The fitter analyzes statistical properties and spectral")
print("   shape to create synthetic data with similar characteristics.")


# =============================================================================
# Section 9: Full Builder Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Complete Builder Example")
print("-" * 60)

# Demonstrate all builder options together
full_dataset = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    # Spectral features
    .with_features(
        wavelength_range=(1100, 2400),
        wavelength_step=4,           # Sparse sampling
        complexity="realistic",
        components=["water", "protein", "lipid", "starch"]
    )
    # Target configuration
    .with_targets(
        distribution="lognormal",
        range=(10, 80),
        component="protein"
    )
    # Metadata for grouping
    .with_metadata(
        n_groups=4,
        n_repetitions=2
    )
    # Train/test split
    .with_partitions(
        train_ratio=0.75,
        shuffle=True
    )
    # Batch variations
    .with_batch_effects(
        enabled=True,
        n_batches=2
    )
    .build()
)

# Get configuration
config = (
    SyntheticDatasetBuilder(n_samples=400, random_state=42)
    .with_features(complexity="realistic")
    .get_config()
)

print(f"\n📊 Full configuration dataset:")
print(f"   Samples: {full_dataset.num_samples}")
print(f"\n   Configuration saved for reproducibility:")
print(f"   - n_samples: {config.n_samples}")
print(f"   - complexity: {config.features.complexity}")
print(f"   - train_ratio: {config.partitions.train_ratio}")


# =============================================================================
# Section 10: Pipeline Integration
# =============================================================================
print("\n" + "-" * 60)
print("Section 10: Complete Pipeline Integration")
print("-" * 60)

# Build a comprehensive synthetic test
print("\nTesting preprocessing methods on synthetic data...")

dataset_test = (
    SyntheticDatasetBuilder(n_samples=300, random_state=42)
    .with_features(
        complexity="realistic",
        components=["water", "protein", "lipid"]
    )
    .with_targets(range=(0, 100), component="protein")
    .with_partitions(train_ratio=0.8)
    .build()
)

result = nirs4all.run(
    pipeline=[
        StandardScaler(),
        ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=dataset_test,
    name="FullBuilderTest",
    verbose=0
)

print(f"\n   Pipeline result: RMSE = {result.best_rmse:.2f}")


# =============================================================================
# Section 11: Visualization of Generated Data
# =============================================================================
print("\n" + "-" * 60)
print("Section 11: Visualization of Generated Data")
print("-" * 60)

# Save summary of what was generated
X_builder = full_dataset.x({}, layout="2d")
y_builder = full_dataset.y({})
summary_path = save_array_summary(
    {"X (spectra)": X_builder, "y (targets)": y_builder},
    EXAMPLE_NAME
)
print_output_location(summary_path, "Data summary")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Builder dataset spectra colored by target
ax1 = axes[0, 0]
wavelengths = np.linspace(1100, 2400, X_builder.shape[1])
colors = y_builder.ravel()
norm_colors = (colors - colors.min()) / (colors.max() - colors.min())
for i in range(min(50, X_builder.shape[0])):
    ax1.plot(wavelengths, X_builder[i], alpha=0.5, linewidth=0.8,
             color=plt.cm.viridis(norm_colors[i]))
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Absorbance")
ax1.set_title("Builder Dataset (colored by target value)")
ax1.grid(True, alpha=0.3)

# Plot 2: Target distribution (lognormal)
ax2 = axes[0, 1]
ax2.hist(y_builder.ravel(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax2.set_xlabel("Target Value")
ax2.set_ylabel("Frequency")
ax2.set_title("Target Distribution (lognormal)")
ax2.axvline(y_builder.mean(), color='red', linestyle='--', label=f'Mean: {y_builder.mean():.1f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Batch effects visualization
ax3 = axes[1, 0]
X_batch = dataset_batch.x({}, layout="2d")
wavelengths_batch = np.linspace(1100, 2400, X_batch.shape[1])  # Create wavelengths for batch dataset
n_per_batch = len(X_batch) // 3
colors_batch = ['blue', 'green', 'orange']
for batch_idx in range(3):
    start = batch_idx * n_per_batch
    end = start + min(20, n_per_batch)
    for i in range(start, end):
        ax3.plot(wavelengths_batch, X_batch[i], alpha=0.4, linewidth=0.7,
                 color=colors_batch[batch_idx],
                 label=f"Batch {batch_idx+1}" if i == start else "")
ax3.set_xlabel("Wavelength (nm)")
ax3.set_ylabel("Absorbance")
ax3.set_title("Batch Effects (3 measurement sessions)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Component spectra (from full builder dataset)
ax4 = axes[1, 1]
X_full = full_dataset.x({}, layout="2d")
mean_spectrum = X_full.mean(axis=0)
std_spectrum = X_full.std(axis=0)
wl_full = np.linspace(1100, 2400, X_full.shape[1])
ax4.fill_between(wl_full, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                 alpha=0.3, color='steelblue', label='±1 std')
ax4.plot(wl_full, mean_spectrum, color='navy', linewidth=2, label='Mean spectrum')
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Absorbance")
ax4.set_title("Full Builder Dataset: Mean ± Std")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Always save the plot
plot_path = get_example_output_path(EXAMPLE_NAME, "synthetic_advanced_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print_output_location(plot_path, "Overview plot")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Builder API Methods:

  SyntheticDatasetBuilder(n_samples, random_state, name)
      .with_features(wavelength_range, complexity, components)
      .with_targets(distribution, range, component)
      .with_classification(n_classes, separation, class_weights)
      .with_metadata(n_groups, n_repetitions, sample_ids)
      .with_sources([source_configs])
      .with_partitions(train_ratio, stratify, shuffle)
      .with_batch_effects(enabled, n_batches)
      .with_output(as_dataset, include_metadata)
      .build()

  NEW - Target Complexity Methods:

      .with_nonlinear_targets(interactions, interaction_strength,
                              hidden_factors, polynomial_degree)
          interactions: "polynomial", "synergistic", "antagonistic"
          Creates non-linear target relationships

      .with_target_complexity(signal_to_confound_ratio, n_confounders,
                              spectral_masking, temporal_drift)
          Adds confounders and partial predictability

      .with_complex_target_landscape(n_regimes, regime_method,
                                      regime_overlap, noise_heteroscedasticity)
          Creates multi-regime targets with subpopulations

Export Functions:

  nirs4all.generate.to_folder(path, n_samples, format, ...)
      Formats: "standard", "single", "fragmented"

  nirs4all.generate.to_csv(path, n_samples, ...)
      Single CSV file with all data

  nirs4all.generate.from_template(X_real, n_samples, ...)
      Generate data matching real data characteristics

Advanced Features:

  Metadata       n_groups for GroupKFold, n_repetitions for
                 replicate measurements

  Multi-source   Combine NIR spectra with auxiliary data
                 (markers, sensors, etc.)

  Batch Effects  Simulate measurement session variations
                 for domain adaptation research

  Template       Analyze real data and generate synthetic
                 data with similar statistical properties

  Non-Linear     Create challenging datasets with:
  Targets        - Polynomial/synergistic/antagonistic effects
                 - Hidden factors (irreducible error)
                 - Confounders (partial predictability)
                 - Multi-regime landscapes (subpopulations)
                 - Heteroscedastic noise

Key Use Cases:

  • Unit tests with reproducible synthetic data
  • Algorithm benchmarking with known ground truth
  • Prototyping before real data is available
  • Teaching NIRS concepts with controllable examples
  • Domain adaptation research with batch effects
  • Method comparison with challenging non-linear targets
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
