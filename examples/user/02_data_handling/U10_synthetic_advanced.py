"""
U10 - Advanced Synthetic Data: Builder API and Customization
=============================================================

Master the full synthetic data generation API for complex scenarios.

This tutorial covers:

* Using ``SyntheticDatasetBuilder`` for full control
* Metadata generation (groups, repetitions)
* Multi-source datasets
* Batch effects simulation
* Exporting to files
* Matching real data characteristics

Prerequisites
-------------
Complete :ref:`U09_synthetic_data` first.

Next Steps
----------
See developer examples for extending the generator.

Duration: ~3 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import tempfile
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.data.synthetic import SyntheticDatasetBuilder
from nirs4all.data import DatasetConfigs

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U10 Advanced Synthetic Data Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


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
# Section 5: Export to Files
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Export to Files")
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
# Section 6: Single CSV Export
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Single CSV Export")
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
# Section 7: Matching Real Data (Template Fitting)
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Matching Real Data Characteristics")
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
# Section 8: Full Builder Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Complete Builder Example")
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
# Section 9: Pipeline Integration
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Complete Pipeline Integration")
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

Export Functions:

  nirs4all.generate.to_folder(path, n_samples, format, ...)
      Formats: "standard", "single", "fragmented"

  nirs4all.generate.to_csv(path, n_samples, ...)
      Single CSV file with all data

  nirs4all.generate.from_template(X_real, n_samples, ...)
      Generate data matching real data characteristics

Advanced Features:

  Metadata      n_groups for GroupKFold, n_repetitions for
                replicate measurements

  Multi-source  Combine NIR spectra with auxiliary data
                (markers, sensors, etc.)

  Batch Effects Simulate measurement session variations
                for domain adaptation research

  Template      Analyze real data and generate synthetic
                data with similar statistical properties

Key Use Cases:

  • Unit tests with reproducible synthetic data
  • Algorithm benchmarking with known ground truth
  • Prototyping before real data is available
  • Teaching NIRS concepts with controllable examples
  • Domain adaptation research with batch effects
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
