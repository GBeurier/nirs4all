"""
U05 - Synthetic Data: Generate Test Datasets
=============================================

Learn to generate synthetic NIRS spectra for testing and prototyping.

This tutorial covers:

* Using ``nirs4all.generate()`` for quick dataset creation
* Convenience functions for regression and classification
* Configuring spectral complexity and components
* Integration with pipelines

Prerequisites
-------------
Complete the getting_started examples first.

Next Steps
----------
See :ref:`U06_synthetic_advanced` for builder API and customization.

Duration: ~2 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# NIRS4All imports
import nirs4all

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U05 Synthetic Data Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "U05_synthetic_data"


# =============================================================================
# Section 1: Basic Generation
# =============================================================================
print("\n" + "=" * 60)
print("U05 - Synthetic NIRS Data Generation")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Basic Generation")
print("-" * 60)

# Generate a simple dataset
dataset = nirs4all.generate(n_samples=500, random_state=42)

print(f"\n📊 Generated dataset:")
print(f"   Type: {type(dataset).__name__}")
print(f"   Samples: {dataset.num_samples}")

# Get the shapes
X_train = dataset.x({"partition": "train"}, layout="2d")
y_train = dataset.y({"partition": "train"})

print(f"   Training features: {X_train.shape}")
print(f"   Training targets: {y_train.shape}")


# =============================================================================
# Section 2: Get Raw Arrays
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Get Raw Arrays")
print("-" * 60)

# For quick experiments, get numpy arrays directly
X, y = nirs4all.generate(n_samples=300, as_dataset=False, random_state=42)

print(f"\n📊 Generated arrays:")
print(f"   Features shape: {X.shape}")
print(f"   Targets shape: {y.shape}")
print(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")


# =============================================================================
# Section 3: Regression Datasets
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Regression Convenience Function")
print("-" * 60)

# Generate regression dataset with specific configuration
# Note: When setting target_range, we need a single target (component 0)
dataset_reg = nirs4all.generate.regression(
    n_samples=500,
    target_range=(0, 100),           # Scale targets to 0-100
    target_component=0,              # Use first component as target
    complexity="realistic",          # Add realistic noise/scatter
    random_state=42
)

y_all = dataset_reg.y({})
print(f"\n📊 Regression dataset:")
print(f"   Samples: {dataset_reg.num_samples}")
print(f"   Target range: [{y_all.min():.1f}, {y_all.max():.1f}]")

# Run a quick pipeline
result_reg = nirs4all.run(
    pipeline=[
        StandardScaler(),
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset=dataset_reg,
    name="SyntheticRegression",
    verbose=0
)

print(f"   Pipeline RMSE: {result_reg.best_rmse:.2f}")


# =============================================================================
# Section 4: Classification Datasets
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Classification Convenience Function")
print("-" * 60)

# Binary classification
dataset_binary = nirs4all.generate.classification(
    n_samples=400,
    n_classes=2,
    class_separation=2.0,            # Well-separated classes
    random_state=42
)

print(f"\n📊 Binary classification:")
y_classes = dataset_binary.y({})
unique, counts = np.unique(y_classes, return_counts=True)
print(f"   Classes: {unique.tolist()}")
print(f"   Counts: {counts.tolist()}")

# Multiclass with imbalanced classes
dataset_multi = nirs4all.generate.classification(
    n_samples=600,
    n_classes=3,
    class_weights=[0.5, 0.3, 0.2],   # Imbalanced
    complexity="simple",
    random_state=42
)

print(f"\n📊 Multiclass (imbalanced):")
y_multi = dataset_multi.y({})
unique, counts = np.unique(y_multi, return_counts=True)
print(f"   Classes: {unique.tolist()}")
print(f"   Counts: {counts.tolist()}")

# Run classification pipeline
result_clf = nirs4all.run(
    pipeline=[
        StandardScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": RandomForestClassifier(n_estimators=50, random_state=42)}
    ],
    dataset=dataset_binary,
    name="SyntheticClassification",
    verbose=0
)

print(f"\n   Classification accuracy: {result_clf.best_score:.3f}")


# =============================================================================
# Section 5: Complexity Levels
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Complexity Levels")
print("-" * 60)

print("\nComparing complexity levels on the same pipeline...")

for complexity in ["simple", "realistic", "complex"]:
    # Use regression() with explicit target_component for sklearn models
    dataset_cx = nirs4all.generate.regression(
        n_samples=300,
        complexity=complexity,
        target_component=0,          # Single target for sklearn
        random_state=42
    )

    result = nirs4all.run(
        pipeline=[
            StandardScaler(),
            ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=10)}
        ],
        dataset=dataset_cx,
        verbose=0
    )

    print(f"   {complexity:10s}: RMSE = {result.best_rmse:.4f}")


# =============================================================================
# Section 6: Using Specific Components
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Specific Spectral Components")
print("-" * 60)

# Generate with specific chemical components
dataset_food = nirs4all.generate(
    n_samples=400,
    components=["water", "protein", "lipid", "starch"],
    complexity="realistic",
    random_state=42
)

print(f"\n📊 Food-like composition dataset:")
print(f"   Components: water, protein, lipid, starch")
print(f"   Samples: {dataset_food.num_samples}")


# =============================================================================
# Section 7: Direct Pipeline Integration
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Direct Pipeline Integration")
print("-" * 60)

# Generate and train in one call
print("\nDirect generation in nirs4all.run()...")

result_direct = nirs4all.run(
    pipeline=[
        StandardScaler(),
        ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=15)}
    ],
    # Generate dataset inline with single target!
    dataset=nirs4all.generate.regression(
        n_samples=600,
        complexity="realistic",
        target_range=(0, 50),
        target_component=0,          # Single target for sklearn
        random_state=123
    ),
    name="InlineGeneration",
    verbose=0
)

print(f"   Generated and trained: RMSE = {result_direct.best_rmse:.3f}")


# =============================================================================
# Section 8: Visualization of Generated Data
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Visualization of Generated Data")
print("-" * 60)

# Always save a summary of what was generated
X_all = dataset.x({}, layout="2d")
y_all_dataset = dataset.y({})
summary_path = save_array_summary(
    {"X (spectra)": X_all, "y (targets)": y_all_dataset},
    EXAMPLE_NAME
)
print_output_location(summary_path, "Data summary")

# Generate plots (always saved, optionally shown)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Sample spectra from basic generation
ax1 = axes[0, 0]
wavelengths = np.linspace(1000, 2500, X_all.shape[1])
for i in range(min(20, X_all.shape[0])):
    ax1.plot(wavelengths, X_all[i], alpha=0.5, linewidth=0.8)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Absorbance")
ax1.set_title("Generated NIRS Spectra (20 samples)")
ax1.grid(True, alpha=0.3)

# Plot 2: Target distribution
ax2 = axes[0, 1]
ax2.hist(y_all_dataset.ravel(), bins=30, edgecolor='black', alpha=0.7)
ax2.set_xlabel("Target Value")
ax2.set_ylabel("Frequency")
ax2.set_title("Target Distribution")
ax2.grid(True, alpha=0.3)

# Plot 3: Classification dataset
ax3 = axes[1, 0]
X_clf = dataset_binary.x({}, layout="2d")
y_clf = dataset_binary.y({})
for class_val in np.unique(y_clf):
    mask = y_clf.ravel() == class_val
    for i in np.where(mask)[0][:10]:
        ax3.plot(wavelengths, X_clf[i], alpha=0.5, linewidth=0.8,
                 label=f"Class {int(class_val)}" if i == np.where(mask)[0][0] else "")
ax3.set_xlabel("Wavelength (nm)")
ax3.set_ylabel("Absorbance")
ax3.set_title("Classification Dataset (2 classes)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Complexity comparison
ax4 = axes[1, 1]
for complexity, color in [("simple", "green"), ("realistic", "blue"), ("complex", "red")]:
    X_cx, _ = nirs4all.generate(n_samples=50, complexity=complexity, as_dataset=False, random_state=42)
    mean_spectrum = X_cx.mean(axis=0)
    ax4.plot(wavelengths, mean_spectrum, label=complexity, color=color, linewidth=2)
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Mean Absorbance")
ax4.set_title("Complexity Levels Comparison")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Always save the plot
plot_path = get_example_output_path(EXAMPLE_NAME, "synthetic_data_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print_output_location(plot_path, "Overview plot")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Synthetic Data Generation API:

  nirs4all.generate(n_samples, ...)
      Main function - returns SpectroDataset or (X, y) tuple

  nirs4all.generate.regression(n_samples, ...)
      Convenience for regression with target scaling

  nirs4all.generate.classification(n_samples, n_classes, ...)
      Convenience for classification with class separation

Key Parameters:
  n_samples        Number of samples to generate
  random_state     Random seed for reproducibility
  complexity       "simple" | "realistic" | "complex"
  components       ["water", "protein", ...] - spectral components
  as_dataset       True=SpectroDataset, False=(X, y) tuple
  target_range     (min, max) for target scaling

Complexity Levels:
  simple      Minimal noise, fast (unit tests)
  realistic   Typical NIR noise/scatter (development)
  complex     High noise, artifacts (robustness testing)

Predefined Components:
  water, protein, lipid, starch, cellulose,
  chlorophyll, oil, nitrogen_compound

Next: U06_synthetic_advanced.py - Builder API and customization
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
