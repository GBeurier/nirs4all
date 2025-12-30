"""
U02 - SHAP with sklearn Wrapper: Advanced Analysis
===================================================

Use SHAP with NIRSPipeline for advanced model analysis.

This tutorial covers:

* SHAP with NIRSPipeline wrapper
* Custom SHAP explainers
* Feature importance extraction
* Integration with sklearn tools

Prerequisites
-------------
Complete :ref:`U01_shap_basics` first.
Requires: pip install shap

Next Steps
----------
See :ref:`U03_feature_selection` for wavelength selection.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.sklearn import NIRSPipeline
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.data import DatasetConfigs

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 SHAP sklearn Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why SHAP with sklearn Wrapper?
# =============================================================================
print("\n" + "=" * 60)
print("U02 - SHAP with sklearn Wrapper")
print("=" * 60)

print("""
Using SHAP with NIRSPipeline gives you:

  🔧 DIRECT SHAP INTEGRATION
     - Use standard SHAP explainers
     - Access underlying model
     - Custom analysis workflows

  📊 FLEXIBILITY
     - Model-specific explainers
     - Custom background samples
     - Fine-grained control

  📈 ADVANCED ANALYSIS
     - Feature importance ranking
     - Interaction effects
     - Custom visualizations
""")


# =============================================================================
# Section 2: Check SHAP Installation
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Check SHAP Installation")
print("-" * 60)

try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ SHAP is installed")
except ImportError:
    SHAP_AVAILABLE = False
    print("✗ SHAP not installed")
    print("  Install with: pip install shap")
    print("  Continuing with limited examples...")


# =============================================================================
# Section 3: Train and Wrap Model
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Train and Wrap Model")
print("-" * 60)

# Train a model
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="shap_sklearn",
    verbose=1,
    save_artifacts=True,
    plots_visible=False
)

print(f"\nTraining complete!")
print(f"  RMSE: {result.best_rmse:.4f}")
r2_val = result.best_r2
print(f"  R²: {r2_val:.4f}" if not np.isnan(r2_val) else "  R²: (see test metrics)")

# Wrap for sklearn
pipe = NIRSPipeline.from_result(result)
print(f"\n✓ NIRSPipeline created: {pipe.model_name}")


# =============================================================================
# Section 4: Get Data for SHAP
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Prepare Data for SHAP")
print("-" * 60)

# Load dataset
dataset = DatasetConfigs("sample_data/regression")
for config, name in dataset.configs:
    ds = dataset.get_dataset(config, name)
    X = ds.x({})
    y = ds.y({})  # y() is a method, needs selector
    break

print(f"Data shape: X={X.shape}, y={y.shape}")

# Define background and test samples
background = X[:50]  # Background samples for SHAP
test_samples = X[50:60]  # Samples to explain

print(f"Background samples: {background.shape}")
print(f"Test samples: {test_samples.shape}")


# =============================================================================
# Section 5: SHAP with KernelExplainer
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: SHAP with KernelExplainer")
print("-" * 60)

if SHAP_AVAILABLE:
    print("""
    KernelExplainer works with any model.
    It uses the NIRSPipeline's predict function.
    """)

    # Create Kernel SHAP explainer
    print("Creating SHAP explainer...")

    # Use kmeans for background summary (faster)
    background_summary = shap.kmeans(background, 10)

    explainer = shap.KernelExplainer(
        pipe.predict,  # Use wrapper's predict
        background_summary
    )

    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(
        test_samples,
        nsamples=100  # Number of samples for estimation
    )

    print(f"SHAP values shape: {shap_values.shape}")

    # Feature importance
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    print(f"\nTop 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
else:
    print("SHAP not available - skipping example")


# =============================================================================
# Section 6: SHAP with Tree Models
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: SHAP with Tree Models")
print("-" * 60)

print("""
Tree models (Random Forest, GradientBoosting) can use
TreeExplainer for faster SHAP computation.
""")

# Train a tree model
pipeline_tree = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)}
]

result_tree = nirs4all.run(
    pipeline=pipeline_tree,
    dataset="sample_data/regression",
    name="shap_tree",
    verbose=1,
    save_artifacts=True,
    plots_visible=False
)

pipe_tree = NIRSPipeline.from_result(result_tree)
print(f"\n✓ Tree model wrapped: {pipe_tree.model_name}")

if SHAP_AVAILABLE:
    # Access underlying model for TreeExplainer
    model = pipe_tree.model_

    # Transform data first (apply preprocessing)
    X_transformed = pipe_tree.transform(X[:100])

    # Use TreeExplainer
    print("Creating TreeExplainer...")
    tree_explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    tree_shap_values = tree_explainer.shap_values(X_transformed[:10])

    print(f"SHAP values shape: {tree_shap_values.shape}")

    # Compare speed vs Kernel
    print("\nTreeExplainer is much faster than KernelExplainer!")
else:
    print("SHAP not available - skipping example")


# =============================================================================
# Section 7: Feature Importance Summary
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Feature Importance Summary")
print("-" * 60)

if SHAP_AVAILABLE:
    print("""
    Extract overall feature importance from SHAP values.
    """)

    # Calculate global importance
    global_importance = np.mean(np.abs(shap_values), axis=0)

    # Group into spectral regions (bins)
    n_features = len(global_importance)
    bin_size = 20
    n_bins = n_features // bin_size

    print(f"\nSpectral region importance (bin_size={bin_size}):")
    for i in range(min(5, n_bins)):  # Show first 5 bins
        start = i * bin_size
        end = start + bin_size
        region_importance = np.mean(global_importance[start:end])
        print(f"  Region {start}-{end}: {region_importance:.4f}")

    # Find most important region
    region_importances = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        region_importances.append((start, end, np.mean(global_importance[start:end])))

    best_region = max(region_importances, key=lambda x: x[2])
    print(f"\nMost important region: {best_region[0]}-{best_region[1]}")
else:
    print("SHAP not available - skipping example")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
SHAP with sklearn Wrapper:

  1. TRAIN AND WRAP:
     result = nirs4all.run(pipeline=..., dataset=...)
     pipe = NIRSPipeline.from_result(result)

  2. KERNEL EXPLAINER (any model):
     background = shap.kmeans(X, 10)
     explainer = shap.KernelExplainer(pipe.predict, background)
     shap_values = explainer.shap_values(X_test)

  3. TREE EXPLAINER (tree models):
     # Access underlying model
     model = pipe.model_
     X_prep = pipe.transform(X)  # Apply preprocessing
     explainer = shap.TreeExplainer(model)
     shap_values = explainer.shap_values(X_prep)

  4. FEATURE IMPORTANCE:
     importance = np.mean(np.abs(shap_values), axis=0)
     top_features = np.argsort(importance)[-10:][::-1]

Explainer Types:
  - KernelExplainer: Any model (slowest)
  - TreeExplainer: Tree models (fastest)
  - LinearExplainer: Linear models (fast)
  - DeepExplainer: Neural networks

Tips:
  - Use shap.kmeans() for efficient background
  - TreeExplainer is 10-100x faster
  - Transform data before TreeExplainer
  - Access pipe.model_ for model-specific explainers

Next: U03_feature_selection.py - Wavelength selection methods
""")
