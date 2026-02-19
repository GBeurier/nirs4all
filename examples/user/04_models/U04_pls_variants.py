"""
U04 - PLS Variants: Specialized Partial Least Squares Methods
==============================================================

Explore PLS variations for different spectroscopy scenarios.

This tutorial covers:

* Standard PLSRegression and PLSDA
* IKPLS (Improved Kernel PLS) for speed (requires: pip install ikpls)
* OPLS/OPLSDA (Orthogonal PLS) for filtering
* SparsePLS for variable selection
* SIMPLS, IntervalPLS, RobustPLS, KernelPLS

Prerequisites
-------------
Complete :ref:`U01_multi_model` first.

Next Steps
----------
See :ref:`05_cross_validation/U01_cv_strategies` for cross-validation methods.

Duration: ~6 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all

# PLS operators from nirs4all
from nirs4all.operators.models.sklearn import (
    IKPLS,  # Improved Kernel PLS (fast) - requires ikpls package
    OPLS,  # Orthogonal PLS
    OPLSDA,  # Orthogonal PLS-DA
    PLSDA,  # PLS Discriminant Analysis
    SIMPLS,  # de Jong 1993 algorithm
    SparsePLS,  # Sparse PLS for variable selection
)
from nirs4all.operators.models.sklearn.ipls import IntervalPLS
from nirs4all.operators.models.sklearn.nlpls import KernelPLS
from nirs4all.operators.models.sklearn.robust_pls import RobustPLS
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate
from nirs4all.utils.backend import IKPLS_AVAILABLE
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 PLS Variants Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Overview of PLS Variants
# =============================================================================
print("\n" + "=" * 60)
print("U04 - PLS Variants")
print("=" * 60)

# Check for optional IKPLS package
if IKPLS_AVAILABLE:
    print("✓ ikpls package detected - IKPLS models available")
else:
    print("✗ ikpls package not installed - IKPLS examples will be skipped")
    print("  Install with: pip install ikpls")

print("""
Partial Least Squares (PLS) has many variants for different use cases:

  📊 STANDARD PLS
     PLSRegression - sklearn standard implementation
     PLSDA         - PLS Discriminant Analysis (classification)

  ⚡ FAST IMPLEMENTATIONS
     IKPLS         - Improved Kernel PLS (faster for large data)
     SIMPLS        - de Jong 1993 algorithm

  🎯 ORTHOGONAL FILTERING
     OPLS          - Removes Y-orthogonal variation
     OPLSDA        - Orthogonal PLS-DA for classification

  🔍 VARIABLE SELECTION
     SparsePLS     - L1 regularization for sparse loadings
     IntervalPLS   - Wavelength interval selection

  🛡️ ROBUST / NONLINEAR
     RobustPLS     - Outlier-resistant PLS
     KernelPLS     - Nonlinear PLS using kernels
""")

# =============================================================================
# Section 2: Standard PLS vs IKPLS
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Standard PLS vs IKPLS")
print("-" * 60)

print("""
IKPLS (Improved Kernel PLS) is a faster implementation.
Useful for large datasets with many samples or features.
""")

# Build pipeline with standard PLS models
pipeline_ikpls = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # Standard sklearn PLS
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": PLSRegression(n_components=15), "name": "PLS-15"},
]

# Add IKPLS models only if ikpls package is available
if IKPLS_AVAILABLE:
    pipeline_ikpls.extend([
        {"model": IKPLS(n_components=5, backend='numpy'), "name": "IKPLS-5"},
        {"model": IKPLS(n_components=10, backend='numpy'), "name": "IKPLS-10"},
        {"model": IKPLS(n_components=15, backend='numpy'), "name": "IKPLS-15"},
    ])
else:
    print("  (Skipping IKPLS models - ikpls package not installed)")

result_ikpls = nirs4all.run(
    pipeline=pipeline_ikpls,
    dataset="sample_data/regression",
    name="IKPLS",
    verbose=1
)

print("\nPLS vs IKPLS comparison:")
for pred in result_ikpls.top(10, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 3: OPLS - Orthogonal PLS
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: OPLS - Orthogonal PLS")
print("-" * 60)

print("""
OPLS separates Y-predictive from Y-orthogonal variation.
Improves interpretability by filtering irrelevant variation.

Parameters:
  n_components   - Number of orthogonal components to remove
  pls_components - Number of predictive PLS components
""")

pipeline_opls = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    # Standard PLS baseline
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},

    # OPLS with different orthogonal components
    {"model": OPLS(n_components=1, pls_components=1, backend='numpy'), "name": "OPLS-1ortho"},
    {"model": OPLS(n_components=2, pls_components=1, backend='numpy'), "name": "OPLS-2ortho"},
    {"model": OPLS(n_components=3, pls_components=1, backend='numpy'), "name": "OPLS-3ortho"},
]

result_opls = nirs4all.run(
    pipeline=pipeline_opls,
    dataset="sample_data/regression",
    name="OPLS",
    verbose=1
)

print("\nOPLS results:")
for pred in result_opls.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 4: SparsePLS - Variable Selection
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: SparsePLS - Variable Selection")
print("-" * 60)

print("""
SparsePLS uses L1 regularization to select relevant wavelengths.
Higher alpha = more sparsity (fewer selected variables).
""")

pipeline_sparse = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    # Standard PLS
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},

    # SparsePLS with different sparsity levels
    {"model": SparsePLS(n_components=5, alpha=0.1, backend='numpy'), "name": "SparsePLS-a0.1"},
    {"model": SparsePLS(n_components=5, alpha=0.5, backend='numpy'), "name": "SparsePLS-a0.5"},
    {"model": SparsePLS(n_components=5, alpha=1.0, backend='numpy'), "name": "SparsePLS-a1.0"},
]

result_sparse = nirs4all.run(
    pipeline=pipeline_sparse,
    dataset="sample_data/regression",
    name="SparsePLS",
    verbose=1
)

print("\nSparsePLS results:")
for pred in result_sparse.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 5: IntervalPLS - Wavelength Regions
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: IntervalPLS - Wavelength Regions")
print("-" * 60)

print("""
IntervalPLS (iPLS) selects optimal wavelength intervals.
Useful for identifying informative spectral regions.

Modes:
  single  - Select single best interval
  forward - Forward selection of intervals
""")

pipeline_ipls = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    # Standard PLS
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},

    # IntervalPLS
    {"model": IntervalPLS(n_components=5, n_intervals=10, mode='single', backend='numpy'),
     "name": "iPLS-single"},
    {"model": IntervalPLS(n_components=5, n_intervals=10, mode='forward', backend='numpy'),
     "name": "iPLS-forward"},
]

result_ipls = nirs4all.run(
    pipeline=pipeline_ipls,
    dataset="sample_data/regression",
    name="IntervalPLS",
    verbose=1
)

print("\nIntervalPLS results:")
for pred in result_ipls.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 6: RobustPLS - Outlier Handling
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: RobustPLS - Outlier Handling")
print("-" * 60)

print("""
RobustPLS down-weights outliers during training.

Weighting schemes:
  huber - Huber loss (moderate outlier handling)
  tukey - Tukey's biweight (aggressive outlier rejection)
""")

pipeline_robust = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": RobustPLS(n_components=5, weighting='huber', max_iter=50, backend='numpy'),
     "name": "RobustPLS-huber"},
    {"model": RobustPLS(n_components=5, weighting='tukey', max_iter=50, backend='numpy'),
     "name": "RobustPLS-tukey"},
]

result_robust = nirs4all.run(
    pipeline=pipeline_robust,
    dataset="sample_data/regression",
    name="RobustPLS",
    verbose=1
)

print("\nRobustPLS results:")
for pred in result_robust.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 7: KernelPLS - Nonlinear PLS
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: KernelPLS - Nonlinear PLS")
print("-" * 60)

print("""
KernelPLS captures nonlinear relationships using kernel methods.

Kernels:
  linear - Standard linear PLS
  rbf    - Radial Basis Function (nonlinear)
  poly   - Polynomial kernel
""")

pipeline_kernel = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": KernelPLS(n_components=5, kernel='linear', backend='numpy'), "name": "KernelPLS-linear"},
    {"model": KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy'), "name": "KernelPLS-rbf"},
]

result_kernel = nirs4all.run(
    pipeline=pipeline_kernel,
    dataset="sample_data/regression",
    name="KernelPLS",
    verbose=1
)

print("\nKernelPLS results:")
for pred in result_kernel.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 8: Classification - PLSDA and OPLSDA
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Classification - PLSDA and OPLSDA")
print("-" * 60)

print("""
PLSDA and OPLSDA are PLS-based classifiers.
""")

pipeline_plsda = [
    StandardScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    # PLSDA
    {"model": PLSDA(n_components=5), "name": "PLSDA-5"},
    {"model": PLSDA(n_components=10), "name": "PLSDA-10"},

    # OPLSDA
    {"model": OPLSDA(n_components=1, pls_components=5), "name": "OPLSDA-1ortho"},
    {"model": OPLSDA(n_components=2, pls_components=5), "name": "OPLSDA-2ortho"},
]

result_plsda = nirs4all.run(
    pipeline=pipeline_plsda,
    dataset="sample_data/classification",
    name="PLSDA",
    verbose=1
)

print("\nPLSDA/OPLSDA results:")
for pred in result_plsda.top(5, display_metrics=['rmse', 'r2']):
    model = pred.get('model_name', 'Unknown')
    accuracy = (1 - pred.get('rmse', 0)) * 100
    print(f"   {model}: Accuracy={accuracy:.1f}%")

# =============================================================================
# Section 9: Comprehensive Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Comprehensive Comparison")
print("-" * 60)

pipeline_all = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),
    FirstDerivative(),

    ShuffleSplit(n_splits=3, random_state=42),

    # Standard
    {"model": PLSRegression(n_components=10), "name": "sklearn-PLS"},

    # Fast - SIMPLS (always available)
    {"model": SIMPLS(n_components=10, backend='numpy'), "name": "SIMPLS"},

    # Orthogonal
    {"model": OPLS(n_components=2, pls_components=1, backend='numpy'), "name": "OPLS"},

    # Variable selection
    {"model": SparsePLS(n_components=5, alpha=0.5, backend='numpy'), "name": "SparsePLS"},

    # Robust
    {"model": RobustPLS(n_components=10, weighting='huber', backend='numpy'), "name": "RobustPLS"},

    # Nonlinear
    {"model": KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy'), "name": "KernelPLS"},
]

# Add IKPLS if available
if IKPLS_AVAILABLE:
    # Insert after sklearn-PLS for proper comparison grouping
    pipeline_all.insert(6, {"model": IKPLS(n_components=10, backend='numpy'), "name": "IKPLS"})

result_all = nirs4all.run(
    pipeline=pipeline_all,
    dataset="sample_data/regression",
    name="AllPLS",
    verbose=1
)

print("\nAll PLS variants ranked:")
for i, pred in enumerate(result_all.top(10, display_metrics=['rmse', 'r2']), 1):
    print(f"   {i}. {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 10: Visualization
# =============================================================================
if args.plots:
    analyzer = PredictionAnalyzer(result_all.predictions)
    fig1 = analyzer.plot_top_k(k=7, rank_metric='rmse')
    fig2 = analyzer.plot_candlestick(variable="model_name", display_partition="test")

    if args.show:
        plt.show()

# =============================================================================
# Validation: Ensure all results are valid (no NaN metrics)
# =============================================================================
import os
import sys

# Add examples dir to find example_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from example_utils import validate_results

# Validate all results - will exit with code 1 if any have NaN metrics
validate_results(
    [result_ikpls, result_opls, result_sparse, result_ipls,
     result_robust, result_kernel, result_plsda, result_all],
    names=["IKPLS", "OPLS", "SparsePLS", "IntervalPLS",
           "RobustPLS", "KernelPLS", "PLSDA", "AllPLS"]
)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
PLS Variant Selection Guide:

  ┌─────────────────┬────────────────────────────────────────┐
  │ Variant         │ When to Use                            │
  ├─────────────────┼────────────────────────────────────────┤
  │ PLSRegression   │ Standard baseline, well-understood     │
  │ IKPLS           │ Large datasets, need speed             │
  │ SIMPLS          │ Alternative fast algorithm             │
  │ OPLS            │ Filter Y-orthogonal variation          │
  │ SparsePLS       │ Variable selection, interpretability   │
  │ IntervalPLS     │ Wavelength region selection            │
  │ RobustPLS       │ Data with outliers                     │
  │ KernelPLS       │ Nonlinear relationships                │
  │ PLSDA/OPLSDA    │ Classification tasks                   │
  └─────────────────┴────────────────────────────────────────┘

Common Parameters:
  n_components  - Number of latent variables
  backend       - 'numpy' or 'jax' (for GPU acceleration)
  alpha         - Sparsity parameter (SparsePLS)
  kernel        - Kernel type (KernelPLS): 'linear', 'rbf', 'poly'
  weighting     - Robust weighting: 'huber', 'tukey'

JAX Backend:
  If JAX is installed, many PLS variants support GPU acceleration:
    IKPLS(n_components=10, backend='jax')

Next: See 05_cross_validation/U01_cv_strategies.py - Cross-validation strategies
""")
