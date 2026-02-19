"""
D03 - PCA Geometry: Preprocessing Quality Analysis
===================================================

PreprocPCAEvaluator analyzes how preprocessing affects the
geometric structure of spectral data using PCA.

This tutorial covers:

* PCA geometry preservation metrics
* PreprocPCAEvaluator usage
* Comparing preprocessing methods
* Visualization of PCA spread

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics
- Understanding of PCA

Next Steps
----------
See 05_advanced_features/D01_metadata_branching for metadata-based partitioning.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import Detrend, FirstDerivative, SavitzkyGolay, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D03 PCA Geometry Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D03 - PCA Geometry: Preprocessing Quality Analysis")
print("=" * 60)

print("""
Good preprocessing should:
  1. Remove noise/artifacts (improve SNR)
  2. Preserve chemical information (geometry)
  3. Reduce unwanted variance (scatter, baseline)

PreprocPCAEvaluator measures these effects using PCA:
  - Variance explained by top components
  - Spread in PCA space (sample separation)
  - Eigenvalue spectrum analysis
""")

# =============================================================================
# Section 1: PreprocPCAEvaluator Basics
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: PreprocPCAEvaluator Basics")
print("-" * 60)

print("""
PreprocPCAEvaluator compares preprocessing effects:

    from nirs4all.operators.analysis import PreprocPCAEvaluator

    evaluator = PreprocPCAEvaluator(
        preprocessings=[SNV(), MSC(), FirstDerivative()],
        n_components=10
    )

    results = evaluator.fit_evaluate(X)
""")

try:
    from nirs4all.operators.analysis import PreprocPCAEvaluator

    evaluator = PreprocPCAEvaluator(
        n_components=10
    )
    print("PreprocPCAEvaluator created")

except ImportError:
    print("PreprocPCAEvaluator not available in this installation")
    print("Demonstrating concepts manually...")

# =============================================================================
# Section 2: Manual PCA Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Manual PCA Analysis")
print("-" * 60)

# Generate sample data
np.random.seed(42)
n_samples = 100
n_features = 200

# Simulated spectral data with noise
X = np.random.randn(n_samples, n_features)
# Add some structure
X[:50] += 2  # Group 1
X[50:] -= 1  # Group 2

print(f"Sample data: {X.shape}")

# Compare preprocessing effects on PCA
preprocessings = {
    'raw': None,
    'snv': SNV(),
    'msc': MSC(),
    'detrend': Detrend(),
    '1st_deriv': FirstDerivative(),
    'savgol': SavitzkyGolay(window_length=11, polyorder=2),
}

pca_results = {}
for name, preproc in preprocessings.items():
    # Apply preprocessing
    X_pp = X.copy() if preproc is None else preproc.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_pp)

    # Compute metrics
    pca_results[name] = {
        'var_explained': pca.explained_variance_ratio_[:3].sum(),
        'spread_pc1': np.std(X_pca[:, 0]),
        'spread_pc2': np.std(X_pca[:, 1]),
        'X_pca': X_pca,
    }

print("\n📊 PCA Metrics (first 3 components):")
print("-" * 50)
print(f"{'Preprocessing':<12} {'Var Explained':<15} {'PC1 Spread':<12} {'PC2 Spread':<12}")
print("-" * 50)
for name, metrics in pca_results.items():
    print(f"{name:<12} {metrics['var_explained']:<15.3f} {metrics['spread_pc1']:<12.3f} {metrics['spread_pc2']:<12.3f}")

# =============================================================================
# Section 3: Variance Explained Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Variance Explained Analysis")
print("-" * 60)

print("""
Variance explained indicates information concentration:
  - High variance in few PCs → structured data
  - Spread across many PCs → noisy/complex data

Good preprocessing often INCREASES variance in top PCs
by removing noise that spreads variance.
""")

# Find best preprocessing by variance concentration
best_var = max(pca_results.items(), key=lambda x: x[1]['var_explained'])
print(f"\n🏆 Highest variance concentration: {best_var[0]}")
print(f"   Top 3 PCs explain: {best_var[1]['var_explained']*100:.1f}% of variance")

# =============================================================================
# Section 4: PCA Spread Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: PCA Spread Analysis")
print("-" * 60)

print("""
PCA spread measures sample separation in reduced space:
  - Larger spread → better sample differentiation
  - Useful for classification/clustering tasks

Scatter corrections (SNV, MSC) often REDUCE spread
by normalizing samples, which can be good or bad.
""")

# Compare spreads
print("\n📊 PC1 vs PC2 Spread Comparison:")
for name, metrics in sorted(pca_results.items(), key=lambda x: -x[1]['spread_pc1']):
    print(f"  {name:<12}: PC1={metrics['spread_pc1']:.2f}, PC2={metrics['spread_pc2']:.2f}")

# =============================================================================
# Section 5: Visualization of PCA Effects
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Visualization of PCA Effects")
print("-" * 60)

if args.plots or args.show:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    colors = ['blue'] * 50 + ['red'] * 50  # Group colors

    for idx, (name, metrics) in enumerate(pca_results.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        X_pca = metrics['X_pca']
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=30)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f"{name}\nVar: {metrics['var_explained']*100:.1f}%")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_geometry.png', dpi=100)
    print("Saved: pca_geometry.png")

# =============================================================================
# Section 6: Eigenvalue Spectrum
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Eigenvalue Spectrum")
print("-" * 60)

print("""
Eigenvalue spectrum reveals data complexity:
  - Sharp decay → low intrinsic dimensionality
  - Slow decay → high complexity or noise

Compare preprocessing effects on spectrum.
""")

# Compute full eigenvalue spectra
eigenvalue_spectra = {}
for name, preproc in preprocessings.items():
    X_pp = X.copy() if preproc is None else preproc.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_pp)
    eigenvalue_spectra[name] = pca_full.explained_variance_ratio_[:20]

print("\nTop 5 eigenvalues per preprocessing:")
print("-" * 60)
for name, spectrum in eigenvalue_spectra.items():
    top5 = ', '.join([f"{v:.3f}" for v in spectrum[:5]])
    print(f"  {name:<12}: {top5}")

# =============================================================================
# Section 7: Preprocessing Recommendations
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Preprocessing Recommendations")
print("-" * 60)

print("""
📋 PCA-Based Preprocessing Selection:

┌─────────────────────┬─────────────────────────────────────┐
│ Observation         │ Recommended Action                  │
├─────────────────────┼─────────────────────────────────────┤
│ Low variance in     │ Use derivatives to enhance          │
│ top PCs             │ spectral features                   │
├─────────────────────┼─────────────────────────────────────┤
│ Poor group          │ Try scatter corrections             │
│ separation          │ (SNV, MSC) to reduce noise          │
├─────────────────────┼─────────────────────────────────────┤
│ High PC1 variance,  │ Baseline drift issue                │
│ PC1 = offset        │ Use Detrend or derivatives          │
├─────────────────────┼─────────────────────────────────────┤
│ Slow eigenvalue     │ Data is high-dimensional            │
│ decay               │ Consider feature selection          │
└─────────────────────┴─────────────────────────────────────┘
""")

# =============================================================================
# Section 8: Integration with Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Integration with Pipeline")
print("-" * 60)

print("""
Use PCA analysis to select preprocessing:

    # Evaluate options
    evaluator = PreprocPCAEvaluator(n_components=10)
    scores = evaluator.compare([SNV(), MSC(), Detrend()], X)

    # Use best in pipeline
    best_preproc = scores.get_best()

    pipeline = [
        best_preproc,
        PLSRegression(n_components=10)
    ]
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

# Find overall best
best_overall = max(pca_results.items(),
                   key=lambda x: x[1]['var_explained'] + x[1]['spread_pc1']/10)

print(f"""
What we learned:
1. PCA reveals preprocessing effects on data structure
2. Variance explained shows information concentration
3. PCA spread indicates sample separation
4. Eigenvalue spectrum reveals complexity

Key metrics:
- Variance in top PCs: higher = more concentrated info
- PC spread: higher = better sample differentiation
- Eigenvalue decay: faster = lower dimensionality

Best preprocessing by combined metrics: {best_overall[0]}

Usage:
    evaluator = PreprocPCAEvaluator(n_components=10)
    results = evaluator.fit_evaluate(X)

Next: 05_advanced_features/D01_metadata_branching.py - Metadata-based partitioning
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
