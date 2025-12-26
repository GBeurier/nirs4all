"""
Q41 Example - SHAP Explanations with sklearn Wrapper
=====================================================
Demonstrates SHAP feature importance analysis using NIRSPipeline
for sklearn/SHAP integration.

This example shows:
1. Training a pipeline and wrapping with NIRSPipeline
2. Computing SHAP values with different explainer types
3. Visualizing spectral feature importance
4. Aggregating SHAP values by wavelength regions
5. Comparing feature importance across models

Requirements:
    pip install shap matplotlib

Phase 5 Implementation - SHAP Integration Example
"""

# Standard library imports
import argparse
import warnings

# Third-party imports
import numpy as np

# Parse command-line arguments early to control plotting
parser = argparse.ArgumentParser(description='Q41 SHAP Analysis Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at end')
args = parser.parse_args()

# Configure matplotlib before importing pyplot
import matplotlib
if not args.plots and not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.sklearn import NIRSPipeline
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay
from nirs4all.data import DatasetConfigs

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def get_sample_data():
    """Load sample data for SHAP analysis."""
    dataset = DatasetConfigs("sample_data/regression")
    for config, name in dataset.configs:
        ds = dataset.get_dataset(config, name)
        X = ds.x({})
        y = ds.y
        break
    return X, y


def example_1_basic_shap():
    """Example 1: Basic SHAP with PLS regression.

    Shows fundamental SHAP workflow with NIRSPipeline wrapper.
    """
    print("\n" + "="*60)
    print("Example 1: Basic SHAP with PLS Regression")
    print("="*60)

    # Check for SHAP
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return None

    # Train a simple pipeline
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="SHAP_PLS_Example",
        verbose=1,
        plots_visible=False
    )

    print(f"\nTraining complete - RMSE: {result.best_rmse:.4f}")

    # Wrap for sklearn/SHAP compatibility
    pipe = NIRSPipeline.from_result(result)
    print(f"Created NIRSPipeline wrapper")

    # Get data for SHAP analysis
    X, y = get_sample_data()

    # Use a subset for efficiency
    n_background = 50
    n_explain = 10

    background = X[:n_background]
    X_explain = X[n_background:n_background + n_explain]

    print(f"\nComputing SHAP values...")
    print(f"  Background samples: {n_background}")
    print(f"  Samples to explain: {n_explain}")

    # Create SHAP explainer
    # Use sampling for efficiency with high-dimensional data
    explainer = shap.KernelExplainer(
        pipe.predict,
        shap.kmeans(background, 10)  # Cluster background for speed
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    print(f"\nSHAP values shape: {shap_values.shape}")
    print(f"Mean |SHAP| (first 10 features): {np.mean(np.abs(shap_values), axis=0)[:10]}")

    # Find most important wavelengths
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    print(f"\nTop 10 most important feature indices: {top_indices}")

    return shap_values, pipe


def example_2_shap_visualization():
    """Example 2: SHAP visualizations for spectral data.

    Creates spectral importance plots and summary visualizations.
    """
    print("\n" + "="*60)
    print("Example 2: SHAP Visualizations for Spectral Data")
    print("="*60)

    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping visualization example.")
        return None

    # Train model
    pipeline = [
        MinMaxScaler(),
        SavitzkyGolay(window_length=11, polyorder=2),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=8)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="SHAP_Viz_Example",
        verbose=1,
        plots_visible=False
    )

    pipe = NIRSPipeline.from_result(result)
    X, y = get_sample_data()

    # Compute SHAP values
    n_bg, n_exp = 40, 20
    background = X[:n_bg]
    X_explain = X[n_bg:n_bg + n_exp]

    print("\nComputing SHAP values for visualization...")
    explainer = shap.KernelExplainer(pipe.predict, shap.kmeans(background, 10))
    shap_values = explainer.shap_values(X_explain, nsamples=100)

    # Create wavelength labels (simulated for sample data)
    n_features = X.shape[1]
    wavelengths = np.linspace(1100, 2500, n_features)  # NIR range

    # -------------------------------------------------------------------------
    # Plot 1: Spectral importance (mean |SHAP| vs wavelength)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    ax1.fill_between(wavelengths, mean_abs_shap, alpha=0.5)
    ax1.plot(wavelengths, mean_abs_shap, linewidth=1.5)
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax1.set_title('Spectral Feature Importance (SHAP)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Mark top regions
    top_5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    for idx in top_5_idx:
        ax1.axvline(wavelengths[idx], color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("shap_spectral_importance.png", dpi=150)
    print("\nSaved: shap_spectral_importance.png")

    if args.plots:
        plt.show()

    # -------------------------------------------------------------------------
    # Plot 2: SHAP waterfall for single sample
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Get SHAP for first sample
    sample_shap = shap_values[0]
    sample_x = X_explain[0]

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:15]

    # Create bar plot
    colors = ['red' if s < 0 else 'blue' for s in sample_shap[sorted_idx]]
    positions = range(len(sorted_idx))

    ax2.barh(positions, sample_shap[sorted_idx], color=colors, alpha=0.7)
    ax2.set_yticks(positions)
    ax2.set_yticklabels([f'λ {wavelengths[i]:.0f} nm' for i in sorted_idx])
    ax2.set_xlabel('SHAP value', fontsize=12)
    ax2.set_title('Feature Contributions (Single Sample)', fontsize=14)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig("shap_waterfall_sample.png", dpi=150)
    print("Saved: shap_waterfall_sample.png")

    if args.plots:
        plt.show()

    # -------------------------------------------------------------------------
    # Plot 3: SHAP summary with wavelength bins
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    # Bin SHAP values into wavelength regions (50 nm bins)
    bin_size = 50  # nm
    bin_edges = np.arange(1100, 2550, bin_size)
    n_bins = len(bin_edges) - 1

    binned_shap = np.zeros((len(X_explain), n_bins))
    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (wavelengths >= low) & (wavelengths < high)
        if np.any(mask):
            binned_shap[:, i] = np.mean(shap_values[:, mask], axis=1)

    # Plot binned importance
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_binned = np.mean(np.abs(binned_shap), axis=0)

    ax3.bar(bin_centers, mean_binned, width=bin_size*0.8, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Wavelength region (nm)', fontsize=12)
    ax3.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax3.set_title(f'Binned Spectral Importance ({bin_size} nm bins)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("shap_binned_importance.png", dpi=150)
    print("Saved: shap_binned_importance.png")

    if args.plots:
        plt.show()

    plt.close('all')

    return shap_values


def example_3_model_comparison():
    """Example 3: Compare feature importance across models.

    Compares SHAP-based importance between PLS and RandomForest.
    """
    print("\n" + "="*60)
    print("Example 3: Compare SHAP Importance Across Models")
    print("="*60)

    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping comparison example.")
        return None

    X, y = get_sample_data()
    wavelengths = np.linspace(1100, 2500, X.shape[1])

    # Split data
    n_bg = 40
    n_exp = 15
    background = X[:n_bg]
    X_explain = X[n_bg:n_bg + n_exp]

    results = {}

    # Model 1: PLS
    print("\nTraining PLS model...")
    pls_pipeline = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ]

    pls_result = nirs4all.run(
        pipeline=pls_pipeline,
        dataset="sample_data/regression",
        name="SHAP_PLS",
        verbose=0,
        plots_visible=False
    )
    pls_pipe = NIRSPipeline.from_result(pls_result)

    print("Computing PLS SHAP values...")
    pls_explainer = shap.KernelExplainer(pls_pipe.predict, shap.kmeans(background, 10))
    pls_shap = pls_explainer.shap_values(X_explain, nsamples=100)
    results['PLS'] = np.mean(np.abs(pls_shap), axis=0)

    # Model 2: Random Forest
    print("\nTraining Random Forest model...")
    rf_pipeline = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)}
    ]

    rf_result = nirs4all.run(
        pipeline=rf_pipeline,
        dataset="sample_data/regression",
        name="SHAP_RF",
        verbose=0,
        plots_visible=False
    )
    rf_pipe = NIRSPipeline.from_result(rf_result)

    print("Computing RF SHAP values...")
    rf_explainer = shap.KernelExplainer(rf_pipe.predict, shap.kmeans(background, 10))
    rf_shap = rf_explainer.shap_values(X_explain, nsamples=100)
    results['RF'] = np.mean(np.abs(rf_shap), axis=0)

    # -------------------------------------------------------------------------
    # Compare models
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, (name, importance) in zip(axes, results.items()):
        ax.fill_between(wavelengths, importance, alpha=0.5)
        ax.plot(wavelengths, importance, linewidth=1.5)
        ax.set_ylabel('Mean |SHAP|', fontsize=11)
        ax.set_title(f'{name} Feature Importance', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Mark top 3 peaks
        top_3 = np.argsort(importance)[::-1][:3]
        for idx in top_3:
            ax.axvline(wavelengths[idx], color='red', linestyle='--', alpha=0.4)

    axes[1].set_xlabel('Wavelength (nm)', fontsize=12)

    plt.suptitle('Model Comparison: SHAP Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_model_comparison.png", dpi=150)
    print("\nSaved: shap_model_comparison.png")

    if args.plots:
        plt.show()

    # Correlation between model importances
    from scipy.stats import spearmanr
    corr, pval = spearmanr(results['PLS'], results['RF'])
    print(f"\nSpearman correlation between PLS and RF importance: {corr:.3f} (p={pval:.3e})")

    plt.close('all')

    return results


def example_4_nirs4all_explain():
    """Example 4: Using nirs4all.explain() directly.

    Shows the integrated explain() function instead of manual SHAP.
    """
    print("\n" + "="*60)
    print("Example 4: Using nirs4all.explain() (Built-in SHAP)")
    print("="*60)

    # Train model
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="Explain_Example",
        verbose=1,
        plots_visible=False
    )

    print(f"\nTraining complete - RMSE: {result.best_rmse:.4f}")

    # Use built-in explain
    print("\nUsing nirs4all.explain()...")
    explain_result = nirs4all.explain(
        model=result.best,
        data="sample_data/regression",
        n_samples=50,
        verbose=1
    )

    # Access results through ExplainResult
    print(f"\nExplainResult summary:")
    print(f"  SHAP values shape: {explain_result.shape}")
    print(f"  Number of features: {len(explain_result.feature_names) if explain_result.feature_names else explain_result.shape[-1]}")

    # Get feature importance
    importance = explain_result.get_feature_importance(top_n=10)
    print(f"\nTop 10 features by importance:")
    for i, (feature, value) in enumerate(importance.items(), 1):
        print(f"  {i}. Feature {feature}: {value:.4f}")

    # Top features list
    print(f"\nTop features (by name): {explain_result.top_features[:10]}")

    return explain_result


def main():
    """Run all SHAP examples."""
    print("\n" + "#"*60)
    print("# Q41: SHAP Explanations with NIRSPipeline")
    print("#"*60)

    # Check for SHAP
    try:
        import shap
        print(f"\nSHAP version: {shap.__version__}")
    except ImportError:
        print("\n⚠️  SHAP not installed!")
        print("Install with: pip install shap")
        print("Some examples will be skipped.")

    # Run examples
    example_1_basic_shap()
    example_2_shap_visualization()
    example_3_model_comparison()
    example_4_nirs4all_explain()

    print("\n" + "#"*60)
    print("# All SHAP Examples Complete!")
    print("#"*60)

    print("\nGenerated plots:")
    print("  - shap_spectral_importance.png")
    print("  - shap_waterfall_sample.png")
    print("  - shap_binned_importance.png")
    print("  - shap_model_comparison.png")

    print("\nKey takeaways:")
    print("  1. NIRSPipeline wraps trained models for SHAP compatibility")
    print("  2. Use shap.KernelExplainer for any model type")
    print("  3. Access underlying model with pipe.model_ for specialized explainers")
    print("  4. Use nirs4all.explain() for integrated SHAP workflow")
    print("  5. Bin wavelengths for clearer spectral importance visualization")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
