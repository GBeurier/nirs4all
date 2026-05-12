"""
Synthetic NIRS Spectra Generation Example
==========================================

This example demonstrates how to generate realistic synthetic NIRS spectra
for training autoencoders, testing preprocessing algorithms, and other ML applications.

Key features demonstrated:
1. Basic spectrum generation with predefined components
2. Custom component library creation
3. Batch/session effects for domain adaptation
4. Visualization of generated data
5. Integration with nirs4all pipelines

Usage:
    python S1_synthetic_generation.py --plots    # Show all plots
    python S1_synthetic_generation.py --save     # Save plots to files
    python S1_synthetic_generation.py --pipeline # Run a simple pipeline on synthetic data
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Synthetic NIRS Generation Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--save', action='store_true', help='Save plots to files')
parser.add_argument('--pipeline', action='store_true', help='Run a pipeline on synthetic data')
args = parser.parse_args()

# ============================================================================
# Import synthetic generation module
# ============================================================================
from synthetic import (
    PREDEFINED_COMPONENTS,
    ComponentLibrary,
    NIRBand,
    SyntheticNIRSGenerator,
    SyntheticRealComparator,
    SyntheticSpectraVisualizer,
    compare_with_real_data,
    plot_synthetic_spectra,
)

print("=" * 70)
print("SYNTHETIC NIRS SPECTRA GENERATION")
print("=" * 70)

# ============================================================================
# Example 1: Basic Generation with Predefined Components
# ============================================================================
print("\n📊 Example 1: Basic Generation with Predefined Components")
print("-" * 50)

# Create generator with default settings (realistic complexity)
generator = SyntheticNIRSGenerator(
    wavelength_start=1000,
    wavelength_end=2500,
    wavelength_step=2,
    complexity="realistic",
    random_state=42
)

print(f"  Wavelength range: {generator.wavelength_start}-{generator.wavelength_end} nm")
print(f"  Number of wavelengths: {generator.n_wavelengths}")
print(f"  Number of components: {generator.library.n_components}")
print(f"  Component names: {generator.library.component_names}")

# Generate spectra
X, Y, E = generator.generate(n_samples=1000, concentration_method="dirichlet")

print(f"\n  Generated spectra shape: {X.shape}")
print(f"  Concentrations shape: {Y.shape}")
print(f"  Component spectra shape: {E.shape}")

# Quick statistics
print(f"\n  Absorbance range: [{X.min():.3f}, {X.max():.3f}]")
print(f"  Mean absorbance: {X.mean():.3f} ± {X.std():.3f}")

# ============================================================================
# Example 2: Different Complexity Levels
# ============================================================================
print("\n📊 Example 2: Comparing Complexity Levels")
print("-" * 50)

complexity_levels = ["simple", "realistic", "complex"]
comparison_data = {}

for complexity in complexity_levels:
    gen = SyntheticNIRSGenerator(complexity=complexity, random_state=42)
    X_comp, Y_comp, _ = gen.generate(n_samples=500)
    comparison_data[complexity] = X_comp

    # Compute noise estimate
    noise_std = np.diff(X_comp, axis=1).std() / np.sqrt(2)
    print(f"  {complexity.capitalize():10s}: Noise σ ≈ {noise_std:.4f}, "
          f"Range: [{X_comp.min():.3f}, {X_comp.max():.3f}]")

# ============================================================================
# Example 3: Custom Component Library
# ============================================================================
print("\n📊 Example 3: Custom Component Library")
print("-" * 50)

# Create a custom library with specific components
custom_library = ComponentLibrary(random_state=123)

# Add some predefined components
custom_library._components["water"] = PREDEFINED_COMPONENTS["water"]
custom_library._components["protein"] = PREDEFINED_COMPONENTS["protein"]

# Add a custom random component
custom_library.add_random_component(
    name="synthetic_compound",
    n_bands=4,
    zones=[(1100, 1300), (1500, 1700), (2000, 2200)]
)

print(f"  Custom library components: {custom_library.component_names}")

# Generate with custom library
custom_generator = SyntheticNIRSGenerator(
    component_library=custom_library,
    complexity="realistic",
    random_state=42
)

X_custom, Y_custom, E_custom = custom_generator.generate(n_samples=500)
print(f"  Generated {X_custom.shape[0]} samples with {Y_custom.shape[1]} components")

# ============================================================================
# Example 4: Batch Effects for Domain Adaptation
# ============================================================================
print("\n📊 Example 4: Batch Effects (Multi-Session Simulation)")
print("-" * 50)

# Generate spectra with batch effects
X_batch, Y_batch, E_batch, metadata = generator.generate(
    n_samples=600,
    include_batch_effects=True,
    n_batches=3,
    return_metadata=True
)

batch_ids = metadata["batch_ids"]
unique_batches = np.unique(batch_ids)
print(f"  Number of batches: {len(unique_batches)}")
for batch_id in unique_batches:
    n_in_batch = np.sum(batch_ids == batch_id)
    print(f"    Batch {batch_id}: {n_in_batch} samples")

# ============================================================================
# Example 5: Different Concentration Methods
# ============================================================================
print("\n📊 Example 5: Concentration Generation Methods")
print("-" * 50)

concentration_methods = ["dirichlet", "uniform", "lognormal", "correlated"]

for method in concentration_methods:
    X_m, Y_m, _ = generator.generate(n_samples=500, concentration_method=method)

    # Compute concentration statistics
    conc_mean = Y_m.mean()
    conc_std = Y_m.std()
    conc_corr = np.corrcoef(Y_m.T)
    off_diag = conc_corr[np.triu_indices(Y_m.shape[1], k=1)]
    avg_corr = np.abs(off_diag).mean()

    print(f"  {method.capitalize():12s}: Mean={conc_mean:.3f}, Std={conc_std:.3f}, "
          f"Avg |corr|={avg_corr:.3f}")

# ============================================================================
# Example 6: Comparison with Real Data
# ============================================================================
print("\n📊 Example 6: Comparison with Real Dataset")
print("-" * 50)

# Try to load sample data. It may be in different formats depending on the dataset.
sample_data_dir = Path(__file__).parent / "sample_data" / "regression"
real_data_loaded = False

# Try different loading strategies
if sample_data_dir.exists():
    import pandas as pd

    # First, try Xcal.csv.gz with semicolon-separated values in first column
    xcal_file = sample_data_dir / "Xcal.csv.gz"
    if xcal_file.exists() and not real_data_loaded:
        try:
            # Load and parse semicolon-separated data
            raw = pd.read_csv(xcal_file, compression='gzip', header=None)
            if raw.shape[1] == 1:
                # Split semicolon-separated values
                data_rows = [row.iloc[0].split(';') for _, row in raw.iterrows()]
                real_spectra = np.array([[float(v) for v in row] for row in data_rows], dtype=np.float64)
                n_wl = real_spectra.shape[1]
                # Generate approximate wavelength range (typical NIR)
                real_wavelengths = np.linspace(1000, 2500, n_wl)
                real_data_loaded = True
                print(f"  Loading real data from: {xcal_file.name}")
                print(f"    Real data: {real_spectra.shape[0]} samples, {real_spectra.shape[1]} wavelengths")
        except Exception as e:
            print(f"    Warning: Could not load {xcal_file.name}: {e}")

if real_data_loaded:
    # Create comparator and add datasets
    comparator = SyntheticRealComparator()
    comparator.add_real_dataset(real_spectra, wavelengths=real_wavelengths, name="real_calibration")
    comparator.add_synthetic_dataset(X, wavelengths=generator.wavelengths, name="synthetic")

    # Get comparison results
    comparison = comparator.compute_comparison()

    # Print summary
    print("\n  Dataset Properties:")
    syn = comparison.get("synthetic", {}).get("synthetic", {})
    real = comparison.get("real", {}).get("real_calibration", {})

    if syn and real:
        print("    Property          | Synthetic  | Real")
        print("    " + "-" * 40)
        print(f"    Global mean       | {syn.get('global_mean', 0):.4f}    | {real.get('global_mean', 0):.4f}")
        print(f"    Mean slope        | {syn.get('mean_slope', 0):+.4f}   | {real.get('mean_slope', 0):+.4f}")
        print(f"    Noise level       | {syn.get('noise_estimate', 0):.5f}  | {real.get('noise_estimate', 0):.5f}")
        print(f"    SNR               | {syn.get('snr_estimate', 0):.1f}       | {real.get('snr_estimate', 0):.1f}")
        print(f"    PCA 95%%           | {syn.get('pca_n_components_95', 0)} comps    | {real.get('pca_n_components_95', 0)} comps")

    # Print similarity scores
    comp_key = "real_calibration_vs_synthetic"
    comp_data = comparison.get("comparison", {}).get(comp_key, {})
    if comp_data:
        score = comp_data.get("similarity_score", 0)
        print(f"\n  Similarity Score: {score:.1f}/100")

    # Get recommendations
    recs = comparator.get_tuning_recommendations()
    if recs:
        print("\n  Tuning Recommendations:")
        for _param, rec_list in list(recs.items())[:3]:
            for rec in rec_list[:1]:
                print(f"    • {rec}")
else:
    # Use synthetic comparison (compare different complexity levels)
    print("  No real data found or loading failed. Comparing complexity levels instead...")

    # Compare simple vs realistic vs complex
    comparator = SyntheticRealComparator()
    comparator.add_synthetic_dataset(comparison_data["simple"], wavelengths=generator.wavelengths, name="simple")
    comparator.add_real_dataset(comparison_data["realistic"], wavelengths=generator.wavelengths, name="realistic")

    comparison = comparator.compute_comparison()
    comp_data = comparison.get("comparison", {}).get("realistic_vs_simple", {})
    score = comp_data.get("similarity_score", 0)
    print(f"\n  Simple vs Realistic similarity: {score:.1f}/100")

    comparator2 = SyntheticRealComparator()
    comparator2.add_synthetic_dataset(comparison_data["simple"], wavelengths=generator.wavelengths, name="simple")
    comparator2.add_real_dataset(comparison_data["complex"], wavelengths=generator.wavelengths, name="complex")

    comparison2 = comparator2.compute_comparison()
    comp_data2 = comparison2.get("comparison", {}).get("complex_vs_simple", {})
    score2 = comp_data2.get("similarity_score", 0)
    print(f"  Simple vs Complex similarity: {score2:.1f}/100")

print("\n  ✓ Comparison complete!")

# ============================================================================
# Visualization
# ============================================================================
if args.plots or args.save:
    print("\n🎨 Generating Visualizations...")
    print("-" * 50)

    # Create visualizer
    viz = SyntheticSpectraVisualizer(
        spectra=X,
        concentrations=Y,
        wavelengths=generator.wavelengths,
        component_names=generator.library.component_names,
        component_spectra=E,
        metadata={}
    )

    # Figure 1: Spectra Overview
    fig1 = viz.plot_spectra_overview(n_display=100, component_idx=0)
    fig1.suptitle("Generated NIRS Spectra - Colored by Water Content", fontsize=14, y=1.02)

    # Figure 2: Spectral Envelope
    fig2 = viz.plot_spectral_envelope()

    # Figure 3: 3D View
    fig3 = viz.plot_spectra_3d(n_display=100, color_by_component=1)

    # Figure 4: Component Library
    fig4 = viz.plot_component_library(stacked=True)

    # Figure 5: Concentration Distributions
    fig5 = viz.plot_concentration_distributions()

    # Figure 6: Concentration Correlations
    fig6 = viz.plot_concentration_correlations()

    # Figure 7: Noise Analysis
    fig7 = viz.plot_noise_analysis()

    # Figure 8: PCA Analysis
    fig8 = viz.plot_pca_analysis()

    # Figure 9: Complexity Comparison
    fig9, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (complexity, X_comp) in zip(axes, comparison_data.items(), strict=False):
        # Plot mean ± std envelope
        mean_spec = X_comp.mean(axis=0)
        std_spec = X_comp.std(axis=0)
        wl = generator.wavelengths

        ax.fill_between(wl, mean_spec - std_spec, mean_spec + std_spec, alpha=0.3, color='blue')
        ax.plot(wl, mean_spec, 'b-', linewidth=2)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"{complexity.capitalize()} Complexity")
        ax.grid(True, alpha=0.3)

    fig9.suptitle("Comparison of Complexity Levels", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Figure 10: Batch Effects (if available)
    viz_batch = SyntheticSpectraVisualizer(
        spectra=X_batch,
        concentrations=Y_batch,
        wavelengths=generator.wavelengths,
        component_names=generator.library.component_names,
        component_spectra=E_batch,
        metadata=metadata
    )
    fig10 = viz_batch.plot_batch_effects()

    # Save figures if requested
    if args.save:
        output_dir = Path(__file__).parent / "charts"
        output_dir.mkdir(exist_ok=True)

        figures = [
            ("spectra_overview", fig1),
            ("spectral_envelope", fig2),
            ("spectra_3d", fig3),
            ("component_library", fig4),
            ("concentration_distributions", fig5),
            ("concentration_correlations", fig6),
            ("noise_analysis", fig7),
            ("pca_analysis", fig8),
            ("complexity_comparison", fig9),
            ("batch_effects", fig10),
        ]

        for name, fig in figures:
            if fig is not None:
                filepath = output_dir / f"synthetic_{name}.png"
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filepath}")

    if args.plots:
        plt.show()

# ============================================================================
# Example 6: Integration with nirs4all Pipeline
# ============================================================================
if args.pipeline:
    print("\n📊 Example 7: Pipeline Integration")
    print("-" * 50)

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import MinMaxScaler

    from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
    from nirs4all.pipeline import PipelineConfigs, PipelineRunner

    # Create a SpectroDataset from synthetic data
    print("  Creating SpectroDataset...")
    dataset = generator.create_dataset(
        n_train=800,
        n_test=200,
        target_component="protein"  # Predict protein content
    )

    print(f"  Dataset created: {dataset.name}")
    print(f"    Train samples: {len(dataset.y({'partition': 'train'}))}")
    print(f"    Test samples: {len(dataset.y({'partition': 'test'}))}")

    # Build a simple pipeline
    pipeline = [
        MinMaxScaler(),
        {"_or_": [StandardNormalVariate, SavitzkyGolay]},
        PLSRegression(n_components=10),
    ]

    # Run pipeline
    print("  Running pipeline...")
    pipeline_config = PipelineConfigs(pipeline, "synthetic_test")
    runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset)

    # Display results
    print("\n  Top 3 models by RMSE:")
    top_models = predictions.top(3, "rmse")
    for idx, pred in enumerate(top_models):
        metrics = pred.get("metrics", {})
        test_metrics = metrics.get("test", {})
        test_rmse = test_metrics.get("rmse", "N/A")
        test_r2 = test_metrics.get("r2", "N/A")
        if isinstance(test_rmse, (int, float)) and isinstance(test_r2, (int, float)):
            print(f"    {idx+1}. RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
        else:
            print(f"    {idx+1}. RMSE={test_rmse}, R²={test_r2}")

    print("\n  ✓ Pipeline completed successfully on synthetic data!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The synthetic NIRS generator provides:

1. REALISTIC SPECTRAL PHYSICS
   - Beer-Lambert law with Voigt peak profiles
   - Predefined components based on NIR band assignments
   - Baseline, scatter, global slope, and instrumental effects

2. CONFIGURABLE COMPLEXITY
   - Simple: Low noise, minimal artifacts (testing)
   - Realistic: Typical lab conditions (training)
   - Complex: High noise, artifacts, strong batch effects (robustness)

3. BATCH EFFECTS
   - Simulates multi-session/multi-instrument variation
   - Useful for domain adaptation research

4. COMPARISON WITH REAL DATA
   - Statistical property comparison (slope, noise, PCA structure)
   - Tuning recommendations to match real spectra characteristics

5. VISUALIZATION
   - Comprehensive plots for quality assessment
   - PCA analysis, noise characterization, etc.

6. PIPELINE INTEGRATION
   - Direct SpectroDataset creation
   - Ready for use with nirs4all pipelines

Files created:
   examples/synthetic/
   ├── __init__.py       # Module exports
   ├── generator.py      # SyntheticNIRSGenerator class
   ├── visualizer.py     # Visualization tools
   └── comparator.py     # Real data comparison tools
""")
print("=" * 70)

if not args.plots and not args.save and not args.pipeline:
    print("\nTip: Run with --plots to see visualizations, --save to save them,")
    print("     or --pipeline to test with a nirs4all pipeline.")
