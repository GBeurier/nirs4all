"""
Q27 Example - Transfer Preprocessing Analysis for Machine/Dataset Transfer
=========================================================================
Demonstrates the use of TransferPreprocessingSelector to find optimal
preprocessing for transfer learning scenarios in NIRS analysis.

This example covers:
1. Basic transfer analysis between two datasets (machine transfer)
2. Using different presets (fast, balanced, thorough, full)
3. Generator-based preprocessing specification
4. Applying transfer recommendations to pipelines
5. Visualization of transfer analysis results

Use Cases:
- Machine transfer: Training on Machine A, predicting on Machine B
- Temporal transfer: Training on Year 1, predicting on Year 2
- Cross-site calibration: Harmonizing data from different laboratories
- Train/test alignment: Ensuring test data lives in same feature space as training

Reference: bench/SPEC_TRANSFER_PREPROCESSING_SELECTION.md
"""

# Standard library imports
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.analysis import (
    TransferPreprocessingSelector,
    get_base_preprocessings,
    list_presets,
)
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Q27 Transfer Analysis Example")
parser.add_argument("--plots", action="store_true", help="Show plots interactively")
parser.add_argument("--show", action="store_true", help="Show all plots")
args = parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def generate_synthetic_nirs_data(
    n_samples: int = 100,
    n_features: int = 200,
    baseline_shift: float = 0.0,
    scatter_variation: float = 0.0,
    noise_level: float = 0.08,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic NIRS-like spectral data.

    This simulates realistic spectral data with optional baseline shifts and
    multiplicative scatter to mimic machine transfer scenarios.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of wavelength points.
        baseline_shift: Additive baseline shift (simulates instrument drift).
        scatter_variation: Multiplicative scatter variation (simulates light scatter).
        noise_level: Standard deviation of random noise.
        random_state: Random seed for reproducibility.

    Returns:
        X: Spectral data (n_samples, n_features).
        y: Target values (n_samples,).
    """
    rng = np.random.RandomState(random_state)

    # Create wavelength axis
    wavelengths = np.linspace(400, 2500, n_features)

    # Base spectral shape (sum of Gaussian absorption peaks)
    # Simulating typical NIRS peaks (water, organics, etc.)
    peaks = [
        (600, 50, 0.5),    # O-H second overtone
        (900, 80, 0.8),    # C-H third overtone
        (1200, 100, 0.6),  # C-H second overtone
        (1700, 120, 0.9),  # O-H first overtone
        (2100, 90, 0.7),   # C-H combination
    ]

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Target value (e.g., protein content, moisture)
        base_target = 15 + 25 * (i / n_samples)
        y[i] = base_target + rng.normal(0, 2)

        # Start with baseline
        spectrum = baseline_shift + rng.uniform(0.1, 0.3)

        # Add absorption peaks with sample-specific variations
        for center, width, height in peaks:
            # Peak height correlates with target
            peak_height = height * (0.8 + 0.4 * (y[i] / 40))
            peak_center = center + rng.uniform(-10, 10)
            gaussian_peak = peak_height * np.exp(
                -((wavelengths - peak_center) ** 2) / (2 * width ** 2)
            )
            spectrum = spectrum + gaussian_peak

        # Apply multiplicative scatter
        scatter = 1 + rng.uniform(-scatter_variation, scatter_variation)
        spectrum = spectrum * scatter

        # Add random noise
        spectrum = spectrum + rng.normal(0, noise_level, n_features)

        X[i] = spectrum

    return X, y


# =============================================================================
# Example 1: Basic Transfer Analysis with Fast Preset
# =============================================================================


def example_1_basic_transfer_analysis():
    """
    Demonstrate basic transfer analysis between two datasets.

    Scenario: You have training data from Machine A and need to apply
    the model to predict samples measured on Machine B. This example
    shows how to find the best preprocessing to align the datasets.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Transfer Analysis (Machine A → Machine B)")
    print("=" * 70)

    # Generate synthetic data simulating two different instruments
    print("\nGenerating synthetic data from two 'machines'...")

    # Machine A (source/training data) - reference instrument
    X_source, y_source = generate_synthetic_nirs_data(
        n_samples=150,
        n_features=200,
        baseline_shift=0.0,
        scatter_variation=0.05,
        random_state=42,
    )

    # Machine B (target/prediction data) - different instrument
    # Has baseline shift and more scatter variation (common in real transfers)
    X_target, y_target = generate_synthetic_nirs_data(
        n_samples=120,
        n_features=200,
        baseline_shift=0.4,  # Systematic baseline shift
        scatter_variation=0.12,  # More scatter variation
        random_state=43,
    )

    print(f"Source data (Machine A): {X_source.shape}")
    print(f"Target data (Machine B): {X_target.shape}")

    # Create transfer selector with fast preset (default)
    print("\nRunning transfer analysis with 'fast' preset...")
    selector = TransferPreprocessingSelector(preset="fast", verbose=1)

    # Analyze transfer
    results = selector.fit(X_source, X_target)

    # Show results
    print("\n" + "-" * 50)
    print(f"Best preprocessing: {results.best.name}")
    print(f"Transfer score: {results.best.transfer_score:.4f}")
    print(f"Distance reduction: {results.best.improvement_pct:.1f}%")
    print(f"Pipeline type: {results.best.pipeline_type}")

    # Get pipeline specification for use in nirs4all
    pipeline_spec = results.to_pipeline_spec()
    print(f"\nPipeline specification: {pipeline_spec}")

    return results


# =============================================================================
# Example 2: Balanced Analysis with Stacking
# =============================================================================


def example_2_balanced_with_stacking():
    """
    Demonstrate balanced preset which includes stacked preprocessing.

    The 'balanced' preset evaluates both single preprocessings and
    stacked combinations (e.g., SNV followed by first derivative).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Balanced Analysis with Stacking")
    print("=" * 70)

    # Generate data with more complex transfer scenario
    X_source, y_source = generate_synthetic_nirs_data(
        n_samples=150,
        n_features=200,
        baseline_shift=0.0,
        scatter_variation=0.05,
        random_state=42,
    )

    X_target, _ = generate_synthetic_nirs_data(
        n_samples=120,
        n_features=200,
        baseline_shift=0.3,
        scatter_variation=0.10,
        noise_level=0.12,
        random_state=44,
    )

    print("\nRunning transfer analysis with 'balanced' preset...")
    selector = TransferPreprocessingSelector(preset="balanced", verbose=1)
    results = selector.fit(X_source, X_target)

    # Show top 5 recommendations
    print("\nTop 5 recommendations:")
    print("-" * 60)
    for i, r in enumerate(results.top_k(5), 1):
        print(f"  {i}. {r.name}")
        print(f"     Type: {r.pipeline_type}")
        print(f"     Score: {r.transfer_score:.4f}")
        print(f"     Improvement: {r.improvement_pct:.1f}%")

    # Count by type
    singles = [r for r in results.ranking if r.pipeline_type == "single"]
    stacked = [r for r in results.ranking if r.pipeline_type == "stacked"]
    print(f"\nEvaluated: {len(singles)} single, {len(stacked)} stacked preprocessings")

    return results


# =============================================================================
# Example 3: Full Analysis with Supervised Validation
# =============================================================================


def example_3_full_with_validation():
    """
    Demonstrate full preset with supervised validation.

    The 'full' preset includes all stages:
    - Stage 1: Single preprocessing evaluation
    - Stage 2: Stacked combinations
    - Stage 3: Feature augmentation
    - Stage 4: Supervised validation (verifies signal preservation)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Full Analysis with Supervised Validation")
    print("=" * 70)

    # Generate data
    X_source, y_source = generate_synthetic_nirs_data(
        n_samples=150, n_features=200, random_state=42
    )
    X_target, y_target = generate_synthetic_nirs_data(
        n_samples=120,
        n_features=200,
        baseline_shift=0.35,
        scatter_variation=0.08,
        random_state=45,
    )

    print("\nRunning transfer analysis with 'full' preset...")
    print("(This includes supervised validation with proxy models)")

    selector = TransferPreprocessingSelector(preset="full", verbose=1)
    results = selector.fit(X_source, X_target, y_source=y_source)

    # Show results with signal scores
    print("\nTop 5 results with signal preservation scores:")
    print("-" * 70)
    for i, r in enumerate(results.top_k(5), 1):
        signal = f"{r.signal_score:.3f}" if r.signal_score is not None else "N/A"
        print(f"  {i}. {r.name}")
        print(f"     Transfer Score: {r.transfer_score:.4f}")
        print(f"     Signal Score: {signal}")
        print(f"     Type: {r.pipeline_type}")

    return results


# =============================================================================
# Example 4: Using Generator-Based Preprocessing Specification
# =============================================================================


def example_4_generator_specification():
    """
    Demonstrate generator-based preprocessing specification.

    The generator DSL allows flexible, constraint-based specification
    of which preprocessings to evaluate.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Generator-Based Preprocessing Specification")
    print("=" * 70)

    # Generate data
    X_source, _ = generate_synthetic_nirs_data(n_samples=100, n_features=150, random_state=42)
    X_target, _ = generate_synthetic_nirs_data(
        n_samples=80, n_features=150, baseline_shift=0.3, random_state=46
    )

    # Example 4a: Simple _or_ specification
    print("\n4a. Simple specification (3 specific preprocessings):")
    selector_a = TransferPreprocessingSelector(
        preset="fast",
        preprocessing_spec={"_or_": ["snv", "msc", "d1"]},
        verbose=0,
    )
    results_a = selector_a.fit(X_source, X_target)
    print(f"   Evaluated: {len(results_a.ranking)} preprocessings")
    print(f"   Best: {results_a.best.name} (score: {results_a.best.transfer_score:.4f})")

    # Example 4b: Arrange specification (generates stacked combinations)
    print("\n4b. Arrange specification (stacked combinations):")
    selector_b = TransferPreprocessingSelector(
        preset="fast",
        preprocessing_spec={
            "_or_": ["snv", "msc", "d1", "savgol"],
            "arrange": 2,  # Generate all 2-step stacked pipelines
        },
        verbose=0,
    )
    results_b = selector_b.fit(X_source, X_target)
    stacked = [r for r in results_b.ranking if r.pipeline_type == "stacked"]
    print(f"   Evaluated: {len(stacked)} stacked combinations")
    print(f"   Best: {results_b.best.name}")

    # Example 4c: With constraints (mutex)
    print("\n4c. With constraints (no double derivatives):")
    selector_c = TransferPreprocessingSelector(
        preset="fast",
        preprocessing_spec={
            "_or_": ["snv", "d1", "d2", "detrend"],
            "arrange": 2,
            "_mutex_": [["d1", "d2"]],  # Don't combine d1 and d2
        },
        verbose=0,
    )
    results_c = selector_c.fit(X_source, X_target)
    stacked_names = [r.name for r in results_c.ranking if r.pipeline_type == "stacked"]
    print(f"   Stacked pipelines: {stacked_names}")
    print("   Note: d1>d2 and d2>d1 are excluded by constraint")

    return results_c


# =============================================================================
# Example 5: Apply Transfer Recommendations to Pipeline
# =============================================================================


def example_5_apply_to_pipeline():
    """
    Demonstrate applying transfer recommendations to a full nirs4all pipeline.

    This is the practical workflow:
    1. Analyze transfer between datasets
    2. Get recommended preprocessing
    3. Use it in a prediction pipeline
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Apply Transfer Recommendations to Pipeline")
    print("=" * 70)

    # Generate source and target data
    X_source, y_source = generate_synthetic_nirs_data(
        n_samples=150, n_features=200, random_state=42
    )
    X_target, y_target = generate_synthetic_nirs_data(
        n_samples=120,
        n_features=200,
        baseline_shift=0.35,
        scatter_variation=0.08,
        random_state=47,
    )

    # Step 1: Analyze transfer
    print("\nStep 1: Analyzing transfer between datasets...")
    selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
    results = selector.fit(X_source, X_target)

    best_pp = results.best.name
    print(f"   Recommended preprocessing: {best_pp}")
    print(f"   Improvement: {results.best.improvement_pct:.1f}%")

    # Step 2: Get pipeline specification
    print("\nStep 2: Getting pipeline specification...")

    # Option A: Single best preprocessing
    single_spec = results.to_pipeline_spec(top_k=1)
    print(f"   Single spec: {single_spec}")

    # Option B: Feature augmentation with top 2
    aug_spec = results.to_pipeline_spec(top_k=2, use_augmentation=True)
    print(f"   Augmentation spec: {aug_spec}")

    # Step 3: Build pipeline with recommended preprocessing
    print("\nStep 3: Building pipeline with recommended preprocessing...")

    # Get preprocessing transforms
    preprocessings = get_base_preprocessings()

    if ">" in best_pp:
        # Stacked preprocessing
        components = best_pp.split(">")
        pp_steps = [preprocessings[c] for c in components]
        print(f"   Using stacked: {' → '.join(components)}")
    else:
        # Single preprocessing
        pp_steps = [preprocessings[best_pp]]
        print(f"   Using single: {best_pp}")

    # Build the pipeline
    pipeline = [
        MinMaxScaler(),
        *pp_steps,  # Apply recommended preprocessing
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
        {"model": PLSRegression(n_components=10)},
        {"model": PLSRegression(n_components=15)},
    ]

    # Step 4: Run on source data (would then apply to target)
    print("\nStep 4: Running pipeline on source dataset...")

    # Save data to temporary location for pipeline
    import tempfile
    import pandas as pd
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "source_data"
        data_path.mkdir()

        # Save in nirs4all format
        pd.DataFrame(X_source).to_csv(
            data_path / "Xcal.csv.gz", index=False, header=False, compression="gzip", sep=";"
        )
        pd.DataFrame(y_source).to_csv(
            data_path / "Ycal.csv.gz", index=False, header=False, compression="gzip", sep=";"
        )

        # Run pipeline
        pipeline_config = PipelineConfigs(pipeline, "Q27_transfer_pipeline")
        dataset_config = DatasetConfigs(str(data_path))

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Show results
        print(f"\n   Generated {predictions.num_predictions} predictions")
        top = predictions.top(n=2, rank_metric="rmse")
        for i, pred in enumerate(top, 1):
            print(
                f"   {i}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}"
            )

    return results


# =============================================================================
# Example 6: Visualization of Results
# =============================================================================


def example_6_visualization(show_plots: bool = False):
    """
    Demonstrate visualization capabilities for transfer analysis.

    Creates:
    - Ranking plot: Bar chart of preprocessing recommendations
    - Metrics comparison: Detailed metrics for top preprocessings
    - Improvement heatmap: Shows improvement across all metrics
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Visualization of Transfer Analysis Results")
    print("=" * 70)

    # Generate data
    X_source, _ = generate_synthetic_nirs_data(
        n_samples=120, n_features=180, random_state=42
    )
    X_target, _ = generate_synthetic_nirs_data(
        n_samples=100,
        n_features=180,
        baseline_shift=0.3,
        scatter_variation=0.1,
        random_state=48,
    )

    # Run analysis
    print("\nRunning transfer analysis...")
    selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
    results = selector.fit(X_source, X_target)

    # Print summary
    print("\n" + results.summary(top_k=5))

    # Create visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Ranking bar chart
    print("   - Creating ranking plot...")
    fig1 = results.plot_ranking(top_k=10)

    # Plot 2: Metrics comparison
    print("   - Creating metrics comparison...")
    fig2 = results.plot_metrics_comparison(top_k=8)

    # Plot 3: Improvement heatmap
    print("   - Creating improvement heatmap...")
    fig3 = results.plot_improvement_heatmap(top_k=10)

    print("\nVisualization complete!")

    if show_plots:
        plt.show()
    else:
        plt.close("all")
        print("(Use --show to display plots)")

    return results


# =============================================================================
# Example 7: Compare Two Regression Pipelines with Transfer Analysis
# =============================================================================


def example_7_compare_pipelines():
    """
    Analyze transfer between two different regression scenarios.

    This demonstrates analyzing transfer between:
    1. Two different datasets (e.g., different years, different machines)
    2. Train/test splits within a single dataset
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Comparing Transfer Between Multiple Dataset Pairs")
    print("=" * 70)

    # Generate three datasets: reference, similar, and different
    print("\nGenerating datasets...")

    # Reference dataset
    X_ref, y_ref = generate_synthetic_nirs_data(
        n_samples=120, n_features=180, random_state=42
    )

    # Similar dataset (small shift)
    X_similar, y_similar = generate_synthetic_nirs_data(
        n_samples=100,
        n_features=180,
        baseline_shift=0.1,
        scatter_variation=0.05,
        random_state=50,
    )

    # Different dataset (large shift)
    X_different, y_different = generate_synthetic_nirs_data(
        n_samples=100,
        n_features=180,
        baseline_shift=0.5,
        scatter_variation=0.15,
        random_state=51,
    )

    print("   Reference dataset: X_ref")
    print("   Similar dataset: X_similar (small shift)")
    print("   Different dataset: X_different (large shift)")

    # Analyze both transfers
    selector = TransferPreprocessingSelector(preset="balanced", verbose=0)

    print("\nAnalyzing Reference → Similar transfer...")
    results_similar = selector.fit(X_ref, X_similar)
    print(f"   Raw centroid distance: {results_similar.raw_metrics['centroid_distance']:.4f}")
    print(f"   Best preprocessing: {results_similar.best.name}")
    print(f"   Improvement: {results_similar.best.improvement_pct:.1f}%")

    print("\nAnalyzing Reference → Different transfer...")
    results_different = selector.fit(X_ref, X_different)
    print(f"   Raw centroid distance: {results_different.raw_metrics['centroid_distance']:.4f}")
    print(f"   Best preprocessing: {results_different.best.name}")
    print(f"   Improvement: {results_different.best.improvement_pct:.1f}%")

    # Compare recommendations
    print("\n" + "-" * 50)
    print("COMPARISON:")
    print("-" * 50)
    print(f"Similar transfer needs: {results_similar.best.name}")
    print(f"Different transfer needs: {results_different.best.name}")

    if results_similar.best.name != results_different.best.name:
        print("\n⚠ Different scenarios require different preprocessing!")
    else:
        print("\n✓ Same preprocessing works for both scenarios")

    return results_similar, results_different


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Q27 - Transfer Preprocessing Analysis for NIRS")
    print("#" * 70)

    # Show available presets
    print("\nAvailable presets:")
    for name, description in list_presets().items():
        print(f"  - {name}: {description}")

    # Run examples
    example_1_basic_transfer_analysis()
    example_2_balanced_with_stacking()
    example_3_full_with_validation()
    example_4_generator_specification()
    example_5_apply_to_pipeline()
    example_6_visualization(show_plots=args.show)
    example_7_compare_pipelines()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

    if args.show:
        plt.show()
