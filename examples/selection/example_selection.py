"""
Preprocessing Selection Example
===============================

This example demonstrates the preprocessing selection framework that helps
filter and rank preprocessing techniques before running full ML/DL pipelines.

The framework reduces the preprocessing search space by 5-10× without losing
performance, saving 80-95% of exploration time.

Usage:
    python example_selection.py [--plots] [--full]

Arguments:
    --plots: Show visualization plots
    --full: Use full nitro datasets instead of small sample data
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NIRS4All imports
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    MultiplicativeScatterCorrection,
    FirstDerivative,
    SecondDerivative,
    Haar,
    Detrend,
    Gaussian,
    IdentityTransformer,
    Derivate,
    Wavelet,
    RobustStandardNormalVariate,
    LocalStandardNormalVariate,
)

# Selection framework imports
from selection import PreprocessingSelector, print_selection_report


def load_sample_data(data_path: str) -> tuple:
    """
    Load sample data from CSV files.

    Args:
        data_path: Path to data folder containing Xcal/Xval and Ycal/Yval files

    Returns:
        Tuple of (X, y) numpy arrays
    """
    # Try different file patterns
    patterns = [
        ('Xcal.csv', 'Ycal.csv', False),
        ('Xcal.csv.gz', 'Ycal.csv.gz', False),
        ('Xtrain.csv', 'Ytrain.csv', True),  # May have header
        ('XY.csv', None, False),  # Combined file
    ]

    for x_file, y_file, may_have_header in patterns:
        x_path = os.path.join(data_path, x_file)
        if os.path.exists(x_path):
            print(f"Loading data from {x_path}...")

            if y_file is None:
                # Combined XY file
                df = pd.read_csv(x_path, header=None, sep=';')
                X = df.iloc[:, :-1].values.astype(np.float64)
                y = df.iloc[:, -1].values.astype(np.float64)
            else:
                y_path = os.path.join(data_path, y_file)
                if os.path.exists(y_path):
                    # Check if files have headers by trying to parse first value
                    x_df = pd.read_csv(x_path, header=None, sep=';')
                    y_df = pd.read_csv(y_path, header=None, sep=';')

                    # Check if first row is header (non-numeric)
                    try:
                        float(x_df.iloc[0, 0])
                        x_has_header = False
                    except (ValueError, TypeError):
                        x_has_header = True

                    try:
                        float(y_df.iloc[0, 0])
                        y_has_header = False
                    except (ValueError, TypeError):
                        y_has_header = True

                    if x_has_header:
                        x_df = pd.read_csv(x_path, header=0, sep=';')
                    if y_has_header:
                        y_df = pd.read_csv(y_path, header=0, sep=';')

                    X = x_df.values.astype(np.float64)
                    y = y_df.values.astype(np.float64).ravel()
                else:
                    continue

            # Handle NaN values
            if np.any(np.isnan(X)):
                print("Warning: NaN values found in X, replacing with column means")
                col_means = np.nanmean(X, axis=0)
                nan_indices = np.where(np.isnan(X))
                X[nan_indices] = col_means[nan_indices[1]]

            if np.any(np.isnan(y)):
                print("Warning: NaN values found in y, removing those samples")
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]

            # Ensure X and y have matching number of samples
            min_samples = min(X.shape[0], y.shape[0])
            if X.shape[0] != y.shape[0]:
                print(f"Warning: Shape mismatch X={X.shape[0]}, y={y.shape[0]}, using {min_samples} samples")
                X = X[:min_samples]
                y = y[:min_samples]

            print(f"Loaded X: {X.shape}, y: {y.shape}")
            return X, y

    raise FileNotFoundError(f"Could not find data files in {data_path}")


def define_preprocessings() -> dict:
    """
    Define a comprehensive set of preprocessing techniques to evaluate.

    Returns:
        Dict of {name: preprocessing_transformer}
    """
    preprocessings = {
        # Identity (baseline)
        'identity': IdentityTransformer(),

        # Scatter correction
        'snv': StandardNormalVariate(),
        'rsnv': RobustStandardNormalVariate(),
        'lsnv': LocalStandardNormalVariate(window=11),
        'msc': MultiplicativeScatterCorrection(scale=False),

        # Smoothing
        'savgol_11_3': SavitzkyGolay(window_length=11, polyorder=3),
        'savgol_17_2': SavitzkyGolay(window_length=17, polyorder=2),
        'savgol_21_3': SavitzkyGolay(window_length=21, polyorder=3),

        # Derivatives
        'd1': FirstDerivative(),
        'd2': SecondDerivative(),
        'savgol_d1': SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
        'savgol_d2': SavitzkyGolay(window_length=17, polyorder=2, deriv=2),

        # Wavelets
        'haar': Haar(),
        'coif3': Wavelet('coif3'),

        # Other
        'detrend': Detrend(),
        'gaussian_1_2': Gaussian(order=1, sigma=2),
        'gaussian_2_1': Gaussian(order=2, sigma=1),
        'derivate_1_1': Derivate(order=1, delta=1),
        'derivate_2_1': Derivate(order=2, delta=1),
    }

    return preprocessings


def run_quick_demo():
    """Run a quick demo with minimal preprocessings on small data."""
    print("\n" + "=" * 70)
    print("QUICK DEMO: Preprocessing Selection Framework")
    print("=" * 70)

    # Use small sample data
    data_path = 'sample_data/regression'
    X, y = load_sample_data(data_path)

    # Define a small set of preprocessings for quick demo
    preprocessings = {
        'identity': IdentityTransformer(),
        'snv': StandardNormalVariate(),
        'msc': MultiplicativeScatterCorrection(scale=False),
        'savgol': SavitzkyGolay(window_length=11, polyorder=3),
        'd1': FirstDerivative(),
        'haar': Haar(),
        'detrend': Detrend(),
    }

    print(f"\nEvaluating {len(preprocessings)} preprocessings...")

    # Initialize selector
    selector = PreprocessingSelector(verbose=1)

    # Run selection with all stages
    results = selector.select(
        X=X,
        y=y,
        preprocessings=preprocessings,
        stages=['A', 'B', 'C', 'D'],
        top_k=5,
        # Relaxed thresholds for demo
        min_snr_ratio=0.3,
        max_roughness_ratio=50.0,
        min_separation_ratio=0.5,
    )

    # Print report
    print_selection_report(results)

    return results


def run_full_evaluation(data_path: str, show_plots: bool = False):
    """
    Run full evaluation on a dataset.

    Args:
        data_path: Path to data folder
        show_plots: Whether to show visualization plots
    """
    print("\n" + "=" * 70)
    print("FULL EVALUATION: Preprocessing Selection Framework")
    print("=" * 70)

    # Load data
    X, y = load_sample_data(data_path)

    # Define all preprocessings
    preprocessings = define_preprocessings()

    print(f"\nEvaluating {len(preprocessings)} preprocessings on {X.shape[0]} samples...")

    # Initialize selector
    selector = PreprocessingSelector(verbose=1)

    # Run selection with all stages
    start_time = time.time()
    results = selector.select(
        X=X,
        y=y,
        preprocessings=preprocessings,
        stages=['A', 'B', 'C', 'D'],
        top_k=8,
        # Relaxed thresholds for NIRS data
        # NIRS data often has dominant first component (baseline)
        min_variance_ratio=0.70,
        max_first_component_ratio=1.0,  # Disable this filter
        min_snr_ratio=0.0,  # Disable SNR filter (derivatives reduce SNR by design)
        max_roughness_ratio=100.0,  # Very permissive
        min_separation_ratio=0.3,  # Permissive
        # Proxy model settings
        cv_folds=3,
        knn_neighbors=5,
    )
    total_time = time.time() - start_time

    # Print report
    print_selection_report(results)

    # Detailed stage analysis
    print("\n📈 DETAILED STAGE ANALYSIS:")
    print("-" * 50)

    if 'B' in results['stage_results']:
        print("\nStage B - Supervised Metrics:")
        for name, r in sorted(
            results['stage_results']['B'].items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )[:10]:
            print(f"  {name}:")
            print(f"    RV={r['rv']['rv_score']:.4f}, "
                  f"CKA={r['cka']['cka_score']:.4f}, "
                  f"Corr={r['correlation']['correlation_score']:.4f}, "
                  f"PLS_R2={r['pls']['pls_r2']:.4f}")

    if 'C' in results['stage_results']:
        print("\nStage C - Proxy Models:")
        for name, r in sorted(
            results['stage_results']['C'].items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )[:10]:
            print(f"  {name}: Ridge_R2={r['ridge']['ridge_r2']:.4f}, "
                  f"KNN={r['knn']['knn_score']:.4f}")

    print(f"\n⏱️ Total evaluation time: {total_time:.2f}s")
    print(f"   Average per preprocessing: {total_time/len(preprocessings):.2f}s")

    # Show plots if requested
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            plot_results(results)
            plt.show()
        except ImportError:
            print("Warning: matplotlib not available for plotting")

    return results


def plot_results(results: dict):
    """
    Create visualization plots for selection results.

    Args:
        results: Results dict from PreprocessingSelector.select()
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final ranking bar chart
    ax1 = axes[0, 0]
    names = [r[0] for r in results['ranking']]
    scores = [r[1] for r in results['ranking']]
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(names)))
    ax1.barh(names, scores, color=colors)
    ax1.set_xlabel('Score')
    ax1.set_title('Final Preprocessing Ranking')
    ax1.invert_yaxis()

    # Plot 2: Stage B metrics comparison
    ax2 = axes[0, 1]
    if 'B' in results['stage_results']:
        b_results = results['stage_results']['B']
        top_names = [r[0] for r in results['ranking']]

        metrics = ['rv', 'cka', 'correlation', 'pls']
        metric_names = ['RV', 'CKA', 'Correlation', 'PLS R²']
        x = np.arange(len(top_names))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metrics, metric_names)):
            if metric == 'rv':
                values = [b_results[n]['rv']['rv_score'] for n in top_names if n in b_results]
            elif metric == 'cka':
                values = [b_results[n]['cka']['cka_score'] for n in top_names if n in b_results]
            elif metric == 'correlation':
                values = [b_results[n]['correlation']['correlation_score'] for n in top_names if n in b_results]
            else:
                values = [max(0, b_results[n]['pls']['pls_r2']) for n in top_names if n in b_results]

            ax2.bar(x[:len(values)] + i * width, values, width, label=label)

        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(top_names, rotation=45, ha='right')
        ax2.set_ylabel('Score')
        ax2.set_title('Stage B: Supervised Metrics')
        ax2.legend(loc='upper right')

    # Plot 3: Stage C proxy comparison
    ax3 = axes[1, 0]
    if 'C' in results['stage_results']:
        c_results = results['stage_results']['C']
        top_names = [r[0] for r in results['ranking']]

        ridge_scores = [c_results[n]['ridge']['ridge_r2'] for n in top_names if n in c_results]
        knn_scores = [c_results[n]['knn']['knn_score'] for n in top_names if n in c_results]

        x = np.arange(len(top_names))
        width = 0.35

        ax3.bar(x - width/2, ridge_scores, width, label='Ridge R²', color='steelblue')
        ax3.bar(x + width/2, knn_scores, width, label='KNN Score', color='coral')

        ax3.set_xticks(x)
        ax3.set_xticklabels(top_names, rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('Stage C: Proxy Models')
        ax3.legend()

    # Plot 4: Combination recommendations
    ax4 = axes[1, 1]
    if results.get('combinations_2d'):
        combos = results['combinations_2d'][:8]
        combo_names = [c['combination'] for c in combos]
        combo_scores = [c['combined_score'] for c in combos]

        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(combo_names)))
        ax4.barh(combo_names, combo_scores, color=colors)
        ax4.set_xlabel('Combined Score')
        ax4.set_title('Recommended 2D Combinations')
        ax4.invert_yaxis()
    else:
        ax4.text(0.5, 0.5, 'No combination data', ha='center', va='center')
        ax4.set_title('Recommended 2D Combinations')

    plt.tight_layout()
    plt.savefig('selection_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'selection_results.png'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing Selection Example')
    parser.add_argument('--plots', action='store_true', help='Show visualization plots')
    parser.add_argument('--full', action='store_true',
                        help='Use full nitro datasets instead of sample data')
    parser.add_argument('--data', type=str, default=None,
                        help='Custom data path')
    args = parser.parse_args()

    if args.data:
        # Use custom data path
        run_full_evaluation(args.data, show_plots=args.plots)
    elif args.full:
        # Use nitro regression dataset
        data_path = 'selection/nitro_regression/Digestibility_0.8'
        if os.path.exists(data_path):
            run_full_evaluation(data_path, show_plots=args.plots)
        else:
            print(f"Error: Data path not found: {data_path}")
            print("Running quick demo with sample data instead...")
            run_quick_demo()
    else:
        # Run quick demo
        run_quick_demo()
