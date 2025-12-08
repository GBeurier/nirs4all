"""
Splitter Selection Runner
=========================

Main script for evaluating and comparing different splitting strategies
for NIRS spectral data. This script:

1. Loads spectral data (X), targets (Y), and metadata (M)
2. Applies multiple splitting strategies respecting sample grouping
3. Trains baseline models on each strategy
4. Compares performance and recommends the best strategy

Usage:
    python run_splitter_selection.py --data_dir path/to/data --output_dir path/to/output

The data directory should contain:
    - X.csv: Spectra matrix (semicolon-separated)
    - Y.csv: Target values
    - M.csv: Metadata with ID and Rep columns (semicolon-separated)
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from splitter_strategies import (
    SPLITTING_STRATEGIES,
    get_splitter,
    list_strategies,
    SimpleSplitter,
    TargetStratifiedSplitter,
    SpectralPCASplitter,
    SpectralDistanceSplitter,
    HybridSplitter,
    AdversarialSplitter,
    KennardStoneSplitter,
    StratifiedGroupKFoldSplitter,
    Nirs4allKennardStoneSplitter,
    Nirs4allSPXYSplitter,
    HAS_NIRS4ALL_SPLITTERS,
    PuchweinSplitter,
    DuplexSplitter,
    ShenkWestSplitter,
    HonigsSplitter,
    HierarchicalClusteringSplitter,
    KMedoidsSplitter
)
from splitter_evaluation import (
    evaluate_strategy,
    compare_strategies,
    identify_best_strategies,
    compute_statistical_tests,
    StrategyResult
)
from splitter_visualization import (
    plot_comparison_comprehensive,
    plot_predictions,
    plot_cv_distribution,
    plot_split_pca,
    plot_residuals,
    create_summary_report
)

warnings.filterwarnings('ignore')


def load_data(data_dir: Path) -> tuple:
    """
    Load X, Y, M data from CSV files.

    Args:
        data_dir: Directory containing X.csv, Y.csv, M.csv

    Returns:
        X: Spectra array (n_samples, n_features)
        y: Target values (n_samples,)
        sample_ids: Sample IDs (n_samples,)
        metadata: Full metadata DataFrame
    """
    # Load X (spectra)
    X_path = data_dir / 'X.csv'
    X_df = pd.read_csv(X_path, sep=';')
    X = X_df.values.astype(np.float32)

    # Load Y (target)
    Y_path = data_dir / 'Y.csv'
    Y_df = pd.read_csv(Y_path)
    y = Y_df.iloc[:, 0].values.astype(np.float32)

    # Load M (metadata)
    M_path = data_dir / 'M.csv'
    metadata = pd.read_csv(M_path, sep=';')
    sample_ids = metadata['ID'].values

    print(f"Data loaded from: {data_dir}")
    print(f"  Spectra shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Unique samples (IDs): {len(np.unique(sample_ids))}")
    print(f"  Repetitions per sample: {len(y) // len(np.unique(sample_ids)):.1f} (avg)")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y, sample_ids, metadata


def get_configured_strategies(
    test_size: float = 0.2,
    n_folds: int = 3,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Get all configured splitting strategies.

    Args:
        test_size: Fraction for test set
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary of strategy configurations
    """
    strategies = {
        'simple': {
            'splitter': SimpleSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state
            ),
            'name': 'Simple Random',
            'category': 'Baseline',
            'description': 'Random split at sample ID level'
        },
        'target_stratified': {
            'splitter': TargetStratifiedSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_bins=5
            ),
            'name': 'Target Stratified',
            'category': 'Target-Based',
            'description': 'Stratified by target value bins (5 bins)'
        },
        'target_stratified_10bins': {
            'splitter': TargetStratifiedSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_bins=10
            ),
            'name': 'Target Stratified (10 bins)',
            'category': 'Target-Based',
            'description': 'Stratified by target value bins (10 bins)'
        },
        'spectral_pca_5c': {
            'splitter': SpectralPCASplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_clusters=5,
                pca_variance=0.95
            ),
            'name': 'Spectral PCA (5 clusters)',
            'category': 'Spectral-Based',
            'description': 'Stratified by 5 PCA clusters of spectra'
        },
        'spectral_pca_10c': {
            'splitter': SpectralPCASplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_clusters=10,
                pca_variance=0.95
            ),
            'name': 'Spectral PCA (10 clusters)',
            'category': 'Spectral-Based',
            'description': 'Stratified by 10 PCA clusters of spectra'
        },
        'spectral_distance': {
            'splitter': SpectralDistanceSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                pca_components=10
            ),
            'name': 'Spectral Distance',
            'category': 'Spectral-Based',
            'description': 'Farthest point sampling for spectral diversity'
        },
        'hybrid_5c_3b': {
            'splitter': HybridSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_spectral_clusters=5,
                n_target_bins=3
            ),
            'name': 'Hybrid (5 clusters, 3 bins)',
            'category': 'Hybrid',
            'description': 'Combined spectral (5) and target (3) stratification'
        },
        'hybrid_8c_5b': {
            'splitter': HybridSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_spectral_clusters=8,
                n_target_bins=5
            ),
            'name': 'Hybrid (8 clusters, 5 bins)',
            'category': 'Hybrid',
            'description': 'Combined spectral (8) and target (5) stratification'
        },
        'adversarial_30': {
            'splitter': AdversarialSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                adversarial_strength=0.3
            ),
            'name': 'Adversarial (30%)',
            'category': 'Robustness',
            'description': 'Challenging test set with 30% outlier samples'
        },
        'adversarial_50': {
            'splitter': AdversarialSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                adversarial_strength=0.5
            ),
            'name': 'Adversarial (50%)',
            'category': 'Robustness',
            'description': 'Challenging test set with 50% outlier samples'
        },
        'kennard_stone': {
            'splitter': KennardStoneSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                pca_components=10
            ),
            'name': 'Kennard-Stone',
            'category': 'Chemometrics',
            'description': 'Classic chemometric sample selection algorithm'
        },
        'stratified_group_kfold': {
            'splitter': StratifiedGroupKFoldSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                n_bins=5
            ),
            'name': 'Stratified Group KFold',
            'category': 'Stratified',
            'description': 'Sklearn StratifiedGroupKFold for stratified CV with groups'
        },
    }

    # Add nirs4all splitters if available
    if HAS_NIRS4ALL_SPLITTERS:
        strategies['nirs4all_kennard_stone'] = {
            'splitter': Nirs4allKennardStoneSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                pca_components=10
            ),
            'name': 'Nirs4all Kennard-Stone',
            'category': 'Chemometrics',
            'description': 'Kennard-Stone from nirs4all library'
        }
        strategies['nirs4all_spxy'] = {
            'splitter': Nirs4allSPXYSplitter(
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                pca_components=10
            ),
            'name': 'Nirs4all SPXY',
            'category': 'Chemometrics',
            'description': 'SPXY sampling from nirs4all library (X+Y based)'
        }

    # Add unsupervised sample selection splitters
    strategies['puchwein'] = {
        'splitter': PuchweinSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            factor_k=0.05,
            pca_components=10
        ),
        'name': 'Puchwein',
        'category': 'Chemometrics',
        'description': 'Puchwein distance-based sample selection'
    }
    strategies['duplex'] = {
        'splitter': DuplexSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            pca_components=10
        ),
        'name': 'Duplex',
        'category': 'Chemometrics',
        'description': 'Duplex alternating train/test selection'
    }
    strategies['shenkwest'] = {
        'splitter': ShenkWestSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            pca_components=10
        ),
        'name': 'Shenk-Westerhaus',
        'category': 'Chemometrics',
        'description': 'Shenk & Westerhaus distance-based selection'
    }
    strategies['honigs'] = {
        'splitter': HonigsSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state
        ),
        'name': 'Honigs',
        'category': 'Chemometrics',
        'description': 'Honigs spectral uniqueness selection'
    }
    strategies['hierarchical_clustering'] = {
        'splitter': HierarchicalClusteringSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            pca_components=10,
            linkage='complete'
        ),
        'name': 'Hierarchical Clustering',
        'category': 'Clustering',
        'description': 'Agglomerative clustering-based selection'
    }
    strategies['kmedoids'] = {
        'splitter': KMedoidsSplitter(
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            pca_components=10
        ),
        'name': 'K-Medoids',
        'category': 'Clustering',
        'description': 'K-Medoids based sample selection'
    }

    return strategies


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    strategies: Dict[str, Dict[str, Any]],
    model_type: str = 'ridge',
    verbose: bool = True,
    **model_kwargs
) -> tuple:
    """
    Run evaluation on all strategies.

    Args:
        X: Spectra array
        y: Target values
        sample_ids: Sample IDs
        strategies: Dictionary of strategy configurations
        model_type: Baseline model type
        verbose: Print progress
        **model_kwargs: Model parameters

    Returns:
        results: List of StrategyResult
        split_results: Dictionary of SplitResult by strategy key
    """
    results = []
    split_results = {}

    print("\n" + "=" * 80)
    print("RUNNING SPLITTER SELECTION EVALUATION")
    print("=" * 80)

    for strategy_key, strategy_info in strategies.items():
        print(f"\n{'=' * 80}")
        print(f"Strategy: {strategy_info['name']}")
        print(f"Category: {strategy_info['category']}")
        print(f"Description: {strategy_info['description']}")
        print("=" * 80)

        splitter = strategy_info['splitter']

        # Perform split
        split_result = splitter.split(X, y, sample_ids)
        split_results[strategy_key] = split_result

        # Print split info
        config = splitter.get_stratification_info()
        print(f"Configuration: {config}")

        # Evaluate
        result = evaluate_strategy(
            X, y, sample_ids,
            split_result,
            strategy_key,
            strategy_info['name'],
            strategy_info['category'],
            model_type=model_type,
            verbose=verbose,
            **model_kwargs
        )

        results.append(result)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results, split_results


def save_results(
    results: list,
    comparison_df: pd.DataFrame,
    best_strategies: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Save results to files.

    Args:
        results: List of StrategyResult
        comparison_df: Comparison DataFrame
        best_strategies: Best strategies by criterion
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison CSV
    comparison_df.to_csv(output_dir / 'strategy_comparison.csv', index=False)

    # Save best strategies JSON
    with open(output_dir / 'best_strategies.json', 'w') as f:
        json.dump(best_strategies, f, indent=2)

    # Save detailed results
    detailed = []
    for r in results:
        detailed.append({
            'strategy_key': r.strategy_key,
            'strategy_name': r.strategy_name,
            'category': r.category,
            'test_rmse': r.test_rmse,
            'test_mae': r.test_mae,
            'test_r2': r.test_r2,
            'cv_rmse_mean': r.cv_rmse_mean,
            'cv_rmse_std': r.cv_rmse_std,
            'cv_r2_mean': r.cv_r2_mean,
            'cv_r2_std': r.cv_r2_std,
            'generalization_gap': r.generalization_gap,
            'n_test': r.n_test,
            'strategy_info': r.strategy_info
        })

    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


def main(
    data_dir: str,
    output_dir: str = None,
    test_size: float = 0.2,
    n_folds: int = 3,
    model_type: str = 'xgboost',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main function for splitter selection.

    Args:
        data_dir: Directory containing X.csv, Y.csv, M.csv
        output_dir: Directory for outputs (default: data_dir/splitter_selection)
        test_size: Fraction for test set
        n_folds: Number of CV folds
        model_type: Baseline model type ('ridge', 'pls', 'knn', 'xgboost')
        random_state: Random seed
        verbose: Print progress

    Returns:
        Dictionary with results summary
    """
    data_path = Path(data_dir)
    if output_dir is None:
        output_path = data_path / 'splitter_selection'
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X, y, sample_ids, metadata = load_data(data_path)

    # Get strategies
    strategies = get_configured_strategies(
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )

    print(f"\n{len(strategies)} splitting strategies configured")

    # Run evaluation
    results, split_results = run_evaluation(
        X, y, sample_ids, strategies,
        model_type=model_type,
        verbose=verbose
    )

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING STRATEGIES")
    print("=" * 80)

    comparison_df = compare_strategies(results)
    best_strategies = identify_best_strategies(comparison_df)

    # Print summary
    print("\n📊 RANKING (by Test RMSE):")
    print(comparison_df[['strategy', 'category', 'test_rmse', 'test_r2',
                          'cv_rmse_mean', 'generalization_gap']].to_string())

    print("\n🏆 BEST STRATEGIES:")
    for criterion, info in best_strategies.items():
        print(f"  {criterion}: {info['strategy']}")

    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (Top 4)")
    print("=" * 80)
    stat_df = compute_statistical_tests(results[:4])
    print(stat_df.to_string())

    # Save results
    save_results(results, comparison_df, best_strategies, output_path)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Comprehensive comparison
    fig = plot_comparison_comprehensive(
        comparison_df, results,
        save_path=str(figures_dir / 'comparison_comprehensive.png')
    )
    plt.close(fig)
    print("  ✓ Comprehensive comparison plot")

    # Predictions
    fig = plot_predictions(
        results, n_top=6,
        save_path=str(figures_dir / 'predictions_top6.png')
    )
    plt.close(fig)
    print("  ✓ Predictions plot (top 6)")

    # CV distribution
    fig = plot_cv_distribution(
        results, comparison_df,
        save_path=str(figures_dir / 'cv_distribution.png')
    )
    plt.close(fig)
    print("  ✓ CV distribution boxplot")

    # Residuals
    fig = plot_residuals(
        results, n_top=4,
        save_path=str(figures_dir / 'residuals_top4.png')
    )
    plt.close(fig)
    print("  ✓ Residual analysis (top 4)")

    # PCA visualization for best strategy
    best_key = comparison_df.iloc[0]['strategy_key']
    best_split = split_results[best_key]
    fig = plot_split_pca(
        X, sample_ids,
        best_split.train_ids, best_split.test_ids,
        best_split.fold_assignments,
        comparison_df.iloc[0]['strategy'],
        save_path=str(figures_dir / 'best_strategy_pca.png')
    )
    plt.close(fig)
    print("  ✓ Best strategy PCA visualization")

    # Summary report
    create_summary_report(
        comparison_df, best_strategies,
        str(output_path / 'SUMMARY_REPORT.txt')
    )

    # Final recommendation
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATION")
    print("=" * 80)
    best = comparison_df.iloc[0]
    print(f"\nRecommended Strategy: {best['strategy']}")
    print(f"  Category: {best['category']}")
    print(f"  Test RMSE: {best['test_rmse']:.4f}")
    print(f"  Test R²: {best['test_r2']:.4f}")
    print(f"  CV RMSE: {best['cv_rmse_mean']:.4f} ± {best['cv_rmse_std']:.4f}")
    print(f"  Generalization Gap: {best['generalization_gap']:+.4f}")

    print(f"\n📁 All results saved to: {output_path}")

    return {
        'best_strategy': best['strategy'],
        'best_strategy_key': best['strategy_key'],
        'test_rmse': best['test_rmse'],
        'test_r2': best['test_r2'],
        'comparison_df': comparison_df,
        'results': results,
        'output_dir': str(output_path)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Splitter Selection for NIRS Spectral Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_splitter_selection.py --data_dir ./data --n_folds 3
  python run_splitter_selection.py --data_dir ./data --model xgboost --test_size 0.2
  python run_splitter_selection.py --data_dir ./data --model ridge --test_size 0.2
        """
    )

    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='nitro_regression_merged/Digestibility_0.8',
        help='Directory containing X.csv, Y.csv, M.csv'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: data_dir/splitter_selection)'
    )
    parser.add_argument(
        '--test_size', '-t',
        type=float,
        default=0.2,
        help='Fraction of samples for test set (default: 0.2)'
    )
    parser.add_argument(
        '--n_folds', '-f',
        type=int,
        default=3,
        help='Number of cross-validation folds (default: 3)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='xgboost',
        choices=['ridge', 'pls', 'knn', 'xgboost'],
        help='Baseline model type (default: xgboost)'
    )
    parser.add_argument(
        '--random_state', '-r',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    result = main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_folds=args.n_folds,
        model_type=args.model,
        random_state=args.random_state,
        verbose=not args.quiet
    )
