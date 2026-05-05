"""
Enhanced Splitter Selection Runner
==================================

Advanced script for comprehensive evaluation of splitting strategies.
This version includes:

1. Multiple models per splitter (Ridge, ElasticNet, PLS, SVR, GBR, XGBoost, MLP)
2. Repeated cross-validation (3 repeats with different seeds)
3. Bootstrap confidence intervals for all metrics
4. Representativeness metrics (spectral coverage, target coverage, leverage)

Usage:
    python run_splitter_selection_enhanced.py --data_dir path/to/data --output_dir path/to/output

Expected evaluation time: 30-60 minutes for typical NIRS datasets (depends on dataset size)
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from splitter_evaluation_enhanced import EnhancedStrategyResult, compare_strategies_enhanced, compute_statistical_tests_enhanced, evaluate_strategy_enhanced, get_model_suite, identify_best_strategies_enhanced
from splitter_strategies import (
    HAS_NIRS4ALL_SPLITTERS,
    SPLITTING_STRATEGIES,
    AdversarialSplitter,
    DuplexSplitter,
    HierarchicalClusteringSplitter,
    HonigsSplitter,
    HybridSplitter,
    KennardStoneSplitter,
    KMedoidsSplitter,
    Nirs4allKennardStoneSplitter,
    Nirs4allSPXYSplitter,
    PuchweinSplitter,
    ShenkWestSplitter,
    SimpleSplitter,
    SpectralDistanceSplitter,
    SpectralPCASplitter,
    SplitResult,
    StratifiedGroupKFoldSplitter,
    TargetStratifiedSplitter,
    get_splitter,
    list_strategies,
)
from splitter_visualization_enhanced import (
    create_summary_report_enhanced,
    plot_bootstrap_confidence,
    plot_comparison_enhanced,
    plot_cv_distribution_enhanced,
    plot_model_comparison,
    plot_predictions_enhanced,
    plot_representativeness,
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
) -> dict[str, dict[str, Any]]:
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

    # Add additional chemometrics splitters
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

def run_enhanced_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    strategies: dict[str, dict[str, Any]],
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    verbose: bool = True
) -> tuple:
    """
    Run enhanced evaluation on all strategies.

    Args:
        X: Spectra array
        y: Target values
        sample_ids: Sample IDs
        strategies: Dictionary of strategy configurations
        n_repeats: Number of CV repetitions
        n_bootstrap: Number of bootstrap samples
        verbose: Print progress

    Returns:
        results: List of EnhancedStrategyResult
        split_results: Dictionary of SplitResult by strategy key
    """
    results = []
    split_results = {}

    models = get_model_suite()
    model_names = list(models.keys())

    print("\n" + "=" * 80)
    print("ENHANCED SPLITTER SELECTION EVALUATION")
    print("=" * 80)
    print("\n📊 Configuration:")
    print(f"   Models: {', '.join(model_names)}")
    print(f"   CV Repeats: {n_repeats}")
    print(f"   Bootstrap Samples: {n_bootstrap}")
    print(f"   Strategies to evaluate: {len(strategies)}")

    total_start = time.time()

    for idx, (strategy_key, strategy_info) in enumerate(strategies.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(strategies)}] Strategy: {strategy_info['name']}")
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

        # Enhanced evaluation
        result = evaluate_strategy_enhanced(
            X, y, sample_ids,
            split_result,
            strategy_key,
            strategy_info['name'],
            strategy_info['category'],
            n_repeats=n_repeats,
            n_bootstrap=n_bootstrap,
            verbose=verbose
        )

        results.append(result)

        # Progress estimate
        elapsed = time.time() - total_start
        avg_per_strategy = elapsed / idx
        remaining = avg_per_strategy * (len(strategies) - idx)
        print(f"\n  ⏱ Progress: {idx}/{len(strategies)} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"Remaining: ~{remaining/60:.1f}min")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE in {total_time/60:.1f} minutes")
    print("=" * 80)

    return results, split_results

def save_enhanced_results(
    results: list,
    comparison_df: pd.DataFrame,
    best_strategies: dict[str, dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Save enhanced results to files.

    Args:
        results: List of EnhancedStrategyResult
        comparison_df: Comparison DataFrame
        best_strategies: Best strategies by criterion
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison CSV
    comparison_df.to_csv(output_dir / 'strategy_comparison_enhanced.csv', index=False)

    # Save best strategies JSON
    with open(output_dir / 'best_strategies_enhanced.json', 'w') as f:
        json.dump(best_strategies, f, indent=2, default=str)

    # Save detailed results
    detailed = []
    for r in results:
        detailed.append({
            'strategy_key': r.strategy_key,
            'strategy_name': r.strategy_name,
            'category': r.category,
            'test_rmse': r.test_rmse,
            'test_rmse_ci': [r.bootstrap_metrics.rmse_ci_lower, r.bootstrap_metrics.rmse_ci_upper],
            'test_mae': r.test_mae,
            'test_r2': r.test_r2,
            'test_r2_ci': [r.bootstrap_metrics.r2_ci_lower, r.bootstrap_metrics.r2_ci_upper],
            'cv_rmse_mean': r.cv_rmse_mean,
            'cv_rmse_std': r.cv_rmse_std,
            'cv_r2_mean': r.cv_r2_mean,
            'cv_r2_std': r.cv_r2_std,
            'generalization_gap': r.generalization_gap,
            'n_test': r.n_test,
            'n_repeats': r.n_repeats,
            'n_folds': r.n_folds,
            'model_results': r.model_test_results,
            'representativeness': {
                'spectral_coverage': r.representativeness.spectral_coverage,
                'target_wasserstein': r.representativeness.target_wasserstein,
                'target_kl_divergence': r.representativeness.target_kl_divergence,
                'leverage_mean': r.representativeness.leverage_mean,
                'leverage_max': r.representativeness.leverage_max,
                'n_high_leverage': r.representativeness.n_high_leverage,
                'hotelling_t2_mean': r.representativeness.hotelling_t2_mean
            },
            'total_time': r.total_time,
            'strategy_info': r.strategy_info
        })

    with open(output_dir / 'detailed_results_enhanced.json', 'w') as f:
        json.dump(detailed, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")

def export_best_split_regression(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    best_result: EnhancedStrategyResult,
    best_split: SplitResult,
    sample_ids: np.ndarray,
    output_dir: Path
) -> None:
    """
    Export the best split data to CSV files for regression.

    Creates:
        - X_train.csv, X_test.csv (spectra)
        - Y_train.csv, Y_test.csv (targets)
        - M_train.csv, M_test.csv (metadata)
        - split_info.json (split configuration and metrics)
    """
    export_dir = output_dir / 'best_split_export'
    export_dir.mkdir(parents=True, exist_ok=True)

    # Get train/test masks
    train_mask = np.isin(sample_ids, best_split.train_ids)
    test_mask = np.isin(sample_ids, best_split.test_ids)

    # Export X
    X_train_df = pd.DataFrame(X[train_mask])
    X_test_df = pd.DataFrame(X[test_mask])
    X_train_df.to_csv(export_dir / 'X_train.csv', sep=';', index=False)
    X_test_df.to_csv(export_dir / 'X_test.csv', sep=';', index=False)

    # Export Y
    Y_train_df = pd.DataFrame({'y': y[train_mask]})
    Y_test_df = pd.DataFrame({'y': y[test_mask]})
    Y_train_df.to_csv(export_dir / 'Y_train.csv', index=False)
    Y_test_df.to_csv(export_dir / 'Y_test.csv', index=False)

    # Export M
    M_train = metadata.iloc[np.where(train_mask)[0]].copy()
    M_test = metadata.iloc[np.where(test_mask)[0]].copy()
    M_train['split'] = 'train'
    M_test['split'] = 'test'
    M_train.to_csv(export_dir / 'M_train.csv', sep=';', index=False)
    M_test.to_csv(export_dir / 'M_test.csv', sep=';', index=False)

    # Export fold assignments
    best_split.fold_assignments.to_csv(export_dir / 'fold_assignments.csv', index=False)

    # Export split info
    split_info = {
        'strategy_key': best_result.strategy_key,
        'strategy_name': best_result.strategy_name,
        'category': best_result.category,
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'n_train_ids': len(best_split.train_ids),
        'n_test_ids': len(best_split.test_ids),
        'test_rmse': best_result.test_rmse,
        'test_rmse_ci': [
            best_result.bootstrap_metrics.rmse_ci_lower,
            best_result.bootstrap_metrics.rmse_ci_upper
        ],
        'test_r2': best_result.test_r2,
        'test_r2_ci': [
            best_result.bootstrap_metrics.r2_ci_lower,
            best_result.bootstrap_metrics.r2_ci_upper
        ],
        'cv_rmse_mean': best_result.cv_rmse_mean,
        'cv_rmse_std': best_result.cv_rmse_std,
        'generalization_gap': best_result.generalization_gap,
        'spectral_coverage': best_result.representativeness.spectral_coverage,
        'target_wasserstein': best_result.representativeness.target_wasserstein,
        'n_high_leverage': best_result.representativeness.n_high_leverage,
        'strategy_info': best_result.strategy_info
    }

    with open(export_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2, default=str)

    print(f"\n📁 Best split exported to: {export_dir}")
    print(f"   - X_train.csv: {X_train_df.shape}")
    print(f"   - X_test.csv: {X_test_df.shape}")
    print(f"   - Y_train.csv: {Y_train_df.shape}")
    print(f"   - Y_test.csv: {Y_test_df.shape}")
    print(f"   - M_train.csv: {M_train.shape}")
    print(f"   - M_test.csv: {M_test.shape}")
    print("   - fold_assignments.csv")
    print("   - split_info.json")

def main(
    data_dir: str,
    output_dir: str = None,
    test_size: float = 0.2,
    n_folds: int = 3,
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Main function for enhanced splitter selection.

    Args:
        data_dir: Directory containing X.csv, Y.csv, M.csv
        output_dir: Directory for outputs (default: data_dir/splitter_selection_enhanced)
        test_size: Fraction for test set
        n_folds: Number of CV folds
        n_repeats: Number of CV repetitions
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        verbose: Print progress

    Returns:
        Dictionary with results summary
    """
    data_path = Path(data_dir)
    output_path = data_path / 'splitter_selection_enhanced' if output_dir is None else Path(output_dir)

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

    # Run enhanced evaluation
    results, split_results = run_enhanced_evaluation(
        X, y, sample_ids, strategies,
        n_repeats=n_repeats,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING STRATEGIES")
    print("=" * 80)

    comparison_df = compare_strategies_enhanced(results)
    best_strategies = identify_best_strategies_enhanced(comparison_df)

    # Print summary
    print("\n📊 RANKING (by Test RMSE with 95% CI):")
    summary_cols = ['strategy', 'category', 'test_rmse', 'test_rmse_ci_lower',
                    'test_rmse_ci_upper', 'test_r2', 'spectral_coverage']
    print(comparison_df[summary_cols].to_string())

    print("\n🏆 BEST STRATEGIES:")
    for criterion, info in best_strategies.items():
        val_str = f"{info.get('value', ''):.4f}" if 'value' in info else ''
        ci_str = info.get('ci', '')
        print(f"  {criterion}: {info['strategy']} {val_str} {ci_str}")

    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (Top 5)")
    print("=" * 80)
    stat_df = compute_statistical_tests_enhanced(results[:5])
    if not stat_df.empty:
        print(stat_df[['strategy_1', 'strategy_2', 'rmse_diff',
                       'cohens_d', 't_pvalue', 't_significant']].to_string())

    # Save results
    save_enhanced_results(results, comparison_df, best_strategies, output_path)

    # Export best split data
    print("\n" + "=" * 80)
    print("EXPORTING BEST SPLIT DATA")
    print("=" * 80)

    best_strategy_key = comparison_df.iloc[0]['strategy_key']
    best_result = results[0]  # Already sorted by test_rmse ascending
    best_split = split_results[best_strategy_key]

    export_best_split_regression(
        X, y, metadata,
        best_result, best_split, sample_ids,
        output_path
    )

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Comprehensive comparison
    fig = plot_comparison_enhanced(
        comparison_df, results,
        save_path=str(figures_dir / 'comparison_comprehensive.png')
    )
    plt.close(fig)
    print("  ✓ Comprehensive comparison plot")

    # 2. Model comparison
    fig = plot_model_comparison(
        results, comparison_df,
        save_path=str(figures_dir / 'model_comparison.png')
    )
    plt.close(fig)
    print("  ✓ Model comparison plot")

    # 3. Representativeness metrics
    fig = plot_representativeness(
        results, comparison_df,
        save_path=str(figures_dir / 'representativeness.png')
    )
    plt.close(fig)
    print("  ✓ Representativeness metrics plot")

    # 4. Bootstrap confidence intervals
    fig = plot_bootstrap_confidence(
        results, comparison_df, n_top=10,
        save_path=str(figures_dir / 'bootstrap_confidence.png')
    )
    plt.close(fig)
    print("  ✓ Bootstrap confidence interval plot")

    # 5. CV distribution
    fig = plot_cv_distribution_enhanced(
        results, comparison_df,
        save_path=str(figures_dir / 'cv_distribution.png')
    )
    plt.close(fig)
    print("  ✓ CV distribution boxplot")

    # 6. Predictions
    fig = plot_predictions_enhanced(
        results, n_top=6,
        save_path=str(figures_dir / 'predictions_top6.png')
    )
    plt.close(fig)
    print("  ✓ Predictions plot (top 6)")

    # 7. Summary report
    create_summary_report_enhanced(
        comparison_df, best_strategies, results,
        str(output_path / 'SUMMARY_REPORT_ENHANCED.txt')
    )

    # Final recommendation
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATION")
    print("=" * 80)
    best = comparison_df.iloc[0]
    print(f"\nRecommended Strategy: {best['strategy']}")
    print(f"  Category: {best['category']}")
    print(f"  Test RMSE: {best['test_rmse']:.4f} [{best['test_rmse_ci_lower']:.4f}, {best['test_rmse_ci_upper']:.4f}]")
    print(f"  Test R²: {best['test_r2']:.4f} [{best['test_r2_ci_lower']:.4f}, {best['test_r2_ci_upper']:.4f}]")
    print(f"  CV RMSE: {best['cv_rmse_mean']:.4f} ± {best['cv_rmse_std']:.4f}")
    print(f"  Generalization Gap: {best['generalization_gap']:+.4f}")
    print(f"  Spectral Coverage: {best['spectral_coverage']*100:.1f}%")
    print(f"  High Leverage Samples: {best['n_high_leverage']}")

    print(f"\n📁 All results saved to: {output_path}")
    print(f"📁 Best split exported to: {output_path / 'best_split_export'}")

    return {
        'best_strategy': best['strategy'],
        'best_strategy_key': best['strategy_key'],
        'test_rmse': best['test_rmse'],
        'test_rmse_ci': [best['test_rmse_ci_lower'], best['test_rmse_ci_upper']],
        'test_r2': best['test_r2'],
        'comparison_df': comparison_df,
        'results': results,
        'output_dir': str(output_path)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enhanced Splitter Selection for NIRS Spectral Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_splitter_selection_enhanced.py --data_dir ./data --n_folds 3
  python run_splitter_selection_enhanced.py --data_dir ./data --n_repeats 5 --n_bootstrap 2000

This enhanced version includes:
  • Multiple models: Ridge, ElasticNet, PLS, SVR, GBR, XGBoost, MLP
  • Repeated CV: 3 repeats with different random seeds
  • Bootstrap CIs: 95% confidence intervals for all metrics
  • Representativeness: Spectral coverage, target coverage, leverage analysis

Expected runtime: 30-60 minutes for typical NIRS datasets.
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
        help='Output directory (default: data_dir/splitter_selection_enhanced)'
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
        '--n_repeats', '-rp',
        type=int,
        default=3,
        help='Number of CV repetitions (default: 3)'
    )
    parser.add_argument(
        '--n_bootstrap', '-b',
        type=int,
        default=1000,
        help='Number of bootstrap samples for CIs (default: 1000)'
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
        n_repeats=args.n_repeats,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        verbose=not args.quiet
    )
