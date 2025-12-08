"""
Classification Splitter Selection Runner
=========================================

Script for comprehensive evaluation of splitting strategies for classification tasks.
Uses the whole dataset (train + test) from classification folders and evaluates
different splitting strategies using StratifiedGroupKFold cross-validation.

Key features:
1. Loads full dataset by merging train/test files
2. Multiple classification models (LogisticRegression, SVC, RF, XGBoost, MLP)
3. StratifiedGroupKFold CV (groups = sample IDs to avoid repetition leakage)
4. Bootstrap confidence intervals for all metrics
5. Exports best split data (X_train, X_test, M_train, M_test, Y_train, Y_test)

Usage:
    python run_splitter_selection_classification.py --data_dir nitro_classif/Digestibility_custom5
    # For classification data
python run_splitter_selection_classification.py --data_dir nitro_classif/Digestibility_custom5

# For regression data (now also exports best split)
python run_splitter_selection_enhanced.py --data_dir nitro_regression_merged/Digestibility_0.8

Expected evaluation time: 15-30 minutes for typical NIRS datasets
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from splitter_strategies import (
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
    KMedoidsSplitter,
    SplitResult
)
from splitter_evaluation_classification import (
    evaluate_classification_strategy,
    compare_classification_strategies,
    identify_best_classification_strategies,
    ClassificationStrategyResult,
    get_classification_model_suite
)

warnings.filterwarnings('ignore')


def load_classification_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load X, Y, M classification data by merging train and test files.

    Expected files in data_dir:
        - Xtrain.csv, Xtest.csv (spectra, semicolon-separated)
        - Ytrain.csv, Ytest.csv (target class)
        - Mtrain.csv, Mtest.csv (metadata, semicolon-separated)

    Args:
        data_dir: Directory containing the data files

    Returns:
        X: Spectra array (n_samples, n_features)
        y: Target classes (n_samples,)
        sample_ids: Sample IDs (n_samples,)
        metadata: Full metadata DataFrame
    """
    # Load training data
    X_train = pd.read_csv(data_dir / 'Xtrain.csv', sep=';').values.astype(np.float32)
    Y_train = pd.read_csv(data_dir / 'Ytrain.csv').iloc[:, 0].values.astype(np.int32)
    M_train = pd.read_csv(data_dir / 'Mtrain.csv', sep=';')
    M_train['original_split'] = 'train'

    # Load test data
    X_test = pd.read_csv(data_dir / 'Xtest.csv', sep=';').values.astype(np.float32)
    Y_test = pd.read_csv(data_dir / 'Ytest.csv').iloc[:, 0].values.astype(np.int32)
    M_test = pd.read_csv(data_dir / 'Mtest.csv', sep=';')
    M_test['original_split'] = 'test'

    # Merge all data
    X = np.vstack([X_train, X_test])
    y = np.concatenate([Y_train, Y_test])
    metadata = pd.concat([M_train, M_test], ignore_index=True)

    # Get sample IDs (for grouping repetitions)
    sample_ids = metadata['ID'].values

    print(f"Data loaded from: {data_dir}")
    print(f"  Spectra shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Classes: {np.unique(y)} ({len(np.unique(y))} classes)")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Unique samples (IDs): {len(np.unique(sample_ids))}")
    print(f"  Repetitions per sample: {len(y) // len(np.unique(sample_ids)):.1f} (avg)")
    print(f"  Original train: {len(Y_train)}, test: {len(Y_test)}")

    return X, y, sample_ids, metadata


def get_classification_strategies(
    test_size: float = 0.2,
    n_folds: int = 3,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Get all configured splitting strategies for classification.

    Note: For classification, we adapt strategies that were designed for regression
    by using class labels instead of continuous target bins.
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
                n_bins=5  # Will naturally use class labels
            ),
            'name': 'Target Stratified',
            'category': 'Target-Based',
            'description': 'Stratified by target class'
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
            'description': 'SPXY sampling from nirs4all library'
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


def run_classification_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    strategies: Dict[str, Dict[str, Any]],
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    verbose: bool = True
) -> Tuple[list, Dict[str, SplitResult]]:
    """
    Run evaluation on all strategies for classification.
    """
    results = []
    split_results = {}

    n_classes = len(np.unique(y))
    models = get_classification_model_suite(n_classes)
    model_names = list(models.keys())

    print("\n" + "=" * 80)
    print("CLASSIFICATION SPLITTER SELECTION EVALUATION")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   Models: {', '.join(model_names)}")
    print(f"   CV Repeats: {n_repeats} (StratifiedGroupKFold)")
    print(f"   Bootstrap Samples: {n_bootstrap}")
    print(f"   Strategies to evaluate: {len(strategies)}")
    print(f"   Classes: {n_classes}")

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

        # Enhanced evaluation for classification
        result = evaluate_classification_strategy(
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


def export_best_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    best_result: ClassificationStrategyResult,
    best_split: SplitResult,
    sample_ids: np.ndarray,
    output_dir: Path
) -> None:
    """
    Export the best split data to CSV files.

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
        'n_classes': best_result.n_classes,
        'test_accuracy': best_result.test_accuracy,
        'test_accuracy_ci': [
            best_result.bootstrap_metrics.accuracy_ci_lower,
            best_result.bootstrap_metrics.accuracy_ci_upper
        ],
        'test_f1_macro': best_result.test_f1_macro,
        'test_f1_macro_ci': [
            best_result.bootstrap_metrics.f1_macro_ci_lower,
            best_result.bootstrap_metrics.f1_macro_ci_upper
        ],
        'test_balanced_accuracy': best_result.test_balanced_accuracy,
        'cv_accuracy_mean': best_result.cv_accuracy_mean,
        'cv_accuracy_std': best_result.cv_accuracy_std,
        'generalization_gap': best_result.generalization_gap,
        'spectral_coverage': best_result.representativeness.spectral_coverage,
        'class_balance_similarity': best_result.representativeness.class_balance_similarity,
        'train_class_distribution': best_result.representativeness.train_class_distribution,
        'test_class_distribution': best_result.representativeness.test_class_distribution,
        'strategy_info': best_result.strategy_info
    }

    with open(export_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2, default=str)

    # Export confusion matrix
    np.savetxt(export_dir / 'confusion_matrix.csv', best_result.confusion_matrix,
               delimiter=',', fmt='%d')

    print(f"\n📁 Best split exported to: {export_dir}")
    print(f"   - X_train.csv: {X_train_df.shape}")
    print(f"   - X_test.csv: {X_test_df.shape}")
    print(f"   - Y_train.csv: {Y_train_df.shape}")
    print(f"   - Y_test.csv: {Y_test_df.shape}")
    print(f"   - M_train.csv: {M_train.shape}")
    print(f"   - M_test.csv: {M_test.shape}")
    print(f"   - fold_assignments.csv")
    print(f"   - split_info.json")
    print(f"   - confusion_matrix.csv")


def save_classification_results(
    results: list,
    comparison_df: pd.DataFrame,
    best_strategies: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Save classification results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison CSV
    comparison_df.to_csv(output_dir / 'strategy_comparison_classification.csv', index=False)

    # Save best strategies JSON
    with open(output_dir / 'best_strategies_classification.json', 'w') as f:
        json.dump(best_strategies, f, indent=2, default=str)

    # Save detailed results
    detailed = []
    for r in results:
        detailed.append({
            'strategy_key': r.strategy_key,
            'strategy_name': r.strategy_name,
            'category': r.category,
            'test_accuracy': r.test_accuracy,
            'test_accuracy_ci': [
                r.bootstrap_metrics.accuracy_ci_lower,
                r.bootstrap_metrics.accuracy_ci_upper
            ],
            'test_f1_macro': r.test_f1_macro,
            'test_f1_macro_ci': [
                r.bootstrap_metrics.f1_macro_ci_lower,
                r.bootstrap_metrics.f1_macro_ci_upper
            ],
            'test_f1_weighted': r.test_f1_weighted,
            'test_precision_macro': r.test_precision_macro,
            'test_recall_macro': r.test_recall_macro,
            'test_balanced_accuracy': r.test_balanced_accuracy,
            'cv_accuracy_mean': r.cv_accuracy_mean,
            'cv_accuracy_std': r.cv_accuracy_std,
            'cv_f1_macro_mean': r.cv_f1_macro_mean,
            'cv_f1_macro_std': r.cv_f1_macro_std,
            'generalization_gap': r.generalization_gap,
            'n_test': r.n_test,
            'n_classes': r.n_classes,
            'n_repeats': r.n_repeats,
            'n_folds': r.n_folds,
            'model_results': r.model_test_results,
            'representativeness': {
                'spectral_coverage': r.representativeness.spectral_coverage,
                'class_balance_similarity': r.representativeness.class_balance_similarity,
                'train_class_distribution': r.representativeness.train_class_distribution,
                'test_class_distribution': r.representativeness.test_class_distribution,
                'leverage_mean': r.representativeness.leverage_mean,
                'leverage_max': r.representativeness.leverage_max,
                'n_high_leverage': r.representativeness.n_high_leverage,
                'hotelling_t2_mean': r.representativeness.hotelling_t2_mean
            },
            'confusion_matrix': r.confusion_matrix.tolist(),
            'total_time': r.total_time,
            'strategy_info': r.strategy_info
        })

    with open(output_dir / 'detailed_results_classification.json', 'w') as f:
        json.dump(detailed, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


def plot_classification_comparison(
    comparison_df: pd.DataFrame,
    results: list,
    save_path: str
) -> plt.Figure:
    """Create comparison plot for classification metrics."""
    # Color scheme by category
    CATEGORY_COLORS = {
        'Baseline': '#7F8C8D',
        'Target-Based': '#2E86AB',
        'Spectral-Based': '#16A085',
        'Hybrid': '#F18F01',
        'Robustness': '#C73E1D',
        'Chemometrics': '#A23B72',
        'Stratified': '#5D5D5D',
        'Clustering': '#8E44AD'
    }

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Classification Splitter Strategy Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)
    colors = [CATEGORY_COLORS.get(cat, '#666666') for cat in comparison_df['category']]
    n_strategies = len(comparison_df)
    y_pos = range(n_strategies)

    # 1. Test Accuracy with CIs
    ax = fig.add_subplot(gs[0, 0])
    xerr_lower = comparison_df['test_accuracy'] - comparison_df['test_accuracy_ci_lower']
    xerr_upper = comparison_df['test_accuracy_ci_upper'] - comparison_df['test_accuracy']
    ax.barh(y_pos, comparison_df['test_accuracy'], xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Test Accuracy (95% CI)', fontweight='bold')
    ax.set_title('Test Accuracy', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2. Test F1 Macro with CIs
    ax = fig.add_subplot(gs[0, 1])
    xerr_lower = comparison_df['test_f1_macro'] - comparison_df['test_f1_macro_ci_lower']
    xerr_upper = comparison_df['test_f1_macro_ci_upper'] - comparison_df['test_f1_macro']
    ax.barh(y_pos, comparison_df['test_f1_macro'], xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Test F1 Macro (95% CI)', fontweight='bold')
    ax.set_title('Test F1 Score (Macro)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. CV Accuracy with std
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(y_pos, comparison_df['cv_accuracy_mean'],
            xerr=comparison_df['cv_accuracy_std'], color=colors, alpha=0.7,
            edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('CV Accuracy (mean ± std)', fontweight='bold')
    ax.set_title('Cross-Validation Accuracy', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Generalization Gap
    ax = fig.add_subplot(gs[0, 3])
    gap_colors = ['#27ae60' if x > 0 else '#e74c3c' for x in comparison_df['generalization_gap']]
    ax.barh(y_pos, comparison_df['generalization_gap'],
            color=gap_colors, alpha=0.6, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Gap (Test - CV)', fontweight='bold')
    ax.set_title('Generalization Gap', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 5. Spectral Coverage
    ax = fig.add_subplot(gs[1, 0])
    ax.barh(y_pos, comparison_df['spectral_coverage'] * 100,
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Coverage (%)', fontweight='bold')
    ax.set_title('Spectral Coverage', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 100])

    # 6. Class Balance Similarity
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(y_pos, comparison_df['class_balance_similarity'] * 100,
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Similarity (%)', fontweight='bold')
    ax.set_title('Class Balance Similarity', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 100])

    # 7. Balanced Accuracy
    ax = fig.add_subplot(gs[1, 2])
    ax.barh(y_pos, comparison_df['test_balanced_accuracy'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Balanced Accuracy', fontweight='bold')
    ax.set_title('Test Balanced Accuracy', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 8. CV Stability
    ax = fig.add_subplot(gs[1, 3])
    ax.barh(y_pos, comparison_df['cv_stability'] * 100,
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('CV Coefficient (%)', fontweight='bold')
    ax.set_title('Fold Stability (lower=better)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrices(
    results: list,
    save_path: str,
    n_top: int = 6
) -> plt.Figure:
    """Plot confusion matrices for top strategies."""
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda r: r.test_accuracy, reverse=True)[:n_top]

    n_cols = 3
    n_rows = (n_top + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle('Confusion Matrices (Top Strategies)', fontsize=14, fontweight='bold')

    axes = axes.flatten() if n_top > 1 else [axes]

    for idx, result in enumerate(sorted_results):
        ax = axes[idx]
        cm = result.confusion_matrix
        n_classes = result.n_classes

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"{result.strategy_name}\nAcc={result.test_accuracy:.3f}",
                     fontsize=10)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))

    # Hide empty subplots
    for idx in range(len(sorted_results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_classification_summary_report(
    comparison_df: pd.DataFrame,
    best_strategies: Dict[str, Dict[str, Any]],
    results: list,
    output_path: str
) -> None:
    """Create a text summary report for classification."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION SPLITTER SELECTION - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("BEST STRATEGIES BY CRITERION:\n")
        f.write("-" * 40 + "\n")
        for criterion, info in best_strategies.items():
            val = info.get('value', '')
            ci = info.get('ci', '')
            f.write(f"  {criterion:20s}: {info['strategy']:30s} {val:.4f} {ci}\n")

        f.write("\n\nFULL RANKING (by Test Accuracy):\n")
        f.write("-" * 80 + "\n")

        for idx, row in comparison_df.iterrows():
            f.write(f"\n{row['strategy']} ({row['category']})\n")
            f.write(f"  Test Accuracy: {row['test_accuracy']:.4f} "
                    f"[{row['test_accuracy_ci_lower']:.4f}, {row['test_accuracy_ci_upper']:.4f}]\n")
            f.write(f"  Test F1 Macro: {row['test_f1_macro']:.4f} "
                    f"[{row['test_f1_macro_ci_lower']:.4f}, {row['test_f1_macro_ci_upper']:.4f}]\n")
            f.write(f"  CV Accuracy: {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}\n")
            f.write(f"  Generalization Gap: {row['generalization_gap']:+.4f}\n")
            f.write(f"  Spectral Coverage: {row['spectral_coverage']:.2%}\n")
            f.write(f"  Class Balance Similarity: {row['class_balance_similarity']:.2%}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main(
    data_dir: str,
    output_dir: str = None,
    test_size: float = 0.2,
    n_folds: int = 3,
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main function for classification splitter selection.

    Args:
        data_dir: Directory containing Xtrain.csv, Ytrain.csv, Mtrain.csv,
                  Xtest.csv, Ytest.csv, Mtest.csv
        output_dir: Directory for outputs (default: data_dir/splitter_selection_classification)
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
    if output_dir is None:
        output_path = data_path / 'splitter_selection_classification'
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("LOADING CLASSIFICATION DATA")
    print("=" * 80)
    X, y, sample_ids, metadata = load_classification_data(data_path)

    # Get strategies
    strategies = get_classification_strategies(
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )

    print(f"\n{len(strategies)} splitting strategies configured")

    # Run evaluation
    results, split_results = run_classification_evaluation(
        X, y, sample_ids, strategies,
        n_repeats=n_repeats,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING STRATEGIES")
    print("=" * 80)

    comparison_df = compare_classification_strategies(results)
    best_strategies = identify_best_classification_strategies(comparison_df)

    # Print summary
    print("\n📊 RANKING (by Test Accuracy with 95% CI):")
    summary_cols = ['strategy', 'category', 'test_accuracy', 'test_accuracy_ci_lower',
                    'test_accuracy_ci_upper', 'test_f1_macro', 'spectral_coverage']
    print(comparison_df[summary_cols].to_string())

    print("\n🏆 BEST STRATEGIES:")
    for criterion, info in best_strategies.items():
        val_str = f"{info.get('value', ''):.4f}" if 'value' in info else ''
        ci_str = info.get('ci', '')
        print(f"  {criterion}: {info['strategy']} {val_str} {ci_str}")

    # Save results
    save_classification_results(results, comparison_df, best_strategies, output_path)

    # Export best split data
    print("\n" + "=" * 80)
    print("EXPORTING BEST SPLIT DATA")
    print("=" * 80)

    best_result = results[0]  # Already sorted by test_accuracy descending
    best_strategy_key = comparison_df.iloc[0]['strategy_key']
    best_split = split_results[best_strategy_key]

    export_best_split(
        X, y, metadata,
        best_result, best_split, sample_ids,
        output_path
    )

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Comparison plot
    fig = plot_classification_comparison(
        comparison_df, results,
        save_path=str(figures_dir / 'comparison_classification.png')
    )
    plt.close(fig)
    print("  ✓ Comparison plot")

    # 2. Confusion matrices
    fig = plot_confusion_matrices(
        results,
        save_path=str(figures_dir / 'confusion_matrices.png'),
        n_top=6
    )
    plt.close(fig)
    print("  ✓ Confusion matrices")

    # 3. Summary report
    create_classification_summary_report(
        comparison_df, best_strategies, results,
        str(output_path / 'SUMMARY_REPORT_CLASSIFICATION.txt')
    )
    print("  ✓ Summary report")

    # Final recommendation
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATION")
    print("=" * 80)
    best = comparison_df.iloc[0]
    print(f"\nRecommended Strategy: {best['strategy']}")
    print(f"  Category: {best['category']}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f} [{best['test_accuracy_ci_lower']:.4f}, {best['test_accuracy_ci_upper']:.4f}]")
    print(f"  Test F1 (macro): {best['test_f1_macro']:.4f}")
    print(f"  Test Balanced Accuracy: {best['test_balanced_accuracy']:.4f}")
    print(f"  CV Accuracy: {best['cv_accuracy_mean']:.4f} ± {best['cv_accuracy_std']:.4f}")
    print(f"  Generalization Gap: {best['generalization_gap']:+.4f}")
    print(f"  Spectral Coverage: {best['spectral_coverage']*100:.1f}%")
    print(f"  Class Balance Similarity: {best['class_balance_similarity']*100:.1f}%")

    print(f"\n📁 All results saved to: {output_path}")
    print(f"📁 Best split exported to: {output_path / 'best_split_export'}")

    return {
        'best_strategy': best['strategy'],
        'best_strategy_key': best['strategy_key'],
        'test_accuracy': best['test_accuracy'],
        'test_accuracy_ci': [best['test_accuracy_ci_lower'], best['test_accuracy_ci_upper']],
        'test_f1_macro': best['test_f1_macro'],
        'comparison_df': comparison_df,
        'results': results,
        'output_dir': str(output_path)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classification Splitter Selection for NIRS Spectral Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_splitter_selection_classification.py --data_dir nitro_classif/Digestibility_custom5
  python run_splitter_selection_classification.py --data_dir nitro_classif/Digestibility_custom5 --n_folds 5

This classification version includes:
  • Merges train/test files to use full dataset
  • Classification models: LogisticRegression, SVC, RF, XGBoost, MLP
  • StratifiedGroupKFold CV (groups = sample IDs to avoid repetition leakage)
  • Metrics: Accuracy, F1 (macro/weighted), Precision, Recall, Balanced Accuracy
  • Exports best split data (X_train, X_test, M_train, M_test, Y_train, Y_test)

Expected runtime: 15-30 minutes for typical NIRS datasets.
        """
    )

    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='nitro_classif/Digestibility_custom5',
        help='Directory containing Xtrain.csv, Ytrain.csv, Mtrain.csv, Xtest.csv, Ytest.csv, Mtest.csv'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: data_dir/splitter_selection_classification)'
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
