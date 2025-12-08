"""
Splitter Visualization Module
=============================

This module provides visualization functions for comparing splitting strategies
and analyzing their performance.

Visualizations include:
- Performance comparison bar charts
- CV distribution boxplots
- Predicted vs actual scatter plots
- Generalization gap analysis
- PCA visualization of splits
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

from splitter_evaluation import StrategyResult


# Color scheme by category
CATEGORY_COLORS = {
    'Baseline': '#7F8C8D',
    'Target-Based': '#2E86AB',
    'Spectral-Based': '#16A085',
    'Hybrid': '#F18F01',
    'Robustness': '#C73E1D',
    'Chemometrics': '#A23B72'
}


def plot_comparison_comprehensive(
    comparison_df: pd.DataFrame,
    results: List[StrategyResult],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create comprehensive comparison plot with multiple metrics.

    Args:
        comparison_df: Comparison DataFrame from compare_strategies
        results: List of StrategyResult objects
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Splitting Strategy Performance Comparison',
                 fontsize=16, fontweight='bold')

    # Get colors
    colors = [CATEGORY_COLORS.get(cat, '#666666')
              for cat in comparison_df['category']]

    # 1. Test RMSE comparison
    ax = axes[0, 0]
    bars = ax.barh(range(len(comparison_df)), comparison_df['test_rmse'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('Test RMSE', fontweight='bold')
    ax.set_title('Test Set Performance', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add values
    for i, v in enumerate(comparison_df['test_rmse']):
        ax.text(v + 0.05, i, f"{v:.3f}", va='center', fontsize=8)

    # 2. CV RMSE with error bars
    ax = axes[0, 1]
    ax.barh(range(len(comparison_df)), comparison_df['cv_rmse_mean'],
            xerr=comparison_df['cv_rmse_std'], color=colors, alpha=0.7,
            edgecolor='black', capsize=3)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('CV RMSE', fontweight='bold')
    ax.set_title('Cross-Validation Performance', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. CV Stability
    ax = axes[0, 2]
    ax.barh(range(len(comparison_df)), comparison_df['cv_stability'] * 100,
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('CV Coefficient (%)', fontweight='bold')
    ax.set_title('Fold Stability (lower is better)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Generalization Gap
    ax = axes[1, 0]
    gap_colors = ['#27ae60' if x < 0 else '#e74c3c'
                  for x in comparison_df['generalization_gap']]
    ax.barh(range(len(comparison_df)), comparison_df['generalization_gap'],
            color=gap_colors, alpha=0.6, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('Test - CV RMSE Gap', fontweight='bold')
    ax.set_title('Generalization Gap', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 5. Test R² comparison
    ax = axes[1, 1]
    ax.barh(range(len(comparison_df)), comparison_df['test_r2'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('Test R²', fontweight='bold')
    ax.set_title('Test Set R² Score', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1])

    # 6. Overfitting indicator
    ax = axes[1, 2]
    ax.barh(range(len(comparison_df)), comparison_df['overfitting'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['strategy'], fontsize=9)
    ax.set_xlabel('Val - Train RMSE', fontweight='bold')
    ax.set_title('Overfitting Indicator', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, label=cat, alpha=0.7,
                                       edgecolor='black')
                       for cat, color in CATEGORY_COLORS.items()
                       if cat in comparison_df['category'].values]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 0.98), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_predictions(
    results: List[StrategyResult],
    n_top: int = 6,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot predicted vs actual for top strategies.

    Args:
        results: List of StrategyResult objects
        n_top: Number of top strategies to plot
        save_path: Optional path to save figure
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure
    """
    # Sort by test RMSE
    results_sorted = sorted(results, key=lambda r: r.test_rmse)[:n_top]

    n_cols = min(3, n_top)
    n_rows = (n_top + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Predicted vs Actual: Top {n_top} Strategies',
                 fontsize=14, fontweight='bold')

    for idx, result in enumerate(results_sorted):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        y_true = result.y_test_true
        y_pred = result.y_test_pred

        # Scatter plot
        color = CATEGORY_COLORS.get(result.category, '#666666')
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color,
                   edgecolor='black', linewidth=0.3)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='Perfect')

        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], [p(min_val), p(max_val)], 'b-',
                alpha=0.5, linewidth=1.5,
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        ax.set_xlabel('Actual', fontweight='bold')
        ax.set_ylabel('Predicted', fontweight='bold')
        ax.set_title(f"{result.strategy_name}\n"
                     f"RMSE: {result.test_rmse:.3f}, R²: {result.test_r2:.3f}",
                     fontweight='bold', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n_top, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_cv_distribution(
    results: List[StrategyResult],
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot CV RMSE distribution as boxplot.

    Args:
        results: List of StrategyResult objects
        comparison_df: Comparison DataFrame
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Order by test RMSE
    order = comparison_df['strategy'].values

    # Collect data
    data = []
    colors = []
    for strategy in order:
        result = next(r for r in results if r.strategy_name == strategy)
        val_rmses = [f.val_rmse for f in result.fold_results]
        data.append(val_rmses)
        colors.append(CATEGORY_COLORS.get(result.category, '#666666'))

    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Splitting Strategy', fontweight='bold', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontweight='bold', fontsize=12)
    ax.set_title('Cross-Validation RMSE Distribution by Strategy',
                 fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=color, label=cat, alpha=0.6,
                                       edgecolor='black')
                       for cat, color in CATEGORY_COLORS.items()
                       if cat in [r.category for r in results]]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_split_pca(
    X: np.ndarray,
    sample_ids: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    fold_df: pd.DataFrame,
    strategy_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualize train/test split in PCA space.

    Args:
        X: Feature matrix
        sample_ids: Sample IDs
        train_ids: Training sample IDs
        test_ids: Test sample IDs
        fold_df: Fold assignments DataFrame
        strategy_name: Name of the strategy
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Aggregate by sample
    df = pd.DataFrame(X)
    df['id'] = sample_ids
    X_agg = df.groupby('id').mean()
    unique_ids = X_agg.index.values
    X_agg = X_agg.values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_agg)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{strategy_name}: Split Visualization in PCA Space',
                 fontsize=14, fontweight='bold')

    # Left: Train vs Test
    ax = axes[0]
    train_mask = np.isin(unique_ids, train_ids)
    test_mask = np.isin(unique_ids, test_ids)

    ax.scatter(X_pca[train_mask, 0], X_pca[train_mask, 1],
               c='#2ecc71', alpha=0.6, s=30, label='Train', edgecolor='black', linewidth=0.3)
    ax.scatter(X_pca[test_mask, 0], X_pca[test_mask, 1],
               c='#e74c3c', alpha=0.8, s=40, label='Test', edgecolor='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Train vs Test Split')
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: Fold assignments
    ax = axes[1]
    n_folds = fold_df[fold_df['split'] != 'test']['fold'].max() + 1
    fold_colors = plt.cm.Set1(np.linspace(0, 1, n_folds))

    # Plot test first (background)
    ax.scatter(X_pca[test_mask, 0], X_pca[test_mask, 1],
               c='gray', alpha=0.3, s=20, label='Test')

    # Plot each fold
    for fold_idx in range(n_folds):
        fold_ids = fold_df[fold_df['fold'] == fold_idx]['ID'].values
        fold_mask = np.isin(unique_ids, fold_ids)
        ax.scatter(X_pca[fold_mask, 0], X_pca[fold_mask, 1],
                   c=[fold_colors[fold_idx]], alpha=0.7, s=30,
                   label=f'Fold {fold_idx}', edgecolor='black', linewidth=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Fold Assignments')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_residuals(
    results: List[StrategyResult],
    n_top: int = 4,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot residual analysis for top strategies.

    Args:
        results: List of StrategyResult objects
        n_top: Number of top strategies to plot
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    results_sorted = sorted(results, key=lambda r: r.test_rmse)[:n_top]

    n_cols = 2
    n_rows = (n_top + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (12, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Residual Analysis: Top Strategies', fontsize=14, fontweight='bold')

    for idx, result in enumerate(results_sorted):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        residuals = result.y_test_pred - result.y_test_true

        # Histogram of residuals
        ax.hist(residuals, bins=30, alpha=0.7,
                color=CATEGORY_COLORS.get(result.category, '#666666'),
                edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(residuals):.3f}')

        ax.set_xlabel('Residual (Predicted - Actual)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{result.strategy_name}\n"
                     f"Std: {np.std(residuals):.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n_top, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_report(
    comparison_df: pd.DataFrame,
    best_strategies: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    Create a text summary report of the analysis.

    Args:
        comparison_df: Comparison DataFrame
        best_strategies: Dictionary of best strategies by criterion
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPLITTING STRATEGY SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("BEST STRATEGIES BY CRITERION\n")
        f.write("-" * 40 + "\n\n")

        f.write("1. Best Test Performance (lowest RMSE):\n")
        bp = best_strategies['test_performance']
        f.write(f"   Strategy: {bp['strategy']}\n")
        f.write(f"   Test RMSE: {bp['test_rmse']:.4f}\n")
        f.write(f"   Test R²: {bp['test_r2']:.4f}\n\n")

        f.write("2. Most Stable Cross-Validation:\n")
        cs = best_strategies['cv_stability']
        f.write(f"   Strategy: {cs['strategy']}\n")
        f.write(f"   CV Stability: {cs['cv_stability']:.4f}\n\n")

        f.write("3. Best Generalization:\n")
        bg = best_strategies['generalization']
        f.write(f"   Strategy: {bg['strategy']}\n")
        f.write(f"   Gap (CV→Test): {bg['generalization_gap']:+.4f}\n\n")

        f.write("4. Highest R²:\n")
        br = best_strategies['r2']
        f.write(f"   Strategy: {br['strategy']}\n")
        f.write(f"   Test R²: {br['test_r2']:.4f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FULL RANKING (by Test RMSE)\n")
        f.write("=" * 80 + "\n\n")

        for idx, row in comparison_df.iterrows():
            f.write(f"{comparison_df.index.get_loc(idx) + 1}. {row['strategy']}\n")
            f.write(f"   Category: {row['category']}\n")
            f.write(f"   Test RMSE: {row['test_rmse']:.4f}\n")
            f.write(f"   CV RMSE: {row['cv_rmse_mean']:.4f} ± {row['cv_rmse_std']:.4f}\n")
            f.write(f"   Test R²: {row['test_r2']:.4f}\n")
            f.write(f"   Gen. Gap: {row['generalization_gap']:+.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")

        best = comparison_df.iloc[0]
        f.write(f"Recommended Strategy: {best['strategy']}\n")
        f.write(f"Reason: Best test set performance with acceptable generalization\n")

    print(f"Report saved to: {output_path}")
