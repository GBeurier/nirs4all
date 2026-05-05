"""
Enhanced Splitter Visualization Module
======================================

This module provides enhanced visualization functions for the comprehensive
splitting strategy evaluation, including:
- Multi-model performance comparison
- Bootstrap confidence interval plots
- Representativeness metric visualizations
- Model ensemble analysis
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from splitter_evaluation_enhanced import EnhancedStrategyResult

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

# Model colors
MODEL_COLORS = {
    'ridge': '#3498db',
    'elasticnet': '#2980b9',
    'pls': '#1abc9c',
    'svr': '#9b59b6',
    'gbr': '#27ae60',
    'xgboost': '#e74c3c',
    'mlp': '#f39c12'
}

def plot_comparison_enhanced(
    comparison_df: pd.DataFrame,
    results: list[EnhancedStrategyResult],
    save_path: str | None = None,
    figsize: tuple[int, int] = (20, 16)
) -> plt.Figure:
    """
    Create comprehensive comparison plot with all enhanced metrics.

    Args:
        comparison_df: Enhanced comparison DataFrame
        results: List of EnhancedStrategyResult objects
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Enhanced Splitting Strategy Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Get colors
    colors = [CATEGORY_COLORS.get(cat, '#666666')
              for cat in comparison_df['category']]

    n_strategies = len(comparison_df)

    # 1. Test RMSE with Bootstrap CIs
    ax = fig.add_subplot(gs[0, 0])
    y_pos = range(n_strategies)

    xerr_lower = comparison_df['test_rmse'] - comparison_df['test_rmse_ci_lower']
    xerr_upper = comparison_df['test_rmse_ci_upper'] - comparison_df['test_rmse']

    ax.barh(y_pos, comparison_df['test_rmse'], xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Test RMSE (95% CI)', fontweight='bold')
    ax.set_title('Test Performance', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2. Test R² with Bootstrap CIs
    ax = fig.add_subplot(gs[0, 1])

    xerr_lower = comparison_df['test_r2'] - comparison_df['test_r2_ci_lower']
    xerr_upper = comparison_df['test_r2_ci_upper'] - comparison_df['test_r2']

    ax.barh(y_pos, comparison_df['test_r2'], xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Test R² (95% CI)', fontweight='bold')
    ax.set_title('Test R² Score', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. CV RMSE with error bars
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(y_pos, comparison_df['cv_rmse_mean'],
            xerr=comparison_df['cv_rmse_std'], color=colors, alpha=0.7,
            edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('CV RMSE (mean ± std)', fontweight='bold')
    ax.set_title('Cross-Validation Performance', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Generalization Gap
    ax = fig.add_subplot(gs[0, 3])
    gap_colors = ['#27ae60' if x < 0 else '#e74c3c'
                  for x in comparison_df['generalization_gap']]
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
    ax.set_title('Spectral Coverage (higher=better)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 100])

    # 6. Target Distribution (Wasserstein)
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(y_pos, comparison_df['target_wasserstein'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Wasserstein Distance', fontweight='bold')
    ax.set_title('Target Distribution Gap (lower=better)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 7. Leverage Analysis
    ax = fig.add_subplot(gs[1, 2])
    ax.barh(y_pos, comparison_df['n_high_leverage'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Count', fontweight='bold')
    ax.set_title('High Leverage Samples (lower=better)', fontweight='bold')
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

    # 9. Per-Model Performance Heatmap
    ax = fig.add_subplot(gs[2, :2])

    # Extract model RMSE columns
    model_cols = [c for c in comparison_df.columns if c.endswith('_rmse')
                  and c not in ['test_rmse', 'cv_rmse_mean', 'cv_rmse_std',
                               'test_rmse_ci_lower', 'test_rmse_ci_upper']]

    if model_cols:
        model_data = comparison_df[model_cols].values
        model_names = [c.replace('_rmse', '').upper() for c in model_cols]

        im = ax.imshow(model_data, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_strategies))
        ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
        ax.set_title('Per-Model Test RMSE (lower=better)', fontweight='bold')

        # Add values
        for i in range(n_strategies):
            for j in range(len(model_names)):
                val = model_data[i, j]
                color = 'white' if val > model_data.mean() else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label='RMSE')

    # 10. Training Time
    ax = fig.add_subplot(gs[2, 2])
    ax.barh(y_pos, comparison_df['total_time_sec'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_title('Total Evaluation Time', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 11. Hotelling's T²
    ax = fig.add_subplot(gs[2, 3])
    ax.barh(y_pos, comparison_df['hotelling_t2_mean'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['strategy'], fontsize=8)
    ax.set_xlabel('Mean T²', fontweight='bold')
    ax.set_title('Hotelling T² (lower=less extrapolation)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add legend at top
    legend_elements = [mpatches.Patch(facecolor=color, label=cat, alpha=0.7,
                                       edgecolor='black')
                       for cat, color in CATEGORY_COLORS.items()
                       if cat in comparison_df['category'].values]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 0.995), fontsize=10, frameon=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_model_comparison(
    results: list[EnhancedStrategyResult],
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Compare individual model performance across strategies.

    Args:
        results: List of EnhancedStrategyResult
        comparison_df: Comparison DataFrame
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Performance Comparison Across Strategies',
                 fontsize=14, fontweight='bold')

    # Get model columns
    model_cols = [c for c in comparison_df.columns if c.endswith('_rmse')
                  and c not in ['test_rmse', 'cv_rmse_mean', 'cv_rmse_std',
                               'test_rmse_ci_lower', 'test_rmse_ci_upper']]
    model_names = [c.replace('_rmse', '') for c in model_cols]

    # 1. Line plot: RMSE by model across strategies
    ax = axes[0, 0]
    strategies = comparison_df['strategy'].values
    x = range(len(strategies))

    for model_name in model_names:
        col = f'{model_name}_rmse'
        if col in comparison_df.columns:
            color = MODEL_COLORS.get(model_name, '#666666')
            ax.plot(x, comparison_df[col], 'o-', label=model_name.upper(),
                   color=color, linewidth=2, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('Model RMSE by Strategy', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Box plot: Model RMSE distribution
    ax = axes[0, 1]
    model_data = []
    model_labels = []
    model_colors_list = []

    for model_name in model_names:
        col = f'{model_name}_rmse'
        if col in comparison_df.columns:
            model_data.append(comparison_df[col].values)
            model_labels.append(model_name.upper())
            model_colors_list.append(MODEL_COLORS.get(model_name, '#666666'))

    bp = ax.boxplot(model_data, labels=model_labels, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], model_colors_list, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('RMSE Distribution by Model', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 3. Best model per strategy
    ax = axes[1, 0]
    best_models = []
    best_rmses = []

    for _idx, row in comparison_df.iterrows():
        best_rmse = float('inf')
        best_model = None
        for model_name in model_names:
            col = f'{model_name}_rmse'
            if col in row and row[col] < best_rmse:
                best_rmse = row[col]
                best_model = model_name
        best_models.append(best_model)
        best_rmses.append(best_rmse)

    colors = [MODEL_COLORS.get(m, '#666666') for m in best_models]
    ax.barh(range(len(strategies)), best_rmses, color=colors,
            alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_xlabel('Best Model RMSE', fontweight='bold')
    ax.set_title('Best Model per Strategy', fontweight='bold')
    ax.invert_yaxis()

    # Add model labels
    for i, (model, rmse) in enumerate(zip(best_models, best_rmses, strict=False)):
        ax.text(rmse + 0.02, i, model.upper(), va='center', fontsize=8)

    ax.grid(axis='x', alpha=0.3)

    # 4. Ensemble vs Best Model
    ax = axes[1, 1]
    ensemble_rmse = comparison_df['test_rmse'].values

    width = 0.35
    x = np.arange(len(strategies))

    ax.bar(x - width/2, best_rmses, width, label='Best Single Model',
           color='#3498db', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, ensemble_rmse, width, label='Ensemble',
           color='#e74c3c', alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('Ensemble vs Best Single Model', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_representativeness(
    results: list[EnhancedStrategyResult],
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Detailed representativeness metrics visualization.

    Args:
        results: List of EnhancedStrategyResult
        comparison_df: Comparison DataFrame
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Split Representativeness Analysis',
                 fontsize=14, fontweight='bold')

    strategies = comparison_df['strategy'].values
    colors = [CATEGORY_COLORS.get(cat, '#666666')
              for cat in comparison_df['category']]
    n = len(strategies)

    # 1. Spectral Coverage
    ax = axes[0, 0]
    ax.barh(range(n), comparison_df['spectral_coverage'] * 100,
            color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(80, color='green', linestyle='--', linewidth=2, label='Good threshold (80%)')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Spectral Coverage (%)', fontweight='bold')
    ax.set_title('Test Spectral Space Coverage by Train', fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 2. Target Wasserstein Distance
    ax = axes[0, 1]
    ax.barh(range(n), comparison_df['target_wasserstein'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Wasserstein Distance', fontweight='bold')
    ax.set_title('Target Distribution Distance (Train vs Test)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. KL Divergence
    ax = axes[0, 2]
    ax.barh(range(n), comparison_df['target_kl_divergence'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('KL Divergence', fontweight='bold')
    ax.set_title('Target KL Divergence (Test || Train)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Mean Leverage
    ax = axes[1, 0]
    ax.barh(range(n), comparison_df['leverage_mean'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Mean Leverage', fontweight='bold')
    ax.set_title('Average Test Sample Leverage', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 5. High Leverage Count
    ax = axes[1, 1]
    ax.barh(range(n), comparison_df['n_high_leverage'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Count', fontweight='bold')
    ax.set_title('High Leverage Test Samples (Extrapolation Risk)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 6. Hotelling's T²
    ax = axes[1, 2]
    ax.barh(range(n), comparison_df['hotelling_t2_mean'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Mean Hotelling T²', fontweight='bold')
    ax.set_title('Test Sample Mahalanobis Distance from Train', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, label=cat, alpha=0.7,
                                       edgecolor='black')
                       for cat, color in CATEGORY_COLORS.items()
                       if cat in comparison_df['category'].values]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 0.99), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_bootstrap_confidence(
    results: list[EnhancedStrategyResult],
    comparison_df: pd.DataFrame,
    n_top: int = 10,
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot bootstrap confidence intervals for top strategies.

    Args:
        results: List of EnhancedStrategyResult
        comparison_df: Comparison DataFrame
        n_top: Number of top strategies to show
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Bootstrap 95% Confidence Intervals (Top {n_top} Strategies)',
                 fontsize=14, fontweight='bold')

    # Limit to top n
    df = comparison_df.head(n_top)
    strategies = df['strategy'].values
    colors = [CATEGORY_COLORS.get(cat, '#666666') for cat in df['category']]
    n = len(strategies)

    # RMSE CI
    ax = axes[0]
    y_pos = np.arange(n)

    rmse_means = df['test_rmse'].values
    rmse_lower = df['test_rmse_ci_lower'].values
    rmse_upper = df['test_rmse_ci_upper'].values

    # Error bars
    xerr_lower = rmse_means - rmse_lower
    xerr_upper = rmse_upper - rmse_means

    ax.errorbar(rmse_means, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='o', capsize=5, capthick=2, markersize=10,
                color='black', ecolor='black', elinewidth=2)

    # Colored markers
    for i, (mean, color) in enumerate(zip(rmse_means, colors, strict=False)):
        ax.scatter(mean, i, c=color, s=150, zorder=5, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('Test RMSE', fontweight='bold', fontsize=12)
    ax.set_title('RMSE with 95% Bootstrap CI', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add CI width annotation
    for i, (lower, upper) in enumerate(zip(rmse_lower, rmse_upper, strict=False)):
        width = upper - lower
        ax.annotate(f'±{width/2:.3f}', xy=(upper + 0.01, i),
                   fontsize=8, va='center')

    # R² CI
    ax = axes[1]

    r2_means = df['test_r2'].values
    r2_lower = df['test_r2_ci_lower'].values
    r2_upper = df['test_r2_ci_upper'].values

    xerr_lower = r2_means - r2_lower
    xerr_upper = r2_upper - r2_means

    ax.errorbar(r2_means, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='o', capsize=5, capthick=2, markersize=10,
                color='black', ecolor='black', elinewidth=2)

    for i, (mean, color) in enumerate(zip(r2_means, colors, strict=False)):
        ax.scatter(mean, i, c=color, s=150, zorder=5, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('Test R²', fontweight='bold', fontsize=12)
    ax.set_title('R² with 95% Bootstrap CI', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_cv_distribution_enhanced(
    results: list[EnhancedStrategyResult],
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Enhanced CV distribution plot showing all repeats.

    Args:
        results: List of EnhancedStrategyResult
        comparison_df: Comparison DataFrame
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Cross-Validation Distribution (All Repeats)',
                 fontsize=14, fontweight='bold')

    # Order by test RMSE
    order = comparison_df['strategy'].values
    results_dict = {r.strategy_name: r for r in results}

    # 1. CV RMSE Boxplot
    ax = axes[0]
    data = []
    colors = []

    for strategy in order:
        result = results_dict.get(strategy)
        if result:
            val_rmses = [f.ensemble_val_rmse for f in result.fold_results
                        if not np.isnan(f.ensemble_val_rmse)]
            data.append(val_rmses)
            colors.append(CATEGORY_COLORS.get(result.category, '#666666'))

    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    showmeans=True, meanline=True, notch=True)

    for patch, color in zip(bp['boxes'], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=20, c='black', zorder=3)

    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Validation RMSE', fontweight='bold')
    ax.set_title('CV RMSE Distribution (notched = 95% CI of median)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # 2. CV R² Boxplot
    ax = axes[1]
    data_r2 = []

    for strategy in order:
        result = results_dict.get(strategy)
        if result:
            val_r2s = [f.ensemble_val_r2 for f in result.fold_results
                      if not np.isnan(f.ensemble_val_r2)]
            data_r2.append(val_r2s)

    bp = ax.boxplot(data_r2, labels=order, patch_artist=True,
                    showmeans=True, meanline=True, notch=True)

    for patch, color in zip(bp['boxes'], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, d in enumerate(data_r2):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=20, c='black', zorder=3)

    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Validation R²', fontweight='bold')
    ax.set_title('CV R² Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_predictions_enhanced(
    results: list[EnhancedStrategyResult],
    n_top: int = 6,
    save_path: str | None = None,
    figsize: tuple[int, int] | None = None
) -> plt.Figure:
    """
    Enhanced prediction plots with density and residuals.

    Args:
        results: List of EnhancedStrategyResult
        n_top: Number of top strategies
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
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

        # Scatter with color by residual magnitude
        residuals = np.abs(y_pred - y_true)
        scatter = ax.scatter(y_true, y_pred, c=residuals, cmap='RdYlGn_r',
                            alpha=0.6, s=30, edgecolor='black', linewidth=0.3)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--',
                linewidth=2, label='Perfect', alpha=0.8)

        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], [p(min_val), p(max_val)], 'b-',
                alpha=0.7, linewidth=2,
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        ax.set_xlabel('Actual', fontweight='bold')
        ax.set_ylabel('Predicted', fontweight='bold')

        # Add bootstrap CI to title
        rmse_ci = f"[{result.bootstrap_metrics.rmse_ci_lower:.3f}, {result.bootstrap_metrics.rmse_ci_upper:.3f}]"
        ax.set_title(f"{result.strategy_name}\n"
                     f"RMSE: {result.test_rmse:.3f} {rmse_ci}\n"
                     f"R²: {result.test_r2:.3f}",
                     fontweight='bold', fontsize=9)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label='|Residual|', shrink=0.8)

    # Hide empty subplots
    for idx in range(n_top, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def create_summary_report_enhanced(
    comparison_df: pd.DataFrame,
    best_strategies: dict[str, dict[str, Any]],
    results: list[EnhancedStrategyResult],
    output_path: str
) -> None:
    """
    Create comprehensive text summary report.

    Args:
        comparison_df: Comparison DataFrame
        best_strategies: Best strategies by criterion
        results: List of EnhancedStrategyResult
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ENHANCED SPLITTING STRATEGY SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        if results:
            r0 = results[0]
            f.write("EVALUATION CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Models evaluated: {len(r0.model_test_results)}\n")
            f.write(f"CV repeats: {r0.n_repeats}\n")
            f.write(f"CV folds: {r0.n_folds}\n")
            f.write(f"Total CV iterations: {r0.n_repeats * r0.n_folds}\n")
            f.write("Bootstrap samples: 1000\n\n")

        f.write("BEST STRATEGIES BY CRITERION\n")
        f.write("-" * 40 + "\n\n")

        for criterion, info in best_strategies.items():
            f.write(f"• {criterion.replace('_', ' ').title()}:\n")
            f.write(f"    Strategy: {info['strategy']}\n")
            if 'value' in info:
                f.write(f"    Value: {info['value']:.4f}\n")
            if 'ci' in info:
                f.write(f"    95% CI: {info['ci']}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-MODEL PERFORMANCE (Best Strategy)\n")
        f.write("=" * 80 + "\n\n")

        best_result = next(r for r in results
                          if r.strategy_name == comparison_df.iloc[0]['strategy'])
        for model, metrics in best_result.model_test_results.items():
            f.write(f"  {model:12s}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}\n")
        f.write(f"  {'ENSEMBLE':12s}: RMSE={best_result.test_rmse:.4f}, R²={best_result.test_r2:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("REPRESENTATIVENESS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Best strategies by representativeness:\n")
        f.write(f"  • Highest spectral coverage: {best_strategies.get('spectral_coverage', {}).get('strategy', 'N/A')}\n")
        f.write(f"  • Best target coverage: {best_strategies.get('target_coverage', {}).get('strategy', 'N/A')}\n")
        f.write(f"  • Lowest extrapolation risk: {best_strategies.get('low_extrapolation', {}).get('strategy', 'N/A')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FULL RANKING (by Test RMSE)\n")
        f.write("=" * 80 + "\n\n")

        for rank, (_idx, row) in enumerate(comparison_df.iterrows(), 1):
            f.write(f"{rank}. {row['strategy']}\n")
            f.write(f"   Category: {row['category']}\n")
            f.write(f"   Test RMSE: {row['test_rmse']:.4f} [{row['test_rmse_ci_lower']:.4f}, {row['test_rmse_ci_upper']:.4f}]\n")
            f.write(f"   Test R²: {row['test_r2']:.4f} [{row['test_r2_ci_lower']:.4f}, {row['test_r2_ci_upper']:.4f}]\n")
            f.write(f"   CV RMSE: {row['cv_rmse_mean']:.4f} ± {row['cv_rmse_std']:.4f}\n")
            f.write(f"   Gen. Gap: {row['generalization_gap']:+.4f}\n")
            f.write(f"   Spectral Coverage: {row['spectral_coverage']*100:.1f}%\n")
            f.write(f"   High Leverage Samples: {row['n_high_leverage']}\n")
            f.write(f"   Evaluation Time: {row['total_time_sec']:.1f}s\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")

        best = comparison_df.iloc[0]
        f.write(f"Recommended Strategy: {best['strategy']}\n\n")
        f.write("Reasons:\n")
        f.write(f"  • Best test set performance (RMSE: {best['test_rmse']:.4f})\n")

        if best['spectral_coverage'] >= 0.8:
            f.write(f"  • Good spectral coverage ({best['spectral_coverage']*100:.1f}%)\n")

        if abs(best['generalization_gap']) < 0.05:
            f.write(f"  • Excellent generalization (gap: {best['generalization_gap']:+.4f})\n")

        if best['n_high_leverage'] <= 5:
            f.write(f"  • Low extrapolation risk ({best['n_high_leverage']} high-leverage samples)\n")

        f.write("\n")
        f.write("Note: Results based on ensemble of multiple models with repeated CV.\n")
        f.write("Bootstrap confidence intervals provide uncertainty estimates.\n")

    print(f"Enhanced report saved to: {output_path}")
