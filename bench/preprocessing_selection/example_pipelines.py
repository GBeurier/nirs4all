"""
Preprocessing Pipeline and Augmentation Selection
==================================================

This example demonstrates:
1. Stacked preprocessing pipelines (order 1-3): e.g., SNV → Detrend → SavGol
2. Feature augmentation: concatenating multiple preprocessed versions as features
3. Comprehensive visualization of results

Usage:
    python example_pipelines.py [--plots] [--full]
"""

import argparse
import sys
import os
import time
import itertools
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from copy import deepcopy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NIRS4All imports
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    MultiplicativeScatterCorrection,
    FirstDerivative,
    Haar,
    Detrend,
    Gaussian,
    IdentityTransformer,
    RobustStandardNormalVariate,
    LocalStandardNormalVariate,
)

# Selection framework imports
from selection import PreprocessingSelector, print_selection_report
from selection.metrics import evaluate_supervised, pls_score
from selection.proxy_models import evaluate_proxies


def load_sample_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load sample data from CSV files."""
    patterns = [
        ('Xcal.csv', 'Ycal.csv'),
        ('Xcal.csv.gz', 'Ycal.csv.gz'),
        ('Xtrain.csv', 'Ytrain.csv'),
    ]

    for x_file, y_file in patterns:
        x_path = os.path.join(data_path, x_file)
        if os.path.exists(x_path):
            print(f"Loading data from {x_path}...")
            y_path = os.path.join(data_path, y_file)

            x_df = pd.read_csv(x_path, header=None, sep=';')
            y_df = pd.read_csv(y_path, header=None, sep=';')

            # Check for headers
            try:
                float(x_df.iloc[0, 0])
            except (ValueError, TypeError):
                x_df = pd.read_csv(x_path, header=0, sep=';')

            try:
                float(y_df.iloc[0, 0])
            except (ValueError, TypeError):
                y_df = pd.read_csv(y_path, header=0, sep=';')

            X = x_df.values.astype(np.float64)
            y = y_df.values.astype(np.float64).ravel()

            # Ensure matching shapes
            min_samples = min(X.shape[0], y.shape[0])
            X, y = X[:min_samples], y[:min_samples]

            print(f"Loaded X: {X.shape}, y: {y.shape}")
            return X, y

    raise FileNotFoundError(f"Could not find data files in {data_path}")


def get_base_preprocessings() -> Dict[str, Any]:
    """Get base preprocessing transformers."""
    return {
        'identity': IdentityTransformer(),
        'snv': StandardNormalVariate(),
        'msc': MultiplicativeScatterCorrection(scale=False),
        'savgol': SavitzkyGolay(window_length=11, polyorder=3),
        'detrend': Detrend(),
        'd1': FirstDerivative(),
        'haar': Haar(),
        'gaussian': Gaussian(order=1, sigma=2),
    }


def apply_pipeline(X: np.ndarray, pipeline: List) -> np.ndarray:
    """
    Apply a sequence of preprocessing steps.

    Args:
        X: Input data
        pipeline: List of (name, transformer) tuples

    Returns:
        Transformed data
    """
    X_out = X.copy()
    for name, transformer in pipeline:
        try:
            # Create fresh copy of transformer to avoid state issues
            t = deepcopy(transformer)
            X_out = t.fit_transform(X_out)
        except Exception as e:
            print(f"  Warning: Pipeline step '{name}' failed: {e}")
            return None
    return X_out


def generate_stacked_pipelines(
    base_preprocessings: Dict[str, Any],
    max_depth: int = 3
) -> Dict[str, List]:
    """
    Generate stacked preprocessing pipelines of various depths.

    Args:
        base_preprocessings: Dict of {name: transformer}
        max_depth: Maximum pipeline depth (1, 2, or 3)

    Returns:
        Dict of {pipeline_name: [(name, transformer), ...]}
    """
    pipelines = {}
    names = list(base_preprocessings.keys())
    transformers = list(base_preprocessings.values())

    # Depth 1: Single preprocessings
    for name, t in base_preprocessings.items():
        pipelines[name] = [(name, t)]

    if max_depth >= 2:
        # Depth 2: Pairs (avoid same preprocessing twice)
        for i, (n1, t1) in enumerate(zip(names, transformers)):
            for j, (n2, t2) in enumerate(zip(names, transformers)):
                if i != j:
                    pipeline_name = f"{n1}>{n2}"
                    pipelines[pipeline_name] = [(n1, t1), (n2, t2)]

    if max_depth >= 3:
        # Depth 3: Triples (avoid same preprocessing in sequence)
        for i, (n1, t1) in enumerate(zip(names, transformers)):
            for j, (n2, t2) in enumerate(zip(names, transformers)):
                for k, (n3, t3) in enumerate(zip(names, transformers)):
                    if i != j and j != k:  # Adjacent can't be same
                        pipeline_name = f"{n1}>{n2}>{n3}"
                        pipelines[pipeline_name] = [(n1, t1), (n2, t2), (n3, t3)]

    return pipelines


def generate_feature_augmentations(
    base_preprocessings: Dict[str, Any],
    X: np.ndarray,
    max_augmentations: int = 3
) -> Dict[str, np.ndarray]:
    """
    Generate feature augmentation variants (stacked features).

    Args:
        base_preprocessings: Dict of {name: transformer}
        X: Original data
        max_augmentations: Maximum number of preprocessings to stack as features

    Returns:
        Dict of {augmentation_name: X_augmented}
    """
    augmentations = {}

    # First, transform all base preprocessings
    transformed = {}
    for name, t in base_preprocessings.items():
        try:
            t_copy = deepcopy(t)
            X_t = t_copy.fit_transform(X.copy())
            if X_t is not None and not np.any(np.isnan(X_t)):
                transformed[name] = X_t
        except Exception:
            pass

    names = list(transformed.keys())

    # Generate combinations of 2 preprocessings as features
    if max_augmentations >= 2:
        for combo in itertools.combinations(names, 2):
            aug_name = f"[{'+'.join(combo)}]"
            try:
                X_aug = np.hstack([transformed[n] for n in combo])
                augmentations[aug_name] = X_aug
            except Exception:
                pass

    # Generate combinations of 3 preprocessings as features
    if max_augmentations >= 3:
        for combo in itertools.combinations(names, 3):
            aug_name = f"[{'+'.join(combo)}]"
            try:
                X_aug = np.hstack([transformed[n] for n in combo])
                augmentations[aug_name] = X_aug
            except Exception:
                pass

    return augmentations


def evaluate_pipelines(
    X: np.ndarray,
    y: np.ndarray,
    pipelines: Dict[str, List],
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate all stacked pipelines.

    Args:
        X: Original data
        y: Target values
        pipelines: Dict of {name: pipeline}
        verbose: Print progress

    Returns:
        Dict of {name: results}
    """
    results = {}
    total = len(pipelines)

    for i, (name, pipeline) in enumerate(pipelines.items()):
        if verbose and i % 20 == 0:
            print(f"  Evaluating pipelines: {i}/{total}...")

        # Apply pipeline
        X_pp = apply_pipeline(X, pipeline)
        if X_pp is None:
            results[name] = {'error': 'Pipeline failed', 'depth': len(pipeline)}
            continue

        # Evaluate with PLS and proxies
        try:
            pls_result = pls_score(X_pp, y, n_components=3)
            supervised = evaluate_supervised(X_pp, y, pls_n_components=3)
            proxy = evaluate_proxies(X_pp, y, cv_folds=3)

            results[name] = {
                'depth': len(pipeline),
                'pls_r2': pls_result['pls_r2'],
                'pls_rmse': pls_result['pls_rmse'],
                'supervised_score': supervised['composite_score'],
                'proxy_score': proxy['composite_score'],
                'ridge_r2': proxy['ridge']['ridge_r2'],
                'combined_score': (supervised['composite_score'] + 2 * proxy['composite_score']) / 3
            }
        except Exception as e:
            results[name] = {'error': str(e), 'depth': len(pipeline)}

    return results


def evaluate_augmentations(
    augmentations: Dict[str, np.ndarray],
    y: np.ndarray,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate all feature augmentations.

    Args:
        augmentations: Dict of {name: X_augmented}
        y: Target values
        verbose: Print progress

    Returns:
        Dict of {name: results}
    """
    results = {}
    total = len(augmentations)

    for i, (name, X_aug) in enumerate(augmentations.items()):
        if verbose and i % 10 == 0:
            print(f"  Evaluating augmentations: {i}/{total}...")

        try:
            # Count number of stacked features
            n_features = name.count('+') + 1

            pls_result = pls_score(X_aug, y, n_components=3)
            supervised = evaluate_supervised(X_aug, y, pls_n_components=3)
            proxy = evaluate_proxies(X_aug, y, cv_folds=3)

            results[name] = {
                'n_stacked': n_features,
                'n_features': X_aug.shape[1],
                'pls_r2': pls_result['pls_r2'],
                'pls_rmse': pls_result['pls_rmse'],
                'supervised_score': supervised['composite_score'],
                'proxy_score': proxy['composite_score'],
                'ridge_r2': proxy['ridge']['ridge_r2'],
                'combined_score': (supervised['composite_score'] + 2 * proxy['composite_score']) / 3
            }
        except Exception as e:
            results[name] = {'error': str(e), 'n_stacked': name.count('+') + 1}

    return results


def plot_pipeline_results(pipeline_results: Dict, augmentation_results: Dict, save_path: str = None):
    """
    Create comprehensive visualization of results.

    Args:
        pipeline_results: Results from evaluate_pipelines
        augmentation_results: Results from evaluate_augmentations
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12))

    # =========================================================================
    # Plot 1: Top Stacked Pipelines by Depth
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)

    # Filter valid results and sort by score
    valid_pipelines = {k: v for k, v in pipeline_results.items() if 'error' not in v}

    # Group by depth
    by_depth = {1: [], 2: [], 3: []}
    for name, res in valid_pipelines.items():
        depth = res['depth']
        if depth in by_depth:
            by_depth[depth].append((name, res['combined_score']))

    # Sort each group and take top 5
    colors = {1: '#2ecc71', 2: '#3498db', 3: '#9b59b6'}
    labels_added = set()

    y_pos = 0
    y_ticks = []
    y_labels = []

    for depth in [1, 2, 3]:
        sorted_items = sorted(by_depth[depth], key=lambda x: x[1], reverse=True)[:5]
        for name, score in sorted_items:
            label = f'Depth {depth}' if depth not in labels_added else None
            ax1.barh(y_pos, score, color=colors[depth], label=label, alpha=0.8)
            labels_added.add(depth)
            y_ticks.append(y_pos)
            y_labels.append(name)
            y_pos += 1
        y_pos += 0.5  # Gap between groups

    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)
    ax1.set_xlabel('Combined Score')
    ax1.set_title('Top 5 Stacked Pipelines by Depth')
    ax1.legend(loc='lower right')
    ax1.invert_yaxis()

    # =========================================================================
    # Plot 2: Pipeline Score vs Depth (Box Plot)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    scores_by_depth = {1: [], 2: [], 3: []}
    for name, res in valid_pipelines.items():
        if 'combined_score' in res:
            scores_by_depth[res['depth']].append(res['combined_score'])

    bp_data = [scores_by_depth[1], scores_by_depth[2], scores_by_depth[3]]
    bp = ax2.boxplot(bp_data, labels=['Depth 1\n(single)', 'Depth 2\n(pair)', 'Depth 3\n(triple)'],
                     patch_artist=True)

    for patch, color in zip(bp['boxes'], [colors[1], colors[2], colors[3]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Combined Score')
    ax2.set_title('Score Distribution by Pipeline Depth')
    ax2.grid(axis='y', alpha=0.3)

    # Add counts
    for i, depth in enumerate([1, 2, 3]):
        count = len(scores_by_depth[depth])
        ax2.text(i + 1, ax2.get_ylim()[1], f'n={count}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # Plot 3: Feature Augmentation Results
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    valid_augs = {k: v for k, v in augmentation_results.items() if 'error' not in v}

    # Sort by combined score
    sorted_augs = sorted(valid_augs.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:15]

    aug_names = [item[0] for item in sorted_augs]
    aug_scores = [item[1]['combined_score'] for item in sorted_augs]
    aug_n_stacked = [item[1]['n_stacked'] for item in sorted_augs]

    # Color by number of stacked features
    aug_colors = ['#e74c3c' if n == 2 else '#f39c12' for n in aug_n_stacked]

    bars = ax3.barh(range(len(aug_names)), aug_scores, color=aug_colors, alpha=0.8)

    ax3.set_yticks(range(len(aug_names)))
    ax3.set_yticklabels(aug_names, fontsize=8)
    ax3.set_xlabel('Combined Score')
    ax3.set_title('Top 15 Feature Augmentations')
    ax3.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='2 stacked'),
                       Patch(facecolor='#f39c12', alpha=0.8, label='3 stacked')]
    ax3.legend(handles=legend_elements, loc='lower right')

    # =========================================================================
    # Plot 4: Comparison - Single vs Stacked vs Augmented
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    # Best scores from each category
    best_single = max([v['combined_score'] for v in valid_pipelines.values() if v['depth'] == 1], default=0)
    best_depth2 = max([v['combined_score'] for v in valid_pipelines.values() if v['depth'] == 2], default=0)
    best_depth3 = max([v['combined_score'] for v in valid_pipelines.values() if v['depth'] == 3], default=0)
    best_aug2 = max([v['combined_score'] for v in valid_augs.values() if v['n_stacked'] == 2], default=0)
    best_aug3 = max([v['combined_score'] for v in valid_augs.values() if v['n_stacked'] == 3], default=0)

    categories = ['Single PP', 'Stacked\n(depth 2)', 'Stacked\n(depth 3)',
                  'Augmented\n(2 features)', 'Augmented\n(3 features)']
    best_scores = [best_single, best_depth2, best_depth3, best_aug2, best_aug3]
    bar_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    bars = ax4.bar(categories, best_scores, color=bar_colors, alpha=0.8)

    # Add value labels
    for bar, score in zip(bars, best_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    ax4.set_ylabel('Best Combined Score')
    ax4.set_title('Best Score by Strategy')
    ax4.set_ylim(0, max(best_scores) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to '{save_path}'")

    return fig


def run_pipeline_evaluation(data_path: str = 'sample_data/regression',
                            max_depth: int = 2,
                            max_augmentations: int = 3,
                            show_plots: bool = True):
    """
    Run comprehensive pipeline and augmentation evaluation.

    Args:
        data_path: Path to data folder
        max_depth: Maximum pipeline depth (1-3)
        max_augmentations: Maximum feature augmentation size
        show_plots: Whether to display plots
    """
    print("\n" + "=" * 70)
    print("PIPELINE & AUGMENTATION EVALUATION")
    print("=" * 70)

    # Load data
    X, y = load_sample_data(data_path)

    # Get base preprocessings
    base_pp = get_base_preprocessings()
    print(f"\nBase preprocessings: {len(base_pp)}")
    for name in base_pp.keys():
        print(f"  - {name}")

    # =========================================================================
    # PART 1: Stacked Pipelines
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PART 1: Stacked Pipelines (max depth = {max_depth})")
    print("=" * 60)

    pipelines = generate_stacked_pipelines(base_pp, max_depth=max_depth)
    print(f"Generated {len(pipelines)} pipeline combinations:")
    print(f"  - Depth 1: {sum(1 for p in pipelines.values() if len(p) == 1)}")
    print(f"  - Depth 2: {sum(1 for p in pipelines.values() if len(p) == 2)}")
    if max_depth >= 3:
        print(f"  - Depth 3: {sum(1 for p in pipelines.values() if len(p) == 3)}")

    print("\nEvaluating pipelines...")
    start_time = time.time()
    pipeline_results = evaluate_pipelines(X, y, pipelines, verbose=True)
    pipeline_time = time.time() - start_time

    # Report results
    valid_results = {k: v for k, v in pipeline_results.items() if 'error' not in v}
    failed = len(pipeline_results) - len(valid_results)

    print(f"\nPipeline evaluation complete: {len(valid_results)} successful, {failed} failed")
    print(f"Time: {pipeline_time:.2f}s")

    # Top results by depth
    for depth in range(1, max_depth + 1):
        depth_results = [(k, v) for k, v in valid_results.items() if v['depth'] == depth]
        top_5 = sorted(depth_results, key=lambda x: x[1]['combined_score'], reverse=True)[:5]

        print(f"\n📊 Top 5 Depth-{depth} Pipelines:")
        for name, res in top_5:
            print(f"  {name}: score={res['combined_score']:.4f}, "
                  f"PLS_R²={res['pls_r2']:.4f}, Ridge_R²={res['ridge_r2']:.4f}")

    # =========================================================================
    # PART 2: Feature Augmentation
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PART 2: Feature Augmentation (max stacked = {max_augmentations})")
    print("=" * 60)

    augmentations = generate_feature_augmentations(base_pp, X, max_augmentations=max_augmentations)
    print(f"Generated {len(augmentations)} augmentation combinations:")
    print(f"  - 2 stacked: {sum(1 for n in augmentations.keys() if n.count('+') == 1)}")
    print(f"  - 3 stacked: {sum(1 for n in augmentations.keys() if n.count('+') == 2)}")

    print("\nEvaluating augmentations...")
    start_time = time.time()
    aug_results = evaluate_augmentations(augmentations, y, verbose=True)
    aug_time = time.time() - start_time

    valid_augs = {k: v for k, v in aug_results.items() if 'error' not in v}
    failed_augs = len(aug_results) - len(valid_augs)

    print(f"\nAugmentation evaluation complete: {len(valid_augs)} successful, {failed_augs} failed")
    print(f"Time: {aug_time:.2f}s")

    # Top results
    top_augs = sorted(valid_augs.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:10]
    print("\n📊 Top 10 Feature Augmentations:")
    for name, res in top_augs:
        print(f"  {name}: score={res['combined_score']:.4f}, "
              f"features={res['n_features']}, PLS_R²={res['pls_r2']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    # Find overall best
    all_results = []
    for name, res in valid_results.items():
        all_results.append(('pipeline', name, res['combined_score'], res))
    for name, res in valid_augs.items():
        all_results.append(('augmentation', name, res['combined_score'], res))

    all_sorted = sorted(all_results, key=lambda x: x[2], reverse=True)

    print("\n🏆 Overall Top 10:")
    for i, (type_, name, score, res) in enumerate(all_sorted[:10]):
        extra = f"depth={res['depth']}" if type_ == 'pipeline' else f"features={res['n_features']}"
        print(f"  {i+1}. [{type_}] {name}: {score:.4f} ({extra})")

    print(f"\n⏱️ Total time: {pipeline_time + aug_time:.2f}s")

    # =========================================================================
    # Visualization
    # =========================================================================
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            fig = plot_pipeline_results(pipeline_results, aug_results,
                                        save_path='selection/pipeline_results.png')
            plt.show()
        except ImportError:
            print("\nWarning: matplotlib not available for plotting")

    return {
        'pipeline_results': pipeline_results,
        'augmentation_results': aug_results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline and Augmentation Evaluation')
    parser.add_argument('--plots', action='store_true', help='Show visualization plots')
    parser.add_argument('--full', action='store_true',
                        help='Use full nitro datasets instead of sample data')
    parser.add_argument('--depth', type=int, default=2,
                        help='Maximum pipeline depth (1-3)')
    parser.add_argument('--aug', type=int, default=3,
                        help='Maximum feature augmentation size (2-3)')
    args = parser.parse_args()

    if args.full:
        data_path = 'selection/nitro_regression/Digestibility_0.8'
        if not os.path.exists(data_path):
            print(f"Error: Data path not found: {data_path}")
            data_path = 'sample_data/regression'
    else:
        data_path = 'sample_data/regression'

    run_pipeline_evaluation(
        data_path=data_path,
        max_depth=min(args.depth, 3),
        max_augmentations=min(args.aug, 3),
        show_plots=args.plots
    )
