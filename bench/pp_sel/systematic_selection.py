"""
Systematic Preprocessing Selection
==================================

A comprehensive, systematic evaluation of preprocessing pipelines:

1. Stage 1 - Exhaustive Unsupervised Evaluation:
   - All single preprocessings (depth 1)
   - All stacked pipelines depth 2 (A → B)
   - All stacked pipelines depth 3 (A → B → C)
   - Compute unsupervised metrics for each
   - Output: CSV + visualization

2. Stage 2 - Diversity Analysis:
   - Take top N candidates
   - Compute pairwise distances (Grassmann, CKA)
   - Identify diverse preprocessing pairs for augmentation

3. Stage 3 - Proxy Model Evaluation:
   - Test top single/stacked with Ridge/KNN
   - Test diverse augmentations (2nd and 3rd order)

4. Final Ranking:
   - Combined ranking of all strategies

Usage:
    python systematic_selection.py [--depth 3] [--top 15] [--plots] [--full]
"""

import argparse
import sys
import os
import time
import warnings
from itertools import permutations, combinations
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Add parent directory to path
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
    Wavelet,
    RobustStandardNormalVariate,
)
from nirs4all.operators.transforms.nirs import AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC



# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PipelineResult:
    """Result of a pipeline evaluation."""
    name: str
    depth: int
    pipeline_type: str  # 'single', 'stacked', 'augmented'
    components: List[str]
    X_transformed: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    proxy_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0


# =============================================================================
# Unsupervised Metrics
# =============================================================================

def compute_pca_metrics(X: np.ndarray, n_components: int = 10) -> Dict[str, float]:
    """Compute PCA-based metrics."""
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_comp < 1:
        return {'variance_ratio': 0.0, 'first_component_ratio': 1.0, 'effective_dim': 0.0}

    pca = PCA(n_components=n_comp)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    first_ratio = pca.explained_variance_ratio_[0]

    # Effective dimensionality (entropy-based)
    var_ratios = pca.explained_variance_ratio_
    var_ratios = var_ratios[var_ratios > 1e-10]
    entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-10))
    effective_dim = np.exp(entropy)

    return {
        'variance_ratio': float(cumvar[-1]),
        'first_component_ratio': float(first_ratio),
        'effective_dim': float(effective_dim),
    }


def compute_snr(X: np.ndarray) -> float:
    """Compute signal-to-noise ratio."""
    signal_var = np.var(np.mean(X, axis=0))
    noise_var = np.mean(np.var(X, axis=0))
    if noise_var < 1e-10:
        return 100.0
    return float(signal_var / noise_var)


def compute_roughness(X: np.ndarray) -> float:
    """Compute spectral roughness (2nd derivative magnitude)."""
    if X.shape[1] < 3:
        return 0.0
    d2 = np.diff(X, n=2, axis=1)
    return float(np.mean(np.abs(d2)))


def compute_separation(X: np.ndarray, n_samples: int = 100) -> float:
    """Compute inter-sample separation score."""
    if X.shape[0] < 2:
        return 0.0

    # Subsample for speed
    if X.shape[0] > n_samples:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Compute pairwise distances
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(X_sub)

    # Mean distance (normalized)
    mean_dist = np.mean(dists[np.triu_indices(len(X_sub), k=1)])

    # Normalize by feature std
    std_norm = np.mean(np.std(X_sub, axis=0)) + 1e-10

    return float(mean_dist / std_norm)


def evaluate_unsupervised(X: np.ndarray) -> Dict[str, float]:
    """Compute all unsupervised metrics."""
    pca_metrics = compute_pca_metrics(X)

    metrics = {
        'variance_ratio': pca_metrics['variance_ratio'],
        'effective_dim': pca_metrics['effective_dim'],
        'snr': compute_snr(X),
        'roughness': compute_roughness(X),
        'separation': compute_separation(X),
    }

    # Compute total score (normalized combination)
    # Higher is better for: variance_ratio, effective_dim, snr, separation
    # Lower is better for: roughness

    # Normalize roughness (invert and clip)
    roughness_score = 1.0 / (1.0 + metrics['roughness'])

    # Normalize SNR (log scale, clipped)
    snr_score = np.clip(np.log1p(metrics['snr']) / 5.0, 0, 1)

    # Normalize effective_dim
    eff_dim_score = np.clip(metrics['effective_dim'] / 10.0, 0, 1)

    # Total score
    metrics['total_score'] = (
        0.25 * metrics['variance_ratio'] +
        0.25 * eff_dim_score +
        0.20 * snr_score +
        0.15 * roughness_score +
        0.15 * np.clip(metrics['separation'] / 10.0, 0, 1)
    )

    return metrics


# =============================================================================
# Distance Metrics for Diversity
# =============================================================================

def compute_grassmann_distance(X1: np.ndarray, X2: np.ndarray, n_components: int = 5) -> float:
    """
    Compute Grassmann distance between PCA subspaces of two datasets.

    Lower distance = more similar preprocessing effects
    """
    n_comp = min(n_components, X1.shape[0] - 1, X1.shape[1], X2.shape[0] - 1, X2.shape[1])
    if n_comp < 1:
        return 1.0

    pca1 = PCA(n_components=n_comp)
    pca2 = PCA(n_components=n_comp)

    pca1.fit(X1)
    pca2.fit(X2)

    # Get principal components (column space)
    U1 = pca1.components_.T  # (features, n_comp)
    U2 = pca2.components_.T

    # Compute principal angles via SVD
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)

    # Grassmann distance = sqrt(sum of squared angles)
    angles = np.arccos(s)
    distance = np.sqrt(np.sum(angles ** 2))

    # Normalize to [0, 1]
    max_dist = np.sqrt(n_comp * (np.pi / 2) ** 2)
    return float(distance / max_dist)


def compute_cka_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute CKA-based distance (1 - CKA similarity).

    CKA = Centered Kernel Alignment
    """
    def centering_matrix(n):
        return np.eye(n) - np.ones((n, n)) / n

    n = X1.shape[0]
    if n != X2.shape[0]:
        min_n = min(n, X2.shape[0])
        X1, X2 = X1[:min_n], X2[:min_n]
        n = min_n

    H = centering_matrix(n)

    # Linear kernels
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T

    # Center kernels
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H

    # HSIC
    hsic = np.sum(K1c * K2c) / (n - 1) ** 2

    # Normalize
    hsic1 = np.sum(K1c * K1c) / (n - 1) ** 2
    hsic2 = np.sum(K2c * K2c) / (n - 1) ** 2

    if hsic1 < 1e-10 or hsic2 < 1e-10:
        return 1.0

    cka = hsic / np.sqrt(hsic1 * hsic2)

    return float(1.0 - cka)


# =============================================================================
# Pipeline Utilities
# =============================================================================

def get_base_preprocessings() -> Dict[str, Any]:
    """Get the base set of preprocessing transforms."""
    return {
        'snv': StandardNormalVariate(),
        'rsnv': RobustStandardNormalVariate(),
        'msc': MultiplicativeScatterCorrection(scale=False),
        'savgol': SavitzkyGolay(window_length=11, polyorder=3),
        'd1': FirstDerivative(),
        'd2': SecondDerivative(),
        'savgol_d1': SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
        'haar': Haar(),
        'detrend': Detrend(),
        'gaussian': Gaussian(order=1, sigma=2),
        'gaussian2': Gaussian(order=2, sigma=2),
        'emsc': EMSC(),
        'area_norm': AreaNormalization(),
        'wav_sym5': Wavelet('sym5'),
        'wav_coif3': Wavelet('coif3'),
        'identity': IdentityTransformer(),
    }

def apply_pipeline(X: np.ndarray, transforms: List) -> np.ndarray:
    """Apply a sequence of transforms to X."""
    from copy import deepcopy
    X_out = X.copy()
    for t in transforms:
        t_copy = deepcopy(t)
        X_out = t_copy.fit_transform(X_out)
    return X_out


def apply_augmentation(X: np.ndarray, transform_list: List[List]) -> np.ndarray:
    """Apply multiple pipelines and concatenate features."""
    transformed = []
    for transforms in transform_list:
        X_t = apply_pipeline(X, transforms)
        transformed.append(X_t)
    return np.hstack(transformed)


def generate_stacked_pipelines(
    preprocessings: Dict[str, Any],
    max_depth: int = 3
) -> List[Tuple[str, List[str], List]]:
    """
    Generate all stacked pipeline combinations.

    Returns:
        List of (name, component_names, transforms)
    """
    names = list(preprocessings.keys())
    pipelines = []

    for depth in range(1, max_depth + 1):
        for combo in permutations(names, depth):
            name = '>'.join(combo)
            transforms = [preprocessings[n] for n in combo]
            pipelines.append((name, list(combo), transforms))

    return pipelines


# =============================================================================
# Proxy Model Evaluation
# =============================================================================

def evaluate_with_proxies(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    is_classification: bool = False
) -> Dict[str, float]:
    """Evaluate with Ridge and KNN proxy models."""
    results = {}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle high-dimensional data
    if X_scaled.shape[1] > X_scaled.shape[0]:
        pca = PCA(n_components=min(50, X_scaled.shape[0] - 1))
        X_scaled = pca.fit_transform(X_scaled)

    # Ridge
    try:
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        scores = cross_val_score(ridge, X_scaled, y, cv=cv_folds, scoring='r2')
        results['ridge_r2'] = float(np.mean(scores))
    except Exception:
        results['ridge_r2'] = 0.0

    # KNN
    try:
        if is_classification:
            knn = KNeighborsClassifier(n_neighbors=5)
            scoring = 'accuracy'
        else:
            knn = KNeighborsRegressor(n_neighbors=5)
            scoring = 'r2'
        scores = cross_val_score(knn, X_scaled, y, cv=cv_folds, scoring=scoring)
        results['knn_score'] = float(np.mean(scores))
    except Exception:
        results['knn_score'] = 0.0

    # Composite
    results['proxy_score'] = 0.6 * results['ridge_r2'] + 0.4 * results['knn_score']

    return results


# =============================================================================
# Main Systematic Selection Class
# =============================================================================

class SystematicSelector:
    """
    Systematic preprocessing selection with exhaustive evaluation.
    """

    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.results: List[PipelineResult] = []
        self.distance_matrix: Optional[pd.DataFrame] = None

    def _log(self, msg: str, level: int = 1):
        if self.verbose >= level:
            print(msg)

    def run_stage1_unsupervised(
        self,
        X: np.ndarray,
        preprocessings: Dict[str, Any],
        max_depth: int = 3
    ) -> pd.DataFrame:
        """
        Stage 1: Exhaustive unsupervised evaluation of all pipelines.

        Returns:
            DataFrame with all results
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 1: Exhaustive Unsupervised Evaluation")
        self._log("=" * 70)

        # Generate all stacked pipelines
        pipelines = generate_stacked_pipelines(preprocessings, max_depth)
        self._log(f"\nGenerated {len(pipelines)} pipeline combinations (depth 1-{max_depth})")

        self.results = []

        for i, (name, components, transforms) in enumerate(pipelines):
            if self.verbose >= 1:
                print(f"\r  Evaluating [{i+1}/{len(pipelines)}] {name}...", end='', flush=True)

            try:
                # Apply pipeline
                X_t = apply_pipeline(X, transforms)

                # Check for valid output
                if np.any(np.isnan(X_t)) or np.any(np.isinf(X_t)):
                    self._log(f"\n  Warning: {name} produced NaN/Inf, skipping", 2)
                    continue

                # Compute metrics
                metrics = evaluate_unsupervised(X_t)

                result = PipelineResult(
                    name=name,
                    depth=len(components),
                    pipeline_type='stacked' if len(components) > 1 else 'single',
                    components=components,
                    X_transformed=X_t,
                    metrics=metrics,
                    total_score=metrics['total_score']
                )
                self.results.append(result)

            except Exception as e:
                self._log(f"\n  Error with {name}: {e}", 2)
                continue

        print()  # Newline after progress

        # Create DataFrame
        df_data = []
        for r in self.results:
            row = {
                'name': r.name,
                'depth': r.depth,
                'type': r.pipeline_type,
                'components': '|'.join(r.components),
                **r.metrics
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

        self._log(f"\n✓ Evaluated {len(self.results)} valid pipelines")
        self._log("\nTop 10 by total score:")
        self._log(df[['name', 'depth', 'total_score', 'variance_ratio', 'effective_dim', 'snr']].head(10).to_string())

        return df

    def run_stage2_diversity(
        self,
        top_k: int = 15
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Stage 2: Compute distances between top candidates for diversity.

        Returns:
            - Distance matrix DataFrame
            - List of diverse pairs (name1, name2, distance)
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 2: Diversity Analysis")
        self._log("=" * 70)

        # Get top results
        top_results = sorted(self.results, key=lambda x: x.total_score, reverse=True)[:top_k]
        names = [r.name for r in top_results]

        self._log(f"\nComputing pairwise distances for top {len(names)} pipelines...")

        # Compute distance matrix
        n = len(top_results)
        grassmann_matrix = np.zeros((n, n))
        cka_matrix = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair_count += 1
                if self.verbose >= 1:
                    print(f"\r  Computing distances [{pair_count}/{total_pairs}]...", end='', flush=True)

                X1 = top_results[i].X_transformed
                X2 = top_results[j].X_transformed

                grassmann_matrix[i, j] = compute_grassmann_distance(X1, X2)
                grassmann_matrix[j, i] = grassmann_matrix[i, j]

                cka_matrix[i, j] = compute_cka_distance(X1, X2)
                cka_matrix[j, i] = cka_matrix[i, j]

        print()

        # Combined distance
        combined_matrix = 0.5 * grassmann_matrix + 0.5 * cka_matrix

        # Create DataFrame
        self.distance_matrix = pd.DataFrame(combined_matrix, index=names, columns=names)

        # Find diverse pairs (high distance = different = good for augmentation)
        diverse_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                diverse_pairs.append((names[i], names[j], combined_matrix[i, j]))

        # Sort by distance (descending = most diverse first)
        diverse_pairs.sort(key=lambda x: x[2], reverse=True)

        self._log(f"\n✓ Distance matrix computed")
        self._log("\nMost diverse pairs (best for augmentation):")
        for p1, p2, d in diverse_pairs[:10]:
            self._log(f"  {p1} + {p2}: distance = {d:.4f}")

        return self.distance_matrix, diverse_pairs

    def run_stage3_proxy_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessings: Dict[str, Any],
        top_k: int = 15,
        n_augmentations: int = 20,
        diverse_pairs: List[Tuple[str, str, float]] = None,
        cv_folds: int = 3
    ) -> pd.DataFrame:
        """
        Stage 3: Evaluate top candidates with proxy models.

        Tests:
        - Top single/stacked pipelines
        - Diverse 2-way augmentations
        - Diverse 3-way augmentations
        """
        self._log("\n" + "=" * 70)
        self._log("STAGE 3: Proxy Model Evaluation")
        self._log("=" * 70)

        # Get top results
        top_results = sorted(self.results, key=lambda x: x.total_score, reverse=True)[:top_k]

        # Detect if classification
        unique_y = np.unique(y)
        is_classification = len(unique_y) < 20 and np.all(unique_y == unique_y.astype(int))
        task_type = "classification" if is_classification else "regression"
        self._log(f"\nDetected task: {task_type}")

        final_results = []

        # Evaluate top single/stacked pipelines
        self._log(f"\nEvaluating top {len(top_results)} pipelines with proxy models...")

        for i, r in enumerate(top_results):
            if self.verbose >= 1:
                print(f"\r  [{i+1}/{len(top_results)}] {r.name}...", end='', flush=True)

            proxy_scores = evaluate_with_proxies(r.X_transformed, y, cv_folds, is_classification)
            r.proxy_scores = proxy_scores
            r.final_score = 0.4 * r.total_score + 0.6 * proxy_scores['proxy_score']
            final_results.append(r)

        print()

        # Generate and evaluate diverse augmentations
        if diverse_pairs:
            self._log(f"\nEvaluating {n_augmentations} diverse 2-way augmentations...")

            # Take most diverse pairs
            aug_pairs = diverse_pairs[:n_augmentations]

            for i, (name1, name2, dist) in enumerate(aug_pairs):
                if self.verbose >= 1:
                    print(f"\r  [{i+1}/{len(aug_pairs)}] {name1} + {name2}...", end='', flush=True)

                # Find the results
                r1 = next((r for r in self.results if r.name == name1), None)
                r2 = next((r for r in self.results if r.name == name2), None)

                if r1 is None or r2 is None:
                    continue

                # Create augmented features
                X_aug = np.hstack([r1.X_transformed, r2.X_transformed])

                # Compute metrics
                metrics = evaluate_unsupervised(X_aug)
                proxy_scores = evaluate_with_proxies(X_aug, y, cv_folds, is_classification)

                aug_result = PipelineResult(
                    name=f"[{name1}+{name2}]",
                    depth=2,
                    pipeline_type='augmented_2',
                    components=[name1, name2],
                    X_transformed=X_aug,
                    metrics=metrics,
                    total_score=metrics['total_score'],
                    proxy_scores=proxy_scores,
                    final_score=0.4 * metrics['total_score'] + 0.6 * proxy_scores['proxy_score']
                )
                final_results.append(aug_result)

            print()

            # 3-way augmentations (top diverse triplets)
            self._log("\nEvaluating diverse 3-way augmentations...")

            # Get top names from diverse pairs
            top_diverse_names = list(set(
                [p[0] for p in diverse_pairs[:10]] + [p[1] for p in diverse_pairs[:10]]
            ))[:8]

            # Generate triplets
            triplet_count = 0
            max_triplets = 15

            for combo in combinations(top_diverse_names, 3):
                if triplet_count >= max_triplets:
                    break

                name1, name2, name3 = combo

                r1 = next((r for r in self.results if r.name == name1), None)
                r2 = next((r for r in self.results if r.name == name2), None)
                r3 = next((r for r in self.results if r.name == name3), None)

                if r1 is None or r2 is None or r3 is None:
                    continue

                if self.verbose >= 1:
                    print(f"\r  [{triplet_count+1}/{max_triplets}] {name1} + {name2} + {name3}...", end='', flush=True)

                X_aug = np.hstack([r1.X_transformed, r2.X_transformed, r3.X_transformed])

                metrics = evaluate_unsupervised(X_aug)
                proxy_scores = evaluate_with_proxies(X_aug, y, cv_folds, is_classification)

                aug_result = PipelineResult(
                    name=f"[{name1}+{name2}+{name3}]",
                    depth=3,
                    pipeline_type='augmented_3',
                    components=[name1, name2, name3],
                    X_transformed=X_aug,
                    metrics=metrics,
                    total_score=metrics['total_score'],
                    proxy_scores=proxy_scores,
                    final_score=0.4 * metrics['total_score'] + 0.6 * proxy_scores['proxy_score']
                )
                final_results.append(aug_result)
                triplet_count += 1

            print()

        # Create final DataFrame
        df_data = []
        for r in final_results:
            row = {
                'name': r.name,
                'type': r.pipeline_type,
                'depth': r.depth,
                'unsupervised_score': r.total_score,
                'ridge_r2': r.proxy_scores.get('ridge_r2', 0),
                'knn_score': r.proxy_scores.get('knn_score', 0),
                'proxy_score': r.proxy_scores.get('proxy_score', 0),
                'final_score': r.final_score,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)

        self._log(f"\n✓ Evaluated {len(final_results)} configurations")

        return df

    def run_full_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessings: Dict[str, Any] = None,
        max_depth: int = 3,
        top_k: int = 15,
        n_augmentations: int = 20,
        cv_folds: int = 3,
        output_dir: str = '.'
    ) -> Dict[str, Any]:
        """
        Run the complete systematic selection pipeline.

        Returns:
            Dict with all results and DataFrames
        """
        if preprocessings is None:
            preprocessings = get_base_preprocessings()

        start_time = time.time()

        # Stage 1
        df_stage1 = self.run_stage1_unsupervised(X, preprocessings, max_depth)

        # Stage 2
        distance_matrix, diverse_pairs = self.run_stage2_diversity(top_k)

        # Stage 3
        df_final = self.run_stage3_proxy_evaluation(
            X, y, preprocessings, top_k, n_augmentations, diverse_pairs, cv_folds
        )

        total_time = time.time() - start_time

        # Save results
        stage1_path = os.path.join(output_dir, 'stage1_unsupervised.csv')
        df_stage1.to_csv(stage1_path, index=False)
        self._log(f"\n📄 Stage 1 results saved to: {stage1_path}")

        final_path = os.path.join(output_dir, 'final_ranking.csv')
        df_final.to_csv(final_path, index=False)
        self._log(f"📄 Final ranking saved to: {final_path}")

        if distance_matrix is not None:
            dist_path = os.path.join(output_dir, 'distance_matrix.csv')
            distance_matrix.to_csv(dist_path)
            self._log(f"📄 Distance matrix saved to: {dist_path}")

        # Print final summary
        self._log("\n" + "=" * 70)
        self._log("FINAL RANKING")
        self._log("=" * 70)
        self._log(f"\n⏱️ Total time: {total_time:.1f}s")
        self._log("\n🏆 Top 15 Configurations:")
        self._log(df_final[['name', 'type', 'unsupervised_score', 'proxy_score', 'final_score']].head(15).to_string())

        return {
            'stage1_df': df_stage1,
            'distance_matrix': distance_matrix,
            'diverse_pairs': diverse_pairs,
            'final_df': df_final,
            'total_time': total_time,
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    stage1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    distance_matrix: pd.DataFrame = None,
    output_path: str = 'systematic_results.png'
):
    """Create comprehensive visualization."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Stage 1 - Score distribution by depth
    ax1 = fig.add_subplot(2, 3, 1)
    for depth in sorted(stage1_df['depth'].unique()):
        subset = stage1_df[stage1_df['depth'] == depth]
        ax1.hist(subset['total_score'], bins=20, alpha=0.5, label=f'Depth {depth}')
    ax1.set_xlabel('Unsupervised Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Stage 1: Score Distribution by Depth')
    ax1.legend()

    # Plot 2: Stage 1 - Top 15 pipelines
    ax2 = fig.add_subplot(2, 3, 2)
    top15 = stage1_df.head(15)
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, 15))
    ax2.barh(range(15), top15['total_score'], color=colors)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(top15['name'], fontsize=8)
    ax2.set_xlabel('Unsupervised Score')
    ax2.set_title('Stage 1: Top 15 Pipelines')
    ax2.invert_yaxis()

    # Plot 3: Distance heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    if distance_matrix is not None:
        im = ax3.imshow(distance_matrix.values, cmap='RdYlBu', aspect='auto')
        ax3.set_xticks(range(len(distance_matrix.columns)))
        ax3.set_yticks(range(len(distance_matrix.index)))
        ax3.set_xticklabels(distance_matrix.columns, rotation=45, ha='right', fontsize=6)
        ax3.set_yticklabels(distance_matrix.index, fontsize=6)
        plt.colorbar(im, ax=ax3, label='Distance')
        ax3.set_title('Stage 2: Preprocessing Distances')
    else:
        ax3.text(0.5, 0.5, 'No distance data', ha='center', va='center')

    # Plot 4: Metrics comparison (radar-like bar chart)
    ax4 = fig.add_subplot(2, 3, 4)
    top5 = stage1_df.head(5)
    metrics = ['variance_ratio', 'effective_dim', 'snr', 'separation']
    x = np.arange(len(metrics))
    width = 0.15

    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[m] if m != 'effective_dim' else row[m] / 10 for m in metrics]
        values = [min(v, 1.0) for v in values]  # Normalize
        ax4.bar(x + i * width, values, width, label=row['name'][:15])

    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(['Variance', 'Eff. Dim (÷10)', 'SNR', 'Separation'])
    ax4.set_ylabel('Score')
    ax4.set_title('Stage 1: Metrics Comparison (Top 5)')
    ax4.legend(fontsize=7, loc='upper right')

    # Plot 5: Final ranking by type
    ax5 = fig.add_subplot(2, 3, 5)
    type_colors = {
        'single': 'steelblue',
        'stacked': 'coral',
        'augmented_2': 'green',
        'augmented_3': 'purple'
    }

    top20 = final_df.head(20)
    colors = [type_colors.get(t, 'gray') for t in top20['type']]
    ax5.barh(range(len(top20)), top20['final_score'], color=colors)
    ax5.set_yticks(range(len(top20)))
    ax5.set_yticklabels(top20['name'], fontsize=7)
    ax5.set_xlabel('Final Score')
    ax5.set_title('Final Ranking (Top 20)')
    ax5.invert_yaxis()

    # Add legend for types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax5.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # Plot 6: Proxy vs Unsupervised scores
    ax6 = fig.add_subplot(2, 3, 6)
    scatter_colors = [type_colors.get(t, 'gray') for t in final_df['type']]
    ax6.scatter(
        final_df['unsupervised_score'],
        final_df['proxy_score'],
        c=scatter_colors,
        alpha=0.6,
        s=50
    )
    ax6.set_xlabel('Unsupervised Score')
    ax6.set_ylabel('Proxy Score')
    ax6.set_title('Unsupervised vs Proxy Performance')

    # Add labels for top 5
    for i in range(min(5, len(final_df))):
        row = final_df.iloc[i]
        ax6.annotate(
            row['name'][:12],
            (row['unsupervised_score'], row['proxy_score']),
            fontsize=7,
            alpha=0.8
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")

    return fig


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from CSV files."""
    patterns = [
        ('Xcal.csv', 'Ycal.csv'),
        ('Xcal.csv.gz', 'Ycal.csv.gz'),
        ('Xtrain.csv', 'Ytrain.csv'),
    ]

    for x_file, y_file in patterns:
        x_path = os.path.join(data_path, x_file)
        y_path = os.path.join(data_path, y_file)

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Loading data from {x_path}...")

            # Read with header detection
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

            # Handle shape mismatch
            min_samples = min(X.shape[0], y.shape[0])
            X, y = X[:min_samples], y[:min_samples]

            # Handle NaN
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[valid], y[valid]

            print(f"Loaded X: {X.shape}, y: {y.shape}")
            return X, y

    raise FileNotFoundError(f"Could not find data in {data_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Systematic Preprocessing Selection')
    parser.add_argument('--depth', type=int, default=3, help='Maximum pipeline depth (1-3)')
    parser.add_argument('--top', type=int, default=15, help='Number of top candidates for Stage 2-3')
    parser.add_argument('--plots', action='store_true', help='Show plots')
    parser.add_argument('--full', action='store_true', help='Use full nitro dataset')
    parser.add_argument('--data', type=str, default=None, help='Custom data path')
    parser.add_argument('--output', type=str, default='selection', help='Output directory')
    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Determine data path
    if args.data:
        data_path = args.data
    elif args.full:
        data_path = str(script_dir / 'nitro_regression' / 'Digestibility_0.8')
    else:
        data_path = str(script_dir / 'nitro_regression' / 'Digestibility_0.8')

    # Ensure output directory exists
    output_dir = script_dir / args.output
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\n📂 Data path: {data_path}")
    X, y = load_data(data_path)

    # Run selection
    selector = SystematicSelector(verbose=1)
    results = selector.run_full_selection(
        X=X,
        y=y,
        max_depth=args.depth,
        top_k=args.top,
        output_dir=args.output
    )

    # Create plots
    fig = plot_results(
        results['stage1_df'],
        results['final_df'],
        results['distance_matrix'],
        output_path=os.path.join(args.output, 'systematic_results.png')
    )

    if args.plots:
        import matplotlib.pyplot as plt
        plt.show()

    print("\n✅ Complete!")


if __name__ == '__main__':
    main()
