"""
Combination Analysis for Preprocessing Selection
=================================================

This module provides methods for analyzing preprocessing stacks and combinations.
These are Stage D methods that help identify complementary preprocessings and
avoid redundant combinations.

Stage D: Combination Analysis
- Mutual Information Redundancy: MI of each preprocessing with Y, penalize redundant combinations
- Grassmann Distance: Angles between latent spaces of different preprocessings
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.linalg import subspace_angles


def mutual_info_redundancy(
    preprocessed_variants: dict,
    y: np.ndarray,
    task: str = 'auto',
    top_k_features: int = 50
) -> dict:
    """
    Analyze preprocessing combinations using mutual information.

    Compute MI of each preprocessing with Y, then penalize redundant combinations.
    Combinations with similar latent spaces are redundant.

    Args:
        preprocessed_variants: Dict of {name: X_preprocessed}
        y: Target values
        task: 'regression' or 'classification' or 'auto'
        top_k_features: Number of top features to use for redundancy

    Returns:
        dict with 'mi_scores', 'redundancy_matrix', 'combination_scores', 'names'
    """
    if task == 'auto':
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'

    mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression

    names = list(preprocessed_variants.keys())
    n_pp = len(names)

    # MI with Y for each preprocessing
    mi_scores = {}
    reduced_X = {}  # Keep top features for redundancy computation

    for name, X in preprocessed_variants.items():
        try:
            mi_values = mi_func(X, y, random_state=42)
            mi_scores[name] = float(np.mean(mi_values))

            # Keep top features for redundancy
            n_top = min(top_k_features, X.shape[1])
            top_indices = np.argsort(mi_values)[-n_top:]
            reduced_X[name] = X[:, top_indices]
        except Exception:
            mi_scores[name] = 0.0
            reduced_X[name] = X[:, :min(top_k_features, X.shape[1])]

    # Pairwise redundancy (correlation between preprocessings)
    redundancy_matrix = np.zeros((n_pp, n_pp))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                # Use mean correlation as proxy for redundancy
                Xi = reduced_X[name_i]
                Xj = reduced_X[name_j]
                try:
                    corr = np.abs(np.corrcoef(Xi.mean(axis=1), Xj.mean(axis=1))[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
                redundancy_matrix[i, j] = corr
                redundancy_matrix[j, i] = corr

    # Score combinations (MI - redundancy)
    combination_scores = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                combo_name = f"{name_i}+{name_j}"
                score = mi_scores[name_i] + mi_scores[name_j] - redundancy_matrix[i, j]
                combination_scores[combo_name] = float(score)

    return {
        'mi_scores': mi_scores,
        'redundancy_matrix': redundancy_matrix,
        'names': names,
        'combination_scores': combination_scores
    }


def grassmann_distance(
    preprocessed_variants: dict,
    n_components: int = 5
) -> dict:
    """
    Compute Grassmann distances between preprocessing latent spaces.

    Compute angles between latent spaces of different preprocessings.
    Combinations with similar latent spaces are redundant.

    Args:
        preprocessed_variants: Dict of {name: X_preprocessed}
        n_components: Number of PCA components for subspace

    Returns:
        dict with 'distance_matrix', 'names', 'similar_pairs', 'diverse_pairs'
    """
    names = list(preprocessed_variants.keys())
    n_pp = len(names)

    # Compute subspaces via PCA
    subspaces = {}
    for name, X in preprocessed_variants.items():
        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 1:
            subspaces[name] = np.zeros((X.shape[1], 1))
            continue

        try:
            pca = PCA(n_components=n_comp)
            pca.fit(X)
            subspaces[name] = pca.components_.T  # (n_features, n_components)
        except Exception:
            subspaces[name] = np.zeros((X.shape[1], 1))

    # Compute pairwise Grassmann distances
    distance_matrix = np.zeros((n_pp, n_pp))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                try:
                    # Ensure same number of features for subspace_angles
                    sub_i = subspaces[name_i]
                    sub_j = subspaces[name_j]

                    # Handle different feature dimensions
                    min_features = min(sub_i.shape[0], sub_j.shape[0])
                    sub_i = sub_i[:min_features, :]
                    sub_j = sub_j[:min_features, :]

                    angles = subspace_angles(sub_i, sub_j)
                    # Grassmann distance = sqrt(sum of squared angles)
                    dist = np.sqrt(np.sum(angles ** 2))
                except Exception:
                    dist = 0.0

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

    # Find similar pairs (low distance = redundant)
    similar_pairs = []
    diverse_pairs = []

    non_zero_distances = distance_matrix[distance_matrix > 0]
    if len(non_zero_distances) > 0:
        threshold_low = np.percentile(non_zero_distances, 25)
        threshold_high = np.percentile(non_zero_distances, 75)

        for i in range(n_pp):
            for j in range(i + 1, n_pp):
                dist = distance_matrix[i, j]
                if dist < threshold_low:
                    similar_pairs.append((names[i], names[j], float(dist)))
                elif dist > threshold_high:
                    diverse_pairs.append((names[i], names[j], float(dist)))

    return {
        'distance_matrix': distance_matrix,
        'names': names,
        'similar_pairs': similar_pairs,
        'diverse_pairs': diverse_pairs
    }


def analyze_combinations(
    preprocessed_variants: dict,
    y: np.ndarray,
    task: str = 'auto',
    n_components: int = 5,
    top_k_features: int = 50
) -> dict:
    """
    Run all Stage D combination analysis methods.

    Args:
        preprocessed_variants: Dict of {name: X_preprocessed}
        y: Target values
        task: 'regression', 'classification', or 'auto'
        n_components: Number of PCA components for Grassmann distance
        top_k_features: Number of top features for MI redundancy

    Returns:
        dict with results from all combination analysis methods
    """
    results = {}

    # Mutual information redundancy
    mi_result = mutual_info_redundancy(
        preprocessed_variants,
        y,
        task=task,
        top_k_features=top_k_features
    )
    results['mutual_info'] = mi_result

    # Grassmann distance
    grassmann_result = grassmann_distance(
        preprocessed_variants,
        n_components=n_components
    )
    results['grassmann'] = grassmann_result

    # Recommend best combinations (high MI, low redundancy, high diversity)
    recommended_combinations = []

    # Sort combinations by MI-based score
    combo_scores = mi_result['combination_scores']
    sorted_combos = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)

    # Add diversity bonus from Grassmann
    names = mi_result['names']
    name_to_idx = {name: i for i, name in enumerate(names)}

    for combo_name, mi_score in sorted_combos[:10]:  # Top 10
        parts = combo_name.split('+')
        if len(parts) == 2:
            i, j = name_to_idx.get(parts[0]), name_to_idx.get(parts[1])
            if i is not None and j is not None:
                grassmann_dist = grassmann_result['distance_matrix'][i, j]
                # Combined score: MI score + diversity bonus
                combined_score = mi_score + 0.1 * grassmann_dist
                recommended_combinations.append({
                    'combination': combo_name,
                    'mi_score': float(mi_score),
                    'grassmann_distance': float(grassmann_dist),
                    'combined_score': float(combined_score)
                })

    # Sort by combined score
    recommended_combinations.sort(key=lambda x: x['combined_score'], reverse=True)
    results['recommended_combinations'] = recommended_combinations

    return results
