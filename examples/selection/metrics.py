"""
Preprocessing Selection Metrics
===============================

This module provides metrics for filtering and ranking preprocessing techniques
before running full ML/DL pipelines. The goal is to reduce the preprocessing
search space by 5-10× without losing performance.

Stage A: Unsupervised Filtering (no target required)
Stage B: Supervised Ranking (uses target but no full model training)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, pairwise_distances


# =============================================================================
# Stage A: Unsupervised Filtering (30-50% elimination)
# =============================================================================

def compute_snr(X: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio.

    SNR = mean(signal) / std(noise)
    Using sample-wise mean as signal and residual std as noise.

    Args:
        X: Spectra data (n_samples, n_features)

    Returns:
        SNR value
    """
    signal = np.mean(X, axis=1)  # Mean spectrum per sample
    residual = X - signal[:, np.newaxis]  # Deviation from mean
    noise = np.std(residual, axis=1)  # Noise per sample

    # Avoid division by zero
    noise = np.where(noise == 0, 1e-10, noise)
    snr = np.mean(np.abs(signal)) / np.mean(noise)
    return snr


def compute_roughness(X: np.ndarray) -> float:
    """
    Compute spectral roughness as mean absolute second derivative.
    High values indicate jagged/noisy spectra.

    Args:
        X: Spectra data (n_samples, n_features)

    Returns:
        Roughness value
    """
    if X.shape[1] < 3:
        return 0.0
    d2 = np.diff(X, n=2, axis=1)  # Second derivative
    roughness = np.mean(np.abs(d2))
    return roughness


def pca_variance_filter(
    X_preprocessed: np.ndarray,
    n_components: int = 10,
    min_variance_ratio: float = 0.90,
    max_first_component_ratio: float = 0.99
) -> dict:
    """
    Filter preprocessing based on PCA variance analysis.

    Eliminates preprocessings that:
    - Destroy too much information (variance << others)
    - Produce artifacts (variance concentrated on 1 component)

    Args:
        X_preprocessed: Transformed data (n_samples, n_features)
        n_components: Number of PCA components to consider
        min_variance_ratio: Minimum cumulative variance to keep
        max_first_component_ratio: Maximum variance for 1st component

    Returns:
        dict with 'variance_score', 'is_valid', 'reason', 'explained_variance_ratio'
    """
    n_comp = min(n_components, X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
    if n_comp < 1:
        return {
            'variance_score': 0.0,
            'is_valid': False,
            'reason': 'Insufficient samples or features for PCA',
            'explained_variance_ratio': np.array([])
        }

    pca = PCA(n_components=n_comp)
    pca.fit(X_preprocessed)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    first_comp_ratio = pca.explained_variance_ratio_[0]
    total_var = cumulative_var[-1]

    is_valid = True
    reason = ""

    if total_var < min_variance_ratio:
        is_valid = False
        reason = f"Variance too low: {total_var:.3f} < {min_variance_ratio}"
    elif first_comp_ratio > max_first_component_ratio:
        is_valid = False
        reason = f"First component too dominant: {first_comp_ratio:.3f} > {max_first_component_ratio}"

    return {
        'variance_score': total_var,
        'is_valid': is_valid,
        'reason': reason,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }


def snr_filter(
    X_original: np.ndarray,
    X_preprocessed: np.ndarray,
    min_snr_ratio: float = 0.8
) -> dict:
    """
    Filter preprocessing based on SNR analysis.

    Eliminates preprocessings that increase relative noise (SNR↓).

    Args:
        X_original: Original spectra
        X_preprocessed: Transformed spectra
        min_snr_ratio: Minimum SNR_after / SNR_before

    Returns:
        dict with 'snr_before', 'snr_after', 'snr_ratio', 'is_valid', 'reason'
    """
    snr_before = compute_snr(X_original)
    snr_after = compute_snr(X_preprocessed)
    snr_ratio = snr_after / snr_before if snr_before != 0 else 0

    is_valid = snr_ratio >= min_snr_ratio
    reason = "" if is_valid else f"SNR degraded: ratio={snr_ratio:.3f} < {min_snr_ratio}"

    return {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_ratio': snr_ratio,
        'is_valid': is_valid,
        'reason': reason
    }


def roughness_filter(
    X_preprocessed: np.ndarray,
    X_original: np.ndarray = None,
    max_roughness_ratio: float = 10.0
) -> dict:
    """
    Filter preprocessing based on roughness analysis.

    Too aggressive derivatives or smoothing can produce artifacts.

    Args:
        X_preprocessed: Transformed spectra
        X_original: Original spectra (for ratio comparison)
        max_roughness_ratio: Maximum acceptable roughness ratio vs original

    Returns:
        dict with 'roughness', 'roughness_ratio', 'is_valid', 'reason'
    """
    roughness = compute_roughness(X_preprocessed)

    if X_original is not None:
        roughness_orig = compute_roughness(X_original)
        roughness_ratio = roughness / roughness_orig if roughness_orig > 0 else float('inf')
    else:
        roughness_ratio = 1.0

    is_valid = roughness_ratio <= max_roughness_ratio
    reason = "" if is_valid else f"Too rough: ratio={roughness_ratio:.3f} > {max_roughness_ratio}"

    return {
        'roughness': roughness,
        'roughness_ratio': roughness_ratio,
        'is_valid': is_valid,
        'reason': reason
    }


def distance_separation_filter(
    X_preprocessed: np.ndarray,
    y: np.ndarray = None,
    min_separation_ratio: float = 1.0,
    n_samples: int = 500
) -> dict:
    """
    Filter preprocessing based on sample separation analysis.

    Computes L2 distances intra-sample (similar samples) vs inter-sample
    (different samples). Good preprocessings should increase separation.

    For regression: groups by Y quantiles
    For classification: groups by class

    Args:
        X_preprocessed: Transformed spectra
        y: Target values (optional, uses random pairs if None)
        min_separation_ratio: Minimum inter/intra distance ratio
        n_samples: Number of pairs to sample for efficiency

    Returns:
        dict with 'intra_distance', 'inter_distance', 'separation_ratio', 'is_valid'
    """
    n = X_preprocessed.shape[0]

    if y is not None:
        # Create groups based on Y
        if len(np.unique(y)) < 10:  # Classification
            groups = y.copy()
        else:  # Regression: quantile-based groups
            groups = np.digitize(y, np.percentile(y, [25, 50, 75]))

        intra_pairs = []
        inter_pairs = []

        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            indices = np.where(mask)[0]
            if len(indices) >= 2:
                # Sample intra pairs
                for _ in range(min(n_samples // len(unique_groups), len(indices))):
                    i, j = np.random.choice(indices, 2, replace=False)
                    intra_pairs.append((i, j))

            # Sample inter pairs
            other_indices = np.where(~mask)[0]
            if len(other_indices) > 0:
                for idx in indices[:min(n_samples // len(unique_groups), len(indices))]:
                    j = np.random.choice(other_indices)
                    inter_pairs.append((idx, j))

        if intra_pairs:
            intra_dists = [np.linalg.norm(X_preprocessed[i] - X_preprocessed[j]) for i, j in intra_pairs]
            intra_distance = np.mean(intra_dists)
        else:
            intra_distance = 1e-10

        if inter_pairs:
            inter_dists = [np.linalg.norm(X_preprocessed[i] - X_preprocessed[j]) for i, j in inter_pairs]
            inter_distance = np.mean(inter_dists)
        else:
            inter_distance = 0
    else:
        # Without Y, just compute overall variance
        n_subset = min(100, n)
        distances = pairwise_distances(X_preprocessed[:n_subset])
        intra_distance = np.mean(distances[np.tril_indices(len(distances), -1)])
        inter_distance = intra_distance  # Cannot distinguish without Y

    separation_ratio = inter_distance / intra_distance if intra_distance > 0 else 0
    is_valid = separation_ratio >= min_separation_ratio
    reason = "" if is_valid else f"Poor separation: ratio={separation_ratio:.3f} < {min_separation_ratio}"

    return {
        'intra_distance': intra_distance,
        'inter_distance': inter_distance,
        'separation_ratio': separation_ratio,
        'is_valid': is_valid,
        'reason': reason
    }


# =============================================================================
# Stage B: Supervised Ranking (Fast, with Y)
# =============================================================================

def rv_coefficient(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    center: bool = True
) -> dict:
    """
    Compute RV coefficient between X and Y.

    RV coefficient (Renyi–Van der Waerden) measures similarity between
    the latent space of X_preprocessed and Y. Works for regression.
    Ultra-fast (just matrix products).

    RV = trace(X'Y Y'X) / sqrt(trace(X'X X'X) * trace(Y'Y Y'Y))

    Args:
        X_preprocessed: Transformed spectra (n_samples, n_features)
        y: Target values (n_samples,) or (n_samples, n_targets)
        center: Whether to center the matrices

    Returns:
        dict with 'rv_score'
    """
    X = X_preprocessed.copy()
    Y = np.atleast_2d(y).T if y.ndim == 1 else y.copy()

    if center:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    # Compute Gram matrices
    XX = X @ X.T  # (n, n)
    YY = Y @ Y.T  # (n, n)

    # RV coefficient
    numerator = np.trace(XX @ YY)
    denominator = np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY))

    rv_score = numerator / denominator if denominator > 0 else 0

    return {'rv_score': rv_score}


def _centering_matrix(n: int) -> np.ndarray:
    """Create centering matrix H = I - 1/n * 1*1'"""
    return np.eye(n) - np.ones((n, n)) / n


def _hsic(K: np.ndarray, L: np.ndarray, H: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion"""
    n = K.shape[0]
    if n <= 1:
        return 0.0
    return np.trace(K @ H @ L @ H) / (n - 1) ** 2


def cka_score(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    kernel: str = 'linear',
    gamma: float = None
) -> dict:
    """
    Compute Centered Kernel Alignment between X and Y.

    CKA is widely used in deep learning to measure X↔Y relationship.
    One of the best metrics for ranking preprocessings before learning.

    CKA = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        kernel: 'linear' or 'rbf'
        gamma: RBF kernel parameter (auto if None)

    Returns:
        dict with 'cka_score'
    """
    Y = np.atleast_2d(y).T if y.ndim == 1 else y
    n = X_preprocessed.shape[0]
    H = _centering_matrix(n)

    if kernel == 'linear':
        K_X = X_preprocessed @ X_preprocessed.T
        K_Y = Y @ Y.T
    elif kernel == 'rbf':
        from sklearn.metrics.pairwise import rbf_kernel
        if gamma is None:
            gamma = 1.0 / X_preprocessed.shape[1]
        K_X = rbf_kernel(X_preprocessed, gamma=gamma)
        K_Y = rbf_kernel(Y, gamma=1.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    hsic_xy = _hsic(K_X, K_Y, H)
    hsic_xx = _hsic(K_X, K_X, H)
    hsic_yy = _hsic(K_Y, K_Y, H)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    cka = hsic_xy / denominator if denominator > 0 else 0

    return {'cka_score': cka}


def correlation_score(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    aggregation: str = 'max',
    top_k: int = 10
) -> dict:
    """
    Compute feature-wise correlations with target and aggregate.

    Preprocessings that increase global correlation are more likely to be useful.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        aggregation: 'max', 'mean', 'sum', or 'l1_norm'
        top_k: Number of top correlations to return

    Returns:
        dict with 'correlation_score', 'top_correlations', 'all_correlations'
    """
    n_features = X_preprocessed.shape[1]
    correlations = np.zeros(n_features)

    y_centered = y - y.mean()
    y_std = y.std()
    if y_std == 0:
        return {
            'correlation_score': 0,
            'top_correlations': [],
            'all_correlations': correlations
        }

    for j in range(n_features):
        x_j = X_preprocessed[:, j]
        x_centered = x_j - x_j.mean()
        x_std = x_j.std()
        if x_std > 0:
            correlations[j] = np.abs(np.dot(x_centered, y_centered) / (len(y) * x_std * y_std))

    if aggregation == 'max':
        score = np.max(correlations)
    elif aggregation == 'mean':
        score = np.mean(correlations)
    elif aggregation == 'sum':
        score = np.sum(correlations)
    elif aggregation == 'l1_norm':
        score = np.linalg.norm(correlations, ord=1)
    else:
        score = np.max(correlations)

    top_indices = np.argsort(correlations)[-top_k:][::-1]
    top_corrs = [(int(i), float(correlations[i])) for i in top_indices]

    return {
        'correlation_score': score,
        'top_correlations': top_corrs,
        'all_correlations': correlations
    }


def pls_score(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    cv_folds: int = None
) -> dict:
    """
    Evaluate preprocessing using fast PLS regression.

    Train a very fast PLS with 1-2 latent variables. No CV needed.
    Rank preprocessings by covariance captured or quick RMSE.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        n_components: Number of PLS latent variables (1-2 is fast)
        cv_folds: Number of CV folds (None = no CV, fit on all data)

    Returns:
        dict with 'pls_r2', 'pls_rmse', 'covariance_captured'
    """
    n_comp = min(n_components, X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
    if n_comp < 1:
        return {
            'pls_r2': 0.0,
            'pls_rmse': float('inf'),
            'covariance_captured': 0.0
        }

    pls = PLSRegression(n_components=n_comp)

    if cv_folds is not None and cv_folds > 1:
        # Quick cross-validation
        try:
            scores = cross_val_score(pls, X_preprocessed, y, cv=cv_folds, scoring='r2')
            pls_r2 = np.mean(scores)
        except Exception:
            pls_r2 = 0.0

        # Fit on all data for RMSE
        pls.fit(X_preprocessed, y)
        y_pred = pls.predict(X_preprocessed).ravel()
        pls_rmse = np.sqrt(mean_squared_error(y, y_pred))
    else:
        # No CV, just fit
        pls.fit(X_preprocessed, y)
        y_pred = pls.predict(X_preprocessed).ravel()
        pls_r2 = r2_score(y, y_pred)
        pls_rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Covariance captured by latent variables
    # Sum of squared covariances between X and Y scores
    try:
        X_scores = pls.x_scores_
        Y_scores = pls.y_scores_
        covariances = [np.cov(X_scores[:, i], Y_scores[:, i])[0, 1] for i in range(n_comp)]
        covariance_captured = sum(c**2 for c in covariances)
    except Exception:
        covariance_captured = 0.0

    return {
        'pls_r2': pls_r2,
        'pls_rmse': pls_rmse,
        'covariance_captured': covariance_captured
    }


# =============================================================================
# Batch Evaluation Functions
# =============================================================================

def evaluate_unsupervised(
    X_original: np.ndarray,
    X_preprocessed: np.ndarray,
    y: np.ndarray = None,
    pca_n_components: int = 10,
    min_variance_ratio: float = 0.90,
    max_first_component_ratio: float = 0.99,
    min_snr_ratio: float = 0.8,
    max_roughness_ratio: float = 10.0,
    min_separation_ratio: float = 1.0
) -> dict:
    """
    Run all Stage A (unsupervised) filters on a preprocessed dataset.

    Args:
        X_original: Original spectra
        X_preprocessed: Transformed spectra
        y: Target values (optional, for distance separation)
        **kwargs: Parameters for individual filters

    Returns:
        dict with results from all filters and overall 'is_valid' status
    """
    results = {}

    # PCA variance filter
    pca_result = pca_variance_filter(
        X_preprocessed,
        n_components=pca_n_components,
        min_variance_ratio=min_variance_ratio,
        max_first_component_ratio=max_first_component_ratio
    )
    results['pca'] = pca_result

    # SNR filter
    snr_result = snr_filter(X_original, X_preprocessed, min_snr_ratio=min_snr_ratio)
    results['snr'] = snr_result

    # Roughness filter
    roughness_result = roughness_filter(
        X_preprocessed,
        X_original=X_original,
        max_roughness_ratio=max_roughness_ratio
    )
    results['roughness'] = roughness_result

    # Distance separation filter (if y provided)
    if y is not None:
        distance_result = distance_separation_filter(
            X_preprocessed,
            y=y,
            min_separation_ratio=min_separation_ratio
        )
        results['distance'] = distance_result

    # Collect filter results (exclude summary keys)
    filter_results = {k: v for k, v in results.items() if isinstance(v, dict)}

    # Overall validity
    is_valid = all(r['is_valid'] for r in filter_results.values() if 'is_valid' in r)
    results['is_valid'] = is_valid

    # Collect reasons for invalidity
    reasons = [r['reason'] for r in filter_results.values() if r.get('reason')]
    results['reasons'] = reasons

    return results


def evaluate_supervised(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    pls_n_components: int = 2,
    pls_cv_folds: int = None,
    correlation_aggregation: str = 'max',
    cka_kernel: str = 'linear'
) -> dict:
    """
    Run all Stage B (supervised) metrics on a preprocessed dataset.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        **kwargs: Parameters for individual metrics

    Returns:
        dict with results from all metrics
    """
    results = {}

    # RV coefficient
    rv_result = rv_coefficient(X_preprocessed, y)
    results['rv'] = rv_result

    # CKA score
    cka_result = cka_score(X_preprocessed, y, kernel=cka_kernel)
    results['cka'] = cka_result

    # Correlation score
    corr_result = correlation_score(X_preprocessed, y, aggregation=correlation_aggregation)
    results['correlation'] = corr_result

    # PLS score
    pls_result = pls_score(X_preprocessed, y, n_components=pls_n_components, cv_folds=pls_cv_folds)
    results['pls'] = pls_result

    # Compute composite score (normalized sum)
    scores = [
        rv_result['rv_score'],
        cka_result['cka_score'],
        corr_result['correlation_score'],
        max(0, pls_result['pls_r2'])  # Clamp negative R2
    ]
    results['composite_score'] = np.mean(scores)

    return results
