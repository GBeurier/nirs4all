"""Array-level sampling utilities.

Provides sampling strategies for numpy arrays, returning selected indices.
These are lower-level than RowSelector (which works on DataFrames) and are
suitable for use in visualization/playground contexts where raw numpy arrays
are processed directly.
"""

from __future__ import annotations

import numpy as np


def random_sample(
    n_total: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Select random sample indices.

    Args:
        n_total: Total number of samples available.
        n_samples: Number to select (capped at n_total).
        seed: Random seed for reproducibility.

    Returns:
        1D int array of selected indices.
    """
    n_select = min(n_samples, n_total)
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=n_select, replace=False)


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int = 42,
    n_bins: int = 5,
) -> np.ndarray:
    """Select stratified sample indices based on y-value quantile bins.

    Falls back to random sampling if stratification is not possible
    (too few unique y values, bins with <2 samples, etc.).

    Args:
        X: Feature matrix (n_total, n_features).
        y: Target values (n_total,).
        n_samples: Number to select (capped at n_total).
        seed: Random seed for reproducibility.
        n_bins: Number of quantile bins for stratification.

    Returns:
        1D int array of selected indices.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    n_total = X.shape[0]
    n_select = min(n_samples, n_total)
    rng = np.random.RandomState(seed)

    n_unique = len(np.unique(y))
    max_bins = min(n_bins, n_unique, n_select // 2)

    if max_bins < 2:
        return rng.choice(n_total, size=n_select, replace=False)

    y_binned = np.digitize(y, np.percentile(y, np.linspace(0, 100, max_bins + 1)[1:-1]))
    bin_counts = np.bincount(y_binned)

    if np.any(bin_counts < 2):
        return rng.choice(n_total, size=n_select, replace=False)

    try:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_select / n_total,
            random_state=seed,
        )
        _, indices = next(sss.split(X, y_binned))
        return indices
    except ValueError:
        return rng.choice(n_total, size=n_select, replace=False)


def kmeans_sample(
    X: np.ndarray,
    n_samples: int,
    seed: int = 42,
    n_init: int = 3,
) -> np.ndarray:
    """Select representative samples via MiniBatchKMeans clustering.

    Finds cluster centers, then selects the nearest unique sample to each center.

    Args:
        X: Feature matrix (n_total, n_features).
        n_samples: Number to select (= number of clusters, capped at n_total).
        seed: Random seed for reproducibility.
        n_init: Number of KMeans initializations.

    Returns:
        1D int array of selected indices.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_total = X.shape[0]
    n_select = min(n_samples, n_total)

    kmeans = MiniBatchKMeans(
        n_clusters=n_select,
        random_state=seed,
        n_init=n_init,
    )
    kmeans.fit(X)

    selected = []
    used_indices: set[int] = set()

    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X - center, axis=1)
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            if int(idx) not in used_indices:
                selected.append(int(idx))
                used_indices.add(int(idx))
                break

    return np.array(selected[:n_select])
