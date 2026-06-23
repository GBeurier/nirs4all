"""Representative subset selection over sample indices.

Each function returns a sorted array of distinct sample *indices* into the
original arrays. They share the conventions:

- ``n_select <= 0`` returns an empty index array.
- ``n_select >= n_samples`` returns every index (``arange``).
- otherwise exactly ``n_select`` distinct indices are returned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def random_sample(n_total: int, n_select: int, seed: int | None = None) -> NDArray[np.intp]:
    """Pick ``n_select`` distinct sample indices uniformly at random.

    Args:
        n_total: Total number of samples to choose from.
        n_select: Number of indices to return.
        seed: Optional seed for reproducible selection.

    Returns:
        Sorted array of distinct indices in ``[0, n_total)``.
    """
    n_total = int(n_total)
    n_select = int(n_select)
    if n_select <= 0:
        return np.empty(0, dtype=np.intp)
    if n_select >= n_total:
        return np.arange(n_total, dtype=np.intp)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_select, replace=False)).astype(np.intp)


def stratified_sample(X: ArrayLike, y: ArrayLike, n_select: int, seed: int | None = None) -> NDArray[np.intp]:
    """Pick ``n_select`` indices spread evenly across the distribution of ``y``.

    The target is split into up to ten rank-based strata (equal-count bins,
    robust to skew and ties and valid for both continuous and discrete
    targets). Indices are drawn from each stratum in proportion to its size so
    the subset mirrors the overall target distribution; any rounding shortfall
    is topped up with a uniform random draw from the unselected samples.

    Args:
        X: Feature matrix. Unused — accepted only so all sampling functions
            share a uniform signature; stratification is on the target.
        y: Target values, one per sample.
        n_select: Number of indices to return.
        seed: Optional seed for reproducible selection.

    Returns:
        Sorted array of distinct sample indices.
    """
    del X  # stratification is on the target only
    y_arr = np.asarray(y).reshape(-1)
    n_total = int(y_arr.shape[0])
    n_select = int(n_select)
    if n_select <= 0:
        return np.empty(0, dtype=np.intp)
    if n_select >= n_total:
        return np.arange(n_total, dtype=np.intp)

    rng = np.random.default_rng(seed)
    n_bins = max(1, min(n_select, 10, n_total))

    # Rank-based equal-count strata: assign each sample to a bin by its position
    # in the sorted target, which handles skew and ties without explicit edges.
    order = np.argsort(y_arr, kind="stable")
    rank = np.empty(n_total, dtype=np.int64)
    rank[order] = np.arange(n_total)
    strata = rank * n_bins // n_total

    chosen: list[NDArray[np.intp]] = []
    for bin_id in range(n_bins):
        idx = np.flatnonzero(strata == bin_id).astype(np.intp)
        if idx.size == 0:
            continue
        take = min(idx.size, int(idx.size / n_total * n_select))
        if take > 0:
            chosen.append(rng.choice(idx, size=take, replace=False))

    out = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.intp)
    if out.size < n_select:
        pool = np.setdiff1d(np.arange(n_total, dtype=np.intp), out)
        extra = rng.choice(pool, size=min(n_select - out.size, pool.size), replace=False)
        out = np.concatenate([out, extra])
    return np.sort(out).astype(np.intp)


def kmeans_sample(X: ArrayLike, n_select: int, seed: int | None = None) -> NDArray[np.intp]:
    """Pick ``n_select`` indices that are representative of the feature space.

    Clusters the samples into ``n_select`` groups with k-means and returns, for
    each non-empty cluster, the index of the sample nearest its centroid. Any
    shortfall (from empty clusters or duplicate medoids) is topped up with a
    uniform random draw from the unselected samples.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``; must be finite.
        n_select: Number of indices to return.
        seed: Optional seed for reproducible clustering and top-up.

    Returns:
        Sorted array of distinct sample indices.
    """
    x_arr = np.asarray(X, dtype=float)
    n_total = int(x_arr.shape[0])
    n_select = int(n_select)
    if n_select <= 0:
        return np.empty(0, dtype=np.intp)
    if n_select >= n_total:
        return np.arange(n_total, dtype=np.intp)

    from sklearn.cluster import KMeans

    estimator = KMeans(n_clusters=n_select, random_state=seed, n_init=10)
    labels = estimator.fit_predict(x_arr)
    centroids = estimator.cluster_centers_

    medoids: list[int] = []
    for cluster_id in range(n_select):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        dists = np.sum((x_arr[members] - centroids[cluster_id]) ** 2, axis=1)
        medoids.append(int(members[int(np.argmin(dists))]))

    out = np.unique(np.asarray(medoids, dtype=np.intp))
    if out.size < n_select:
        rng = np.random.default_rng(seed)
        pool = np.setdiff1d(np.arange(n_total, dtype=np.intp), out)
        extra = rng.choice(pool, size=min(n_select - out.size, pool.size), replace=False)
        out = np.sort(np.concatenate([out, extra]))
    return out.astype(np.intp)
