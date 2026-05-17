"""Vendored SPXY K-fold splitter (formerly `nirs4all.operators.splitters.SPXYFold`).

Breaks the circular dependency between aom-nirs and nirs4all. The
implementation mirrors `nirs4all/operators/splitters/splitters.py:670`
(commit history preserved in nirs4all's git log; the algorithm itself
is documented in Galvao et al. 2005, *Talanta*, 67, 736-740).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection._split import BaseCrossValidator


class _CustomSplitter(BaseCrossValidator, ABC):
    """Minimal abstract base mirroring nirs4all's CustomSplitter."""

    @abstractmethod
    def split(self, X, y=None, groups=None):  # pragma: no cover - abstract
        ...

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):  # pragma: no cover - abstract
        ...


class SPXYFold(_CustomSplitter):
    """SPXY-based K-Fold cross-validation splitter.

    Assigns samples to folds using a joint X-Y distance (SPXY algorithm)
    to ensure each fold is spatially representative of the full feature
    and target space. When ``y_metric`` is None it falls back to pure
    Kennard-Stone (X-only).

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be >= 2.
    metric : str, default="euclidean"
        Distance metric for X-space (any metric supported by
        ``scipy.spatial.distance.cdist``).
    y_metric : str or None, default="euclidean"
        Distance metric for Y-space. Use ``"hamming"`` for categorical
        targets, ``None`` for pure Kennard-Stone.
    pca_components : int or None, default=None
        If provided, apply PCA before computing X distances.
    random_state : int or None, default=None
        Used only when ``pca_components`` is set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        metric: str = "euclidean",
        y_metric: str | None = "euclidean",
        pca_components: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = int(n_splits)
        self.metric = metric
        self.y_metric = y_metric
        self.pca_components = pca_components
        self.random_state = random_state

    # ------------------------------------------------------------------ math

    def _compute_distance_matrix(self, X, y):
        if self.pca_components is not None:
            pca = PCA(self.pca_components, random_state=self.random_state)
            X = pca.fit_transform(X)

        D_X = cdist(X, X, metric=self.metric)
        max_D_X = D_X.max()
        if max_D_X > 0:
            D_X = D_X / max_D_X

        if y is not None and self.y_metric is not None:
            y = np.atleast_1d(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if self.y_metric == "hamming":
                D_Y = (y != y.T).astype(float)
                if y.shape[1] > 1:
                    D_Y = np.any(y[:, None, :] != y[None, :, :], axis=2).astype(float)
            else:
                D_Y = cdist(y, y, metric=self.y_metric)
                max_D_Y = D_Y.max()
                if max_D_Y > 0:
                    D_Y = D_Y / max_D_Y
            return D_X + D_Y
        return D_X

    def _assign_to_folds(self, D, n_splits):
        n_samples = D.shape[0]
        fold_assignment = np.full(n_samples, -1, dtype=int)

        if n_splits >= n_samples:
            for i in range(n_samples):
                fold_assignment[i] = i % n_splits
            return fold_assignment

        centroid_distances = D.mean(axis=1)
        init_indices = np.argsort(centroid_distances)[-n_splits:]
        for fold_idx, sample_idx in enumerate(init_indices):
            fold_assignment[sample_idx] = fold_idx

        remaining = set(range(n_samples)) - set(init_indices)
        fold_sizes = np.ones(n_splits, dtype=int)
        target_size = n_samples // n_splits
        max_size = target_size + (1 if n_samples % n_splits > 0 else 0)
        fold_members = [[idx] for idx in init_indices]

        while remaining:
            for fold_idx in range(n_splits):
                if not remaining:
                    break
                if fold_sizes[fold_idx] >= max_size:
                    continue
                remaining_list = list(remaining)
                min_distances = np.array([
                    D[r, fold_members[fold_idx]].min()
                    for r in remaining_list
                ])
                best_idx = remaining_list[np.argmax(min_distances)]
                fold_assignment[best_idx] = fold_idx
                fold_members[fold_idx].append(best_idx)
                fold_sizes[fold_idx] += 1
                remaining.remove(best_idx)
        return fold_assignment

    # ----------------------------------------------------------------- sklearn API

    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.y_metric is not None and y is None:
            raise ValueError(
                f"y is required when y_metric='{self.y_metric}'. "
                "Set y_metric=None for X-only (Kennard-Stone) splitting."
            )

        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        D = self._compute_distance_matrix(X, y if self.y_metric else None)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} with only {n_samples} samples."
            )

        fold_assignment = self._assign_to_folds(D, self.n_splits)
        for fold_idx in range(self.n_splits):
            test_indices = np.where(fold_assignment == fold_idx)[0]
            train_indices = np.where(fold_assignment != fold_idx)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


__all__ = ["SPXYFold"]
