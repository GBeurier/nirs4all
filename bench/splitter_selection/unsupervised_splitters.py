"""
Unsupervised Sample Selection Splitters
========================================

This module implements unsupervised sample selection methods for splitting
spectral data into train/test sets. These methods are adapted from the work
by Fonseca Diaz et al. for use with the splitter selection pipeline.

Methods implemented:
- PuchweinSplitter: Distance-based sample selection (Puchwein algorithm)
- DuplexSplitter: Alternating selection for balanced train/test sets
- ShenkWestSplitter: Shenk & Westerhaus distance-based selection
- HonigsSplitter: Selection based on spectral uniqueness
- HierarchicalClusteringSplitter: Agglomerative clustering-based selection
- KMedoidsSplitter: K-Medoids based sample selection

References:
    Valeria Fonseca Diaz, Bart De Ketelaere, Ben Aernouts, Wouter Saeys,
    "Cost-efficient unsupervised sample selection for multivariate calibration",
    Chemometrics and Intelligent Laboratory Systems, Volume 215, 2021, 104352.
    https://doi.org/10.1016/j.chemolab.2021.104352

    Original implementation: https://github.com/vfonsecad/unsupervised-sample-selection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances


@dataclass
class SplitResult:
    """Container for split results."""
    train_ids: np.ndarray
    test_ids: np.ndarray
    fold_assignments: pd.DataFrame  # columns: ID, fold, split (train/val/test)
    strategy_info: Dict[str, Any]


class BaseSplitter(ABC):
    """Base class for all splitting strategies."""

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42
    ):
        """
        Initialize the splitter.

        Args:
            test_size: Fraction of samples (IDs) for test set
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        """
        Perform the train/test split and create CV folds.

        Args:
            X: Spectra array (n_samples, n_features)
            y: Target values (n_samples,)
            sample_ids: Sample IDs (n_samples,) - repetitions share same ID

        Returns:
            SplitResult with train/test IDs and fold assignments
        """
        pass

    def get_stratification_info(self) -> Dict[str, Any]:
        """Return information about the stratification strategy."""
        return {
            'strategy_name': self.__class__.__name__,
            'test_size': self.test_size,
            'n_folds': self.n_folds,
            'random_state': self.random_state
        }

    def _aggregate_by_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate data by sample ID (mean of repetitions).

        Returns:
            X_agg: Mean spectra per sample (n_unique_samples, n_features)
            y_agg: Mean target per sample (n_unique_samples,)
            unique_ids: Unique sample IDs (n_unique_samples,)
        """
        df = pd.DataFrame({'y': y, 'id': sample_ids})
        df_X = pd.DataFrame(X, index=df.index)
        df_X['id'] = sample_ids

        # Aggregate
        X_agg = df_X.groupby('id').mean().values
        y_agg = df.groupby('id')['y'].mean().values
        unique_ids = df.groupby('id').first().reset_index()['id'].values

        return X_agg, y_agg, unique_ids


class PuchweinSplitter(BaseSplitter):
    """
    Puchwein algorithm for sample selection.

    This method selects samples based on distance thresholds, starting from
    samples farthest from the mean and progressively removing similar samples.

    Reference:
        Puchwein, G. (1988). Selection of calibration samples for near-infrared
        spectrometry by factor analysis of spectra.
        Analytical Chemistry, 60(6), 569-573.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        factor_k: float = 0.05,
        pca_components: Optional[int] = None
    ):
        """
        Initialize Puchwein splitter.

        Args:
            test_size: Fraction of samples for test set
            n_folds: Number of CV folds
            random_state: Random seed
            factor_k: Factor by which to multiply the distance threshold
            pca_components: Number of PCA components (None for auto)
        """
        super().__init__(test_size, n_folds, random_state)
        self.factor_k = factor_k
        self.pca_components = pca_components

    def _puchwein_selection(
        self,
        X: np.ndarray,
        n_select: int
    ) -> np.ndarray:
        """
        Puchwein sample selection algorithm.

        Args:
            X: Feature matrix (n_samples, n_features)
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        n_samples = X.shape[0]
        n_components = X.shape[1]

        # Compute distances to mean
        x_mean = X.mean(axis=0).reshape(1, -1)
        hh = distance.cdist(X, x_mean, metric='euclidean')
        hh_id = np.argsort(hh, axis=0).flatten()

        # Compute pairwise distances (ordered by distance to mean)
        d = distance.cdist(X, X, metric='euclidean')[hh_id, :][:, hh_id]
        d_ini = self.factor_k * max(n_components - 2, 1)

        m = 1
        n_sel = n_samples
        sel = list(range(n_samples))

        # Iteratively reduce selection until we reach target
        while n_sel > n_select:
            dm = m * d_ini
            min_d = d[-1, :] <= dm
            sel = [hh_id[-1]]

            for ii in range(n_samples - 1, -1, -1):
                if ii not in np.where(min_d)[0]:
                    sel.append(hh_id[ii])
                    min_d = np.logical_or(min_d, d[ii, :] <= dm)

            n_sel = len(sel)
            m += 1

        return np.array(sel)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Apply PCA if specified
        if self.pca_components is not None:
            n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_agg)
        else:
            # Use Mahalanobis-like transformation (standardize by eigenvalues)
            pca = PCA(n_components=min(X_agg.shape[0] - 1, X_agg.shape[1]),
                      random_state=self.random_state)
            X_pca = pca.fit_transform(X_agg)
            # Scale by sqrt of eigenvalues for Mahalanobis distance
            X_transformed = X_pca / np.sqrt(pca.explained_variance_ + 1e-10)

        # Number of training samples to select
        n_train = int(len(unique_ids) * (1 - self.test_size))

        # Puchwein selection for training set
        train_indices = self._puchwein_selection(X_transformed, n_train)
        test_indices = np.array([i for i in range(len(unique_ids)) if i not in train_indices])

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Random fold assignment for training
        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'factor_k': self.factor_k,
                'algorithm': 'puchwein'
            }
        )


class DuplexSplitter(BaseSplitter):
    """
    Duplex algorithm for sample selection.

    This method alternates between selecting samples for training and test sets
    based on maximum distance, ensuring both sets have good coverage of the
    spectral space.

    Reference:
        Snee, R.D. (1977). Validation of regression models: methods and examples.
        Technometrics, 19(4), 415-428.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: Optional[int] = None
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components

    def _duplex_selection(
        self,
        X: np.ndarray,
        n_train: int
    ) -> tuple:
        """
        Duplex sample selection algorithm.

        Args:
            X: Feature matrix
            n_train: Number of samples for training set

        Returns:
            train_indices, test_indices
        """
        n = X.shape[0]
        n_vector = list(range(n))
        model = []  # training set
        test = []   # test set
        half = n // 2
        n_test = n - n_train

        # Handle edge cases
        if n_train <= half:
            temp_n = n_train
        else:
            temp_n = n_test

        def furthest_point_to_set(xp, xs):
            """Find point in xp furthest from all points in xs."""
            D = distance.cdist(xp, xs, metric='euclidean')
            D_min = np.argmax(np.amin(D, axis=1).flatten())
            return D_min

        # Initial selection: two furthest points for model
        current_n_vector = n_vector.copy()
        D = distance.cdist(X[n_vector, :], X[n_vector, :], metric='euclidean')
        id_d = np.unravel_index(np.argmax(D), D.shape)

        for ii in id_d:
            model.append(current_n_vector[ii])
            n_vector.remove(current_n_vector[ii])

        # Two furthest points for test
        current_n_vector = n_vector.copy()
        D = distance.cdist(X[n_vector, :], X[n_vector, :], metric='euclidean')
        id_d = np.unravel_index(np.argmax(D), D.shape)

        for ii in id_d:
            test.append(current_n_vector[ii])
            n_vector.remove(current_n_vector[ii])

        icount = len(model)

        # Alternate between model and test
        while icount < temp_n and len(n_vector) > 0:
            # Model
            current_n_vector = n_vector.copy()
            id_d = furthest_point_to_set(X[n_vector, :], X[model, :])
            model.append(current_n_vector[id_d])
            n_vector.remove(current_n_vector[id_d])

            # Test
            if len(n_vector) > 0:
                current_n_vector = n_vector.copy()
                id_d = furthest_point_to_set(X[n_vector, :], X[test, :])
                test.append(current_n_vector[id_d])
                n_vector.remove(current_n_vector[id_d])

            icount = len(model)

        # Assign remaining samples
        for sample in n_vector:
            test.append(sample)

        # If n_train > half, swap model and test
        if n_train > half:
            return np.array(test), np.array(model)
        else:
            return np.array(model), np.array(test)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Apply PCA if specified
        if self.pca_components is not None:
            n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_agg)
        else:
            pca = PCA(n_components=min(X_agg.shape[0] - 1, X_agg.shape[1]),
                      random_state=self.random_state)
            X_pca = pca.fit_transform(X_agg)
            X_transformed = X_pca / np.sqrt(pca.explained_variance_ + 1e-10)

        n_train = int(len(unique_ids) * (1 - self.test_size))

        # Duplex selection
        train_indices, test_indices = self._duplex_selection(X_transformed, n_train)

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'algorithm': 'duplex'
            }
        )


class ShenkWestSplitter(BaseSplitter):
    """
    Shenk & Westerhaus algorithm for sample selection.

    This method uses a distance threshold to select representative samples,
    automatically determining the threshold using bisection to achieve
    the target number of samples.

    Reference:
        Shenk, J.S. & Westerhaus, M.O. (1991). Population definition, sample
        selection, and calibration procedures for near infrared reflectance
        spectroscopy. Crop Science, 31(2), 469-474.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: Optional[int] = None,
        remove_outliers: bool = False
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components
        self.remove_outliers = remove_outliers

    def _shenkwest_selection(
        self,
        X: np.ndarray,
        n_select: int
    ) -> np.ndarray:
        """
        Shenk & Westerhaus sample selection algorithm.

        Args:
            X: Feature matrix (already transformed)
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        n_samples = X.shape[0]
        n_components = X.shape[1]

        # Standardize
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        n_vector = np.arange(n_samples)

        # Remove outliers if specified
        if self.remove_outliers:
            hh = distance.cdist(X_std, X_std.mean(axis=0).reshape(1, -1), metric='euclidean')
            hh = hh / n_components
            valid_mask = (hh <= 3).flatten()
            X_std = X_std[valid_mask]
            n_vector = n_vector[valid_mask]

        # Find optimal distance threshold using bisection
        a = 1e-11
        b = n_components

        while abs(a - b) > 0.001:
            factor_d_min = (a + b) / 2
            d = (distance.cdist(X_std, X_std, metric='euclidean') / n_components) < factor_d_min
            idx = np.argmax(d.sum(axis=0))
            current_trues = d[:, idx]
            current_rate = current_trues.sum() / len(n_vector)

            if current_rate < 1 / n_select:
                a = factor_d_min
            else:
                b = factor_d_min

        factor_d_min = (a + b) / 2

        # Select samples
        d = (distance.cdist(X_std, X_std, metric='euclidean') / n_components) < factor_d_min
        model = []
        current_n_vector = n_vector.copy()

        while d.shape[1] > 1 and len(model) < n_select:
            d_sum = d.sum(axis=0)
            idx = np.argmax(d_sum)
            current_trues = d[:, idx]

            if d_sum.max() == 1:
                # Add remaining samples if we haven't reached n_select
                remaining_needed = n_select - len(model)
                if remaining_needed > 0 and len(current_n_vector) > 0:
                    model.extend([int(x) for x in current_n_vector[:remaining_needed]])
                break

            model.append(int(current_n_vector[idx]))
            knn = np.where(current_trues)[0]
            keep_mask = ~current_trues
            current_n_vector = current_n_vector[keep_mask]
            d = d[keep_mask][:, keep_mask]

        # If we still don't have enough samples, add remaining ones
        if len(model) < n_select:
            remaining_indices = [i for i in range(n_samples) if i not in model]
            remaining_needed = n_select - len(model)
            model.extend(remaining_indices[:remaining_needed])

        return np.array(model, dtype=int)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Apply PCA
        if self.pca_components is not None:
            n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_agg)
        else:
            pca = PCA(n_components=min(X_agg.shape[0] - 1, X_agg.shape[1]),
                      random_state=self.random_state)
            X_pca = pca.fit_transform(X_agg)
            X_transformed = X_pca / np.sqrt(pca.explained_variance_ + 1e-10)

        n_train = int(len(unique_ids) * (1 - self.test_size))

        # ShenkWest selection for training set
        train_indices = self._shenkwest_selection(X_transformed, n_train)
        test_indices = np.array([i for i in range(len(unique_ids)) if i not in train_indices])

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'remove_outliers': self.remove_outliers,
                'algorithm': 'shenkwest'
            }
        )


class HonigsSplitter(BaseSplitter):
    """
    Honigs algorithm for sample selection.

    This method selects samples based on spectral uniqueness, iteratively
    choosing samples with maximum absolute values and removing their
    contribution from the remaining spectra.

    Reference:
        Honigs, D.E., Hieftje, G.M., Mark, H.L., Hirschfeld, T.B. (1985).
        Unique-sample selection via near-infrared spectral subtraction.
        Analytical Chemistry, 57(12), 2299-2303.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42
    ):
        super().__init__(test_size, n_folds, random_state)

    def _honigs_selection(
        self,
        X: np.ndarray,
        n_select: int
    ) -> np.ndarray:
        """
        Honigs sample selection algorithm.

        Args:
            X: Feature matrix (original spectra)
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        xx = X.copy()
        n = np.arange(n_samples)
        p = np.arange(n_features)

        model = []

        for _ in range(min(n_select, min(n_samples, n_features))):
            if xx.shape[0] == 0 or xx.shape[1] == 0:
                break

            # Find maximum absolute value
            axx = np.abs(xx)
            idx = np.unravel_index(np.argmax(axx), axx.shape)

            model.append(n[idx[0]])

            # Remove contribution and update matrices
            if xx.shape[0] > 1 and xx.shape[1] > 1:
                weight = xx[:, idx[1]] / (xx[idx[0], idx[1]] + 1e-10)
                weight = weight.reshape(-1, 1)
                current_xx = xx[idx[0], :].reshape(1, -1)
                xx2 = weight @ current_xx
                xx = xx - xx2

                # Remove selected sample and wavelength
                n = np.delete(n, idx[0])
                p = np.delete(p, idx[1])
                xx = np.delete(np.delete(xx, idx[0], axis=0), idx[1], axis=1)
            else:
                break

        return np.array(model)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        n_train = int(len(unique_ids) * (1 - self.test_size))

        # Honigs selection for training set
        train_indices = self._honigs_selection(X_agg, n_train)
        test_indices = np.array([i for i in range(len(unique_ids)) if i not in train_indices])

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'algorithm': 'honigs'
            }
        )


class HierarchicalClusteringSplitter(BaseSplitter):
    """
    Hierarchical clustering-based sample selection.

    This method uses agglomerative clustering to group samples, then selects
    the most central sample from each cluster for the training set.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: Optional[int] = None,
        linkage: str = 'complete'
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components
        self.linkage = linkage

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Apply PCA
        if self.pca_components is not None:
            n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_agg)
        else:
            pca = PCA(n_components=min(X_agg.shape[0] - 1, X_agg.shape[1]),
                      random_state=self.random_state)
            X_pca = pca.fit_transform(X_agg)
            X_transformed = X_pca / np.sqrt(pca.explained_variance_ + 1e-10)

        n_train = int(len(unique_ids) * (1 - self.test_size))
        n_clusters = n_train

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage
        ).fit(X_transformed)

        assign_clusters = clustering.labels_
        chosen_samples = []

        # Select most central sample from each cluster
        for cluster in np.unique(assign_clusters):
            cluster_samples_id = np.where(assign_clusters == cluster)[0]
            current_cluster_samples = X_transformed[cluster_samples_id, :]

            if len(cluster_samples_id) > 1:
                distances = pairwise_distances(current_cluster_samples, metric='euclidean')
                # Select sample with minimum maximum distance to others
                chosen_idx = np.argmin(np.amax(distances, axis=1))
                chosen_samples.append(cluster_samples_id[chosen_idx])
            else:
                chosen_samples.append(cluster_samples_id[0])

        train_indices = np.array(chosen_samples)
        test_indices = np.array([i for i in range(len(unique_ids)) if i not in train_indices])

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'n_clusters': n_clusters,
                'linkage': self.linkage,
                'algorithm': 'hierarchical_clustering'
            }
        )


class KMedoidsSplitter(BaseSplitter):
    """
    K-Medoids based sample selection.

    Similar to K-Means, but selects actual data points as cluster centers
    (medoids) instead of computing centroids. This ensures the selected
    samples are real observations from the dataset.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: Optional[int] = None
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Apply PCA
        if self.pca_components is not None:
            n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_agg)
        else:
            pca = PCA(n_components=min(X_agg.shape[0] - 1, X_agg.shape[1]),
                      random_state=self.random_state)
            X_pca = pca.fit_transform(X_agg)
            X_transformed = X_pca / np.sqrt(pca.explained_variance_ + 1e-10)

        n_train = int(len(unique_ids) * (1 - self.test_size))

        # Use KMeans to get initial clusters
        km = KMeans(n_clusters=n_train, init='k-means++', random_state=self.random_state, n_init=10)
        assign_clusters = km.fit_predict(X_transformed)

        chosen_samples = []

        # Select medoid (most central actual point) from each cluster
        for cluster in np.unique(assign_clusters):
            cluster_samples_id = np.where(assign_clusters == cluster)[0]
            current_cluster_samples = X_transformed[cluster_samples_id, :]

            if len(cluster_samples_id) > 1:
                distances = pairwise_distances(current_cluster_samples, metric='euclidean')
                # Select sample with minimum maximum distance to others
                chosen_idx = np.argmin(np.amax(distances, axis=1))
                chosen_samples.append(cluster_samples_id[chosen_idx])
            else:
                chosen_samples.append(cluster_samples_id[0])

        train_indices = np.array(chosen_samples)
        test_indices = np.array([i for i in range(len(unique_ids)) if i not in train_indices])

        train_ids = unique_ids[train_indices]
        test_ids = unique_ids[test_indices]

        # Create fold assignments
        fold_assignments = []

        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        train_shuffled = self.rng.permutation(train_ids)
        for i, id_ in enumerate(train_shuffled):
            fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'n_clusters': n_train,
                'algorithm': 'kmedoids'
            }
        )


# Export all splitters
__all__ = [
    'PuchweinSplitter',
    'DuplexSplitter',
    'ShenkWestSplitter',
    'HonigsSplitter',
    'HierarchicalClusteringSplitter',
    'KMedoidsSplitter'
]
