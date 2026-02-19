"""
Splitter Strategies for NIRS Spectral Data
===========================================

This module provides various splitting strategies for train/test/fold partitioning
of spectral data. All strategies respect the constraint that samples with the same
ID (and their repetitions) must stay together in the same partition.

Strategies implemented:
- SimpleSplitter: Random split at the sample (ID) level
- TargetStratifiedSplitter: Stratified by target value bins
- SpectralPCASplitter: Stratified by PCA clusters of spectra
- SpectralDistanceSplitter: Maximizes spectral diversity in test set
- HybridSplitter: Combines spectral and target stratification
- AdversarialSplitter: Creates challenging test sets for robustness evaluation
- StratifiedGroupKFoldSplitter: Uses sklearn's StratifiedGroupKFold
- Nirs4allKennardStoneSplitter: Wrapper for nirs4all's KennardStoneSplitter
- Nirs4allSPXYSplitter: Wrapper for nirs4all's SPXYSplitter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# Import nirs4all splitters
try:
    from nirs4all.operators.splitters import KennardStoneSplitter as N4AKennardStone
    from nirs4all.operators.splitters import SPXYSplitter as N4ASPXY
    HAS_NIRS4ALL_SPLITTERS = True
except ImportError:
    HAS_NIRS4ALL_SPLITTERS = False
    N4AKennardStone = None
    N4ASPXY = None

@dataclass
class SplitResult:
    """Container for split results."""
    train_ids: np.ndarray
    test_ids: np.ndarray
    fold_assignments: pd.DataFrame  # columns: ID, fold, split (train/val/test)
    strategy_info: dict[str, Any]

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

    def get_stratification_info(self) -> dict[str, Any]:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

class SimpleSplitter(BaseSplitter):
    """
    Simple random splitting at the sample (ID) level.

    Randomly assigns samples to train/test, then creates random CV folds
    within the training set. All repetitions of a sample stay together.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42
    ):
        super().__init__(test_size, n_folds, random_state)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Get unique sample IDs
        unique_ids = np.unique(sample_ids)
        n_samples = len(unique_ids)
        n_test = int(n_samples * self.test_size)

        # Shuffle and split
        shuffled_ids = self.rng.permutation(unique_ids)
        test_ids = shuffled_ids[:n_test]
        train_ids = shuffled_ids[n_test:]

        # Create fold assignments for training set
        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Training set - assign to folds
        train_shuffled = self.rng.permutation(train_ids)
        fold_sizes = np.diff(np.linspace(0, len(train_ids), self.n_folds + 1, dtype=int))

        fold_idx = 0
        current_count = 0
        for id_ in train_shuffled:
            fold_assignments.append({'ID': id_, 'fold': fold_idx, 'split': 'train'})
            current_count += 1
            if current_count >= fold_sizes[fold_idx] and fold_idx < self.n_folds - 1:
                fold_idx += 1
                current_count = 0

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info=self.get_stratification_info()
        )

class TargetStratifiedSplitter(BaseSplitter):
    """
    Stratified splitting based on target value bins.

    Ensures that the distribution of target values is similar across
    train/test and all CV folds.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        n_bins: int = 5
    ):
        super().__init__(test_size, n_folds, random_state)
        self.n_bins = n_bins

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        _, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Create target bins for stratification
        y_bins = pd.qcut(y_agg, q=self.n_bins, labels=False, duplicates='drop')

        # Split into train/test with stratification
        n_test = int(len(unique_ids) * self.test_size)

        # Stratified split
        test_ids = []
        train_ids = []

        for bin_val in np.unique(y_bins):
            bin_mask = y_bins == bin_val
            bin_ids = unique_ids[bin_mask]
            n_bin_test = max(1, int(len(bin_ids) * self.test_size))

            shuffled = self.rng.permutation(bin_ids)
            test_ids.extend(shuffled[:n_bin_test])
            train_ids.extend(shuffled[n_bin_test:])

        test_ids = np.array(test_ids)
        train_ids = np.array(train_ids)

        # Create stratified folds for training set
        train_mask = np.isin(unique_ids, train_ids)
        train_y_bins = y_bins[train_mask]
        train_unique_ids = unique_ids[train_mask]

        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Stratified fold assignment
        for fold_idx in range(self.n_folds):
            fold_ids = []
            for bin_val in np.unique(train_y_bins):
                bin_mask = train_y_bins == bin_val
                bin_ids = train_unique_ids[bin_mask]

                # Distribute bins across folds
                fold_size = len(bin_ids) // self.n_folds
                start_idx = fold_idx * fold_size
                end_idx = start_idx + fold_size if fold_idx < self.n_folds - 1 else len(bin_ids)

                shuffled = self.rng.permutation(bin_ids)
                fold_ids.extend(shuffled[start_idx:end_idx])

            for id_ in fold_ids:
                fold_assignments.append({'ID': id_, 'fold': fold_idx, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)
        # Remove duplicates, keeping first occurrence
        fold_df = fold_df.drop_duplicates(subset='ID', keep='first')

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={**self.get_stratification_info(), 'n_bins': self.n_bins}
        )

class SpectralPCASplitter(BaseSplitter):
    """
    Splitting based on PCA clustering of spectra.

    Clusters spectra using PCA + KMeans, then stratifies by cluster
    to ensure spectral diversity in both train and test sets.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        n_clusters: int = 5,
        pca_variance: float = 0.95
    ):
        super().__init__(test_size, n_folds, random_state)
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # PCA dimensionality reduction
        pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
        X_pca = pca.fit_transform(X_agg)

        # Cluster the PCA space
        n_clusters = min(self.n_clusters, len(unique_ids) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X_pca)

        # Stratified split by cluster
        test_ids = []
        train_ids = []

        for cluster_val in np.unique(clusters):
            cluster_mask = clusters == cluster_val
            cluster_ids = unique_ids[cluster_mask]
            n_cluster_test = max(1, int(len(cluster_ids) * self.test_size))

            shuffled = self.rng.permutation(cluster_ids)
            test_ids.extend(shuffled[:n_cluster_test])
            train_ids.extend(shuffled[n_cluster_test:])

        test_ids = np.array(test_ids)
        train_ids = np.array(train_ids)

        # Create fold assignments
        train_mask = np.isin(unique_ids, train_ids)
        train_clusters = clusters[train_mask]
        train_unique_ids = unique_ids[train_mask]

        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Stratified fold assignment by cluster
        fold_id_sets = [set() for _ in range(self.n_folds)]
        for cluster_val in np.unique(train_clusters):
            cluster_mask = train_clusters == cluster_val
            cluster_ids = train_unique_ids[cluster_mask]
            shuffled = self.rng.permutation(cluster_ids)

            # Distribute evenly across folds
            for i, id_ in enumerate(shuffled):
                fold_id_sets[i % self.n_folds].add(id_)

        for fold_idx, fold_ids in enumerate(fold_id_sets):
            for id_ in fold_ids:
                fold_assignments.append({'ID': id_, 'fold': fold_idx, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'n_clusters': n_clusters,
                'pca_variance': self.pca_variance,
                'n_pca_components': X_pca.shape[1]
            }
        )

class SpectralDistanceSplitter(BaseSplitter):
    """
    Splitting that maximizes spectral diversity in the test set.

    Uses farthest point sampling to select spectrally diverse test samples,
    ensuring the test set covers the full spectral space.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: int = 10
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

        # PCA for distance computation
        n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_agg)

        # Farthest point sampling for diverse test set
        n_test = int(len(unique_ids) * self.test_size)

        # Start with random sample
        selected_indices = [self.rng.randint(len(unique_ids))]
        remaining_indices = set(range(len(unique_ids))) - set(selected_indices)

        while len(selected_indices) < n_test:
            # Compute distances to selected points
            selected_points = X_pca[selected_indices]
            remaining_list = list(remaining_indices)
            remaining_points = X_pca[remaining_list]

            # Find point farthest from all selected points
            distances = cdist(remaining_points, selected_points).min(axis=1)
            farthest_idx = remaining_list[np.argmax(distances)]

            selected_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        test_ids = unique_ids[selected_indices]
        train_ids = unique_ids[list(remaining_indices)]

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
                'pca_components': n_components,
                'sampling_method': 'farthest_point'
            }
        )

class HybridSplitter(BaseSplitter):
    """
    Hybrid splitting combining spectral clustering and target stratification.

    Creates joint clusters based on both spectral features and target values,
    then stratifies by these combined clusters.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        n_spectral_clusters: int = 5,
        n_target_bins: int = 3,
        spectral_weight: float = 0.5
    ):
        super().__init__(test_size, n_folds, random_state)
        self.n_spectral_clusters = n_spectral_clusters
        self.n_target_bins = n_target_bins
        self.spectral_weight = spectral_weight

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Get spectral clusters
        pca = PCA(n_components=0.95, random_state=self.random_state)
        X_pca = pca.fit_transform(X_agg)

        n_clusters = min(self.n_spectral_clusters, len(unique_ids) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        spectral_clusters = kmeans.fit_predict(X_pca)

        # Get target bins
        target_bins = pd.qcut(y_agg, q=self.n_target_bins, labels=False, duplicates='drop')

        # Create combined strata (spectral_cluster * n_target_bins + target_bin)
        combined_strata = spectral_clusters * (max(target_bins) + 1) + target_bins

        # Stratified split by combined strata
        test_ids = []
        train_ids = []

        for stratum in np.unique(combined_strata):
            stratum_mask = combined_strata == stratum
            stratum_ids = unique_ids[stratum_mask]
            n_stratum_test = max(1, int(len(stratum_ids) * self.test_size))

            if len(stratum_ids) <= 1:
                # Keep in training if too few samples
                train_ids.extend(stratum_ids)
            else:
                shuffled = self.rng.permutation(stratum_ids)
                test_ids.extend(shuffled[:n_stratum_test])
                train_ids.extend(shuffled[n_stratum_test:])

        test_ids = np.array(test_ids)
        train_ids = np.array(train_ids)

        # Create fold assignments
        train_mask = np.isin(unique_ids, train_ids)
        train_strata = combined_strata[train_mask]
        train_unique_ids = unique_ids[train_mask]

        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Stratified fold assignment
        fold_id_sets = [set() for _ in range(self.n_folds)]
        for stratum in np.unique(train_strata):
            stratum_mask = train_strata == stratum
            stratum_ids = train_unique_ids[stratum_mask]
            shuffled = self.rng.permutation(stratum_ids)

            for i, id_ in enumerate(shuffled):
                fold_id_sets[i % self.n_folds].add(id_)

        for fold_idx, fold_ids in enumerate(fold_id_sets):
            for id_ in fold_ids:
                fold_assignments.append({'ID': id_, 'fold': fold_idx, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'n_spectral_clusters': n_clusters,
                'n_target_bins': self.n_target_bins,
                'spectral_weight': self.spectral_weight,
                'n_combined_strata': len(np.unique(combined_strata))
            }
        )

class AdversarialSplitter(BaseSplitter):
    """
    Adversarial splitting for robustness evaluation.

    Creates challenging test sets by selecting samples that are most different
    from the training set in terms of spectral characteristics.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        adversarial_strength: float = 0.5
    ):
        super().__init__(test_size, n_folds, random_state)
        self.adversarial_strength = adversarial_strength

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # PCA for distance computation
        pca = PCA(n_components=min(20, X_agg.shape[1], X_agg.shape[0] - 1),
                  random_state=self.random_state)
        X_pca = pca.fit_transform(X_agg)

        n_test = int(len(unique_ids) * self.test_size)

        # Compute distance to centroid
        centroid = X_pca.mean(axis=0)
        distances_to_centroid = np.linalg.norm(X_pca - centroid, axis=1)

        # Mix adversarial (far from centroid) with random selection
        n_adversarial = int(n_test * self.adversarial_strength)
        n_random = n_test - n_adversarial

        # Select adversarial samples (farthest from centroid)
        sorted_indices = np.argsort(distances_to_centroid)[::-1]
        adversarial_indices = sorted_indices[:n_adversarial]

        # Select random samples from remaining
        remaining = set(range(len(unique_ids))) - set(adversarial_indices)
        random_indices = self.rng.choice(list(remaining), size=n_random, replace=False)

        test_indices = np.concatenate([adversarial_indices, random_indices])
        train_indices = np.array(list(set(range(len(unique_ids))) - set(test_indices)))

        test_ids = unique_ids[test_indices]
        train_ids = unique_ids[train_indices]

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
                'adversarial_strength': self.adversarial_strength,
                'n_adversarial_samples': n_adversarial
            }
        )

class KennardStoneSplitter(BaseSplitter):
    """
    Kennard-Stone algorithm for sample selection.

    Classic algorithm in chemometrics that selects samples to maximize
    coverage of the spectral space. Test set is selected first to ensure
    good coverage, then training set is the remainder.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: int = 10
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components

    def _kennard_stone(self, X: np.ndarray, n_select: int) -> np.ndarray:
        """
        Kennard-Stone selection algorithm.

        Args:
            X: Feature matrix (n_samples, n_features)
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        n_samples = X.shape[0]

        # Compute pairwise distances
        dist_matrix = cdist(X, X)

        # Start with the two most distant samples
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        selected = list(max_dist_idx)
        remaining = set(range(n_samples)) - set(selected)

        while len(selected) < n_select:
            # Find point with maximum minimum distance to selected points
            min_distances = np.full(n_samples, np.inf)
            for idx in remaining:
                min_dist = min(dist_matrix[idx, s] for s in selected)
                min_distances[idx] = min_dist

            # Select the point with maximum minimum distance
            for idx in selected:
                min_distances[idx] = -np.inf

            new_idx = np.argmax(min_distances)
            selected.append(new_idx)
            remaining.remove(new_idx)

        return np.array(selected)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # PCA
        n_components = min(self.pca_components, X_agg.shape[1], X_agg.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_agg)

        n_test = int(len(unique_ids) * self.test_size)

        # Kennard-Stone selection for test set
        test_indices = self._kennard_stone(X_pca, n_test)
        train_indices = np.array(list(set(range(len(unique_ids))) - set(test_indices)))

        test_ids = unique_ids[test_indices]
        train_ids = unique_ids[train_indices]

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
                'pca_components': n_components,
                'algorithm': 'kennard_stone'
            }
        )

class StratifiedGroupKFoldSplitter(BaseSplitter):
    """
    Splitter using sklearn's StratifiedGroupKFold.

    Uses stratified k-fold cross-validation with non-overlapping groups.
    Target values are binned for stratification.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        n_bins: int = 5
    ):
        super().__init__(test_size, n_folds, random_state)
        self.n_bins = n_bins

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        _, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Create target bins for stratification
        y_bins = pd.qcut(y_agg, q=self.n_bins, labels=False, duplicates='drop')

        # First split: train/test using GroupShuffleSplit
        n_test = int(len(unique_ids) * self.test_size)
        shuffled_ids = self.rng.permutation(unique_ids)
        test_ids = shuffled_ids[:n_test]
        train_ids = shuffled_ids[n_test:]

        # Get training data indices for StratifiedGroupKFold
        train_mask = np.isin(unique_ids, train_ids)
        train_y_bins = y_bins[train_mask]
        train_unique_ids = unique_ids[train_mask]

        # Use StratifiedGroupKFold for fold assignment
        sgkf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_assignments = []

        # Test set
        for id_ in test_ids:
            fold_assignments.append({'ID': id_, 'fold': -1, 'split': 'test'})

        # Assign folds using StratifiedGroupKFold
        # We need X_train for the split method, but we only care about the fold indices
        X_train_ids = np.arange(len(train_unique_ids)).reshape(-1, 1)

        try:
            for fold_idx, (_, val_idx) in enumerate(sgkf.split(X_train_ids, train_y_bins, train_unique_ids)):
                for idx in val_idx:
                    fold_assignments.append({
                        'ID': train_unique_ids[idx],
                        'fold': fold_idx,
                        'split': 'train'
                    })
        except ValueError:
            # Fallback to simple fold assignment if stratification fails
            train_shuffled = self.rng.permutation(train_unique_ids)
            for i, id_ in enumerate(train_shuffled):
                fold_assignments.append({'ID': id_, 'fold': i % self.n_folds, 'split': 'train'})

        fold_df = pd.DataFrame(fold_assignments)
        fold_df = fold_df.drop_duplicates(subset='ID', keep='first')

        return SplitResult(
            train_ids=train_ids,
            test_ids=test_ids,
            fold_assignments=fold_df,
            strategy_info={
                **self.get_stratification_info(),
                'n_bins': self.n_bins,
                'algorithm': 'stratified_group_kfold'
            }
        )

class Nirs4allKennardStoneSplitter(BaseSplitter):
    """
    Wrapper for nirs4all's KennardStoneSplitter.

    Uses the Kennard-Stone algorithm implementation from nirs4all library.
    Respects sample grouping by aggregating repetitions before selection.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: int = None,
        metric: str = 'euclidean'
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components
        self.metric = metric

        if not HAS_NIRS4ALL_SPLITTERS:
            raise ImportError("nirs4all splitters not available. Install nirs4all.")

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Use nirs4all's KennardStoneSplitter
        n4a_splitter = N4AKennardStone(
            test_size=self.test_size,
            random_state=self.random_state,
            pca_components=self.pca_components,
            metric=self.metric
        )

        # Get train/test split
        for train_idx, test_idx in n4a_splitter.split(X_agg, y_agg.reshape(-1, 1)):
            train_ids = unique_ids[train_idx]
            test_ids = unique_ids[test_idx]
            break

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
                'pca_components': self.pca_components,
                'metric': self.metric,
                'algorithm': 'nirs4all_kennard_stone'
            }
        )

class Nirs4allSPXYSplitter(BaseSplitter):
    """
    Wrapper for nirs4all's SPXYSplitter.

    SPXY (Sample set Partitioning based on joint X-Y distances) selects
    samples considering both spectral (X) and target (Y) information.
    Respects sample grouping by aggregating repetitions before selection.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_folds: int = 3,
        random_state: int = 42,
        pca_components: int = None,
        metric: str = 'euclidean'
    ):
        super().__init__(test_size, n_folds, random_state)
        self.pca_components = pca_components
        self.metric = metric

        if not HAS_NIRS4ALL_SPLITTERS:
            raise ImportError("nirs4all splitters not available. Install nirs4all.")

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray
    ) -> SplitResult:
        # Aggregate by sample
        X_agg, y_agg, unique_ids = self._aggregate_by_sample(X, y, sample_ids)

        # Use nirs4all's SPXYSplitter
        n4a_splitter = N4ASPXY(
            test_size=self.test_size,
            random_state=self.random_state,
            pca_components=self.pca_components,
            metric=self.metric
        )

        # Get train/test split
        for train_idx, test_idx in n4a_splitter.split(X_agg, y_agg.reshape(-1, 1)):
            train_ids = unique_ids[train_idx]
            test_ids = unique_ids[test_idx]
            break

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
                'pca_components': self.pca_components,
                'metric': self.metric,
                'algorithm': 'nirs4all_spxy'
            }
        )

# Import unsupervised sample selection splitters (imported after class definitions to avoid circular imports)
from unsupervised_splitters import DuplexSplitter, HierarchicalClusteringSplitter, HonigsSplitter, KMedoidsSplitter, PuchweinSplitter, ShenkWestSplitter

# Strategy registry
SPLITTING_STRATEGIES = {
    'simple': {
        'class': SimpleSplitter,
        'name': 'Simple Random',
        'category': 'Baseline',
        'description': 'Random split at sample ID level'
    },
    'target_stratified': {
        'class': TargetStratifiedSplitter,
        'name': 'Target Stratified',
        'category': 'Target-Based',
        'description': 'Stratified by target value bins'
    },
    'spectral_pca': {
        'class': SpectralPCASplitter,
        'name': 'Spectral PCA',
        'category': 'Spectral-Based',
        'description': 'Stratified by PCA clusters of spectra'
    },
    'spectral_distance': {
        'class': SpectralDistanceSplitter,
        'name': 'Spectral Distance',
        'category': 'Spectral-Based',
        'description': 'Farthest point sampling for spectral diversity'
    },
    'hybrid': {
        'class': HybridSplitter,
        'name': 'Hybrid Spectral+Target',
        'category': 'Hybrid',
        'description': 'Combined spectral and target stratification'
    },
    'adversarial': {
        'class': AdversarialSplitter,
        'name': 'Adversarial',
        'category': 'Robustness',
        'description': 'Challenging test sets with outlier samples'
    },
    'kennard_stone': {
        'class': KennardStoneSplitter,
        'name': 'Kennard-Stone',
        'category': 'Chemometrics',
        'description': 'Classic chemometric sample selection algorithm'
    },
    'stratified_group_kfold': {
        'class': StratifiedGroupKFoldSplitter,
        'name': 'Stratified Group KFold',
        'category': 'Stratified',
        'description': 'Sklearn StratifiedGroupKFold for stratified CV with groups'
    },
    'nirs4all_kennard_stone': {
        'class': Nirs4allKennardStoneSplitter,
        'name': 'Nirs4all Kennard-Stone',
        'category': 'Chemometrics',
        'description': 'Kennard-Stone from nirs4all library'
    },
    'nirs4all_spxy': {
        'class': Nirs4allSPXYSplitter,
        'name': 'Nirs4all SPXY',
        'category': 'Chemometrics',
        'description': 'SPXY sampling from nirs4all library (X+Y based)'
    },
    'puchwein': {
        'class': PuchweinSplitter,
        'name': 'Puchwein',
        'category': 'Chemometrics',
        'description': 'Puchwein distance-based sample selection'
    },
    'duplex': {
        'class': DuplexSplitter,
        'name': 'Duplex',
        'category': 'Chemometrics',
        'description': 'Duplex alternating train/test selection'
    },
    'shenkwest': {
        'class': ShenkWestSplitter,
        'name': 'Shenk-Westerhaus',
        'category': 'Chemometrics',
        'description': 'Shenk & Westerhaus distance-based selection'
    },
    'honigs': {
        'class': HonigsSplitter,
        'name': 'Honigs',
        'category': 'Chemometrics',
        'description': 'Honigs spectral uniqueness selection'
    },
    'hierarchical_clustering': {
        'class': HierarchicalClusteringSplitter,
        'name': 'Hierarchical Clustering',
        'category': 'Clustering',
        'description': 'Agglomerative clustering-based selection'
    },
    'kmedoids': {
        'class': KMedoidsSplitter,
        'name': 'K-Medoids',
        'category': 'Clustering',
        'description': 'K-Medoids based sample selection'
    }
}

def get_splitter(
    strategy_name: str,
    test_size: float = 0.2,
    n_folds: int = 3,
    random_state: int = 42,
    **kwargs
) -> BaseSplitter:
    """
    Factory function to get a splitter by name.

    Args:
        strategy_name: Name of the splitting strategy
        test_size: Fraction of samples for test set
        n_folds: Number of CV folds
        random_state: Random seed
        **kwargs: Additional arguments for specific splitters

    Returns:
        Configured splitter instance
    """
    if strategy_name not in SPLITTING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(SPLITTING_STRATEGIES.keys())}")

    splitter_class = SPLITTING_STRATEGIES[strategy_name]['class']
    return splitter_class(
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state,
        **kwargs
    )

def list_strategies() -> dict[str, dict[str, str]]:
    """List all available splitting strategies with descriptions."""
    return {
        key: {
            'name': info['name'],
            'category': info['category'],
            'description': info['description']
        }
        for key, info in SPLITTING_STRATEGIES.items()
    }
