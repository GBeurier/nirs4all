import importlib
import random as rd
from abc import ABC, abstractmethod
from math import ceil, floor

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import KBinsDiscretizer
from twinning import twin


def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """
    Validation helper to check if the train/test sizes are meaningful w.r.t. the
    size of the data (n_samples).
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )

    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError(
            "The sum of train_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "train_size." % (n_train + n_test, n_samples)
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, test_size={} and train_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, test_size, train_size)
        )

    # Ensure that the sum of n_train and n_test equals n_samples
    if n_train + n_test != n_samples:
        n_test = n_samples - n_train

    return n_train, n_test


class CustomSplitter(BaseCrossValidator, ABC):
    """
    Abstract base class for custom splitters.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def split(self, X, y=None, groups=None):
        pass

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        pass


class SystematicCircularSplitter(CustomSplitter):
    """
    Implements the systematic circular sampling method.
    """

    def __init__(self, test_size, random_state=None):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = 1  # Since it's a single split

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("Y data are required to use systematic circular sampling")

        if self.random_state is not None:
            rd.seed(self.random_state)

        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, None)

        ordered_idx = np.argsort(y[:, 0], axis=0)
        rotated_idx = np.roll(ordered_idx, rd.randint(0, n_samples))

        step = n_samples / n_train
        indices = [round(step * i) for i in range(n_train)]

        index_train = rotated_idx[indices]
        index_test = np.delete(rotated_idx, indices)
        yield index_train, index_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class KBinsStratifiedSplitter(CustomSplitter):
    """
    Implements stratified sampling using KBins discretization.
    """

    def __init__(self, test_size, random_state=None, n_bins=10, strategy="uniform", encode="ordinal"):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.n_splits = 1  # Single split

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("Y data are required to use KBins stratified sampling")

        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy,
                                       subsample=200000)
        y_discrete = discretizer.fit_transform(y)

        split_model = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        for train_idx, test_idx in split_model.split(X, y_discrete):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class BinnedStratifiedGroupKFold(CustomSplitter):
    """
    Stratified Group K-Fold cross-validator with binned continuous targets.

    This splitter combines:
    - KBinsDiscretizer to bin continuous y values into discrete categories
    - StratifiedGroupKFold to ensure stratified splits while respecting groups

    This is useful for regression tasks where you want stratified sampling
    (balanced target distribution across folds) while ensuring samples from
    the same group are never split across train and test sets.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_bins : int, default=10
        Number of bins for discretizing continuous y values.
        More bins = finer stratification but may fail with small datasets.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins:
        - 'uniform': All bins have identical widths.
        - 'quantile': All bins have the same number of points (recommended for
          imbalanced distributions).
        - 'kmeans': Values in each bin have the same nearest center of a 1D
          k-means cluster.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting.

    random_state : int or None, default=None
        Random state for reproducibility when shuffle=True.

    Examples
    --------
    Basic usage with regression targets and groups:

    >>> from nirs4all.operators.splitters import BinnedStratifiedGroupKFold
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)  # Continuous target
    >>> groups = np.repeat(np.arange(20), 5)  # 20 groups, 5 samples each
    >>> splitter = BinnedStratifiedGroupKFold(n_splits=5, n_bins=5)
    >>> for train_idx, test_idx in splitter.split(X, y, groups):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    With quantile binning for imbalanced targets:

    >>> splitter = BinnedStratifiedGroupKFold(
    ...     n_splits=3,
    ...     n_bins=10,
    ...     strategy='quantile',
    ...     shuffle=True,
    ...     random_state=42
    ... )

    Notes
    -----
    - The number of bins should be chosen based on the dataset size and the
      number of unique groups. Too many bins may cause stratification to fail.
    - Groups are never split across folds - all samples from a group will be
      in either train or test, never both.
    - Stratification is approximate when groups have varying sizes.

    See Also
    --------
    KBinsStratifiedSplitter : Single train/test split with binned stratification.
    sklearn.model_selection.StratifiedGroupKFold : For categorical targets.
    """

    def __init__(
        self,
        n_splits=5,
        n_bins=10,
        strategy="quantile",
        shuffle=False,
        random_state=None
    ):
        super().__init__()
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.strategy = strategy
        self.shuffle = shuffle
        self.random_state = random_state

        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        if strategy not in ("uniform", "quantile", "kmeans"):
            raise ValueError(
                f"strategy must be 'uniform', 'quantile', or 'kmeans', got '{strategy}'"
            )

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Continuous target values to be binned for stratification.

        groups : array-like of shape (n_samples,)
            Group labels for samples. Samples with the same group label
            will always be in the same fold.

        Yields
        ------
        train : ndarray
            Training set indices for this fold.
        test : ndarray
            Test set indices for this fold.
        """
        if y is None:
            raise ValueError("y is required for BinnedStratifiedGroupKFold")
        if groups is None:
            raise ValueError("groups is required for BinnedStratifiedGroupKFold")

        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Bin the continuous y values
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
            subsample=200000
        )
        y_binned = discretizer.fit_transform(y).ravel().astype(int)

        # Use sklearn's StratifiedGroupKFold with binned y
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for train_idx, test_idx in sgkf.split(X, y_binned, groups):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.
        groups : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of folds.
        """
        return self.n_splits


class KMeansSplitter(CustomSplitter):
    """
    Implements sampling using K-Means clustering.
    """

    def __init__(self, test_size, random_state=None, pca_components=None, metric="euclidean"):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.pca_components = pca_components
        self.metric = metric
        self.n_splits = 1  # Single split

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, None)

        if self.pca_components is not None:
            pca = PCA(self.pca_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X)
        else:
            X_transformed = X

        kmean = KMeans(n_clusters=n_train, random_state=self.random_state, n_init=10)
        kmean.fit(X_transformed)
        centroids = kmean.cluster_centers_

        index_train = np.zeros(n_samples, dtype=int)
        for i, centroid in enumerate(centroids):
            tmp_array = cdist(X_transformed, [centroid], metric=self.metric).flatten()
            closest_idx = np.argmin(tmp_array)
            index_train[i] = closest_idx

        index_train = np.unique(index_train).astype(int)
        index_test = np.delete(np.arange(n_samples), index_train)

        # Ensure that the number of training and testing samples is correct
        if len(index_train) > n_train:
            index_train = index_train[:n_train]
        if len(index_test) > n_test:
            index_test = index_test[:n_test]

        yield index_train, index_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class KennardStoneSplitter(CustomSplitter):
    """
    Implements the Kennard-Stone sampling method based on maximum minimum distance.
    """

    def __init__(self, test_size, random_state=None, pca_components=None, metric="euclidean"):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.pca_components = pca_components
        self.metric = metric
        self.n_splits = 1  # Single split

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, _ = _validate_shuffle_split(n_samples, self.test_size, None)

        if self.pca_components is not None:
            pca = PCA(self.pca_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X)
        else:
            X_transformed = X

        if n_train < 2:
            raise ValueError("Train sample size should be at least 2.")

        distance = cdist(X_transformed, X_transformed, metric=self.metric)
        train_indices, test_indices = self._max_min_distance_split(distance, n_train)
        yield train_indices, test_indices

    def _max_min_distance_split(self, distance, train_size):
        index_train = np.array([], dtype=int)
        index_test = np.arange(distance.shape[0], dtype=int)

        # Select the two farthest points
        first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
        index_train = np.append(index_train, first_2pts[0])
        index_train = np.append(index_train, first_2pts[1])

        # Remove selected points from test indices
        index_test = np.delete(index_test, np.where(index_test == first_2pts[0]))
        index_test = np.delete(index_test, np.where(index_test == first_2pts[1]))

        for _ in range(train_size - 2):
            min_distances = distance[index_train].min(axis=0)
            next_point = np.argmax(min_distances[index_test])
            selected = index_test[next_point]
            index_train = np.append(index_train, selected)
            index_test = np.delete(index_test, next_point)

        return index_train, index_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SPXYSplitter(CustomSplitter):
    """
    Implements the SPXY sampling method.
    """

    def __init__(self, test_size, random_state=None, pca_components=None, metric="euclidean"):
        """
        metric : str or callable, optional
            The distance metric to use. If a string, the distance function can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'.
        """
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.pca_components = pca_components
        self.metric = metric
        self.n_splits = 1  # Single split

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("Y data are required to use SPXY sampling")

        n_samples = _num_samples(X)
        n_train, _ = _validate_shuffle_split(n_samples, self.test_size, None)

        if self.pca_components is not None:
            pca = PCA(self.pca_components, random_state=self.random_state)
            X_transformed = pca.fit_transform(X)
            y_transformed = pca.fit_transform(y.reshape(-1, 1)) if y.ndim == 1 else pca.fit_transform(y)
        else:
            X_transformed = X
            y_transformed = y

        if n_train < 2:
            raise ValueError("Train sample size should be at least 2.")

        distance_features = cdist(X_transformed, X_transformed, metric=self.metric)
        distance_features /= distance_features.max()

        distance_labels = cdist(y_transformed, y_transformed, metric=self.metric)
        distance_labels /= distance_labels.max()

        distance = distance_features + distance_labels

        train_indices, test_indices = self._max_min_distance_split(distance, n_train)
        yield train_indices, test_indices

    def _max_min_distance_split(self, distance, train_size):
        index_train = np.array([], dtype=int)
        index_test = np.arange(distance.shape[0], dtype=int)

        # Select the two farthest points
        first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
        index_train = np.append(index_train, first_2pts[0])
        index_train = np.append(index_train, first_2pts[1])

        # Remove selected points from test indices
        index_test = np.delete(index_test, np.where(index_test == first_2pts[0]))
        index_test = np.delete(index_test, np.where(index_test == first_2pts[1]))

        for _ in range(train_size - 2):
            min_distances = distance[index_train].min(axis=0)
            next_point = np.argmax(min_distances[index_test])
            selected = index_test[next_point]
            index_train = np.append(index_train, selected)
            index_test = np.delete(index_test, next_point)

        return index_train, index_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SPlitSplitter(CustomSplitter):
    """
    Implements the SPlit sampling.
    """

    def __init__(self, test_size, random_state=None):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = 1  # Single split

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        # n_features = X.shape[1]
        # n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, None)

        r = int(1 / self.test_size)
        index_test = twin(X, r)
        index_train = np.delete(np.arange(n_samples), index_test)
        yield index_train, index_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SPXYGFold(CustomSplitter):
    """
    SPXY-based K-Fold splitter with group awareness.

    Combines:
    - SPXY (joint X-Y distance) or Kennard-Stone (X-only) selection
    - Group constraints (samples in same group stay together)
    - K-fold cross-validation

    This splitter extends the SPXY algorithm to support:
    1. Classification tasks (using appropriate distance metrics for categorical y)
    2. Group-aware splitting (treating groups as atomic units)
    3. K-fold cross-validation (not just single train/test split)

    The algorithm ensures uniform coverage of the feature space (and optionally
    target space) across all folds, which is particularly useful for spectroscopy
    data where sample distribution matters for model generalization.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds for cross-validation. Use 1 for single train/test split.
        Must be at least 2 for cross-validation.

    test_size : float, default=None
        Proportion of samples for test set. Only used when n_splits=1.
        If None with n_splits=1, defaults to 0.25.

    metric : str, default="euclidean"
        Distance metric for X-space. Any metric supported by scipy.spatial.distance.cdist:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    y_metric : str or None, default="euclidean"
        Distance metric for Y-space.
        - "euclidean": For regression (continuous y) - default SPXY behavior
        - "hamming": For classification (categorical y) - treats all class
          differences equally
        - None: Ignore Y (pure Kennard-Stone, X-only selection)

    aggregation : str, default="mean"
        Method for group aggregation when groups are provided:
        - "mean": Use group centroid (mean of all samples in group)
        - "median": Use group median (robust to outliers)

    pca_components : int or None, default=None
        If provided, apply PCA to reduce X dimensionality before distance
        computation. Useful for high-dimensional spectral data.

    random_state : int or None, default=None
        Random state for reproducibility. Only used for tie-breaking when
        multiple samples have equal distances.

    Examples
    --------
    Basic K-Fold with SPXY:

    >>> from nirs4all.operators.splitters import SPXYGFold
    >>> splitter = SPXYGFold(n_splits=5)
    >>> for train_idx, test_idx in splitter.split(X, y):
    ...     X_train, X_test = X[train_idx], X[test_idx]

    Single train/test split (backward compatible with SPXYSplitter):

    >>> splitter = SPXYGFold(n_splits=1, test_size=0.25)
    >>> train_idx, test_idx = next(splitter.split(X, y))

    Classification with Hamming distance for y:

    >>> splitter = SPXYGFold(n_splits=5, y_metric="hamming")
    >>> for train_idx, test_idx in splitter.split(X, y_class):
    ...     pass

    Group-aware splitting:

    >>> splitter = SPXYGFold(n_splits=5)
    >>> for train_idx, test_idx in splitter.split(X, y, groups=sample_ids):
    ...     pass  # Samples with same group stay together

    Pure Kennard-Stone (X-only):

    >>> splitter = SPXYGFold(n_splits=5, y_metric=None)
    >>> for train_idx, test_idx in splitter.split(X):
    ...     pass

    References
    ----------
    .. [1] Kennard, R.W. & Stone, L.A. (1969). "Computer Aided Design of
       Experiments." Technometrics, 11(1), 137-148.

    .. [2] Galvão, R.K.H., et al. (2005). "A method for calibration and
       validation subset partitioning." Talanta, 67(4), 736-740.
    """

    def __init__(
        self,
        n_splits=5,
        test_size=None,
        metric="euclidean",
        y_metric="euclidean",
        aggregation="mean",
        pca_components=None,
        random_state=None
    ):
        super().__init__()
        self.n_splits = n_splits
        self.test_size = test_size
        self.metric = metric
        self.y_metric = y_metric
        self.aggregation = aggregation
        self.pca_components = pca_components
        self.random_state = random_state

        # Validate parameters
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation must be 'mean' or 'median', got {aggregation}")

    def _aggregate_groups(self, X, y, groups):
        """Aggregate samples by group, returning representatives and index mapping.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs) or None
            Target values.
        groups : array-like of shape (n_samples,)
            Group labels for each sample.

        Returns
        -------
        X_rep : ndarray of shape (n_groups, n_features)
            Representative features for each group.
        y_rep : ndarray of shape (n_groups,) or (n_groups, n_outputs) or None
            Representative targets for each group.
        group_indices : list of lists
            For each group, the list of sample indices belonging to it.
        unique_groups : ndarray
            Unique group labels in order.
        """
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Compute group representatives for X
        X_rep = np.zeros((n_groups, X.shape[1]))
        group_indices = []

        for i, g in enumerate(unique_groups):
            mask = groups == g
            indices = np.where(mask)[0].tolist()
            group_indices.append(indices)

            if self.aggregation == "mean":
                X_rep[i] = X[mask].mean(axis=0)
            else:  # median
                X_rep[i] = np.median(X[mask], axis=0)

        # Compute group representatives for y
        y_rep = None
        if y is not None:
            y = np.atleast_1d(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            y_rep = np.zeros((n_groups, y.shape[1]))
            for i, g in enumerate(unique_groups):
                mask = groups == g
                if self.y_metric == "hamming":
                    # For classification: use mode (most common value)
                    from scipy import stats
                    for j in range(y.shape[1]):
                        mode_result = stats.mode(y[mask, j], keepdims=True)
                        y_rep[i, j] = mode_result.mode[0]
                else:
                    # For regression: use mean/median
                    if self.aggregation == "mean":
                        y_rep[i] = y[mask].mean(axis=0)
                    else:
                        y_rep[i] = np.median(y[mask], axis=0)

        return X_rep, y_rep, group_indices, unique_groups

    def _compute_distance_matrix(self, X, y):
        """Compute combined X+Y distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs) or None
            Target values.

        Returns
        -------
        D : ndarray of shape (n_samples, n_samples)
            Combined distance matrix.
        """
        # Apply PCA if requested
        if self.pca_components is not None:
            pca = PCA(self.pca_components, random_state=self.random_state)
            X = pca.fit_transform(X)

        # Compute X distance
        D_X = cdist(X, X, metric=self.metric)
        max_D_X = D_X.max()
        if max_D_X > 0:
            D_X = D_X / max_D_X

        # Compute Y distance if requested
        if y is not None and self.y_metric is not None:
            y = np.atleast_1d(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if self.y_metric == "hamming":
                # For classification: binary distance (0 if same class, 1 if different)
                # Works correctly for multi-class: any difference = 1
                D_Y = (y != y.T).astype(float)
                if y.shape[1] > 1:
                    # Multi-output: average across outputs
                    D_Y = np.any(y[:, None, :] != y[None, :, :], axis=2).astype(float)
            else:
                # For regression: standard distance metric
                D_Y = cdist(y, y, metric=self.y_metric)
                max_D_Y = D_Y.max()
                if max_D_Y > 0:
                    D_Y = D_Y / max_D_Y

            D = D_X + D_Y
        else:
            D = D_X

        return D

    def _assign_to_folds(self, D, n_splits):
        """Assign samples/groups to folds using alternating max-min algorithm.

        Parameters
        ----------
        D : ndarray of shape (n_samples, n_samples)
            Distance matrix.
        n_splits : int
            Number of folds.

        Returns
        -------
        fold_assignment : ndarray of shape (n_samples,)
            Fold index for each sample.
        """
        n_samples = D.shape[0]
        fold_assignment = np.full(n_samples, -1, dtype=int)

        if n_splits >= n_samples:
            # More folds than samples: assign one sample per fold
            for i in range(n_samples):
                fold_assignment[i] = i % n_splits
            return fold_assignment

        # Initialize: find k samples farthest from centroid
        centroid_distances = D.mean(axis=1)
        init_indices = np.argsort(centroid_distances)[-n_splits:]

        # Assign initial samples to folds (one per fold)
        for fold_idx, sample_idx in enumerate(init_indices):
            fold_assignment[sample_idx] = fold_idx

        # Track which samples are assigned and fold sizes
        remaining = set(range(n_samples)) - set(init_indices)
        fold_sizes = np.ones(n_splits, dtype=int)
        target_size = n_samples // n_splits
        max_size = target_size + (1 if n_samples % n_splits > 0 else 0)

        # Lists of samples in each fold
        fold_members = [list([idx]) for idx in init_indices]

        # Alternating assignment: cycle through folds
        while remaining:
            for fold_idx in range(n_splits):
                if not remaining:
                    break
                if fold_sizes[fold_idx] >= max_size:
                    continue

                # Compute min distance from remaining samples to this fold's members
                remaining_list = list(remaining)
                min_distances = np.array([
                    D[r, fold_members[fold_idx]].min()
                    for r in remaining_list
                ])

                # Select sample with maximum min-distance (most distant from fold)
                best_idx = remaining_list[np.argmax(min_distances)]
                fold_assignment[best_idx] = fold_idx
                fold_members[fold_idx].append(best_idx)
                fold_sizes[fold_idx] += 1
                remaining.remove(best_idx)

        return fold_assignment

    def _single_split(self, D, test_size):
        """Perform single train/test split using max-min algorithm.

        This replicates the original SPXYSplitter behavior for backward compatibility.

        Parameters
        ----------
        D : ndarray of shape (n_samples, n_samples)
            Distance matrix.
        test_size : float
            Proportion of samples for test set.

        Returns
        -------
        train_indices : ndarray
            Indices of training samples.
        test_indices : ndarray
            Indices of test samples.
        """
        n_samples = D.shape[0]
        n_train, _ = _validate_shuffle_split(n_samples, test_size, None, default_test_size=0.25)

        if n_train < 2:
            raise ValueError("Train sample size should be at least 2.")

        index_train = np.array([], dtype=int)
        index_test = np.arange(n_samples, dtype=int)

        # Select the two farthest points
        first_2pts = np.unravel_index(np.argmax(D), D.shape)
        index_train = np.append(index_train, first_2pts[0])
        index_train = np.append(index_train, first_2pts[1])

        # Remove selected points from test indices
        index_test = np.delete(index_test, np.where(index_test == first_2pts[0]))
        index_test = np.delete(index_test, np.where(index_test == first_2pts[1]))

        for _ in range(n_train - 2):
            min_distances = D[index_train].min(axis=0)
            next_point = np.argmax(min_distances[index_test])
            selected = index_test[next_point]
            index_train = np.append(index_train, selected)
            index_test = np.delete(index_test, next_point)

        return index_train, index_test

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values. Required if y_metric is not None.
        groups : array-like of shape (n_samples,), default=None
            Group labels for samples. Samples with the same group label
            will always be in the same fold.

        Yields
        ------
        train : ndarray
            Training set indices for this fold.
        test : ndarray
            Test set indices for this fold.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Validate y requirement
        if self.y_metric is not None and y is None:
            raise ValueError(
                f"y is required when y_metric='{self.y_metric}'. "
                "Set y_metric=None for X-only (Kennard-Stone) splitting."
            )

        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        # Handle groups
        if groups is not None:
            groups = np.asarray(groups)
            X_rep, y_rep, group_indices, unique_groups = self._aggregate_groups(X, y, groups)
            D = self._compute_distance_matrix(X_rep, y_rep if self.y_metric else None)
            n_units = len(unique_groups)
        else:
            D = self._compute_distance_matrix(X, y if self.y_metric else None)
            group_indices = [[i] for i in range(n_samples)]
            n_units = n_samples

        # Single split mode (backward compatible with SPXYSplitter)
        if self.n_splits == 1:
            test_size = self.test_size if self.test_size is not None else 0.25
            train_units, test_units = self._single_split(D, test_size)

            # Map back to sample indices
            train_indices = np.concatenate([group_indices[u] for u in train_units])
            test_indices = np.concatenate([group_indices[u] for u in test_units])

            yield train_indices, test_indices
            return

        # K-fold mode
        if self.n_splits > n_units:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} with only {n_units} "
                f"{'groups' if groups is not None else 'samples'}."
            )

        fold_assignment = self._assign_to_folds(D, self.n_splits)

        for fold_idx in range(self.n_splits):
            test_units = np.where(fold_assignment == fold_idx)[0]
            train_units = np.where(fold_assignment != fold_idx)[0]

            # Map back to sample indices
            train_indices = np.concatenate([group_indices[u] for u in train_units])
            test_indices = np.concatenate([group_indices[u] for u in test_units])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.
        groups : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of folds.
        """
        return self.n_splits
