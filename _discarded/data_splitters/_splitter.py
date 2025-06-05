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
from sklearn.model_selection import StratifiedShuffleSplit
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
