# nirs4all/data_splitters/_helpers.py

import importlib
from sklearn.utils import indexable
from sklearn.model_selection import ShuffleSplit

from ._splitter import (
    SystematicCircularSplitter,
    KBinsStratifiedSplitter,
    KMeansSplitter,
    KennardStoneSplitter,
    SPXYSplitter,
    SPlitSplitter,
)

def train_test_split_idx(
    *x,
    y=None,
    test_size=None,
    method="random",
    random_state=None,
    metric="euclidean",
    pca_components=None,
    n_bins=10,
    train_size=None,
):
    """
    Split the data into training and test sets based on the specified method.

    Parameters
    ----------
    *x : array-like
        Input arrays to be split. Can be one or more arrays.
    y : array-like, optional
        The target variable.
    test_size : float or int, optional
        If float, represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of samples to include in the test split.
        Defaults to None.
    method : {'random', 'stratified', 'k_mean', 'kennard_stone', 'spxy', 'circular', 'SPlit'}, optional
        The method used for splitting the data. Defaults to 'random'.
    random_state : int, RandomState instance or None, optional
        Determines random number generation for dataset shuffling. Pass an int for reproducible output
        across multiple function calls. Defaults to None.
    metric : str, optional
        The distance metric to use. Defaults to 'euclidean'.
    pca_components : int or None, optional
        The number of components to keep in PCA transformation. If None, no PCA transformation is applied.
        Defaults to None.
    n_bins : int, optional
        The number of bins for stratified sampling. Defaults to 10.
    train_size : float or int, optional
        If float, represents the proportion of the dataset to include in the train split.
        If int, represents the absolute number of samples to include in the train split.
        Defaults to None.

    Returns
    -------
    train_index : ndarray
        The indices of the samples in the training set.
    test_index : ndarray
        The indices of the samples in the test set.

    Raises
    ------
    ValueError
        If `method` is not one of the supported methods.

    """
    data = indexable(*x)[0]

    if method == "random":
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
    elif method == "stratified":
        splitter = KBinsStratifiedSplitter(
            test_size=test_size,
            random_state=random_state,
            n_bins=n_bins,
        )
    elif method == "k_mean":
        splitter = KMeansSplitter(
            test_size=test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "kennard_stone":
        splitter = KennardStoneSplitter(
            test_size=test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "spxy":
        splitter = SPXYSplitter(
            test_size=test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "circular":
        splitter = SystematicCircularSplitter(
            test_size=test_size,
            random_state=random_state,
        )
    elif method == "SPlit":
        splitter = SPlitSplitter(
            test_size=test_size,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "Argument 'method' must be one of: ['k_mean', 'kennard_stone', 'random', "
            "'SPlit', 'spxy', 'stratified', 'circular']."
        )

    splits = list(splitter.split(data, y))

    # Since n_splits=1, we take the first split
    train_index, test_index = splits[0]

    return train_index, test_index
