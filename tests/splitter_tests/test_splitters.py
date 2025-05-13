import numpy as np
import pytest
from nirs4all.data_splitters._splitter import (
    SystematicCircularSplitter,
    KBinsStratifiedSplitter,
    KMeansSplitter,
    KennardStoneSplitter,
    SPXYSplitter,
    CustomSplitter,
    SPlitSplitter
)
from sklearn.datasets import make_classification
import sys


def test_CustomSplitter():
    class DummySplitter(CustomSplitter):
        def split(self, X, y=None, groups=None):
            pass

        def get_n_splits(self, X=None, y=None, groups=None):
            return 1

    splitter = DummySplitter()
    assert splitter.get_n_splits() == 1


def test_SystematicCircularSplitter():
    X = np.random.rand(100, 10)
    y = np.sort(np.random.rand(100, 1), axis=0)
    splitter = SystematicCircularSplitter(test_size=0.2, random_state=42)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) + len(test_idx) == 100
    assert len(train_idx) == 80


def test_KBinsStratifiedSplitter():
    X, y = make_classification(n_samples=100, n_features=10)
    y = y.reshape(-1, 1)
    splitter = KBinsStratifiedSplitter(test_size=0.2, random_state=42, n_bins=5)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) + len(test_idx) == 100


def test_KMeansSplitter():
    X = np.random.rand(100, 10)
    splitter = KMeansSplitter(test_size=0.2, random_state=42, pca_components=5)
    splits = list(splitter.split(X))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) + len(test_idx) >= 99
    assert len(train_idx) + len(test_idx) <= 100

def test_KMeansSplitter_small_train_size():
    """
    Test KMeansSplitter with a small training size.
    """
    X = np.random.rand(10, 10)
    splitter = KMeansSplitter(test_size=0.3, random_state=42)
    splits = list(splitter.split(X))
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 7 or len(train_idx) == 8
    assert len(test_idx) == 3 or len(test_idx) == 2


def test_KennardStoneSplitter():
    X = np.random.rand(100, 10)
    splitter = KennardStoneSplitter(test_size=0.2, random_state=42, pca_components=5)
    splits = list(splitter.split(X))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 80


def test_SPXYSplitter():
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    splitter = SPXYSplitter(test_size=0.2, random_state=42, pca_components=1)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 80


def test_SPXYSplitter_invalid_pca_components():
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    splitter = SPXYSplitter(test_size=0.2, random_state=42, pca_components=5)
    with pytest.raises(ValueError):
        list(splitter.split(X, y=y))


def test_SPlitSplitter(monkeypatch):
    X = np.random.rand(100, 10)
    splitter = SPlitSplitter(test_size=0.2, random_state=42)
    splits = list(splitter.split(X))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) + len(test_idx) == 100
    assert len(test_idx) == 20