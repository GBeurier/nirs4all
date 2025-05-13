import numpy as np
import pytest
from sklearn.datasets import make_classification  
from nirs4all.data_splitters._helpers import train_test_split_idx


def test_train_test_split_idx_random():
    X, y = make_classification(n_samples=100, n_features=10)
    train_idx, test_idx = train_test_split_idx(
        X, y=y, test_size=0.2, method='random', random_state=42
    )
    assert len(train_idx) + len(test_idx) == 100
    assert len(test_idx) == 20


def test_train_test_split_idx_stratified():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, n_informative=5
    )
    y = y.reshape(-1, 1)
    train_idx, test_idx = train_test_split_idx(
        X, y=y, test_size=0.2, method='stratified', random_state=42
    )
    assert len(train_idx) + len(test_idx) == 100
    assert len(test_idx) == 20


def test_train_test_split_idx_k_mean():
    X, y = make_classification(n_samples=100, n_features=10)
    train_idx, test_idx = train_test_split_idx(
        X, test_size=0.2, method='k_mean', random_state=42
    )
    assert len(train_idx) + len(test_idx) == 100 or len(train_idx) + len(test_idx) == 99
    assert len(test_idx) == 100 - len(train_idx) or len(test_idx) == 99 - len(train_idx)


def test_train_test_split_idx_kennard_stone():
    X, y = make_classification(n_samples=100, n_features=10)
    train_idx, test_idx = train_test_split_idx(
        X, test_size=0.2, method='kennard_stone', random_state=42
    )
    assert len(train_idx) + len(test_idx) == 100
    assert len(train_idx) == 80


def test_train_test_split_idx_spxy():
    X, y = make_classification(n_samples=100, n_features=10)
    y = y.reshape(-1, 1)
    train_idx, test_idx = train_test_split_idx(
        X, y=y, test_size=0.2, method='spxy', random_state=42, pca_components=1
    )
    assert len(train_idx) + len(test_idx) == 100
    assert len(train_idx) == 80


def test_train_test_split_idx_circular():
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    train_idx, test_idx = train_test_split_idx(
        X, y=y, test_size=0.2, method='circular', random_state=42
    )
    assert len(train_idx) + len(test_idx) == 100
    assert len(train_idx) == 80


def test_train_test_split_idx_invalid_method():
    X, y = make_classification(n_samples=100, n_features=10)
    with pytest.raises(ValueError):
        train_test_split_idx(
            X, y=y, test_size=0.2, method='invalid_method', random_state=42
        )
