# tests/test_standard.py

import numpy as np
import pytest
from nirs4all.transformations._standard import (
    Baseline,
    baseline,
    Detrend,
    detrend,
    Gaussian,
    gaussian
)
import scipy.sparse


def test_Baseline_inverse_transform(random_data):
    transformer = Baseline()
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    X_inverse = transformer.inverse_transform(X_transformed)
    np.testing.assert_array_almost_equal(random_data, X_inverse)


def test_Baseline_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    transformer = Baseline()
    with pytest.raises(TypeError):
        transformer.fit(X_sparse)


def test_Detrend_bp(random_data):
    transformer = Detrend(bp=[0, 50])
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_Detrend_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    transformer = Detrend()
    with pytest.raises(ValueError):
        transformer.fit(X_sparse)


def test_Gaussian_order_zero(random_data):
    transformer = Gaussian(order=0)
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_Gaussian_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    transformer = Gaussian()
    with pytest.raises(ValueError):
        transformer.fit(X_sparse)
