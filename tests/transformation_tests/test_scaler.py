# tests/test_scaler.py

import numpy as np
import pytest
from nirs4all.transformations._scaler import (
    Normalize,
    norml,
    Derivate,
    derivate,
    SimpleScale,
    spl_norml
)
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
import scipy.sparse


def test_Normalize_invalid_feature_range(random_data):
    with pytest.warns(SyntaxWarning):
        normalizer = Normalize(feature_range=(1, -1))
        normalizer.fit(random_data)

    with pytest.raises(ValueError):
        normalizer = Normalize(feature_range=(1, 1))
        normalizer.fit(random_data)


def test_Normalize_inverse_transform(random_data):
    normalizer = Normalize(feature_range=(0, 1))
    normalizer.fit(random_data)
    X_normalized = normalizer.transform(random_data)
    X_inverse = normalizer.inverse_transform(X_normalized)
    np.testing.assert_array_almost_equal(random_data, X_inverse)


def test_Normalize_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    normalizer = Normalize()
    with pytest.raises(TypeError):
        normalizer.fit(X_sparse)


def test_Derivate_order_zero(random_data):
    derivative = Derivate(order=0)
    derivative.fit(random_data)
    X_derivative = derivative.transform(random_data)
    np.testing.assert_array_equal(X_derivative, random_data)


def test_Derivate_order_two(random_data):
    derivative = Derivate(order=2)
    derivative.fit(random_data)
    X_derivative = derivative.transform(random_data)
    assert X_derivative.shape == random_data.shape


def test_Derivate_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    derivative = Derivate()
    with pytest.raises(ValueError):
        derivative.fit(X_sparse)


def test_SimpleScale_inverse_transform(random_data):
    scaler = SimpleScale()
    scaler.fit(random_data)
    X_scaled = scaler.transform(random_data)
    X_inverse = scaler.inverse_transform(X_scaled)
    np.testing.assert_array_almost_equal(random_data, X_inverse)


def test_SimpleScale_sparse_input():
    X_sparse = scipy.sparse.csr_matrix(np.random.rand(10, 10))
    scaler = SimpleScale()
    with pytest.raises(TypeError):
        scaler.fit(X_sparse)
