
import numpy as np
import pytest
import scipy.sparse
from nirs4all.transformations._nirs import (
    wavelet_transform,
    Wavelet,
    Haar,
    savgol,
    SavitzkyGolay,
    MultiplicativeScatterCorrection,
    msc
)
from sklearn.exceptions import NotFittedError


def test_wavelet_transform(random_data):
    transformed = wavelet_transform(random_data, wavelet='db1')
    assert transformed.shape == random_data.shape


def test_wavelet_transform_invalid_wavelet(random_data):
    with pytest.raises(ValueError):
        wavelet_transform(random_data, wavelet='invalid_wavelet')


def test_Wavelet_transform(random_data):
    transformer = Wavelet(wavelet='db1')
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_Wavelet_invalid_input():
    X = scipy.sparse.csr_matrix(np.random.rand(10, 100))
    transformer = Wavelet(wavelet='db1')
    with pytest.raises(ValueError):
        transformer.fit(X)


def test_Haar_transform(random_data):
    transformer = Haar()
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_savgol(random_data):
    filtered = savgol(random_data, window_length=11, polyorder=3)
    assert filtered.shape == random_data.shape


def test_SavitzkyGolay_transform(random_data):
    transformer = SavitzkyGolay(window_length=11, polyorder=3)
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_MultiplicativeScatterCorrection_transform(random_data):
    transformer = MultiplicativeScatterCorrection()
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    assert X_transformed.shape == random_data.shape


def test_MultiplicativeScatterCorrection_inverse_transform(random_data):
    transformer = MultiplicativeScatterCorrection()
    transformer.fit(random_data)
    X_transformed = transformer.transform(random_data)
    X_inverse = transformer.inverse_transform(X_transformed)
    np.testing.assert_array_almost_equal(random_data, X_inverse)


def test_msc(random_data):
    corrected_spectra = msc(random_data)
    assert corrected_spectra.shape == random_data.shape


