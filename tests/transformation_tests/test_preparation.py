# tests/test_preparation.py

import numpy as np
import pytest
from nirs4all.transformations._preparation import CropTransformer, ResampleTransformer


def test_CropTransformer_full_range(random_data):
    cropper = CropTransformer()
    X_cropped = cropper.transform(random_data)
    assert X_cropped.shape == random_data.shape


def test_CropTransformer_end_exceeds(random_data):
    cropper = CropTransformer(start=10, end=200)  # End exceeds data shape
    X_cropped = cropper.transform(random_data)
    assert X_cropped.shape[1] == random_data.shape[1] - 10


def test_CropTransformer_invalid_input():
    cropper = CropTransformer()
    with pytest.raises(ValueError):
        cropper.transform([1, 2, 3])  # Not a numpy array


def test_ResampleTransformer_same_size(random_data):
    resampler = ResampleTransformer(num_samples=random_data.shape[1])
    X_resampled = resampler.transform(random_data)
    assert X_resampled.shape == random_data.shape


def test_ResampleTransformer_invalid_input():
    resampler = ResampleTransformer(num_samples=50)
    with pytest.raises(ValueError):
        resampler.transform([1, 2, 3])  # Not a numpy array


def test_ResampleTransformer_invalid_dimensions():
    resampler = ResampleTransformer(num_samples=50)
    with pytest.raises(ValueError):
        resampler.transform(np.random.rand(10, 10, 10))  # Not a 2D array
