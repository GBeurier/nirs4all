# tests/test_spline_augmentation.py

import numpy as np
import pytest
from nirs4all.transformations._spline_augmentation import (
    Spline_Smoothing,
    Spline_X_Perturbations,
    Spline_Y_Perturbations,
    Spline_X_Simplification,
    Spline_Curve_Simplification,
)
from sklearn.exceptions import NotFittedError


def test_Spline_Smoothing_apply_on_global(random_data):
    augmenter = Spline_Smoothing()
    X_augmented = augmenter.augment(random_data, apply_on="global")
    assert X_augmented.shape == random_data.shape


def test_Spline_X_Perturbations_invalid_density(random_data):
    with pytest.raises(ValueError):
        augmenter = Spline_X_Perturbations(perturbation_density=1.5)
        augmenter.transform(random_data)


def test_Spline_X_Perturbations_apply_on_global(random_data):
    augmenter = Spline_X_Perturbations(random_state=42)
    X_augmented = augmenter.augment(random_data, apply_on="global")
    assert X_augmented.shape == random_data.shape


def test_Spline_Y_Perturbations_apply_on_global(random_data):
    augmenter = Spline_Y_Perturbations(random_state=42)
    X_augmented = augmenter.augment(random_data, apply_on="global")
    assert X_augmented.shape == random_data.shape


def test_Spline_X_Simplification_non_uniform(random_data):
    augmenter = Spline_X_Simplification(random_state=42, uniform=False)
    X_augmented = augmenter.transform(random_data)
    assert X_augmented.shape == random_data.shape


def test_Spline_Curve_Simplification_apply_on_features(random_data):
    augmenter = Spline_Curve_Simplification(random_state=42)
    X_augmented = augmenter.augment(random_data, apply_on="features")
    assert X_augmented.shape == random_data.shape


def test_Spline_Curve_Simplification_uniform(random_data):
    augmenter = Spline_Curve_Simplification(random_state=42, uniform=True)
    X_augmented = augmenter.transform(random_data)
    assert X_augmented.shape == random_data.shape
