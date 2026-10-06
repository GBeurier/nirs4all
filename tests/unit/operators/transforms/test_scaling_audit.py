"""Analytic/sklearn references for the remaining assigned transform findings."""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from nirs4all.operators.transforms.feature_selection import CARS
from nirs4all.operators.transforms.scalers import Derivate, Normalize, RobustStandardNormalVariate, SimpleScale, StandardNormalVariate, derivate, norml, spl_norml
from nirs4all.operators.transforms.signal_conversion import ToAbsorbance


@pytest.mark.parametrize("order", [1, 2])
def test_derivative_uses_wavelength_axis_and_is_batch_invariant(order):
    """OPO-02: polynomial reference, with the existing gradient edge convention."""
    spacing = 0.25
    wavelengths = np.arange(20) * spacing
    X = np.vstack([wavelengths ** 2 + 10, wavelengths ** 2 - 100, wavelengths ** 2 + 300])
    original = X.copy()
    operator = Derivate(order=order, delta=spacing)
    actual = operator.fit_transform(X)
    expected = 2 * wavelengths if order == 1 else np.full_like(wavelengths, 2.)
    np.testing.assert_allclose(actual[:, 2:-2], np.broadcast_to(expected[2:-2], actual[:, 2:-2].shape), atol=1e-13)
    np.testing.assert_allclose(operator.transform(X[1:2]), actual[1:2], atol=1e-13)
    np.testing.assert_allclose(operator.transform(X[::-1]), actual[::-1], atol=1e-13)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(derivate(X, order=order, delta=spacing), actual)


def test_explicit_sample_axis_derivative_is_available():
    X = np.arange(6.)[:, None] * np.array([[2., 3., 4.]])
    expected = np.broadcast_to([2., 3., 4.], X.shape)
    np.testing.assert_allclose(Derivate(axis=0).fit_transform(X), expected)
    np.testing.assert_allclose(derivate(X, axis=0), expected)


@pytest.mark.parametrize("transformer", [StandardNormalVariate, RobustStandardNormalVariate])
def test_axis_zero_snv_retains_documented_stateless_batch_statistics(transformer):
    """OPO-11: document transductive behavior without changing statistics."""
    training = np.array([[0., 1.], [1., 3.], [2., 5.]])
    test = np.array([[10., 20.], [11., 21.], [12., 22.]])
    operator = transformer(axis=0).fit(training)
    np.testing.assert_array_equal(operator.transform(test[:1]), np.zeros((1, 2)))
    np.testing.assert_allclose(operator.transform(test), transformer(axis=0).fit_transform(test))
    assert not np.allclose(operator.transform(test), np.broadcast_to(operator.transform(test[:1]), test.shape))


@pytest.mark.parametrize("bounds", [(0., 1.), (2., 4.), (-1., 2.)])
def test_functional_normalize_maps_matrix_extrema_to_requested_bounds(bounds):
    """OPO-12: preserve the matrix-wide functional range contract."""
    X = np.array([[2., 4.], [8., 10.]])
    expected = bounds[0] + (bounds[1] - bounds[0]) * (X - 2) / 8
    np.testing.assert_allclose(norml(X, feature_range=bounds), expected)


def test_normalize_set_params_refits_using_current_range():
    """OPO-13: constructor-derived state cannot defeat sklearn tuning."""
    X = np.array([[2., 0., 4.], [8., 0., 4.], [10., 0., 4.]])
    operator = Normalize().set_params(feature_range=(0, 1))
    assert operator.user_defined
    np.testing.assert_allclose(operator.fit_transform(X), MinMaxScaler().fit_transform(X))
    operator.set_params(feature_range=(-1, 1))
    assert not operator.user_defined
    norms = np.linalg.norm(X, axis=0)
    np.testing.assert_allclose(operator.fit_transform(X), X / np.where(norms == 0, 1., norms))


@pytest.mark.parametrize("transformer", [Normalize, SimpleScale, lambda: Normalize((0, 1))])
def test_constant_columns_are_finite_and_inverse_roundtrips_held_out(transformer):
    """OPO-14: columns that were constant during calibration stay usable."""
    training = np.array([[2., 0., 4.], [8., 0., 4.], [10., 0., 4.]])
    operator = transformer().fit(training)
    corrected = operator.transform(training)
    assert np.isfinite(corrected).all()
    held_out = np.array([[3., 1., 5.]])
    np.testing.assert_allclose(operator.inverse_transform(operator.transform(held_out)), held_out)
    if isinstance(operator, SimpleScale) or operator.user_defined:
        np.testing.assert_allclose(corrected, MinMaxScaler().fit_transform(training))


def test_functional_normalizers_handle_constant_and_zero_columns():
    X = np.array([[2., 0., 4.], [8., 0., 4.]])
    assert np.isfinite(norml(X)).all()
    np.testing.assert_allclose(spl_norml(X), MinMaxScaler().fit_transform(X))
    np.testing.assert_array_equal(norml(np.ones((3, 2)), feature_range=(2, 4)), np.full((3, 2), 2.))


@pytest.mark.parametrize("target_scale", [1e-3, 1e-5, 1e-9])
def test_cars_selection_is_invariant_to_target_units(target_scale):
    """OPO-08: real PLS-derived weights must be valid probabilities."""
    rng = np.random.default_rng(77)
    X = rng.normal(size=(48, 14))
    y = 3 * X[:, 0] + X[:, 4] + rng.normal(size=48) * 0.2
    options = {"n_components": 3, "n_sampling_runs": 6, "cv_folds": 3, "random_state": 0}
    reference = CARS(**options).fit(X, y)
    scaled = CARS(**options).fit(X, y * target_scale)
    np.testing.assert_array_equal(scaled.selected_indices_, reference.selected_indices_)
    np.testing.assert_array_equal(scaled.transform(X), reference.transform(X))


def test_cars_constant_target_uses_uniform_sampling():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 8))
    operator = CARS(n_components=2, n_sampling_runs=4, cv_folds=3, random_state=0).fit(X, np.zeros(24))
    assert operator.transform(X).shape[0] == 24
    assert np.isfinite(operator.transform(X)).all()


@pytest.mark.parametrize("source_type,factor", [("reflectance", 1.), ("transmittance", 1.), ("reflectance%", 100.)])
def test_absorbance_opt_out_refuses_nonpositive_input(source_type, factor):
    """OPO-16: clipping is explicit; no silent epsilon substitution on opt-out."""
    X = np.array([[-0.2, 0., 0.5]]) * factor
    with pytest.raises(ValueError, match="Non-positive"):
        ToAbsorbance(source_type=source_type, clip_negative=False).fit_transform(X)
    np.testing.assert_allclose(ToAbsorbance(source_type=source_type).fit_transform(X), [[10., 10., -np.log10(0.5)]])


def test_absorbance_opt_out_preserves_positive_values_below_epsilon():
    X = np.array([[1e-15, 0.1, 0.5]])
    np.testing.assert_allclose(ToAbsorbance(clip_negative=False).fit_transform(X), -np.log10(X))
