"""Physical wavelength order must not change or break spectral warping."""

import numpy as np
import pytest

from nirs4all.operators.augmentation.spectral import LocalWavelengthWarp, SmoothMagnitudeWarp


@pytest.mark.parametrize("operator", [LocalWavelengthWarp, SmoothMagnitudeWarp])
@pytest.mark.parametrize("order", [np.arange(15, -1, -1), np.random.default_rng(55).permutation(16)])
def test_warp_restores_descending_or_permuted_channel_order(operator, order):
    X = np.random.default_rng(54).normal(size=(3, 16))
    wavelengths = np.linspace(1000, 2400, 16)
    expected = operator(random_state=5).fit_transform(X, wavelengths=wavelengths)
    actual = operator(random_state=5).fit_transform(X[:, order], wavelengths=wavelengths[order])
    np.testing.assert_allclose(actual, expected[:, order], rtol=0, atol=0)
    assert actual.shape == X.shape and np.isfinite(actual).all()


@pytest.mark.parametrize("operator", [LocalWavelengthWarp, SmoothMagnitudeWarp])
def test_duplicate_coordinates_are_rejected_clearly(operator):
    with pytest.raises(ValueError, match="duplicate wavelengths"):
        operator(random_state=5).fit_transform(np.ones((2, 4)), wavelengths=np.array([1., 2., 2., 3.]))


@pytest.mark.parametrize("operator", [LocalWavelengthWarp, SmoothMagnitudeWarp])
def test_warp_needs_at_least_two_control_points(operator):
    with pytest.raises(ValueError, match="n_control_points must be at least 2"):
        operator(n_control_points=1).fit_transform(np.ones((2, 4)))
