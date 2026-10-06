"""Independent numerical references for OPO-01/03/15 and duplicate AOM-02."""

import numpy as np
import pytest
from scipy.linalg import null_space

from nirs4all.operators.models._aom_nirs.pls.preprocessing import ExtendedMSC
from nirs4all.operators.transforms import EPO, ExtendedMultiplicativeScatterCorrection, MultiplicativeScatterCorrection


@pytest.mark.parametrize("transformer", [ExtendedMultiplicativeScatterCorrection, ExtendedMSC])
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_emsc_recovers_known_reference_from_additive_and_multiplicative_scatter(transformer, scale, degree):
    """Analytic generating model, with zero-mean interference in calibration."""
    axis = np.linspace(-1., 1., 101)
    reference = 5. + np.exp(-((axis - 0.2) / 0.15) ** 2) + np.sin(axis * 8)
    polynomial = np.polynomial.polynomial.polyvander(axis, degree)
    offsets = np.linspace(-0.5, 0.5, 6)[:, None] * np.arange(1, degree + 2)
    slopes = np.linspace(0.5, 1.5, 6)
    training = slopes[:, None] * reference + offsets @ polynomial.T
    original = training.copy()
    operator = transformer(degree=degree, scale=scale)
    corrected = operator.fit_transform(training)
    np.testing.assert_allclose(operator.reference_, reference, atol=1e-14)
    np.testing.assert_allclose(corrected, np.broadcast_to(reference, corrected.shape), atol=2e-13)
    np.testing.assert_array_equal(training, original)
    # Prediction-time scatter coefficients differ from calibration coefficients.
    held_out = 1.8 * reference + np.arange(1, degree + 2) @ polynomial.T
    np.testing.assert_allclose(operator.transform(held_out[None, :])[0], reference, atol=2e-13)


@pytest.mark.parametrize("transformer", [ExtendedMultiplicativeScatterCorrection, ExtendedMSC])
def test_emsc_matches_orthogonal_baseline_reference_on_arbitrary_spectra(transformer):
    """Project out the baseline first, then infer each sample's reference slope."""
    rng = np.random.default_rng(118)
    training = 5 + rng.normal(size=(15, 40))
    held_out = 5 + rng.normal(size=(4, 40))
    reference = training.mean(axis=0)
    polynomial = np.polynomial.polynomial.polyvander(np.linspace(-1., 1., 40), 2)
    complement = null_space(polynomial.T)
    projected_reference = reference @ complement
    slopes = (held_out @ complement @ projected_reference) / (projected_reference @ projected_reference)
    nuisance = (held_out - slopes[:, None] * reference) @ np.linalg.pinv(polynomial).T @ polynomial.T
    expected = (held_out - nuisance) / slopes[:, None]
    np.testing.assert_allclose(transformer().fit(training).transform(held_out), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("transformer", [ExtendedMultiplicativeScatterCorrection, ExtendedMSC])
def test_emsc_refuses_unidentifiable_polynomial_reference(transformer):
    with pytest.raises(ValueError, match="independent"):
        transformer().fit(np.ones((5, 30)))


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("dependent_parameters", [True, False])
def test_epo_reuses_analytic_spectral_interference_projection(scale, dependent_parameters):
    nuisance = np.tile([-1., 1.], 12)
    signal = np.repeat([-2., 0., 2.], 8)
    interference = np.array([1., -2., 0.5, 0.])
    useful = np.array([2., 1., 0., 3.])
    training = 5. + np.outer(nuisance, interference) + np.outer(signal, useful)
    external = np.column_stack([nuisance, 2 * nuisance]) if dependent_parameters else nuisance
    reference_basis = null_space(interference[None, :])
    projector = reference_basis @ reference_basis.T
    operator = EPO(scale=scale).fit(training, external)
    mean = training.mean(axis=0) if scale else np.zeros(4)
    expected = (training - mean) @ projector + mean
    np.testing.assert_allclose(operator.transform(training), expected, atol=2e-14)
    np.testing.assert_allclose(EPO(scale=scale).fit_transform(training, external), expected, atol=2e-14)
    held_out = 5. + 10 * interference + 2 * useful
    np.testing.assert_allclose(operator.transform(held_out[None, :]), ((held_out - mean) @ projector + mean)[None, :], atol=3e-14)
    assert not np.allclose(operator.transform(held_out[None, :]), held_out[None, :])


def test_epo_constant_external_parameter_is_identity_and_default_does_not_mutate():
    rng = np.random.default_rng(72)
    X = rng.normal(size=(12, 20))
    original = X.copy()
    np.testing.assert_allclose(EPO().fit_transform(X, np.ones(12)), X, atol=1e-14)
    np.testing.assert_array_equal(X, original)


def test_msc_inverse_has_explicit_noninvertibility_error():
    """OPO-15: no missing a_/b_/scaler_ attributes or invented inverse."""
    X = np.array([[1., 3., 2.], [2., 5., 4.]])
    operator = MultiplicativeScatterCorrection().fit(X)
    with pytest.raises(NotImplementedError, match="not invertible"):
        operator.inverse_transform(operator.transform(X))
