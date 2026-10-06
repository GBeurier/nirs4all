"""Scientific regressions for independent streams, modes and correlated data."""

import numpy as np
import pytest

from nirs4all.operators.augmentation import TemperatureAugmenter
from nirs4all.synthesis.accelerated import AcceleratorBackend, create_accelerated_arrays, generate_spectra_batch_accelerated
from nirs4all.synthesis.environmental import EnvironmentalEffectsConfig, TemperatureConfig
from nirs4all.synthesis.generator import SyntheticNIRSGenerator
from nirs4all.synthesis.instruments import MultiScanConfig
from nirs4all.synthesis.products import ComponentVariation, ProductGenerator, ProductTemplate, VariationType
from nirs4all.synthesis.sources import MultiSourceGenerator


def test_repeated_effects_advance_and_fresh_generators_replay():
    generators = [SyntheticNIRSGenerator(random_state=42) for _ in range(2)]
    x = np.ones((64, generators[0].n_wavelengths))
    for effect in ["_apply_path_length", "_add_noise"]:
        first, second = [getattr(g, effect)(x) for g in generators]
        np.testing.assert_array_equal(first, second)
        third, fourth = [getattr(g, effect)(x) for g in generators]
        np.testing.assert_array_equal(third, fourth)
        assert not np.array_equal(first, third)
    assert generators[0]._path_length_op.random_state != generators[0]._noise_op.random_state


def test_multi_source_preserves_axes_and_independent_noise():
    sources = [{"name": name, "type": "nir", "wavelength_range": (1100, 1120), "wavelength_step": .5} for name in ["a", "b"]]
    source_generator = MultiSourceGenerator(random_state=42)
    result = source_generator.generate(48, sources=sources)
    assert not np.array_equal(result.sources["a"], result.sources["b"])
    repeated = MultiSourceGenerator(random_state=42).generate(48, sources=sources)
    np.testing.assert_array_equal(result.sources["a"], repeated.sources["a"])
    dataset = MultiSourceGenerator(random_state=42).create_dataset(48, sources=sources + [{"name": "aux", "type": "aux", "n_features": 3}])
    assert dataset.n_sources == 3
    assert len(dataset.headers(0)) == 41 and len(dataset.headers(2)) == 3
    np.testing.assert_allclose(dataset.wavelengths_nm(0), np.arange(1100, 1120.1, .5))


@pytest.mark.parametrize("external", [False, True])
def test_temperature_values_actually_applied(external):
    config = EnvironmentalEffectsConfig(temperature=TemperatureConfig(sample_temperature=35, temperature_variation=0), enable_temperature=True)
    axis = np.linspace(1300, 1550, 64)
    generator = SyntheticNIRSGenerator(wavelengths=axis, random_state=4, environmental_config=config)
    temperatures = np.array([25., 35., 45.])
    spectra = np.tile(np.exp(-((axis - 1450) / 35)**2), (3, 1))
    expected = np.vstack([TemperatureAugmenter(temperature_delta=t - config.temperature.reference_temperature,
        reference_temperature=config.temperature.reference_temperature, temperature_range=None).transform(spectra[i:i+1], wavelengths=axis)
        for i, t in enumerate(temperatures)])
    np.testing.assert_allclose(generator._apply_temperature(spectra, temperatures), expected)
    if external:
        concentrations = generator.generate_concentrations(3)
        x, metadata = generator.generate_from_concentrations(concentrations, temperatures=temperatures)
        np.testing.assert_array_equal(metadata["temperatures"], temperatures)
    else:
        x, _, _, metadata = generator.generate(3, temperatures=temperatures, return_metadata=True)
        np.testing.assert_array_equal(metadata["environmental_config"]["temperatures"], temperatures)
    assert np.all(np.isfinite(x))
    with pytest.raises(ValueError, match="one finite value"):
        generator._apply_temperature(spectra, np.array([30.]))


@pytest.mark.parametrize("mode", ["transmittance", "reflectance", "transflectance", "atr", "interactance"])
def test_measurement_mode_dispatched_in_both_generation_routes(mode):
    generator = SyntheticNIRSGenerator(random_state=4, measurement_mode=mode)
    assert generator.measurement_mode_simulator.config.mode.value == mode
    calls = []
    def apply(absorption, wavelengths):
        calls.append(absorption.copy())
        return absorption + .123
    generator.measurement_mode_simulator.apply = apply
    generator.generate(4)
    generator.generate_from_concentrations(generator.generate_concentrations(4))
    assert len(calls) == 2


def test_measurement_modes_change_measured_spectra():
    x_trans = SyntheticNIRSGenerator(random_state=4, measurement_mode="transmittance").generate(24)[0]
    x_refl = SyntheticNIRSGenerator(random_state=4, measurement_mode="reflectance").generate(24)[0]
    assert not np.allclose(x_trans, x_refl)
    assert np.all(np.isfinite(x_refl))


@pytest.mark.parametrize("rho", [-.85, .85])
def test_product_correlation_has_requested_sign_and_strength(rho):
    template = ProductTemplate(name="test", description="test", category="test", domain="food", components=[
        ComponentVariation("water", VariationType.UNIFORM, min_value=.2, max_value=.4),
        ComponentVariation("protein", VariationType.CORRELATED, correlated_with="water", correlation=rho, min_value=.1, max_value=.3)])
    compositions = ProductGenerator(template, random_state=4)._sample_compositions(20000)
    correlation = np.corrcoef(compositions.T)[0, 1]
    assert abs(correlation - rho) < .025
    assert np.mean((compositions[:, 1] == .1) | (compositions[:, 1] == .3)) < .01


def test_correlated_compositions_preserve_latent_negative_relation_and_support_psd():
    generator = SyntheticNIRSGenerator(random_state=4)
    matrix = np.eye(5)
    matrix[0, 1] = matrix[1, 0] = -1
    values = generator.generate_concentrations(1000, method="correlated", correlation_matrix=matrix)
    np.testing.assert_allclose(values.sum(axis=1), 1)
    assert np.std(np.log(values[:, 0] / values[:, 1])) > 1.8
    matrix[0, 1] = matrix[1, 0] = 1
    positive = generator.generate_concentrations(1000, method="correlated", correlation_matrix=matrix)
    np.testing.assert_allclose(positive[:, 0], positive[:, 1])
    matrix[0, 1] = matrix[1, 0] = 2
    with pytest.raises(ValueError, match="positive semidefinite"):
        generator.generate_concentrations(10, method="correlated", correlation_matrix=matrix)


@pytest.mark.parametrize("method", ["median", "weighted", "savgol"])
def test_outlier_rejection_does_not_replace_selected_averaging_method(method):
    generator = SyntheticNIRSGenerator(random_state=4, multi_scan_config=MultiScanConfig(n_scans=5, averaging_method=method))
    scans = np.random.default_rng(8).normal(size=(3, 5, 16))
    averaged = np.full((3, 16), .123)
    np.testing.assert_array_equal(generator._reject_scan_outliers(scans, averaged, 100), averaged)


def test_numpy_acceleration_seed_and_noise_rms():
    arrays = [create_accelerated_arrays(AcceleratorBackend.NUMPY, seed=4) for _ in range(2)]
    axis = np.linspace(1000, 2000, 64)
    outputs = [generate_spectra_batch_accelerated(4000, axis, np.ones((1, 64)), np.ones((4000, 1)), .1, a) for a in arrays]
    np.testing.assert_array_equal(*outputs)
    assert abs(np.std(outputs[0] - 1) - .1) < .002
