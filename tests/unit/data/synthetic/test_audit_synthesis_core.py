"""Regressions for synthesis grids, target regimes, metadata and fitted state."""

from dataclasses import asdict

import numpy as np
import pytest

from nirs4all.synthesis.builder import SyntheticDatasetBuilder
from nirs4all.synthesis.config import (
    BatchEffectConfig,
    ConfounderConfig,
    MetadataConfig,
    MultiRegimeConfig,
    NonLinearConfig,
    SyntheticDatasetConfig,
    TargetConfig,
)
from nirs4all.synthesis.fitter import FittedParameters, RealDataFitter
from nirs4all.synthesis.generator import SyntheticNIRSGenerator
from nirs4all.synthesis.targets import NonLinearTargetConfig, NonLinearTargetProcessor


def test_spectral_target_regimes_receive_generated_spectra():
    builder = SyntheticDatasetBuilder(96, random_state=14).with_complex_target_landscape(n_regimes=3, regime_method="spectral")
    x, y = builder.build_arrays()
    assert x.shape[0] == y.shape[0] == 96 and np.isfinite(y).all()
    np.testing.assert_array_equal(builder.state._X, x)


def test_concentration_regimes_follow_composition_structure_and_spectral_fallback():
    axis = np.linspace(0, 1, 96)
    compositions = np.column_stack((axis, 1 - axis))
    processor = NonLinearTargetProcessor(NonLinearTargetConfig(n_regimes=3, regime_method="concentration"))
    regimes = processor._assign_regimes(compositions, None)
    np.testing.assert_array_equal(np.bincount(regimes), [32, 32, 32])
    assert abs(np.corrcoef(regimes, axis)[0, 1]) > .94
    fallback = NonLinearTargetProcessor(NonLinearTargetConfig(n_regimes=3, regime_method="spectral"))
    np.testing.assert_array_equal(fallback._assign_regimes(compositions, None), regimes)


@pytest.mark.parametrize("separation", [3., 3.5, 5.])
def test_high_separation_threshold_classification_is_valid(separation):
    x, y = SyntheticDatasetBuilder(96, random_state=14).with_classification(
        n_classes=3, separation=separation, separation_method="threshold").build_arrays()
    assert x.shape[0] == y.size == 96
    np.testing.assert_array_equal(np.bincount(y), [32, 32, 32])


@pytest.mark.parametrize("step", [.1, .3, .7, 44.827586206896555])
def test_regular_grid_never_overshoots_end(step):
    generator = SyntheticNIRSGenerator(wavelength_start=1100, wavelength_end=2400, wavelength_step=step, random_state=12)
    assert generator.wavelengths[-1] <= 2400 + 1e-10
    assert np.all(np.diff(generator.wavelengths) > 0)


def test_template_builder_keeps_nonuniform_axis_and_fitted_statistics():
    axis = np.array([1100., 1100.5, 1200.25, 1360., 1590.75, 1700., 1830.1, 2000., 2400.])
    x = np.random.default_rng(12).uniform(.1, 1, (24, len(axis)))
    builder = SyntheticDatasetBuilder(12, random_state=14).fit_to(x, wavelengths=axis)
    assert builder.state.custom_params is not None
    dataset = builder.build()
    np.testing.assert_allclose(dataset.wavelengths_nm(), axis, atol=1e-10)
    assert len(set(dataset.headers(0))) == len(axis)
    unmatched = SyntheticDatasetBuilder(12, random_state=14).fit_to(x, wavelengths=axis, match_statistics=False)
    assert unmatched.state.custom_params is None


def test_fitted_parameters_persistence_and_generator_kwargs(tmp_path):
    parameters = FittedParameters(noise_base=.123, path_length_std=.33, baseline_amplitude=.7,
                                  preprocessing_type="second_derivative", is_preprocessed=True, inferred_instrument="foss_xds")
    path = tmp_path / "fit.json"
    parameters.save(str(path))
    restored = FittedParameters.load(str(path))
    assert restored.preprocessing_type == "second_derivative" and restored.is_preprocessed
    generator = SyntheticNIRSGenerator(**restored.to_generator_kwargs(), random_state=8)
    assert generator.params["noise_base"] == .123 and generator.params["path_length_std"] == .33
    fitter = RealDataFitter()
    fitter.fitted_params = restored
    matched = fitter.create_matched_generator(random_state=8)
    assert matched.params["baseline_amplitude"] == .7 and matched.instrument is not None


def test_matched_generator_retains_fitted_environment_scattering_and_edge_configs():
    parameters = FittedParameters(temperature_config={"temperature_variation": 3.},
        moisture_config={"water_activity": .7}, particle_size_config={"mean_size_um": 80., "std_size_um": 20.},
        emsc_config={"multiplicative_scatter_std": .12},
        edge_artifacts_config={"stray_light": {"enabled": True, "stray_fraction": .004}})
    generator = SyntheticNIRSGenerator(**parameters.to_generator_kwargs(), random_state=8)
    assert generator.environmental_config.temperature.temperature_variation == 3.
    assert generator.environmental_config.moisture.water_activity == .7
    assert generator.scattering_effects_config.particle_size.distribution.mean_size_um == 80.
    assert generator.scattering_effects_config.emsc.multiplicative_scatter_std == .12
    assert generator.edge_artifacts_config.enable_stray_light and generator.edge_artifacts_config.stray_fraction == .004


@pytest.mark.parametrize("channels", [2, 3, 16, 20])
def test_artifact_injection_is_valid_on_small_grids(channels):
    generator = SyntheticNIRSGenerator(wavelengths=np.linspace(1100, 2400, channels), complexity="realistic",
                                       custom_params={"artifact_prob": 1.}, random_state=0)
    spectra, _, _, metadata = generator.generate(200, return_metadata=True)
    assert spectra.shape == (200, channels) and np.isfinite(spectra).all()
    assert set(metadata["artifact_types"]) == {"spike", "dead_band", "saturation"}


def test_metadata_keeps_sample_alignment_after_shuffle():
    builder = SyntheticDatasetBuilder(96, random_state=14).with_metadata(n_groups=3, n_repetitions=(2, 3), sample_id_prefix="S")
    dataset = builder.build()
    assert {"sample_id", "bio_sample_id", "repetition", "group"} <= set(dataset.metadata_columns)
    ids = builder.state._sample_metadata.sample_ids
    order = np.random.default_rng(14).permutation(96)
    np.testing.assert_array_equal(dataset.metadata_column("sample_id"), ids[order])
    np.testing.assert_allclose(dataset.x({}, layout="2d"), builder.state._X[order])


def test_stratified_classification_uses_target_labels():
    dataset = SyntheticDatasetBuilder(96, random_state=14).with_classification(n_classes=3, separation=5, separation_method="threshold").with_partitions(
        train_ratio=.5, stratify=True).build()
    np.testing.assert_array_equal(np.bincount(dataset.y({"partition": "train"}).ravel().astype(int)), [16, 16, 16])
    np.testing.assert_array_equal(np.bincount(dataset.y({"partition": "test"}).ravel().astype(int)), [16, 16, 16])


def test_config_roundtrip_preserves_target_complexity_and_batch_parameters():
    config = SyntheticDatasetConfig(n_samples=64, random_state=12,
        nonlinear=NonLinearConfig("polynomial", .7, 2, 3), confounders=ConfounderConfig(.5, 2, .1, True),
        multi_regime=MultiRegimeConfig(3, "spectral", .1, .4), targets=TargetConfig(component_indices=[0, 2]),
        metadata=MetadataConfig(n_groups=2, group_names=["A", "B"]), batch_effects=BatchEffectConfig(True, 2, .12, .13))
    builder = SyntheticDatasetBuilder.from_config(config)
    assert asdict(builder.get_config()) == asdict(config)
    x, y = builder.build_arrays()
    assert x.shape[0] == y.shape[0] == 64 and y.shape[1] == 2


def test_builder_merges_custom_physics_across_fluent_calls_and_zero_batch_effects():
    builder = SyntheticDatasetBuilder(32, random_state=14).with_features(noise_base=.123).with_features(path_length_std=.33)
    assert builder.state.custom_params == {"noise_base": .123, "path_length_std": .33}
    generator = SyntheticNIRSGenerator(custom_params={"batch_offset_std": 0., "batch_gain_std": 0.}, random_state=14)
    offsets, gains = generator.generate_batch_effects(3, [10, 10, 10])
    assert np.all(offsets == 0) and np.all(gains == 1)
