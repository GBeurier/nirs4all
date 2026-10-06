"""Registry conformance, exact benchmark axes and multi-sensor coverage."""

import numpy as np
import pytest

from nirs4all.synthesis._aggregates import expand_aggregate, get_aggregate, list_aggregates
from nirs4all.synthesis.benchmarks import create_synthetic_matching_benchmark, get_benchmark_info, list_benchmark_datasets
from nirs4all.synthesis.components import ComponentLibrary
from nirs4all.synthesis.domains import APPLICATION_DOMAINS
from nirs4all.synthesis.generator import SyntheticNIRSGenerator
from nirs4all.synthesis.prior import NIRSPriorConfig, PriorSampler
from nirs4all.synthesis.wavenumber import (
    classify_wavelength_extended,
    get_all_zones_extended,
    get_all_zones_wavelength,
    get_zone_wavelength_range,
)


@pytest.mark.parametrize("name", list_aggregates())
def test_every_aggregate_resolves_to_available_spectral_components(name):
    aggregate = get_aggregate(name)
    for varying in [False, True]:
        composition = expand_aggregate(name, variability=varying, random_state=4)
        library = ComponentLibrary.from_predefined(list(composition))
        spectrum = library.compute_all(np.linspace(1100, 2400, 32))
        assert np.all(np.isfinite(spectrum))
    assert set(aggregate.variability) <= set(aggregate.components)


def test_pharmaceutical_surrogates_are_explicit():
    assert "not a loratadine spectrum" in get_aggregate("tablet_loratadine_low_dose").description
    for name in ["tablet_generic_ir_low_dose", "tablet_generic_ir_medium_dose", "tablet_generic_ir_high_load", "capsule_generic_powder"]:
        assert "paracetamol spectral proxy" in get_aggregate(name).description
    assert "salt effects not modeled" in get_aggregate("tablet_metformin").description
    assert "hydrate effects not modeled" in get_aggregate("tablet_amoxicillin").description


@pytest.mark.parametrize("domain", list(NIRSPriorConfig().domain_weights))
def test_prior_short_domains_use_registered_component_sets(domain):
    sampler = PriorSampler(random_state=4)
    canonical = sampler._canonical_domain(domain)
    assert canonical in APPLICATION_DOMAINS
    sample = sampler.sample_for_domain(domain, n_samples=12)
    assert set(sample["components"]) <= set(APPLICATION_DOMAINS[canonical].typical_components)
    assert sample["domain_category"] == APPLICATION_DOMAINS[canonical].category.value
    ComponentLibrary.from_predefined(sample["components"])


def test_unknown_prior_domain_has_valid_generic_fallback():
    components = PriorSampler(random_state=4).sample_components("custom_unsupported_domain", n_components=5)
    ComponentLibrary.from_predefined(components)
    assert "starch" in components


def test_wavelength_zone_ranges_ascend_and_visible_specific_zones_win():
    expected = (1600., 1e7 / 5500)
    np.testing.assert_allclose(get_zone_wavelength_range("1st_overtones_CH"), expected)
    assert all(lo < hi for lo, hi, _ in get_all_zones_wavelength())
    assert all(lo < hi for lo, hi, _, _ in get_all_zones_extended())
    assert classify_wavelength_extended(450)[0] == "blue_absorption"
    assert classify_wavelength_extended(660)[0] == "red_absorption"


@pytest.mark.parametrize("name", list_benchmark_datasets())
def test_benchmark_match_has_exact_channel_count(name):
    info = get_benchmark_info(name)
    x, concentrations, spectra = create_synthetic_matching_benchmark(name, n_samples=8, random_state=4)
    assert x.shape == (8, info.n_wavelengths)
    assert spectra.shape[1] == info.n_wavelengths
    assert concentrations.shape[0] == 8
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("instrument", ["foss_xds", "metrohm_ds2500", "asd_fieldspec"])
def test_multi_sensor_noise_uses_wavelength_covering_detector(instrument):
    generator = SyntheticNIRSGenerator(instrument=instrument, random_state=4)
    axis = generator.wavelengths
    output = generator._apply_detector_effects(np.zeros((32, len(axis))), axis)
    assert np.all(np.isfinite(output))
    responses = []
    for sensor, simulator in generator._sensor_detectors:
        covered = (axis >= sensor.wavelength_range[0]) & (axis <= sensor.wavelength_range[1])
        responses.append(np.where(covered, simulator.response.get_response_at(axis), -np.inf))
    assert responses
    covered_response = np.max(responses, axis=0)
    primary_response = generator.detector_simulator.response.get_response_at(axis)
    repaired = (primary_response == 0) & (covered_response > .15)
    assert np.any(repaired)
    # Each actual selected simulator stores its applied inverse-response noise
    # factors. Covered channels cannot inherit the primary out-of-band factor.
    scales = np.concatenate([simulator._response_noise_scaling for _, simulator in generator._sensor_detectors
                             if hasattr(simulator, "_response_noise_scaling")])
    assert np.count_nonzero(scales < 1 / .15) >= np.count_nonzero(repaired)
