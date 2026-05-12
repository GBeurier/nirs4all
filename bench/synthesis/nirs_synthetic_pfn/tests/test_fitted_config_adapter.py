from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from nirsyntheticpfn.adapters.fitted_config_adapter import (
    FittedConfigAdapterError,
    _edge_artifacts_config,
    _environmental_config,
    _scattering_config,
    build_dataset_run_from_fitted_config,
    fitted_config_to_prior_record,
)
from nirsyntheticpfn.adapters.fitted_residual_effects import fit_observable_residual_effects

from nirs4all.synthesis.fitter import RealDataFitter


def test_fitted_config_maps_to_a2_dataset_run_and_reports_unsupported_fields() -> None:
    fitted_config = _full_fitted_config()

    run = build_dataset_run_from_fitted_config(
        fitted_config,
        n_samples=16,
        random_seed=1234,
    )

    assert run.X.shape == (16, run.wavelengths.size)
    assert run.y.shape == (16,)
    assert np.isfinite(run.X).all()
    assert np.isfinite(run.y).all()
    assert run.validation_summary["status"] == "passed"
    assert run.builder_config["adapter_version"] == "A3-real-fit-via-A2"
    assert run.builder_config["features"]["complexity"] == "realistic"
    assert run.builder_config["nuisance"]["custom_params"]["noise_base"] == 0.003
    assert run.builder_config["fitted_config_mapping"]["effect_reconstruction"]["mode"] == (
        "calibrated_fitted_nuisance_effect_controls"
    )
    assert run.metadata["real_fit_adapter"]["source"] == "RealDataFitter.to_full_config"
    assert run.metadata["real_fit_adapter"]["unsupported_fields"] == run.validation_summary["unsupported_fields"]
    assert run.prior_config["source_prior_config"]["_raw_fitted_config"]["noise_base"] == 0.003

    unsupported_fields = {
        field["field"] for field in run.validation_summary["unsupported_fields"]
    }
    assert "moisture_config" not in unsupported_fields
    assert "emsc_config" not in unsupported_fields
    assert "preprocessing_type" in unsupported_fields
    assert "particle_size_config.std_size_um" not in unsupported_fields
    assert "edge_artifacts_config.edge_curvature" not in unsupported_fields


def test_fitted_effect_reconstruction_reports_raw_and_effective_controls() -> None:
    fitted_config = _full_fitted_config()
    fitted_config.update({
        "complexity": "complex",
        "path_length_std": 0.18,
        "baseline_amplitude": 0.09,
        "scatter_alpha_std": 0.14,
        "global_slope_mean": 0.20,
        "particle_size_config": {"mean_size_um": 20.0, "std_size_um": 6.0},
        "moisture_config": {"moisture_content": 0.04},
        "edge_artifacts_config": {
            "edge_curvature": {
                "enabled": True,
                "curvature_type": "smile",
                "left_severity": 2.0,
                "right_severity": 1.5,
            },
        },
    })

    run = build_dataset_run_from_fitted_config(fitted_config, n_samples=8, random_seed=1234)
    reconstruction = run.builder_config["fitted_config_mapping"]["effect_reconstruction"]
    fields = reconstruction["fields"]

    assert run.builder_config["features"]["complexity"] == "realistic"
    assert fields["complexity"]["raw"] == "complex"
    assert fields["complexity"]["effective"] == "realistic"
    assert fields["path_length_std"]["raw"] == 0.18
    assert fields["path_length_std"]["effective"] == 0.05
    assert fields["global_slope_mean"]["effective"] == 0.05
    assert fields["particle_size_config"]["raw"]["mean_size_um"] == 20.0
    assert fields["particle_size_config"]["effective"]["mean_size_um"] == 100.0
    assert fields["particle_size_config"]["valid_range_um"] == [100.0, 1000.0]
    assert fields["moisture_config"]["effective"] == {}
    assert fields["edge_artifacts_config"]["mapping"] == "range_safe_executable_edge_artifacts"
    assert fields["edge_artifacts_config"]["effective"]["detector_rolloff"]["enabled"] is False
    assert fields["edge_artifacts_config"]["effective"]["edge_curvature"]["enabled"] is True
    assert fields["edge_artifacts_config"]["effective"]["edge_curvature"]["left_severity"] == 1.0
    assert fields["edge_artifacts_config"]["effective"]["edge_curvature"]["right_severity"] == 1.0
    assert run.builder_config["nuisance"]["fitted_effect_overrides"] == reconstruction["nested_overrides"]
    assert any(
        "calibrated generator controls" in note
        for note in run.validation_summary["adapter_assumptions"]
    )


def test_fitted_edge_effect_overrides_execute_with_physical_caps() -> None:
    fitted_config = _full_fitted_config()
    fitted_config["edge_artifacts_config"] = {
        "detector_rolloff": {"enabled": True, "severity": 2.0},
        "stray_light": {"enabled": True, "stray_fraction": 0.2, "wavelength_dependent": False},
        "edge_curvature": {
            "enabled": True,
            "curvature_type": "convex",
            "left_severity": 2.0,
            "right_severity": -0.5,
        },
        "truncated_peaks": {"enabled": True, "left_amplitude": 0.2, "right_amplitude": -0.1},
    }

    run = build_dataset_run_from_fitted_config(fitted_config, n_samples=8, random_seed=1234)
    edge_reconstruction = run.builder_config["fitted_config_mapping"]["effect_reconstruction"]["fields"][
        "edge_artifacts_config"
    ]
    effective = edge_reconstruction["effective"]

    assert effective["detector_rolloff"]["enabled"] is True
    assert effective["detector_rolloff"]["severity"] == edge_reconstruction["anchor_rolloff_cap"]
    assert effective["stray_light"]["enabled"] is True
    assert effective["stray_light"]["stray_fraction"] == 0.02
    assert effective["stray_light"]["wavelength_dependent"] is False
    assert effective["edge_curvature"]["enabled"] is True
    assert effective["edge_curvature"]["curvature_type"] == "smile"
    assert effective["edge_curvature"]["left_severity"] == 1.0
    assert effective["edge_curvature"]["right_severity"] == 0.0
    assert effective["truncated_peaks"]["enabled"] is True
    assert effective["truncated_peaks"]["left_amplitude"] == 0.1
    assert effective["truncated_peaks"]["right_amplitude"] == 0.0


def test_explicit_fitted_boundary_components_are_added_without_n_components_imputation() -> None:
    fitted_config = _full_fitted_config()
    fitted_config["n_components"] = 99
    fitted_config["boundary_components_config"] = {
        "components": [{
            "name": "boundary_left",
            "band_center": 1050.0,
            "bandwidth": 140.0,
            "amplitude": 0.4,
            "edge": "left",
        }]
    }

    run = build_dataset_run_from_fitted_config(fitted_config, n_samples=8, random_seed=1234)
    reconstruction = run.builder_config["fitted_config_mapping"]["effect_reconstruction"]
    boundary = reconstruction["fields"]["boundary_components_config"]
    generation_metadata = run.metadata["generation_metadata"]
    unsupported_fields = {field["field"] for field in run.validation_summary["unsupported_fields"]}

    assert boundary["mapping"] == "explicit_fitted_boundary_components_added_to_component_library"
    assert boundary["effective"]["components"][0]["name"] == "boundary_left"
    assert boundary["effective"]["components"][0]["band_center"] == 1050.0
    assert reconstruction["nested_overrides"]["boundary_components_config"] == boundary["effective"]
    assert generation_metadata["n_components"] == 4
    assert generation_metadata["a3_source_component_count"] == 3
    assert "boundary_left" in generation_metadata["component_names"]
    assert "n_components" in unsupported_fields


def test_fitted_config_instrument_override_is_explicitly_reported() -> None:
    run = build_dataset_run_from_fitted_config(
        _full_fitted_config() | {"instrument": "asd_fieldspec"},
        n_samples=8,
        random_seed=1234,
        instrument_override="foss_xds",
        instrument_override_reason="test source provenance",
    )

    resolution = run.metadata["real_fit_adapter"]["instrument_resolution"]
    assert run.metadata["instrument"]["key"] == "foss_xds"
    assert run.builder_config["features"]["fitted_instrument"] == "asd_fieldspec"
    assert resolution["fitted_instrument"] == "asd_fieldspec"
    assert resolution["effective_instrument"] == "foss_xds"
    assert resolution["source"] == "instrument_override"
    assert resolution["reason"] == "test source provenance"


def test_fitted_config_component_override_is_oracle_metadata() -> None:
    run = build_dataset_run_from_fitted_config(
        _full_fitted_config() | {"components": ["starch", "protein"]},
        n_samples=8,
        random_seed=1234,
        component_override=["starch", "protein", "moisture"],
        component_override_reason="test source component provenance",
    )

    resolution = run.metadata["real_fit_adapter"]["component_resolution"]
    assert run.latent_metadata["component_keys"] == ["starch", "protein", "moisture"]
    assert run.builder_config["features"]["components"] == ["starch", "protein", "moisture"]
    assert run.builder_config["features"]["fitted_components"] == ["starch", "protein"]
    assert resolution["fitted_components"] == ["starch", "protein"]
    assert resolution["effective_components"] == ["starch", "protein", "moisture"]
    assert resolution["source"] == "source_provenance_component_override"
    assert resolution["reason"] == "test source component provenance"
    assert resolution["fitted_only_executable_contract"] is False
    assert any("oracle ablation" in note for note in run.validation_summary["adapter_assumptions"])


def test_fitted_residual_effects_config_is_serializable_finite_and_non_oracle() -> None:
    source = build_dataset_run_from_fitted_config(
        _full_fitted_config(),
        n_samples=16,
        random_seed=222,
    )
    config = fit_observable_residual_effects(source.X, source.wavelengths, _full_fitted_config())

    json.dumps(config)
    serialized = json.dumps(config)
    fit = config["fit"]
    assert config["source"] == "observed_fitted_spectra"
    assert config["oracle_provenance"] is False
    assert config["no_oracle"] is True
    assert config["enabled"] is True
    assert config["version"].startswith("A3.2")
    assert set(config["observed_distributions"]) >= {
        "derivative_std",
        "peak_count",
        "peak_density",
        "baseline_curvature",
    }
    assert fit["local_details"]["model"] == "clustered_peak_ridge_distributions_not_rows"
    assert fit["local_details"]["templates"]
    assert np.isfinite(_numeric_leaves(config)).all()
    assert "latent_metadata" not in serialized
    assert "source_rows" not in serialized


def test_fitted_residual_effects_document_quantile_matching_and_bound_json_shape() -> None:
    source = build_dataset_run_from_fitted_config(
        _full_fitted_config(),
        n_samples=24,
        random_seed=223,
    )
    config = fit_observable_residual_effects(source.X, source.wavelengths, _full_fitted_config())
    serialized = json.dumps(config)
    application = config["application"]

    assert application["match_observed_metric_distributions"] is True
    assert application["quantile_matching"] == {
        "scope": "fitted_observed_metric_distributions",
        "sampling": "inverse_cdf_quantile_sampling",
        "oracle_provenance": False,
        "row_level_values_serialized": False,
        "auditable": True,
    }
    assert len(config["fit"]["local_details"]["templates"]) <= 64
    assert "source_provenance" not in serialized
    assert "instrument_override" not in serialized
    assert "component_override" not in serialized
    assert "latent_metadata" not in serialized
    assert "source_rows" not in serialized
    assert "residual_rows" not in serialized
    _assert_bounded_config_lists(config)


def test_fitted_residual_effects_execute_additively_and_change_X() -> None:
    fitted_config = _full_fitted_config()
    source = build_dataset_run_from_fitted_config(fitted_config, n_samples=16, random_seed=333)
    residual_config = fit_observable_residual_effects(source.X, source.wavelengths, fitted_config)
    with_residuals = dict(fitted_config)
    with_residuals["fitted_residual_effects_config"] = residual_config

    baseline = build_dataset_run_from_fitted_config(fitted_config, n_samples=16, random_seed=444)
    regenerated = build_dataset_run_from_fitted_config(with_residuals, n_samples=16, random_seed=444)

    assert baseline.X.shape == regenerated.X.shape
    assert not np.allclose(baseline.X, regenerated.X)
    residual_metadata = regenerated.metadata["generation_metadata"]["a3_fitted_residual_effects"]
    reconstruction = regenerated.builder_config["fitted_config_mapping"]["effect_reconstruction"]
    assert residual_metadata["applied"] is True
    assert residual_metadata["source"] == "observed_fitted_spectra"
    assert residual_metadata["oracle_provenance"] is False
    assert residual_metadata["no_oracle"] is True
    assert residual_metadata["local_detail_templates"] > 0
    assert residual_metadata["active_detail_counts"]["total"] > 0
    assert set(residual_metadata["target_observed_distributions"]) >= {
        "derivative_std",
        "peak_count",
        "peak_density",
        "baseline_curvature",
    }
    assert residual_metadata["max_abs_applied"] > 0.0
    assert reconstruction["fields"]["fitted_residual_effects_config"]["effective"]["enabled"] is True
    assert reconstruction["fields"]["fitted_residual_effects_config"]["effective"]["no_oracle"] is True
    assert (
        regenerated.metadata["real_fit_adapter"]["effect_reconstruction"]["fields"][
            "fitted_residual_effects_config"
        ]["mapping"]
        == "observed_residual_distribution_additive_reconstruction"
    )
    unsupported_fields = {field["field"] for field in regenerated.validation_summary["unsupported_fields"]}
    assert "fitted_residual_effects_config" not in unsupported_fields


def test_fitted_residual_effects_are_deterministic_for_same_seed() -> None:
    fitted_config = _full_fitted_config()
    source = build_dataset_run_from_fitted_config(fitted_config, n_samples=16, random_seed=557)
    residual_config = fit_observable_residual_effects(source.X, source.wavelengths, fitted_config)
    with_residuals = dict(fitted_config)
    with_residuals["fitted_residual_effects_config"] = residual_config

    first = build_dataset_run_from_fitted_config(with_residuals, n_samples=16, random_seed=558)
    second = build_dataset_run_from_fitted_config(with_residuals, n_samples=16, random_seed=558)

    assert np.allclose(first.X, second.X)
    assert np.allclose(first.y, second.y)
    assert np.allclose(
        np.asarray(first.latent_metadata["concentrations"]),
        np.asarray(second.latent_metadata["concentrations"]),
    )
    assert (
        first.metadata["generation_metadata"]["a3_fitted_residual_effects"]
        == second.metadata["generation_metadata"]["a3_fitted_residual_effects"]
    )


def test_fitted_residual_effects_do_not_change_y_or_concentrations() -> None:
    fitted_config = _full_fitted_config()
    target_config = {"type": "regression", "n_targets": 1, "nonlinearity": "moderate"}
    source = build_dataset_run_from_fitted_config(
        fitted_config,
        n_samples=16,
        random_seed=555,
        target_config=target_config,
    )
    residual_config = fit_observable_residual_effects(source.X, source.wavelengths, fitted_config)
    with_residuals = dict(fitted_config)
    with_residuals["fitted_residual_effects_config"] = residual_config

    baseline = build_dataset_run_from_fitted_config(
        fitted_config,
        n_samples=16,
        random_seed=556,
        target_config=target_config,
    )
    regenerated = build_dataset_run_from_fitted_config(
        with_residuals,
        n_samples=16,
        random_seed=556,
        target_config=target_config,
    )

    assert not np.allclose(baseline.X, regenerated.X)
    assert np.allclose(baseline.y, regenerated.y)
    assert np.allclose(
        np.asarray(baseline.latent_metadata["concentrations"]),
        np.asarray(regenerated.latent_metadata["concentrations"]),
    )


def test_fitted_config_instrument_override_is_not_fitted_only_contract() -> None:
    run = build_dataset_run_from_fitted_config(
        _full_fitted_config() | {"instrument": "asd_fieldspec"},
        n_samples=8,
        random_seed=1234,
        instrument_override="foss_xds",
        instrument_override_reason="test source provenance",
    )

    assert run.metadata["real_fit_adapter"]["instrument_resolution"]["fitted_only_executable_contract"] is False


def test_nested_fitted_config_values_are_executable_mappings() -> None:
    fitted_config = _full_fitted_config()
    fitted_config["temperature_config"] = {
        "sample_temperature": 31.0,
        "temperature_variation": 2.5,
        "reference_temperature": 22.0,
        "enable_shift": False,
        "enable_intensity": False,
        "enable_broadening": False,
        "region_specific": False,
    }
    fitted_config["moisture_config"] = {
        "water_activity": 0.62,
        "moisture_content": 0.04,
        "free_water_fraction": 0.2,
        "bound_water_shift": 18.0,
        "reference_aw": 0.45,
    }
    fitted_config["particle_size_config"] = {
        "mean_size_um": 80.0,
        "std_size_um": 12.0,
        "reference_size_um": 100.0,
        "size_effect_strength": 0.8,
        "wavelength_exponent": 1.1,
        "include_path_length_effect": False,
        "path_length_sensitivity": 0.25,
    }
    fitted_config["emsc_config"] = {
        "multiplicative_scatter_std": 0.02,
        "additive_scatter_std": 0.03,
        "polynomial_order": 3,
        "include_wavelength_terms": True,
        "wavelength_coef_std": 0.007,
    }
    fitted_config["edge_artifacts_config"] = {
        "detector_rolloff": {"enabled": True, "detector_model": "ingaas_standard", "severity": 0.2},
        "stray_light": {"enabled": True, "stray_fraction": 0.002, "wavelength_dependent": False},
        "edge_curvature": {
            "enabled": True,
            "curvature_type": "smile",
            "left_severity": 0.4,
            "right_severity": 0.2,
        },
        "truncated_peaks": {"enabled": True, "left_amplitude": 0.03, "right_amplitude": 0.04},
    }
    nuisance = {
        "temperature_c": 25.0,
        "particle_size_um": 150.0,
        "custom_params": {"scatter_alpha_std": 0.1, "scatter_beta_std": 0.2},
        "edge_artifacts": {"rolloff_severity": 0.3},
    }

    env = _environmental_config(fitted_config, nuisance)
    scattering = _scattering_config(fitted_config, nuisance)
    edge = _edge_artifacts_config(fitted_config, nuisance)
    run = build_dataset_run_from_fitted_config(fitted_config, n_samples=8, random_seed=1234)
    unsupported_fields = {field["field"] for field in run.validation_summary["unsupported_fields"]}

    assert env.temperature.sample_temperature == 31.0
    assert env.temperature.reference_temperature == 22.0
    assert env.temperature.enable_shift is False
    assert env.temperature.enable_intensity is False
    assert env.temperature.enable_broadening is False
    assert env.temperature.region_specific is False
    assert env.moisture.moisture_content == 0.04
    assert env.moisture.bound_water_shift == 18.0
    assert env.moisture.reference_aw == 0.45
    assert scattering.particle_size.reference_size_um == 100.0
    assert scattering.particle_size.wavelength_exponent == 1.1
    assert scattering.particle_size.include_path_length_effect is False
    assert scattering.particle_size.path_length_sensitivity == 0.25
    assert scattering.emsc.polynomial_order == 3
    assert scattering.emsc.wavelength_coef_std == 0.007
    assert run.builder_config["fitted_config_mapping"]["effect_reconstruction"]["fields"]["emsc_config"][
        "effective"
    ]["wavelength_coef_std"] == 0.007
    assert edge.detector_model == "ingaas_standard"
    assert edge.stray_wavelength_dependent is False
    assert edge.curvature_type == "smile"
    assert edge.right_curvature_severity == 0.2
    assert edge.left_peak_amplitude == 0.03
    assert "temperature_config.enable_shift" not in unsupported_fields
    assert "particle_size_config.wavelength_exponent" not in unsupported_fields
    assert "emsc_config.wavelength_coef_std" not in unsupported_fields
    assert "edge_artifacts_config.truncated_peaks.left_amplitude" not in unsupported_fields


def test_nested_fitted_config_reports_unsupported_subfields() -> None:
    fitted_config = _full_fitted_config()
    fitted_config["temperature_config"] = {"custom_regions": {"water": [1400, 1500]}}
    fitted_config["moisture_config"] = {"temperature_interaction": True}
    fitted_config["particle_size_config"] = {"distribution": "lognormal"}
    fitted_config["emsc_config"] = {"reference_spectrum": [1.0, 2.0, 3.0]}
    fitted_config["edge_artifacts_config"] = {
        "detector_rolloff": {"calibration_file": "detector.json"},
        "unknown_artifact": {"enabled": True},
    }

    run = build_dataset_run_from_fitted_config(fitted_config, n_samples=8, random_seed=1234)

    unsupported_fields = {field["field"] for field in run.validation_summary["unsupported_fields"]}
    assert "temperature_config.custom_regions" in unsupported_fields
    assert "moisture_config.temperature_interaction" in unsupported_fields
    assert "particle_size_config.distribution" in unsupported_fields
    assert "emsc_config.reference_spectrum" in unsupported_fields
    assert "edge_artifacts_config.detector_rolloff.calibration_file" in unsupported_fields
    assert "edge_artifacts_config.unknown_artifact" in unsupported_fields


def test_fitted_config_strict_mode_fails_when_unsupported_fields_exist() -> None:
    with pytest.raises(FittedConfigAdapterError) as exc:
        build_dataset_run_from_fitted_config(
            _full_fitted_config(),
            n_samples=8,
            random_seed=7,
            fail_on_unsupported=True,
        )

    assert exc.value.validation_summary["failures"][0]["reason"] == "unsupported_fitted_fields"
    assert exc.value.validation_summary["unsupported_fields"]


def test_fitted_config_without_components_fails_explicitly() -> None:
    fitted_config = _full_fitted_config()
    fitted_config["components"] = []

    with pytest.raises(FittedConfigAdapterError) as exc:
        fitted_config_to_prior_record(fitted_config, n_samples=8, random_seed=7)

    assert exc.value.validation_summary["failures"][0]["reason"] == "missing_fitted_components"
    assert "unsupported_fields" in exc.value.validation_summary


def test_fit_synthetic_then_regenerate_smoke_contract() -> None:
    source = build_dataset_run_from_fitted_config(
        _full_fitted_config(),
        n_samples=20,
        random_seed=44,
    )
    params = RealDataFitter().fit(source.X, wavelengths=source.wavelengths, name="a3_test_source")
    fitted_config = params.to_full_config()

    regenerated = build_dataset_run_from_fitted_config(
        fitted_config,
        n_samples=20,
        random_seed=45,
    )

    assert regenerated.X.shape == source.X.shape
    assert regenerated.wavelengths.shape == source.wavelengths.shape
    assert np.all(np.diff(regenerated.wavelengths) > 0)
    assert regenerated.validation_summary["status"] == "passed"
    assert "unsupported_fields" in regenerated.validation_summary


def test_report_marks_failed_similarity_as_blocking_gate() -> None:
    result = {
        "status": "passed",
        "contract_status": "passed",
        "scientific_status": "failed",
        "source": _summary("source"),
        "regenerated": _summary("regenerated"),
        "unsupported_fields": [],
        "adapter_assumptions": [],
        "comparison": {
            "mean_spectrum_mae": 0.1,
            "mean_spectrum_rmse": 0.2,
            "global_mean_abs_gap": 0.05,
        },
        "scorecard": {
            "overall_pass": False,
            "metric_results": [{
                "metric": "correlation_length",
                "value": 0.2,
                "threshold": 0.7,
                "passed": False,
                "details": {},
            }],
        },
    }

    report = _load_real_fit_experiment().render_markdown(
        result,
        n_samples=4,
        source_seed=1,
        regen_seed=2,
        git_status={"returncode": 0, "entry_count": 0, "by_status": {}, "sample": []},
    )

    assert "Scientific similarity status: `failed`" in report
    assert "A3 gate blocked" in report
    assert "Blocked: A3 contract is executable" in report
    assert "Fitted Nuisance/Effect Reconstruction" in report


def test_report_keeps_oracle_pass_separate_from_fitted_only_gate() -> None:
    result = {
        "status": "passed",
        "contract_status": "passed",
        "scientific_status": "failed",
        "variant_name": "fitted_instrument",
        "source": _summary("source") | {"components": ["starch", "protein", "moisture"]},
        "regenerated": _summary("regenerated") | {"components": ["starch", "protein"]},
        "fitted_config": {"components": ["starch", "protein"]},
        "unsupported_fields": [],
        "adapter_assumptions": [],
        "comparison": {
            "mean_spectrum_mae": 0.1,
            "mean_spectrum_rmse": 0.2,
            "global_mean_abs_gap": 0.05,
        },
        "scorecard": {
            "overall_pass": False,
            "metric_results": [{
                "metric": "correlation_length",
                "value": 0.2,
                "threshold": 0.7,
                "passed": False,
                "details": {},
            }],
        },
        "ablations": [{
            "variant_name": "source_components_override",
            "status": "passed",
            "contract_status": "passed",
            "scientific_status": "passed",
            "diagnostic_scope": "oracle_source_provenance",
            "regenerated": _summary("oracle") | {
                "instrument": "foss_xds",
                "components": ["starch", "protein", "moisture"],
            },
            "comparison": {
                "mean_spectrum_mae": 0.01,
                "mean_spectrum_rmse": 0.02,
                "global_mean_abs_gap": 0.005,
            },
            "scorecard": {"overall_pass": True, "metric_results": []},
            "instrument_override_reason": None,
            "component_override_reason": "oracle source components",
        }],
    }

    report = _load_real_fit_experiment().render_markdown(
        result,
        n_samples=4,
        source_seed=1,
        regen_seed=2,
        git_status={"returncode": 0, "entry_count": 0, "by_status": {}, "sample": []},
    )

    assert "Fitted-only gate variant: `fitted_instrument`" in report
    assert "Oracle/source-provenance success does not satisfy the fitted-only A3 gate" in report


def test_primary_variant_keeps_fitted_only_gate_ahead_of_oracle_ablations() -> None:
    module = _load_real_fit_experiment()
    variants = [
        {"variant_name": "source_components_override", "scientific_status": "passed"},
        {"variant_name": "fitted_instrument", "scientific_status": "failed"},
    ]

    primary = module._primary_variant(variants)

    assert primary["variant_name"] == "fitted_instrument"


def _full_fitted_config() -> dict[str, object]:
    return {
        "wavelength_start": 1100.0,
        "wavelength_end": 2500.0,
        "wavelength_step": 8.0,
        "complexity": "realistic",
        "noise_base": 0.003,
        "noise_signal_dep": 0.008,
        "scatter_alpha_std": 0.04,
        "scatter_beta_std": 0.01,
        "path_length_std": 0.05,
        "baseline_amplitude": 0.02,
        "tilt_std": 0.01,
        "global_slope_mean": 0.05,
        "global_slope_std": 0.03,
        "instrument": "foss_xds",
        "measurement_mode": "reflectance",
        "domain": "agriculture_grain",
        "components": ["starch", "protein", "moisture"],
        "n_components": 5,
        "temperature_config": {"temperature_variation": 2.5},
        "moisture_config": {"water_activity": 0.5, "moisture_content": 0.02},
        "particle_size_config": {
            "mean_size_um": 80.0,
            "std_size_um": 12.0,
            "size_effect_strength": 1.0,
        },
        "emsc_config": {
            "multiplicative_scatter_std": 0.02,
            "additive_scatter_std": 0.03,
            "polynomial_order": 2,
            "include_wavelength_terms": True,
        },
        "edge_artifacts_config": {
            "detector_rolloff": {"enabled": True, "severity": 0.2},
            "edge_curvature": {"enabled": True, "left_severity": 0.4},
        },
        "boundary_components_config": {"left": {"center": 1110.0}},
        "preprocessing_type": "raw_absorbance",
        "is_preprocessed": False,
    }


def _numeric_leaves(value: object) -> np.ndarray:
    leaves: list[float] = []
    if isinstance(value, dict):
        for nested in value.values():
            leaves.extend(_numeric_leaves(nested).tolist())
    elif isinstance(value, list):
        for nested in value:
            leaves.extend(_numeric_leaves(nested).tolist())
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        leaves.append(float(value))
    return np.asarray(leaves, dtype=float)


def _assert_bounded_config_lists(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_bounded_config_lists(nested, (*path, str(key)))
        return
    if isinstance(value, list):
        assert len(value) <= 64, ".".join(path)
        for nested in value:
            _assert_bounded_config_lists(nested, path)


def _summary(name: str) -> dict[str, object]:
    return {
        "domain": "agriculture_grain",
        "instrument": "foss_xds",
        "mode": "reflectance",
        "target_type": "regression",
        "validation_summary": {
            "status": "passed",
            "checks": {
                "shape": True,
                "finite": True,
                "wavelengths_monotonic": True,
                "target_contract": True,
            },
            "summary": {
                "X_shape": [4, 3],
                "y_shape": [4],
                "wavelength_range_nm": [1100.0, 1116.0],
            },
        },
        "name": name,
    }


def _load_real_fit_experiment() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments" / "exp00_real_fit_adapter.py"
    spec = importlib.util.spec_from_file_location("exp00_real_fit_adapter", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
