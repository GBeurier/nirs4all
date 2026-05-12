"""R2c sentinel matrix remediation contract tests for the bench-only builder adapter.

These tests guard the bench-only opt-in remediation path:

- the default code path (``remediation_profile=None``) is unchanged;
- the only supported profile is ``"r2c_sentinel_matrix_v1"``;
- the per-domain composition + optical-path scale are deterministic and never
  read real spectra, labels, splits, targets, or AUC;
- audit flags emitted under ``metadata["r2c_mechanistic_remediation"]`` are all
  false for non-oracle conditions and document the mechanistic constants;
- the petrochem_fuels remediation reduces uncalibrated_raw mean / std /
  amplitude relative to the unremediated baseline (mechanistic effect, not a
  fit to any real-data metric).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from nirsyntheticpfn.adapters.builder_adapter import (
    _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_ABSORBANCE,
    _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SOURCE,
    _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SUPPORT_NM,
    _R9C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
    _R9C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
    _R9C_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM,
    _R9C_PETROCHEM_FUELS_CONSTANTS_SOURCE,
    _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE,
    _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM,
    _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM,
    _R9C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
    _R9C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
    _R9C_PETROCHEM_FUELS_SUPPORT_NM,
    _R9D_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
    _R9D_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM,
    _R9D_PETROCHEM_FUELS_CONSTANTS_SOURCE,
    _R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE,
    _R9D_PETROCHEM_FUELS_RENORM_EPSILON,
    _R9D_PETROCHEM_FUELS_SHAPE_CLIP,
    _R9D_PETROCHEM_FUELS_SUPPORT_NM,
    _R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE,
    _R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE,
    _R9E_PETROCHEM_FUELS_SUPPORT_NM,
    _R9F_PETROCHEM_FUELS_CONSTANTS_SOURCE,
    _R9F_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE,
    _R9F_PETROCHEM_FUELS_SUPPORT_NM,
    ALL_REMEDIATION_PROFILES,
    R2C_REMEDIATION_PROFILES,
    R2D_REMEDIATION_PROFILES,
    R2F_REMEDIATION_PROFILES,
    R2G_REMEDIATION_PROFILES,
    R2H_REMEDIATION_PROFILES,
    R2I_REMEDIATION_PROFILES,
    R2J_REMEDIATION_PROFILES,
    R2K_REMEDIATION_PROFILES,
    R2L_REMEDIATION_PROFILES,
    R2M_REMEDIATION_PROFILES,
    R2N_REMEDIATION_PROFILES,
    R2O_REMEDIATION_PROFILES,
    R2P_REMEDIATION_PROFILES,
    R2Q_REMEDIATION_PROFILES,
    R2R_REMEDIATION_PROFILES,
    R2S_REMEDIATION_PROFILES,
    R2T_REMEDIATION_PROFILES,
    R2U_REMEDIATION_PROFILES,
    R2V_REMEDIATION_PROFILES,
    R2W_REMEDIATION_PROFILES,
    R2X_REMEDIATION_PROFILES,
    R2Y_REMEDIATION_PROFILES,
    R2Z_REMEDIATION_PROFILES,
    R3A_REMEDIATION_PROFILES,
    R3B_REMEDIATION_PROFILES,
    R3C_REMEDIATION_PROFILES,
    R3D_REMEDIATION_PROFILES,
    R3E_REMEDIATION_PROFILES,
    R3F_REMEDIATION_PROFILES,
    R3G_REMEDIATION_PROFILES,
    R4A_REMEDIATION_PROFILES,
    R4B_REMEDIATION_PROFILES,
    R4C_REMEDIATION_PROFILES,
    R5A_REMEDIATION_PROFILES,
    R5B_REMEDIATION_PROFILES,
    R5C_REMEDIATION_PROFILES,
    R6A_REMEDIATION_PROFILES,
    R7A_REMEDIATION_PROFILES,
    R8A_REMEDIATION_PROFILES,
    R8B_REMEDIATION_PROFILES,
    R9B_REMEDIATION_PROFILES,
    R9C_REMEDIATION_PROFILES,
    R9D_REMEDIATION_PROFILES,
    R9E_REMEDIATION_PROFILES,
    R9F_REMEDIATION_PROFILES,
    R9H_REMEDIATION_PROFILES,
    R9I_REMEDIATION_PROFILES,
    R9J_REMEDIATION_PROFILES,
    R9K_REMEDIATION_PROFILES,
    _convolve_rows,
    _gaussian_kernel,
    build_synthetic_dataset_run,
)
from nirsyntheticpfn.adapters.prior_adapter import (
    BENCH_ONLY_COMPONENT_ALIAS_SCOPE,
    PriorCanonicalizationError,
    canonicalize_domain,
    canonicalize_prior_config,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config


def _typical_domain_components(domain_key: str, n: int) -> list[str]:
    components: list[str] = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n:
            break
    if len(components) < n:
        raise AssertionError(f"not enough valid components for {domain_key}")
    return components


def _fuel_diesel_source(*, seed: int) -> dict[str, object]:
    """Source config for petrochem_fuels with the bench-only ``diesel`` alias."""
    return {
        "domain": "petrochem_fuels",
        "domain_category": "petrochemical",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (900, 1700),
        "spectral_resolution": 4.0,
        "measurement_mode": "transmittance",
        "matrix_type": "liquid",
        "temperature": 25.0,
        "particle_size": 2.0,
        "noise_level": 1.0,
        "components": ["diesel", "alkane", "aromatic"],
        "n_samples": 80,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _dairy_emulsion_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "food_dairy",
        "domain_category": "food",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (1100, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "transflectance",
        "matrix_type": "emulsion",
        "temperature": 25.0,
        "particle_size": 2.0,
        "noise_level": 1.0,
        "components": _typical_domain_components("food_dairy", 4),
        "n_samples": 80,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _grain_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "grain",
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (1100, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "powder",
        "temperature": 25.0,
        "particle_size": 75.0,
        "noise_level": 1.0,
        "components": _typical_domain_components("agriculture_grain", 3),
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _assert_audit_non_oracle(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2c_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2d(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2d_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2f(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2f_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2g(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2g_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2h(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2h_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2i(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2i_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2j(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2j_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2k(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2k_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2l(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2l_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2m(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2m_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2n(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2n_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2p(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2p_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2q(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2q_sentinel_matrix_remediation"


def _assert_audit_non_oracle_r2r(audit: dict[str, Any]) -> None:
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert audit["scope"] == "bench_only_r2r_sentinel_matrix_remediation"


def _beer_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "beverage_wine",
        "domain_category": "food",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (1100, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "transmittance",
        "matrix_type": "liquid",
        "temperature": 25.0,
        "particle_size": 2.0,
        "noise_level": 1.0,
        "components": ["water", "ethanol", "glucose", "fructose", "glycerol"],
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _corn_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "agriculture_grain",
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (1100, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "powder",
        "temperature": 25.0,
        "particle_size": 75.0,
        "noise_level": 1.0,
        "components": ["starch", "protein", "moisture", "lipid", "cellulose"],
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _r3a_corn_source(*, seed: int) -> dict[str, object]:
    source = _corn_source(seed=seed)
    source["_r3a_corn_readout_route"] = {
        "enabled": True,
        "route_marker": "corn",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3b_corn_source(*, seed: int) -> dict[str, object]:
    source = _corn_source(seed=seed)
    source["_r3b_corn_readout_route"] = {
        "enabled": True,
        "route_marker": "corn",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _juice_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "beverage_juice",
        "domain_category": "food",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (900, 1100),
        "spectral_resolution": 4.0,
        "measurement_mode": "transmittance",
        "matrix_type": "liquid",
        "temperature": 25.0,
        "particle_size": 2.0,
        "noise_level": 1.0,
        "components": [
            "water",
            "glucose",
            "fructose",
            "sucrose",
            "citric_acid",
            "malic_acid",
            "carotenoid",
        ],
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _fruit_puree_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "fruit",
        "domain_category": "agriculture",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (900, 1100),
        "spectral_resolution": 4.0,
        "measurement_mode": "transflectance",
        "matrix_type": "paste",
        "temperature": 25.0,
        "particle_size": 35.0,
        "noise_level": 1.0,
        "components": [
            "water",
            "glucose",
            "fructose",
            "sucrose",
            "cellulose",
            "starch",
            "malic_acid",
            "citric_acid",
            "carotenoid",
        ],
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _soil_source(*, seed: int) -> dict[str, object]:
    return {
        "domain": "environmental_soil",
        "domain_category": "environmental",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (1100, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "powder",
        "temperature": 25.0,
        "particle_size": 75.0,
        "noise_level": 1.0,
        "components": [
            "moisture",
            "carbonates",
            "kaolinite",
            "gypsum",
            "cellulose",
            "lignin",
            "protein",
        ],
        "n_samples": 60,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def test_default_path_is_unchanged_when_remediation_profile_is_none() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))

    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    explicit_none = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile=None,
    )

    np.testing.assert_allclose(baseline.X, explicit_none.X)
    np.testing.assert_allclose(baseline.y, explicit_none.y)
    np.testing.assert_allclose(baseline.wavelengths, explicit_none.wavelengths)
    assert baseline.metadata == explicit_none.metadata
    assert "r2c_mechanistic_remediation" not in baseline.metadata
    assert "r2c_mechanistic_remediation" not in explicit_none.metadata


def test_supported_profiles_only_contain_v1() -> None:
    assert R2C_REMEDIATION_PROFILES == ("r2c_sentinel_matrix_v1",)


def test_unknown_remediation_profile_raises_value_error() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=7))
    with pytest.raises(ValueError):
        build_synthetic_dataset_run(
            record,
            n_samples=20,
            random_seed=99,
            remediation_profile="not_a_real_profile",
        )


def test_remediation_is_deterministic_for_same_seed() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=7))

    first = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2c_sentinel_matrix_v1",
    )
    second = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.y, second.y)
    audit_a = first.metadata["r2c_mechanistic_remediation"]
    audit_b = second.metadata["r2c_mechanistic_remediation"]
    assert audit_a == audit_b
    assert audit_a["concentration_seed"] == audit_b["concentration_seed"]
    assert audit_a["spectra_seed"] == audit_b["spectra_seed"]


def test_petrochem_fuels_remediation_reduces_mean_std_amplitude_vs_baseline() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))

    baseline = build_synthetic_dataset_run(record, n_samples=64, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=64,
        random_seed=4242,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    base_mean = float(baseline.X.mean())
    rem_mean = float(remediated.X.mean())
    base_std = float(baseline.X.std(ddof=0))
    rem_std = float(remediated.X.std(ddof=0))
    base_amp = float(np.median(np.percentile(baseline.X, 95.0, axis=1) - np.percentile(baseline.X, 5.0, axis=1)))
    rem_amp = float(np.median(np.percentile(remediated.X, 95.0, axis=1) - np.percentile(remediated.X, 5.0, axis=1)))

    # Mechanistic short liquid pathlength + tight diesel-centered Dirichlet
    # both reduce raw absorbance scale.
    assert rem_mean < base_mean, (
        f"remediated mean {rem_mean:.4f} must be < baseline {base_mean:.4f}"
    )
    assert rem_std < base_std, (
        f"remediated std {rem_std:.4f} must be < baseline {base_std:.4f}"
    )
    assert rem_amp < base_amp, (
        f"remediated amplitude {rem_amp:.4f} must be < baseline {base_amp:.4f}"
    )


def test_petrochem_fuels_remediation_audit_records_mechanistic_constants() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=4242,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2c_sentinel_matrix_v1"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "short_liquid_optical_path_scale"
    assert params["composition_source"] == "textbook_diesel_composition"
    assert params["spectra_source"] == "beer_lambert_short_path"
    assert isinstance(params["alphas"], dict)
    assert params["alpha_sum"] > params["n_components"], (
        "tight Dirichlet must have alpha_sum > n_components for low variance"
    )
    low, high = params["path_factor_range"]
    assert 0.0 < low < high < 1.0, (
        "short pathlength scale must be a strictly attenuating range in (0, 1)"
    )
    _assert_audit_non_oracle(audit)


def test_food_dairy_remediation_audit_records_emulsion_rule() -> None:
    record = canonicalize_prior_config(_dairy_emulsion_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=2024,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["domain_key"] == "food_dairy"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_milk_emulsion_centered"
    assert params["spectra_rule"] == "transflectance_raw_intensity_scale"
    assert params["composition_source"] == "textbook_dairy_emulsion_composition"
    assert params["spectra_source"] == "double_pass_emulsion_attenuation"
    _assert_audit_non_oracle(audit)


def test_remediation_does_not_change_targets_or_split_for_unsupported_domain() -> None:
    """Domains without an R2c rule keep concentrations/spectra unchanged but still audit."""
    record = canonicalize_prior_config(_grain_source(seed=21))

    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=512)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=512,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    # Audit must report enabled but applied=False because the grain domain has
    # no R2c remediation rule. X is therefore identical to the baseline.
    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["domain_key"] == canonicalize_domain("grain")
    assert audit["applied_to_concentrations"] is False
    assert audit["applied_to_spectra"] is False
    np.testing.assert_allclose(remediated.X, baseline.X)


def test_diesel_alias_is_recorded_in_source_prior_config() -> None:
    """The bench-only ``diesel`` alias must surface in the canonical record audit."""
    record = canonicalize_prior_config(_fuel_diesel_source(seed=7))

    audit = record.source_prior_config["_bench_component_aliases"]
    assert audit["scope"] == BENCH_ONLY_COMPONENT_ALIAS_SCOPE
    assert audit["applied"] is True
    assert audit["non_oracle"] is True
    assert audit["no_target_or_label"] is True
    assert audit["real_stat_capture"] is False
    translations = audit["translations"]
    assert any(
        t["raw_component"] == "diesel" and t["canonical_component"] == "oil"
        for t in translations
    )
    # Canonical components must contain the translated label, not "diesel".
    assert "oil" in record.component_keys
    assert "diesel" not in record.component_keys


def test_diesel_alias_does_not_weaken_non_sequence_components_validation() -> None:
    source = _fuel_diesel_source(seed=7)
    source["components"] = "diesel"

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts == {"invalid_components": 1}


def test_component_aliases_do_not_accept_unknown_components_outside_alias_table() -> None:
    source = _fuel_diesel_source(seed=7)
    source["components"] = ["__not_a_registered_component__", "alkane", "aromatic"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts["invalid_component"] == 1


def test_diesel_alias_is_domain_scoped() -> None:
    source = _grain_source(seed=7)
    source["components"] = ["diesel", "starch", "protein"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts["domain_component_mismatch"] == 1


def test_remediation_audit_does_not_modify_thresholds_or_metrics() -> None:
    """Strict guard against any oracle / threshold / metric leak in the audit."""
    record = canonicalize_prior_config(_fuel_diesel_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=2024,
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    params = audit["transform_params"]
    assert not (set(params) & forbidden_keys)


# ---------------------------------------------------------------------------
# R2d sentinel matrix remediation profile (opt-in superset of R2c).
# ---------------------------------------------------------------------------


def test_r2d_profile_is_opt_in_and_listed_in_supported_profiles() -> None:
    assert R2D_REMEDIATION_PROFILES == ("r2d_sentinel_matrix_v1",)
    assert "r2d_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    # R2c profile must remain stable in the union too.
    assert "r2c_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES


def test_r2d_default_path_is_unchanged_when_remediation_profile_is_none() -> None:
    record = canonicalize_prior_config(_beer_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata


def test_r2d_unknown_profile_rejected() -> None:
    record = canonicalize_prior_config(_beer_source(seed=7))
    with pytest.raises(ValueError):
        build_synthetic_dataset_run(
            record,
            n_samples=20,
            random_seed=99,
            remediation_profile="r2e_made_up",
        )


def test_r2d_diesel_rule_reuses_r2c_mechanistic_constants_for_diesel_domain() -> None:
    """R2d must reuse the R2c DIESEL mechanistic constants (alphas, range, labels).

    The per-sample random draw differs because the profile name is mixed into
    the deterministic seed namespace; that is by design (audit isolation per
    profile). The mechanistic rule constants must match exactly.
    """
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    r2c_run = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=4242,
        remediation_profile="r2c_sentinel_matrix_v1",
    )
    r2d_run = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    r2c_audit = r2c_run.metadata["r2c_mechanistic_remediation"]
    r2d_audit = r2d_run.metadata["r2c_mechanistic_remediation"]
    assert r2d_audit["profile"] == "r2d_sentinel_matrix_v1"
    assert r2d_audit["scope"] == "bench_only_r2d_sentinel_matrix_remediation"
    for key in (
        "composition_rule",
        "spectra_rule",
        "composition_source",
        "spectra_source",
        "alphas",
        "path_factor_range",
    ):
        assert (
            r2c_audit["transform_params"][key] == r2d_audit["transform_params"][key]
        ), f"r2d must reuse r2c {key!r} for diesel domain"


def test_r2d_milk_rule_reuses_r2c_mechanistic_constants_for_dairy_domain() -> None:
    record = canonicalize_prior_config(_dairy_emulsion_source(seed=13))
    r2c_run = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=2024,
        remediation_profile="r2c_sentinel_matrix_v1",
    )
    r2d_run = build_synthetic_dataset_run(
        record,
        n_samples=32,
        random_seed=2024,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    r2c_audit = r2c_run.metadata["r2c_mechanistic_remediation"]
    r2d_audit = r2d_run.metadata["r2c_mechanistic_remediation"]
    for key in (
        "composition_rule",
        "spectra_rule",
        "composition_source",
        "spectra_source",
        "alphas",
        "path_factor_range",
    ):
        assert (
            r2c_audit["transform_params"][key] == r2d_audit["transform_params"][key]
        ), f"r2d must reuse r2c {key!r} for dairy domain"


def test_r2d_beer_rule_applies_concentrations_and_spectra() -> None:
    record = canonicalize_prior_config(_beer_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2d_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_wine"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_beer_centered"
    assert params["spectra_rule"] == "long_liquid_optical_path_scale"
    assert params["composition_source"] == "textbook_beer_composition"
    assert params["spectra_source"] == "beer_lambert_long_path"
    low, high = params["path_factor_range"]
    # Long liquid path => strictly amplifying scale.
    assert 1.0 < low < high
    assert params["alpha_sum"] > params["n_components"]
    _assert_audit_non_oracle_r2d(audit)


def test_r2d_beer_remediation_amplifies_mean_and_amplitude_vs_baseline() -> None:
    record = canonicalize_prior_config(_beer_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=48, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    base_amp = float(
        np.median(
            np.percentile(baseline.X, 95.0, axis=1)
            - np.percentile(baseline.X, 5.0, axis=1)
        )
    )
    rem_amp = float(
        np.median(
            np.percentile(remediated.X, 95.0, axis=1)
            - np.percentile(remediated.X, 5.0, axis=1)
        )
    )
    # Beer-Lambert long-path scale is strictly > 1, so the per-row amplitude
    # of the remediated spectra must exceed the baseline.
    assert rem_amp > base_amp


def test_r2d_corn_rule_applies_concentrations_smoothing_and_scatter() -> None:
    record = canonicalize_prior_config(_corn_source(seed=21))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2d_sentinel_matrix_v1"
    assert audit["domain_key"] == "agriculture_grain"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_corn_grain_centered"
    assert params["spectra_rule"] == "powder_reflectance_smoothing_and_scatter"
    assert params["composition_source"] == "textbook_corn_grain_composition"
    assert params["spectra_source"] == "instrumental_broadening_and_powder_scatter"
    assert params["smoothing_fwhm_nm"] == 12.0
    assert params["smoothing_kernel_size"] >= 3
    assert params["smoothing_sigma_bins"] > 0.0
    low, high = params["path_factor_range"]
    assert low < high
    _assert_audit_non_oracle_r2d(audit)


def test_r2d_corn_remediation_reduces_derivative_std_vs_baseline() -> None:
    """CORN smoothing must reduce the median per-row derivative std."""
    record = canonicalize_prior_config(_corn_source(seed=21))
    baseline = build_synthetic_dataset_run(record, n_samples=48, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    base_dstd = float(np.median(np.diff(baseline.X, axis=1).std(axis=1, ddof=0)))
    rem_dstd = float(np.median(np.diff(remediated.X, axis=1).std(axis=1, ddof=0)))
    assert rem_dstd < base_dstd, (
        f"r2d corn smoothing must reduce derivative std; got {rem_dstd:.6f} "
        f"vs baseline {base_dstd:.6f}"
    )


def test_gaussian_row_convolution_preserves_constant_spectra() -> None:
    """Smoothing must not create artificial edge attenuation on flat spectra."""
    X = np.full((3, 25), 7.5, dtype=float)
    kernel = _gaussian_kernel(sigma_bins=2.0)

    smoothed = _convolve_rows(X, kernel)

    np.testing.assert_allclose(smoothed, X, rtol=1e-12, atol=1e-12)


def test_r2d_remediation_is_deterministic_for_same_seed() -> None:
    record = canonicalize_prior_config(_corn_source(seed=21))
    first = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    second = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.y, second.y)
    assert (
        first.metadata["r2c_mechanistic_remediation"]
        == second.metadata["r2c_mechanistic_remediation"]
    )


def test_r2d_remediation_audit_does_not_modify_thresholds_or_metrics() -> None:
    record = canonicalize_prior_config(_beer_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=2024,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    audit = remediated.metadata["r2c_mechanistic_remediation"]
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    params = audit["transform_params"]
    assert not (set(params) & forbidden_keys)


# ---------------------------------------------------------------------------
# R2f sentinel matrix remediation profile (opt-in superset of R2d for juice).
# ---------------------------------------------------------------------------


def test_r2f_profile_is_opt_in_and_listed_in_supported_profiles() -> None:
    assert R2F_REMEDIATION_PROFILES == ("r2f_sentinel_matrix_v1",)
    assert "r2f_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2d_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2c_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_juice_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata


def test_r2f_unknown_profile_still_rejected() -> None:
    record = canonicalize_prior_config(_juice_source(seed=7))
    with pytest.raises(ValueError):
        build_synthetic_dataset_run(
            record,
            n_samples=20,
            random_seed=99,
            remediation_profile="r2f_not_a_profile",
        )


def test_r2f_reuses_r2d_constants_for_existing_sentinel_domains() -> None:
    cases = (
        ("diesel", _fuel_diesel_source(seed=11), ("path_factor_range",)),
        ("milk", _dairy_emulsion_source(seed=13), ("path_factor_range",)),
        ("beer", _beer_source(seed=17), ("path_factor_range",)),
        (
            "corn",
            _corn_source(seed=21),
            (
                "path_factor_range",
                "smoothing_fwhm_nm",
                "smoothing_kernel_size",
                "smoothing_sigma_bins",
            ),
        ),
    )
    for label, source, extra_keys in cases:
        record = canonicalize_prior_config(source)
        r2d_run = build_synthetic_dataset_run(
            record,
            n_samples=32,
            random_seed=4242,
            remediation_profile="r2d_sentinel_matrix_v1",
        )
        r2f_run = build_synthetic_dataset_run(
            record,
            n_samples=32,
            random_seed=4242,
            remediation_profile="r2f_sentinel_matrix_v1",
        )
        r2d_params = r2d_run.metadata["r2c_mechanistic_remediation"][
            "transform_params"
        ]
        r2f_audit = r2f_run.metadata["r2c_mechanistic_remediation"]
        r2f_params = r2f_audit["transform_params"]
        assert r2f_audit["profile"] == "r2f_sentinel_matrix_v1"
        assert r2f_audit["scope"] == "bench_only_r2f_sentinel_matrix_remediation"
        for key in (
            "composition_rule",
            "spectra_rule",
            "composition_source",
            "spectra_source",
            "alphas",
            *extra_keys,
        ):
            assert r2d_params[key] == r2f_params[key], (
                f"r2f must reuse r2d {key!r} for {label}"
            )


def test_r2f_beverage_juice_applies_concentrations_and_spectra() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2f_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2f_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_juice"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_fruit_juice_centered"
    assert params["spectra_rule"] == "moderate_liquid_cuvette_path_scale"
    assert params["composition_source"] == "textbook_fruit_juice_composition"
    assert params["spectra_source"] == "beer_lambert_moderate_cuvette_path"
    assert set(params["alphas"]) == {
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "citric_acid",
        "malic_acid",
        "carotenoid",
    }
    for forbidden in ("pectin", "polyphenols", "anthocyanin", "tannins"):
        assert forbidden not in params["alphas"]
    assert params["alphas"]["water"] > params["alphas"]["fructose"]
    assert params["alphas"]["fructose"] > params["alphas"]["citric_acid"]
    assert params["alphas"]["citric_acid"] > params["alphas"]["carotenoid"]
    low, high = params["path_factor_range"]
    assert 1.0 < low < high <= 1.25
    assert params["alpha_sum"] > params["n_components"]
    _assert_audit_non_oracle_r2f(audit)


def test_r2f_beverage_juice_components_are_valid_for_domain() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2f_sentinel_matrix_v1",
    )

    params = remediated.metadata["r2c_mechanistic_remediation"]["transform_params"]
    valid_components = set(get_domain_config("beverage_juice").typical_components)
    assert set(params["alphas"]).issubset(valid_components)


def test_r2d_profile_does_not_pick_up_beverage_juice_rule() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=4242)
    r2d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2d_sentinel_matrix_v1",
    )

    audit = r2d_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2d_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_juice"
    assert audit["applied_to_concentrations"] is False
    assert audit["applied_to_spectra"] is False
    np.testing.assert_allclose(r2d_run.X, baseline.X)
    np.testing.assert_allclose(r2d_run.y, baseline.y)


def test_r2f_beverage_juice_remediation_amplifies_amplitude_vs_baseline() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=48, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    base_amp = float(
        np.median(
            np.percentile(baseline.X, 95.0, axis=1)
            - np.percentile(baseline.X, 5.0, axis=1)
        )
    )
    rem_amp = float(
        np.median(
            np.percentile(remediated.X, 95.0, axis=1)
            - np.percentile(remediated.X, 5.0, axis=1)
        )
    )
    assert rem_amp > base_amp


def test_r2f_remediation_is_deterministic_for_same_seed() -> None:
    record = canonicalize_prior_config(_juice_source(seed=21))
    first = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    second = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=99,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.y, second.y)
    assert (
        first.metadata["r2c_mechanistic_remediation"]
        == second.metadata["r2c_mechanistic_remediation"]
    )


def test_r2f_remediation_audit_does_not_modify_thresholds_or_metrics() -> None:
    record = canonicalize_prior_config(_juice_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=2024,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    audit = remediated.metadata["r2c_mechanistic_remediation"]
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    params = audit["transform_params"]
    assert not (set(params) & forbidden_keys)


# ---------------------------------------------------------------------------
# R2g sentinel matrix remediation profile (opt-in superset of R2f for soil).
# ---------------------------------------------------------------------------


def test_r2g_profile_is_opt_in_listed_and_accepted() -> None:
    assert R2G_REMEDIATION_PROFILES == ("r2g_sentinel_matrix_v1",)
    assert "r2g_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2f_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_soil_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata

    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    assert (
        remediated.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r2g_sentinel_matrix_v1"
    )


def test_r2g_reuses_r2f_constants_for_existing_sentinel_domains() -> None:
    cases = (
        ("diesel", _fuel_diesel_source(seed=11), ("path_factor_range",)),
        ("milk", _dairy_emulsion_source(seed=13), ("path_factor_range",)),
        ("beer", _beer_source(seed=17), ("path_factor_range",)),
        (
            "corn",
            _corn_source(seed=21),
            (
                "path_factor_range",
                "smoothing_fwhm_nm",
                "smoothing_kernel_size",
                "smoothing_sigma_bins",
            ),
        ),
        ("juice", _juice_source(seed=23), ("path_factor_range",)),
    )
    for label, source, extra_keys in cases:
        record = canonicalize_prior_config(source)
        r2f_run = build_synthetic_dataset_run(
            record,
            n_samples=32,
            random_seed=4242,
            remediation_profile="r2f_sentinel_matrix_v1",
        )
        r2g_run = build_synthetic_dataset_run(
            record,
            n_samples=32,
            random_seed=4242,
            remediation_profile="r2g_sentinel_matrix_v1",
        )
        r2f_params = r2f_run.metadata["r2c_mechanistic_remediation"][
            "transform_params"
        ]
        r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
        r2g_params = r2g_audit["transform_params"]
        assert r2g_audit["profile"] == "r2g_sentinel_matrix_v1"
        assert r2g_audit["scope"] == "bench_only_r2g_sentinel_matrix_remediation"
        for key in (
            "composition_rule",
            "spectra_rule",
            "composition_source",
            "spectra_source",
            "alphas",
            *extra_keys,
        ):
            assert r2f_params[key] == r2g_params[key], (
                f"r2g must reuse r2f {key!r} for {label}"
            )


def test_r2g_environmental_soil_applies_concentrations_smoothing_and_scatter() -> None:
    record = canonicalize_prior_config(_soil_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2g_sentinel_matrix_v1"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_mineral_organic_topsoil_centered"
    assert params["spectra_rule"] == "diffuse_powder_smoothing_and_scatter_compression"
    assert params["composition_source"] == "mechanistic_mineral_organic_topsoil_composition"
    assert (
        params["spectra_source"]
        == "diffuse_reflectance_powder_path_scatter_compression"
    )
    assert params["smoothing_fwhm_nm"] == 24.0
    assert params["smoothing_kernel_size"] >= 3
    low, high = params["path_factor_range"]
    assert (low, high) == (0.55, 0.75)
    assert params["alpha_sum"] > params["n_components"]
    _assert_audit_non_oracle_r2g(audit)


def test_r2g_environmental_soil_uses_only_valid_domain_components() -> None:
    record = canonicalize_prior_config(_soil_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )

    params = remediated.metadata["r2c_mechanistic_remediation"]["transform_params"]
    valid_components = set(get_domain_config("environmental_soil").typical_components)
    assert set(params["alphas"]) == {
        "moisture",
        "carbonates",
        "kaolinite",
        "gypsum",
        "cellulose",
        "lignin",
        "protein",
    }
    assert set(params["alphas"]).issubset(valid_components)


def test_r2g_soil_remediation_reduces_derivative_std_vs_baseline() -> None:
    record = canonicalize_prior_config(_soil_source(seed=21))
    baseline = build_synthetic_dataset_run(record, n_samples=48, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    base_dstd = float(np.median(np.diff(baseline.X, axis=1).std(axis=1, ddof=0)))
    rem_dstd = float(np.median(np.diff(remediated.X, axis=1).std(axis=1, ddof=0)))
    assert rem_dstd < base_dstd, (
        f"r2g soil smoothing must reduce derivative std; got {rem_dstd:.6f} "
        f"vs baseline {base_dstd:.6f}"
    )


def test_r2g_remediation_audit_does_not_modify_thresholds_or_metrics() -> None:
    record = canonicalize_prior_config(_soil_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=2024,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    audit = remediated.metadata["r2c_mechanistic_remediation"]
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    params = audit["transform_params"]
    assert not (set(params) & forbidden_keys)


# ---------------------------------------------------------------------------
# R2h sentinel matrix remediation profile (opt-in BERRY/juice readout).
# ---------------------------------------------------------------------------


def test_r2h_profile_is_opt_in_listed_and_accepted() -> None:
    assert R2H_REMEDIATION_PROFILES == ("r2h_sentinel_matrix_v1",)
    assert "r2h_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2g_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_juice_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata

    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2h_sentinel_matrix_v1",
    )
    assert (
        remediated.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r2h_sentinel_matrix_v1"
    )


def test_r2h_reuses_r2g_constants_for_non_juice_domains() -> None:
    for source in (_beer_source(seed=17), _soil_source(seed=19)):
        record = canonicalize_prior_config(source)
        r2g_run = build_synthetic_dataset_run(
            record,
            n_samples=24,
            random_seed=4242,
            remediation_profile="r2g_sentinel_matrix_v1",
        )
        r2h_run = build_synthetic_dataset_run(
            record,
            n_samples=24,
            random_seed=4242,
            remediation_profile="r2h_sentinel_matrix_v1",
        )
        r2g_params = r2g_run.metadata["r2c_mechanistic_remediation"][
            "transform_params"
        ]
        r2h_audit = r2h_run.metadata["r2c_mechanistic_remediation"]
        r2h_params = r2h_audit["transform_params"]
        assert r2h_audit["profile"] == "r2h_sentinel_matrix_v1"
        assert r2h_audit["scope"] == "bench_only_r2h_sentinel_matrix_remediation"
        for key in (
            "composition_rule",
            "spectra_rule",
            "composition_source",
            "spectra_source",
            "alphas",
        ):
            assert r2g_params[key] == r2h_params[key]


def test_r2h_beverage_juice_applies_percent_transmittance_readout() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=48, random_seed=4242)
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2h_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_juice"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_cloudy_berry_juice_centered"
    assert params["spectra_rule"] == "cloudy_berry_percent_transmittance_readout"
    assert (
        params["spectra_source"]
        == "beer_lambert_percent_transmittance_with_turbidity"
    )
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "apparent_percent_transmittance_intensity"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["absorbance_path_factor_range"] == [4.25, 4.75]
    assert params["detector_baseline_percent"] == 30.0
    assert params["detector_dynamic_percent"] == 20.0
    assert params["turbidity_offset_percent_range"] == [-20.0, 20.0]
    assert params["output_clip_percent"] == [0.0, 100.0]
    assert float(np.min(remediated.X)) >= 0.0
    assert float(np.max(remediated.X)) <= 100.0
    assert float(np.mean(remediated.X)) > float(np.mean(baseline.X)) * 10.0
    _assert_audit_non_oracle_r2h(audit)


def test_r2h_beverage_juice_uses_only_valid_domain_components() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    params = remediated.metadata["r2c_mechanistic_remediation"]["transform_params"]
    valid_components = set(get_domain_config("beverage_juice").typical_components)
    assert set(params["alphas"]) == {
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "citric_acid",
        "malic_acid",
        "carotenoid",
    }
    assert set(params["alphas"]).issubset(valid_components)


def test_r2h_remediation_audit_does_not_modify_thresholds_or_metrics() -> None:
    record = canonicalize_prior_config(_juice_source(seed=13))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=2024,
        remediation_profile="r2h_sentinel_matrix_v1",
    )
    audit = remediated.metadata["r2c_mechanistic_remediation"]
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    params = audit["transform_params"]
    assert not (set(params) & forbidden_keys)


# ---------------------------------------------------------------------------
# R2i sentinel matrix remediation profile (FruitPuree paste/transflectance).
# ---------------------------------------------------------------------------


def test_r2i_profile_is_opt_in_listed_and_accepted() -> None:
    assert R2I_REMEDIATION_PROFILES == ("r2i_sentinel_matrix_v1",)
    assert "r2i_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2h_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata

    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    assert (
        remediated.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r2i_sentinel_matrix_v1"
    )


def test_r2i_fruit_puree_applies_paste_transflectance_rule() -> None:
    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2i_sentinel_matrix_v1"
    assert audit["domain_key"] == "agriculture_fruit"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_semi_solid_fruit_puree_centered"
    )
    assert params["spectra_rule"] == (
        "semi_solid_fruit_puree_short_path_scatter_smoothing"
    )
    assert params["spectra_rule"] != "cloudy_berry_percent_transmittance_readout"
    assert params["composition_source"] == "textbook_fruit_puree_tissue_composition"
    assert (
        params["spectra_source"]
        == "semi_solid_paste_transflectance_scatter_compression"
    )
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "semi_solid_puree_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["path_factor_range"] == [0.12, 0.22]
    assert params["smoothing_fwhm_nm"] == 16.0
    assert params["additive_baseline_range"] == [0.002, 0.008]
    assert params["alpha_sum"] > params["n_components"]
    _assert_audit_non_oracle_r2i(audit)


def test_r2i_fruit_puree_uses_only_valid_fruit_domain_components() -> None:
    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )

    params = remediated.metadata["r2c_mechanistic_remediation"]["transform_params"]
    valid_components = set(get_domain_config("agriculture_fruit").typical_components)
    assert set(params["alphas"]) == {
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "cellulose",
        "starch",
        "malic_acid",
        "citric_acid",
        "carotenoid",
    }
    assert set(params["alphas"]).issubset(valid_components)


def test_r2i_keeps_beverage_juice_on_r2f_clear_juice_rule() -> None:
    record = canonicalize_prior_config(_juice_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    params = audit["transform_params"]
    assert audit["profile"] == "r2i_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_juice"
    assert params["spectra_rule"] == "moderate_liquid_cuvette_path_scale"
    assert params["composition_rule"] == "tight_dirichlet_fruit_juice_centered"
    assert params["readout_space"] == "uncalibrated_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["spectra_rule"] != "cloudy_berry_percent_transmittance_readout"


def test_r2h_profile_does_not_remediate_fruit_puree_domain() -> None:
    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=4242)
    r2h_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    audit = r2h_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2h_sentinel_matrix_v1"
    assert audit["domain_key"] == "agriculture_fruit"
    assert audit["applied_to_concentrations"] is False
    assert audit["applied_to_spectra"] is False
    assert audit["transform_params"] == {}
    np.testing.assert_allclose(r2h_run.X, baseline.X)
    np.testing.assert_allclose(r2h_run.y, baseline.y)


# ---------------------------------------------------------------------------
# R2j sentinel matrix remediation profile (DIESEL micro-path raw readout).
# ---------------------------------------------------------------------------


def test_r2j_profile_is_opt_in_listed_and_accepted() -> None:
    assert R2J_REMEDIATION_PROFILES == ("r2j_sentinel_matrix_v1",)
    assert "r2j_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2i_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    assert "r2c_mechanistic_remediation" not in baseline.metadata

    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2j_sentinel_matrix_v1",
    )
    assert (
        remediated.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r2j_sentinel_matrix_v1"
    )


def test_r2j_fuel_applies_micro_path_absorbance_floor_rule() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2j_run = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2j_sentinel_matrix_v1",
    )

    r2i_params = r2i_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r2j_audit = r2j_run.metadata["r2c_mechanistic_remediation"]
    params = r2j_audit["transform_params"]
    assert r2j_audit["enabled"] is True
    assert r2j_audit["profile"] == "r2j_sentinel_matrix_v1"
    assert r2j_audit["scope"] == "bench_only_r2j_sentinel_matrix_remediation"
    assert r2j_audit["domain_key"] == "petrochem_fuels"
    assert r2j_audit["applied_to_concentrations"] is True
    assert r2j_audit["applied_to_spectra"] is True
    assert r2i_params["spectra_rule"] == "short_liquid_optical_path_scale"
    assert r2i_params["path_factor_range"] == [0.45, 0.65]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_transmission_absorbance_floor"
    assert params["composition_source"] == "textbook_diesel_composition"
    assert params["spectra_source"] == "beer_lambert_micro_path_with_detector_floor"
    assert params["path_factor_range"] == [0.03, 0.05]
    assert params["additive_baseline_range"] == [0.0005, 0.002]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "micro_path_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert float(r2j_run.X.mean()) < float(r2i_run.X.mean()) * 0.25
    _assert_audit_non_oracle_r2j(r2j_audit)


def test_r2j_inherits_r2i_fruit_puree_rule_unchanged() -> None:
    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2j_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2j_sentinel_matrix_v1",
    )

    r2i_audit = r2i_run.metadata["r2c_mechanistic_remediation"]
    r2j_audit = r2j_run.metadata["r2c_mechanistic_remediation"]
    assert r2j_audit["profile"] == "r2i_sentinel_matrix_v1"
    assert r2j_audit["scope"] == "bench_only_r2i_sentinel_matrix_remediation"
    r2i_params = dict(r2i_audit["transform_params"])
    r2j_params = dict(r2j_audit["transform_params"])
    for params in (r2i_params, r2j_params):
        params.pop("path_factor_min", None)
        params.pop("path_factor_max", None)
        params.pop("additive_baseline_min", None)
        params.pop("additive_baseline_max", None)
    assert r2j_params == r2i_params
    np.testing.assert_allclose(r2j_run.X, r2i_run.X)
    np.testing.assert_allclose(r2j_run.y, r2i_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _beer_source(seed=17),
        _corn_source(seed=18),
        _juice_source(seed=19),
        _soil_source(seed=20),
    ),
)
def test_r2j_reuses_r2i_rules_for_non_fuel_domains(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2j_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2j_sentinel_matrix_v1",
    )

    r2i_audit = r2i_run.metadata["r2c_mechanistic_remediation"]
    r2j_audit = r2j_run.metadata["r2c_mechanistic_remediation"]
    assert r2i_audit["domain_key"] != "petrochem_fuels"
    assert r2j_audit["profile"] == "r2i_sentinel_matrix_v1"
    assert r2j_audit["scope"] == "bench_only_r2i_sentinel_matrix_remediation"
    assert r2j_audit["domain_key"] == r2i_audit["domain_key"]

    r2i_params = dict(r2i_audit["transform_params"])
    r2j_params = dict(r2j_audit["transform_params"])
    for params in (r2i_params, r2j_params):
        params.pop("path_factor_min", None)
        params.pop("path_factor_max", None)
        params.pop("additive_baseline_min", None)
        params.pop("additive_baseline_max", None)
        params.pop("absorbance_path_factor_min", None)
        params.pop("absorbance_path_factor_max", None)
        params.pop("turbidity_offset_percent_min", None)
        params.pop("turbidity_offset_percent_max", None)
    assert r2j_params == r2i_params
    np.testing.assert_allclose(r2j_run.X, r2i_run.X)
    np.testing.assert_allclose(r2j_run.y, r2i_run.y)


# ---------------------------------------------------------------------------
# R2k sentinel matrix remediation profile (DIESEL CH overtone contrast).
# ---------------------------------------------------------------------------


def test_r2k_profile_is_opt_in_listed_and_accepted() -> None:
    assert R2K_REMEDIATION_PROFILES == ("r2k_sentinel_matrix_v1",)
    assert "r2k_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2i_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2k_sentinel_matrix_v1",
    )
    assert (
        remediated.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r2k_sentinel_matrix_v1"
    )


def test_r2k_fuel_preserves_more_derivative_structure_than_r2j() -> None:
    record = canonicalize_prior_config(_fuel_diesel_source(seed=11))
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2j_run = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2j_sentinel_matrix_v1",
    )
    r2k_run = build_synthetic_dataset_run(
        record,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2k_sentinel_matrix_v1",
    )

    r2k_audit = r2k_run.metadata["r2c_mechanistic_remediation"]
    params = r2k_audit["transform_params"]
    assert r2k_audit["enabled"] is True
    assert r2k_audit["profile"] == "r2k_sentinel_matrix_v1"
    assert r2k_audit["scope"] == "bench_only_r2k_sentinel_matrix_remediation"
    assert r2k_audit["domain_key"] == "petrochem_fuels"
    assert r2k_audit["applied_to_concentrations"] is True
    assert r2k_audit["applied_to_spectra"] is True
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert (
        params["spectra_source"]
        == "beer_lambert_micro_path_with_fixed_ch_overtone_contrast"
    )
    assert params["path_factor_range"] == [0.055, 0.085]
    assert params["feature_contrast_range"] == [0.24, 0.34]
    assert params["continuum_smoothing_fwhm_nm"] == 96.0
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0, 1720.0]
    assert params["ch_overtone_width_nm"] == 34.0
    assert params["ch_overtone_gain_range"] == [0.10, 0.18]
    assert params["additive_baseline_range"] == [0.0005, 0.002]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "micro_path_ch_overtone_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["contrast_source"] == "fixed_hydrocarbon_ch_overtone_prior"

    r2j_deriv = float(np.diff(r2j_run.X, axis=1).std())
    r2k_deriv = float(np.diff(r2k_run.X, axis=1).std())
    r2i_mean = float(r2i_run.X.mean())
    assert r2k_deriv > r2j_deriv * 2.5
    assert float(r2k_run.X.mean()) < r2i_mean * 0.5
    assert float(r2k_run.X.mean()) > float(r2j_run.X.mean())
    assert not np.allclose(r2k_run.X, r2j_run.X)
    _assert_audit_non_oracle_r2k(r2k_audit)


def test_r2k_reuses_r2i_rules_for_non_fuel_domains() -> None:
    record = canonicalize_prior_config(_fruit_puree_source(seed=11))
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2k_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2k_sentinel_matrix_v1",
    )

    r2i_audit = r2i_run.metadata["r2c_mechanistic_remediation"]
    r2k_audit = r2k_run.metadata["r2c_mechanistic_remediation"]
    assert r2k_audit["profile"] == "r2i_sentinel_matrix_v1"
    assert r2k_audit["scope"] == "bench_only_r2i_sentinel_matrix_remediation"
    assert r2k_audit["domain_key"] == r2i_audit["domain_key"]
    np.testing.assert_allclose(r2k_run.X, r2i_run.X)
    np.testing.assert_allclose(r2k_run.y, r2i_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _beer_source(seed=17),
        _corn_source(seed=18),
        _juice_source(seed=19),
        _soil_source(seed=20),
        _fruit_puree_source(seed=21),
    ),
)
def test_r2k_non_fuel_draws_are_identical_to_r2i(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2i_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    r2k_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2k_sentinel_matrix_v1",
    )

    r2i_audit = r2i_run.metadata["r2c_mechanistic_remediation"]
    r2k_audit = r2k_run.metadata["r2c_mechanistic_remediation"]
    assert r2i_audit["domain_key"] != "petrochem_fuels"
    assert r2k_audit["profile"] == "r2i_sentinel_matrix_v1"
    assert r2k_audit == r2i_audit
    np.testing.assert_allclose(r2k_run.X, r2i_run.X)
    np.testing.assert_allclose(r2k_run.y, r2i_run.y)


# ---------------------------------------------------------------------------
# R2l sentinel matrix remediation profile (LUCAS-only soil readout).
# ---------------------------------------------------------------------------


def _lucas_soil_source(*, seed: int) -> dict[str, object]:
    source = _soil_source(seed=seed)
    source["_r2l_lucas_soil_route"] = {
        "enabled": True,
        "route_marker": "lucas",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2m_milk_source(*, seed: int, variant: str = "shortwave") -> dict[str, object]:
    source = _dairy_emulsion_source(seed=seed)
    source["_r2m_milk_readout_route"] = {
        "enabled": True,
        "route_marker": "milk",
        "variant": variant,
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2n_manure21_source(*, seed: int) -> dict[str, object]:
    source = _soil_source(seed=seed)
    source["particle_size"] = 90.0
    source["_r2n_manure21_readout_route"] = {
        "enabled": True,
        "route_marker": "manure21",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2o_beer_source(*, seed: int) -> dict[str, object]:
    source = _beer_source(seed=seed)
    source["_r2o_beer_readout_route"] = {
        "enabled": True,
        "route_marker": "beer",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2p_phosphorus_source(*, seed: int) -> dict[str, object]:
    source = _soil_source(seed=seed)
    source["_r2p_phosphorus_readout_route"] = {
        "enabled": True,
        "route_marker": "phosphorus",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2q_lucas_ph_organic_source(*, seed: int) -> dict[str, object]:
    source = _soil_source(seed=seed)
    source["_r2q_lucas_ph_organic_readout_route"] = {
        "enabled": True,
        "route_marker": "lucas_ph_organic",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2r_fruit_puree_source(*, seed: int) -> dict[str, object]:
    source = _fruit_puree_source(seed=seed)
    source["_r2r_fruitpuree_readout_route"] = {
        "enabled": True,
        "route_marker": "fruitpuree",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r2s_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r2s_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3c_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3c_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3d_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3e_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3e_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3f_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3f_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r3g_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3g_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r4a_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r4a_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r4b_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r4b_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r4c_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r4c_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r2l_profile_is_opt_in_listed_and_accepted_for_lucas_marked_soil() -> None:
    assert R2L_REMEDIATION_PROFILES == ("r2l_sentinel_matrix_v1",)
    assert "r2l_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2k_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_lucas_soil_source(seed=11))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2l_sentinel_matrix_v1"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "lucas_mineral_albedo_absorbance_floor_scatter_readout"
    )
    assert (
        params["spectra_source"]
        == "fixed_mineral_albedo_floor_plus_diffuse_scatter_residual"
    )
    assert params["path_factor_range"] == [0.2, 0.25]
    assert params["additive_baseline_range"] == [0.30103, 0.30103]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "lucas_raw_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["baseline_source"] == "mineral_albedo_A_equals_minus_log10_0p5"
    _assert_audit_non_oracle_r2l(audit)


def test_r2l_unmarked_soil_is_routed_back_to_r2g() -> None:
    record = canonicalize_prior_config(_soil_source(seed=20))
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2l_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2l_audit = r2l_run.metadata["r2c_mechanistic_remediation"]
    assert r2l_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2l_audit["scope"] == "bench_only_r2g_sentinel_matrix_remediation"
    assert r2l_audit == r2g_audit
    np.testing.assert_allclose(r2l_run.X, r2g_run.X)
    np.testing.assert_allclose(r2l_run.y, r2g_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _fuel_diesel_source(seed=17),
        _beer_source(seed=18),
        _corn_source(seed=19),
        _juice_source(seed=20),
        _fruit_puree_source(seed=21),
    ),
)
def test_r2l_non_soil_draws_are_identical_to_r2k(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2k_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2k_sentinel_matrix_v1",
    )
    r2l_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    r2k_audit = r2k_run.metadata["r2c_mechanistic_remediation"]
    r2l_audit = r2l_run.metadata["r2c_mechanistic_remediation"]
    assert r2l_audit["domain_key"] != "environmental_soil"
    assert r2l_audit["profile"] == r2k_audit["profile"]
    assert r2l_audit == r2k_audit
    np.testing.assert_allclose(r2l_run.X, r2k_run.X)
    np.testing.assert_allclose(r2l_run.y, r2k_run.y)


def test_r2m_profile_is_opt_in_listed_and_accepted_for_milk_marked_dairy() -> None:
    assert R2M_REMEDIATION_PROFILES == ("r2m_sentinel_matrix_v1",)
    assert "r2m_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2l_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_r2m_milk_source(seed=22))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2m_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2m_sentinel_matrix_remediation"
    assert audit["domain_key"] == "food_dairy"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "milk_emulsion_scatter_inverse_transflectance_readout"
    )
    assert (
        params["spectra_source"]
        == "fat_globule_scatter_inverse_beer_lambert_transflectance"
    )
    assert params["path_factor_range"] == [0.55, 0.85]
    assert params["milk_readout_variant"] == "shortwave"
    assert params["detector_dynamic_range"] == [1.8, 2.6]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "milk_raw_transflectance_intensity"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["scatter_source"] == "fixed_fat_globule_mie_scatter_prior"
    assert params["provenance_source"] == "exp09_dataset_token_milk_route"
    assert params["milk_readout_route_source"] == "exp09_dataset_token"
    assert params["milk_readout_route_marker"] == "milk"
    assert params["milk_readout_route_non_oracle"] is True
    assert params["milk_readout_route_real_stat_capture"] is False
    assert params["milk_readout_route_thresholds_modified"] is False
    _assert_audit_non_oracle_r2m(audit)


def test_r2m_labels_route_uses_fullrange_milk_detector_prior() -> None:
    record = canonicalize_prior_config(_r2m_milk_source(seed=23, variant="fullrange"))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    params = remediated.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert params["milk_readout_variant"] == "fullrange"
    assert params["detector_dynamic_range"] == [1.0, 1.8]
    assert params["output_clip_intensity"] == [0.0, 3.0]


def test_r2m_unmarked_dairy_is_routed_back_to_r2l() -> None:
    record = canonicalize_prior_config(_dairy_emulsion_source(seed=24))
    r2l_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )
    r2m_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    r2l_audit = r2l_run.metadata["r2c_mechanistic_remediation"]
    r2m_audit = r2m_run.metadata["r2c_mechanistic_remediation"]
    assert r2m_audit["profile"] == r2l_audit["profile"]
    assert r2m_audit == r2l_audit
    np.testing.assert_allclose(r2m_run.X, r2l_run.X)
    np.testing.assert_allclose(r2m_run.y, r2l_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _fuel_diesel_source(seed=25),
        _beer_source(seed=26),
        _corn_source(seed=27),
        _juice_source(seed=28),
        _fruit_puree_source(seed=29),
        _lucas_soil_source(seed=30),
        _soil_source(seed=31),
    ),
)
def test_r2m_non_dairy_draws_are_identical_to_r2l(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2l_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )
    r2m_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    r2l_audit = r2l_run.metadata["r2c_mechanistic_remediation"]
    r2m_audit = r2m_run.metadata["r2c_mechanistic_remediation"]
    assert r2m_audit["domain_key"] != "food_dairy"
    assert r2m_audit["profile"] == r2l_audit["profile"]
    assert r2m_audit == r2l_audit
    np.testing.assert_allclose(r2m_run.X, r2l_run.X)
    np.testing.assert_allclose(r2m_run.y, r2l_run.y)


# ---------------------------------------------------------------------------
# R2n sentinel matrix remediation profile (MANURE21-only manure readout).
# ---------------------------------------------------------------------------


def test_r2n_profile_is_opt_in_listed_and_accepted_for_manure21_marked_soil() -> None:
    assert R2N_REMEDIATION_PROFILES == ("r2n_sentinel_matrix_v1",)
    assert "r2n_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2m_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_r2n_manure21_source(seed=32))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2n_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2n_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_dried_manure_organic_mineral_centered"
    )
    assert params["spectra_rule"] == (
        "dried_manure_organic_mineral_albedo_scatter_readout"
    )
    assert params["composition_source"] == (
        "textbook_dried_manure_organic_mineral_composition"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_diffuse_scatter_residual"
    )
    assert params["path_factor_range"] == [0.3, 0.42]
    assert params["additive_baseline_range"] == [0.6, 0.78]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert params["manure21_readout_route_source"] == "exp09_dataset_token"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    _assert_audit_non_oracle_r2n(audit)


def test_r2n_unmarked_manure_like_soil_is_routed_back_to_r2g() -> None:
    source = _soil_source(seed=33)
    source["product_key"] = "manure21"
    record = canonicalize_prior_config(source)
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    assert r2n_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2n_audit["scope"] == "bench_only_r2g_sentinel_matrix_remediation"
    assert r2n_audit == r2g_audit
    np.testing.assert_allclose(r2n_run.X, r2g_run.X)
    np.testing.assert_allclose(r2n_run.y, r2g_run.y)


def test_r2n_non_compliant_manure21_route_is_ignored() -> None:
    source = _r2n_manure21_source(seed=34)
    route = dict(cast("dict[str, object]", source["_r2n_manure21_readout_route"]))
    route["thresholds_modified"] = True
    source["_r2n_manure21_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    assert r2n_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2n_audit == r2g_audit
    np.testing.assert_allclose(r2n_run.X, r2g_run.X)
    np.testing.assert_allclose(r2n_run.y, r2g_run.y)


def test_r2n_marker_on_agriculture_grain_does_not_trigger_manure_readout() -> None:
    source = _corn_source(seed=35)
    source["_r2n_manure21_readout_route"] = {
        "enabled": True,
        "route_marker": "manure21",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    record = canonicalize_prior_config(source)
    r2m_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2m_sentinel_matrix_v1",
    )
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    r2m_audit = r2m_run.metadata["r2c_mechanistic_remediation"]
    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    assert r2n_audit["profile"] != "r2n_sentinel_matrix_v1"
    assert r2n_audit["transform_params"]["spectra_rule"] != (
        "dried_manure_organic_mineral_albedo_scatter_readout"
    )
    assert r2n_audit == r2m_audit
    np.testing.assert_allclose(r2n_run.X, r2m_run.X)
    np.testing.assert_allclose(r2n_run.y, r2m_run.y)


# ---------------------------------------------------------------------------
# R2o sentinel matrix remediation profile (BEER-only fermented-liquid readout).
# ---------------------------------------------------------------------------


def test_r2o_profile_is_opt_in_listed_and_accepted_for_beer_marked_wine() -> None:
    assert R2O_REMEDIATION_PROFILES == ("r2o_sentinel_matrix_v1",)
    assert "r2o_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2n_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_r2o_beer_source(seed=36))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2o_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2o_sentinel_matrix_remediation"
    assert audit["domain_key"] == "beverage_wine"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_beer_centered"
    assert params["spectra_rule"] == "fermented_beer_turbid_cuvette_absorbance_readout"
    assert params["composition_source"] == "textbook_beer_composition"
    assert (
        params["spectra_source"]
        == "beer_lambert_long_path_with_fixed_haze_carbonation"
    )
    assert params["path_factor_range"] == [1.75, 2.35]
    assert params["haze_absorbance_baseline_range"] == [1.75, 2.15]
    assert params["haze_slope_absorbance_range"] == [0.06, 0.18]
    assert params["carbonation_residual_absorbance_range"] == [0.0, 0.05]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "fermented_beer_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_beer_route"
    assert params["beer_readout_route_source"] == "exp09_dataset_token"
    assert params["beer_readout_route_marker"] == "beer"
    assert params["beer_readout_route_non_oracle"] is True
    assert params["beer_readout_route_real_stat_capture"] is False
    assert params["beer_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False


def test_r2o_unmarked_beer_domain_is_routed_back_to_r2n_inheritance() -> None:
    record = canonicalize_prior_config(_beer_source(seed=37))
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    assert r2o_audit["profile"] == r2n_audit["profile"]
    assert r2o_audit["transform_params"]["spectra_rule"] != (
        "fermented_beer_turbid_cuvette_absorbance_readout"
    )
    assert r2o_audit == r2n_audit
    np.testing.assert_allclose(r2o_run.X, r2n_run.X)
    np.testing.assert_allclose(r2o_run.y, r2n_run.y)


def test_r2o_non_compliant_beer_route_is_ignored() -> None:
    source = _r2o_beer_source(seed=38)
    route = dict(cast("dict[str, object]", source["_r2o_beer_readout_route"]))
    route["real_stat_capture"] = True
    source["_r2o_beer_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    assert r2o_audit["profile"] == r2n_audit["profile"]
    assert r2o_audit["transform_params"]["spectra_rule"] != (
        "fermented_beer_turbid_cuvette_absorbance_readout"
    )
    assert r2o_audit == r2n_audit
    np.testing.assert_allclose(r2o_run.X, r2n_run.X)
    np.testing.assert_allclose(r2o_run.y, r2n_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _r2n_manure21_source(seed=39),
        _r2m_milk_source(seed=40),
        _lucas_soil_source(seed=41),
        _soil_source(seed=42),
        _fuel_diesel_source(seed=43),
        _juice_source(seed=44),
        _fruit_puree_source(seed=45),
    ),
)
def test_r2o_non_beer_draws_are_identical_to_r2n(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2n_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    r2n_audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    assert r2o_audit["profile"] == r2n_audit["profile"]
    assert r2o_audit["transform_params"].get("spectra_rule") != (
        "fermented_beer_turbid_cuvette_absorbance_readout"
    )
    assert r2o_audit == r2n_audit
    np.testing.assert_allclose(r2o_run.X, r2n_run.X)
    np.testing.assert_allclose(r2o_run.y, r2n_run.y)


# ---------------------------------------------------------------------------
# R2p sentinel matrix remediation profile (PHOSPHORUS-only mineral readout).
# ---------------------------------------------------------------------------


def test_r2p_profile_is_opt_in_listed_and_accepted_for_phosphorus_marked_soil() -> None:
    assert R2P_REMEDIATION_PROFILES == ("r2p_sentinel_matrix_v1",)
    assert "r2p_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2o_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_r2p_phosphorus_source(seed=46))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2p_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2p_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_mineral_organic_topsoil_centered"
    )
    assert params["spectra_rule"] == (
        "phosphorus_mineral_fertilizer_albedo_residual_readout"
    )
    assert params["composition_source"] == (
        "mechanistic_mineral_organic_topsoil_composition"
    )
    assert params["spectra_source"] == (
        "fixed_phosphate_mineral_albedo_plus_inverted_residual"
    )
    assert params["path_factor_range"] == [0.95, 1.05]
    assert params["additive_baseline_range"] == [0.195, 0.21]
    assert params["centered_residual_readout"] == (
        "inverted_about_phosphate_mineral_albedo"
    )
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "phosphorus_raw_mineral_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_phosphorus_route"
    assert params["phosphorus_readout_route_source"] == "exp09_dataset_token"
    assert params["phosphorus_readout_route_marker"] == "phosphorus"
    assert params["phosphorus_readout_route_non_oracle"] is True
    assert params["phosphorus_readout_route_real_stat_capture"] is False
    assert params["phosphorus_readout_route_thresholds_modified"] is False
    _assert_audit_non_oracle_r2p(audit)


def test_r2p_phosphorus_readout_is_distinct_from_r2g_and_r2l_lucas() -> None:
    phosphorus_record = canonicalize_prior_config(_r2p_phosphorus_source(seed=47))
    lucas_record = canonicalize_prior_config(_lucas_soil_source(seed=47))

    r2g_run = build_synthetic_dataset_run(
        phosphorus_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2p_run = build_synthetic_dataset_run(
        phosphorus_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2l_run = build_synthetic_dataset_run(
        lucas_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    r2p_params = r2p_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r2l_params = r2l_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert r2p_params["spectra_rule"] != r2g_run.metadata[
        "r2c_mechanistic_remediation"
    ]["transform_params"]["spectra_rule"]
    assert r2p_params["spectra_rule"] != r2l_params["spectra_rule"]
    assert r2p_params["readout_space"] != r2l_params["readout_space"]
    assert r2p_params["additive_baseline_range"] != r2l_params[
        "additive_baseline_range"
    ]
    assert not np.allclose(r2p_run.X, r2g_run.X)
    assert not np.allclose(r2p_run.X, r2l_run.X)


def test_r2p_unmarked_soil_is_routed_back_to_r2o_inheritance() -> None:
    record = canonicalize_prior_config(_soil_source(seed=48))
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    r2p_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    r2p_audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    assert r2p_audit["profile"] == r2o_audit["profile"]
    assert r2p_audit["transform_params"].get("spectra_rule") != (
        "phosphorus_mineral_fertilizer_albedo_residual_readout"
    )
    assert r2p_audit == r2o_audit
    np.testing.assert_allclose(r2p_run.X, r2o_run.X)
    np.testing.assert_allclose(r2p_run.y, r2o_run.y)


def test_r2p_non_compliant_phosphorus_route_is_ignored() -> None:
    source = _r2p_phosphorus_source(seed=49)
    route = dict(cast("dict[str, object]", source["_r2p_phosphorus_readout_route"]))
    route["thresholds_modified"] = True
    source["_r2p_phosphorus_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    r2p_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    r2p_audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    assert r2p_audit["profile"] == r2o_audit["profile"]
    assert r2p_audit["transform_params"].get("spectra_rule") != (
        "phosphorus_mineral_fertilizer_albedo_residual_readout"
    )
    assert r2p_audit == r2o_audit
    np.testing.assert_allclose(r2p_run.X, r2o_run.X)
    np.testing.assert_allclose(r2p_run.y, r2o_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _r2o_beer_source(seed=50),
        _r2n_manure21_source(seed=51),
        _r2m_milk_source(seed=52),
        _lucas_soil_source(seed=53),
        _fuel_diesel_source(seed=54),
        _juice_source(seed=55),
        _fruit_puree_source(seed=56),
    ),
)
def test_r2p_non_phosphorus_draws_are_identical_to_r2o(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2o_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    r2p_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    r2o_audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    r2p_audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    assert r2p_audit["profile"] == r2o_audit["profile"]
    assert r2p_audit["transform_params"].get("spectra_rule") != (
        "phosphorus_mineral_fertilizer_albedo_residual_readout"
    )
    assert r2p_audit == r2o_audit
    np.testing.assert_allclose(r2p_run.X, r2o_run.X)
    np.testing.assert_allclose(r2p_run.y, r2o_run.y)


# ---------------------------------------------------------------------------
# R2q sentinel matrix remediation profile (LUCAS pH Organic-only humic readout).
# ---------------------------------------------------------------------------


def test_r2q_profile_is_opt_in_listed_and_accepted_for_lucas_ph_organic_soil() -> None:
    assert R2Q_REMEDIATION_PROFILES == ("r2q_sentinel_matrix_v1",)
    assert "r2q_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2p_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    record = canonicalize_prior_config(_r2q_lucas_ph_organic_source(seed=57))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2q_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2q_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_lucas_ph_organic_topsoil_centered"
    )
    assert params["spectra_rule"] == "lucas_ph_organic_humic_albedo_oh_readout"
    assert params["composition_source"] == (
        "fixed_lucas_ph_organic_humic_topsoil_composition"
    )
    assert params["spectra_source"] == (
        "fixed_humic_dark_albedo_plus_oh_residual_readout"
    )
    assert params["path_factor_range"] == [0.22, 0.32]
    assert params["additive_baseline_range"] == [0.405, 0.455]
    assert params["humic_slope_absorbance_range"] == [0.015, 0.045]
    assert params["oh_band_absorbance_range"] == [0.005, 0.025]
    assert params["oh_band_centers_nm"] == [1450.0, 1940.0]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "lucas_ph_organic_raw_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_lucas_ph_organic_route"
    assert params["matrix_source"] == "fixed_lucas_humic_organic_topsoil_prior"
    assert params["scatter_source"] == "fixed_organic_topsoil_diffuse_scatter_prior"
    assert params["lucas_ph_organic_readout_route_source"] == "exp09_dataset_token"
    assert params["lucas_ph_organic_readout_route_marker"] == "lucas_ph_organic"
    assert params["lucas_ph_organic_readout_route_non_oracle"] is True
    assert params["lucas_ph_organic_readout_route_real_stat_capture"] is False
    assert params["lucas_ph_organic_readout_route_thresholds_modified"] is False
    _assert_audit_non_oracle_r2q(audit)


def test_r2q_lucas_ph_organic_readout_is_distinct_from_r2p_phosphorus_and_r2l_lucas() -> None:
    organic_record = canonicalize_prior_config(_r2q_lucas_ph_organic_source(seed=58))
    phosphorus_record = canonicalize_prior_config(_r2p_phosphorus_source(seed=58))
    lucas_record = canonicalize_prior_config(_lucas_soil_source(seed=58))

    r2q_run = build_synthetic_dataset_run(
        organic_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )
    r2p_run = build_synthetic_dataset_run(
        phosphorus_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2l_run = build_synthetic_dataset_run(
        lucas_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    r2q_params = r2q_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r2p_params = r2p_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r2l_params = r2l_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert r2q_params["spectra_rule"] != r2p_params["spectra_rule"]
    assert r2q_params["spectra_rule"] != r2l_params["spectra_rule"]
    assert r2q_params["readout_space"] != r2p_params["readout_space"]
    assert r2q_params["readout_space"] != r2l_params["readout_space"]
    assert r2q_params["additive_baseline_range"] != r2p_params[
        "additive_baseline_range"
    ]
    assert r2q_params["additive_baseline_range"] != r2l_params[
        "additive_baseline_range"
    ]
    assert not np.allclose(r2q_run.X, r2p_run.X)
    assert not np.allclose(r2q_run.X, r2l_run.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2p_phosphorus_source(seed=59),
        _lucas_soil_source(seed=60),
        _r2o_beer_source(seed=61),
        _r2n_manure21_source(seed=62),
        _r2m_milk_source(seed=63),
        _fuel_diesel_source(seed=64),
        _juice_source(seed=65),
        _fruit_puree_source(seed=66),
    ),
)
def test_r2q_non_target_draws_are_identical_to_r2p(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2p_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2q_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    r2p_audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    r2q_audit = r2q_run.metadata["r2c_mechanistic_remediation"]
    assert r2q_audit["profile"] == r2p_audit["profile"]
    assert r2q_audit["transform_params"].get("spectra_rule") != (
        "lucas_ph_organic_humic_albedo_oh_readout"
    )
    assert r2q_audit == r2p_audit
    np.testing.assert_allclose(r2q_run.X, r2p_run.X)
    np.testing.assert_allclose(r2q_run.y, r2p_run.y)


def test_r2q_non_compliant_lucas_ph_organic_route_is_ignored() -> None:
    source = _r2q_lucas_ph_organic_source(seed=67)
    route = dict(
        cast("dict[str, object]", source["_r2q_lucas_ph_organic_readout_route"])
    )
    route["real_stat_capture"] = True
    source["_r2q_lucas_ph_organic_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2p_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2q_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    r2p_audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    r2q_audit = r2q_run.metadata["r2c_mechanistic_remediation"]
    assert r2q_audit["profile"] == r2p_audit["profile"]
    assert r2q_audit["transform_params"].get("spectra_rule") != (
        "lucas_ph_organic_humic_albedo_oh_readout"
    )
    assert r2q_audit == r2p_audit
    np.testing.assert_allclose(r2q_run.X, r2p_run.X)
    np.testing.assert_allclose(r2q_run.y, r2p_run.y)


# ---------------------------------------------------------------------------
# R2r sentinel matrix remediation profile (explicit FruitPuree route only).
# ---------------------------------------------------------------------------


def test_r2r_profile_is_opt_in_listed_and_requires_fruitpuree_route() -> None:
    assert R2R_REMEDIATION_PROFILES == ("r2r_sentinel_matrix_v1",)
    assert "r2r_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2q_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    unmarked_record = canonicalize_prior_config(_fruit_puree_source(seed=68))
    marked_record = canonicalize_prior_config(_r2r_fruit_puree_source(seed=68))

    inherited = build_synthetic_dataset_run(
        unmarked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2r_sentinel_matrix_v1",
    )
    remediated = build_synthetic_dataset_run(
        marked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    inherited_audit = inherited.metadata["r2c_mechanistic_remediation"]
    assert inherited_audit["profile"] != "r2r_sentinel_matrix_v1"
    assert inherited_audit["transform_params"].get("spectra_rule") != (
        "strawberry_puree_transflectance_residual_readout"
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2r_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2r_sentinel_matrix_remediation"
    assert audit["domain_key"] == "agriculture_fruit"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_strawberry_puree_cellular_matrix"
    assert params["spectra_rule"] == "strawberry_puree_transflectance_residual_readout"
    assert params["composition_source"] == (
        "textbook_strawberry_puree_cellular_composition"
    )
    assert params["spectra_source"] == (
        "fixed_puree_transflectance_albedo_residual_readout"
    )
    assert params["path_factor_range"] == [0.045, 0.075]
    assert params["additive_baseline_range"] == [0.006, 0.009]
    assert params["scatter_slope_absorbance_range"] == [-0.0015, 0.0015]
    assert params["water_band_absorbance_range"] == [0.001, 0.0025]
    assert params["sugar_solids_band_absorbance_range"] == [0.0003, 0.0012]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "strawberry_puree_raw_transflectance_residual_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_fruitpuree_route"
    assert params["matrix_source"] == "fixed_semi_solid_strawberry_puree_prior"
    assert params["scatter_source"] == "fixed_seed_skin_pectin_cellular_scatter_prior"
    assert params["fruitpuree_readout_route_source"] == "exp09_dataset_token"
    assert params["fruitpuree_readout_route_marker"] == "fruitpuree"
    assert params["fruitpuree_readout_route_non_oracle"] is True
    assert params["fruitpuree_readout_route_real_stat_capture"] is False
    assert params["fruitpuree_readout_route_thresholds_modified"] is False
    assert params["spectra_rule"] != "cloudy_berry_percent_transmittance_readout"
    _assert_audit_non_oracle_r2r(audit)
    assert not np.allclose(remediated.X, inherited.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2q_lucas_ph_organic_source(seed=69),
        _r2p_phosphorus_source(seed=70),
        _r2o_beer_source(seed=71),
        _r2n_manure21_source(seed=72),
        _r2m_milk_source(seed=73),
        _fuel_diesel_source(seed=74),
        _juice_source(seed=75),
        _fruit_puree_source(seed=76),
    ),
)
def test_r2r_non_explicit_fruitpuree_draws_are_identical_to_r2q(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2q_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )
    r2r_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    r2q_audit = r2q_run.metadata["r2c_mechanistic_remediation"]
    r2r_audit = r2r_run.metadata["r2c_mechanistic_remediation"]
    assert r2r_audit["profile"] == r2q_audit["profile"]
    assert r2r_audit["transform_params"].get("spectra_rule") != (
        "strawberry_puree_transflectance_residual_readout"
    )
    assert r2r_audit == r2q_audit
    np.testing.assert_allclose(r2r_run.X, r2q_run.X)
    np.testing.assert_allclose(r2r_run.y, r2q_run.y)


def test_r2r_non_compliant_fruitpuree_route_is_ignored() -> None:
    source = _r2r_fruit_puree_source(seed=77)
    route = dict(cast("dict[str, object]", source["_r2r_fruitpuree_readout_route"]))
    route["real_stat_capture"] = True
    source["_r2r_fruitpuree_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2q_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2q_sentinel_matrix_v1",
    )
    r2r_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    r2q_audit = r2q_run.metadata["r2c_mechanistic_remediation"]
    r2r_audit = r2r_run.metadata["r2c_mechanistic_remediation"]
    assert r2r_audit["profile"] == r2q_audit["profile"]
    assert r2r_audit["transform_params"].get("spectra_rule") != (
        "strawberry_puree_transflectance_residual_readout"
    )
    assert r2r_audit == r2q_audit
    np.testing.assert_allclose(r2r_run.X, r2q_run.X)
    np.testing.assert_allclose(r2r_run.y, r2q_run.y)


# ---------------------------------------------------------------------------
# R2s sentinel matrix remediation profile (explicit DIESEL-only readout).
# ---------------------------------------------------------------------------


def test_r2s_profile_is_opt_in_listed_and_requires_diesel_route() -> None:
    assert R2S_REMEDIATION_PROFILES == ("r2s_sentinel_matrix_v1",)
    assert "r2s_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    unmarked_record = canonicalize_prior_config(_fuel_diesel_source(seed=78))
    marked_record = canonicalize_prior_config(_r2s_diesel_source(seed=78))

    inherited = build_synthetic_dataset_run(
        unmarked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    remediated = build_synthetic_dataset_run(
        marked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    inherited_audit = inherited.metadata["r2c_mechanistic_remediation"]
    assert inherited_audit["profile"] == "r2k_sentinel_matrix_v1"

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2s_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2s_sentinel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.03, 0.045]
    assert params["feature_contrast_range"] == [0.24, 0.34]
    assert params["ch_overtone_gain_range"] == [0.12, 0.2]
    assert (
        params["spectra_source"]
        == "beer_lambert_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"
    )
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert not np.allclose(remediated.X, inherited.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2r_fruit_puree_source(seed=79),
        _r2q_lucas_ph_organic_source(seed=80),
        _r2p_phosphorus_source(seed=81),
        _r2o_beer_source(seed=82),
        _r2n_manure21_source(seed=83),
        _r2m_milk_source(seed=84),
        _juice_source(seed=85),
    ),
)
def test_r2s_non_diesel_draws_are_identical_to_r2r(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2r_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2r_sentinel_matrix_v1",
    )
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    r2r_audit = r2r_run.metadata["r2c_mechanistic_remediation"]
    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    assert r2s_audit["profile"] == r2r_audit["profile"]
    assert r2s_audit["transform_params"].get("spectra_rule") != (
        "micro_path_fuel_ch_overtone_contrast_readout"
    )
    assert r2s_audit == r2r_audit
    np.testing.assert_allclose(r2s_run.X, r2r_run.X)
    np.testing.assert_allclose(r2s_run.y, r2r_run.y)


def test_r2s_non_compliant_diesel_route_is_ignored() -> None:
    source = _r2s_diesel_source(seed=86)
    route = dict(cast("dict[str, object]", source["_r2s_diesel_readout_route"]))
    route["thresholds_modified"] = True
    source["_r2s_diesel_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2k_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2k_sentinel_matrix_v1",
    )
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    r2k_audit = r2k_run.metadata["r2c_mechanistic_remediation"]
    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    assert r2s_audit["profile"] == r2k_audit["profile"]
    assert r2s_audit["transform_params"].get("provenance_source") != (
        "exp09_dataset_token_diesel_route"
    )
    assert r2s_audit == r2k_audit
    np.testing.assert_allclose(r2s_run.X, r2k_run.X)
    np.testing.assert_allclose(r2s_run.y, r2k_run.y)


# ---------------------------------------------------------------------------
# R2t sentinel matrix remediation profile (explicit MANURE21-only heterogeneity).
# ---------------------------------------------------------------------------


def test_r2t_profile_is_opt_in_listed_and_requires_manure21_route() -> None:
    assert R2T_REMEDIATION_PROFILES == ("r2t_sentinel_matrix_v1",)
    assert "r2t_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    unmarked_record = canonicalize_prior_config(_soil_source(seed=87))
    marked_record = canonicalize_prior_config(_r2n_manure21_source(seed=87))

    inherited = build_synthetic_dataset_run(
        unmarked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2t_sentinel_matrix_v1",
    )
    remediated = build_synthetic_dataset_run(
        marked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    inherited_audit = inherited.metadata["r2c_mechanistic_remediation"]
    assert inherited_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert inherited_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_heterogeneous_scatter_patch_readout"
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2t_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2t_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_dried_manure_organic_mineral_centered"
    )
    assert params["spectra_rule"] == (
        "dried_manure_heterogeneous_scatter_patch_readout"
    )
    assert params["composition_source"] == (
        "textbook_dried_manure_organic_mineral_composition"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_particle_scatter_moisture_mineral_lumps"
    )
    assert params["path_factor_range"] == [0.36, 0.54]
    assert params["additive_baseline_range"] == [0.66, 0.9]
    assert params["scatter_slope_absorbance_range"] == [-0.26, 0.26]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.12]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.1]
    assert params["mineral_ash_absorbance_range"] == [-0.09, 0.09]
    assert params["heterogeneous_terms_centered"] is True
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_particle_size_moisture_organic_mineral_lump_prior"
    )
    assert (
        params["scatter_source"]
        == "fixed_dried_manure_particle_size_diffuse_scatter_prior"
    )
    assert params["manure21_readout_route_source"] == "exp09_dataset_token"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert not np.allclose(remediated.X, inherited.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=88),
        _r2r_fruit_puree_source(seed=89),
        _r2q_lucas_ph_organic_source(seed=90),
        _r2p_phosphorus_source(seed=91),
        _r2o_beer_source(seed=92),
        _r2m_milk_source(seed=93),
        _juice_source(seed=94),
    ),
)
def test_r2t_non_manure_draws_are_identical_to_r2s(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2t_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2t_audit = r2t_run.metadata["r2c_mechanistic_remediation"]
    assert r2t_audit["profile"] == r2s_audit["profile"]
    assert r2t_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_heterogeneous_scatter_patch_readout"
    )
    assert r2t_audit == r2s_audit
    np.testing.assert_allclose(r2t_run.X, r2s_run.X)
    np.testing.assert_allclose(r2t_run.y, r2s_run.y)


def test_r2t_non_compliant_manure21_route_is_ignored() -> None:
    source = _r2n_manure21_source(seed=95)
    route = dict(cast("dict[str, object]", source["_r2n_manure21_readout_route"]))
    route["real_stat_capture"] = True
    source["_r2n_manure21_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2t_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2t_audit = r2t_run.metadata["r2c_mechanistic_remediation"]
    assert r2t_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2t_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_heterogeneous_scatter_patch_readout"
    )
    assert r2t_audit == r2g_audit
    np.testing.assert_allclose(r2t_run.X, r2g_run.X)
    np.testing.assert_allclose(r2t_run.y, r2g_run.y)


# ---------------------------------------------------------------------------
# R2u sentinel matrix remediation profile (explicit MANURE21-only centered scatter).
# ---------------------------------------------------------------------------


def test_r2u_profile_is_opt_in_listed_keeps_r2t_available_and_requires_manure21_route() -> None:
    assert R2U_REMEDIATION_PROFILES == ("r2u_sentinel_matrix_v1",)
    assert "r2u_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    unmarked_record = canonicalize_prior_config(_soil_source(seed=96))
    marked_record = canonicalize_prior_config(_r2n_manure21_source(seed=96))

    inherited = build_synthetic_dataset_run(
        unmarked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2u_sentinel_matrix_v1",
    )
    remediated = build_synthetic_dataset_run(
        marked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    inherited_audit = inherited.metadata["r2c_mechanistic_remediation"]
    assert inherited_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert inherited_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_bounded_centered_scatter_readout"
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2u_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2u_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_dried_manure_organic_mineral_centered"
    )
    assert params["spectra_rule"] == "dried_manure_bounded_centered_scatter_readout"
    assert params["composition_source"] == (
        "textbook_dried_manure_organic_mineral_composition"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.52, 0.7]
    assert params["additive_baseline_range"] == [0.74, 0.86]
    assert params["scatter_slope_absorbance_range"] == [-0.18, 0.18]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.085]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.075]
    assert params["mineral_ash_absorbance_range"] == [-0.06, 0.06]
    assert params["heterogeneous_terms_centered"] is True
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_centered_particle_moisture_mineral_scatter_prior"
    )
    assert (
        params["scatter_source"]
        == "fixed_dried_manure_bounded_centered_scatter_prior"
    )
    assert (
        params["albedo_source"]
        == "fixed_dark_organic_reflectance_prior_r2n_continuum"
    )
    assert params["manure21_readout_route_source"] == "exp09_dataset_token"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert not np.allclose(remediated.X, inherited.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=97),
        _r2r_fruit_puree_source(seed=98),
        _r2q_lucas_ph_organic_source(seed=99),
        _r2p_phosphorus_source(seed=100),
        _r2o_beer_source(seed=101),
        _r2m_milk_source(seed=102),
        _juice_source(seed=103),
    ),
)
def test_r2u_non_manure_draws_are_identical_to_r2s(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2u_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2u_audit = r2u_run.metadata["r2c_mechanistic_remediation"]
    assert r2u_audit["profile"] == r2s_audit["profile"]
    assert r2u_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_bounded_centered_scatter_readout"
    )
    assert r2u_audit == r2s_audit
    np.testing.assert_allclose(r2u_run.X, r2s_run.X)
    np.testing.assert_allclose(r2u_run.y, r2s_run.y)


def test_r2u_non_compliant_manure21_route_is_ignored() -> None:
    source = _r2n_manure21_source(seed=104)
    route = dict(cast("dict[str, object]", source["_r2n_manure21_readout_route"]))
    route["thresholds_modified"] = True
    source["_r2n_manure21_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2u_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2u_audit = r2u_run.metadata["r2c_mechanistic_remediation"]
    assert r2u_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2u_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_bounded_centered_scatter_readout"
    )
    assert r2u_audit == r2g_audit
    np.testing.assert_allclose(r2u_run.X, r2g_run.X)
    np.testing.assert_allclose(r2u_run.y, r2g_run.y)


# ---------------------------------------------------------------------------
# R2v sentinel matrix remediation profile (explicit MANURE21-only balanced scatter).
# ---------------------------------------------------------------------------


def test_r2v_profile_is_opt_in_listed_keeps_r2t_r2u_available_and_requires_manure21_route() -> None:
    assert R2V_REMEDIATION_PROFILES == ("r2v_sentinel_matrix_v1",)
    assert "r2v_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2u_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    unmarked_record = canonicalize_prior_config(_soil_source(seed=105))
    marked_record = canonicalize_prior_config(_r2n_manure21_source(seed=105))

    inherited = build_synthetic_dataset_run(
        unmarked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2v_sentinel_matrix_v1",
    )
    remediated = build_synthetic_dataset_run(
        marked_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    inherited_audit = inherited.metadata["r2c_mechanistic_remediation"]
    assert inherited_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert inherited_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_balanced_centered_scatter_readout"
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2v_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2v_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == (
        "tight_dirichlet_dried_manure_organic_mineral_centered"
    )
    assert params["spectra_rule"] == "dried_manure_balanced_centered_scatter_readout"
    assert params["composition_source"] == (
        "textbook_dried_manure_organic_mineral_composition"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_balanced_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.6, 0.76]
    assert params["additive_baseline_range"] == [0.74, 0.86]
    assert params["scatter_slope_absorbance_range"] == [-0.16, 0.16]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.105]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.095]
    assert params["mineral_ash_absorbance_range"] == [-0.075, 0.075]
    assert params["heterogeneous_terms_centered"] is True
    assert params["balanced_centered_draws"] is True
    assert params["readout_centering_range_nm"] == [1100.0, 2500.0]
    assert params["readout_centering_grid"] == "uniform_wavenumber"
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_balanced_centered_particle_moisture_mineral_scatter_prior"
    )
    assert (
        params["scatter_source"]
        == "fixed_dried_manure_balanced_centered_scatter_prior"
    )
    assert (
        params["albedo_source"]
        == "fixed_dark_organic_reflectance_prior_r2n_continuum"
    )
    assert params["manure21_readout_route_source"] == "exp09_dataset_token"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert not np.allclose(remediated.X, inherited.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=106),
        _r2r_fruit_puree_source(seed=107),
        _r2q_lucas_ph_organic_source(seed=108),
        _r2p_phosphorus_source(seed=109),
        _r2o_beer_source(seed=110),
        _r2m_milk_source(seed=111),
        _juice_source(seed=112),
    ),
)
def test_r2v_non_manure_draws_are_identical_to_r2s(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2v_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2v_audit = r2v_run.metadata["r2c_mechanistic_remediation"]
    assert r2v_audit["profile"] == r2s_audit["profile"]
    assert r2v_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_balanced_centered_scatter_readout"
    )
    assert r2v_audit == r2s_audit
    np.testing.assert_allclose(r2v_run.X, r2s_run.X)
    np.testing.assert_allclose(r2v_run.y, r2s_run.y)


def test_r2v_non_compliant_manure21_route_is_ignored() -> None:
    source = _r2n_manure21_source(seed=113)
    route = dict(cast("dict[str, object]", source["_r2n_manure21_readout_route"]))
    route["real_stat_capture"] = True
    source["_r2n_manure21_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    r2v_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    r2g_audit = r2g_run.metadata["r2c_mechanistic_remediation"]
    r2v_audit = r2v_run.metadata["r2c_mechanistic_remediation"]
    assert r2v_audit["profile"] == "r2g_sentinel_matrix_v1"
    assert r2v_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_balanced_centered_scatter_readout"
    )
    assert r2v_audit == r2g_audit
    np.testing.assert_allclose(r2v_run.X, r2g_run.X)
    np.testing.assert_allclose(r2v_run.y, r2g_run.y)


# ---------------------------------------------------------------------------
# R2w sentinel matrix remediation profile (explicit MANURE21-only albedo variance).
# ---------------------------------------------------------------------------


def test_r2w_profile_is_opt_in_listed_and_records_non_oracle_manure21_route() -> None:
    assert R2W_REMEDIATION_PROFILES == ("r2w_sentinel_matrix_v1",)
    assert "r2w_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2v_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r2n_manure21_source(seed=114)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2w_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2w_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_albedo_variance_centered_scatter_readout"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_variance_plus_balanced_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.8, 1.0]
    assert params["additive_baseline_range"] == [0.72, 1.0]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.14]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.13]
    assert params["mineral_ash_absorbance_range"] == [-0.1, 0.1]
    assert params["heterogeneous_terms_centered"] is True
    assert params["balanced_centered_draws"] is True
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_albedo_variance_centered_particle_moisture_mineral_scatter_prior"
    )
    assert params["albedo_source"] == "fixed_wide_dark_organic_mineral_albedo_prior"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=115),
        _r2r_fruit_puree_source(seed=116),
        _r2q_lucas_ph_organic_source(seed=117),
        _r2p_phosphorus_source(seed=118),
        _r2o_beer_source(seed=119),
        _r2m_milk_source(seed=120),
        _juice_source(seed=121),
    ),
)
def test_r2w_non_manure_draws_are_identical_to_r2s(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2w_audit = r2w_run.metadata["r2c_mechanistic_remediation"]
    assert r2w_audit["profile"] == r2s_audit["profile"]
    assert r2w_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_albedo_variance_centered_scatter_readout"
    )
    assert r2w_audit == r2s_audit
    np.testing.assert_allclose(r2w_run.X, r2s_run.X)
    np.testing.assert_allclose(r2w_run.y, r2s_run.y)


# ---------------------------------------------------------------------------
# R2x sentinel matrix remediation profile (explicit MANURE21-only albedo dispersion).
# ---------------------------------------------------------------------------


def test_r2x_profile_is_opt_in_listed_and_records_non_oracle_manure21_route() -> None:
    assert R2X_REMEDIATION_PROFILES == ("r2x_sentinel_matrix_v1",)
    assert "r2x_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2w_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r2n_manure21_source(seed=122)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2x_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2x_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2x_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_coarse_albedo_dispersion_centered_readout"
    )
    assert params["spectra_source"] == (
        "fixed_coarse_dark_organic_albedo_dispersion_plus_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.8, 1.0]
    assert params["additive_baseline_range"] == [0.7, 1.02]
    assert params["scatter_slope_absorbance_range"] == [-0.16, 0.16]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.14]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.13]
    assert params["mineral_ash_absorbance_range"] == [-0.1, 0.1]
    assert params["balanced_centered_draws"] is True
    assert params["readout_centering_grid"] == "uniform_wavenumber"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_coarse_albedo_dispersion_centered_particle_moisture_mineral_scatter_prior"
    )
    assert params["albedo_source"] == (
        "fixed_wide_dark_organic_mineral_albedo_dispersion_prior"
    )
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=123),
        _r2r_fruit_puree_source(seed=124),
        _r2q_lucas_ph_organic_source(seed=125),
        _r2p_phosphorus_source(seed=126),
        _r2o_beer_source(seed=127),
        _r2m_milk_source(seed=128),
        _juice_source(seed=129),
    ),
)
def test_r2x_non_manure_draws_are_identical_to_r2s(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2x_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2x_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2x_audit = r2x_run.metadata["r2c_mechanistic_remediation"]
    assert r2x_audit["profile"] == r2s_audit["profile"]
    assert r2x_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_coarse_albedo_dispersion_centered_readout"
    )
    assert r2x_audit == r2s_audit
    np.testing.assert_allclose(r2x_run.X, r2s_run.X)
    np.testing.assert_allclose(r2x_run.y, r2s_run.y)


# ---------------------------------------------------------------------------
# R2y sentinel matrix remediation profile (explicit MANURE21-only soft low-frequency dispersion).
# ---------------------------------------------------------------------------


def test_r2y_profile_is_opt_in_listed_and_records_non_oracle_manure21_route() -> None:
    assert R2Y_REMEDIATION_PROFILES == ("r2y_sentinel_matrix_v1",)
    assert "r2y_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2w_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r2n_manure21_source(seed=130)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2y_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2y_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2y_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_soft_low_frequency_albedo_dispersion_centered_readout"
    )
    assert params["spectra_source"] == (
        "fixed_soft_low_frequency_dark_organic_albedo_dispersion_plus_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.82, 1.02]
    assert params["additive_baseline_range"] == [0.7, 1.02]
    assert params["scatter_slope_absorbance_range"] == [-0.15, 0.15]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.14]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.13]
    assert params["mineral_ash_absorbance_range"] == [-0.1, 0.1]
    assert params["balanced_centered_draws"] is True
    assert params["readout_centering_grid"] == "uniform_wavenumber"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_soft_low_frequency_albedo_dispersion_centered_particle_moisture_mineral_scatter_prior"
    )
    assert params["albedo_source"] == (
        "fixed_soft_wide_dark_organic_mineral_albedo_dispersion_prior"
    )
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=131),
        _r2r_fruit_puree_source(seed=132),
        _r2q_lucas_ph_organic_source(seed=133),
        _r2p_phosphorus_source(seed=134),
        _r2o_beer_source(seed=135),
        _r2m_milk_source(seed=136),
        _juice_source(seed=137),
    ),
)
def test_r2y_non_manure_draws_are_identical_to_r2s(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2y_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2y_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2y_audit = r2y_run.metadata["r2c_mechanistic_remediation"]
    assert r2y_audit["profile"] == r2s_audit["profile"]
    assert r2y_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_soft_low_frequency_albedo_dispersion_centered_readout"
    )
    assert r2y_audit == r2s_audit
    np.testing.assert_allclose(r2y_run.X, r2s_run.X)
    np.testing.assert_allclose(r2y_run.y, r2s_run.y)


# ---------------------------------------------------------------------------
# R2z sentinel matrix remediation profile (explicit MANURE21-only compositional heterogeneity).
# ---------------------------------------------------------------------------


def test_r2z_profile_is_opt_in_listed_and_records_non_oracle_manure21_route() -> None:
    assert R2Z_REMEDIATION_PROFILES == ("r2z_sentinel_matrix_v1",)
    assert "r2z_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2w_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r2n_manure21_source(seed=138)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2z_sentinel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2z_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2z_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_compositional_heterogeneity_centered_readout"
    )
    assert params["spectra_source"] == (
        "fixed_mean_neutral_compositional_heterogeneity_plus_smooth_centered_scatter_bands"
    )
    assert params["path_factor_range"] == [0.84, 1.02]
    assert params["additive_baseline_range"] == [0.71, 1.01]
    assert params["scatter_slope_absorbance_range"] == [-0.1, 0.1]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.12]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.11]
    assert params["mineral_ash_absorbance_range"] == [-0.08, 0.08]
    assert params["composition_heterogeneity"] == (
        "mean_neutral_dirichlet_concentration_scaled"
    )
    assert params["composition_alpha_concentration_scale"] == 0.72
    assert params["balanced_centered_draws"] is True
    assert params["readout_centering_grid"] == "uniform_wavenumber"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["heterogeneity_source"]
        == "fixed_dried_manure_mean_neutral_compositional_heterogeneity_smooth_centered_scatter_prior"
    )
    assert params["albedo_source"] == (
        "fixed_wide_dark_organic_mineral_albedo_prior_r2w_envelope"
    )
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=139),
        _r2r_fruit_puree_source(seed=140),
        _r2q_lucas_ph_organic_source(seed=141),
        _r2p_phosphorus_source(seed=142),
        _r2o_beer_source(seed=143),
        _r2m_milk_source(seed=144),
        _juice_source(seed=145),
    ),
)
def test_r2z_non_manure_draws_are_identical_to_r2s(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2s_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2z_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2z_sentinel_matrix_v1",
    )

    r2s_audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    r2z_audit = r2z_run.metadata["r2c_mechanistic_remediation"]
    assert r2z_audit["profile"] == r2s_audit["profile"]
    assert r2z_audit["transform_params"].get("spectra_rule") != (
        "dried_manure_compositional_heterogeneity_centered_readout"
    )
    assert r2z_audit == r2s_audit
    np.testing.assert_allclose(r2z_run.X, r2s_run.X)
    np.testing.assert_allclose(r2z_run.y, r2s_run.y)


# ---------------------------------------------------------------------------
# R3a CORN matrix remediation profile (explicit CORN-only powder readout).
# ---------------------------------------------------------------------------


def test_r3a_profile_is_opt_in_listed_and_records_non_oracle_corn_route() -> None:
    assert R3A_REMEDIATION_PROFILES == ("r3a_corn_matrix_v1",)
    assert "r3a_corn_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r2w_sentinel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3a_corn_source(seed=146)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3a_corn_matrix_v1"
    assert audit["scope"] == "bench_only_r3a_corn_matrix_remediation"
    assert audit["domain_key"] == "agriculture_grain"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_corn_grain_powder_centered"
    assert params["composition_source"] == "textbook_corn_grain_powder_composition"
    assert params["spectra_rule"] == "corn_powder_albedo_baseline_smoothing_readout"
    assert params["spectra_source"] == (
        "fixed_corn_meal_diffuse_reflectance_albedo_plus_particle_smoothing"
    )
    assert params["path_factor_range"] == [0.22, 0.34]
    assert params["additive_baseline_range"] == [0.34, 0.43]
    assert params["scatter_slope_absorbance_range"] == [-0.04, 0.04]
    assert params["moisture_band_absorbance_range"] == [0.0, 0.025]
    assert params["starch_band_absorbance_range"] == [0.0, 0.025]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "corn_powder_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_corn_route"
    assert params["matrix_source"] == "fixed_corn_grain_powder_prior"
    assert params["scatter_source"] == (
        "fixed_particle_size_diffuse_reflectance_smoothing_prior"
    )
    assert params["albedo_source"] == "fixed_corn_meal_reflectance_albedo_prior"
    assert params["corn_readout_route_source"] == "exp09_dataset_token"
    assert params["corn_readout_route_marker"] == "corn"
    assert params["corn_readout_route_non_oracle"] is True
    assert params["corn_readout_route_real_stat_capture"] is False
    assert params["corn_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3a_corn_readout_lifts_continuum_and_suppresses_derivative_vs_r2w() -> None:
    record = canonicalize_prior_config(_r3a_corn_source(seed=147))
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )

    assert not np.allclose(r3a_run.X, r2w_run.X)
    assert float(r3a_run.X.mean()) > float(r2w_run.X.mean())
    assert float(np.diff(r3a_run.X, axis=1).std()) < float(
        np.diff(r2w_run.X, axis=1).std()
    )
    assert r3a_run.y.shape == r2w_run.y.shape


def test_r3a_unmarked_corn_is_routed_back_to_r2w_inheritance() -> None:
    record = canonicalize_prior_config(_corn_source(seed=148))
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )

    r2w_audit = r2w_run.metadata["r2c_mechanistic_remediation"]
    r3a_audit = r3a_run.metadata["r2c_mechanistic_remediation"]
    assert r3a_audit["profile"] == r2w_audit["profile"]
    assert r3a_audit["transform_params"].get("spectra_rule") != (
        "corn_powder_albedo_baseline_smoothing_readout"
    )
    assert r3a_audit == r2w_audit
    np.testing.assert_allclose(r3a_run.X, r2w_run.X)
    np.testing.assert_allclose(r3a_run.y, r2w_run.y)


def test_r3a_non_compliant_corn_route_is_ignored() -> None:
    source = _r3a_corn_source(seed=149)
    route = dict(cast("dict[str, object]", source["_r3a_corn_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3a_corn_readout_route"] = route
    record = canonicalize_prior_config(source)
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )

    assert r3a_run.metadata["r2c_mechanistic_remediation"] == r2w_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3a_run.X, r2w_run.X)
    np.testing.assert_allclose(r3a_run.y, r2w_run.y)


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=150),
        _r2r_fruit_puree_source(seed=151),
        _r2q_lucas_ph_organic_source(seed=152),
        _r2p_phosphorus_source(seed=153),
        _r2o_beer_source(seed=154),
        _r2n_manure21_source(seed=155),
        _r2m_milk_source(seed=156),
        _juice_source(seed=157),
        _soil_source(seed=158),
    ),
)
def test_r3a_non_corn_draws_are_identical_to_r2w(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )

    r2w_audit = r2w_run.metadata["r2c_mechanistic_remediation"]
    r3a_audit = r3a_run.metadata["r2c_mechanistic_remediation"]
    assert r3a_audit["profile"] == r2w_audit["profile"]
    assert r3a_audit["transform_params"].get("spectra_rule") != (
        "corn_powder_albedo_baseline_smoothing_readout"
    )
    assert r3a_audit == r2w_audit
    np.testing.assert_allclose(r3a_run.X, r2w_run.X)
    np.testing.assert_allclose(r3a_run.y, r2w_run.y)


# ---------------------------------------------------------------------------
# R3b CORN matrix remediation profile (higher path dispersion, coarser smoothing).
# ---------------------------------------------------------------------------


def test_r3b_profile_is_opt_in_listed_and_records_non_oracle_corn_route() -> None:
    assert R3B_REMEDIATION_PROFILES == ("r3b_corn_matrix_v1",)
    assert "r3b_corn_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3a_corn_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3b_corn_source(seed=159)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3b_corn_matrix_v1"
    assert audit["scope"] == "bench_only_r3b_corn_matrix_remediation"
    assert audit["domain_key"] == "agriculture_grain"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_corn_grain_powder_centered"
    assert params["spectra_rule"] == (
        "corn_powder_albedo_path_dispersion_smoothing_readout"
    )
    assert params["spectra_source"] == (
        "fixed_corn_meal_albedo_plus_coarse_particle_path_dispersion_smoothing"
    )
    assert params["path_factor_range"] == [0.9, 1.35]
    assert params["additive_baseline_range"] == [0.34, 0.43]
    assert params["scatter_slope_absorbance_range"] == [-0.012, 0.012]
    assert params["moisture_band_absorbance_range"] == [0.035, 0.105]
    assert params["starch_band_absorbance_range"] == [0.035, 0.105]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "corn_powder_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_corn_route"
    assert params["scatter_source"] == (
        "fixed_coarse_particle_size_path_dispersion_smoothing_prior"
    )
    assert params["corn_readout_route_source"] == "exp09_dataset_token"
    assert params["corn_readout_route_marker"] == "corn"
    assert params["corn_readout_route_non_oracle"] is True
    assert params["corn_readout_route_real_stat_capture"] is False
    assert params["corn_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3b_corn_recovers_amplitude_vs_r3a_while_staying_smoother_than_r2w() -> None:
    r2w_record = canonicalize_prior_config(_r3b_corn_source(seed=160))
    r3a_record = canonicalize_prior_config(_r3a_corn_source(seed=160))
    r3b_record = canonicalize_prior_config(_r3b_corn_source(seed=160))
    r2w_run = build_synthetic_dataset_run(
        r2w_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3a_run = build_synthetic_dataset_run(
        r3a_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3a_corn_matrix_v1",
    )
    r3b_run = build_synthetic_dataset_run(
        r3b_record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )

    assert not np.allclose(r3b_run.X, r3a_run.X)
    r3a_amplitude = float(np.median(np.ptp(r3a_run.X, axis=1)))
    r3b_amplitude = float(np.median(np.ptp(r3b_run.X, axis=1)))
    assert r3b_amplitude > r3a_amplitude
    assert float(np.diff(r3b_run.X, axis=1).std()) < float(
        np.diff(r2w_run.X, axis=1).std()
    )
    assert r3b_run.y.shape == r3a_run.y.shape


def test_r3b_unmarked_or_non_compliant_corn_is_routed_back_to_r2w() -> None:
    unmarked = canonicalize_prior_config(_corn_source(seed=161))
    r2w_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3b_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )
    assert r3b_unmarked.metadata["r2c_mechanistic_remediation"] == r2w_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3b_unmarked.X, r2w_run.X)

    source = _r3b_corn_source(seed=162)
    route = dict(cast("dict[str, object]", source["_r3b_corn_readout_route"]))
    route["thresholds_modified"] = True
    source["_r3b_corn_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3b_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )
    r2w_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    assert r3b_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r2w_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3b_non_compliant.X, r2w_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r2s_diesel_source(seed=163),
        _r2r_fruit_puree_source(seed=164),
        _r2q_lucas_ph_organic_source(seed=165),
        _r2p_phosphorus_source(seed=166),
        _r2o_beer_source(seed=167),
        _r2n_manure21_source(seed=168),
        _r2m_milk_source(seed=169),
        _juice_source(seed=170),
        _soil_source(seed=171),
    ),
)
def test_r3b_non_corn_draws_are_identical_to_r2w(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r2w_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r2w_sentinel_matrix_v1",
    )
    r3b_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )

    assert r3b_run.metadata["r2c_mechanistic_remediation"] == r2w_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3b_run.X, r2w_run.X)
    np.testing.assert_allclose(r3b_run.y, r2w_run.y)


# ---------------------------------------------------------------------------
# R3c DIESEL matrix remediation profile (lower-offset R2s-derived readout).
# ---------------------------------------------------------------------------


def test_r3c_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R3C_REMEDIATION_PROFILES == ("r3c_diesel_matrix_v1",)
    assert "r3c_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3b_corn_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3c_diesel_source(seed=172)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3c_diesel_matrix_v1"
    assert audit["scope"] == "bench_only_r3c_diesel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.024, 0.036]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3c_diesel_lowers_offset_and_damps_amplitude_vs_r2s_without_r2j_floor() -> None:
    record_r2s = canonicalize_prior_config(_r2s_diesel_source(seed=173))
    record_r3c = canonicalize_prior_config(_r3c_diesel_source(seed=173))
    r2s_run = build_synthetic_dataset_run(
        record_r2s,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r3c_run = build_synthetic_dataset_run(
        record_r3c,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )

    r3c_params = r3c_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert r3c_params["spectra_rule"] != "micro_path_fuel_transmission_absorbance_floor"
    assert float(r3c_run.X.mean()) < float(r2s_run.X.mean())
    assert float(r3c_run.X.std()) < float(r2s_run.X.std())
    assert float(np.median(np.ptp(r3c_run.X, axis=1))) < float(
        np.median(np.ptp(r2s_run.X, axis=1))
    )
    assert float(np.diff(r3c_run.X, axis=1).std()) > 0.0
    assert not np.allclose(r3c_run.X, r2s_run.X)


def test_r3c_unmarked_or_non_compliant_diesel_is_routed_back_to_r3b() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=174))
    r3b_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )
    r3c_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    assert r3c_unmarked.metadata["r2c_mechanistic_remediation"] == r3b_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3c_unmarked.X, r3b_run.X)

    source = _r3c_diesel_source(seed=175)
    route = dict(cast("dict[str, object]", source["_r3c_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3c_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3c_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    r3b_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )
    assert r3c_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3b_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3c_non_compliant.X, r3b_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=176),
        _r2r_fruit_puree_source(seed=177),
        _r2q_lucas_ph_organic_source(seed=178),
        _r2p_phosphorus_source(seed=179),
        _r2o_beer_source(seed=180),
        _r2n_manure21_source(seed=181),
        _r2m_milk_source(seed=182),
        _juice_source(seed=183),
        _soil_source(seed=184),
    ),
)
def test_r3c_non_diesel_draws_are_identical_to_r3b(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3b_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3b_corn_matrix_v1",
    )
    r3c_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )

    assert r3c_run.metadata["r2c_mechanistic_remediation"] == r3b_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3c_run.X, r3b_run.X)
    np.testing.assert_allclose(r3c_run.y, r3b_run.y)


# ---------------------------------------------------------------------------
# R3d DIESEL matrix remediation profile (shorter path/lower detector offset).
# ---------------------------------------------------------------------------


def test_r3d_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R3D_REMEDIATION_PROFILES == ("r3d_diesel_matrix_v1",)
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3c_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=185)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3d_diesel_matrix_v1"
    assert audit["scope"] == "bench_only_r3d_diesel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3d_diesel_reduces_broadband_offset_vs_r3c_without_zeroing_derivative() -> None:
    record_r3c = canonicalize_prior_config(_r3c_diesel_source(seed=186))
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=186))
    r3c_run = build_synthetic_dataset_run(
        record_r3c,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )

    assert float(r3d_run.X.mean()) < float(r3c_run.X.mean())
    assert float(r3d_run.X.std()) < float(r3c_run.X.std())
    assert float(np.median(np.ptp(r3d_run.X, axis=1))) < float(
        np.median(np.ptp(r3c_run.X, axis=1))
    )
    assert float(np.diff(r3d_run.X, axis=1).std()) > 0.0
    assert not np.allclose(r3d_run.X, r3c_run.X)


def test_r3d_unmarked_or_non_compliant_diesel_is_routed_back_to_r3c() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=187))
    r3c_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    r3d_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r3d_unmarked.metadata["r2c_mechanistic_remediation"] == r3c_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3d_unmarked.X, r3c_run.X)

    source = _r3d_diesel_source(seed=188)
    route = dict(cast("dict[str, object]", source["_r3d_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3d_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3c_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    assert r3d_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3c_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3d_non_compliant.X, r3c_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=189),
        _r2r_fruit_puree_source(seed=190),
        _r2q_lucas_ph_organic_source(seed=191),
        _r2p_phosphorus_source(seed=192),
        _r2o_beer_source(seed=193),
        _r2n_manure21_source(seed=194),
        _r2m_milk_source(seed=195),
        _juice_source(seed=196),
        _soil_source(seed=197),
    ),
)
def test_r3d_non_diesel_draws_are_identical_to_r3c(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3c_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3c_diesel_matrix_v1",
    )
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )

    assert r3d_run.metadata["r2c_mechanistic_remediation"] == r3c_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3d_run.X, r3c_run.X)
    np.testing.assert_allclose(r3d_run.y, r3c_run.y)


# ---------------------------------------------------------------------------
# R3e DIESEL matrix remediation profile (minimal path/near-zero offset).
# ---------------------------------------------------------------------------


def test_r3e_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R3E_REMEDIATION_PROFILES == ("r3e_diesel_matrix_v1",)
    assert "r3e_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3e_diesel_source(seed=198)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3e_diesel_matrix_v1"
    assert audit["scope"] == "bench_only_r3e_diesel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.004, 0.009]
    assert params["additive_baseline_range"] == [0.0, 0.0001]
    assert params["feature_contrast_range"] == [0.20, 0.28]
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3e_diesel_reduces_broadband_offset_vs_r3d_without_zeroing_derivative() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=199))
    record_r3e = canonicalize_prior_config(_r3e_diesel_source(seed=199))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3e_run = build_synthetic_dataset_run(
        record_r3e,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )

    assert float(r3e_run.X.mean()) < float(r3d_run.X.mean())
    assert float(r3e_run.X.std()) < float(r3d_run.X.std())
    assert float(np.diff(r3e_run.X, axis=1).std()) > 0.0
    assert not np.allclose(r3e_run.X, r3d_run.X)


def test_r3e_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=200))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3e_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    assert r3e_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3e_unmarked.X, r3d_run.X)

    source = _r3e_diesel_source(seed=201)
    route = dict(cast("dict[str, object]", source["_r3e_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3e_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3e_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r3e_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3e_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=202),
        _r2r_fruit_puree_source(seed=203),
        _r2q_lucas_ph_organic_source(seed=204),
        _r2p_phosphorus_source(seed=205),
        _r2o_beer_source(seed=206),
        _r2n_manure21_source(seed=207),
        _r2m_milk_source(seed=208),
        _juice_source(seed=209),
        _soil_source(seed=210),
    ),
)
def test_r3e_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3e_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )

    assert r3e_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3e_run.X, r3d_run.X)
    np.testing.assert_allclose(r3e_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R3f DIESEL matrix remediation profile (restored residual contrast with lower offset).
# ---------------------------------------------------------------------------


def test_r3f_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R3F_REMEDIATION_PROFILES == ("r3f_diesel_matrix_v1",)
    assert "r3f_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3e_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3f_diesel_source(seed=211)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3f_diesel_matrix_v1"
    assert audit["scope"] == "bench_only_r3f_diesel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.009, 0.016]
    assert params["additive_baseline_range"] == [0.0, 0.00012]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3f_diesel_restores_derivative_contrast_vs_r3e_without_r3d_offset() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=212))
    record_r3e = canonicalize_prior_config(_r3e_diesel_source(seed=212))
    record_r3f = canonicalize_prior_config(_r3f_diesel_source(seed=212))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3e_run = build_synthetic_dataset_run(
        record_r3e,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    r3f_run = build_synthetic_dataset_run(
        record_r3f,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )

    assert r3f_run.metadata["r2c_mechanistic_remediation"]["transform_params"][
        "additive_baseline_range"
    ] < r3d_run.metadata["r2c_mechanistic_remediation"]["transform_params"][
        "additive_baseline_range"
    ]
    assert float(np.diff(r3f_run.X, axis=1).std()) > float(
        np.diff(r3e_run.X, axis=1).std()
    )
    assert float(np.diff(r3f_run.X, axis=1).std()) > 0.0
    assert not np.allclose(r3f_run.X, r3e_run.X)


def test_r3f_unmarked_or_non_compliant_diesel_is_routed_back_to_r3e() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=213))
    r3e_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    r3f_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    assert r3f_unmarked.metadata["r2c_mechanistic_remediation"] == r3e_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3f_unmarked.X, r3e_run.X)

    source = _r3f_diesel_source(seed=214)
    route = dict(cast("dict[str, object]", source["_r3f_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3f_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3f_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    r3e_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    assert r3f_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3e_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3f_non_compliant.X, r3e_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=215),
        _r2r_fruit_puree_source(seed=216),
        _r2q_lucas_ph_organic_source(seed=217),
        _r2p_phosphorus_source(seed=218),
        _r2o_beer_source(seed=219),
        _r2n_manure21_source(seed=220),
        _r2m_milk_source(seed=221),
        _juice_source(seed=222),
        _soil_source(seed=223),
    ),
)
def test_r3f_non_diesel_draws_are_identical_to_r3e(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3e_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3e_diesel_matrix_v1",
    )
    r3f_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )

    assert r3f_run.metadata["r2c_mechanistic_remediation"] == r3e_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3f_run.X, r3e_run.X)
    np.testing.assert_allclose(r3f_run.y, r3e_run.y)


def test_r3g_profile_is_opt_in_listed_and_records_fixed_hydrocarbon_envelope() -> None:
    assert R3G_REMEDIATION_PROFILES == ("r3g_diesel_matrix_v1",)
    assert "r3g_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES
    assert "r3f_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3g_diesel_source(seed=224)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3g_diesel_matrix_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r3g_diesel_matrix_v1"
    assert audit["scope"] == "bench_only_r3g_diesel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.003, 0.007]
    assert params["additive_baseline_range"] == [0.0, 0.00012]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["fixed_envelope_absorbance_range"] == [0.0005, 0.001]
    assert params["fixed_envelope_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert params["fixed_envelope_centered"] is True
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r3g_diesel_adds_shape_envelope_without_r3d_offset() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=225))
    record_r3f = canonicalize_prior_config(_r3f_diesel_source(seed=225))
    record_r3g = canonicalize_prior_config(_r3g_diesel_source(seed=225))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r3f_run = build_synthetic_dataset_run(
        record_r3f,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    r3g_run = build_synthetic_dataset_run(
        record_r3g,
        n_samples=48,
        random_seed=4242,
        remediation_profile="r3g_diesel_matrix_v1",
    )

    params = r3g_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert params["fixed_envelope_centered"] is True
    assert float(r3g_run.X.mean()) < float(r3d_run.X.mean())
    assert float(np.diff(r3g_run.X, axis=1).std()) > 0.0
    assert not np.allclose(r3g_run.X, r3f_run.X)


def test_r3g_unmarked_or_non_compliant_diesel_is_routed_back_to_r3f() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=226))
    r3f_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    r3g_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3g_diesel_matrix_v1",
    )
    assert r3g_unmarked.metadata["r2c_mechanistic_remediation"] == r3f_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3g_unmarked.X, r3f_run.X)

    source = _r3g_diesel_source(seed=227)
    route = dict(cast("dict[str, object]", source["_r3g_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r3g_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r3g_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3g_diesel_matrix_v1",
    )
    r3f_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    assert r3g_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3f_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_allclose(r3g_non_compliant.X, r3f_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=228),
        _r2r_fruit_puree_source(seed=229),
        _r2q_lucas_ph_organic_source(seed=230),
        _r2p_phosphorus_source(seed=231),
        _r2o_beer_source(seed=232),
        _r2n_manure21_source(seed=233),
        _r2m_milk_source(seed=234),
        _juice_source(seed=235),
        _soil_source(seed=236),
    ),
)
def test_r3g_non_diesel_draws_are_identical_to_r3f(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3f_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3f_diesel_matrix_v1",
    )
    r3g_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3g_diesel_matrix_v1",
    )

    assert r3g_run.metadata["r2c_mechanistic_remediation"] == r3f_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_allclose(r3g_run.X, r3f_run.X)
    np.testing.assert_allclose(r3g_run.y, r3f_run.y)


# ---------------------------------------------------------------------------
# R4a DIESEL basis remediation profile (R3d inheritance, support-aware shape).
# ---------------------------------------------------------------------------


def test_r4a_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R4A_REMEDIATION_PROFILES == ("r4a_diesel_basis_v1",)
    assert "r4a_diesel_basis_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4a_diesel_source(seed=240)),
        n_samples=24,
        random_seed=4242,
        remediation_profile="r4a_diesel_basis_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r4a_diesel_basis_v1"
    assert audit["scope"] == "bench_only_r4a_diesel_basis_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    # Inherited R3d path/baseline (no broadband level change in R4a).
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    # New R4a CH overtone basis: support-only, no 1720 nm.
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 46.0
    assert params["ch_overtone_gain_range"] == [0.055, 0.105]
    # Damping windows inside the over-structured 1100-1500 nm region.
    assert params["damping_windows_nm"] == [
        [1180.0, 70.0, 1.0],
        [1425.0, 85.0, 1.0],
    ]
    assert params["damping_strength_range"] == [0.30, 0.50]
    # Short-continuum hydrocarbon hump centered on the 750-1550 nm support.
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 90.0
    assert params["continuum_hump_amplitude_range"] == [0.00025, 0.00065]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # No fixed envelope (R4a does not inherit R3g).
    assert "fixed_envelope_absorbance_range" not in params
    assert "fixed_envelope_centers_nm" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r4a_diesel_changes_morphology_vs_r3d_without_global_inversion() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=241))
    record_r4a = canonicalize_prior_config(_r4a_diesel_source(seed=241))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=64,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        record_r4a,
        n_samples=64,
        random_seed=4242,
        remediation_profile="r4a_diesel_basis_v1",
    )

    # R4a must produce different spectra than R3d.
    assert not np.allclose(r4a_run.X, r3d_run.X)

    wl = r4a_run.wavelengths
    short_band = (wl >= 900.0) & (wl <= 1050.0)
    structured_band = (wl >= 1100.0) & (wl <= 1500.0)
    assert short_band.any()
    assert structured_band.any()

    # 900-1050 nm energy/variation increases under R4a (continuum hump support).
    r4a_short = r4a_run.X[:, short_band]
    r3d_short = r3d_run.X[:, short_band]
    assert float(r4a_short.std()) > float(r3d_short.std())
    assert float(r4a_short.mean()) > float(r3d_short.mean())

    # 1100-1500 nm structure is damped under R4a vs R3d.
    r4a_struct = r4a_run.X[:, structured_band]
    r3d_struct = r3d_run.X[:, structured_band]
    r4a_struct_amp = float(np.median(np.ptp(r4a_struct, axis=1)))
    r3d_struct_amp = float(np.median(np.ptp(r3d_struct, axis=1)))
    assert r4a_struct_amp < r3d_struct_amp

    # No global inversion: positive correlation between mean curves.
    r4a_mean = r4a_run.X.mean(axis=0)
    r3d_mean = r3d_run.X.mean(axis=0)
    if r4a_mean.std() > 0 and r3d_mean.std() > 0:
        corr = float(np.corrcoef(r4a_mean, r3d_mean)[0, 1])
        assert corr > 0.0

    # Derivative structure is not zeroed: still positive.
    assert float(np.diff(r4a_run.X, axis=1).std()) > 0.0


def test_r4a_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=242))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r4a_diesel_basis_v1",
    )
    assert r4a_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4a_unmarked.X, r3d_run.X)

    source = _r4a_diesel_source(seed=243)
    route = dict(cast("dict[str, object]", source["_r4a_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r4a_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r4a_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r4a_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r4a_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=244),
        _r2r_fruit_puree_source(seed=245),
        _r2q_lucas_ph_organic_source(seed=246),
        _r2p_phosphorus_source(seed=247),
        _r2o_beer_source(seed=248),
        _r2n_manure21_source(seed=249),
        _r2m_milk_source(seed=250),
        _juice_source(seed=251),
        _soil_source(seed=252),
    ),
)
def test_r4a_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4242,
        remediation_profile="r4a_diesel_basis_v1",
    )

    assert r4a_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4a_run.X, r3d_run.X)
    np.testing.assert_array_equal(r4a_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R4b DIESEL derivative-restore remediation profile (R3d inheritance,
# narrower CH width / lower damping than R4a to restore derivative structure).
# ---------------------------------------------------------------------------


def test_r4b_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R4B_REMEDIATION_PROFILES == ("r4b_diesel_derivative_restore_v1",)
    assert "r4b_diesel_derivative_restore_v1" in ALL_REMEDIATION_PROFILES
    assert "r4a_diesel_basis_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4b_diesel_source(seed=340)),
        n_samples=24,
        random_seed=4343,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r4b_diesel_derivative_restore_v1"
    assert audit["scope"] == "bench_only_r4b_diesel_derivative_restore_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    params = audit["transform_params"]
    # Inherited R3d path/baseline byte-for-byte.
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    # R4b CH overtone basis: same support-only centers as R4a, narrower width.
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 38.0
    assert params["ch_overtone_gain_range"] == [0.085, 0.145]
    # Narrower / weaker damping windows than R4a.
    assert params["damping_windows_nm"] == [
        [1180.0, 52.0, 0.75],
        [1425.0, 62.0, 0.85],
    ]
    assert params["damping_strength_range"] == [0.10, 0.22]
    # Narrower lower-amplitude short-continuum hump on the same support.
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 75.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00035]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    assert "fixed_envelope_absorbance_range" not in params
    assert "fixed_envelope_centers_nm" not in params
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r4b_diesel_differs_from_r4a_and_r3d_and_restores_derivative_std() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=341))
    record_r4a = canonicalize_prior_config(_r4a_diesel_source(seed=341))
    record_r4b = canonicalize_prior_config(_r4b_diesel_source(seed=341))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=64,
        random_seed=4343,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        record_r4a,
        n_samples=64,
        random_seed=4343,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r4b_run = build_synthetic_dataset_run(
        record_r4b,
        n_samples=64,
        random_seed=4343,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )

    # R4b spectra differ from both R3d and R4a.
    assert not np.allclose(r4b_run.X, r3d_run.X)
    assert not np.allclose(r4b_run.X, r4a_run.X)

    # Derivative structure restored vs R4a (the original R4b motivation).
    r4b_deriv_std = float(np.diff(r4b_run.X, axis=1).std())
    r4a_deriv_std = float(np.diff(r4a_run.X, axis=1).std())
    assert r4b_deriv_std > r4a_deriv_std


def test_r4b_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=342))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4b_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )
    assert r4b_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4b_unmarked.X, r3d_run.X)

    source = _r4b_diesel_source(seed=343)
    route = dict(cast("dict[str, object]", source["_r4b_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r4b_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r4b_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r4b_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r4b_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=344),
        _r2r_fruit_puree_source(seed=345),
        _r2o_beer_source(seed=346),
        _r2m_milk_source(seed=347),
    ),
)
def test_r4b_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4b_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4343,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )

    assert r4b_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4b_run.X, r3d_run.X)
    np.testing.assert_array_equal(r4b_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R4c DIESEL balanced-derivative remediation profile (R3d inheritance,
# narrower CH width / weaker damping than R4b to push derivative closer to
# R3d while keeping the R4b gap/mean shift improvement).
# ---------------------------------------------------------------------------


def test_r4c_profile_is_opt_in_listed_and_records_non_oracle_diesel_route() -> None:
    assert R4C_REMEDIATION_PROFILES == ("r4c_diesel_balanced_derivative_v1",)
    assert "r4c_diesel_balanced_derivative_v1" in ALL_REMEDIATION_PROFILES
    assert "r4b_diesel_derivative_restore_v1" in ALL_REMEDIATION_PROFILES
    assert "r4a_diesel_basis_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=440)),
        n_samples=24,
        random_seed=4444,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r4c_diesel_balanced_derivative_v1"
    assert (
        audit["scope"]
        == "bench_only_r4c_diesel_balanced_derivative_remediation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    # Inherited R3d path/baseline byte-for-byte.
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    # R4c CH overtone basis: same support-only centers as R4a/R4b, narrower width.
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    # Narrower / weaker damping windows than R4b.
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    # Narrower lower-amplitude short-continuum hump on the same support.
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # No fixed envelope (R4c does not inherit R3g).
    assert "fixed_envelope_absorbance_range" not in params
    assert "fixed_envelope_centers_nm" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r4c_diesel_differs_from_r4b_and_r3d_and_increases_derivative_std_vs_r4b() -> None:
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=441))
    record_r4b = canonicalize_prior_config(_r4b_diesel_source(seed=441))
    record_r4c = canonicalize_prior_config(_r4c_diesel_source(seed=441))
    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=64,
        random_seed=4444,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4b_run = build_synthetic_dataset_run(
        record_r4b,
        n_samples=64,
        random_seed=4444,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )
    r4c_run = build_synthetic_dataset_run(
        record_r4c,
        n_samples=64,
        random_seed=4444,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )

    # R4c spectra differ from both R3d and R4b on-target.
    assert not np.allclose(r4c_run.X, r3d_run.X)
    assert not np.allclose(r4c_run.X, r4b_run.X)

    # Derivative structure pushed closer to R3d than R4b: R4c std > R4b std.
    r4c_deriv_std = float(np.diff(r4c_run.X, axis=1).std())
    r4b_deriv_std = float(np.diff(r4b_run.X, axis=1).std())
    assert r4c_deriv_std > r4b_deriv_std


def test_r4c_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=442))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4c_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    assert r4c_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4c_unmarked.X, r3d_run.X)

    source = _r4c_diesel_source(seed=443)
    route = dict(cast("dict[str, object]", source["_r4c_diesel_readout_route"]))
    route["real_stat_capture"] = True
    source["_r4c_diesel_readout_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r4c_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r4c_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r4c_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=444),
        _r2r_fruit_puree_source(seed=445),
        _r2o_beer_source(seed=446),
        _r2m_milk_source(seed=447),
    ),
)
def test_r4c_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4c_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=4444,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )

    assert r4c_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r4c_run.X, r3d_run.X)
    np.testing.assert_array_equal(r4c_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R5 DIESEL readout-space remediation profiles. R5a is byte-identical to R4c
# on the same seed and explicit DIESEL route; R5b/R5c apply a deterministic
# readout-space transform on top of the R4c absorbance pipeline.
# ---------------------------------------------------------------------------


def _r5a_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r5a_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r5b_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r5b_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def _r5c_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r5c_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


_R5_PROFILE_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "r5a_diesel_absorbance_readout_v1",
        "bench_only_r5a_diesel_absorbance_readout_remediation",
        "absorbance",
        "uncalibrated_raw_absorbance",
    ),
    (
        "r5b_diesel_transmittance_readout_v1",
        "bench_only_r5b_diesel_transmittance_readout_remediation",
        "transmittance",
        "uncalibrated_raw_transmittance",
    ),
    (
        "r5c_diesel_blank_referenced_intensity_v1",
        "bench_only_r5c_diesel_blank_referenced_intensity_remediation",
        "blank_referenced_intensity",
        "uncalibrated_raw_blank_referenced_intensity",
    ),
)


def _r5_source_for(profile: str, *, seed: int) -> dict[str, object]:
    if profile == "r5a_diesel_absorbance_readout_v1":
        return _r5a_diesel_source(seed=seed)
    if profile == "r5b_diesel_transmittance_readout_v1":
        return _r5b_diesel_source(seed=seed)
    if profile == "r5c_diesel_blank_referenced_intensity_v1":
        return _r5c_diesel_source(seed=seed)
    raise AssertionError(f"unknown R5 profile {profile!r}")


def test_r5_profiles_are_opt_in_listed() -> None:
    assert R5A_REMEDIATION_PROFILES == ("r5a_diesel_absorbance_readout_v1",)
    assert R5B_REMEDIATION_PROFILES == ("r5b_diesel_transmittance_readout_v1",)
    assert R5C_REMEDIATION_PROFILES == (
        "r5c_diesel_blank_referenced_intensity_v1",
    )
    for profile_id, _scope, _transform, _readout_space in _R5_PROFILE_CASES:
        assert profile_id in ALL_REMEDIATION_PROFILES
    assert "r4c_diesel_balanced_derivative_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES


@pytest.mark.parametrize(
    "profile_id,scope,transform,readout_space",
    _R5_PROFILE_CASES,
)
def test_r5_profile_records_non_oracle_diesel_route_and_readout_metadata(
    profile_id: str, scope: str, transform: str, readout_space: str
) -> None:
    record = canonicalize_prior_config(_r5_source_for(profile_id, seed=540))
    remediated = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=5454,
        remediation_profile=profile_id,
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == profile_id
    assert audit["scope"] == scope
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    # R5 inherits the full R4c absorbance pipeline byte-for-byte.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # Readout-space metadata is the only thing R5 changes.
    assert params["readout_space"] == readout_space
    assert params["readout_space_transform"] == transform
    assert params["readout_space_transform_clip"] == [0.0, 1.0]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_source"] == "exp09_dataset_token"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r5a_diesel_is_byte_identical_to_r4c_on_explicit_diesel_route() -> None:
    seed = 541
    rs = 5454
    n = 32
    record_r4c = canonicalize_prior_config(_r4c_diesel_source(seed=seed))
    record_r5a = canonicalize_prior_config(_r5a_diesel_source(seed=seed))
    r4c_run = build_synthetic_dataset_run(
        record_r4c,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r5a_run = build_synthetic_dataset_run(
        record_r5a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r5a_diesel_absorbance_readout_v1",
    )

    np.testing.assert_array_equal(r5a_run.X, r4c_run.X)
    np.testing.assert_array_equal(r5a_run.y, r4c_run.y)
    r5a_params = r5a_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert r5a_params["readout_space_transform"] == "absorbance"


def test_r5b_and_r5c_differ_from_r5a_and_lie_in_unit_interval() -> None:
    seed = 542
    rs = 5454
    n = 32
    record_r5a = canonicalize_prior_config(_r5a_diesel_source(seed=seed))
    record_r5b = canonicalize_prior_config(_r5b_diesel_source(seed=seed))
    record_r5c = canonicalize_prior_config(_r5c_diesel_source(seed=seed))
    r5a_run = build_synthetic_dataset_run(
        record_r5a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r5a_diesel_absorbance_readout_v1",
    )
    r5b_run = build_synthetic_dataset_run(
        record_r5b,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r5b_diesel_transmittance_readout_v1",
    )
    r5c_run = build_synthetic_dataset_run(
        record_r5c,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r5c_diesel_blank_referenced_intensity_v1",
    )

    assert not np.allclose(r5b_run.X, r5a_run.X)
    assert not np.allclose(r5c_run.X, r5a_run.X)
    assert not np.allclose(r5b_run.X, r5c_run.X)
    assert float(r5b_run.X.min()) >= 0.0
    assert float(r5b_run.X.max()) <= 1.0
    assert float(r5c_run.X.min()) >= 0.0
    assert float(r5c_run.X.max()) <= 1.0
    np.testing.assert_allclose(
        r5b_run.X,
        np.clip(np.power(10.0, -r5a_run.X), 0.0, 1.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        r5c_run.X,
        np.clip(1.0 - np.power(10.0, -r5a_run.X), 0.0, 1.0),
        atol=1e-12,
    )
    # T = 10**-A and (1 - T) are complementary on the absorbance pipeline.
    np.testing.assert_allclose(r5b_run.X + r5c_run.X, 1.0, atol=1e-12)
    np.testing.assert_array_equal(r5b_run.y, r5a_run.y)
    np.testing.assert_array_equal(r5c_run.y, r5a_run.y)


@pytest.mark.parametrize(
    "profile_id,scope,transform,readout_space",
    _R5_PROFILE_CASES,
)
def test_r5_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d(
    profile_id: str, scope: str, transform: str, readout_space: str
) -> None:
    del scope, transform, readout_space
    rs = 5454
    n = 24
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=543))
    r3d_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r5_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=n,
        random_seed=rs,
        remediation_profile=profile_id,
    )
    assert r5_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_unmarked.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r5_unmarked.X, r3d_unmarked.X)

    source = _r5_source_for(profile_id, seed=544)
    route_key = f"_{profile_id.split('_')[0]}_diesel_readout_route"
    route = dict(cast("dict[str, object]", source[route_key]))
    route["real_stat_capture"] = True
    source[route_key] = route
    non_compliant = canonicalize_prior_config(source)
    r5_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=n,
        random_seed=rs,
        remediation_profile=profile_id,
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r5_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r5_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "profile_id,scope,transform,readout_space",
    _R5_PROFILE_CASES,
)
@pytest.mark.parametrize(
    "non_diesel_source",
    (
        _r3b_corn_source(seed=545),
        _r2r_fruit_puree_source(seed=546),
        _r2o_beer_source(seed=547),
        _r2m_milk_source(seed=548),
        _r2n_manure21_source(seed=549),
        _soil_source(seed=550),
    ),
)
def test_r5_non_diesel_draws_are_identical_to_r3d(
    profile_id: str,
    scope: str,
    transform: str,
    readout_space: str,
    non_diesel_source: dict[str, object],
) -> None:
    del scope, transform, readout_space
    record = canonicalize_prior_config(non_diesel_source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=5454,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r5_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=5454,
        remediation_profile=profile_id,
    )

    assert r5_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r5_run.X, r3d_run.X)
    np.testing.assert_array_equal(r5_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R6a DIESEL centered hydrocarbon shape remediation profile.
# R6a inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r6a_diesel_shape_route, R6a applies the full R4c balanced
# derivative pipeline byte-for-byte (using the same R4c seed source so y is
# identical and the R4c portion of the spectra RNG sequence stays aligned)
# and adds a small fixed mean-neutral hydrocarbon shape envelope on the
# 750-1550 nm support that is identically zero outside the support and
# zero-mean over the support.
# ---------------------------------------------------------------------------


def _r6a_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r6a_diesel_shape_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r6a_profile_is_opt_in_listed_and_records_non_oracle_diesel_shape_route() -> None:
    assert R6A_REMEDIATION_PROFILES == (
        "r6a_diesel_centered_hydrocarbon_shape_v1",
    )
    assert "r6a_diesel_centered_hydrocarbon_shape_v1" in ALL_REMEDIATION_PROFILES
    assert "r5a_diesel_absorbance_readout_v1" in ALL_REMEDIATION_PROFILES
    assert "r4c_diesel_balanced_derivative_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r6a_diesel_source(seed=640)),
        n_samples=24,
        random_seed=6464,
        remediation_profile="r6a_diesel_centered_hydrocarbon_shape_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r6a_diesel_centered_hydrocarbon_shape_v1"
    assert (
        audit["scope"]
        == "bench_only_r6a_diesel_centered_hydrocarbon_shape_remediation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    # Inherited R4c absorbance pipeline parameters byte-for-byte.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # R6a shape envelope metadata.
    assert params["shape_envelope_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert params["shape_envelope_widths_nm"] == [30.0, 34.0, 42.0, 46.0]
    assert params["shape_envelope_weights"] == [0.65, 1.00, 0.55, 0.72]
    assert params["shape_envelope_support_nm"] == [750.0, 1550.0]
    assert params["shape_envelope_absorbance_range"] == [0.00020, 0.00050]
    assert params["shape_envelope_zero_mean_on_support"] is True
    assert params["shape_envelope_application_stage"] == "after_r4c_output_clip"
    assert (
        params["output_clip_absorbance_applies_to"]
        == "r4c_pipeline_before_shape_envelope"
    )
    assert params["shape_envelope_final_output_clip_absorbance"] is None
    assert params["shape_envelope_final_min_absorbance"] == pytest.approx(
        float(remediated.X.min())
    )
    assert params["shape_envelope_final_max_absorbance"] == pytest.approx(
        float(remediated.X.max())
    )
    assert params["shape_envelope_absorbance_min"] >= 0.00020
    assert params["shape_envelope_absorbance_max"] <= 0.00050
    # R6a is not a readout-space transform.
    assert "readout_space_transform" not in params
    assert "fixed_envelope_absorbance_range" not in params
    assert "fixed_envelope_centers_nm" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["provenance_source"] == "exp09_dataset_token_diesel_shape_route"
    )
    # The R6a route is a shape route, not a readout-space route.
    assert params["diesel_shape_route_source"] == "exp09_dataset_token"
    assert params["diesel_shape_route_marker"] == "diesel"
    assert params["diesel_shape_route_non_oracle"] is True
    assert params["diesel_shape_route_real_stat_capture"] is False
    assert params["diesel_shape_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r6a_diesel_adds_only_zero_mean_shape_envelope_on_top_of_r4c() -> None:
    seed = 641
    rs = 6464
    n = 32
    record_r4c = canonicalize_prior_config(_r4c_diesel_source(seed=seed))
    # R6a uses the same fuel diesel source, but with the dedicated shape route.
    record_r6a = canonicalize_prior_config(_r6a_diesel_source(seed=seed))
    r4c_run = build_synthetic_dataset_run(
        record_r4c,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r6a_run = build_synthetic_dataset_run(
        record_r6a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r6a_diesel_centered_hydrocarbon_shape_v1",
    )

    # y is identical because R6a uses the same R4c seed source.
    np.testing.assert_array_equal(r6a_run.y, r4c_run.y)
    # X differs only by a small additive shape envelope.
    assert not np.allclose(r6a_run.X, r4c_run.X)
    delta = r6a_run.X - r4c_run.X
    params = r6a_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    amp_high = float(params["shape_envelope_absorbance_range"][1])
    assert float(np.abs(delta).max()) <= amp_high + 1e-12
    assert params["shape_envelope_application_stage"] == "after_r4c_output_clip"
    assert (
        params["output_clip_absorbance_applies_to"]
        == "r4c_pipeline_before_shape_envelope"
    )
    assert params["shape_envelope_final_output_clip_absorbance"] is None
    assert params["shape_envelope_final_min_absorbance"] == pytest.approx(
        float(r6a_run.X.min())
    )
    assert params["shape_envelope_final_max_absorbance"] == pytest.approx(
        float(r6a_run.X.max())
    )
    # Outside the 750-1550 nm support the addition is identically zero.
    wl = r6a_run.wavelengths
    support_mask = (wl >= 750.0) & (wl <= 1550.0)
    if (~support_mask).any():
        assert float(np.abs(delta[:, ~support_mask]).max()) == 0.0
    # On the support the per-row mean of the addition is zero by construction.
    assert support_mask.any()
    support_means = delta[:, support_mask].mean(axis=1)
    np.testing.assert_allclose(support_means, 0.0, atol=1e-12)


def test_r6a_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=642))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r6a_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r6a_diesel_centered_hydrocarbon_shape_v1",
    )
    assert r6a_unmarked.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r6a_unmarked.X, r3d_run.X)

    source = _r6a_diesel_source(seed=643)
    route = dict(cast("dict[str, object]", source["_r6a_diesel_shape_route"]))
    route["real_stat_capture"] = True
    source["_r6a_diesel_shape_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r6a_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r6a_diesel_centered_hydrocarbon_shape_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r6a_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r6a_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=644),
        _r2r_fruit_puree_source(seed=645),
        _r2o_beer_source(seed=646),
        _r2m_milk_source(seed=647),
        _r2n_manure21_source(seed=648),
        _soil_source(seed=649),
    ),
)
def test_r6a_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r6a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=6464,
        remediation_profile="r6a_diesel_centered_hydrocarbon_shape_v1",
    )

    assert r6a_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r6a_run.X, r3d_run.X)
    np.testing.assert_array_equal(r6a_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R7a DIESEL support-centered residual transfer remediation profile.
# R7a inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r7a_diesel_residual_route, R7a applies the R4a-like absorbance
# base (R3d micro-path continuum and detector offset, support-only CH overtone
# centers 1150/1210/1390/1460 nm at width 46 nm and gain 0.055-0.105, R4a
# damping windows and strength range, 975 nm short-continuum hump on the
# 750-1550 nm support) and adds a bounded support-centered residual transfer
# in [0.08, 0.18] before the final non-negative absorbance clip. The audit
# trail records the clip rule, clip fraction, and pre/post-clip min/max so
# any non-zero clip activity remains observable.
# ---------------------------------------------------------------------------


def _r7a_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r7a_diesel_residual_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r7a_profile_is_opt_in_listed_and_records_non_oracle_residual_route() -> None:
    assert R7A_REMEDIATION_PROFILES == (
        "r7a_diesel_support_centered_residual_transfer_v1",
    )
    assert (
        "r7a_diesel_support_centered_residual_transfer_v1"
        in ALL_REMEDIATION_PROFILES
    )
    assert "r6a_diesel_centered_hydrocarbon_shape_v1" in ALL_REMEDIATION_PROFILES
    assert "r4a_diesel_basis_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r7a_diesel_source(seed=720)),
        n_samples=24,
        random_seed=7474,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r7a_diesel_support_centered_residual_transfer_v1"
    assert (
        audit["scope"]
        == "bench_only_r7a_diesel_support_centered_residual_transfer_remediation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    # Inherited R4a-like absorbance base parameters.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 46.0
    assert params["ch_overtone_gain_range"] == [0.055, 0.105]
    assert params["damping_windows_nm"] == [
        [1180.0, 70.0, 1.0],
        [1425.0, 85.0, 1.0],
    ]
    assert params["damping_strength_range"] == [0.30, 0.50]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 90.0
    assert params["continuum_hump_amplitude_range"] == [0.00025, 0.00065]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # R7a residual transfer metadata.
    assert params["support_centered_residual_transfer_range"] == [0.08, 0.18]
    assert params["support_centered_residual_transfer_support_nm"] == [
        750.0,
        1550.0,
    ]
    assert params["support_centered_residual_transfer_source"] == (
        "fixed_synthetic_hydrocarbon_residual_transfer_prior"
    )
    assert params["support_centered_residual_transfer_centering"] == (
        "row_center_on_support_zero_outside"
    )
    assert params["support_centered_residual_transfer_application_stage"] == (
        "before_final_clip_after_r4a_base"
    )
    assert 0.08 <= params["support_centered_residual_transfer_min"] <= 0.18
    assert 0.08 <= params["support_centered_residual_transfer_max"] <= 0.18
    # Final clip metadata.
    assert params["final_clip_rule"] == "nonnegative_lower_bound_no_upper_bound"
    assert params["output_clip_absorbance"] == [0.0, None]
    assert params["final_min_absorbance_after_clip"] >= 0.0 - 1e-12
    assert params["final_min_absorbance_after_clip"] == pytest.approx(
        float(remediated.X.min())
    )
    assert params["final_max_absorbance_after_clip"] == pytest.approx(
        float(remediated.X.max())
    )
    assert params["final_max_absorbance_before_clip"] == pytest.approx(
        params["final_max_absorbance_after_clip"]
    )
    assert params["final_min_absorbance_before_clip"] <= params[
        "final_min_absorbance_after_clip"
    ]
    assert 0.0 <= params["final_clip_fraction"] <= 1.0
    # R7a is not a readout-space transform and not a shape envelope.
    assert "readout_space_transform" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "fixed_envelope_absorbance_range" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["provenance_source"]
        == "exp09_dataset_token_diesel_residual_transfer_route"
    )
    # The R7a route is a residual transfer route, not a readout / shape route.
    assert params["diesel_residual_transfer_route_source"] == "exp09_dataset_token"
    assert params["diesel_residual_transfer_route_marker"] == "diesel"
    assert params["diesel_residual_transfer_route_non_oracle"] is True
    assert params["diesel_residual_transfer_route_real_stat_capture"] is False
    assert params["diesel_residual_transfer_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    assert "diesel_shape_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"


def test_r7a_diesel_changes_morphology_vs_r4a_and_r3d_with_finite_nonneg_x() -> None:
    seed = 721
    rs = 7474
    n = 32
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=seed))
    record_r4a = canonicalize_prior_config(_r4a_diesel_source(seed=seed))
    record_r7a = canonicalize_prior_config(_r7a_diesel_source(seed=seed))

    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        record_r4a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r7a_run = build_synthetic_dataset_run(
        record_r7a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )

    # R7a changes X vs both R4a and R3d on explicit DIESEL routing.
    assert not np.allclose(r7a_run.X, r3d_run.X)
    assert not np.allclose(r7a_run.X, r4a_run.X)
    # X stays finite and non-negative after the final clip.
    assert np.isfinite(r7a_run.X).all()
    assert float(r7a_run.X.min()) >= 0.0
    # y is deterministic given the R7a seed source and stable across reruns.
    rerun = build_synthetic_dataset_run(
        record_r7a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )
    np.testing.assert_array_equal(r7a_run.y, rerun.y)
    np.testing.assert_array_equal(r7a_run.X, rerun.X)


def test_r7a_diesel_residual_support_and_clip_metadata_are_recorded() -> None:
    seed = 722
    rs = 7474
    n = 32
    record_r7a = canonicalize_prior_config(_r7a_diesel_source(seed=seed))
    r7a_run = build_synthetic_dataset_run(
        record_r7a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )

    params = r7a_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    wl = r7a_run.wavelengths
    support_low, support_high = params[
        "support_centered_residual_transfer_support_nm"
    ]
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()
    assert params["support_centered_residual_transfer_centering"] == (
        "row_center_on_support_zero_outside"
    )
    assert params["support_centered_residual_transfer_application_stage"] == (
        "before_final_clip_after_r4a_base"
    )

    # Final clip metadata must be reported and consistent with the final X.
    assert params["final_clip_fraction"] >= 0.0
    assert params["final_clip_fraction"] <= 1.0
    assert params["final_min_absorbance_after_clip"] >= 0.0
    assert params["final_min_absorbance_after_clip"] == pytest.approx(
        float(r7a_run.X.min())
    )
    assert params["final_max_absorbance_after_clip"] == pytest.approx(
        float(r7a_run.X.max())
    )
    if params["final_min_absorbance_before_clip"] >= 0.0:
        assert params["final_clip_fraction"] == pytest.approx(0.0)
    else:
        assert params["final_clip_fraction"] > 0.0


def test_r7a_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=723))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r7a_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )
    assert r7a_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r7a_unmarked.X, r3d_run.X)

    source = _r7a_diesel_source(seed=724)
    route = dict(cast("dict[str, object]", source["_r7a_diesel_residual_route"]))
    route["real_stat_capture"] = True
    source["_r7a_diesel_residual_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r7a_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r7a_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r7a_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=725),
        _r2r_fruit_puree_source(seed=726),
        _r2o_beer_source(seed=727),
        _r2m_milk_source(seed=728),
        _r2n_manure21_source(seed=729),
        _soil_source(seed=730),
    ),
)
def test_r7a_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r7a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=7474,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )

    assert r7a_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r7a_run.X, r3d_run.X)
    np.testing.assert_array_equal(r7a_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R8a DIESEL mean-preserving micro-path modulation remediation profile.
# R8a inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r8a_diesel_micro_path_route, R8a applies the R4a-like absorbance
# base (R3d micro-path continuum and detector offset, support-only CH overtone
# centers 1150/1210/1390/1460 nm at width 46 nm and gain 0.055-0.105, R4a
# damping windows and strength range, 975 nm short-continuum hump on the
# 750-1550 nm support) and, AFTER the standard R4a non-negative absorbance
# clip, multiplies the support of the base by exp(strength * shape) where
# ``shape`` is a robust synthetic-only normalization of the residual on the
# support clipped to [-1, 1]. The support row mean is exactly preserved by a
# multiplicative renormalization. Outside the support the readout is
# identical to the R4a base. A guard non-negative clip is recorded for audit
# but is expected to be a no-op.
# ---------------------------------------------------------------------------


def _r8a_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r8a_diesel_micro_path_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r8a_profile_is_opt_in_listed_and_records_non_oracle_micro_path_route() -> None:
    assert R8A_REMEDIATION_PROFILES == (
        "r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    assert (
        "r8a_diesel_mean_preserving_micro_path_modulation_v1"
        in ALL_REMEDIATION_PROFILES
    )
    assert (
        "r7a_diesel_support_centered_residual_transfer_v1"
        in ALL_REMEDIATION_PROFILES
    )
    assert "r4a_diesel_basis_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r8a_diesel_source(seed=820)),
        n_samples=24,
        random_seed=8484,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert (
        audit["profile"]
        == "r8a_diesel_mean_preserving_micro_path_modulation_v1"
    )
    assert (
        audit["scope"]
        == "bench_only_r8a_diesel_mean_preserving_micro_path_modulation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    params = audit["transform_params"]
    # Inherited R4a-like absorbance base parameters.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 46.0
    assert params["ch_overtone_gain_range"] == [0.055, 0.105]
    assert params["damping_windows_nm"] == [
        [1180.0, 70.0, 1.0],
        [1425.0, 85.0, 1.0],
    ]
    assert params["damping_strength_range"] == [0.30, 0.50]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 90.0
    assert params["continuum_hump_amplitude_range"] == [0.00025, 0.00065]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # R8a modulation metadata (mean-preserving multiplicative micro-path).
    assert params["support_centered_micro_path_modulation_strength_range"] == [
        0.10,
        0.30,
    ]
    assert params["support_centered_micro_path_modulation_support_nm"] == [
        750.0,
        1550.0,
    ]
    assert (
        params["support_centered_micro_path_modulation_normalization"]
        == "p95_abs"
    )
    assert (
        params["support_centered_micro_path_modulation_normalization_epsilon"]
        > 0.0
    )
    assert params["support_centered_micro_path_modulation_shape_clip"] == [
        -1.0,
        1.0,
    ]
    assert (
        params["support_centered_micro_path_modulation_centering"]
        == "row_center_on_support_zero_outside"
    )
    assert (
        params["support_centered_micro_path_modulation_application_stage"]
        == "after_base_nonnegative_clip"
    )
    assert (
        params["support_centered_micro_path_modulation_normalization_source"]
        == "synthetic_internal_residual_only"
    )
    assert (
        params["support_centered_micro_path_modulation_source"]
        == "synthetic_internal_residual_only"
    )
    assert (
        0.10
        <= params["support_centered_micro_path_modulation_strength_min"]
        <= 0.30
    )
    assert (
        0.10
        <= params["support_centered_micro_path_modulation_strength_max"]
        <= 0.30
    )
    assert (
        -1.0
        <= params["support_centered_micro_path_modulation_shape_min"]
        <= 1.0
    )
    assert (
        -1.0
        <= params["support_centered_micro_path_modulation_shape_max"]
        <= 1.0
    )
    # Modulation is positive (exp of bounded shape).
    assert (
        params["support_centered_micro_path_modulation_modulation_min"] > 0.0
    )
    assert (
        params["support_centered_micro_path_modulation_modulation_max"]
        >= params["support_centered_micro_path_modulation_modulation_min"]
    )
    # Mean preservation: support row mean delta is essentially zero (<= float
    # round-off; reasonable tolerance keeps this robust to numerics).
    assert (
        params[
            "support_centered_micro_path_modulation_support_mean_abs_delta_max"
        ]
        <= 1.0e-9
    )
    # Guard clip is expected to be a no-op since modulation is positive on a
    # non-negative base.
    assert (
        params[
            "support_centered_micro_path_modulation_guard_clip_fraction"
        ]
        == pytest.approx(0.0)
    )
    assert (
        params[
            "support_centered_micro_path_modulation_guard_clip_rule"
        ]
        == "nonnegative_lower_bound_no_upper_bound"
    )
    assert (
        params[
            "support_centered_micro_path_modulation_min_after_guard_clip"
        ]
        >= 0.0
    )
    assert (
        params[
            "support_centered_micro_path_modulation_max_after_guard_clip"
        ]
        == pytest.approx(float(remediated.X.max()))
    )
    assert (
        params[
            "support_centered_micro_path_modulation_n_support_bins"
        ]
        > 0
    )
    # R8a is not a readout-space transform, not a shape envelope, and not the
    # raw R7a residual transfer addition: those keys must not leak.
    assert "readout_space_transform" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "fixed_envelope_absorbance_range" not in params
    assert "support_centered_residual_transfer_range" not in params
    assert "support_centered_residual_transfer_support_nm" not in params
    assert "support_centered_residual_transfer_centering" not in params
    assert "support_centered_residual_transfer_application_stage" not in params
    assert "diesel_residual_transfer_route_source" not in params
    assert "diesel_residual_transfer_route_marker" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["provenance_source"]
        == "exp09_dataset_token_diesel_micro_path_modulation_route"
    )
    # R8a route fields use the dedicated micro-path modulation route key.
    assert (
        params["diesel_micro_path_modulation_route_source"]
        == "exp09_dataset_token"
    )
    assert (
        params["diesel_micro_path_modulation_route_marker"] == "diesel"
    )
    assert params["diesel_micro_path_modulation_route_non_oracle"] is True
    assert (
        params["diesel_micro_path_modulation_route_real_stat_capture"]
        is False
    )
    assert (
        params["diesel_micro_path_modulation_route_thresholds_modified"]
        is False
    )
    assert "diesel_readout_route_source" not in params
    assert "diesel_shape_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    # X is finite and non-negative.
    assert np.isfinite(remediated.X).all()
    assert float(remediated.X.min()) >= 0.0


def test_r8a_diesel_changes_morphology_vs_r3d_r4a_r7a_with_finite_nonneg_x() -> None:
    seed = 821
    rs = 8484
    n = 32
    record_r3d = canonicalize_prior_config(_r3d_diesel_source(seed=seed))
    record_r4a = canonicalize_prior_config(_r4a_diesel_source(seed=seed))
    record_r7a = canonicalize_prior_config(
        {
            **_fuel_diesel_source(seed=seed),
            "_r7a_diesel_residual_route": {
                "enabled": True,
                "route_marker": "diesel",
                "source": "exp09_dataset_token",
                "non_oracle": True,
                "no_target_or_label": True,
                "real_stat_capture": False,
                "thresholds_modified": False,
            },
        }
    )
    record_r8a = canonicalize_prior_config(_r8a_diesel_source(seed=seed))

    r3d_run = build_synthetic_dataset_run(
        record_r3d,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        record_r4a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r7a_run = build_synthetic_dataset_run(
        record_r7a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r7a_diesel_support_centered_residual_transfer_v1",
    )
    r8a_run = build_synthetic_dataset_run(
        record_r8a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )

    # R8a changes X vs R3d, R4a, and R7a on explicit DIESEL routing.
    assert not np.allclose(r8a_run.X, r3d_run.X)
    assert not np.allclose(r8a_run.X, r4a_run.X)
    assert not np.allclose(r8a_run.X, r7a_run.X)
    # X stays finite and non-negative after the modulation + guard clip.
    assert np.isfinite(r8a_run.X).all()
    assert float(r8a_run.X.min()) >= 0.0
    # y is deterministic given the R8a seed source and stable across reruns.
    rerun = build_synthetic_dataset_run(
        record_r8a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    np.testing.assert_array_equal(r8a_run.y, rerun.y)
    np.testing.assert_array_equal(r8a_run.X, rerun.X)


def test_r8a_diesel_support_mean_preserved_vs_r4a_base_within_tolerance() -> None:
    seed = 822
    rs = 8484
    n = 32
    record_r4a = canonicalize_prior_config(_r4a_diesel_source(seed=seed))
    record_r8a = canonicalize_prior_config(_r8a_diesel_source(seed=seed))

    r4a_run = build_synthetic_dataset_run(
        record_r4a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r8a_run = build_synthetic_dataset_run(
        record_r8a,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    params = r8a_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params[
        "support_centered_micro_path_modulation_support_nm"
    ]
    wl = r8a_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    r4a_support_mean = r4a_run.X[:, support_mask].mean(axis=1)
    r8a_support_mean = r8a_run.X[:, support_mask].mean(axis=1)
    np.testing.assert_allclose(
        r8a_support_mean,
        r4a_support_mean,
        rtol=1e-6,
        atol=1e-9,
    )
    # Outside the support, the readout is identical to R4a.
    np.testing.assert_allclose(
        r8a_run.X[:, ~support_mask],
        r4a_run.X[:, ~support_mask],
    )
    # R8a actually changed the support shape (so the mean preservation is a
    # real renormalization, not a trivial identity).
    assert not np.allclose(
        r8a_run.X[:, support_mask],
        r4a_run.X[:, support_mask],
    )


def test_r8a_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=823))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r8a_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    assert r8a_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r8a_unmarked.X, r3d_run.X)

    source = _r8a_diesel_source(seed=824)
    route = dict(cast("dict[str, object]", source["_r8a_diesel_micro_path_route"]))
    route["real_stat_capture"] = True
    source["_r8a_diesel_micro_path_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r8a_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r8a_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    assert (
        r8a_non_compliant.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r3d_diesel_matrix_v1"
    )
    np.testing.assert_array_equal(r8a_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=825),
        _r2r_fruit_puree_source(seed=826),
        _r2o_beer_source(seed=827),
        _r2m_milk_source(seed=828),
        _r2n_manure21_source(seed=829),
        _soil_source(seed=830),
    ),
)
def test_r8a_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r8a_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8484,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )

    assert r8a_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r8a_run.X, r3d_run.X)
    np.testing.assert_array_equal(r8a_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R8b DIESEL R4c-base mean-preserving micro-path modulation remediation profile.
# R8b inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r8b_diesel_micro_path_route, R8b applies the R4c balanced-derivative
# absorbance base and then applies the R8a support-mean-preserving modulation.
# ---------------------------------------------------------------------------


def _r8b_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r8b_diesel_micro_path_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r8b_profile_is_opt_in_listed_and_records_non_oracle_micro_path_route() -> None:
    assert R8B_REMEDIATION_PROFILES == (
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    )
    assert (
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        in ALL_REMEDIATION_PROFILES
    )
    assert "r4c_diesel_balanced_derivative_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r8b_diesel_source(seed=920)),
        n_samples=24,
        random_seed=8585,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert (
        audit["profile"]
        == "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
    )
    assert (
        audit["scope"]
        == "bench_only_r8b_diesel_r4c_base_mean_preserving_micro_path_modulation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    params = audit["transform_params"]
    # R4c balanced-derivative base, not the R8a/R4a base.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    assert params["support_centered_micro_path_modulation_strength_range"] == [
        0.10,
        0.30,
    ]
    assert params["support_centered_micro_path_modulation_support_nm"] == [
        750.0,
        1550.0,
    ]
    assert params["support_centered_micro_path_modulation_normalization"] == "p95_abs"
    assert params["support_centered_micro_path_modulation_shape_clip"] == [
        -1.0,
        1.0,
    ]
    assert (
        params["support_centered_micro_path_modulation_application_stage"]
        == "after_base_nonnegative_clip"
    )
    assert (
        params["support_centered_micro_path_modulation_normalization_source"]
        == "synthetic_internal_residual_only"
    )
    assert (
        params[
            "support_centered_micro_path_modulation_support_mean_abs_delta_max"
        ]
        <= 1.0e-9
    )
    assert (
        params["support_centered_micro_path_modulation_guard_clip_fraction"]
        == pytest.approx(0.0)
    )
    assert (
        params["support_centered_micro_path_modulation_guard_clip_rule"]
        == "nonnegative_lower_bound_no_upper_bound"
    )
    assert (
        params["support_centered_micro_path_modulation_min_after_guard_clip"]
        >= 0.0
    )
    assert (
        params["support_centered_micro_path_modulation_max_after_guard_clip"]
        == pytest.approx(float(remediated.X.max()))
    )
    assert "readout_space_transform" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "support_centered_residual_transfer_range" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert (
        params["provenance_source"]
        == "exp09_dataset_token_diesel_r8b_micro_path_modulation_route"
    )
    assert params["diesel_micro_path_modulation_route_source"] == "exp09_dataset_token"
    assert params["diesel_micro_path_modulation_route_marker"] == "diesel"
    assert params["diesel_micro_path_modulation_route_non_oracle"] is True
    assert params["diesel_micro_path_modulation_route_real_stat_capture"] is False
    assert params["diesel_micro_path_modulation_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert np.isfinite(remediated.X).all()
    assert float(remediated.X.min()) >= 0.0


def test_r8b_diesel_differs_from_r4c_and_r8a_with_finite_nonnegative_x() -> None:
    seed = 921
    rs = 8585
    n = 32
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r8a_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r8a_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r8a_diesel_mean_preserving_micro_path_modulation_v1",
    )
    r8b_record = canonicalize_prior_config(_r8b_diesel_source(seed=seed))
    r8b_run = build_synthetic_dataset_run(
        r8b_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )

    assert not np.allclose(r8b_run.X, r4c_run.X)
    assert not np.allclose(r8b_run.X, r8a_run.X)
    assert np.isfinite(r8b_run.X).all()
    assert float(r8b_run.X.min()) >= 0.0
    rerun = build_synthetic_dataset_run(
        r8b_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )
    np.testing.assert_array_equal(r8b_run.y, rerun.y)
    np.testing.assert_array_equal(r8b_run.X, rerun.X)


def test_r8b_diesel_support_mean_preserved_vs_r4c_base_within_tolerance() -> None:
    seed = 922
    rs = 8585
    n = 32
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r8b_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r8b_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )
    params = r8b_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params[
        "support_centered_micro_path_modulation_support_nm"
    ]
    wl = r8b_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    np.testing.assert_allclose(
        r8b_run.X[:, support_mask].mean(axis=1),
        r4c_run.X[:, support_mask].mean(axis=1),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(r8b_run.X[:, ~support_mask], r4c_run.X[:, ~support_mask])
    assert not np.allclose(r8b_run.X[:, support_mask], r4c_run.X[:, support_mask])


def test_r8b_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=923))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8585,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r8b_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8585,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )
    assert r8b_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r8b_unmarked.X, r3d_run.X)

    source = _r8b_diesel_source(seed=924)
    route = dict(cast("dict[str, object]", source["_r8b_diesel_micro_path_route"]))
    route["thresholds_modified"] = True
    source["_r8b_diesel_micro_path_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r8b_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8585,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8585,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r8b_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    assert (
        r8b_non_compliant.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r3d_diesel_matrix_v1"
    )
    np.testing.assert_array_equal(r8b_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=925),
        _r2r_fruit_puree_source(seed=926),
        _r2o_beer_source(seed=927),
        _r2m_milk_source(seed=928),
        _r2n_manure21_source(seed=929),
        _soil_source(seed=930),
    ),
)
def test_r8b_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8585,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r8b_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8585,
        remediation_profile=(
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ),
    )

    assert r8b_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r8b_run.X, r3d_run.X)
    np.testing.assert_array_equal(r8b_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9b DIESEL support-level mechanistic intercept remediation profile.
# R9b inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r9b_diesel_support_intercept_route, R9b applies the R4c balanced-
# derivative absorbance base byte-for-byte and then adds a single small fixed
# mechanistic absorbance intercept on the 750-1550 nm support after the R4c
# non-negative output clip; outside the support the readout is byte-identical
# to the R4c base. The intercept value is a pre-declared mechanistic constant,
# NOT chosen from any R9a or R9b mean-shift residual delta.
# ---------------------------------------------------------------------------


def _r9b_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp11_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9b_diesel_support_intercept_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp11_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9b_intercept_constant_is_pre_declared_and_not_audit_delta() -> None:
    """The R9b intercept must be a pre-declared mechanistic constant pulled
    from a documented module-level binding, not derived from any R9a/R9b
    mean-shift residual or audit metric.
    """
    assert R9B_REMEDIATION_PROFILES == ("r9b_diesel_support_intercept_v1",)
    assert _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_ABSORBANCE == 0.002
    # Bounded to the small mechanistic blank-cell/detector reference range
    # (1e-3 to 5e-3 absorbance units after blank referencing); strictly
    # positive so the R4c non-negative output clip is not triggered, and
    # well below the R4a basis profile's absorbance scale so it cannot
    # smooth derivatives the way R4a does.
    assert 1.0e-3 <= _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_ABSORBANCE <= 5.0e-3
    assert _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SUPPORT_NM == (750.0, 1550.0)
    assert _R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SOURCE == (
        "fixed_blank_cell_detector_support_level_intercept_prior"
    )


def test_r9b_profile_is_opt_in_listed_and_records_non_oracle_intercept_route() -> None:
    assert R9B_REMEDIATION_PROFILES == ("r9b_diesel_support_intercept_v1",)
    assert "r9b_diesel_support_intercept_v1" in ALL_REMEDIATION_PROFILES
    assert "r4c_diesel_balanced_derivative_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9b_diesel_source(seed=940)),
        n_samples=24,
        random_seed=8686,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9b_diesel_support_intercept_v1"
    assert audit["scope"] == "bench_only_r9b_diesel_support_intercept_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    params = audit["transform_params"]
    # R9b inherits the full R4c balanced-derivative base byte-for-byte.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert 1720.0 not in params["ch_overtone_centers_nm"]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    # R9b-specific transform_params.
    assert params["support_intercept_absorbance"] == pytest.approx(0.002)
    assert params["support_intercept_support_nm"] == [750.0, 1550.0]
    assert params["support_intercept_n_support_bins"] >= 1
    assert params["support_intercept_application_stage"] == "after_r4c_output_clip"
    assert params["support_intercept_off_support_unchanged"] is True
    assert params["support_intercept_value_origin"] == (
        "pre_declared_mechanistic_constant_not_audit_delta"
    )
    assert params["support_intercept_source"] == (
        "fixed_blank_cell_detector_support_level_intercept_prior"
    )
    # Guard clip is recorded but must be a no-op (R4c output is non-negative
    # and intercept is positive).
    assert params["support_intercept_guard_clip_fraction"] == pytest.approx(0.0)
    assert params["support_intercept_guard_clip_rule"] == (
        "nonnegative_lower_bound_no_upper_bound"
    )
    assert params["support_intercept_min_after_guard_clip"] >= 0.0
    # Support mean must move by exactly the intercept, within fp tolerance.
    assert params["support_intercept_support_mean_delta_min"] == pytest.approx(
        0.002, abs=1e-12
    )
    assert params["support_intercept_support_mean_delta_max"] == pytest.approx(
        0.002, abs=1e-12
    )
    # R9b must NOT be a readout-space transform, shape envelope, residual
    # transfer, or modulation profile.
    assert "readout_space_transform" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "support_centered_residual_transfer_range" not in params
    assert "support_centered_micro_path_modulation_strength_range" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == (
        "exp11_dataset_token_diesel_r9b_support_intercept_route"
    )
    assert params["diesel_support_intercept_route_source"] == "exp11_dataset_token"
    assert params["diesel_support_intercept_route_marker"] == "diesel"
    assert params["diesel_support_intercept_route_non_oracle"] is True
    assert params["diesel_support_intercept_route_real_stat_capture"] is False
    assert params["diesel_support_intercept_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    assert "diesel_micro_path_modulation_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert np.isfinite(remediated.X).all()
    assert float(remediated.X.min()) >= 0.0


def test_r9b_diesel_differs_from_r4c_only_on_support_with_finite_x() -> None:
    seed = 941
    rs = 8686
    n = 32
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r9b_record = canonicalize_prior_config(_r9b_diesel_source(seed=seed))
    r9b_run = build_synthetic_dataset_run(
        r9b_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )

    assert not np.allclose(r9b_run.X, r4c_run.X)
    assert np.isfinite(r9b_run.X).all()
    assert float(r9b_run.X.min()) >= 0.0
    # Determinism: same seed/source/profile must reproduce X and y exactly.
    rerun = build_synthetic_dataset_run(
        r9b_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )
    np.testing.assert_array_equal(r9b_run.y, rerun.y)
    np.testing.assert_array_equal(r9b_run.X, rerun.X)


def test_r9b_diesel_support_lowers_absolute_support_mean_versus_real_target_proxy_via_added_intercept() -> None:
    """R9b is the R4c base plus a small positive support-only intercept; the
    mean absorbance over the support must rise by exactly the intercept and
    derivative structure inside the support must be unchanged.
    """
    seed = 942
    rs = 8686
    n = 32
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r9b_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9b_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )
    params = r9b_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    intercept = float(params["support_intercept_absorbance"])
    support_low, support_high = params["support_intercept_support_nm"]
    wl = r9b_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    # Off-support cells are byte-identical to the R4c base.
    np.testing.assert_array_equal(
        r9b_run.X[:, ~support_mask], r4c_run.X[:, ~support_mask]
    )
    # Support cells differ from R4c by exactly the intercept (the guard clip
    # is a no-op, so the support delta is the deterministic +intercept add).
    np.testing.assert_allclose(
        r9b_run.X[:, support_mask] - r4c_run.X[:, support_mask],
        intercept,
        rtol=0.0,
        atol=1e-12,
    )
    # Per-row support mean must rise by exactly the intercept.
    np.testing.assert_allclose(
        r9b_run.X[:, support_mask].mean(axis=1)
        - r4c_run.X[:, support_mask].mean(axis=1),
        intercept,
        rtol=0.0,
        atol=1e-12,
    )
    # Derivative structure inside the support is unchanged: a constant
    # additive on contiguous support cells has zero effect on np.diff inside
    # that block. Compare diffs over the strict support interior (drop the
    # first support index so the np.diff index points to in-support
    # neighbours).
    support_indices = np.flatnonzero(support_mask)
    interior = support_indices[1:]
    np.testing.assert_allclose(
        np.diff(r9b_run.X, axis=1)[:, interior - 1],
        np.diff(r4c_run.X, axis=1)[:, interior - 1],
        rtol=0.0,
        atol=1e-12,
    )
    # No clip event: the global min stays non-negative and the support
    # intercept did not push any absorbance below zero.
    assert float(r9b_run.X.min()) >= 0.0


def test_r9b_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=943))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9b_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )
    assert r9b_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9b_unmarked.X, r3d_run.X)

    source = _r9b_diesel_source(seed=944)
    route = dict(
        cast("dict[str, object]", source["_r9b_diesel_support_intercept_route"])
    )
    route["thresholds_modified"] = True
    source["_r9b_diesel_support_intercept_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9b_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9b_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    assert (
        r9b_non_compliant.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r3d_diesel_matrix_v1"
    )
    np.testing.assert_array_equal(r9b_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=945),
        _r2r_fruit_puree_source(seed=946),
        _r2o_beer_source(seed=947),
        _r2m_milk_source(seed=948),
        _r2n_manure21_source(seed=949),
        _soil_source(seed=950),
    ),
)
def test_r9b_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9b_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8686,
        remediation_profile="r9b_diesel_support_intercept_v1",
    )

    assert r9b_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r9b_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9b_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9c DIESEL support-level shape mechanism remediation profile.
# R9c inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r9c_diesel_support_shape_route, R9c applies the R3d micro-path /
# baseline / CH-overtone / clip pipeline byte-for-byte and then adds, on the
# 750-1550 nm support only and AFTER the R3d non-negative output clip, a fixed
# mechanistic shape modulation (Gaussian CH band sums plus support-localized
# damping windows plus a small support hump). Constants are pre-declared
# general liquid-hydrocarbon NIR prior values, NOT chosen from any R9a/R9b
# mean-shift residual delta.
# ---------------------------------------------------------------------------


def _r9c_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp12_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9c_diesel_support_shape_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp12_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9c_constants_are_pre_declared_general_liquid_hydrocarbon_prior() -> None:
    """The R9c shape constants must come from a general liquid-hydrocarbon NIR
    prior, not from any R9a/R9b residual delta or audit metric."""
    assert R9C_REMEDIATION_PROFILES == (
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    assert _R9C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM == (
        1150.0,
        1210.0,
        1390.0,
        1460.0,
    )
    assert 1720.0 not in _R9C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM
    assert _R9C_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM == (40.0, 40.0, 44.0, 48.0)
    assert _R9C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE == (0.075, 0.135)
    assert _R9C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM == (
        (1180.0, 56.0, 0.55),
        (1425.0, 72.0, 0.85),
    )
    assert _R9C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE == (0.14, 0.28)
    assert _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM == 975.0
    assert _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM == 84.0
    assert _R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE == (
        0.00018,
        0.00048,
    )
    assert _R9C_PETROCHEM_FUELS_SUPPORT_NM == (750.0, 1550.0)
    assert _R9C_PETROCHEM_FUELS_CONSTANTS_SOURCE == (
        "predeclared_general_liquid_hydrocarbon_nir_prior"
    )


def test_r9c_profile_is_opt_in_listed_and_records_non_oracle_support_shape_route() -> None:
    assert R9C_REMEDIATION_PROFILES == (
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    assert "r9c_diesel_selective_ch_bandwidth_damping_v1" in ALL_REMEDIATION_PROFILES
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9c_diesel_source(seed=1140)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9c_diesel_selective_ch_bandwidth_damping_v1"
    assert audit["scope"] == (
        "bench_only_r9c_diesel_selective_ch_bandwidth_damping_remediation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    params = audit["transform_params"]
    # R9c inherits the R3d micro-path / baseline / clip pipeline byte-for-byte
    # (NOT R4c). The R3d rule retains the 5-band CH overtone family with a
    # single scalar width, so the legacy single-width API is unchanged for the
    # R3d portion of the rule.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_width_nm"] == 34.0
    # R9c-specific transform_params (the new per-band widths sit on the R9c
    # support_shape_widths_nm key, not on the legacy ch_overtone_width_nm key).
    assert params["support_shape_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert params["support_shape_widths_nm"] == [40.0, 40.0, 44.0, 48.0]
    assert params["support_shape_gain_range"] == [0.075, 0.135]
    assert params["support_shape_damping_windows_nm"] == [
        [1180.0, 56.0, 0.55],
        [1425.0, 72.0, 0.85],
    ]
    assert params["support_shape_damping_strength_range"] == [0.14, 0.28]
    assert params["support_shape_hump_center_nm"] == 975.0
    assert params["support_shape_hump_width_nm"] == 84.0
    assert params["support_shape_hump_amplitude_range"] == [0.00018, 0.00048]
    assert params["support_shape_support_nm"] == [750.0, 1550.0]
    assert params["support_shape_n_support_bins"] >= 1
    assert params["support_shape_application_stage"] == "after_r3d_output_clip"
    assert params["support_shape_off_support_unchanged"] is True
    assert params["support_shape_value_origin"] == (
        "predeclared_mechanistic_constants_not_audit_delta"
    )
    assert params["support_shape_constants_source"] == (
        "predeclared_general_liquid_hydrocarbon_nir_prior"
    )
    assert params["support_shape_mechanism"] == (
        "selective_ch_bandwidth_damping_support_shape_only"
    )
    # Audit flags: every R9c forbidden mechanism must be explicitly false.
    for key, expected in (
        ("support_shape_calibration", False),
        ("support_shape_uses_real_stats", False),
        ("support_shape_uses_pca", False),
        ("support_shape_captures_noise", False),
        ("support_shape_uses_labels", False),
        ("support_shape_uses_targets", False),
        ("support_shape_uses_splits", False),
        ("support_shape_uses_ml", False),
        ("support_shape_uses_dl", False),
        ("support_shape_mutates_thresholds", False),
        ("support_shape_mutates_metrics", False),
        ("support_shape_adds_offset", False),
        ("support_shape_only", True),
    ):
        assert params[key] is expected, f"audit flag {key!r} = {params[key]!r}"
    # Guard clip is recorded but expected to be a no-op (positive additive
    # shape on top of a non-negative R3d base).
    assert params["support_shape_guard_clip_fraction"] == pytest.approx(0.0)
    assert params["support_shape_guard_clip_rule"] == (
        "nonnegative_lower_bound_no_upper_bound"
    )
    assert params["support_shape_min_after_guard_clip"] >= 0.0
    # R9c must NOT be a readout-space transform, support intercept, shape
    # envelope, residual transfer, or modulation profile.
    assert "readout_space_transform" not in params
    assert "support_intercept_absorbance" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "support_centered_residual_transfer_range" not in params
    assert "support_centered_micro_path_modulation_strength_range" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == (
        "exp12_dataset_token_diesel_r9c_support_shape_route"
    )
    assert params["diesel_support_shape_route_source"] == "exp12_dataset_token"
    assert params["diesel_support_shape_route_marker"] == "diesel"
    assert params["diesel_support_shape_route_non_oracle"] is True
    assert params["diesel_support_shape_route_real_stat_capture"] is False
    assert params["diesel_support_shape_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    assert "diesel_support_intercept_route_source" not in params
    assert "diesel_micro_path_modulation_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert np.isfinite(remediated.X).all()
    assert float(remediated.X.min()) >= 0.0


def test_r9c_diesel_differs_from_r4c_and_r4a_and_is_deterministic() -> None:
    seed = 1141
    rs = 8787
    n = 32
    r4a_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4a_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r9c_record = canonicalize_prior_config(_r9c_diesel_source(seed=seed))
    r9c_run = build_synthetic_dataset_run(
        r9c_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )

    # Compliant DIESEL: R9c differs from both R4c and R4a.
    assert np.any(r9c_run.X != r4c_run.X)
    assert np.any(r9c_run.X != r4a_run.X)
    assert np.isfinite(r9c_run.X).all()
    assert float(r9c_run.X.min()) >= 0.0
    # Determinism: same seed/source/profile reproduces X and y exactly.
    rerun = build_synthetic_dataset_run(
        r9c_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    np.testing.assert_array_equal(r9c_run.y, rerun.y)
    np.testing.assert_array_equal(r9c_run.X, rerun.X)


def test_r9c_diesel_off_support_byte_identical_to_r3d_and_support_delta_is_not_a_constant_offset() -> None:
    """On compliant DIESEL rows, R9c must leave off-support cells byte-equal
    to the R3d base, and the on-support delta must be a wavelength-dependent
    shape (non-zero std along the wavelength axis), not a scalar offset.
    """
    seed = 1142
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    params = r9c_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params["support_shape_support_nm"]
    wl = r9c_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    # Off-support cells are byte-identical to the R3d base.
    np.testing.assert_array_equal(
        r9c_run.X[:, ~support_mask], r3d_run.X[:, ~support_mask]
    )
    # On-support cells differ from R3d.
    assert np.any(r9c_run.X[:, support_mask] != r3d_run.X[:, support_mask])
    # The support delta must NOT be a constant scalar offset: it must vary
    # across wavelengths (non-zero std along the wavelength axis on the
    # support).
    delta = r9c_run.X[:, support_mask] - r3d_run.X[:, support_mask]
    assert delta.std() > 0.0
    per_row_wavelength_std = delta.std(axis=1)
    assert np.all(per_row_wavelength_std > 0.0)
    # Mean per-row wavelength std must be at least an order of magnitude
    # above floating-point noise: the R9c shape is not a numerical artefact.
    assert float(per_row_wavelength_std.mean()) > 1.0e-6
    # No clip event: the global min stays non-negative.
    assert float(r9c_run.X.min()) >= 0.0


def test_r9c_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=1143))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9c_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    assert r9c_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9c_unmarked.X, r3d_run.X)

    source = _r9c_diesel_source(seed=1144)
    route = dict(
        cast("dict[str, object]", source["_r9c_diesel_support_shape_route"])
    )
    route["thresholds_modified"] = True
    source["_r9c_diesel_support_shape_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9c_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9c_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    assert (
        r9c_non_compliant.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r3d_diesel_matrix_v1"
    )
    np.testing.assert_array_equal(r9c_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1145),
        _r2r_fruit_puree_source(seed=1146),
        _r2o_beer_source(seed=1147),
        _r2m_milk_source(seed=1148),
        _r2n_manure21_source(seed=1149),
        _soil_source(seed=1150),
    ),
)
def test_r9c_non_diesel_draws_are_identical_to_r3d(source: dict[str, object]) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9c_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
    )

    assert r9c_run.metadata["r2c_mechanistic_remediation"] == r3d_run.metadata[
        "r2c_mechanistic_remediation"
    ]
    np.testing.assert_array_equal(r9c_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9c_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9d DIESEL energy-normalized mean-neutral support redistribution profile.
# R9d inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r9d_diesel_support_redistribution_route, R9d applies the R3d
# micro-path / baseline / CH-overtone / clip pipeline byte-for-byte and then,
# AFTER the R3d non-negative output clip and ON the 750-1550 nm DIESEL real
# basis support window only, applies a multiplicative ``exp(strength * shape)``
# modulation built from PRE-DECLARED MECHANISTIC CONSTANTS (Gaussian CH overtone
# bands at 1150/1210/1390/1460 nm with per-band widths 40/40/44/48 nm, mean-
# subtracted on the support, max-abs normalized, clipped to [-1, 1]). After the
# multiplicative factor, each row support is multiplicatively renormalized so
# that the post-redistribution support mean equals the pre-redistribution
# support mean exactly within numerical tolerance. Off-support cells are byte-
# identical to the R3d base by construction.
# ---------------------------------------------------------------------------


def _r9d_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp13_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9d_diesel_support_redistribution_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp13_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9d_constants_are_pre_declared_general_liquid_hydrocarbon_prior() -> None:
    """The R9d redistribution constants must come from a general liquid-
    hydrocarbon NIR energy redistribution prior, not from any R9a/R9b/R9c
    mean-shift residual or audit metric."""
    assert R9D_REMEDIATION_PROFILES == (
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    assert _R9D_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM == (
        1150.0,
        1210.0,
        1390.0,
        1460.0,
    )
    assert 1720.0 not in _R9D_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM
    assert _R9D_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM == (40.0, 40.0, 44.0, 48.0)
    assert _R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE == (
        0.035,
        0.095,
    )
    assert _R9D_PETROCHEM_FUELS_SHAPE_CLIP == (-1.0, 1.0)
    assert _R9D_PETROCHEM_FUELS_RENORM_EPSILON == 1e-12
    assert _R9D_PETROCHEM_FUELS_SUPPORT_NM == (750.0, 1550.0)
    assert _R9D_PETROCHEM_FUELS_CONSTANTS_SOURCE == (
        "predeclared_general_liquid_hydrocarbon_nir_energy_redistribution_prior"
    )


def test_r9d_profile_is_opt_in_listed_and_records_non_oracle_redistribution_route() -> None:
    assert R9D_REMEDIATION_PROFILES == (
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    assert (
        "r9d_diesel_energy_normalized_support_redistribution_v1"
        in ALL_REMEDIATION_PROFILES
    )
    assert "r3d_diesel_matrix_v1" in ALL_REMEDIATION_PROFILES

    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9d_diesel_source(seed=1240)),
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == (
        "r9d_diesel_energy_normalized_support_redistribution_v1"
    )
    assert audit["scope"] == (
        "bench_only_r9d_diesel_energy_normalized_support_redistribution_remediation"
    )
    assert audit["domain_key"] == "petrochem_fuels"
    params = audit["transform_params"]
    # R9d inherits the R3d micro-path / baseline / clip pipeline byte-for-byte
    # (NOT R4c). The R3d rule retains the 5-band CH overtone family with a
    # single scalar width, so the legacy single-width API is unchanged for the
    # R3d portion of the rule.
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == (
        "micro_path_fuel_ch_overtone_contrast_readout"
    )
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_width_nm"] == 34.0
    # R9d-specific transform_params (the new redistribution keys are
    # support_redistribution_*, distinct from R9c support_shape_* and from the
    # legacy ch_overtone_width_nm key).
    assert params["support_redistribution_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
    ]
    assert params["support_redistribution_widths_nm"] == [40.0, 40.0, 44.0, 48.0]
    assert params["support_redistribution_log_strength_range"] == [0.035, 0.095]
    assert params["support_redistribution_shape_clip"] == [-1.0, 1.0]
    assert params["support_redistribution_renorm_epsilon"] == 1e-12
    assert params["support_redistribution_support_nm"] == [750.0, 1550.0]
    assert params["support_redistribution_n_support_bins"] >= 1
    assert params["support_redistribution_application_stage"] == (
        "after_r3d_output_clip"
    )
    assert params["support_redistribution_off_support_unchanged"] is True
    assert params["support_redistribution_value_origin"] == (
        "predeclared_mechanistic_constants_not_audit_delta"
    )
    assert params["support_redistribution_constants_source"] == (
        "predeclared_general_liquid_hydrocarbon_nir_energy_redistribution_prior"
    )
    assert params["support_redistribution_mechanism"] == (
        "energy_normalized_mean_neutral_support_redistribution"
    )
    assert params["support_redistribution_shape_normalization"] == "max_abs"
    # Mean-neutral and energy-normalized flags must be true.
    assert params["support_redistribution_mean_neutral"] is True
    assert params["support_redistribution_energy_normalized"] is True
    # Strength range exposes its draw bounds.
    assert (
        _R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE[0]
        <= params["support_redistribution_strength_min"]
    )
    assert (
        params["support_redistribution_strength_max"]
        <= _R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE[1]
    )
    # The mean-subtracted shape is mean-neutral on the support by
    # construction (modulo float arithmetic).
    assert params["support_redistribution_shape_support_mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    # Both lobes must be present: mean-subtracted Gaussian sums always have
    # both positive and negative segments.
    assert params["support_redistribution_shape_has_positive_lobe"] is True
    assert params["support_redistribution_shape_has_negative_lobe"] is True
    # Per-row support mean is preserved within numerical tolerance.
    assert params["support_redistribution_support_mean_abs_error_max"] == (
        pytest.approx(0.0, abs=1e-10)
    )
    # Audit flags: every R9d forbidden mechanism must be explicitly false.
    for key, expected in (
        ("support_redistribution_calibration", False),
        ("support_redistribution_uses_real_stats", False),
        ("support_redistribution_uses_pca", False),
        ("support_redistribution_captures_noise", False),
        ("support_redistribution_uses_labels", False),
        ("support_redistribution_uses_targets", False),
        ("support_redistribution_uses_splits", False),
        ("support_redistribution_uses_ml", False),
        ("support_redistribution_uses_dl", False),
        ("support_redistribution_mutates_thresholds", False),
        ("support_redistribution_mutates_metrics", False),
        ("support_redistribution_adds_offset", False),
        ("support_redistribution_only", True),
    ):
        assert params[key] is expected, f"audit flag {key!r} = {params[key]!r}"
    # R9d must NOT be a readout-space transform, support intercept, support
    # shape envelope, residual transfer, or multiplicative micro-path
    # modulation profile.
    assert "readout_space_transform" not in params
    assert "support_intercept_absorbance" not in params
    assert "shape_envelope_absorbance_range" not in params
    assert "support_centered_residual_transfer_range" not in params
    assert (
        "support_centered_micro_path_modulation_strength_range" not in params
    )
    assert "support_shape_centers_nm" not in params
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == (
        "exp13_dataset_token_diesel_r9d_support_redistribution_route"
    )
    assert params["diesel_support_redistribution_route_source"] == (
        "exp13_dataset_token"
    )
    assert params["diesel_support_redistribution_route_marker"] == "diesel"
    assert params["diesel_support_redistribution_route_non_oracle"] is True
    assert (
        params["diesel_support_redistribution_route_real_stat_capture"]
        is False
    )
    assert (
        params["diesel_support_redistribution_route_thresholds_modified"]
        is False
    )
    assert "diesel_readout_route_source" not in params
    assert "diesel_support_intercept_route_source" not in params
    assert "diesel_support_shape_route_source" not in params
    assert "diesel_micro_path_modulation_route_source" not in params
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False, f"audit flag {key!r} must be False"
    assert np.isfinite(remediated.X).all()
    assert float(remediated.X.min()) >= 0.0


def test_r9d_diesel_off_support_byte_identical_to_r3d_and_support_mean_preserved() -> None:
    """On compliant DIESEL rows, R9d must leave off-support cells byte-equal
    to the R3d base, modify on-support cells, preserve per-row support mean
    within numerical tolerance, and remain non-negative and deterministic.
    """
    seed = 1241
    rs = 8787
    n = 32
    r3d_record = canonicalize_prior_config(_r3d_diesel_source(seed=seed))
    r9d_record = canonicalize_prior_config(_r9d_diesel_source(seed=seed))
    r3d_run = build_synthetic_dataset_run(
        r3d_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9d_run = build_synthetic_dataset_run(
        r9d_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )
    params = r9d_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params["support_redistribution_support_nm"]
    wl = r9d_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    # Off-support cells are byte-identical to the R3d base.
    np.testing.assert_array_equal(
        r9d_run.X[:, ~support_mask], r3d_run.X[:, ~support_mask]
    )
    # On-support cells differ from R3d on at least some rows / wavelengths.
    assert np.any(r9d_run.X[:, support_mask] != r3d_run.X[:, support_mask])
    # Per-row support mean is preserved (mean-neutral redistribution +
    # multiplicative renormalization). The error budget is at least a few
    # orders of magnitude tighter than the renormalization epsilon.
    r3d_support_mean = r3d_run.X[:, support_mask].mean(axis=1)
    r9d_support_mean = r9d_run.X[:, support_mask].mean(axis=1)
    np.testing.assert_allclose(
        r9d_support_mean, r3d_support_mean, atol=1e-10, rtol=0.0
    )
    # Non-negativity is preserved on the entire output.
    assert float(r9d_run.X.min()) >= 0.0
    # Determinism: same seed/source/profile reproduces X and y exactly.
    rerun = build_synthetic_dataset_run(
        r9d_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )
    np.testing.assert_array_equal(r9d_run.y, rerun.y)
    np.testing.assert_array_equal(r9d_run.X, rerun.X)


def test_r9d_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    """Missing or non-compliant R9d markers must fall back byte-identically to
    R3d (audit metadata, X, and y all equal to the R3d run)."""
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=1243))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9d_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )
    assert r9d_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9d_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9d_unmarked.y, r3d_run.y)

    source = _r9d_diesel_source(seed=1244)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9d_diesel_support_redistribution_route"],
        )
    )
    route["thresholds_modified"] = True
    source["_r9d_diesel_support_redistribution_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9d_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    assert (
        r9d_non_compliant.metadata["r2c_mechanistic_remediation"]["profile"]
        == "r3d_diesel_matrix_v1"
    )
    np.testing.assert_array_equal(r9d_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1245),
        _r2r_fruit_puree_source(seed=1246),
        _r2o_beer_source(seed=1247),
        _r2m_milk_source(seed=1248),
        _r2n_manure21_source(seed=1249),
        _soil_source(seed=1250),
    ),
)
def test_r9d_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    """For every non-DIESEL domain, R9d must fall back byte-identically to
    R3d on X, y, and audit metadata."""
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )

    assert r9d_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9d_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9d_run.y, r3d_run.y)


def test_r9d_diesel_differs_from_r3d_r4c_and_r4a() -> None:
    """On compliant DIESEL rows, R9d differs from the R3d base, the R4a
    basis, and the R4c balanced-derivative base."""
    seed = 1242
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4a_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4a_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4a_diesel_basis_v1",
    )
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r9d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ),
    )

    assert np.any(r9d_run.X != r3d_run.X)
    assert np.any(r9d_run.X != r4a_run.X)
    assert np.any(r9d_run.X != r4c_run.X)
    assert np.isfinite(r9d_run.X).all()
    assert float(r9d_run.X.min()) >= 0.0


# ---------------------------------------------------------------------------
# R9e DIESEL pathlength/reference attenuation profile.
# R9e inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r9e_diesel_reference_attenuation_route, R9e applies the R3d
# micro-path / baseline / CH-overtone / clip pipeline byte-for-byte and then,
# AFTER the R3d non-negative output clip and ON the 750-1550 nm support only,
# applies one positive multiplicative attenuation factor per row in [0.970,
# 0.985]. Off-support cells are byte-identical to R3d by construction.
# ---------------------------------------------------------------------------


def _r9e_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp15_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9e_diesel_reference_attenuation_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp15_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9e_constants_are_pre_declared_generic_reference_pathlength_prior() -> None:
    assert R9E_REMEDIATION_PROFILES == (
        "r9e_diesel_pathlength_reference_attenuation_v1",
    )
    assert _R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE == (
        0.970,
        0.985,
    )
    assert _R9E_PETROCHEM_FUELS_SUPPORT_NM == (750.0, 1550.0)
    assert _R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE == (
        "predeclared_generic_blank_reference_pathlength_attenuation_prior"
    )


def test_r9e_profile_is_listed_and_records_non_oracle_reference_route() -> None:
    assert "r9e_diesel_pathlength_reference_attenuation_v1" in ALL_REMEDIATION_PROFILES
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9e_diesel_source(seed=1340)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9e_diesel_pathlength_reference_attenuation_v1"
    assert (
        audit["scope"]
        == "bench_only_r9e_diesel_pathlength_reference_attenuation_remediation"
    )
    params = audit["transform_params"]
    assert params["composition_rule"] == "tight_dirichlet_diesel_centered"
    assert params["spectra_rule"] == (
        "micro_path_fuel_ch_overtone_contrast_readout"
    )
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["ch_overtone_width_nm"] == 34.0
    assert params["support_reference_attenuation_factor_range"] == [0.97, 0.985]
    assert params["support_reference_attenuation_support_nm"] == [750.0, 1550.0]
    assert params["support_reference_attenuation_n_support_bins"] >= 1
    assert params["support_reference_attenuation_application_stage"] == (
        "after_r3d_output_clip"
    )
    assert params["support_reference_attenuation_off_support_unchanged"] is True
    assert params["support_reference_attenuation_constants_source"] == (
        "predeclared_generic_blank_reference_pathlength_attenuation_prior"
    )
    assert params["support_reference_attenuation_mechanism"] == (
        "positive_pathlength_reference_attenuation_support_only"
    )
    assert (
        0.970 <= params["support_reference_attenuation_factor_min"] <= 0.985
    )
    assert (
        0.970 <= params["support_reference_attenuation_factor_max"] <= 0.985
    )
    assert params["support_reference_attenuation_support_ratio_min"] >= 0.970
    assert params["support_reference_attenuation_support_ratio_max"] <= 0.985
    assert params["support_reference_attenuation_guard_clip_fraction"] == 0.0
    assert params["support_reference_attenuation_route_key"] == (
        "_r9e_diesel_reference_attenuation_route"
    )
    for key, expected in (
        ("support_reference_attenuation_calibration", False),
        ("support_reference_attenuation_uses_real_stats", False),
        ("support_reference_attenuation_uses_pca", False),
        ("support_reference_attenuation_captures_noise", False),
        ("support_reference_attenuation_uses_labels", False),
        ("support_reference_attenuation_uses_targets", False),
        ("support_reference_attenuation_uses_splits", False),
        ("support_reference_attenuation_uses_ml", False),
        ("support_reference_attenuation_uses_dl", False),
        ("support_reference_attenuation_mutates_thresholds", False),
        ("support_reference_attenuation_mutates_metrics", False),
        ("support_reference_attenuation_adds_offset", False),
        ("support_reference_attenuation_no_additional_clip", True),
        ("support_reference_attenuation_uses_r9d_shape", False),
        ("support_reference_attenuation_renormalizes_support_mean", False),
        ("support_reference_attenuation_readout_transform", False),
    ):
        assert params[key] is expected, f"audit flag {key!r} = {params[key]!r}"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == (
        "exp15_dataset_token_diesel_r9e_reference_attenuation_route"
    )
    assert params["diesel_reference_attenuation_route_source"] == (
        "exp15_dataset_token"
    )
    assert params["diesel_reference_attenuation_route_marker"] == "diesel"
    assert params["diesel_reference_attenuation_route_non_oracle"] is True
    assert params["diesel_reference_attenuation_route_real_stat_capture"] is False
    assert params["diesel_reference_attenuation_route_thresholds_modified"] is False
    assert "diesel_readout_route_source" not in params
    assert "diesel_support_redistribution_route_source" not in params
    assert float(remediated.X.min()) >= 0.0


def test_r9e_compliant_diesel_attenuates_support_only_and_is_deterministic() -> None:
    seed = 1341
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9e_record = canonicalize_prior_config(_r9e_diesel_source(seed=seed))
    r9e_run = build_synthetic_dataset_run(
        r9e_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )
    params = r9e_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params["support_reference_attenuation_support_nm"]
    wl = r9e_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    np.testing.assert_array_equal(
        r9e_run.X[:, ~support_mask], r3d_run.X[:, ~support_mask]
    )
    assert np.any(r9e_run.X[:, support_mask] != r3d_run.X[:, support_mask])
    positive = r3d_run.X[:, support_mask] > 0.0
    ratios = r9e_run.X[:, support_mask][positive] / r3d_run.X[:, support_mask][
        positive
    ]
    assert ratios.size > 0
    assert float(ratios.min()) >= 0.970
    assert float(ratios.max()) <= 0.985
    support_mean_ratios = (
        r9e_run.X[:, support_mask].mean(axis=1)
        / r3d_run.X[:, support_mask].mean(axis=1)
    )
    assert float(support_mean_ratios.min()) >= 0.970
    assert float(support_mean_ratios.max()) <= 0.985
    assert float(r9e_run.X.min()) >= 0.0

    rerun = build_synthetic_dataset_run(
        r9e_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )
    np.testing.assert_array_equal(r9e_run.y, rerun.y)
    np.testing.assert_array_equal(r9e_run.X, rerun.X)


def test_r9e_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=1343))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9e_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )
    assert r9e_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9e_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9e_unmarked.y, r3d_run.y)

    source = _r9e_diesel_source(seed=1344)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9e_diesel_reference_attenuation_route"],
        )
    )
    route["real_stat_capture"] = True
    source["_r9e_diesel_reference_attenuation_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9e_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9e_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9e_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1345),
        _r2r_fruit_puree_source(seed=1346),
        _r2o_beer_source(seed=1347),
        _r2m_milk_source(seed=1348),
        _r2n_manure21_source(seed=1349),
        _soil_source(seed=1350),
    ),
)
def test_r9e_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9e_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )

    assert r9e_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9e_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9e_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9f DIESEL pre-offset pathlength/reference attenuation profile.
# R9f inherits R3d for non-DIESEL rows. On explicit DIESEL rows that carry the
# dedicated _r9f_diesel_pre_offset_reference_attenuation_route, R9f attenuates
# only the Beer-Lambert continuum/path component on 750-1550 nm before detector
# offset addition and before the existing R3d output clip. Feature residuals,
# additive offsets, readout transforms, R9d shape, support-mean renormalization,
# and negative intercepts remain out of scope.
# ---------------------------------------------------------------------------


def _r9f_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp16_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9f_diesel_pre_offset_reference_attenuation_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp16_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9f_constants_are_pre_declared_generic_reference_pathlength_prior() -> None:
    assert R9F_REMEDIATION_PROFILES == (
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
    )
    assert _R9F_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE == (
        0.970,
        0.985,
    )
    assert _R9F_PETROCHEM_FUELS_SUPPORT_NM == (750.0, 1550.0)
    assert _R9F_PETROCHEM_FUELS_CONSTANTS_SOURCE == (
        "predeclared_generic_blank_reference_pathlength_attenuation_prior"
    )


def test_r9f_profile_records_pre_offset_reference_route_metadata() -> None:
    assert (
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        in ALL_REMEDIATION_PROFILES
    )
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9f_diesel_source(seed=1351)),
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == (
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
    )
    assert audit["scope"] == (
        "bench_only_r9f_diesel_pre_offset_pathlength_reference_attenuation_remediation"
    )
    params = audit["transform_params"]
    assert params["support_reference_attenuation_factor_range"] == [0.97, 0.985]
    assert params["support_reference_attenuation_support_nm"] == [750.0, 1550.0]
    assert params["support_reference_attenuation_application_stage"] == (
        "before_additive_baseline_and_output_clip_on_continuum_path_component"
    )
    assert params["support_reference_attenuation_component_only"] is True
    assert params["support_reference_attenuation_offset_unchanged"] is True
    assert params["support_reference_attenuation_feature_residual_unchanged"] is True
    assert params["support_reference_attenuation_no_additional_clip"] is True
    assert params["support_reference_attenuation_route_key"] == (
        "_r9f_diesel_pre_offset_reference_attenuation_route"
    )
    assert params["support_reference_attenuation_constants_source"] == (
        "predeclared_generic_blank_reference_pathlength_attenuation_prior"
    )
    assert (
        0.970
        <= params["support_reference_attenuation_factor_min"]
        <= 0.985
    )
    assert (
        0.970
        <= params["support_reference_attenuation_factor_max"]
        <= 0.985
    )
    assert params["support_reference_attenuation_component_ratio_min"] >= 0.970
    assert params["support_reference_attenuation_component_ratio_max"] <= 0.985
    for key, expected in (
        ("support_reference_attenuation_calibration", False),
        ("support_reference_attenuation_uses_real_stats", False),
        ("support_reference_attenuation_uses_pca", False),
        ("support_reference_attenuation_captures_noise", False),
        ("support_reference_attenuation_uses_labels", False),
        ("support_reference_attenuation_uses_targets", False),
        ("support_reference_attenuation_uses_splits", False),
        ("support_reference_attenuation_uses_ml", False),
        ("support_reference_attenuation_uses_dl", False),
        ("support_reference_attenuation_mutates_thresholds", False),
        ("support_reference_attenuation_mutates_metrics", False),
        ("support_reference_attenuation_adds_offset", False),
        ("support_reference_attenuation_uses_r9d_shape", False),
        ("support_reference_attenuation_renormalizes_support_mean", False),
        ("support_reference_attenuation_readout_transform", False),
    ):
        assert params[key] is expected, f"audit flag {key!r} = {params[key]!r}"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == (
        "exp16_dataset_token_diesel_r9f_pre_offset_reference_attenuation_route"
    )
    assert params["diesel_pre_offset_reference_attenuation_route_source"] == (
        "exp16_dataset_token"
    )
    assert params["diesel_pre_offset_reference_attenuation_route_marker"] == "diesel"
    assert params["diesel_pre_offset_reference_attenuation_route_non_oracle"] is True
    assert (
        params["diesel_pre_offset_reference_attenuation_route_real_stat_capture"]
        is False
    )
    assert (
        params["diesel_pre_offset_reference_attenuation_route_thresholds_modified"]
        is False
    )
    assert float(remediated.X.min()) >= 0.0


def test_r9f_compliant_diesel_changes_support_and_differs_from_r9e() -> None:
    seed = 1352
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9e_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9e_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9e_diesel_pathlength_reference_attenuation_v1",
    )
    r9f_record = canonicalize_prior_config(_r9f_diesel_source(seed=seed))
    r9f_run = build_synthetic_dataset_run(
        r9f_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )
    params = r9f_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    support_low, support_high = params["support_reference_attenuation_support_nm"]
    wl = r9f_run.wavelengths
    support_mask = (wl >= float(support_low)) & (wl <= float(support_high))
    assert support_mask.any()
    assert (~support_mask).any()

    np.testing.assert_array_equal(
        r9f_run.X[:, ~support_mask], r3d_run.X[:, ~support_mask]
    )
    assert np.any(r9f_run.X[:, support_mask] != r3d_run.X[:, support_mask])
    assert np.any(r9f_run.X[:, support_mask] != r9e_run.X[:, support_mask])
    assert np.any(r9f_run.X != r3d_run.X)
    assert np.any(r9f_run.X != r9e_run.X)
    assert float(r9f_run.X.min()) >= 0.0

    rerun = build_synthetic_dataset_run(
        r9f_record,
        n_samples=n,
        random_seed=rs,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )
    np.testing.assert_array_equal(r9f_run.y, rerun.y)
    np.testing.assert_array_equal(r9f_run.X, rerun.X)


def test_r9f_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked = canonicalize_prior_config(_fuel_diesel_source(seed=1353))
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9f_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )
    assert r9f_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9f_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9f_unmarked.y, r3d_run.y)

    source = _r9f_diesel_source(seed=1354)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9f_diesel_pre_offset_reference_attenuation_route"],
        )
    )
    route["thresholds_modified"] = True
    source["_r9f_diesel_pre_offset_reference_attenuation_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9f_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9f_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9f_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1355),
        _r2r_fruit_puree_source(seed=1356),
        _r2o_beer_source(seed=1357),
        _r2m_milk_source(seed=1358),
        _r2n_manure21_source(seed=1359),
        _soil_source(seed=1360),
    ),
)
def test_r9f_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9f_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )

    assert r9f_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9f_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9f_run.y, r3d_run.y)


def test_r9f_off_support_wavelength_grid_is_identical_to_r3d() -> None:
    source = _r9f_diesel_source(seed=1361)
    source["wavelength_range"] = (1600, 1700)
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9f_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile=(
            "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
        ),
    )

    params = r9f_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert params["support_reference_attenuation_n_support_bins"] == 0
    np.testing.assert_array_equal(r9f_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9f_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9h DIESEL support-CH-center/drop-1720 isolation profile.
# R9h inherits R3d for non-DIESEL and non-compliant DIESEL rows. On explicit
# DIESEL rows carrying _r9h_diesel_support_ch_center_route, it changes ONLY
# ch_overtone_centers_nm to 1150/1210/1390/1460 nm while keeping R3d width,
# gain, path/baseline/feature/readout/clip and adding no R4/R9 support extras.
# ---------------------------------------------------------------------------


def _r9h_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp18_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9h_diesel_support_ch_center_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp18_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9h_profile_records_support_ch_center_route_metadata() -> None:
    assert R9H_REMEDIATION_PROFILES == (
        "r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )
    assert (
        "r9h_diesel_support_ch_center_drop1720_isolation_v1"
        in ALL_REMEDIATION_PROFILES
    )
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9h_diesel_source(seed=1370)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9h_diesel_support_ch_center_drop1720_isolation_v1"
    assert audit["scope"] == (
        "bench_only_r9h_diesel_support_ch_center_drop1720_isolation_remediation"
    )
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert params["ch_overtone_width_nm"] == 34.0
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["readout_space"] == (
        "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["output_clip_absorbance"] == [0.0, None]
    assert params["support_ch_center_drop1720_isolation_route_key"] == (
        "_r9h_diesel_support_ch_center_route"
    )
    assert params["diesel_support_ch_center_route_source"] == "exp18_dataset_token"
    assert params["diesel_support_ch_center_route_marker"] == "diesel"
    assert params["diesel_support_ch_center_route_non_oracle"] is True
    assert params["diesel_support_ch_center_route_real_stat_capture"] is False
    assert params["diesel_support_ch_center_route_thresholds_modified"] is False
    for key, expected in (
        ("support_ch_center_drop1720_isolation_calibration", False),
        ("support_ch_center_drop1720_isolation_uses_real_stats", False),
        ("support_ch_center_drop1720_isolation_uses_pca", False),
        ("support_ch_center_drop1720_isolation_captures_noise", False),
        ("support_ch_center_drop1720_isolation_uses_labels", False),
        ("support_ch_center_drop1720_isolation_uses_targets", False),
        ("support_ch_center_drop1720_isolation_uses_splits", False),
        ("support_ch_center_drop1720_isolation_uses_ml", False),
        ("support_ch_center_drop1720_isolation_uses_dl", False),
        ("support_ch_center_drop1720_isolation_mutates_thresholds", False),
        ("support_ch_center_drop1720_isolation_mutates_metrics", False),
        ("support_ch_center_drop1720_isolation_adds_damping", False),
        ("support_ch_center_drop1720_isolation_adds_continuum_hump", False),
        ("support_ch_center_drop1720_isolation_adds_support_intercept", False),
        ("support_ch_center_drop1720_isolation_adds_support_shape", False),
        ("support_ch_center_drop1720_isolation_adds_redistribution", False),
        ("support_ch_center_drop1720_isolation_adds_attenuation", False),
        ("support_ch_center_drop1720_isolation_readout_transform", False),
        ("support_ch_center_drop1720_isolation_extra_guard_clip", False),
    ):
        assert params[key] is expected


def test_r9h_compliant_diesel_changes_only_support_ch_centers_vs_r3d() -> None:
    seed = 1371
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r4b_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4b_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4b_diesel_derivative_restore_v1",
    )
    r4c_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r4c_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
    )
    r9h_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9h_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )

    assert np.any(r9h_run.X != r3d_run.X)
    assert np.any(r9h_run.X != r4b_run.X)
    assert np.any(r9h_run.X != r4c_run.X)
    np.testing.assert_array_equal(r9h_run.y, r3d_run.y)
    params = r9h_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r3d_params = r3d_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    for key in (
        "path_factor_range",
        "path_factor_min",
        "path_factor_max",
        "feature_contrast_range",
        "feature_contrast_min",
        "feature_contrast_max",
        "ch_overtone_width_nm",
        "ch_overtone_gain_range",
        "ch_overtone_gain_min",
        "ch_overtone_gain_max",
        "additive_baseline_range",
        "additive_baseline_min",
        "additive_baseline_max",
        "readout_space",
        "output_clip_absorbance",
    ):
        assert params[key] == r3d_params[key]
    assert params["ch_overtone_centers_nm"] == [1150.0, 1210.0, 1390.0, 1460.0]
    assert r3d_params["ch_overtone_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
        1720.0,
    ]
    for forbidden in (
        "damping_windows_nm",
        "continuum_hump_center_nm",
        "continuum_hump_width_nm",
        "continuum_hump_amplitude_range",
        "support_reference_attenuation_factor_range",
        "support_shape_centers_nm",
        "support_redistribution_centers_nm",
        "support_intercept_absorbance",
        "readout_space_transform",
    ):
        assert forbidden not in params


def test_r9h_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked_source = _fuel_diesel_source(seed=1372)
    unmarked_source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp18_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    unmarked = canonicalize_prior_config(unmarked_source)
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9h_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )
    assert r9h_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9h_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9h_unmarked.y, r3d_run.y)

    source = _r9h_diesel_source(seed=1373)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9h_diesel_support_ch_center_route"],
        )
    )
    route["real_stat_capture"] = True
    source["_r9h_diesel_support_ch_center_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9h_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9h_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9h_non_compliant.X, r3d_non_compliant.X)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1374),
        _r2r_fruit_puree_source(seed=1375),
        _r2o_beer_source(seed=1376),
        _r2m_milk_source(seed=1377),
        _r2n_manure21_source(seed=1378),
        _soil_source(seed=1379),
    ),
)
def test_r9h_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9h_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )

    assert r9h_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9h_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9h_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9i DIESEL CH width/gain isolation profile.
# R9i inherits R3d for non-DIESEL and non-compliant DIESEL rows. On explicit
# DIESEL rows carrying _r9i_diesel_ch_width_gain_route, it changes ONLY
# ch_overtone_width_nm to 36.0 and ch_overtone_gain_range to 0.092-0.155,
# while keeping R3d CH centers and adding no R4/R9 support extras.
# ---------------------------------------------------------------------------


def _r9i_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp19_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9i_diesel_ch_width_gain_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp19_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9i_profile_records_ch_width_gain_route_metadata() -> None:
    assert R9I_REMEDIATION_PROFILES == (
        "r9i_diesel_ch_width_gain_isolation_v1",
    )
    assert "r9i_diesel_ch_width_gain_isolation_v1" in ALL_REMEDIATION_PROFILES
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9i_diesel_source(seed=1470)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9i_diesel_ch_width_gain_isolation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9i_diesel_ch_width_gain_isolation_v1"
    assert audit["scope"] == (
        "bench_only_r9i_diesel_ch_width_gain_isolation_remediation"
    )
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["ch_overtone_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
        1720.0,
    ]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["readout_space"] == (
        "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["output_clip_absorbance"] == [0.0, None]
    assert params["diesel_ch_width_gain_isolation_route_key"] == (
        "_r9i_diesel_ch_width_gain_route"
    )
    assert params["diesel_ch_width_gain_route_source"] == "exp19_dataset_token"
    assert params["diesel_ch_width_gain_route_marker"] == "diesel"
    assert params["diesel_ch_width_gain_route_non_oracle"] is True
    assert params["diesel_ch_width_gain_route_real_stat_capture"] is False
    assert params["diesel_ch_width_gain_route_thresholds_modified"] is False
    for key, expected in (
        ("diesel_ch_width_gain_isolation_only", True),
        ("diesel_ch_width_gain_isolation_calibration", False),
        ("diesel_ch_width_gain_isolation_uses_real_stats", False),
        ("diesel_ch_width_gain_isolation_uses_pca", False),
        ("diesel_ch_width_gain_isolation_captures_noise", False),
        ("diesel_ch_width_gain_isolation_uses_labels", False),
        ("diesel_ch_width_gain_isolation_uses_targets", False),
        ("diesel_ch_width_gain_isolation_uses_splits", False),
        ("diesel_ch_width_gain_isolation_uses_ml", False),
        ("diesel_ch_width_gain_isolation_uses_dl", False),
        ("diesel_ch_width_gain_isolation_mutates_thresholds", False),
        ("diesel_ch_width_gain_isolation_mutates_metrics", False),
        ("diesel_ch_width_gain_isolation_changes_ch_centers", False),
        ("diesel_ch_width_gain_isolation_adds_damping", False),
        ("diesel_ch_width_gain_isolation_adds_continuum_hump", False),
        ("diesel_ch_width_gain_isolation_adds_support_intercept", False),
        ("diesel_ch_width_gain_isolation_adds_support_shape", False),
        ("diesel_ch_width_gain_isolation_adds_redistribution", False),
        ("diesel_ch_width_gain_isolation_adds_attenuation", False),
        ("diesel_ch_width_gain_isolation_readout_transform", False),
        ("diesel_ch_width_gain_isolation_extra_guard_clip", False),
    ):
        assert params[key] is expected
    for forbidden in (
        "damping_windows_nm",
        "continuum_hump_center_nm",
        "continuum_hump_width_nm",
        "continuum_hump_amplitude_range",
        "support_reference_attenuation_factor_range",
        "pre_offset_reference_attenuation_factor_range",
        "support_shape_centers_nm",
        "support_redistribution_centers_nm",
        "support_intercept_absorbance",
        "readout_space_transform",
    ):
        assert forbidden not in params


def test_r9i_compliant_diesel_differs_from_r3d_and_r9h_only_by_width_gain() -> None:
    seed = 1471
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9h_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9h_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
    )
    r9i_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9i_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9i_diesel_ch_width_gain_isolation_v1",
    )

    assert np.any(r9i_run.X != r3d_run.X)
    assert np.any(r9i_run.X != r9h_run.X)
    np.testing.assert_array_equal(r9i_run.y, r3d_run.y)
    params = r9i_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r3d_params = r3d_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    for key in (
        "path_factor_range",
        "path_factor_min",
        "path_factor_max",
        "feature_contrast_range",
        "feature_contrast_min",
        "feature_contrast_max",
        "additive_baseline_range",
        "additive_baseline_min",
        "additive_baseline_max",
        "readout_space",
        "output_clip_absorbance",
        "ch_overtone_centers_nm",
    ):
        assert params[key] == r3d_params[key]
    assert params["ch_overtone_width_nm"] == 36.0
    assert params["ch_overtone_gain_range"] == [0.092, 0.155]
    assert r3d_params["ch_overtone_width_nm"] == 34.0
    assert r3d_params["ch_overtone_gain_range"] == [0.11, 0.18]


def test_r9i_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked_source = _fuel_diesel_source(seed=1472)
    unmarked_source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp19_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    unmarked = canonicalize_prior_config(unmarked_source)
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9i_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9i_diesel_ch_width_gain_isolation_v1",
    )
    assert r9i_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9i_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9i_unmarked.y, r3d_run.y)

    source = _r9i_diesel_source(seed=1473)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9i_diesel_ch_width_gain_route"],
        )
    )
    route["thresholds_modified"] = True
    source["_r9i_diesel_ch_width_gain_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9i_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9i_diesel_ch_width_gain_isolation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9i_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9i_non_compliant.X, r3d_non_compliant.X)
    np.testing.assert_array_equal(r9i_non_compliant.y, r3d_non_compliant.y)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1474),
        _r2r_fruit_puree_source(seed=1475),
        _r2o_beer_source(seed=1476),
        _r2m_milk_source(seed=1477),
        _r2n_manure21_source(seed=1478),
        _soil_source(seed=1479),
    ),
)
def test_r9i_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9i_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9i_diesel_ch_width_gain_isolation_v1",
    )

    assert r9i_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9i_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9i_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9j DIESEL residual damping-only isolation profile.
# R9j inherits R3d for non-DIESEL and non-compliant DIESEL rows. On explicit
# DIESEL rows carrying _r9j_diesel_residual_damping_route, it changes ONLY
# damping_windows_nm and damping_strength_range, while keeping R3d CH centers,
# width, gain, and adding no continuum hump or support/readout extras.
# ---------------------------------------------------------------------------


def _r9j_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp20_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9j_diesel_residual_damping_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp20_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9j_profile_records_residual_damping_route_metadata() -> None:
    assert R9J_REMEDIATION_PROFILES == (
        "r9j_diesel_residual_damping_isolation_v1",
    )
    assert "r9j_diesel_residual_damping_isolation_v1" in ALL_REMEDIATION_PROFILES
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9j_diesel_source(seed=1570)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9j_diesel_residual_damping_isolation_v1"
    assert audit["scope"] == (
        "bench_only_r9j_diesel_residual_damping_isolation_remediation"
    )
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["ch_overtone_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
        1720.0,
    ]
    assert params["ch_overtone_width_nm"] == 34.0
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["readout_space"] == (
        "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["output_clip_absorbance"] == [0.0, None]
    assert params["diesel_residual_damping_isolation_route_key"] == (
        "_r9j_diesel_residual_damping_route"
    )
    assert params["diesel_residual_damping_route_source"] == "exp20_dataset_token"
    assert params["diesel_residual_damping_route_marker"] == "diesel"
    assert params["diesel_residual_damping_route_non_oracle"] is True
    assert params["diesel_residual_damping_route_real_stat_capture"] is False
    assert params["diesel_residual_damping_route_thresholds_modified"] is False
    for key, expected in (
        ("diesel_residual_damping_isolation_only", True),
        ("diesel_residual_damping_isolation_calibration", False),
        ("diesel_residual_damping_isolation_uses_real_stats", False),
        ("diesel_residual_damping_isolation_uses_pca", False),
        ("diesel_residual_damping_isolation_captures_noise", False),
        ("diesel_residual_damping_isolation_uses_labels", False),
        ("diesel_residual_damping_isolation_uses_targets", False),
        ("diesel_residual_damping_isolation_uses_splits", False),
        ("diesel_residual_damping_isolation_uses_ml", False),
        ("diesel_residual_damping_isolation_uses_dl", False),
        ("diesel_residual_damping_isolation_mutates_thresholds", False),
        ("diesel_residual_damping_isolation_mutates_metrics", False),
        ("diesel_residual_damping_isolation_changes_ch_centers", False),
        ("diesel_residual_damping_isolation_changes_ch_width_gain", False),
        ("diesel_residual_damping_isolation_adds_damping", True),
        ("diesel_residual_damping_isolation_adds_continuum_hump", False),
        ("diesel_residual_damping_isolation_adds_support_intercept", False),
        ("diesel_residual_damping_isolation_adds_support_shape", False),
        ("diesel_residual_damping_isolation_adds_redistribution", False),
        ("diesel_residual_damping_isolation_adds_attenuation", False),
        ("diesel_residual_damping_isolation_readout_transform", False),
        ("diesel_residual_damping_isolation_extra_guard_clip", False),
    ):
        assert params[key] is expected
    for forbidden in (
        "continuum_hump_center_nm",
        "continuum_hump_width_nm",
        "continuum_hump_amplitude_range",
        "support_reference_attenuation_factor_range",
        "pre_offset_reference_attenuation_factor_range",
        "support_shape_centers_nm",
        "support_redistribution_centers_nm",
        "support_intercept_absorbance",
        "readout_space_transform",
    ):
        assert forbidden not in params


def test_r9j_compliant_diesel_differs_from_r3d_only_by_residual_damping() -> None:
    seed = 1571
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9j_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9j_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )

    assert np.any(r9j_run.X != r3d_run.X)
    np.testing.assert_array_equal(r9j_run.y, r3d_run.y)
    params = r9j_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r3d_params = r3d_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    for key in (
        "path_factor_range",
        "path_factor_min",
        "path_factor_max",
        "feature_contrast_range",
        "feature_contrast_min",
        "feature_contrast_max",
        "additive_baseline_range",
        "additive_baseline_min",
        "additive_baseline_max",
        "readout_space",
        "output_clip_absorbance",
        "ch_overtone_centers_nm",
        "ch_overtone_width_nm",
        "ch_overtone_gain_range",
        "ch_overtone_gain_min",
        "ch_overtone_gain_max",
    ):
        assert params[key] == r3d_params[key]
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.60],
        [1425.0, 54.0, 0.70],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert "continuum_hump_center_nm" not in params


def test_r9j_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked_source = _fuel_diesel_source(seed=1572)
    unmarked_source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp20_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    unmarked = canonicalize_prior_config(unmarked_source)
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9j_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )
    assert r9j_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9j_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9j_unmarked.y, r3d_run.y)

    source = _r9j_diesel_source(seed=1573)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9j_diesel_residual_damping_route"],
        )
    )
    route["real_stat_capture"] = True
    source["_r9j_diesel_residual_damping_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9j_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9j_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9j_non_compliant.X, r3d_non_compliant.X)
    np.testing.assert_array_equal(r9j_non_compliant.y, r3d_non_compliant.y)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1574),
        _r2r_fruit_puree_source(seed=1575),
        _r2o_beer_source(seed=1576),
        _r2m_milk_source(seed=1577),
        _r2n_manure21_source(seed=1578),
        _soil_source(seed=1579),
    ),
)
def test_r9j_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9j_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )

    assert r9j_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9j_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9j_run.y, r3d_run.y)


# ---------------------------------------------------------------------------
# R9k DIESEL continuum-hump-only isolation profile.
# R9k inherits R3d for non-DIESEL and non-compliant DIESEL rows. On explicit
# DIESEL rows carrying _r9k_diesel_continuum_hump_route, it changes ONLY
# the R4c continuum hump, while keeping R3d CH centers, width, gain, and
# adding no damping, attenuation, support extras, readout transform, or clip.
# ---------------------------------------------------------------------------


def _r9k_diesel_source(*, seed: int) -> dict[str, object]:
    source = _fuel_diesel_source(seed=seed)
    source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp21_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    source["_r9k_diesel_continuum_hump_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp21_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    return source


def test_r9k_profile_records_continuum_hump_route_metadata() -> None:
    assert R9K_REMEDIATION_PROFILES == (
        "r9k_diesel_continuum_hump_isolation_v1",
    )
    assert "r9k_diesel_continuum_hump_isolation_v1" in ALL_REMEDIATION_PROFILES
    remediated = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9k_diesel_source(seed=1670)),
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9k_diesel_continuum_hump_isolation_v1",
    )

    audit = remediated.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9k_diesel_continuum_hump_isolation_v1"
    assert audit["scope"] == (
        "bench_only_r9k_diesel_continuum_hump_isolation_remediation"
    )
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["ch_overtone_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
        1720.0,
    ]
    assert params["ch_overtone_width_nm"] == 34.0
    assert params["ch_overtone_gain_range"] == [0.11, 0.18]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    assert params["path_factor_range"] == [0.01, 0.018]
    assert params["additive_baseline_range"] == [5e-05, 0.00035]
    assert params["feature_contrast_range"] == [0.22, 0.31]
    assert params["readout_space"] == (
        "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["output_clip_absorbance"] == [0.0, None]
    assert params["diesel_continuum_hump_isolation_route_key"] == (
        "_r9k_diesel_continuum_hump_route"
    )
    assert params["diesel_continuum_hump_route_source"] == "exp21_dataset_token"
    assert params["diesel_continuum_hump_route_marker"] == "diesel"
    assert params["diesel_continuum_hump_route_non_oracle"] is True
    assert params["diesel_continuum_hump_route_real_stat_capture"] is False
    assert params["diesel_continuum_hump_route_thresholds_modified"] is False
    for key, expected in (
        ("diesel_continuum_hump_isolation_only", True),
        ("diesel_continuum_hump_isolation_calibration", False),
        ("diesel_continuum_hump_isolation_uses_real_stats", False),
        ("diesel_continuum_hump_isolation_uses_pca", False),
        ("diesel_continuum_hump_isolation_captures_noise", False),
        ("diesel_continuum_hump_isolation_uses_labels", False),
        ("diesel_continuum_hump_isolation_uses_targets", False),
        ("diesel_continuum_hump_isolation_uses_splits", False),
        ("diesel_continuum_hump_isolation_uses_ml", False),
        ("diesel_continuum_hump_isolation_uses_dl", False),
        ("diesel_continuum_hump_isolation_mutates_thresholds", False),
        ("diesel_continuum_hump_isolation_mutates_metrics", False),
        ("diesel_continuum_hump_isolation_changes_ch_centers", False),
        ("diesel_continuum_hump_isolation_changes_ch_width_gain", False),
        ("diesel_continuum_hump_isolation_adds_damping", False),
        ("diesel_continuum_hump_isolation_adds_continuum_hump", True),
        ("diesel_continuum_hump_isolation_adds_support_intercept", False),
        ("diesel_continuum_hump_isolation_adds_support_shape", False),
        ("diesel_continuum_hump_isolation_adds_redistribution", False),
        ("diesel_continuum_hump_isolation_adds_attenuation", False),
        ("diesel_continuum_hump_isolation_readout_transform", False),
        ("diesel_continuum_hump_isolation_extra_guard_clip", False),
    ):
        assert params[key] is expected
    for forbidden in (
        "damping_windows_nm",
        "damping_strength_range",
        "support_reference_attenuation_factor_range",
        "pre_offset_reference_attenuation_factor_range",
        "support_shape_centers_nm",
        "support_redistribution_centers_nm",
        "support_intercept_absorbance",
        "readout_space_transform",
    ):
        assert forbidden not in params


def test_r9k_compliant_diesel_differs_from_r3d_only_by_continuum_hump() -> None:
    seed = 1671
    rs = 8787
    n = 32
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r3d_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9j_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9j_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )
    r9k_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9k_diesel_source(seed=seed)),
        n_samples=n,
        random_seed=rs,
        remediation_profile="r9k_diesel_continuum_hump_isolation_v1",
    )

    assert np.any(r9k_run.X != r3d_run.X)
    assert np.any(r9k_run.X != r9j_run.X)
    np.testing.assert_array_equal(r9k_run.y, r3d_run.y)
    params = r9k_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    r3d_params = r3d_run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    for key in (
        "path_factor_range",
        "path_factor_min",
        "path_factor_max",
        "feature_contrast_range",
        "feature_contrast_min",
        "feature_contrast_max",
        "additive_baseline_range",
        "additive_baseline_min",
        "additive_baseline_max",
        "readout_space",
        "output_clip_absorbance",
        "ch_overtone_centers_nm",
        "ch_overtone_width_nm",
        "ch_overtone_gain_range",
        "ch_overtone_gain_min",
        "ch_overtone_gain_max",
    ):
        assert params[key] == r3d_params[key]
    assert params["continuum_hump_center_nm"] == 975.0
    assert params["continuum_hump_width_nm"] == 72.0
    assert params["continuum_hump_amplitude_range"] == [0.00010, 0.00032]
    assert params["continuum_hump_support_nm"] == [750.0, 1550.0]
    assert "damping_windows_nm" not in params
    assert "damping_strength_range" not in params


def test_r9k_unmarked_or_non_compliant_diesel_is_routed_back_to_r3d() -> None:
    unmarked_source = _fuel_diesel_source(seed=1672)
    unmarked_source["_r3d_diesel_readout_route"] = {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp21_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }
    unmarked = canonicalize_prior_config(unmarked_source)
    r3d_run = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9k_unmarked = build_synthetic_dataset_run(
        unmarked,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9k_diesel_continuum_hump_isolation_v1",
    )
    assert r9k_unmarked.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9k_unmarked.X, r3d_run.X)
    np.testing.assert_array_equal(r9k_unmarked.y, r3d_run.y)

    source = _r9k_diesel_source(seed=1673)
    route = dict(
        cast(
            "dict[str, object]",
            source["_r9k_diesel_continuum_hump_route"],
        )
    )
    route["real_stat_capture"] = True
    source["_r9k_diesel_continuum_hump_route"] = route
    non_compliant = canonicalize_prior_config(source)
    r9k_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9k_diesel_continuum_hump_isolation_v1",
    )
    r3d_non_compliant = build_synthetic_dataset_run(
        non_compliant,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    assert r9k_non_compliant.metadata["r2c_mechanistic_remediation"] == (
        r3d_non_compliant.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9k_non_compliant.X, r3d_non_compliant.X)
    np.testing.assert_array_equal(r9k_non_compliant.y, r3d_non_compliant.y)


@pytest.mark.parametrize(
    "source",
    (
        _r3b_corn_source(seed=1674),
        _r2r_fruit_puree_source(seed=1675),
        _r2o_beer_source(seed=1676),
        _r2m_milk_source(seed=1677),
        _r2n_manure21_source(seed=1678),
        _soil_source(seed=1679),
    ),
)
def test_r9k_non_diesel_draws_are_identical_to_r3d(
    source: dict[str, object],
) -> None:
    record = canonicalize_prior_config(source)
    r3d_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9k_run = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=8787,
        remediation_profile="r9k_diesel_continuum_hump_isolation_v1",
    )

    assert r9k_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )
    np.testing.assert_array_equal(r9k_run.X, r3d_run.X)
    np.testing.assert_array_equal(r9k_run.y, r3d_run.y)
