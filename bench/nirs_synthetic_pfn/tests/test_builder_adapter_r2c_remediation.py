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
