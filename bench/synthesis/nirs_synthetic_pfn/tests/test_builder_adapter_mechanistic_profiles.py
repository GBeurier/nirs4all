"""R2a mechanistic profile contract tests for the bench-only builder adapter."""

from __future__ import annotations

import numpy as np
import pytest
from nirsyntheticpfn.adapters.builder_adapter import (
    ALL_REMEDIATION_PROFILES,
    R2A_MECHANISTIC_PROFILES,
    R9L_REMEDIATION_PROFILES,
    R9M_REMEDIATION_PROFILES,
    build_synthetic_dataset_run,
)
from nirsyntheticpfn.adapters.prior_adapter import (
    canonicalize_domain,
    canonicalize_prior_config,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config


def _valid_source(domain_alias: str, *, seed: int) -> dict[str, object]:
    domain_key = canonicalize_domain(domain_alias)
    components: list[str] = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == 3:
            break
    if len(components) < 3:
        raise AssertionError(f"not enough valid components for {domain_key}")
    return {
        "domain": domain_alias,
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (400, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": components,
        "n_samples": 100,
        "target_config": {"type": "regression", "n_targets": 1, "nonlinearity": "none"},
        "random_state": seed,
    }


def _diesel_route(source: str) -> dict[str, object]:
    return {
        "enabled": True,
        "route_marker": "diesel",
        "source": source,
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }


def _r9l_diesel_source(*, seed: int, compliant: bool = True) -> dict[str, object]:
    source = _valid_source("petrochem_fuels", seed=seed)
    source["_r3d_diesel_readout_route"] = _diesel_route("exp22_dataset_token")
    route = _diesel_route("exp22_dataset_token")
    if not compliant:
        route["real_stat_capture"] = True
    source["_r9l_diesel_residual_damping_clean_attenuation_route"] = route
    return source


def _r9m_diesel_source(*, seed: int, compliant: bool = True) -> dict[str, object]:
    source = _valid_source("petrochem_fuels", seed=seed)
    source["_r3d_diesel_readout_route"] = _diesel_route("exp23_dataset_token")
    route = _diesel_route("exp23_dataset_token")
    if not compliant:
        route["thresholds_modified"] = True
    source["_r9m_diesel_width_gain_damping_clean_attenuation_route"] = route
    return source


def _r9j_diesel_source(*, seed: int) -> dict[str, object]:
    source = _valid_source("petrochem_fuels", seed=seed)
    source["_r3d_diesel_readout_route"] = _diesel_route("exp20_dataset_token")
    source["_r9j_diesel_residual_damping_route"] = _diesel_route("exp20_dataset_token")
    return source


def test_default_path_is_unchanged_when_profile_is_none() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=42))

    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=31415)
    explicit_none = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=31415,
        mechanistic_profile=None,
    )

    np.testing.assert_allclose(baseline.X, explicit_none.X)
    np.testing.assert_allclose(baseline.y, explicit_none.y)
    np.testing.assert_allclose(baseline.wavelengths, explicit_none.wavelengths)
    audit = baseline.metadata["r2a_mechanistic_profile"]
    assert audit["enabled"] is False
    assert audit["profile"] is None
    _assert_audit_non_oracle(audit)


@pytest.mark.parametrize(
    "profile",
    [p for p in R2A_MECHANISTIC_PROFILES if p != "r2a_baseline"],
)
def test_each_profile_changes_x_and_records_non_oracle_audit(profile: str) -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=11))

    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    profiled = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        mechanistic_profile=profile,
    )

    assert profiled.X.shape == baseline.X.shape
    assert np.isfinite(profiled.X).all()
    np.testing.assert_allclose(profiled.y, baseline.y)
    np.testing.assert_allclose(profiled.wavelengths, baseline.wavelengths)
    assert not np.allclose(profiled.X, baseline.X)
    audit = profiled.metadata["r2a_mechanistic_profile"]
    assert audit["enabled"] is True
    assert audit["profile"] == profile
    assert isinstance(audit["seed"], int)
    _assert_audit_non_oracle(audit)


def test_baseline_profile_is_identity_control() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=11))

    baseline = build_synthetic_dataset_run(record, n_samples=24, random_seed=2024)
    control = build_synthetic_dataset_run(
        record,
        n_samples=24,
        random_seed=2024,
        mechanistic_profile="r2a_baseline",
    )

    np.testing.assert_allclose(control.X, baseline.X)
    audit = control.metadata["r2a_mechanistic_profile"]
    assert audit["enabled"] is True
    assert audit["profile"] == "r2a_baseline"
    _assert_audit_non_oracle(audit)


def test_profiles_are_deterministic_for_same_seed() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=7))

    first = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=99,
        mechanistic_profile="r2a_emsc_like_scatter",
    )
    second = build_synthetic_dataset_run(
        record,
        n_samples=20,
        random_seed=99,
        mechanistic_profile="r2a_emsc_like_scatter",
    )

    np.testing.assert_allclose(first.X, second.X)
    assert first.metadata["r2a_mechanistic_profile"]["seed"] == (second.metadata["r2a_mechanistic_profile"]["seed"])


def test_unknown_profile_raises_value_error() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=7))

    with pytest.raises(ValueError):
        build_synthetic_dataset_run(
            record,
            n_samples=20,
            random_seed=99,
            mechanistic_profile="not_a_real_profile",
        )


def test_r9l_profile_records_controlled_combination_contract() -> None:
    assert R9L_REMEDIATION_PROFILES == ("r9l_diesel_residual_damping_clean_attenuation_v1",)
    assert "r9l_diesel_residual_damping_clean_attenuation_v1" in ALL_REMEDIATION_PROFILES
    run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9l_diesel_source(seed=20260501)),
        n_samples=18,
        random_seed=20260501,
        remediation_profile="r9l_diesel_residual_damping_clean_attenuation_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9l_diesel_residual_damping_clean_attenuation_v1"
    assert audit["scope"] == ("bench_only_r9l_diesel_residual_damping_clean_attenuation_remediation")
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
        [1180.0, 46.0, 0.6],
        [1425.0, 54.0, 0.7],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["support_reference_attenuation_factor_range"] == [0.97, 0.985]
    assert params["support_reference_attenuation_support_nm"] == [750.0, 1550.0]
    assert params["support_reference_attenuation_application_stage"] == ("after_r3d_output_clip")
    assert params["support_reference_attenuation_only"] is False
    assert params["support_reference_attenuation_route_key"] == ("_r9l_diesel_residual_damping_clean_attenuation_route")
    assert "continuum_hump_center_nm" not in params
    assert "pre_offset_reference_attenuation_factor_range" not in params
    for key, expected in (
        ("diesel_residual_damping_clean_attenuation_adds_damping", True),
        ("diesel_residual_damping_clean_attenuation_adds_clean_attenuation", True),
        ("diesel_residual_damping_clean_attenuation_adds_continuum_hump", False),
        ("diesel_residual_damping_clean_attenuation_adds_pre_offset_attenuation", False),
        ("diesel_residual_damping_clean_attenuation_changes_ch_centers", False),
        ("diesel_residual_damping_clean_attenuation_changes_ch_width_gain", False),
        ("diesel_residual_damping_clean_attenuation_integration", False),
        ("diesel_residual_damping_clean_attenuation_no_calibration", True),
        ("diesel_residual_damping_clean_attenuation_no_real_stats", True),
        ("diesel_residual_damping_clean_attenuation_no_pca", True),
        ("diesel_residual_damping_clean_attenuation_no_noise_capture", True),
        ("diesel_residual_damping_clean_attenuation_no_ml_dl", True),
        (
            "diesel_residual_damping_clean_attenuation_no_labels_targets_splits",
            True,
        ),
        (
            "diesel_residual_damping_clean_attenuation_no_threshold_metric_mutation",
            True,
        ),
    ):
        assert params[key] is expected


def test_r9l_equals_r9j_off_support_and_attenuates_support_only() -> None:
    seed = 20260502
    r9j_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9j_diesel_source(seed=seed)),
        n_samples=18,
        random_seed=seed,
        remediation_profile="r9j_diesel_residual_damping_isolation_v1",
    )
    r9l_run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9l_diesel_source(seed=seed)),
        n_samples=18,
        random_seed=seed,
        remediation_profile="r9l_diesel_residual_damping_clean_attenuation_v1",
    )

    support = (r9l_run.wavelengths >= 750.0) & (r9l_run.wavelengths <= 1550.0)
    np.testing.assert_array_equal(r9l_run.y, r9j_run.y)
    np.testing.assert_array_equal(r9l_run.X[:, ~support], r9j_run.X[:, ~support])
    assert np.any(r9l_run.X[:, support] != r9j_run.X[:, support])
    ratio = np.divide(
        r9l_run.X[:, support],
        r9j_run.X[:, support],
        out=np.ones_like(r9l_run.X[:, support]),
        where=r9j_run.X[:, support] != 0.0,
    )
    nonzero_ratio = ratio[r9j_run.X[:, support] != 0.0]
    assert float(nonzero_ratio.min()) >= 0.970 - 1e-12
    assert float(nonzero_ratio.max()) <= 0.985 + 1e-12


def test_r9l_unmarked_or_non_compliant_diesel_falls_back_to_r3d() -> None:
    seed = 20260503
    r3d_source = _r9l_diesel_source(seed=seed)
    r3d_source.pop("_r9l_diesel_residual_damping_clean_attenuation_route")
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(r3d_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9l_unmarked = build_synthetic_dataset_run(
        canonicalize_prior_config(r3d_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r9l_diesel_residual_damping_clean_attenuation_v1",
    )
    np.testing.assert_array_equal(r9l_unmarked.X, r3d_run.X)
    assert r9l_unmarked.metadata["r2c_mechanistic_remediation"] == (r3d_run.metadata["r2c_mechanistic_remediation"])

    non_compliant_source = _r9l_diesel_source(seed=seed, compliant=False)
    r3d_non_compliant = build_synthetic_dataset_run(
        canonicalize_prior_config(non_compliant_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9l_non_compliant = build_synthetic_dataset_run(
        canonicalize_prior_config(non_compliant_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r9l_diesel_residual_damping_clean_attenuation_v1",
    )
    np.testing.assert_array_equal(r9l_non_compliant.X, r3d_non_compliant.X)
    assert r9l_non_compliant.metadata["r2c_mechanistic_remediation"] == (r3d_non_compliant.metadata["r2c_mechanistic_remediation"])


def test_r9m_profile_records_final_controlled_combination_contract() -> None:
    assert R9M_REMEDIATION_PROFILES == ("r9m_diesel_width_gain_damping_clean_attenuation_v1",)
    assert "r9m_diesel_width_gain_damping_clean_attenuation_v1" in ALL_REMEDIATION_PROFILES
    run = build_synthetic_dataset_run(
        canonicalize_prior_config(_r9m_diesel_source(seed=20260501)),
        n_samples=18,
        random_seed=20260501,
        remediation_profile="r9m_diesel_width_gain_damping_clean_attenuation_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r9m_diesel_width_gain_damping_clean_attenuation_v1"
    assert audit["scope"] == ("bench_only_r9m_diesel_width_gain_damping_clean_attenuation_remediation")
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
    assert params["damping_windows_nm"] == [
        [1180.0, 46.0, 0.6],
        [1425.0, 54.0, 0.7],
    ]
    assert params["damping_strength_range"] == [0.05, 0.15]
    assert params["support_reference_attenuation_factor_range"] == [0.97, 0.985]
    assert params["support_reference_attenuation_support_nm"] == [750.0, 1550.0]
    assert params["support_reference_attenuation_application_stage"] == ("after_r3d_output_clip")
    assert params["support_reference_attenuation_only"] is False
    assert params["support_reference_attenuation_route_key"] == ("_r9m_diesel_width_gain_damping_clean_attenuation_route")
    assert "continuum_hump_center_nm" not in params
    assert "pre_offset_reference_attenuation_factor_range" not in params
    for key, expected in (
        ("diesel_width_gain_damping_clean_attenuation_changes_ch_centers", False),
        ("diesel_width_gain_damping_clean_attenuation_changes_ch_width_gain", True),
        ("diesel_width_gain_damping_clean_attenuation_adds_damping", True),
        ("diesel_width_gain_damping_clean_attenuation_adds_clean_attenuation", True),
        ("diesel_width_gain_damping_clean_attenuation_adds_continuum_hump", False),
        (
            "diesel_width_gain_damping_clean_attenuation_adds_pre_offset_attenuation",
            False,
        ),
        ("diesel_width_gain_damping_clean_attenuation_adds_support_intercept", False),
        ("diesel_width_gain_damping_clean_attenuation_adds_support_shape", False),
        ("diesel_width_gain_damping_clean_attenuation_adds_redistribution", False),
        ("diesel_width_gain_damping_clean_attenuation_readout_transform", False),
        ("diesel_width_gain_damping_clean_attenuation_extra_guard_clip", False),
        ("diesel_width_gain_damping_clean_attenuation_integration", False),
        ("diesel_width_gain_damping_clean_attenuation_no_calibration", True),
        ("diesel_width_gain_damping_clean_attenuation_no_real_stats", True),
        ("diesel_width_gain_damping_clean_attenuation_no_pca", True),
        ("diesel_width_gain_damping_clean_attenuation_no_noise_capture", True),
        ("diesel_width_gain_damping_clean_attenuation_no_ml_dl", True),
        (
            "diesel_width_gain_damping_clean_attenuation_no_labels_targets_splits",
            True,
        ),
        (
            "diesel_width_gain_damping_clean_attenuation_no_threshold_metric_mutation",
            True,
        ),
    ):
        assert params[key] is expected


def test_r9m_unmarked_or_non_compliant_diesel_falls_back_to_r3d() -> None:
    seed = 20260503
    r3d_source = _r9m_diesel_source(seed=seed)
    r3d_source.pop("_r9m_diesel_width_gain_damping_clean_attenuation_route")
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(r3d_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9m_unmarked = build_synthetic_dataset_run(
        canonicalize_prior_config(r3d_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r9m_diesel_width_gain_damping_clean_attenuation_v1",
    )
    np.testing.assert_array_equal(r9m_unmarked.X, r3d_run.X)
    assert r9m_unmarked.metadata["r2c_mechanistic_remediation"] == (r3d_run.metadata["r2c_mechanistic_remediation"])

    non_compliant_source = _r9m_diesel_source(seed=seed, compliant=False)
    r3d_non_compliant = build_synthetic_dataset_run(
        canonicalize_prior_config(non_compliant_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9m_non_compliant = build_synthetic_dataset_run(
        canonicalize_prior_config(non_compliant_source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r9m_diesel_width_gain_damping_clean_attenuation_v1",
    )
    np.testing.assert_array_equal(r9m_non_compliant.X, r3d_non_compliant.X)
    assert r9m_non_compliant.metadata["r2c_mechanistic_remediation"] == (r3d_non_compliant.metadata["r2c_mechanistic_remediation"])


def test_r9m_non_diesel_falls_back_to_r3d() -> None:
    seed = 20260504
    source = _valid_source("grain", seed=seed)
    r3d_run = build_synthetic_dataset_run(
        canonicalize_prior_config(source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r3d_diesel_matrix_v1",
    )
    r9m_run = build_synthetic_dataset_run(
        canonicalize_prior_config(source),
        n_samples=12,
        random_seed=seed,
        remediation_profile="r9m_diesel_width_gain_damping_clean_attenuation_v1",
    )
    np.testing.assert_array_equal(r9m_run.X, r3d_run.X)
    assert r9m_run.metadata["r2c_mechanistic_remediation"] == (
        r3d_run.metadata["r2c_mechanistic_remediation"]
    )


def _assert_audit_non_oracle(audit: dict[str, object]) -> None:
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
    assert audit["scope"] == "bench_only_r2a_sentinel_mechanistic_ablation"
