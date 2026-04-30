"""R2a mechanistic profile contract tests for the bench-only builder adapter."""

from __future__ import annotations

import numpy as np
import pytest
from nirsyntheticpfn.adapters.builder_adapter import (
    R2A_MECHANISTIC_PROFILES,
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
    assert first.metadata["r2a_mechanistic_profile"]["seed"] == (
        second.metadata["r2a_mechanistic_profile"]["seed"]
    )


def test_unknown_profile_raises_value_error() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=7))

    with pytest.raises(ValueError):
        build_synthetic_dataset_run(
            record,
            n_samples=20,
            random_seed=99,
            mechanistic_profile="not_a_real_profile",
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
