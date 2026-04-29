from __future__ import annotations

from dataclasses import replace

import numpy as np
from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun, build_synthetic_dataset_run
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config
from nirsyntheticpfn.evaluation.prior_checks import (
    PHASE_A_GATE_OVERRIDE,
    validate_prior_predictive_run,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config


def test_prior_predictive_validation_passes_valid_regression_run() -> None:
    run = _build_run("grain", target_type="regression", target_size=1, seed=42)

    validation = validate_prior_predictive_run(run, preset="grain")

    assert validation.validation_status == "passed"
    assert validation.downstream_training_status == "allowed"
    assert validation.phase_a_gate_override == PHASE_A_GATE_OVERRIDE
    assert validation.failed_or_blocking_checks == []
    statuses = {check.name: check.status for check in validation.checks}
    assert statuses["concentration_sums_and_ranges"] == "passed"
    assert statuses["target_distribution"] == "passed"
    assert statuses["spectral_snr"] == "passed"
    assert statuses["derivative_statistics"] == "passed"
    assert statuses["baseline_curvature"] == "passed"
    assert statuses["peak_density"] == "passed"


def test_prior_predictive_validation_blocks_bad_concentration_sums() -> None:
    run = _build_run("grain", target_type="regression", target_size=1, seed=43)
    latent_metadata = dict(run.latent_metadata)
    latent_metadata["concentrations"] = np.asarray(latent_metadata["concentrations"]) * 0.5
    broken = replace(run, latent_metadata=latent_metadata)

    validation = validate_prior_predictive_run(broken, preset="bad_concentrations")

    assert validation.validation_status == "blocked"
    assert validation.downstream_training_status == "blocked"
    assert "concentration_sums_and_ranges" in validation.failed_or_blocking_checks


def test_prior_predictive_validation_blocks_imbalanced_classification() -> None:
    run = _build_run("tablets", target_type="classification", target_size=3, seed=44)
    broken = replace(run, y=np.zeros_like(run.y))

    validation = validate_prior_predictive_run(broken, preset="bad_classes")

    assert validation.validation_status == "blocked"
    assert "target_distribution" in validation.failed_or_blocking_checks
    target_check = next(check for check in validation.checks if check.name == "target_distribution")
    assert target_check.status == "failed"


def test_prior_predictive_validation_blocks_non_integer_class_labels() -> None:
    run = _build_run("tablets", target_type="classification", target_size=3, seed=46)
    broken = replace(run, y=run.y.astype(float) + 0.25)

    validation = validate_prior_predictive_run(broken, preset="bad_class_labels")

    assert validation.validation_status == "blocked"
    assert validation.downstream_training_status == "blocked"
    target_check = next(check for check in validation.checks if check.name == "target_distribution")
    assert target_check.status == "failed"
    assert "non-integer" in target_check.message


def test_prior_predictive_validation_blocks_unsupported_checks() -> None:
    run = _build_run("grain", target_type="regression", target_size=1, seed=47)
    latent_metadata = dict(run.latent_metadata)
    latent_metadata.pop("concentrations")
    broken = replace(run, latent_metadata=latent_metadata)

    validation = validate_prior_predictive_run(broken, preset="missing_concentrations")

    assert validation.validation_status == "blocked"
    assert validation.downstream_training_status == "blocked"
    assert "concentration_sums_and_ranges" in validation.failed_or_blocking_checks
    assert "concentration_sums_and_ranges" in validation.summary["unsupported_checks"]


def test_prior_predictive_validation_returns_blocked_for_invalid_empty_wavelength_grid() -> None:
    run = _build_run("grain", target_type="regression", target_size=1, seed=48)
    broken = replace(run, wavelengths=np.asarray([], dtype=float))

    validation = validate_prior_predictive_run(broken, preset="bad_wavelengths")

    assert validation.validation_status == "blocked"
    assert validation.downstream_training_status == "blocked"
    assert validation.summary["wavelength_range_nm"] is None
    assert "wavelengths_and_mode" in validation.failed_or_blocking_checks
    assert "spectral_statistics" in validation.failed_or_blocking_checks
    assert "spectral_statistics" in validation.summary["unsupported_checks"]


def test_prior_predictive_validation_checks_declared_nonlinearity() -> None:
    run = _build_run(
        "grain",
        target_type="regression",
        target_size=1,
        seed=45,
        nonlinearity="moderate",
        n_samples=80,
    )

    validation = validate_prior_predictive_run(run, preset="nonlinear_grain")

    nonlinear_check = next(check for check in validation.checks if check.name == "nonlinear_target_behavior")
    assert validation.validation_status == "passed"
    assert nonlinear_check.status == "passed"
    assert nonlinear_check.metrics["linear_residual_ratio"] >= 0.02


def _build_run(
    domain_alias: str,
    *,
    target_type: str,
    target_size: int,
    seed: int,
    nonlinearity: str = "none",
    n_samples: int = 40,
) -> SyntheticDatasetRun:
    source = _valid_source(
        domain_alias,
        target_type=target_type,
        target_size=target_size,
        seed=seed,
        nonlinearity=nonlinearity,
    )
    record = canonicalize_prior_config(source)
    return build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)


def _valid_source(
    domain_alias: str,
    *,
    target_type: str,
    target_size: int,
    seed: int,
    nonlinearity: str,
) -> dict[str, object]:
    domain_key = canonicalize_domain(domain_alias)
    components = _first_valid_domain_components(domain_key, max(3, target_size))
    if target_type == "classification":
        target_config: dict[str, object] = {
            "type": "classification",
            "n_classes": target_size,
            "separation": "moderate",
        }
    else:
        target_config = {
            "type": "regression",
            "n_targets": target_size,
            "nonlinearity": nonlinearity,
        }
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
        "target_config": target_config,
        "random_state": seed,
    }


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    components = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n_components:
            return components
    raise AssertionError(f"Not enough valid components for {domain_key}")
