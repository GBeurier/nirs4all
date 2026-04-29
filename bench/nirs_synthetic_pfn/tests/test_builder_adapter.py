from __future__ import annotations

import numpy as np
import pytest
from nirsyntheticpfn.adapters.builder_adapter import (
    PriorDatasetAdapterError,
    build_synthetic_dataset_run,
    prior_to_builder_config,
)
from nirsyntheticpfn.adapters.prior_adapter import (
    canonicalize_domain,
    canonicalize_prior_config,
    sample_canonical_prior,
)

from nirs4all.synthesis.domains import get_domain_config


def test_regression_dataset_run_shape_finite_range_seed_and_metadata() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=42))

    first = build_synthetic_dataset_run(record, n_samples=24, random_seed=31415)
    second = build_synthetic_dataset_run(record, n_samples=24, random_seed=31415)

    assert first.X.shape == (24, first.wavelengths.size)
    assert first.y.shape == (24,)
    assert np.isfinite(first.X).all()
    assert np.isfinite(first.y).all()
    assert np.all(np.diff(first.wavelengths) > 0)
    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.y, second.y)
    np.testing.assert_allclose(first.wavelengths, second.wavelengths)

    target_min, target_max = first.builder_config["target"]["range"]
    assert float(np.min(first.y)) >= target_min
    assert float(np.max(first.y)) <= target_max
    assert first.validation_summary["status"] == "passed"
    assert first.validation_summary["checks"]["concentrations_row_normalized"] is True
    np.testing.assert_allclose(
        np.sum(np.asarray(first.latent_metadata["concentrations"]), axis=1),
        1.0,
    )
    assert first.latent_metadata["concentration_transform"]["row_normalized"] is True
    assert first.metadata["domain"]["key"] == "agriculture_grain"
    assert first.metadata["instrument"]["key"] == "foss_xds"
    assert first.metadata["mode"] == "reflectance"
    assert first.metadata["target"]["type"] == "regression"
    assert first.metadata["nuisance"]["matrix_type"] == "solid"
    assert first.metadata["prior_config"]["source_prior_config"]["domain"] == "grain"
    assert "validation_summary" in first.metadata
    assert "builder_config" in first.metadata


def test_classification_dataset_run_labels_and_metadata() -> None:
    source = _valid_source("tablets", seed=7)
    source["target_config"] = {
        "type": "classification",
        "n_classes": 3,
        "separation": "moderate",
    }
    record = canonicalize_prior_config(source)

    run = build_synthetic_dataset_run(record, n_samples=36, random_seed=2718)

    labels = set(np.unique(run.y).astype(int).tolist())
    assert labels.issubset({0, 1, 2})
    assert len(labels) >= 2
    assert labels == {0, 1, 2}
    assert run.metadata["target"]["type"] == "classification"
    assert run.metadata["target"]["n_classes"] == 3
    assert run.metadata["target"]["separation_method"] == "composition_quantile"
    assert run.validation_summary["checks"]["target_contract"] is True


def test_repaired_prior_wavelength_grid_is_bounded_by_effective_range() -> None:
    record = sample_canonical_prior(random_state=4, repair_domain_components=True)

    run = build_synthetic_dataset_run(record, n_samples=24, random_seed=4)

    expected_low, expected_high = run.builder_config["features"]["wavelength_range"]
    assert float(run.wavelengths[0]) >= expected_low
    assert float(run.wavelengths[-1]) <= expected_high
    assert run.validation_summary["checks"]["wavelengths_monotonic"] is True


def test_repaired_prior_classification_emits_declared_classes() -> None:
    record = sample_canonical_prior(random_state=24, repair_domain_components=True)

    run = build_synthetic_dataset_run(record, n_samples=24, random_seed=24)

    assert run.metadata["target"]["type"] == "classification"
    assert set(np.unique(run.y).astype(int).tolist()) == set(
        range(run.metadata["target"]["n_classes"])
    )
    assert run.validation_summary["checks"]["target_contract"] is True


def test_classification_requires_at_least_one_sample_per_declared_class() -> None:
    source = _valid_source("tablets", seed=7)
    source["target_config"] = {
        "type": "classification",
        "n_classes": 4,
        "separation": "moderate",
    }
    record = canonicalize_prior_config(source)

    with pytest.raises(PriorDatasetAdapterError) as exc:
        prior_to_builder_config(record, n_samples=3, random_seed=7)

    assert exc.value.validation_summary["failures"][0]["reason"] == "invalid_n_samples"


def test_prior_to_builder_config_requires_supported_target_mapping() -> None:
    record = canonicalize_prior_config(_valid_source("grain", seed=42))
    broken_record = record.__class__(
        **{
            **record.to_dict(),
            "component_keys": tuple(record.component_keys),
            "target_prior": {
                **record.target_prior,
                "executable_mapping": "not_executable",
            },
        }
    )

    with pytest.raises(PriorDatasetAdapterError) as exc:
        prior_to_builder_config(broken_record, n_samples=8, random_seed=1)

    assert exc.value.validation_summary["status"] == "failed"
    assert exc.value.validation_summary["failures"][0]["reason"] == "unsupported_target_mapping"


def test_a1_provenance_repairs_are_preserved_in_metadata() -> None:
    source = _valid_source("blood", seed=99)
    source["_raw_prior_config"] = {"domain": "blood", "components": ["water", "protein"]}
    source["_canonical_repairs"] = {
        "components_resampled_from_canonical_domain": {
            "raw_domain": "blood",
            "canonical_domain": "biomedical_tissue",
        }
    }
    record = canonicalize_prior_config(source)

    run = build_synthetic_dataset_run(record, n_samples=12, random_seed=99)

    provenance = run.metadata["provenance_a1"]
    assert provenance["_raw_prior_config"]["domain"] == "blood"
    assert (
        provenance["_canonical_repairs"]["components_resampled_from_canonical_domain"][
            "canonical_domain"
        ]
        == "biomedical_tissue"
    )


def _valid_source(domain_alias: str, *, seed: int) -> dict[str, object]:
    domain_key = canonicalize_domain(domain_alias)
    components = _first_valid_domain_components(domain_key, 3)
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
        "target_config": {
            "type": "regression",
            "n_targets": 1,
            "nonlinearity": "none",
        },
        "random_state": seed,
    }


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    from nirs4all.synthesis.components import get_component

    components = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n_components:
            return components
    raise AssertionError(f"Not enough valid components for {domain_key}")
