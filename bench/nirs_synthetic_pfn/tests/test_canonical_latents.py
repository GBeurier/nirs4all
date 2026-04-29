"""Phase C1 contract tests for ``CanonicalLatentBatch``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from nirsyntheticpfn.adapters.builder_adapter import build_synthetic_dataset_run
from nirsyntheticpfn.adapters.prior_adapter import (
    canonicalize_domain,
    canonicalize_prior_config,
)
from nirsyntheticpfn.data import CanonicalLatentBatch, CanonicalLatentBatchError

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config


def _valid_source(domain_alias: str, *, seed: int) -> dict[str, Any]:
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
        raise AssertionError(f"Not enough valid components for {domain_key}")
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


def _build_run(domain_alias: str = "grain", seed: int = 31415, n_samples: int = 24):
    record = canonicalize_prior_config(_valid_source(domain_alias, seed=7))
    return build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)


def test_from_synthetic_dataset_run_populates_contract() -> None:
    run = _build_run()

    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    n = run.X.shape[0]
    assert len(batch.latent_ids) == n
    assert batch.concentrations.shape == (n, len(run.latent_metadata["component_keys"]))
    np.testing.assert_allclose(batch.concentrations.sum(axis=1), 1.0, atol=1e-6)
    assert batch.latent_features.shape == (n, len(batch.latent_feature_names))
    assert "temperature_c" in batch.latent_feature_names
    assert "particle_size_um" in batch.latent_feature_names
    assert "noise_level" in batch.latent_feature_names
    assert batch.target_clean.shape == batch.target_noisy.shape
    np.testing.assert_array_equal(batch.target_clean, batch.target_noisy)
    assert len(batch.batch_ids) == n
    assert len(batch.group_ids) == n
    assert batch.split_labels is None
    assert batch.target_metadata["target_clean_equals_target_noisy"] is True
    assert batch.view_metadata == {
        "intended_contract": "SpectralViewBatch",
        "rendered_view_count": 0,
        "note": "C1 only declares intent; spectral views are produced by C2.",
    }
    assert batch.provenance["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert batch.provenance["claims"]["realism"] is False
    assert batch.provenance["claims"]["transfer"] is False
    assert batch.domain_metadata["domain_key"] == "agriculture_grain"
    assert batch.instrument_metadata["instrument_key"] == "foss_xds"
    assert batch.instrument_metadata["measurement_mode"] == "reflectance"
    assert batch.component_metadata["spectra_reference_source"] == "builder_config.features.components"
    assert batch.component_metadata["spectra_reference_keys"] == run.builder_config["features"]["components"]
    assert batch.optical_metadata["measurement_mode"] == "reflectance"
    assert "optical_geometry_source" in batch.optical_metadata


def test_latent_ids_are_deterministic_across_identical_runs() -> None:
    run_a = _build_run(seed=20260429, n_samples=16)
    run_b = _build_run(seed=20260429, n_samples=16)

    batch_a = CanonicalLatentBatch.from_synthetic_dataset_run(run_a)
    batch_b = CanonicalLatentBatch.from_synthetic_dataset_run(run_b)

    assert batch_a.latent_ids == batch_b.latent_ids
    assert len(set(batch_a.latent_ids)) == len(batch_a.latent_ids)


def test_latent_ids_change_when_seed_changes() -> None:
    run_a = _build_run(seed=11)
    run_b = _build_run(seed=22)

    batch_a = CanonicalLatentBatch.from_synthetic_dataset_run(run_a)
    batch_b = CanonicalLatentBatch.from_synthetic_dataset_run(run_b)

    assert batch_a.latent_ids != batch_b.latent_ids


def test_subset_with_split_label_propagates_label_and_arrays() -> None:
    run = _build_run(seed=5, n_samples=20)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    indices = [0, 3, 7, 9]
    sub = batch.subset(indices, split_label="train")

    assert len(sub.latent_ids) == len(indices)
    assert sub.latent_ids == tuple(batch.latent_ids[i] for i in indices)
    assert sub.split_labels == tuple(["train"] * len(indices))
    np.testing.assert_allclose(sub.concentrations, batch.concentrations[indices])
    np.testing.assert_allclose(sub.latent_features, batch.latent_features[indices])
    np.testing.assert_array_equal(sub.target_clean, batch.target_clean[np.asarray(indices)])
    np.testing.assert_array_equal(sub.target_noisy, batch.target_noisy[np.asarray(indices)])
    assert sub.batch_ids == tuple(batch.batch_ids[i] for i in indices)


def test_subset_rejects_out_of_range_indices() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError):
        batch.subset([0, 99])


@pytest.mark.parametrize(
    "indices",
    [
        [0.0, 1.0],
        np.asarray([[0, 1], [2, 3]]),
    ],
)
def test_subset_rejects_non_integer_or_non_1d_indices(indices: Any) -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        batch.subset(indices)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_indices" in reasons


def test_from_run_rejects_mismatched_batch_ids_without_padding() -> None:
    run = _build_run(seed=5, n_samples=8)
    latent_metadata = dict(run.latent_metadata)
    latent_metadata["batch_ids"] = [0, 1]
    bad_run = replace(run, latent_metadata=latent_metadata)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        CanonicalLatentBatch.from_synthetic_dataset_run(bad_run)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("shape_mismatch", "batch_ids") in failures
    assert ("shape_mismatch", "group_ids") in failures


def test_from_run_rejects_non_numeric_concentrations_as_contract_error() -> None:
    run = _build_run(seed=5, n_samples=8)
    latent_metadata = dict(run.latent_metadata)
    latent_metadata["concentrations"] = [["bad"] * len(run.latent_metadata["component_keys"])] * 8
    bad_run = replace(run, latent_metadata=latent_metadata)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        CanonicalLatentBatch.from_synthetic_dataset_run(bad_run)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_numeric", "concentrations") in failures


def test_validation_rejects_concentration_row_sum_violation() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    bad_concentrations = batch.concentrations.copy()
    bad_concentrations[0] = bad_concentrations[0] * 0.5

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, concentrations=bad_concentrations)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "row_sum_mismatch" in reasons


def test_validation_rejects_negative_concentrations() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    bad = batch.concentrations.copy()
    bad[0, 0] = -0.1
    bad[0] = bad[0] / bad[0].sum() if bad[0].sum() > 0 else bad[0]

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, concentrations=bad)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "negative_concentrations" in reasons or "row_sum_mismatch" in reasons


def test_validation_rejects_non_finite_concentrations() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    bad = batch.concentrations.copy()
    bad[0, 0] = np.nan

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, concentrations=bad)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "concentrations") in failures


def test_validation_rejects_non_finite_latent_features() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    bad = batch.latent_features.copy()
    bad[0, 0] = np.inf

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, latent_features=bad)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "latent_features") in failures


def test_validation_rejects_non_numeric_arrays_as_contract_errors() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as concentrations_exc:
        replace(batch, concentrations=np.asarray([["bad"] * len(batch.component_keys)] * len(batch.latent_ids), dtype=object))
    with pytest.raises(CanonicalLatentBatchError) as features_exc:
        replace(batch, latent_features=np.asarray([["bad"] * len(batch.latent_feature_names)] * len(batch.latent_ids), dtype=object))
    with pytest.raises(CanonicalLatentBatchError) as targets_exc:
        replace(batch, target_clean=np.asarray(["bad"] * len(batch.latent_ids), dtype=object))

    concentration_failures = {(f["reason"], f["field"]) for f in concentrations_exc.value.failures}
    feature_failures = {(f["reason"], f["field"]) for f in features_exc.value.failures}
    target_failures = {(f["reason"], f["field"]) for f in targets_exc.value.failures}
    assert ("non_numeric", "concentrations") in concentration_failures
    assert ("non_numeric", "latent_features") in feature_failures
    assert ("non_numeric", "target_clean") in target_failures


def test_validation_rejects_mismatched_target_shapes() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, target_noisy=batch.target_clean[:-1])

    reasons = {f["reason"] for f in exc.value.failures}
    assert "shape_mismatch" in reasons


def test_validation_rejects_non_finite_targets() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    bad = batch.target_noisy.copy()
    bad[0] = np.nan

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, target_noisy=bad)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "target_noisy") in failures


def test_validation_rejects_duplicate_latent_ids() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    duplicated = list(batch.latent_ids)
    duplicated[1] = duplicated[0]

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, latent_ids=tuple(duplicated))

    reasons = {f["reason"] for f in exc.value.failures}
    assert "duplicate_latent_ids" in reasons


def test_validation_rejects_invalid_component_and_feature_names() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as component_exc:
        replace(batch, component_keys=(batch.component_keys[0], batch.component_keys[0]))
    with pytest.raises(CanonicalLatentBatchError) as feature_exc:
        replace(
            batch,
            latent_feature_names=(
                batch.latent_feature_names[0],
                batch.latent_feature_names[0],
            ),
        )

    component_reasons = {f["reason"] for f in component_exc.value.failures}
    feature_reasons = {f["reason"] for f in feature_exc.value.failures}
    assert "duplicate_component_keys" in component_reasons
    assert "shape_mismatch" in component_reasons
    assert "duplicate_latent_feature_names" in feature_reasons
    assert "shape_mismatch" in feature_reasons


def test_validation_rejects_empty_required_metadata() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, provenance={})

    reasons = {f["reason"] for f in exc.value.failures}
    assert "empty_metadata" in reasons


def test_validation_rejects_empty_physical_metadata_namespaces() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    with pytest.raises(CanonicalLatentBatchError) as exc:
        replace(batch, optical_metadata={}, split_metadata={})

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("empty_metadata", "optical_metadata") in failures
    assert ("empty_metadata", "split_metadata") in failures


def test_to_light_dict_omits_heavy_arrays() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    light = batch.to_light_dict()

    for heavy_field in ("concentrations", "latent_features", "target_clean", "target_noisy"):
        assert heavy_field not in light
    assert light["n_samples"] == len(batch.latent_ids)
    assert light["n_components"] == len(batch.component_keys)
    assert light["n_latent_features"] == len(batch.latent_feature_names)
    assert light["latent_ids"] == list(batch.latent_ids)
    assert light["domain_metadata"]["domain_key"] == batch.domain_metadata["domain_key"]
    assert light["view_metadata"]["intended_contract"] == "SpectralViewBatch"


def test_to_dict_includes_arrays_as_lists() -> None:
    run = _build_run(seed=5, n_samples=8)
    batch = CanonicalLatentBatch.from_synthetic_dataset_run(run)

    payload = batch.to_dict()

    assert isinstance(payload["concentrations"], list)
    assert isinstance(payload["latent_features"], list)
    assert isinstance(payload["target_clean"], list)
    assert isinstance(payload["target_noisy"], list)
    assert len(payload["concentrations"]) == len(batch.latent_ids)
