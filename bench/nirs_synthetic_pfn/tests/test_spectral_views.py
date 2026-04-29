"""Phase C2 contract tests for ``SpectralViewBatch``."""

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
from nirsyntheticpfn.data import (
    CanonicalLatentBatch,
    SpectralViewBatch,
    SpectralViewBatchError,
)

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


def _build_run(domain_alias: str = "grain", seed: int = 31415, n_samples: int = 16):
    record = canonicalize_prior_config(_valid_source(domain_alias, seed=7))
    return build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)


def _build_pair(seed: int = 31415, n_samples: int = 16):
    run = _build_run(seed=seed, n_samples=n_samples)
    latents = CanonicalLatentBatch.from_synthetic_dataset_run(run)
    return run, latents


def test_from_synthetic_dataset_run_populates_contract() -> None:
    run, latents = _build_pair()

    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    n = run.X.shape[0]
    assert views.X.shape == (n, run.wavelengths.size)
    np.testing.assert_array_equal(views.X, np.asarray(run.X, dtype=float))
    np.testing.assert_array_equal(
        views.wavelengths, np.asarray(run.wavelengths, dtype=float)
    )
    assert views.latent_ids == latents.latent_ids
    assert len(views.view_ids) == n
    assert len(set(views.view_ids)) == n
    assert views.instrument_key == run.builder_config["features"]["instrument"]
    assert views.measurement_mode == run.builder_config["features"]["measurement_mode"]
    assert views.view_config["phase"] == "C2"
    assert views.view_config["instrument_key"] == views.instrument_key
    assert views.view_config["measurement_mode"] == views.measurement_mode
    assert views.preprocessing_state["preprocessing_applied"] is False
    assert views.noise_state["noise_added_in_view"] is False
    assert views.noise_state["source_noise_level"] == float(
        run.builder_config["nuisance"]["noise_level"]
    )
    assert views.metadata["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert views.metadata["claims"]["realism"] is False
    assert views.metadata["claims"]["transfer"] is False
    assert views.metadata["source_contract"] == "CanonicalLatentBatch"
    assert views.metadata["source_latent_count"] == n
    assert "a2_spectral_validation_summary" in views.metadata
    assert "a2_validation_summary" not in views.metadata


def test_to_dict_roundtrip_contains_X_and_wavelengths_as_lists() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    payload = views.to_dict()

    assert isinstance(payload["X"], list)
    assert len(payload["X"]) == len(views.latent_ids)
    assert isinstance(payload["wavelengths"], list)
    assert len(payload["wavelengths"]) == int(views.wavelengths.size)
    assert payload["latent_ids"] == list(views.latent_ids)
    assert payload["view_ids"] == list(views.view_ids)


def test_assert_aligned_to_passes_for_same_run() -> None:
    run, latents = _build_pair(seed=5, n_samples=10)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    views.assert_aligned_to(latents)


def test_assert_aligned_to_rejects_misaligned_latents() -> None:
    run, latents = _build_pair(seed=5, n_samples=10)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)
    other = latents.subset([2, 4, 6])

    with pytest.raises(SpectralViewBatchError) as exc:
        views.assert_aligned_to(other)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "alignment_mismatch" in reasons


def test_view_ids_are_deterministic_across_identical_runs() -> None:
    run_a, latents_a = _build_pair(seed=20260429, n_samples=12)
    run_b, latents_b = _build_pair(seed=20260429, n_samples=12)

    views_a = SpectralViewBatch.from_synthetic_dataset_run(run_a, latents_a)
    views_b = SpectralViewBatch.from_synthetic_dataset_run(run_b, latents_b)

    assert views_a.view_ids == views_b.view_ids
    assert views_a.latent_ids == views_b.latent_ids
    assert len(set(views_a.view_ids)) == len(views_a.view_ids)


def test_view_ids_change_when_view_config_changes() -> None:
    run, latents = _build_pair(seed=11, n_samples=8)

    default_views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)
    custom_views = SpectralViewBatch.from_synthetic_dataset_run(
        run,
        latents,
        view_config={
            "phase": "C2",
            "view_kind": "alt",
            "tag": "alt",
        },
    )

    assert default_views.view_ids != custom_views.view_ids
    assert len(set(custom_views.view_ids)) == len(custom_views.view_ids)


def test_from_run_rejects_latent_count_mismatch() -> None:
    run, latents = _build_pair(seed=5, n_samples=10)
    smaller = latents.subset([0, 1, 2])

    with pytest.raises(SpectralViewBatchError) as exc:
        SpectralViewBatch.from_synthetic_dataset_run(run, smaller)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("shape_mismatch", "latent_ids") in failures


def test_from_run_rejects_latents_from_different_run_same_count() -> None:
    run, _latents = _build_pair(seed=5, n_samples=10)
    _other_run, other_latents = _build_pair(seed=6, n_samples=10)

    with pytest.raises(SpectralViewBatchError) as exc:
        SpectralViewBatch.from_synthetic_dataset_run(run, other_latents)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("alignment_mismatch", "latent_ids") in failures


def test_validation_rejects_non_2d_X() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad_X = views.X.reshape(-1)
    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, X=bad_X)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("shape_mismatch", "X") in failures


def test_validation_rejects_non_finite_X() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad = views.X.copy()
    bad[0, 0] = np.nan

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, X=bad)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "X") in failures


def test_validation_rejects_non_finite_wavelengths() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad_w = views.wavelengths.copy()
    bad_w[1] = np.inf

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, wavelengths=bad_w)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "wavelengths") in failures


def test_validation_rejects_non_monotonic_wavelengths() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad_w = views.wavelengths.copy()
    bad_w[2] = bad_w[1] - 10.0

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, wavelengths=bad_w)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "non_monotonic_wavelengths" in reasons


def test_validation_rejects_wavelengths_size_mismatch() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, wavelengths=views.wavelengths[:-1])
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("shape_mismatch", "wavelengths") in failures


def test_validation_rejects_invalid_id_lengths() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc_latent:
        replace(views, latent_ids=views.latent_ids[:-1])
    with pytest.raises(SpectralViewBatchError) as exc_view:
        replace(views, view_ids=views.view_ids[:-1])

    latent_failures = {(f["reason"], f["field"]) for f in exc_latent.value.failures}
    view_failures = {(f["reason"], f["field"]) for f in exc_view.value.failures}
    assert ("shape_mismatch", "latent_ids") in latent_failures
    assert ("shape_mismatch", "view_ids") in view_failures


def test_validation_rejects_duplicate_view_ids() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    duplicated = list(views.view_ids)
    duplicated[1] = duplicated[0]

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, view_ids=tuple(duplicated))

    reasons = {f["reason"] for f in exc.value.failures}
    assert "duplicate_view_ids" in reasons


@pytest.mark.parametrize(
    "leak_key",
    [
        "y",
        "target",
        "target_clean",
        "target_noisy",
        "concentrations",
        "latent_features",
    ],
)
def test_validation_rejects_metadata_leakage(leak_key: str) -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad_metadata = dict(views.metadata)
    bad_metadata[leak_key] = "not allowed"

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, metadata=bad_metadata)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "metadata_leakage" in reasons


@pytest.mark.parametrize(
    ("field_name", "payload"),
    [
        ("metadata", {"safe": {"target": [1.0, 2.0]}}),
        ("view_config", {"phase": "C2", "nested": {"y_min": 0.0}}),
        ("preprocessing_state", {"phase": "C2", "latent_features": [0.0]}),
        ("noise_state", {"phase": "C2", "concentration_row_sum_min": 1.0}),
    ],
)
def test_validation_rejects_nested_namespace_leakage(
    field_name: str,
    payload: dict[str, Any],
) -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc:
        if field_name == "metadata":
            replace(views, metadata=payload)
        elif field_name == "view_config":
            replace(views, view_config=payload)
        elif field_name == "preprocessing_state":
            replace(views, preprocessing_state=payload)
        elif field_name == "noise_state":
            replace(views, noise_state=payload)
        else:
            raise AssertionError(f"unexpected field_name={field_name}")

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", field_name) in failures


def test_a2_spectral_validation_summary_does_not_leak_targets_or_latents() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    summary = views.metadata["a2_spectral_validation_summary"]
    serialized = str(summary)

    assert "y_" not in serialized
    assert "target" not in serialized
    assert "concentration" not in serialized
    assert summary["summary"]["X_shape"] == list(run.X.shape)
    assert summary["summary"]["wavelength_range_nm"] == [
        float(run.wavelengths[0]),
        float(run.wavelengths[-1]),
    ]


def test_validation_rejects_missing_risk_gates() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    bad_metadata = dict(views.metadata)
    bad_metadata["risk_gates"] = {"A3_failed_documented": True}

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, metadata=bad_metadata)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "missing_risk_gates" in reasons


def test_validation_rejects_empty_required_dicts() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, view_config={}, preprocessing_state={})

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("empty_metadata", "view_config") in failures
    assert ("empty_metadata", "preprocessing_state") in failures


def test_validation_rejects_invalid_instrument_or_mode() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc:
        replace(views, instrument_key="", measurement_mode="")
    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_instrument_key" in reasons
    assert "invalid_measurement_mode" in reasons


def test_subset_slices_arrays_and_ids() -> None:
    run, latents = _build_pair(seed=5, n_samples=12)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    indices = [0, 3, 5]
    sub = views.subset(indices)

    assert len(sub.latent_ids) == len(indices)
    assert sub.latent_ids == tuple(views.latent_ids[i] for i in indices)
    assert sub.view_ids == tuple(views.view_ids[i] for i in indices)
    np.testing.assert_array_equal(sub.X, views.X[np.asarray(indices)])
    np.testing.assert_array_equal(sub.wavelengths, views.wavelengths)


def test_subset_rejects_out_of_range_indices() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError):
        views.subset([0, 99])


@pytest.mark.parametrize(
    "indices",
    [
        [0.0, 1.0],
        np.asarray([[0, 1], [2, 3]]),
    ],
)
def test_subset_rejects_invalid_indices(indices: Any) -> None:
    run, latents = _build_pair(seed=5, n_samples=8)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    with pytest.raises(SpectralViewBatchError) as exc:
        views.subset(indices)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_indices" in reasons


def test_to_light_dict_omits_X() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)

    light = views.to_light_dict()

    assert "X" not in light
    assert light["n_samples"] == len(views.latent_ids)
    assert light["n_wavelengths"] == int(views.wavelengths.size)
    assert light["latent_ids"] == list(views.latent_ids)
    assert light["view_ids"] == list(views.view_ids)
    assert light["view_config"]["phase"] == "C2"
    assert light["metadata"]["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
