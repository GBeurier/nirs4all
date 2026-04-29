"""Phase D1 contract tests for ``NIRSPriorTask``."""

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
    NIRSPriorTask,
    NIRSPriorTaskError,
    SpectralViewBatch,
    SpectralViewBatchError,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config


def _valid_source(
    domain_alias: str,
    *,
    seed: int,
    target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "target_config": target_config
        or {
            "type": "regression",
            "n_targets": 1,
            "nonlinearity": "none",
        },
        "random_state": seed,
    }


def _build_run(
    domain_alias: str = "grain",
    seed: int = 31415,
    n_samples: int = 16,
    target_config: dict[str, Any] | None = None,
):
    record = canonicalize_prior_config(
        _valid_source(domain_alias, seed=7, target_config=target_config)
    )
    return build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)


def _build_pair(
    seed: int = 31415,
    n_samples: int = 16,
    target_config: dict[str, Any] | None = None,
):
    run = _build_run(seed=seed, n_samples=n_samples, target_config=target_config)
    latents = CanonicalLatentBatch.from_synthetic_dataset_run(run)
    views = SpectralViewBatch.from_synthetic_dataset_run(run, latents)
    return run, latents, views


def _default_split(n_total: int) -> tuple[list[int], list[int]]:
    half = n_total // 2
    context_indices = list(range(half))
    query_indices = list(range(half, n_total))
    return context_indices, query_indices


# ---------------------------------------------------------------------------
# Construction / contract surface
# ---------------------------------------------------------------------------


def test_from_batches_populates_split_arrays_ids_and_semantics() -> None:
    _run, latents, views = _build_pair(seed=11, n_samples=12)
    ctx, query = _default_split(len(latents.latent_ids))

    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    assert task.X_context.shape == (len(ctx), int(views.wavelengths.size))
    assert task.X_query.shape == (len(query), int(views.wavelengths.size))
    assert task.y_context.shape == (len(ctx),)
    assert task.y_query.shape == (len(query),)
    np.testing.assert_array_equal(task.X_context, views.X[np.asarray(ctx)])
    np.testing.assert_array_equal(task.X_query, views.X[np.asarray(query)])
    np.testing.assert_array_equal(
        task.y_context, np.asarray(latents.target_noisy, dtype=float)[np.asarray(ctx)]
    )
    np.testing.assert_array_equal(
        task.y_query, np.asarray(latents.target_noisy, dtype=float)[np.asarray(query)]
    )
    np.testing.assert_array_equal(task.wavelengths_context, views.wavelengths)
    np.testing.assert_array_equal(task.wavelengths_query, views.wavelengths)

    assert task.context_latent_ids == tuple(latents.latent_ids[i] for i in ctx)
    assert task.query_latent_ids == tuple(latents.latent_ids[i] for i in query)
    assert task.context_view_ids == tuple(views.view_ids[i] for i in ctx)
    assert task.query_view_ids == tuple(views.view_ids[i] for i in query)
    assert set(task.context_latent_ids).isdisjoint(set(task.query_latent_ids))
    assert set(task.context_view_ids).isdisjoint(set(task.query_view_ids))

    assert task.domain_key == latents.domain_metadata["domain_key"]
    assert task.instrument_context == latents.instrument_metadata["instrument_key"]
    assert task.instrument_query == latents.instrument_metadata["instrument_key"]
    assert task.measurement_mode == latents.instrument_metadata["measurement_mode"]
    assert task.target_type == "regression"
    assert task.target_name.startswith("target__")

    semantics = task.target_semantics
    assert semantics["target_source"] == "target_noisy"
    assert semantics["target_name"] == task.target_name
    assert semantics["target_type"] == "regression"
    assert semantics["target_clean_equals_target_noisy"] is True
    assert semantics["n_outputs"] == 1
    assert semantics["output_names"] == [task.target_name]
    assert semantics["multi_output_supported"] is True

    assert task.provenance["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert task.provenance["claims"]["realism"] is False
    assert task.provenance["claims"]["transfer"] is False
    assert task.provenance["limitations"]["context_query_sampler_implemented"] is False
    assert task.provenance["limitations"]["multi_output_supported"] is True
    assert task.provenance["n_outputs"] == 1

    assert task.split_policy["n_context"] == len(ctx)
    assert task.split_policy["n_query"] == len(query)
    assert task.split_policy["indices_disjoint"] is True


def test_from_batches_supports_target_clean_source() -> None:
    _run, latents, views = _build_pair(seed=11, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))

    task = NIRSPriorTask.from_batches(
        latents, views, ctx, query, target_source="target_clean"
    )

    np.testing.assert_array_equal(
        task.y_context, np.asarray(latents.target_clean, dtype=float)[np.asarray(ctx)]
    )
    assert task.target_semantics["target_source"] == "target_clean"
    assert task.provenance["target_source"] == "target_clean"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_task_id_is_deterministic_for_identical_inputs() -> None:
    _run_a, latents_a, views_a = _build_pair(seed=2026, n_samples=12)
    _run_b, latents_b, views_b = _build_pair(seed=2026, n_samples=12)
    ctx, query = _default_split(len(latents_a.latent_ids))

    task_a = NIRSPriorTask.from_batches(latents_a, views_a, ctx, query, task_seed=42)
    task_b = NIRSPriorTask.from_batches(latents_b, views_b, ctx, query, task_seed=42)

    assert task_a.task_id == task_b.task_id


def test_task_id_changes_when_split_or_task_seed_changes() -> None:
    _run, latents, views = _build_pair(seed=2026, n_samples=12)
    ctx, query = _default_split(len(latents.latent_ids))

    base = NIRSPriorTask.from_batches(latents, views, ctx, query, task_seed=1)
    same_seed_other_split = NIRSPriorTask.from_batches(
        latents, views, ctx[:-1], query + [ctx[-1]], task_seed=1
    )
    other_seed_same_split = NIRSPriorTask.from_batches(
        latents, views, ctx, query, task_seed=2
    )

    assert base.task_id != same_seed_other_split.task_id
    assert base.task_id != other_seed_same_split.task_id
    assert same_seed_other_split.task_id != other_seed_same_split.task_id


def test_task_id_ignores_X_y_values_but_changes_with_target_type() -> None:
    _run, latents, views = _build_pair(seed=2026, n_samples=12)
    ctx, query = _default_split(len(latents.latent_ids))

    base = NIRSPriorTask.from_batches(
        latents,
        views,
        ctx,
        query,
        target_name="shared_target",
        task_seed=42,
    )
    changed_X = replace(views, X=np.asarray(views.X, dtype=float) + 1000.0)
    changed_y = replace(
        latents,
        target_clean=np.asarray(latents.target_clean, dtype=float) + 1000.0,
        target_noisy=np.asarray(latents.target_noisy, dtype=float) + 1000.0,
    )

    assert (
        NIRSPriorTask.from_batches(
            latents,
            changed_X,
            ctx,
            query,
            target_name="shared_target",
            task_seed=42,
        ).task_id
        == base.task_id
    )
    assert (
        NIRSPriorTask.from_batches(
            changed_y,
            views,
            ctx,
            query,
            target_name="shared_target",
            task_seed=42,
        ).task_id
        == base.task_id
    )

    labels = np.arange(len(latents.latent_ids)) % 2
    classification_latents = replace(
        latents,
        target_clean=labels,
        target_noisy=labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": 2,
        },
    )
    classification = NIRSPriorTask.from_batches(
        classification_latents,
        views,
        ctx,
        query,
        target_name="shared_target",
        task_seed=42,
    )

    assert classification.task_id != base.task_id


def test_task_id_changes_with_output_structure_not_values() -> None:
    _run, latents, views = _build_pair(seed=2026, n_samples=12)
    ctx, query = _default_split(len(latents.latent_ids))

    base = NIRSPriorTask.from_batches(
        latents,
        views,
        ctx,
        query,
        target_name="shared_target",
        task_seed=42,
    )

    multi_target = np.column_stack([latents.target_noisy, latents.target_noisy * 2.0])
    multi_latents = replace(
        latents,
        target_clean=multi_target,
        target_noisy=multi_target,
    )
    multi = NIRSPriorTask.from_batches(
        multi_latents,
        views,
        ctx,
        query,
        target_name="shared_target",
        task_seed=42,
    )

    renamed_outputs = replace(
        multi_latents,
        target_metadata={
            **dict(multi_latents.target_metadata),
            "output_names": ["alpha", "beta"],
        },
    )
    renamed = NIRSPriorTask.from_batches(
        renamed_outputs,
        views,
        ctx,
        query,
        target_name="shared_target",
        task_seed=42,
    )

    assert multi.task_id != base.task_id
    assert renamed.task_id != multi.task_id


# ---------------------------------------------------------------------------
# Index validation
# ---------------------------------------------------------------------------


def test_from_batches_rejects_overlapping_context_query_indices() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(latents, views, [0, 1, 2, 3], [3, 4, 5])

    reasons = {f["reason"] for f in exc.value.failures}
    assert "overlapping_indices" in reasons


def test_from_batches_rejects_duplicate_indices_within_a_split() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(latents, views, [0, 1, 1, 2], [3, 4, 5])

    reasons = {f["reason"] for f in exc.value.failures}
    assert "duplicate_indices" in reasons


def test_from_batches_rejects_empty_split() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)

    with pytest.raises(NIRSPriorTaskError) as exc_ctx:
        NIRSPriorTask.from_batches(latents, views, [], [0, 1, 2])
    with pytest.raises(NIRSPriorTaskError) as exc_query:
        NIRSPriorTask.from_batches(latents, views, [0, 1, 2], [])

    ctx_reasons = {f["reason"] for f in exc_ctx.value.failures}
    query_reasons = {f["reason"] for f in exc_query.value.failures}
    assert "empty_split" in ctx_reasons
    assert "empty_split" in query_reasons


def test_from_batches_rejects_out_of_range_indices() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=8)

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(latents, views, [0, 1, 2, 3], [4, 5, 6, 99])

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_indices" in reasons


def test_from_batches_rejects_non_integer_indices() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=8)

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(
            latents,
            views,
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([3.0, 4.0]),
        )

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_indices" in reasons


def test_from_batches_rejects_non_1d_indices() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=8)

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(
            latents,
            views,
            np.asarray([[0, 1], [2, 3]]),
            np.asarray([4, 5]),
        )

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_indices" in reasons


@pytest.mark.parametrize("bad_seed", [True, 1.5])
def test_from_batches_rejects_non_integer_task_seed(bad_seed: Any) -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=8)
    ctx, query = _default_split(len(latents.latent_ids))

    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(latents, views, ctx, query, task_seed=bad_seed)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_task_seed", "task_seed") in failures


# ---------------------------------------------------------------------------
# Misalignment
# ---------------------------------------------------------------------------


def test_from_batches_rejects_misaligned_view_and_latent_batch() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    other_view = views.subset([0, 1, 2, 3, 4])

    with pytest.raises(SpectralViewBatchError) as exc:
        NIRSPriorTask.from_batches(latents, other_view, [0, 1], [2, 3])

    reasons = {f["reason"] for f in exc.value.failures}
    assert "alignment_mismatch" in reasons


# ---------------------------------------------------------------------------
# Validation via dataclasses.replace
# ---------------------------------------------------------------------------


def test_validation_rejects_non_finite_X_context_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad = task.X_context.copy()
    bad[0, 0] = np.nan
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, X_context=bad)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "X_context") in failures


def test_validation_rejects_non_finite_y_query_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad = task.y_query.copy()
    bad[0] = np.inf
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, y_query=bad)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("non_finite", "y_query") in failures


def test_validation_rejects_X_y_shape_mismatch_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    truncated_y = task.y_context[:-1].copy()
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, y_context=truncated_y)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "shape_mismatch" in reasons


def test_validation_rejects_non_monotonic_wavelengths_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=8)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad = task.wavelengths_context.copy()
    bad[2] = bad[1] - 5.0
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, wavelengths_context=bad)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "non_monotonic_wavelengths" in reasons


def test_validation_rejects_overlapping_split_ids_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    duplicated = list(task.query_latent_ids)
    duplicated[0] = task.context_latent_ids[0]

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, query_latent_ids=tuple(duplicated))

    reasons = {f["reason"] for f in exc.value.failures}
    assert "overlapping_split_ids" in reasons


# ---------------------------------------------------------------------------
# Leakage and risk gates
# ---------------------------------------------------------------------------


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
def test_validation_rejects_metadata_context_leakage(leak_key: str) -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_metadata = dict(task.metadata_context)
    bad_metadata[leak_key] = "not allowed"

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, metadata_context=bad_metadata)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", "metadata_context") in failures


def test_validation_rejects_nested_metadata_query_leakage() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_metadata = {**dict(task.metadata_query), "nested": {"target_clean": [0.1, 0.2]}}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, metadata_query=bad_metadata)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", "metadata_query") in failures


@pytest.mark.parametrize("leak_key", ["sample_target", "row_latent_features"])
def test_validation_rejects_prefixed_or_suffixed_metadata_leakage(leak_key: str) -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_metadata = {**dict(task.metadata_query), leak_key: "not allowed"}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, metadata_query=bad_metadata)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", "metadata_query") in failures


def test_validation_rejects_per_sample_leakage_in_prior_config() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_prior = {**dict(task.prior_config), "concentrations": [[0.5, 0.3, 0.2]]}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, prior_config=bad_prior)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", "prior_config") in failures


@pytest.mark.parametrize("leak_key", ["target_clean", "target_noisy", "latent_features"])
def test_validation_rejects_other_per_sample_leakage_in_prior_config(
    leak_key: str,
) -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_prior = {**dict(task.prior_config), leak_key: [0.1, 0.2]}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, prior_config=bad_prior)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("metadata_leakage", "prior_config") in failures


def test_prior_config_accepts_legitimate_target_prior_keys() -> None:
    """``target_prior`` / ``concentration_prior`` are config descriptors, not leaks."""
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    enriched_prior = {
        **dict(task.prior_config),
        "target_prior": {"type": "regression"},
        "concentration_prior": {"min_value": 0.0},
    }

    rebuilt = replace(task, prior_config=enriched_prior)
    assert rebuilt.prior_config["target_prior"] == {"type": "regression"}
    assert rebuilt.prior_config["concentration_prior"] == {"min_value": 0.0}


def test_validation_rejects_missing_risk_gates_in_provenance() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_provenance = dict(task.provenance)
    bad_provenance["risk_gates"] = {"A3_failed_documented": True}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, provenance=bad_provenance)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "missing_risk_gates" in reasons


def test_validation_rejects_missing_risk_gates_dict_in_provenance() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_provenance = {k: v for k, v in task.provenance.items() if k != "risk_gates"}

    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, provenance=bad_provenance)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "missing_risk_gates" in reasons


def test_validation_rejects_non_bool_target_clean_equals_target_noisy_semantics() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_semantics = {
        **dict(task.target_semantics),
        "target_clean_equals_target_noisy": "true",
    }
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, target_semantics=bad_semantics)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("incomplete_target_semantics", "target_semantics") in failures


# ---------------------------------------------------------------------------
# Classification / multi-output (D3)
# ---------------------------------------------------------------------------


def test_from_batches_accepts_classification_integer_like_labels() -> None:
    _run, latents, views = _build_pair(
        seed=5,
        n_samples=12,
        target_config={"type": "classification", "n_classes": 3, "separation": "moderate"},
    )
    ctx, query = _default_split(len(latents.latent_ids))

    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    assert task.target_type == "classification"
    assert task.target_semantics["target_type"] == "classification"
    assert task.target_semantics["n_outputs"] == 1
    assert task.target_semantics["multi_output_supported"] is True
    assert np.all(task.y_context == np.round(task.y_context))
    assert np.all(task.y_query == np.round(task.y_query))


def test_from_batches_rejects_classification_non_integer_labels() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    n = len(latents.latent_ids)
    bad_labels = np.linspace(0.0, 1.0, n)
    classification_latents = replace(
        latents,
        target_clean=bad_labels,
        target_noisy=bad_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": 2,
        },
    )

    ctx, query = _default_split(n)
    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(classification_latents, views, ctx, query)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_classification_labels", "target") in failures


def test_from_batches_accepts_multi_output_regression_target() -> None:
    """D3: multi-output regression task is accepted with 2D y arrays."""
    _run, latents, views = _build_pair(seed=5, n_samples=12)
    n = len(latents.latent_ids)
    multi_target = np.column_stack(
        [latents.target_noisy, latents.target_noisy * 2.0, latents.target_noisy + 1.0]
    )
    multi_latents = replace(latents, target_clean=multi_target, target_noisy=multi_target)
    assert multi_latents.target_noisy.shape == (n, 3)

    ctx, query = _default_split(n)
    task = NIRSPriorTask.from_batches(multi_latents, views, ctx, query)

    assert task.y_context.ndim == 2
    assert task.y_query.ndim == 2
    assert task.y_context.shape == (len(ctx), 3)
    assert task.y_query.shape == (len(query), 3)

    np.testing.assert_array_equal(task.y_context, multi_target[np.asarray(ctx)])
    np.testing.assert_array_equal(task.y_query, multi_target[np.asarray(query)])

    semantics = task.target_semantics
    assert semantics["target_type"] == "regression"
    assert semantics["n_outputs"] == 3
    assert semantics["multi_output_supported"] is True
    assert isinstance(semantics["output_names"], list)
    assert len(semantics["output_names"]) == 3
    assert all(isinstance(name, str) and name for name in semantics["output_names"])

    assert task.provenance["n_outputs"] == 3
    assert task.provenance["limitations"]["multi_output_supported"] is True
    assert task.provenance["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert task.provenance["claims"]["realism"] is False
    assert task.provenance["claims"]["transfer"] is False

    light = task.to_light_dict()
    assert light["target_semantics"]["n_outputs"] == 3
    assert "y_context" not in light
    assert "y_query" not in light

    payload = task.to_dict()
    assert isinstance(payload["y_context"], list)
    assert len(payload["y_context"]) == len(ctx)
    assert len(payload["y_context"][0]) == 3


def test_from_batches_accepts_multi_output_classification_integer_labels() -> None:
    """D3: multi-output classification with integer-like labels is accepted."""
    _run, latents, views = _build_pair(seed=5, n_samples=16)
    n = len(latents.latent_ids)
    label_a = np.asarray([i % 2 for i in range(n)], dtype=float)
    label_b = np.asarray([i % 3 for i in range(n)], dtype=float)
    multi_labels = np.column_stack([label_a, label_b])
    multi_latents = replace(
        latents,
        target_clean=multi_labels,
        target_noisy=multi_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": [2, 3],
        },
    )

    ctx, query = _default_split(n)
    task = NIRSPriorTask.from_batches(multi_latents, views, ctx, query)

    assert task.target_type == "classification"
    assert task.target_semantics["n_outputs"] == 2
    assert task.target_semantics["multi_output_supported"] is True
    assert task.y_context.ndim == 2
    assert task.y_query.ndim == 2
    assert task.y_context.shape == (len(ctx), 2)
    assert np.all(task.y_context == np.round(task.y_context))
    assert np.all(task.y_query == np.round(task.y_query))


def test_from_batches_rejects_multi_output_classification_non_integer_labels() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    n = len(latents.latent_ids)
    label_a = np.asarray([i % 2 for i in range(n)], dtype=float)
    label_b = np.linspace(0.0, 1.0, n)  # non-integer
    multi_labels = np.column_stack([label_a, label_b])
    multi_latents = replace(
        latents,
        target_clean=multi_labels,
        target_noisy=multi_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": [2, 2],
        },
    )

    ctx, query = _default_split(n)
    with pytest.raises(NIRSPriorTaskError) as exc:
        NIRSPriorTask.from_batches(multi_latents, views, ctx, query)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_classification_labels", "target") in failures


def test_from_batches_normalises_2d_single_column_target_to_1d() -> None:
    """A 2D ``(n, 1)`` target is collapsed to a 1D 1D single-output array."""
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    n = len(latents.latent_ids)
    column_target = np.asarray(latents.target_noisy, dtype=float).reshape(n, 1)
    column_latents = replace(latents, target_clean=column_target, target_noisy=column_target)

    ctx, query = _default_split(n)
    task = NIRSPriorTask.from_batches(column_latents, views, ctx, query)

    assert task.y_context.ndim == 1
    assert task.y_query.ndim == 1
    assert task.target_semantics["n_outputs"] == 1


def test_validation_normalises_2d_single_column_y_via_replace() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    rebuilt = replace(
        task,
        y_context=task.y_context.reshape(-1, 1),
        y_query=task.y_query.reshape(-1, 1),
    )

    assert rebuilt.y_context.ndim == 1
    assert rebuilt.y_query.ndim == 1
    assert rebuilt.target_semantics["n_outputs"] == 1


def test_validation_rejects_inconsistent_y_dimensionality_via_replace() -> None:
    """Cross-split: y_context 1D and y_query 2D must be rejected."""
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    bad_y_query = np.column_stack([task.y_query, task.y_query])
    with pytest.raises(NIRSPriorTaskError) as exc:
        replace(task, y_query=bad_y_query)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "shape_mismatch" in reasons


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_light_dict_omits_heavy_arrays_and_labels() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    light = task.to_light_dict()

    for heavy in ("X_context", "X_query", "y_context", "y_query"):
        assert heavy not in light
    assert light["n_context"] == len(ctx)
    assert light["n_query"] == len(query)
    assert light["n_wavelengths_context"] == int(views.wavelengths.size)
    assert light["n_wavelengths_query"] == int(views.wavelengths.size)
    assert light["context_latent_ids"] == list(task.context_latent_ids)
    assert light["query_latent_ids"] == list(task.query_latent_ids)
    assert light["target_semantics"]["target_source"] == "target_noisy"
    assert light["provenance"]["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }


def test_to_dict_includes_arrays_and_labels_as_lists() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=10)
    ctx, query = _default_split(len(latents.latent_ids))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)

    payload = task.to_dict()

    assert isinstance(payload["X_context"], list)
    assert isinstance(payload["X_query"], list)
    assert isinstance(payload["y_context"], list)
    assert isinstance(payload["y_query"], list)
    assert isinstance(payload["wavelengths_context"], list)
    assert len(payload["X_context"]) == len(ctx)
    assert len(payload["y_query"]) == len(query)
