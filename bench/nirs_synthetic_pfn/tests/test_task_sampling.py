"""Phase D2 contract tests for the context/query sampler."""

from __future__ import annotations

import json
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
    ContextQuerySplit,
    ContextQuerySplitConfig,
    ContextQuerySplitError,
    NIRSPriorTask,
    SpectralViewBatch,
    sample_context_query_split,
    sample_nirs_prior_task,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _with_groups(latents: CanonicalLatentBatch, n_groups: int) -> CanonicalLatentBatch:
    n = len(latents.latent_ids)
    new_groups = tuple(f"group_{i % n_groups}" for i in range(n))
    return replace(latents, group_ids=new_groups, batch_ids=new_groups)


def _with_classification_labels(
    latents: CanonicalLatentBatch, n_classes: int
) -> CanonicalLatentBatch:
    n = len(latents.latent_ids)
    labels = np.asarray([i % n_classes for i in range(n)], dtype=float)
    return replace(
        latents,
        target_clean=labels,
        target_noisy=labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": n_classes,
        },
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_missing_seed() -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(strategy="random", seed=None)  # type: ignore[arg-type]
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_seed", "seed") in failures


@pytest.mark.parametrize("bad_seed", [True, False, 1.5, "0"])
def test_config_rejects_non_int_seed(bad_seed: Any) -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(strategy="random", seed=bad_seed)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_seed", "seed") in failures


def test_config_accepts_numpy_integer_seed() -> None:
    config = ContextQuerySplitConfig(strategy="random", seed=np.int64(1))
    assert int(config.seed) == 1


def test_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(strategy="bogus", seed=0)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_strategy", "strategy") in failures


def test_config_rejects_conflicting_size_specs() -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(
            strategy="random", seed=0, n_context=5, context_fraction=0.5
        )
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("conflicting_size", "n_context") in failures


def test_config_rejects_invalid_target_source() -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(
            strategy="random", seed=0, target_source="bogus"
        )
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_target_source", "target_source") in failures


def test_config_rejects_invalid_fraction_range() -> None:
    with pytest.raises(ContextQuerySplitError) as exc:
        ContextQuerySplitConfig(strategy="random", seed=0, context_fraction=0.0)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("invalid_fraction", "context_fraction") in failures


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


def test_random_split_produces_disjoint_indices_with_exact_sizes() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=20)
    config = ContextQuerySplitConfig(strategy="random", seed=42, n_context=8, n_query=8)

    split = sample_context_query_split(latents, config)

    assert isinstance(split, ContextQuerySplit)
    assert split.context_indices.dtype == np.intp
    assert split.query_indices.dtype == np.intp
    assert split.context_indices.size == 8
    assert split.query_indices.size == 8
    assert set(split.context_indices.tolist()).isdisjoint(set(split.query_indices.tolist()))
    assert split.context_indices.min() >= 0
    assert split.query_indices.max() < len(latents.latent_ids)
    assert split.split_policy["phase"] == "D2"
    assert split.split_policy["strategy"] == "random"
    assert split.split_policy["n_context"] == 8
    assert split.split_policy["n_query"] == 8
    assert split.split_policy["indices_disjoint"] is True


def test_random_split_default_is_balanced_50_50() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=10)
    config = ContextQuerySplitConfig(strategy="random", seed=0)

    split = sample_context_query_split(latents, config)

    assert split.context_indices.size == 5
    assert split.query_indices.size == 5
    assert split.context_indices.size + split.query_indices.size == len(latents.latent_ids)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_config_gives_identical_indices_and_task() -> None:
    _run, latents, views = _build_pair(seed=11, n_samples=18)
    config = ContextQuerySplitConfig(strategy="random", seed=2026, n_context=9, n_query=9)

    split_a = sample_context_query_split(latents, config)
    split_b = sample_context_query_split(latents, config)
    np.testing.assert_array_equal(split_a.context_indices, split_b.context_indices)
    np.testing.assert_array_equal(split_a.query_indices, split_b.query_indices)
    assert split_a.split_policy == split_b.split_policy

    task_a = sample_nirs_prior_task(latents, views, config)
    task_b = sample_nirs_prior_task(latents, views, config)
    assert task_a.task_id == task_b.task_id


def test_different_seeds_produce_different_random_splits() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=20)
    config_a = ContextQuerySplitConfig(strategy="random", seed=1)
    config_b = ContextQuerySplitConfig(strategy="random", seed=2)

    split_a = sample_context_query_split(latents, config_a)
    split_b = sample_context_query_split(latents, config_b)

    assert not np.array_equal(split_a.context_indices, split_b.context_indices)


# ---------------------------------------------------------------------------
# from_batches integration via sampler
# ---------------------------------------------------------------------------


def test_sample_nirs_prior_task_returns_valid_task_with_d2_provenance() -> None:
    _run, latents, views = _build_pair(seed=11, n_samples=20)
    config = ContextQuerySplitConfig(strategy="random", seed=7, n_context=10, n_query=10)

    task = sample_nirs_prior_task(latents, views, config)

    assert isinstance(task, NIRSPriorTask)
    assert task.X_context.shape[0] == 10
    assert task.X_query.shape[0] == 10
    assert task.split_policy["phase"] == "D2"
    assert task.split_policy["strategy"] == "random"
    assert task.split_policy["seed"] == 7
    assert task.task_seed == 7
    # D2 sampler flips the limitation flag.
    assert task.provenance["limitations"]["context_query_sampler_implemented"] is True
    # Risk gates remain negative.
    assert task.provenance["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert task.provenance["claims"]["realism"] is False
    assert task.provenance["claims"]["transfer"] is False


def test_d1_default_limitation_flag_remains_false() -> None:
    """Calling NIRSPriorTask.from_batches directly keeps the D1 default."""
    _run, latents, views = _build_pair(seed=11, n_samples=10)
    ctx = list(range(5))
    query = list(range(5, 10))
    task = NIRSPriorTask.from_batches(latents, views, ctx, query)
    assert task.split_policy["phase"] == "D1"
    assert task.provenance["limitations"]["context_query_sampler_implemented"] is False


# ---------------------------------------------------------------------------
# Infeasible sizes
# ---------------------------------------------------------------------------


def test_random_split_rejects_oversized_request() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=10)
    config = ContextQuerySplitConfig(
        strategy="random", seed=0, n_context=8, n_query=8
    )
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(latents, config)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "infeasible_size" in reasons


def test_split_rejects_too_small_batch() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=10)
    tiny = replace(
        latents,
        latent_ids=(latents.latent_ids[0],),
        concentrations=latents.concentrations[:1],
        latent_features=latents.latent_features[:1],
        target_clean=latents.target_clean[:1],
        target_noisy=latents.target_noisy[:1],
        batch_ids=(latents.batch_ids[0],),
        group_ids=(latents.group_ids[0],),
    )
    config = ContextQuerySplitConfig(strategy="random", seed=0)
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(tiny, config)
    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("infeasible_size", "latent_batch") in failures


def test_invalid_input_types_are_rejected() -> None:
    _run, latents, _views = _build_pair(seed=11, n_samples=10)
    with pytest.raises(ContextQuerySplitError):
        sample_context_query_split(object(), ContextQuerySplitConfig(  # type: ignore[arg-type]
            strategy="random", seed=0
        ))
    with pytest.raises(ContextQuerySplitError):
        sample_context_query_split(latents, {"strategy": "random"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stratified classification
# ---------------------------------------------------------------------------


def test_stratified_classification_preserves_classes_in_both_splits() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=24)
    classification_latents = _with_classification_labels(latents, n_classes=3)
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=42,
        n_context=12,
        n_query=12,
    )

    split = sample_context_query_split(classification_latents, config)

    assert split.context_indices.size == 12
    assert split.query_indices.size == 12
    labels = np.asarray(classification_latents.target_noisy, dtype=int)
    ctx_classes = set(labels[split.context_indices].tolist())
    query_classes = set(labels[split.query_indices].tolist())
    assert ctx_classes == {0, 1, 2}
    assert query_classes == {0, 1, 2}
    assert split.diagnostics["n_classes_context"] == 3
    assert split.diagnostics["n_classes_query"] == 3
    assert split.diagnostics["shared_label_count"] == 3


def test_stratified_classification_keeps_query_classes_when_query_is_subset() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=30)
    labels = np.asarray([0] * 10 + [1] * 10 + [2] * 10, dtype=float)
    classification_latents = replace(
        latents,
        target_clean=labels,
        target_noisy=labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": 3,
        },
    )
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=42,
        n_context=15,
        n_query=3,
        shuffle=False,
    )

    split = sample_context_query_split(classification_latents, config)

    observed = set(labels[split.query_indices].astype(int).tolist())
    assert split.query_indices.size == 3
    assert observed == {0, 1, 2}
    assert split.diagnostics["class_counts_query"] == {"0": 1, "1": 1, "2": 1}


def test_stratified_classification_rejects_regression_target() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    config = ContextQuerySplitConfig(strategy="stratified_classification", seed=0)
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(latents, config)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "strategy_target_mismatch" in reasons


def test_stratified_classification_rejects_non_integer_labels() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    n = len(latents.latent_ids)
    bad_labels = np.linspace(0.0, 1.0, n)
    bad = replace(
        latents,
        target_clean=bad_labels,
        target_noisy=bad_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": 2,
        },
    )
    config = ContextQuerySplitConfig(strategy="stratified_classification", seed=0)
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(bad, config)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_classification_labels" in reasons


def test_stratified_classification_rejects_singleton_class() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    n = len(latents.latent_ids)
    labels = np.zeros(n, dtype=float)
    labels[0] = 1.0  # singleton class for label=1
    bad = replace(
        latents,
        target_clean=labels,
        target_noisy=labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": 2,
        },
    )
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=0,
        n_context=6,
        n_query=6,
    )
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(bad, config)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "infeasible_stratification" in reasons


# ---------------------------------------------------------------------------
# Group holdout
# ---------------------------------------------------------------------------


def test_group_holdout_produces_disjoint_groups() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=24)
    grouped = _with_groups(latents, n_groups=4)
    config = ContextQuerySplitConfig(strategy="group_holdout", seed=11)

    split = sample_context_query_split(grouped, config)

    ctx_groups = {grouped.group_ids[i] for i in split.context_indices.tolist()}
    query_groups = {grouped.group_ids[i] for i in split.query_indices.tolist()}
    assert ctx_groups.isdisjoint(query_groups)
    assert split.diagnostics["groups_disjoint"] is True
    assert split.diagnostics["n_groups_context"] >= 1
    assert split.diagnostics["n_groups_query"] >= 1
    assert split.split_policy["strategy"] == "group_holdout"


def test_group_holdout_rejects_single_group() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    grouped = _with_groups(latents, n_groups=1)
    config = ContextQuerySplitConfig(strategy="group_holdout", seed=11)
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(grouped, config)
    reasons = {f["reason"] for f in exc.value.failures}
    assert "infeasible_grouping" in reasons


def test_group_holdout_reports_actual_sizes_when_exact_sizes_are_atomic_infeasible() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=10)
    grouped = _with_groups(latents, n_groups=2)
    config = ContextQuerySplitConfig(
        strategy="group_holdout",
        seed=11,
        n_context=1,
        n_query=1,
        shuffle=False,
    )

    split = sample_context_query_split(grouped, config)

    ctx_groups = {grouped.group_ids[i] for i in split.context_indices.tolist()}
    query_groups = {grouped.group_ids[i] for i in split.query_indices.tolist()}
    assert ctx_groups.isdisjoint(query_groups)
    assert split.context_indices.size == 5
    assert split.query_indices.size == 5
    assert split.split_policy["requested_n_context"] == 1
    assert split.split_policy["requested_n_query"] == 1
    assert split.split_policy["exact_size_match"] is False
    assert split.split_policy["diagnostics_summary"]["n_context"] == 5
    assert split.split_policy["diagnostics_summary"]["n_query"] == 5
    assert split.split_policy["diagnostics_summary"]["exact_size_match"] is False


def test_group_holdout_seed_changes_partition() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=24)
    grouped = _with_groups(latents, n_groups=6)

    split_a = sample_context_query_split(
        grouped, ContextQuerySplitConfig(strategy="group_holdout", seed=1)
    )
    split_b = sample_context_query_split(
        grouped, ContextQuerySplitConfig(strategy="group_holdout", seed=2)
    )
    # With 6 groups and different seeds the partitions should differ.
    assert not np.array_equal(split_a.context_indices, split_b.context_indices)


# ---------------------------------------------------------------------------
# Diagnostics & non-leakage
# ---------------------------------------------------------------------------


def test_diagnostics_present_for_regression() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=20)
    config = ContextQuerySplitConfig(strategy="random", seed=0)

    split = sample_context_query_split(latents, config)

    expected = {
        "strategy",
        "n_total",
        "n_context",
        "n_query",
        "indices_disjoint",
        "latent_ids_disjoint",
        "constant_target_context",
        "constant_target_query",
        "target_range_context",
        "target_range_query",
    }
    assert expected.issubset(split.diagnostics.keys())
    assert isinstance(split.diagnostics["target_range_context"], list)
    assert len(split.diagnostics["target_range_context"]) == 2
    assert "target_range_context" not in split.split_policy["diagnostics_summary"]
    assert "target_range_query" not in split.split_policy["diagnostics_summary"]


def test_diagnostics_can_be_disabled() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=20)
    config = ContextQuerySplitConfig(strategy="random", seed=0, diagnostics=False)
    split = sample_context_query_split(latents, config)
    assert split.diagnostics == {}
    # split_policy still carries a non-leaky summary.
    assert "diagnostics_summary" in split.split_policy


def test_split_policy_is_json_serialisable_and_non_leaky() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=24)
    classification_latents = _with_classification_labels(latents, n_classes=3)
    config = ContextQuerySplitConfig(
        strategy="stratified_classification", seed=0, n_context=12, n_query=12
    )
    split = sample_context_query_split(classification_latents, config)

    # Round-trip through JSON to prove serialisability.
    payload = json.dumps(split.split_policy)
    restored = json.loads(payload)
    assert restored["phase"] == "D2"

    # split_policy must not embed per-sample y / latent values.
    forbidden = {"y", "target_clean", "target_noisy", "concentrations", "latent_features"}

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert str(key).lower() not in forbidden
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(split.split_policy)


# ---------------------------------------------------------------------------
# sample_nirs_prior_task input validation
# ---------------------------------------------------------------------------


def test_sample_nirs_prior_task_rejects_non_view_input() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=10)
    config = ContextQuerySplitConfig(strategy="random", seed=0)
    with pytest.raises(ContextQuerySplitError):
        sample_nirs_prior_task(latents, object(), config)  # type: ignore[arg-type]


def test_sample_nirs_prior_task_target_name_propagation() -> None:
    _run, latents, views = _build_pair(seed=5, n_samples=12)
    config = ContextQuerySplitConfig(strategy="random", seed=0, n_context=6, n_query=6)
    task = sample_nirs_prior_task(latents, views, config, target_name="custom_target")
    assert task.target_name == "custom_target"
    assert task.target_semantics["target_name"] == "custom_target"


# ---------------------------------------------------------------------------
# Phase D3: multi-output regression / classification
# ---------------------------------------------------------------------------


def _with_multi_output_regression(
    latents: CanonicalLatentBatch, n_outputs: int = 3
) -> CanonicalLatentBatch:
    base = np.asarray(latents.target_noisy, dtype=float)
    columns = [base * float(i + 1) for i in range(n_outputs)]
    multi = np.column_stack(columns)
    return replace(latents, target_clean=multi, target_noisy=multi)


def _with_multi_output_classification(
    latents: CanonicalLatentBatch,
    n_classes_per_output: list[int],
) -> CanonicalLatentBatch:
    n = len(latents.latent_ids)
    columns = [
        np.asarray([i % nc for i in range(n)], dtype=float)
        for nc in n_classes_per_output
    ]
    multi = np.column_stack(columns)
    return replace(
        latents,
        target_clean=multi,
        target_noisy=multi,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": list(n_classes_per_output),
        },
    )


def test_random_split_supports_multi_output_regression_diagnostics() -> None:
    """Multi-output regression sampling: 2D y, aggregated diagnostics."""
    _run, latents, views = _build_pair(seed=11, n_samples=20)
    multi = _with_multi_output_regression(latents, n_outputs=3)
    config = ContextQuerySplitConfig(strategy="random", seed=42, n_context=10, n_query=10)

    split = sample_context_query_split(multi, config)

    assert split.diagnostics["n_outputs"] == 3
    assert isinstance(split.diagnostics["target_range_context"], list)
    assert len(split.diagnostics["target_range_context"]) == 3
    assert all(
        len(rng) == 2 and isinstance(rng[0], float) and isinstance(rng[1], float)
        for rng in split.diagnostics["target_range_context"]
    )

    # split_policy.diagnostics_summary must NOT carry per-output ranges
    # or any per-sample target value.
    summary = split.split_policy["diagnostics_summary"]
    assert "target_range_context" not in summary
    assert "target_range_query" not in summary
    forbidden = {"y", "target_clean", "target_noisy", "concentrations", "latent_features"}

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert str(key).lower() not in forbidden
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    _walk(split.split_policy)

    # End-to-end task constructed via the sampler.
    task = sample_nirs_prior_task(multi, views, config)
    assert task.y_context.ndim == 2
    assert task.y_context.shape == (10, 3)
    assert task.target_semantics["n_outputs"] == 3
    assert task.target_semantics["multi_output_supported"] is True
    assert task.provenance["risk_gates"] == {
        "A3_failed_documented": True,
        "B2_realism_failed": True,
    }
    assert task.provenance["claims"]["realism"] is False
    assert task.provenance["claims"]["transfer"] is False


def test_stratified_classification_supports_multi_output_via_joint_labels() -> None:
    """Multi-output classification stratifies on joint label tuples."""
    _run, latents, _views = _build_pair(seed=5, n_samples=24)
    multi = _with_multi_output_classification(latents, n_classes_per_output=[2, 3])
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=42,
        n_context=12,
        n_query=12,
    )

    split = sample_context_query_split(multi, config)

    labels = np.asarray(multi.target_noisy, dtype=int)
    ctx_pairs = {tuple(row) for row in labels[split.context_indices].tolist()}
    query_pairs = {tuple(row) for row in labels[split.query_indices].tolist()}
    # All joint labels (2 * 3 = 6) seen in both splits.
    assert len(ctx_pairs) == 6
    assert len(query_pairs) == 6

    diag = split.diagnostics
    assert diag["n_outputs"] == 2
    assert diag["n_joint_labels"] == 6
    assert diag["n_classes_context"] == 6
    assert diag["n_classes_query"] == 6
    assert diag["shared_joint_label_count"] == 6
    assert "joint_label_count_context" in diag
    assert "joint_label_count_query" in diag
    # Joint label keys are aggregate (e.g. "0,1"), not per-sample values.
    for key in diag["joint_label_count_context"]:
        parts = key.split(",")
        assert len(parts) == 2
        assert all(p.lstrip("-").isdigit() for p in parts)


def test_stratified_multi_output_classification_rejects_singleton_joint_label() -> None:
    """Joint label classes with <2 samples make stratification infeasible."""
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    n = len(latents.latent_ids)
    label_a = np.zeros(n, dtype=float)
    label_b = np.zeros(n, dtype=float)
    label_a[0] = 1.0
    label_b[0] = 1.0  # joint (1, 1) has count 1 → singleton
    multi_labels = np.column_stack([label_a, label_b])
    bad = replace(
        latents,
        target_clean=multi_labels,
        target_noisy=multi_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": [2, 2],
        },
    )
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=0,
        n_context=6,
        n_query=6,
    )
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(bad, config)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "infeasible_stratification" in reasons


def test_stratified_classification_rejects_multi_output_non_integer_labels() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=12)
    n = len(latents.latent_ids)
    label_a = np.asarray([i % 2 for i in range(n)], dtype=float)
    label_b = np.linspace(0.0, 1.0, n)  # non-integer
    multi_labels = np.column_stack([label_a, label_b])
    bad = replace(
        latents,
        target_clean=multi_labels,
        target_noisy=multi_labels,
        target_metadata={
            **dict(latents.target_metadata),
            "type": "classification",
            "n_classes": [2, 2],
        },
    )
    config = ContextQuerySplitConfig(
        strategy="stratified_classification",
        seed=0,
        n_context=6,
        n_query=6,
    )
    with pytest.raises(ContextQuerySplitError) as exc:
        sample_context_query_split(bad, config)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_classification_labels" in reasons


def test_split_policy_serialisable_for_multi_output_regression() -> None:
    _run, latents, _views = _build_pair(seed=5, n_samples=20)
    multi = _with_multi_output_regression(latents, n_outputs=2)
    config = ContextQuerySplitConfig(strategy="random", seed=0, n_context=10, n_query=10)
    split = sample_context_query_split(multi, config)
    payload = json.dumps(split.split_policy)
    restored = json.loads(payload)
    assert restored["phase"] == "D2"
    assert restored["diagnostics_summary"]["n_outputs"] == 2
