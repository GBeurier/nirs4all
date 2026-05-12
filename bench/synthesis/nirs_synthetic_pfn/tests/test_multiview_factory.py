"""Phase C3 contract tests for ``build_same_latent_spectral_views``."""

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
    SpectralViewVariantConfig,
    build_same_latent_spectral_views,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config

_LEAKY_KEYS = {
    "y",
    "target",
    "targets",
    "concentration",
    "concentrations",
    "target_clean",
    "target_noisy",
    "latent_feature",
    "latent_features",
}
_LEAKY_PREFIXES = tuple(f"{k}_" for k in _LEAKY_KEYS)


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


def _build_pair(
    *,
    seed: int = 31415,
    n_samples: int = 12,
    domain_alias: str = "grain",
) -> tuple[Any, CanonicalLatentBatch]:
    record = canonicalize_prior_config(_valid_source(domain_alias, seed=7))
    run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)
    latents = CanonicalLatentBatch.from_synthetic_dataset_run(run)
    return run, latents


def _is_leaky_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in _LEAKY_KEYS:
        return True
    return any(normalized.startswith(prefix) for prefix in _LEAKY_PREFIXES)


def _walk_keys(value: Any, *, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_leaky_key(str(key)):
                paths.append(child_path)
            paths.extend(_walk_keys(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_walk_keys(child, path=f"{path}[{index}]"))
    return paths


def test_default_factory_produces_at_least_two_aligned_views() -> None:
    run, latents = _build_pair(seed=12345, n_samples=10)

    views = build_same_latent_spectral_views(run, latents, random_seed=42)

    assert len(views) >= 2
    for view in views:
        assert isinstance(view, SpectralViewBatch)
        assert view.latent_ids == latents.latent_ids
        assert view.X.shape == (len(latents.latent_ids), int(run.wavelengths.size))
        assert np.all(np.isfinite(view.X))
        np.testing.assert_array_equal(
            view.wavelengths, np.asarray(run.wavelengths, dtype=float)
        )
        assert view.instrument_key == run.builder_config["features"]["instrument"]
        assert view.measurement_mode == run.builder_config["features"]["measurement_mode"]
        assert view.metadata["risk_gates"] == {
            "A3_failed_documented": True,
            "B2_realism_failed": True,
        }
        assert view.metadata["claims"]["realism"] is False
        assert view.metadata["claims"]["transfer"] is False
        view.assert_aligned_to(latents)


def test_default_factory_is_deterministic_under_same_seed() -> None:
    run_a, latents_a = _build_pair(seed=20260429, n_samples=8)
    run_b, latents_b = _build_pair(seed=20260429, n_samples=8)

    views_a = build_same_latent_spectral_views(run_a, latents_a, random_seed=7)
    views_b = build_same_latent_spectral_views(run_b, latents_b, random_seed=7)

    assert len(views_a) == len(views_b)
    for view_a, view_b in zip(views_a, views_b, strict=True):
        np.testing.assert_array_equal(view_a.X, view_b.X)
        assert view_a.view_ids == view_b.view_ids
        assert view_a.latent_ids == view_b.latent_ids


def test_view_ids_differ_between_variants_within_same_call() -> None:
    run, latents = _build_pair(seed=11, n_samples=6)

    views = build_same_latent_spectral_views(run, latents, random_seed=0)

    assert len(views) >= 2
    all_view_ids = [vid for view in views for vid in view.view_ids]
    assert len(set(all_view_ids)) == len(all_view_ids)
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            assert set(views[i].view_ids).isdisjoint(set(views[j].view_ids))


def test_non_identity_variant_changes_X_and_records_state() -> None:
    run, latents = _build_pair(seed=2024, n_samples=10)

    views = build_same_latent_spectral_views(run, latents, random_seed=1)

    identity_view = next(
        v for v in views if v.preprocessing_state.get("preprocessing_applied") is False
    )
    transformed_view = next(
        v for v in views if v.preprocessing_state.get("preprocessing_applied") is True
    )

    np.testing.assert_array_equal(identity_view.X, np.asarray(run.X, dtype=float))
    assert not np.array_equal(transformed_view.X, np.asarray(run.X, dtype=float))
    assert transformed_view.preprocessing_state["method"] in {"center", "snv"}
    assert transformed_view.noise_state["noise_added_in_view"] in {True, False}
    if transformed_view.noise_state["noise_added_in_view"]:
        assert transformed_view.noise_state["view_noise_std"] > 0.0
        assert transformed_view.noise_state["noise_distribution"] == "gaussian"


def test_factory_constructions_have_no_leakage_in_namespaces() -> None:
    run, latents = _build_pair(seed=5, n_samples=8)

    views = build_same_latent_spectral_views(run, latents, random_seed=3)

    for view in views:
        for namespace_name in (
            "view_config",
            "preprocessing_state",
            "noise_state",
            "metadata",
        ):
            namespace = getattr(view, namespace_name)
            leaky_paths = _walk_keys(namespace)
            assert leaky_paths == [], (
                f"{namespace_name} leaks at: {leaky_paths}"
            )


def test_factory_rejects_duplicate_view_keys() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)

    variants = (
        SpectralViewVariantConfig(view_key="dup"),
        SpectralViewVariantConfig(view_key="dup", preprocessing="snv", noise_std=0.0),
    )
    with pytest.raises(SpectralViewBatchError) as exc:
        build_same_latent_spectral_views(run, latents, variants, random_seed=0)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "duplicate_view_key" in reasons


def test_factory_rejects_negative_noise_std() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)

    variants = (
        SpectralViewVariantConfig(view_key="ok"),
        SpectralViewVariantConfig(view_key="bad", noise_std=-0.1),
    )
    with pytest.raises(SpectralViewBatchError) as exc:
        build_same_latent_spectral_views(run, latents, variants, random_seed=0)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "invalid_noise_std" in reasons


def test_factory_rejects_unsupported_instrument_override() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)

    variants = (
        SpectralViewVariantConfig(view_key="other", instrument_key="some_other"),
    )
    with pytest.raises(SpectralViewBatchError) as exc:
        build_same_latent_spectral_views(run, latents, variants, random_seed=0)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "rerender_unsupported" in reasons


def test_factory_rejects_empty_variants() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)

    with pytest.raises(SpectralViewBatchError) as exc:
        build_same_latent_spectral_views(run, latents, variants=(), random_seed=0)

    reasons = {f["reason"] for f in exc.value.failures}
    assert "empty_variants" in reasons


def test_factory_rejects_latent_arrays_that_do_not_match_run() -> None:
    run, latents = _build_pair(seed=5, n_samples=6)
    bad_concentrations = latents.concentrations[::-1].copy()
    mismatched = replace(latents, concentrations=bad_concentrations)

    with pytest.raises(SpectralViewBatchError) as exc:
        build_same_latent_spectral_views(run, mismatched, random_seed=0)

    failures = {(f["reason"], f["field"]) for f in exc.value.failures}
    assert ("same_latent_mismatch", "concentrations") in failures


def test_custom_variants_produce_expected_count_and_alignment() -> None:
    run, latents = _build_pair(seed=42, n_samples=9)

    variants = (
        SpectralViewVariantConfig(view_key="raw"),
        SpectralViewVariantConfig(
            view_key="centered", preprocessing="center", noise_std=0.0
        ),
        SpectralViewVariantConfig(
            view_key="snv_only", preprocessing="snv", noise_std=0.0
        ),
    )
    views = build_same_latent_spectral_views(run, latents, variants, random_seed=99)

    assert len(views) == 3
    np.testing.assert_array_equal(views[0].X, np.asarray(run.X, dtype=float))
    assert not np.array_equal(views[1].X, np.asarray(run.X, dtype=float))
    assert not np.array_equal(views[2].X, np.asarray(run.X, dtype=float))
    np.testing.assert_allclose(
        views[1].X.mean(axis=1), np.zeros(views[1].X.shape[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        views[2].X.mean(axis=1), np.zeros(views[2].X.shape[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        views[2].X.std(axis=1, ddof=0),
        np.ones(views[2].X.shape[0]),
        atol=1e-10,
    )
    for view in views:
        assert view.latent_ids == latents.latent_ids
