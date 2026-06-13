"""Unit tests for N7 fit-influence policy resolution."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.fit_influence import (
    FitInfluenceError,
    FitInfluencePolicy,
    resolve_fit_influence,
)


def test_auto_uses_uniform_rows_for_constant_cardinality():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2", "S2"],
        backend_supports_sample_weight=True,
    )

    assert resolution.effective_mode == "uniform_rows"
    assert resolution.sample_weight is None
    assert resolution.resample_indices is None


def test_auto_uniform_rows_records_derived_row_warning():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2", "S2"],
        backend_supports_sample_weight=True,
        row_is_derived=[True, True, True, True],
    )

    assert resolution.effective_mode == "uniform_rows"
    assert "derived rows" in resolution.warnings[0]


def test_auto_uses_equal_sample_influence_when_backend_supports_weights():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S1", "S2"],
        backend_supports_sample_weight=True,
    )

    assert resolution.effective_mode == "equal_sample_influence"
    assert resolution.sample_weight is not None
    np.testing.assert_allclose(resolution.sample_weight, [1 / 3, 1 / 3, 1 / 3, 1.0])
    assert resolution.sample_weight[:3].sum() == pytest.approx(1.0)
    assert resolution.sample_weight[3:].sum() == pytest.approx(1.0)


def test_auto_resamples_when_backend_lacks_weight_support():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S1", "S2"],
        backend_supports_sample_weight=False,
    )

    assert resolution.effective_mode == "resample_equalized"
    assert resolution.sample_weight is None
    assert resolution.resample_indices is not None
    # S2 is repeated to match S1's cardinality of three rows.
    np.testing.assert_array_equal(resolution.resample_indices, np.array([0, 1, 2, 3, 3, 3]))


def test_strict_weight_support_fails_without_backend_support():
    with pytest.raises(FitInfluenceError, match="strict_weight_support"):
        resolve_fit_influence(
            ["S1", "S1", "S1", "S2"],
            policy=FitInfluencePolicy(mode="strict_weight_support"),
            backend_supports_sample_weight=False,
        )


def test_backend_loss_weight_requires_backend_support():
    with pytest.raises(FitInfluenceError, match="sample_weight support"):
        resolve_fit_influence(
            ["S1", "S1", "S2"],
            policy=FitInfluencePolicy(mode="backend_loss_weight"),
            backend_supports_sample_weight=False,
        )


def test_backend_loss_weight_returns_equal_sample_weights():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2"],
        policy=FitInfluencePolicy(mode="backend_loss_weight"),
        backend_supports_sample_weight=True,
    )

    assert resolution.effective_mode == "backend_loss_weight"
    assert resolution.sample_weight is not None
    np.testing.assert_allclose(resolution.sample_weight, [0.5, 0.5, 1.0])


def test_scorer_only_keeps_fit_unweighted_and_returns_scorer_weights():
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2"],
        policy=FitInfluencePolicy(mode="scorer_only"),
        backend_supports_sample_weight=False,
    )

    assert resolution.sample_weight is None
    assert resolution.scorer_weight is not None
    np.testing.assert_allclose(resolution.scorer_weight, [0.5, 0.5, 1.0])


def test_policy_round_trip_and_resolution_manifest():
    policy = FitInfluencePolicy(mode="auto", allowed_fallbacks=("resample_equalized",), random_state=12)
    restored = FitInfluencePolicy.from_dict(policy.to_dict())
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2"],
        policy=restored,
        backend_supports_sample_weight=False,
    )

    assert restored == policy
    assert restored.fingerprint() == policy.fingerprint()
    manifest = resolution.to_manifest()
    assert manifest["policy"]["allowed_fallbacks"] == ["resample_equalized"]
    assert manifest["effective_mode"] == "resample_equalized"
    assert manifest["has_resample_indices"] is True


def test_policy_from_minimal_dict_keeps_default_fallbacks():
    policy = FitInfluencePolicy.from_dict({"mode": "auto"})
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2"],
        policy=policy,
        backend_supports_sample_weight=False,
    )

    assert "resample_equalized" in policy.allowed_fallbacks
    assert resolution.effective_mode == "resample_equalized"


def test_auto_fails_when_required_fallback_is_not_allowed():
    with pytest.raises(FitInfluenceError, match="fallback"):
        resolve_fit_influence(
            ["S1", "S1", "S2"],
            policy=FitInfluencePolicy(mode="auto", allowed_fallbacks=("uniform_rows",)),
            backend_supports_sample_weight=False,
        )


def test_explicit_mode_respects_disallowed_fallback():
    # An explicit (non-auto) mode that the backend cannot honour must still obey
    # allowed_fallbacks: resample_equalized is not allowed here, so it must fail.
    with pytest.raises(FitInfluenceError, match="fallback"):
        resolve_fit_influence(
            ["S1", "S1", "S2"],
            policy=FitInfluencePolicy(mode="equal_sample_influence", allowed_fallbacks=("uniform_rows",)),
            backend_supports_sample_weight=False,
        )


def test_explicit_mode_uses_allowed_fallback():
    # Same explicit mode, but the fallback is allowed: resolution falls back cleanly.
    resolution = resolve_fit_influence(
        ["S1", "S1", "S2"],
        policy=FitInfluencePolicy(mode="equal_sample_influence", allowed_fallbacks=("resample_equalized",)),
        backend_supports_sample_weight=False,
    )

    assert resolution.effective_mode == "resample_equalized"
    assert resolution.resample_indices is not None
    assert resolution.sample_weight is None
