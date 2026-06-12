"""Unit tests for late-fusion meta-feature alignment utilities."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.meta_features import (
    MetaFeatureAlignmentError,
    PredictionFeatureVector,
    align_prediction_feature_vectors,
)
from nirs4all.operators.data.merge import MetaFeaturePlan


def test_strict_alignment_uses_unit_ids_not_row_position():
    """Strict alignment reorders each branch by explicit physical_sample_id."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S2", "S1"], [20.0, 10.0]),
            PredictionFeatureVector("raman", ["S1", "S2"], [1.0, 2.0]),
        ],
        target_unit_ids=["S1", "S2"],
    )

    assert aligned.unit_ids == ["S1", "S2"]
    assert aligned.feature_names == ["nir", "raman"]
    np.testing.assert_array_equal(aligned.X, np.array([[10.0, 1.0], [20.0, 2.0]]))
    assert aligned.mask is None


def test_strict_alignment_rejects_missing_units():
    """Strict mode refuses silently incomplete prediction coverage."""
    with pytest.raises(MetaFeatureAlignmentError, match="Strict meta-feature alignment"):
        align_prediction_feature_vectors(
            [
                PredictionFeatureVector("nir", ["S1", "S2"], [10.0, 20.0]),
                PredictionFeatureVector("raman", ["S1"], [1.0]),
            ],
            target_unit_ids=["S1", "S2"],
        )


def test_duplicate_unit_ids_are_rejected():
    """Alignment requires one prediction per feature/unit pair."""
    with pytest.raises(MetaFeatureAlignmentError, match="duplicate unit ids"):
        align_prediction_feature_vectors(
            [
                PredictionFeatureVector("nir", ["S1", "S1"], [10.0, 11.0]),
                PredictionFeatureVector("raman", ["S1"], [1.0]),
            ]
        )


def test_drop_incomplete_drops_missing_rows():
    """drop_incomplete keeps only units covered by every prediction branch."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S1", "S2", "S3"], [10.0, 20.0, 30.0]),
            PredictionFeatureVector("raman", ["S1", "S3"], [1.0, 3.0]),
        ],
        plan=MetaFeaturePlan(missing_prediction_policy="drop_incomplete"),
        target_unit_ids=["S1", "S2", "S3"],
    )

    assert aligned.unit_ids == ["S1", "S3"]
    assert aligned.dropped_units == ["S2"]
    np.testing.assert_array_equal(aligned.X, np.array([[10.0, 1.0], [30.0, 3.0]]))


def test_drop_branch_drops_incomplete_features():
    """drop_branch removes branches that cannot cover the requested unit set."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S1", "S2"], [10.0, 20.0]),
            PredictionFeatureVector("raman", ["S1"], [1.0]),
        ],
        plan=MetaFeaturePlan(missing_prediction_policy="drop_branch"),
        target_unit_ids=["S1", "S2"],
    )

    assert aligned.feature_names == ["nir"]
    assert aligned.dropped_features == ["raman"]
    np.testing.assert_array_equal(aligned.X, np.array([[10.0], [20.0]]))


@pytest.mark.parametrize("policy", ["mask", "pad", "partial_model"])
def test_mask_like_policies_return_mask_and_nan(policy):
    """mask/pad/partial_model preserve rows and expose missing prediction masks."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S1", "S2"], [10.0, 20.0]),
            PredictionFeatureVector("raman", ["S1"], [1.0]),
        ],
        plan=MetaFeaturePlan(missing_prediction_policy=policy),
        target_unit_ids=["S1", "S2"],
    )

    assert aligned.unit_ids == ["S1", "S2"]
    assert aligned.mask is not None
    np.testing.assert_array_equal(aligned.mask, np.array([[True, True], [True, False]]))
    assert np.isnan(aligned.X[1, 1])


def test_impute_declared_fills_declared_values():
    """impute_declared requires explicit values and records no mask."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S1", "S2"], [10.0, 20.0]),
            PredictionFeatureVector("raman", ["S1"], [1.0]),
        ],
        plan=MetaFeaturePlan(missing_prediction_policy="impute_declared"),
        target_unit_ids=["S1", "S2"],
        impute_values={"raman": -1.0},
    )

    np.testing.assert_array_equal(aligned.X, np.array([[10.0, 1.0], [20.0, -1.0]]))
    assert aligned.mask is None


def test_impute_declared_requires_values():
    """impute_declared never picks implicit prediction fill values."""
    with pytest.raises(MetaFeatureAlignmentError, match="declared impute value"):
        align_prediction_feature_vectors(
            [
                PredictionFeatureVector("nir", ["S1", "S2"], [10.0, 20.0]),
                PredictionFeatureVector("raman", ["S1"], [1.0]),
            ],
            plan=MetaFeaturePlan(missing_prediction_policy="impute_declared"),
            target_unit_ids=["S1", "S2"],
        )


def test_manifest_records_schema_without_prediction_values():
    """Aligned meta-feature manifests keep replay metadata but not arrays."""
    aligned = align_prediction_feature_vectors(
        [
            PredictionFeatureVector("nir", ["S1"], [10.0]),
            PredictionFeatureVector("raman", ["S1"], [1.0]),
        ],
    )

    manifest = aligned.to_manifest()

    assert manifest["shape"] == [1, 2]
    assert manifest["unit_ids"] == ["S1"]
    assert manifest["feature_names"] == ["nir", "raman"]
    assert manifest["has_mask"] is False
    assert manifest["meta_feature_plan"]["missing_prediction_policy"] == "strict"
    assert "X" not in manifest
