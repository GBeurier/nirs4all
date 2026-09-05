"""Inline structured chart data remains numerical, aligned and accessible."""

import numpy as np
import pytest

from nirs4all.analysis.playground_execution import execute_preview
from nirs4all.analysis.playground_projections import (
    display_indices,
    pca_projection,
    repetition_projection,
    umap_projection,
)
from nirs4all.analysis.playground_types import PreviewBatch
from nirs4all.analysis.projections import compute_pca_projection


def test_pca_exactly_reuses_owner_with_targets_and_fold_labels():
    batch = PreviewBatch.from_arrays(np.random.default_rng(9).normal(size=(20, 15)), y=np.arange(20))
    labels = list(range(20))
    result = pca_projection(batch, {"fold_labels": labels})
    owner = compute_pca_projection(batch.x, max_components=10, variance_threshold=0.999)
    np.testing.assert_array_equal(result["coordinates"], owner["coordinates"])
    assert result["n_components_999"] == owner["n_components_threshold"]
    assert result["y"] == result["fold_labels"] == labels


def test_repetition_pairs_first_and_three_member_mean_references():
    batch = PreviewBatch.from_arrays([[0, 0], [3, 4], [0, 0], [3, 0], [6, 0]],
                                    metadata={"bio_sample": ["pair", "pair", "triple", "triple", "triple"]}, y=[1, 1, 2, 2, 2])
    result = repetition_projection(batch, options={"distance_metric": "euclidean"})
    np.testing.assert_allclose([point["distance"] for point in result["data"]], [0, 5, 3, 0, 3])
    assert [point["sample_index"] for point in result["data"]] == list(range(5))
    assert result["statistics"]["mean_distance"] == 2.2
    assert result["total_repetitions"] == 5


def test_repetition_uses_actual_pca_coordinates_not_raw_spectra():
    batch = PreviewBatch.from_arrays([[0, 0], [100, 100]], sample_ids=["a_rep1", "a_rep2"])
    result = repetition_projection(batch, pca={"coordinates": [[0, 0], [3, 4]]})
    assert [point["distance"] for point in result["data"]] == [0, 5]
    assert result["effective_space"] == "pca"
    absent = repetition_projection(batch)
    assert absent["effective_space"] == "spectra"
    assert absent["diagnostics"][0]["requested_space"] == "pca"


def test_mahalanobis_covariance_real_and_budget_checked_before_covariance(monkeypatch):
    from scipy.spatial.distance import mahalanobis

    values = np.array([[1, 2], [3, 6], [7, 4]], dtype=float)
    batch = PreviewBatch.from_arrays(values, metadata={"bio_sample": ["g"] * 3})
    result = repetition_projection(batch, options={"distance_metric": "mahalanobis"})
    precision = np.linalg.inv(np.cov(values, rowvar=False) + np.eye(2) * 1e-6)
    np.testing.assert_allclose([point["distance"] for point in result["data"]], [mahalanobis(row, values.mean(axis=0), precision) for row in values])
    monkeypatch.setattr(np, "cov", lambda *args, **kwargs: pytest.fail("must admit covariance first"))
    with pytest.raises(ValueError, match="host budget"):
        repetition_projection(batch, options={"distance_metric": "mahalanobis", "max_covariance_cells": 3})


def test_optional_umap_absence_is_real_capability_not_fake_projection(monkeypatch):
    import nirs4all.analysis.playground_projections as projections

    monkeypatch.setattr(projections.importlib.util, "find_spec", lambda name: None)
    result = umap_projection(PreviewBatch.from_arrays(np.ones((20, 3))))
    assert result["available"] is False
    assert "coordinates" not in result


def test_decimation_changes_only_display_data_not_pca_or_statistics():
    batch = PreviewBatch.from_arrays(np.random.default_rng(1).normal(size=(30, 401)), wavelengths=np.linspace(1000, 2000, 401))
    full = execute_preview(batch)
    small = execute_preview(batch, options={"max_wavelengths_returned": 40})
    assert small["processed"]["spectra"].shape == (30, 40)
    assert small["processed"]["shape"] == [30, 401]
    assert small["processed"]["statistics"] == full["processed"]["statistics"]
    np.testing.assert_array_equal(small["pca"]["coordinates"], full["pca"]["coordinates"])
    indices = small["processed"]["display_feature_indices"]
    assert indices[0] == 0 and indices[-1] == 400
    np.testing.assert_array_equal(small["processed"]["spectra"], batch.x[:, indices])


@pytest.mark.parametrize("maximum", [None, 0, 1, 2, 10, 100])
def test_decimation_historical_small_count_policy(maximum):
    np.testing.assert_array_equal(display_indices(np.arange(10), np.ones((4, 10)), maximum), np.arange(10))


def test_invalid_repetition_projection_alignment_and_pattern():
    batch = PreviewBatch.from_arrays([[1, 2], [3, 4]], sample_ids=["a_1", "a_2"])
    with pytest.raises(ValueError, match="match current"):
        repetition_projection(batch, pca={"coordinates": [[0, 0]]})
    with pytest.raises(ValueError, match="Unknown"):
        repetition_projection(batch, options={"distance_metric": "typo"})
