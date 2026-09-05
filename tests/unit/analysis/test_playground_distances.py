"""Numerical and identity regressions for the extracted 0.9.1 definitions."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from nirs4all.analysis.playground_distances import paired_spectral_distances, repetition_variance


@pytest.fixture
def pairs():
    rng = np.random.default_rng(19)
    return rng.normal(size=(12, 3)), rng.normal(size=(12, 3))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "spectral_angle", "correlation", "mahalanobis", "pca_distance"])
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_paired_distances_preserve_historical_definitions(pairs, metric, scale):
    left, right = pairs
    if metric == "euclidean":
        expected = np.linalg.norm(left - right, axis=1)
    elif metric == "manhattan":
        expected = np.sum(np.abs(left - right), axis=1)
    elif metric == "cosine":
        expected = np.diag(cdist(left, right, "cosine"))
    elif metric == "spectral_angle":
        expected = np.arccos(np.clip(np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-10), -1, 1))
    elif metric == "correlation":
        expected = np.array([1 - np.corrcoef(a, b)[0, 1] for a, b in zip(left, right, strict=True)])
    elif metric == "mahalanobis":
        covariance = LedoitWolf().fit(np.vstack([left, right]))
        differences = left - right
        expected = np.sqrt(np.sum(differences @ covariance.precision_ * differences, axis=1))
    else:
        projection = PCA(n_components=3).fit(np.vstack([left, right]))
        expected = np.linalg.norm(projection.transform(left) - projection.transform(right), axis=1)
    if scale == "log":
        expected = np.log1p(expected)
    result = paired_spectral_distances(left, right, metric=metric, scale=scale)
    np.testing.assert_allclose(result["distances"], expected, rtol=1e-14, atol=1e-14)
    assert result["sample_indices"] == list(range(12))
    assert result["effective_metric"] == metric
    assert result["diagnostics"] == []


@pytest.mark.parametrize("metric", ["mahalanobis", "pca_distance"])
def test_small_sample_historical_approximation_is_explicit(metric):
    result = paired_spectral_distances([[1, 2]], [[4, 6]], metric=metric)
    assert result["distances"].tolist() == [5]
    assert result["metric"] == metric
    assert result["effective_metric"] == "euclidean"
    assert "approximation" in result["diagnostics"][0]["code"]


def test_estimator_failure_never_retries_as_another_metric(pairs, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("estimator failed")

    monkeypatch.setattr(LedoitWolf, "fit", broken)
    with pytest.raises(RuntimeError, match="estimator failed"):
        paired_spectral_distances(*pairs, metric="mahalanobis")


def test_zero_vectors_and_constant_correlation_are_explicit():
    cosine = paired_spectral_distances([[0, 0]], [[1, 2]], metric="cosine")
    assert np.isnan(cosine["distances"][0])
    assert cosine["diagnostics"] == [{"code": "undefined_distance", "sample_indices": [0]}]
    correlation = paired_spectral_distances([[2, 2]], [[1, 2]], metric="correlation")
    assert correlation["distances"].tolist() == [1]
    assert correlation["diagnostics"][0]["sample_indices"] == [0]


@pytest.mark.parametrize("reference,expected", [
    ("group_mean", [2.5, 2.5, 2.5, 2.5]),
    ("first", [0, 5, 0, 5]),
    ("leave_one_out", [5, 5, 5, 5]),
])
def test_repetition_order_original_indices_and_references(reference, expected):
    values = [[1, 2], [10, 20], [4, 6], [13, 24], [100, 100]]
    result = repetition_variance(values, ["b", "a", "b", "a", "singleton"], reference=reference)
    np.testing.assert_array_equal(result["distances"], expected)
    assert result["sample_indices"] == [1, 3, 0, 2]
    assert result["group_ids"] == ["a", "a", "b", "b"]
    assert result["n_groups"] == 2
    assert result["per_group"]["a"]["count"] == 2


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine", "spectral_angle", "correlation"])
def test_repetition_three_members_leave_one_out_matches_scalar_definition(pairs, metric):
    spectra = pairs[0][:3]
    result = repetition_variance(spectra, ["g"] * 3, reference="leave_one_out", metric=metric)
    expected = []
    for index, spectrum in enumerate(spectra):
        ref = np.delete(spectra, index, axis=0).mean(axis=0)
        if metric == "spectral_angle":
            expected.append(np.arccos(np.clip(np.dot(spectrum, ref) / (np.linalg.norm(spectrum) * np.linalg.norm(ref)), -1, 1)))
        else:
            scipy_metric = "cityblock" if metric == "manhattan" else metric
            expected.append(cdist(spectrum[None], ref[None], scipy_metric)[0, 0])
    np.testing.assert_allclose(result["distances"], expected, atol=1e-14, rtol=1e-14)


def test_repetition_historical_aliases_and_zero_vectors_are_not_silent():
    alias = repetition_variance([[1, 2], [4, 6]], ["a", "a"], reference="selected", metric="mahalanobis")
    assert alias["effective_reference"] == "group_mean"
    assert alias["effective_metric"] == "euclidean"
    assert len(alias["diagnostics"]) == 2
    zero = repetition_variance([[0, 0], [0, 0]], ["a", "a"], metric="cosine")
    assert zero["distances"].tolist() == [0, 0]
    assert zero["diagnostics"][0]["sample_indices"] == [0, 1]
    constant = repetition_variance([[1, 1], [2, 2]], ["a", "a"], metric="correlation")
    assert constant["distances"].tolist() == [1, 1]
    assert constant["diagnostics"][0]["sample_indices"] == [0, 1]
    singletons = repetition_variance([[1, 2], [3, 4]], ["a", "b"])
    assert singletons["distances"].tolist() == []
    assert singletons["n_groups"] == 0


@pytest.mark.parametrize("kwargs", [{"metric": "typo"}, {"scale": "typo"}])
def test_unknown_pair_options_fail_before_calculation(pairs, kwargs):
    with pytest.raises(ValueError, match="Unknown"):
        paired_spectral_distances(*pairs, **kwargs)


@pytest.mark.parametrize("values", [[[np.inf]], [[np.nan]], [1, 2], []])
def test_invalid_pair_observations_are_not_replaced(values):
    with pytest.raises(ValueError):
        paired_spectral_distances(values, values)


def test_alignment_and_group_options_are_validated():
    with pytest.raises(ValueError, match="Shape mismatch"):
        paired_spectral_distances([[1]], [[1, 2]])
    for groups in ([], [["a"]], ["a", "b"]):
        with pytest.raises(ValueError, match="group_ids"):
            repetition_variance([[1, 2]], groups)
    for kwargs in ({"reference": "typo"}, {"metric": "typo"}):
        with pytest.raises(ValueError, match="Unknown"):
            repetition_variance([[1, 2]], ["a"], **kwargs)
