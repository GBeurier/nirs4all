"""Studio 0.9.1 descriptive definitions, without importing Studio at runtime."""

import numpy as np
import pytest

from nirs4all.analysis.playground_statistics import distance_statistics, spectral_statistics


def test_population_statistics_and_linear_percentiles_preserve_full_axis():
    values = np.array([[1, 5, 9], [3, 7, 11], [9, 1, 4], [7, 3, 0]], dtype=float)
    original = values.copy()
    result = spectral_statistics(values)
    for name, operation in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        np.testing.assert_array_equal(result[name], operation(values, axis=0))
        assert result["global"][name] == operation(values)
    for percentile in (5, 25, 50, 75, 95):
        np.testing.assert_array_equal(result[f"p{percentile}"], np.percentile(values, percentile, axis=0))
    assert result["global"]["n_samples"] == 4
    assert result["global"]["n_features"] == 3
    assert result["diagnostics"]["std_ddof"] == 0
    np.testing.assert_array_equal(values, original)


def test_non_finite_data_is_visible_not_replaced_or_dropped():
    result = spectral_statistics([[1, np.nan, 2], [3, 4, np.inf]])
    assert result["mean"][0] == 2
    assert np.isnan(result["mean"][1])
    assert np.isinf(result["mean"][2])
    assert result["diagnostics"]["non_finite_feature_indices"] == [1, 2]
    assert result["diagnostics"]["nan_count"] == result["diagnostics"]["inf_count"] == 1


def test_one_row_and_8193_features_are_valid_no_display_decimation():
    result = spectral_statistics(np.arange(8193)[None])
    assert len(result["mean"]) == 8193
    np.testing.assert_array_equal(result["std"], np.zeros(8193))


@pytest.mark.parametrize("values", [[], [1, 2], [[1], [2, 3]], np.zeros((2, 0)), np.zeros((2, 3, 1))])
def test_invalid_matrix_shape(values):
    with pytest.raises(ValueError):
        spectral_statistics(values)


def test_distance_summary_distinguishes_empty_real_zero_and_undefined():
    assert distance_statistics([])["count"] == 0
    assert distance_statistics([0])["count"] == 1
    assert distance_statistics([0])["undefined_count"] == 0
    result = distance_statistics([1, np.nan])
    assert result["undefined_count"] == 1
    assert np.isnan(result["mean"])


def test_distance_summary_quantile_definition():
    result = distance_statistics([1, 2, 4, 8])
    np.testing.assert_array_equal(list(result["quantiles"].values()), np.percentile([1, 2, 4, 8], [50, 75, 90, 95]))
    with pytest.raises(ValueError, match="one-dimensional"):
        distance_statistics([[1, 2]])
