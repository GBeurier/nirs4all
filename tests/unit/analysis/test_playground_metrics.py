"""The complete historical descriptor set, with real owner computations."""

import numpy as np
import pytest
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import linregress

from nirs4all.analysis.playground_execution import execute_preview
from nirs4all.analysis.playground_metrics import ALL_METRICS, FAST_METRICS, compute_descriptors, descriptor_statistics
from nirs4all.analysis.playground_types import PreviewBatch
from nirs4all.operators.filters import HighLeverageFilter, XOutlierFilter


@pytest.fixture
def batch():
    return PreviewBatch.from_arrays(np.random.default_rng(92).normal(size=(30, 7)), wavelengths=np.linspace(1100, 1200, 7), header_unit="nm")


def test_all_24_historical_descriptors_have_real_values_and_origins(batch):
    result = compute_descriptors(batch, options={"metrics": list(ALL_METRICS)})
    assert len(ALL_METRICS) == 24
    assert result["computed_metrics"] == list(ALL_METRICS)
    assert result["errors"] == {}
    assert result["diagnostics"] == []
    for name in ALL_METRICS:
        assert len(result["values"][name]) == 30
        assert np.isfinite(result["values"][name]).all()
    assert result["sample_origins"].tolist() == list(range(30))


def test_amplitude_energy_noise_and_quality_match_historical_equations(batch):
    x, wl = batch.x, batch.wavelengths
    result = compute_descriptors(batch)["values"]
    expected = {
        "global_min": np.nanmin(x, axis=1), "global_max": np.nanmax(x, axis=1),
        "dynamic_range": np.nanmax(x, axis=1) - np.nanmin(x, axis=1), "mean_intensity": np.nanmean(x, axis=1),
        "l2_norm": np.linalg.norm(x, axis=1), "rms_energy": np.sqrt(np.nanmean(x**2, axis=1)),
        "auc": np.trapezoid(x, wl, axis=1), "abs_auc": np.trapezoid(np.abs(x), wl, axis=1),
        "hf_variance": np.nanvar(np.diff(x, axis=1), axis=1), "snr_estimate": np.abs(np.nanmean(x, axis=1)) / np.nanstd(x, axis=1),
        "smoothness": 1 / np.nanvar(np.diff(x, axis=1), axis=1),
        "nan_count": np.isnan(x).sum(axis=1), "inf_count": np.isinf(x).sum(axis=1),
        "saturation_count": (x >= np.nanmax(x) * 0.99).sum(axis=1), "zero_count": (x == 0).sum(axis=1),
    }
    assert set(expected) == set(FAST_METRICS)
    for name, values in expected.items():
        np.testing.assert_array_equal(result[name], values)


def test_shape_definitions_use_index_not_nm_axis(batch):
    result = compute_descriptors(batch, options={"metrics": ["baseline_slope", "baseline_offset", "peak_count", "peak_prominence_max"]})
    expected = {key: [] for key in result["values"]}
    for row in batch.x:
        regression = linregress(np.arange(7), row)
        expected["baseline_slope"].append(regression.slope)
        expected["baseline_offset"].append(regression.intercept)
        peaks, _ = find_peaks(row, prominence=np.std(row) * 0.5)
        expected["peak_count"].append(len(peaks))
        peaks, _ = find_peaks(row)
        expected["peak_prominence_max"].append(np.max(peak_prominences(row, peaks)[0]) if len(peaks) else 0)
    assert result["definitions"]["baseline_axis"] == "feature_index"
    for key in expected:
        np.testing.assert_array_equal(result["values"][key], expected[key])


@pytest.mark.parametrize("name,method", [("hotelling_t2", "pca_leverage"), ("q_residual", "pca_residual"), ("distance_to_centroid", "mahalanobis"), ("lof_score", "lof")])
def test_chemometric_values_exactly_delegate_to_existing_filter(batch, name, method):
    result = compute_descriptors(batch, options={"metrics": [name]})
    expected = XOutlierFilter(method=method, n_components=5, contamination=0.1).fit(batch.x)._distances_
    np.testing.assert_array_equal(result["values"][name], expected)
    assert result["errors"] == {}


def test_high_leverage_uses_owner_public_values(batch):
    expected = HighLeverageFilter(method="hat", n_components=5).fit(batch.x).get_leverages(batch.x)
    actual = compute_descriptors(batch, options={"metrics": ["leverage"]})
    np.testing.assert_array_equal(actual["values"]["leverage"], expected)


def test_unknown_and_failed_metrics_are_exposed_not_silently_dropped(batch, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("owner witness failure")

    monkeypatch.setattr(XOutlierFilter, "fit", broken)
    result = compute_descriptors(batch, options={"metrics": ["global_min", "typo", "q_residual"]})
    assert result["computed_metrics"] == ["global_min"]
    assert "Unknown" in result["errors"]["typo"]
    assert result["errors"]["q_residual"] == "owner witness failure"


def test_covariance_limit_precedes_owner_fit(batch, monkeypatch):
    monkeypatch.setattr(XOutlierFilter, "fit", lambda *args, **kwargs: pytest.fail("budget admission must precede fit"))
    result = compute_descriptors(batch, options={"metrics": ["distance_to_centroid"], "max_covariance_cells": 48})
    assert result["values"] == {}
    assert "host budget" in result["errors"]["distance_to_centroid"]


def test_missing_values_preserve_quality_counts_and_explain_cleaning():
    batch = PreviewBatch.from_arrays([[1, np.nan, 2], [2, 4, 6], [4, 3, 5], [1, 1, 2]])
    result = compute_descriptors(batch, options={"metrics": ["nan_count", "l2_norm", "peak_count", "q_residual"]})
    assert result["values"]["nan_count"] == [1, 0, 0, 0]
    assert np.isnan(result["values"]["l2_norm"][0])
    assert {entry["metric"] for entry in result["diagnostics"] if entry["code"] == "historical_nan_to_num_input_policy"} == {"peak_count", "q_residual"}
    summary = descriptor_statistics(np.array([np.nan]))
    assert summary["valid_count"] == 0 and summary["nan_count"] == 1
    assert summary["mean"] == 0  # Historical empty summary is labelled count=0.


def test_inline_metrics_are_computed_by_default_owner_and_no_target_needed(batch):
    result = execute_preview(batch, options={"compute_metrics": True})
    assert result["metrics"]["computed_metrics"] == list(FAST_METRICS)
    assert result["metrics"]["definitions"]["auc_axis_unit"] == "nm"
    assert result["metrics"]["errors"] == {}
    assert result["processed"]["y"] is None
    with pytest.raises(ValueError, match="list"):
        compute_descriptors(batch, options={"metrics": "mean_intensity"})


def test_auc_preserves_descending_axis_sign_and_exact_grid():
    batch = PreviewBatch.from_arrays([[1, 2, 3]], wavelengths=[1100.000321, 1000.000123, 800.000421], header_unit="cm-1")
    result = compute_descriptors(batch, options={"metrics": ["auc", "abs_auc"]})
    assert result["values"]["auc"][0] < 0
    assert result["values"]["auc"] == result["values"]["abs_auc"]
    assert result["values"]["auc"][0] == np.trapezoid(batch.x[0], batch.wavelengths)
