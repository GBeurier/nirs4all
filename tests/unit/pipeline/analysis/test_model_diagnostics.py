"""Tests for nirs4all.pipeline.analysis.model_diagnostics."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.analysis.model_diagnostics import (
    bias_variance_decomposition,
    estimate_train_size,
    learning_curve_points,
    robustness_axes,
)


class TestBiasVarianceDecomposition:
    def test_constant_predictions_have_zero_variance(self):
        result = bias_variance_decomposition({"s1": [(1.0, 2.0), (1.0, 2.0)]})
        assert result is not None
        assert result.variance == pytest.approx(0.0)
        assert result.bias_squared == pytest.approx(1.0)
        assert result.total_error == pytest.approx(1.0)
        assert result.n_samples == 1

    def test_unbiased_spread_has_zero_bias(self):
        result = bias_variance_decomposition({"s1": [(2.0, 1.0), (2.0, 3.0)]})
        assert result is not None
        assert result.bias_squared == pytest.approx(0.0)
        assert result.variance == pytest.approx(1.0)

    def test_single_prediction_samples_are_excluded(self):
        assert bias_variance_decomposition({"s1": [(1.0, 1.5)]}) is None

    def test_non_finite_pairs_ignored(self):
        result = bias_variance_decomposition({
            "s1": [(1.0, float("nan")), (1.0, 2.0), (1.0, 2.0)],
        })
        assert result is not None
        assert result.variance == pytest.approx(0.0)

    def test_aggregates_over_samples(self):
        result = bias_variance_decomposition({
            "a": [(0.0, 1.0), (0.0, 1.0)],   # bias²=1, var=0
            "b": [(0.0, -1.0), (0.0, 1.0)],  # bias²=0, var=1
        })
        assert result is not None
        assert result.bias_squared == pytest.approx(0.5)
        assert result.variance == pytest.approx(0.5)
        assert result.n_samples == 2


class TestRobustnessAxes:
    def test_empty_input(self):
        assert robustness_axes([]) == []

    def test_axes_normalized_across_set(self):
        chains = [
            {"fold_scores": [1.0, 1.0, 1.0], "train_score": 0.9, "val_score": 0.9, "score": 0.9, "fold_count": 5},
            {"fold_scores": [0.5, 1.5, 1.0], "train_score": 0.9, "val_score": 0.5, "score": 0.5, "fold_count": 3},
        ]
        profiles = robustness_axes(chains)
        stable, unstable = profiles
        assert stable["cv_stability"]["value"] == pytest.approx(1.0)
        assert unstable["cv_stability"]["value"] == pytest.approx(0.0)
        assert stable["train_test_gap"]["value"] == pytest.approx(1.0)
        assert unstable["train_test_gap"]["value"] == pytest.approx(0.0)
        assert stable["score_absolute"]["value"] == pytest.approx(1.0)
        assert unstable["score_absolute"]["value"] == pytest.approx(0.0)
        assert stable["fold_count_ratio"]["value"] == pytest.approx(1.0)
        assert unstable["fold_count_ratio"]["value"] == pytest.approx(3 / 5)

    def test_lower_better_flips_score_axis(self):
        chains = [
            {"fold_scores": [], "train_score": None, "val_score": None, "score": 0.2, "fold_count": 5},
            {"fold_scores": [], "train_score": None, "val_score": None, "score": 0.8, "fold_count": 5},
        ]
        low, high = robustness_axes(chains, lower_better=True)
        assert low["score_absolute"]["value"] == pytest.approx(1.0)
        assert high["score_absolute"]["value"] == pytest.approx(0.0)


class TestLearningCurve:
    def test_estimate_train_size_kfold(self):
        # 5-fold, 20 val samples -> total ≈ 25, train ≈ 5... (20*5/4=25)
        assert estimate_train_size(20, 5) == 5
        assert estimate_train_size(0, 5) == 0
        # fold_count < 2 falls back to 5
        assert estimate_train_size(20, 1) == 5

    def test_points_sorted_and_aggregated(self):
        points = learning_curve_points({
            100: [{"train": 0.9, "val": 0.8}, {"train": 0.7, "val": 0.6}],
            50: [{"train": 0.5, "val": None}],
        })
        assert [p["train_size"] for p in points] == [50, 100]
        small, large = points
        assert small["train_mean"] == pytest.approx(0.5)
        assert small["train_std"] is None
        assert small["val_mean"] is None
        assert large["train_mean"] == pytest.approx(0.8)
        assert large["val_mean"] == pytest.approx(0.7)
        assert large["count"] == 2
