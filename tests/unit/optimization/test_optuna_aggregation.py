"""Unit tests for OptunaManager._aggregate_scores (BUG-2 regression test)."""

import numpy as np
import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestAggregateScores:
    """Tests for _aggregate_scores method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_mean_returns_average_not_sum(self, manager):
        """BUG-2 regression: 'mean' must return np.mean, not np.sum."""
        scores = [0.5, 0.6, 0.7]
        result = manager._aggregate_scores(scores, "mean")
        assert result == pytest.approx(0.6)
        assert result != pytest.approx(1.8)  # Was the old bug

    def test_mean_single_score(self, manager):
        result = manager._aggregate_scores([0.42], "mean")
        assert result == pytest.approx(0.42)

    def test_best_returns_minimum(self, manager):
        scores = [0.5, 0.6, 0.7]
        result = manager._aggregate_scores(scores, "best")
        assert result == pytest.approx(0.5)

    def test_best_with_inf(self, manager):
        scores = [float('inf'), 0.5, 0.6]
        result = manager._aggregate_scores(scores, "best")
        assert result == pytest.approx(0.5)

    def test_robust_best_excludes_inf(self, manager):
        scores = [float('inf'), 0.5, 0.6]
        result = manager._aggregate_scores(scores, "robust_best")
        assert result == pytest.approx(0.5)

    def test_robust_best_all_inf(self, manager):
        scores = [float('inf'), float('inf')]
        result = manager._aggregate_scores(scores, "robust_best")
        assert result == float('inf')

    def test_unknown_eval_mode_raises(self, manager):
        """Unknown eval_mode must raise ValueError, not silently fallback.

        Note: 'avg' is normalized to 'mean' upstream in finetune(), but
        _aggregate_scores itself must reject it — it only accepts canonical values.
        """
        with pytest.raises(ValueError, match="Unknown eval_mode 'avg'"):
            manager._aggregate_scores([0.5, 0.6], "avg")

    def test_unknown_eval_mode_sum_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            manager._aggregate_scores([0.5, 0.6], "sum")
