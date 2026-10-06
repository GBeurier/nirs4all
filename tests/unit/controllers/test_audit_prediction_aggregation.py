"""Independent references for the two local legacy weighting repairs."""

import numpy as np
import pytest

from nirs4all.controllers.shared.prediction_aggregator import PredictionAggregator
from nirs4all.operators.data.merge import AggregationStrategy


@pytest.mark.parametrize("metric,scores,expected", [("r2", [0.9, 0.1], 0.1), ("accuracy", [0.9, 0.1], 0.1), ("rmse", [0.9, 0.1], 0.9)])
def test_fold_weights_follow_metric_direction(metric, scores, expected):
    actual = PredictionAggregator.aggregate_folds([np.zeros((3, 1)), np.ones((3, 1))], scores, "weighted_mean", metric)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("scores", [[-0.2, -0.5], [0.0, 0.0]])
@pytest.mark.parametrize("proba", [False, True])
def test_nonpositive_higher_better_weights_use_uniform_mean(scores, proba):
    predictions = {"a": np.array([[0.2, 0.8], [0.6, 0.4]]) if proba else np.array([2.0, 4.0]),
                   "b": np.array([[0.8, 0.2], [0.4, 0.6]]) if proba else np.array([4.0, 8.0])}
    actual = PredictionAggregator.aggregate(predictions, AggregationStrategy.WEIGHTED_MEAN, dict(zip(predictions, scores, strict=True)), proba, "r2")
    expected = np.mean(list(predictions.values()), axis=0)
    if not proba:
        expected = expected.reshape(-1, 1)
    np.testing.assert_allclose(actual, expected)
