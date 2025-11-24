"""Tests for predictions scores management."""

import json
import numpy as np
import pytest
from nirs4all.data.predictions import Predictions
from nirs4all.data._predictions.ranker import PredictionRanker
from nirs4all.core import metrics as evaluator

@pytest.fixture
def base_prediction_params():
    """Base parameters for creating predictions."""
    return {
        "dataset_name": "test_dataset",
        "dataset_path": "/path/to/dataset",
        "config_name": "test_config",
        "config_path": "/path/to/config",
        "pipeline_uid": "pipe_123",
        "step_idx": 0,
        "op_counter": 1,
        "model_name": "PLS_5",
        "model_classname": "PLSRegression",
        "fold_id": 0,
        "n_samples": 3,
        "n_features": 10,
    }

class TestPredictionsScores:
    """Test suite for Predictions scores functionality."""

    def test_add_prediction_with_scores(self, base_prediction_params):
        """Test adding a prediction with pre-computed scores."""
        predictions = Predictions()

        scores = {
            "train": {"rmse": 0.1, "r2": 0.9},
            "val": {"rmse": 0.2, "r2": 0.8},
            "test": {"rmse": 0.3, "r2": 0.7}
        }

        pred_id = predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            test_score=0.3,
            scores=scores,
            **base_prediction_params
        )

        assert pred_id is not None

        # Verify storage
        df = predictions.to_dataframe()
        row = df.filter(pl.col("id") == pred_id).to_dicts()[0]
        assert "scores" in row
        assert row["scores"] is not None

        stored_scores = json.loads(row["scores"])
        assert stored_scores["test"]["rmse"] == 0.3
        assert stored_scores["test"]["r2"] == 0.7

    def test_top_uses_precomputed_scores(self, base_prediction_params):
        """Test that top() uses pre-computed scores for ranking."""
        predictions = Predictions()

        # Add prediction with specific scores
        # We set y_true/y_pred to values that would give a DIFFERENT score
        # to verify that the pre-computed score is used.

        # Real RMSE of these arrays is sqrt(0.01) = 0.1
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])

        scores = {
            "test": {"rmse": 999.0, "r2": -999.0} # Fake scores
        }

        predictions.add_prediction(
            partition="test",
            y_true=y_true,
            y_pred=y_pred,
            test_score=999.0,
            scores=scores,
            **base_prediction_params
        )

        # Rank by RMSE
        results = predictions.top(1, rank_partition="test", rank_metric="rmse")

        assert len(results) == 1
        # Should use the fake score, not the calculated one
        assert results[0]["rank_score"] == 999.0

        # Rank by R2
        results = predictions.top(1, rank_partition="test", rank_metric="r2")
        assert results[0]["rank_score"] == -999.0

    def test_top_fallback_calculation(self, base_prediction_params):
        """Test that top() falls back to calculation if score is missing."""
        predictions = Predictions()

        # Real RMSE is 0.1
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])

        # Scores dict missing 'mae'
        scores = {
            "test": {"rmse": 0.1}
        }

        predictions.add_prediction(
            partition="test",
            y_true=y_true,
            y_pred=y_pred,
            test_score=0.1,
            scores=scores,
            **base_prediction_params
        )

        # Rank by MAE (not in scores)
        results = predictions.top(1, rank_partition="test", rank_metric="mae")

        assert len(results) == 1
        # Should have calculated MAE: mean(|0.1|, |0.1|) = 0.1
        assert abs(results[0]["rank_score"] - 0.1) < 1e-6

    def test_display_metrics_from_scores(self, base_prediction_params):
        """Test that display metrics are pulled from scores."""
        predictions = Predictions()

        scores = {
            "test": {"rmse": 0.1, "custom_metric": 123.45}
        }

        predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            test_score=0.1,
            scores=scores,
            **base_prediction_params
        )

        results = predictions.top(
            1,
            rank_partition="test",
            rank_metric="rmse",
            display_metrics=["custom_metric"]
        )

        assert len(results) == 1
        assert results[0]["custom_metric"] == 123.45

import polars as pl
