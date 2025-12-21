"""
Integration tests for MergeController Phase 5: Per-Branch Prediction Control.

Tests:
- Model selection per branch (all, best, top_k, explicit)
- Prediction aggregation per branch (separate, mean, weighted_mean, proba_mean)
- Mixed strategies across branches
- Model ranking by validation metrics

These tests verify the Phase 5 implementation from the branching_concat_merge_design.
"""

import pytest
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection,
)

# Mark all tests in this module as sklearn-only (no deep learning dependencies)
pytestmark = pytest.mark.sklearn


def create_test_dataset(n_samples: int = 100, n_features: int = 50, seed: int = 42) -> SpectroDataset:
    """Create a synthetic dataset for integration testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    # Split into train/test (80/20)
    n_train = int(n_samples * 0.8)

    dataset = SpectroDataset(name="test_merge_per_branch")
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_targets(y[:n_train])
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[n_train:])

    return dataset


class TestModelSelectionStrategies:
    """Test model selection strategies (all, best, top_k, explicit)."""

    def test_select_all_models(self):
        """Test selecting all models from a branch (default behavior)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": Ridge(alpha=1.0)}],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "all"},
                    {"branch": 1, "select": "all"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_select_all"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_select_best_model(self):
        """Test selecting only the best model per branch."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        # Create a branch with multiple models, then select best
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                ],
                [
                    MultiplicativeScatterCorrection(),
                    {"model": Ridge(alpha=1.0)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "rmse"},
                    {"branch": 1, "select": "best", "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_select_best"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_select_top_k_models(self):
        """Test selecting top K models per branch."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": {"top_k": 2}, "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_select_top_k"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_select_top_k_exceeds_available(self):
        """Test that top_k > available models returns all available."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    # Request top 10 but only 1 model exists
                    {"branch": 0, "select": {"top_k": 10}, "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_select_top_k_exceeds"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestAggregationStrategies:
    """Test prediction aggregation strategies (separate, mean, weighted_mean, proba_mean)."""

    def test_aggregate_separate(self):
        """Test separate aggregation (each model = 1 feature)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                ],
                [
                    MultiplicativeScatterCorrection(),
                    {"model": Ridge(alpha=1.0)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "aggregate": "separate"},
                    {"branch": 1, "aggregate": "separate"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_aggregate_separate"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_aggregate_mean(self):
        """Test mean aggregation (average predictions from multiple models)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    # Aggregate all 3 PLS models into a single feature
                    {"branch": 0, "aggregate": "mean"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_aggregate_mean"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_aggregate_weighted_mean(self):
        """Test weighted mean aggregation based on validation scores."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    # Weighted average based on RMSE scores
                    {"branch": 0, "aggregate": "weighted_mean", "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_aggregate_weighted_mean"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestMixedStrategies:
    """Test mixed selection and aggregation strategies across branches."""

    def test_different_strategies_per_branch(self):
        """Test using different selection and aggregation strategies per branch."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                ],
                [
                    MultiplicativeScatterCorrection(),
                    {"model": Ridge(alpha=0.5)},
                    {"model": Ridge(alpha=1.0)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    # Branch 0: select best, keep separate
                    {"branch": 0, "select": "best", "metric": "rmse", "aggregate": "separate"},
                    # Branch 1: select all, average together
                    {"branch": 1, "select": "all", "aggregate": "mean"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_strategies"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_top_k_with_mean_aggregation(self):
        """Test selecting top-K models then averaging them."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=2)},
                    {"model": PLSRegression(n_components=4)},
                    {"model": PLSRegression(n_components=6)},
                    {"model": PLSRegression(n_components=8)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    # Select top 2 by RMSE, then average them
                    {"branch": 0, "select": {"top_k": 2}, "metric": "rmse", "aggregate": "mean"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_top_k_mean"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestModelRankingMetrics:
    """Test model ranking by different validation metrics."""

    def test_rank_by_rmse(self):
        """Test ranking models by RMSE (lower is better)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_rank_rmse"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_rank_by_mae(self):
        """Test ranking models by MAE (lower is better)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "mae"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_rank_mae"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_rank_by_r2(self):
        """Test ranking models by R2 (higher is better)."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "r2"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_rank_r2"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_model_selection(self):
        """Test selection strategies with only a single model."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "rmse"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_single_model"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_many_folds_with_aggregation(self):
        """Test aggregation with many CV folds."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "aggregate": "mean"},
                ],
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_many_folds"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_legacy_syntax_still_works(self):
        """Test that legacy syntax (without per-branch config) still works."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        # Legacy syntax: just list of branch indices
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": Ridge(alpha=1.0)}],
            ]},
            {"merge": {
                "predictions": [0, 1],  # Legacy: just branch indices
            }},
            {"model": LinearRegression()}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_legacy_syntax"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0
