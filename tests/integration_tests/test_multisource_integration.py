"""
Integration tests for multi-source regression (Q6 example).

Tests multi-source regression (multiple X arrays) and model reuse.
Based on Q6_multisource.py example.
"""

import pytest
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.analyzers import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

from tests.unit.utils.test_data_generator import TestDataManager


class TestMultisourceIntegration:
    """Integration tests for multi-source regression."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_multi_source_regression(self, test_data_manager):
        """Test Q6 style multi-source regression."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {
                "feature_augmentation": {
                    "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
                    "size": [(2, 3), (1, 3)],
                    "count": 3
                }
            },
            ShuffleSplit(n_splits=2, random_state=42),
            MinMaxScaler(feature_range=(0.1, 0.8)),
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            {"model": PLSRegression(n_components=10), "name": "PLS_10"},
            ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_source_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions for multiple sources
        assert predictions.num_predictions > 0

        # Test top models
        top_models = predictions.top_k(3, 'rmse')
        assert len(top_models) >= 1

        for pred in top_models:
            assert 'rmse' in pred
            assert np.isfinite(pred['rmse'])

    def test_multi_source_with_heatmap_analysis(self, test_data_manager):
        """Test multi-source regression with analyzer heatmap."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": [Gaussian, StandardNormalVariate]},
            ShuffleSplit(n_splits=2, random_state=42),
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            ElasticNet(alpha=0.1, random_state=42),
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_source_heatmap_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test PredictionAnalyzer creation
        analyzer = PredictionAnalyzer(predictions)
        assert analyzer is not None

    def test_multi_source_model_reuse(self, test_data_manager):
        """Test model reuse with multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"model": PLSRegression(n_components=8), "name": "PLS_8"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_source_reuse_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best model
        best_prediction = predictions.top_k(1, partition="test")[0]
        model_id = best_prediction['id']

        print(f"Best model ID for multi-source: {model_id}")

        # Test prediction reuse (on same data for simplicity)
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(dataset_folder)

        pred_result, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)

        # Verify predictions
        assert pred_result is not None
        assert np.isfinite(pred_result).all()

    def test_multi_source_different_models(self, test_data_manager):
        """Test multiple models on multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=1, random_state=42),
            {"model": PLSRegression(n_components=5)},
            {"model": PLSRegression(n_components=10)},
            {"model": ElasticNet(alpha=0.1, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_source_models_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all models
        assert predictions.num_predictions >= 3

        # Verify different models
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 2
