"""
Integration tests for basic nirs4all pipeline functionality.

Tests basic sklearn models training based on Q1/Q2 examples using synthetic data.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

from tests.unit.utils.test_data_generator import TestDataManager


class TestBasicPipelineIntegration:
    """Test basic pipeline functionality with sklearn models."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    def test_simple_pls_pipeline(self, test_data_manager):
        """Test simple PLS regression pipeline (Q1 style)."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_simple_pls")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        # Verify we got predictions
        assert predictions.num_predictions > 0
        assert len(predictions_per_datasets) > 0

        # Verify metrics exist
        best_pred = predictions.get_best(ascending=True)  # Lower RMSE is better
        assert 'val_score' in best_pred
        assert 'test_score' in best_pred

        # Verify predictions are reasonable (not NaN/Inf)
        assert np.isfinite(best_pred['val_score'])
        assert np.isfinite(best_pred['test_score'])

    def test_multiple_models_pipeline(self, test_data_manager):
        """Test pipeline with multiple models (Q2 style)."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(feature_range=(0.1, 0.8)),
            MultiplicativeScatterCorrection(),
            ShuffleSplit(n_splits=2, random_state=42),
            {"y_processing": MinMaxScaler()},
            {"model": PLSRegression(n_components=10)},
            {"model": PLSRegression(n_components=15)},
            {"model": RandomForestRegressor(n_estimators=10, random_state=42)},
            {"model": ElasticNet(random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_multiple_models")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple models
        assert predictions.num_predictions >= 4

        # Test top_k functionality
        top_3 = predictions.top_k(3, 'rmse')
        assert len(top_3) == 3

        # Verify each model has different performance
        rmse_values = [pred['rmse'] for pred in top_3]
        assert len(set(rmse_values)) > 1  # Should have different RMSE values

    def test_preprocessing_pipeline(self, test_data_manager):
        """Test pipeline with preprocessing transformations."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Test various preprocessing steps
        pipeline = [
            StandardScaler(),
            Detrend(),
            Gaussian(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=8)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_preprocessing")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

        # Verify preprocessing information is stored
        best_pred = predictions.get_best(ascending=True)
        assert 'preprocessings' in best_pred
        # Should contain some preprocessing steps
        assert len(best_pred['preprocessings']) > 0

    def test_feature_augmentation_basic(self, test_data_manager):
        """Test basic feature augmentation functionality."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        list_of_preprocessors = [
            Detrend, FirstDerivative, Gaussian, StandardNormalVariate
        ]

        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": {
                "_or_": list_of_preprocessors,
                "size": 1,
                "count": 3
            }},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_feature_augmentation")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have multiple predictions from different augmentations
        assert predictions.num_predictions >= 3

        # Each prediction should have different preprocessing
        pd_predictions = predictions.to_pandas()
        n_unique_preprocessings = pd_predictions['preprocessings'].nunique()
        assert n_unique_preprocessings == 3

    def test_y_processing(self, test_data_manager):
        """Test y-axis preprocessing functionality."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"y_processing": StandardScaler()},
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_y_processing")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

        # Y-processing should be documented
        best_pred = predictions.get_best(ascending=True)
        # Check if y_processing information is available
        assert 'y_pred' in best_pred
        assert np.isfinite(best_pred['y_pred']).all()

    @pytest.mark.sklearn
    def test_various_sklearn_models(self, test_data_manager):
        """Test various sklearn models work with the pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        models_to_test = [
            PLSRegression(n_components=5),
            RandomForestRegressor(n_estimators=5, random_state=42),
            ElasticNet(alpha=1.0, random_state=42),
            GradientBoostingRegressor(n_estimators=5, random_state=42),
            # Note: SVR and MLPRegressor can be slow, so using minimal configs
            SVR(kernel='linear', C=1.0),
            MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42),
        ]

        for model in models_to_test:
            pipeline = [
                MinMaxScaler(),
                ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                {"model": model},
            ]

            pipeline_config = PipelineConfigs(pipeline, f"test_{model.__class__.__name__}")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_files=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0, f"No predictions for {model.__class__.__name__}"

            # Check that prediction is reasonable
            best_pred = predictions.get_best(ascending=True)
            assert np.isfinite(best_pred['val_score']), f"Invalid RMSE for {model.__class__.__name__}"

    def test_prediction_analyzer_basic(self, test_data_manager):
        """Test basic functionality of PredictionAnalyzer."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
            {"model": PLSRegression(n_components=10)},
            {"model": RandomForestRegressor(n_estimators=5, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_analyzer")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test PredictionAnalyzer creation
        analyzer = PredictionAnalyzer(predictions)
        assert analyzer is not None

        # Test top_k functionality
        top_models = predictions.top_k(2, 'rmse')
        assert len(top_models) == 2

        # Test prediction string formatting
        for pred in top_models:
            short_str = Predictions.pred_short_string(pred, metrics=['rmse'])
            assert isinstance(short_str, str)
            assert len(short_str) > 0

    def test_error_handling_invalid_model(self, test_data_manager):
        """Test that invalid models are handled gracefully."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # This should handle the case where a model configuration is invalid
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": "invalid_model_string"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_invalid_model")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)

        # This might raise an error or return empty predictions
        # The exact behavior depends on the implementation
        try:
            predictions, _ = runner.run(pipeline_config, dataset_config)
            # If no error is raised, check that we handle it gracefully
            assert isinstance(predictions, Predictions)
        except Exception as e:
            # If an error is raised, it should be informative
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_minimal_pipeline(self, test_data_manager):
        """Test minimal working pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Absolute minimal pipeline
        pipeline = [
            {"model": PLSRegression(n_components=3)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_minimal")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

    def test_chart_operations(self, test_data_manager):
        """Test chart operations in pipeline (should not break execution)."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            "chart_2d",  # These might create charts or be ignored in headless mode
            MinMaxScaler(),
            "chart_3d",
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_charts")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should still work despite chart operations
        assert predictions.num_predictions > 0
