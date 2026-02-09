"""
Complete integration test suite for nirs4all functionality.

This test suite validates the entire nirs4all pipeline based on the Q examples,
testing regression, classification, preprocessing, predictions, and multi-dataset scenarios.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ShuffleSplit, RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.operators.transforms.targets import RangeDiscretizer
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

from tests.fixtures.data_generators import TestDataManager


class TestNirs4allIntegration:
    """Comprehensive integration tests for nirs4all functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with all datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression", n_train=40, n_val=12)
        manager.create_classification_dataset("classification", n_train=45, n_val=15)
        manager.create_multi_target_dataset("multi_target", n_train=36, n_val=12)
        manager.create_regression_dataset("regression_2", n_train=36, n_val=12)
        manager.create_regression_dataset("regression_3", n_train=36, n_val=12)
        yield manager
        manager.cleanup()

    def test_q1_style_regression_pipeline(self, test_data_manager):
        """Test Q1-style regression pipeline with feature augmentation."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Q1-style pipeline with preprocessing and multiple models
        list_of_preprocessors = [Detrend, FirstDerivative, Gaussian, StandardNormalVariate]

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": {"_or_": list_of_preprocessors, "pick": 1, "count": 2}},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=4)},
            {"model": PLSRegression(n_components=7)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_q1_regression")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions were generated
        assert predictions.num_predictions > 0

        # Test top functionality
        top_models = predictions.top(n=2, rank_metric='rmse', display_metrics=['rmse'])
        assert len(top_models) <= 2

        # Verify metrics are reasonable for regression
        for pred in top_models:
            assert 'rmse' in pred
            # assert 'r2' in pred
            # assert 'mae' in pred
            assert np.isfinite(pred['rmse'])
            assert pred['rmse'] > 0  # RMSE should be positive

        # Test PredictionAnalyzer
        analyzer = PredictionAnalyzer(predictions)
        assert analyzer is not None

    def test_q2_style_multiple_models(self, test_data_manager):
        """Test Q2-style pipeline with multiple sklearn models."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(feature_range=(0.1, 0.8)),
            MultiplicativeScatterCorrection(),
            ShuffleSplit(n_splits=1, random_state=42),
            {"y_processing": MinMaxScaler()},
            {"model": PLSRegression(6)},
            {"model": RandomForestRegressor(n_estimators=3, random_state=42)},
            {"model": ElasticNet(random_state=42)},
            {"model": GradientBoostingRegressor(n_estimators=3, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_q2_models")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple models
        assert predictions.num_predictions >= 4

        # Test that different models produce different results
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 4

    def test_preprocessing_transformations(self, test_data_manager):
        """Test various preprocessing transformations."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Test individual transformations
        transformations = [
            Detrend(),
            FirstDerivative(),
            Gaussian(),
            StandardNormalVariate(),
            SavitzkyGolay(),
        ]

        for transform in transformations:
            pipeline = [
                transform,
                ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                {"model": PLSRegression(n_components=3)},
            ]

            pipeline_config = PipelineConfigs(pipeline, f"test_{transform.__class__.__name__}")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0, f"No predictions for {transform.__class__.__name__}"

            # Verify preprocessing was applied
            best_pred = predictions.get_best(ascending=True)
            assert 'preprocessings' in best_pred

    def test_q7_style_classification_from_regression(self, test_data_manager):
        """Test Q7-style classification by converting regression targets."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Convert regression to classification using RangeDiscretizer
        pipeline = [
            StandardScaler(),
            {"feature_augmentation": [Gaussian, StandardNormalVariate]},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"y_processing": RangeDiscretizer([15, 25, 35, 45])},  # Convert continuous to categories
            {"model": RandomForestClassifier(max_depth=5, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_q7_classification")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

        # Verify classification metrics are used
        best_pred = predictions.get_best(ascending=False)  # Higher accuracy is better
        print(best_pred)
        # assert 'accuracy' in best_pred
        assert 0 <= best_pred['val_score'] <= 1  # Validation score should be between 0 and 1

    def test_native_classification_data(self, test_data_manager):
        """Test classification pipeline with native classification data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": [Detrend, Gaussian]},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": RandomForestClassifier(max_depth=5, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_native_classification")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

        # Verify classification metrics
        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
        assert 0 <= best_pred['val_score'] <= 1  # Validation score should be between 0 and 1
        unique_pred_values = np.unique(best_pred['y_pred'])
        assert len(unique_pred_values) <= 10  # Should be a small number for classification

    def test_q3_style_multi_dataset(self, test_data_manager):
        """Test Q3-style pipeline with multiple datasets."""
        temp_dir = test_data_manager.get_temp_directory()
        dataset_paths = [
            str(temp_dir / "regression"),
            str(temp_dir / "regression_2"),
            str(temp_dir / "regression_3")
        ]

        pipeline = [
            MinMaxScaler(feature_range=(0.1, 0.8)),
            {"feature_augmentation": [
                MultiplicativeScatterCorrection(),
                Gaussian(),
            ]},
            ShuffleSplit(n_splits=1, random_state=42),
            {"y_processing": MinMaxScaler()},
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_q3_multi")
        dataset_config = DatasetConfigs(dataset_paths)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple datasets
        assert len(datasets_predictions) == 3
        assert predictions.num_predictions > 0

        # Verify each dataset has predictions
        for name, dataset_pred in datasets_predictions.items():
            assert 'run_predictions' in dataset_pred
            assert dataset_pred['run_predictions'].num_predictions > 0

    def test_y_processing_variations(self, test_data_manager):
        """Test different y-processing options."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        y_processors = [
            MinMaxScaler(),
            StandardScaler(),
        ]

        for y_proc in y_processors:
            pipeline = [
                MinMaxScaler(),
                ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                {"y_processing": y_proc},
                {"model": PLSRegression(n_components=3)},
            ]

            pipeline_config = PipelineConfigs(pipeline, f"test_y_proc_{y_proc.__class__.__name__}")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0

    def test_cross_validation_strategies(self, test_data_manager):
        """Test different cross-validation strategies."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        cv_strategies = [
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        ]

        for cv in cv_strategies:
            pipeline = [
                MinMaxScaler(),
                cv,
                {"model": PLSRegression(n_components=3)},
            ]

            pipeline_config = PipelineConfigs(pipeline, f"test_cv_{cv.__class__.__name__}")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0

    def test_chart_operations_compatibility(self, test_data_manager):
        """Test that chart operations don't break the pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            "chart_2d",
            MinMaxScaler(),
            "chart_3d",
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_charts")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should work despite chart operations
        assert predictions.num_predictions > 0

    def test_minimal_pipeline_configurations(self, test_data_manager):
        """Test minimal working pipeline configurations."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Ultra minimal
        minimal_pipeline = [
            {"model": PLSRegression(n_components=3)},
        ]

        pipeline_config = PipelineConfigs(minimal_pipeline, "test_minimal")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

    def test_complex_feature_augmentation(self, test_data_manager):
        """Test complex feature augmentation scenarios."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": {
                "_or_": [Detrend, FirstDerivative, Gaussian, StandardNormalVariate],
                "pick": (1, 2),
                "count": 2
            }},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_complex_augmentation")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have multiple predictions from different augmentations
        assert predictions.num_predictions >= 2

        # Verify different preprocessing combinations
        all_preds = predictions.to_dicts()
        preprocessing_sets = {str(pred['preprocessings']) for pred in all_preds}
        assert len(preprocessing_sets) > 1

    def test_predictions_analysis_functionality(self, test_data_manager):
        """Test prediction analysis and reporting functionality."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": {"_or_": [Gaussian, StandardNormalVariate], "pick": 1, "count": 2}},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
            {"model": RandomForestRegressor(n_estimators=3, random_state=42)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_analysis")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test Predictions functionality
        assert predictions.num_predictions > 0

        # Test top
        top_3 = predictions.top(n=3, rank_metric='rmse', display_metrics=['rmse', 'r2'])
        assert len(top_3) <= 6

        # Test get_best
        best = predictions.get_best(ascending=True)
        assert best is not None

        # Test string formatting
        for pred in top_3:
            short_str = Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])
            assert isinstance(short_str, str)
            assert len(short_str) > 0

        # Test PredictionAnalyzer creation
        analyzer = PredictionAnalyzer(predictions)
        assert analyzer is not None

    @pytest.mark.sklearn
    def test_comprehensive_sklearn_models(self, test_data_manager):
        """Test comprehensive sklearn model coverage."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        sklearn_models = [
            PLSRegression(n_components=3),
            RandomForestRegressor(n_estimators=3, random_state=42),
            ElasticNet(alpha=0.5, random_state=42),
            GradientBoostingRegressor(n_estimators=3, random_state=42),
            SVR(kernel='linear', C=1.0),  # Use linear kernel for speed
        ]

        for model in sklearn_models:
            pipeline = [
                MinMaxScaler(),
                ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                {"model": model},
            ]

            pipeline_config = PipelineConfigs(pipeline, f"test_{model.__class__.__name__}")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, continue_on_error=True)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0, f"No predictions for {model.__class__.__name__}"

            # Verify model produces reasonable results
            best_pred = predictions.get_best(ascending=True)
            assert np.isfinite(best_pred['test_score']), f"Invalid RMSE for {model.__class__.__name__}"

    def test_error_handling_robustness(self, test_data_manager):
        """Test error handling and robustness."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Test with potentially problematic configurations
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=3)},
            {"model": PLSRegression(n_components=5)},  # Multiple models should work
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_error_handling")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, continue_on_error=True)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should handle multiple models gracefully
        assert predictions.num_predictions >= 1

    def test_data_shape_validation(self, test_data_manager):
        """Test that the synthetic data has expected properties."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Load the data directly to verify properties
        from nirs4all.data.loaders import load_csv

        X_df, _, _, _, _ = load_csv(str(Path(dataset_folder) / "Xcal.csv.gz"), delimiter=';', has_header=False)
        y_df, _, _, _, _ = load_csv(str(Path(dataset_folder) / "Ycal.csv.gz"), delimiter=';', has_header=False)

        # Verify shapes
        assert X_df.shape[0] == y_df.shape[0]  # Same number of samples
        assert X_df.shape[1] > 1  # Multiple features
        assert y_df.shape[1] == 1  # Single target

        # Verify data ranges are reasonable
        X_array = X_df.to_numpy()
        y_array = y_df.to_numpy()

        assert not np.isnan(X_array).any()  # No NaN values
        assert not np.isnan(y_array).any()  # No NaN values
        assert np.isfinite(X_array).all()   # All finite values
        assert np.isfinite(y_array).all()   # All finite values
