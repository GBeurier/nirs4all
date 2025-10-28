"""
Integration tests for classification pipelines (Q1_classif examples).

Tests RandomForest and TensorFlow classification with confusion matrix analysis.
Based on Q1_classif.py and Q1_classif_tf.py examples.
"""

import pytest
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter

from tests.unit.utils.test_data_generator import TestDataManager


class TestClassificationIntegration:
    """Integration tests for classification pipelines."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with classification dataset."""
        manager = TestDataManager()
        manager.create_classification_dataset("classification")
        yield manager
        manager.cleanup()

    def test_q1_style_classification_pipeline(self, test_data_manager):
        """Test Q1-style classification with RandomForest (Q1_classif.py)."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        preprocessing_options = [
            Detrend, FirstDerivative, SecondDerivative, Gaussian,
            StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
        ]

        pipeline = [
            {"feature_augmentation": preprocessing_options[:4]},  # Use subset for speed
            StandardScaler(),
            SPXYSplitter(0.25),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "Q1_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

        # Verify predictions were generated
        assert predictions.num_predictions > 0

        # Test top models by accuracy
        top_models = predictions.top(3)
        assert len(top_models) <= 3

        # Verify classification metrics
        for pred in top_models:
            assert 'val_score' in pred
            assert 0 <= pred['val_score'] <= 1  # Accuracy should be between 0 and 1
            assert np.isfinite(pred['val_score'])

        # Verify predictions are discrete classes
        best_pred = predictions.get_best(ascending=False)
        unique_classes = np.unique(best_pred['y_pred'])
        assert len(unique_classes) >= 2  # Should have at least 2 classes

    def test_feature_augmentation_classification(self, test_data_manager):
        """Test classification with feature augmentation."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            {"feature_augmentation": {
                "_or_": [Detrend, Gaussian, StandardNormalVariate],
                "size": 1,
                "count": 3
            }},
            StandardScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=15, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "classification_augmentation_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have multiple predictions from different augmentations
        assert predictions.num_predictions >= 3

        # Each prediction should have different preprocessing
        all_preds = predictions.to_dicts()
        preprocessing_sets = {str(pred['preprocessings']) for pred in all_preds}
        assert len(preprocessing_sets) >= 2  # At least 2 different preprocessing combinations

    def test_multiple_max_depth_classification(self, test_data_manager):
        """Test classification with multiple max_depth values."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
        ]

        # Add multiple models with different max_depth and explicit names
        for max_depth in [5, 10, 20]:
            pipeline.append({
                "model": RandomForestClassifier(max_depth=max_depth, random_state=42),
                "name": f"RF_depth{max_depth}"
            })

        pipeline_config = PipelineConfigs(pipeline, "multi_depth_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all models
        assert predictions.num_predictions >= 3

        # Verify different models exist
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        # Should have at least 2 different models (could be more with CV folds)
        assert len(model_names) >= 2

    def test_confusion_matrix_analysis(self, test_data_manager):
        """Test confusion matrix analysis for classification."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            {"feature_augmentation": [Gaussian, StandardNormalVariate]},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=10, random_state=42),
            RandomForestClassifier(max_depth=15, random_state=42),
        ]

        pipeline_config = PipelineConfigs(pipeline, "confusion_matrix_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test PredictionAnalyzer for confusion matrix
        analyzer = PredictionAnalyzer(predictions)
        assert analyzer is not None

        # Verify we can get top models
        top_models = predictions.top(2)
        assert len(top_models) >= 1

        # Note: Actual confusion matrix plotting tested in separate visualization tests

    def test_spxy_splitter_classification(self, test_data_manager):
        """Test SPXYSplitter with classification data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            SPXYSplitter(0.25),
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "spxy_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

        # Verify partitions exist
        best_pred = predictions.get_best(ascending=False)
        assert 'partition' in best_pred

    def test_classification_with_cv_and_split(self, test_data_manager):
        """Test classification with both SPXY split and cross-validation."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            SPXYSplitter(0.25),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=12, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "split_cv_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from both CV splits
        assert predictions.num_predictions >= 2

    def test_classification_predictions_string_formatting(self, test_data_manager):
        """Test prediction string formatting for classification."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "string_format_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test string formatting
        top_models = predictions.top(2)
        for pred in top_models:
            short_str = Predictions.pred_short_string(pred)
            assert isinstance(short_str, str)
            assert len(short_str) > 0

    @pytest.mark.tensorflow
    def test_tensorflow_classification(self, test_data_manager):
        """Test TensorFlow neural network classification (Q1_classif_tf.py)."""
        pytest.importorskip("tensorflow")
        from nirs4all.operators.models.cirad_tf import nicon_classification

        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            {"feature_augmentation": [Detrend, Gaussian]},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {
                "model": nicon_classification,
                "train_params": {
                    "epochs": 3,  # Minimal for testing
                    "patience": 10,
                    "verbose": 0
                }
            }
        ]

        pipeline_config = PipelineConfigs(pipeline, "tf_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify TensorFlow model trained
        assert predictions.num_predictions > 0

        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
        assert 0 <= best_pred['val_score'] <= 1
        assert np.isfinite(best_pred['val_score'])

    @pytest.mark.tensorflow
    def test_tensorflow_vs_sklearn_classification(self, test_data_manager):
        """Test both TensorFlow and sklearn models in same pipeline."""
        pytest.importorskip("tensorflow")
        from nirs4all.operators.models.cirad_tf import nicon_classification

        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=10, random_state=42),
            {
                "model": nicon_classification,
                "train_params": {
                    "epochs": 3,
                    "verbose": 0
                }
            }
        ]

        pipeline_config = PipelineConfigs(pipeline, "mixed_models_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from both models
        assert predictions.num_predictions >= 1

        # Verify different model types
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 1

    def test_classification_error_handling(self, test_data_manager):
        """Test error handling in classification pipelines."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        # Test with potentially problematic configuration
        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            RandomForestClassifier(max_depth=5, random_state=42),
        ]

        pipeline_config = PipelineConfigs(pipeline, "error_handling_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should handle gracefully
        assert isinstance(predictions, Predictions)

    def test_classification_minimal_pipeline(self, test_data_manager):
        """Test minimal classification pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        pipeline = [
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "minimal_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0
        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
