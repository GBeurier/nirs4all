"""
Integration tests for prediction and model reuse (Q5 examples).

Tests model persistence, prediction with entry, and prediction with model ID.
Based on Q5_predict.py and Q5_predict_NN.py examples.
"""

import pytest
import numpy as np
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
)

from tests.unit.utils.test_data_generator import TestDataManager


class TestPredictionReuseIntegration:
    """Integration tests for prediction and model reuse functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with regression datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        manager.create_regression_dataset("regression_val")  # For prediction
        yield manager
        manager.cleanup()

    def test_model_persistence_and_prediction_with_entry(self, test_data_manager):
        """Test Q5 style: Train model, save, and predict with prediction entry."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Build and train pipeline
        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay(), Gaussian()]},
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            {"model": PLSRegression(n_components=10), "name": "PLS_10"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "train_for_reuse")
        dataset_config = DatasetConfigs(train_folder)

        # Train with save_files=True to enable persistence
        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best prediction
        best_prediction = predictions.top_k(1, partition="test")[0]
        model_id = best_prediction['id']

        print(f"Best model ID: {model_id}")
        print(f"Best model: {best_prediction['model_name']}")

        # Get reference predictions from training
        reference_predictions = np.array(best_prediction['y_pred'][:5]).flatten()
        print(f"Reference predictions: {reference_predictions}")

        # Method 1: Predict using prediction entry
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        # Make predictions using the best prediction entry
        method1_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)
        method1_array = np.array(method1_predictions[:5]).flatten()
        print(f"Method 1 predictions: {method1_array}")

        # Verify predictions are valid
        assert method1_predictions is not None
        assert len(method1_predictions) > 0
        method1_predictions = np.array(method1_predictions)
        assert np.isfinite(method1_predictions).all()
        assert method1_predictions.shape[0] > 0

    def test_prediction_with_model_id(self, test_data_manager):
        """Test prediction using model ID string."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Train pipeline
        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
            {"model": PLSRegression(n_components=8), "name": "PLS_8"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "train_for_id_reuse")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best model ID
        best_prediction = predictions.top_k(1, partition="test")[0]
        model_id = best_prediction['id']

        print(f"Using model ID: {model_id}")

        # Method 2: Predict using model ID
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        method2_predictions, _ = predictor.predict(model_id, prediction_dataset, verbose=0)

        # Verify predictions
        assert method2_predictions is not None
        assert len(method2_predictions) > 0
        assert np.isfinite(method2_predictions).all()


    def test_prediction_consistency(self, test_data_manager):
        """Test that predictions are consistent when using same model."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Simple pipeline for consistency check
        pipeline = [
            MinMaxScaler(),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "consistency_test")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Predict twice with same model
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        pred1, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)
        pred2, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)

        # Should be identical
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)

    @pytest.mark.tensorflow
    def test_tensorflow_model_reuse(self, test_data_manager):
        """Test Q5_predict_NN style: TensorFlow model persistence and reuse."""
        pytest.importorskip("tensorflow")
        from nirs4all.operators.models.tensorflow.nicon import nicon

        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Build pipeline with TensorFlow model
        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": [StandardNormalVariate(), Gaussian()]},
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
            {
                "model": nicon,
                "train_params": {
                    "epochs": 3,  # Minimal for testing
                    "patience": 10,
                    "verbose": 0
                }
            }
        ]

        pipeline_config = PipelineConfigs(pipeline, "tf_train_for_reuse")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best TensorFlow model
        best_prediction = predictions.top_k(1, partition="test")[0]

        # Predict with TensorFlow model
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        tf_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)

        # Verify TensorFlow predictions
        assert tf_predictions is not None
        assert len(tf_predictions) > 0
        assert np.isfinite(tf_predictions).all()

    def test_prediction_with_different_preprocessing(self, test_data_manager):
        """Test prediction with models having different preprocessing."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Pipeline with multiple preprocessing options
        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": [Gaussian(), StandardNormalVariate(), Haar()]},
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_preproc_reuse")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test prediction with each preprocessing variant
        top_3 = predictions.top_k(3, partition="test")

        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        for pred_entry in top_3:
            pred_result, _ = predictor.predict(pred_entry, prediction_dataset, verbose=0)
            assert pred_result is not None
            assert np.isfinite(pred_result).all()
            print(f"Preprocessing: {pred_entry['preprocessings']}")

    def test_prediction_with_multiple_models(self, test_data_manager):
        """Test prediction reuse with multiple trained models."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            {"model": PLSRegression(n_components=10), "name": "PLS_10"},
            {"model": GradientBoostingRegressor(n_estimators=5, random_state=42), "name": "GBR"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_model_reuse")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Test prediction with each model type
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        # Get predictions from different models
        all_preds = predictions.to_dicts()
        unique_models = {pred['model_name']: pred for pred in all_preds}

        for model_name, pred_entry in unique_models.items():
            pred_result, _ = predictor.predict(pred_entry, prediction_dataset, verbose=0)
            assert pred_result is not None
            assert np.isfinite(pred_result).all()
            print(f"Model: {model_name}")

    def test_prediction_error_handling_missing_model(self, test_data_manager):
        """Test error handling when model file is missing."""
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        # Create fake prediction entry with non-existent model ID
        fake_prediction = {
            'id': 'nonexistent_model_id_12345',
            'model_name': 'FakeModel',
            'preprocessings': []
        }

        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        # Should handle missing model gracefully
        with pytest.raises(Exception):  # Could be FileNotFoundError or custom exception
            predictor.predict(fake_prediction, prediction_dataset, verbose=0)

    def test_prediction_with_fold_id(self, test_data_manager):
        """Test prediction using specific fold from cross-validation."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")
        predict_folder = str(test_data_manager.get_temp_directory() / "regression_val")

        pipeline = [
            MinMaxScaler(),
            RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "fold_specific_reuse")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get predictions from different folds
        all_preds = predictions.to_dicts()
        fold_ids = list({pred.get('fold_id') for pred in all_preds})

        # Test prediction with specific fold
        predictor = PipelineRunner(save_files=False, verbose=0)
        prediction_dataset = DatasetConfigs(predict_folder)

        for fold_id in fold_ids[:2]:  # Test first 2 folds
            fold_pred = [p for p in all_preds if p.get('fold_id') == fold_id][0]
            pred_result, _ = predictor.predict(fold_pred, prediction_dataset, verbose=0)
            assert pred_result is not None
            assert np.isfinite(pred_result).all()

    @pytest.mark.skip(reason="Prediction with numpy arrays not fully supported yet.")
    def test_prediction_with_new_data_format(self, test_data_manager):
        """Test prediction with different data formats."""
        train_folder = str(test_data_manager.get_temp_directory() / "regression")

        # Train model
        pipeline = [
            MinMaxScaler(),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "format_test")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_files=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Create numpy array data for prediction with same number of features as training data
        np.random.seed(42)
        n_features = best_prediction['n_features']
        X_new = np.random.randn(20, n_features)  # 20 samples, same features as training

        # Predict with numpy arrays
        predictor = PipelineRunner(save_files=False, verbose=0)

        # For prediction, pass numpy array as a dict or tuple
        # Since we only have X_new, pass it as test_x
        pred_result, _ = predictor.predict(
            best_prediction,
            {"test_x": X_new},  # Pass as dict for prediction
            verbose=0
        )
        assert pred_result is not None
        assert len(pred_result) == 20
