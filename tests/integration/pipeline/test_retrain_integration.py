"""
Integration tests for retrain and transfer learning functionality.

Tests the full retrain workflow including:
- Full retrain on new data
- Transfer mode (reuse preprocessing)
- Fine-tuning
- Extract and modify pipelines
"""

import pytest
import numpy as np
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner, StepMode
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay

from tests.fixtures.data_generators import TestDataManager


class TestRetrainIntegration:
    """Integration tests for retrain functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with regression datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("train_data")
        manager.create_regression_dataset("retrain_data")
        yield manager
        manager.cleanup()

    @pytest.fixture
    def trained_model(self, test_data_manager):
        """Train a baseline model for retrain tests."""
        train_folder = str(test_data_manager.get_temp_directory() / "train_data")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            StandardNormalVariate(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "baseline_for_retrain")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_pred = predictions.top(n=1, rank_partition="test")[0]

        return {
            "runner": runner,
            "best_prediction": best_pred,
            "train_folder": train_folder,
        }

    def test_full_retrain(self, test_data_manager, trained_model):
        """Test full retrain mode."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Full retrain
        retrain_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='full',
            dataset_name='retrain_full',
            verbose=0
        )

        # Verify we got predictions
        assert retrain_preds.num_predictions > 0

        best_retrain = retrain_preds.top(n=1, rank_partition="test", display_metrics=['rmse'])[0]
        assert 'rmse' in best_retrain
        assert best_retrain['rmse'] > 0
        assert np.isfinite(best_retrain['rmse'])

    def test_transfer_mode(self, test_data_manager, trained_model):
        """Test transfer mode - reuse preprocessing."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Transfer retrain
        transfer_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='transfer',
            dataset_name='transfer_test',
            verbose=0
        )

        # Verify we got predictions
        assert transfer_preds.num_predictions > 0

        best_transfer = transfer_preds.top(n=1, rank_partition="test", display_metrics=['rmse'])[0]
        assert 'rmse' in best_transfer
        assert np.isfinite(best_transfer['rmse'])

    def test_transfer_with_new_model(self, test_data_manager, trained_model):
        """Test transfer mode with a different model type."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Transfer with new model
        new_model = RandomForestRegressor(n_estimators=10, random_state=42)

        transfer_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='transfer',
            new_model=new_model,
            dataset_name='transfer_new_model',
            verbose=0
        )

        # Verify we got predictions
        assert transfer_preds.num_predictions > 0

        best_transfer = transfer_preds.top(n=1, rank_partition="test", display_metrics=['rmse'])[0]
        assert 'rmse' in best_transfer
        # Model should be different from original
        assert best_transfer['model_name'] != best_pred['model_name']

    def test_finetune_mode(self, test_data_manager, trained_model):
        """Test finetune mode."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Finetune
        finetune_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='finetune',
            epochs=5,
            dataset_name='finetune_test',
            verbose=0
        )

        # Verify we got predictions
        assert finetune_preds.num_predictions > 0

        best_finetune = finetune_preds.top(n=1, rank_partition="test", display_metrics=['rmse'])[0]
        assert 'rmse' in best_finetune
        assert np.isfinite(best_finetune['rmse'])

    def test_extract_pipeline(self, trained_model):
        """Test extracting pipeline for inspection."""
        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Extract pipeline
        extracted = runner.extract(best_pred)

        # Verify extraction
        assert len(extracted) > 0
        assert extracted.model_step_index is not None

        # Should be able to get model step
        model_step = extracted.get_model_step()
        assert model_step is not None

    def test_extract_and_modify(self, test_data_manager, trained_model):
        """Test extracting, modifying, and running pipeline."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Extract pipeline
        extracted = runner.extract(best_pred)

        # Modify model
        new_model = PLSRegression(n_components=8)
        extracted.set_model(new_model)

        # Run modified pipeline
        modified_preds, _ = runner.run(
            pipeline=extracted.steps,
            dataset=retrain_dataset,
            pipeline_name='modified_pipeline'
        )

        # Verify we got predictions
        assert modified_preds.num_predictions > 0

    def test_step_mode_control(self, test_data_manager, trained_model):
        """Test fine-grained step mode control."""
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_data")
        retrain_dataset = DatasetConfigs(retrain_folder)

        runner = trained_model["runner"]
        best_pred = trained_model["best_prediction"]

        # Define step modes - use existing preprocessing
        step_modes = [
            StepMode(step_index=1, mode='predict'),  # Use existing MinMaxScaler
            StepMode(step_index=2, mode='predict'),  # Use existing y_processing
        ]

        controlled_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='full',
            step_modes=step_modes,
            dataset_name='controlled_retrain',
            verbose=0
        )

        # Verify we got predictions
        assert controlled_preds.num_predictions > 0


class TestExportBundleIntegration:
    """Integration tests for bundle export functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager."""
        manager = TestDataManager()
        manager.create_regression_dataset("bundle_test")
        yield manager
        manager.cleanup()

    @pytest.fixture
    def trained_model_for_export(self, test_data_manager):
        """Train a model for export tests."""
        train_folder = str(test_data_manager.get_temp_directory() / "bundle_test")

        pipeline = [
            MinMaxScaler(),
            StandardNormalVariate(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5), "name": "PLS_export"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "export_test")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_pred = predictions.top(n=1, rank_partition="test")[0]

        return {
            "runner": runner,
            "best_prediction": best_pred,
            "train_folder": train_folder,
        }

    def test_export_n4a_bundle(self, trained_model_for_export, tmp_path):
        """Test exporting to .n4a bundle."""
        runner = trained_model_for_export["runner"]
        best_pred = trained_model_for_export["best_prediction"]

        bundle_path = tmp_path / "model.n4a"

        result_path = runner.export(
            source=best_pred,
            output_path=bundle_path,
            format='n4a'
        )

        # Verify bundle was created
        assert result_path.exists()
        assert result_path.suffix == '.n4a'
        assert result_path.stat().st_size > 0

        # Verify it's a valid ZIP
        import zipfile
        assert zipfile.is_zipfile(result_path)

        # Verify contents
        with zipfile.ZipFile(result_path, 'r') as zf:
            names = zf.namelist()
            assert 'manifest.json' in names
            assert 'pipeline.json' in names

    def test_export_n4a_py_script(self, trained_model_for_export, tmp_path):
        """Test exporting to portable Python script."""
        runner = trained_model_for_export["runner"]
        best_pred = trained_model_for_export["best_prediction"]

        script_path = tmp_path / "model.n4a.py"

        result_path = runner.export(
            source=best_pred,
            output_path=script_path,
            format='n4a.py'
        )

        # Verify script was created
        assert result_path.exists()
        assert str(result_path).endswith('.n4a.py')
        assert result_path.stat().st_size > 0

        # Verify it's valid Python (can be compiled)
        with open(result_path, 'r') as f:
            content = f.read()

        # Should compile without syntax errors
        compile(content, str(result_path), 'exec')

        # Verify expected content
        assert 'ARTIFACTS' in content
        assert 'def predict(' in content
        assert 'joblib' in content

    def test_predict_from_bundle(self, trained_model_for_export, tmp_path):
        """Test predicting from exported bundle."""
        runner = trained_model_for_export["runner"]
        best_pred = trained_model_for_export["best_prediction"]
        train_folder = trained_model_for_export["train_folder"]

        # Export bundle
        bundle_path = tmp_path / "predict_test.n4a"
        runner.export(best_pred, bundle_path, format='n4a')

        # Create prediction dataset
        predict_dataset = DatasetConfigs(train_folder)

        # Predict from bundle
        bundle_predictions, _ = runner.predict(
            prediction_obj=str(bundle_path),
            dataset=predict_dataset,
            verbose=0
        )

        # Verify predictions
        assert bundle_predictions is not None
        assert len(bundle_predictions) > 0
        assert np.isfinite(bundle_predictions).all()

    def test_retrain_from_bundle(self, test_data_manager, trained_model_for_export, tmp_path):
        """Test retraining from exported bundle."""
        test_data_manager.create_regression_dataset("retrain_from_bundle")
        retrain_folder = str(test_data_manager.get_temp_directory() / "retrain_from_bundle")

        runner = trained_model_for_export["runner"]
        best_pred = trained_model_for_export["best_prediction"]

        # Export bundle
        bundle_path = tmp_path / "retrain_test.n4a"
        runner.export(best_pred, bundle_path, format='n4a')

        # Retrain from bundle
        retrain_dataset = DatasetConfigs(retrain_folder)

        retrain_preds, _ = runner.retrain(
            source=str(bundle_path),
            dataset=retrain_dataset,
            mode='transfer',
            dataset_name='bundle_retrain',
            verbose=0
        )

        # Verify predictions
        assert retrain_preds.num_predictions > 0


class TestMultiplePreprocessingRetrain:
    """Tests for retrain with multiple preprocessing options."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager."""
        manager = TestDataManager()
        manager.create_regression_dataset("multi_pp_test")
        manager.create_regression_dataset("multi_pp_retrain")
        yield manager
        manager.cleanup()

    def test_retrain_with_feature_augmentation(self, test_data_manager):
        """Test retraining a model with feature augmentation."""
        train_folder = str(test_data_manager.get_temp_directory() / "multi_pp_test")
        retrain_folder = str(test_data_manager.get_temp_directory() / "multi_pp_retrain")

        # Train with feature augmentation
        pipeline = [
            MinMaxScaler(),
            {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay()]},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multi_pp_baseline")
        dataset_config = DatasetConfigs(train_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_pred = predictions.top(n=1, rank_partition="test")[0]

        # Retrain
        retrain_dataset = DatasetConfigs(retrain_folder)

        retrain_preds, _ = runner.retrain(
            source=best_pred,
            dataset=retrain_dataset,
            mode='full',
            dataset_name='multi_pp_retrain',
            verbose=0
        )

        assert retrain_preds.num_predictions > 0
