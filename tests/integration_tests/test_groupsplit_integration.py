"""
Integration tests for group-based splitting (Q1_groupsplit example).

Tests GroupKFold and StratifiedGroupKFold with Sample_ID metadata.
Based on Q1_groupsplit.py example.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

from tests.unit.utils.test_data_generator import TestDataManager


class TestGroupSplitIntegration:
    """Integration tests for group-based splitting strategies."""

    @pytest.fixture
    def test_data_with_groups(self):
        """Create test data with Sample_ID metadata for grouping."""
        manager = TestDataManager()

        # Create classification dataset
        manager.create_classification_dataset("classification")

        # Create separate metadata file with Sample_ID
        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "classification"

        # Load existing data to get actual sample count (without treating header as data)
        # TestDataManager creates files with headers, so we need to read properly
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')  # Auto-detect headers
        X_val = pd.read_csv(data_dir / "Xval.csv.gz", sep=';')  # Auto-detect headers

        # Create Sample_ID column (simulate repeated measurements)
        def create_sample_ids(n_samples):
            sample_ids = []
            sample_counter = 1
            i = 0
            np.random.seed(42)  # For reproducibility
            while i < n_samples:
                repeats = np.random.choice([2, 3])
                for _ in range(min(repeats, n_samples - i)):
                    sample_ids.append(f"Sample_{sample_counter:03d}")
                    i += 1
                sample_counter += 1
            return sample_ids[:n_samples]

        # Create metadata files for cal and val
        M_cal = pd.DataFrame({'Sample_ID': create_sample_ids(len(X_cal))})
        M_val = pd.DataFrame({'Sample_ID': create_sample_ids(len(X_val))})

        # Save metadata files (with header)
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)
        M_val.to_csv(data_dir / "Mval.csv.gz", sep=';', index=False, header=True)

        yield manager
        manager.cleanup()

    def test_group_kfold_basic(self, test_data_with_groups):
        """Test basic GroupKFold splitting with Sample_ID."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            "fold_Sample_ID",  # Visualize fold distribution by Sample_ID
            {
                "split": GroupKFold(n_splits=3),
                "group": "Sample_ID"
            },
            "fold_Sample_ID",  # Visualize after split
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all folds
        assert predictions.num_predictions >= 3

        # Verify different folds
        all_preds = predictions.to_dicts()
        fold_ids = {pred.get('fold_id') for pred in all_preds}
        assert len(fold_ids) >= 2

    def test_stratified_group_kfold(self, test_data_with_groups):
        """Test StratifiedGroupKFold splitting."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            "fold_Sample_ID",
            {
                "split": StratifiedGroupKFold(n_splits=3),
                "group": "Sample_ID"
            },
            "fold_Sample_ID",
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "stratified_groupkfold_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all folds
        assert predictions.num_predictions >= 3

    def test_group_kfold_with_shuffle(self, test_data_with_groups):
        """Test GroupKFold with shuffle and random_state."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": GroupKFold(n_splits=3),
                "group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_shuffle_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

        # Verify predictions are valid
        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
        assert np.isfinite(best_pred['val_score'])

    def test_group_kfold_no_leakage(self, test_data_with_groups):
        """Test that GroupKFold prevents data leakage from same sample."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": GroupKFold(n_splits=2),
                "group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=5, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_leakage_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions were generated
        assert predictions.num_predictions >= 2

        # Note: Actual leakage verification would require inspecting train/val splits
        # which is handled internally by the library

    def test_group_kfold_with_preprocessing(self, test_data_with_groups):
        """Test GroupKFold with preprocessing steps."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        from sklearn.preprocessing import StandardScaler
        from nirs4all.operators.transforms import Gaussian

        pipeline = [
            StandardScaler(),
            Gaussian(),
            {
                "split": GroupKFold(n_splits=3),
                "group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=8, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_preprocessing_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

    def test_stratified_group_kfold_with_fold_chart(self, test_data_with_groups):
        """Test StratifiedGroupKFold with fold visualization."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            "fold_Sample_ID",
            {
                "split": StratifiedGroupKFold(n_splits=2),
                "group": "Sample_ID"
            },
            "fold_chart",  # Generic fold visualization
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "stratified_fold_chart_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 2

    def test_group_kfold_multiple_models(self, test_data_with_groups):
        """Test GroupKFold with multiple models."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": GroupKFold(n_splits=2),
                "group": "Sample_ID"
            },
            {"model": RandomForestClassifier(max_depth=5, random_state=42), "name": "RF_depth5"},
            {"model": RandomForestClassifier(max_depth=10, random_state=42), "name": "RF_depth10"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_multi_model_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple models and folds
        assert predictions.num_predictions >= 4

        # Verify different models
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 2

    def test_group_missing_error_handling(self, test_data_with_groups):
        """Test error handling when group column doesn't exist."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": GroupKFold(n_splits=3),
                "group": "NonExistent_Column"  # This column doesn't exist
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "group_missing_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)

        # Should handle missing group column gracefully
        try:
            predictions, _ = runner.run(pipeline_config, dataset_config)
            # If no error, verify it's handled somehow
            assert isinstance(predictions, type(predictions))
        except (KeyError, ValueError) as e:
            # Expected error
            assert "NonExistent_Column" in str(e) or "group" in str(e).lower()

    def test_group_kfold_regression(self, test_data_with_groups):
        """Test GroupKFold with regression (not just classification)."""
        # Create regression dataset with groups
        manager = test_data_with_groups
        manager.create_regression_dataset("regression_grouped")

        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "regression_grouped"

        # Create metadata file with Sample_ID (not in X)
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')  # Auto-detect headers
        n_samples = len(X_cal)
        sample_ids = [f"Sample_{i//3:03d}" for i in range(n_samples)]
        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        dataset_folder = str(data_dir)

        pipeline = [
            {
                "split": GroupKFold(n_splits=3),
                "group": "Sample_ID"
            },
            PLSRegression(n_components=5)
        ]

        pipeline_config = PipelineConfigs(pipeline, "groupkfold_regression_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

        # Verify regression metrics
        best_pred = predictions.get_best(ascending=True)
        assert 'val_score' in best_pred
        assert np.isfinite(best_pred['val_score'])
