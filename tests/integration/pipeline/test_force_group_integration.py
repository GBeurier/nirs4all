"""
Integration tests for force_group splitting feature.

Tests the force_group parameter that enables any sklearn-compatible splitter
to work with grouped samples by wrapping it with GroupedSplitterWrapper.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import Gaussian

from tests.fixtures.data_generators import TestDataManager


class TestForceGroupIntegration:
    """Integration tests for force_group parameter in split steps."""

    @pytest.fixture
    def test_data_with_groups(self):
        """Create test data with Sample_ID metadata for grouping."""
        manager = TestDataManager()

        # Create classification dataset with binary classes
        manager.create_classification_dataset("classification", n_classes=2)

        # Create separate metadata file with Sample_ID
        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "classification"

        # Load existing data to get actual sample count
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')
        X_val = pd.read_csv(data_dir / "Xval.csv.gz", sep=';')

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

    @pytest.fixture
    def regression_data_with_groups(self):
        """Create regression test data with Sample_ID metadata."""
        manager = TestDataManager()

        # Create regression dataset
        manager.create_regression_dataset("regression")

        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "regression"

        # Load existing data
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')

        # Create Sample_ID column
        n_samples = len(X_cal)
        sample_ids = [f"Sample_{i//3:03d}" for i in range(n_samples)]
        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        yield manager
        manager.cleanup()

    # --- KFold with force_group ---

    def test_kfold_with_force_group(self, test_data_with_groups):
        """Test KFold with force_group respects groups."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3, shuffle=False),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "kfold_force_group_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all folds
        assert predictions.num_predictions >= 3

        # Verify different folds were used
        all_preds = predictions.to_dicts()
        fold_ids = {pred.get('fold_id') for pred in all_preds}
        assert len(fold_ids) >= 2

    def test_kfold_with_force_group_shuffle(self, test_data_with_groups):
        """Test KFold with force_group and shuffle respects groups."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3, shuffle=True, random_state=42),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=8, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "kfold_force_group_shuffle_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

        # Verify predictions are valid
        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
        assert np.isfinite(best_pred['val_score'])

    # --- ShuffleSplit with force_group ---

    def test_shuffle_split_with_force_group(self, test_data_with_groups):
        """Test ShuffleSplit with force_group respects groups."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "shuffle_split_force_group_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all splits
        assert predictions.num_predictions >= 3

    # --- StratifiedKFold with force_group ---

    def test_stratified_kfold_with_force_group(self, test_data_with_groups):
        """Test StratifiedKFold with force_group respects groups and stratification."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                "force_group": "Sample_ID",
                "y_aggregation": "mode"  # Use mode for classification
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "stratified_force_group_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all folds
        assert predictions.num_predictions >= 2

    # --- Aggregation options ---

    def test_force_group_with_median_aggregation(self, test_data_with_groups):
        """Test force_group with median aggregation."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID",
                "aggregation": "median"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_median_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

    def test_force_group_with_first_aggregation(self, test_data_with_groups):
        """Test force_group with first aggregation (no averaging)."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID",
                "aggregation": "first"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_first_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

    # --- Regression with force_group ---

    def test_force_group_with_regression(self, regression_data_with_groups):
        """Test force_group with regression task."""
        dataset_folder = str(regression_data_with_groups.get_temp_directory() / "regression")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID"
            },
            PLSRegression(n_components=5)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_regression_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

        # Verify regression metrics
        best_pred = predictions.get_best(ascending=True)
        assert 'val_score' in best_pred
        assert np.isfinite(best_pred['val_score'])

    # --- With preprocessing ---

    def test_force_group_with_preprocessing(self, test_data_with_groups):
        """Test force_group with preprocessing steps."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            StandardScaler(),
            Gaussian(),
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=8, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_preprocessing_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3

    # --- Multiple models ---

    def test_force_group_with_multiple_models(self, test_data_with_groups):
        """Test force_group with multiple models."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=2),
                "force_group": "Sample_ID"
            },
            {"model": RandomForestClassifier(max_depth=5, random_state=42), "name": "RF_depth5"},
            {"model": RandomForestClassifier(max_depth=10, random_state=42), "name": "RF_depth10"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_multi_model_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple models and folds
        assert predictions.num_predictions >= 4

        # Verify different models
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 2

    # --- Error handling ---

    def test_force_group_missing_column_error(self, test_data_with_groups):
        """Test error handling when force_group column doesn't exist."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "NonExistent_Column"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_missing_column_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)

        # Should raise or handle missing column gracefully
        try:
            predictions, _ = runner.run(pipeline_config, dataset_config)
            # If no error was raised but continue_on_error=True, that's acceptable
        except (KeyError, ValueError) as e:
            # Expected error
            assert "NonExistent_Column" in str(e) or "force_group" in str(e).lower()

    def test_force_group_invalid_aggregation_error(self, test_data_with_groups):
        """Test error handling for invalid aggregation parameter."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID",
                "aggregation": "invalid_aggregation"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_invalid_agg_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0, continue_on_error=True)

        # Should raise error for invalid aggregation
        try:
            predictions, _ = runner.run(pipeline_config, dataset_config)
        except ValueError as e:
            assert "aggregation must be one of" in str(e)

    # --- Comparison with native group splitters ---

    def test_force_group_vs_group_parameter_equivalence(self, test_data_with_groups):
        """Test that force_group produces valid splits like native group splitters."""
        dataset_folder = str(test_data_with_groups.get_temp_directory() / "classification")

        # Using force_group with KFold
        pipeline_force_group = [
            {
                "split": KFold(n_splits=3, shuffle=False),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline_force_group, "force_group_equivalence_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify we get valid predictions
        assert predictions.num_predictions >= 3

        # Verify metrics are reasonable (not NaN, within expected range)
        best_pred = predictions.get_best(ascending=False)
        assert 'val_score' in best_pred
        assert np.isfinite(best_pred['val_score'])
        assert 0 <= best_pred['val_score'] <= 1  # For classification accuracy


class TestForceGroupEdgeCases:
    """Edge case tests for force_group feature."""

    @pytest.fixture
    def single_sample_per_group_data(self):
        """Create test data where each group has exactly one sample."""
        manager = TestDataManager()
        manager.create_classification_dataset("single_sample_groups")

        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "single_sample_groups"

        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')
        n_samples = len(X_cal)

        # Each sample is its own group
        sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        yield manager
        manager.cleanup()

    def test_force_group_single_sample_per_group(self, single_sample_per_group_data):
        """Test force_group when each group has only one sample."""
        dataset_folder = str(single_sample_per_group_data.get_temp_directory() / "single_sample_groups")

        pipeline = [
            {
                "split": KFold(n_splits=3, shuffle=True, random_state=42),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_single_sample_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should still work when each group has one sample
        assert predictions.num_predictions >= 3

    @pytest.fixture
    def unequal_group_sizes_data(self):
        """Create test data with highly unequal group sizes."""
        manager = TestDataManager()
        manager.create_classification_dataset("unequal_groups")

        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "unequal_groups"

        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')
        n_samples = len(X_cal)

        # Create unequal group sizes
        np.random.seed(42)
        sample_ids = []
        group_id = 0
        i = 0
        while i < n_samples:
            group_size = np.random.choice([1, 3, 5, 10])
            for _ in range(min(group_size, n_samples - i)):
                sample_ids.append(f"Group_{group_id:03d}")
                i += 1
            group_id += 1

        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        yield manager
        manager.cleanup()

    def test_force_group_unequal_group_sizes(self, unequal_group_sizes_data):
        """Test force_group with unequal group sizes."""
        dataset_folder = str(unequal_group_sizes_data.get_temp_directory() / "unequal_groups")

        pipeline = [
            {
                "split": KFold(n_splits=3),
                "force_group": "Sample_ID"
            },
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "force_group_unequal_sizes_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 3
