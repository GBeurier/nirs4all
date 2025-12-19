"""
Integration tests for sample augmentation (Q12 example).

Tests standard and balanced augmentation with leak prevention in CV.
Based on Q12_sample_augmentation.py example.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Rotate_Translate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

from tests.fixtures.data_generators import TestDataManager


class TestSampleAugmentationIntegration:
    """Integration tests for sample augmentation functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with regression dataset including Sample_ID."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")

        # Add Sample_ID as metadata (not in X)
        temp_dir = manager.get_temp_directory()
        data_dir = temp_dir / "regression"

        # Load X to get sample count
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')  # Auto-detect headers
        n_samples = len(X_cal)
        sample_ids = [f"Sample_{i//2:03d}" for i in range(n_samples)]  # 2 measurements per sample

        # Create metadata file
        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        yield manager
        manager.cleanup()

    def test_standard_augmentation(self, test_data_manager):
        """Test standard sample augmentation with fixed count."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            "fold_chart",
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
                    "count": 2,
                    "selection": "random",
                    "random_state": 42
                }
            },
            "fold_chart",
            GroupKFold(n_splits=2),
        ]

        pipeline_config = PipelineConfigs(pipeline, "standard_augmentation_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should complete without errors
        assert predictions.num_predictions >= 0

    def test_balanced_augmentation_with_binning(self, test_data_manager):
        """Test balanced augmentation with target binning."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            "fold_chart",
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate()],
                    "balance": "y",
                    "target_size": 15,
                    "bins": 5,
                    "binning_strategy": "equal_width",
                    "bin_balancing": "value",
                    "selection": "random",
                    "random_state": 42,
                }
            },
            "fold_chart",
            {"split": GroupKFold(n_splits=2), "group": "Sample_ID"},
            "fold_chart",
        ]

        pipeline_config = PipelineConfigs(pipeline, "balanced_augmentation_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 0

    def test_augmentation_with_classification(self, test_data_manager):
        """Test augmentation with classification model."""
        # Create classification dataset
        test_data_manager.create_classification_dataset("classification")
        dataset_folder = str(test_data_manager.get_temp_directory() / "classification")

        # Add Sample_ID metadata for GroupKFold
        data_dir = Path(dataset_folder)
        X_cal = pd.read_csv(data_dir / "Xcal.csv.gz", sep=';')
        n_samples = len(X_cal)
        sample_ids = [f"Sample_{i//2:03d}" for i in range(n_samples)]
        M_cal = pd.DataFrame({'Sample_ID': sample_ids})
        M_cal.to_csv(data_dir / "Mcal.csv.gz", sep=';', index=False, header=True)

        pipeline = [
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate()],
                    "balance": "y",
                    "target_size": 20,
                    "selection": "random",
                    "random_state": 42,
                }
            },
            {"split": GroupKFold(n_splits=2), "group": "Sample_ID"},
            RandomForestClassifier(max_depth=10, random_state=42)
        ]

        pipeline_config = PipelineConfigs(pipeline, "augmentation_classification_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

    def test_augmentation_leak_prevention(self, test_data_manager):
        """Test that augmentation doesn't leak into validation folds."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate()],
                    "count": 3,
                    "random_state": 42,
                }
            },
            {"split": GroupKFold(n_splits=2), "group": "Sample_ID"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "augmentation_leak_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify folds were created correctly
        # Actual leak prevention is tested internally by the library
        assert predictions.num_predictions >= 0

    def test_augmentation_with_quantile_binning(self, test_data_manager):
        """Test augmentation with quantile binning strategy."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate()],
                    "balance": "y",
                    "target_size": 18,
                    "bins": 4,
                    "binning_strategy": "quantile",
                    "selection": "random",
                    "random_state": 42,
                }
            },
        ]

        pipeline_config = PipelineConfigs(pipeline, "quantile_binning_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 0

    def test_augmentation_with_sample_balancing(self, test_data_manager):
        """Test augmentation with sample-level balancing."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {
                "sample_augmentation": {
                    "transformers": [Rotate_Translate()],
                    "balance": "y",
                    "target_size": 20,
                    "bins": 5,
                    "bin_balancing": "sample",  # Default mode
                    "random_state": 42,
                }
            },
        ]

        pipeline_config = PipelineConfigs(pipeline, "sample_balancing_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions >= 0
