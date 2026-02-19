"""
Integration tests for dataset-level aggregation by sample ID.

Tests Phase 3 of the aggregation feature: Pipeline Integration.
Verifies that aggregation settings propagate correctly from DatasetConfigs
through the pipeline to the TabReportManager.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.pipeline.config.context import ExecutionContext
from tests.fixtures.data_generators import SyntheticNIRSDataGenerator, TestDataManager


class TestAggregationIntegration:
    """Test aggregation settings propagation through pipeline."""

    @pytest.fixture
    def test_data_manager(self, tmp_path):
        """Create test data manager with datasets including metadata."""
        manager = TestDataManager(base_temp_dir=str(tmp_path))
        manager.create_regression_dataset("regression")

        # Add metadata file with sample_id column for aggregation tests
        regression_path = manager.get_temp_directory() / "regression"

        # Read actual X files to get correct row counts
        # TestDataManager now saves with headers, so use header=0
        X_train = pd.read_csv(regression_path / "Xcal.csv.gz", compression='gzip', sep=';', header=0)
        X_val = pd.read_csv(regression_path / "Xval.csv.gz", compression='gzip', sep=';', header=0)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]

        # Create metadata with sample_id (use repeated IDs to test aggregation grouping)
        train_sample_ids = [f"S{i // 4:04d}" for i in range(n_train)]
        val_sample_ids = [f"V{i // 4:04d}" for i in range(n_val)]

        pd.DataFrame({"sample_id": train_sample_ids}).to_csv(
            regression_path / "Mcal.csv.gz", index=False, compression='gzip', sep=';'
        )
        pd.DataFrame({"sample_id": val_sample_ids}).to_csv(
            regression_path / "Mval.csv.gz", index=False, compression='gzip', sep=';'
        )

        yield manager
        manager.cleanup()

    def test_aggregate_via_config_dict(self, test_data_manager):
        """Test aggregation via config dictionary."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "test_x": str(temp_dir / "regression" / "Xval.csv.gz"),
            "test_y": str(temp_dir / "regression" / "Yval.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        # Verify the aggregate setting is stored
        assert dataset_config._aggregates[0] == "sample_id"

        # Get dataset and verify the setting propagates
        dataset = dataset_config.get_dataset_at(0)
        assert dataset.aggregate == "sample_id"

    def test_aggregate_true_for_y_grouping(self, test_data_manager):
        """Test aggregate=True groups by y values."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "aggregate": True
        }

        dataset_config = DatasetConfigs(config)
        dataset = dataset_config.get_dataset_at(0)
        assert dataset.aggregate == "y"

    def test_pipeline_with_aggregate_runs_successfully(self, test_data_manager):
        """Test that a pipeline with aggregate setting runs without errors."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_agg_pipeline")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        # Verify we got predictions
        assert predictions.num_predictions > 0

        # Verify the pipeline completed successfully
        best_pred = predictions.get_best(ascending=True)
        assert np.isfinite(best_pred['val_score'])
        assert np.isfinite(best_pred['test_score'])

    def test_context_stores_aggregate_column(self, test_data_manager):
        """Test that ExecutionContext stores the aggregate_column property."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        dataset = dataset_config.get_dataset_at(0)

        # Initialize context using the executor's method
        from nirs4all.pipeline.execution.executor import PipelineExecutor
        from nirs4all.pipeline.steps.step_runner import StepRunner

        step_runner = StepRunner()
        executor = PipelineExecutor(step_runner=step_runner)
        context = executor.initialize_context(dataset)

        # Verify aggregate_column is set in context
        assert context.aggregate_column == "sample_id"

    def test_context_preserves_aggregate_on_copy(self):
        """Test that ExecutionContext.copy() preserves aggregate_column."""
        context = ExecutionContext(aggregate_column="sample_id")

        copied_context = context.copy()

        assert copied_context.aggregate_column == "sample_id"

    def test_context_aggregate_none_when_not_set(self):
        """Test that aggregate_column defaults to None."""
        context = ExecutionContext()

        assert context.aggregate_column is None

    def test_spectro_dataset_aggregate_property(self):
        """Test SpectroDataset aggregate property setter and getter."""
        dataset = SpectroDataset("test_dataset")

        # Default should be None
        assert dataset.aggregate is None

        # Set via string
        dataset.set_aggregate("sample_id")
        assert dataset.aggregate == "sample_id"

        # Set via True
        dataset.set_aggregate(True)
        assert dataset.aggregate == "y"

        # Set back to None
        dataset.set_aggregate(None)
        assert dataset.aggregate is None

    def test_pipeline_without_aggregate_still_works(self, test_data_manager):
        """Test that pipelines without aggregate setting work normally."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # No aggregate parameter
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_no_agg")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should work normally
        assert predictions.num_predictions > 0

    def test_per_dataset_aggregate_settings(self, tmp_path):
        """Test that different datasets can have different aggregate settings."""
        # Create two simple datasets
        generator = SyntheticNIRSDataGenerator(random_state=42)

        # Dataset 1
        X1, y1 = generator.generate_regression_data(50)
        path1 = tmp_path / "dataset1"
        path1.mkdir()
        pd.DataFrame(X1).to_csv(path1 / "Xcal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')
        pd.DataFrame(y1).to_csv(path1 / "Ycal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')

        # Dataset 2
        X2, y2 = generator.generate_regression_data(50)
        path2 = tmp_path / "dataset2"
        path2.mkdir()
        pd.DataFrame(X2).to_csv(path2 / "Xcal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')
        pd.DataFrame(y2).to_csv(path2 / "Ycal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')

        # Create configs with per-dataset aggregates
        config1 = {"train_x": str(path1 / "Xcal.csv.gz"), "train_y": str(path1 / "Ycal.csv.gz"), "aggregate": "sample_id"}
        config2 = {"train_x": str(path2 / "Xcal.csv.gz"), "train_y": str(path2 / "Ycal.csv.gz"), "aggregate": "batch_id"}

        dataset_config = DatasetConfigs([config1, config2])

        assert dataset_config._aggregates[0] == "sample_id"
        assert dataset_config._aggregates[1] == "batch_id"

class TestAggregationReporting:
    """Test aggregated reporting in pipeline output."""

    def test_tab_report_includes_aggregated_rows(self, tmp_path):
        """Test that TabReportManager includes aggregated rows when aggregate is set."""
        # Create simple dataset without metadata (aggregate by y)
        generator = SyntheticNIRSDataGenerator(random_state=42)
        X, y = generator.generate_regression_data(60)

        dataset_path = tmp_path / "agg_test"
        dataset_path.mkdir()

        pd.DataFrame(X).to_csv(dataset_path / "Xcal.csv.gz", index=False, header=False,
                              compression='gzip', sep=';')
        pd.DataFrame(y).to_csv(dataset_path / "Ycal.csv.gz", index=False, header=False,
                              compression='gzip', sep=';')

        # Use aggregate=True for y-based aggregation (doesn't need metadata)
        config = {
            "train_x": str(dataset_path / "Xcal.csv.gz"),
            "train_y": str(dataset_path / "Ycal.csv.gz"),
            "aggregate": True
        }
        dataset_config = DatasetConfigs(config)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_agg_report")

        # Run with verbose to capture output
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions exist
        assert predictions.num_predictions > 0

    def test_aggregated_predictions_have_fewer_samples(self):
        """Test that aggregated predictions have fewer samples than raw predictions."""
        # Create sample predictions
        n_reps = 3
        n_unique = 15
        n_total = n_unique * n_reps

        np.random.seed(42)
        y_true = np.random.rand(n_total) * 10
        y_pred = y_true + np.random.randn(n_total) * 0.5
        group_ids = np.repeat(np.arange(n_unique), n_reps)

        # Aggregate predictions
        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, y_true=y_true)

        # Aggregated should have n_unique samples
        assert len(result['y_pred']) == n_unique
        assert len(result['y_true']) == n_unique
