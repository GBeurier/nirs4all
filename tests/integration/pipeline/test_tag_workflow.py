"""
Integration tests for tag workflow.

Tests the full tag workflow including:
- TagController pipeline integration
- Tag column creation and value persistence
- Tag-based filtering in DataSelector
- Training and prediction mode support
- Interaction with other pipeline steps
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.filters.x_outlier import XOutlierFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.pipeline.config.context import DataSelector
from tests.fixtures.data_generators import TestDataManager


class TestTagWorkflowBasic:
    """Test basic tag workflow functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression", n_train=80, n_val=20)
        yield manager
        manager.cleanup()

    def test_tag_step_creates_tag_column(self, test_data_manager):
        """Tag step should create tag column in dataset indexer."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {"tag": YOutlierFilter(method="iqr", threshold=1.5)},
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_tag_basic")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        # Verify we got predictions (pipeline ran successfully)
        assert predictions.num_predictions > 0

        # Get the dataset that was used (predictions_per_datasets is dict with dataset names as keys)
        dataset_info = list(predictions_per_datasets.values())[0]
        dataset = dataset_info["dataset"]

        # Verify tag column was created
        tag_columns = dataset._indexer._store.get_tag_column_names()
        assert "y_outlier_iqr" in tag_columns

    def test_tag_with_custom_name(self, test_data_manager):
        """Tag with custom tag_name should use that name."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {"tag": YOutlierFilter(method="iqr", tag_name="is_y_outlier")},
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_tag_custom_name")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        dataset = list(predictions_per_datasets.values())[0]["dataset"]
        tag_columns = dataset._indexer._store.get_tag_column_names()
        assert "is_y_outlier" in tag_columns

    def test_tag_named_dict_format(self, test_data_manager):
        """Tag with named dict should use dict keys as tag names."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {"tag": {
                "y_outliers": YOutlierFilter(method="iqr"),
                "x_outliers": XOutlierFilter(method="pca_residual", n_components=5)
            }},
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_tag_named_dict")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        dataset = list(predictions_per_datasets.values())[0]["dataset"]
        tag_columns = dataset._indexer._store.get_tag_column_names()
        assert "y_outliers" in tag_columns
        assert "x_outliers" in tag_columns

class TestTagDataSelectorIntegration:
    """Test tag_filters in DataSelector."""

    def test_with_tag_filter_method(self):
        """DataSelector.with_tag_filter should add tag filter."""
        selector = DataSelector(partition="train")
        new_selector = selector.with_tag_filter("is_outlier", False)

        assert "is_outlier" in new_selector.tag_filters
        assert new_selector.tag_filters["is_outlier"] is False

    def test_chained_tag_filters(self):
        """Multiple tag filters can be chained."""
        selector = DataSelector(partition="train")
        new_selector = (
            selector
            .with_tag_filter("is_outlier", False)
            .with_tag_filter("cluster_id", [1, 2, 3])
        )

        assert len(new_selector.tag_filters) == 2
        assert new_selector.tag_filters["is_outlier"] is False
        assert new_selector.tag_filters["cluster_id"] == [1, 2, 3]

    def test_tag_filter_copy(self):
        """Copy should preserve tag_filters."""
        selector = DataSelector(partition="train", tag_filters={"test": True})
        copied = selector.copy()

        assert copied.tag_filters == {"test": True}
        # Verify it's a copy not a reference
        copied.tag_filters["new"] = False
        assert "new" not in selector.tag_filters

class TestTagWorkflowWithFiltering:
    """Test tag workflow combined with filtering."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with dataset containing outliers."""
        manager = TestDataManager()
        # Create dataset with known outliers
        manager.create_regression_dataset("regression", n_train=80, n_val=20)
        yield manager
        manager.cleanup()

    def test_tag_then_filter_pipeline(self, test_data_manager):
        """Pipeline with tag step followed by sample_filter should work."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            # First tag outliers
            {"tag": {"y_outliers": YOutlierFilter(method="iqr", threshold=1.5)}},
            # Then filter them out
            {"sample_filter": {
                "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
                "mode": "any"
            }},
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_tag_then_filter")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        # Pipeline should complete successfully
        assert predictions.num_predictions > 0

        # Tag column should exist
        dataset = list(predictions_per_datasets.values())[0]["dataset"]
        tag_columns = dataset._indexer._store.get_tag_column_names()
        assert "y_outliers" in tag_columns

class TestTagMultipleFilters:
    """Test tag step with multiple filters."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression", n_train=80, n_val=20)
        yield manager
        manager.cleanup()

    def test_tag_list_of_filters(self, test_data_manager):
        """Tag with list of filters should create multiple tag columns."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {"tag": [
                YOutlierFilter(method="iqr", tag_name="iqr_outlier"),
                YOutlierFilter(method="zscore", tag_name="zscore_outlier")
            ]},
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_tag_list")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        dataset = list(predictions_per_datasets.values())[0]["dataset"]
        tag_columns = dataset._indexer._store.get_tag_column_names()
        assert "iqr_outlier" in tag_columns
        assert "zscore_outlier" in tag_columns
