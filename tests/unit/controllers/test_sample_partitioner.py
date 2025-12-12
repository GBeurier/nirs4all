"""
Unit tests for SamplePartitionerController.

Tests the sample partitioner branching functionality including:
- Controller matching for sample_partitioner syntax
- Filter creation for Y and X outlier detection
- Branch name generation
- Sample partitioning into outliers/inliers branches
- Integration with model training
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from nirs4all.controllers.data.sample_partitioner import (
    SamplePartitionerController,
    _create_partition_filter,
    _generate_branch_names,
)
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext
)
from nirs4all.pipeline.execution.result import StepOutput
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestSamplePartitionerControllerMatches:
    """Test SamplePartitionerController.matches() method."""

    def test_matches_sample_partitioner_by_keyword(self):
        """Should match when 'by' is 'sample_partitioner'."""
        step = {"branch": {"by": "sample_partitioner", "filter": {}}}
        assert SamplePartitionerController.matches(step, None, "branch") is True

    def test_matches_with_filter_config(self):
        """Should match with filter configuration."""
        step = {
            "branch": {
                "by": "sample_partitioner",
                "filter": {"method": "y_outlier", "threshold": 2.0}
            }
        }
        assert SamplePartitionerController.matches(step, None, "branch") is True

    def test_not_matches_outlier_excluder(self):
        """Should not match outlier_excluder syntax."""
        step = {"branch": {"by": "outlier_excluder", "strategies": []}}
        assert SamplePartitionerController.matches(step, None, "branch") is False

    def test_not_matches_regular_branch(self):
        """Should not match regular branch syntax."""
        step = {"branch": [["step1"], ["step2"]]}
        assert SamplePartitionerController.matches(step, None, "branch") is False

    def test_not_matches_named_branch(self):
        """Should not match named branch syntax."""
        step = {"branch": {"snv": ["snv"], "msc": ["msc"]}}
        assert SamplePartitionerController.matches(step, None, "branch") is False

    def test_not_matches_non_branch_keyword(self):
        """Should not match other keywords."""
        step = {"preprocessing": {"by": "sample_partitioner"}}
        assert SamplePartitionerController.matches(step, None, "preprocessing") is False


class TestSamplePartitionerPriority:
    """Test SamplePartitionerController priority."""

    def test_priority_higher_than_outlier_excluder(self):
        """SamplePartitionerController should have priority higher than OutlierExcluderController."""
        # OutlierExcluderController has priority 4, SamplePartitionerController should be 3
        assert SamplePartitionerController.priority == 3


class TestFilterCreation:
    """Test partition filter creation."""

    def test_create_y_outlier_filter(self):
        """Create Y outlier filter."""
        filter_config = {"method": "y_outlier", "threshold": 2.0}
        filter_obj = _create_partition_filter(filter_config)
        assert filter_obj is not None
        assert hasattr(filter_obj, 'fit')
        assert hasattr(filter_obj, 'get_mask')

    def test_create_x_outlier_filter(self):
        """Create X outlier filter (default isolation_forest)."""
        filter_config = {"method": "x_outlier", "contamination": 0.1}
        filter_obj = _create_partition_filter(filter_config)
        assert filter_obj is not None

    def test_create_isolation_forest_filter(self):
        """Create isolation_forest filter directly."""
        filter_config = {"method": "isolation_forest", "contamination": 0.05}
        filter_obj = _create_partition_filter(filter_config)
        assert filter_obj is not None

    def test_create_mahalanobis_filter(self):
        """Create mahalanobis filter."""
        filter_config = {"method": "mahalanobis", "threshold": 3.0}
        filter_obj = _create_partition_filter(filter_config)
        assert filter_obj is not None

    def test_create_lof_filter(self):
        """Create LOF filter."""
        filter_config = {"method": "lof", "contamination": 0.05}
        filter_obj = _create_partition_filter(filter_config)
        assert filter_obj is not None

    def test_default_filter_is_y_outlier(self):
        """Default filter method should be y_outlier."""
        filter_config = {}  # No method specified
        filter_obj = _create_partition_filter(filter_config)
        # Should be YOutlierFilter
        assert "YOutlierFilter" in type(filter_obj).__name__

    def test_unknown_method_raises_error(self):
        """Unknown method should raise ValueError."""
        filter_config = {"method": "unknown_method"}
        with pytest.raises(ValueError, match="Unknown partition method"):
            _create_partition_filter(filter_config)


class TestBranchNameGeneration:
    """Test branch name generation."""

    def test_y_outlier_names(self):
        """Y outlier should generate y_outliers/y_inliers."""
        names = _generate_branch_names({"method": "y_outlier"})
        assert names == ("y_outliers", "y_inliers")

    def test_x_outlier_names(self):
        """X outlier methods should generate x_outliers/x_inliers."""
        names = _generate_branch_names({"method": "x_outlier"})
        assert names == ("x_outliers", "x_inliers")

        names = _generate_branch_names({"method": "isolation_forest"})
        assert names == ("x_outliers", "x_inliers")

        names = _generate_branch_names({"method": "mahalanobis"})
        assert names == ("x_outliers", "x_inliers")

    def test_custom_branch_names(self):
        """Custom branch names should override defaults."""
        names = _generate_branch_names({
            "method": "y_outlier",
            "branch_names": ["bad_samples", "good_samples"]
        })
        assert names == ("bad_samples", "good_samples")


class TestSamplePartitionerExecution:
    """Test SamplePartitionerController.execute() method."""

    @pytest.fixture
    def controller(self):
        return SamplePartitionerController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with spectral data."""
        dataset = Mock()
        dataset.name = "test_dataset"

        # Mock indexer
        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataset._indexer = indexer

        # Mock X and Y data
        X = np.random.randn(10, 100)
        y = np.random.randn(10)
        dataset.x.return_value = X
        dataset.y.return_value = y

        # Mock features for snapshot/restore
        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None
        runtime.next_op = Mock(return_value=1)
        return runtime

    @patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter")
    def test_execute_creates_two_branches(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute should create exactly two branches (outliers and inliers)."""
        # Mock filter that marks 3 samples as outliers
        mock_filter = Mock()
        mock_mask = np.array([True, True, False, True, False, True, True, False, True, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier", "threshold": 1.5}
                }
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should have exactly 2 branches
        assert "branch_contexts" in result_context.custom
        branches = result_context.custom["branch_contexts"]
        assert len(branches) == 2

        # Check branch names
        names = [b["name"] for b in branches]
        assert "y_outliers" in names
        assert "y_inliers" in names

    @patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter")
    def test_branch_contexts_have_partition_info(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Each branch context should have sample_partition info."""
        mock_filter = Mock()
        mock_mask = np.array([True, True, False, True, False, True, True, False, True, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier"}
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]

        # Outliers branch
        outliers_branch = next(b for b in branches if b["name"] == "y_outliers")
        outliers_context = outliers_branch["context"]
        assert "sample_partition" in outliers_context.custom
        assert outliers_context.custom["sample_partition"]["partition_type"] == "outliers"
        # Outliers are where mask is False (indices 2, 4, 7)
        assert outliers_context.custom["sample_partition"]["n_samples"] == 3

        # Inliers branch
        inliers_branch = next(b for b in branches if b["name"] == "y_inliers")
        inliers_context = inliers_branch["context"]
        assert "sample_partition" in inliers_context.custom
        assert inliers_context.custom["sample_partition"]["partition_type"] == "inliers"
        # Inliers are where mask is True (7 samples)
        assert inliers_context.custom["sample_partition"]["n_samples"] == 7

    @patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter")
    def test_sample_indices_correctly_partitioned(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Sample indices should be correctly split between branches."""
        mock_filter = Mock()
        # Samples 2, 4, 7 are outliers (mask=False)
        mock_mask = np.array([True, True, False, True, False, True, True, False, True, True])
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier"}
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]

        outliers_branch = next(b for b in branches if b["name"] == "y_outliers")
        inliers_branch = next(b for b in branches if b["name"] == "y_inliers")

        outlier_indices = set(outliers_branch["context"].custom["sample_partition"]["sample_indices"])
        inlier_indices = set(inliers_branch["context"].custom["sample_partition"]["sample_indices"])

        # Verify correct partition
        assert outlier_indices == {2, 4, 7}
        assert inlier_indices == {0, 1, 3, 5, 6, 8, 9}

        # Verify no overlap
        assert outlier_indices.isdisjoint(inlier_indices)

        # Verify complete coverage
        assert outlier_indices.union(inlier_indices) == set(range(10))

    def test_branch_mode_flag_set(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute should set in_branch_mode flag."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier"}
                }
            },
            metadata={}
        )

        with patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter") as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_mask.return_value = np.ones(10, dtype=bool)
            mock_filter_class.return_value = mock_filter

            result_context, _ = controller.execute(
                step_info=step_info,
                dataset=mock_dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )

        assert result_context.custom.get("in_branch_mode") is True
        assert result_context.custom.get("sample_partitioner_active") is True


class TestSamplePartitionerPredict:
    """Test SamplePartitionerController in prediction mode."""

    def test_supports_prediction_mode(self):
        """Controller should support prediction mode."""
        assert SamplePartitionerController.supports_prediction_mode() is True

    def test_use_multi_source(self):
        """Controller should support multi-source datasets."""
        assert SamplePartitionerController.use_multi_source() is True


class TestBranchMultiplication:
    """Test nested branch context multiplication."""

    @pytest.fixture
    def controller(self):
        return SamplePartitionerController()

    def test_multiply_branch_contexts(self, controller):
        """Test multiplication of existing and new branch contexts."""
        # Existing branches (e.g., from preprocessing branch)
        existing = [
            {"branch_id": 0, "name": "snv", "context": Mock()},
            {"branch_id": 1, "name": "msc", "context": Mock()},
        ]

        # New branches (from sample partitioner)
        new = [
            {"branch_id": 0, "name": "y_outliers", "context": Mock(), "partition_info": {"type": "outliers"}},
            {"branch_id": 1, "name": "y_inliers", "context": Mock(), "partition_info": {"type": "inliers"}},
        ]

        # Mock context copy method
        for item in existing + new:
            mock_ctx = Mock()
            mock_selector = Mock()
            mock_selector.branch_id = None
            mock_selector.branch_name = None
            mock_ctx.selector = mock_selector
            mock_ctx.copy = Mock(return_value=mock_ctx)
            item["context"] = mock_ctx

        result = controller._multiply_branch_contexts(existing, new)

        # Should have 2 x 2 = 4 combinations
        assert len(result) == 4

        # Check names are combined
        names = [r["name"] for r in result]
        assert "snv_y_outliers" in names
        assert "snv_y_inliers" in names
        assert "msc_y_outliers" in names
        assert "msc_y_inliers" in names

        # Check flattened IDs are sequential
        ids = [r["branch_id"] for r in result]
        assert ids == [0, 1, 2, 3]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def controller(self):
        return SamplePartitionerController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        dataset._indexer = indexer

        X = np.random.randn(5, 50)
        dataset.x.return_value = X
        dataset.y.return_value = np.random.randn(5)

        dataset._features = Mock()
        dataset._features.sources = []

        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0
        runtime.step_number = 1
        runtime.operation_count = 0
        runtime.saver = None
        runtime.next_op = Mock(return_value=1)
        return runtime

    @patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter")
    def test_no_outliers_detected(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """When no outliers detected, outliers branch has 0 samples."""
        mock_filter = Mock()
        mock_mask = np.ones(5, dtype=bool)  # All True = no outliers
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier"}
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]
        outliers_branch = next(b for b in branches if b["name"] == "y_outliers")
        inliers_branch = next(b for b in branches if b["name"] == "y_inliers")

        assert outliers_branch["partition_info"]["n_samples"] == 0
        assert inliers_branch["partition_info"]["n_samples"] == 5

    @patch("nirs4all.controllers.data.sample_partitioner.YOutlierFilter")
    def test_all_samples_are_outliers(
        self, mock_filter_class, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """When all samples are outliers, inliers branch has 0 samples."""
        mock_filter = Mock()
        mock_mask = np.zeros(5, dtype=bool)  # All False = all outliers
        mock_filter.get_mask.return_value = mock_mask
        mock_filter_class.return_value = mock_filter

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier"}
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]
        outliers_branch = next(b for b in branches if b["name"] == "y_outliers")
        inliers_branch = next(b for b in branches if b["name"] == "y_inliers")

        assert outliers_branch["partition_info"]["n_samples"] == 5
        assert inliers_branch["partition_info"]["n_samples"] == 0
