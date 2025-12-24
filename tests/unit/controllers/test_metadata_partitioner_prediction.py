"""
Unit tests for MetadataPartitionerController prediction mode (Phase 4).

Tests the prediction mode routing functionality including:
- Sample routing based on metadata values in prediction mode
- Handling of unknown metadata values
- Bundle export and import with partitioner routing info
- Disjoint branch merge in prediction mode
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from nirs4all.controllers.data.metadata_partitioner import (
    MetadataPartitionerController,
    MetadataPartitionConfig,
    _parse_metadata_partition_config,
    _build_partition_groups,
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


class TestMetadataPartitionerPredictionMode:
    """Test MetadataPartitionerController in prediction mode."""

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with metadata."""
        dataset = Mock()
        dataset.metadata = pd.DataFrame({
            "site": ["A", "A", "B", "B", "C"],
            "sample_id": [0, 1, 2, 3, 4]
        })
        dataset.num_samples = 5
        return dataset

    @pytest.fixture
    def mock_context(self):
        """Create mock execution context."""
        context = Mock()
        context.custom = {}

        # Set up selector mock
        selector = Mock()
        selector.processing = []
        selector.branch_path = None

        # The copy() should return a proper mock context with dict custom
        def make_copy():
            new_ctx = Mock()
            new_ctx.custom = {}
            new_ctx.selector = Mock()
            new_ctx.selector.processing = []
            new_ctx.selector.branch_path = None
            new_ctx.selector.with_branch = Mock(return_value=new_ctx.selector)
            return new_ctx

        context.copy = make_copy
        context.selector = selector
        context.selector.with_branch = Mock(return_value=selector)
        return context

    @pytest.fixture
    def mock_runtime_context(self):
        """Create mock runtime context."""
        runtime = Mock()
        runtime.step_number = 1
        runtime.trace_recorder = None
        runtime.step_runner = None
        runtime.artifact_loader = None
        runtime.artifact_load_counter = {}
        return runtime

    def test_supports_prediction_mode(self):
        """MetadataPartitionerController should support prediction mode."""
        assert MetadataPartitionerController.supports_prediction_mode() is True

    def test_execute_dispatches_to_prediction_mode(self, mock_dataset, mock_context, mock_runtime_context):
        """Execute should dispatch to prediction mode when mode='predict'."""
        controller = MetadataPartitionerController()

        step = {
            "branch": [Mock()],
            "by": "metadata_partitioner",
            "column": "site",
        }
        step_info = Mock()
        step_info.original_step = step

        # Mock the _execute_prediction_mode method
        with patch.object(controller, '_execute_prediction_mode') as mock_predict:
            mock_predict.return_value = (mock_context, StepOutput())

            controller.execute(
                step_info=step_info,
                dataset=mock_dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="predict"
            )

            mock_predict.assert_called_once()

    def test_prediction_mode_routes_samples(self, mock_dataset, mock_context, mock_runtime_context):
        """Prediction mode should route samples by metadata values (structural test).

        This test verifies the _execute_prediction_mode method exists and can be called.
        Full integration testing should use real dataset objects.
        """
        controller = MetadataPartitionerController()

        # Verify the prediction mode method exists and has the expected signature
        assert hasattr(controller, '_execute_prediction_mode')

        # Verify the method accepts the expected parameters
        import inspect
        sig = inspect.signature(controller._execute_prediction_mode)
        expected_params = ['step_info', 'dataset', 'context', 'runtime_context', 'config']
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"

    def test_prediction_mode_missing_metadata_column_raises_error(self, mock_context, mock_runtime_context):
        """Prediction mode should raise error if metadata column is missing."""
        controller = MetadataPartitionerController()

        # Dataset with different column
        dataset = Mock()
        dataset.metadata = pd.DataFrame({"other_column": [1, 2, 3]})
        dataset.num_samples = 3

        step = {
            "branch": [],
            "by": "metadata_partitioner",
            "column": "site",  # Not in metadata
        }
        step_info = Mock()
        step_info.original_step = step

        controller._snapshot_features = Mock(return_value=[])

        with pytest.raises(ValueError, match="not found in prediction data"):
            controller._execute_prediction_mode(
                step_info=step_info,
                dataset=dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                config=_parse_metadata_partition_config(step),
                mode="predict"
            )

    def test_prediction_mode_no_metadata_raises_error(self, mock_context, mock_runtime_context):
        """Prediction mode should raise error if dataset has no metadata."""
        controller = MetadataPartitionerController()

        dataset = Mock()
        dataset.metadata = None
        dataset.num_samples = 3

        step = {
            "branch": [],
            "by": "metadata_partitioner",
            "column": "site",
        }
        step_info = Mock()
        step_info.original_step = step

        with pytest.raises(ValueError, match="Dataset has no metadata"):
            controller._execute_prediction_mode(
                step_info=step_info,
                dataset=dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                config=_parse_metadata_partition_config(step),
                mode="predict"
            )


class TestBundlePartitionerRouting:
    """Test bundle partitioner routing info."""

    def test_bundle_metadata_includes_partitioner_routing(self):
        """BundleMetadata should include partitioner_routing field."""
        from nirs4all.pipeline.bundle.loader import BundleMetadata

        data = {
            "bundle_format_version": "1.0",
            "partitioner_routing": {
                "1": {
                    "column": "site",
                    "partitions": ["A", "B", "C"],
                }
            }
        }

        metadata = BundleMetadata.from_dict(data)
        assert metadata.partitioner_routing is not None
        assert "1" in metadata.partitioner_routing
        assert metadata.partitioner_routing["1"]["column"] == "site"

    def test_bundle_loader_has_partitioner_routing_check(self):
        """BundleLoader should have has_partitioner_routing method."""
        from nirs4all.pipeline.bundle.loader import BundleLoader

        # Create mock loader
        with patch.object(BundleLoader, '__init__', lambda x, y: None):
            loader = BundleLoader.__new__(BundleLoader)
            loader.metadata = Mock()
            loader.metadata.partitioner_routing = {}

            assert loader.has_partitioner_routing() is False

            loader.metadata.partitioner_routing = {"1": {"column": "site"}}
            assert loader.has_partitioner_routing() is True

    def test_bundle_loader_get_required_metadata_columns(self):
        """BundleLoader should return required metadata columns."""
        from nirs4all.pipeline.bundle.loader import BundleLoader

        with patch.object(BundleLoader, '__init__', lambda x, y: None):
            loader = BundleLoader.__new__(BundleLoader)
            loader.metadata = Mock()
            loader.metadata.partitioner_routing = {
                "1": {"column": "site"},
                "3": {"column": "variety"},
            }

            columns = loader.get_required_metadata_columns()
            assert "site" in columns
            assert "variety" in columns
            assert len(columns) == 2


class TestDisjointMergePredictionMode:
    """Test disjoint merge controller in prediction mode."""

    def test_detect_disjoint_branches_with_metadata_partition(self):
        """Should detect metadata partitioner branches as disjoint."""
        from nirs4all.controllers.data.merge import detect_disjoint_branches, BranchType

        context = Mock()
        context.custom = {
            "metadata_partition": {
                "column": "site",
                "sample_indices": [0, 1, 2],
            }
        }

        branch_contexts = [
            {
                "branch_id": 0,
                "context": context,
                "partition_info": {"sample_indices": [0, 1, 2]},
            }
        ]

        analysis = detect_disjoint_branches(branch_contexts)
        assert analysis.is_disjoint is True
        assert analysis.branch_type == BranchType.METADATA_PARTITIONER

    def test_disjoint_merge_prediction_mode_called(self):
        """Disjoint merge should call prediction mode handler."""
        from nirs4all.controllers.data.merge import MergeController

        controller = MergeController()

        # This is a structural test - just verify the method exists
        assert hasattr(controller, '_execute_disjoint_branch_merge_predict_mode')


class TestLoadPartitionRoutingInfo:
    """Test loading partition routing info from trace."""

    def test_load_from_trace_metadata(self):
        """Should load partition info from trace metadata."""
        controller = MetadataPartitionerController()

        # Mock trace with partition info
        mock_step = Mock()
        mock_step.metadata = {"partitions": ["A", "B", "C"]}

        mock_trace = Mock()
        mock_trace.get_step = Mock(return_value=mock_step)

        runtime_context = Mock()
        runtime_context.trace = mock_trace
        runtime_context.step_number = 1

        result = controller._load_partition_routing_info(runtime_context)

        assert result is not None
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_returns_none_when_no_trace(self):
        """Should return None when no trace available."""
        controller = MetadataPartitionerController()

        runtime_context = Mock()
        runtime_context.trace = None
        runtime_context.step_number = 1
        runtime_context.artifact_loader = None

        result = controller._load_partition_routing_info(runtime_context)

        assert result is None
