"""
Unit tests for TagController.

Tests the sample tagging functionality including:
- Controller matching for tag keyword
- Filter parsing (single, list, dict formats)
- Tag column creation and value assignment
- Training and prediction mode support
- Tag name resolution from filter attributes
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from nirs4all.controllers.data.tag import TagController
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.operators.filters.x_outlier import XOutlierFilter
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext
)
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestTagControllerMatches:
    """Test TagController.matches() method."""

    def test_matches_tag_keyword(self):
        """Should match when keyword is 'tag'."""
        step = {"tag": YOutlierFilter()}
        assert TagController.matches(step, None, "tag") is True

    def test_matches_tag_with_list(self):
        """Should match with list of filters."""
        step = {"tag": [YOutlierFilter(), XOutlierFilter()]}
        assert TagController.matches(step, None, "tag") is True

    def test_matches_tag_with_dict(self):
        """Should match with named dict of filters."""
        step = {"tag": {"outliers": YOutlierFilter()}}
        assert TagController.matches(step, None, "tag") is True

    def test_not_matches_other_keywords(self):
        """Should not match other keywords."""
        step = {"sample_filter": YOutlierFilter()}
        assert TagController.matches(step, None, "sample_filter") is False

    def test_not_matches_branch_keyword(self):
        """Should not match branch keyword."""
        step = {"branch": [["step1"], ["step2"]]}
        assert TagController.matches(step, None, "branch") is False


class TestTagControllerProperties:
    """Test TagController class properties."""

    def test_priority(self):
        """Controller should have priority 5 (same as ExcludeController)."""
        assert TagController.priority == 5

    def test_use_multi_source_false(self):
        """Controller should be dataset-level, not per-source."""
        assert TagController.use_multi_source() is False

    def test_supports_prediction_mode_true(self):
        """Controller should support prediction mode to tag prediction samples."""
        assert TagController.supports_prediction_mode() is True


class TestFilterParsing:
    """Test filter parsing from different configuration formats."""

    @pytest.fixture
    def controller(self):
        return TagController()

    def test_parse_single_filter(self, controller):
        """Single filter should be parsed correctly."""
        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        taggers = controller._parse_taggers(filter_obj)

        assert len(taggers) == 1
        tag_name, parsed_filter = taggers[0]
        assert tag_name == "y_outlier_iqr"  # Default from exclusion_reason
        assert isinstance(parsed_filter, YOutlierFilter)

    def test_parse_list_of_filters(self, controller):
        """List of filters should each get their own tag."""
        filters = [
            YOutlierFilter(method="iqr"),
            XOutlierFilter(method="mahalanobis")
        ]
        taggers = controller._parse_taggers(filters)

        assert len(taggers) == 2
        assert taggers[0][0] == "y_outlier_iqr"
        assert taggers[1][0] == "x_outlier_mahalanobis"

    def test_parse_named_dict(self, controller):
        """Named dict should use dict keys as tag names."""
        config = {
            "is_y_outlier": YOutlierFilter(),
            "is_x_outlier": XOutlierFilter()
        }
        taggers = controller._parse_taggers(config)

        assert len(taggers) == 2
        names = [t[0] for t in taggers]
        assert "is_y_outlier" in names
        assert "is_x_outlier" in names

    def test_parse_filter_with_tag_name(self, controller):
        """Filter with tag_name attribute should use that name."""
        filter_obj = YOutlierFilter(method="iqr", tag_name="custom_tag")
        taggers = controller._parse_taggers(filter_obj)

        assert len(taggers) == 1
        assert taggers[0][0] == "custom_tag"

    def test_parse_non_filter_raises_error(self, controller):
        """Non-SampleFilter should raise TypeError."""
        with pytest.raises(TypeError, match="must be a SampleFilter"):
            controller._parse_taggers("not_a_filter")

    def test_parse_dict_with_non_filter_raises_error(self, controller):
        """Dict with non-SampleFilter value should raise TypeError."""
        with pytest.raises(TypeError, match="must be a SampleFilter"):
            controller._parse_taggers({"tag": "not_a_filter"})


class TestTagNameResolution:
    """Test tag name resolution from filters."""

    @pytest.fixture
    def controller(self):
        return TagController()

    def test_tag_name_from_attribute(self, controller):
        """Should use tag_name attribute if set."""
        filter_obj = YOutlierFilter(tag_name="my_custom_tag")
        name = controller._get_tag_name(filter_obj)
        assert name == "my_custom_tag"

    def test_tag_name_fallback_to_exclusion_reason(self, controller):
        """Should fall back to exclusion_reason if tag_name not set."""
        filter_obj = YOutlierFilter(method="zscore")
        name = controller._get_tag_name(filter_obj)
        assert name == "y_outlier_zscore"

    def test_tag_name_custom_reason(self, controller):
        """Should use custom reason for tag name."""
        filter_obj = YOutlierFilter(reason="bad_samples")
        name = controller._get_tag_name(filter_obj)
        assert name == "bad_samples"


class TestTagControllerExecution:
    """Test TagController.execute() method."""

    @pytest.fixture
    def controller(self):
        return TagController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with spectral data."""
        dataset = Mock()
        dataset.name = "test_dataset"

        # Mock indexer
        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        dataset._indexer = indexer

        # Mock indexer store
        store = Mock()
        store.has_tag_column.return_value = False
        indexer._store = store

        # Mock X and y data
        dataset.x.return_value = np.random.randn(5, 100)
        dataset.y.return_value = np.array([1.0, 2.0, 100.0, 3.0, 4.0])  # Index 2 is outlier

        # Mock tag methods
        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

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

        def next_op():
            runtime.operation_count += 1
            return runtime.operation_count

        runtime.next_op = next_op

        # Mock step_runner with verbose
        step_runner = Mock()
        step_runner.verbose = 0
        runtime.step_runner = step_runner

        # Mock saver for artifact persistence
        saver = Mock()
        artifact = ("tag_test", Mock())
        saver.persist_artifact.return_value = artifact
        runtime.saver = saver

        return runtime

    def test_execute_single_filter(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with single filter should create tag column."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={"tag": YOutlierFilter(method="iqr", threshold=1.5)},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should create tag column
        mock_dataset.add_tag.assert_called_once()
        # Should set tag values
        mock_dataset.set_tag.assert_called_once()

        # Should persist artifact in train mode
        assert len(artifacts) == 1

    def test_execute_multiple_filters(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with multiple filters should create multiple tags."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={
                "tag": [
                    YOutlierFilter(method="iqr"),
                    YOutlierFilter(method="zscore")
                ]
            },
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should create two tag columns
        assert mock_dataset.add_tag.call_count == 2
        # Should set two tag values
        assert mock_dataset.set_tag.call_count == 2
        # Should persist two artifacts
        assert len(artifacts) == 2

    def test_execute_named_dict(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with named dict should use dict keys as tag names."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={
                "tag": {"my_outliers": YOutlierFilter(method="iqr")}
            },
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Check tag name used
        call_args = mock_dataset.add_tag.call_args
        assert call_args[0][0] == "my_outliers"

    def test_execute_prediction_mode(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute in prediction mode should compute fresh tags."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={"tag": YOutlierFilter(method="iqr")},
            metadata={}
        )

        # Add loaded binary for prediction mode
        mock_filter = Mock(spec=SampleFilter)
        mock_filter.get_mask.return_value = np.array([True, True, False, True, True])
        loaded_binaries = [("tag_y_outlier_iqr", mock_filter)]

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="predict",
            loaded_binaries=loaded_binaries
        )

        # Should use loaded filter instead of fitting new one
        mock_filter.get_mask.assert_called_once()
        # Should not persist artifacts in prediction mode
        assert len(artifacts) == 0

    def test_execute_no_samples(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with no samples should return early."""
        # Return empty indices
        mock_dataset._indexer.x_indices.return_value = np.array([])

        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={"tag": YOutlierFilter()},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should not create tags or persist artifacts
        mock_dataset.add_tag.assert_not_called()
        assert len(artifacts) == 0

    def test_execute_empty_config_raises_error(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with empty config should raise ValueError."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={"tag": {}},
            metadata={}
        )

        with pytest.raises(ValueError, match="requires at least one filter"):
            controller.execute(
                step_info=step_info,
                dataset=mock_dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )


class TestTagValueSemantics:
    """Test that tag values have correct semantics (True = flagged as outlier)."""

    @pytest.fixture
    def controller(self):
        return TagController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        dataset._indexer = indexer

        store = Mock()
        store.has_tag_column.return_value = False
        indexer._store = store

        dataset.x.return_value = np.random.randn(5, 50)
        dataset.y.return_value = np.array([1.0, 2.0, 100.0, 3.0, 4.0])

        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

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
        runtime.next_op = Mock(return_value=1)

        step_runner = Mock()
        step_runner.verbose = 0
        runtime.step_runner = step_runner

        saver = Mock()
        saver.persist_artifact.return_value = ("test", Mock())
        runtime.saver = saver

        return runtime

    def test_tag_true_means_flagged(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Tag value True should mean sample is flagged (e.g., outlier)."""
        # Mock filter that flags sample at index 2
        with patch.object(YOutlierFilter, 'fit', return_value=None):
            with patch.object(YOutlierFilter, 'get_mask') as mock_get_mask:
                # get_mask returns True=keep, False=exclude
                # So sample 2 has mask=False (outlier)
                mock_get_mask.return_value = np.array([True, True, False, True, True])

                step_info = ParsedStep(
                    operator=None,
                    keyword="tag",
                    step_type=StepType.WORKFLOW,
                    original_step={"tag": YOutlierFilter()},
                    metadata={}
                )

                controller.execute(
                    step_info=step_info,
                    dataset=mock_dataset,
                    context=mock_context,
                    runtime_context=mock_runtime_context,
                    mode="train"
                )

                # Check set_tag was called with inverted mask
                # (True for outliers, False for non-outliers)
                set_tag_call = mock_dataset.set_tag.call_args
                tag_values = set_tag_call[0][2]  # Third positional arg

                # Expected: [False, False, True, False, False]
                # (only sample 2 is True because it's the outlier)
                expected = [False, False, True, False, False]
                assert tag_values == expected


class TestExistingTagColumn:
    """Test behavior when tag column already exists."""

    @pytest.fixture
    def controller(self):
        return TagController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with existing tag column."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2])
        dataset._indexer = indexer

        store = Mock()
        # Tag column already exists
        store.has_tag_column.return_value = True
        indexer._store = store

        dataset.x.return_value = np.random.randn(3, 50)
        dataset.y.return_value = np.array([1.0, 2.0, 3.0])

        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

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
        runtime.next_op = Mock(return_value=1)

        step_runner = Mock()
        step_runner.verbose = 0
        runtime.step_runner = step_runner

        saver = Mock()
        saver.persist_artifact.return_value = ("test", Mock())
        runtime.saver = saver

        return runtime

    def test_existing_tag_not_recreated(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Existing tag column should not be recreated."""
        step_info = ParsedStep(
            operator=None,
            keyword="tag",
            step_type=StepType.WORKFLOW,
            original_step={"tag": YOutlierFilter()},
            metadata={}
        )

        controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # add_tag should not be called since column exists
        mock_dataset.add_tag.assert_not_called()
        # But set_tag should still be called to update values
        mock_dataset.set_tag.assert_called_once()
