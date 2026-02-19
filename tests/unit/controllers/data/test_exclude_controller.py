"""
Unit tests for ExcludeController.

Tests the sample exclusion functionality including:
- Controller matching for exclude keyword
- Filter parsing (single and list formats)
- Exclusion mask combination (any/all modes)
- Tag storage for analysis
- Prediction mode behavior (should not run)
- Cascade to augmented samples
"""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest

from nirs4all.controllers.data.exclude import ExcludeController
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.operators.filters.x_outlier import XOutlierFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, RuntimeContext, StepMetadata
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestExcludeControllerMatches:
    """Test ExcludeController.matches() method."""

    def test_matches_exclude_keyword(self):
        """Should match when keyword is 'exclude'."""
        step = {"exclude": YOutlierFilter()}
        assert ExcludeController.matches(step, None, "exclude") is True

    def test_matches_exclude_with_list(self):
        """Should match with list of filters."""
        step = {"exclude": [YOutlierFilter(), XOutlierFilter()]}
        assert ExcludeController.matches(step, None, "exclude") is True

    def test_matches_exclude_with_mode(self):
        """Should match with mode option."""
        step = {"exclude": [YOutlierFilter()], "mode": "all"}
        assert ExcludeController.matches(step, None, "exclude") is True

    def test_not_matches_tag_keyword(self):
        """Should not match tag keyword."""
        step = {"tag": YOutlierFilter()}
        assert ExcludeController.matches(step, None, "tag") is False

    def test_not_matches_sample_filter_keyword(self):
        """Should not match old sample_filter keyword."""
        step = {"sample_filter": {"filters": [YOutlierFilter()]}}
        assert ExcludeController.matches(step, None, "sample_filter") is False

    def test_not_matches_branch_keyword(self):
        """Should not match branch keyword."""
        step = {"branch": [["step1"], ["step2"]]}
        assert ExcludeController.matches(step, None, "branch") is False

class TestExcludeControllerProperties:
    """Test ExcludeController class properties."""

    def test_priority(self):
        """Controller should have priority 5 (same as TagController)."""
        assert ExcludeController.priority == 5

    def test_use_multi_source_false(self):
        """Controller should be dataset-level, not per-source."""
        assert ExcludeController.use_multi_source() is False

    def test_supports_prediction_mode_false(self):
        """Controller should NOT support prediction mode (never exclude prediction samples)."""
        assert ExcludeController.supports_prediction_mode() is False

class TestFilterParsing:
    """Test filter parsing from different configuration formats."""

    @pytest.fixture
    def controller(self):
        return ExcludeController()

    def test_parse_single_filter(self, controller):
        """Single filter should be parsed correctly."""
        step = {"exclude": YOutlierFilter(method="iqr", threshold=1.5)}
        filters, mode, cascade = controller._parse_config(step)

        assert len(filters) == 1
        assert isinstance(filters[0], YOutlierFilter)
        assert mode == "any"  # default
        assert cascade is True  # default

    def test_parse_list_of_filters(self, controller):
        """List of filters should be parsed."""
        step = {
            "exclude": [
                YOutlierFilter(method="iqr"),
                XOutlierFilter(method="mahalanobis")
            ]
        }
        filters, mode, cascade = controller._parse_config(step)

        assert len(filters) == 2
        assert isinstance(filters[0], YOutlierFilter)
        assert isinstance(filters[1], XOutlierFilter)

    def test_parse_mode_any(self, controller):
        """Mode 'any' should be parsed."""
        step = {"exclude": [YOutlierFilter()], "mode": "any"}
        filters, mode, cascade = controller._parse_config(step)
        assert mode == "any"

    def test_parse_mode_all(self, controller):
        """Mode 'all' should be parsed."""
        step = {"exclude": [YOutlierFilter()], "mode": "all"}
        filters, mode, cascade = controller._parse_config(step)
        assert mode == "all"

    def test_parse_cascade_false(self, controller):
        """cascade_to_augmented=False should be parsed."""
        step = {"exclude": YOutlierFilter(), "cascade_to_augmented": False}
        filters, mode, cascade = controller._parse_config(step)
        assert cascade is False

    def test_invalid_mode_raises_error(self, controller):
        """Invalid mode should raise ValueError."""
        step = {"exclude": YOutlierFilter(), "mode": "invalid"}
        with pytest.raises(ValueError, match="mode must be 'any' or 'all'"):
            controller._parse_config(step)

    def test_parse_non_filter_raises_error(self, controller):
        """Non-SampleFilter should raise TypeError."""
        step = {"exclude": "not_a_filter"}
        with pytest.raises(TypeError, match="must be a SampleFilter"):
            controller._parse_config(step)

class TestExcludeControllerExecution:
    """Test ExcludeController.execute() method."""

    @pytest.fixture
    def controller(self):
        return ExcludeController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with spectral data."""
        dataset = Mock()
        dataset.name = "test_dataset"

        # Mock indexer
        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        indexer.mark_excluded.return_value = 1  # number excluded
        dataset._indexer = indexer

        # Mock indexer store for tag operations
        store = Mock()
        store.has_tag_column.return_value = False
        indexer._store = store

        # Mock X and y data (index 2 has outlier value)
        dataset.x.return_value = np.random.randn(5, 100)
        dataset.y.return_value = np.array([1.0, 2.0, 100.0, 3.0, 4.0])

        # Mock tag methods
        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

        return dataset

    @pytest.fixture
    def mock_context(self):
        ctx = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )
        # Add with_partition method
        ctx.with_partition = Mock(return_value=ctx)
        return ctx

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
        artifact = ("exclude_test", Mock())
        saver.persist_artifact.return_value = artifact
        runtime.saver = saver

        return runtime

    def test_exclude_single_filter(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute with single filter should mark samples as excluded."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should mark samples as excluded
        mock_dataset._indexer.mark_excluded.assert_called_once()
        # Should create tag for analysis
        mock_dataset.add_tag.assert_called_once()
        mock_dataset.set_tag.assert_called_once()
        # Should persist artifact
        assert len(artifacts) == 1

    def test_exclude_multiple_filters_mode_any(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Mode 'any' should exclude if ANY filter flags the sample."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={
                "exclude": [
                    YOutlierFilter(method="iqr"),
                    YOutlierFilter(method="zscore")
                ],
                "mode": "any"
            },
            metadata={}
        )

        # Mock filters - first flags samples 2,3, second flags sample 2
        with patch.object(YOutlierFilter, 'fit', return_value=None):
            with patch.object(YOutlierFilter, 'get_mask', side_effect=[
                np.array([True, True, False, False, True]),  # samples 2,3 flagged
                np.array([True, True, False, True, True]),   # sample 2 flagged
            ]):
                result_context, artifacts = controller.execute(
                    step_info=step_info,
                    dataset=mock_dataset,
                    context=mock_context,
                    runtime_context=mock_runtime_context,
                    mode="train"
                )

        # Check mark_excluded was called with samples 2 and 3 (any mode)
        call_args = mock_dataset._indexer.mark_excluded.call_args
        excluded_samples = call_args[0][0]
        # In "any" mode, we use np.all - keep only if ALL filters say keep
        # Filter 1: keeps [0,1,4], Filter 2: keeps [0,1,3,4]
        # All keep: [0,1,4] -> exclude: [2,3]
        assert 2 in excluded_samples
        assert 3 in excluded_samples
        # Should have 2 tags and 2 artifacts
        assert mock_dataset.add_tag.call_count == 2
        assert len(artifacts) == 2

    def test_exclude_multiple_filters_mode_all(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Mode 'all' should exclude only if ALL filters flag the sample."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={
                "exclude": [
                    YOutlierFilter(method="iqr"),
                    YOutlierFilter(method="zscore")
                ],
                "mode": "all"
            },
            metadata={}
        )

        # Mock filters - first flags samples 2,3, second flags sample 2
        with patch.object(YOutlierFilter, 'fit', return_value=None):
            with patch.object(YOutlierFilter, 'get_mask', side_effect=[
                np.array([True, True, False, False, True]),  # samples 2,3 flagged
                np.array([True, True, False, True, True]),   # sample 2 flagged
            ]):
                result_context, artifacts = controller.execute(
                    step_info=step_info,
                    dataset=mock_dataset,
                    context=mock_context,
                    runtime_context=mock_runtime_context,
                    mode="train"
                )

        # Check mark_excluded was called only with sample 2 (all mode)
        call_args = mock_dataset._indexer.mark_excluded.call_args
        excluded_samples = call_args[0][0]
        # In "all" mode, we use np.any - keep if ANY filter says keep
        # Filter 1: keeps [0,1,4], Filter 2: keeps [0,1,3,4]
        # Any keep: [0,1,3,4] -> exclude: [2]
        assert excluded_samples == [2]

    def test_exclude_not_applied_in_prediction(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute in prediction mode should do nothing (never exclude prediction samples)."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter()},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="predict"
        )

        # Should not call any methods on dataset
        mock_dataset._indexer.mark_excluded.assert_not_called()
        mock_dataset.add_tag.assert_not_called()
        # No artifacts in prediction mode
        assert len(artifacts) == 0

    def test_exclude_cascades_to_augmented(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Exclusion should cascade to augmented samples by default."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter()},
            metadata={}
        )

        controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Check cascade_to_augmented was True
        call_args = mock_dataset._indexer.mark_excluded.call_args
        assert call_args[1]['cascade_to_augmented'] is True

    def test_exclude_no_cascade_option(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """cascade_to_augmented=False should prevent cascading."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter(), "cascade_to_augmented": False},
            metadata={}
        )

        controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Check cascade_to_augmented was False
        call_args = mock_dataset._indexer.mark_excluded.call_args
        assert call_args[1]['cascade_to_augmented'] is False

class TestExclusionTagStorage:
    """Test that exclusion tags are stored for analysis."""

    @pytest.fixture
    def controller(self):
        return ExcludeController()

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2, 3, 4])
        indexer.mark_excluded.return_value = 1
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
        ctx = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )
        ctx.with_partition = Mock(return_value=ctx)
        return ctx

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

    def test_tag_name_has_excluded_prefix(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Tag names should have 'excluded_' prefix."""
        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter(method="iqr")},
            metadata={}
        )

        controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Check tag name starts with 'excluded_'
        call_args = mock_dataset.add_tag.call_args
        tag_name = call_args[0][0]
        assert tag_name.startswith("excluded_")

    def test_tag_values_true_for_excluded(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Tag value True should mean sample was flagged for exclusion."""
        # Mock filter that flags sample at index 2
        with patch.object(YOutlierFilter, 'fit', return_value=None):
            with patch.object(YOutlierFilter, 'get_mask') as mock_get_mask:
                # get_mask returns True=keep, False=exclude
                mock_get_mask.return_value = np.array([True, True, False, True, True])

                step_info = ParsedStep(
                    operator=None,
                    keyword="exclude",
                    step_type=StepType.WORKFLOW,
                    original_step={"exclude": YOutlierFilter()},
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
                set_tag_call = mock_dataset.set_tag.call_args
                tag_values = set_tag_call[0][2]  # Third positional arg

                # Expected: [False, False, True, False, False]
                expected = [False, False, True, False, False]
                assert tag_values == expected

class TestEdgeCases:
    """Test edge cases for ExcludeController."""

    @pytest.fixture
    def controller(self):
        return ExcludeController()

    @pytest.fixture
    def mock_context(self):
        ctx = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )
        ctx.with_partition = Mock(return_value=ctx)
        return ctx

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

    def test_no_samples_returns_early(self, controller, mock_context, mock_runtime_context):
        """Execute with no samples should return early."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([])  # No samples
        dataset._indexer = indexer

        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter()},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should not call mark_excluded
        indexer.mark_excluded.assert_not_called()
        assert len(artifacts) == 0

    def test_no_y_values_returns_early(self, controller, mock_context, mock_runtime_context):
        """Execute with no y values should return early."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2])
        dataset._indexer = indexer

        dataset.x.return_value = np.random.randn(3, 50)
        dataset.y.return_value = None  # No y values

        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": YOutlierFilter()},
            metadata={}
        )

        result_context, artifacts = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should not call mark_excluded
        indexer.mark_excluded.assert_not_called()
        assert len(artifacts) == 0

    def test_empty_config_raises_error(self, controller, mock_context, mock_runtime_context):
        """Execute with empty config should raise ValueError."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2])
        dataset._indexer = indexer

        dataset.x.return_value = np.random.randn(3, 50)
        dataset.y.return_value = np.array([1.0, 2.0, 3.0])

        step_info = ParsedStep(
            operator=None,
            keyword="exclude",
            step_type=StepType.WORKFLOW,
            original_step={"exclude": []},  # Empty list
            metadata={}
        )

        with pytest.raises(ValueError, match="requires at least one filter"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )

    def test_all_samples_excluded_warning(self, controller, mock_context, mock_runtime_context):
        """Should warn when all samples would be excluded."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2])
        indexer.mark_excluded.return_value = 2  # 2 excluded, 1 kept
        dataset._indexer = indexer

        store = Mock()
        store.has_tag_column.return_value = False
        indexer._store = store

        dataset.x.return_value = np.random.randn(3, 50)
        dataset.y.return_value = np.array([1.0, 2.0, 3.0])
        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

        # Mock filter that flags ALL samples
        with patch.object(YOutlierFilter, 'fit', return_value=None):
            with patch.object(YOutlierFilter, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = np.array([False, False, False])  # All excluded

                step_info = ParsedStep(
                    operator=None,
                    keyword="exclude",
                    step_type=StepType.WORKFLOW,
                    original_step={"exclude": YOutlierFilter()},
                    metadata={}
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    controller.execute(
                        step_info=step_info,
                        dataset=dataset,
                        context=mock_context,
                        runtime_context=mock_runtime_context,
                        mode="train"
                    )

                    # Should have emitted a warning
                    assert len(w) == 1
                    assert "exclude ALL" in str(w[0].message)

                # Should still mark some samples (not all) as excluded
                call_args = indexer.mark_excluded.call_args
                excluded = call_args[0][0]
                # At least one sample should be kept
                assert len(excluded) < 3

    def test_filter_error_handled_gracefully(self, controller, mock_context, mock_runtime_context):
        """Filter error should be handled with neutral mask."""
        dataset = Mock()
        dataset.name = "test_dataset"

        indexer = Mock()
        indexer.x_indices.return_value = np.array([0, 1, 2])
        indexer.mark_excluded.return_value = 0
        dataset._indexer = indexer

        store = Mock()
        store.has_tag_column.return_value = False
        indexer._store = store

        dataset.x.return_value = np.random.randn(3, 50)
        dataset.y.return_value = np.array([1.0, 2.0, 3.0])
        dataset.add_tag = Mock()
        dataset.set_tag = Mock()

        # Mock filter that raises error
        with patch.object(YOutlierFilter, 'fit', side_effect=ValueError("insufficient data")):
            step_info = ParsedStep(
                operator=None,
                keyword="exclude",
                step_type=StepType.WORKFLOW,
                original_step={"exclude": YOutlierFilter()},
                metadata={}
            )

            # Enable verbose to trigger warning log
            mock_runtime_context.step_runner.verbose = 1

            # Should not raise, should use neutral mask
            result_context, artifacts = controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=mock_context,
                runtime_context=mock_runtime_context,
                mode="train"
            )

            # Should still complete without error
            # With neutral mask (all keep), no samples excluded
            call_args = indexer.mark_excluded.call_args
            # Called but with empty list
            if call_args is not None:
                excluded = call_args[0][0]
                assert len(excluded) == 0
