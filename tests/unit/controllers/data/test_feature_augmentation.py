"""
Unit tests for FeatureAugmentationController action modes.

Tests the three action modes:
- extend: Add new processings to set (linear growth, no chaining)
- add: Chain on all + keep originals (multiplicative with originals)
- replace: Chain on all + discard originals (multiplicative without originals)
"""

import pytest
import numpy as np
from copy import deepcopy
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

from nirs4all.controllers.data.feature_augmentation import (
    FeatureAugmentationController,
    VALID_ACTIONS,
)
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext,
)
from nirs4all.pipeline.execution.result import StepResult


@dataclass
class MockParsedStep:
    """Mock ParsedStep for testing."""
    operator: Any
    original_step: dict
    step_type: str = "feature_augmentation"


class MockStepResult:
    """Mock step execution result."""
    def __init__(self, updated_context: ExecutionContext, artifacts: List = None):
        self.updated_context = updated_context
        self.artifacts = artifacts or []


class TestFeatureAugmentationControllerMatches:
    """Test FeatureAugmentationController.matches() method."""

    def test_matches_feature_augmentation_keyword(self):
        """Should match when keyword is 'feature_augmentation'."""
        step = {"feature_augmentation": [Mock()]}
        assert FeatureAugmentationController.matches(step, None, "feature_augmentation") is True

    def test_not_matches_other_keywords(self):
        """Should not match other keywords."""
        assert FeatureAugmentationController.matches({}, None, "model") is False
        assert FeatureAugmentationController.matches({}, None, "transform") is False
        assert FeatureAugmentationController.matches({}, None, "sample_augmentation") is False


class TestValidActions:
    """Test action validation."""

    def test_valid_actions_defined(self):
        """Verify valid actions are defined."""
        assert "extend" in VALID_ACTIONS
        assert "add" in VALID_ACTIONS
        assert "replace" in VALID_ACTIONS
        assert len(VALID_ACTIONS) == 3

    def test_invalid_action_raises_error(self):
        """Invalid action mode should raise ValueError."""
        controller = FeatureAugmentationController()

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw"]

        # Create context
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        # Create step with invalid action
        step_info = MockParsedStep(
            operator=Mock(),
            original_step={
                "feature_augmentation": [Mock()],
                "action": "invalid_mode"
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()

        with pytest.raises(ValueError, match="Invalid action"):
            controller.execute(step_info, mock_dataset, context, runtime_context)


class TestExtendMode:
    """Test 'extend' action mode."""

    def test_extend_mode_no_chaining(self):
        """Extend mode should add processings without chaining."""
        controller = FeatureAugmentationController()

        # Track what operations are executed with what processing
        execution_log = []

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            execution_log.append({
                "operation": op,
                "processing": deepcopy(ctx.selector.processing)
            })
            # Simulate adding a new processing
            return MockStepResult(ctx, [])

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.n_sources = 1

        # Simulate that after operations, dataset has new processings
        call_count = [0]

        def get_processings(sdx):
            call_count[0] += 1
            return ["raw", "raw_SNV", "raw_Gaussian"]

        mock_dataset.features_processings = get_processings

        # Create context with existing processing
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        # Create step with extend action
        op1 = Mock(name="SNV")
        op2 = Mock(name="Gaussian")
        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [op1, op2],
                "action": "extend"
            }
        )

        # Setup runtime context
        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # Both operations should have been executed from base processing
        assert len(execution_log) == 2

        # Context should be updated with all processings
        assert updated_context.selector.processing == [["raw", "raw_SNV", "raw_Gaussian"]]

    def test_extend_mode_skips_none_operations(self):
        """Extend mode should skip None operations."""
        controller = FeatureAugmentationController()

        execution_count = [0]

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            execution_count[0] += 1
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [None, Mock(), None],
                "action": "extend"
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        controller.execute(step_info, mock_dataset, context, runtime_context)

        # Only one operation should be executed (the non-None one)
        assert execution_count[0] == 1


class TestAddMode:
    """Test 'add' action mode (current/legacy behavior)."""

    def test_add_mode_chains_on_originals(self):
        """Add mode should chain operations on original processings."""
        controller = FeatureAugmentationController()

        execution_log = []

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            execution_log.append({
                "operation": op,
                "processing": deepcopy(ctx.selector.processing)
            })
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw_A", "raw_A_SNV", "raw_A_Gaussian"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw_A"]])
        )

        op1 = Mock(name="SNV")
        op2 = Mock(name="Gaussian")
        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [op1, op2],
                "action": "add"
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # Both operations should start from original processing
        assert len(execution_log) == 2
        assert execution_log[0]["processing"] == [["raw_A"]]
        assert execution_log[1]["processing"] == [["raw_A"]]

        # Result should include originals + chained versions
        assert updated_context.selector.processing == [["raw_A", "raw_A_SNV", "raw_A_Gaussian"]]

    def test_add_mode_is_default(self):
        """Add mode should be the default when no action is specified."""
        controller = FeatureAugmentationController()

        execution_log = []

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            execution_log.append({
                "processing": deepcopy(ctx.selector.processing)
            })
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw", "raw_SNV"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        # No action specified - should default to add
        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [Mock()]
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        controller.execute(step_info, mock_dataset, context, runtime_context)

        # Should behave like add mode (chain from original)
        assert execution_log[0]["processing"] == [["raw"]]


class TestReplaceMode:
    """Test 'replace' action mode."""

    def test_replace_mode_excludes_originals(self):
        """Replace mode should exclude original processings from result."""
        controller = FeatureAugmentationController()

        execution_log = []

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            execution_log.append({
                "operation": op,
                "processing": deepcopy(ctx.selector.processing)
            })
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        # Dataset has original + new processings
        mock_dataset.features_processings.return_value = [
            "raw_A", "raw_A_SNV", "raw_A_Gaussian"
        ]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw_A"]])
        )

        op1 = Mock(name="SNV")
        op2 = Mock(name="Gaussian")
        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [op1, op2],
                "action": "replace"
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # Operations should chain from original
        assert len(execution_log) == 2
        assert execution_log[0]["processing"] == [["raw_A"]]
        assert execution_log[1]["processing"] == [["raw_A"]]

        # Result should EXCLUDE original, only include chained versions
        assert "raw_A" not in updated_context.selector.processing[0]
        assert "raw_A_SNV" in updated_context.selector.processing[0]
        assert "raw_A_Gaussian" in updated_context.selector.processing[0]

    def test_replace_mode_multi_source(self):
        """Replace mode should work correctly with multiple sources."""
        controller = FeatureAugmentationController()

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 2

        # Different processings per source
        def get_processings(sdx):
            if sdx == 0:
                return ["raw_A", "raw_A_SNV"]
            else:
                return ["raw_B", "raw_B_SNV"]

        mock_dataset.features_processings = get_processings

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw_A"], ["raw_B"]])
        )

        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [Mock()],
                "action": "replace"
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # Each source should have originals excluded
        assert "raw_A" not in updated_context.selector.processing[0]
        assert "raw_A_SNV" in updated_context.selector.processing[0]
        assert "raw_B" not in updated_context.selector.processing[1]
        assert "raw_B_SNV" in updated_context.selector.processing[1]


class TestEmptyOperations:
    """Test edge cases with empty operations."""

    @pytest.mark.parametrize("action", ["extend", "add", "replace"])
    def test_empty_operations_list(self, action):
        """Empty operations list should return unchanged context."""
        controller = FeatureAugmentationController()

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [],
                "action": action
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # No changes should occur
        assert artifacts == []


class TestArtifactCollection:
    """Test artifact collection across all modes."""

    @pytest.mark.parametrize("action", ["extend", "add", "replace"])
    def test_artifacts_collected_from_all_operations(self, action):
        """All modes should collect artifacts from substep executions."""
        controller = FeatureAugmentationController()

        artifact1 = ("artifact1", b"data1")
        artifact2 = ("artifact2", b"data2")
        call_count = [0]

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            # Return different artifacts for different operations based on call order
            call_count[0] += 1
            if call_count[0] == 1:
                return MockStepResult(ctx, [artifact1])
            else:
                return MockStepResult(ctx, [artifact2])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw", "raw_op1", "raw_op2"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        op1 = Mock()
        op2 = Mock()
        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [op1, op2],
                "action": action
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        updated_context, artifacts = controller.execute(
            step_info, mock_dataset, context, runtime_context
        )

        # Both artifacts should be collected
        assert len(artifacts) == 2
        assert artifact1 in artifacts
        assert artifact2 in artifacts


class TestSubstepNumberIncrement:
    """Test that substep_number is incremented correctly."""

    @pytest.mark.parametrize("action", ["extend", "add", "replace"])
    def test_substep_number_incremented(self, action):
        """substep_number should be incremented for each operation."""
        controller = FeatureAugmentationController()

        substep_numbers = []

        def mock_execute(op, ds, ctx, rtc, **kwargs):
            substep_numbers.append(rtc.substep_number)
            return MockStepResult(ctx, [])

        mock_dataset = Mock()
        mock_dataset.n_sources = 1
        mock_dataset.features_processings.return_value = ["raw", "raw_A", "raw_B", "raw_C"]

        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        step_info = MockParsedStep(
            operator=None,
            original_step={
                "feature_augmentation": [Mock(), Mock(), Mock()],
                "action": action
            }
        )

        runtime_context = RuntimeContext()
        runtime_context.step_runner = Mock()
        runtime_context.step_runner.execute = mock_execute
        runtime_context.substep_number = 0

        controller.execute(step_info, mock_dataset, context, runtime_context)

        # substep_number should have been incremented each time
        assert substep_numbers == [1, 2, 3]
        assert runtime_context.substep_number == 3


class TestNormalizeGeneratorSpec:
    """Test normalize_generator_spec static method."""

    def test_non_dict_passthrough(self):
        """Non-dict specs should pass through unchanged."""
        assert FeatureAugmentationController.normalize_generator_spec("string") == "string"
        assert FeatureAugmentationController.normalize_generator_spec(42) == 42
        assert FeatureAugmentationController.normalize_generator_spec(None) is None

    def test_explicit_pick_honored(self):
        """Explicit pick parameter should be honored."""
        spec = {"_or_": [1, 2, 3], "pick": 2}
        result = FeatureAugmentationController.normalize_generator_spec(spec)
        assert result == spec

    def test_explicit_arrange_honored(self):
        """Explicit arrange parameter should be honored."""
        spec = {"_or_": [1, 2, 3], "arrange": 2}
        result = FeatureAugmentationController.normalize_generator_spec(spec)
        assert result == spec

    def test_size_converted_to_pick(self):
        """Legacy 'size' should be converted to 'pick' when _or_ is present."""
        spec = {"_or_": [1, 2, 3], "size": 2}
        result = FeatureAugmentationController.normalize_generator_spec(spec)
        assert "pick" in result
        assert result["pick"] == 2
        assert "size" not in result

    def test_size_without_or_unchanged(self):
        """'size' without '_or_' should remain unchanged."""
        spec = {"size": 2, "other": "value"}
        result = FeatureAugmentationController.normalize_generator_spec(spec)
        assert result == spec
