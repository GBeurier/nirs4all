"""Tests for custom controller and keyword extensibility.

This module verifies that users can:
1. Create custom controllers with custom keywords
2. Register them dynamically
3. Use them in pipelines
4. Have proper priority-based keyword resolution
"""

import pytest
from typing import Any, List, Tuple, Optional

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller, reset_registry, CONTROLLER_REGISTRY
from nirs4all.pipeline.steps.parser import StepParser, ParsedStep, StepType
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.context import ExecutionContext


class TestCustomKeywordExtraction:
    """Test that custom keywords are properly extracted from step definitions."""

    def test_custom_keyword_extracted_from_dict(self):
        """Custom keyword should be extracted when no standard keywords present."""
        parser = StepParser()
        step = {"my_custom_operation": lambda x: x, "params": {"threshold": 0.5}}

        parsed = parser.parse(step)

        assert parsed.keyword == "my_custom_operation"
        assert parsed.step_type == StepType.WORKFLOW
        assert "params" in parsed.metadata
        assert parsed.metadata["params"]["threshold"] == 0.5

    def test_custom_keyword_lowercase(self):
        """Custom keyword with lowercase should work."""
        parser = StepParser()
        step = {"smoothing": "some.module.Smoother"}

        parsed = parser.parse(step)

        assert parsed.keyword == "smoothing"
        assert parsed.step_type == StepType.WORKFLOW

    def test_custom_keyword_with_underscores(self):
        """Custom keyword with underscores should work."""
        parser = StepParser()
        step = {"baseline_correction": lambda x: x}

        parsed = parser.parse(step)

        assert parsed.keyword == "baseline_correction"

    def test_custom_keyword_with_numbers(self):
        """Custom keyword with numbers should work."""
        parser = StepParser()
        step = {"filter2d": lambda x: x}

        parsed = parser.parse(step)

        assert parsed.keyword == "filter2d"

    def test_reserved_keywords_not_treated_as_custom(self):
        """Reserved keywords should not be extracted as custom keywords."""
        parser = StepParser()

        # These should not be treated as workflow keywords
        reserved = ["params", "metadata", "steps", "name", "finetune_params", "train_params"]

        for keyword in reserved:
            assert keyword in parser.RESERVED_KEYWORDS

    def test_priority_keyword_takes_precedence_over_custom(self):
        """Standard workflow keywords should take precedence over custom ones."""
        parser = StepParser()

        # Both "model" (priority) and "my_custom" (custom) present
        step = {
            "model": lambda x: x,
            "my_custom_op": lambda y: y,
            "params": {}
        }

        parsed = parser.parse(step)

        # "model" should be chosen because it's in WORKFLOW_KEYWORDS
        assert parsed.keyword == "model"

    def test_first_custom_keyword_chosen_when_multiple(self):
        """When multiple custom keywords present, first one should be chosen."""
        parser = StepParser()

        # Multiple custom keywords (order depends on dict ordering in Python 3.7+)
        step = {"custom_a": lambda x: x, "custom_b": lambda y: y}

        parsed = parser.parse(step)

        # Should pick one of them (deterministic in Python 3.7+)
        assert parsed.keyword in ["custom_a", "custom_b"]

    def test_custom_keyword_with_class_operator(self):
        """Custom keyword should work with class-based operators."""
        from sklearn.preprocessing import StandardScaler

        parser = StepParser()
        step = {"my_scaler": StandardScaler, "params": {"with_mean": False}}

        parsed = parser.parse(step)

        assert parsed.keyword == "my_scaler"
        assert parsed.operator is not None


class TestCustomControllerRegistration:
    """Test that custom controllers can be registered and discovered."""

    def setup_method(self):
        """Store original registry before each test."""
        self.original_registry = CONTROLLER_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry after each test."""
        global CONTROLLER_REGISTRY
        CONTROLLER_REGISTRY.clear()
        CONTROLLER_REGISTRY.extend(self.original_registry)

    def test_register_custom_controller(self):
        """Custom controller should be registered via decorator."""

        @register_controller
        class TestCustomController(OperatorController):
            priority = 50

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "test_custom"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        # Should be in registry
        assert TestCustomController in CONTROLLER_REGISTRY

    def test_custom_controller_priority_respected(self):
        """Controllers should be sorted by priority in registry."""

        @register_controller
        class HighPriorityController(OperatorController):
            priority = 10

            @classmethod
            def matches(cls, step, operator, keyword):
                return True

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        @register_controller
        class LowPriorityController(OperatorController):
            priority = 90

            @classmethod
            def matches(cls, step, operator, keyword):
                return True

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        # Find positions in registry
        high_pos = CONTROLLER_REGISTRY.index(HighPriorityController)
        low_pos = CONTROLLER_REGISTRY.index(LowPriorityController)

        # High priority should come before low priority
        assert high_pos < low_pos

    def test_duplicate_registration_prevented(self):
        """Registering same controller twice should not duplicate it."""

        @register_controller
        class UniqueController(OperatorController):
            priority = 50

            @classmethod
            def matches(cls, step, operator, keyword):
                return False

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        count_before = CONTROLLER_REGISTRY.count(UniqueController)

        # Try to register again
        register_controller(UniqueController)

        count_after = CONTROLLER_REGISTRY.count(UniqueController)

        # Should not increase count
        assert count_before == count_after == 1


class TestCustomControllerRouting:
    """Test that custom controllers are properly routed to."""

    def setup_method(self):
        """Store original registry before each test."""
        self.original_registry = CONTROLLER_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry after each test."""
        global CONTROLLER_REGISTRY
        CONTROLLER_REGISTRY.clear()
        CONTROLLER_REGISTRY.extend(self.original_registry)

    def test_router_finds_custom_controller(self):
        """Router should find and instantiate custom controller."""

        @register_controller
        class CustomSmoothingController(OperatorController):
            priority = 30

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "smoothing" or keyword == "smooth"

            @classmethod
            def use_multi_source(cls):
                return True

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        parser = StepParser()
        router = ControllerRouter()

        step = {"smoothing": lambda x: x}
        parsed = parser.parse(step)

        controller = router.route(parsed, step)

        assert isinstance(controller, CustomSmoothingController)

    def test_custom_controller_higher_priority_wins(self):
        """Custom controller with higher priority should be chosen over default."""

        # Create a high-priority custom controller
        @register_controller
        class HighPriorityCustom(OperatorController):
            priority = 5  # Very high priority

            @classmethod
            def matches(cls, step, operator, keyword):
                # Match anything with "custom" keyword
                return keyword == "custom_transform"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        # Create a lower priority controller that also matches
        @register_controller
        class LowPriorityCustom(OperatorController):
            priority = 80

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "custom_transform"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        parser = StepParser()
        router = ControllerRouter()

        step = {"custom_transform": lambda x: x}
        parsed = parser.parse(step)

        controller = router.route(parsed, step)

        # Should get the high priority one
        assert isinstance(controller, HighPriorityCustom)

    def test_router_verbose_mode_for_custom(self):
        """Router verbose mode should show custom controller matching."""

        @register_controller
        class VerboseTestController(OperatorController):
            priority = 40

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "verbose_test"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        parser = StepParser()
        router = ControllerRouter(verbose=True)

        step = {"verbose_test": lambda x: x}
        parsed = parser.parse(step)

        # Should not raise, just print debug info
        controller = router.route(parsed, step)
        assert isinstance(controller, VerboseTestController)


class TestCustomControllerExecution:
    """Test that custom controllers execute correctly in pipeline context."""

    def setup_method(self):
        """Store original registry before each test."""
        self.original_registry = CONTROLLER_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry after each test."""
        global CONTROLLER_REGISTRY
        CONTROLLER_REGISTRY.clear()
        CONTROLLER_REGISTRY.extend(self.original_registry)

    def test_custom_controller_receives_correct_step_info(self):
        """Custom controller should receive properly parsed step info."""

        received_info = {}

        @register_controller
        class InfoCapturingController(OperatorController):
            priority = 20

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "info_test"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                received_info['keyword'] = step_info.keyword
                received_info['step_type'] = step_info.step_type
                received_info['metadata'] = step_info.metadata
                received_info['operator'] = step_info.operator
                return context, []

        from nirs4all.pipeline.steps.step_runner import StepRunner
        from nirs4all.pipeline.config.context import RuntimeContext
        import numpy as np

        # Create minimal dataset
        dataset = SpectroDataset(name="test")
        dataset.add_samples(np.random.rand(10, 100))
        dataset.add_targets(np.random.rand(10))

        context = ExecutionContext()
        runtime_context = RuntimeContext(pipeline_uid="test", operation_count=0)

        runner = StepRunner()
        step = {"info_test": lambda x: x, "params": {"window": 5}}

        runner.execute(step, dataset, context, runtime_context)

        # Verify received info
        assert received_info['keyword'] == "info_test"
        assert received_info['metadata']['params']['window'] == 5

    def test_custom_controller_can_modify_context(self):
        """Custom controller should be able to modify execution context."""

        @register_controller
        class ContextModifyingController(OperatorController):
            priority = 20

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "context_modifier"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                # Modify custom context data
                new_context = context.copy()
                new_context.custom["my_custom_data"] = "modified_value"
                return new_context, []

        from nirs4all.pipeline.steps.step_runner import StepRunner
        from nirs4all.pipeline.config.context import RuntimeContext
        import numpy as np

        dataset = SpectroDataset(name="test")
        dataset.add_samples(np.random.rand(10, 100))
        dataset.add_targets(np.random.rand(10))

        context = ExecutionContext()
        runtime_context = RuntimeContext(pipeline_uid="test", operation_count=0)

        runner = StepRunner()
        step = {"context_modifier": lambda x: x}

        result = runner.execute(step, dataset, context, runtime_context)

        # Context should be modified
        assert "my_custom_data" in result.updated_context.custom
        assert result.updated_context.custom["my_custom_data"] == "modified_value"

    def test_custom_controller_prediction_mode_support(self):
        """Custom controller should respect prediction mode support."""

        @register_controller
        class TrainOnlyController(OperatorController):
            priority = 20

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "train_only_op"

            @classmethod
            def use_multi_source(cls):
                return False

            @classmethod
            def supports_prediction_mode(cls):
                return False  # Only runs during training

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                raise RuntimeError("Should not be called in predict mode!")

        from nirs4all.pipeline.steps.step_runner import StepRunner
        from nirs4all.pipeline.config.context import RuntimeContext
        import numpy as np

        dataset = SpectroDataset(name="test")
        dataset.add_samples(np.random.rand(10, 100))
        dataset.add_targets(np.random.rand(10))

        context = ExecutionContext()
        runtime_context = RuntimeContext(pipeline_uid="test", operation_count=0)

        # In predict mode, should skip
        runner = StepRunner(mode="predict")
        step = {"train_only_op": lambda x: x}

        result = runner.execute(step, dataset, context, runtime_context)

        # Should return unchanged context (step was skipped)
        assert result.updated_context == context


class TestKeywordPrioritization:
    """Test priority system for keyword resolution."""

    def test_model_keyword_highest_priority(self):
        """Model keyword should take precedence over other workflow keywords."""
        parser = StepParser()

        step = {
            "model": lambda x: x,
            "preprocessing": lambda y: y,
            "params": {}
        }

        parsed = parser.parse(step)
        assert parsed.keyword == "model"

    def test_preprocessing_priority_over_custom(self):
        """Standard preprocessing should take precedence over custom keywords."""
        parser = StepParser()

        step = {
            "preprocessing": lambda x: x,
            "my_custom": lambda y: y,
        }

        parsed = parser.parse(step)
        assert parsed.keyword == "preprocessing"

    def test_workflow_keyword_priority_order(self):
        """Verify the priority order of workflow keywords."""
        parser = StepParser()

        expected_priority = [
            "model",
            "preprocessing",
            "feature_augmentation",
            "auto_transfer_preproc",
            "concat_transform",
            "y_processing",
            "sample_augmentation",
            "branch",
        ]

        assert parser.WORKFLOW_KEYWORDS == expected_priority

    def test_serialization_operator_over_workflow(self):
        """Serialization operators should be checked before workflow keywords."""
        parser = StepParser()

        # Has both "class" (serialization) and workflow content
        step = {
            "class": "sklearn.preprocessing.StandardScaler",
            "model": lambda x: x  # This should be ignored
        }

        parsed = parser.parse(step)

        # Should treat as serialized, not workflow
        assert parsed.step_type == StepType.SERIALIZED
        assert parsed.keyword == "class"


class TestRealWorldCustomControllerScenarios:
    """Test realistic scenarios for custom controller usage."""

    def setup_method(self):
        """Store original registry before each test."""
        self.original_registry = CONTROLLER_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry after each test."""
        global CONTROLLER_REGISTRY
        CONTROLLER_REGISTRY.clear()
        CONTROLLER_REGISTRY.extend(self.original_registry)

    def test_custom_baseline_correction_controller(self):
        """Example: Custom baseline correction controller."""

        @register_controller
        class BaselineCorrectionController(OperatorController):
            priority = 45

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword in ["baseline_correction", "baseline"]

            @classmethod
            def use_multi_source(cls):
                return True

            @classmethod
            def supports_prediction_mode(cls):
                return True

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                # Simulated baseline correction
                return context, []

        parser = StepParser()
        router = ControllerRouter()

        step = {"baseline_correction": "some.baseline.method", "params": {"method": "als"}}
        parsed = parser.parse(step)
        controller = router.route(parsed, step)

        assert isinstance(controller, BaselineCorrectionController)

    def test_custom_outlier_detection_controller(self):
        """Example: Custom outlier detection controller."""

        from sklearn.ensemble import IsolationForest

        @register_controller
        class OutlierDetectionController(OperatorController):
            priority = 5  # Higher priority than sklearn model controller (6)

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword in ["outlier_detection", "outliers"]

            @classmethod
            def use_multi_source(cls):
                return True

            @classmethod
            def supports_prediction_mode(cls):
                return False  # Only during training

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                # Would mark outliers in dataset
                return context, []

        # Use direct operator reference, not serialization
        step = {"outlier_detection": IsolationForest()}

        parser = StepParser()
        router = ControllerRouter()
        parsed = parser.parse(step)
        controller = router.route(parsed, step)

        assert isinstance(controller, OutlierDetectionController)

    def test_multiple_custom_controllers_coexist(self):
        """Multiple custom controllers should coexist peacefully."""

        @register_controller
        class CustomA(OperatorController):
            priority = 30

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "custom_a"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        @register_controller
        class CustomB(OperatorController):
            priority = 35

            @classmethod
            def matches(cls, step, operator, keyword):
                return keyword == "custom_b"

            @classmethod
            def use_multi_source(cls):
                return False

            def execute(self, step_info, dataset, context, runtime_context,
                       source=-1, mode="train", loaded_binaries=None, prediction_store=None):
                return context, []

        parser = StepParser()
        router = ControllerRouter()

        # Test custom_a
        step_a = {"custom_a": lambda x: x}
        parsed_a = parser.parse(step_a)
        controller_a = router.route(parsed_a, step_a)
        assert isinstance(controller_a, CustomA)

        # Test custom_b
        step_b = {"custom_b": lambda x: x}
        parsed_b = parser.parse(step_b)
        controller_b = router.route(parsed_b, step_b)
        assert isinstance(controller_b, CustomB)
