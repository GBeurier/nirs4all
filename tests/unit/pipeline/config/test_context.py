"""
Tests for pipeline context classes.

This module provides comprehensive tests for:
- DataSelector: Immutable data selection
- PipelineState: Mutable pipeline state
- StepMetadata: Controller coordination
- ExecutionContext: Composite context with extensibility
"""

import pytest
from copy import deepcopy

from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext
)


class TestDataSelector:
    """Tests for DataSelector class."""

    def test_default_initialization(self):
        """Test DataSelector default values."""
        selector = DataSelector()
        assert selector.partition == "all"
        assert selector.processing == [["raw"]]
        assert selector.layout == "2d"
        assert selector.concat_source is True
        assert selector.fold_id is None
        assert selector.include_augmented is False

    def test_custom_initialization(self):
        """Test DataSelector with custom values."""
        selector = DataSelector(
            partition="train",
            processing=[["snv"], ["savgol"]],
            layout="3d",
            concat_source=False,
            fold_id=2,
            include_augmented=True
        )
        assert selector.partition == "train"
        assert selector.processing == [["snv"], ["savgol"]]
        assert selector.layout == "3d"
        assert selector.concat_source is False
        assert selector.fold_id == 2
        assert selector.include_augmented is True

    def test_with_partition_creates_new_instance(self):
        """Test with_partition returns new immutable instance."""
        selector = DataSelector(partition="train")
        new_selector = selector.with_partition("test")

        assert selector.partition == "train"  # Original unchanged
        assert new_selector.partition == "test"
        assert id(selector) != id(new_selector)

    def test_with_processing_creates_new_instance(self):
        """Test with_processing returns new immutable instance."""
        selector = DataSelector(processing=[["raw"]])
        new_selector = selector.with_processing([["snv"], ["savgol"]])

        assert selector.processing == [["raw"]]
        assert new_selector.processing == [["snv"], ["savgol"]]
        assert id(selector) != id(new_selector)

    def test_with_layout_creates_new_instance(self):
        """Test with_layout returns new immutable instance."""
        selector = DataSelector(layout="2d")
        new_selector = selector.with_layout("3d")

        assert selector.layout == "2d"
        assert new_selector.layout == "3d"

    def test_with_fold_creates_new_instance(self):
        """Test with_fold returns new immutable instance."""
        selector = DataSelector(fold_id=None)
        new_selector = selector.with_fold(5)

        assert selector.fold_id is None
        assert new_selector.fold_id == 5

    def test_with_augmented_creates_new_instance(self):
        """Test with_augmented returns new immutable instance."""
        selector = DataSelector(include_augmented=False)
        new_selector = selector.with_augmented(True)

        assert selector.include_augmented is False
        assert new_selector.include_augmented is True

    def test_immutability(self):
        """Test that DataSelector is truly immutable."""
        selector = DataSelector(partition="train")

        with pytest.raises(AttributeError):
            selector.partition = "test"  # type: ignore

    def test_mapping_protocol(self):
        """Test that DataSelector behaves like a dict."""
        selector = DataSelector(partition="train", fold_id=1)

        # Test __getitem__
        assert selector["partition"] == "train"
        assert selector["fold_id"] == 1

        # Test dict conversion
        d = dict(selector)
        assert d["partition"] == "train"
        assert d["fold_id"] == 1
        assert "processing" in d  # Default value

        # Test missing key
        with pytest.raises(KeyError):
            _ = selector["non_existent"]

        # Test None value exclusion
        selector_none = DataSelector(fold_id=None)
        with pytest.raises(KeyError):
            _ = selector_none["fold_id"]

        assert "fold_id" not in dict(selector_none)


class TestPipelineState:
    """Tests for PipelineState class."""

    def test_default_initialization(self):
        """Test PipelineState default values."""
        state = PipelineState()
        assert state.y_processing == "numeric"
        assert state.step_number == 0
        assert state.mode == "train"

    def test_custom_initialization(self):
        """Test PipelineState with custom values."""
        state = PipelineState(
            y_processing="encoded_LabelEncoder_001",
            step_number=5,
            mode="predict"
        )
        assert state.y_processing == "encoded_LabelEncoder_001"
        assert state.step_number == 5
        assert state.mode == "predict"

    def test_mutability(self):
        """Test that PipelineState is mutable."""
        state = PipelineState()

        state.y_processing = "encoded"
        state.step_number = 3
        state.mode = "explain"

        assert state.y_processing == "encoded"
        assert state.step_number == 3
        assert state.mode == "explain"

    def test_copy_creates_independent_instance(self):
        """Test copy creates independent mutable copy."""
        state = PipelineState(y_processing="original", step_number=1)
        state_copy = state.copy()

        state_copy.y_processing = "modified"
        state_copy.step_number = 2

        assert state.y_processing == "original"  # Original unchanged
        assert state.step_number == 1
        assert state_copy.y_processing == "modified"
        assert state_copy.step_number == 2


class TestStepMetadata:
    """Tests for StepMetadata class."""

    def test_default_initialization(self):
        """Test StepMetadata default values."""
        metadata = StepMetadata()
        assert metadata.keyword == ""
        assert metadata.step_id == ""
        assert metadata.augment_sample is False
        assert metadata.add_feature is False
        assert metadata.replace_processing is False
        assert metadata.target_samples == []
        assert metadata.target_features == []

    def test_custom_initialization(self):
        """Test StepMetadata with custom values."""
        metadata = StepMetadata(
            keyword="transform",
            step_id="001",
            augment_sample=True,
            add_feature=False,
            replace_processing=True,
            target_samples=[1, 2, 3],
            target_features=[10, 20]
        )
        assert metadata.keyword == "transform"
        assert metadata.step_id == "001"
        assert metadata.augment_sample is True
        assert metadata.add_feature is False
        assert metadata.replace_processing is True
        assert metadata.target_samples == [1, 2, 3]
        assert metadata.target_features == [10, 20]

    def test_mutability(self):
        """Test that StepMetadata is mutable."""
        metadata = StepMetadata()

        metadata.keyword = "model"
        metadata.step_id = "002"
        metadata.augment_sample = True
        metadata.target_samples = [42]

        assert metadata.keyword == "model"
        assert metadata.step_id == "002"
        assert metadata.augment_sample is True
        assert metadata.target_samples == [42]

    def test_copy_creates_independent_instance(self):
        """Test copy creates independent mutable copy."""
        metadata = StepMetadata(
            keyword="original",
            target_samples=[1, 2]
        )
        metadata_copy = metadata.copy()

        metadata_copy.keyword = "modified"
        metadata_copy.target_samples.append(3)

        assert metadata.keyword == "original"  # Original unchanged
        assert metadata.target_samples == [1, 2]
        assert metadata_copy.keyword == "modified"
        assert metadata_copy.target_samples == [1, 2, 3]


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_default_initialization(self):
        """Test ExecutionContext with default values."""
        context = ExecutionContext()

        assert context.selector.partition == "all"
        assert context.state.y_processing == "numeric"
        assert context.metadata.keyword == ""
        assert context.custom == {}

    def test_custom_initialization(self):
        """Test ExecutionContext with custom components."""
        selector = DataSelector(partition="train")
        state = PipelineState(y_processing="encoded")
        metadata = StepMetadata(keyword="model")
        custom = {"my_data": 42}

        context = ExecutionContext(
            selector=selector,
            state=state,
            metadata=metadata,
            custom=custom
        )

        assert context.selector.partition == "train"
        assert context.state.y_processing == "encoded"
        assert context.metadata.keyword == "model"
        assert context.custom == {"my_data": 42}

    def test_copy_creates_deep_copy(self):
        """Test copy creates deep independent copy."""
        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(y_processing="numeric"),
            metadata=StepMetadata(keyword="transform"),
            custom={"key": "value"}
        )

        context_copy = context.copy()

        # Modify copy
        context_copy.selector = context_copy.selector.with_partition("test")
        context_copy.state.y_processing = "encoded"
        context_copy.metadata.keyword = "model"
        context_copy.custom["key"] = "modified"

        # Verify original unchanged
        assert context.selector.partition == "train"
        assert context.state.y_processing == "numeric"
        assert context.metadata.keyword == "transform"
        assert context.custom == {"key": "value"}

    def test_with_partition_creates_new_context(self):
        """Test with_partition creates new context with updated partition."""
        context = ExecutionContext(
            selector=DataSelector(partition="train")
        )

        new_context = context.with_partition("test")

        assert context.selector.partition == "train"
        assert new_context.selector.partition == "test"

    def test_with_processing_creates_new_context(self):
        """Test with_processing creates new context with updated processing."""
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        new_context = context.with_processing([["snv"], ["savgol"]])

        assert context.selector.processing == [["raw"]]
        assert new_context.selector.processing == [["snv"], ["savgol"]]

    def test_get_selector(self):
        """Test get_selector returns correct selector."""
        selector = DataSelector(partition="train")
        context = ExecutionContext(selector=selector)

        retrieved = context.get_selector()
        assert retrieved.partition == "train"

    def test_custom_data_extensibility(self):
        """Test custom dict for controller-specific data."""
        context = ExecutionContext()

        # Controller A stores data
        context.custom["controller_a"] = {"threshold": 0.5, "iterations": 10}

        # Controller B stores data
        context.custom["controller_b"] = {"learning_rate": 0.01}

        # Both are isolated
        assert context.custom["controller_a"]["threshold"] == 0.5
        assert context.custom["controller_b"]["learning_rate"] == 0.01

    def test_custom_data_survives_copy(self):
        """Test custom data is deep copied."""
        context = ExecutionContext()
        context.custom["data"] = {"value": [1, 2, 3]}

        context_copy = context.copy()
        context_copy.custom["data"]["value"].append(4)

        assert context.custom["data"]["value"] == [1, 2, 3]
        assert context_copy.custom["data"]["value"] == [1, 2, 3, 4]

    def test_processing_chains_shared_reference(self):
        """Test that processing chains are the same list reference initially."""
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        # Get selector and verify processing
        selector = context.get_selector()
        assert selector.processing == [["raw"]]

    def test_deepcopy_compatibility(self):
        """Test compatibility with copy.deepcopy."""
        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(y_processing="numeric"),
            custom={"nested": {"data": [1, 2, 3]}}
        )

        context_copy = deepcopy(context)

        # Modify copy
        context_copy.custom["nested"]["data"].append(4)

        # Verify original unchanged
        assert context.custom["nested"]["data"] == [1, 2, 3]
        assert context_copy.custom["nested"]["data"] == [1, 2, 3, 4]


class TestContextIntegration:
    """Integration tests for context usage patterns."""

    def test_controller_isolation_pattern(self):
        """Test typical controller isolation pattern."""
        # Original context
        context = ExecutionContext(
            selector=DataSelector(partition="all", processing=[["raw"]]),
            state=PipelineState(y_processing="numeric")
        )

        # Controller creates isolated train context
        train_context = context.with_partition("train")
        train_context.state.step_number = 1

        # Original unchanged
        assert context.selector.partition == "all"
        assert context.state.step_number == 0

        # Train context updated
        assert train_context.selector.partition == "train"
        assert train_context.state.step_number == 1

    def test_processing_evolution_pattern(self):
        """Test typical processing chain evolution."""
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]])
        )

        # Transformer updates processing
        new_processing = [["standard_scaler_001"]]
        context = context.with_processing(new_processing)

        assert context.selector.processing == [["standard_scaler_001"]]

    def test_controller_communication_pattern(self):
        """Test controller coordination via metadata."""
        context = ExecutionContext()

        # SampleAugmentationController sets flags
        context.metadata.augment_sample = True
        context.metadata.target_samples = [42]

        # TransformerMixinController reads flags
        if context.metadata.augment_sample:
            targets = context.metadata.target_samples
            assert targets == [42]

    def test_multi_partition_pattern(self):
        """Test creating multiple partition contexts."""
        base_context = ExecutionContext(
            selector=DataSelector(processing=[["snv"]])
        )

        # Create train, test, val contexts
        train_ctx = base_context.with_partition("train")
        test_ctx = base_context.with_partition("test")
        val_ctx = base_context.with_partition("val")

        assert train_ctx.selector.partition == "train"
        assert test_ctx.selector.partition == "test"
        assert val_ctx.selector.partition == "val"

        # All share same processing
        assert train_ctx.selector.processing == [["snv"]]
        assert test_ctx.selector.processing == [["snv"]]
        assert val_ctx.selector.processing == [["snv"]]
