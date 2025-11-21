"""
Unit tests for group-based cross-validation with metadata.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from unittest.mock import Mock
from nirs4all.data.dataset import SpectroDataset
from nirs4all.controllers.splitters.split import CrossValidatorController
from nirs4all.pipeline.steps.parser import ParsedStep, StepType
from nirs4all.pipeline.config.context import ExecutionContext, DataSelector, PipelineState, StepMetadata, RuntimeContext


def make_step_info(operator, step=None):
    """Helper to create ParsedStep for testing."""
    if step is None:
        step = {}
    return ParsedStep(
        operator=operator,
        keyword="",
        step_type=StepType.DIRECT,
        original_step=step,
        metadata={}
    )


def make_mock_runtime_context():
    """Create a mock runtime context without saver."""
    mock_runtime = Mock()
    mock_runtime.saver = None  # No saver, so controller will use fallback
    return mock_runtime


class TestGroupSplitSyntax:
    """Test new split syntax with group parameter."""

    def test_matches_split_keyword(self):
        """Test controller matches on 'split' keyword."""
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group": "batch"}
        assert controller.matches(step, None, "split")

    def test_matches_split_in_dict(self):
        """Test controller matches dict with 'split' key."""
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group": "batch"}
        assert controller.matches(step, None, "")

    def test_backward_compatible_matching(self):
        """Test backward compatibility with direct operator."""
        controller = CrossValidatorController()
        splitter = GroupKFold()
        assert controller.matches(splitter, splitter, "")

    def test_no_match_without_split(self):
        """Test no match for non-splitter objects."""
        controller = CrossValidatorController()
        assert not controller.matches({"other": "value"}, None, "")

    def test_no_match_none_operator(self):
        """Test no match when operator is None and no keyword."""
        controller = CrossValidatorController()
        assert not controller.matches({}, None, "")


class TestGroupSplitExecution:
    """Test execution with metadata groups."""

    @pytest.fixture
    def dataset_with_metadata(self):
        """Create dataset with metadata for testing."""
        dataset = SpectroDataset(name="test")
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        metadata = pd.DataFrame({
            'batch': [1]*25 + [2]*25 + [3]*25 + [4]*25,
            'location': ['A']*50 + ['B']*50,
            'sample_id': range(100)
        })

        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)
        dataset.add_metadata(metadata)
        return dataset

    def test_group_split_with_batch(self, dataset_with_metadata):
        """Test GroupKFold with batch column."""
        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        context, step_output = controller.execute(
            step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
            context=context, runtime_context=make_mock_runtime_context(), mode="train"
        )

        # Verify folds created
        assert dataset_with_metadata._folds is not None
        assert len(dataset_with_metadata._folds) == 4

        # Verify no batch leakage between train/val
        for train_idx, val_idx in dataset_with_metadata._folds:
            train_batches = dataset_with_metadata.metadata_column("batch")[train_idx]
            val_batches = dataset_with_metadata.metadata_column("batch")[val_idx]
            # No overlap in batches
            assert len(set(train_batches) & set(val_batches)) == 0

        # Verify binary filename includes group
        assert len(step_output.outputs) == 1
        assert "group-batch" in step_output.outputs[0][1]

    def test_default_group_column(self, dataset_with_metadata):
        """Test default to first metadata column."""
        step = {"split": GroupKFold(n_splits=4)}  # No group specified
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        # Should use first column (batch) by default
        context, step_output = controller.execute(
            step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
            context=context, runtime_context=make_mock_runtime_context(), mode="train"
        )

        assert dataset_with_metadata._folds is not None
        assert "group-batch" in step_output.outputs[0][1]

    def test_group_shuffle_split(self, dataset_with_metadata):
        """Test GroupShuffleSplit with location column."""
        step = {"split": GroupShuffleSplit(n_splits=3, test_size=0.5, random_state=42), "group": "location"}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        context, step_output = controller.execute(
            step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
            context=context, runtime_context=make_mock_runtime_context(), mode="train"
        )

        # Verify folds created
        assert dataset_with_metadata._folds is not None
        assert len(dataset_with_metadata._folds) == 3

        # Verify no location leakage
        for train_idx, val_idx in dataset_with_metadata._folds:
            train_locations = set(dataset_with_metadata.metadata_column("location")[train_idx])
            val_locations = set(dataset_with_metadata.metadata_column("location")[val_idx])
            assert len(train_locations & val_locations) == 0

    def test_invalid_group_column(self, dataset_with_metadata):
        """Test error on invalid group column."""
        step = {"split": GroupKFold(n_splits=4), "group": "nonexistent"}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        with pytest.raises(ValueError, match="not found in metadata"):
            controller.execute(
                step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
                context=context, runtime_context=make_mock_runtime_context(), mode="train"
            )

    def test_no_metadata_error(self):
        """Test error when no metadata available."""
        dataset = SpectroDataset(name="test")
        dataset.add_samples(np.random.rand(100, 10), {"partition": "train"})
        dataset.add_targets(np.random.rand(100))

        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        with pytest.raises(ValueError, match="no metadata"):
            controller.execute(step_info=make_step_info(step["split"], step), dataset=dataset,
                             context=context, runtime_context=make_mock_runtime_context(), mode="train")

    def test_non_string_group_type(self, dataset_with_metadata):
        """Test error when group is not a string."""
        step = {"split": GroupKFold(n_splits=4), "group": 123}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        with pytest.raises(TypeError, match="must be a string"):
            controller.execute(
                step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
                context=context, runtime_context=make_mock_runtime_context(), mode="train"
            )

    def test_non_grouped_splitter(self, dataset_with_metadata):
        """Test non-grouped splitter still works."""
        step = KFold(n_splits=5)  # No groups needed
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        context, step_output = controller.execute(
            step_info=make_step_info(step), dataset=dataset_with_metadata,
            context=context, runtime_context=make_mock_runtime_context(), mode="train"
        )

        assert dataset_with_metadata._folds is not None
        assert len(dataset_with_metadata._folds) == 5

    def test_prediction_mode(self, dataset_with_metadata):
        """Test prediction mode doesn't fail."""
        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()
        context = ExecutionContext(
            selector=DataSelector(processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        context, step_output = controller.execute(
            step_info=make_step_info(step["split"], step), dataset=dataset_with_metadata,
            context=context, runtime_context=make_mock_runtime_context(), mode="predict"
        )

        # Should create dummy folds for prediction mode
        assert dataset_with_metadata._folds is not None
        assert len(step_output.outputs) == 0  # No binaries in predict mode


class TestSerialization:
    """Test serialization of new syntax."""

    def test_serialize_split_with_group(self):
        """Test serialization preserves group parameter."""
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
        from nirs4all.pipeline.config.component_serialization import serialize_component

        pipeline = [
            {"split": GroupKFold(n_splits=5), "group": "batch_id"}
        ]

        config = PipelineConfigs(pipeline)
        # Use serialize_component directly (serializable_steps was removed in refactoring)
        serialized = config.steps[0]

        # Verify structure preserved
        assert "split" in serialized[0]
        assert "group" in serialized[0]
        assert serialized[0]["group"] == "batch_id"

    def test_roundtrip_serialization(self):
        """Test save/load roundtrip."""
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
        import json

        original = [
            {"split": {"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}, "group": "sample"}
        ]

        config = PipelineConfigs(original)
        # Steps are already serialized in PipelineConfigs
        serialized = json.dumps(config.steps[0])
        deserialized = json.loads(serialized)

        assert deserialized[0]["group"] == "sample"
        assert "GroupKFold" in deserialized[0]["split"]["class"]

    def test_backward_compatible_serialization(self):
        """Test that old format still works."""
        from nirs4all.pipeline.config.component_serialization import serialize_component

        old_format = GroupKFold(n_splits=5)
        # serialize_component doesn't take include_runtime parameter in refactored version
        serialized = serialize_component(old_format)

        # Should serialize without error
        assert "class" in serialized or isinstance(serialized, str)
        if isinstance(serialized, dict):
            assert "GroupKFold" in serialized["class"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
