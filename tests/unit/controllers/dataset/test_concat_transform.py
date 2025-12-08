"""
Unit tests for ConcatAugmentationController.

Tests cover:
- Replace mode (top-level, add_feature=False)
- Add mode (inside feature_augmentation, add_feature=True)
- Single transformers
- Chained transformers
- Multi-processing handling
- Serialization/prediction mode
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD

from nirs4all.controllers.data.concat_transform import ConcatAugmentationController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.steps.parser import ParsedStep, StepType
from nirs4all.pipeline.config.context import (
    ExecutionContext, DataSelector, PipelineState, StepMetadata, RuntimeContext
)


def make_step_info(step_dict):
    """Helper to create ParsedStep for testing."""
    return ParsedStep(
        operator=None,
        keyword="concat_transform",
        step_type=StepType.WORKFLOW,
        original_step=step_dict,
        metadata={}
    )


@pytest.fixture
def mock_runtime_context():
    """Mock RuntimeContext."""
    runtime_ctx = RuntimeContext()
    runtime_ctx.operation_count = 0
    runtime_ctx.step_number = 1
    runtime_ctx.substep_number = 0

    # Configure saver mock
    runtime_ctx.saver = Mock()
    runtime_ctx.saver.persist_artifact = Mock(return_value={"name": "mock", "step": 1})

    return runtime_ctx


@pytest.fixture
def simple_dataset():
    """Create a simple dataset with 10 samples and 50 features."""
    dataset = SpectroDataset("test")

    # Add 10 samples with 50 features
    np.random.seed(42)
    x_data = np.random.rand(10, 50)

    dataset.add_samples(x_data, {"partition": "train"})

    return dataset


@pytest.fixture
def multi_processing_dataset():
    """Create a dataset with multiple processings (raw, snv, savgol)."""
    dataset = SpectroDataset("test_multi_proc")

    # Add 10 samples with 50 features
    np.random.seed(42)
    x_raw = np.random.rand(10, 50)

    dataset.add_samples(x_raw, {"partition": "train"})

    # Add SNV and SavGol processings
    x_snv = x_raw / np.std(x_raw, axis=1, keepdims=True)
    x_savgol = np.gradient(x_raw, axis=1)

    dataset.add_features([x_snv], ["snv"], source=0)
    dataset.add_features([x_savgol], ["savgol"], source=0)

    return dataset


class TestControllerMatching:
    """Test the matches() method."""

    def test_matches_concat_transform_keyword(self):
        """Test that controller matches concat_transform keyword."""
        controller = ConcatAugmentationController()
        step = {"concat_transform": [PCA(10)]}

        assert controller.matches(step, None, "concat_transform") is True

    def test_does_not_match_other_keywords(self):
        """Test that controller doesn't match other keywords."""
        controller = ConcatAugmentationController()

        assert controller.matches({}, None, "model") is False
        assert controller.matches({}, None, "feature_augmentation") is False
        assert controller.matches({}, None, "preprocessing") is False


class TestReplaceMode:
    """Test replace mode (top-level, add_feature=False)."""

    def test_single_transformer_replaces_processing(self, simple_dataset, mock_runtime_context):
        """Test that a single transformer replaces the processing."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(n_components=10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)  # Replace mode
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Check that features were replaced
        assert simple_dataset.num_features == 10  # PCA output
        # Check processing name was updated
        processings = simple_dataset.features_processings(0)
        assert any("concat" in p or "PCA" in p for p in processings)

    def test_multiple_transformers_concatenate(self, simple_dataset, mock_runtime_context):
        """Test that multiple transformers concatenate their outputs."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(n_components=10), TruncatedSVD(n_components=5)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)  # Replace mode
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Check that features were concatenated (10 + 5 = 15)
        assert simple_dataset.num_features == 15

    def test_replaces_all_processings(self, multi_processing_dataset, mock_runtime_context):
        """Test that replace mode applies to all processings."""
        controller = ConcatAugmentationController()

        # Before: 3 processings (raw, snv, savgol) each with 50 features
        assert len(multi_processing_dataset.features_processings(0)) == 3

        step = {"concat_transform": [PCA(n_components=5), TruncatedSVD(n_components=3)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw", "snv", "savgol"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, multi_processing_dataset, context, mock_runtime_context, mode="train"
        )

        # After: still 3 processings but each with 8 features (5 + 3)
        assert len(multi_processing_dataset.features_processings(0)) == 3
        assert multi_processing_dataset.num_features == 8  # 5 + 3


class TestAddMode:
    """Test add mode (inside feature_augmentation, add_feature=True)."""

    def test_adds_new_processing(self, simple_dataset, mock_runtime_context):
        """Test that add mode adds a new processing without replacing."""
        controller = ConcatAugmentationController()

        # Before: 1 processing (raw) with 50 features
        assert len(simple_dataset.features_processings(0)) == 1

        step = {"concat_transform": [PCA(n_components=10), TruncatedSVD(n_components=5)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=True)  # Add mode
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # After: 2 processings (raw + concat)
        processings = simple_dataset.features_processings(0)
        assert len(processings) == 2
        assert "raw" in processings
        assert any("concat" in p for p in processings)


class TestChainedTransformers:
    """Test chained transformer execution."""

    def test_chain_executes_sequentially(self, simple_dataset, mock_runtime_context):
        """Test that chains execute transformers sequentially."""
        controller = ConcatAugmentationController()

        # Chain: StandardScaler -> PCA
        step = {"concat_transform": [[StandardScaler(), PCA(n_components=10)]]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Output should be 10 features from PCA
        assert simple_dataset.num_features == 10

        # Should have artifacts for both transformers in chain
        assert len(artifacts) == 2

    def test_mixed_single_and_chain(self, simple_dataset, mock_runtime_context):
        """Test mixing single transformers with chains."""
        controller = ConcatAugmentationController()

        step = {
            "concat_transform": [
                PCA(n_components=10),           # Single: 10 features
                [StandardScaler(), PCA(n_components=5)]  # Chain: 5 features
            ]
        }
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Output should be 15 features (10 + 5)
        assert simple_dataset.num_features == 15


class TestSerialization:
    """Test serialization and prediction mode."""

    def test_artifacts_created_in_train_mode(self, simple_dataset, mock_runtime_context):
        """Test that artifacts are created during training."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(n_components=10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Should have created artifact
        assert mock_runtime_context.saver.persist_artifact.called

    def test_predict_mode_loads_binaries(self, simple_dataset, mock_runtime_context):
        """Test that predict mode loads pre-fitted transformers."""
        controller = ConcatAugmentationController()

        # First train to get fitted PCA
        pca = PCA(n_components=10)
        pca.fit(simple_dataset.x({"partition": "train"}))

        # Simulate loaded binaries
        loaded_binaries = [("raw_PCA_0", pca)]

        step = {"concat_transform": [PCA(n_components=10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context,
            mode="predict", loaded_binaries=loaded_binaries
        )

        # Should not create new artifacts in predict mode
        assert not mock_runtime_context.saver.persist_artifact.called


class TestConfigParsing:
    """Test configuration parsing."""

    def test_list_config(self, simple_dataset, mock_runtime_context):
        """Test list format configuration."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        # Should not raise
        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

    def test_dict_config_with_operations(self, simple_dataset, mock_runtime_context):
        """Test dict format with operations key."""
        controller = ConcatAugmentationController()

        step = {
            "concat_transform": {
                "operations": [PCA(10)],
                "name": "custom_name"
            }
        }
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # Should use custom name
        processings = simple_dataset.features_processings(0)
        assert any("custom_name" in p for p in processings)

    def test_empty_operations_is_noop(self, simple_dataset, mock_runtime_context):
        """Test that empty operations list is a no-op."""
        controller = ConcatAugmentationController()

        original_features = simple_dataset.num_features

        step = {"concat_transform": []}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata()
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        # No changes should have been made
        assert simple_dataset.num_features == original_features


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_operation_uses_concat_naming(self, simple_dataset, mock_runtime_context):
        """Test that even single operation uses concat naming convention."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        updated_context, artifacts = controller.execute(
            step_info, simple_dataset, context, mock_runtime_context, mode="train"
        )

        processings = simple_dataset.features_processings(0)
        assert any("concat" in p or "PCA" in p for p in processings)

    def test_predict_mode_raises_if_binary_missing(self, simple_dataset, mock_runtime_context):
        """Test that predict mode raises if binary is missing."""
        controller = ConcatAugmentationController()

        step = {"concat_transform": [PCA(10)]}
        step_info = make_step_info(step)

        context = ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(),
            metadata=StepMetadata(add_feature=False)
        )

        # Provide binaries list but with wrong key
        loaded_binaries = [("wrong_key", PCA(10))]

        with pytest.raises(ValueError, match="not found"):
            controller.execute(
                step_info, simple_dataset, context, mock_runtime_context,
                mode="predict", loaded_binaries=loaded_binaries
            )
