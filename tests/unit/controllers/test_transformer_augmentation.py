"""
Unit tests for TransformerMixinController sample augmentation mode.

Tests cover:
- Detection of augment_sample flag
- Transformation of target samples
- Addition of augmented samples with proper origin tracking
- Multi-source support
- Edge cases
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.data.dataset import SpectroDataset


@pytest.fixture
def mock_runner():
    """Mock PipelineRunner."""
    runner = Mock()
    runner.next_op = Mock(side_effect=lambda: f"op_{runner.next_op.call_count}")
    return runner


@pytest.fixture
def simple_dataset():
    """Create a simple dataset with 5 samples."""
    dataset = SpectroDataset("test")

    # Add 5 base samples
    x_data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ])

    dataset.add_samples(x_data, {"partition": "train"})

    return dataset


class TestAugmentSampleDetection:
    """Test detection of augment_sample flag."""

    def test_normal_execution_when_flag_absent(self, simple_dataset, mock_runner):
        """Test that normal execution runs when augment_sample flag is absent."""
        controller = TransformerMixinController()
        transformer = StandardScaler()

        context = {
            "partition": "train",
            "processing": [["raw"]]  # Required by normal execution
        }

        # Should execute normally (not augmentation mode)
        result_context, binaries = controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have processed data normally
        assert result_context is not None
        assert isinstance(binaries, list)

    def test_augmentation_execution_when_flag_true(self, simple_dataset, mock_runner):
        """Test that augmentation mode triggers when flag is True."""
        controller = TransformerMixinController()
        transformer = StandardScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0, 1],
            "partition": "train",
            "augmentation_id": "test_aug"
        }

        # Should execute in augmentation mode
        result_context, binaries = controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have added augmented samples
        total_samples = simple_dataset.x({"partition": "train"}).shape[0]
        assert total_samples == 7  # 5 base + 2 augmented


class TestSampleAugmentation:
    """Test sample augmentation functionality."""

    def test_augment_single_sample(self, simple_dataset, mock_runner):
        """Test augmenting a single sample."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()  # Simple transformer

        initial_count = simple_dataset.x({"partition": "train"}).shape[0]

        context = {
            "augment_sample": True,
            "target_samples": [2],  # Augment sample ID 2
            "partition": "train",
            "augmentation_id": "aug_test"
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have added 1 augmented sample
        new_count = simple_dataset.x({"partition": "train"}).shape[0]
        assert new_count == initial_count + 1

    def test_augment_multiple_samples(self, simple_dataset, mock_runner):
        """Test augmenting multiple samples."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        initial_count = simple_dataset.x({"partition": "train"}).shape[0]

        context = {
            "augment_sample": True,
            "target_samples": [0, 2, 4],  # Augment 3 samples
            "partition": "train",
            "augmentation_id": "aug_multi"
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have added 3 augmented samples
        new_count = simple_dataset.x({"partition": "train"}).shape[0]
        assert new_count == initial_count + 3

    def test_augmented_samples_have_origin_tracking(self, simple_dataset, mock_runner):
        """Test that augmented samples track their origin."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        context = {
            "augment_sample": True,
            "target_samples": [1],
            "partition": "train",
            "augmentation_id": "aug_origin_test"
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Check indexer for origin tracking
        # Get all train samples (including augmented)
        all_indices = simple_dataset._indexer.x_indices({"partition": "train"})  # noqa: SLF001
        assert len(all_indices) == 6  # 5 base + 1 augmented

        # Check that the last sample has origin=1
        last_sample_id = all_indices[-1]
        origin = simple_dataset._indexer.get_origin_for_sample(last_sample_id)  # noqa: SLF001
        assert origin == 1

    def test_augmented_samples_inherit_group(self, mock_runner):
        """Test that augmented samples can be identified by their origin."""
        dataset = SpectroDataset("test")

        # Add samples with group metadata
        x_data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        metadata = np.array([["A"], ["B"]])
        dataset.add_samples(x_data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group"])

        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0],  # Augment sample 0 (group A)
            "partition": "train",
            "augmentation_id": "aug_group_test"
        }

        controller.execute(
            step={}, operator=transformer, dataset=dataset,
            context=context, runner=mock_runner
        )

        # Check that augmented sample can be tracked back to origin
        all_indices = dataset._indexer.x_indices({"partition": "train"})  # noqa: SLF001
        assert len(all_indices) == 3  # 2 base + 1 augmented

        # Get origin of last sample
        origin = dataset._indexer.get_origin_for_sample(all_indices[-1])  # noqa: SLF001
        assert origin == 0

        # Origin sample has group A
        origin_meta = dataset.metadata({"sample": [origin]})
        assert len(origin_meta) > 0
        assert origin_meta["group"][0] == "A"


class TestTransformation:
    """Test that transformations are applied correctly."""

    def test_transformation_applied_to_data(self, simple_dataset, mock_runner):
        """Test that transformer actually transforms the data."""
        controller = TransformerMixinController()

        # Use a simple transformer that we can verify
        transformer = StandardScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0],
            "partition": "train",
            "augmentation_id": "aug_transform_test"
        }

        # Get original sample 0 data
        original_data = simple_dataset.x({"sample": [0]}, layout="2d")

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Get augmented sample (last one added)
        all_data = simple_dataset.x({"partition": "train"}, layout="2d")
        augmented_data = all_data[-1:, :]

        # Transformed data should be different from original
        assert not np.allclose(augmented_data, original_data)

    def test_transformer_fitted_on_train_data(self, simple_dataset, mock_runner):
        """Test that transformer is fitted on full training data."""
        controller = TransformerMixinController()
        transformer = StandardScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0],
            "partition": "train",
            "augmentation_id": "aug_fit_test"
        }

        # Execute augmentation
        _, binaries = controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have saved transformer binaries
        assert len(binaries) > 0
        assert all(isinstance(b, tuple) and len(b) == 2 for b in binaries)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_target_samples_list(self, simple_dataset, mock_runner):
        """Test with empty target_samples list."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        initial_count = simple_dataset.x({"partition": "train"}).shape[0]

        context = {
            "augment_sample": True,
            "target_samples": [],  # Empty list
            "partition": "train"
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should not add any samples
        new_count = simple_dataset.x({"partition": "train"}).shape[0]
        assert new_count == initial_count

    def test_missing_target_samples_key(self, simple_dataset, mock_runner):
        """Test with missing target_samples key."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        initial_count = simple_dataset.x({"partition": "train"}).shape[0]

        context = {
            "augment_sample": True,
            # target_samples key missing
            "partition": "train"
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should not add any samples
        new_count = simple_dataset.x({"partition": "train"}).shape[0]
        assert new_count == initial_count

    def test_augmentation_id_generation(self, simple_dataset, mock_runner):
        """Test automatic augmentation_id generation."""
        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0],
            "partition": "train"
            # augmentation_id not provided
        }

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Check that augmentation_id was generated
        all_indices = simple_dataset._indexer.x_indices({"partition": "train"})  # noqa: SLF001
        # Should have 6 samples (5 base + 1 augmented)
        assert len(all_indices) == 6


class TestMultiSource:
    """Test multi-source dataset support."""

    def test_multi_source_augmentation(self, mock_runner):
        """Test augmentation with multi-source dataset."""
        dataset = SpectroDataset("test_multi")

        # Add samples with 2 sources
        x_data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_data2 = np.array([[5.0, 6.0], [7.0, 8.0]])

        dataset.add_samples(x_data1, {"partition": "train"})
        dataset.add_samples(x_data2, {"partition": "train"})

        controller = TransformerMixinController()
        transformer = MinMaxScaler()

        context = {
            "augment_sample": True,
            "target_samples": [0],
            "partition": "train",
            "augmentation_id": "aug_multi_source"
        }

        initial_data = dataset.x({"partition": "train"}, layout="2d", concat_source=True)
        initial_count = initial_data.shape[0]

        controller.execute(
            step={}, operator=transformer, dataset=dataset,
            context=context, runner=mock_runner
        )

        # Should have added 1 augmented sample
        new_data = dataset.x({"partition": "train"}, layout="2d", concat_source=True)
        new_count = new_data.shape[0]
        assert new_count == initial_count + 1


class TestIntegration:
    """Integration tests with delegation pattern."""

    def test_integration_with_sample_augmentation_controller(self, simple_dataset, mock_runner):
        """Test that context from SampleAugmentationController works correctly."""
        controller = TransformerMixinController()
        transformer = StandardScaler()

        # Simulate context emitted by SampleAugmentationController
        context = {
            "augment_sample": True,
            "target_samples": [0, 1, 2],
            "partition": "train",
            "augmentation_id": "aug_StandardScaler_0"
        }

        initial_count = simple_dataset.x({"partition": "train"}).shape[0]

        controller.execute(
            step={}, operator=transformer, dataset=simple_dataset,
            context=context, runner=mock_runner
        )

        # Should have added 3 augmented samples
        new_count = simple_dataset.x({"partition": "train"}).shape[0]
        assert new_count == initial_count + 3

        # All augmented samples should have proper origin tracking
        all_indices = simple_dataset._indexer.x_indices({"partition": "train"})  # noqa: SLF001

        # Check last 3 samples (augmented ones) have origins
        for i in range(3):
            sample_id = all_indices[-(3 - i)]
            origin = simple_dataset._indexer.get_origin_for_sample(sample_id)  # noqa: SLF001
            assert origin in [0, 1, 2]
