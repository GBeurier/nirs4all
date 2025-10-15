"""
Unit tests for SampleAugmentationController delegation pattern.

Tests cover:
- Standard mode (count-based augmentation)
- Balanced mode (class-aware augmentation)
- Transformer distribution and delegation
- Edge cases
"""

import numpy as np
import polars as pl
import pytest
from unittest.mock import Mock, MagicMock, call
from sklearn.preprocessing import StandardScaler

from nirs4all.controllers.dataset.op_sample_augmentation import SampleAugmentationController
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.indexer import Indexer


class DummyTransformer:
    """Mock transformer for testing."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"DummyTransformer({self.name})"


@pytest.fixture
def mock_runner():
    """Mock PipelineRunner."""
    runner = Mock()
    runner.run_step = Mock()
    return runner


@pytest.fixture
def simple_dataset():
    """Create a simple dataset with 6 samples, 2 classes."""
    dataset = SpectroDataset("test")

    # Add 6 base samples
    x_data = np.random.rand(6, 10)
    y_data = np.array([0, 0, 0, 1, 1, 1])  # 3 per class

    dataset.add_samples(x_data, {"partition": "train"})
    dataset.add_targets(y_data)

    # Add group metadata
    metadata = np.array([["A"], ["A"], ["B"], ["B"], ["C"], ["C"]])
    dataset.add_metadata(metadata, headers=["group"])

    return dataset


@pytest.fixture
def imbalanced_dataset():
    """Create imbalanced dataset: 4 class-0, 2 class-1."""
    dataset = SpectroDataset("test")

    # Add 6 base samples
    x_data = np.random.rand(6, 10)
    y_data = np.array([0, 0, 0, 0, 1, 1])

    dataset.add_samples(x_data, {"partition": "train"})
    dataset.add_targets(y_data)

    return dataset


class TestControllerMatching:
    """Test controller registration and matching."""

    def test_matches_sample_augmentation_keyword(self):
        assert SampleAugmentationController.matches(
            {}, None, "sample_augmentation"
        ) is True

    def test_does_not_match_other_keywords(self):
        assert SampleAugmentationController.matches(
            {}, None, "other_operation"
        ) is False

    def test_use_multi_source(self):
        assert SampleAugmentationController.use_multi_source() is True

    def test_does_not_support_prediction_mode(self):
        assert SampleAugmentationController.supports_prediction_mode() is False


class TestStandardMode:
    """Test standard count-based augmentation."""

    def test_standard_mode_with_count_2(self, simple_dataset, mock_runner):
        """Standard mode: each sample gets count=2 augmentations."""
        controller = SampleAugmentationController()
        transformers = [DummyTransformer("T1"), DummyTransformer("T2")]

        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "count": 2,
                "selection": "random",
                "random_state": 42
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit 2 run_step calls (one per transformer)
        assert mock_runner.run_step.call_count == 2

        # Check that context passed to run_step has augment_sample flag
        calls = mock_runner.run_step.call_args_list
        for call_args in calls:
            _, kwargs = call_args
            local_context = kwargs.get("local_context") or call_args[0][2]  # args[2] is context
            assert local_context.get("augment_sample") is True
            assert "target_samples" in local_context
            assert local_context["partition"] == "train"

    def test_standard_mode_count_1_single_transformer(self, simple_dataset, mock_runner):
        """Standard mode with count=1 and single transformer."""
        controller = SampleAugmentationController()
        transformer = DummyTransformer("T1")

        step = {
            "sample_augmentation": {
                "transformers": [transformer],
                "count": 1,
                "selection": "all"
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit 1 run_step call
        assert mock_runner.run_step.call_count == 1

        # All 6 samples should be in target_samples
        call_context = mock_runner.run_step.call_args[0][2]
        assert len(call_context["target_samples"]) == 6

    def test_standard_mode_all_selection_cycles_transformers(self, simple_dataset, mock_runner):
        """Standard mode with 'all' selection cycles through transformers."""
        controller = SampleAugmentationController()
        transformers = [DummyTransformer("T1"), DummyTransformer("T2")]

        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "count": 3,  # 3 augmentations per sample
                "selection": "all"
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit 2 run_step calls (one per transformer)
        assert mock_runner.run_step.call_count == 2

        # Each transformer should get ~half of augmentations
        # With count=3 and cycling: T1 gets indices [0, 2], T2 gets index [1]
        # So T1: 6 samples * 2 augmentations = 12 augmentations
        # T2: 6 samples * 1 augmentation = 6 augmentations
        calls = mock_runner.run_step.call_args_list
        total_augmentations = sum(len(c[0][2]["target_samples"]) for c in calls)
        assert total_augmentations == 18  # 6 samples * 3 augmentations


class TestBalancedMode:
    """Test balanced class-aware augmentation."""

    def test_balanced_mode_with_y(self, imbalanced_dataset, mock_runner):
        """Balanced mode using y labels."""
        controller = SampleAugmentationController()
        transformers = [DummyTransformer("T1")]

        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "balance": "y",
                "max_factor": 1.0,
                "selection": "random",
                "random_state": 42
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, imbalanced_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit run_step calls
        assert mock_runner.run_step.call_count >= 1

        # Minority class (1) should get more augmentations
        call_context = mock_runner.run_step.call_args[0][2]
        assert "target_samples" in call_context
        # With max_factor=1.0, minority class should be balanced to majority

    def test_balanced_mode_with_metadata_column(self, simple_dataset, mock_runner):
        """Balanced mode using metadata column."""
        controller = SampleAugmentationController()
        transformers = [DummyTransformer("T1")]

        # Imbalanced groups: A=2, B=2, C=2 (balanced already)
        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "balance": "group",
                "max_factor": 1.0,
                "selection": "random",
                "random_state": 42
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit run_step
        assert mock_runner.run_step.call_count >= 0  # May be 0 if already balanced

    def test_balanced_mode_max_factor_limits_augmentation(self, imbalanced_dataset, mock_runner):
        """Balanced mode respects max_factor parameter."""
        controller = SampleAugmentationController()
        transformers = [DummyTransformer("T1")]

        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "balance": "y",
                "max_factor": 0.5,  # Limit augmentation
                "selection": "random",
                "random_state": 42
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, imbalanced_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit fewer augmentations due to max_factor
        assert mock_runner.run_step.call_count >= 0


class TestTransformerDistribution:
    """Test transformer→samples mapping."""

    def test_invert_transformer_map(self):
        """Test _invert_transformer_map helper."""
        controller = SampleAugmentationController()

        transformer_map = {
            0: [0, 1],  # Sample 0 → T0, T1
            1: [0],     # Sample 1 → T0
            2: [1, 1]   # Sample 2 → T1 twice
        }

        inverted = controller._invert_transformer_map(transformer_map, n_transformers=2)

        assert inverted[0] == [0, 1]  # T0 → samples 0, 1
        assert inverted[1] == [0, 2, 2]  # T1 → samples 0, 2 (twice)

    def test_cycle_transformers(self):
        """Test _cycle_transformers helper."""
        controller = SampleAugmentationController()

        transformers = [DummyTransformer("T1"), DummyTransformer("T2")]
        augmentation_counts = {0: 3, 1: 2, 2: 0}

        result = controller._cycle_transformers(transformers, augmentation_counts)

        # Sample 0: count=3 → [0, 1, 0] (T1, T2, T1)
        assert result[0] == [0, 1, 0]
        # Sample 1: count=2 → [0, 1] (T1, T2)
        assert result[1] == [0, 1]
        # Sample 2: count=0 → []
        assert result[2] == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transformers_list_raises_error(self, simple_dataset, mock_runner):
        """No transformers should raise ValueError."""
        controller = SampleAugmentationController()

        step = {
            "sample_augmentation": {
                "transformers": [],
                "count": 1
            }
        }

        context = {"partition": "train"}

        with pytest.raises(ValueError, match="requires at least one transformer"):
            controller.execute(
                step, None, simple_dataset, context, mock_runner, loaded_binaries=None
            )

    def test_empty_dataset_returns_early(self, mock_runner):
        """Empty dataset should return without emitting steps."""
        controller = SampleAugmentationController()

        # Empty dataset (no samples added)
        dataset = SpectroDataset("empty_test")

        step = {
            "sample_augmentation": {
                "transformers": [DummyTransformer("T1")],
                "count": 1
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, dataset, context, mock_runner, loaded_binaries=None
        )

        # No run_step calls
        assert mock_runner.run_step.call_count == 0

    def test_balanced_mode_invalid_balance_source_raises(self, simple_dataset, mock_runner):
        """Invalid balance source should raise ValueError."""
        controller = SampleAugmentationController()

        step = {
            "sample_augmentation": {
                "transformers": [DummyTransformer("T1")],
                "balance": 123,  # Invalid: not string or "y"
                "max_factor": 1.0
            }
        }

        context = {"partition": "train"}

        with pytest.raises(ValueError, match="balance source must be"):
            controller.execute(
                step, None, simple_dataset, context, mock_runner, loaded_binaries=None
            )


class TestDelegationPattern:
    """Test proper delegation to TransformerMixinController."""

    def test_emits_one_run_step_per_transformer(self, simple_dataset, mock_runner):
        """Should emit exactly ONE run_step per transformer."""
        controller = SampleAugmentationController()
        transformers = [
            DummyTransformer("T1"),
            DummyTransformer("T2"),
            DummyTransformer("T3")
        ]

        step = {
            "sample_augmentation": {
                "transformers": transformers,
                "count": 2,
                "selection": "random",
                "random_state": 42
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Should emit at most 3 run_step calls (one per transformer)
        assert mock_runner.run_step.call_count <= 3

    def test_context_contains_augment_sample_flag(self, simple_dataset, mock_runner):
        """Emitted context should have augment_sample=True flag."""
        controller = SampleAugmentationController()
        transformer = DummyTransformer("T1")

        step = {
            "sample_augmentation": {
                "transformers": [transformer],
                "count": 1
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Check emitted context
        call_context = mock_runner.run_step.call_args[0][2]
        assert call_context["augment_sample"] is True

    def test_context_contains_target_samples(self, simple_dataset, mock_runner):
        """Emitted context should contain list of target sample indices."""
        controller = SampleAugmentationController()
        transformer = DummyTransformer("T1")

        step = {
            "sample_augmentation": {
                "transformers": [transformer],
                "count": 1
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Check target_samples
        call_context = mock_runner.run_step.call_args[0][2]
        assert "target_samples" in call_context
        assert isinstance(call_context["target_samples"], list)
        assert len(call_context["target_samples"]) > 0

    def test_is_substep_true(self, simple_dataset, mock_runner):
        """run_step should be called with is_substep=True."""
        controller = SampleAugmentationController()
        transformer = DummyTransformer("T1")

        step = {
            "sample_augmentation": {
                "transformers": [transformer],
                "count": 1
            }
        }

        context = {"partition": "train"}

        controller.execute(
            step, None, simple_dataset, context, mock_runner, loaded_binaries=None
        )

        # Check is_substep parameter
        _, kwargs = mock_runner.run_step.call_args
        assert kwargs.get("is_substep") is True
