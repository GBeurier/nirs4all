"""
Unit tests for CrossValidatorController with augmented samples.

Tests cover:
- Leak prevention: splits only use base samples
- Augmented samples excluded from fold generation
- Group-based splitting with augmented samples
- Integration with sample augmentation
"""

import numpy as np
import pytest
from unittest.mock import Mock
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from nirs4all.controllers.sklearn.op_split import CrossValidatorController
from nirs4all.data.dataset import SpectroDataset


@pytest.fixture
def mock_runner():
    """Mock PipelineRunner."""
    return Mock()


@pytest.fixture
def dataset_with_augmentation():
    """Create dataset with base and augmented samples."""
    dataset = SpectroDataset("test")

    # Add 6 base samples
    x_data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0]
    ])
    y_data = np.array([0, 0, 0, 1, 1, 1])

    dataset.add_samples(x_data, {"partition": "train"})
    dataset.add_targets(y_data)

    # Add 3 augmented samples (from samples 0, 1, 2)
    aug_data = np.array([
        [1.1, 2.1, 3.1],
        [2.1, 3.1, 4.1],
        [3.1, 4.1, 5.1]
    ])
    dataset.augment_samples(
        data=aug_data,
        processings=["raw"],
        augmentation_id="aug_test",
        selector={"partition": "train"},
        count=[1, 1, 1, 0, 0, 0]
    )

    return dataset


@pytest.fixture
def dataset_with_augmentation_and_groups():
    """Create dataset with base samples, augmented samples, and group metadata."""
    dataset = SpectroDataset("test")

    # Add 6 base samples with groups
    x_data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0]
    ])
    y_data = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array([["A"], ["A"], ["B"], ["B"], ["C"], ["C"]])

    dataset.add_samples(x_data, {"partition": "train"})
    dataset.add_targets(y_data)
    dataset.add_metadata(groups, headers=["group"])

    # Add 2 augmented samples (from samples 0, 3)
    aug_data = np.array([
        [1.1, 2.1, 3.1],
        [4.1, 5.1, 6.1]
    ])
    dataset.augment_samples(
        data=aug_data,
        processings=["raw"],
        augmentation_id="aug_group_test",
        selector={"partition": "train"},
        count=[1, 0, 0, 1, 0, 0]
    )

    return dataset


class TestLeakPrevention:
    """Test that splits exclude augmented samples to prevent data leakage."""

    def test_kfold_excludes_augmented_samples(self, dataset_with_augmentation, mock_runner):
        """Test that KFold only splits on base samples."""
        controller = CrossValidatorController()
        splitter = KFold(n_splits=3, shuffle=False)

        context = {"partition": "train"}

        controller.execute(
            step={}, operator=splitter, dataset=dataset_with_augmentation,
            context=context, runner=mock_runner
        )

        # Get the folds
        folds = dataset_with_augmentation._folds  # noqa: SLF001

        # Should have 3 folds
        assert len(folds) == 3

        # Each fold should split only the 6 base samples (not 9 total samples)
        for train_idx, val_idx in folds:
            # Train + val should equal 6 (base samples only)
            assert len(train_idx) + len(val_idx) == 6
            # All indices should be < 6 (base sample indices)
            assert all(idx < 6 for idx in train_idx)
            assert all(idx < 6 for idx in val_idx)

    def test_stratified_kfold_excludes_augmented(self, dataset_with_augmentation, mock_runner):
        """Test that StratifiedKFold only uses base samples."""
        controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=2, shuffle=False)

        context = {"partition": "train"}

        controller.execute(
            step={}, operator=splitter, dataset=dataset_with_augmentation,
            context=context, runner=mock_runner
        )

        folds = dataset_with_augmentation._folds  # noqa: SLF001

        # Should have 2 folds
        assert len(folds) == 2

        # Each fold should only use base samples
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 6
            assert all(idx < 6 for idx in train_idx)
            assert all(idx < 6 for idx in val_idx)

    def test_group_kfold_excludes_augmented(self, dataset_with_augmentation_and_groups, mock_runner):
        """Test that GroupKFold only uses base samples."""
        controller = CrossValidatorController()
        splitter = GroupKFold(n_splits=3)

        step = {"group": "group"}  # Specify group column
        context = {"partition": "train"}

        controller.execute(
            step=step, operator=splitter, dataset=dataset_with_augmentation_and_groups,
            context=context, runner=mock_runner
        )

        folds = dataset_with_augmentation_and_groups._folds  # noqa: SLF001

        # Should have 3 folds (one per group: A, B, C)
        assert len(folds) == 3

        # Each fold should only use base samples (6 total)
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 6
            assert all(idx < 6 for idx in train_idx)
            assert all(idx < 6 for idx in val_idx)


class TestDatasetCounts:
    """Test that correct number of samples are used for splitting."""

    def test_x_shape_reflects_base_samples_only(self, dataset_with_augmentation, mock_runner):
        """Test that X used for splitting has only base samples."""
        controller = CrossValidatorController()
        splitter = KFold(n_splits=2, shuffle=False)

        context = {"partition": "train"}

        # Get X shape before splitting
        X_all = dataset_with_augmentation.x({"partition": "train"}, layout="2d")
        assert X_all.shape[0] == 9  # 6 base + 3 augmented

        # Execute split
        controller.execute(
            step={}, operator=splitter, dataset=dataset_with_augmentation,
            context=context, runner=mock_runner
        )

        # Folds should be based on 6 samples only
        folds = dataset_with_augmentation._folds  # noqa: SLF001
        total_samples_in_folds = sum(len(train) + len(val) for train, val in folds)

        # Each fold uses all 6 base samples (split differently)
        assert total_samples_in_folds == 12  # 2 folds * 6 samples

    def test_y_shape_matches_base_samples(self, dataset_with_augmentation, mock_runner):
        """Test that y used for splitting has only base samples."""
        controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=2, shuffle=False)

        context = {"partition": "train"}

        # y should have 6 values (base samples only) when used for splitting
        controller.execute(
            step={}, operator=splitter, dataset=dataset_with_augmentation,
            context=context, runner=mock_runner
        )

        folds = dataset_with_augmentation._folds  # noqa: SLF001

        # Check that stratification worked on 6 samples
        # (3 class 0, 3 class 1)
        for train_idx, val_idx in folds:
            assert len(train_idx) == 3
            assert len(val_idx) == 3


class TestGroupMetadata:
    """Test group metadata handling with augmented samples."""

    def test_groups_extracted_from_base_samples_only(
        self, dataset_with_augmentation_and_groups, mock_runner
    ):
        """Test that groups are extracted only from base samples."""
        controller = CrossValidatorController()
        splitter = GroupKFold(n_splits=3)

        step = {"group": "group"}
        context = {"partition": "train"}

        controller.execute(
            step=step, operator=splitter, dataset=dataset_with_augmentation_and_groups,
            context=context, runner=mock_runner
        )

        folds = dataset_with_augmentation_and_groups._folds  # noqa: SLF001

        # Should have 3 folds (one per unique group: A, B, C)
        assert len(folds) == 3

        # Groups A, B, C each have 2 base samples
        # Each fold should hold out one group (2 samples)
        for train_idx, val_idx in folds:
            assert len(val_idx) == 2  # One group held out
            assert len(train_idx) == 4  # Two groups in training

    def test_groups_length_matches_base_samples(
        self, dataset_with_augmentation_and_groups, mock_runner
    ):
        """Test that groups array length matches base samples count."""
        controller = CrossValidatorController()
        splitter = GroupKFold(n_splits=2)

        step = {"group": "group"}
        context = {"partition": "train"}

        # Should not raise error about mismatched lengths
        controller.execute(
            step=step, operator=splitter, dataset=dataset_with_augmentation_and_groups,
            context=context, runner=mock_runner
        )

        # If we got here, the groups length matched X.shape[0] correctly


class TestIntegration:
    """Integration tests with sample augmentation workflow."""

    def test_split_after_augmentation(self, mock_runner):
        """Test splitting after sample augmentation in pipeline."""
        dataset = SpectroDataset("test")

        # Add base samples
        x_data = np.random.rand(10, 5)
        y_data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dataset.add_samples(x_data, {"partition": "train"})
        dataset.add_targets(y_data)

        # Simulate sample augmentation (add 5 augmented samples)
        aug_data = np.random.rand(5, 5)
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug_pipeline",
            selector={"partition": "train"},
            count=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

        # Now split (should only use 10 base samples)
        controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=5, shuffle=False)

        context = {"partition": "train"}
        controller.execute(
            step={}, operator=splitter, dataset=dataset,
            context=context, runner=mock_runner
        )

        folds = dataset._folds  # noqa: SLF001

        # Should have 5 folds
        assert len(folds) == 5

        # Each fold should use only 10 base samples
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 10
            assert len(val_idx) == 2  # 10 samples / 5 folds

    def test_multiple_augmentations_then_split(self, mock_runner):
        """Test splitting after multiple rounds of augmentation on base samples only."""
        dataset = SpectroDataset("test")

        # Add 4 base samples with unique metadata
        x_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_data = np.array([0, 0, 1, 1])
        dataset.add_samples(x_data, {"partition": "train", "base": True})
        dataset.add_targets(y_data)

        # First augmentation: Create 4 augmented samples (one per base)
        aug_data1 = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1]])
        dataset.augment_samples(
            data=aug_data1,
            processings=["raw"],
            augmentation_id="aug1",
            selector={"partition": "train", "base": True},  # Only augment base samples
            count=1
        )

        # Second augmentation: Augment the same 4 base samples again
        aug_data2 = np.array([[1.2, 2.2], [3.2, 4.2], [5.2, 6.2], [7.2, 8.2]])
        dataset.augment_samples(
            data=aug_data2,
            processings=["raw"],
            augmentation_id="aug2",
            selector={"partition": "train", "base": True},  # Only augment base samples
            count=1
        )        # Total: 4 base + 4 from aug1 + 4 from aug2 = 12 samples
        total_data = dataset.x({"partition": "train"}, layout="2d", concat_source=True)
        assert total_data.shape[0] == 12

        # Split should use only 4 base samples
        controller = CrossValidatorController()
        splitter = KFold(n_splits=2, shuffle=False)

        context = {"partition": "train"}
        controller.execute(
            step={}, operator=splitter, dataset=dataset,
            context=context, runner=mock_runner
        )

        folds = dataset._folds  # noqa: SLF001

        # Each fold should use only 4 base samples
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 4
            assert len(train_idx) == 2
            assert len(val_idx) == 2


class TestEdgeCases:
    """Test edge cases."""

    def test_split_with_no_augmentation(self, mock_runner):
        """Test that split works normally when no augmentation present."""
        dataset = SpectroDataset("test")

        # Add only base samples (no augmentation)
        x_data = np.random.rand(6, 3)
        y_data = np.array([0, 0, 0, 1, 1, 1])  # Add targets
        dataset.add_samples(x_data, {"partition": "train"})
        dataset.add_targets(y_data)  # Required for StratifiedKFold

        controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=3, shuffle=False)  # Use StratifiedKFold which needs y

        context = {"partition": "train"}
        controller.execute(
            step={}, operator=splitter, dataset=dataset,
            context=context, runner=mock_runner
        )

        folds = dataset._folds  # noqa: SLF001

        # Should work normally with 6 samples
        assert len(folds) == 3
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 6

    def test_split_all_samples_augmented(self, mock_runner):
        """Test split when all samples have augmented versions."""
        dataset = SpectroDataset("test")

        # Add 4 base samples
        x_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_data = np.array([0, 0, 1, 1])  # Add targets
        dataset.add_samples(x_data, {"partition": "train"})
        dataset.add_targets(y_data)  # Required for StratifiedKFold

        # Augment all samples
        aug_data = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1]])
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug_all",
            selector={"partition": "train"},
            count=[1, 1, 1, 1]
        )

        # Total: 8 samples (4 base + 4 augmented)
        total_data = dataset.x({"partition": "train"}, layout="2d", concat_source=True)
        assert total_data.shape[0] == 8

        # Split should still use only 4 base samples
        controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=2, shuffle=False)  # Use StratifiedKFold

        context = {"partition": "train"}
        controller.execute(
            step={}, operator=splitter, dataset=dataset,
            context=context, runner=mock_runner
        )

        folds = dataset._folds  # noqa: SLF001

        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 4
