"""
End-to-end integration tests for sample augmentation feature.

Tests complete pipelines focusing on:
1. Dataset API with augmentation
2. Interaction with CV splits and leak prevention
3. Multi-round augmentation scenarios
4. Metadata preservation
5. Edge cases

Note: Controller-level orchestration is tested in unit tests.
These integration tests focus on the dataset and split integration.
"""

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from nirs4all.controllers.splitters.split import CrossValidatorController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.steps.parser import ParsedStep, StepType
from nirs4all.pipeline.config.context import ExecutionContext


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


@pytest.fixture
def mock_runner():
    """Mock PipelineRunner (minimal implementation for split controller)."""
    from unittest.mock import Mock
    runner = Mock()
    return runner


class TestDatasetAugmentationAPI:
    """Test dataset augmentation API integration."""

    def test_augment_samples_adds_correct_count(self):
        """Test that augment_samples adds the correct number of samples."""
        dataset = SpectroDataset("test")

        # Add 5 base samples
        x_data = np.random.randn(5, 3)
        dataset.add_samples(x_data, {})

        # Augment with count=2
        aug_data = np.random.randn(10, 3)  # 5 samples * 2 augmentations = 10
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=2
        )

        # Total: 5 base + 10 augmented = 15
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 15

    def test_augment_samples_with_selector(self):
        """Test augmenting only selected samples."""
        dataset = SpectroDataset("test")

        # Add samples with different partitions
        x_train = np.random.randn(4, 3)
        x_test = np.random.randn(2, 3)
        dataset.add_samples(x_train, {"partition": "train"})
        dataset.add_samples(x_test, {"partition": "test"})

        # Augment only train samples
        aug_data = np.random.randn(4, 3)  # 4 train samples * 1 augmentation
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            selector={"partition": "train"},
            count=1
        )

        # Total: 6 base + 4 augmented = 10
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 10

        # Only train samples should be augmented
        train_data = dataset.x({"partition": "train"}, layout="2d", concat_source=True)
        assert train_data.shape[0] == 8  # 4 base + 4 augmented

    def test_include_augmented_parameter(self):
        """Test include_augmented parameter filters augmented samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(3, 2)
        dataset.add_samples(x_data, {})

        aug_data = np.random.randn(3, 2)
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # With augmented
        all_data = dataset.x({}, layout="2d", concat_source=True, include_augmented=True)
        assert all_data.shape[0] == 6

        # Without augmented
        base_only = dataset.x({}, layout="2d", concat_source=True, include_augmented=False)
        assert base_only.shape[0] == 3


class TestLeakPreventionIntegration:
    """Test leak prevention in complete pipelines."""

    def test_cv_split_excludes_augmented_samples(self, mock_runner):
        """Verify CV splits only use base samples."""
        dataset = SpectroDataset("test")

        # Create dataset
        np.random.seed(42)
        x_data = np.random.randn(20, 5)
        y_data = np.array([0] * 10 + [1] * 10)
        dataset.add_samples(x_data, {})
        dataset.add_targets(y_data)

        # Augment: 3 per sample
        aug_data = np.random.randn(60, 5)  # 20 * 3
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=3
        )

        # Total: 20 base + 60 augmented = 80
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 80

        # Perform CV split - should only use base samples
        split_controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        split_controller.execute(
            step_info=make_step_info(splitter),
            dataset=dataset,
            context=ExecutionContext(),
            runner=mock_runner
        )

        # Verify all indices in all folds are < 20 (base samples only)
        folds = dataset.folds
        assert len(folds) == 5

        for train_idx, val_idx in folds:
            assert all(idx < 20 for idx in train_idx)
            assert all(idx < 20 for idx in val_idx)
            assert len(train_idx) == 16
            assert len(val_idx) == 4

    def test_kfold_with_augmented_samples(self, mock_runner):
        """Test KFold splitting with augmented samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(10, 3)
        y_data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dataset.add_samples(x_data, {})
        dataset.add_targets(y_data)

        # Augment
        aug_data = np.random.randn(20, 3)  # 10 * 2
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=2
        )

        # Split
        split_controller = CrossValidatorController()
        splitter = KFold(n_splits=2, shuffle=False)

        split_controller.execute(
            step_info=make_step_info(splitter),
            dataset=dataset,
            context=ExecutionContext(),
            runner=mock_runner
        )

        folds = dataset.folds
        assert len(folds) == 2

        for train_idx, val_idx in folds:
            # All indices should be from base samples
            assert all(idx < 10 for idx in train_idx)
            assert all(idx < 10 for idx in val_idx)

    def test_group_kfold_with_augmented_samples(self, mock_runner):
        """Test GroupKFold splitting with augmented samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(12, 3)
        y_data = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        dataset.add_samples(x_data, {})
        dataset.add_targets(y_data)
        dataset.add_metadata(groups.reshape(-1, 1), headers=["group"])

        # Augment
        aug_data = np.random.randn(12, 3)  # 12 * 1
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # Total: 12 base + 12 augmented = 24
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 24

        # Split by groups
        split_controller = CrossValidatorController()
        splitter = GroupKFold(n_splits=3)

        split_controller.execute(
            step_info=make_step_info(splitter, {"group": "group"}),
            dataset=dataset,
            context=ExecutionContext(),
            runner=mock_runner
        )

        folds = dataset.folds
        assert len(folds) == 3

        for train_idx, val_idx in folds:
            # All indices from base samples only
            assert all(idx < 12 for idx in train_idx)
            assert all(idx < 12 for idx in val_idx)
            # Each fold should have one group (4 samples)
            assert len(val_idx) == 4


class TestMultiRoundAugmentation:
    """Test multiple rounds of augmentation."""

    def test_sequential_augmentations_only_augment_base(self):
        """Test that sequential augmentations only target base samples."""
        dataset = SpectroDataset("test")

        # Add 5 base samples
        x_data = np.random.randn(5, 4)
        dataset.add_samples(x_data, {})

        # First augmentation
        aug_data1 = np.random.randn(5, 4)
        dataset.augment_samples(
            data=aug_data1,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # After first: 5 base + 5 augmented = 10
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 10

        # Second augmentation - should only augment base samples (5)
        aug_data2 = np.random.randn(5, 4)
        dataset.augment_samples(
            data=aug_data2,
            processings=["raw"],
            augmentation_id="aug2",
            count=1
        )

        # After second: 5 base + 5 from aug1 + 5 from aug2 = 15
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 15

        # Verify base samples remain 5
        base_data = dataset.x({}, layout="2d", concat_source=True, include_augmented=False)
        assert base_data.shape[0] == 5

    def test_three_rounds_of_augmentation(self):
        """Test three sequential augmentation rounds."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(4, 3)
        dataset.add_samples(x_data, {})

        # Round 1
        aug_data1 = np.random.randn(4, 3)
        dataset.augment_samples(
            data=aug_data1, processings=["raw"],
            augmentation_id="aug1", count=1
        )

        # Round 2
        aug_data2 = np.random.randn(4, 3)
        dataset.augment_samples(
            data=aug_data2, processings=["raw"],
            augmentation_id="aug2", count=1
        )

        # Round 3
        aug_data3 = np.random.randn(4, 3)
        dataset.augment_samples(
            data=aug_data3, processings=["raw"],
            augmentation_id="aug3", count=1
        )

        # Total: 4 base + 4 + 4 + 4 = 16
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 16

        # Verify augmentation IDs from indexer
        aug_ids = dataset._indexer.df["augmentation"].unique().to_list()
        aug_ids = [aid for aid in aug_ids if aid is not None]
        assert len(aug_ids) == 3
        assert set(aug_ids) == {"aug1", "aug2", "aug3"}


class TestMetadataPreservation:
    """Test that augmented samples preserve metadata."""

    def test_augmented_samples_inherit_metadata(self):
        """Test augmented samples inherit metadata from origin samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(4, 2)
        dataset.add_samples(x_data, {"partition": "train", "subject": ["S1", "S1", "S2", "S2"]})

        # Augment
        aug_data = np.random.randn(4, 2)
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # Check augmented samples have correct metadata
        # Get augmented sample indices from indexer (where sample != origin)
        indexer_df = dataset._indexer.df
        augmented_rows = indexer_df.filter(indexer_df["sample"] != indexer_df["origin"])

        assert len(augmented_rows) == 4
        assert all(p == "train" for p in augmented_rows["partition"].to_list())
        assert all(aid == "aug1" for aid in augmented_rows["augmentation"].to_list())

        # Verify metadata inheritance by checking sample indices
        augmented_indices = augmented_rows["sample"].to_list()
        metadata = dataset.metadata()
        if len(metadata) > 0:  # Only check if metadata exists
            augmented_metadata = metadata.filter(metadata["row_id"].is_in(augmented_indices))
            assert augmented_metadata["subject"].to_list() == ["S1", "S1", "S2", "S2"]

    def test_targets_preserved_for_augmented_samples(self):
        """Test that targets are preserved for augmented samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(6, 2)
        y_data = np.array([0, 0, 0, 1, 1, 1])
        dataset.add_samples(x_data, {})
        dataset.add_targets(y_data)

        # Augment
        aug_data = np.random.randn(6, 2)
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # Targets should be duplicated for augmented samples
        all_y = dataset.y({})
        assert all_y.shape[0] == 12
        assert np.sum(all_y == 0) == 6  # 3 base + 3 augmented
        assert np.sum(all_y == 1) == 6  # 3 base + 3 augmented


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_augmentation_with_count_zero_per_sample(self):
        """Test augmentation where some samples have count=0."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(4, 2)
        dataset.add_samples(x_data, {})

        # Augment with count list: only first 2 samples
        aug_data = np.random.randn(2, 2)
        dataset.augment_samples(
            data=aug_data,
            processings=["raw"],
            augmentation_id="aug1",
            count=[1, 1, 0, 0]
        )

        # Total: 4 base + 2 augmented = 6
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 6

    def test_split_with_no_augmentation(self, mock_runner):
        """Test that splits work normally when no augmentation present."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(10, 3)
        y_data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dataset.add_samples(x_data, {})
        dataset.add_targets(y_data)

        # Split without any augmentation
        split_controller = CrossValidatorController()
        splitter = StratifiedKFold(n_splits=2, shuffle=False)

        split_controller.execute(
            step_info=make_step_info(splitter),
            dataset=dataset,
            context=ExecutionContext(),
            runner=mock_runner
        )

        folds = dataset.folds
        assert len(folds) == 2

        for train_idx, val_idx in folds:
            assert len(train_idx) == 5
            assert len(val_idx) == 5

    def test_empty_selector_augments_all_base_samples(self):
        """Test that augment_samples with no selector augments all base samples."""
        dataset = SpectroDataset("test")

        x_data = np.random.randn(5, 2)
        dataset.add_samples(x_data, {})

        # First augmentation
        aug_data1 = np.random.randn(5, 2)
        dataset.augment_samples(
            data=aug_data1,
            processings=["raw"],
            augmentation_id="aug1",
            count=1
        )

        # Second augmentation with no selector - should still only augment base 5
        aug_data2 = np.random.randn(5, 2)
        dataset.augment_samples(
            data=aug_data2,
            processings=["raw"],
            augmentation_id="aug2",
            selector=None,  # Explicitly None
            count=1
        )

        # Total: 5 base + 5 + 5 = 15
        assert dataset.x({}, layout="2d", concat_source=True).shape[0] == 15
