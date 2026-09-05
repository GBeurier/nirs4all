"""Real filter and splitter preview conservation, including held-out rows."""

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit, StratifiedKFold

from nirs4all.analysis.playground_partition import filter_batch, select_sample_indices, split_batch
from nirs4all.analysis.playground_steps import augment_batch, transform_batch
from nirs4all.analysis.playground_types import PreviewBatch
from nirs4all.operators.filters import MetadataFilter, YOutlierFilter
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate


@pytest.fixture
def batch():
    rng = np.random.default_rng(4)
    return PreviewBatch.from_arrays(rng.normal(size=(24, 15)), y=np.arange(24, dtype=float) / 2,
                                    metadata={"subject": [f"s{i // 2}" for i in range(24)],
                                              "quality": ["bad" if i % 3 == 0 else "good" for i in range(24)]},
                                    sample_ids=[f"row{i}" for i in range(24)],
                                    partitions=["test" if i % 4 == 0 else "train" for i in range(24)])


@pytest.mark.parametrize("mode", ["remove", "tag"])
def test_metadata_filter_owner_mask_and_alignment(batch, mode):
    operator = MetadataFilter(column="quality", values_to_exclude=["bad"])
    expected = operator.fit(batch.x, batch.y).get_mask(batch.x, batch.y, metadata=batch.metadata)
    result, info = filter_batch(batch, operator, mode=mode)
    np.testing.assert_array_equal(info["mask"], expected)
    assert info["removed"] == 8
    if mode == "tag":
        assert result is batch
    else:
        for name in ("x", "y", "sample_ids", "origins", "partitions"):
            np.testing.assert_array_equal(getattr(result, name), getattr(batch, name)[expected])


def test_target_filter_real_owner(batch):
    operator = YOutlierFilter(method="iqr", threshold=1.5)
    result, info = filter_batch(batch, operator)
    expected = operator.fit(batch.x, batch.y).get_mask(batch.x, batch.y)
    np.testing.assert_array_equal(result.x, batch.x[expected])
    np.testing.assert_array_equal(info["mask"], expected)


def test_manual_selection_after_sampling_keeps_input_origins(batch):
    sampled = batch.take([4, 7, 12, 20])
    result, info = select_sample_indices(sampled, [1, 3])
    assert result.origins.tolist() == [7, 20]
    assert info["sample_origins"] == [4, 12]
    empty, _ = select_sample_indices(sampled, [])
    assert empty.x.shape == (0, 15)


@pytest.mark.parametrize("order", ["before", "after"])
def test_preprocessing_and_cv_both_orders_keep_heldout_outside_folds(batch, order):
    processed = batch
    if order == "after":
        processed = transform_batch(transform_batch(batch, StandardNormalVariate()), SavitzkyGolay(window_length=5))
    actual, messages = split_batch(processed, KFold(n_splits=3, shuffle=True, random_state=8))
    if order == "before":
        processed = transform_batch(transform_batch(batch, StandardNormalVariate()), SavitzkyGolay(window_length=5))
    train_indices = np.flatnonzero(batch.partitions == "train")
    expected = list(KFold(n_splits=3, shuffle=True, random_state=8).split(batch.x[train_indices]))
    for item, (train, val) in zip(actual["folds"], expected, strict=True):
        assert item["train_indices"] == train_indices[train].tolist()
        assert item["test_indices"] == train_indices[val].tolist()
        assert item["test_origins"] == train_indices[val].tolist()
    assert messages == []
    assert all(actual["fold_labels"][index] == -1 for index in np.flatnonzero(batch.partitions == "test"))
    np.testing.assert_array_equal(processed.origins, batch.origins)


def test_train_only_split_after_filter_and_augmentation_uses_row_partitions(batch):
    selected, _ = select_sample_indices(batch, [0, 1, 2, 4, 6, 7, 9, 10])
    augmented, _ = augment_batch(selected, StandardNormalVariate(), copies=1)
    result, _ = split_batch(augmented, KFold(n_splits=2))
    for fold in result["folds"]:
        for key in ("train_indices", "test_indices"):
            assert all(augmented.partitions[index] == "train" for index in fold[key])


def test_grouped_owner_and_repetition_constraints(batch):
    result, messages = split_batch(batch, GroupKFold(n_splits=3), repetition="subject")
    assert result["effective_group_mode"] == "repetition_only"
    assert any("repetition" in message for message in messages)
    for fold in result["folds"]:
        train_groups = set(batch.metadata["subject"][fold["train_indices"]])
        test_groups = set(batch.metadata["subject"][fold["test_indices"]])
        assert not train_groups & test_groups


def test_non_native_grouped_wrapper_is_used(batch):
    result, _ = split_batch(batch, KFold(n_splits=3), group_by="subject")
    assert result["effective_group_mode"] == "group_by_only"
    for fold in result["folds"]:
        assert not set(batch.metadata["subject"][fold["train_indices"]]) & set(batch.metadata["subject"][fold["test_indices"]])


def test_first_test_split_uses_requested_partition_and_no_synthetic_targets(batch):
    unlabeled = PreviewBatch.from_arrays(batch.x)
    result, _ = split_batch(unlabeled, ShuffleSplit(n_splits=2, test_size=0.25, random_state=42), kind="test_split", split_index=1)
    assert result["kind"] == "test_split"
    assert result["folds"][0]["test_count"] == 6
    assert "y_train_stats" not in result["folds"][0]
    assert set(result["fold_labels"]) == {-1, 1}


def test_stratified_preview_preserves_owner_quantile_binning():
    from nirs4all.data.binning import BinningCalculator

    batch = PreviewBatch.from_arrays(np.arange(240).reshape(60, 4), y=np.linspace(0.1, 99, 60))
    result, _ = split_batch(batch, StratifiedKFold(n_splits=3))
    bins, _ = BinningCalculator.bin_continuous_targets(batch.y.astype(float), bins=5, strategy="quantile")
    expected = list(StratifiedKFold(n_splits=3).split(batch.x, bins))
    for item, (train, test) in zip(result["folds"], expected, strict=True):
        assert item["train_indices"] == train.tolist()
        assert item["test_indices"] == test.tolist()


def test_invalid_selection_and_split_contracts(batch):
    with pytest.raises(ValueError, match="effective group"):
        split_batch(batch, GroupKFold(n_splits=3))
    with pytest.raises(ValueError, match="split_index"):
        split_batch(batch, KFold(n_splits=3), split_index=9)
    with pytest.raises(ValueError, match="Unknown split"):
        split_batch(batch, KFold(), kind="typo")
    for indices in ([True], [1.5], [-1], [999]):
        with pytest.raises(ValueError):
            select_sample_indices(batch, indices)


@pytest.mark.parametrize("train,test", [([0, 0], [1]), ([0], [-1]), ([0.5], [1]), ([0], [0])])
def test_splitter_indices_are_never_truncated_or_silently_duplicated(batch, train, test):
    class InvalidSplitter:
        def split(self, X, y=None):
            yield train, test

    with pytest.raises(ValueError, match="Splitter returned"):
        split_batch(batch, InvalidSplitter())
