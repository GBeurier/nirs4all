"""Batch origin lookup must preserve matrix alignment and fold isolation."""

from unittest.mock import patch

import numpy as np
import pytest

from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.indexer import Indexer
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, StepMetadata


def test_batch_origins_preserve_order_duplicates_missing_and_observe_additions():
    indexer = Indexer()
    assert indexer.get_origins_for_samples([]) == []
    indexer.add_samples(3, partition="train")
    indexer.add_samples(2, partition="train", origin_indices=[1, 0], augmentation="noise")
    requested = [4, 0, 3, 4, 2, 99]
    expected = [indexer.get_origin_for_sample(sample) for sample in requested]
    with patch.object(indexer._store, "query", side_effect=AssertionError("per-row query")):
        assert indexer.get_origins_for_samples(requested) == expected == [0, 0, 1, 0, 2, None]
    indexer.add_samples(1, partition="test")
    assert indexer.get_origins_for_samples([5, 4]) == [5, 0]


def augmented_dataset():
    dataset = SpectroDataset("batch-origin-order")
    dataset.add_samples(np.arange(18, dtype=float).reshape(6, 3), {"partition": "train"})
    dataset.add_targets(np.arange(6, dtype=float) + 0.25)
    dataset.augment_samples(
        data=np.arange(9, dtype=float).reshape(3, 3),
        processings=["raw"], augmentation_id="noise",
        selector={"partition": "train"}, count=[1, 1, 1, 0, 0, 0],
    )
    return dataset


def test_target_batch_follows_x_order_with_interleaved_base_and_augmented_rows():
    dataset = augmented_dataset()
    indexer = dataset._indexer
    # Interleave physical index rows without changing sample IDs/data storage.
    # X is still base-first, so directly selecting origin values is incorrect.
    indexer._store._df = indexer.df[[6, 0, 7, 1, 8, 2, 3, 4, 5]]
    with patch.object(indexer, "get_origin_for_sample", side_effect=AssertionError("per-row query")):
        with patch.object(indexer, "get_origins_for_samples", wraps=indexer.get_origins_for_samples) as batch:
            np.testing.assert_array_equal(
                dataset.y({"partition": "train"}).ravel(),
                [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 0.25, 1.25, 2.25],
            )
            assert batch.call_count == 1


@pytest.mark.parametrize("exclude", [False, True])
def test_fold_batch_keeps_children_only_with_training_origins(exclude):
    dataset = augmented_dataset()
    indexer = dataset._indexer
    dataset.set_folds([([1, 3, 5], [0, 2, 4]), ([0, 2, 4], [1, 3, 5])])
    context = ExecutionContext(
        selector=DataSelector(partition="train"), state=PipelineState(), metadata=StepMetadata(),
    )
    if exclude:
        context.custom["outlier_exclusion"] = {"mask": [False], "sample_indices": [0]}
    active = indexer.x_indices({"partition": "train"}, include_augmented=True)
    with patch.object(indexer, "get_origin_for_sample", side_effect=AssertionError("per-row query")):
        with patch.object(indexer, "get_origins_for_samples", wraps=indexer.get_origins_for_samples) as batch:
            folds = SklearnModelController()._remap_folds_to_positions(dataset, context, mode="train")
            assert batch.call_count == 1  # Independent of number of samples/folds.
    expected_second = [2, 4, 8] if exclude else [0, 2, 4, 6, 8]
    assert [(active[train].tolist(), active[val].tolist()) for train, val in folds] == [
        ([1, 3, 5, 7], [0, 2, 4]), (expected_second, [1, 3, 5]),
    ]
    for train, val in folds:
        train_origins = set(indexer.get_origins_for_samples(active[train].tolist()))
        val_origins = set(indexer.get_origins_for_samples(active[val].tolist()))
        assert train_origins.isdisjoint(val_origins)
