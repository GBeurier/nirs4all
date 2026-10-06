"""Repetition groups retain physical row identities after exclusion and augmentation."""

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml.folds import _build_group_folds, _repetition_grain, _repetition_groups_for_pool

GROUPS = np.array(["A", "B", "A", "C", "B", "C", "D", "D"])


def repetition_dataset(excluded, augmented):
    dataset = SpectroDataset("repetition-alignment")
    dataset.add_samples(np.arange(24, dtype=float).reshape(8, 3), {"partition": ["train"] * 6 + ["test"] * 2})
    dataset.add_targets(np.arange(8, dtype=float))
    dataset.add_metadata(GROUPS.reshape(-1, 1), headers=["sample_id"])
    dataset.set_repetition("sample_id")
    if augmented:
        dataset.augment_samples(np.ones((2, 3)), ["raw"], "audit", {}, count=[1, 1, 0, 0, 0, 0, 0, 0])
    if excluded:
        dataset._indexer.update_by_indices(excluded, {"excluded": True})
    return dataset


@pytest.mark.parametrize("excluded", [[], [0], [1], [7], [1, 3]])
@pytest.mark.parametrize("augmented", [False, True])
def test_repetition_groups_follow_physical_pool_order(excluded, augmented):
    dataset = repetition_dataset(excluded, augmented)
    pool = [sample for sample in range(7, -1, -1) if sample not in excluded]
    np.testing.assert_array_equal(_repetition_groups_for_pool(dataset, pool), GROUPS[pool])
    assert _repetition_grain(dataset, pool) == {sample: GROUPS[sample] for sample in pool}


def test_repetition_groups_select_only_the_requested_subset():
    dataset = repetition_dataset([1, 3], True)
    pool = [5, 2, 4, 0]
    np.testing.assert_array_equal(_repetition_groups_for_pool(dataset, pool), ["C", "A", "B", "A"])
    assert _repetition_groups_for_pool(dataset, []).size == 0


@pytest.mark.parametrize("splitter", [GroupKFold(n_splits=2), KFold(n_splits=2)])
def test_group_folds_keep_noncontiguous_ids_and_repetition_members_together(splitter):
    dataset = repetition_dataset([1, 3], True)
    pool = [7, 5, 4, 2, 0, 6]
    folds = _build_group_folds(splitter, dataset, pool)
    validated = []
    for train, validation in folds:
        assert set(train) | set(validation) == set(pool)
        assert not set(train) & set(validation)
        assert not set(GROUPS[train]) & set(GROUPS[validation])
        validated.extend(validation)
    assert sorted(validated) == sorted(pool)
