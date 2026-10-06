"""Metadata separation preserves physical row identities after exclusion."""

from unittest.mock import Mock

import numpy as np
import pytest

from nirs4all.controllers.data.branch import BranchController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext

GROUPS = np.array(["A", "B", "A", "C", "B", "C", "D", "D"])


def metadata_dataset(excluded, augmented):
    dataset = SpectroDataset("metadata-exclusion-alignment")
    dataset.add_samples(np.arange(24, dtype=float).reshape(8, 3), {"partition": ["train"] * 6 + ["test"] * 2})
    dataset.add_targets(np.arange(8, dtype=float))
    dataset.add_metadata(GROUPS.reshape(-1, 1), headers=["group"])
    if augmented:
        dataset.augment_samples(np.ones((2, 3)), ["raw"], "audit", {}, count=[1, 1, 0, 0, 0, 0, 0, 0])
    if excluded:
        dataset._indexer.update_by_indices(excluded, {"excluded": True})
    return dataset


@pytest.mark.parametrize("excluded", [[], [0], [1], [5], [7], [1, 3]])
@pytest.mark.parametrize("augmented", [False, True])
@pytest.mark.parametrize("mode,subset", [("train", None), ("predict", None), ("train", [1, 2, 4, 6, 7])])
def test_metadata_groups_and_universe_keep_physical_ids(excluded, augmented, mode, subset, monkeypatch):
    dataset = metadata_dataset(excluded, augmented)
    context = ExecutionContext(selector=DataSelector(partition=None, include_augmented=True))
    if subset is not None:
        context.selector["sample"] = subset
    controller = BranchController()
    execute = Mock(return_value=(context, None))
    monkeypatch.setattr(controller, "_execute_separation_branches", execute)

    controller._execute_by_metadata({"by_metadata": "group", "steps": []}, dataset, context, Mock(), mode=mode)

    # Expected membership comes from the original labelled rows, independently
    # of the accessor's filtered positions or the controller's group output.
    universe = [sample for sample in range(8) if sample not in excluded and (subset is None or sample in subset)]
    selected = [sample for sample in universe if mode != "train" or sample < 6]
    names = sorted(set(GROUPS[selected]))
    captured = execute.call_args.kwargs
    np.testing.assert_array_equal(captured["sample_indices"], selected)
    actual_train = {name: [int(captured["sample_indices"][position]) for position in positions] for name, positions in captured["groups"].items()}
    assert actual_train == {name: [sample for sample in selected if GROUPS[sample] == name] for name in names}
    assert captured["universe_groups"] == {name: [sample for sample in universe if GROUPS[sample] == name] for name in names}
    np.testing.assert_array_equal(dataset._indexer.x_indices(None, include_augmented=False, include_excluded=True), np.arange(8))


@pytest.mark.parametrize("explicit_mapping", [False, True])
def test_metadata_min_samples_filters_branches_without_reindexing_rows(explicit_mapping, monkeypatch):
    dataset = metadata_dataset([1], True)
    context = ExecutionContext(selector=DataSelector(partition=None))
    controller = BranchController()
    execute = Mock(return_value=(context, None))
    monkeypatch.setattr(controller, "_execute_separation_branches", execute)
    definition = {"by_metadata": "group", "steps": [], "min_samples": 2}
    if explicit_mapping:
        definition["values"] = {"AC": ["A", "C"], "BD": ["B", "D"]}

    controller._execute_by_metadata(definition, dataset, context, Mock())

    captured = execute.call_args.kwargs
    selected = captured["sample_indices"]
    actual_train = {name: selected[positions].tolist() for name, positions in captured["groups"].items()}
    expected = {"AC": [0, 2, 3, 5]} if explicit_mapping else {"A": [0, 2], "C": [3, 5]}
    assert actual_train == expected
    assert captured["universe_groups"] == expected


def test_metadata_branch_missing_column_lists_available_columns():
    dataset = metadata_dataset([1], False)
    with pytest.raises(ValueError, match=r"Metadata column 'unknown' not found.*Available columns: \['group'\]"):
        BranchController()._execute_by_metadata({"by_metadata": "unknown"}, dataset, ExecutionContext(selector=DataSelector(partition=None)), Mock())


def test_metadata_branch_without_training_samples_is_empty(monkeypatch):
    dataset = metadata_dataset(list(range(6)), True)
    controller = BranchController()
    execute = Mock()
    monkeypatch.setattr(controller, "_execute_separation_branches", execute)
    context = ExecutionContext(selector=DataSelector(partition=None))
    returned_context, output = controller._execute_by_metadata({"by_metadata": "group"}, dataset, context, Mock())
    assert returned_context is context
    assert output.artifacts == []
    execute.assert_not_called()
