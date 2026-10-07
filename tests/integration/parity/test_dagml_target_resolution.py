"""Target selection keeps identities and widths while avoiding a duplicate filter."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nirs4all.core.task_type import TaskType
from nirs4all.data._dataset.target_accessor import TargetAccessor
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.targets import Targets
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.resolver import MaterializationResolver


def _dataset(width: int = 1, cls: type[SpectroDataset] = SpectroDataset) -> SpectroDataset:
    dataset = cls("target-resolution")
    dataset.set_task_type(TaskType.REGRESSION)
    dataset.add_samples(np.arange(18, dtype=float).reshape(6, 3), {"partition": "train"})
    dataset.add_targets(np.arange(6 * width, dtype=float).reshape(6, width) + 10)
    return dataset


@pytest.mark.parametrize("width", [1, 2, 3])
def test_complete_base_targets_filter_once_and_restore_duplicates(width: int) -> None:
    dataset = _dataset(width)
    identity = mint_identity(dataset)
    requested = [identity.to_wire(index) for index in [4, 1, 4, 0]]
    resolver = MaterializationResolver(dataset, identity)
    with patch.object(dataset._indexer._store, "query", wraps=dataset._indexer._store.query) as query:
        result = resolver.resolve_targets(requested, target_id="measurement")
    assert query.call_count == 1
    expected = (np.arange(6 * width).reshape(6, width) + 10)[[4, 1, 4, 0]]
    assert result == {"target_id": "measurement", "sample_ids": requested, "values": expected.ravel().tolist() if width == 1 else expected.tolist()}


def test_resolution_reads_changed_targets_and_exclusion_each_call() -> None:
    dataset = _dataset(2)
    identity = mint_identity(dataset)
    requested = [identity.to_wire(index) for index in [3, 1]]
    resolver = MaterializationResolver(dataset, identity)
    assert resolver.resolve_targets(requested)["values"] == [[16, 17], [12, 13]]
    dataset._targets._data["numeric"] += 100
    assert resolver.resolve_targets(requested)["values"] == [[116, 117], [112, 113]]
    dataset._indexer.mark_excluded([1], reason="diagnostic")
    with pytest.raises(ValueError, match="cannot change target width"):
        resolver.resolve_targets(requested)
    assert resolver.resolve_targets(requested, include_excluded=True)["values"] == [[116, 117], [112, 113]]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_excluded_rows_cannot_be_reshaped_into_another_target_width(width: int) -> None:
    dataset = _dataset(width)
    identity = mint_identity(dataset)
    dataset._indexer.mark_excluded([1], reason="diagnostic")
    requested = [identity.to_wire(index) for index in [0, 1, 2]]
    with pytest.raises(ValueError, match="cannot change target width"):
        MaterializationResolver(dataset, identity).resolve_targets(requested)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_unlabeled_rows_cannot_change_target_width(width: int) -> None:
    dataset = _dataset(width)
    dataset.add_samples(np.ones((2, 3)), {"partition": "test"})
    identity = mint_identity(dataset)
    requested = [identity.to_wire(index) for index in [0, 1, 6]]
    with pytest.raises(ValueError, match="cannot change target width"):
        MaterializationResolver(dataset, identity).resolve_targets(requested)


def test_augmented_children_require_origin_ids_even_when_shape_would_fit() -> None:
    dataset = _dataset(3)
    dataset.add_samples_batch(np.ones((1, 1, 3)), [{"partition": "train", "origin": 0, "augmentation": "noise"}])
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    requested = [identity.to_wire(index) for index in [0, 1, 6]]
    with pytest.raises(ValueError, match="cannot change target width"):
        resolver.resolve_targets(requested, include_excluded=True)
    assert resolver.resolve_targets([identity.to_wire(0), identity.to_wire(0)])["values"] == [[10, 11, 12], [10, 11, 12]]


def test_subclass_target_access_is_preserved() -> None:
    class OffsetDataset(SpectroDataset):
        def y(self, selector, include_augmented=True, include_excluded=False):
            return super().y(selector, include_augmented, include_excluded) + 500

    dataset = _dataset(2, OffsetDataset)
    identity = mint_identity(dataset)
    requested = [identity.to_wire(4), identity.to_wire(1)]
    assert MaterializationResolver(dataset, identity).resolve_targets(requested)["values"] == [[518, 519], [512, 513]]


def test_unknown_ids_and_empty_requests_keep_refusals() -> None:
    dataset = _dataset()
    resolver = MaterializationResolver(dataset, mint_identity(dataset))
    with pytest.raises((KeyError, ValueError)):
        resolver.resolve_targets(["foreign-sample"])
    with pytest.raises(ValueError):
        resolver.resolve_targets([])


def test_classification_numeric_encoding_is_preserved() -> None:
    dataset = SpectroDataset("class-target-resolution")
    dataset.add_samples(np.arange(12, dtype=float).reshape(4, 3), {"partition": "train"})
    dataset.add_targets(np.array(["oak", "pine", "oak", "birch"]))
    encoded = dataset.y({}, include_augmented=False).ravel()
    identity = mint_identity(dataset)
    requested = [identity.to_wire(index) for index in [3, 0, 2, 3]]
    assert MaterializationResolver(dataset, identity).resolve_targets(requested)["values"] == encoded[[3, 0, 2, 3]].tolist()


def test_replaced_live_accessor_block_preserves_actual_target_values() -> None:
    dataset = _dataset(2)
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    replacement = Targets()
    replacement.set_task_type(TaskType.REGRESSION)
    replacement.add_targets(np.arange(12, dtype=float).reshape(6, 2) + 1000)
    dataset._target_accessor._block = replacement
    requested = [identity.to_wire(index) for index in [4, 1, 4]]
    assert dataset._targets is not replacement
    assert resolver.resolve_targets(requested)["values"] == [[1008, 1009], [1002, 1003], [1008, 1009]]


def test_same_alias_accessor_override_preserves_actual_target_values() -> None:
    class OffsetAccessor(TargetAccessor):
        def y(self, selector=None, include_augmented=True, include_excluded=False):
            return super().y(selector, include_augmented, include_excluded) + 500

    dataset = _dataset(2)
    identity = mint_identity(dataset)
    dataset._target_accessor = OffsetAccessor(dataset._indexer, dataset._targets)
    requested = [identity.to_wire(index) for index in [4, 1, 4]]
    assert MaterializationResolver(dataset, identity).resolve_targets(requested)["values"] == [[518, 519], [512, 513], [518, 519]]


def test_same_alias_accessor_refusal_is_preserved() -> None:
    class RefusingAccessor(TargetAccessor):
        def y(self, selector=None, include_augmented=True, include_excluded=False):
            raise ValueError("live accessor refusal")

    dataset = _dataset(2)
    identity = mint_identity(dataset)
    dataset._target_accessor = RefusingAccessor(dataset._indexer, dataset._targets)
    requested = [identity.to_wire(index) for index in [4, 1, 4]]
    with pytest.raises(ValueError, match="live accessor refusal"):
        MaterializationResolver(dataset, identity).resolve_targets(requested)
