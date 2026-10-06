"""DT1-19: failed late appends preserve every source and the indexer."""

import numpy as np
import pytest

from nirs4all.data.dataset import SpectroDataset


def _snapshot(dataset):
    return ([a.copy() for a in dataset.x({}, layout="3d", concat_source=False)], dataset._indexer.df.clone(),
            [dataset.headers(i)[:] for i in range(2)], [dataset.header_unit(i) for i in range(2)])


@pytest.mark.parametrize("failure", ["second_dtype", "second_header_unit", "empty_headers", "second_dimensions"])
@pytest.mark.parametrize("shared", [False, True, "restored"])
def test_late_raw_append_failure_is_transactional(failure, shared):
    dataset = SpectroDataset()
    dataset.add_samples([np.arange(12).reshape(4, 3), np.arange(8).reshape(4, 2)], headers=[["1000", "1100", "1200"], ["4000", "4100"]], header_unit=["nm", "cm-1"])
    snapshots = []
    if shared:
        for source in dataset._features.sources:
            snapshots.append(source._storage.ensure_shared().acquire())
            if shared == "restored":
                source._storage.restore_from_shared(snapshots[-1].acquire())
    before = _snapshot(dataset)
    arrays = [np.ones((2, 3)), np.ones((2, 2))]
    headers = [["1001", "1101", "1201"], ["4001", "4101"]]
    units = ["nm", "cm-1"]
    if failure == "second_dtype":
        arrays[1] = np.full((2, 2), "not numeric")
    elif failure == "second_header_unit":
        units[1] = "invalid unit"
    elif failure == "empty_headers":
        headers = []
    else:
        arrays[1] = np.ones((2, 1, 2))
    with pytest.raises((ValueError, IndexError)):
        dataset.add_samples(arrays, headers=headers, header_unit=units)
    after = _snapshot(dataset)
    assert after[1].equals(before[1])
    assert after[2:] == before[2:]
    for old, new in zip(before[0], after[0], strict=True):
        np.testing.assert_array_equal(old, new)
    for snapshot in snapshots:
        assert snapshot.refcount == 2
        snapshot.release()
    # A corrected retry appends exactly once, without phantom rows.
    dataset.add_samples([np.ones((2, 3)), np.ones((2, 2))])
    assert dataset.num_samples == dataset._indexer.df.height == 6
    assert [source.num_samples for source in dataset._features.sources] == [6, 6]


@pytest.mark.parametrize("failure", ["second_dtype", "second_processing_count", "second_row_count"])
def test_late_batch_append_failure_is_transactional(failure):
    dataset = SpectroDataset()
    dataset.add_samples([np.ones((4, 3)), np.ones((4, 2))], headers=[["1", "2", "3"], ["4", "5"]])
    before = _snapshot(dataset)
    arrays = [np.ones((2, 1, 3)), np.ones((2, 1, 2))]
    if failure == "second_dtype":
        arrays[1] = np.full((2, 1, 2), "bad dtype")
    elif failure == "second_processing_count":
        arrays[1] = np.ones((2, 2, 2))
    else:
        arrays[1] = np.ones((1, 1, 2))
    with pytest.raises(ValueError):
        dataset.add_samples_batch(arrays, [{"origin": 0}, {"origin": 1}])
    after = _snapshot(dataset)
    assert after[1].equals(before[1])
    assert after[2:] == before[2:]
    for old, new in zip(before[0], after[0], strict=True):
        np.testing.assert_array_equal(old, new)


def test_failed_first_append_leaves_empty_dataset():
    dataset = SpectroDataset()
    with pytest.raises(ValueError):
        dataset.add_samples([np.ones((2, 3)), np.full((2, 2), "bad")])
    assert dataset.n_sources == dataset.num_samples == dataset._indexer.df.height == 0
