"""Regression tests for bin boundaries and bounded cache lifecycle."""

import numpy as np
import pytest

from nirs4all.data.binning import BinningCalculator
from nirs4all.data.performance.cache import DataCache


@pytest.mark.parametrize("strategy", ["quantile", "equal_width"])
@pytest.mark.parametrize("bins", [2, 5, 10])
def test_binning_uses_exactly_requested_number_of_intervals(strategy, bins):
    values = np.linspace(0, 1, 100)
    labels, edges = BinningCalculator.bin_continuous_targets(values, bins, strategy)
    assert edges.shape == (bins + 1,)
    assert np.unique(labels).tolist() == list(range(bins))
    assert labels[0] == 0
    assert labels[-1] == bins - 1
    assert np.bincount(labels).min() >= 9  # Minimum is part of the first interval, never a singleton.


@pytest.mark.parametrize("strategy", ["quantile", "equal_width"])
def test_binning_boundaries_and_constant_values(strategy):
    labels, _ = BinningCalculator.bin_continuous_targets(np.array([0., .2, .4, .6, .8, 1.]), 5, strategy)
    assert labels.min() == 0
    assert labels.max() == 4
    assert labels[0] == labels[1]
    labels, _ = BinningCalculator.bin_continuous_targets(np.ones(20), 5, strategy)
    assert np.unique(labels).tolist() == [0]


@pytest.mark.parametrize("max_entries", [0, -1])
def test_cache_refuses_nonpositive_capacity_without_hanging(max_entries):
    with pytest.raises(ValueError, match="max_entries must be positive"):
        DataCache(max_entries=max_entries)


def test_cache_clear_releases_each_entry_once_and_resets_size():
    released = []
    cache = DataCache(max_entries=2, on_evict=released.append)
    a = np.arange(3)
    b = np.arange(5)
    cache.set("a", a)
    cache.set("b", b)
    assert cache.total_size > 0
    cache.clear()
    assert len(released) == 2
    assert released[0] is a and released[1] is b
    assert cache.total_size == 0
    assert cache.stats()["entries"] == 0
    cache.clear()
    assert len(released) == 2
    cache.set("new", np.ones(2))
    assert cache.stats()["entries"] == 1
