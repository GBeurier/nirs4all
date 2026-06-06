"""Tests for SpectroDataset.describe()."""

from __future__ import annotations

import numpy as np

from nirs4all.data.dataset import SpectroDataset


def test_describe_with_targets():
    ds = SpectroDataset("t")
    ds.add_samples(np.random.default_rng(0).random((10, 50)))
    ds.add_targets(np.random.default_rng(1).random(10))
    info = ds.describe()
    assert info["num_samples"] == 10
    assert info["num_features"] == 50
    assert info["n_sources"] == 1
    assert info["task_type"] == "regression"
    assert info["has_targets"] is True
    assert info["num_targets"] == 1
    assert isinstance(info["signal_types"], list)
    assert isinstance(info["metadata_columns"], list)


def test_describe_without_targets():
    ds = SpectroDataset("empty")
    ds.add_samples(np.random.default_rng(0).random((4, 8)))
    info = ds.describe()
    assert info["task_type"] is None
    assert info["has_targets"] is False
    assert info["num_targets"] == 0


def test_describe_is_json_safe():
    import json

    ds = SpectroDataset("t")
    ds.add_samples(np.random.default_rng(0).random((5, 6)))
    json.dumps(ds.describe())  # must not raise
