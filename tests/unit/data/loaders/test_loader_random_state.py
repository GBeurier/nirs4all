"""Tests for loader random-state behavior."""

from __future__ import annotations

import numpy as np

from nirs4all.data.loaders.loader import create_synthetic_dataset


def test_create_synthetic_dataset_does_not_mutate_global_numpy_rng():
    """create_synthetic_dataset should use local RNG, not np.random.seed()."""
    X = np.arange(200, dtype=float).reshape(100, 2)
    y = np.arange(100, dtype=float)

    np.random.seed(12345)
    expected_next = np.random.random()

    np.random.seed(12345)
    _ = create_synthetic_dataset(
        {
            "X": X,
            "y": y,
            "train": 0.8,
            "random_state": 42,
        }
    )
    observed_next = np.random.random()

    assert observed_next == expected_next
