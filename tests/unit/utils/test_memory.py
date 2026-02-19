"""Tests for nirs4all.utils.memory module."""

import numpy as np
import pytest

from nirs4all.utils.memory import (
    estimate_cache_entry_bytes,
    estimate_dataset_bytes,
    format_bytes,
    get_process_rss_mb,
)


class TestEstimateDatasetBytes:
    """Test estimate_dataset_bytes against manual nbytes sums."""

    def test_single_source_dataset(self):
        from nirs4all.data.dataset import SpectroDataset

        dataset = SpectroDataset("test")
        data = np.random.rand(100, 50).astype(np.float32)
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(np.random.rand(100).astype(np.float64))

        estimated = estimate_dataset_bytes(dataset)

        # Feature: 100 samples x 1 processing x 50 features x 4 bytes = 20000
        expected_features = 100 * 1 * 50 * 4
        # Targets: "raw" (float64, 100 * 8) + "numeric" (float64, 100 * 8)
        expected_targets = sum(
            arr.nbytes for arr in dataset._targets._data.values()
        )

        assert estimated == expected_features + expected_targets

    def test_multi_source_dataset(self):
        from nirs4all.data.dataset import SpectroDataset

        dataset = SpectroDataset("test")
        data = [
            np.random.rand(50, 30).astype(np.float32),
            np.random.rand(50, 20).astype(np.float32),
        ]
        dataset.add_samples(data, {"partition": "train"})

        estimated = estimate_dataset_bytes(dataset)

        expected = (50 * 1 * 30 * 4) + (50 * 1 * 20 * 4)
        assert estimated == expected

    def test_empty_dataset(self):
        from nirs4all.data.dataset import SpectroDataset

        dataset = SpectroDataset("empty")
        assert estimate_dataset_bytes(dataset) == 0

class TestEstimateCacheEntryBytes:
    """Test estimate_cache_entry_bytes for various types."""

    def test_numpy_array(self):
        arr = np.zeros((100, 50), dtype=np.float32)
        assert estimate_cache_entry_bytes(arr) == arr.nbytes

    def test_tuple_of_arrays(self):
        a = np.zeros((100, 50), dtype=np.float32)
        b = np.zeros((100,), dtype=np.float64)
        entry = (a, b)

        assert estimate_cache_entry_bytes(entry) == a.nbytes + b.nbytes

    def test_dataset(self):
        from nirs4all.data.dataset import SpectroDataset

        dataset = SpectroDataset("test")
        data = np.random.rand(10, 20).astype(np.float32)
        dataset.add_samples(data, {"partition": "train"})

        result = estimate_cache_entry_bytes(dataset)
        assert result == estimate_dataset_bytes(dataset)

    def test_plain_object_uses_getsizeof(self):
        result = estimate_cache_entry_bytes("hello world")
        assert result > 0

    def test_nested_list(self):
        a = np.zeros((10, 5), dtype=np.float32)
        entry = [a, [a, a]]
        expected = a.nbytes + a.nbytes + a.nbytes
        assert estimate_cache_entry_bytes(entry) == expected

class TestGetProcessRssMb:
    """Test get_process_rss_mb returns a positive value."""

    def test_returns_positive(self):
        rss = get_process_rss_mb()
        assert rss > 0

    def test_returns_float(self):
        rss = get_process_rss_mb()
        assert isinstance(rss, float)

class TestFormatBytes:
    """Test format_bytes for various scales."""

    def test_bytes(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"

    def test_kilobytes(self):
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_bytes(1024 ** 2) == "1.0 MB"
        assert format_bytes(int(152.6 * 1024 ** 2)) == "152.6 MB"

    def test_gigabytes(self):
        assert format_bytes(1024 ** 3) == "1.0 GB"
        assert format_bytes(int(2.5 * 1024 ** 3)) == "2.5 GB"

    def test_negative(self):
        assert format_bytes(-1024) == "-1.0 KB"
