"""Unit tests for nirs4all.data.selection.sampling."""

import numpy as np
import pytest

from nirs4all.data.selection.sampling import (
    kmeans_sample,
    random_sample,
    stratified_sample,
)


def _valid(indices: np.ndarray, n_total: int, n_select: int) -> None:
    assert indices.dtype == np.intp
    assert indices.size == n_select
    assert np.array_equal(indices, np.sort(indices))  # sorted
    assert indices.size == np.unique(indices).size  # distinct
    assert indices.min(initial=0) >= 0
    assert indices.max(initial=n_total - 1) < n_total


# --- random_sample ---------------------------------------------------------

def test_random_sample_basic():
    idx = random_sample(100, 10, seed=0)
    _valid(idx, 100, 10)


def test_random_sample_is_deterministic_with_seed():
    assert np.array_equal(random_sample(100, 10, seed=7), random_sample(100, 10, seed=7))


def test_random_sample_returns_all_when_oversized():
    assert np.array_equal(random_sample(5, 9, seed=0), np.arange(5))


def test_random_sample_empty_for_nonpositive():
    assert random_sample(5, 0).size == 0
    assert random_sample(5, -3).size == 0


# --- stratified_sample -----------------------------------------------------

def test_stratified_sample_count_and_validity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = rng.normal(size=200)
    idx = stratified_sample(X, y, 20, seed=1)
    _valid(idx, 200, 20)


def test_stratified_sample_spreads_across_target():
    # Monotonic y → the subset should span low and high ends, not cluster.
    y = np.arange(300, dtype=float)
    X = np.zeros((300, 3))
    idx = stratified_sample(X, y, 30, seed=2)
    selected_y = y[idx]
    assert selected_y.min() < 30  # represents the bottom decile
    assert selected_y.max() > 270  # represents the top decile


def test_stratified_sample_edges():
    y = np.arange(10, dtype=float)
    X = np.zeros((10, 2))
    assert np.array_equal(stratified_sample(X, y, 99, seed=0), np.arange(10))
    assert stratified_sample(X, y, 0, seed=0).size == 0


# --- kmeans_sample ---------------------------------------------------------

def test_kmeans_sample_count_and_validity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6))
    idx = kmeans_sample(X, 12, seed=3)
    _valid(idx, 150, 12)


def test_kmeans_sample_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5))
    assert np.array_equal(kmeans_sample(X, 10, seed=5), kmeans_sample(X, 10, seed=5))


def test_kmeans_sample_tops_up_with_duplicate_rows():
    # Many identical rows force empty/degenerate clusters; top-up must still
    # return exactly n_select distinct indices.
    X = np.vstack([np.zeros((40, 4)), np.ones((10, 4))])
    idx = kmeans_sample(X, 15, seed=0)
    _valid(idx, 50, 15)


@pytest.mark.parametrize("fn_args", [("random",), ("stratified",), ("kmeans",)])
def test_all_strategies_return_all_when_oversized(fn_args):
    n = 8
    if fn_args[0] == "random":
        out = random_sample(n, 100, seed=0)
    elif fn_args[0] == "stratified":
        out = stratified_sample(np.zeros((n, 2)), np.arange(n, dtype=float), 100, seed=0)
    else:
        out = kmeans_sample(np.random.default_rng(0).normal(size=(n, 3)), 100, seed=0)
    assert np.array_equal(out, np.arange(n))
