"""Tests for PipelineExecutor step-cacheability decisions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.execution.step_cache import StepCache
from nirs4all.pipeline.steps.parser import StepParser


class _DummyController:
    def __init__(self, cacheable: bool):
        self._cacheable = cacheable

    def supports_step_cache(self) -> bool:
        return self._cacheable

class _DummyRouter:
    """Minimal router for unit-testing _is_step_cacheable decisions."""

    _CACHEABLE_KEYWORDS = {"preprocessing", "transform", "feature_selection"}

    def route(self, parsed_step, step=None):  # noqa: D401 - test stub
        return _DummyController(parsed_step.keyword in self._CACHEABLE_KEYWORDS)

def _make_executor() -> PipelineExecutor:
    step_runner = SimpleNamespace(
        parser=StepParser(),
        router=_DummyRouter(),
    )
    return PipelineExecutor(step_runner=step_runner)  # type: ignore[arg-type]

def test_subpipeline_all_transform_steps_is_cacheable():
    executor = _make_executor()
    step = [
        {"preprocessing": None},
        {"preprocessing": None},
    ]
    assert executor._is_step_cacheable(step) is True

def test_subpipeline_with_non_cacheable_step_is_not_cacheable():
    executor = _make_executor()
    step = [
        {"preprocessing": None},
        {"model": None},
    ]
    assert executor._is_step_cacheable(step) is False

def test_subpipeline_with_only_skips_is_not_cacheable():
    executor = _make_executor()
    step = [None, None]
    assert executor._is_step_cacheable(step) is False

def test_nested_subpipeline_all_transform_steps_is_cacheable():
    executor = _make_executor()
    step = [
        [{"preprocessing": None}],
        {"preprocessing": None},
    ]
    assert executor._is_step_cacheable(step) is True

def test_nested_subpipeline_with_model_is_not_cacheable():
    executor = _make_executor()
    step = [
        [{"preprocessing": None}],
        [{"model": None}],
    ]
    assert executor._is_step_cacheable(step) is False

def _make_dataset(name: str = "test_ds", n_samples: int = 12, n_features: int = 8) -> SpectroDataset:
    ds = SpectroDataset(name)
    X = np.random.rand(n_samples, n_features).astype(np.float64)
    y = np.random.rand(n_samples)
    ds.add_samples(X, indexes={"partition": "train"})
    ds.add_targets(y)
    return ds

def test_step_cache_data_hash_changes_when_exclusions_change():
    executor = _make_executor()
    ds = _make_dataset()
    h_before = executor._step_cache_data_hash(ds)

    ds._indexer.mark_excluded([0], reason="unit-test")
    h_after = executor._step_cache_data_hash(ds)

    assert h_before != h_after


@pytest.mark.parametrize("column,value", [("partition", "test"), ("group", 1), ("branch", 1)])
def test_index_mutations_cannot_reuse_a_preprocessing_snapshot(column: str, value: str | int):
    executor = _make_executor()
    ds = _make_dataset()
    feature_hash = ds.content_hash()
    before = executor._step_cache_data_hash(ds)
    assert before == executor._step_cache_data_hash(ds)
    cache = StepCache(max_size_mb=1)
    cache.put("preprocessing", before, ds)
    assert cache.get("preprocessing", before) is not None

    ds._indexer.update_by_indices([0], {column: value})

    assert ds.content_hash() == feature_hash
    after = executor._step_cache_data_hash(ds)
    assert after != before
    assert cache.get("preprocessing", after) is None


def test_tag_mutation_changes_cache_key_without_changing_features():
    executor = _make_executor()
    ds = _make_dataset()
    ds.add_tag("selected", "bool")
    before = executor._step_cache_data_hash(ds)
    feature_hash = ds.content_hash()

    ds.set_tag("selected", [0], True)

    assert ds.content_hash() == feature_hash
    assert executor._step_cache_data_hash(ds) != before


def test_index_row_order_changes_cache_key_without_changing_features():
    executor = _make_executor()
    ds = _make_dataset()
    before = executor._step_cache_data_hash(ds)
    feature_hash = ds.content_hash()

    ds._indexer._store._df = ds._indexer.df.reverse()

    assert ds.content_hash() == feature_hash
    assert executor._step_cache_data_hash(ds) != before


def test_unhashable_index_refuses_feature_only_cache_key(monkeypatch: pytest.MonkeyPatch):
    def broken_hash_rows(self, seed=0):
        raise ValueError("index hashing failed")

    monkeypatch.setattr(pl.DataFrame, "hash_rows", broken_hash_rows)
    with pytest.raises(RuntimeError, match="dataset index could not be hashed") as caught:
        _make_executor()._step_cache_data_hash(_make_dataset())
    assert isinstance(caught.value.__cause__, ValueError)
