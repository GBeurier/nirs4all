"""Regression tests for run() cache configuration defaults."""

from __future__ import annotations

import importlib

from nirs4all.config.cache_config import CacheConfig
from nirs4all.data.predictions import Predictions


class _DummyRunner:
    """Minimal PipelineRunner test double for run() unit tests."""

    instances: list["_DummyRunner"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cache_config = None
        _DummyRunner.instances.append(self)

    def run(self, pipeline, dataset, pipeline_name, refit):
        return Predictions(), {}


def test_run_sets_default_cache_config_when_cache_none(monkeypatch):
    """run(cache=None) should still initialize a default CacheConfig."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)

    run_module.run(
        pipeline=[{"model": "DummyModel"}],
        dataset="dummy_dataset",
        cache=None,
        verbose=0,
    )

    runner = _DummyRunner.instances[-1]
    assert isinstance(runner.cache_config, CacheConfig)
    assert runner.cache_config.step_cache_enabled is False
    assert runner.cache_config.use_cow_snapshots is True


def test_run_uses_explicit_cache_config(monkeypatch):
    """run(cache=...) should preserve the caller-provided CacheConfig."""
    run_module = importlib.import_module("nirs4all.api.run")

    _DummyRunner.instances.clear()
    monkeypatch.setattr(run_module, "PipelineRunner", _DummyRunner)
    explicit_cache = CacheConfig(step_cache_enabled=True, use_cow_snapshots=False)

    run_module.run(
        pipeline=[{"model": "DummyModel"}],
        dataset="dummy_dataset",
        cache=explicit_cache,
        verbose=0,
    )

    runner = _DummyRunner.instances[-1]
    assert runner.cache_config is explicit_cache
