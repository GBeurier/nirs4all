"""Cooperative cancellation via the should_stop hook."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.api.run import run
from nirs4all.pipeline.execution.orchestrator import RunCancelledError


@pytest.fixture()
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 50))
    y = X[:, :5].sum(axis=1)
    return X, y


def _run(pipeline, xy, **kwargs):
    X, y = xy
    engine = kwargs.pop("engine", "legacy")
    return run(
        pipeline=pipeline,
        dataset=(X, y),
        engine=engine,
        verbose=0,
        save_artifacts=False,
        save_charts=False,
        plots_visible=False,
        **kwargs,
    )


def test_immediate_cancel_raises(xy, tmp_path):
    with pytest.raises(RunCancelledError):
        _run([KFold(n_splits=3), PLSRegression(5)], xy,
             workspace_path=tmp_path, should_stop=lambda: True)


def test_cancel_between_variants(xy, tmp_path):
    calls: dict[str, int] = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(RunCancelledError):
        _run([KFold(n_splits=3), {"_or_": [PLSRegression(3), PLSRegression(5)]}], xy,
             workspace_path=tmp_path, should_stop=stop_after_two)
    # dataset boundary + first variant ran; second variant check fired the stop
    assert calls["n"] >= 3


def test_no_hook_runs_to_completion(xy, tmp_path):
    result = _run([KFold(n_splits=3), PLSRegression(5)], xy, workspace_path=tmp_path)
    assert result is not None
