"""Isolated real-runtime gate for the DAG-ML shadow default.

This module is intentionally run in its own pytest process in CI, where the
fallback meter naturally starts at zero.  The test still snapshots that
process-local meter so it remains valid when run after selector tests that
deliberately exercise an explicit legacy rollback.
"""

from __future__ import annotations

import importlib
import warnings

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.engine import LegacyEngineUsageWarning, legacy_fallback_metrics, resolve_engine
from nirs4all.pipeline.runner import PipelineRunner

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]


def _forbid_legacy_runner(*_args: object, **_kwargs: object) -> None:
    """Fail immediately if a shadow-default run reaches legacy orchestration."""

    raise AssertionError("DAG-ML shadow default constructed a legacy PipelineRunner")


def test_dagml_shadow_default_never_constructs_legacy_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one supported public shape through a simulated DAG-ML default.

    The public call deliberately supplies neither ``engine`` nor
    ``allow_legacy_fallback``.  This proves the candidate default and its
    ordinary fail-closed policy rather than an explicitly selected fast path.
    """

    monkeypatch.delenv("N4A_ENGINE", raising=False)
    engine_module = importlib.import_module("nirs4all.pipeline.engine")
    monkeypatch.setattr(engine_module, "DEFAULT_ENGINE", "dag-ml")

    assert resolve_engine() == "dag-ml"
    fallback_metrics_before = legacy_fallback_metrics()

    # Patch the class implementation rather than one imported alias: no path
    # may instantiate PipelineRunner while the DAG-ML candidate is selected.
    monkeypatch.setattr(PipelineRunner, "__init__", _forbid_legacy_runner)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyEngineUsageWarning)
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            verbose=0,
            save_artifacts=False,
            save_charts=False,
        )

    assert isinstance(result, RunResult)
    assert result._is_dagml_engine()  # noqa: SLF001
    assert legacy_fallback_metrics() == fallback_metrics_before
