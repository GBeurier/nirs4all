"""The ADR-17 engine selector is wired into the public run() entry point (migration seam).

These assert the public API gates the dag-ml backend before any execution — production stays on
the legacy engine by default, and an explicit dag-ml request fails loudly while the backend is
built out. No dataset is loaded (resolve_engine runs first), so they are fast and dag-ml-free.
"""

from __future__ import annotations

import pytest
from sklearn.cross_decomposition import PLSRegression

import nirs4all
from nirs4all.pipeline.engine import resolve_engine

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402


def test_resolve_engine_default_is_legacy() -> None:
    assert resolve_engine(None) == "legacy"
    assert resolve_engine("legacy") == "legacy"


def test_run_gates_dagml_engine_before_execution() -> None:
    with pytest.raises(NotImplementedError, match="dag-ml"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")


def test_run_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="bogus")
