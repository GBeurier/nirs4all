"""The ADR-17 engine selector is wired into the public run() entry point (migration seam).

These assert the public API resolves the engine before execution — production stays on the legacy
engine by default, an explicit ``engine="dag-ml"`` dispatches to the operational dag-ml backend,
and an unknown engine is rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.cross_decomposition import PLSRegression

import nirs4all
from nirs4all.pipeline.engine import resolve_engine

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"


def test_resolve_engine_default_is_legacy() -> None:
    assert resolve_engine(None) == "legacy"
    assert resolve_engine("legacy") == "legacy"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dispatches_to_dagml_engine() -> None:
    """`engine="dag-ml"` resolves to the operational backend (no longer gated). With no splitter the
    dag-ml path fails loudly for the right reason via the catchable DagMlUnsupported(NotImplementedError) —
    proving it dispatched and hit the real no-splitter check (the "cross-validator" message), not a generic
    gate. The catchable type is what the fallback cutover relies on. Needs the CLI binary: run_via_dagml
    checks for it before the splitter."""
    with pytest.raises(NotImplementedError, match="cross-validator"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")


def test_run_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="bogus")
