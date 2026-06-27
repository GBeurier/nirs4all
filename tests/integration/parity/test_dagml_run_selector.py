"""The ADR-17 engine selector is wired into the public run() entry point (migration seam).

These assert the public API resolves the engine before execution — production stays on the legacy
engine by default, an explicit ``engine="dag-ml"`` dispatches to the operational dag-ml backend, and
an unknown engine is rejected. They also pre-certify the cutover FALLBACK: a supported shape runs
NATIVE on dag-ml, while a catchable unsupported shape falls back to legacy (warning + valid result)
instead of raising — the exact "try dag-ml; on a catchable unsupported-error → legacy" flip path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.engine import resolve_engine

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"

_FALLBACK_WARNING = "falling back to the legacy engine"


def test_resolve_engine_default_is_legacy() -> None:
    assert resolve_engine(None) == "legacy"
    assert resolve_engine("legacy") == "legacy"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dispatches_to_dagml_engine_native() -> None:
    """`engine="dag-ml"` runs a SUPPORTED shape (transforms + KFold + one model) NATIVELY on the
    dag-ml backend — proving the public API dispatched to it and produced a real RunResult, with NO
    cutover fallback to legacy (no warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [SNV(), KFold(n_splits=3), {"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert not any(_FALLBACK_WARNING in str(w.message) for w in caught)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_dagml_falls_back_to_legacy_on_unsupported_shape() -> None:
    """`engine="dag-ml"` with a no-splitter shape is a catchable coverage-boundary case
    (DagMlUnsupported, a NotImplementedError subclass): with the cutover fallback wired it must NOT
    raise — it warns and re-runs on the legacy engine, returning a valid legacy RunResult."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(
            [{"model": PLSRegression(n_components=2)}],
            dataset_path("regression"),
            engine="dag-ml",
        )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert any(_FALLBACK_WARNING in str(w.message) for w in caught)


def test_run_dagml_propagates_non_catchable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback catches ONLY DagMlUnsupported/NotImplementedError. A genuine bug in the dag-ml
    path (any other exception) MUST still propagate — it is never swallowed into a legacy run."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _boom(*_args: object, **_kwargs: object) -> RunResult:
        raise RuntimeError("genuine dag-ml backend bug")

    monkeypatch.setattr(run_backend, "run_via_dagml", _boom)
    with pytest.raises(RuntimeError, match="genuine dag-ml backend bug"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="dag-ml")


def test_run_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown"):
        nirs4all.run([{"model": PLSRegression(n_components=2)}], dataset_path("regression"), engine="bogus")
