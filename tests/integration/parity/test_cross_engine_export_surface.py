"""CROSS-ENGINE EXPORT SURFACE (B-011 slice): the export/error/workspace contract both engines share.

The dag-ml backend returns native scores with NO workspace (no SQLite store, no artifacts dir), while a
legacy run owns a real workspace. ``RunResult`` branches on that difference at EVERY export entry point, and
the planned ADR-17 cutover (``try dag-ml → except NotImplementedError → legacy``) depends on the branch
raising the RIGHT exception TYPE. These tests pin that surface directly on constructed ``RunResult`` objects —
NO pipeline run, NO ``dag-ml-cli`` binary, fully deterministic — so the cross-engine contract is locked even
on a build where the native path is unavailable. They complement the run-based exactness tests in
``test_conformance_export_roundtrip`` / ``test_dagml_native_export_model`` (both ``slow`` / binary-gated),
which is why this module is FAST (``parity`` only, not ``slow``).

Three contracts, each currently exercised only INDIRECTLY (or asymmetrically) elsewhere:

* :meth:`RunResult._is_dagml_engine` — the workspace-marker predicate that gates every export branch. Used
  load-bearingly by ``_conformance_helpers.dual_engine_runner`` and ``_no_workspace_export_error``, but never
  unit-tested for its truth table (the ``any(...)`` over datasets + the ``isinstance(info, dict)`` guard).
* The no-workspace ERROR contract — the SAME "no workspace + no export spec" call diverges BY ENGINE: dag-ml
  → a CATCHABLE :class:`NotImplementedError` (the cutover redirects export to legacy); legacy → a
  :class:`RuntimeError` that is NOT a ``NotImplementedError`` (a genuine misuse the cutover must let
  PROPAGATE). The subclass relationship ``NotImplementedError ⊂ RuntimeError`` makes the "legacy is NOT
  catchable as the dag-ml gap" assertion load-bearing, not cosmetic.
* ``export_model`` selector fast-rejection — ``export()`` already pins that an explicit ``source=``/``chain_id=``
  on a dag-ml run fails BEFORE the legacy refit materializes; ``export_model`` had no such test (asymmetry).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.rt import RtError

pytestmark = [pytest.mark.parity]


def _result(*, engine: str, export_spec: dict[str, Any] | None = None) -> RunResult:
    """A minimal detached ``RunResult`` tagged for ``engine``, with no workspace and no runner.

    Mirrors how the dag-ml backend builds its in-memory result (empty :class:`Predictions`, a
    ``per_dataset[name]["engine"]`` marker, an optional ``_dagml_export_spec``) and how a detached legacy
    result looks once its runner/store are gone — so an export call lands on the no-workspace branch
    deterministically. ``engine`` is the marker the export gating reads; ``export_spec`` (dag-ml only) toggles
    the legacy-refit bridge path.
    """
    return RunResult(
        predictions=Predictions(),
        per_dataset={"toy": {"engine": engine}},
        _dagml_export_spec=export_spec,
    )


def _call_export(result: RunResult, out: Path) -> Any:
    """Invoke ``export`` with an explicit ``source`` so the call reaches the no-workspace branch.

    ``source`` MUST be non-``None``: with ``source=None`` and an empty :class:`Predictions`, ``export``
    raises ``ValueError('No predictions available to export')`` BEFORE the workspace check — passing an
    explicit (dummy) source skips that guard so the engine-specific no-workspace error is what surfaces.
    """
    return result.export(out, source={"prediction_id": "p0"})


def _call_export_model(result: RunResult, out: Path) -> Any:
    """Invoke ``export_model`` with an explicit ``source`` so the call reaches the no-workspace branch (see :func:`_call_export`)."""
    return result.export_model(out, source={"prediction_id": "p0"})


_EXPORT_ENTRY_POINTS = [
    pytest.param(_call_export, id="export"),
    pytest.param(_call_export_model, id="export_model"),
]


# ---------------------------------------------------------------------------
# A. _is_dagml_engine(): the workspace-marker predicate that gates every export branch.
# ---------------------------------------------------------------------------


def test_is_dagml_engine_detects_dagml_marker() -> None:
    """A ``per_dataset`` value tagged ``engine == "dag-ml"`` makes the result a dag-ml result."""
    assert _result(engine="dag-ml")._is_dagml_engine() is True  # noqa: SLF001


def test_is_dagml_engine_false_for_legacy_marker() -> None:
    """A ``legacy`` engine marker is NOT a dag-ml result — the predicate must not over-trigger and route a real legacy run through the dag-ml export gap."""
    assert _result(engine="legacy")._is_dagml_engine() is False  # noqa: SLF001


def test_is_dagml_engine_false_for_empty_per_dataset() -> None:
    """An empty ``per_dataset`` (no datasets executed) is NOT a dag-ml result — ``any(...)`` over nothing is ``False``."""
    result = RunResult(predictions=Predictions(), per_dataset={})
    assert result._is_dagml_engine() is False  # noqa: SLF001


def test_is_dagml_engine_true_if_any_dataset_is_dagml() -> None:
    """A MIXED multi-dataset result (one legacy, one dag-ml) is a dag-ml result — the ``any(...)`` semantic.

    Export of such a result cannot rely on a workspace for the dag-ml dataset, so gating to the dag-ml path
    (the catchable error / legacy-refit bridge) is the safe direction; a per-dataset value flipping to dag-ml
    must never be masked by a sibling legacy dataset.
    """
    result = RunResult(
        predictions=Predictions(),
        per_dataset={"a": {"engine": "legacy"}, "b": {"engine": "dag-ml"}},
    )
    assert result._is_dagml_engine() is True  # noqa: SLF001


def test_is_dagml_engine_ignores_non_mapping_entries() -> None:
    """A non-dict ``per_dataset`` value must not crash the predicate (the ``isinstance(info, dict)`` guard).

    Real ``per_dataset`` values are dicts, but the predicate guards each entry so a stray string/``None``
    payload yields a clean boolean instead of an ``AttributeError`` from ``.get`` on a non-mapping. A dag-ml
    dict among non-dict entries is still detected; non-dict-only entries are simply not dag-ml.
    """
    detected = RunResult(predictions=Predictions(), per_dataset={"a": "legacy", "b": None, "c": {"engine": "dag-ml"}})
    assert detected._is_dagml_engine() is True  # noqa: SLF001
    not_detected = RunResult(predictions=Predictions(), per_dataset={"a": "legacy", "b": None})
    assert not_detected._is_dagml_engine() is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# B. The no-workspace ERROR contract: the SAME call diverges by engine, and the legacy
#    error is deliberately NOT catchable as the dag-ml gap (NotImplementedError ⊂ RuntimeError).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", _EXPORT_ENTRY_POINTS)
def test_no_workspace_export_diverges_by_engine(call: Callable[[RunResult, Path], Any], tmp_path: Path) -> None:
    """The IDENTICAL no-workspace export call raises a DIFFERENT exception class per engine.

    The contract the ADR-17 cutover (``try dag-ml → except NotImplementedError → legacy``) is built on:

    * dag-ml (no export spec) → a CATCHABLE :class:`NotImplementedError` — the cutover catches it and
      redirects the export to the legacy engine.
    * legacy (detached, no workspace) → a :class:`RuntimeError` that is NOT a ``NotImplementedError`` — a
      genuine misuse (a result that never came from a workspace run) that the cutover MUST let PROPAGATE.

    Because ``NotImplementedError`` is a SUBCLASS of ``RuntimeError``, ``pytest.raises(RuntimeError)`` would
    also accept a (wrong) ``NotImplementedError`` from the legacy path — so the load-bearing assertion is the
    explicit ``not isinstance(..., NotImplementedError)``: if the legacy error ever became a
    ``NotImplementedError`` the cutover would silently swallow a real misuse and refit on legacy.
    """
    with pytest.raises(NotImplementedError, match="no workspace artifacts") as dagml_exc:
        call(_result(engine="dag-ml"), tmp_path / "dagml.n4a")

    with pytest.raises(RuntimeError) as legacy_exc:
        call(_result(engine="legacy"), tmp_path / "legacy.n4a")

    assert isinstance(dagml_exc.value, NotImplementedError)
    assert not isinstance(legacy_exc.value, NotImplementedError), (
        "legacy no-workspace export must raise a plain RuntimeError that the cutover's "
        "`except NotImplementedError` does NOT catch — else a genuine legacy misuse is silently "
        "redirected to a legacy refit instead of surfacing"
    )
    assert "no workspace path available" in str(legacy_exc.value)
    assert RtError.invalid_request(dagml_exc.value, verb="export").cause == "invalid_request"
    assert RtError.invalid_request(legacy_exc.value, verb="export").cause == "invalid_request"


# ---------------------------------------------------------------------------
# C. export_model selector fast-rejection: parity with the existing export() test, and proof of NO refit.
# ---------------------------------------------------------------------------


def test_dagml_export_model_rejects_source_before_legacy_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``source=`` on a dag-ml ``export_model`` fails fast and does NOT materialize the legacy refit.

    ``source=`` names a legacy workspace record; a dag-ml run has none, so the transitional export bridge must
    raise a catchable :class:`NotImplementedError` BEFORE the on-demand legacy refit runs (otherwise an
    invalid request would burn a full pipeline re-fit before failing). This is the ``export_model`` companion
    to ``test_conformance_export_roundtrip``'s ``export()`` selector-rejection test — the same fail-fast
    contract, on the lightweight model-only entry point.
    """
    result = _result(engine="dag-ml", export_spec={"pipeline": [], "dataset": object()})

    def _unexpected_delegate() -> object:
        raise AssertionError("dag-ml export_model delegate must not materialize for an explicit source= selector")

    monkeypatch.setattr(result, "_dagml_export_delegate", _unexpected_delegate)

    with pytest.raises(NotImplementedError, match="export_model does not support an explicit source") as excinfo:
        result.export_model(tmp_path / "model.joblib", source={"prediction_id": "p0"})

    assert result._dagml_legacy_result is None  # noqa: SLF001 -- no legacy refit was triggered
    assert RtError.invalid_request(excinfo.value, verb="export").to_dict()["cause"] == "invalid_request"
