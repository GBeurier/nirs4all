"""CROSS-ENGINE EXPORT SURFACE (B-011 slice): the export/error/workspace contract both engines share.

The dag-ml backend returns native scores with NO workspace (no SQLite store, no artifacts dir), while a
legacy run owns a real workspace. ``RunResult`` branches on that difference at EVERY export entry point:
dag-ml exports now use captured native artifacts or raise a stable ``RtError`` refusal, and the legacy refit
bridge is hidden behind ``compatibility="legacy-refit"``. These tests pin that surface directly on
constructed ``RunResult`` objects — NO pipeline run, NO ``dag-ml-cli`` binary, fully deterministic — so the
cross-engine contract is locked even on a build where the native path is unavailable. They complement the
run-based exactness tests in
``test_conformance_export_roundtrip`` / ``test_dagml_native_export_model`` (both ``slow`` / binary-gated),
which is why this module is FAST (``parity`` only, not ``slow``).

Three contracts, each currently exercised only INDIRECTLY (or asymmetrically) elsewhere:

* :meth:`RunResult._is_dagml_engine` — the workspace-marker predicate that gates every export branch. Used
  load-bearingly by ``_conformance_helpers.dual_engine_runner`` and ``_no_workspace_export_error``, but never
  unit-tested for its truth table (the ``any(...)`` over datasets + the ``isinstance(info, dict)`` guard).
* The no-workspace ERROR contract — dag-ml returns a structured ``RtError`` refusal that points to native
  artifacts / explicit compatibility; legacy returns a plain ``RuntimeError`` for genuine misuse.
* ``export_model`` selector fast-rejection — ``export()`` already pins that an explicit ``source=``/``chain_id=``
  on a dag-ml run fails BEFORE the legacy refit materializes; ``export_model`` had no such test (asymmetry).
"""

from __future__ import annotations

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
    deterministically. ``engine`` is the marker the export gating reads; ``export_spec`` (dag-ml only) is the
    frozen input payload used only by ``compatibility="legacy-refit"``.
    """
    return RunResult(
        predictions=Predictions(),
        per_dataset={"toy": {"engine": engine}},
        _dagml_export_spec=export_spec,
    )


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
    (native artifacts or structured refusal) is the safe direction; a per-dataset value flipping to dag-ml
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
# B. The no-workspace ERROR contract: dag-ml gets a structured refusal, while legacy
#    keeps a plain misuse RuntimeError.
# ---------------------------------------------------------------------------


def test_no_workspace_export_diverges_by_engine() -> None:
    """The no-workspace export error remains engine-specific.

    dag-ml gets the structured V1 export refusal; legacy keeps a plain RuntimeError because a detached legacy
    result with no workspace is genuine misuse, not a compatibility opportunity.
    """
    dagml_error = _result(engine="dag-ml")._no_workspace_export_error()  # noqa: SLF001
    assert isinstance(dagml_error, RtError)
    dagml_payload = dagml_error.to_dict()
    assert dagml_payload["cause"] == "unsupported_capability"
    assert dagml_payload["unsupported_capability"] == "dagml_native_export"
    assert "nirs4all-tools" in dagml_payload["mitigation"]
    assert "compatibility='legacy-refit'" in dagml_payload["mitigation"]

    legacy_error = _result(engine="legacy")._no_workspace_export_error()  # noqa: SLF001
    assert isinstance(legacy_error, RuntimeError)
    assert not isinstance(legacy_error, NotImplementedError)
    assert "no workspace path available" in str(legacy_error)


# ---------------------------------------------------------------------------
# C. export_model selector fast-rejection: parity with the existing export() test, and proof of NO refit.
# ---------------------------------------------------------------------------


def test_dagml_export_model_rejects_source_before_legacy_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``source=`` on a dag-ml ``export_model`` fails fast and does NOT materialize the legacy refit.

    ``source=`` names a legacy workspace record; a dag-ml run has none, so the export path must raise before
    any explicit compatibility refit could run. This is the ``export_model`` companion to
    ``test_conformance_export_roundtrip``'s ``export()`` selector-rejection test — the same fail-fast
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


def test_dagml_export_legacy_refit_requires_named_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy refit bridge is reachable only through ``compatibility="legacy-refit"``."""
    result = _result(engine="dag-ml", export_spec={"pipeline": [], "dataset": object()})
    calls: list[tuple[str, Path, str | None, int | None]] = []

    class _Delegate:
        def export(self, output_path: str | Path, *, format: str = "n4a") -> Path:
            calls.append(("export", Path(output_path), format, None))
            return Path(output_path)

        def export_model(self, output_path: str | Path, *, format: str | None = None, fold: int | None = None) -> Path:
            calls.append(("export_model", Path(output_path), format, fold))
            return Path(output_path)

    monkeypatch.setattr(result, "_dagml_export_delegate", lambda: _Delegate())

    out = tmp_path / "compat.n4a"
    assert result.export(out, compatibility="legacy-refit") == out
    model_out = tmp_path / "compat.joblib"
    assert result.export_model(model_out, format="joblib", fold=2, compatibility="legacy-refit") == model_out
    assert calls == [
        ("export", out, "n4a", None),
        ("export_model", model_out, "joblib", 2),
    ]
