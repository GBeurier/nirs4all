"""Failure types and fail-loud guards for the dag-ml host backend.

``DagMlUnsupported`` is the catchable error every unsupported-or-failed dag-ml shape raises;
``DagMlUnavailable`` is the narrow error a dag-ml-backend PREFLIGHT raises when NEITHER execution
mechanism is installed (no in-process PyO3 extension AND no dag-ml-cli binary); ``_cli_child_error``
extracts the real cause from a dag-ml-cli failure; ``_raise_run_failure`` decides
propagate-vs-fallback for a non-zero subprocess run from the adapter's structured ``error_kind``
(:func:`_run_failure_kind`); ``_reject_multi_model`` rejects a multi-model pipeline UP FRONT.
"""

from __future__ import annotations

from typing import Any

# Structured error-classification protocol between the process adapter and the host. When node
# execution raises, the adapter emits a JSONL error frame (``type == "error"``) carrying ``error_kind``:
#   * ``"unsupported"`` — the node deliberately raised :class:`DagMlUnsupported` (a shape the dag-ml
#     path cannot run) → the host re-raises ``DagMlUnsupported`` so ``run()`` falls back to legacy;
#   * ``"error"`` — ANY other exception (a GENUINE bug, e.g. a model fit failure) → the host PROPAGATES
#     the real error, matching the in-process path (where an operator exception surfaces as the raw
#     bridge error, never swallowed into a fallback).
# It is a dedicated STRUCTURED field on the captured frame — NOT free text in stdout — so a genuine
# error whose message merely *contains* the marker words cannot spoof the classification.
_ERROR_FRAME_TYPE = "error"
_ERROR_KIND_KEY = "error_kind"
ERROR_KIND_UNSUPPORTED = "unsupported"
ERROR_KIND_GENERIC = "error"


class DagMlUnsupported(NotImplementedError):
    """A pipeline shape (or a dag-ml-cli run of one) the dag-ml backend cannot execute.

    A subclass of ``NotImplementedError`` so the cutover fallback
    (``try dag-ml; except NotImplementedError → legacy``) catches it: an unsupported-or-failed
    dag-ml shape must NEVER escape ``run_via_dagml`` as a bare ``RuntimeError``/``ValueError``
    (which the fallback would not catch and would hard-fail in production). Raised both for shapes
    rejected UP FRONT (no splitter, multiple model nodes, …) and for a dag-ml-cli run that failed
    mid-execution because the SHAPE is unsupported — in either case the fallback redirects the
    pipeline to the legacy engine. A GENUINE operator bug (a model fit failure) is NOT this — it
    propagates as a real error (see :func:`_raise_run_failure`).
    """


class _OperatorLoweringUnsupported(DagMlUnsupported):
    """A flat-single ``_or_`` whose LOWERING (DSL serialization / variant_label fingerprinting) is unsupported.

    The DISTINCT signal the native operator-generation path (#23 Phase 7) raises from its narrow LOWERING
    guard ONLY, so the routing branch can catch JUST this and demote to the Python-expand path while a
    RUNTIME ``DagMlUnsupported`` — e.g. :func:`_raise_run_failure` classifying a non-zero run as
    ``error_kind == "unsupported"`` AFTER the lowering guard — PROPAGATES untouched. Catching the broad
    :class:`DagMlUnsupported` at the routing branch would mask such a runtime demotion (a real coverage
    boundary the host has not pre-checked), so the two are kept separate by type, not by message.
    """


class DagMlUnavailable(RuntimeError):
    """NEITHER dag-ml execution mechanism is installed, so the dag-ml backend cannot run at all.

    Raised by the dag-ml-backend availability probes: the run-level PREFLIGHT
    (:func:`~nirs4all.pipeline.dagml.run_backend.preflight_dagml_backend`) when BOTH mechanisms are
    missing (the in-process PyO3 extension ``dag_ml._dag_ml`` does not import/load AND the ``dag-ml-cli``
    binary does not exist), and the in-process/subprocess router's SUBPROCESS branch when in-process is
    not selected and ``dag-ml-cli`` is absent. Since the ADR-17 cutover made ``dag-ml``
    the default engine + a hard dependency, ``run()`` catches THIS (alongside ``DagMlUnsupported`` /
    ``NotImplementedError``) and falls back to the legacy engine WITH A WARNING — so a wheel install
    that is somehow missing the native backend degrades transparently instead of hard-failing.

    Deliberately NARROW: it is raised only by those explicit availability probes, NEVER by
    wrapping a run in a blanket ``except ImportError/FileNotFoundError`` (which would swallow a genuine
    dag-ml bug). A real dag-ml runtime/operator error is NOT this — it propagates untouched.

    NOT a subclass of ``DagMlUnsupported``/``NotImplementedError``: "the backend is not installed" is a
    distinct condition from "this pipeline shape is unsupported", and ``run()`` catches it explicitly.
    """


def _cli_child_error(stdout: str) -> str:
    """The child adapter's actual error line(s) from a dag-ml-cli failure, for an informative message.

    The captured ``stdout``+``stderr`` ends with the process-adapter traceback; the last
    ``Error:``/``ValueError:``/``…Error:`` line is the real cause (e.g. ``ValueError: data view is
    missing``). Surfacing it beats the bare ``rc=1`` (the legacy E-QUALITY message names the cause),
    and lets the reader see WHY the shape failed. Falls back to the trailing slice when no error line
    is found, so nothing is ever hidden.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    causes = [line for line in lines if line.startswith("Error:") or ("Error:" in line and not line.startswith("File "))]
    return causes[-1] if causes else stdout[-800:]


def _reject_multi_model(steps: list[Any]) -> None:
    """Reject a concrete pipeline carrying MORE THAN ONE ``{"model": ...}`` step (fail loud UP FRONT).

    The dag-ml CV+refit assembly (:func:`assemble_cv_refit_dsl` → ``model_node_id``) binds the data
    source to exactly ONE model node; several top-level model steps compile to several model nodes,
    only the first of which gets a data binding, so the others reach the runtime with an empty data
    view and crash mid-execution inside the adapter (``node_runner._sample_ids`` → ``data view is
    missing`` → a bare ``rc=1``). Detecting it here turns that into a CATCHABLE up-front error.

    Several models in one nirs4all pipeline (e.g. ``U02``'s three ``{"model": PLS(n)}`` steps) is a
    legacy multi-model run — the legacy engine fans them out into separate chains; the dag-ml backend
    runs ONE model per pipeline (a model sweep uses ``{"model": {"_or_": [...]}}``, which expands to
    one model PER variant). Until per-model fan-out is wired, this shape falls back to legacy.
    """
    model_steps = [step for step in steps if isinstance(step, dict) and "model" in step]
    if len(model_steps) > 1:
        raise DagMlUnsupported(
            f"engine='dag-ml' runs ONE model per pipeline, but this pipeline has {len(model_steps)} "
            "top-level {'model': ...} steps; the dag-ml CV+refit binds data to a single model node, so "
            "multiple model nodes crash mid-run (data view is missing). Use a model sweep "
            "({'model': {'_or_': [...]}}) — which runs one model per variant — or the legacy engine."
        )


def _run_failure_kind(outcome: dict[str, Any]) -> str | None:
    """The ``error_kind`` the adapter emitted in a structured error frame for this run, or ``None``.

    Reads the dedicated structured field off the captured JSONL frames (``outcome["results"]``) — NOT a
    substring of stdout — so the classification cannot be spoofed by an error message that happens to
    contain the marker words. The last error frame wins (the failing node's). ``None`` when no node-level
    error frame was emitted (a dag-ml CLI/planner-level rejection before/around node execution).
    """
    kinds = [
        frame.get(_ERROR_KIND_KEY)
        for frame in outcome.get("results", [])
        if isinstance(frame, dict) and frame.get("type") == _ERROR_FRAME_TYPE and frame.get(_ERROR_KIND_KEY)
    ]
    return kinds[-1] if kinds else None


def _raise_run_failure(outcome: dict[str, Any], context: str) -> None:
    """Raise for a non-zero subprocess run: PROPAGATE a genuine operator bug, FALL BACK on an unsupported shape.

    Mirrors the in-process path, where an operator exception propagates as the raw bridge error (a real
    bug) while a deliberately-unsupported condition is a catchable ``DagMlUnsupported``. The node handler
    emits a STRUCTURED ``error_kind`` (:func:`_run_failure_kind`) so the two are never conflated:

    * ``error_kind == "unsupported"`` — the node raised :class:`DagMlUnsupported` (a shape the dag-ml path
      cannot run). Re-raise ``DagMlUnsupported`` so the cutover fallback redirects it to the legacy engine.
    * ``error_kind == "error"`` (a genuine bug — e.g. ``PLSRegression(n_components > n_features)``) **or no
      structured kind** (a dag-ml CLI/planner-level crash with no node error frame) — PROPAGATE a plain
      ``RuntimeError`` carrying the child cause, so it is NOT swallowed into a legacy fallback (parity with
      in-process). The host precheck already caught the unsupported-SHAPE cases up front, so an unmarked
      run failure here is a real error, not a coverage gap.
    """
    cause = _cli_child_error(outcome["stdout"])
    if _run_failure_kind(outcome) == ERROR_KIND_UNSUPPORTED:
        raise DagMlUnsupported(f"{context} (rc={outcome['returncode']}): {cause}")
    raise RuntimeError(f"{context} (rc={outcome['returncode']}): {cause}")
