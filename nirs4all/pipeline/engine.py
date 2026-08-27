"""Execution-engine selector for the nirs4all core (interim legacy-default posture).

Seam for the **nirs4all-core → dag-ml** migration. The default production engine is **legacy**: the
public-maintained nirs4all stays pure-Python by default, so :func:`nirs4all.run` runs through the
in-process *legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`) unless another engine is
selected. The dag-ml backend (:mod:`nirs4all.pipeline.dagml.run_backend`) — which runs the pipeline
natively (Rust) and returns a ``RunResult`` of dag-ml's native scores — stays
**fully selectable** via
``engine="dag-ml"`` or ``$N4A_ENGINE=dag-ml``; the whole dag-ml integration (in-process path, native
generator coverage, conformance pack, hard dependency) is intact and runnable out of the box.

An unavailable or unsupported dag-ml request is fail-closed by default.  A
legacy rerun is an explicit R2 rollback action through
``run(..., engine="dag-ml", allow_legacy_fallback=True)``; it is never an
implicit substitute for a claimed native execution.

This is the interim posture: the maintainer keeps the public Python version as the default until the
planned global refactoring lands; at that point the legacy-DROP cutover flips the default back to
dag-ml (the ADR-17 end state). The side-by-side comparison mode (``"dual"``) is intentionally
limited to the strict :func:`nirs4all.run` oracle subset; other public operations reject it.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`
(``legacy``, interim). Pass ``engine="dag-ml"`` (or ``$N4A_ENGINE=dag-ml``) to run on the dag-ml
    backend. ``engine="native"`` separately exposes the verified, fail-closed
    Methods subset: raw-array PLS run/session/predict and Archive V2 replay.
    See ``dag-ml/docs/migration-nirs4all/``.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from threading import Lock
from typing import Any, Literal, cast

Engine = Literal["legacy", "dag-ml", "dual", "native"]

DEFAULT_ENGINE: Engine = "legacy"
ENGINE_ENV_VAR = "N4A_ENGINE"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "dual", "native")

LegacyFallbackReason = Literal["backend_unavailable", "unsupported_shape"]


class LegacyFallbackWarning(RuntimeWarning):
    """Structured evidence that an explicit native-to-legacy rollback occurred.

    This warning can only be emitted for ``engine="dag-ml"`` when the caller
    supplied ``allow_legacy_fallback=True``.  It deliberately carries no input
    data or pipeline representation: diagnostics must remain safe to collect
    without disclosing scientific data.
    """

    engine: Literal["dag-ml"] = "dag-ml"

    def __init__(self, *, reason: LegacyFallbackReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        if reason == "backend_unavailable":
            message = f"the dag-ml backend is not available ({detail}); falling back to the legacy engine"
        else:
            message = f"engine='dag-ml' does not support this pipeline shape ({detail}); falling back to the legacy engine"
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        """Return the safe, structured diagnostic payload for this rollback.

        ``detail`` is deliberately omitted: backend exception text may include
        user-supplied configuration.  Callers that display the warning already
        receive that text through the ordinary warning channel.
        """

        return {"engine": self.engine, "reason": self.reason}


_legacy_fallback_counts: Counter[LegacyFallbackReason] = Counter()
_legacy_fallback_lock = Lock()


def record_legacy_fallback(*, reason: LegacyFallbackReason, detail: str) -> LegacyFallbackWarning:
    """Record one caller-authorized rollback and return its warning payload.

    This intentionally has no persistence or telemetry side effect.  Product
    callers decide whether to inspect :func:`legacy_fallback_metrics`; the
    counter merely makes the opt-in rollback observable in the current process.
    """

    with _legacy_fallback_lock:
        _legacy_fallback_counts[reason] += 1
    return LegacyFallbackWarning(reason=reason, detail=detail)


def legacy_fallback_metrics() -> dict[str, int]:
    """Return process-local counts for explicit legacy rollback events.

    The returned mapping is a snapshot with a stable ``total`` plus one count
    per declared rollback reason.  No run that failed closed contributes here.
    """

    with _legacy_fallback_lock:
        unavailable = _legacy_fallback_counts["backend_unavailable"]
        unsupported = _legacy_fallback_counts["unsupported_shape"]
    return {
        "total": unavailable + unsupported,
        "backend_unavailable": unavailable,
        "unsupported_shape": unsupported,
    }


class DualRunUnsupported(NotImplementedError):
    """The strict side-by-side oracle cannot prove support for a requested run.

    The dual engine never falls back to legacy: callers must select ``engine="legacy"`` or
    ``engine="dag-ml"`` explicitly if they want either non-oracle behavior.
    """


class DualRunMismatchError(RuntimeError):
    """Legacy and native results disagreed under the explicit dual-run contract.

    ``report`` contains the compared semantic fields, their absolute/relative tolerances,
    measured wall-clock timings, and every mismatch.  Timings are diagnostic only in R1;
    no performance budget is asserted here.
    """

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        mismatches = self.report.get("mismatches", [])
        super().__init__(f"engine='dual' detected {len(mismatches)} legacy/native mismatch(es): {mismatches}")


def resolve_engine(engine: str | None = None) -> Engine:
    """Resolve the requested execution engine, defaulting to ``legacy`` (interim, pre-refactoring).

    The default is the pure-Python ``legacy`` orchestrator: the public-maintained nirs4all stays
    pure-Python by default until the planned global refactoring lands (then the legacy-DROP cutover
    flips the default back to ``dag-ml``). The dag-ml backend stays fully selectable here via
    ``engine="dag-ml"`` or ``$N4A_ENGINE=dag-ml``.

    Args:
        engine: Explicit engine name. When ``None``, falls back to the
            ``$N4A_ENGINE`` environment variable, then :data:`DEFAULT_ENGINE`.

    Returns:
        The validated engine name. ``"legacy"`` (the default), ``"dag-ml"`` and the narrow
        ``"dual"`` oracle mode are dispatched by :func:`nirs4all.run`. ``"native"`` is a
        fail-closed Methods subset for raw-array training, sessions, prediction, and Archive V2
        replay; unsupported operations refuse it before execution.

    Raises:
        ValueError: If the name is not one of :data:`ENGINES`.
    """
    name = (engine or os.environ.get(ENGINE_ENV_VAR) or DEFAULT_ENGINE).strip().lower()
    if name not in ENGINES:
        raise ValueError(f"unknown nirs4all engine {name!r}; valid engines: {list(ENGINES)}")
    return cast(Engine, name)


def require_legacy_engine(operation: str, engine: str | None = None) -> Engine:
    """Resolve an API backend selector and reject operations not yet backed by dag-ml."""
    selected = resolve_engine(engine)
    if selected != "legacy":
        if selected == "dual":
            raise DualRunUnsupported(f"nirs4all.{operation} does not support engine='dual'; the strict dual oracle is implemented only for nirs4all.run on its documented subset.")
        if selected == "dag-ml":
            raise NotImplementedError(
                f"nirs4all.{operation} does not have a dag-ml execution path yet; it does not "
                "have an execution path under dag-ml. Use "
                "engine='legacy' for this transition release. nirs4all.run supports "
                "engine='dag-ml' with documented fallback boundaries."
            )
        raise NotImplementedError(
            f"nirs4all.{operation} does not have an execution path for engine='{selected}' yet; use "
            "engine='legacy' for this transition release. nirs4all.run supports "
            "engine='dag-ml' with documented fallback boundaries."
        )
    return selected
