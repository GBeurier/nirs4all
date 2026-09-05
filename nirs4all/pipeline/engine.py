"""Execution-engine selector for the nirs4all core (V1 native cutover posture).

Seam for the **nirs4all-core → native** migration. The default strict/product profile is
**native**. The general :func:`nirs4all.run` API additionally performs declaration-only
capability selection when no selector is supplied: portable requests use native,
other requests use DAG-ML. It never retries execution on another engine. The in-process
*legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`) remains available as an explicit
compatibility path via the public ``engine="legacy"`` selector only.
The explicit ``engine="native"`` lane is the fail-closed Archive V2/N4MM producer for the
portable Methods subset; it never falls back to :class:`~nirs4all.pipeline.PipelineRunner`.

The side-by-side comparison mode (``"dual"``) is intentionally limited to the strict
:func:`nirs4all.run` oracle subset; other public operations remain unavailable.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`
(``native``). Ambient legacy/dual selection is rejected: pass ``engine="legacy"`` explicitly for
compatibility runs. See
``dag-ml/docs/migration-nirs4all/``.

Product-owned internal boundaries resolve with ``execution_profile="strict"``.  That profile is
deliberately not ambient-configurable: it rejects both direct and environment-selected legacy
execution, the dual legacy oracle, and opt-in legacy fallback.  The default
``"rollback-capable"`` profile keeps the frozen public Python ``run`` contract and explicit
compatibility lane available until the R4 removal decision.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from collections.abc import Mapping
from typing import Any, Literal, cast

Engine = Literal["legacy", "dag-ml", "native", "dual"]
ExecutionProfile = Literal["rollback-capable", "strict"]

DEFAULT_ENGINE: Engine = "native"
DEFAULT_EXECUTION_PROFILE: ExecutionProfile = "rollback-capable"
ENGINE_ENV_VAR = "N4A_ENGINE"
LEGACY_USAGE_COUNTER_ENV_VAR = "N4A_LEGACY_USAGE_COUNTER"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "native", "dual")
EXECUTION_PROFILES: tuple[ExecutionProfile, ...] = ("rollback-capable", "strict")
_LEGACY_USAGE_COUNTER_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_LEGACY_USAGE_COUNTS: dict[str, int] = {"legacy": 0, "dual": 0}
_LEGACY_USAGE_COUNTS_LOCK = threading.Lock()


class ExecutionProfileError(ValueError):
    """A fail-closed execution-profile preflight rejected the requested path."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class DualRunUnsupported(NotImplementedError):
    """The strict side-by-side oracle cannot prove support for a requested run.

    The dual engine never falls back to legacy. Callers must select
    ``engine="legacy"`` explicitly when they need the temporary rollback lane.
    """


class DualRunMismatchError(RuntimeError):
    """The native and explicit legacy oracle legs disagreed."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        mismatches = self.report.get("mismatches", [])
        super().__init__(f"engine='dual' detected {len(mismatches)} native/legacy mismatch(es): {mismatches}")


class LegacyEngineUsageWarning(UserWarning):
    """Stable structured warning for an explicit legacy-bearing API request."""

    def __init__(self, *, engine: Engine, operation: str) -> None:
        self.diagnostic: dict[str, str | int] = {
            "schema_version": 1,
            "code": "nirs4all.explicit_legacy_engine",
            "engine": engine,
            "operation": operation,
        }
        super().__init__(json.dumps(self.diagnostic, sort_keys=True, separators=(",", ":")))


def report_explicit_legacy_engine(
    requested_engine: str | None,
    selected_engine: Engine,
    *,
    operation: str,
) -> None:
    """Make an explicit legacy-bearing public request visible and optionally count it.

    The counter is process-local and disabled unless
    ``N4A_LEGACY_USAGE_COUNTER`` is set to a conventional true value. It records
    only the selected engine, never request inputs, scientific data, paths, or
    network telemetry.
    """
    if requested_engine is None or selected_engine not in {"legacy", "dual"}:
        return

    if os.environ.get(LEGACY_USAGE_COUNTER_ENV_VAR, "").strip().lower() in _LEGACY_USAGE_COUNTER_TRUE_VALUES:
        with _LEGACY_USAGE_COUNTS_LOCK:
            _LEGACY_USAGE_COUNTS[selected_engine] += 1

    warnings.warn(
        LegacyEngineUsageWarning(engine=selected_engine, operation=operation),
        stacklevel=3,
    )


def get_legacy_engine_usage_counts() -> dict[str, int]:
    """Return a data-free snapshot of the opt-in process-local support counter."""
    with _LEGACY_USAGE_COUNTS_LOCK:
        legacy = _LEGACY_USAGE_COUNTS["legacy"]
        dual = _LEGACY_USAGE_COUNTS["dual"]
    return {"legacy": legacy, "dual": dual, "total": legacy + dual}


def resolve_engine(
    engine: str | None = None,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    allow_fallback: bool = False,
) -> Engine:
    """Resolve the requested execution engine, defaulting to ``native``.

    The V1 default is the fail-closed portable Methods Archive V2 producer. The pure-Python legacy
    orchestrator remains available only when selected explicitly via ``engine="legacy"``.

    Args:
        engine: Explicit engine name. When ``None``, falls back to the
            ``$N4A_ENGINE`` environment variable, then :data:`DEFAULT_ENGINE`.
        execution_profile: ``"rollback-capable"`` preserves explicit public Python compatibility.
            Product paths must pass ``"strict"`` explicitly; no environment variable can weaken it.
        allow_fallback: Whether the caller requested dag-ml to legacy fallback.  Strict execution
            rejects this before dataset access or meaningful computation.

    Returns:
        The validated engine name. ``"native"`` (the default), ``"dag-ml"``, ``"legacy"``, and
        the narrow ``"dual"`` oracle are dispatched by :func:`nirs4all.run`.

    Raises:
        ValueError: If the name is not one of :data:`ENGINES`.
        ExecutionProfileError: If the profile is unknown or a strict request could reach legacy.
    """
    if not isinstance(execution_profile, str):
        raise ExecutionProfileError(
            "profile_invalid_type",
            "nirs4all execution profile must be a string",
        )
    normalized_profile = execution_profile.strip().lower()
    if normalized_profile not in EXECUTION_PROFILES:
        raise ExecutionProfileError(
            "profile_unknown",
            f"unknown nirs4all execution profile {normalized_profile!r}; valid profiles: {list(EXECUTION_PROFILES)}",
        )
    requested = engine if engine is not None else os.environ.get(ENGINE_ENV_VAR, DEFAULT_ENGINE)
    name = requested.strip().lower()
    if name not in ENGINES:
        raise ValueError(f"unknown nirs4all engine {name!r}; valid engines: {list(ENGINES)}")
    if normalized_profile == "strict":
        if name in {"legacy", "dual"}:
            raise ExecutionProfileError(
                "legacy_execution_forbidden",
                f"execution_profile='strict' forbids engine={name!r}; use a native product engine",
            )
        if allow_fallback:
            raise ExecutionProfileError(
                "legacy_fallback_forbidden",
                "execution_profile='strict' forbids allow_fallback=True",
            )
    if engine is None and name in {"legacy", "dual"}:
        raise ExecutionProfileError(
            "ambient_legacy_execution_forbidden",
            f"ambient {ENGINE_ENV_VAR}={name!r} cannot select a legacy execution path; "
            "pass engine='legacy' explicitly for the rollback-capable compatibility lane",
        )
    if allow_fallback and name != "legacy":
        raise ExecutionProfileError(
            "legacy_fallback_forbidden",
            "allow_fallback=True is no longer an execution path; pass engine='legacy' explicitly",
        )
    return cast(Engine, name)
