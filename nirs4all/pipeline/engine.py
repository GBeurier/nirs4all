"""Execution-engine selector for the nirs4all core (V1 dag-ml cutover posture).

Seam for the **nirs4all-core → dag-ml** migration. The default production engine is **dag-ml**:
:func:`nirs4all.run` runs through the native dag-ml backend
(:mod:`nirs4all.pipeline.dagml.run_backend`) unless another engine is selected. The in-process
*legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`) remains available as an explicit
compatibility path via ``engine="legacy"`` or ``$N4A_ENGINE=legacy``.
The explicit ``engine="native"`` lane is the fail-closed Archive V2/N4MM producer for the
portable Methods subset; it never falls back to :class:`~nirs4all.pipeline.PipelineRunner`.

The side-by-side comparison mode (``"dual"``) is intentionally limited to the strict
:func:`nirs4all.run` oracle subset; other public operations remain unavailable.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`
(``dag-ml``). Pass ``engine="legacy"`` (or ``$N4A_ENGINE=legacy``) only for compatibility runs. See
``dag-ml/docs/migration-nirs4all/``.

Product callers must also select ``execution_profile="strict"``.  That profile is deliberately
not ambient-configurable: it rejects both direct and environment-selected legacy execution, the
dual legacy oracle, and opt-in legacy fallback.  The default ``"rollback-capable"`` profile keeps
the public Python compatibility contract available until the R4 removal decision.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal, cast

Engine = Literal["legacy", "dag-ml", "native", "dual"]
ExecutionProfile = Literal["rollback-capable", "strict"]

DEFAULT_ENGINE: Engine = "dag-ml"
DEFAULT_EXECUTION_PROFILE: ExecutionProfile = "rollback-capable"
ENGINE_ENV_VAR = "N4A_ENGINE"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "native", "dual")
EXECUTION_PROFILES: tuple[ExecutionProfile, ...] = ("rollback-capable", "strict")


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


def resolve_engine(
    engine: str | None = None,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    allow_fallback: bool = False,
) -> Engine:
    """Resolve the requested execution engine, defaulting to ``dag-ml``.

    The V1 default is the dag-ml backend. The pure-Python legacy orchestrator remains available only
    when selected explicitly via ``engine="legacy"`` or ``$N4A_ENGINE=legacy``. The ``"native"``
    engine explicitly selects the fail-closed portable Methods Archive V2 producer.

    Args:
        engine: Explicit engine name. When ``None``, falls back to the
            ``$N4A_ENGINE`` environment variable, then :data:`DEFAULT_ENGINE`.
        execution_profile: ``"rollback-capable"`` preserves explicit public Python compatibility.
            Product paths must pass ``"strict"`` explicitly; no environment variable can weaken it.
        allow_fallback: Whether the caller requested dag-ml to legacy fallback.  Strict execution
            rejects this before dataset access or meaningful computation.

    Returns:
        The validated engine name. ``"dag-ml"`` (the default), ``"native"``, ``"legacy"``, and
        the narrow ``"dual"`` oracle are dispatched by :func:`nirs4all.run`.

    Raises:
        ValueError: If the name is not one of :data:`ENGINES`.
        ExecutionProfileError: If the profile is unknown or a strict request could reach legacy.
    """
    normalized_profile = execution_profile.strip().lower()
    if normalized_profile not in EXECUTION_PROFILES:
        raise ExecutionProfileError(
            "profile_unknown",
            f"unknown nirs4all execution profile {normalized_profile!r}; valid profiles: {list(EXECUTION_PROFILES)}",
        )
    name = (engine or os.environ.get(ENGINE_ENV_VAR) or DEFAULT_ENGINE).strip().lower()
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
    return cast(Engine, name)
