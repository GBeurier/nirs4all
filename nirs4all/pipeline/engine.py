"""Execution-engine selector for the nirs4all core (ADR-17 cutover/rollback skeleton).

Seam for the **nirs4all-core → dag-ml** migration. Today the only production
engine is the in-process *legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`).
The dag-ml-backed engine (``"dag-ml"``) and the side-by-side comparison mode
(``"dual"``) are reserved for the migration and not yet implemented, so
:func:`resolve_engine` refuses them with a clear ``NotImplementedError``.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`.

This module is intentionally **not wired** into :func:`nirs4all.run` yet — it is
the inert selection point the bridge will plug into. See
``dag-ml/docs/migration-nirs4all/WORKING_STRATEGY.md``.
"""

from __future__ import annotations

import os
from typing import Literal, cast

Engine = Literal["legacy", "dag-ml", "dual"]

DEFAULT_ENGINE: Engine = "legacy"
ENGINE_ENV_VAR = "N4A_ENGINE"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "dual")


def resolve_engine(engine: str | None = None) -> Engine:
    """Resolve the requested execution engine, defaulting to ``legacy``.

    Args:
        engine: Explicit engine name. When ``None``, falls back to the
            ``$N4A_ENGINE`` environment variable, then :data:`DEFAULT_ENGINE`.

    Returns:
        The validated engine name. Only ``"legacy"`` is runnable today.

    Raises:
        ValueError: If the name is not one of :data:`ENGINES`.
        NotImplementedError: If a recognised but not-yet-implemented engine
            (``"dag-ml"`` or ``"dual"``) is requested.
    """
    name = (engine or os.environ.get(ENGINE_ENV_VAR) or DEFAULT_ENGINE).strip().lower()
    if name not in ENGINES:
        raise ValueError(f"unknown nirs4all engine {name!r}; valid engines: {list(ENGINES)}")
    if name != DEFAULT_ENGINE:
        raise NotImplementedError(
            f"the {name!r} engine (dag-ml core) is under construction in the "
            f"nirs4all-core -> dag-ml migration; only {DEFAULT_ENGINE!r} is available today "
            f"(see dag-ml/docs/migration-nirs4all/)"
        )
    return cast(Engine, name)
