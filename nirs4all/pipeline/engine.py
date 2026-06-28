"""Execution-engine selector for the nirs4all core (ADR-17 cutover/rollback skeleton).

Seam for the **nirs4all-core → dag-ml** migration (ADR-17 cutover). The default production engine is
now **dag-ml**: :func:`nirs4all.run` dispatches to the dag-ml backend
(:mod:`nirs4all.pipeline.dagml.run_backend`), which runs the pipeline natively (Rust) and returns a
``RunResult`` of dag-ml's native scores, falling back to the in-process *legacy* orchestrator
(:class:`~nirs4all.pipeline.PipelineRunner`) for any shape the dag-ml path cannot yet honor. The
legacy orchestrator stays directly selectable with ``engine="legacy"``. The side-by-side comparison
mode (``"dual"``) is reserved and :func:`resolve_engine` refuses it with a clear ``NotImplementedError``.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE` (``dag-ml``
since the ADR-17 cutover). Pass ``engine="legacy"`` to force the legacy orchestrator. See
``dag-ml/docs/migration-nirs4all/``.
"""

from __future__ import annotations

import os
from typing import Literal, cast

Engine = Literal["legacy", "dag-ml", "dual"]

DEFAULT_ENGINE: Engine = "dag-ml"
ENGINE_ENV_VAR = "N4A_ENGINE"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "dual")


def resolve_engine(engine: str | None = None) -> Engine:
    """Resolve the requested execution engine, defaulting to ``dag-ml`` (ADR-17 cutover).

    Args:
        engine: Explicit engine name. When ``None``, falls back to the
            ``$N4A_ENGINE`` environment variable, then :data:`DEFAULT_ENGINE`.

    Returns:
        The validated engine name. ``"dag-ml"`` (the default) and ``"legacy"`` are both runnable.

    Raises:
        ValueError: If the name is not one of :data:`ENGINES`.
        NotImplementedError: If the reserved-but-unimplemented ``"dual"`` engine is requested.
    """
    name = (engine or os.environ.get(ENGINE_ENV_VAR) or DEFAULT_ENGINE).strip().lower()
    if name not in ENGINES:
        raise ValueError(f"unknown nirs4all engine {name!r}; valid engines: {list(ENGINES)}")
    if name == "dual":
        raise NotImplementedError(
            "the 'dual' engine (side-by-side legacy vs dag-ml comparison) is not implemented yet; "
            "use 'legacy' or 'dag-ml' (see dag-ml/docs/migration-nirs4all/)"
        )
    return cast(Engine, name)
