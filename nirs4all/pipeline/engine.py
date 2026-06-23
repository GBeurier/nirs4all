"""Execution-engine selector for the nirs4all core (ADR-17 cutover/rollback skeleton).

Seam for the **nirs4all-core → dag-ml** migration. The default production engine is the in-process
*legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`). The ``"dag-ml"`` engine is
**wired**: :func:`nirs4all.run` dispatches it to the dag-ml-cli backend
(:mod:`nirs4all.pipeline.dagml.run_backend`), which runs the pipeline natively (Rust) and returns a
``RunResult`` of dag-ml's native scores. The side-by-side comparison mode (``"dual"``) is still
reserved and :func:`resolve_engine` refuses it with a clear ``NotImplementedError``.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`. The
default stays ``legacy`` (ADR-17): production behaviour is unchanged unless ``engine="dag-ml"`` is
explicitly requested. See ``dag-ml/docs/migration-nirs4all/``.
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
    if name == "dual":
        raise NotImplementedError(
            "the 'dual' engine (side-by-side legacy vs dag-ml comparison) is not implemented yet; "
            "use 'legacy' or 'dag-ml' (see dag-ml/docs/migration-nirs4all/)"
        )
    return cast(Engine, name)
