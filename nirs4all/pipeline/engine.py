"""Execution-engine selector for the nirs4all core (interim legacy-default posture).

Seam for the **nirs4all-core → dag-ml** migration. The default production engine is **legacy**: the
public-maintained nirs4all stays pure-Python by default, so :func:`nirs4all.run` runs through the
in-process *legacy* orchestrator (:class:`~nirs4all.pipeline.PipelineRunner`) unless another engine is
selected. The dag-ml backend (:mod:`nirs4all.pipeline.dagml.run_backend`) — which runs the pipeline
natively (Rust) and returns a ``RunResult`` of dag-ml's native scores, with a transparent fallback to
the legacy orchestrator for any shape it cannot yet honor — stays **fully selectable** via
``engine="dag-ml"`` or ``$N4A_ENGINE=dag-ml``; the whole dag-ml integration (in-process path, native
generator coverage, conformance pack, hard dependency) is intact and runnable out of the box.

This is the interim posture: the maintainer keeps the public Python version as the default until the
planned global refactoring lands; at that point the legacy-DROP cutover flips the default back to
dag-ml (the ADR-17 end state). The side-by-side comparison mode (``"dual"``) is reserved and
:func:`resolve_engine` refuses it with a clear ``NotImplementedError``.

Selection precedence: explicit argument > ``$N4A_ENGINE`` env var > :data:`DEFAULT_ENGINE`
(``legacy``, interim). Pass ``engine="dag-ml"`` (or ``$N4A_ENGINE=dag-ml``) to run on the dag-ml
backend. See ``dag-ml/docs/migration-nirs4all/``.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal, cast

Engine = Literal["legacy", "dag-ml", "dual"]

DEFAULT_ENGINE: Engine = "legacy"
ENGINE_ENV_VAR = "N4A_ENGINE"
ENGINES: tuple[Engine, ...] = ("legacy", "dag-ml", "dual")


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
        The validated engine name. ``"legacy"`` (the default) and ``"dag-ml"`` are both runnable.

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


def require_legacy_engine(operation: str, engine: str | None = None) -> Engine:
    """Resolve an API backend selector and reject operations not yet backed by dag-ml."""
    selected = resolve_engine(engine)
    if selected == "dag-ml":
        env_requested = (os.environ.get(ENGINE_ENV_VAR) or "").strip().lower()
        if engine is None and env_requested == "dag-ml":
            warnings.warn(
                f"{ENGINE_ENV_VAR}=dag-ml is ignored for nirs4all.{operation} in this transition "
                "release because this helper does not have a dag-ml execution path yet; using "
                "engine='legacy'. Pass engine='dag-ml' explicitly to fail fast.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "legacy"
        raise NotImplementedError(
            f"nirs4all.{operation} does not have a dag-ml execution path yet; "
            "use engine='legacy' for this transition release. nirs4all.run supports "
            "engine='dag-ml' with documented fallback boundaries."
        )
    return selected
