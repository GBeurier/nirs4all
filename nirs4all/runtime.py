"""Public runtime seam for nirs4all — the ``inspect`` accessor + the ``RtResult`` envelope re-exports (L10).

The small **public** surface other runtimes (Studio REST, Web WASM, the CLI) consume so they never reach
into the private ``nirs4all.pipeline.dagml`` package:

* :func:`list_controller_manifests` — the ``inspect`` verb's controller-manifest list (over the existing
  :func:`nirs4all.pipeline.dagml_bridge.controller_manifests`, already JSON-ready + shaped to dag-ml's
  ``controller_manifest.v1.schema.json``); Studio's ``GET /api/operators/manifests`` proxies it.
* :class:`RtResult` / :class:`RtRunRequest` / :class:`RtError` + :func:`from_native_dir` — the runtime
  envelopes (``LOCK-RT``), re-exported from :mod:`nirs4all.pipeline.dagml.rt` so callers import them from a
  stable public path.

This is the V1 public seam only — NOT the eventual consolidated ``nirs4all/runtime/`` namespace or the
published contracts package (those are deferred to GOV / ``LOCK-REL``). It adds no new behaviour: every
function here forwards to an existing surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nirs4all.pipeline.dagml.rt import RtError, RtResult, RtRunRequest

__all__ = [
    "RtError",
    "RtResult",
    "RtRunRequest",
    "from_native_dir",
    "list_controller_manifests",
]


def list_controller_manifests() -> list[dict[str, Any]]:
    """The host-controller manifests (the ``inspect`` verb surface) as JSON-ready dicts.

    Forwards :func:`nirs4all.pipeline.dagml_bridge.controller_manifests` verbatim — the static kind-level
    set (``transform`` / ``y_transform`` / ``model`` / ``prediction_join`` / ``meta_model``), each already
    shaped to dag-ml's ``controller_manifest.v1.schema.json``. The per-operator manifest ledger waits on the
    CTRL-000 ``OperatorController → ControllerManifest`` adapter; until then this is the authoritative
    kind-level view. Returns a fresh list each call (the bridge builds it on demand).
    """
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    return controller_manifests()


def from_native_dir(run_dir: str | Path) -> RtResult:
    """Project a native results directory into an :class:`RtResult` (the public seam over the dag-ml reader).

    Thin re-export of :meth:`RtResult.from_native_dir` so a consumer reads a native run
    (``manifest.json`` + ``score_set.json`` + ``predictions.parquet``) into the runtime envelope without
    importing the private ``pipeline.dagml`` package. Pure projection (hash-validated read, no recompute).
    """
    return RtResult.from_native_dir(run_dir)
