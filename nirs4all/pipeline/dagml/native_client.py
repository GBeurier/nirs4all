"""Thin native training/replay client for the DAG-ML Python facade.

This module is the nirs4all-side seam for roadmap W2-PY.  It deliberately does
not compile nirs4all pipelines, run Python fallback logic, or reimplement any
DAG-ML contract validation.  Its only responsibility is to load the installed
``dag_ml`` facade, check that the required native entry points exist, and call
them with the already-built JSON/control-plane objects.  Contract proofs such
as DAG-ML D10 ``cache_namespace_fingerprints`` are forwarded as part of those
objects; nirs4all does not recalculate, normalize or filter them.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .errors import DagMlUnavailable, DagMlUnsupported


class DagMLNativeCoverageError(DagMlUnsupported):
    """The installed DAG-ML facade lacks a native capability required by nirs4all.

    This is a coverage error, not an availability error: ``dag_ml`` may import
    successfully while still being too old to expose the training/replay
    contracts needed by the W2 native path.  Keeping this type catchable through
    ``DagMlUnsupported`` preserves the current fail-loud fallback boundary until
    P7 removes fallback for covered shapes.
    """


class _DagMLFacade(Protocol):
    """Runtime subset of the ``dag_ml`` facade consumed by this client."""

    def contract_manifest_json(self) -> str: ...

    def execute_training(
        self,
        request: Any,
        data_envelopes: Any,
        relations: Any,
        training_influence: Any,
        op_callback: Any,
        *,
        outcome_id: str,
        run_id: str,
        bundle_id: str,
        warnings: Any = (),
        diagnostics: Any = None,
    ) -> Any: ...

    def replay_loaded_predictor_package(
        self,
        package: Any,
        request: Any,
        data_envelopes: Any,
        artifact_handles: Any,
        op_callback: Any,
        *,
        outcome_id: str,
        run_id: str,
        warnings: Any = (),
        diagnostics: Any = None,
    ) -> Any: ...


@dataclass(frozen=True)
class DagMLNativeCapabilities:
    """Installed DAG-ML native capabilities visible from nirs4all."""

    package_version: str | None
    contract_manifest: dict[str, Any] | None
    training: bool
    loaded_predictor_replay: bool


class DagMLNativeClient:
    """Minimal adapter around the installed ``dag_ml`` Python facade.

    The constructor is side-effect free.  Importing ``dag_ml`` is deferred until
    :meth:`capabilities` or a native call is made, so importing nirs4all itself
    does not require the DAG-ML wheel to be importable.
    """

    def __init__(self, module_name: str = "dag_ml") -> None:
        self._module_name = module_name
        self._facade: _DagMLFacade | None = None

    def capabilities(self) -> DagMLNativeCapabilities:
        """Return the installed native training/replay capability snapshot."""

        facade = self._load_facade()
        manifest = self._contract_manifest(facade)
        package_version = None
        if manifest is not None:
            raw_version = manifest.get("python_package_version")
            package_version = raw_version if isinstance(raw_version, str) else None
        return DagMLNativeCapabilities(
            package_version=package_version,
            contract_manifest=manifest,
            training=self._has_callable(facade, "execute_training"),
            loaded_predictor_replay=self._has_callable(facade, "replay_loaded_predictor_package"),
        )

    def execute_training(
        self,
        request: Any,
        data_envelopes: Any,
        relations: Any,
        training_influence: Any,
        op_callback: Any,
        *,
        outcome_id: str,
        run_id: str,
        bundle_id: str,
        warnings: Any = (),
        diagnostics: Any = None,
    ) -> Any:
        """Execute native DAG-ML training through the installed facade.

        DAG-ML owns validation of any cache namespace proofs embedded in the
        request, bundle or payload objects.  This seam forwards them unchanged.
        """

        facade = self._require_callable("execute_training")
        return facade.execute_training(
            request,
            data_envelopes,
            relations,
            training_influence,
            op_callback,
            outcome_id=outcome_id,
            run_id=run_id,
            bundle_id=bundle_id,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def replay_loaded_predictor_package(
        self,
        package: Any,
        request: Any,
        data_envelopes: Any,
        artifact_handles: Any,
        op_callback: Any,
        *,
        outcome_id: str,
        run_id: str,
        warnings: Any = (),
        diagnostics: Any = None,
    ) -> Any:
        """Replay a loaded portable predictor package through the installed facade.

        ``request`` is intentionally forwarded unchanged: DAG-ML owns the
        canonical validation of ``phase`` and currently accepts native package
        replay for ``PREDICT`` and ``EXPLAIN``.  The same boundary applies to
        D10 cache namespace proofs embedded in replay payloads or requests.
        """

        facade = self._require_callable("replay_loaded_predictor_package")
        return facade.replay_loaded_predictor_package(
            package,
            request,
            data_envelopes,
            artifact_handles,
            op_callback,
            outcome_id=outcome_id,
            run_id=run_id,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _load_facade(self) -> _DagMLFacade:
        if self._facade is not None:
            return self._facade
        try:
            module = importlib.import_module(self._module_name)
        except ImportError as error:
            raise DagMlUnavailable(f"native DAG-ML Python facade '{self._module_name}' is not importable") from error
        self._facade = cast(_DagMLFacade, module)
        return self._facade

    def _require_callable(self, name: str) -> _DagMLFacade:
        facade = self._load_facade()
        if not self._has_callable(facade, name):
            raise DagMLNativeCoverageError(f"native DAG-ML facade '{self._module_name}' does not expose {name}(); install a DAG-ML build with W1 training/replay bindings")
        return facade

    @staticmethod
    def _has_callable(facade: _DagMLFacade, name: str) -> bool:
        return callable(getattr(facade, name, None))

    @staticmethod
    def _contract_manifest(facade: _DagMLFacade) -> dict[str, Any] | None:
        manifest_json = getattr(facade, "contract_manifest_json", None)
        if not callable(manifest_json):
            return None
        manifest = json.loads(manifest_json())
        if not isinstance(manifest, dict):
            raise DagMLNativeCoverageError("DAG-ML contract manifest must be a JSON object")
        return manifest


def native_client() -> DagMLNativeClient:
    """Create the default native DAG-ML client."""

    return DagMLNativeClient()


__all__ = [
    "DagMLNativeCapabilities",
    "DagMLNativeClient",
    "DagMLNativeCoverageError",
    "native_client",
]
