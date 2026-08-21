"""Methods-only PREDICT replay from a validated Core Archive V2.

This is intentionally a composition boundary, not a second package or model
reader: Core validates and returns the opaque Package V2 member; DAG-ML parses
and validates the package and owns replay; the official ``pls4all`` binding
imports N4MM only for the invocation.  No legacy ``PipelineRunner`` is
consulted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .methods_replay import MethodsN4mmReplayCallbacks, MethodsPortableReplayError

if TYPE_CHECKING:
    from .resolver import MaterializationResolver


class NativeArchiveReplayError(RuntimeError):
    """The native Archive V2 → Methods PREDICT boundary could not be executed."""


def replay_methods_archive_v2(
    archive_path: str | Path,
    request: Any,
    data_envelopes: Any,
    resolver: MaterializationResolver,
    *,
    outcome_id: str,
    run_id: str,
    fallback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay a portable Methods Package V2 stored in a Core Archive V2.

    ``request`` and ``data_envelopes`` remain strict DAG-ML contracts supplied
    by the caller.  In particular, a fresh target-free cohort is represented by
    a signed current envelope, never by a reused training relation or fabricated
    target hash.  Archive contents are validated before this function sees any
    Package bytes; Package/replay validation runs before a host data callback.
    """

    try:
        from nirs4all_core import read_portable_predictor_package_v2
    except ImportError as error:  # pragma: no cover - depends on optional wheel
        raise NativeArchiveReplayError(
            "native Archive V2 replay requires nirs4all-core >= 0.3.14"
        ) from error
    try:
        import dag_ml
    except ImportError as error:  # pragma: no cover - depends on optional wheel
        raise NativeArchiveReplayError(
            "native Archive V2 replay requires dag-ml with portable artifact callbacks"
        ) from error

    package_bytes = read_portable_predictor_package_v2(str(archive_path))
    if not isinstance(package_bytes, bytes):
        raise NativeArchiveReplayError("Core Archive V2 reader did not return package bytes")
    try:
        package_json = package_bytes.decode("utf-8")
        package = dag_ml.PortablePredictorPackage(package_json)
        package_document = package.to_dict()
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise NativeArchiveReplayError(
            "Core Archive V2 package member is not a validated DAG-ML Package V2"
        ) from error
    if package_document.get("schema_version") != 2:
        raise NativeArchiveReplayError("Methods archive replay requires PortablePredictorPackage V2")
    target_names_by_node = _target_names_by_node(package_document)
    callbacks = MethodsN4mmReplayCallbacks(
        resolver,
        target_names_by_node=target_names_by_node,
        fallback=fallback,
    )
    try:
        outcome = dag_ml.replay_loaded_predictor_package(
            package,
            request,
            data_envelopes,
            {},
            callbacks.op_callback,
            outcome_id=outcome_id,
            run_id=run_id,
            artifact_callback=callbacks.artifact_callback,
        )
        if callbacks.active_handle_count:
            raise NativeArchiveReplayError(
                "DAG-ML replay returned while native Methods handles were still retained"
            )
        return outcome.to_dict()
    except MethodsPortableReplayError as error:
        raise NativeArchiveReplayError(str(error)) from error
    finally:
        callbacks.close()


def _target_names_by_node(package: dict[str, Any]) -> dict[str, list[str]]:
    """Derive one unambiguous target schema per executable Methods node."""

    bindings = package.get("output_bindings")
    if not isinstance(bindings, list):
        raise NativeArchiveReplayError("Package V2 has no output binding list")
    targets: dict[str, list[str]] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise NativeArchiveReplayError("Package V2 output binding is not an object")
        node_id = binding.get("node_id")
        target_names = binding.get("target_names")
        if not isinstance(node_id, str) or not isinstance(target_names, list) or not all(
            isinstance(target_name, str) and target_name for target_name in target_names
        ):
            raise NativeArchiveReplayError("Package V2 output binding lacks a target schema")
        previous = targets.setdefault(node_id, list(target_names))
        if previous != target_names:
            raise NativeArchiveReplayError(
                f"Package V2 has incompatible target schemas for node `{node_id}`"
            )
    if not targets:
        raise NativeArchiveReplayError("Package V2 has no replayable output bindings")
    return targets


__all__ = ["NativeArchiveReplayError", "replay_methods_archive_v2"]
