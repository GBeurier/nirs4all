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
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .fit_identity import normalize_predict_identity
from .methods_replay import MethodsN4mmReplayCallbacks, MethodsPortableReplayError
from .raw_replay_lowerer import RawArrayMethodsReplayCompiler, RawArrayMethodsReplayError

if TYPE_CHECKING:
    from .resolver import MaterializationResolver


class NativeArchiveReplayError(RuntimeError):
    """The native Archive V2 → Methods PREDICT boundary could not be executed."""


def validate_methods_archive_v2(archive_path: str | Path) -> None:
    """Validate a Core Archive V2 and its DAG-ML Package V2 without replaying.

    This intentionally performs no Methods hydration.  It is used by the
    public native session constructor so a successfully opened session has
    already crossed the Core/DAG-ML schema and integrity boundary.
    """

    _load_methods_archive_package(archive_path)


def write_methods_archive_v2(
    archive_path: str | Path,
    *,
    archive_id: str,
    outcome: Any,
    package: Any,
) -> dict[str, str]:
    """Persist one native Methods Package V2 as a Core Archive V2.

    DAG-ML alone assembles the closed manifest and the six companion members;
    Core alone validates and writes the ZIP.  This boundary deliberately does
    not accept arbitrary members, retrofit host sidecars, or calculate member
    hashes in nirs4all.
    """

    try:
        import dag_ml
    except ImportError as error:  # pragma: no cover - depends on optional wheel
        raise NativeArchiveReplayError(
            "native Archive V2 writing requires the DAG-ML Python facade"
        ) from error
    try:
        from nirs4all_core import write_archive_v2_from_native_payloads
    except ImportError as error:  # pragma: no cover - depends on optional wheel
        raise NativeArchiveReplayError(
            "native Archive V2 writing requires an nirs4all-core wheel with the Archive V2 writer"
        ) from error
    try:
        assemble = getattr(dag_ml, "build_archive_v2_native_portable_payloads", None)
        if not callable(assemble):
            raise NativeArchiveReplayError(
                "installed DAG-ML lacks native Archive V2 payload assembly; upgrade DAG-ML"
            )
        manifest, members = assemble(archive_id, outcome, package)
        reference = write_archive_v2_from_native_payloads(
            str(archive_path), manifest, members
        )
    except Exception as error:  # DAG-ML/Core expose distinct native error subclasses.
        raise NativeArchiveReplayError(
            "DAG-ML/Core refused the native Archive V2 write inputs"
        ) from error
    if not isinstance(reference, dict) or set(reference) != {"archive_id", "archive_sha256"}:
        raise NativeArchiveReplayError("Core Archive V2 writer returned an invalid reference")
    if reference["archive_id"] != archive_id:
        raise NativeArchiveReplayError("Core Archive V2 writer returned a mismatched archive id")
    return {"archive_id": str(reference["archive_id"]), "archive_sha256": str(reference["archive_sha256"])}


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

    dag_ml, package, package_document = _load_methods_archive_package(archive_path)
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
        document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
        if not isinstance(document, dict):
            raise NativeArchiveReplayError("DAG-ML replay did not return an outcome object")
        return cast(dict[str, Any], document)
    except MethodsPortableReplayError as error:
        raise NativeArchiveReplayError(str(error)) from error
    finally:
        callbacks.close()


def predict_methods_archive_v2_raw(
    archive_path: str | Path,
    X: Any,
    *,
    sample_ids: Any,
    groups: Any = None,
    metadata: Any = None,
    outcome_id: str = "outcome:nirs4all.archive_predict",
    run_id: str = "run:nirs4all.archive_predict",
) -> np.ndarray:
    """Predict a fresh raw-array cohort from a Core Archive V2 Methods package.

    This is the first public composition that creates a current cohort replay
    request.  It requires explicit stable identities, signs the request through
    DAG-ML and delegates N4MM import/prediction to ``pls4all``.  It never
    invokes a legacy pipeline or fabricates target content.
    """

    dag_ml, package, _package_document = _load_methods_archive_package(archive_path)
    identity = normalize_predict_identity(
        X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        require_explicit_sample_ids=True,
    )
    compiler = RawArrayMethodsReplayCompiler(
        package,
        outcome_id=outcome_id,
        run_id=run_id,
    )
    try:
        replay = compiler.compile_replay(
            None, X, mode="predict", identity_frame=identity
        )
        outcome = dag_ml.replay_loaded_predictor_package(
            package,
            replay.request,
            replay.data_envelopes,
            replay.artifact_handles,
            replay.op_callback,
            outcome_id=replay.outcome_id,
            run_id=replay.run_id,
            artifact_callback=replay.artifact_callback,
        )
        return _decode_exact_raw_prediction(outcome, identity.sample_ids)
    except (MethodsPortableReplayError, RawArrayMethodsReplayError) as error:
        raise NativeArchiveReplayError(str(error)) from error
    finally:
        if "replay" in locals() and replay.cleanup is not None:
            replay.cleanup()


def _load_methods_archive_package(
    archive_path: str | Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load opaque V2 bytes through Core, then validate the DAG-ML package."""

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
    return dag_ml, package, package_document


def _decode_exact_raw_prediction(outcome: Any, sample_ids: tuple[str, ...]) -> np.ndarray:
    """Return one exact final prediction block for the requested cohort order."""

    document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if not isinstance(document, dict):
        raise NativeArchiveReplayError("DAG-ML replay did not return an outcome object")
    outputs = document.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise NativeArchiveReplayError("raw Methods archive replay requires exactly one output")
    blocks = outputs[0].get("predictions")
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], dict):
        raise NativeArchiveReplayError("raw Methods archive replay requires exactly one final prediction block")
    block = blocks[0]
    if block.get("sample_ids") != list(sample_ids):
        raise NativeArchiveReplayError(
            "DAG-ML replay prediction identities do not exactly match the current cohort"
        )
    values = np.asarray(block.get("values"), dtype=float)
    if values.ndim != 2 or values.shape[0] != len(sample_ids) or not np.isfinite(values).all():
        raise NativeArchiveReplayError("DAG-ML replay prediction values are not a finite aligned matrix")
    return cast(np.ndarray, values)


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


__all__ = [
    "NativeArchiveReplayError",
    "predict_methods_archive_v2_raw",
    "replay_methods_archive_v2",
    "validate_methods_archive_v2",
    "write_methods_archive_v2",
]
