"""Methods-only PREDICT replay from a validated Core Archive V2.

This is intentionally a composition boundary, not a second package or model
reader: Core validates and returns the opaque Package V2 member; DAG-ML parses
and validates the package and owns replay; the official ``pls4all`` binding
imports N4MM only for the invocation.  No legacy ``PipelineRunner`` is
consulted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .fit_identity import normalize_predict_identity
from .methods_replay import MethodsN4mmReplayCallbacks, MethodsPortableReplayError
from .methods_runtime import resolve_methods_library_path
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
        methods_library_path=resolve_methods_library_path(),
    )
    try:
        replay = compiler.compile_replay(
            None, X, mode="predict", identity_frame=identity
        )
        outcome = _execute_compiled_methods_replay(dag_ml, package, replay)
        return _decode_exact_raw_prediction(outcome, identity.sample_ids)
    except (MethodsPortableReplayError, RawArrayMethodsReplayError) as error:
        raise NativeArchiveReplayError(str(error)) from error
    finally:
        if "replay" in locals() and replay.cleanup is not None:
            replay.cleanup()


def project_methods_archive_v2_conformal_presentation(
    archive_path: str | Path,
    X: Any,
    *,
    sample_ids: Any,
    groups: Any = None,
    metadata: Any = None,
    outcome_id: str = "outcome:nirs4all.archive_conformal_presentation",
    run_id: str = "run:nirs4all.archive_conformal_presentation",
) -> dict[str, Any]:
    """Return DAG-ML's exact scalar conformal presentation for one replay.

    This is a transport projection, not a second conformal implementation:
    DAG-ML owns interval construction and its provenance validation.  The
    adapter verifies only that the returned presentation closes exactly over
    the freshly replayed point block and current sample identities.
    """

    dag_ml, package, package_document = _load_methods_archive_package(archive_path)
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
        methods_library_path=resolve_methods_library_path(),
    )
    try:
        replay = compiler.compile_replay(None, X, mode="predict", identity_frame=identity)
        outcome = _execute_compiled_methods_replay(dag_ml, package, replay)
        projector = getattr(dag_ml, "build_conformal_presentation_v1", None)
        if not callable(projector):
            raise NativeArchiveReplayError(
                "installed DAG-ML lacks the native conformal presentation projector; upgrade DAG-ML"
            )
        presentation = projector(package, replay.request, outcome)
        _validate_conformal_presentation_transport(
            presentation,
            package_document=package_document,
            outcome=outcome,
            sample_ids=identity.sample_ids,
        )
        return cast(dict[str, Any], presentation)
    except (MethodsPortableReplayError, RawArrayMethodsReplayError) as error:
        raise NativeArchiveReplayError(str(error)) from error
    finally:
        if "replay" in locals() and replay.cleanup is not None:
            replay.cleanup()


def _execute_compiled_methods_replay(dag_ml: Any, package: Any, replay: Any) -> Any:
    """Execute a raw Methods replay through the no-callback DAG-ML entry point.

    The raw compiler carries a library path and a full current cohort precisely
    so DAG-ML can materialize its own data view. Passing that contract through
    the generic callback path loses the cohort identities and can only work by
    importing a Python-side N4MM decoder. Public Archive V2 replay must instead
    use the registered native controller directly.
    """

    if replay.methods_inputs is None or replay.methods_library_path is None:
        raise NativeArchiveReplayError(
            "raw Methods Archive V2 replay requires a current native input cohort and libn4m path"
        )
    execute = getattr(dag_ml, "replay_loaded_methods_predictor_package", None)
    if not callable(execute):
        raise NativeArchiveReplayError(
            "installed DAG-ML lacks no-callback Methods package replay; upgrade DAG-ML"
        )
    return execute(
        package,
        replay.request,
        replay.data_envelopes,
        replay.methods_inputs,
        methods_library_path=replay.methods_library_path,
        outcome_id=replay.outcome_id,
        run_id=replay.run_id,
        warnings=replay.warnings,
        diagnostics=replay.diagnostics,
    )


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


def _validate_conformal_presentation_transport(
    presentation: Any,
    *,
    package_document: dict[str, Any],
    outcome: Any,
    sample_ids: tuple[str, ...],
) -> None:
    """Prove a presentation is the exact selected scalar replay output."""

    required = {
        "schema_version",
        "package_fingerprint",
        "replay_outcome_fingerprint",
        "binding_id",
        "target_name",
        "sample_ids",
        "point_predictions",
        "intervals",
        "calibration_fingerprint",
        "presentation_fingerprint",
    }
    if not isinstance(presentation, dict) or set(presentation) != required:
        raise NativeArchiveReplayError("DAG-ML conformal presentation has an unsupported shape")
    if presentation.get("schema_version") != 1 or presentation.get("sample_ids") != list(sample_ids):
        raise NativeArchiveReplayError(
            "DAG-ML conformal presentation identities do not match the current cohort"
        )
    calibration = package_document.get("conformal_calibration")
    if not isinstance(calibration, dict):
        raise NativeArchiveReplayError(
            "DAG-ML produced a conformal presentation without Package V2 calibration"
        )
    if (
        presentation.get("package_fingerprint") != package_document.get("package_fingerprint")
        or presentation.get("calibration_fingerprint")
        != calibration.get("calibration_fingerprint")
    ):
        raise NativeArchiveReplayError(
            "DAG-ML conformal presentation provenance does not match Package V2"
        )
    document = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if not isinstance(document, dict) or presentation.get("replay_outcome_fingerprint") != document.get(
        "outcome_fingerprint"
    ):
        raise NativeArchiveReplayError(
            "DAG-ML conformal presentation provenance does not match replay"
        )
    outputs = document.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise NativeArchiveReplayError("DAG-ML conformal presentation requires one replay output")
    binding = outputs[0].get("binding")
    blocks = outputs[0].get("predictions")
    if (
        not isinstance(binding, dict)
        or presentation.get("binding_id") != binding.get("binding_id")
        or not isinstance(blocks, list)
        or len(blocks) != 1
        or not isinstance(blocks[0], dict)
        or blocks[0].get("sample_ids") != list(sample_ids)
    ):
        raise NativeArchiveReplayError(
            "DAG-ML conformal presentation does not close its selected point block"
        )
    values = blocks[0].get("values")
    points = presentation.get("point_predictions")
    if (
        not isinstance(values, list)
        or any(not isinstance(row, list) or len(row) != 1 for row in values)
        or points != [row[0] for row in values]
    ):
        raise NativeArchiveReplayError(
            "DAG-ML conformal presentation points are not the exact replay values"
        )
    intervals = presentation.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise NativeArchiveReplayError("DAG-ML conformal presentation has no interval blocks")
    seen_coverages: set[float] = set()
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {"coverage", "lower", "upper", "qhat"}:
            raise NativeArchiveReplayError("DAG-ML conformal presentation interval is malformed")
        coverage = interval.get("coverage")
        lower, upper = interval.get("lower"), interval.get("upper")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not 0.0 < float(coverage) < 1.0
            or float(coverage) in seen_coverages
            or not isinstance(lower, list)
            or not isinstance(upper, list)
            or len(lower) != len(sample_ids)
            or len(upper) != len(sample_ids)
        ):
            raise NativeArchiveReplayError("DAG-ML conformal presentation interval coverage is invalid")
        seen_coverages.add(float(coverage))
        for point, lower_value, upper_value in zip(points, lower, upper, strict=True):
            if lower_value is None or upper_value is None:
                continue
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in (point, lower_value, upper_value)
            ) or float(lower_value) > float(point) or float(point) > float(upper_value):
                raise NativeArchiveReplayError("DAG-ML conformal presentation interval does not contain its point")


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
    "project_methods_archive_v2_conformal_presentation",
    "replay_methods_archive_v2",
    "validate_methods_archive_v2",
    "write_methods_archive_v2",
]
