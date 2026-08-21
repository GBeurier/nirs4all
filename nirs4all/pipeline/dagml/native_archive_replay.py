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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .fit_identity import normalize_predict_identity
from .methods_replay import MethodsN4mmReplayCallbacks, MethodsPortableReplayError
from .raw_replay_lowerer import RawArrayMethodsReplayCompiler, RawArrayMethodsReplayError

if TYPE_CHECKING:
    from .resolver import MaterializationResolver


class NativeArchiveReplayError(RuntimeError):
    """The native Archive V2 → Methods PREDICT boundary could not be executed."""


@dataclass(frozen=True)
class NativeArchiveConformalInterval:
    """One finite native conformal interval view returned by DAG-ML replay.

    The values are decoded from the exact interval block emitted by DAG-ML;
    this adapter neither calibrates nor recomputes endpoint arithmetic.
    """

    coverage: float
    lower: np.ndarray
    upper: np.ndarray
    qhat: float | np.ndarray
    calibration_fingerprint: str


@dataclass(frozen=True)
class NativeArchivePrediction:
    """One identity-aligned native Archive V2 PREDICT result."""

    values: np.ndarray
    sample_ids: tuple[str, ...]
    intervals: dict[float, NativeArchiveConformalInterval]
    conformal_guarantee_status: dict[str, Any] | None


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
        manifest, members = dag_ml.build_archive_v2_native_portable_payloads(
            archive_id, outcome, package
        )
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
        return outcome.to_dict()
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

    return predict_methods_archive_v2_raw_result(
        archive_path,
        X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        outcome_id=outcome_id,
        run_id=run_id,
    ).values


def predict_methods_archive_v2_raw_result(
    archive_path: str | Path,
    X: Any,
    *,
    sample_ids: Any,
    groups: Any = None,
    metadata: Any = None,
    outcome_id: str = "outcome:nirs4all.archive_predict",
    run_id: str = "run:nirs4all.archive_predict",
) -> NativeArchivePrediction:
    """Replay a native archive and retain its exact conformal result blocks.

    This is the public result-bearing form of
    :func:`predict_methods_archive_v2_raw`.  It exposes only interval blocks
    already materialized and validated by DAG-ML; it never recalibrates.
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
        return _decode_exact_raw_prediction(
            outcome,
            identity.sample_ids,
            package_document,
        )
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


def _decode_exact_raw_prediction(
    outcome: Any,
    sample_ids: tuple[str, ...],
    package: dict[str, Any],
) -> NativeArchivePrediction:
    """Decode one exact final block and any already-materialized intervals."""

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
    binding = outputs[0].get("binding")
    binding_id = binding.get("binding_id") if isinstance(binding, dict) else None
    intervals, guarantee_status = _decode_native_conformal_intervals(
        document,
        package,
        sample_ids=sample_ids,
        prediction_shape=values.shape,
        binding_id=binding_id,
    )
    return NativeArchivePrediction(
        values=values,
        sample_ids=sample_ids,
        intervals=intervals,
        conformal_guarantee_status=guarantee_status,
    )


def _decode_native_conformal_intervals(
    outcome: dict[str, Any],
    package: dict[str, Any],
    *,
    sample_ids: tuple[str, ...],
    prediction_shape: tuple[int, ...],
    binding_id: Any,
) -> tuple[dict[float, NativeArchiveConformalInterval], dict[str, Any] | None]:
    """Project DAG-ML's exact finite interval blocks into public result views."""

    calibration = package.get("conformal_calibration")
    raw_blocks = outcome.get("conformal_intervals", [])
    if calibration is None:
        if raw_blocks not in (None, []):
            raise NativeArchiveReplayError(
                "DAG-ML emitted conformal intervals without portable calibration state"
            )
        return {}, None
    if not isinstance(calibration, dict):
        raise NativeArchiveReplayError("Package V2 conformal calibration is not an object")
    if not isinstance(raw_blocks, list):
        raise NativeArchiveReplayError("DAG-ML conformal intervals are not an array")
    if not isinstance(binding_id, str) or not binding_id:
        raise NativeArchiveReplayError(
            "DAG-ML conformal replay output has no selected binding identity"
        )
    matching = [
        block
        for block in raw_blocks
        if isinstance(block, dict)
        and block.get("binding_id") == binding_id
        and block.get("sample_ids") == list(sample_ids)
    ]
    if len(matching) != 1 or len(raw_blocks) != 1:
        raise NativeArchiveReplayError(
            "DAG-ML conformal intervals do not exactly cover the selected prediction block"
        )
    block = matching[0]
    fingerprint = block.get("calibration_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != calibration.get("calibration_fingerprint"):
        raise NativeArchiveReplayError(
            "DAG-ML conformal interval calibration fingerprint does not match Package V2"
        )
    if block.get("point_prediction_fingerprint") in (None, ""):
        raise NativeArchiveReplayError(
            "DAG-ML conformal interval has no point-prediction fingerprint"
        )
    radii = _conformal_radii_by_coverage(calibration, prediction_shape[1])
    raw_intervals = block.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise NativeArchiveReplayError("DAG-ML conformal interval block is empty")
    intervals: dict[float, NativeArchiveConformalInterval] = {}
    for raw_interval in raw_intervals:
        if not isinstance(raw_interval, dict):
            raise NativeArchiveReplayError("DAG-ML conformal interval entry is not an object")
        coverage = raw_interval.get("coverage")
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0.0 < float(coverage) < 1.0:
            raise NativeArchiveReplayError("DAG-ML conformal interval has an invalid coverage")
        coverage = float(coverage)
        if coverage in intervals or coverage not in radii:
            raise NativeArchiveReplayError(
                "DAG-ML conformal interval coverage does not exactly match Package V2 quantiles"
            )
        cells = raw_interval.get("cells")
        if not isinstance(cells, list) or len(cells) != prediction_shape[0]:
            raise NativeArchiveReplayError("DAG-ML conformal interval rows do not match predictions")
        lower = np.empty(prediction_shape, dtype=float)
        upper = np.empty(prediction_shape, dtype=float)
        for row_index, row in enumerate(cells):
            if not isinstance(row, list) or len(row) != prediction_shape[1]:
                raise NativeArchiveReplayError("DAG-ML conformal interval cells are ragged")
            for column_index, cell in enumerate(row):
                if not isinstance(cell, dict):
                    raise NativeArchiveReplayError("DAG-ML conformal interval cell is not an object")
                if cell.get("status") == "unbounded":
                    raise NativeArchiveReplayError(
                        "native Archive V2 result cannot coerce an unbounded conformal interval into finite endpoints"
                    )
                if set(cell) != {"status", "lower", "upper"} or cell.get("status") != "finite":
                    raise NativeArchiveReplayError("DAG-ML conformal interval cell is malformed")
                lower_value, upper_value = cell.get("lower"), cell.get("upper")
                if (
                    not isinstance(lower_value, (int, float))
                    or isinstance(lower_value, bool)
                    or not isinstance(upper_value, (int, float))
                    or isinstance(upper_value, bool)
                    or not math.isfinite(float(lower_value))
                    or not math.isfinite(float(upper_value))
                    or float(lower_value) > float(upper_value)
                ):
                    raise NativeArchiveReplayError("DAG-ML conformal interval endpoints are invalid")
                lower[row_index, column_index] = float(lower_value)
                upper[row_index, column_index] = float(upper_value)
        intervals[coverage] = NativeArchiveConformalInterval(
            coverage=coverage,
            lower=lower,
            upper=upper,
            qhat=radii[coverage],
            calibration_fingerprint=fingerprint,
        )
    if set(intervals) != set(radii):
        raise NativeArchiveReplayError(
            "DAG-ML conformal interval coverages do not exactly close the Package V2 calibration"
        )
    status = {
        "version": 2,
        "status": "active",
        "method": "split_absolute_residual",
        "unit": "physical_sample",
        "coverage": sorted(intervals),
        "calibrated_coverages": sorted(intervals),
        "multi_target": calibration.get("multi_target_policy"),
        "calibration_fingerprint": fingerprint,
        "source": "dag_ml_portable_predictor_package_v2",
    }
    return intervals, status


def _conformal_radii_by_coverage(
    calibration: dict[str, Any], target_count: int
) -> dict[float, float | np.ndarray]:
    """Read finite native radii directly from the persisted calibration."""

    quantiles = calibration.get("quantiles")
    policy = calibration.get("multi_target_policy")
    if not isinstance(quantiles, list) or not quantiles:
        raise NativeArchiveReplayError("Package V2 conformal calibration has no quantiles")
    if policy not in {"marginal", "joint_max"}:
        raise NativeArchiveReplayError("Package V2 conformal calibration has an unsupported policy")
    radii_by_coverage: dict[float, float | np.ndarray] = {}
    for quantile in quantiles:
        if not isinstance(quantile, dict):
            raise NativeArchiveReplayError("Package V2 conformal quantile is not an object")
        coverage = quantile.get("coverage")
        raw_radii = quantile.get("radii")
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0.0 < float(coverage) < 1.0:
            raise NativeArchiveReplayError("Package V2 conformal quantile has an invalid coverage")
        coverage = float(coverage)
        expected_count = target_count if policy == "marginal" else 1
        if coverage in radii_by_coverage or not isinstance(raw_radii, list) or len(raw_radii) != expected_count:
            raise NativeArchiveReplayError("Package V2 conformal quantile radii do not match its policy")
        values: list[float] = []
        for radius in raw_radii:
            if not isinstance(radius, dict) or set(radius) != {"status", "value"} or radius.get("status") != "finite":
                raise NativeArchiveReplayError(
                    "native Archive V2 result cannot coerce an unbounded conformal radius"
                )
            value = radius.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise NativeArchiveReplayError("Package V2 conformal radius is invalid")
            values.append(float(value))
        radii_by_coverage[coverage] = values[0] if policy == "joint_max" else np.asarray(values)
    return radii_by_coverage


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
    "NativeArchiveConformalInterval",
    "NativeArchivePrediction",
    "NativeArchiveReplayError",
    "predict_methods_archive_v2_raw",
    "predict_methods_archive_v2_raw_result",
    "replay_methods_archive_v2",
    "write_methods_archive_v2",
]
