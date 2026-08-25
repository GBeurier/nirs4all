"""Identity-bound calibration replay inputs for the native Methods lane.

This module only compiles host data into the existing DAG-ML PREDICT and
conformal-truth contracts.  It never computes residuals, quantiles,
fingerprints, or interval bounds: those remain owned by DAG-ML.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .estimator import DagMLReplayExecution
from .fit_identity import DagMLCalibrationIdentityFrame, normalize_calibration_identity
from .raw_replay_lowerer import (
    RawArrayMethodsReplayCompiler,
    RawArrayMethodsReplayError,
    _single_output_binding,
    validate_native_methods_package,
)


class NativeConformalCalibrationError(RuntimeError):
    """A raw calibration cohort cannot be represented by the native lane."""


@dataclass(frozen=True)
class NativeConformalCalibrationReplay:
    """One native PREDICT replay plus identity-keyed calibration truth.

    ``truth`` is intentionally a JSON-shaped value for
    ``dag_ml.TrainingResult.attach_conformal_calibration``.  It has no
    provenance fields: DAG-ML derives and validates those from the source
    outcome, replay and relation set at attachment time.
    """

    execution: DagMLReplayExecution
    binding_id: str
    calibration_relations: dict[str, Any]
    truth: dict[str, list[Any]]


def compile_methods_conformal_calibration_replay(
    package: Any,
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[Any] | None,
    groups: Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    methods_library_path: str | None = None,
    dagml_module: str = "dag_ml",
) -> NativeConformalCalibrationReplay:
    """Compile a finite, explicitly identified calibration cohort.

    The replay itself remains PREDICT, so the Methods provider never receives
    targets as execution inputs.  Its envelope nevertheless carries the
    measured-target fingerprint required by DAG-ML, and the same stable sample
    ids occur in the point prediction request and in ``truth``.
    """

    if sample_ids is None:
        raise NativeConformalCalibrationError(
            "native conformal calibration requires explicit sample_ids"
        )
    values = np.ascontiguousarray(np.asarray(X, dtype=float))
    if values.ndim != 2 or values.shape[0] == 0:
        raise NativeConformalCalibrationError(
            "native conformal calibration X must be a non-empty two-dimensional matrix"
        )
    if not np.isfinite(values).all():
        raise NativeConformalCalibrationError(
            "native conformal calibration X contains a non-finite value"
        )
    truth_values = np.ascontiguousarray(np.asarray(y, dtype=float))
    if truth_values.ndim == 1:
        truth_values = truth_values.reshape(-1, 1)
    if truth_values.ndim != 2 or truth_values.shape[0] != values.shape[0]:
        raise NativeConformalCalibrationError(
            "native conformal calibration y must be a row-aligned one- or two-dimensional matrix"
        )
    if not np.isfinite(truth_values).all():
        raise NativeConformalCalibrationError(
            "native conformal calibration y contains a non-finite value"
        )

    try:
        identity_frame = normalize_calibration_identity(
            values,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=True,
        )
        document = validate_native_methods_package(package)
        binding = _single_output_binding(document)
    except (RawArrayMethodsReplayError, TypeError, ValueError) as error:
        raise NativeConformalCalibrationError(str(error)) from error
    target_names = binding["target_names"]
    if truth_values.shape[1] != len(target_names):
        raise NativeConformalCalibrationError(
            "native conformal calibration y width does not match the portable output binding"
        )

    compiler = RawArrayMethodsReplayCompiler(
        document,
        dagml_module=dagml_module,
        methods_library_path=methods_library_path,
        outcome_id="outcome:nirs4all.raw_calibration_predict",
        run_id="run:nirs4all.raw_calibration_predict",
        request_id="replay:nirs4all.raw_calibration_predict",
    )
    try:
        execution = compiler.compile_replay(
            None,
            values,
            mode="predict",
            identity_frame=identity_frame,
        )
    except RawArrayMethodsReplayError as error:
        raise NativeConformalCalibrationError(str(error)) from error
    return NativeConformalCalibrationReplay(
        execution=execution,
        binding_id=binding["binding_id"],
        calibration_relations=_calibration_relations(execution, identity_frame),
        truth={
            "sample_ids": list(identity_frame.sample_ids),
            "values": truth_values.tolist(),
        },
    )


def _calibration_relations(
    execution: DagMLReplayExecution,
    identity_frame: DagMLCalibrationIdentityFrame,
) -> dict[str, Any]:
    """Extract the one relation authority shared by every replay envelope."""

    relations = [
        dict(cast(Mapping[str, Any], envelope.get("coordinator_relations")))
        for envelope in execution.data_envelopes.values()
        if isinstance(envelope, Mapping)
        and isinstance(envelope.get("coordinator_relations"), Mapping)
    ]
    if not relations or len(relations) != len(execution.data_envelopes):
        raise NativeConformalCalibrationError(
            "native calibration replay has no coordinator relation authority"
        )
    first = relations[0]
    if any(relation != first for relation in relations[1:]):
        raise NativeConformalCalibrationError(
            "native calibration replay has conflicting coordinator relation authorities"
        )
    records = first.get("records")
    if not isinstance(records, list) or [record.get("sample_id") for record in records if isinstance(record, Mapping)] != list(identity_frame.sample_ids):
        raise NativeConformalCalibrationError(
            "native calibration replay relations do not exactly cover sample_ids"
        )
    return first


__all__ = [
    "NativeConformalCalibrationError",
    "NativeConformalCalibrationReplay",
    "compile_methods_conformal_calibration_replay",
]
