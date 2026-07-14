"""Public conformal calibration entry point."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from nirs4all.api.result import PredictResult
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml.conformal_contracts import (
    CalibratedRunResult,
    ConformalCalibrationSpec,
    ConformalMethod,
    ConformalMetricSet,
    ConformalMultiTarget,
    calibrate_replayed_predictions,
    conformal_guarantee_status,
    evaluate_conformal_prediction,
    parse_conformal_calibration_spec,
)
from nirs4all.pipeline.dagml.conformal_store import (
    attach_conformal_result_to_bundle,
    export_conformal_result_bundle,
    load_conformal_result_archive,
    load_conformal_result_store,
    save_conformal_result_store,
)

ConformalUnit = Literal["physical_sample"]
CONFORMAL_CALIBRATION_METHODS: tuple[ConformalMethod, ...] = ("split_absolute_residual",)
CONFORMAL_CALIBRATION_UNITS: tuple[ConformalUnit, ...] = ("physical_sample",)
CONFORMAL_MULTI_TARGET_POLICIES: tuple[ConformalMultiTarget, ...] = ("marginal", "joint_max")
CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES: tuple[ConformalMultiTarget, ...] = ("marginal", "joint_max")


def _strict_json_mapping(payload: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    """Return a strict JSON-native mapping without Python object stringification."""

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip() or "\x00" in key:
            raise ValueError(f"{label} keys must be canonical non-empty strings")
        if key in normalized:
            raise ValueError(f"{label} contains duplicate keys")
        normalized[key] = _strict_json_value(value, f"{label}[{key}]")
    return normalized


def _strict_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be JSON-native and finite")
        return value
    if isinstance(value, (bytes, tuple, set, frozenset)):
        raise ValueError(f"{label} must be JSON-native")
    if isinstance(value, Mapping):
        return _strict_json_mapping(value, label)
    if isinstance(value, list):
        return [_strict_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{label} must be JSON-native")


def _canonical_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _canonical_metadata_columns(value: Any, label: str) -> str | list[str]:
    if isinstance(value, str):
        return _canonical_non_empty_string(value, label)
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a canonical string or a sequence of canonical strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, column in enumerate(value):
        canonical = _canonical_non_empty_string(column, f"{label}[{index}]")
        if canonical in seen:
            raise ValueError(f"{label} contains duplicate column names")
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


@dataclass(frozen=True)
class ConformalCalibrationData:
    """Typed public calibration evidence for ``nirs4all.calibrate()``.

    Use either explicit replayed arrays (``y_true``/``y_pred``/``sample_ids``) or
    a dataset-backed source (``dataset`` + ``selector``) with one replay lane
    such as ``y_pred`` or ``predictor``. The object serializes to the mapping
    consumed by the existing runtime; it does not create implicit splits or
    discover predictors automatically.
    """

    y_true: Any | None = None
    y_pred: Any | None = None
    sample_ids: Any = None
    groups: Any = None
    metadata: Any = None
    dataset: Any | None = None
    selector: Mapping[str, Any] | None = None
    sample_id_column: str | None = None
    group_column: str | None = None
    metadata_columns: str | list[str] | tuple[str, ...] | None = None
    include_augmented: bool = False
    predictor: Any | None = None
    predictor_bundle: str | Path | None = None
    model_bundle: str | Path | None = None
    predictor_path: str | Path | None = None
    model_path: str | Path | None = None
    predictor_result: Any | None = None
    predictor_chain_id: str | None = None
    workspace_chain_id: str | None = None
    workspace_path: str | Path | None = None
    predictor_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``nirs4all.calibrate()``."""

        if self.dataset is None:
            if self.y_true is None or self.y_pred is None:
                raise ValueError("ConformalCalibrationData requires y_true/y_pred or dataset/selector")
            payload: dict[str, Any] = {
                "y_pred": self.y_pred,
                "y_true": self.y_true,
            }
        else:
            if self.y_true is not None:
                raise ValueError("ConformalCalibrationData dataset-backed form derives y_true from dataset/selector")
            if self.selector is None:
                raise ValueError("ConformalCalibrationData dataset-backed form requires selector")
            if not isinstance(self.include_augmented, bool):
                raise ValueError("ConformalCalibrationData.include_augmented must be a boolean")
            selector = _strict_json_mapping(self.selector, "ConformalCalibrationData.selector")
            predictor_bundle = _single_alias_value(
                "predictor_bundle",
                (("predictor_bundle", self.predictor_bundle), ("model_bundle", self.model_bundle), ("predictor_path", self.predictor_path), ("model_path", self.model_path)),
            )
            predictor_chain_id = _single_alias_value(
                "predictor_chain_id",
                (("predictor_chain_id", self.predictor_chain_id), ("workspace_chain_id", self.workspace_chain_id)),
            )
            if predictor_chain_id is not None:
                predictor_chain_id = _canonical_non_empty_string(
                    predictor_chain_id,
                    "ConformalCalibrationData.predictor_chain_id",
                )
            replay_lanes = (
                self.y_pred is not None,
                self.predictor is not None,
                predictor_bundle is not None,
                self.predictor_result is not None,
                predictor_chain_id is not None,
            )
            if sum(replay_lanes) > 1:
                raise ValueError("ConformalCalibrationData dataset-backed form accepts at most one of y_pred, predictor, predictor_bundle, predictor_result, or predictor_chain_id")
            payload = {
                "dataset": self.dataset,
                "include_augmented": self.include_augmented,
                "selector": selector,
            }
            _set_if_not_none(payload, "y_pred", self.y_pred)
            _set_if_not_none(
                payload,
                "sample_id_column",
                None
                if self.sample_id_column is None
                else _canonical_non_empty_string(
                    self.sample_id_column,
                    "ConformalCalibrationData.sample_id_column",
                ),
            )
            _set_if_not_none(
                payload,
                "group_column",
                None
                if self.group_column is None
                else _canonical_non_empty_string(
                    self.group_column,
                    "ConformalCalibrationData.group_column",
                ),
            )
            _set_if_not_none(
                payload,
                "metadata_columns",
                None
                if self.metadata_columns is None
                else _canonical_metadata_columns(
                    self.metadata_columns,
                    "ConformalCalibrationData.metadata_columns",
                ),
            )
            _set_if_not_none(payload, "predictor", self.predictor)
            _set_if_not_none(payload, "predictor_bundle", predictor_bundle)
            _set_if_not_none(payload, "predictor_result", self.predictor_result)
            _set_if_not_none(payload, "predictor_chain_id", predictor_chain_id)
            _set_if_not_none(payload, "workspace_path", self.workspace_path)
        _set_if_not_none(payload, "sample_ids", self.sample_ids)
        _set_if_not_none(payload, "groups", self.groups)
        _set_if_not_none(payload, "metadata", self.metadata)
        _set_if_not_none(payload, "predictor_fingerprint", self.predictor_fingerprint)
        return payload


def _canonical_mapping(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError(f"{label} keys must be canonical non-empty strings")
        if key in normalized:
            raise ValueError(f"{label} contains duplicate keys")
        normalized[key] = value
    return normalized


class Nirs4AllCalibrationNotImplementedError(NotImplementedError):
    """Structured boundary for public calibration forms that are still planned."""

    def __init__(
        self,
        calibration_spec: ConformalCalibrationSpec,
        *,
        missing_gates: tuple[str, ...] = (
            "automatic predictor lookup for dataset/replay calibration forms",
            "native predictor bundle calibration fitting",
        ),
    ) -> None:
        self.calibration_spec = calibration_spec
        self.missing_gates = missing_gates
        super().__init__(
            "nirs4all.calibrate() currently supports already replayed arrays and explicit SpectroDataset cohorts "
            "or DatasetConfigs-backed cohorts with replayed calibration predictions or an in-memory predictor "
            "or a public predictor replay source and explicit physical sample ids; "
            f"validated calibration contract fingerprint: {calibration_spec.fingerprint}; "
            f"missing gates for broader public forms: {', '.join(missing_gates)}."
        )


def calibrate(
    calibration_data: ConformalCalibrationData | PredictResult | Mapping[str, Any] | tuple[Any, ...] | None = None,
    *,
    y_true: Any | None = None,
    y_pred_calibration: Any | None = None,
    y_pred: Any | None = None,
    calibration_sample_ids: Any | None = None,
    prediction_sample_ids: Any | None = None,
    coverage: float | list[float] | tuple[float, ...] = 0.9,
    method: str = "split_absolute_residual",
    unit: str = "physical_sample",
    group_by: Any | None = None,
    multi_target: str = "marginal",
    calibration_groups: Any | None = None,
    calibration_metadata: Any | None = None,
    prediction_groups: Any | None = None,
    prediction_metadata: Any | None = None,
    result_metadata: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    predictor_fingerprint: str | None = None,
    store_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    workspace_path: str | Path | None = None,
    workspace_name: str = "",
    workspace_conformal_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
    overwrite_store: bool = False,
    as_predict_result: bool = False,
) -> CalibratedRunResult | PredictResult:
    """Calibrate replayed point predictions with split conformal intervals.

    This first public surface is intentionally narrow: callers must provide
    calibration targets, replayed calibration predictions or an explicit
    predictor replay source over a selected dataset-backed cohort, prediction
    outputs, and explicit physical sample ids. Automatic predictor discovery
    remains a planned follow-up gate. When ``group_by`` is provided, callers
    must also provide row-aligned prediction group evidence via
    ``prediction_groups`` and/or ``prediction_metadata`` so intervals can be
    routed to a calibrated group without fallback.
    """

    spec = parse_conformal_calibration_spec(
        {
            "coverage": coverage,
            "group_by": group_by,
            "method": method,
            "multi_target": multi_target,
            "unit": unit,
        }
    )
    calibration_replay_source: dict[str, Any] | None = None
    if calibration_data is not None:
        extracted = _extract_calibration_data(calibration_data, spec)
        if extracted.get("y_true") is not None:
            y_true = extracted["y_true"]
        if extracted.get("y_pred") is not None:
            y_pred_calibration = extracted["y_pred"]
        if extracted.get("sample_ids") is not None:
            calibration_sample_ids = extracted["sample_ids"]
        if extracted.get("groups") is not None:
            calibration_groups = extracted["groups"]
        if extracted.get("metadata") is not None:
            calibration_metadata = extracted["metadata"]
        if predictor_fingerprint is None and extracted.get("predictor_fingerprint") is not None:
            predictor_fingerprint = extracted["predictor_fingerprint"]
        if extracted.get("calibration_replay_source") is not None:
            calibration_replay_source = dict(extracted["calibration_replay_source"])
        if y_pred_calibration is None and _is_explicit_dataset_calibration_data(calibration_data):
            raise ValueError("Dataset-backed calibration_data requires replayed calibration predictions in y_pred/y_pred_calibration or an explicit predictor replay source")
    if calibration_replay_source is None:
        calibration_replay_source = _calibration_replay_source("explicit_replayed_arrays")

    missing = [
        name
        for name, value in (
            ("y_true", y_true),
            ("y_pred_calibration", y_pred_calibration),
            ("y_pred", y_pred),
            ("calibration_sample_ids", calibration_sample_ids),
            ("prediction_sample_ids", prediction_sample_ids),
        )
        if value is None
    ]
    if missing:
        raise Nirs4AllCalibrationNotImplementedError(spec)

    result = calibrate_replayed_predictions(
        y_true_calibration=y_true,
        y_pred_calibration=y_pred_calibration,
        y_pred=y_pred,
        spec=spec,
        calibration_sample_ids=calibration_sample_ids,
        prediction_sample_ids=prediction_sample_ids,
        calibration_groups=calibration_groups,
        calibration_metadata=calibration_metadata,
        prediction_groups=prediction_groups,
        prediction_metadata=prediction_metadata,
        result_metadata=_strict_json_mapping(result_metadata, "calibrate.result_metadata"),
        target_name=target_name,
        predictor_fingerprint=predictor_fingerprint,
        calibration_replay_source=calibration_replay_source,
    )
    if store_path is not None:
        save_conformal_result_store(result, store_path, overwrite=overwrite_store)
    if bundle_path is not None:
        export_conformal_result_bundle(result, bundle_path, overwrite=overwrite_store)
    if workspace_path is not None:
        save_workspace_calibrated_result(
            workspace_path,
            result,
            name=workspace_name,
            conformal_id=workspace_conformal_id,
            metadata=workspace_metadata,
        )
    if not as_predict_result:
        return result
    return cast(PredictResult, result.to_predict_result())


def load_calibrated_result(path: str | Path) -> CalibratedRunResult:
    """Load a verified conformal calibration result store or bundle."""

    candidate = Path(path)
    if candidate.is_file():
        return load_conformal_result_archive(candidate)
    return load_conformal_result_store(candidate)


def export_calibrated_result(result: CalibratedRunResult, path: str | Path, *, overwrite: bool = False) -> Path:
    """Export a calibrated result as a zipped ``.n4a`` conformal bundle."""

    return export_conformal_result_bundle(result, path, overwrite=overwrite)


def save_workspace_calibrated_result(
    workspace_path: str | Path,
    result: CalibratedRunResult,
    *,
    name: str = "",
    conformal_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
    prediction_id: str | None = None,
) -> str:
    """Persist a calibrated conformal result in a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        return cast(str, store.save_conformal_result(
            result,
            name=name,
            conformal_id=conformal_id,
            metadata=metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            prediction_id=prediction_id,
        ))
    finally:
        store.close()


def load_workspace_calibrated_result(workspace_path: str | Path, conformal_id: str) -> CalibratedRunResult:
    """Load a verified calibrated conformal result from a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        return cast(CalibratedRunResult, store.load_conformal_result(conformal_id))
    finally:
        store.close()


def load_workspace_calibrated_predict_result(workspace_path: str | Path, conformal_id: str) -> PredictResult:
    """Load a workspace conformal result as a public ``PredictResult``."""

    return cast(PredictResult, load_workspace_calibrated_result(workspace_path, conformal_id).to_predict_result())


def attach_calibrated_result_to_bundle(
    model_bundle_path: str | Path,
    calibrated: CalibratedRunResult | str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Attach a calibrated conformal result sidecar to an existing model ``.n4a`` bundle."""

    result = load_calibrated_result(str(calibrated)) if isinstance(calibrated, (str, Path)) else calibrated
    return cast(Path, attach_conformal_result_to_bundle(model_bundle_path, result, output_path, overwrite=overwrite))


def predict_calibrated(
    calibrated: CalibratedRunResult | str | Path,
    *,
    y_pred: Any,
    prediction_sample_ids: Any,
    result_metadata: Mapping[str, Any] | None = None,
    as_predict_result: bool = True,
) -> CalibratedRunResult | PredictResult:
    """Apply an existing conformal calibrator to already computed predictions."""

    source = load_calibrated_result(str(calibrated)) if isinstance(calibrated, (str, Path)) else calibrated
    predictions = _as_1d_float_array(y_pred, "y_pred")
    sample_ids = _normalize_prediction_sample_ids(prediction_sample_ids, predictions.size)
    calibration_replay_source = source.calibration_replay_source
    metadata = _strict_json_mapping(result_metadata, "predict_calibrated.result_metadata")
    if calibration_replay_source is not None:
        metadata["calibration_replay_source"] = dict(calibration_replay_source)
    metadata["conformal_guarantee_status"] = conformal_guarantee_status(
        source.artifact,
        effective_engine="nirs4all.python.replayed_array_apply",
        calibration_replay_source=calibration_replay_source,
        source_calibrated_result_fingerprint=source.fingerprint,
    )
    metadata["source_calibrated_result_fingerprint"] = source.fingerprint
    result = CalibratedRunResult(
        artifact=source.artifact,
        prediction=source.artifact.calibrator.apply_block(predictions),
        sample_ids=sample_ids,
        metadata=metadata,
    )
    if not as_predict_result:
        return result
    return cast(PredictResult, result.to_predict_result())


def conformal_metrics(
    calibrated: CalibratedRunResult | PredictResult | str | Path,
    *,
    y_true: Any,
) -> dict[float, ConformalMetricSet]:
    """Compute observed coverage, widths and interval scores for conformal predictions."""

    if isinstance(calibrated, PredictResult):
        if not calibrated.intervals:
            raise ValueError("PredictResult does not contain conformal intervals")
        from nirs4all.pipeline.dagml.conformal_contracts import CalibratedPredictionBlock

        return evaluate_conformal_prediction(
            y_true=y_true,
            prediction=CalibratedPredictionBlock(
                y_pred=np.asarray(calibrated.y_pred, dtype=float).reshape(-1),
                intervals=calibrated.intervals,
            ),
        )
    source = load_calibrated_result(str(calibrated)) if isinstance(calibrated, (str, Path)) else calibrated
    return source.metrics(y_true)


def _extract_calibration_data(calibration_data: Any, spec: ConformalCalibrationSpec) -> dict[str, Any]:
    """Extract replayed calibration arrays from supported public forms."""

    if isinstance(calibration_data, ConformalCalibrationData):
        calibration_data = calibration_data.to_dict()
    if isinstance(calibration_data, PredictResult):
        return {
            "calibration_replay_source": _calibration_replay_source("predict_result"),
            "y_pred": calibration_data.y_pred,
            "sample_ids": calibration_data.sample_indices,
        }
    if isinstance(calibration_data, tuple):
        return _extract_tuple_calibration_data(calibration_data)
    if not isinstance(calibration_data, Mapping):
        raise TypeError("calibration_data must be a mapping, tuple or PredictResult for the replayed-array calibration surface")

    if "dataset" in calibration_data or "spectro_dataset" in calibration_data:
        return _extract_dataset_backed_calibration_data(calibration_data, spec)

    return _extract_replayed_array_calibration_data(calibration_data, spec)


def _extract_replayed_array_calibration_data(calibration_data: Mapping[str, Any], spec: ConformalCalibrationSpec) -> dict[str, Any]:
    """Extract replayed-array calibration evidence from a raw mapping."""

    y_pred = _single_mapping_alias_value(
        calibration_data,
        "y_pred",
        ("y_pred", "y_pred_calibration", "calibration_predictions"),
        "replayed-array calibration_data",
    )
    groups = _single_mapping_alias_value(
        calibration_data,
        "groups",
        ("groups", "calibration_groups"),
        "replayed-array calibration_data",
    )
    metadata = _single_mapping_alias_value(
        calibration_data,
        "metadata",
        ("metadata", "calibration_metadata"),
        "replayed-array calibration_data",
    )
    sample_ids = _single_mapping_alias_value(
        calibration_data,
        "sample_ids",
        ("sample_ids", "calibration_sample_ids", "physical_sample_ids", "prediction_sample_ids"),
        "replayed-array calibration_data",
    )
    if sample_ids is None:
        sample_ids = _extract_calibration_sample_ids_from_metadata(metadata)
    if not ("y_true" in calibration_data and y_pred is not None):
        accepted = {
            "calibration_groups",
            "calibration_metadata",
            "calibration_predictions",
            "calibration_sample_ids",
            "groups",
            "metadata",
            "physical_sample_ids",
            "prediction_sample_ids",
            "predictor_fingerprint",
            "sample_ids",
            "y_pred",
            "y_pred_calibration",
            "y_true",
        }
        unsupported = sorted(set(calibration_data) - accepted)
        if unsupported:
            raise Nirs4AllCalibrationNotImplementedError(spec)
    return {
        "calibration_replay_source": _calibration_replay_source("replayed_arrays"),
        "groups": groups,
        "metadata": metadata,
        "predictor_fingerprint": calibration_data.get("predictor_fingerprint"),
        "sample_ids": sample_ids,
        "y_pred": y_pred,
        "y_true": calibration_data.get("y_true"),
    }


def _extract_tuple_calibration_data(calibration_data: tuple[Any, ...]) -> dict[str, Any]:
    """Extract replayed calibration evidence from a compact tuple form."""

    if not 2 <= len(calibration_data) <= 5:
        raise ValueError("calibration_data tuple must be (y_true, y_pred, sample_ids, groups, metadata) with optional trailing fields omitted")
    y_true, y_pred, *rest = calibration_data
    sample_ids = rest[0] if len(rest) >= 1 else None
    groups = rest[1] if len(rest) >= 2 else None
    metadata = rest[2] if len(rest) >= 3 else None
    return {
        "calibration_replay_source": _calibration_replay_source("tuple_replayed_arrays"),
        "groups": groups,
        "metadata": metadata,
        "sample_ids": sample_ids,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def _extract_dataset_backed_calibration_data(
    calibration_data: Mapping[str, Any],
    spec: ConformalCalibrationSpec,
) -> dict[str, Any]:
    """Extract explicit calibration evidence from a selected dataset-backed cohort."""

    spectro = _coerce_calibration_dataset_source(
        _first_present(calibration_data, "dataset", "spectro_dataset"),
        spec,
    )
    if any(key in calibration_data for key in ("y_true", "target", "targets", "X", "features")):
        raise ValueError("Dataset-backed calibration_data mappings must not also provide X/y_true arrays")

    selector = calibration_data.get("selector")
    if selector is None:
        raise ValueError("Dataset-backed calibration_data mapping requires an explicit selector")
    if not isinstance(selector, Mapping):
        raise ValueError("Dataset-backed calibration_data selector must be a mapping")
    selector_dict = _strict_json_mapping(selector, "Dataset-backed calibration_data.selector")
    include_augmented = calibration_data.get("include_augmented", False)
    if not isinstance(include_augmented, bool):
        raise ValueError("Dataset-backed calibration_data.include_augmented must be a boolean")

    y_true = spectro.y(selector_dict, include_augmented=include_augmented)
    y_array = np.asarray(y_true)
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        y_true = y_array[:, 0]

    sample_ids = _single_mapping_alias_value(
        calibration_data,
        "sample_ids",
        ("sample_ids", "calibration_sample_ids", "prediction_sample_ids", "physical_sample_ids"),
        "Dataset-backed calibration_data",
    )
    if sample_ids is None:
        sample_id_column = _first_present(calibration_data, "sample_id_column", "physical_sample_id_column")
        if sample_id_column is not None:
            sample_ids = spectro.metadata_column(
                _canonical_non_empty_string(
                    sample_id_column,
                    "Dataset-backed calibration_data.sample_id_column",
                ),
                selector_dict,
                include_augmented=include_augmented,
            )
    if sample_ids is None:
        raise ValueError("Dataset-backed calibration_data requires sample_ids or sample_id_column")

    groups = _single_mapping_alias_value(
        calibration_data,
        "groups",
        ("groups", "calibration_groups"),
        "Dataset-backed calibration_data",
    )
    if groups is None:
        group_column = calibration_data.get("group_column")
        if group_column is not None:
            groups = spectro.metadata_column(
                _canonical_non_empty_string(
                    group_column,
                    "Dataset-backed calibration_data.group_column",
                ),
                selector_dict,
                include_augmented=include_augmented,
            )

    metadata = _single_mapping_alias_value(
        calibration_data,
        "metadata",
        ("metadata", "calibration_metadata"),
        "Dataset-backed calibration_data",
    )
    if metadata is None:
        metadata_columns = calibration_data.get("metadata_columns")
        if metadata_columns is not None:
            metadata_columns = _canonical_metadata_columns(
                metadata_columns,
                "Dataset-backed calibration_data.metadata_columns",
            )
            if isinstance(metadata_columns, str):
                metadata_columns = [metadata_columns]
            metadata_frame = spectro.metadata(
                selector_dict,
                columns=list(metadata_columns),
                include_augmented=include_augmented,
            )
            metadata = metadata_frame.to_dicts() if hasattr(metadata_frame, "to_dicts") else metadata_frame

    calibration_predictions = _first_present(calibration_data, "y_pred", "y_pred_calibration", "calibration_predictions")
    predictor = _first_present(calibration_data, "predictor", "model", "estimator")
    predictor_bundle = _first_present(calibration_data, "predictor_bundle", "model_bundle", "predictor_path", "model_path")
    predictor_result = _first_present(calibration_data, "predictor_result", "run_result", "prediction_entry", "prediction")
    predictor_chain_id = _first_present(calibration_data, "predictor_chain_id", "workspace_chain_id")
    if predictor_chain_id is not None:
        predictor_chain_id = _canonical_non_empty_string(
            predictor_chain_id,
            "Dataset-backed calibration_data.predictor_chain_id",
        )
    replay_source: dict[str, Any] | None = None
    replay_inputs = [
        calibration_predictions is not None,
        predictor is not None,
        predictor_bundle is not None,
        predictor_result is not None,
        predictor_chain_id is not None,
    ]
    if sum(replay_inputs) > 1:
        raise ValueError("Dataset-backed calibration_data must provide exactly one of y_pred, predictor, predictor_bundle, predictor_result, or predictor_chain_id")
    if calibration_predictions is None and predictor is not None:
        X = spectro.x(
            selector_dict,
            layout="2d",
            concat_source=True,
            include_augmented=include_augmented,
        )
        if isinstance(X, list):
            raise ValueError("Dataset-backed calibration_data predictor replay requires a single 2D feature matrix")
        calibration_predictions = _predict_with_optional_identity(
            predictor,
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
        replay_source = _calibration_replay_source(
            "dataset_predictor",
            dataset_backed=True,
            predictor_type=_public_type_name(predictor),
        )
    if calibration_predictions is None and predictor_bundle is not None:
        X = spectro.x(
            selector_dict,
            layout="2d",
            concat_source=True,
            include_augmented=include_augmented,
        )
        if isinstance(X, list):
            raise ValueError("Dataset-backed calibration_data predictor replay requires a single 2D feature matrix")
        calibration_predictions = _predict_with_saved_predictor(
            predictor_bundle,
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
        replay_source = _calibration_replay_source(
            "dataset_predictor_bundle",
            dataset_backed=True,
            predictor_bundle=str(predictor_bundle),
        )
    if calibration_predictions is None and predictor_result is not None:
        X = spectro.x(
            selector_dict,
            layout="2d",
            concat_source=True,
            include_augmented=include_augmented,
        )
        if isinstance(X, list):
            raise ValueError("Dataset-backed calibration_data predictor replay requires a single 2D feature matrix")
        calibration_predictions = _predict_with_result_predictor(
            predictor_result,
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            workspace_path=calibration_data.get("workspace_path"),
        )
        replay_source = _calibration_replay_source(
            "dataset_predictor_result",
            dataset_backed=True,
            predictor_fingerprint=_default_replayed_predictor_fingerprint(predictor_result=predictor_result),
            workspace_path=str(calibration_data["workspace_path"]) if calibration_data.get("workspace_path") is not None else None,
        )
    if calibration_predictions is None and predictor_chain_id is not None:
        X = spectro.x(
            selector_dict,
            layout="2d",
            concat_source=True,
            include_augmented=include_augmented,
        )
        if isinstance(X, list):
            raise ValueError("Dataset-backed calibration_data predictor replay requires a single 2D feature matrix")
        calibration_predictions = _predict_with_workspace_chain(
            predictor_chain_id,
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            workspace_path=calibration_data.get("workspace_path"),
        )
        replay_source = _calibration_replay_source(
            "dataset_predictor_chain_id",
            dataset_backed=True,
            predictor_chain_id=predictor_chain_id,
            workspace_path=str(calibration_data["workspace_path"]) if calibration_data.get("workspace_path") is not None else None,
        )
    if calibration_predictions is not None and replay_source is None:
        replay_source = _calibration_replay_source("dataset_y_pred", dataset_backed=True)

    return {
        "calibration_replay_source": replay_source,
        "groups": groups,
        "metadata": metadata,
        "predictor_fingerprint": calibration_data.get("predictor_fingerprint")
        or _default_replayed_predictor_fingerprint(
            predictor_bundle=predictor_bundle,
            predictor_result=predictor_result,
            predictor_chain_id=predictor_chain_id,
            workspace_path=calibration_data.get("workspace_path"),
        ),
        "sample_ids": sample_ids,
        "y_pred": calibration_predictions,
        "y_true": y_true,
    }


def _coerce_calibration_dataset_source(
    source: Any,
    spec: ConformalCalibrationSpec,
) -> SpectroDataset:
    """Resolve a calibration dataset source through existing nirs4all loaders."""

    if isinstance(source, SpectroDataset):
        return source
    if isinstance(source, DatasetConfigs):
        configs = source
    elif isinstance(source, (str, Path, Mapping)):
        try:
            configs = DatasetConfigs(cast(dict[str, Any] | list[dict[str, Any]] | str | list[str], source))
        except Exception as exc:
            raise Nirs4AllCalibrationNotImplementedError(spec) from exc
    else:
        raise Nirs4AllCalibrationNotImplementedError(spec)
    if len(configs.configs) != 1:
        raise ValueError("calibration_data.dataset must resolve to exactly one dataset")
    return configs.get_dataset_at(0)


def _is_explicit_dataset_calibration_data(calibration_data: Any) -> bool:
    return isinstance(calibration_data, Mapping) and ("dataset" in calibration_data or "spectro_dataset" in calibration_data)


def _predict_with_optional_identity(
    predictor: Any,
    X: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Any:
    if not hasattr(predictor, "predict"):
        raise TypeError("Dataset-backed calibration_data predictor must expose predict(X)")
    predict = predictor.predict
    accepted = _accepted_kwargs(predict)
    kwargs: dict[str, Any] = {}
    if sample_ids is not None and ("sample_ids" in accepted or "**" in accepted):
        kwargs["sample_ids"] = sample_ids
    if groups is not None and ("groups" in accepted or "**" in accepted):
        kwargs["groups"] = groups
    if metadata is not None and ("metadata" in accepted or "**" in accepted):
        kwargs["metadata"] = metadata
    if kwargs:
        return predict(X, **kwargs)
    return predict(X)


def _predict_with_saved_predictor(
    predictor_bundle: Any,
    X: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Any:
    if not isinstance(predictor_bundle, (str, Path)):
        raise TypeError("Dataset-backed calibration_data predictor_bundle must be a path string or Path")
    return _predict_with_public_predict(
        model=predictor_bundle,
        chain_id=None,
        X=X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        workspace_path=None,
    )


def _predict_with_result_predictor(
    predictor_result: Any,
    X: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    workspace_path: Any = None,
) -> Any:
    model = _coerce_predictor_result_model(predictor_result)
    return _predict_with_public_predict(
        model=model,
        chain_id=None,
        X=X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        workspace_path=workspace_path,
    )


def _predict_with_workspace_chain(
    predictor_chain_id: Any,
    X: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    workspace_path: Any = None,
) -> Any:
    canonical_predictor_chain_id = _canonical_non_empty_string(
        predictor_chain_id,
        "Dataset-backed calibration_data.predictor_chain_id",
    )
    if workspace_path is None or not str(workspace_path).strip():
        raise ValueError("Dataset-backed calibration_data predictor_chain_id requires an explicit workspace_path")
    return _predict_with_public_predict(
        model=None,
        chain_id=canonical_predictor_chain_id,
        X=X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        workspace_path=workspace_path,
    )


def _predict_with_public_predict(
    *,
    model: Any | None,
    chain_id: str | None,
    X: Any,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
    workspace_path: Any = None,
) -> Any:
    from nirs4all.api.predict import predict

    data: dict[str, Any] = {"X": X}
    if sample_ids is not None:
        data["sample_ids"] = sample_ids
    if groups is not None:
        data["groups"] = groups
    if metadata is not None:
        data["metadata"] = metadata
    kwargs: dict[str, Any] = {
        "data": data,
        "name": "calibration_prediction",
        "all_predictions": False,
        "verbose": 0,
    }
    if workspace_path is not None:
        kwargs["workspace_path"] = workspace_path
    if chain_id is not None:
        kwargs["chain_id"] = chain_id
    else:
        kwargs["model"] = model
    result = predict(**kwargs)
    return result.y_pred


def _coerce_predictor_result_model(predictor_result: Any) -> Any:
    if isinstance(predictor_result, Mapping):
        return predictor_result
    best = getattr(predictor_result, "best", None)
    if isinstance(best, Mapping):
        return best
    if callable(best):
        candidate = best()
        if isinstance(candidate, Mapping):
            return candidate
    raise TypeError("Dataset-backed calibration_data predictor_result must be a prediction mapping or RunResult-like object exposing .best")


def _default_replayed_predictor_fingerprint(
    *,
    predictor_bundle: Any = None,
    predictor_result: Any = None,
    predictor_chain_id: Any = None,
    workspace_path: Any = None,
) -> str | None:
    if predictor_bundle is not None:
        return str(predictor_bundle)
    if predictor_chain_id is not None:
        canonical_predictor_chain_id = _canonical_non_empty_string(
            predictor_chain_id,
            "Dataset-backed calibration_data.predictor_chain_id",
        )
        if workspace_path is not None:
            return f"workspace:{workspace_path}#chain:{canonical_predictor_chain_id}"
        return f"chain:{canonical_predictor_chain_id}"
    if predictor_result is not None:
        model = _coerce_predictor_result_model(predictor_result)
        for key in ("predictor_fingerprint", "model_fingerprint", "artifact_fingerprint", "chain_id", "prediction_id", "id"):
            value = model.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{str(value).strip()}"
    return None


def _calibration_replay_source(
    kind: str,
    *,
    dataset_backed: bool = False,
    predictor_bundle: str | None = None,
    predictor_chain_id: str | None = None,
    predictor_fingerprint: str | None = None,
    predictor_type: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    route_by_kind = {
        "dataset_predictor": "predictor.predict",
        "dataset_predictor_bundle": "nirs4all.predict",
        "dataset_predictor_chain_id": "nirs4all.predict",
        "dataset_predictor_result": "nirs4all.predict",
        "dataset_y_pred": "provided_dataset_predictions",
        "explicit_replayed_arrays": "provided_arrays",
        "predict_result": "provided_predict_result",
        "replayed_arrays": "provided_arrays",
        "tuple_replayed_arrays": "provided_tuple",
    }
    payload: dict[str, Any] = {
        "dataset_backed": bool(dataset_backed),
        "kind": kind,
        "requires_model_replay": kind.startswith("dataset_predictor"),
        "route": route_by_kind.get(kind, "provided_arrays"),
        "version": 1,
    }
    _set_if_not_none(payload, "predictor_bundle", predictor_bundle)
    _set_if_not_none(payload, "predictor_chain_id", predictor_chain_id)
    _set_if_not_none(payload, "predictor_fingerprint", predictor_fingerprint)
    _set_if_not_none(payload, "predictor_type", predictor_type)
    _set_if_not_none(payload, "workspace_path", workspace_path)
    return payload


def _public_type_name(value: Any) -> str:
    cls = type(value)
    if cls.__module__ in {"builtins", "__main__"}:
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _accepted_kwargs(callable_obj: Any) -> set[str]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set()
    accepted: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepted.add("**")
        elif parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            accepted.add(parameter.name)
    return accepted


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _single_mapping_alias_value(mapping: Mapping[str, Any], canonical: str, aliases: tuple[str, ...], label: str) -> Any | None:
    present = [(alias, mapping[alias]) for alias in aliases if alias in mapping and mapping[alias] is not None]
    if not present:
        return None
    if len(present) > 1:
        names = ", ".join(alias for alias, _value in present)
        raise ValueError(f"{label} accepts only one alias for {canonical}; got {names}")
    return present[0][1]


def _extract_calibration_sample_ids_from_metadata(metadata: Any) -> Any:
    """Extract physical sample ids from prediction metadata when present."""

    if isinstance(metadata, Mapping):
        for key in ("physical_sample_id", "physical_sample_ids", "sample_ids"):
            if metadata.get(key) is not None:
                return metadata[key]
    return None


def _as_1d_float_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _normalize_prediction_sample_ids(sample_ids: Any, expected_size: int) -> tuple[str, ...]:
    if sample_ids is None:
        raise ValueError("prediction_sample_ids are required")
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("prediction_sample_ids must be a row-aligned sequence, not a scalar string")
    try:
        values = list(sample_ids)
    except TypeError as exc:
        raise ValueError("prediction_sample_ids must be a row-aligned sequence") from exc
    if len(values) != expected_size:
        raise ValueError("prediction_sample_ids length must match predictions")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
            raise ValueError("prediction_sample_ids must contain canonical non-empty strings")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("prediction_sample_ids must be unique physical sample identifiers")
    return tuple(normalized)


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _single_alias_value(canonical: str, values: tuple[tuple[str, Any], ...]) -> Any | None:
    present = [(name, value) for name, value in values if value is not None]
    if not present:
        return None
    if len(present) > 1:
        names = ", ".join(name for name, _value in present)
        raise ValueError(f"ConformalCalibrationData dataset-backed form accepts only one alias for {canonical}; got {names}")
    return present[0][1]


__all__ = [
    "CONFORMAL_CALIBRATION_METHODS",
    "CONFORMAL_CALIBRATION_UNITS",
    "CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES",
    "CONFORMAL_MULTI_TARGET_POLICIES",
    "ConformalCalibrationData",
    "ConformalMethod",
    "Nirs4AllCalibrationNotImplementedError",
    "ConformalMultiTarget",
    "ConformalUnit",
    "attach_calibrated_result_to_bundle",
    "calibrate",
    "conformal_metrics",
    "export_calibrated_result",
    "load_calibrated_result",
    "load_workspace_calibrated_predict_result",
    "load_workspace_calibrated_result",
    "predict_calibrated",
    "save_workspace_calibrated_result",
]
