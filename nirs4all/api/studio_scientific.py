"""Bounded scientific-library entry point for the Studio CPython stdio host.

This module is not a Studio backend.  It accepts one already-resolved, closed
JSON value, runs the supported scientific operation, and returns a bounded
JSON-native value.  Studio's Rust sidecar remains responsible for transport,
jobs, cancellation, events, persistence, and all workspace access.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.core import TaskType, detect_task_type

from .run import run as _run

STUDIO_SCIENTIFIC_JOB_SCHEMA: Final = "nirs4all.studio-scientific-job.v1"
STUDIO_SCIENTIFIC_RESULT_SCHEMA: Final = "nirs4all.studio-scientific-job-result.v1"
MAX_STUDIO_SCIENTIFIC_REQUEST_BYTES: Final = 65_536
MAX_STUDIO_SCIENTIFIC_RESPONSE_BYTES: Final = 8_192
MAX_STUDIO_SCIENTIFIC_SAMPLES: Final = 128
MAX_STUDIO_SCIENTIFIC_FEATURES: Final = 256
MAX_STUDIO_SCIENTIFIC_CELLS: Final = 16_384
_MAX_JSON_DEPTH: Final = 8
_MAX_TEXT_BYTES: Final = 256
_FORBIDDEN_DAGML_SIDE_CHANNELS: Final = (
    "N4A_DAGML_DATASET_PATH",
    "N4A_DAGML_DATASET_PICKLE",
    "N4A_DAGML_GRAPH_PATH",
    "N4A_DAGML_METHODS_SNV",
    "N4A_DAGML_RESULT_CAPTURE",
    "N4A_DAGML_SAMPLE_META_PATH",
    "N4A_RANDOM_STATE",
)


class StudioScientificJobError(ValueError):
    """Typed refusal at the closed Studio scientific-library boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _refuse(code: str, message: str) -> StudioScientificJobError:
    return StudioScientificJobError(code, message)


def _plain_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise _refuse("json_too_deep", "scientific request exceeds the nesting limit")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise _refuse("non_finite_number", "scientific request contains a non-finite number")
        return
    if type(value) is list:
        for item in value:
            _plain_json(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise _refuse("non_json_value", "scientific request object keys must be strings")
            _plain_json(item, depth=depth + 1)
        return
    raise _refuse("non_json_value", "scientific request must contain only plain JSON values")


def _object(value: object, required: set[str], allowed: set[str], where: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _refuse("invalid_shape", f"{where} must be a JSON object")
    result = cast(dict[str, object], value)
    missing = sorted(required - result.keys())
    unknown = sorted(result.keys() - allowed)
    if missing:
        raise _refuse("missing_field", f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise _refuse("unknown_field", f"{where} contains unknown fields: {', '.join(unknown)}")
    return result


def _text(value: object, where: str, *, path_safe: bool = False) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise _refuse("invalid_text", f"{where} must be a non-empty bounded string")
    if any(ord(character) < 32 for character in value):
        raise _refuse("invalid_text", f"{where} must not contain control characters")
    if path_safe and ("/" in value or "\\" in value or value in {".", ".."}):
        raise _refuse("path_forbidden", f"{where} must be an identifier, not a path")
    return value


def _integer(value: object, where: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _refuse("invalid_integer", f"{where} must be an integer in {minimum}..={maximum}")
    return value


def _number(value: object, where: str) -> float:
    if type(value) not in {int, float}:
        raise _refuse("invalid_number", f"{where} must be a finite JSON number")
    try:
        number = float(cast(int | float, value))
    except OverflowError as error:
        raise _refuse("non_finite_number", f"{where} must be finite") from error
    if not math.isfinite(number):
        raise _refuse("non_finite_number", f"{where} must be finite")
    return number


def _matrix(value: object) -> NDArray[np.float64]:
    if type(value) is not list or not value:
        raise _refuse("invalid_dataset", "dataset.X must be a non-empty array of rows")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if type(row) is not list or not row:
            raise _refuse("invalid_dataset", f"dataset.X[{row_index}] must be a non-empty array")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise _refuse("invalid_dataset", "dataset.X rows must have equal lengths")
        rows.append([_number(item, f"dataset.X[{row_index}]") for item in row])
    if len(rows) > MAX_STUDIO_SCIENTIFIC_SAMPLES or width is None or width > MAX_STUDIO_SCIENTIFIC_FEATURES:
        raise _refuse("dataset_too_large", "dataset exceeds the Studio scientific host dimensions")
    if len(rows) * width > MAX_STUDIO_SCIENTIFIC_CELLS:
        raise _refuse("dataset_too_large", "dataset exceeds the Studio scientific host cell budget")
    return cast(NDArray[np.float64], np.asarray(rows, dtype=np.float64))


def _target(value: object, samples: int) -> NDArray[np.float64]:
    if type(value) is not list or len(value) != samples:
        raise _refuse("invalid_dataset", "dataset.y must contain one scalar target per sample")
    target = np.asarray([_number(item, "dataset.y") for item in value], dtype=np.float64)
    if detect_task_type(target) is not TaskType.REGRESSION:
        raise _refuse("unsupported_task", "the v1 Studio scientific host supports regression targets only")
    return cast(NDArray[np.float64], target)


def _ambient_runtime_preflight() -> None:
    if os.environ.get("N4A_NATIVE_RESULTS", "").strip().lower() not in {"", "0", "false", "no"}:
        raise _refuse("ambient_persistence_forbidden", "N4A_NATIVE_RESULTS is forbidden in the Studio scientific host")
    if os.environ.get("N4A_DAGML_CLI"):
        raise _refuse("external_runtime_forbidden", "an external dag-ml CLI is forbidden in the Studio scientific host")
    if os.environ.get("N4A_DAGML_INPROCESS", "").strip().lower() in {"0", "false", "off"}:
        raise _refuse("external_runtime_forbidden", "the Studio scientific host requires the packaged in-process dag-ml runtime")
    active_side_channels = [variable for variable in _FORBIDDEN_DAGML_SIDE_CHANNELS if os.environ.get(variable)]
    if active_side_channels:
        raise _refuse(
            "ambient_runtime_forbidden",
            f"ambient dag-ml side channels are forbidden in the Studio scientific host: {', '.join(active_side_channels)}",
        )
    prefix = Path(sys.prefix).resolve()
    try:
        Path(__file__).resolve().relative_to(prefix)
    except ValueError as error:
        raise _refuse("sibling_runtime_forbidden", "the nirs4all Studio callable must be installed inside the active Python prefix") from error
    try:
        import dag_ml._dag_ml as _dag_ml
    except ImportError as error:
        raise _refuse("native_runtime_unavailable", "the packaged in-process dag-ml runtime is unavailable") from error
    origin = getattr(_dag_ml, "__file__", None)
    if not isinstance(origin, str):
        raise _refuse("native_runtime_unavailable", "the packaged in-process dag-ml runtime has no file identity")
    try:
        Path(origin).resolve().relative_to(prefix)
    except ValueError as error:
        raise _refuse("sibling_runtime_forbidden", "the dag-ml runtime must be installed inside the active Python prefix") from error


def _validated_job(request: object) -> tuple[str, str, NDArray[np.float64], NDArray[np.float64], int, bool, int, bool, int]:
    _plain_json(request)
    try:
        encoded = json.dumps(request, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _refuse("non_json_value", "scientific request is not strict JSON") from error
    if len(encoded) > MAX_STUDIO_SCIENTIFIC_REQUEST_BYTES:
        raise _refuse("request_too_large", "scientific request exceeds 65536 bytes")

    root_keys = {"schema", "operation", "job_id", "engine", "allow_fallback", "dataset", "pipeline", "options"}
    root = _object(request, root_keys, root_keys, "request")
    if root["schema"] != STUDIO_SCIENTIFIC_JOB_SCHEMA or root["operation"] != "run":
        raise _refuse("unsupported_contract", "scientific request schema and operation must be the v1 run contract")
    if root["engine"] != "dag-ml":
        raise _refuse("engine_forbidden", "the Studio scientific host requires engine='dag-ml'")
    if root["allow_fallback"] is not False:
        raise _refuse("fallback_forbidden", "the Studio scientific host requires allow_fallback=false")
    job_id = _text(root["job_id"], "job_id", path_safe=True)

    dataset = _object(root["dataset"], {"name", "task_type", "X", "y"}, {"name", "task_type", "X", "y"}, "dataset")
    _text(dataset["name"], "dataset.name", path_safe=True)
    if dataset["task_type"] != "regression":
        raise _refuse("unsupported_task", "dataset.task_type must be regression")
    X = _matrix(dataset["X"])
    y = _target(dataset["y"], X.shape[0])

    pipeline = _object(root["pipeline"], {"kind", "n_components", "scale", "cross_validation"}, {"kind", "n_components", "scale", "cross_validation"}, "pipeline")
    if pipeline["kind"] != "pls_regression":
        raise _refuse("unsupported_pipeline", "pipeline.kind must be pls_regression")
    components = _integer(pipeline["n_components"], "pipeline.n_components", 1, MAX_STUDIO_SCIENTIFIC_FEATURES)
    if type(pipeline["scale"]) is not bool:
        raise _refuse("invalid_shape", "pipeline.scale must be a boolean")

    cv = _object(pipeline["cross_validation"], {"kind", "n_splits", "shuffle"}, {"kind", "n_splits", "shuffle"}, "pipeline.cross_validation")
    if cv["kind"] != "kfold":
        raise _refuse("unsupported_pipeline", "pipeline.cross_validation.kind must be kfold")
    splits = _integer(cv["n_splits"], "pipeline.cross_validation.n_splits", 2, 10)
    if splits > X.shape[0] or X.shape[0] < 4:
        raise _refuse("invalid_cross_validation", "kfold requires at least four samples and no more splits than samples")
    if type(cv["shuffle"]) is not bool:
        raise _refuse("invalid_shape", "pipeline.cross_validation.shuffle must be a boolean")
    smallest_train = X.shape[0] - math.ceil(X.shape[0] / splits)
    if components > min(X.shape[1], smallest_train):
        raise _refuse("invalid_components", "PLS components exceed the smallest training fold")

    options = _object(root["options"], {"name", "random_state"}, {"name", "random_state"}, "options")
    run_name = _text(options["name"], "options.name", path_safe=True)
    random_state = _integer(options["random_state"], "options.random_state", 0, 2_147_483_647)
    return job_id, run_name, X, y, components, pipeline["scale"], splits, cv["shuffle"], random_state


def _finite_result_number(value: object, where: str) -> float:
    if type(value) not in {int, float}:
        raise _refuse("invalid_scientific_result", f"{where} is not a JSON number")
    try:
        number = float(cast(int | float, value))
    except OverflowError as error:
        raise _refuse("invalid_scientific_result", f"{where} is not finite") from error
    if not math.isfinite(number):
        raise _refuse("invalid_scientific_result", f"{where} is not finite")
    return number


def studio_scientific_job_v1(request: object) -> dict[str, object]:
    """Execute one resolved, bounded Studio regression job through dag-ml.

    The request and response schemas are documented in
    ``docs/source/reference/public_interfaces.md``.  The callable never opens a
    caller path, workspace, socket, HTTP server, scheduler, or durable store.
    """

    job_id, run_name, X, y, components, scale, splits, shuffle, random_state = _validated_job(request)
    _ambient_runtime_preflight()
    pipeline = [
        KFold(n_splits=splits, shuffle=shuffle, random_state=random_state if shuffle else None),
        PLSRegression(n_components=components, scale=scale),
    ]
    result: Any = _run(
        pipeline=pipeline,
        dataset=(X, y),
        name=run_name,
        verbose=0,
        save_artifacts=False,
        save_charts=False,
        plots_visible=False,
        # KFold is the only stochastic v1 component and is seeded above.  Do
        # not ask run_via_dagml to seed process globals: that would mutate
        # PYTHONHASHSEED (and optional framework state) inside the stdio host.
        random_state=None,
        refit=True,
        cache=None,
        project=None,
        engine="dag-ml",
        results_path=None,
        allow_fallback=False,
    )
    try:
        selected = dict(result.cv_best)
        response: dict[str, object] = {
            "schema": STUDIO_SCIENTIFIC_RESULT_SCHEMA,
            "job_id": job_id,
            "engine": "dag-ml",
            "result": {
                "model": "pls_regression",
                "task_type": "regression",
                "metric": _text(selected.get("metric"), "result.metric"),
                "validation_score": _finite_result_number(selected.get("val_score"), "result.validation_score"),
                "training_score": _finite_result_number(selected.get("train_score"), "result.training_score"),
                "prediction_count": _integer(result.num_predictions, "result.prediction_count", 1, 1_000_000),
            },
        }
        encoded = json.dumps(response, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_STUDIO_SCIENTIFIC_RESPONSE_BYTES:
            raise _refuse("response_too_large", "scientific response exceeds 8192 bytes")
        return response
    finally:
        result.close()


__all__ = [
    "MAX_STUDIO_SCIENTIFIC_REQUEST_BYTES",
    "MAX_STUDIO_SCIENTIFIC_RESPONSE_BYTES",
    "STUDIO_SCIENTIFIC_JOB_SCHEMA",
    "STUDIO_SCIENTIFIC_RESULT_SCHEMA",
    "StudioScientificJobError",
    "studio_scientific_job_v1",
]
