"""Versioned general scientific call for the Rust-owned Studio stdio host.

Rust authorizes paths and jobs; this synchronous call owns only library work.
Canonical operator declarations are trusted scientific code, not a sandbox.
Custom package imports require a separately authorized package manifest and
are not enabled by user-supplied module names in this closed contract.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np

from nirs4all.pipeline.config.component_serialization import deserialize_component

from .result import RunResult
from .run import _run_strict_product
from .studio_scientific import StudioScientificJobError, _ambient_runtime_preflight

STUDIO_GENERAL_JOB_SCHEMA = "nirs4all.studio-scientific-job.v2"
STUDIO_GENERAL_RESULT_SCHEMA = "nirs4all.studio-scientific-job-result.v2"
MAX_GENERAL_REQUEST_BYTES = 8 * 1024 * 1024
MAX_GENERAL_RESPONSE_BYTES = 256 * 1024
_PACKAGE_PREFIXES = frozenset({"nirs4all", "sklearn", "numpy", "scipy", "xgboost", "lightgbm", "catboost", "torch", "tensorflow"})
_MODULE_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_OPTIONS = frozenset({
    "name", "random_state", "verbose", "save_charts", "save_artifacts", "workspace_path",
    "project", "refit", "cache", "report_naming", "keep_datasets", "n_jobs", "max_generation_count",
    "continue_on_error", "results_path",
})


def _validate_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise StudioScientificJobError("json_too_deep", "general scientific request exceeds nesting depth 64")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item, depth + 1)
        return
    raise StudioScientificJobError("non_json_value", "general scientific request must contain finite plain JSON values")


def _validate_operator_imports(value: Any) -> None:
    """Validate before any canonical declaration can instantiate Python code."""
    if isinstance(value, str) and _MODULE_PATH.fullmatch(value):
        if value.partition(".")[0] not in _PACKAGE_PREFIXES:
            raise StudioScientificJobError("operator_package_forbidden", f"operator package requires explicit authorization: {value}")
    elif isinstance(value, list):
        for item in value:
            _validate_operator_imports(item)
    elif isinstance(value, dict):
        for key in ("class", "function", "instance", "enum"):
            if key in value:
                declaration = value[key]
                if not isinstance(declaration, str) or not _MODULE_PATH.fullmatch(declaration):
                    raise StudioScientificJobError("invalid_operator", f"{key} must be a qualified approved operator name")
        for item in value.values():
            _validate_operator_imports(item)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _inline_dataset_arrays(value: Any) -> Any:
    """Restore the ndarray wire representation; leave file configs untouched."""
    if isinstance(value, list):
        return [_inline_dataset_arrays(item) for item in value]
    if isinstance(value, dict) and "X" in value and "y" in value:
        return {**value, "X": np.asarray(value["X"]), "y": np.asarray(value["y"])}
    return value


def studio_scientific_job_v2(request: object) -> dict[str, Any]:
    """Run one canonical general request with Rust-owned job/path authority.

    Required fields: schema, operation='run', job_id, pipeline (canonical
    step list or list of pipelines), dataset (library path/config), options
    containing an absolute Rust-authorized workspace_path. Optional engine
    and allow_fallback must be 'dag-ml' and false. No HTTP, queue, polling,
    scheduler, or cancellation owner is introduced here.

    Responses contain bounded summaries and durable result paths, never
    estimator objects or full prediction arrays. Unsupported capabilities
    propagate without trying a different engine. The V1 closed portable
    callable remains unchanged and is not silently promoted to this contract.
    """
    _validate_json(request)
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_GENERAL_REQUEST_BYTES:
        raise StudioScientificJobError("request_too_large", "general scientific request exceeds 8 MiB")
    required = {"schema", "operation", "job_id", "pipeline", "dataset", "options"}
    if not isinstance(request, dict) or not required <= request.keys() or request.keys() - required - {"engine", "allow_fallback"}:
        raise StudioScientificJobError("invalid_shape", "general scientific request has missing or unknown fields")
    if request["schema"] != STUDIO_GENERAL_JOB_SCHEMA or request["operation"] != "run":
        raise StudioScientificJobError("unsupported_contract", "expected the v2 general run contract")
    if request.get("engine", "dag-ml") != "dag-ml" or request.get("allow_fallback", False) is not False:
        raise StudioScientificJobError("engine_forbidden", "general Studio execution requires DAG-ML without fallback")
    job_id = request["job_id"]
    if not isinstance(job_id, str) or not job_id or len(job_id.encode("utf-8")) > 256 or any(ord(char) < 32 for char in job_id):
        raise StudioScientificJobError("invalid_job_id", "job_id must be a non-empty bounded identifier")
    options = request["options"]
    if not isinstance(options, dict) or options.keys() - _OPTIONS:
        raise StudioScientificJobError("unknown_option", "general Studio options must belong to the public run allowlist")
    workspace = options.get("workspace_path")
    if not isinstance(workspace, str) or not Path(workspace).is_absolute():
        raise StudioScientificJobError("workspace_required", "Rust must supply an absolute authorized workspace_path")
    if not isinstance(request["pipeline"], list) or not request["pipeline"]:
        raise StudioScientificJobError("invalid_pipeline", "pipeline must contain canonical steps or pipelines")
    if not isinstance(request["dataset"], (str, dict, list)):
        raise StudioScientificJobError("invalid_dataset", "dataset must be a canonical library path or config")
    _validate_operator_imports(request["pipeline"])
    _ambient_runtime_preflight()
    pipeline = deserialize_component(request["pipeline"])
    result = cast(RunResult, _run_strict_product(
        pipeline, _inline_dataset_arrays(request["dataset"]), engine="dag-ml", allow_fallback=False,
        **{"verbose": 0, "save_artifacts": True, **options},
    ))
    try:
        children = getattr(result, "runs", (result,))
        run_ids = []
        native_results = []
        for child in children:
            for metadata in child.per_dataset.values():
                identifier = metadata.get("run_id")
                if isinstance(identifier, str) and identifier not in run_ids:
                    run_ids.append(identifier)
            if child._dagml_results_dir is not None:
                native_results.append(str(child._dagml_results_dir))
        if not run_ids:
            raise StudioScientificJobError("missing_persistence", "general scientific result omitted durable run IDs")
        selected = result.cv_best
        response = {
            "schema": STUDIO_GENERAL_RESULT_SCHEMA,
            "job_id": job_id,
            "engine": result.execution_engine,
            "result": {
                "run_ids": run_ids,
                "workspace_path": workspace,
                "native_results_dirs": native_results,
                "metric": selected.get("metric"),
                "validation_score": _finite_or_none(result.cv_best_score),
                "prediction_count": result.num_predictions,
                "model_names": result.get_models(),
                "dataset_names": result.get_datasets(),
                "native_score_sets_available": all(child._dagml_score_set is not None for child in children),
            },
        }
        encoded_response = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded_response) > MAX_GENERAL_RESPONSE_BYTES:
            raise StudioScientificJobError("response_too_large", "general scientific response exceeds 256 KiB")
        return response
    finally:
        result.close()
