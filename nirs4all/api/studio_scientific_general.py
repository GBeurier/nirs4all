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
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np

from nirs4all.pipeline.config.component_serialization import deserialize_component

from .result import RunResult
from .run import _run_strict_product
from .studio_scientific import StudioScientificJobError, _ambient_runtime_preflight

STUDIO_GENERAL_JOB_SCHEMA = "nirs4all.studio-scientific-job.v2"
STUDIO_GENERAL_RESULT_SCHEMA = "nirs4all.studio-scientific-job-result.v2"
STUDIO_MULTIMODAL_DATASET_SCHEMA = "nirs4all.studio-multimodal-dataset.v1"
MAX_MULTIMODAL_ARRAY_BYTES = 32 * 1024 * 1024
MAX_GENERAL_REQUEST_BYTES = 8 * 1024 * 1024
MAX_GENERAL_RESPONSE_BYTES = 256 * 1024
_PACKAGE_PREFIXES = frozenset({"nirs4all", "sklearn", "numpy", "scipy", "xgboost", "lightgbm", "catboost", "torch", "tensorflow"})
# Product presets authorize these declarations, not the whole optional package.
# Document editing must not import TabPFN or require its runtime dependencies.
_OPTIONAL_PRODUCT_OPERATORS = frozenset({"tabpfn.TabPFNRegressor", "tabpfn.TabPFNClassifier"})
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
        if value.partition(".")[0] not in _PACKAGE_PREFIXES and value not in _OPTIONAL_PRODUCT_OPERATORS:
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


def _preflight_optional_product_operators(value: Any) -> None:
    """Require optional preset operators only at execution, before construction."""
    if isinstance(value, str) and value in _OPTIONAL_PRODUCT_OPERATORS:
        module_name, _, operator_name = value.rpartition(".")
        try:
            module = import_module(module_name)
            getattr(module, operator_name)
        except (ImportError, AttributeError) as exc:
            raise StudioScientificJobError(
                "dependency_missing",
                f"Cannot execute {value}: install a compatible tabpfn package and its "
                f"runtime dependencies in the authorized scientific environment ({exc}). "
                "The preset can still be imported and edited without this optional dependency.",
            ) from exc
    elif isinstance(value, list):
        for item in value:
            _preflight_optional_product_operators(item)
    elif isinstance(value, dict):
        for item in value.values():
            _preflight_optional_product_operators(item)


def validate_studio_pipeline_config(value: Any) -> None:
    """Validate canonical Studio declarations before any operator is imported.

    Shared by the Rust-owned document adapter and scientific host. This is a
    package authorization boundary for trusted scientific code, not a sandbox.
    """
    _validate_json(value)
    if not isinstance(value, list) or not value:
        raise StudioScientificJobError("invalid_pipeline", "pipeline must contain canonical steps or pipelines")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_GENERAL_REQUEST_BYTES:
        raise StudioScientificJobError("request_too_large", "canonical pipeline exceeds 8 MiB")
    _validate_operator_imports(value)


def _inline_dataset_arrays(value: Any) -> Any:
    """Restore typed in-memory datasets; leave ordinary file configs untouched."""
    if isinstance(value, list):
        return [_inline_dataset_arrays(item) for item in value]
    if isinstance(value, dict) and value.get("schema") == STUDIO_MULTIMODAL_DATASET_SCHEMA:
        if set(value) != {"schema", "cohort"} or not isinstance(value["cohort"], dict):
            raise StudioScientificJobError("invalid_dataset", "multimodal dataset descriptor requires only a cohort object")
        from nirs4all_io import MultimodalDataset

        try:
            _validate_multimodal_array_budget(value["cohort"])
            return MultimodalDataset.from_dict(value["cohort"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudioScientificJobError("invalid_dataset", "multimodal cohort failed IO validation") from exc
    if isinstance(value, dict) and "X" in value and "y" in value:
        return {**value, "X": np.asarray(value["X"]), "y": np.asarray(value["y"])}
    return value


def _validate_multimodal_array_budget(cohort: dict[str, Any]) -> None:
    """Bound declared array storage before IO materializes untrusted dtypes."""
    remaining = MAX_MULTIMODAL_ARRAY_BYTES
    sample_ids = cohort.get("sample_ids")
    cohort_rows = len(sample_ids) if isinstance(sample_ids, list) else 0
    source_arrays = {
        id(source["array"])
        for source in cohort.get("sources", [])
        if isinstance(source, dict) and source.get("source_kind") != "ragged_series" and isinstance(source.get("array"), dict)
    } if isinstance(cohort.get("sources"), list) else set()
    pending: list[Any] = [cohort]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if {"dtype", "shape", "values"} <= item.keys():
                try:
                    dtype = np.dtype(item["dtype"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Invalid multimodal array dtype") from exc
                shape = item["shape"]
                if not isinstance(shape, list) or not shape or len(shape) > 8 or any(type(dim) is not int or dim < 0 for dim in shape):
                    raise ValueError("Invalid multimodal array shape")
                try:
                    actual_shape = np.asarray(item["values"], dtype=object).shape
                except ValueError as exc:
                    raise ValueError("Multimodal array values are not rectangular") from exc
                zero_axis = shape.index(0) if 0 in shape else None
                expected_shape = tuple(shape if zero_axis is None else shape[:zero_axis + 1])
                if actual_shape != expected_shape:
                    raise ValueError("Multimodal array values disagree with declared shape")
                # A zero-row source can be left-aligned to a populated cohort.
                # Bound its full aligned extent before IO creates placeholders.
                size = max(dtype.itemsize, 1)
                dimensions = shape.copy()
                if id(item) in source_arrays:
                    dimensions[0] = max(dimensions[0], cohort_rows)
                for dim in dimensions:
                    size *= max(dim, 1)
                    if size > remaining:
                        raise ValueError("Multimodal array allocation exceeds budget")
                remaining -= size
            else:
                pending.extend(item.values())


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
    validate_studio_pipeline_config(request["pipeline"])
    multimodal = isinstance(request["dataset"], dict) and request["dataset"].get("schema") == STUDIO_MULTIMODAL_DATASET_SCHEMA
    if multimodal and (options.get("refit") is False or options.get("save_artifacts") is False):
        raise StudioScientificJobError("invalid_option", "multimodal Studio runs require refit and saved artifacts for replay")
    dataset = _inline_dataset_arrays(request["dataset"])
    _ambient_runtime_preflight()
    _preflight_optional_product_operators(request["pipeline"])
    pipeline = deserialize_component(request["pipeline"])
    run_options = {"verbose": 0, "save_artifacts": True, **options}
    if multimodal:
        run_options["refit"] = True
    result = cast(RunResult, _run_strict_product(
        pipeline, dataset, engine="dag-ml", allow_fallback=False,
        **run_options,
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
        if not run_ids and not multimodal:
            raise StudioScientificJobError("missing_persistence", "general scientific result omitted durable run IDs")
        archive_path = None
        if multimodal:
            export_dir = Path(workspace) / "exports"
            export_dir.mkdir(exist_ok=True)
            if export_dir.is_symlink() or export_dir.resolve() != export_dir:
                raise StudioScientificJobError("invalid_workspace", "multimodal export directory must be canonical")
            merged = next((
                child for child in children
                if any(
                    metadata.get("producer_node", "").startswith("merge:")
                    for metadata in child.per_dataset.values()
                )
            ), result)
            archive_path = str(merged.export(export_dir / f"studio-multimodal-{uuid4().hex}.n4a"))
        selected = result.cv_best or result.best
        evaluations = [
            {"run_id": metadata.get("run_id"), "dataset": dataset_name, **metadata["evaluation"]}
            for child in children for dataset_name, metadata in child.per_dataset.items()
            if "evaluation" in metadata
        ]
        response = {
            "schema": STUDIO_GENERAL_RESULT_SCHEMA,
            "job_id": job_id,
            "engine": result.execution_engine,
            "result": {
                "run_ids": run_ids,
                "workspace_path": workspace,
                "native_results_dirs": native_results,
                **({"archive_path": archive_path} if archive_path is not None else {}),
                "metric": selected.get("metric"),
                "validation_score": _finite_or_none(result.cv_best_score),
                "evaluations": evaluations,
                "chart_reports": [path for child in children for metadata in child.per_dataset.values() for path in metadata.get("chart_reports", [])],
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
