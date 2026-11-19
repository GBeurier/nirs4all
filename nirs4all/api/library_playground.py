"""Closed-contract Playground callable for a Rust-owned Studio stdio host.

The caller owns path authorization, transport, cancellation and any request or
response byte budget.  This module only validates one closed JSON request and
delegates scientific work to the library's stateless Playground facade.  It
opens no listener, queue or store and imposes no content-size ceiling.

A warm host process keeps the last loaded dataset between requests.  It is
reused only for an identical dataset request whose input files still have the
same path, size, modification time and inode; scientific steps work on copies
extracted from it, so a reused dataset gives the same result as a fresh load.

Response contract additions (``nirs4all.studio-playground-result.v1``):

* With ``options.omit_unchanged_processed_spectra`` set to ``true``, an
  ``execute`` result whose processed spectra are bitwise identical to its
  original spectra omits ``processed.spectra`` and carries
  ``processed_is_original: true``; the caller reads ``original.spectra`` for
  both.  Without the option, both matrices are always returned.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from typing import Any, Final, cast

import numpy as np

from nirs4all.analysis.playground_dataset import playground_metadata_columns
from nirs4all.analysis.playground_distances import (
    DISTANCE_METRICS,
    paired_spectral_distances,
    repetition_variance,
)
from nirs4all.analysis.playground_facade import preview_arrays, preview_spectro_dataset
from nirs4all.analysis.playground_metrics import ALL_METRICS
from nirs4all.analysis.playground_prepare import PreviewStep
from nirs4all.analysis.playground_types import PreviewLimits
from nirs4all.api.dataset_documents import normalize_dataset_document
from nirs4all.api.dataset_inspection import _references, load_dataset_for_analysis
from nirs4all.pipeline.config.component_serialization import deserialize_component

from .studio_scientific import StudioScientificJobError
from .studio_scientific_general import _validate_json, _validate_operator_imports

STUDIO_PLAYGROUND_JOB_SCHEMA: Final = "nirs4all.studio-playground-job.v1"
STUDIO_PLAYGROUND_RESULT_SCHEMA: Final = "nirs4all.studio-playground-result.v1"
_OPERATIONS = frozenset({"execute", "metadata_columns", "diff", "repetition_variance", "validate", "capabilities"})
_OPTION_KEYS = frozenset({
    "auto_detect_repetitions", "bio_sample_column", "bio_sample_pattern", "compute_metrics",
    "compute_pca", "compute_repetitions", "compute_statistics", "compute_umap", "dataset_repetition",
    "distance_metric", "lof_contamination", "max_covariance_cells", "max_response_cells", "max_steps",
    "max_wavelengths_returned", "metrics", "n_pca_components", "omit_unchanged_processed_spectra",
    "saturation_threshold", "split_index", "subset_mode", "max_samples_displayed", "umap_params", "use_cache",
})
# One warm host process serves one request at a time; it keeps its last loaded
# dataset as (request key, input file stamps, dataset, reader evidence).
_loaded_dataset: tuple[str, tuple[tuple[str, int, int, int], ...], Any, dict[str, Any]] | None = None


def _refuse(code: str, message: str) -> StudioScientificJobError:
    return StudioScientificJobError(code, message)


def _object(value: Any, *, required: set[str], allowed: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _refuse("invalid_shape", f"{where} must be a JSON object")
    result = cast(dict[str, Any], value)
    missing = sorted(required - result.keys())
    unknown = sorted(result.keys() - allowed)
    if missing:
        raise _refuse("missing_field", f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise _refuse("unknown_field", f"{where} contains unknown fields: {', '.join(unknown)}")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _refuse("invalid_integer", f"{where} must be an integer >= {minimum}")
    return value


def _limits(value: Any) -> PreviewLimits:
    if value is None:
        return PreviewLimits()
    item = _object(value, required=set(), allowed={"max_samples", "max_features", "max_cells"}, where="payload.limits")
    try:
        return PreviewLimits(**item)
    except (TypeError, ValueError) as error:
        raise _refuse("invalid_limits", str(error)) from error


def _sampling(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _object(value, required=set(), allowed={"method", "n_samples", "seed"}, where="payload.sampling")
    method = item.get("method", "random")
    if method not in {"all", "random", "stratified", "kmeans"}:
        raise _refuse("invalid_sampling", "sampling.method must be all, random, stratified or kmeans")
    return {"method": method, "n_samples": _integer(item.get("n_samples", 100), "sampling.n_samples", minimum=1),
            "seed": _integer(item.get("seed", 42), "sampling.seed")}


def _options(value: Any) -> dict[str, Any]:
    item = _object(value or {}, required=set(), allowed=set(_OPTION_KEYS), where="payload.options")
    # The historical flag is accepted as an explicit no-op; this callable keeps
    # no results (only the loaded dataset) and never reintroduces the unsafe
    # prefix cache.
    if item.get("use_cache") not in {None, True, False}:
        raise _refuse("invalid_option", "options.use_cache must be a boolean")
    if item.get("omit_unchanged_processed_spectra") not in {None, True, False}:
        raise _refuse("invalid_option", "options.omit_unchanged_processed_spectra must be a boolean")
    return item


def _steps(value: Any) -> list[PreviewStep]:
    if type(value) is not list:
        raise _refuse("invalid_steps", "payload.steps must be an array of steps")
    result: list[PreviewStep] = []
    keys = {"id", "type", "name", "params", "enabled", "operator"}
    for index, raw in enumerate(value):
        item = _object(raw, required={"id", "type", "name"}, allowed=keys, where=f"payload.steps[{index}]")
        if item["type"] not in {"preprocessing", "augmentation", "splitting", "filter"}:
            raise _refuse("invalid_step", f"payload.steps[{index}].type is unsupported")
        if (not isinstance(item["id"], str) or not item["id"] or len(item["id"].encode("utf-8")) > 256
                or any(ord(char) < 32 for char in item["id"])
                or not isinstance(item["name"], str) or not item["name"] or len(item["name"].encode("utf-8")) > 256
                or any(ord(char) < 32 for char in item["name"])):
            raise _refuse("invalid_step", f"payload.steps[{index}] identifiers must be non-empty strings")
        params = item.get("params", {})
        if type(params) is not dict or type(item.get("enabled", True)) is not bool:
            raise _refuse("invalid_step", f"payload.steps[{index}] params/enabled are invalid")
        declaration = item.get("operator")
        operator = None
        if item["name"] != "SampleIndexFilter":
            if type(declaration) is not dict:
                raise _refuse("missing_operator", f"payload.steps[{index}] requires one canonical operator declaration")
            _validate_operator_imports(declaration)
            try:
                operator = deserialize_component(declaration)
            except Exception as error:
                raise _refuse("invalid_operator", f"payload.steps[{index}] operator is invalid: {error}") from error
            expected_method = "split" if item["type"] == "splitting" else "get_mask" if item["type"] == "filter" else "fit_transform"
            if not callable(getattr(operator, expected_method, None)):
                raise _refuse("invalid_operator", f"payload.steps[{index}] operator does not implement {expected_method}()")
            native_identity = declaration.get("class") == f"n4m:{getattr(operator, '_method_id', '')}"
            if not native_identity and item["name"] != operator.__class__.__name__:
                raise _refuse("operator_identity_mismatch", f"payload.steps[{index}] name does not match its canonical operator")
        elif declaration is not None:
            raise _refuse("invalid_operator", "SampleIndexFilter is an explicit index operation and takes no operator declaration")
        elif item["type"] != "filter":
            raise _refuse("invalid_operator", "SampleIndexFilter is only valid as a filter step")
        result.append(PreviewStep(id=item["id"], type=item["type"], name=item["name"], operator=operator,
                                  params=cast(dict[str, Any], params), enabled=item.get("enabled", True)))
    return result


def _input_stamps(config: dict[str, Any]) -> tuple[tuple[str, int, int, int], ...] | None:
    """Return the identity of every declared input file, or None when one cannot be read."""
    try:
        paths = dict.fromkeys(_references(normalize_dataset_document(config)))
        stats = [(str(path), path.stat()) for path in paths]
    except Exception:
        # The loader reports the actual problem; such a request is never reused.
        return None
    return tuple((path, stat.st_size, stat.st_mtime_ns, stat.st_ino) for path, stat in stats) or None


def _dataset(value: Any) -> tuple[Any, dict[str, Any]]:
    global _loaded_dataset
    item = _object(value, required={"config"}, allowed={"config", "load_limits", "max_input_bytes"}, where="payload.dataset")
    if type(item["config"]) is not dict:
        raise _refuse("invalid_dataset", "payload.dataset.config must be one authorized canonical dataset object")
    maximum = _integer(item.get("max_input_bytes", 512 * 1024 * 1024), "dataset.max_input_bytes", minimum=1)
    load_limits = item.get("load_limits")
    if load_limits is not None and type(load_limits) is not dict:
        raise _refuse("invalid_dataset", "dataset.load_limits must be an object")
    key = json.dumps({"config": item["config"], "load_limits": load_limits, "max_input_bytes": maximum},
                     sort_keys=True, separators=(",", ":"))
    stamps = _input_stamps(item["config"])
    if stamps is not None and _loaded_dataset is not None and _loaded_dataset[:2] == (key, stamps):
        return _loaded_dataset[2], dict(_loaded_dataset[3])
    # Release the previous dataset before loading another one.
    _loaded_dataset = None
    try:
        dataset, reader = load_dataset_for_analysis(item["config"], load_limits=load_limits, max_input_bytes=maximum)
    except Exception as error:
        raise _refuse("dataset_load_failed", str(error)) from error
    # A file that changed while it was read is not kept.
    if stamps is not None and _input_stamps(item["config"]) == stamps:
        _loaded_dataset = (key, stamps, dataset, dict(reader))
    return dataset, reader


def _omit_unchanged_processed_spectra(result: dict[str, Any]) -> None:
    """Send the spectra once when processing left them bitwise unchanged."""
    original, processed = result.get("original"), result.get("processed")
    if not isinstance(original, dict) or not isinstance(processed, dict):
        return
    before, after = original.get("spectra"), processed.get("spectra")
    if (isinstance(before, np.ndarray) and isinstance(after, np.ndarray) and before.dtype == after.dtype
            and before.shape == after.shape and before.tobytes() == after.tobytes()):
        del processed["spectra"]
        result["processed_is_original"] = True


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"data", "dataset", "selection", "steps", "sampling", "options", "limits"}
    if payload.keys() - allowed or ("data" in payload) == ("dataset" in payload):
        raise _refuse("invalid_shape", "execute requires exactly one of payload.data or payload.dataset")
    limits = _limits(payload.get("limits"))
    steps = _steps(payload.get("steps", []))
    sampling = _sampling(payload.get("sampling"))
    options = dict(_options(payload.get("options")))
    omit_unchanged = options.pop("omit_unchanged_processed_spectra", False)
    if "data" in payload:
        data = _object(payload["data"], required={"x"},
                       allowed={"x", "y", "wavelengths", "sample_ids", "metadata", "partitions", "header_unit"}, where="payload.data")
        result = preview_arrays(data["x"], y=data.get("y"), wavelengths=data.get("wavelengths"),
                                sample_ids=data.get("sample_ids"), metadata=data.get("metadata"), partitions=data.get("partitions"),
                                header_unit=data.get("header_unit"), steps=steps, sampling=sampling, options=options, limits=limits)
    else:
        dataset, reader = _dataset(payload["dataset"])
        selection = _object(payload.get("selection", {}), required=set(),
                            allowed={"partition", "source_index", "target_index"}, where="payload.selection")
        result = preview_spectro_dataset(
            dataset,
            partition=selection.get("partition", "all"),
            source_index=_integer(selection.get("source_index", 0), "selection.source_index"),
            target_index=_integer(selection.get("target_index", 0), "selection.target_index"),
            steps=steps,
            sampling=sampling,
            options=options,
            limits=limits,
        )
        result["dataset_reader"] = reader
    if omit_unchanged:
        _omit_unchanged_processed_spectra(result)
    return result


def _dispatch(operation: str, payload: dict[str, Any]) -> Any:
    if operation == "execute":
        return _execute(payload)
    if operation == "capabilities":
        if payload:
            raise _refuse("unknown_field", "capabilities payload must be empty")
        defaults = PreviewLimits()
        return {"operations": sorted(_OPERATIONS), "sampling_methods": ["all", "random", "stratified", "kmeans"],
                "distance_metrics": sorted(DISTANCE_METRICS), "spectral_descriptors": list(ALL_METRICS),
                "stateless": True, "cache": False,
                "default_limits": {"max_samples": defaults.max_samples, "max_features": defaults.max_features, "max_cells": defaults.max_cells}}
    if operation == "validate":
        if payload.keys() - {"steps"}:
            raise _refuse("unknown_field", "validate accepts only payload.steps")
        steps = _steps(payload.get("steps", []))
        return {"valid": True, "steps": [{"id": step.id, "name": step.name, "type": step.type} for step in steps]}
    if operation == "diff":
        item = _object(payload, required={"reference", "final"}, allowed={"reference", "final", "metric", "scale"}, where="payload")
        return paired_spectral_distances(item["reference"], item["final"], metric=item.get("metric", "euclidean"), scale=item.get("scale", "linear"))
    if operation == "repetition_variance":
        item = _object(payload, required={"x", "group_ids"}, allowed={"x", "group_ids", "reference", "metric"}, where="payload")
        return repetition_variance(item["x"], item["group_ids"], reference=item.get("reference", "group_mean"), metric=item.get("metric", "euclidean"))
    if operation == "metadata_columns":
        if payload.keys() - {"dataset", "partition", "max_unique_values"} or "dataset" not in payload:
            raise _refuse("invalid_shape", "metadata_columns requires payload.dataset and bounded options")
        dataset, reader = _dataset(payload["dataset"])
        result = playground_metadata_columns(dataset, partition=payload.get("partition", "train"),
                                             max_unique_values=_integer(payload.get("max_unique_values", 200), "max_unique_values", minimum=1))
        result["dataset_reader"] = reader
        return result
    raise _refuse("unsupported_operation", f"Unsupported Playground operation: {operation}")


def _json_native(value: Any, diagnostics: dict[str, int]) -> Any:
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist(), diagnostics)
    if isinstance(value, np.generic):
        return _json_native(value.item(), diagnostics)
    if isinstance(value, float) and not math.isfinite(value):
        diagnostics["non_finite_values_encoded_as_null"] = diagnostics.get("non_finite_values_encoded_as_null", 0) + 1
        return None
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_native(item, diagnostics) for item in value]
    if isinstance(value, list):
        return [_json_native(item, diagnostics) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item, diagnostics) for key, item in value.items()}
    return value


def studio_playground_job_v1(request: object) -> dict[str, Any]:
    """Run one closed Playground request for an attested stdio caller."""
    _validate_json(request)
    root = _object(request, required={"schema", "operation", "request_id", "payload"},
                   allowed={"schema", "operation", "request_id", "payload"}, where="request")
    if root["schema"] != STUDIO_PLAYGROUND_JOB_SCHEMA:
        raise _refuse("unsupported_contract", "expected the Studio Playground v1 contract")
    operation = root["operation"]
    request_id = root["request_id"]
    if operation not in _OPERATIONS:
        raise _refuse("unsupported_operation", "Playground operation is not supported")
    if not isinstance(request_id, str) or not request_id or len(request_id.encode("utf-8")) > 256 or any(ord(char) < 32 for char in request_id):
        raise _refuse("invalid_request_id", "request_id must be a non-empty bounded identifier")
    if type(root["payload"]) is not dict:
        raise _refuse("invalid_shape", "payload must be a JSON object")
    payload_value = cast(dict[str, Any], root["payload"])
    payload = _object(payload_value, required=set(), allowed=set(payload_value), where="payload")
    try:
        value = _dispatch(operation, payload)
    except StudioScientificJobError:
        raise
    except (MemoryError, OverflowError) as error:
        raise _refuse("resource_limit", str(error) or type(error).__name__) from error
    except (TypeError, ValueError, IndexError) as error:
        raise _refuse("invalid_playground_request", str(error)) from error
    wire_diagnostics: dict[str, int] = {}
    response = _json_native({"schema": STUDIO_PLAYGROUND_RESULT_SCHEMA, "request_id": request_id,
                             "operation": operation, "result": value}, wire_diagnostics)
    if wire_diagnostics:
        response["wire_diagnostics"] = wire_diagnostics
    return cast(dict[str, Any], response)


__all__ = [
    "STUDIO_PLAYGROUND_JOB_SCHEMA",
    "STUDIO_PLAYGROUND_RESULT_SCHEMA",
    "studio_playground_job_v1",
]
