"""Pure normalization helpers for Inspector route payloads."""

from __future__ import annotations

import json
import math
from typing import Any

from .inspector_json import sanitize_dict


def _parse_json_like(value: Any) -> Any:
    """Parse JSON-encoded strings commonly returned by chain summary views."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def _split_preprocessing_steps(value: Any) -> list[str]:
    """Split the serialized preprocessing string into stable step labels."""
    if value is None:
        return []

    steps: list[str] = []
    for segment in str(value).split(" | "):
        step = segment.strip()
        if step and step not in steps:
            steps.append(step)
    return steps


def _coerce_target_index(value: Any) -> int | None:
    """Coerce a target index carried by native/legacy metadata."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        index = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        index = int(value)
    elif isinstance(value, str) and value.strip():
        try:
            index = int(value.strip())
        except ValueError:
            return None
    else:
        return None
    if index < 0:
        return None
    return index


def _format_target_label(index: int, target_names: list[str]) -> str:
    fallback_label = f"Target {index + 1}"
    if not target_names:
        return fallback_label
    if len(target_names) == 1:
        return f"{target_names[0]} ({fallback_label})"
    return f"{fallback_label} ({len(target_names)} names)"


def _extract_result_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return result metadata from a normalized chain record when present."""
    variant_params = record.get("variant_params")
    if isinstance(variant_params, dict):
        metadata = variant_params.get("result_metadata")
        if isinstance(metadata, dict):
            return metadata
    metadata = record.get("result_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _build_available_targets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the Inspector target catalog from normalized chain result metadata."""
    buckets: dict[int, dict[str, Any]] = {}

    for record in records:
        metadata = _extract_result_metadata(record)
        dimensions = metadata.get("dimensions")
        dimensions = dimensions if isinstance(dimensions, dict) else {}
        target_name = str(metadata.get("target_name") or "").strip()
        index = _coerce_target_index(dimensions.get("target_index"))
        if index is None:
            index = _coerce_target_index(metadata.get("target_index"))
        if index is None and target_name:
            index = 0
        if index is None:
            continue

        bucket = buckets.setdefault(index, {"index": index, "count": 0, "target_names": set()})
        bucket["count"] += 1
        if target_name:
            bucket["target_names"].add(target_name)

    targets: list[dict[str, Any]] = []
    for index in sorted(buckets):
        bucket = buckets[index]
        target_names = sorted(bucket["target_names"])
        targets.append(
            {
                "index": index,
                "label": _format_target_label(index, target_names),
                "count": bucket["count"],
                "target_names": target_names,
            }
        )
    return targets


def _extract_model_params_from_expanded_config(
    expanded_config: Any,
    model_step_idx: Any,
) -> dict[str, Any] | None:
    """Extract model params from a serialized expanded pipeline config."""
    steps = _parse_json_like(expanded_config)
    if not isinstance(steps, list):
        return None

    try:
        idx = int(model_step_idx) - 1
    except Exception:
        return None

    if idx < 0 or idx >= len(steps):
        return None

    step = steps[idx]

    if isinstance(step, dict) and "model" in step:
        model_spec = step.get("model")
        if isinstance(model_spec, dict):
            params = model_spec.get("params")
            if isinstance(params, dict):
                return params
        return None

    if isinstance(step, dict):
        params = step.get("params")
        if isinstance(params, dict):
            return params

    return None


def _merge_variant_params(
    existing_params: dict[str, Any] | None,
    step_params: dict[str, Any] | None,
    best_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge existing, static pipeline, and best params into a single view."""
    merged: dict[str, Any] = {}
    existing_result_metadata = None
    if isinstance(existing_params, dict):
        merged.update(existing_params)
        existing_result_metadata = existing_params.get("result_metadata")
    if isinstance(step_params, dict):
        merged.update(step_params)
    if isinstance(best_params, dict):
        merged.update(best_params)
    if existing_result_metadata is not None:
        merged["result_metadata"] = existing_result_metadata
    return merged or None


def _load_pipeline_metadata_map(store: Any, pipeline_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load pipeline metadata needed to enrich chain summaries.

    Issues a single ``list_pipelines`` query for the whole workspace and
    keeps only the rows whose ``pipeline_id`` is in *pipeline_ids* — avoiding
    one DB round-trip per distinct pipeline (the former N+1).
    """
    wanted = {pid for pid in pipeline_ids if pid}
    if not wanted:
        return {}

    list_pipelines = getattr(store, "list_pipelines", None)
    if not callable(list_pipelines):
        return {}

    pipeline_map: dict[str, dict[str, Any]] = {}
    df = list_pipelines()
    for row in df.iter_rows(named=True):
        pipeline_id = str(row.get("pipeline_id") or "")
        if pipeline_id in wanted:
            pipeline_map[pipeline_id] = dict(row)

    return pipeline_map


def _normalize_chain_record(
    raw: dict[str, Any],
    pipeline_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize a chain-summary row for frontend consumption."""
    pipeline_map = pipeline_map or {}
    record = sanitize_dict(dict(raw))
    for json_field in ("best_params", "branch_path", "variant_params", "score_maps"):
        record[json_field] = _parse_json_like(record.get(json_field))
    pipeline_id = str(record.get("pipeline_id") or "")
    pipeline = pipeline_map.get(pipeline_id, {})
    step_params = _extract_model_params_from_expanded_config(
        pipeline.get("expanded_config"),
        record.get("model_step_idx"),
    )
    best_params = record.get("best_params")
    best_params = best_params if isinstance(best_params, dict) else None
    record["best_params"] = best_params
    existing_variant_params = record.get("variant_params")
    existing_variant_params = existing_variant_params if isinstance(existing_variant_params, dict) else None
    record["variant_params"] = _merge_variant_params(existing_variant_params, step_params, best_params)
    record["pipeline_name"] = pipeline.get("name")
    record["preprocessing_steps"] = _split_preprocessing_steps(record.get("preprocessings"))
    return record


def _normalize_chain_records(store: Any, df: Any) -> list[dict[str, Any]]:
    """Normalize chain rows and enrich them with pipeline-derived metadata."""
    rows = [dict(row) for row in df.iter_rows(named=True)]
    pipeline_map = _load_pipeline_metadata_map(
        store,
        [str(row.get("pipeline_id") or "") for row in rows],
    )
    return [_normalize_chain_record(row, pipeline_map) for row in rows]


def _matches_task_type_filter(raw_task_type: Any, expected: str | None) -> bool:
    """Match broad frontend task filters against backend task variants."""
    if not expected:
        return True

    actual = str(raw_task_type or "").strip().lower()
    if not actual:
        return False

    expected_norm = expected.strip().lower()
    if expected_norm == "classification":
        return "classification" in actual
    if expected_norm == "regression":
        return "regression" in actual
    return actual == expected_norm


def _is_classification_task(task_type: Any) -> bool:
    """Return True for classification-like task types."""
    return "classification" in str(task_type or "").strip().lower()


def _flatten_numeric_params(params: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric hyperparameters, preserving nested keys with dot notation."""
    flattened: dict[str, float] = {}
    if not isinstance(params, dict):
        return flattened

    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_params(value, name))
            continue

        if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            flattened[name] = float(value)

    return flattened
