"""Portable typed classification target codec shared by data and storage.

Sorted vocabulary positions encode target values only; probability columns keep
exactly their model-owned class order. Numeric historical rows remain unchanged.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def classification_task(task_type: Any) -> bool:
    """Recognize the classification task names persisted by the SDK."""
    return isinstance(task_type, str) and task_type in {"classification", "binary_classification", "multiclass_classification"}


def encode_target_labels(values: np.ndarray | None) -> str | None:
    """Encode homogeneous string/int64 labels without floating-point conversion."""
    if values is None:
        return None
    array = np.asarray(values)
    if array.dtype.kind == "f":
        return None  # Preserve existing numeric classification storage.
    if array.ndim not in (1, 2) or array.size > 16_777_216:
        raise ValueError("classification label arrays must be bounded vectors or matrices")
    labels = [value.item() if isinstance(value, np.generic) else value for value in array.ravel()]
    if array.dtype.kind == "U" or labels and all(type(value) is str for value in labels):
        label_type = "str"
    elif array.dtype.kind in "iu" or labels and all(type(value) is int for value in labels):
        if any(type(value) is not int or not -(1 << 63) <= value < (1 << 63) for value in labels):
            raise ValueError("classification integer labels must fit signed int64")
        label_type = "int64"
    else:
        raise ValueError("classification labels must be homogeneous strings or signed int64 integers")
    vocabulary = sorted(set(labels))
    positions = {label: index for index, label in enumerate(vocabulary)}
    # Retain historical flat mono-y reads, including an input column vector.
    shape = [len(array)] if array.ndim == 2 and array.shape[1] == 1 else list(array.shape)
    return json.dumps({"schema_version": 1, "label_type": label_type, "vocabulary": vocabulary,
                       "indices": [positions[label] for label in labels], "shape": shape},
                      ensure_ascii=True, separators=(",", ":"))


def decode_target_array(row: dict[str, Any], field: str) -> np.ndarray | None:
    """Decode one portable target column, refusing malformed typed label data.

    Older numeric Parquet rows need no label columns. The same decoder serves
    workspace reloads and standalone ``Predictions.from_parquet`` reads.
    """
    if field not in {"y_true", "y_pred"}:
        raise ValueError("unknown prediction target field")
    encoded = row.get(f"{field}_labels")
    values = row.get(field)
    if encoded is None:
        if values is None:
            return None
        array = np.array(values, dtype=np.float64)
        shape = row.get(f"{field}_shape")
        return array.reshape(shape) if shape is not None else array
    if values is not None or not classification_task(row.get("task_type")) or not isinstance(encoded, str):
        raise ValueError("typed classification labels conflict with numeric targets or task type")
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid classification label encoding") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "label_type", "vocabulary", "indices", "shape"}:
        raise ValueError("invalid classification label encoding fields")
    if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
            or type(payload["label_type"]) is not str or payload["label_type"] not in {"str", "int64"}):
        raise ValueError("unsupported classification label encoding")
    vocabulary, indices, shape = payload["vocabulary"], payload["indices"], payload["shape"]
    label_type = payload["label_type"]
    if not isinstance(vocabulary, list) or not all(type(label) is (str if label_type == "str" else int) for label in vocabulary):
        raise ValueError("classification vocabulary must be homogeneous and typed")
    if label_type == "int64" and any(not -(1 << 63) <= label < (1 << 63) for label in vocabulary):
        raise ValueError("classification vocabulary integer exceeds int64")
    if vocabulary != sorted(set(vocabulary)):
        raise ValueError("classification vocabulary must be sorted and unique")
    if not isinstance(shape, list) or len(shape) not in (1, 2) or any(type(size) is not int or not 0 <= size <= 16_777_216 for size in shape):
        raise ValueError("invalid classification label shape")
    count = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    if count > 16_777_216 or not isinstance(indices, list) or len(indices) != count:
        raise ValueError("classification label shape or size differs from its indices")
    if any(type(index) is not int or not 0 <= index < len(vocabulary) for index in indices):
        raise ValueError("classification label index outside vocabulary")
    if row.get(f"{field}_shape") is not None and row[f"{field}_shape"] != shape:
        raise ValueError("classification label shape differs from target shape")
    return np.asarray([vocabulary[index] for index in indices], dtype=str if label_type == "str" else np.int64).reshape(shape)
