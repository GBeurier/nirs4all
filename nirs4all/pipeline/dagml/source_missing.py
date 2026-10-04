"""Explicit missing-source prediction policy shared by live and archive hosts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def predict_present_rows(predict: Callable[[Any], Any], block: Any, presence: np.ndarray, output_width: int) -> np.ndarray:
    """Predict observed rows in original target space; absent rows remain zero.

    The callable must include any captured inverse target transform. No encoder
    or model is called when this source is absent for every requested row.
    """
    mask = np.asarray(presence)
    if mask.dtype.kind != "b" or mask.shape != (len(block),):
        raise ValueError("source presence must be a boolean mask aligned with the raw rows")
    if type(output_width) is not int or output_width < 1:
        raise ValueError("source prediction width must be a positive integer")
    values = np.zeros((len(mask), output_width), dtype=float)
    if mask.any():
        observed = block if mask.all() else block.take_rows(mask) if hasattr(block, "take_rows") else block[mask]
        prediction = np.asarray(predict(observed), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if prediction.shape != (int(mask.sum()), output_width) or not np.isfinite(prediction).all():
            raise ValueError("observed source predictions must be finite and match the declared target width")
        values[mask] = prediction
    return values


def append_source_presence(predictions: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Append one declared availability feature to one source's predictions."""
    values, mask = np.asarray(predictions, dtype=float), np.asarray(presence)
    if values.ndim != 2 or mask.dtype.kind != "b" or mask.shape != (len(values),):
        raise ValueError("source predictions and boolean presence must have aligned rows")
    if not np.isfinite(values).all() or np.any(values[~mask] != 0):
        raise ValueError("missing-source predictions must be zero and observed predictions finite")
    return np.column_stack([values, mask.astype(float)])


def prediction_source_presence_metadata(dataset: Any, sample_ids: list[int]) -> dict[str, dict[int, Any]]:
    """Declare IO availability before a native Test/PREDICT cohort is signed."""
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    if not isinstance(dataset, MultimodalSpectroDataset):
        raise ValueError("prediction availability requires a typed multimodal cohort")
    if len(sample_ids) != len(set(sample_ids)) or any(type(row) is not int or not 0 <= row < len(dataset.cohort) for row in sample_ids):
        raise ValueError("prediction availability requires unique cohort row indices")
    masks = dataset.cohort.source_presence()
    return {"prediction_source_presence": {
        row: {name: bool(masks[name][row]) for name in dataset.source_names} for row in sample_ids
    }}


def build_prediction_availability(dataset: Any, identity: Any, sample_ids: list[int], *,
                                  target_values: Any, classification: bool) -> dict[str, Any]:
    """Sign IO masks and encoded class IDs without inventing folds or labels."""
    presence = prediction_source_presence_metadata(dataset, sample_ids)["prediction_source_presence"]
    values = np.asarray(target_values, dtype=float).reshape(len(sample_ids), -1)
    mask = np.asarray(dataset.cohort.target_mask)
    if mask.dtype.kind != "b":
        raise ValueError("target availability requires an explicit boolean IO mask")
    mask = mask.reshape(len(dataset.cohort), -1)[sample_ids]
    names = list(dataset.cohort.target_names)
    if values.shape != mask.shape or values.shape[1] != len(names) or not np.isfinite(values[mask]).all():
        raise ValueError("target availability must match named finite observed targets")
    result = {
        "schema_version": 1,
        "sample_ids": [identity.to_wire(row) for row in sample_ids],
        "source_presence": {name: [presence[row][name] for row in sample_ids] for name in dataset.source_names},
        "target_names": names, "target_validity_masks": mask.tolist(),
    }
    if classification:
        if values.shape[1] != 1 or not mask.all():
            raise ValueError("partial-source classification requires complete mono-target labels")
        classes = np.unique(values[:, 0])
        if len(classes) < 2:
            raise ValueError("partial-source classification requires at least two native classes")
        result.update(class_labels=classes.tolist(), sample_labels=values[:, 0].tolist())
    return result


def checked_target_validity(resolved: dict[str, Any], sample_ids: list[str], availability: dict[str, Any], *,
                            fitting: bool) -> np.ndarray:
    """Verify actual target masks against signed Train authority for fit rows."""
    values = np.asarray(resolved["values"]).reshape(len(sample_ids), -1)
    raw = resolved.get("validity_masks")
    mask = np.ones(values.shape, dtype=bool) if raw is None else np.asarray(raw)
    if mask.dtype.kind != "b" or mask.shape != values.shape or resolved.get("target_names") != availability["target_names"]:
        raise ValueError("actual target validity disagrees with the signed target schema")
    if fitting:
        rows = dict(zip(availability["sample_ids"], availability["target_validity_masks"], strict=True))
        if any(sample not in rows for sample in sample_ids) or not np.array_equal(mask, np.asarray([rows[sample] for sample in sample_ids])):
            raise ValueError("actual fit target validity changed after native admission")
    return mask


def checked_source_presence(resolver: Any, sample_ids: list[str], source_index: int, source_name: str,
                            availability: dict[str, Any], *, fold_label: str | None = None,
                            fitting: bool = False) -> np.ndarray:
    """Resolve true IO presence, checking signed Train rows before fitting."""
    mask = np.asarray(resolver.resolve_source_presence(sample_ids, source_index, fold_label=fold_label))
    if mask.dtype.kind != "b" or mask.shape != (len(sample_ids),):
        raise ValueError("actual source presence must be aligned and boolean")
    rows = dict(zip(availability["sample_ids"], availability["source_presence"][source_name], strict=True))
    for row, sample in enumerate(sample_ids):
        if sample in rows and bool(mask[row]) != rows[sample]:
            raise ValueError("actual source presence changed after native admission")
        if fitting and sample not in rows:
            raise ValueError("native fit row is outside signed Train availability")
    return mask


def checked_prediction_features(spec: dict[str, Any], presence: np.ndarray) -> np.ndarray:
    """Consume native sparse-join features; zero cells are never distributions."""
    values = np.asarray(spec["values"], dtype=float)
    masks, native_presence = np.asarray(spec.get("feature_validity_masks")), np.asarray(spec.get("source_presence"))
    if (values.ndim != 2 or values.shape[0] != len(spec["sample_ids"])
            or type(spec.get("prediction_width")) is not int or spec["prediction_width"] != values.shape[1]
            or spec["prediction_width"] < 1 or not np.isfinite(values).all() or masks.dtype.kind != "b"
            or masks.shape != values.shape or native_presence.dtype.kind != "b"
            or native_presence.shape != (len(values),) or not np.array_equal(native_presence, presence)
            or not np.array_equal(masks, np.broadcast_to(presence[:, None], values.shape))
            or np.any(values[~masks] != 0)):
        raise ValueError("native prediction features require exact paired availability masks and zero absent cells")
    return values
