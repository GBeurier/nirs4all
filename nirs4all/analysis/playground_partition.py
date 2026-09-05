"""Owner-backed filter and split previews with aligned held-out identities."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone

from nirs4all.analysis.playground_types import PreviewBatch, positive_count
from nirs4all.controllers.splitters.split import resolve_split_groups
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.splitters import GroupedSplitterWrapper


def filter_batch(batch: PreviewBatch, operator: Any, *, mode: str = "remove") -> tuple[PreviewBatch, dict[str, Any]]:
    """Apply SampleFilter fit/get_mask once; tagging never removes rows."""
    if mode not in {"remove", "tag"}:
        raise ValueError(f"Unknown filter mode: {mode}")
    fitted = clone(operator)
    fitted.fit(batch.x.copy(), batch.y.copy() if batch.y is not None else None)
    kwargs = {"metadata": batch.metadata} if "metadata" in inspect.signature(fitted.get_mask).parameters else {}
    mask = np.asarray(fitted.get_mask(batch.x, batch.y, **kwargs))
    if mask.ndim != 1 or len(mask) != len(batch.x) or mask.dtype.kind != "b":
        raise ValueError("Filter must return one boolean per sample")
    return _apply_mask(batch, mask, mode, str(fitted.exclusion_reason))


def select_sample_indices(batch: PreviewBatch, indices: ArrayLike, *, selection: str = "keep",
                          mode: str = "remove") -> tuple[PreviewBatch, dict[str, Any]]:
    """Historical explicit selection, without requiring a new filter parser."""
    if selection not in {"keep", "remove"} or mode not in {"tag", "remove"}:
        raise ValueError("Unknown sample selection or filter mode")
    selected = np.asarray(indices)
    if selected.ndim != 1 or selected.size and selected.dtype.kind not in "iu":
        raise ValueError("Selected sample indices must be integers")
    if selected.size and (selected.min() < 0 or selected.max() >= len(batch.x)):
        raise ValueError("Selected sample index out of range")
    mask = np.isin(np.arange(len(batch.x)), selected)
    if selection == "remove":
        mask = ~mask
    return _apply_mask(batch, mask, mode, f"Sample index {selection} selection")


def _apply_mask(batch: PreviewBatch, mask: np.ndarray, mode: str, reason: str) -> tuple[PreviewBatch, dict[str, Any]]:
    selected = np.flatnonzero(mask)
    rejected = np.flatnonzero(~mask)
    return (batch if mode == "tag" else batch.take(selected)), {
        "mask": mask, "kept": len(selected), "removed": len(rejected), "filter_mode": mode,
        "reason": reason, "sample_indices": rejected.tolist(),
        "sample_origins": batch.origins[rejected].tolist() if batch.origins is not None else rejected.tolist(),
    }


def split_batch(batch: PreviewBatch, operator: Any, *, kind: str = "cv_folds", group_by: Any = None,
                legacy_group: Any = None, repetition: str | None = None, ignore_repetition: bool = False,
                aggregation: str = "mean", y_aggregation: str | None = None,
                split_index: int | None = None) -> tuple[dict[str, Any], list[str]]:
    """Preview real owner splitter folds, keeping source test rows outside CV.

    Labels use current row positions and each fold also supplies input origins.
    Group constraints are resolved by the same library contract as training.
    """
    if kind not in {"cv_folds", "test_split"}:
        raise ValueError(f"Unknown split kind: {kind}")
    if split_index is not None:
        positive_count(split_index, "split_index", allow_zero=True)
    indices = np.arange(len(batch.x))
    if kind == "cv_folds" and batch.partitions is not None:
        indices = np.flatnonzero(batch.partitions == "train")
    selected = batch.take(indices)
    if len(selected.x) == 0:
        raise ValueError("Split has no training observations")
    # sklearn CV splitters are not estimators and expose no get_params. The
    # explicit safe=False clone contract deep-copies those input objects.
    fitted = clone(operator, safe=False)
    dataset = SpectroDataset(name="playground_split")
    dataset.add_samples(selected.x, {"partition": "train"})
    if selected.y is not None:
        dataset.add_targets(selected.y)
    if selected.metadata:
        dataset.add_metadata(pd.DataFrame(selected.metadata))
    if repetition:
        dataset.set_repetition(repetition)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_split_groups(dataset, fitted, group_by, legacy_group=legacy_group,
                                        ignore_repetition=ignore_repetition, context={"partition": "train"},
                                        include_augmented=False)
    split_y = selected.y
    if (split_y is not None and "Stratified" in fitted.__class__.__name__
            and np.issubdtype(split_y.dtype, np.number) and np.isfinite(split_y).all()):
        bins = min(5, len(np.unique(split_y)))
        if bins > 1:
            from nirs4all.data.binning import BinningCalculator

            split_y, _ = BinningCalculator.bin_continuous_targets(split_y.astype(float), bins=bins, strategy="quantile")
    if resolved.requires_wrapper:
        fitted = GroupedSplitterWrapper(splitter=fitted, aggregation=aggregation, y_aggregation=y_aggregation)
    kwargs: dict[str, Any] = {}
    if split_y is not None and ("y" in inspect.signature(fitted.split).parameters or resolved.effective_groups is not None):
        kwargs["y"] = split_y
    if resolved.effective_groups is not None:
        kwargs["groups"] = resolved.effective_groups
    folds = []
    labels = np.full(len(batch.x), -1, dtype=int)
    for fold, (train_local, test_local) in enumerate(fitted.split(selected.x, **kwargs)):
        local_rows = []
        for part in (train_local, test_local):
            rows = np.asarray(part)
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or (
                rows.size and (rows.min() < 0 or rows.max() >= len(indices) or len(np.unique(rows)) != len(rows))
            ):
                raise ValueError("Splitter returned invalid or duplicate row indices")
            local_rows.append(rows)
        train, test = indices[local_rows[0]], indices[local_rows[1]]
        if np.intersect1d(train, test).size:
            raise ValueError("Splitter returned overlapping training and validation rows")
        item: dict[str, Any] = {"fold_index": fold, "train_count": len(train), "test_count": len(test),
                                "train_indices": train.tolist(), "test_indices": test.tolist()}
        for part, rows in (("train", train), ("test", test)):
            if batch.origins is not None:
                item[f"{part}_origins"] = batch.origins[rows].tolist()
            if batch.y is not None and len(rows):
                values = batch.y[rows]
                item[f"y_{part}_stats"] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                          "min": float(np.min(values)), "max": float(np.max(values))}
        folds.append(item)
        if split_index is None or fold == split_index:
            labels[test] = fold
    if split_index is not None and split_index >= len(folds):
        raise ValueError("split_index is out of range")
    mode = "combined" if resolved.uses_repetition and resolved.uses_group_by else (
        "repetition_only" if resolved.uses_repetition else "group_by_only" if resolved.uses_group_by else "none")
    group_parts = [resolved.group_by] if isinstance(resolved.group_by, str) else list(resolved.group_by or [])
    effective_parts = ([repetition] if resolved.uses_repetition and repetition else []) + group_parts
    return {"splitter_name": operator.__class__.__name__, "n_folds": len(folds), "folds": folds,
            "fold_labels": labels.tolist(), "split_index": split_index, "kind": kind,
            "repetition_column": repetition, "group_by": " + ".join(group_parts) or None,
            "effective_group_mode": mode, "effective_group_label": " + ".join(effective_parts) or None}, list(dict.fromkeys(str(w.message) for w in caught))
