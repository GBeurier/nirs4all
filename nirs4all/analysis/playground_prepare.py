"""Preparation and row-lineage bookkeeping for exploratory pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from nirs4all.analysis.playground_steps import sample_batch
from nirs4all.analysis.playground_types import PreviewBatch, positive_count


@dataclass(frozen=True)
class PreviewStep:
    """One explicit, already resolved operator and its preview-only options."""

    id: str
    type: str
    name: str
    operator: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


def prepare_batch(batch: PreviewBatch, sampling: dict[str, Any] | None,
                  options: dict[str, Any]) -> tuple[PreviewBatch, dict[str, Any] | None, list[str]]:
    """Apply visible-subset then sampling policies without losing full-row IDs."""
    if batch.origins is None:
        batch = replace(batch, origins=np.arange(len(batch.x), dtype=np.intp))
    total = len(batch.x)
    subset = None
    messages: list[str] = []
    mode = options.get("subset_mode", "all")
    if mode not in {"all", "visible"}:
        raise ValueError(f"Unknown subset_mode: {mode}")
    if mode == "visible":
        count = positive_count(options.get("max_samples_displayed", 200), "max_samples_displayed")
        batch, subset_messages = sample_batch(batch, method="stratified" if batch.y is not None else "random",
                                              n_samples=min(count, total), seed=42)
        messages.extend(subset_messages)
        subset = {"subset_mode": mode, "total_samples": total, "displayed_samples": len(batch.x)}
    if sampling:
        batch, sampling_messages = sample_batch(batch, **sampling)
        messages.extend(sampling_messages)
    return batch, subset, messages


def partition_summary(batch: PreviewBatch) -> dict[str, Any] | None:
    """Expose exact row memberships, even when augmentation interleaves parts."""
    if batch.partitions is None:
        return None
    train, test = np.flatnonzero(batch.partitions == "train"), np.flatnonzero(batch.partitions == "test")
    return {"has_test": bool(len(test)), "n_train": len(train), "n_test": len(test),
            "train_indices": train.tolist(), "test_indices": test.tolist(), "scope": "current_rows"}


def remap_folds(folds: dict[str, Any] | None, parents: np.ndarray,
                batch: PreviewBatch) -> dict[str, Any] | None:
    """Map old row positions to surviving/new copies without recomputing a split."""
    if folds is None:
        return None
    result = {**folds, "folds": []}
    labels = np.asarray(folds["fold_labels"])
    result["fold_labels"] = labels[parents].tolist()
    for fold in folds["folds"]:
        item = dict(fold)
        for part in ("train", "test"):
            rows = np.flatnonzero(np.isin(parents, fold[f"{part}_indices"]))
            item[f"{part}_indices"] = rows.tolist()
            item[f"{part}_count"] = len(rows)
            if batch.origins is not None:
                item[f"{part}_origins"] = batch.origins[rows].tolist()
            item.pop(f"y_{part}_stats", None)
            if batch.y is not None and len(rows):
                values = batch.y[rows]
                item[f"y_{part}_stats"] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                          "min": float(np.min(values)), "max": float(np.max(values))}
        result["folds"].append(item)
    return result


def establish_test_partition(batch: PreviewBatch, folds: dict[str, Any]) -> PreviewBatch:
    """Use the explicitly selected (otherwise first) holdout for later CV.

    The complete first-split preview is retained; this records which of its
    possible holdouts determines the train/test distinction for later steps.
    """
    index = folds.get("split_index") or 0
    if not folds["folds"]:
        raise ValueError("Splitter returned no folds")
    chosen = folds["folds"][index]
    partitions = np.full(len(batch.x), "unassigned", dtype=object)
    partitions[chosen["train_indices"]] = "train"
    partitions[chosen["test_indices"]] = "test"
    folds["partition_fold_index"] = index
    return replace(batch, partitions=partitions)
