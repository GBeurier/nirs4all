"""Extract aligned Playground inputs from an assembled scientific dataset.

The application host selects and authorizes the dataset before calling this
module.  Extraction preserves source/target choice, physical row identity,
metadata and real train/test membership; it never invents missing targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from nirs4all.analysis.playground_types import PreviewBatch, PreviewLimits, positive_count


@dataclass(frozen=True)
class PlaygroundDatasetSelection:
    """An extracted preview batch plus auditable selection evidence."""

    batch: PreviewBatch
    evidence: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...] = ()


def _matrix(dataset: Any, partition: str, source_index: int) -> np.ndarray:
    sources = dataset.x(
        {"partition": partition},
        layout="2d",
        concat_source=False,
    )
    values = sources[source_index] if isinstance(sources, list) else sources
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"Dataset {partition} source must be a two-dimensional matrix")
    return matrix


def _target(
    dataset: Any,
    partition: str,
    rows: int,
    target_index: int,
    diagnostics: list[dict[str, Any]],
) -> np.ndarray | None:
    if rows == 0 or not dataset.describe()["has_targets"]:
        return None
    values = np.asarray(dataset.y({"partition": partition}))
    if values.size == 0:
        return None
    if values.ndim == 1:
        if target_index != 0:
            raise ValueError("target_index is out of range")
        selected = values
    else:
        values = values.reshape(values.shape[0], -1)
        if target_index >= values.shape[1]:
            raise ValueError("target_index is out of range")
        selected = values[:, target_index]
    if len(selected) != rows:
        # The legacy target accessor treats an empty index selection as all
        # targets.  Omit that ambiguous target cohort instead of attaching
        # another partition's observations to these spectra.
        diagnostics.append({"code": "target_partition_mismatch", "partition": partition,
                            "policy": "targets_omitted_not_imputed"})
        return None
    return np.asarray(selected)


def _metadata(dataset: Any, partition: str, rows: int) -> dict[str, np.ndarray]:
    if rows == 0:
        return {}
    frame = dataset.metadata({"partition": partition})
    if frame is None or len(frame) == 0:
        return {}
    values = {str(name): np.asarray(column) for name, column in frame.to_dict(as_series=False).items()}
    for name, column in values.items():
        if column.ndim != 1 or len(column) != rows:
            raise ValueError(f"Metadata column {name!r} does not match the {partition} source cohort")
    return values


def _row_ids(dataset: Any, partition: str, rows: int, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    for name in ("physical_sample_id", "sample_id", "sample", "id"):
        values = metadata.get(name)
        if values is not None and len(values) == rows:
            return values.copy(), f"metadata.{name}"
    values = np.asarray(dataset.index_column("sample", {"partition": partition}))
    if values.ndim != 1 or len(values) != rows:
        raise ValueError(f"Dataset row identifiers do not match the {partition} source cohort")
    return values, "dataset.index.sample"


def _merge_metadata(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    n_train: int,
    n_test: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in sorted(train.keys() | test.keys()):
        before = train.get(name, np.full(n_train, None, dtype=object))
        after = test.get(name, np.full(n_test, None, dtype=object))
        result[name] = np.concatenate([before, after])
    return result


def extract_playground_dataset(
    dataset: Any,
    *,
    partition: str = "all",
    source_index: int = 0,
    target_index: int = 0,
    limits: PreviewLimits | None = None,
) -> PlaygroundDatasetSelection:
    """Extract one source/target without changing scientific row semantics."""
    if partition not in {"train", "test", "all"}:
        raise ValueError("partition must be train, test or all")
    source = positive_count(source_index, "source_index", allow_zero=True)
    target = positive_count(target_index, "target_index", allow_zero=True)
    description = dataset.describe()
    if source >= dataset.n_sources:
        raise ValueError("source_index is out of range")
    if target >= description["num_targets"] and (description["num_targets"] or target):
        raise ValueError("target_index is out of range")
    budget = limits or PreviewLimits()

    diagnostics: list[dict[str, Any]] = []
    parts: dict[str, tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray], np.ndarray, str]] = {}
    for name in ("train", "test"):
        x = _matrix(dataset, name, source)
        if len(x):
            budget.admit(*x.shape)
        metadata = _metadata(dataset, name, len(x))
        ids, id_source = _row_ids(dataset, name, len(x), metadata) if len(x) else (np.asarray([], dtype=int), "empty_partition")
        parts[name] = (x, _target(dataset, name, len(x), target, diagnostics), metadata, ids, id_source)

    selected_names = (partition,) if partition != "all" else tuple(name for name in ("train", "test") if len(parts[name][0]))
    if not selected_names:
        raise ValueError("Selected dataset partition is empty")
    if partition != "all" and not len(parts[partition][0]):
        raise ValueError(f"Selected {partition} partition is empty")
    matrices = [parts[name][0] for name in selected_names]
    x = matrices[0].copy() if len(matrices) == 1 else np.concatenate(matrices, axis=0)
    budget.admit(*x.shape)

    target_parts = [parts[name][1] for name in selected_names]
    if all(values is not None for values in target_parts):
        y = np.concatenate([values for values in target_parts if values is not None])
    elif any(values is not None for values in target_parts):
        y = None
        diagnostics.append({"code": "incomplete_target_cohort", "policy": "targets_omitted_not_imputed"})
    else:
        y = None

    if len(selected_names) == 1:
        metadata = {name: values.copy() for name, values in parts[selected_names[0]][2].items()}
    else:
        metadata = _merge_metadata(parts["train"][2], parts["test"][2], len(parts["train"][0]), len(parts["test"][0]))
    sample_ids = np.concatenate([parts[name][3] for name in selected_names])
    partitions = np.concatenate([np.full(len(parts[name][0]), name, dtype=object) for name in selected_names])

    headers = dataset.headers(source)
    original_headers = list(headers) if headers is not None else None
    wavelengths = None
    header_unit = None
    if headers is not None and len(headers) == x.shape[1]:
        try:
            wavelengths = np.asarray(headers, dtype=float)
            header_unit = dataset.header_unit(source)
        except (TypeError, ValueError):
            diagnostics.append({"code": "non_numeric_feature_axis", "policy": "feature_index"})
    elif headers is not None:
        raise ValueError("Dataset headers do not match the selected source feature count")

    batch = PreviewBatch.from_arrays(
        x,
        wavelengths=wavelengths,
        y=y,
        sample_ids=sample_ids,
        metadata=cast(dict[str, ArrayLike], metadata),
        partitions=partitions,
        header_unit=header_unit,
        limits=budget,
    )
    evidence = {
        "partition": partition,
        "source_index": source,
        "target_index": target,
        "n_sources": dataset.n_sources,
        "num_targets": description["num_targets"],
        "n_train": int(np.count_nonzero(partitions == "train")),
        "n_test": int(np.count_nonzero(partitions == "test")),
        "sample_id_sources": [parts[name][4] for name in selected_names],
        "original_headers": original_headers,
        "repetition_column": getattr(dataset, "repetition", None),
    }
    return PlaygroundDatasetSelection(batch=batch, evidence=evidence, diagnostics=tuple(diagnostics))


def playground_metadata_columns(
    dataset: Any,
    *,
    partition: str = "train",
    max_unique_values: int = 200,
) -> dict[str, Any]:
    """Return bounded metadata choices for the existing Playground controls."""
    if partition not in {"train", "test", "all"}:
        raise ValueError("partition must be train, test or all")
    maximum = positive_count(max_unique_values, "max_unique_values")
    selector = {} if partition == "all" else {"partition": partition}
    frame = dataset.metadata(selector)
    columns = []
    if frame is not None and len(frame):
        for name in frame.columns:
            values = frame[name].drop_nulls().unique(maintain_order=True).to_list()
            columns.append({"name": name, "dtype": str(frame[name].dtype), "unique_values": values[:maximum], "n_unique": len(values)})
    return {
        "columns": columns,
        "partition": partition,
        "repetition_column": getattr(dataset, "repetition", None),
        "max_unique_values": maximum,
    }
