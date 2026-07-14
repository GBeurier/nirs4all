"""Fit-time sample identity normalization for native DAG-ML estimator paths.

The existing bridge mints identities from ``SpectroDataset`` content.  The
sklearn-style native estimator also receives raw ``X``/``y`` arrays, so P2 needs
an explicit, testable identity frame before P3 compiles data envelopes.  This
module does not build DAG-ML contracts; it only validates row-aligned identity,
group and metadata inputs and exposes helper maps in the shape the existing
envelope builders already consume.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .identity import validate_data_id


@dataclass(frozen=True)
class DagMLFitIdentityFrame:
    """Normalized row-aligned identities for one estimator ``fit`` call."""

    n_samples: int
    sample_ids: tuple[str, ...]
    groups: tuple[str | None, ...]
    metadata_rows: tuple[dict[str, Any], ...]
    explicit_sample_ids: bool
    fingerprint: str

    def metadata_by_sample_int(self) -> dict[str, dict[int, Any]]:
        """Return ``{column: {sample_position: value}}`` for envelope builders."""

        columns: dict[str, dict[int, Any]] = {}
        for index, row in enumerate(self.metadata_rows):
            for column, value in row.items():
                columns.setdefault(column, {})[index] = value
        return columns

    def group_by_sample_int(self) -> dict[int, str]:
        """Return non-null groups keyed by sample position."""

        return {index: group for index, group in enumerate(self.groups) if group is not None}

    def metadata_by_sample_id(self) -> dict[str, dict[str, Any]]:
        """Return row metadata keyed by the normalized sample id."""

        return {sample_id: dict(row) for sample_id, row in zip(self.sample_ids, self.metadata_rows, strict=True) if row}


def normalize_fit_identity(
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    require_explicit_sample_ids: bool = False,
) -> DagMLFitIdentityFrame:
    """Normalize sample identities, groups and metadata for a native fit.

    Without explicit ``sample_ids`` this emits compatibility ids derived from a
    content fingerprint plus row position.  They are deterministic for the
    provided arrays but are not strong enough for future leakage/exchangeability
    claims; callers can set ``require_explicit_sample_ids=True`` to fail closed.
    """

    n_samples = _infer_n_samples(X, y)
    explicit = sample_ids is not None
    if sample_ids is None:
        if require_explicit_sample_ids:
            raise ValueError("native DAG-ML fit requires explicit sample_ids for this estimator")
        normalized_sample_ids = _compat_sample_ids(X, y, n_samples)
    else:
        normalized_sample_ids = _normalize_sample_ids(sample_ids, n_samples)
    normalized_groups = _normalize_groups(groups, n_samples)
    metadata_rows = _normalize_metadata(metadata, n_samples)
    fingerprint = _identity_fingerprint(
        normalized_sample_ids,
        normalized_groups,
        metadata_rows,
        explicit_sample_ids=explicit,
    )
    return DagMLFitIdentityFrame(
        n_samples=n_samples,
        sample_ids=normalized_sample_ids,
        groups=normalized_groups,
        metadata_rows=metadata_rows,
        explicit_sample_ids=explicit,
        fingerprint=fingerprint,
    )


def _infer_n_samples(X: Any, y: Any) -> int:
    x_shape = getattr(X, "shape", None)
    if x_shape is not None and len(x_shape) >= 1:
        n_samples = int(x_shape[0])
    else:
        n_samples = len(X)
    try:
        y_len = len(y)
    except TypeError:
        y_len = n_samples
    if y_len != n_samples:
        raise ValueError(f"X and y must have the same number of samples, got {n_samples} and {y_len}")
    if n_samples <= 0:
        raise ValueError("native DAG-ML fit requires at least one sample")
    return n_samples


def _normalize_sample_ids(sample_ids: Sequence[Any], n_samples: int) -> tuple[str, ...]:
    if len(sample_ids) != n_samples:
        raise ValueError(f"sample_ids length must be {n_samples}, got {len(sample_ids)}")
    normalized = tuple(validate_data_id(str(value)) for value in sample_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample_ids must be unique")
    return normalized


def _normalize_groups(groups: Sequence[Any] | None, n_samples: int) -> tuple[str | None, ...]:
    if groups is None:
        return (None,) * n_samples
    if len(groups) != n_samples:
        raise ValueError(f"groups length must be {n_samples}, got {len(groups)}")
    normalized: list[str | None] = []
    for value in groups:
        if value is None:
            normalized.append(None)
        else:
            group = str(value)
            if not group:
                raise ValueError("group ids must be non-empty when provided")
            normalized.append(group)
    return tuple(normalized)


def _normalize_metadata(
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None,
    n_samples: int,
) -> tuple[dict[str, Any], ...]:
    if metadata is None:
        return tuple({} for _ in range(n_samples))
    if isinstance(metadata, Mapping):
        rows: list[dict[str, Any]] = [{} for _ in range(n_samples)]
        for column, values in metadata.items():
            column_name = _normalize_metadata_key(column)
            if len(values) != n_samples:
                raise ValueError(f"metadata column {column_name!r} length must be {n_samples}, got {len(values)}")
            for index, value in enumerate(values):
                rows[index][column_name] = _normalize_metadata_value(value)
        return tuple(rows)
    if len(metadata) != n_samples:
        raise ValueError(f"metadata rows length must be {n_samples}, got {len(metadata)}")
    rows = []
    for row in metadata:
        if not isinstance(row, Mapping):
            raise TypeError("metadata rows must be mappings")
        rows.append({_normalize_metadata_key(column): _normalize_metadata_value(value) for column, value in row.items()})
    return tuple(rows)


def _normalize_metadata_key(column: Any) -> str:
    key = str(column)
    if not key:
        raise ValueError("metadata column names must be non-empty")
    return key


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata float values must be finite")
        return value
    raise TypeError("metadata values must be JSON scalar values")


def _compat_sample_ids(X: Any, y: Any, n_samples: int) -> tuple[str, ...]:
    digest = _content_fingerprint(X, y)
    return tuple(validate_data_id(f"n4a.{digest}.s{index}") for index in range(n_samples))


def _content_fingerprint(X: Any, y: Any) -> str:
    hasher = hashlib.sha256()
    _update_array_hash(hasher, np.asarray(X), "X")
    _update_array_hash(hasher, np.asarray(y), "y")
    return hasher.hexdigest()


def _update_array_hash(hasher: Any, array: np.ndarray, label: str) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(label.encode("utf-8"))
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.tobytes())


def _identity_fingerprint(
    sample_ids: tuple[str, ...],
    groups: tuple[str | None, ...],
    metadata_rows: tuple[dict[str, Any], ...],
    *,
    explicit_sample_ids: bool,
) -> str:
    payload = {
        "explicit_sample_ids": explicit_sample_ids,
        "groups": groups,
        "metadata_rows": metadata_rows,
        "sample_ids": sample_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "DagMLFitIdentityFrame",
    "normalize_fit_identity",
]
