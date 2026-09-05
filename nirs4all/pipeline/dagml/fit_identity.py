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
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .identity import validate_data_id

# Public cross-runtime identity profile for the native Methods PREDICT path.
#
# The old raw-array hash included NumPy's dtype spelling and host byte order;
# Rust IO cannot reproduce either property without carrying Python internals.
# New native cohorts therefore use an unambiguous f64, little-endian preimage.
# Existing packages retain their already-attested SHA-256 values; this profile
# applies only when a new X-only cohort is materialized for native replay.
MATRIX_F64_LE_FINGERPRINT_PROFILE = "n4a-matrix-f64-le.v1"
_MATRIX_F64_LE_FINGERPRINT_PREFIX = (
    MATRIX_F64_LE_FINGERPRINT_PROFILE.encode("ascii") + b"\0"
)


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


@dataclass(frozen=True)
class DagMLPredictIdentityFrame:
    """Normalized identity and feature-content proof for one X-only replay.

    PREDICT must not manufacture target data merely to reuse a training identity.
    This frame therefore binds exactly the supplied feature rows, stable sample
    ids and optional group/metadata columns, and deliberately has no target
    fingerprint.
    """

    n_samples: int
    sample_ids: tuple[str, ...]
    groups: tuple[str | None, ...]
    metadata_rows: tuple[dict[str, Any], ...]
    explicit_sample_ids: bool
    data_content_fingerprint: str
    fingerprint: str

    def metadata_by_sample_id(self) -> dict[str, dict[str, Any]]:
        """Return row metadata keyed by the exact public sample id."""

        return {sample_id: dict(row) for sample_id, row in zip(self.sample_ids, self.metadata_rows, strict=True) if row}

    def group_by_sample_id(self) -> dict[str, str]:
        """Return non-null groups keyed by the exact public sample id."""

        return {
            sample_id: group
            for sample_id, group in zip(self.sample_ids, self.groups, strict=True)
            if group is not None
        }


@dataclass(frozen=True)
class DagMLCalibrationIdentityFrame:
    """Explicit PREDICT identities plus an attested measured-target proof.

    Calibration is evaluated through a PREDICT replay so a Methods provider
    does not receive targets as execution inputs.  Unlike ordinary inference,
    its replay provenance must still bind the separately supplied measured
    truth.  The target fingerprint has the same raw-array profile as native
    training and is never synthesized from a sentinel.
    """

    n_samples: int
    sample_ids: tuple[str, ...]
    groups: tuple[str | None, ...]
    metadata_rows: tuple[dict[str, Any], ...]
    explicit_sample_ids: bool
    data_content_fingerprint: str
    target_content_fingerprint: str
    fingerprint: str

    def metadata_by_sample_id(self) -> dict[str, dict[str, Any]]:
        """Return row metadata keyed by the exact public sample id."""

        return {sample_id: dict(row) for sample_id, row in zip(self.sample_ids, self.metadata_rows, strict=True) if row}

    def group_by_sample_id(self) -> dict[str, str]:
        """Return non-null groups keyed by the exact public sample id."""

        return {
            sample_id: group
            for sample_id, group in zip(self.sample_ids, self.groups, strict=True)
            if group is not None
        }


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


def normalize_predict_identity(
    X: Any,
    *,
    sample_ids: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    require_explicit_sample_ids: bool = False,
) -> DagMLPredictIdentityFrame:
    """Normalize an X-only PREDICT cohort without creating a target sentinel.

    Compatibility sample ids are deterministic over features only.  Callers
    serving persisted artifacts can require explicit ids, which is the only
    mode suitable for relation-sensitive replay and presentation joins.
    """

    n_samples = _infer_x_n_samples(X)
    explicit = sample_ids is not None
    data_content_fingerprint = feature_content_fingerprint(X)
    if sample_ids is None:
        if require_explicit_sample_ids:
            raise ValueError("native DAG-ML predict requires explicit sample_ids for this estimator")
        normalized_sample_ids = _compat_sample_ids(X, None, n_samples)
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
    return DagMLPredictIdentityFrame(
        n_samples=n_samples,
        sample_ids=normalized_sample_ids,
        groups=normalized_groups,
        metadata_rows=metadata_rows,
        explicit_sample_ids=explicit,
        data_content_fingerprint=data_content_fingerprint,
        fingerprint=fingerprint,
    )


def normalize_calibration_identity(
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    require_explicit_sample_ids: bool = False,
) -> DagMLCalibrationIdentityFrame:
    """Bind a measured calibration cohort without changing PREDICT inputs."""

    predict = normalize_predict_identity(
        X,
        sample_ids=sample_ids,
        groups=groups,
        metadata=metadata,
        require_explicit_sample_ids=require_explicit_sample_ids,
    )
    targets = np.ascontiguousarray(np.asarray(y))
    if targets.ndim not in (1, 2) or targets.shape[0] != predict.n_samples:
        raise ValueError("native DAG-ML calibration identity requires row-aligned one- or two-dimensional y")
    if not np.issubdtype(targets.dtype, np.number) or not np.isfinite(targets).all():
        raise ValueError("native DAG-ML calibration identity requires finite numeric y")
    return DagMLCalibrationIdentityFrame(
        n_samples=predict.n_samples,
        sample_ids=predict.sample_ids,
        groups=predict.groups,
        metadata_rows=predict.metadata_rows,
        explicit_sample_ids=predict.explicit_sample_ids,
        data_content_fingerprint=predict.data_content_fingerprint,
        target_content_fingerprint=target_content_fingerprint(y),
        fingerprint=predict.fingerprint,
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


def _infer_x_n_samples(X: Any) -> int:
    x_shape = getattr(X, "shape", None)
    n_samples = int(x_shape[0]) if x_shape is not None and len(x_shape) >= 1 else len(X)
    if n_samples <= 0:
        raise ValueError("native DAG-ML predict requires at least one sample")
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


def _compat_sample_ids(X: Any, y: Any | None, n_samples: int) -> tuple[str, ...]:
    digest = feature_content_fingerprint(X) if y is None else _content_fingerprint(X, y)
    return tuple(validate_data_id(f"n4a.{digest}.s{index}") for index in range(n_samples))


def feature_content_fingerprint(X: Any) -> str:
    """Return the native X-only content identity for an inference cohort.

    This is deliberately the same byte-level identity used by the raw-array
    DAG-ML lowering for its ``data_content_fingerprint``. Absence of a target
    is represented only by the nullable replay target field; it must never
    change the feature identity through a sentinel marker.
    """

    matrix = np.asarray(X, dtype=np.dtype("<f8"), order="C")
    if matrix.ndim != 2:
        raise ValueError(
            "native DAG-ML feature-content identity requires a rank-2 matrix"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "native DAG-ML feature-content identity requires finite feature values"
        )
    rows, cols = matrix.shape
    hasher = hashlib.sha256()
    hasher.update(_MATRIX_F64_LE_FINGERPRINT_PREFIX)
    hasher.update(struct.pack("<QQ", rows, cols))
    hasher.update(matrix.tobytes(order="C"))
    return hasher.hexdigest()


def target_content_fingerprint(y: Any) -> str:
    """Return the exact raw-array target proof used by native training."""

    array = np.ascontiguousarray(np.asarray(y))
    hasher = hashlib.sha256()
    hasher.update(b"y")
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _content_fingerprint(X: Any, y: Any | None) -> str:
    hasher = hashlib.sha256()
    _update_array_hash(hasher, np.asarray(X), "X")
    if y is not None:
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
    "DagMLCalibrationIdentityFrame",
    "DagMLPredictIdentityFrame",
    "feature_content_fingerprint",
    "normalize_calibration_identity",
    "normalize_fit_identity",
    "normalize_predict_identity",
    "target_content_fingerprint",
]
