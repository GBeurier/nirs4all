"""Aligned, host-independent inputs for spectral exploration.

These objects are not training plans. Sample origins identify input rows and
may repeat after augmentation; they must never be confused with new wire IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nirs4all.analysis.playground_statistics import spectral_matrix


def positive_count(value: Any, name: str, *, allow_zero: bool = False) -> int:
    """Validate counters without truncating floats or treating booleans as ints."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")
    return integer


@dataclass(frozen=True)
class PreviewLimits:
    """Configurable admission bounds, not a promise about peak process RSS.

    Defaults retain historical 10000 x 10000 preview dimensions. Hosts can
    tighten or explicitly raise them, and must also bound serialized input.
    """

    max_samples: int = 10_000
    max_features: int = 10_000
    max_cells: int = 100_000_000

    def __post_init__(self) -> None:
        for name in ("max_samples", "max_features", "max_cells"):
            positive_count(getattr(self, name), name)

    def admit(self, rows: int, columns: int) -> None:
        """Check prospective shape before allocating an expanded matrix."""
        positive_count(rows, "rows", allow_zero=True)
        positive_count(columns, "columns")
        if rows > self.max_samples or columns > self.max_features or rows * columns > self.max_cells:
            raise ValueError("Preview shape exceeds host limits; explicitly raise PreviewLimits for this workload")


@dataclass(frozen=True)
class PreviewBatch:
    """One matrix with aligned optional targets, metadata, partitions and axes."""

    x: NDArray[np.float64]
    wavelengths: NDArray[np.float64]
    y: NDArray[Any] | None = None
    sample_ids: NDArray[Any] | None = None
    metadata: dict[str, NDArray[Any]] = field(default_factory=dict)
    origins: NDArray[np.intp] | None = None
    partitions: NDArray[Any] | None = None
    header_unit: str | None = None
    axis_kind: str = "wavelength"

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[1] == 0:
            raise ValueError("Preview X must be two-dimensional with at least one feature")
        if self.wavelengths.ndim != 1 or len(self.wavelengths) != self.x.shape[1]:
            raise ValueError("wavelengths must match the feature axis")
        columns = [("y", self.y), ("sample_ids", self.sample_ids), ("origins", self.origins),
                   ("partitions", self.partitions), *[(f"metadata.{key}", value) for key, value in self.metadata.items()]]
        for name, values in columns:
            if values is not None and (values.ndim != 1 or len(values) != len(self.x)):
                raise ValueError(f"{name} must have one value per sample")

    @classmethod
    def from_arrays(cls, x: ArrayLike, *, wavelengths: ArrayLike | None = None,
                    y: ArrayLike | None = None, sample_ids: ArrayLike | None = None,
                    metadata: dict[str, ArrayLike] | None = None, partitions: ArrayLike | None = None,
                    header_unit: str | None = None, limits: PreviewLimits | None = None) -> PreviewBatch:
        """Validate inputs; absent y stays absent, never generated from row IDs."""
        budget = limits or PreviewLimits()
        # Admit the declared rectangular extent before materializing float64.
        # Rectangularity and all side-column lengths are checked immediately
        # afterwards; the host remains responsible for its decoder allocation.
        shape = getattr(x, "shape", None)
        if shape is not None and len(shape) == 2:
            budget.admit(*shape)
        elif isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple, np.ndarray)):
            budget.admit(len(x), len(x[0]))
        matrix = spectral_matrix(x)
        budget.admit(*matrix.shape)
        return cls(
            x=matrix, wavelengths=np.asarray(wavelengths, dtype=float) if wavelengths is not None else np.arange(matrix.shape[1], dtype=float),
            y=np.asarray(y) if y is not None else None,
            sample_ids=np.asarray(sample_ids) if sample_ids is not None else None,
            metadata={key: np.asarray(value) for key, value in (metadata or {}).items()},
            origins=np.arange(len(matrix), dtype=np.intp),
            partitions=np.asarray(partitions) if partitions is not None else None,
            header_unit=header_unit, axis_kind="wavelength" if wavelengths is not None else "feature_index",
        )

    def take(self, indices: ArrayLike) -> PreviewBatch:
        """Select or repeat rows consistently, preserving their input origins."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise ValueError("sample indices must be a one-dimensional integer array")
        if selected.size and (selected.min() < 0 or selected.max() >= len(self.x)):
            raise ValueError("sample index out of range")
        return replace(self, x=self.x[selected], y=self.y[selected] if self.y is not None else None,
                       sample_ids=self.sample_ids[selected] if self.sample_ids is not None else None,
                       metadata={key: value[selected] for key, value in self.metadata.items()},
                       origins=self.origins[selected] if self.origins is not None else None,
                       partitions=self.partitions[selected] if self.partitions is not None else None)
