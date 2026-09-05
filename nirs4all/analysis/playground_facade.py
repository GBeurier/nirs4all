"""Pure library facade for complete Playground scientific previews."""

from __future__ import annotations

from typing import Any

from nirs4all.analysis.playground_dataset import (
    extract_playground_dataset,
    playground_metadata_columns,
)
from nirs4all.analysis.playground_execution import execute_preview
from nirs4all.analysis.playground_prepare import PreviewStep
from nirs4all.analysis.playground_types import PreviewBatch, PreviewLimits


def preview_arrays(
    x: Any,
    *,
    y: Any = None,
    wavelengths: Any = None,
    sample_ids: Any = None,
    metadata: dict[str, Any] | None = None,
    partitions: Any = None,
    header_unit: str | None = None,
    steps: list[PreviewStep] | None = None,
    sampling: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    limits: PreviewLimits | None = None,
) -> dict[str, Any]:
    """Execute a stateless preview from already decoded aligned arrays."""
    batch = PreviewBatch.from_arrays(
        x,
        y=y,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        metadata=metadata,
        partitions=partitions,
        header_unit=header_unit,
        limits=limits,
    )
    return execute_preview(batch, steps, sampling=sampling, options=options, limits=limits)


def preview_spectro_dataset(
    dataset: Any,
    *,
    partition: str = "all",
    source_index: int = 0,
    target_index: int = 0,
    steps: list[PreviewStep] | None = None,
    sampling: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    limits: PreviewLimits | None = None,
) -> dict[str, Any]:
    """Execute a preview from one explicit assembled-dataset selection."""
    selection = extract_playground_dataset(
        dataset,
        partition=partition,
        source_index=source_index,
        target_index=target_index,
        limits=limits,
    )
    effective_options = dict(options or {})
    repetition = selection.evidence["repetition_column"]
    if repetition and "dataset_repetition" not in effective_options:
        effective_options["dataset_repetition"] = repetition
    result = execute_preview(
        selection.batch,
        steps,
        sampling=sampling,
        options=effective_options,
        limits=limits,
    )
    result["dataset_selection"] = selection.evidence
    result["diagnostics"] = list(selection.diagnostics)
    return result


__all__ = [
    "playground_metadata_columns",
    "preview_arrays",
    "preview_spectro_dataset",
]
