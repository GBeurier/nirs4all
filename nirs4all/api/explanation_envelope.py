"""The ``nirs4all.explanation.v1`` envelope shared by every SDK explain backend.

The schema is owned by nirs4all-methods (``docs/explain/explanation_v1.schema.json``).
Hosts fill it from their attribution result and recorded metadata. Axis values
and units come only from recorded dataset headers; unknown or derived feature
spaces are labelled ``feature_index`` with a null unit. ``rows.sample_indices[i]``
is the index, within the supplied data, of explained row ``i``; explained-row
positions and source indices are never interchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

EXPLANATION_SCHEMA = "nirs4all.explanation.v1"
_AXIS_KINDS = {"nm": "wavelength", "cm-1": "wavenumber"}


def feature_index_axis(n_features: int) -> dict[str, Any]:
    """Return the axis of features without a recorded physical coordinate."""
    return {"values": [float(index) for index in range(n_features)], "unit": None, "kind": "feature_index"}


def recorded_axis(dataset: Any, n_features: int) -> dict[str, Any]:
    """Return the recorded spectral axis of a single-source dataset.

    Args:
        dataset: Materialized dataset whose rows are explained.
        n_features: Number of explained raw input columns.

    Returns:
        A wavelength (``nm``) or wavenumber (``cm-1``) axis when the dataset has
        one source with numeric headers in that explicitly declared unit, otherwise a
        ``feature_index`` axis with a null unit.
    """
    # header_unit() falls back to cm-1 when no unit was declared; never report that default.
    if getattr(dataset, "n_sources", None) != 1 or dataset.headers(0) is None or not dataset.header_unit_recorded(0):
        return feature_index_axis(n_features)
    unit = dataset.header_unit(0)
    if unit not in _AXIS_KINDS:
        return feature_index_axis(n_features)
    try:
        values = np.asarray(dataset.float_headers(0), dtype=np.float64)
    except ValueError:
        # Non-numeric headers carry names, not coordinates.
        return feature_index_axis(n_features)
    if values.shape != (n_features,) or not np.isfinite(values).all():
        return feature_index_axis(n_features)
    return {"values": values.tolist(), "unit": unit, "kind": _AXIS_KINDS[unit]}


def describe_groups(names: Sequence[str], axis: dict[str, Any], feature_group: np.ndarray | None, n_groups: int, sources: Sequence[str]) -> list[dict[str, Any]]:
    """Describe contiguous feature groups for the envelope.

    Args:
        names: Explained feature names.
        axis: Envelope axis of the explained features.
        feature_group: Group of each feature, or ``None`` for one group per feature.
        n_groups: Number of groups.
        sources: Group source per group (``per_feature``, ``fixed_width``, ``ranges`` or ``remainder``).

    Returns:
        One record per group with half-open ``[feature_start, feature_end)``
        bounds and inclusive axis coordinates.
    """
    n_features = len(names)
    if feature_group is None:
        starts = np.arange(n_features)
        ends = starts + 1
    else:
        positions = np.arange(n_features)
        starts = np.full(n_groups, n_features)
        ends = np.zeros(n_groups, dtype=np.int64)
        np.minimum.at(starts, feature_group, positions)
        np.maximum.at(ends, feature_group, positions + 1)
    values = axis["values"]
    groups = []
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        label = names[start] if end - start == 1 else f"{names[start]}–{names[end - 1]}"
        groups.append({"label": label, "feature_start": start, "feature_end": end, "axis_start": values[start], "axis_end": values[end - 1], "source": sources[index]})
    return groups


def _nested(values: np.ndarray) -> list[Any]:
    nested: list[Any] = np.asarray(values, dtype=np.float64).tolist()
    return nested


def build_explanation_v1(
    *,
    kind: str,
    engine: dict[str, str],
    exactness: str,
    additive: bool,
    group_aggregation: str,
    attribution_space: str,
    output: dict[str, Any],
    axis: dict[str, Any],
    feature_names: Sequence[str],
    groups: list[dict[str, Any]],
    sample_indices: Sequence[int],
    attributions: np.ndarray,
    std_error: np.ndarray | None,
    expected: np.ndarray,
    prediction: np.ndarray,
    additivity_max_abs_error: float | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a JSON-native ``nirs4all.explanation.v1`` envelope.

    Args:
        kind: Attribution kind, such as ``permutation_shap`` or ``tree_shap``.
        engine: Producing engine name, version and optional ABI.
        exactness: ``exact``, ``sampled``, ``surrogate`` or ``perturbation``.
        additive: Whether attributions plus expected values equal predictions.
        group_aggregation: ``group_game``, ``sum_of_feature_values`` or ``none``.
        attribution_space: ``raw_input`` or ``transformed_features``.
        output: Explained output ``{"index", "label", "space"}``; ``index``
            addresses the last axis of ``attributions``.
        axis: Axis of the explained features.
        feature_names: Explained feature names.
        groups: Group records from :func:`describe_groups`.
        sample_indices: Source index (within the supplied data) of each explained row.
        attributions: ``(rows, groups, outputs)`` attributions.
        std_error: Optional Monte-Carlo standard errors with the same shape;
            undefined entries are stored as null.
        expected: ``(rows, outputs)`` expected values.
        prediction: ``(rows, outputs)`` explained predictions.
        additivity_max_abs_error: Maximum additivity error, ``None`` when not additive.
        settings: Scalar explainer settings.

    Returns:
        The envelope dictionary.
    """
    selected = np.asarray(attributions, dtype=np.float64)[:, :, output["index"]]
    errors = None
    if std_error is not None:
        errors = [[[float(value) if np.isfinite(value) else None for value in group] for group in row] for row in np.asarray(std_error, dtype=np.float64)]
    return {
        "schema": EXPLANATION_SCHEMA,
        "kind": kind,
        "engine": dict(engine),
        "exactness": exactness,
        "additive": additive,
        "group_aggregation": group_aggregation,
        "attribution_space": attribution_space,
        "output": dict(output),
        "axis": dict(axis),
        "feature_names": list(feature_names),
        "groups": groups,
        "rows": {"count": len(sample_indices), "sample_indices": [int(index) for index in sample_indices], "source": "dataset_rows"},
        "attributions": _nested(attributions),
        "std_error": errors,
        "expected_value": _nested(expected),
        "prediction": _nested(prediction),
        "additivity_max_abs_error": None if additivity_max_abs_error is None else float(additivity_max_abs_error),
        "settings": dict(settings),
        "summary": {"output_index": output["index"], "mean_abs": np.abs(selected).mean(axis=0).tolist(), "mean_signed": selected.mean(axis=0).tolist()},
    }
