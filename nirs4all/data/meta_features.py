"""Meta-feature alignment utilities for late fusion and stacking (N8)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nirs4all.operators.data.merge import MetaFeaturePlan, MissingPredictionPolicy


class MetaFeatureAlignmentError(ValueError):
    """Raised when meta-features cannot be aligned by explicit unit ids."""


@dataclass(frozen=True)
class PredictionFeatureVector:
    """One model/branch prediction vector with explicit alignment units."""

    name: str
    unit_ids: Sequence[Any]
    values: Sequence[Any] | np.ndarray


@dataclass(frozen=True)
class AlignedMetaFeatures:
    """Aligned matrix returned by :func:`align_prediction_feature_vectors`."""

    X: np.ndarray
    unit_ids: list[str]
    feature_names: list[str]
    mask: np.ndarray | None
    dropped_units: list[str]
    dropped_features: list[str]
    plan: MetaFeaturePlan

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-safe manifest without array values."""
        return {
            "shape": list(self.X.shape),
            "unit_ids": list(self.unit_ids),
            "feature_names": list(self.feature_names),
            "has_mask": self.mask is not None,
            "dropped_units": list(self.dropped_units),
            "dropped_features": list(self.dropped_features),
            "meta_feature_plan": self.plan.to_dict(),
        }


def align_prediction_feature_vectors(
    vectors: Sequence[PredictionFeatureVector],
    *,
    plan: MetaFeaturePlan | None = None,
    target_unit_ids: Sequence[Any] | None = None,
    impute_values: Mapping[str, float] | None = None,
) -> AlignedMetaFeatures:
    """Align branch/model prediction vectors by explicit unit id.

    The function never aligns by row position. In ``strict`` mode every feature
    vector must provide every target unit exactly once.
    """
    if not vectors:
        raise MetaFeatureAlignmentError("At least one prediction vector is required for meta-feature alignment.")
    plan = plan or MetaFeaturePlan()
    policy = MissingPredictionPolicy(plan.missing_prediction_policy)
    keyed = [_as_keyed_vector(vector) for vector in vectors]
    feature_names = [vector.name for vector in vectors]
    dropped_features: list[str] = []

    if target_unit_ids is None:
        unit_sets = [set(mapping) for mapping in keyed]
        if policy is MissingPredictionPolicy.DROP_INCOMPLETE:
            target_units = sorted(set.intersection(*unit_sets))
        else:
            target_units = sorted(set.union(*unit_sets))
    else:
        target_units = [str(unit) for unit in target_unit_ids]

    if not target_units:
        raise MetaFeatureAlignmentError("Meta-feature alignment produced no target units.")

    if policy is MissingPredictionPolicy.STRICT:
        _raise_on_missing(keyed, feature_names, target_units)

    active_keyed: list[dict[str, float]] = []
    active_names: list[str] = []
    if policy is MissingPredictionPolicy.DROP_BRANCH:
        for name, mapping in zip(feature_names, keyed, strict=True):
            if all(unit in mapping for unit in target_units):
                active_names.append(name)
                active_keyed.append(mapping)
            else:
                dropped_features.append(name)
        if not active_keyed:
            raise MetaFeatureAlignmentError("All prediction branches were dropped by missing_prediction_policy='drop_branch'.")
    else:
        active_names = feature_names
        active_keyed = keyed

    rows: list[list[float]] = []
    masks: list[list[bool]] = []
    kept_units: list[str] = []
    dropped_units: list[str] = []
    for unit in target_units:
        row: list[float] = []
        mask_row: list[bool] = []
        missing_names: list[str] = []
        for name, mapping in zip(active_names, active_keyed, strict=True):
            if unit in mapping:
                row.append(mapping[unit])
                mask_row.append(True)
            else:
                missing_names.append(name)
                row.append(_missing_value(policy, name, impute_values))
                mask_row.append(False)
        if missing_names and policy is MissingPredictionPolicy.DROP_INCOMPLETE:
            dropped_units.append(unit)
            continue
        kept_units.append(unit)
        rows.append(row)
        masks.append(mask_row)

    if not rows:
        raise MetaFeatureAlignmentError("Meta-feature alignment dropped every target unit.")

    X = np.asarray(rows, dtype=float)
    mask = np.asarray(masks, dtype=bool)
    if policy not in {MissingPredictionPolicy.MASK, MissingPredictionPolicy.PAD, MissingPredictionPolicy.PARTIAL_MODEL}:
        mask_out: np.ndarray | None = None
    else:
        mask_out = mask
    return AlignedMetaFeatures(
        X=X,
        unit_ids=kept_units,
        feature_names=active_names,
        mask=mask_out,
        dropped_units=dropped_units,
        dropped_features=dropped_features,
        plan=plan,
    )


def _as_keyed_vector(vector: PredictionFeatureVector) -> dict[str, float]:
    unit_ids = [str(unit) for unit in vector.unit_ids]
    values = np.asarray(vector.values, dtype=float).reshape(-1)
    if len(unit_ids) != len(values):
        raise MetaFeatureAlignmentError(
            f"Prediction vector {vector.name!r} has {len(values)} values but {len(unit_ids)} unit ids."
        )
    if len(set(unit_ids)) != len(unit_ids):
        raise MetaFeatureAlignmentError(f"Prediction vector {vector.name!r} contains duplicate unit ids.")
    return dict(zip(unit_ids, (float(value) for value in values), strict=True))


def _raise_on_missing(
    keyed: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
    target_units: Sequence[str],
) -> None:
    problems: list[str] = []
    for name, mapping in zip(feature_names, keyed, strict=True):
        missing = [unit for unit in target_units if unit not in mapping]
        if missing:
            problems.append(f"{name}: missing {missing[:5]}")
    if problems:
        raise MetaFeatureAlignmentError(
            "Strict meta-feature alignment requires every prediction branch to cover every unit; "
            + "; ".join(problems)
        )


def _missing_value(
    policy: MissingPredictionPolicy,
    feature_name: str,
    impute_values: Mapping[str, float] | None,
) -> float:
    if policy is MissingPredictionPolicy.IMPUTE_DECLARED:
        if impute_values is None or feature_name not in impute_values:
            raise MetaFeatureAlignmentError(
                f"Missing prediction for feature {feature_name!r} requires a declared impute value."
            )
        return float(impute_values[feature_name])
    if policy in {MissingPredictionPolicy.MASK, MissingPredictionPolicy.PAD, MissingPredictionPolicy.PARTIAL_MODEL}:
        return float("nan")
    if policy is MissingPredictionPolicy.DROP_BRANCH:
        return float("nan")
    if policy is MissingPredictionPolicy.DROP_INCOMPLETE:
        return float("nan")
    raise MetaFeatureAlignmentError(f"Unsupported missing prediction policy {policy.value!r}.")


__all__ = [
    "AlignedMetaFeatures",
    "MetaFeatureAlignmentError",
    "PredictionFeatureVector",
    "align_prediction_feature_vectors",
]
