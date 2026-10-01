"""Identity-bound splitter declarations for native PLS studies within folds.

Only the split structure is materialized here. DAG-ML runs every optimizer,
fit, metric and selection; Methods owns the fitted numerical models.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from .envelope import build_fold_set
from .fit_identity import DagMLFitIdentityFrame
from .raw_training_lowerer import identity_from_fit_frame
from .steps import FrozenDagMlSplitStep, _split_pipeline

if TYPE_CHECKING:
    from .native_pls_phase_controls import NativePlsPhaseControls


@dataclass(frozen=True)
class NativePlsFoldPlan:
    """Declared outer/inner folds, before the native runtime attests them."""

    pipeline: list[Any]
    outer_fold_set: dict[str, Any]
    inner_fold_sets: dict[str, Any]
    refit_inner_fold_set: dict[str, Any]


def normalize_fold_resume_package(value: Any) -> str | None:
    """Accept only a complete package carrying the separate fold ledger."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("native fold HPO resume_package must be a complete Package V2 mapping")
    bundle = value.get("execution_bundle")
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(bundle.get("methods_hpo_fold_state"), Mapping)
        or bundle.get("methods_hpo_resume_state") is not None
    ):
        raise ValueError("native fold HPO resume_package requires the complete Package V2 fold study ledger")
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError("native fold HPO resume_package must contain finite JSON values") from error


def validate_fold_splitter(splitter: Any) -> None:
    """Keep the local native profile within explicit partitioning splitters."""

    if type(splitter) not in {KFold, GroupKFold}:
        raise ValueError("native fold HPO requires an exact sklearn KFold or GroupKFold declaration")
    if getattr(splitter, "shuffle", False) and getattr(splitter, "random_state", None) is None:
        raise ValueError("native fold HPO requires an explicit splitter random_state when shuffle=True")


def prepare_native_pls_fold_plan(
    profile: NativePlsPhaseControls, identity: DagMLFitIdentityFrame, *, n_features: int, split_groups: Any,
) -> NativePlsFoldPlan:
    """Run the declared splitter on each scoped ID pool, never on outer targets."""

    steps, splitter = _split_pipeline(profile.pipeline)
    validate_fold_splitter(splitter)
    group_values = identity.groups
    has_groups = any(group is not None for group in group_values)
    if type(splitter) is GroupKFold:
        if split_groups is None or not has_groups or any(group is None for group in group_values):
            raise ValueError("native fold HPO GroupKFold requires complete dataset groups")
    elif has_groups:
        raise ValueError("native fold HPO with declared groups requires GroupKFold")
    wire_identity = identity_from_fit_frame(identity)
    # GroupKFold's ordering depends on the declared label dtype. Keep numeric
    # labels numeric for splitting; only the wire identity normalizes to str.
    declared_groups = np.asarray(split_groups) if type(splitter) is GroupKFold else None

    def split(pool: list[int]) -> list[tuple[list[int], list[int]]]:
        local_splitter = copy.deepcopy(splitter)
        kwargs = {"groups": declared_groups[pool]} if declared_groups is not None else {}
        return [
            ([pool[int(row)] for row in train], [pool[int(row)] for row in validation])
            for train, validation in local_splitter.split(np.arange(len(pool)), **kwargs)
        ]

    def fold_set(folds: list[tuple[list[int], list[int]]], *, name: str, prefix: str | None = None) -> dict[str, Any]:
        value = build_fold_set(wire_identity, folds, set_id=name)
        if has_groups:
            value["sample_groups"] = {sample: group_values[wire_identity.to_int(sample)] for sample in value["sample_ids"]}
        if prefix is not None:
            for fold in value["folds"]:
                fold["fold_id"] = f"{prefix}.inner.{fold['fold_id']}"
        return value

    pool = list(range(identity.n_samples))
    outer_folds = split(pool)
    outer = fold_set(outer_folds, name="folds.outer")
    inner = {}
    scoped_training_sizes: list[int] = []
    for outer_fold, (train, _) in zip(outer["folds"], outer_folds, strict=True):
        identifier = outer_fold["fold_id"]
        inner_folds = split(train)
        scoped_training_sizes.extend(len(rows) for rows, _ in inner_folds)
        inner[identifier] = {
            "parent_outer_fold_id": identifier,
            "inner_fold_set": fold_set(inner_folds, name=f"folds.{identifier}.inner", prefix=identifier),
        }
    refit_folds = split(pool)
    scoped_training_sizes.extend(len(rows) for rows, _ in refit_folds)
    refit = fold_set(refit_folds, name="folds.refit.inner", prefix="refit")
    required_components = (
        3 if any(axis["name"] == "n_components" for axis in profile.search_axes)
        else profile.train_params.get("n_components", profile.base_params["n_components"])
    )
    if min(scoped_training_sizes) <= required_components or n_features < required_components:
        raise ValueError("native fold HPO inner training pools cannot support the declared PLS component count")
    frozen = FrozenDagMlSplitStep(
        splitter=copy.deepcopy(splitter), sample_pool=tuple(pool),
        folds=tuple((tuple(train), tuple(validation)) for train, validation in outer_folds),
    )
    return NativePlsFoldPlan([frozen, *steps], outer, inner, refit)
