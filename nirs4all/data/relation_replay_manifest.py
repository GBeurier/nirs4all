"""Replay manifest contracts for relational representations (N9 minimal)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nirs4all.data.fit_influence import FitInfluencePolicy
from nirs4all.data.raw_multisource import AlignedMaterialization, RawMultiSourceDataset, RepresentationPlan
from nirs4all.data.reduction import ReductionPlan
from nirs4all.operators.data.merge import StackingFitContract


class RelationReplayManifestError(ValueError):
    """Raised when a relational replay manifest is malformed."""


@dataclass(frozen=True)
class RelationReplayManifest:
    """Bundle/workspace-ready manifest for replaying relational representations."""

    staging_manifest: dict[str, Any] | None = None
    materialization_manifest: dict[str, Any] | None = None
    representation_plan: RepresentationPlan | None = None
    reduction_plans: tuple[ReductionPlan, ...] = ()
    fit_influence_policy: FitInfluencePolicy | None = None
    stacking_fit_contract: StackingFitContract | None = None
    extra_fingerprints: dict[str, str] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if self.representation_plan is None and self.materialization_manifest is not None:
            plan_payload = self.materialization_manifest.get("representation_plan")
            if isinstance(plan_payload, Mapping):
                object.__setattr__(self, "representation_plan", RepresentationPlan.from_dict(plan_payload))
        if self.materialization_manifest is not None and self.representation_plan is None:
            raise RelationReplayManifestError("materialization_manifest requires a replayable representation_plan.")
        if self.staging_manifest is None and self.materialization_manifest is None and self.representation_plan is None:
            raise RelationReplayManifestError("RelationReplayManifest requires staging, materialization, or representation_plan data.")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable manifest."""
        payload: dict[str, Any] = {
            "version": self.version,
            "staging_manifest": _json_safe(self.staging_manifest),
            "materialization_manifest": _json_safe(self.materialization_manifest),
            "representation_plan": self.representation_plan.to_dict() if self.representation_plan is not None else None,
            "reduction_plans": [plan.to_dict() for plan in self.reduction_plans],
            "fit_influence_policy": self.fit_influence_policy.to_dict() if self.fit_influence_policy is not None else None,
            "stacking_fit_contract": self.stacking_fit_contract.to_dict() if self.stacking_fit_contract is not None else None,
            "extra_fingerprints": dict(sorted(self.extra_fingerprints.items())),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RelationReplayManifest:
        """Reconstruct a manifest from :meth:`to_dict` output."""
        representation_plan = None
        if data.get("representation_plan") is not None:
            representation_plan = RepresentationPlan.from_dict(data["representation_plan"])
        fit_influence_policy = None
        if data.get("fit_influence_policy") is not None:
            fit_influence_policy = FitInfluencePolicy.from_dict(dict(data["fit_influence_policy"]))
        stacking_fit_contract = None
        if data.get("stacking_fit_contract") is not None:
            stacking_fit_contract = StackingFitContract.from_dict(dict(data["stacking_fit_contract"]))
        manifest = cls(
            staging_manifest=dict(data["staging_manifest"]) if data.get("staging_manifest") is not None else None,
            materialization_manifest=dict(data["materialization_manifest"]) if data.get("materialization_manifest") is not None else None,
            representation_plan=representation_plan,
            reduction_plans=tuple(ReductionPlan.from_dict(item) for item in data.get("reduction_plans", ())),
            fit_influence_policy=fit_influence_policy,
            stacking_fit_contract=stacking_fit_contract,
            extra_fingerprints={str(k): str(v) for k, v in dict(data.get("extra_fingerprints", {})).items()},
            version=int(data.get("version", 1)),
        )
        expected_fingerprint = data.get("fingerprint")
        if expected_fingerprint is not None and str(expected_fingerprint) != manifest.fingerprint():
            raise RelationReplayManifestError("RelationReplayManifest fingerprint does not match its content.")
        return manifest

    def fingerprint(self) -> str:
        """Stable SHA-256 over the replay contract."""
        payload = self.to_dict(include_fingerprint=False)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_relation_replay_manifest(
    *,
    staging: RawMultiSourceDataset | Mapping[str, Any] | None = None,
    materialization: AlignedMaterialization | Mapping[str, Any] | None = None,
    representation_plan: RepresentationPlan | Mapping[str, Any] | None = None,
    reduction_plans: Sequence[ReductionPlan | Mapping[str, Any]] = (),
    fit_influence_policy: FitInfluencePolicy | Mapping[str, Any] | None = None,
    stacking_fit_contract: StackingFitContract | Mapping[str, Any] | None = None,
    extra_fingerprints: Mapping[str, str] | None = None,
) -> RelationReplayManifest:
    """Build a replay manifest from live objects or existing manifests."""
    staging_manifest = staging.to_manifest() if isinstance(staging, RawMultiSourceDataset) else dict(staging) if staging is not None else None
    materialization_manifest = (
        materialization.to_manifest()
        if isinstance(materialization, AlignedMaterialization)
        else dict(materialization)
        if materialization is not None
        else None
    )
    resolved_plan = _coerce_representation_plan(representation_plan)
    resolved_reductions = tuple(
        item if isinstance(item, ReductionPlan) else ReductionPlan.from_dict(item)
        for item in reduction_plans
    )
    resolved_fit = _coerce_fit_influence_policy(fit_influence_policy)
    resolved_stacking = _coerce_stacking_fit_contract(stacking_fit_contract)
    return RelationReplayManifest(
        staging_manifest=staging_manifest,
        materialization_manifest=materialization_manifest,
        representation_plan=resolved_plan,
        reduction_plans=resolved_reductions,
        fit_influence_policy=resolved_fit,
        stacking_fit_contract=resolved_stacking,
        extra_fingerprints={str(k): str(v) for k, v in dict(extra_fingerprints or {}).items()},
    )


def _coerce_representation_plan(value: RepresentationPlan | Mapping[str, Any] | None) -> RepresentationPlan | None:
    if value is None:
        return None
    if isinstance(value, RepresentationPlan):
        return value
    return RepresentationPlan.from_dict(value)


def _coerce_fit_influence_policy(value: FitInfluencePolicy | Mapping[str, Any] | None) -> FitInfluencePolicy | None:
    if value is None:
        return None
    if isinstance(value, FitInfluencePolicy):
        return value
    return FitInfluencePolicy.from_dict(dict(value))


def _coerce_stacking_fit_contract(value: StackingFitContract | Mapping[str, Any] | None) -> StackingFitContract | None:
    if value is None:
        return None
    if isinstance(value, StackingFitContract):
        return value
    return StackingFitContract.from_dict(dict(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "RelationReplayManifest",
    "RelationReplayManifestError",
    "build_relation_replay_manifest",
]
