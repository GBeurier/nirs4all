"""Representation-fusion operator configuration for raw multi-source staging."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nirs4all.data.raw_multisource import AlignedMaterialization, RawMultiSourceDataset, RepresentationPlan


@dataclass(frozen=True)
class RepFusionConfig:
    """Configuration for the ``rep_fusion`` relational materialisation step."""

    plan: RepresentationPlan

    @property
    def representation(self) -> str:
        """Representation name selected by the plan."""
        return self.plan.representation

    def materialize(self, dataset: RawMultiSourceDataset) -> AlignedMaterialization:
        """Materialise this configuration on a raw multi-source staging dataset."""
        return dataset.materialize(self.plan)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable config manifest."""
        return {"representation_plan": self.plan.to_dict()}

    @classmethod
    def from_step_value(cls, value: str | Mapping[str, Any] | RepresentationPlan | RepFusionConfig) -> RepFusionConfig:
        """Parse pipeline syntax into a ``RepFusionConfig``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, RepresentationPlan):
            return cls(value)
        if isinstance(value, str):
            return cls(RepresentationPlan(value))
        if isinstance(value, Mapping):
            plan_payload = value.get("representation_plan", value.get("plan", value))
            if not isinstance(plan_payload, (str, Mapping, RepresentationPlan)):
                raise ValueError("rep_fusion representation_plan must be a string, mapping, or RepresentationPlan.")
            return cls(RepresentationPlan.from_step_value(plan_payload))
        raise ValueError(
            f"Invalid rep_fusion config type: {type(value).__name__}. "
            "Expected a representation string, mapping, RepresentationPlan, or RepFusionConfig."
        )


__all__ = ["RepFusionConfig"]
