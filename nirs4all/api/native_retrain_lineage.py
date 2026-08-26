"""Signed provenance for the supported native Methods full-retrain lane.

The parent result is not a fitted-model handle: every field below is copied
from the parent DAG-ML outcome before the child fit starts.  The child outcome
then includes this object in its self-fingerprinted diagnostics, which makes
the relationship durable through Package V2 and Archive V2 without inventing
an archive retraining recipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DIAGNOSTIC_KEY = "nirs4all_native_retrain_lineage"
SEED_DIAGNOSTIC_KEY = "nirs4all_native_seed"
_FINGERPRINT_FIELDS = (
    "source_outcome_fingerprint",
    "source_training_request_fingerprint",
    "source_effective_plan_fingerprint",
    "source_selected_variant_fingerprint",
)


@dataclass(frozen=True)
class NativeRetrainLineage:
    """The immutable parent evidence for one native full retrain."""

    source_outcome_fingerprint: str
    source_training_request_fingerprint: str
    source_effective_plan_fingerprint: str
    source_selected_variant_id: str
    source_selected_variant_fingerprint: str
    source_seed: int

    @classmethod
    def from_source_outcome(cls, outcome: Mapping[str, Any]) -> NativeRetrainLineage:
        """Build lineage only from an attested, completed native outcome."""

        if not isinstance(outcome, Mapping):
            raise ValueError("native retrain source does not retain a structured native outcome")
        refit = outcome.get("refit")
        if not isinstance(refit, Mapping) or refit.get("status") != "completed":
            raise ValueError("native retrain source does not retain a completed native refit")
        diagnostics = outcome.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("native retrain source does not retain native training diagnostics")
        seed = diagnostics.get(SEED_DIAGNOSTIC_KEY)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("native retrain source does not retain an attested native seed")
        values = {
            "schema_version": 1,
            "operation": "full_refit",
            "source_outcome_fingerprint": outcome.get("outcome_fingerprint"),
            "source_training_request_fingerprint": outcome.get("training_request_fingerprint"),
            "source_effective_plan_fingerprint": outcome.get("effective_plan_fingerprint"),
            "source_selected_variant_id": outcome.get("selected_variant_id"),
            "source_selected_variant_fingerprint": outcome.get("selected_variant_fingerprint"),
            "source_seed": seed,
        }
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeRetrainLineage:
        """Strictly parse persisted lineage without accepting extension keys."""

        expected = {
            "schema_version",
            "operation",
            "source_outcome_fingerprint",
            "source_training_request_fingerprint",
            "source_effective_plan_fingerprint",
            "source_selected_variant_id",
            "source_selected_variant_fingerprint",
            "source_seed",
        }
        actual = set(value)
        if actual != expected:
            raise ValueError("native retrain lineage has an invalid field set")
        if value.get("schema_version") != 1 or value.get("operation") != "full_refit":
            raise ValueError("native retrain lineage has an unsupported schema or operation")
        normalized: dict[str, Any] = {}
        for field in _FINGERPRINT_FIELDS:
            fingerprint = value.get(field)
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
                raise ValueError(f"native retrain lineage has an invalid {field}")
            normalized[field] = fingerprint
        variant_id = value.get("source_selected_variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError("native retrain lineage has an invalid source_selected_variant_id")
        seed = value.get("source_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("native retrain lineage has an invalid source_seed")
        return cls(source_selected_variant_id=variant_id, source_seed=seed, **normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact versioned diagnostic representation."""

        return {
            "schema_version": 1,
            "operation": "full_refit",
            "source_outcome_fingerprint": self.source_outcome_fingerprint,
            "source_training_request_fingerprint": self.source_training_request_fingerprint,
            "source_effective_plan_fingerprint": self.source_effective_plan_fingerprint,
            "source_selected_variant_id": self.source_selected_variant_id,
            "source_selected_variant_fingerprint": self.source_selected_variant_fingerprint,
            "source_seed": self.source_seed,
        }


__all__ = ["DIAGNOSTIC_KEY", "SEED_DIAGNOSTIC_KEY", "NativeRetrainLineage"]
