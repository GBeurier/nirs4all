"""Fit-influence policy resolution for repeated and derived rows (N7)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np


class FitInfluenceError(ValueError):
    """Raised when a fit-influence policy cannot be resolved safely."""


class FitInfluenceMode(StrEnum):
    """Supported fit-influence strategies."""

    AUTO = "auto"
    UNIFORM_ROWS = "uniform_rows"
    EQUAL_SAMPLE_INFLUENCE = "equal_sample_influence"
    RESAMPLE_EQUALIZED = "resample_equalized"
    SCORER_ONLY = "scorer_only"
    BACKEND_LOSS_WEIGHT = "backend_loss_weight"
    STRICT_WEIGHT_SUPPORT = "strict_weight_support"


@dataclass(frozen=True)
class FitInfluencePolicy:
    """Replayable contract for how rows influence model fitting."""

    mode: str = FitInfluenceMode.AUTO.value
    strict_weight_support: bool = False
    allowed_fallbacks: tuple[str, ...] = field(
        default_factory=lambda: (
            FitInfluenceMode.EQUAL_SAMPLE_INFLUENCE.value,
            FitInfluenceMode.RESAMPLE_EQUALIZED.value,
            FitInfluenceMode.UNIFORM_ROWS.value,
        )
    )
    random_state: int | None = None
    version: int = 1

    def __post_init__(self) -> None:
        mode = FitInfluenceMode(self.mode)
        if mode is FitInfluenceMode.STRICT_WEIGHT_SUPPORT:
            object.__setattr__(self, "mode", FitInfluenceMode.AUTO.value)
            object.__setattr__(self, "strict_weight_support", True)
        else:
            object.__setattr__(self, "mode", mode.value)
        for fallback in self.allowed_fallbacks:
            FitInfluenceMode(fallback)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable policy manifest."""
        return {
            "version": self.version,
            "mode": self.mode,
            "strict_weight_support": self.strict_weight_support,
            "allowed_fallbacks": list(self.allowed_fallbacks),
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitInfluencePolicy:
        """Build a policy from a manifest mapping."""
        allowed_fallbacks = data.get("allowed_fallbacks")
        return cls(
            mode=str(data.get("mode", FitInfluenceMode.AUTO.value)),
            strict_weight_support=bool(data.get("strict_weight_support", False)),
            allowed_fallbacks=(
                tuple(str(item) for item in allowed_fallbacks)
                if allowed_fallbacks is not None
                else FitInfluencePolicy().allowed_fallbacks
            ),
            random_state=data.get("random_state"),
            version=int(data.get("version", 1)),
        )

    def fingerprint(self) -> str:
        """Stable SHA-256 of the policy contract."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FitInfluenceResolution:
    """Resolved fit-influence decision for one materialised training matrix."""

    policy: FitInfluencePolicy
    effective_mode: str
    sample_ids: list[str]
    sample_weight: np.ndarray | None = None
    scorer_weight: np.ndarray | None = None
    resample_indices: np.ndarray | None = None
    warnings: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        """Return audit metadata without large arrays."""
        return {
            "policy": self.policy.to_dict(),
            "effective_mode": self.effective_mode,
            "n_rows": len(self.sample_ids),
            "n_samples": len(set(self.sample_ids)),
            "has_sample_weight": self.sample_weight is not None,
            "has_scorer_weight": self.scorer_weight is not None,
            "has_resample_indices": self.resample_indices is not None,
            "warnings": list(self.warnings),
        }


def resolve_fit_influence(
    sample_ids: Sequence[Any],
    *,
    policy: FitInfluencePolicy | str | None = None,
    backend_supports_sample_weight: bool,
    row_is_derived: Sequence[bool] | None = None,
) -> FitInfluenceResolution:
    """Resolve effective fit influence for a materialised representation.

    The returned ``sample_weight`` is intended for model loss functions that
    support it. When a backend lacks weight support, ``auto`` falls back to a
    deterministic resampling plan if that fallback is allowed.
    """
    resolved_policy = _coerce_policy(policy)
    samples = [str(sample) for sample in sample_ids]
    if not samples:
        raise FitInfluenceError("Fit influence requires at least one row.")
    derived_flags = list(row_is_derived) if row_is_derived is not None else [False] * len(samples)
    if len(derived_flags) != len(samples):
        raise FitInfluenceError("row_is_derived must match sample_ids length.")

    counts = Counter(samples)
    constant_cardinality = len(set(counts.values())) == 1
    has_derived_rows = any(derived_flags)
    requested = FitInfluenceMode(resolved_policy.mode)
    if requested is FitInfluenceMode.AUTO:
        requested = _auto_mode(
            constant_cardinality=constant_cardinality,
            has_derived_rows=has_derived_rows,
            backend_supports_sample_weight=backend_supports_sample_weight,
            policy=resolved_policy,
        )

    warnings: list[str] = []
    if requested is FitInfluenceMode.UNIFORM_ROWS:
        if has_derived_rows:
            warnings.append("uniform_rows applied to derived rows; sample influence follows row multiplicity by policy.")
        return FitInfluenceResolution(
            policy=resolved_policy,
            effective_mode=requested.value,
            sample_ids=samples,
            warnings=tuple(warnings),
        )

    if requested in {FitInfluenceMode.EQUAL_SAMPLE_INFLUENCE, FitInfluenceMode.BACKEND_LOSS_WEIGHT}:
        if backend_supports_sample_weight:
            return FitInfluenceResolution(
                policy=resolved_policy,
                effective_mode=requested.value,
                sample_ids=samples,
                sample_weight=_equal_sample_weights(samples),
            )
        if resolved_policy.strict_weight_support or requested is FitInfluenceMode.BACKEND_LOSS_WEIGHT:
            raise FitInfluenceError("FitInfluencePolicy requires sample_weight support, but backend does not support it.")
        _require_fallback(resolved_policy, FitInfluenceMode.RESAMPLE_EQUALIZED)
        warnings.append("Backend lacks sample_weight support; falling back to resample_equalized.")
        return FitInfluenceResolution(
            policy=resolved_policy,
            effective_mode=FitInfluenceMode.RESAMPLE_EQUALIZED.value,
            sample_ids=samples,
            resample_indices=_equalized_resample_indices(samples),
            warnings=tuple(warnings),
        )

    if requested is FitInfluenceMode.RESAMPLE_EQUALIZED:
        return FitInfluenceResolution(
            policy=resolved_policy,
            effective_mode=requested.value,
            sample_ids=samples,
            resample_indices=_equalized_resample_indices(samples),
        )

    if requested is FitInfluenceMode.SCORER_ONLY:
        return FitInfluenceResolution(
            policy=resolved_policy,
            effective_mode=requested.value,
            sample_ids=samples,
            scorer_weight=_equal_sample_weights(samples),
        )

    raise FitInfluenceError(f"Unsupported fit influence mode {requested.value!r}.")


def _coerce_policy(policy: FitInfluencePolicy | str | None) -> FitInfluencePolicy:
    if policy is None:
        return FitInfluencePolicy()
    if isinstance(policy, FitInfluencePolicy):
        return policy
    return FitInfluencePolicy(mode=policy)


def _auto_mode(
    *,
    constant_cardinality: bool,
    has_derived_rows: bool,
    backend_supports_sample_weight: bool,
    policy: FitInfluencePolicy,
) -> FitInfluenceMode:
    if constant_cardinality:
        return FitInfluenceMode.UNIFORM_ROWS
    if backend_supports_sample_weight:
        _require_fallback(policy, FitInfluenceMode.EQUAL_SAMPLE_INFLUENCE)
        return FitInfluenceMode.EQUAL_SAMPLE_INFLUENCE
    if policy.strict_weight_support:
        raise FitInfluenceError("Variable cardinalities require sample_weight support under strict_weight_support.")
    _require_fallback(policy, FitInfluenceMode.RESAMPLE_EQUALIZED)
    return FitInfluenceMode.RESAMPLE_EQUALIZED


def _require_fallback(policy: FitInfluencePolicy, mode: FitInfluenceMode) -> None:
    if mode.value not in policy.allowed_fallbacks and policy.mode == FitInfluenceMode.AUTO.value:
        raise FitInfluenceError(f"FitInfluencePolicy auto cannot use fallback {mode.value!r}.")


def _equal_sample_weights(sample_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(sample_ids)
    return cast(np.ndarray, np.asarray([1.0 / counts[sample] for sample in sample_ids], dtype=float))


def _equalized_resample_indices(sample_ids: Sequence[str]) -> np.ndarray:
    by_sample: dict[str, list[int]] = {}
    for idx, sample in enumerate(sample_ids):
        by_sample.setdefault(sample, []).append(idx)
    target = max(len(indices) for indices in by_sample.values())
    out: list[int] = []
    for sample in sorted(by_sample):
        indices = by_sample[sample]
        for slot in range(target):
            out.append(indices[slot % len(indices)])
    return cast(np.ndarray, np.asarray(out, dtype=int))


__all__ = [
    "FitInfluenceError",
    "FitInfluenceMode",
    "FitInfluencePolicy",
    "FitInfluenceResolution",
    "resolve_fit_influence",
]
