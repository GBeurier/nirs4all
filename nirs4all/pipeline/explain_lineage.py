"""Derive explainability lineage from relation materialization/replay manifests.

Heterogeneous multi-source repetition runs materialise an aligned feature matrix
from per-source observations (see :mod:`nirs4all.data.raw_multisource`). When SHAP
explanations are computed on such a matrix, the explained columns are *derived*
features (per-source aggregates, cartesian combos, stacked repetitions) rather
than a single raw wavelength axis. This module turns a materialization or replay
manifest into the provenance metadata carried by
:class:`nirs4all.api.result.ExplainResult` so that the scientific meaning stays
visible: an aggregated MIR feature is never presented as if it were one raw
observation wavelength.

The single entry point is :func:`derive_relation_explain_lineage`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Feature-level explanation level per executable representation. The value
#: describes what a single explained *column* means, which is distinct from the
#: row unit level (e.g. ``per_source_aggregate`` rows are physical samples but
#: each column is an aggregate over one source's repetitions).
_EXPLANATION_LEVEL_BY_REPRESENTATION: dict[str, str] = {
    "per_source_aggregate": "source_aggregate",
    "sample_aggregate": "sample_aggregate",
    "per_source_observation": "raw_observation",
    "stack_fixed": "stack",
    "stack_padded_masked": "stack",
    "cartesian_full": "combo",
    "cartesian_mc": "combo",
    "cartesian_augmentation": "combo",
}

#: Explanation levels whose features are derived/aggregated and therefore must
#: not be presented as raw observation wavelengths.
_DERIVED_LEVELS: frozenset[str] = frozenset({"source_aggregate", "sample_aggregate", "combo", "stack"})

#: Source-prefix separators understood when parsing feature headers. The live
#: materializer emits ``MIR:1000``; manifests authored elsewhere may use the
#: ``MIR__1000`` convention. Longer separators are tried first.
_SOURCE_SEPARATORS: tuple[str, ...] = ("__", ":")


@dataclass(frozen=True)
class RelationExplainLineage:
    """Explainability provenance derived from a relation manifest.

    Attributes:
        explanation_level: Human-readable feature level (e.g. ``source_aggregate``,
            ``raw_observation``, ``combo``, ``stack``) or ``None`` when unknown.
        feature_lineage: Mapping from explained feature name to its provenance
            payload (representation, source id, fingerprint, lineage stages...).
        lineage_warning: Warning emitted when explained features are derived or
            aggregated rather than raw observation wavelengths, else ``None``.
        feature_names: Feature names resolved from the manifest headers when the
            caller did not supply names, else ``None``.
    """

    explanation_level: str | None = None
    feature_lineage: dict[str, dict[str, Any]] = field(default_factory=dict)
    lineage_warning: str | None = None
    feature_names: list[str] | None = None


def derive_relation_explain_lineage(
    manifest: Mapping[str, Any] | None,
    *,
    feature_names: Sequence[str] | None = None,
    n_features: int | None = None,
) -> RelationExplainLineage | None:
    """Derive explainability lineage from a relation manifest.

    Accepts either a direct materialization manifest (as produced by
    :meth:`nirs4all.data.raw_multisource.AlignedMaterialization.to_manifest`) or a
    relation replay manifest that wraps one under ``materialization_manifest``.

    Args:
        manifest: Materialization or replay manifest, or ``None``.
        feature_names: Feature names already chosen by SHAP. When ``None`` the
            manifest headers are used (subject to ``n_features`` matching).
        n_features: SHAP feature width. When provided, manifest headers are only
            adopted as feature names if their count matches this width.

    Returns:
        A :class:`RelationExplainLineage`, or ``None`` when ``manifest`` carries
        no materialization data to interpret.
    """
    materialization = _materialization_manifest(manifest)
    if materialization is None:
        return None

    representation = materialization.get("representation")
    plan = _representation_plan(materialization, manifest)
    unit_level = plan.get("unit_level") if plan is not None else None
    stage = plan.get("stage") if plan is not None else None
    lineage_stages = _lineage_stages(plan.get("lineage")) if plan is not None else None
    fingerprint = materialization.get("fingerprint")

    headers = _header_names(materialization.get("model_headers"))
    if headers is None or (n_features is not None and len(headers) != n_features):
        raw_headers = _header_names(materialization.get("headers"))
        if raw_headers is not None and (n_features is None or len(raw_headers) == n_features):
            headers = raw_headers

    level = _explanation_level(representation, stage, unit_level)
    warning = _lineage_warning(level, representation)

    effective_names = list(feature_names) if feature_names is not None else None
    if effective_names is not None and n_features is not None and len(effective_names) != n_features:
        effective_names = None
    if effective_names is None and feature_names is None and headers is not None and (n_features is None or len(headers) == n_features):
        effective_names = list(headers)

    known_sources = _known_sources(materialization)

    feature_lineage: dict[str, dict[str, Any]] = {}
    if effective_names is not None:
        for index, name in enumerate(effective_names):
            source_id, source_feature, feature_role = _split_feature_header(name, known_sources)
            feature_lineage[name] = {
                "feature_name": name,
                "feature_index": index,
                "feature_role": feature_role,
                "representation": representation,
                "explanation_level": level,
                "materialization_fingerprint": fingerprint,
                "source_id": source_id,
                "source_feature": source_feature,
                "unit_level": unit_level,
                "stage": stage,
                "lineage_stages": lineage_stages,
            }

    return RelationExplainLineage(
        explanation_level=level,
        feature_lineage=feature_lineage,
        lineage_warning=warning,
        feature_names=effective_names,
    )


def _materialization_manifest(manifest: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the embedded materialization manifest, unwrapping replay manifests."""
    if not isinstance(manifest, Mapping):
        return None
    inner = manifest.get("materialization_manifest")
    if isinstance(inner, Mapping):
        return inner
    if manifest.get("representation") is not None and any(key in manifest for key in ("headers", "shape", "fingerprint")):
        return manifest
    return None


def _header_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return None


def _representation_plan(
    materialization: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Locate the representation plan on the materialization or replay manifest."""
    plan = materialization.get("representation_plan")
    if isinstance(plan, Mapping):
        return plan
    if isinstance(manifest, Mapping):
        top_plan = manifest.get("representation_plan")
        if isinstance(top_plan, Mapping):
            return top_plan
    return None


def _explanation_level(representation: Any, stage: Any, unit_level: Any) -> str | None:
    """Map a representation to its human-readable feature explanation level."""
    if isinstance(representation, str) and representation in _EXPLANATION_LEVEL_BY_REPRESENTATION:
        return _EXPLANATION_LEVEL_BY_REPRESENTATION[representation]
    if stage:
        return str(stage)
    if unit_level:
        return str(unit_level)
    return None


def _lineage_stages(value: Any) -> list[Any] | None:
    """Return representation-plan lineage as a list without splitting strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _lineage_warning(level: str | None, representation: Any) -> str | None:
    """Build a warning when explained features are derived rather than raw."""
    if level not in _DERIVED_LEVELS:
        return None
    rep = str(representation) if representation else "relation"
    if level == "sample_aggregate":
        return (
            f"Explained features are sample-level aggregates over source repetitions (representation '{rep}', "
            "level 'sample_aggregate'); SHAP attributions describe aggregated sample signals, not individual "
            "raw observation wavelengths."
        )
    if level in {"source_aggregate", "sample_aggregate"}:
        return (
            f"Explained features are per-source aggregates over repetitions (representation '{rep}', "
            f"level '{level}'); SHAP attributions describe aggregated source signals, not individual "
            "raw observation wavelengths."
        )
    if level == "combo":
        return (
            f"Explained features come from cartesian combination rows (representation '{rep}'); SHAP "
            "attributions describe derived combo observations, not raw single-observation wavelengths."
        )
    return (
        f"Explained features are stacked repetition blocks (representation '{rep}'); SHAP attributions "
        "describe rep-indexed stacked columns, not a single raw wavelength axis."
    )


def _known_sources(materialization: Mapping[str, Any]) -> set[str]:
    """Collect the distinct source ids referenced by the manifest."""
    sources: set[str] = set()
    for source_id in materialization.get("source_ids") or ():
        if source_id:
            sources.add(str(source_id))
    for row in materialization.get("lineage") or ():
        if not isinstance(row, Mapping):
            continue
        source_id = row.get("source_id")
        if source_id:
            sources.add(str(source_id))
        observations = row.get("source_observations")
        if isinstance(observations, Mapping):
            sources.update(str(key) for key in observations)
    return sources


def _split_feature_header(name: str, known_sources: set[str]) -> tuple[str | None, str, str]:
    text = str(name)
    if text.startswith("mask:"):
        source_id, source_feature = _split_source(text[len("mask:"):], known_sources)
        return source_id, source_feature, "presence_mask"
    source_id, source_feature = _split_source(text, known_sources)
    return source_id, source_feature, "signal"


def _split_source(name: str, known_sources: set[str]) -> tuple[str | None, str]:
    """Split a source-prefixed feature header into ``(source_id, source_feature)``.

    Known source ids are matched first (longest-first to avoid prefix collisions);
    otherwise the header is split on the first recognised separator. Returns
    ``(None, name)`` when no source prefix can be inferred.
    """
    text = str(name)
    for source in sorted(known_sources, key=len, reverse=True):
        for separator in _SOURCE_SEPARATORS:
            prefix = f"{source}{separator}"
            if text.startswith(prefix):
                return source, text[len(prefix):]
    for separator in _SOURCE_SEPARATORS:
        if separator in text:
            source, _, rest = text.partition(separator)
            if source:
                return source, rest
    return None, text


__all__ = ["RelationExplainLineage", "derive_relation_explain_lineage"]
