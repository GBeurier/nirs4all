"""Raw source-specific staging for heterogeneous multi-source repetitions (N3).

:class:`RawMultiSourceDataset` holds heterogeneous per-source spectra *before*
any rectangular materialisation. It is the in-memory staging object that lets
nirs4all carry e.g. ``MIR=2`` / ``RAMAN=3`` / ``NIRS=2`` measurements per
physical sample without fabricating a fake rectangular matrix or refactoring
:class:`~nirs4all.data.features.Features` into a ragged container (an explicit
non-goal of the first delivery, per the roadmap).

The object owns:

* ``X_by_source`` -- one rectangular block *per source* (rows are that source's
  raw observations, in their input order);
* ``headers_by_source`` -- the feature headers per source;
* a validated :class:`~nirs4all.data.relations.NormalizedObservationTable`
  resolving sample / source / observation identity once;
* ``targets_by_sample`` and ``metadata_by_level`` derived from the table;
* a deterministic source/row mapping (so the staging is replayable);
* a content :meth:`fingerprint`.

Crucially, there is **no implicit ragged -> rectangular coercion**. Converting
to an aligned matrix must be requested explicitly via
:meth:`RawMultiSourceDataset.materialize`, naming a replayable
:class:`RepresentationPlan`. N5a implements the non-cartesian representations;
N5b adds bounded cartesian representations with lineage, reduction and
fit-influence contracts.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from nirs4all.data.relations import (
    NormalizedObservationTable,
    Partition,
    RelationValidationError,
    RepetitionSpec,
    RepOrder,
    SourceObservations,
    UnitLevel,
    build_relation_table,
)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

#: Representations that :meth:`RawMultiSourceDataset.materialize` can execute now.
EXECUTABLE_REPRESENTATIONS: frozenset[str] = frozenset(
    {
        "per_source_aggregate",
        "per_source_observation",
        "sample_aggregate",
        "stack_fixed",
        "stack_padded_masked",
        "cartesian_full",
        "cartesian_mc",
        "cartesian_augmentation",
    }
)

#: Cartesian representations executable once their N6/N7/N9 contracts are present.
CARTESIAN_REPRESENTATIONS: frozenset[str] = frozenset(
    {
        "cartesian_full",
        "cartesian_mc",
        "cartesian_augmentation",
    }
)
#: Representations that are part of the design but not executable in this phase.
DECLARED_REPRESENTATIONS: frozenset[str] = frozenset()
ALL_REPRESENTATIONS: frozenset[str] = EXECUTABLE_REPRESENTATIONS | DECLARED_REPRESENTATIONS


class RepresentationUnitLevel(StrEnum):
    """Unit level produced by a materialised representation."""

    SAMPLE = "sample"
    SOURCE_OBSERVATION = "source_observation"
    SOURCE_AGGREGATE = "source_aggregate"
    STACK = "stack"
    COMBO = "combo"


class RepresentationStage(StrEnum):
    """Execution stage of a representation plan."""

    RAW_MULTISOURCE = "raw_multisource"
    SOURCE_AGGREGATE = "source_aggregate"
    SOURCE_OBSERVATION = "source_observation"
    SAMPLE_AGGREGATE = "sample_aggregate"
    STACK = "stack"
    COMBO = "combo"


class ComboSelection(StrEnum):
    """How cartesian combinations are selected."""

    NONE = "none"
    DETERMINISTIC_ALL = "deterministic_all"
    RANDOM_SEEDED = "random_seeded"
    STRATIFIED = "stratified"
    MATCH_BY = "match_by"
    ZIP = "zip"


@dataclass(frozen=True)
class CombinationPlan:
    """Replayable selection and cap contract for cartesian representations."""

    combo_selection: str = ComboSelection.DETERMINISTIC_ALL.value
    max_combos_per_sample: int | None = None
    max_total_combos: int | None = None
    max_total_rows: int | None = None
    memory_budget: int | str | None = None
    random_state: int | None = None
    train_only: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        ComboSelection(self.combo_selection)
        if self.combo_selection == ComboSelection.NONE.value:
            raise RelationValidationError("CombinationPlan.combo_selection cannot be 'none'.", code="REL-E019")
        _validate_non_negative_optional("max_combos_per_sample", self.max_combos_per_sample)
        _validate_non_negative_optional("max_total_combos", self.max_total_combos)
        _validate_non_negative_optional("max_total_rows", self.max_total_rows)
        _memory_budget_bytes(self.memory_budget)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable combination manifest."""
        return {
            "version": self.version,
            "combo_selection": self.combo_selection,
            "max_combos_per_sample": self.max_combos_per_sample,
            "max_total_combos": self.max_total_combos,
            "max_total_rows": self.max_total_rows,
            "memory_budget": self.memory_budget,
            "random_state": self.random_state,
            "train_only": self.train_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CombinationPlan:
        """Build a combination plan from a manifest mapping."""
        return cls(
            combo_selection=str(data.get("combo_selection", ComboSelection.DETERMINISTIC_ALL.value)),
            max_combos_per_sample=data.get("max_combos_per_sample"),
            max_total_combos=data.get("max_total_combos"),
            max_total_rows=data.get("max_total_rows"),
            memory_budget=data.get("memory_budget"),
            random_state=data.get("random_state"),
            train_only=bool(data.get("train_only", False)),
            version=int(data.get("version", 1)),
        )

    def fingerprint(self) -> str:
        """Stable SHA-256 of the combination contract."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


_PLAN_DEFAULTS: dict[str, tuple[RepresentationUnitLevel, RepresentationStage, tuple[str, ...]]] = {
    "per_source_aggregate": (
        RepresentationUnitLevel.SAMPLE,
        RepresentationStage.SOURCE_AGGREGATE,
        ("raw_observation", "source_aggregate", "sample"),
    ),
    "per_source_observation": (
        RepresentationUnitLevel.SOURCE_OBSERVATION,
        RepresentationStage.SOURCE_OBSERVATION,
        ("raw_observation", "source_observation"),
    ),
    "sample_aggregate": (
        RepresentationUnitLevel.SAMPLE,
        RepresentationStage.SAMPLE_AGGREGATE,
        ("raw_observation", "source_aggregate", "sample_aggregate"),
    ),
    "stack_fixed": (
        RepresentationUnitLevel.STACK,
        RepresentationStage.STACK,
        ("raw_observation", "fixed_stack", "sample"),
    ),
    "stack_padded_masked": (
        RepresentationUnitLevel.STACK,
        RepresentationStage.STACK,
        ("raw_observation", "padded_stack", "mask", "sample"),
    ),
    "cartesian_full": (
        RepresentationUnitLevel.COMBO,
        RepresentationStage.COMBO,
        ("raw_observation", "cartesian_combo"),
    ),
    "cartesian_mc": (
        RepresentationUnitLevel.COMBO,
        RepresentationStage.COMBO,
        ("raw_observation", "cartesian_combo", "random_seeded"),
    ),
    "cartesian_augmentation": (
        RepresentationUnitLevel.COMBO,
        RepresentationStage.COMBO,
        ("raw_observation", "cartesian_combo", "augmentation"),
    ),
}


@dataclass(frozen=True)
class RepresentationPlan:
    """Replayable plan for raw multi-source representation materialisation."""

    representation: str
    unit_level: str | None = None
    stage: str | None = None
    lineage: Sequence[str] | None = None
    combo_selection: str = "none"
    missing_source_policy: str = "error"
    missing_repetition_policy: str = "error"
    max_combos_per_sample: int | None = None
    max_total_combos: int | None = None
    max_total_rows: int | None = None
    memory_budget: int | str | None = None
    random_state: int | None = None
    combination_plan: CombinationPlan | None = None
    version: int = 1

    def __post_init__(self) -> None:
        name = str(self.representation)
        if name not in ALL_REPRESENTATIONS:
            raise RelationValidationError(
                f"Unknown representation {name!r}. Executable: {sorted(EXECUTABLE_REPRESENTATIONS)}; "
                f"declared (later phases): {sorted(DECLARED_REPRESENTATIONS)}.",
                code="REL-E019",
            )
        unit_level, stage, lineage = _PLAN_DEFAULTS[name]
        object.__setattr__(self, "representation", name)
        object.__setattr__(self, "unit_level", str(self.unit_level or unit_level.value))
        object.__setattr__(self, "stage", str(self.stage or stage.value))
        raw_lineage = self.lineage or lineage
        if isinstance(raw_lineage, str):
            raw_lineage = (raw_lineage,)
        object.__setattr__(self, "lineage", tuple(str(item) for item in raw_lineage))
        combo_selection = _default_combo_selection(name, str(self.combo_selection))
        object.__setattr__(self, "combo_selection", combo_selection)
        object.__setattr__(self, "missing_source_policy", str(self.missing_source_policy))
        object.__setattr__(self, "missing_repetition_policy", str(self.missing_repetition_policy))
        random_state = self.random_state
        if name == "cartesian_mc" and random_state is None:
            random_state = 0
        object.__setattr__(self, "random_state", random_state)
        _validate_non_negative_optional("max_total_rows", self.max_total_rows)
        _validate_non_negative_optional("max_combos_per_sample", self.max_combos_per_sample)
        _validate_non_negative_optional("max_total_combos", self.max_total_combos)
        _memory_budget_bytes(self.memory_budget)
        combination_plan = _normalize_combination_plan(self)
        object.__setattr__(self, "combination_plan", combination_plan)

    @property
    def is_cartesian(self) -> bool:
        """Whether the plan is a cartesian representation declared for N5b+."""
        return self.representation in CARTESIAN_REPRESENTATIONS

    @property
    def is_executable(self) -> bool:
        """Whether the plan can be materialised in the current implementation."""
        return self.representation in EXECUTABLE_REPRESENTATIONS

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable representation manifest."""
        payload: dict[str, Any] = {
            "version": self.version,
            "representation": self.representation,
            "unit_level": self.unit_level,
            "stage": self.stage,
            "lineage": list(self.lineage or ()),
            "combo_selection": self.combo_selection,
            "missing_source_policy": self.missing_source_policy,
            "missing_repetition_policy": self.missing_repetition_policy,
            "max_combos_per_sample": self.max_combos_per_sample,
            "max_total_combos": self.max_total_combos,
            "max_total_rows": self.max_total_rows,
            "memory_budget": self.memory_budget,
            "random_state": self.random_state,
        }
        if self.combination_plan is not None:
            payload["combination_plan"] = self.combination_plan.to_dict()
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentationPlan:
        """Build a plan from a representation manifest."""
        if "representation" not in data:
            raise RelationValidationError("RepresentationPlan manifest is missing 'representation'.", code="REL-E019")
        expected_fingerprint = data.get("fingerprint")
        plan = cls(
            representation=str(data["representation"]),
            unit_level=data.get("unit_level"),
            stage=data.get("stage"),
            lineage=data.get("lineage"),
            combo_selection=data.get("combo_selection", "none"),
            missing_source_policy=data.get("missing_source_policy", "error"),
            missing_repetition_policy=data.get("missing_repetition_policy", "error"),
            max_combos_per_sample=data.get("max_combos_per_sample"),
            max_total_combos=data.get("max_total_combos"),
            max_total_rows=data.get("max_total_rows"),
            memory_budget=data.get("memory_budget"),
            random_state=data.get("random_state"),
            combination_plan=CombinationPlan.from_dict(data["combination_plan"]) if data.get("combination_plan") is not None else None,
            version=int(data.get("version", 1)),
        )
        if expected_fingerprint is not None and str(expected_fingerprint) != plan.fingerprint():
            raise RelationValidationError("RepresentationPlan manifest fingerprint does not match its content.", code="REL-E019")
        return plan

    @classmethod
    def from_step_value(cls, value: str | Mapping[str, Any] | RepresentationPlan) -> RepresentationPlan:
        """Parse compact pipeline/operator syntax into a plan."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise RelationValidationError(
            f"Invalid representation plan type: {type(value).__name__}; expected string, mapping, or RepresentationPlan.",
            code="REL-E019",
        )

    def fingerprint(self) -> str:
        """Stable SHA-256 of the replay contract."""
        payload = self.to_dict(include_fingerprint=False)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AlignedMaterialization:
    """Result of an explicit rectangular materialisation.

    Attributes:
        representation: The representation name that produced ``X``.
        X: The rectangular feature matrix ``(n_samples, n_features)``.
        sample_ids: Physical sample ids, aligned to ``X`` rows (sorted order).
        headers: Feature headers, aligned to ``X`` columns (source-prefixed).
        targets: Sample-level targets aligned to ``sample_ids`` (``None`` where
            no target is declared).
    """

    representation: str
    X: np.ndarray
    sample_ids: list[str]
    headers: list[str]
    targets: list[Any]
    representation_plan: RepresentationPlan | None = None
    unit_ids: list[str] | None = None
    source_ids: list[str | None] | None = None
    feature_mask: np.ndarray | None = None
    lineage: list[dict[str, Any]] | None = None
    partitions: list[str] | None = None

    @cached_property
    def fingerprint(self) -> str:
        """Stable SHA-256 over the materialised arrays and replay plan."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.X, dtype=np.float64).tobytes())
        if self.feature_mask is not None:
            digest.update(np.ascontiguousarray(self.feature_mask, dtype=np.bool_).tobytes())
        payload = {
            "representation": self.representation,
            "plan": self.representation_plan.to_dict() if self.representation_plan is not None else None,
            "sample_ids": self.sample_ids,
            "unit_ids": self.unit_ids,
            "source_ids": self.source_ids,
            "partitions": self.partitions,
            "headers": self.headers,
            "targets": _json_safe(self.targets),
            "arrays": digest.hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable replay/export manifest without arrays."""
        model_matrix, model_headers = self.to_feature_matrix()
        return {
            "representation": self.representation,
            "representation_plan": self.representation_plan.to_dict() if self.representation_plan is not None else None,
            "fingerprint": self.fingerprint,
            "shape": list(self.X.shape),
            "model_shape": list(model_matrix.shape),
            "has_feature_mask": self.feature_mask is not None,
            "sample_ids": list(self.sample_ids),
            "unit_ids": list(self.unit_ids) if self.unit_ids is not None else None,
            "source_ids": list(self.source_ids) if self.source_ids is not None else None,
            "partitions": list(self.partitions) if self.partitions is not None else None,
            "headers": list(self.headers),
            "model_headers": list(model_headers),
            "targets": _json_safe(self.targets),
            "lineage": _json_safe(self.lineage),
        }

    def to_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return the numeric matrix and headers consumed by legacy models.

        Masked relational representations keep the original padded values in
        ``X`` plus a boolean ``feature_mask``. Legacy estimators cannot consume
        a side-channel mask, so the model matrix fills missing values with zero
        and appends deterministic mask columns.
        """
        if self.feature_mask is None:
            return np.asarray(self.X).copy(), list(self.headers)

        if self.feature_mask.shape != self.X.shape:
            raise RelationValidationError(
                f"feature_mask shape {self.feature_mask.shape!r} does not match X shape {self.X.shape!r}.",
                code="REL-E019",
            )

        values = np.nan_to_num(np.asarray(self.X, dtype=float), nan=0.0)
        mask_values = np.asarray(self.feature_mask, dtype=float)
        mask_headers = [f"mask:{header}" for header in self.headers]
        return np.concatenate([values, mask_values], axis=1), [*self.headers, *mask_headers]

    def to_spectro_dataset(self, name: str = "relation_materialized") -> SpectroDataset:
        """Convert this explicit materialisation to the legacy rectangular dataset API."""
        from nirs4all.data.dataset import SpectroDataset

        dataset = SpectroDataset(name=name)
        model_matrix, model_headers = self.to_feature_matrix()
        n_rows = int(self.X.shape[0])
        partitions = list(self.partitions) if self.partitions is not None else [Partition.TRAIN.value] * n_rows
        if len(partitions) != n_rows:
            raise RelationValidationError(
                f"Materialization declares {len(partitions)} partitions for {n_rows} rows.",
                code="REL-E021",
            )

        base_partition = partitions[0] if partitions else Partition.TRAIN.value
        dataset.add_samples(model_matrix, {"partition": base_partition}, headers=model_headers, header_unit="text")

        for partition in sorted(set(partitions)):
            if partition == base_partition:
                continue
            sample_indices = [idx for idx, value in enumerate(partitions) if value == partition]
            dataset._indexer.update_by_indices(sample_indices, {"partition": partition})

        if any(target is not None for target in self.targets):
            target_values = np.asarray([np.nan if target is None else target for target in self.targets])
            if target_values.dtype == object:
                with contextlib.suppress(TypeError, ValueError):
                    target_values = target_values.astype(float)
            dataset.add_targets(target_values)

        metadata_rows = self._spectro_metadata_rows(partitions)
        if metadata_rows:
            headers = list(metadata_rows[0])
            dataset.add_metadata(np.asarray([[row.get(header, "") for header in headers] for row in metadata_rows], dtype=object), headers=headers)
            dataset.set_repetition("physical_sample_id")
            dataset.set_aggregate("physical_sample_id")

        dataset._relation_materialization_manifest = self.to_manifest()
        if self.feature_mask is not None:
            dataset._relation_feature_mask = np.asarray(self.feature_mask, dtype=bool).copy()
        return dataset

    def _spectro_metadata_rows(self, partitions: Sequence[str]) -> list[dict[str, Any]]:
        unit_level = self.representation_plan.unit_level if self.representation_plan is not None else self.representation
        unit_ids = self.unit_ids if self.unit_ids is not None else self.sample_ids
        source_ids = self.source_ids if self.source_ids is not None else [None] * len(self.sample_ids)
        lineage = self.lineage if self.lineage is not None else [{} for _ in self.sample_ids]
        rows: list[dict[str, Any]] = []
        for idx, sample_id in enumerate(self.sample_ids):
            unit_id = unit_ids[idx] if idx < len(unit_ids) else sample_id
            source_id = source_ids[idx] if idx < len(source_ids) else None
            line = lineage[idx] if idx < len(lineage) else {}
            rows.append(
                {
                    "physical_sample_id": sample_id,
                    "origin_sample_id": line.get("origin_sample_id", sample_id) if isinstance(line, Mapping) else sample_id,
                    "unit_level": unit_level,
                    "unit_id": unit_id,
                    "derived_unit_id": line.get("derived_unit_id", unit_id) if isinstance(line, Mapping) else unit_id,
                    "source_id": source_id or "",
                    "relation_partition": partitions[idx],
                    "relation_row_id": f"{self.representation}:{idx}",
                    "representation": self.representation,
                    "representation_fingerprint": self.fingerprint,
                    "relation_lineage": json.dumps(_json_safe(line), sort_keys=True, separators=(",", ":")),
                }
            )
        return rows


class RawMultiSourceDataset:
    """Heterogeneous per-source staging dataset (roadmap N3).

    See the module docstring. Construct via :meth:`from_sources` (the normal
    path) or directly from an already-built relation table.
    """

    def __init__(
        self,
        relation_table: NormalizedObservationTable,
        X_by_source: Mapping[str, np.ndarray],
        *,
        headers_by_source: Mapping[str, Sequence[str]] | None = None,
        spec: RepetitionSpec | None = None,
    ) -> None:
        """Initialise and validate the staging dataset.

        Args:
            relation_table: A validated normalised observation table whose
                observation records carry ``source_row`` indices.
            X_by_source: Mapping ``source_id -> (n_obs_source, n_features)`` raw
                feature block, in the source's input row order.
            headers_by_source: Optional feature headers per source; generated
                from the array width when omitted.
            spec: The :class:`RepetitionSpec` that produced the table, if any.

        Raises:
            RelationValidationError: If the per-source arrays are inconsistent
                with the relation table.
        """
        self.relation_table = relation_table
        self.spec = spec
        self.X_by_source: dict[str, np.ndarray] = {
            source_id: np.asarray(array) for source_id, array in X_by_source.items()
        }
        self._validate_feature_blocks()
        self.headers_by_source: dict[str, list[str]] = self._resolve_headers(headers_by_source)
        self._validate_alignment()
        self.metadata_by_level: dict[str, dict[str, dict[str, Any]]] = self._build_metadata_by_level()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        spec: RepetitionSpec,
        X_by_source: Mapping[str, np.ndarray],
        keys_by_source: Mapping[str, Sequence[Any]],
        *,
        headers_by_source: Mapping[str, Sequence[str]] | None = None,
        rep_by_source: Mapping[str, Sequence[Any]] | None = None,
        targets_by_source: Mapping[str, Sequence[Any]] | None = None,
        metadata_by_source: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
        partition: Partition | str = Partition.TRAIN,
    ) -> RawMultiSourceDataset:
        """Build a staging dataset by joining per-source observations on a key.

        Args:
            spec: The source-aware repetition specification.
            X_by_source: Per-source raw feature blocks (input row order).
            keys_by_source: Per-source ``link_by`` key values (one per row).
            headers_by_source: Optional feature headers per source.
            rep_by_source: Optional explicit repetition indices per source.
            targets_by_source: Optional per-row targets per source (sample-level;
                may be declared on a single source and shared via the join).
            metadata_by_source: Optional ``source_id -> {col -> per-row values}``.
            partition: Partition label applied to every produced row.

        Returns:
            A validated :class:`RawMultiSourceDataset`.

        Raises:
            RelationValidationError: If the join or the array shapes are invalid.
        """
        if set(X_by_source) != set(keys_by_source):
            raise RelationValidationError(
                f"X_by_source declares sources {sorted(X_by_source)} but keys_by_source declares "
                f"{sorted(keys_by_source)}; every source needs both a feature block and keys.",
                code="REL-E021",
            )
        declared_sources = set(X_by_source)
        for name, mapping in (
            ("rep_by_source", rep_by_source),
            ("targets_by_source", targets_by_source),
            ("metadata_by_source", metadata_by_source),
        ):
            if mapping is None:
                continue
            extra = set(mapping) - declared_sources
            if extra:
                raise RelationValidationError(
                    f"{name} declares source(s) {sorted(extra)} that are absent from X_by_source. "
                    "Auxiliary mappings must use the same source ids as the feature blocks.",
                    code="REL-E021",
                )
        sources: list[SourceObservations] = []
        for source_id in sorted(X_by_source):
            array = np.asarray(X_by_source[source_id])
            keys = list(keys_by_source[source_id])
            n_rows = int(array.shape[0]) if array.ndim >= 1 else 0
            if len(keys) != n_rows:
                raise RelationValidationError(
                    f"Source {source_id!r} has {n_rows} feature rows but {len(keys)} link_by keys; "
                    "each observation row needs exactly one key.",
                    code="REL-E021",
                )
            sources.append(
                SourceObservations(
                    source_id=source_id,
                    sample_ids=keys,
                    rep_ids=list(rep_by_source[source_id]) if rep_by_source and source_id in rep_by_source else None,
                    targets=list(targets_by_source[source_id]) if targets_by_source and source_id in targets_by_source else None,
                    metadata={k: list(v) for k, v in metadata_by_source[source_id].items()}
                    if metadata_by_source and source_id in metadata_by_source
                    else None,
                )
            )
        table = build_relation_table(spec, sources, partition=partition)
        return cls(table, X_by_source, headers_by_source=headers_by_source, spec=spec)

    # -- basic accessors ---------------------------------------------------

    @property
    def name(self) -> str:
        """Dataset identifier used by legacy pipeline logging."""
        return "raw_multisource"

    @property
    def source_ids(self) -> list[str]:
        """Sorted unique source ids (deterministic)."""
        return cast(list[str], self.relation_table.source_ids)

    @property
    def physical_sample_ids(self) -> list[str]:
        """Sorted unique physical sample ids (deterministic)."""
        return cast(list[str], self.relation_table.physical_sample_ids)

    @property
    def n_samples(self) -> int:
        """Number of physical samples."""
        return len(self.physical_sample_ids)

    @property
    def n_observations(self) -> int:
        """Total number of raw observations across all sources."""
        return sum(1 for r in self.relation_table.records if r.unit_level is UnitLevel.OBSERVATION)

    @property
    def num_samples(self) -> int:
        """Legacy-compatible sample count for pipeline shape logging."""
        return self.n_samples

    @property
    def num_features(self) -> int:
        """Legacy-compatible rectangular width for the default materialized block."""
        return sum(self.feature_dims().values())

    @property
    def aggregate(self) -> str | None:
        """Legacy-compatible aggregation setting before explicit materialisation."""
        return None

    def features_sources(self) -> int:
        """Legacy-compatible number of feature sources."""
        return len(self.source_ids)

    def content_hash(self, source_index: int | None = None) -> str:
        """Legacy-compatible content hash for pipeline bookkeeping."""
        if source_index is not None:
            source_id = self.source_ids[source_index]
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(self.X_by_source[source_id], dtype=np.float64).tobytes())
            return digest.hexdigest()
        return self.fingerprint()

    def short_preprocessings_str(self) -> str:
        """Display label used by chain persistence before materialisation."""
        return "raw_multisource"

    def cardinalities(self) -> dict[tuple[str, str], int]:
        """Observation counts keyed by ``(physical_sample_id, source_id)``."""
        return cast(dict[tuple[str, str], int], self.relation_table.cardinalities())

    def targets_by_sample(self) -> dict[str, Any]:
        """Sample-level target value per physical sample."""
        return cast(dict[str, Any], self.relation_table.targets_by_sample())

    def feature_dims(self) -> dict[str, int]:
        """Feature width per source."""
        return {source_id: int(array.shape[1]) if array.ndim >= 2 else 0 for source_id, array in self.X_by_source.items()}

    def aligned_row_order(self) -> list[tuple[str, str, int]]:
        """Deterministic ``(physical_sample_id, source_id, source_row)`` order.

        The order is canonical -- sorted by sample, then source, then repetition
        -- so it does not depend on the input row order or source order. This is
        the replayable source/row mapping the staging guarantees.
        """
        observations = [r for r in self.relation_table.records if r.unit_level is UnitLevel.OBSERVATION]
        observations.sort(key=lambda r: (r.physical_sample_id, r.source_id, r.rep_id))
        return [(r.physical_sample_id, r.source_id, int(r.source_row if r.source_row is not None else -1)) for r in observations]

    # -- materialisation ---------------------------------------------------

    def materialize(self, representation: str | Mapping[str, Any] | RepresentationPlan | None = None) -> AlignedMaterialization:
        """Materialise an explicit rectangular representation.

        There is **no default**: a representation must be named, by design, so a
        heterogeneous staging dataset is never silently coerced into a (possibly
        meaningless) rectangular matrix. Executable representations include
        per-source aggregates/observations, fixed or padded stacks, and bounded
        cartesian combo materialisations.

        Args:
            representation: The representation to build. ``None`` is rejected.

        Returns:
            An :class:`AlignedMaterialization`.

        Raises:
            RelationValidationError: If no representation is named, if a declared
                representation is not yet executable, or if the name is unknown.
        """
        if representation is None:
            raise RelationValidationError(
                "RawMultiSourceDataset.materialize() requires an explicit representation; heterogeneous "
                "staging is never coerced to a rectangular matrix implicitly. "
                f"Executable now: {sorted(EXECUTABLE_REPRESENTATIONS)}.",
                code="REL-E019",
            )
        plan = RepresentationPlan.from_step_value(representation)
        name = plan.representation
        if not plan.is_executable:
            raise RelationValidationError(
                f"Representation {name!r} is declared in the design but not executable until N6/N7 and "
                "the extended N9 replay contract are implemented; "
                f"only {sorted(EXECUTABLE_REPRESENTATIONS)} are materialisable now.",
                code="REL-E019",
            )
        if name in {"per_source_aggregate", "sample_aggregate"}:
            return self._materialize_per_source_aggregate(plan)
        if name == "per_source_observation":
            return self._materialize_per_source_observation(plan)
        if name == "stack_fixed":
            return self._materialize_stack_fixed(plan)
        if name == "stack_padded_masked":
            return self._materialize_stack_padded_masked(plan)
        if name in CARTESIAN_REPRESENTATIONS:
            return self._materialize_cartesian(plan)
        raise RelationValidationError(f"Representation {name!r} is not routable.", code="REL-E019")

    def _materialize_per_source_aggregate(self, plan: RepresentationPlan) -> AlignedMaterialization:
        """Build the per-source mean aggregate (one row per physical sample).

        For each sample, each source's repetition rows are averaged into a single
        vector; the per-source vectors are concatenated in sorted source order.
        Heterogeneous repetition counts collapse to one vector per source, so the
        output is rectangular and aligned. Requires every source to be present
        for every sample (strict); missing-source policies are a later phase.
        """
        sources = self.source_ids
        samples = self.physical_sample_ids
        by_key = self._observation_groups()
        headers, _offsets = self._source_block_headers()
        self._enforce_materialization_caps(plan, n_rows=len(samples), n_features=len(headers))
        rows: list[np.ndarray] = []
        lineage: list[dict[str, Any]] = []
        for sample in samples:
            parts: list[np.ndarray] = []
            row_lineage: dict[str, Any] = {
                "unit_id": sample,
                "physical_sample_id": sample,
                "representation": plan.representation,
                "source_observations": {},
            }
            for source_id in sources:
                records = by_key.get((sample, source_id))
                if not records:
                    parts.append(self._missing_source_vector(plan, sample, source_id))
                    row_lineage["source_observations"][source_id] = []
                    continue
                array = self.X_by_source[source_id]
                vectors = np.stack([array[r.source_row] for r in records if r.source_row is not None], axis=0)
                parts.append(vectors.mean(axis=0))
                row_lineage["source_observations"][source_id] = [r.observation_id for r in records]
            rows.append(np.concatenate(parts))
            lineage.append(row_lineage)

        X = np.vstack(rows) if rows else np.empty((0, 0), dtype=float)
        targets_map = self.targets_by_sample()
        targets = [targets_map.get(sample) for sample in samples]
        return AlignedMaterialization(
            representation=plan.representation,
            X=X,
            sample_ids=list(samples),
            headers=headers,
            targets=targets,
            representation_plan=plan,
            unit_ids=list(samples),
            source_ids=[None for _ in samples],
            lineage=lineage,
            partitions=self._partitions_for_samples(samples),
        )

    def _materialize_per_source_observation(self, plan: RepresentationPlan) -> AlignedMaterialization:
        """Build one sparse/padded row per raw source observation."""
        observations = self._sorted_observation_records()
        headers, offsets = self._source_block_headers()
        self._enforce_materialization_caps(plan, n_rows=len(observations), n_features=len(headers), include_mask=True)
        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        sample_ids: list[str] = []
        unit_ids: list[str] = []
        source_ids: list[str | None] = []
        partitions: list[str] = []
        targets_map = self.targets_by_sample()
        targets: list[Any] = []
        lineage: list[dict[str, Any]] = []
        for record in observations:
            row = np.full(len(headers), np.nan, dtype=float)
            mask = np.zeros(len(headers), dtype=bool)
            start, end = offsets[record.source_id]
            row[start:end] = self.X_by_source[record.source_id][record.source_row]
            mask[start:end] = True
            unit_id = f"{record.physical_sample_id}|{record.source_id}|rep{record.rep_id}"
            rows.append(row)
            masks.append(mask)
            sample_ids.append(record.physical_sample_id)
            unit_ids.append(unit_id)
            source_ids.append(record.source_id)
            partitions.append(self._partition_value(record.partition))
            targets.append(targets_map.get(record.physical_sample_id))
            lineage.append(
                {
                    "unit_id": unit_id,
                    "physical_sample_id": record.physical_sample_id,
                    "source_id": record.source_id,
                    "rep_id": record.rep_id,
                    "observation_id": record.observation_id,
                    "source_row": record.source_row,
                }
            )
        X = np.vstack(rows) if rows else np.empty((0, len(headers)), dtype=float)
        feature_mask = np.vstack(masks) if masks else np.empty((0, len(headers)), dtype=bool)
        return AlignedMaterialization(
            representation=plan.representation,
            X=X,
            sample_ids=sample_ids,
            headers=headers,
            targets=targets,
            representation_plan=plan,
            unit_ids=unit_ids,
            source_ids=source_ids,
            feature_mask=feature_mask,
            lineage=lineage,
            partitions=partitions,
        )

    def _materialize_stack_fixed(self, plan: RepresentationPlan) -> AlignedMaterialization:
        """Build one row per sample by flattening fixed per-source repetitions."""
        by_key = self._observation_groups()
        samples = self.physical_sample_ids
        sources = self.source_ids
        counts_by_source: dict[str, int] = {}
        for source_id in sources:
            counts = [len(by_key.get((sample, source_id), [])) for sample in samples]
            unique = set(counts)
            if len(unique) != 1 or 0 in unique:
                raise RelationValidationError(
                    f"Cannot materialise 'stack_fixed': source {source_id!r} has incompatible per-sample "
                    f"cardinalities {counts}. Use 'stack_padded_masked' for variable or missing repetitions.",
                    code="REL-E019",
                )
            counts_by_source[source_id] = counts[0]
        headers = self._stack_headers(counts_by_source)
        self._enforce_materialization_caps(plan, n_rows=len(samples), n_features=len(headers))
        rows: list[np.ndarray] = []
        lineage: list[dict[str, Any]] = []
        for sample in samples:
            parts: list[np.ndarray] = []
            row_lineage: dict[str, Any] = {"unit_id": sample, "physical_sample_id": sample, "source_observations": {}}
            for source_id in sources:
                records = by_key[(sample, source_id)]
                array = self.X_by_source[source_id]
                parts.extend(array[record.source_row] for record in records if record.source_row is not None)
                row_lineage["source_observations"][source_id] = [record.observation_id for record in records]
            rows.append(np.concatenate(parts) if parts else np.empty((0,), dtype=float))
            lineage.append(row_lineage)
        X = np.vstack(rows) if rows else np.empty((0, len(headers)), dtype=float)
        targets_map = self.targets_by_sample()
        return AlignedMaterialization(
            representation=plan.representation,
            X=X,
            sample_ids=list(samples),
            headers=headers,
            targets=[targets_map.get(sample) for sample in samples],
            representation_plan=plan,
            unit_ids=list(samples),
            source_ids=[None for _ in samples],
            lineage=lineage,
            partitions=self._partitions_for_samples(samples),
        )

    def _materialize_stack_padded_masked(self, plan: RepresentationPlan) -> AlignedMaterialization:
        """Build one row per sample with repetition slots padded by NaN and a mask."""
        by_key = self._observation_groups()
        samples = self.physical_sample_ids
        sources = self.source_ids
        counts_by_source = {
            source_id: max((len(by_key.get((sample, source_id), [])) for sample in samples), default=0)
            for source_id in sources
        }
        headers = self._stack_headers(counts_by_source)
        self._enforce_materialization_caps(plan, n_rows=len(samples), n_features=len(headers), include_mask=True)
        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        lineage: list[dict[str, Any]] = []
        for sample in samples:
            row_parts: list[np.ndarray] = []
            mask_parts: list[np.ndarray] = []
            row_lineage: dict[str, Any] = {"unit_id": sample, "physical_sample_id": sample, "source_observations": {}}
            for source_id in sources:
                width = self.feature_dims()[source_id]
                records = by_key.get((sample, source_id), [])
                missing_source_vector = self._missing_source_vector(plan, sample, source_id) if not records else None
                source_slots: list[np.ndarray] = []
                source_masks: list[np.ndarray] = []
                for slot in range(counts_by_source[source_id]):
                    if slot < len(records):
                        record = records[slot]
                        source_slots.append(self.X_by_source[source_id][record.source_row])
                        source_masks.append(np.ones(width, dtype=bool))
                    elif missing_source_vector is not None:
                        source_slots.append(missing_source_vector.copy())
                        source_masks.append(np.zeros(width, dtype=bool))
                    else:
                        source_slots.append(np.full(width, np.nan, dtype=float))
                        source_masks.append(np.zeros(width, dtype=bool))
                if source_slots:
                    row_parts.extend(source_slots)
                    mask_parts.extend(source_masks)
                row_lineage["source_observations"][source_id] = [record.observation_id for record in records]
            rows.append(np.concatenate(row_parts) if row_parts else np.empty((0,), dtype=float))
            masks.append(np.concatenate(mask_parts) if mask_parts else np.empty((0,), dtype=bool))
            lineage.append(row_lineage)
        X = np.vstack(rows) if rows else np.empty((0, len(headers)), dtype=float)
        feature_mask = np.vstack(masks) if masks else np.empty((0, len(headers)), dtype=bool)
        targets_map = self.targets_by_sample()
        return AlignedMaterialization(
            representation=plan.representation,
            X=X,
            sample_ids=list(samples),
            headers=headers,
            targets=[targets_map.get(sample) for sample in samples],
            representation_plan=plan,
            unit_ids=list(samples),
            source_ids=[None for _ in samples],
            feature_mask=feature_mask,
            lineage=lineage,
            partitions=self._partitions_for_samples(samples),
        )

    def _materialize_cartesian(self, plan: RepresentationPlan) -> AlignedMaterialization:
        """Build one row per selected source-observation cartesian combo."""
        assert plan.combination_plan is not None
        sources = self.source_ids
        samples = self.physical_sample_ids
        headers, _offsets = self._source_block_headers()
        by_key = self._observation_groups()
        targets_map = self.targets_by_sample()
        selected_by_sample: dict[str, list[tuple[tuple[Any, np.ndarray, np.ndarray], ...]]] = {}
        n_rows = 0
        has_missing = False
        for sample in samples:
            options = self._cartesian_options_for_sample(plan, sample, sources, by_key)
            combos = self._select_cartesian_combos(plan, sample, options)
            selected_by_sample[sample] = combos
            n_rows += len(combos)
            has_missing = has_missing or any(component[0] is None for combo in combos for component in combo)
        self._enforce_combo_caps(plan, n_rows)
        self._enforce_materialization_caps(plan, n_rows=n_rows, n_features=len(headers), include_mask=has_missing)

        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        sample_ids: list[str] = []
        unit_ids: list[str] = []
        partitions: list[str] = []
        targets: list[Any] = []
        lineage: list[dict[str, Any]] = []
        for sample in samples:
            sample_partition = self._partition_for_sample(sample)
            for combo_index, combo in enumerate(selected_by_sample[sample]):
                vectors = [component[1] for component in combo]
                mask_parts = [component[2] for component in combo]
                row = np.concatenate(vectors) if vectors else np.empty((0,), dtype=float)
                mask = np.concatenate(mask_parts) if mask_parts else np.empty((0,), dtype=bool)
                unit_id = self._cartesian_unit_id(sample, combo)
                rows.append(row)
                masks.append(mask)
                sample_ids.append(sample)
                unit_ids.append(unit_id)
                partitions.append(sample_partition)
                targets.append(targets_map.get(sample))
                lineage.append(self._cartesian_lineage(plan, sample, combo_index, unit_id, combo))

        X = np.vstack(rows) if rows else np.empty((0, len(headers)), dtype=float)
        feature_mask = np.vstack(masks) if has_missing and masks else None
        return AlignedMaterialization(
            representation=plan.representation,
            X=X,
            sample_ids=sample_ids,
            headers=headers,
            targets=targets,
            representation_plan=plan,
            unit_ids=unit_ids,
            source_ids=[None for _ in sample_ids],
            feature_mask=feature_mask,
            lineage=lineage,
            partitions=partitions,
        )

    def _cartesian_options_for_sample(
        self,
        plan: RepresentationPlan,
        sample: str,
        sources: Sequence[str],
        by_key: Mapping[tuple[str, str], list[Any]],
    ) -> list[list[tuple[Any, np.ndarray, np.ndarray]]]:
        options_by_source: list[list[tuple[Any, np.ndarray, np.ndarray]]] = []
        for source_id in sources:
            records = by_key.get((sample, source_id), [])
            width = self.feature_dims()[source_id]
            if not records:
                vector = self._missing_source_vector(plan, sample, source_id)
                options_by_source.append([(None, vector, np.zeros(width, dtype=bool))])
                continue
            source_options: list[tuple[Any, np.ndarray, np.ndarray]] = []
            array = self.X_by_source[source_id]
            for record in records:
                if record.source_row is None:
                    continue
                source_options.append((record, array[record.source_row], np.ones(width, dtype=bool)))
            options_by_source.append(source_options)
        return options_by_source

    def _select_cartesian_combos(
        self,
        plan: RepresentationPlan,
        sample: str,
        options_by_source: Sequence[Sequence[tuple[Any, np.ndarray, np.ndarray]]],
    ) -> list[tuple[tuple[Any, np.ndarray, np.ndarray], ...]]:
        all_combos = [tuple(combo) for combo in itertools.product(*options_by_source)]
        if not all_combos:
            return []
        selection = ComboSelection(plan.combo_selection)
        max_per_sample = plan.combination_plan.max_combos_per_sample if plan.combination_plan is not None else None
        if selection is ComboSelection.DETERMINISTIC_ALL:
            if max_per_sample is not None and len(all_combos) > max_per_sample:
                raise RelationValidationError(
                    f"Representation {plan.representation!r} would produce {len(all_combos)} combos for sample "
                    f"{sample!r}, exceeding max_combos_per_sample={max_per_sample}.",
                    code="REL-E019",
                )
            return all_combos
        if selection is ComboSelection.RANDOM_SEEDED:
            limit = max_per_sample if max_per_sample is not None else 1
            n_select = min(limit, len(all_combos))
            rng = np.random.default_rng(_stable_sample_seed(plan.random_state, sample))
            selected_indices = sorted(rng.choice(len(all_combos), size=n_select, replace=False).tolist())
            return [all_combos[idx] for idx in selected_indices]
        raise RelationValidationError(
            f"combo_selection={selection.value!r} is declared but not executable for {plan.representation!r} yet.",
            code="REL-E019",
        )

    def _cartesian_unit_id(
        self,
        sample: str,
        combo: Sequence[tuple[Any, np.ndarray, np.ndarray]],
    ) -> str:
        tokens = []
        for record, _vector, _mask in combo:
            if record is None:
                tokens.append("missing")
            else:
                tokens.append(f"{record.source_id}{record.rep_id}")
        return f"{sample}::{'x'.join(tokens)}"

    def _cartesian_lineage(
        self,
        plan: RepresentationPlan,
        sample: str,
        combo_index: int,
        unit_id: str,
        combo: Sequence[tuple[Any, np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        records = [component[0] for component in combo if component[0] is not None]
        partitions = sorted({record.partition.value for record in records})
        return {
            "unit_id": unit_id,
            "unit_level": "combo",
            "derived_unit_id": unit_id,
            "physical_sample_id": sample,
            "origin_sample_id": sample,
            "representation": plan.representation,
            "combo_index": combo_index,
            "component_observation_ids": [record.observation_id for record in records],
            "rep_ids_by_source": {record.source_id: record.rep_id for record in records},
            "partition": partitions[0] if len(partitions) == 1 else partitions,
            "combination_plan": plan.combination_plan.to_dict() if plan.combination_plan is not None else None,
            "augmentation": "cartesian_augmentation" if plan.representation == "cartesian_augmentation" else None,
        }

    def _enforce_combo_caps(self, plan: RepresentationPlan, n_combos: int) -> None:
        combination_plan = plan.combination_plan
        if (
            combination_plan is not None
            and combination_plan.max_total_combos is not None
            and n_combos > combination_plan.max_total_combos
        ):
            raise RelationValidationError(
                f"Representation {plan.representation!r} would produce {n_combos} combos, exceeding "
                f"max_total_combos={combination_plan.max_total_combos}.",
                code="REL-E019",
            )

    def _sorted_observation_records(self) -> list[Any]:
        observations = [r for r in self.relation_table.records if r.unit_level is UnitLevel.OBSERVATION]
        observations.sort(key=lambda r: (r.physical_sample_id, r.source_id, r.rep_id, r.observation_id))
        return observations

    def _observation_groups(self) -> dict[tuple[str, str], list[Any]]:
        by_key: dict[tuple[str, str], list[Any]] = {}
        for record in self._sorted_observation_records():
            by_key.setdefault((record.physical_sample_id, record.source_id), []).append(record)
        return by_key

    def _partitions_for_samples(self, sample_ids: Sequence[str]) -> list[str]:
        """Return one unambiguous partition per physical sample."""
        return [self._partition_for_sample(sample_id) for sample_id in sample_ids]

    def _partition_for_sample(self, sample_id: str) -> str:
        partitions = {
            self._partition_value(record.partition)
            for record in self.relation_table.records
            if record.unit_level is UnitLevel.OBSERVATION and record.physical_sample_id == sample_id
        }
        if not partitions:
            return "train"
        if len(partitions) != 1:
            raise RelationValidationError(
                f"Physical sample {sample_id!r} spans multiple partitions {sorted(partitions)}; "
                "materialized rows require one partition per derived unit.",
                code="REL-E021",
            )
        return next(iter(partitions))

    @staticmethod
    def _partition_value(partition: Partition | str) -> str:
        return partition.value if isinstance(partition, Partition) else str(partition)

    def _source_block_headers(self) -> tuple[list[str], dict[str, tuple[int, int]]]:
        headers: list[str] = []
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for source_id in self.source_ids:
            source_headers = [f"{source_id}:{h}" for h in self.headers_by_source[source_id]]
            headers.extend(source_headers)
            offsets[source_id] = (cursor, cursor + len(source_headers))
            cursor += len(source_headers)
        return headers, offsets

    def _stack_headers(self, counts_by_source: Mapping[str, int]) -> list[str]:
        headers: list[str] = []
        for source_id in self.source_ids:
            for rep_idx in range(counts_by_source[source_id]):
                headers.extend(f"{source_id}:rep{rep_idx}:{h}" for h in self.headers_by_source[source_id])
        return headers

    def _missing_source_vector(self, plan: RepresentationPlan, sample: str, source_id: str) -> np.ndarray:
        policy = plan.missing_source_policy.lower()
        width = self.feature_dims()[source_id]
        if policy in {"impute_declared", "nan", "pad", "mask", "allow", "allow_missing"}:
            return cast(np.ndarray, np.full(width, np.nan, dtype=float))
        if policy == "zero":
            return cast(np.ndarray, np.zeros(width, dtype=float))
        if policy in {"drop_incomplete", "drop_branch", "partial_model"}:
            raise RelationValidationError(
                f"Cannot materialise {plan.representation!r}: missing_source_policy={plan.missing_source_policy!r} "
                "requires a model-selection or row-dropping adapter that is not part of this representation. "
                "Use 'impute_declared', 'mask', or 'pad' for explicit rectangular replay.",
                code="REL-E007",
            )
        spec_policy = self.spec.missing_source_policy.value if self.spec is not None else "none"
        raise RelationValidationError(
            f"Cannot materialise {plan.representation!r}: physical sample {sample!r} has no observation for "
            f"source {source_id!r}. Plan missing_source_policy={plan.missing_source_policy!r}; "
            f"relation spec missing_source_policy={spec_policy!r}.",
            code="REL-E007",
        )

    def _enforce_materialization_caps(
        self,
        plan: RepresentationPlan,
        *,
        n_rows: int,
        n_features: int,
        include_mask: bool = False,
    ) -> None:
        if plan.max_total_rows is not None and n_rows > plan.max_total_rows:
            raise RelationValidationError(
                f"Representation {plan.representation!r} would produce {n_rows} rows, exceeding "
                f"max_total_rows={plan.max_total_rows}.",
                code="REL-E019",
            )
        budget = _memory_budget_bytes(plan.memory_budget)
        if budget is None:
            return
        required = n_rows * n_features * np.dtype(np.float64).itemsize
        if include_mask:
            required += n_rows * n_features * np.dtype(np.bool_).itemsize
        if required > budget:
            raise RelationValidationError(
                f"Representation {plan.representation!r} would require about {required} bytes, exceeding "
                f"memory_budget={plan.memory_budget!r}.",
                code="REL-E019",
            )

    # -- fingerprint / manifest -------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the table, spec and per-source contents.

        The digest binds each feature vector to its *identity*
        ``(physical_sample_id, source_id)`` rather than its raw row position, so
        source order and input row order never change it. Repetition order is
        honoured per :attr:`RepetitionSpec.rep_order`: ``exchangeable`` reps are
        hashed as an order-independent multiset (shuffled reps -> same digest);
        ``ordered`` reps are hashed in ``rep_id`` order.
        """
        rep_order = self.spec.rep_order if self.spec is not None else RepOrder.EXCHANGEABLE
        groups: dict[tuple[str, str], list[tuple[int, bytes]]] = {}
        for record in self.relation_table.records:
            if record.unit_level is not UnitLevel.OBSERVATION or record.source_row is None:
                continue
            vector = np.ascontiguousarray(self.X_by_source[record.source_id][record.source_row], dtype=np.float64)
            groups.setdefault((record.physical_sample_id, record.source_id), []).append((record.rep_id, vector.tobytes()))

        feature_digest = hashlib.sha256()
        for sample, source_id in sorted(groups):
            feature_digest.update(f"{sample}|{source_id}|".encode())
            entries = groups[(sample, source_id)]
            if rep_order is RepOrder.ORDERED:
                ordered = [vec for _, vec in sorted(entries, key=lambda item: item[0])]
            else:
                ordered = sorted(vec for _, vec in entries)
            for vec in ordered:
                feature_digest.update(vec)
                feature_digest.update(b";")
            feature_digest.update(b"#")

        payload = {
            "relation": self.relation_table.fingerprint(),
            "spec": self.spec.to_dict() if self.spec is not None else {},
            "features": feature_digest.hexdigest(),
            "headers": {source_id: list(headers) for source_id, headers in sorted(self.headers_by_source.items())},
            "feature_dims": dict(sorted(self.feature_dims().items())),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        """Serialisable manifest (no feature arrays) for future workspace work.

        This carries enough to describe the staging -- spec, source ids, sample
        ids, cardinalities, headers, feature dims, targets and fingerprints --
        without the heavy arrays. It is intentionally JSON-serialisable; the
        array persistence (workspace / ``.n4a``) is a later phase (N9).
        """
        cardinalities = {f"{sample}|{source}": count for (sample, source), count in self.cardinalities().items()}
        return {
            "spec": self.spec.to_dict() if self.spec is not None else None,
            "relation_fingerprint": self.relation_table.fingerprint(),
            "fingerprint": self.fingerprint(),
            "source_ids": self.source_ids,
            "physical_sample_ids": self.physical_sample_ids,
            "cardinalities": cardinalities,
            "headers_by_source": {source_id: list(headers) for source_id, headers in self.headers_by_source.items()},
            "feature_dims": self.feature_dims(),
            "targets_by_sample": _json_safe(self.targets_by_sample()),
            "metadata_by_level": _json_safe(self.metadata_by_level),
        }

    # -- internals ---------------------------------------------------------

    def _validate_feature_blocks(self) -> None:
        """Ensure each source block is an explicit 2D feature matrix."""
        for source_id, array in self.X_by_source.items():
            if array.ndim != 2:
                raise RelationValidationError(
                    f"Source {source_id!r} feature block must be a 2D array shaped "
                    f"(n_observations, n_features), got shape {array.shape!r}.",
                    code="REL-E021",
                )

    def _resolve_headers(self, headers_by_source: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]]:
        resolved: dict[str, list[str]] = {}
        for source_id, array in self.X_by_source.items():
            width = int(array.shape[1]) if array.ndim >= 2 else 0
            if headers_by_source is not None and source_id in headers_by_source:
                headers = list(headers_by_source[source_id])
                if len(headers) != width:
                    raise RelationValidationError(
                        f"Source {source_id!r} has {width} feature columns but {len(headers)} headers.",
                        code="REL-E021",
                    )
                resolved[source_id] = headers
            else:
                resolved[source_id] = [f"{source_id}_f{i}" for i in range(width)]
        return resolved

    def _validate_alignment(self) -> None:
        """Ensure every observation maps to a valid row of its source's array."""
        table_sources = set(self.relation_table.source_ids)
        array_sources = set(self.X_by_source)
        missing = table_sources - array_sources
        if missing:
            raise RelationValidationError(
                f"Relation table references sources {sorted(missing)} with no feature block in X_by_source.",
                code="REL-E021",
            )
        extra = array_sources - table_sources
        if extra:
            raise RelationValidationError(
                f"X_by_source contains source blocks {sorted(extra)} that are not referenced by the relation table.",
                code="REL-E021",
            )
        used_rows: dict[str, list[int]] = {source_id: [] for source_id in table_sources}
        for record in self.relation_table.records:
            if record.unit_level is not UnitLevel.OBSERVATION:
                continue
            if record.source_row is None:
                raise RelationValidationError(
                    f"Observation {record.observation_id!r} carries no source_row index; build the relation "
                    "table via build_relation_table so staging can map rows back to features.",
                    code="REL-E021",
                )
            array = self.X_by_source[record.source_id]
            n_rows = int(array.shape[0]) if array.ndim >= 1 else 0
            if not (0 <= int(record.source_row) < n_rows):
                raise RelationValidationError(
                    f"Observation {record.observation_id!r} maps to source_row {record.source_row} but source "
                    f"{record.source_id!r} only has {n_rows} feature rows.",
                    code="REL-E021",
                )
            used_rows[record.source_id].append(int(record.source_row))
        for source_id in sorted(table_sources):
            n_rows = int(self.X_by_source[source_id].shape[0])
            used = used_rows[source_id]
            counts = {row: used.count(row) for row in set(used)}
            duplicate = sorted(row for row, count in counts.items() if count > 1)
            if duplicate:
                raise RelationValidationError(
                    f"Source {source_id!r} source_row index/indices {duplicate[:5]} are referenced by "
                    "multiple observations; staging requires a one-to-one source row mapping.",
                    code="REL-E021",
                )
            missing_rows = sorted(set(range(n_rows)) - set(used))
            if missing_rows:
                raise RelationValidationError(
                    f"Source {source_id!r} has unreferenced feature row(s) {missing_rows[:5]}; every "
                    "source row must be represented by exactly one observation.",
                    code="REL-E021",
                )

    def _build_metadata_by_level(self) -> dict[str, dict[str, dict[str, Any]]]:
        sample_meta: dict[str, dict[str, Any]] = {}
        for record in self.relation_table.records:
            if not record.metadata:
                continue
            target = sample_meta.setdefault(record.physical_sample_id, {})
            for key, value in record.metadata.items():
                target.setdefault(key, value)
        return {"sample": sample_meta}

    def __len__(self) -> int:
        return self.n_observations

    def __repr__(self) -> str:
        return (
            f"RawMultiSourceDataset(samples={self.n_samples}, sources={self.source_ids}, "
            f"observations={self.n_observations})"
        )


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable copy of simple Python / numpy containers."""
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


def _validate_non_negative_optional(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise RelationValidationError(f"{name} must be non-negative.", code="REL-E019")


def _default_combo_selection(representation: str, combo_selection: str) -> str:
    if combo_selection != ComboSelection.NONE.value:
        return combo_selection
    if representation == "cartesian_full":
        return ComboSelection.DETERMINISTIC_ALL.value
    if representation == "cartesian_mc":
        return ComboSelection.RANDOM_SEEDED.value
    if representation == "cartesian_augmentation":
        return ComboSelection.DETERMINISTIC_ALL.value
    return combo_selection


def _normalize_combination_plan(plan: RepresentationPlan) -> CombinationPlan | None:
    raw = plan.combination_plan
    if raw is None and not plan.is_cartesian:
        return None
    if isinstance(raw, CombinationPlan):
        return raw
    if isinstance(raw, Mapping):
        return CombinationPlan.from_dict(raw)
    if raw is not None:
        raise RelationValidationError(
            f"combination_plan must be a mapping or CombinationPlan, got {type(raw).__name__}.",
            code="REL-E019",
        )
    return CombinationPlan(
        combo_selection=plan.combo_selection,
        max_combos_per_sample=plan.max_combos_per_sample,
        max_total_combos=plan.max_total_combos,
        max_total_rows=plan.max_total_rows,
        memory_budget=plan.memory_budget,
        random_state=plan.random_state,
        train_only=plan.representation == "cartesian_augmentation",
    )


def _stable_sample_seed(random_state: int | None, sample: str) -> int:
    payload = f"{0 if random_state is None else random_state}|{sample}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def _memory_budget_bytes(value: int | str | None) -> int | None:
    """Parse a simple byte budget value used by representation plans."""
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise RelationValidationError("memory_budget must be non-negative.", code="REL-E019")
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)(?:\s*([kmgt]?b?))?", text)
    if match is None:
        raise RelationValidationError(
            f"memory_budget={value!r} is not parseable; use an integer byte count or values like '10MB'.",
            code="REL-E019",
        )
    amount = int(match.group(1))
    unit = match.group(2) or "b"
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    return amount * multipliers[unit]


def replay_materialization(
    dataset: RawMultiSourceDataset,
    manifest: Mapping[str, Any],
    *,
    validate_fingerprint: bool = False,
) -> AlignedMaterialization:
    """Replay a materialisation from a minimal N9 representation manifest.

    By default this reuses only the replayable :class:`RepresentationPlan`, so
    prediction data may have different row/cardinality counts. The feature-space
    contract (number and names of columns) is still validated because a bundle
    model cannot safely consume a different rectangular representation. Set
    ``validate_fingerprint=True`` when replaying the exact same dataset and
    expecting byte-identical materialisation output.
    """
    plan_payload = manifest.get("representation_plan", manifest)
    if plan_payload is None:
        raise RelationValidationError("Materialization manifest has no representation_plan.", code="REL-E019")
    if not isinstance(plan_payload, Mapping):
        raise RelationValidationError("Materialization representation_plan must be a mapping.", code="REL-E019")
    materialized = dataset.materialize(RepresentationPlan.from_dict(plan_payload))
    _validate_replayed_feature_space(materialized, manifest)
    if validate_fingerprint:
        expected_shape = manifest.get("shape")
        if expected_shape is not None and list(materialized.X.shape) != list(expected_shape):
            raise RelationValidationError(
                f"Replayed materialization shape {list(materialized.X.shape)} does not match manifest shape {expected_shape}.",
                code="REL-E019",
            )
        expected_fingerprint = manifest.get("fingerprint")
        if expected_fingerprint is not None and str(expected_fingerprint) != materialized.fingerprint:
            raise RelationValidationError("Replayed materialization fingerprint does not match manifest.", code="REL-E019")
    return materialized


def _validate_replayed_feature_space(materialized: AlignedMaterialization, manifest: Mapping[str, Any]) -> None:
    """Ensure prediction replay preserves the trained rectangular feature space."""
    expected_mask = manifest.get("has_feature_mask")
    if expected_mask is not None and bool(expected_mask) != (materialized.feature_mask is not None):
        raise RelationValidationError(
            "Replayed relation materialization produced a different mask contract than the bundle manifest. "
            "Missing-source and masked-representation policies must preserve whether mask features are present.",
            code="REL-E019",
        )

    expected_shape = manifest.get("shape")
    if isinstance(expected_shape, Sequence) and len(expected_shape) >= 2:
        expected_width = int(expected_shape[1])
        actual_width = int(materialized.X.shape[1]) if materialized.X.ndim >= 2 else 0
        if actual_width != expected_width:
            raise RelationValidationError(
                "Replayed relation materialization produced a different feature-space width "
                f"({actual_width}) than the bundle manifest ({expected_width}). This usually means a "
                "declared source is missing, a source feature width changed, or the missing-source policy "
                "is not replayable for this bundle.",
                code="REL-E019",
            )
    expected_headers = manifest.get("headers")
    if isinstance(expected_headers, Sequence) and not isinstance(expected_headers, (str, bytes)):
        expected = [str(header) for header in expected_headers]
        actual = [str(header) for header in materialized.headers]
        if actual != expected:
            raise RelationValidationError(
                "Replayed relation materialization produced different feature-space headers than the bundle manifest. "
                "Source order, source widths and missing-source replay policy must preserve the trained headers.",
                code="REL-E019",
            )

    expected_model_shape = manifest.get("model_shape")
    if isinstance(expected_model_shape, Sequence) and len(expected_model_shape) >= 2:
        model_matrix, model_headers = materialized.to_feature_matrix()
        expected_model_width = int(expected_model_shape[1])
        actual_model_width = int(model_matrix.shape[1]) if model_matrix.ndim >= 2 else 0
        if actual_model_width != expected_model_width:
            raise RelationValidationError(
                "Replayed relation materialization produced a different model feature-space width "
                f"({actual_model_width}) than the bundle manifest ({expected_model_width}).",
                code="REL-E019",
            )
        expected_model_headers = manifest.get("model_headers")
        if isinstance(expected_model_headers, Sequence) and not isinstance(expected_model_headers, (str, bytes)):
            expected = [str(header) for header in expected_model_headers]
            actual = [str(header) for header in model_headers]
            if actual != expected:
                raise RelationValidationError(
                    "Replayed relation materialization produced different model feature-space headers than the bundle manifest.",
                    code="REL-E019",
                )


__all__ = [
    "RawMultiSourceDataset",
    "AlignedMaterialization",
    "RepresentationPlan",
    "RepresentationUnitLevel",
    "RepresentationStage",
    "ComboSelection",
    "CombinationPlan",
    "CARTESIAN_REPRESENTATIONS",
    "EXECUTABLE_REPRESENTATIONS",
    "DECLARED_REPRESENTATIONS",
    "ALL_REPRESENTATIONS",
    "replay_materialization",
]
