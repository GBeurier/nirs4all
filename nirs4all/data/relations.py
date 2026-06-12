"""Relational identity model for heterogeneous multi-source repetitions.

This module is the *internal foundation* (roadmap phases N1/N2) for supporting
spectral repetitions that differ per source, e.g. ``MIR=2``, ``RAMAN=3``,
``NIRS=2`` measurements per physical sample. It does **not** materialise
aligned :class:`~nirs4all.data.features.Features`; that is deferred to phase N3.
The goal here is to resolve sample/source/observation identities *once* and to
validate them, so that split, reshape, scoring and export do not each recompute
their own notion of "sample".

The canonical identity columns mirror the design document:

* ``physical_sample_id`` -- the physical sample, the statistical unit for
  splitting, evaluation and selection. ``sample_id`` is only a user-facing YAML
  alias that maps onto this stable key.
* ``internal_sample_id`` -- a deterministic integer assigned per physical sample.
* ``source_id`` -- the source / modality / instrument (``MIR``, ``RAMAN`` ...).
* ``observation_id`` -- one real spectral measurement of a source for a sample.
* ``rep_id`` -- the repetition index within a given source.
* ``derived_unit_id`` -- a derived unit (e.g. one cartesian combo). ``None`` for
  raw observations.
* ``origin_sample_id`` -- provenance for derived / augmented rows; equals
  ``physical_sample_id`` for base rows.
* ``unit_level`` / ``unit_id`` -- the typed level of a row and its key.
* ``row_id`` -- a global row counter.
* ``partition`` -- ``train`` / ``test`` / ``predict``.
* ``target_id`` -- the sample-level target value (constant per physical sample).
* ``sample_influence_weight`` -- an auditable fit-influence column (default
  ``1.0``). The *effective* weight is derived later (phase N7); N1 only carries a
  computable column.
* ``quality_flag`` -- per-row quality marker.

Nothing in this module mutates the legacy internal index; it is a standalone,
in-memory contract. See :class:`NormalizedObservationTable` and
:class:`SampleRelationPlan`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Canonical column names (single source of truth across the codebase)
# ---------------------------------------------------------------------------

PHYSICAL_SAMPLE_ID = "physical_sample_id"
INTERNAL_SAMPLE_ID = "internal_sample_id"
UNIT_LEVEL = "unit_level"
UNIT_ID = "unit_id"
SOURCE_ID = "source_id"
OBSERVATION_ID = "observation_id"
REP_ID = "rep_id"
ORIGIN_SAMPLE_ID = "origin_sample_id"
DERIVED_UNIT_ID = "derived_unit_id"
ROW_ID = "row_id"
PARTITION = "partition"
TARGET_ID = "target_id"
SAMPLE_INFLUENCE_WEIGHT = "sample_influence_weight"
QUALITY_FLAG = "quality_flag"
COMPONENT_OBSERVATION_IDS = "component_observation_ids"

#: Ordered canonical schema of the normalised observation table.
RELATION_TABLE_COLUMNS: tuple[str, ...] = (
    ROW_ID,
    PHYSICAL_SAMPLE_ID,
    INTERNAL_SAMPLE_ID,
    SOURCE_ID,
    OBSERVATION_ID,
    REP_ID,
    UNIT_LEVEL,
    UNIT_ID,
    DERIVED_UNIT_ID,
    ORIGIN_SAMPLE_ID,
    PARTITION,
    TARGET_ID,
    SAMPLE_INFLUENCE_WEIGHT,
    QUALITY_FLAG,
    COMPONENT_OBSERVATION_IDS,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UnitLevel(StrEnum):
    """Typed level of a relational unit.

    Attributes:
        PHYSICAL_SAMPLE: One physical sample (the statistical unit).
        SOURCE_SAMPLE: One ``(physical_sample, source)`` pair.
        OBSERVATION: One ``(physical_sample, source, rep)`` measurement.
        COMBO: A derived cartesian unit drawn from several observations.
        ROW: A generic materialised row (representation output).
    """

    PHYSICAL_SAMPLE = "physical_sample"
    SOURCE_SAMPLE = "source_sample"
    OBSERVATION = "observation"
    COMBO = "combo"
    ROW = "row"


class Partition(StrEnum):
    """Partition assignment of a relational row."""

    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class RepOrder(StrEnum):
    """Whether repetitions within a source are exchangeable or ordered.

    Attributes:
        EXCHANGEABLE: Repetitions carry no stable order (the default for NIRS
            replicate measurements); ``rep_id`` is assigned by sorted key.
        ORDERED: Repetition order is semantically meaningful and preserved.
    """

    EXCHANGEABLE = "exchangeable"
    ORDERED = "ordered"


class MissingSourcePolicy(StrEnum):
    """Policy when a source is entirely absent for a sample."""

    STRICT = "strict"
    IMPUTE_DECLARED = "impute_declared"
    DROP_INCOMPLETE = "drop_incomplete"
    DROP_BRANCH = "drop_branch"
    MASK = "mask"
    PAD = "pad"
    PARTIAL_MODEL = "partial_model"


class MissingRepetitionPolicy(StrEnum):
    """Policy when the observed repetition count differs from ``expected``."""

    STRICT = "strict"
    PAD = "pad"
    DROP = "drop"
    TRUNCATE = "truncate"
    IMPUTE_DECLARED = "impute_declared"


class QualityFlag(StrEnum):
    """Per-row quality marker."""

    OK = "ok"
    OUTLIER = "outlier"
    IMPUTED = "imputed"
    PADDED = "padded"
    MISSING = "missing"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RelationValidationError(ValueError):
    """Raised when a relational configuration or table is incoherent.

    Subclasses :class:`ValueError` so that boundary handlers that already catch
    ``ValueError`` (loaders, config parsing) treat it uniformly. Carries an
    optional machine-readable ``code`` mirroring the existing ``REP-Exxx`` style.

    Attributes:
        code: Short actionable error code, e.g. ``"REL-E001"``.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message if code is None else f"{message} [Error: {code}]")
        self.code = code


# ---------------------------------------------------------------------------
# RepetitionSpec (N2) -- source-aware repetition configuration
# ---------------------------------------------------------------------------


@dataclass
class SourceRepetitionSpec:
    """Per-source repetition declaration.

    Attributes:
        rep_col: Optional metadata column holding the repetition index for this
            source. ``None`` means repetition indices are assigned positionally
            within each ``(sample, source)`` group.
        expected: Expected number of repetitions for this source, or ``None`` to
            infer from the data. Validated when ``strict_cardinality`` is set.
    """

    rep_col: str | None = None
    expected: int | None = None

    def __post_init__(self) -> None:
        if self.expected is not None and (not isinstance(self.expected, int) or self.expected < 1):
            raise RelationValidationError(
                f"SourceRepetitionSpec.expected must be a positive integer, got {self.expected!r}",
                code="REL-E010",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifests."""
        out: dict[str, Any] = {}
        if self.rep_col is not None:
            out["rep_col"] = self.rep_col
        if self.expected is not None:
            out["expected"] = self.expected
        return out


@dataclass
class RepetitionSpec:
    """Source-aware repetition specification (roadmap N2).

    Describes how repetitions are organised *per source* rather than along a
    single global row axis. This is the explicit alternative to the legacy
    uniform ``repetition=`` shortcut, and the only contract under which
    heterogeneous sources (``MIR=2``, ``RAMAN=3`` ...) may be joined.

    Attributes:
        sample_id: Name of the user-facing column identifying the physical
            sample. Mapped onto the canonical ``physical_sample_id``.
        target_level: Granularity at which targets live. Only
            ``"physical_sample"`` is supported (targets are sample-level).
        sources: Mapping ``source_id -> SourceRepetitionSpec``.
        missing_repetition_policy: Policy for repetition-count mismatches.
        missing_source_policy: Policy for entirely missing sources.
        rep_order: Whether repetitions are exchangeable or ordered.
        strict_cardinality: If ``True``, every declared ``expected`` count must
            match exactly; otherwise mismatches are tolerated (and recorded).
        link_by: Explicit join key across sources. Defaults to ``sample_id``.
            When set, sources are joined by key, never positionally.
    """

    sample_id: str
    target_level: str = "physical_sample"
    sources: dict[str, SourceRepetitionSpec] = field(default_factory=dict)
    missing_repetition_policy: MissingRepetitionPolicy = MissingRepetitionPolicy.STRICT
    missing_source_policy: MissingSourcePolicy = MissingSourcePolicy.STRICT
    rep_order: RepOrder = RepOrder.EXCHANGEABLE
    strict_cardinality: bool = False
    link_by: str | None = None

    def __post_init__(self) -> None:
        self.missing_repetition_policy = MissingRepetitionPolicy(self.missing_repetition_policy)
        self.missing_source_policy = MissingSourcePolicy(self.missing_source_policy)
        self.rep_order = RepOrder(self.rep_order)
        if self.target_level != "physical_sample":
            raise RelationValidationError(
                f"RepetitionSpec.target_level must be 'physical_sample', got {self.target_level!r}. "
                "Targets are sample-level in nirs4all.",
                code="REL-E011",
            )
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise RelationValidationError(
                "RepetitionSpec.sample_id must be a non-empty column name.",
                code="REL-E012",
            )
        if self.link_by is not None:
            if not isinstance(self.link_by, str) or not self.link_by.strip():
                raise RelationValidationError(
                    "RepetitionSpec.link_by must be a non-empty column name when provided.",
                    code="REL-E012",
                )
            if self.link_by != self.sample_id:
                raise RelationValidationError(
                    f"Distinct link_by columns are not executable yet (sample_id={self.sample_id!r}, "
                    f"link_by={self.link_by!r}). Use the physical sample column as link_by for this phase.",
                    code="REL-E022",
                )

    @property
    def join_key(self) -> str:
        """The effective join key (``link_by`` if set, else ``sample_id``)."""
        return self.link_by or self.sample_id

    def source_spec(self, source_id: str) -> SourceRepetitionSpec:
        """Return the declaration for ``source_id`` (empty default if absent)."""
        return self.sources.get(source_id, SourceRepetitionSpec())

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> RepetitionSpec:
        """Build a :class:`RepetitionSpec` from a config mapping.

        Accepts the experimental YAML shape::

            repetition_spec:
              sample_id: sample_id
              link_by: sample_id
              rep_order: exchangeable
              strict_cardinality: true
              missing_repetition_policy: strict
              missing_source_policy: strict
              sources:
                MIR:   {rep_col: rep, expected: 2}
                RAMAN: {expected: 3}

        Args:
            cfg: The ``repetition_spec`` mapping.

        Returns:
            A validated :class:`RepetitionSpec`.

        Raises:
            RelationValidationError: If the mapping is malformed.
        """
        if not isinstance(cfg, Mapping):
            raise RelationValidationError(
                f"repetition_spec must be a mapping, got {type(cfg).__name__}.",
                code="REL-E013",
            )
        sample_id = cfg.get("sample_id")
        if sample_id is None:
            raise RelationValidationError(
                "repetition_spec requires a 'sample_id' field.",
                code="REL-E012",
            )
        sources_cfg = cfg.get("sources") or {}
        if not isinstance(sources_cfg, Mapping):
            raise RelationValidationError(
                "repetition_spec.sources must be a mapping of source_id -> spec.",
                code="REL-E014",
            )
        sources: dict[str, SourceRepetitionSpec] = {}
        for name, sspec in sources_cfg.items():
            if sspec is None:
                sources[str(name)] = SourceRepetitionSpec()
            elif isinstance(sspec, SourceRepetitionSpec):
                sources[str(name)] = sspec
            elif isinstance(sspec, Mapping):
                sources[str(name)] = SourceRepetitionSpec(
                    rep_col=sspec.get("rep_col"),
                    expected=sspec.get("expected"),
                )
            elif isinstance(sspec, int):
                sources[str(name)] = SourceRepetitionSpec(expected=sspec)
            else:
                raise RelationValidationError(
                    f"repetition_spec.sources['{name}'] must be a mapping, int or null, "
                    f"got {type(sspec).__name__}.",
                    code="REL-E014",
                )
        return cls(
            sample_id=str(sample_id),
            target_level=cfg.get("target_level", "physical_sample"),
            sources=sources,
            missing_repetition_policy=cfg.get("missing_repetition_policy", MissingRepetitionPolicy.STRICT),
            missing_source_policy=cfg.get("missing_source_policy", MissingSourcePolicy.STRICT),
            rep_order=cfg.get("rep_order", RepOrder.EXCHANGEABLE),
            strict_cardinality=bool(cfg.get("strict_cardinality", False)),
            link_by=cfg.get("link_by"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifests / fingerprints."""
        return {
            "sample_id": self.sample_id,
            "target_level": self.target_level,
            "link_by": self.join_key,
            "rep_order": self.rep_order.value,
            "strict_cardinality": self.strict_cardinality,
            "missing_repetition_policy": self.missing_repetition_policy.value,
            "missing_source_policy": self.missing_source_policy.value,
            "sources": {name: spec.to_dict() for name, spec in sorted(self.sources.items())},
        }


# ---------------------------------------------------------------------------
# Relation config parsing (N2) -- bridge legacy dataset dicts -> RepetitionSpec
# ---------------------------------------------------------------------------


@dataclass
class RelationConfig:
    """Parsed relational configuration extracted from a dataset config dict.

    This is the bridge between the (additive, inert) schema fields carried in the
    legacy dataset dict -- ``experimental_relation_pipeline``, ``repetition_spec``,
    ``relations`` and per-source ``link_by`` -- and the executable
    :class:`RepetitionSpec` / join contract.

    Attributes:
        enabled: Whether the experimental relation pipeline is explicitly opted
            in (``experimental_relation_pipeline: true``).
        spec: The parsed :class:`RepetitionSpec`, if a ``repetition_spec`` /
            ``relations`` block was provided.
        link_by: The effective join key, resolved from the spec, the ``relations``
            block, or a per-source ``link_by``.
    """

    enabled: bool = False
    spec: RepetitionSpec | None = None
    link_by: str | None = None

    @property
    def is_relational(self) -> bool:
        """Whether any relational contract is declared (flag, spec or link_by)."""
        return self.enabled or self.spec is not None or self.link_by is not None


def parse_relation_config(config: Mapping[str, Any]) -> RelationConfig | None:
    """Parse the experimental relational fields out of a dataset config dict.

    Reads the additive fields emitted by
    :meth:`~nirs4all.data.schema.config.DatasetConfigSchema` (and any equivalent
    hand-written dict). Returns ``None`` when nothing relational is declared so
    legacy callers can cheaply skip the relational path.

    Args:
        config: The (legacy-shaped) dataset configuration dict.

    Returns:
        A :class:`RelationConfig` if any relational field is present, else
        ``None``.

    Raises:
        RelationValidationError: If a declared ``repetition_spec`` / ``relations``
            block is malformed.
    """
    if not isinstance(config, Mapping):
        return None

    enabled = bool(config.get("experimental_relation_pipeline"))

    spec_cfg: Any = config.get("repetition_spec")
    relations_cfg = config.get("relations")
    if relations_cfg is not None and not isinstance(relations_cfg, Mapping):
        raise RelationValidationError(
            f"relations must be a mapping, got {type(relations_cfg).__name__}.",
            code="REL-E018",
        )
    # `relations` may itself carry the repetition spec, or be the spec inline.
    if spec_cfg is None and relations_cfg is not None:
        if isinstance(relations_cfg.get("repetition_spec"), Mapping):
            spec_cfg = relations_cfg["repetition_spec"]
        elif "sample_id" in relations_cfg:
            spec_cfg = relations_cfg

    spec: RepetitionSpec | None = None
    if spec_cfg is not None:
        try:
            spec = RepetitionSpec.from_config(spec_cfg)
        except RelationValidationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise RelationValidationError(
                f"Could not parse repetition_spec/relations block: {exc}",
                code="REL-E018",
            ) from exc

    sources_link_by = _link_by_from_sources(config.get("_sources"))
    relations_link_by: str | None = None
    if relations_cfg is not None:
        raw_lb = relations_cfg.get("link_by")
        relations_link_by = str(raw_lb) if raw_lb else None

    candidates = [
        (name, value)
        for name, value in (
            ("repetition_spec", spec.join_key if spec is not None else None),
            ("sources.link_by", sources_link_by),
            ("relations.link_by", relations_link_by),
        )
        if value is not None
    ]
    distinct_link_by = {value for _name, value in candidates}
    if len(distinct_link_by) > 1:
        details = ", ".join(f"{name}={value!r}" for name, value in candidates)
        raise RelationValidationError(
            f"Conflicting link_by declarations: {details}. Use one shared key column or make "
            "repetition_spec.link_by explicit.",
            code="REL-E017",
        )
    link_by = candidates[0][1] if candidates else None

    if not enabled and spec is None and link_by is None:
        return None
    return RelationConfig(enabled=enabled, spec=spec, link_by=link_by)


def _link_by_from_sources(sources_meta: Any) -> str | None:
    """Return the unique per-source ``link_by`` declared in a ``_sources`` list."""
    if isinstance(sources_meta, Sequence) and not isinstance(sources_meta, (str, bytes)):
        values: list[str] = []
        for src in sources_meta:
            if isinstance(src, Mapping) and src.get("link_by"):
                values.append(str(src["link_by"]))
        distinct = set(values)
        if len(distinct) > 1:
            raise RelationValidationError(
                f"Per-source link_by declarations disagree: {sorted(distinct)}. Heterogeneous sources "
                "must share one executable join key.",
                code="REL-E017",
            )
        if values:
            return values[0]
    return None


# ---------------------------------------------------------------------------
# Observation record + normalised table (N1)
# ---------------------------------------------------------------------------


def make_observation_id(source_id: str, physical_sample_id: str, rep_id: int) -> str:
    """Return the canonical observation id ``"<source>:<sample>:<rep>"``."""
    return f"{source_id}:{physical_sample_id}:{rep_id}"


@dataclass
class ObservationRecord:
    """One row of the :class:`NormalizedObservationTable`.

    Carries the full canonical identity schema. For raw observations,
    ``unit_level`` is :attr:`UnitLevel.OBSERVATION` and ``derived_unit_id`` /
    ``component_observation_ids`` are empty. Derived combos set those fields and
    use :attr:`UnitLevel.COMBO`.
    """

    physical_sample_id: str
    internal_sample_id: int
    source_id: str
    rep_id: int
    observation_id: str
    row_id: int
    partition: Partition = Partition.TRAIN
    target_id: Any = None
    origin_sample_id: str | None = None
    unit_level: UnitLevel = UnitLevel.OBSERVATION
    unit_id: str = ""
    derived_unit_id: str | None = None
    sample_influence_weight: float = 1.0
    quality_flag: QualityFlag = QualityFlag.OK
    component_observation_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Original row index of this observation within its source's input order.
    #: Internal mapping field used by phase-N3 staging to index back into the
    #: per-source feature matrices. It is intentionally *not* part of the
    #: canonical :data:`RELATION_TABLE_COLUMNS` schema and does not affect the
    #: fingerprint (source/row order must not change identity).
    source_row: int | None = None

    def __post_init__(self) -> None:
        if self.origin_sample_id is None:
            self.origin_sample_id = self.physical_sample_id
        if not self.unit_id:
            self.unit_id = self.observation_id

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical column mapping for this record."""
        return {
            ROW_ID: self.row_id,
            PHYSICAL_SAMPLE_ID: self.physical_sample_id,
            INTERNAL_SAMPLE_ID: self.internal_sample_id,
            SOURCE_ID: self.source_id,
            OBSERVATION_ID: self.observation_id,
            REP_ID: self.rep_id,
            UNIT_LEVEL: self.unit_level.value,
            UNIT_ID: self.unit_id,
            DERIVED_UNIT_ID: self.derived_unit_id,
            ORIGIN_SAMPLE_ID: self.origin_sample_id,
            PARTITION: self.partition.value,
            TARGET_ID: self.target_id,
            SAMPLE_INFLUENCE_WEIGHT: self.sample_influence_weight,
            QUALITY_FLAG: self.quality_flag.value,
            COMPONENT_OBSERVATION_IDS: list(self.component_observation_ids),
        }


@dataclass
class ComboRecord:
    """Lineage descriptor for a derived cartesian combination.

    Attributes:
        derived_unit_id: Stable combo id ``"<sample>::<rep_a>x<rep_b>..."``.
        physical_sample_id: The originating physical sample.
        origin_sample_id: Provenance (equals ``physical_sample_id``).
        component_observation_ids: Ordered component observation ids.
        rep_ids_by_source: Mapping ``source_id -> rep_id`` for the components.
    """

    derived_unit_id: str
    physical_sample_id: str
    origin_sample_id: str
    component_observation_ids: tuple[str, ...]
    rep_ids_by_source: dict[str, int]


class NormalizedObservationTable:
    """In-memory canonical table of observations / combos / samples.

    The table is the single resolution of sample/source/observation identity.
    It is intentionally backed by plain Python records (not the polars index) so
    it can be built and validated without touching legacy storage. Phase N3 will
    consume it to materialise aligned features.
    """

    def __init__(self, records: Sequence[ObservationRecord]):
        self.records: list[ObservationRecord] = list(records)

    # -- basic accessors ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    @property
    def physical_sample_ids(self) -> list[str]:
        """Sorted unique physical sample ids."""
        return sorted({r.physical_sample_id for r in self.records})

    @property
    def source_ids(self) -> list[str]:
        """Sorted unique source ids."""
        return sorted({r.source_id for r in self.records})

    def internal_id_map(self) -> dict[str, int]:
        """Mapping ``physical_sample_id -> internal_sample_id``."""
        return {r.physical_sample_id: r.internal_sample_id for r in self.records}

    def records_for_sample(self, physical_sample_id: str) -> list[ObservationRecord]:
        """All records belonging to a physical sample (in row order)."""
        return [r for r in self.records if r.physical_sample_id == physical_sample_id]

    def cardinalities(self) -> dict[tuple[str, str], int]:
        """Observation counts keyed by ``(physical_sample_id, source_id)``."""
        counts: Counter[tuple[str, str]] = Counter()
        for r in self.records:
            if r.unit_level is UnitLevel.OBSERVATION:
                counts[(r.physical_sample_id, r.source_id)] += 1
        return dict(counts)

    def targets_by_sample(self) -> dict[str, Any]:
        """Sample-level target value per physical sample."""
        out: dict[str, Any] = {}
        for r in self.records:
            if r.physical_sample_id not in out and r.target_id is not None:
                out[r.physical_sample_id] = r.target_id
        return out

    def to_columns(self) -> dict[str, list[Any]]:
        """Columnar view keyed by :data:`RELATION_TABLE_COLUMNS`."""
        cols: dict[str, list[Any]] = {c: [] for c in RELATION_TABLE_COLUMNS}
        for r in self.records:
            row = r.to_dict()
            for c in RELATION_TABLE_COLUMNS:
                cols[c].append(row[c])
        return cols

    # -- influence ---------------------------------------------------------

    def with_equal_sample_influence(self) -> NormalizedObservationTable:
        """Return a copy whose weights equalise per-sample influence.

        Each physical sample receives total influence ``1.0``, split uniformly
        across its observation rows (``1 / n_obs(sample)``). This is an auditable
        derivation; the default table keeps ``1.0`` per row (phase N7 owns the
        effective policy).
        """
        per_sample: Counter[str] = Counter()
        for r in self.records:
            if r.unit_level is UnitLevel.OBSERVATION:
                per_sample[r.physical_sample_id] += 1
        new_records: list[ObservationRecord] = []
        for r in self.records:
            weight = 1.0 / per_sample[r.physical_sample_id] if per_sample[r.physical_sample_id] else 1.0
            clone = ObservationRecord(**{**r.__dict__, "metadata": dict(r.metadata)})
            clone.sample_influence_weight = weight
            new_records.append(clone)
        return NormalizedObservationTable(new_records)

    # -- combo lineage -----------------------------------------------------

    def enumerate_combos(self, physical_sample_id: str) -> list[ComboRecord]:
        """Enumerate the full cartesian combos for a physical sample.

        Each combo is the deterministic cross-product of one observation per
        source (in :data:`source_ids` order), carrying full lineage. This is the
        combo-lineage helper referenced by N1; cartesian *materialisation* and
        cost caps belong to later phases.

        Args:
            physical_sample_id: The sample to enumerate.

        Returns:
            A deterministic list of :class:`ComboRecord`. Empty if any source is
            missing for the sample.
        """
        by_source: dict[str, list[ObservationRecord]] = defaultdict(list)
        for r in self.records_for_sample(physical_sample_id):
            if r.unit_level is UnitLevel.OBSERVATION:
                by_source[r.source_id].append(r)
        source_order = sorted(by_source)
        # Require every known source to be present for a full cartesian combo.
        if not source_order or set(source_order) != set(self.source_ids):
            return []
        for s in source_order:
            by_source[s].sort(key=lambda rec: rec.rep_id)

        combos: list[ComboRecord] = []

        def _recurse(idx: int, chosen: list[ObservationRecord]) -> None:
            if idx == len(source_order):
                comp_ids = tuple(rec.observation_id for rec in chosen)
                rep_by_src = {rec.source_id: rec.rep_id for rec in chosen}
                token = "x".join(f"{rec.source_id}{rec.rep_id}" for rec in chosen)
                combos.append(
                    ComboRecord(
                        derived_unit_id=f"{physical_sample_id}::{token}",
                        physical_sample_id=physical_sample_id,
                        origin_sample_id=physical_sample_id,
                        component_observation_ids=comp_ids,
                        rep_ids_by_source=rep_by_src,
                    )
                )
                return
            for rec in by_source[source_order[idx]]:
                _recurse(idx + 1, [*chosen, rec])

        _recurse(0, [])
        return combos

    # -- fingerprint -------------------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the canonical, order-independent table.

        Records are canonicalised and sorted so that source order, shuffled
        repetitions and insertion order do not change the digest.
        """
        canon = sorted(_canonical_record(r) for r in self.records)
        payload = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- validation --------------------------------------------------------

    def validate(self, spec: RepetitionSpec | None = None) -> None:
        """Validate the structural invariants of the table.

        Checks:

        * uniqueness of ``(physical_sample_id, source_id, rep_id)``;
        * constant target per physical sample;
        * non-contradictory sample-level metadata;
        * declared cardinalities when ``spec`` provides ``expected`` and
          ``strict_cardinality`` is set.

        Args:
            spec: Optional :class:`RepetitionSpec` providing expected
                cardinalities and the strictness flag.

        Raises:
            RelationValidationError: On any violated invariant.
        """
        self._validate_uniqueness()
        self._validate_constant_target()
        self._validate_metadata_consistency()
        if spec is not None:
            self._validate_cardinalities(spec)

    def _validate_uniqueness(self) -> None:
        seen: Counter[tuple[str, str, int]] = Counter()
        for r in self.records:
            if r.unit_level is UnitLevel.OBSERVATION:
                seen[(r.physical_sample_id, r.source_id, r.rep_id)] += 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        if dupes:
            sample, source, rep = next(iter(dupes))
            raise RelationValidationError(
                f"Duplicate observation key (physical_sample_id={sample!r}, source_id={source!r}, "
                f"rep_id={rep}) appears {dupes[(sample, source, rep)]} times. Each "
                "(sample, source, rep) must be unique; deduplicate or assign distinct rep ids.",
                code="REL-E001",
            )

    def _validate_constant_target(self) -> None:
        per_sample: dict[str, Any] = {}
        for r in self.records:
            if r.target_id is None:
                continue
            if r.physical_sample_id not in per_sample:
                per_sample[r.physical_sample_id] = r.target_id
            elif not _targets_equal(per_sample[r.physical_sample_id], r.target_id):
                raise RelationValidationError(
                    f"Physical sample {r.physical_sample_id!r} has contradictory targets "
                    f"({per_sample[r.physical_sample_id]!r} vs {r.target_id!r}). Targets are "
                    "sample-level and must be constant across a sample's observations.",
                    code="REL-E002",
                )

    def _validate_metadata_consistency(self) -> None:
        per_sample: dict[str, dict[str, Any]] = {}
        for r in self.records:
            if not r.metadata:
                continue
            ref = per_sample.setdefault(r.physical_sample_id, {})
            for key, value in r.metadata.items():
                if key in ref:
                    if not _values_equal(ref[key], value):
                        raise RelationValidationError(
                            f"Physical sample {r.physical_sample_id!r} has contradictory sample-level "
                            f"metadata for {key!r} ({ref[key]!r} vs {value!r}). Sample-level metadata "
                            "must be consistent across a sample's observations.",
                            code="REL-E003",
                        )
                else:
                    ref[key] = value

    def _validate_cardinalities(self, spec: RepetitionSpec) -> None:
        cards = self.cardinalities()
        for source_id, sspec in spec.sources.items():
            if sspec.expected is None:
                continue
            observed_by_sample = {
                sample_id: cards.get((sample_id, source_id), 0)
                for sample_id in self.physical_sample_ids
            }
            bad = {sample_id: count for sample_id, count in observed_by_sample.items() if count != sspec.expected}
            if bad and spec.strict_cardinality:
                raise RelationValidationError(
                    f"Source {source_id!r} declares expected={sspec.expected} repetitions but observed "
                    f"mismatches {bad}. With strict_cardinality=True every sample must match.",
                    code="REL-E004",
                )


# ---------------------------------------------------------------------------
# Per-source observation inputs + link_by join builder (N2)
# ---------------------------------------------------------------------------


@dataclass
class SourceObservations:
    """Per-source observation columns fed to :func:`build_relation_table`.

    This is the decoupled, IO-free input to the join contract: callers extract
    the join-key values (one per observation row), optional explicit repetition
    indices, optional targets and optional sample-level metadata columns.

    Attributes:
        source_id: The source / modality name.
        sample_ids: The join-key value per observation row (the physical sample
            this row belongs to). ``None`` here means "no key column available".
        rep_ids: Optional explicit repetition index per row.
        targets: Optional per-row target value.
        metadata: Optional mapping ``column -> per-row values``.
    """

    source_id: str
    sample_ids: Sequence[Any] | None
    rep_ids: Sequence[Any] | None = None
    targets: Sequence[Any] | None = None
    metadata: Mapping[str, Sequence[Any]] | None = None

    @classmethod
    def from_frame(
        cls,
        source_id: str,
        frame: Any,
        *,
        sample_col: str | None,
        rep_col: str | None = None,
        target_col: str | None = None,
        metadata_cols: Sequence[str] | None = None,
    ) -> SourceObservations:
        """Build :class:`SourceObservations` from a column container.

        Works with anything exposing ``frame[column]`` returning a sequence
        (pandas ``DataFrame``, dict of lists, polars frame ...). This is the
        seam a future loader (phase N3) uses to assemble per-source observations
        from already-read tables.

        Args:
            source_id: The source / modality name.
            frame: A column container (``frame[col]`` -> values).
            sample_col: Column holding the physical sample id, or ``None`` to
                declare that the source exposes no key (positional join refused).
            rep_col: Optional column holding the repetition index.
            target_col: Optional column holding the per-row target.
            metadata_cols: Optional sample-level metadata columns to carry.

        Returns:
            A :class:`SourceObservations` instance.
        """

        def _col(name: str | None) -> list[Any] | None:
            if name is None:
                return None
            if not _frame_has_column(frame, name):
                raise RelationValidationError(
                    f"Source {source_id!r} is missing required column {name!r}. "
                    f"Available columns: {_available_columns(frame)}.",
                    code="REL-E005",
                )
            col = frame[name]
            if hasattr(col, "tolist"):
                return list(col.tolist())
            return list(col)

        metadata: dict[str, list[Any]] | None = None
        if metadata_cols:
            metadata = {name: _col(name) or [] for name in metadata_cols}

        return cls(
            source_id=source_id,
            sample_ids=_col(sample_col),
            rep_ids=_col(rep_col),
            targets=_col(target_col),
            metadata=metadata,
        )


def build_relation_table(
    spec: RepetitionSpec,
    sources: Sequence[SourceObservations],
    *,
    partition: Partition | str = Partition.TRAIN,
) -> NormalizedObservationTable:
    """Build a :class:`NormalizedObservationTable` by joining sources on a key.

    This is the real ``link_by`` join contract (roadmap N2): sources are joined
    by their key values, never by row position. Source order and per-row order
    are irrelevant -- only the keys matter -- so shuffled sources are supported.

    Args:
        spec: The source-aware repetition specification.
        sources: One :class:`SourceObservations` per source.
        partition: Partition label applied to every produced row.

    Returns:
        A validated :class:`NormalizedObservationTable`.

    Raises:
        RelationValidationError: If the join is not executable as a true join
            (missing key column), if keys are ambiguous, or if any structural
            invariant fails.
    """
    partition = Partition(partition)
    if not sources:
        raise RelationValidationError("build_relation_table requires at least one source.", code="REL-E020")

    # Refuse positional joins: every source must expose a key column.
    missing_key = [s.source_id for s in sources if s.sample_ids is None]
    if missing_key:
        raise RelationValidationError(
            f"link_by join requires a sample-id column for every source, but sources {missing_key} "
            f"expose none. Positional joins are refused when link_by={spec.join_key!r} is required; "
            "provide the key column or drop the relation spec.",
            code="REL-E005",
        )

    # Assign deterministic internal ids ordered by the stringified physical sample id.
    all_keys: set[str] = set()
    for s in sources:
        assert s.sample_ids is not None
        for i, key in enumerate(s.sample_ids):
            all_keys.add(_key_str(key, source_id=s.source_id, row_index=i, key_name=spec.join_key))
    internal_ids = {key: idx for idx, key in enumerate(sorted(all_keys))}

    records: list[ObservationRecord] = []
    row_counter = 0

    for s in sources:
        assert s.sample_ids is not None
        n = len(s.sample_ids)
        _check_column_length("rep_ids", s.rep_ids, n, s.source_id)
        _check_column_length("targets", s.targets, n, s.source_id)
        if s.metadata:
            for col, values in s.metadata.items():
                _check_column_length(f"metadata[{col!r}]", values, n, s.source_id)

        # Group rows of this source by physical sample to assign rep ids.
        rows_by_sample: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(s.sample_ids):
            rows_by_sample[_key_str(key, source_id=s.source_id, row_index=i, key_name=spec.join_key)].append(i)

        for skey, row_indices in rows_by_sample.items():
            rep_values = _resolve_rep_ids(s, row_indices, spec.rep_order)
            for i, rep_id in zip(row_indices, rep_values, strict=True):
                target = s.targets[i] if s.targets is not None else None
                meta = {col: values[i] for col, values in s.metadata.items()} if s.metadata else {}
                obs_id = make_observation_id(s.source_id, skey, rep_id)
                records.append(
                    ObservationRecord(
                        physical_sample_id=skey,
                        internal_sample_id=internal_ids[skey],
                        source_id=s.source_id,
                        rep_id=rep_id,
                        observation_id=obs_id,
                        row_id=row_counter,
                        partition=partition,
                        target_id=target,
                        metadata=meta,
                        source_row=i,
                    )
                )
                row_counter += 1

    table = NormalizedObservationTable(records)
    table.validate(spec)
    _validate_missing_sources(table, spec)
    return table


def _resolve_rep_ids(
    source: SourceObservations,
    row_indices: Sequence[int],
    rep_order: RepOrder,
) -> list[int]:
    """Resolve repetition ids for one ``(source, sample)`` group.

    Explicit ``rep_ids`` are honoured verbatim. Otherwise positions are assigned
    ``0..k-1`` -- preserving order for ``ORDERED`` and using stable enumeration
    for ``EXCHANGEABLE``.
    """
    if source.rep_ids is not None:
        out: list[int] = []
        for i in row_indices:
            raw = source.rep_ids[i]
            try:
                out.append(int(raw))
            except (TypeError, ValueError) as exc:
                raise RelationValidationError(
                    f"Source {source.source_id!r} has a non-integer rep id {raw!r}; rep ids must be "
                    "integers (or omit rep_col to assign them positionally).",
                    code="REL-E006",
                ) from exc
        return out
    # Positional assignment. row_indices already reflect source row order.
    return list(range(len(row_indices)))


def _validate_missing_sources(table: NormalizedObservationTable, spec: RepetitionSpec) -> None:
    """Enforce ``missing_source_policy=strict`` (every sample has every source)."""
    if spec.missing_source_policy is not MissingSourcePolicy.STRICT:
        return
    declared_sources = set(spec.sources) or set(table.source_ids)
    present: dict[str, set[str]] = defaultdict(set)
    for r in table.records:
        if r.unit_level is UnitLevel.OBSERVATION:
            present[r.physical_sample_id].add(r.source_id)
    for sample_id in table.physical_sample_ids:
        missing = declared_sources - present.get(sample_id, set())
        if missing:
            raise RelationValidationError(
                f"Physical sample {sample_id!r} is missing source(s) {sorted(missing)} with "
                "missing_source_policy=strict. Provide the source or relax the policy "
                "(impute_declared / drop_incomplete / mask / pad).",
                code="REL-E007",
            )


# ---------------------------------------------------------------------------
# SampleRelationPlan (N1)
# ---------------------------------------------------------------------------


@dataclass
class SampleRelationPlan:
    """Resolved relational plan binding a spec to a normalised table.

    Holds the single resolution of sample identity that downstream phases
    (split, scoring, refit, export) must consume instead of recomputing.

    Attributes:
        table: The validated :class:`NormalizedObservationTable`.
        spec: The :class:`RepetitionSpec` that produced it (if any).
        rep_order: Effective repetition order.
    """

    table: NormalizedObservationTable
    spec: RepetitionSpec | None = None
    rep_order: RepOrder = RepOrder.EXCHANGEABLE

    @classmethod
    def from_sources(
        cls,
        spec: RepetitionSpec,
        sources: Sequence[SourceObservations],
        *,
        partition: Partition | str = Partition.TRAIN,
    ) -> SampleRelationPlan:
        """Build and validate a plan from per-source observations."""
        table = build_relation_table(spec, sources, partition=partition)
        return cls(table=table, spec=spec, rep_order=spec.rep_order)

    @property
    def physical_to_internal(self) -> dict[str, int]:
        """Stable ``physical_sample_id -> internal_sample_id`` mapping."""
        return self.table.internal_id_map()

    def declared_cardinalities(self) -> dict[str, int | None]:
        """Declared ``expected`` count per source (``None`` if not declared)."""
        if self.spec is None:
            return {}
        return {name: s.expected for name, s in self.spec.sources.items()}

    def fingerprint(self) -> str:
        """Deterministic digest combining the spec and the table."""
        spec_payload = self.spec.to_dict() if self.spec is not None else {}
        payload = json.dumps(
            {"spec": spec_payload, "table": self.table.fingerprint(), "rep_order": self.rep_order.value},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Guardrails (N0)
# ---------------------------------------------------------------------------

#: Repetition mechanisms that must not compose implicitly (exclusivity matrix).
REPETITION_MECHANISMS: tuple[str, ...] = ("repetition", "rep_to_sources", "rep_to_pp", "rep_fusion")


def detect_repetition_mechanisms(
    steps: Sequence[Any],
    *,
    has_global_repetition: bool = False,
) -> set[str]:
    """Detect which repetition mechanisms a pipeline activates.

    Args:
        steps: The pipeline step list (dicts / objects). Keyword steps such as
            ``{"rep_to_sources": ...}`` are recognised by their keys.
        has_global_repetition: Whether the dataset declares a global, uniform
            ``repetition=`` column.

    Returns:
        The set of active mechanism names (subset of
        :data:`REPETITION_MECHANISMS`).
    """
    active: set[str] = set()
    if has_global_repetition:
        active.add("repetition")
    for step in steps or []:
        if isinstance(step, Mapping):
            for key in step:
                if key in ("rep_to_sources", "rep_to_pp", "rep_fusion"):
                    active.add(key)
    return active


def check_repetition_exclusivity(
    steps: Sequence[Any],
    *,
    has_global_repetition: bool = False,
) -> None:
    """Enforce the repetition exclusivity matrix.

    Legacy uniform ``repetition=``, ``rep_to_sources``, ``rep_to_pp`` and the
    relational ``rep_fusion`` are mutually exclusive paths; combining them is
    rejected so they never compose implicitly.

    Args:
        steps: The pipeline step list.
        has_global_repetition: Whether a global ``repetition=`` is configured.

    Raises:
        RelationValidationError: If more than one mechanism is active.
    """
    active = detect_repetition_mechanisms(steps, has_global_repetition=has_global_repetition)
    if len(active) > 1:
        ordered = [m for m in REPETITION_MECHANISMS if m in active]
        raise RelationValidationError(
            f"Conflicting repetition mechanisms are active together: {ordered}. These paths are "
            "mutually exclusive -- use exactly one of repetition= (uniform legacy), rep_to_sources, "
            "rep_to_pp, or rep_fusion (relational).",
            code="REL-E008",
        )


def audit_source_lengths(
    lengths: Sequence[int],
    *,
    relation_mode: bool = False,
    link_by: str | None = None,
) -> None:
    """Audit multi-source row counts for positional alignment coherence.

    In the legacy positional loader, sources are concatenated by row position,
    which is only valid when every source has the same number of rows. Unequal
    counts mean the sources are heterogeneous and cannot be aligned positionally;
    this previously produced a silently incoherent dataset.

    Args:
        lengths: Row count per source.
        relation_mode: Whether an explicit relational pipeline is active. When
            ``True`` the heterogeneous case is handled by the relation table, so
            this audit defers (it does not bless positional alignment).
        link_by: The declared join key, if any (used to tailor the message).

    Raises:
        RelationValidationError: If sources have unequal lengths and no relation
            plan can resolve them.
    """
    if len(lengths) <= 1:
        return
    if len(set(lengths)) == 1:
        return
    if relation_mode:
        # A relational plan owns heterogeneous alignment (materialised in N3).
        return
    hint = (
        f"A link_by={link_by!r} key is declared but is not executed as a true join in the legacy "
        "loader; true heterogeneous-source joins are handled by the experimental relation pipeline "
        "materialisation phase."
        if link_by
        else "Declare a relation spec (link_by / repetition_spec) for the experimental relation "
        "pipeline before loading heterogeneous sources."
    )
    raise RelationValidationError(
        f"Multi-source feature blocks have different row counts {list(lengths)} and cannot be aligned "
        f"positionally. {hint}",
        code="REL-E009",
    )


def audit_link_by_executable(
    *,
    link_by: str | None,
    relation_mode: bool,
    available_columns: Sequence[str] | None = None,
) -> None:
    """Ensure a declared ``link_by`` is an executable join in relation mode.

    A ``link_by`` that is parsed but silently ignored is a correctness trap. In
    the relational profile a declared key must resolve to a real column.

    Args:
        link_by: The declared join key (or ``None``).
        relation_mode: Whether the experimental relational pipeline is active.
        available_columns: Columns available to satisfy the join, if known.

    Raises:
        RelationValidationError: If ``link_by`` cannot be executed as a join.
    """
    if not link_by or not relation_mode:
        return
    if available_columns is not None and link_by not in available_columns:
        raise RelationValidationError(
            f"link_by={link_by!r} is declared but no such column is available to join on "
            f"(available: {list(available_columns)}). A declared link_by must be an executable join.",
            code="REL-E005",
        )


def audit_link_by_alignment(
    source_keys: Mapping[str, Sequence[Any] | None],
    *,
    link_by: str | None = None,
    require_unique: bool = True,
) -> None:
    """Validate that declared ``link_by`` keys make positional alignment safe.

    This is the *positional* safety contract, distinct from the real relational
    join (:func:`build_relation_table`). When several sources are about to be
    concatenated by row position but a ``link_by`` key is declared, the keys must
    establish an unambiguous 1:1 correspondence. Concretely:

    * every source must expose the key (no positional-only source);
    * keys must be unique within each source (when ``require_unique``); otherwise
      the positional correspondence is ambiguous (heterogeneous repetitions);
    * every source must cover exactly the same set of keys; otherwise the sources
      are divergent and cannot be aligned by position.

    Non-unique keys (repetitions) and divergent / missing keys are *not* silently
    tolerated here -- they require the real relational join, not positional
    concatenation.

    Args:
        source_keys: Mapping ``source_id -> per-row key values``. A ``None`` value
            marks a source that exposes no key column at all.
        link_by: The declared join key (used only to tailor messages).
        require_unique: Whether keys must be unique within each source.

    Raises:
        RelationValidationError: If positional alignment is unsafe.
    """
    if len(source_keys) <= 1:
        return
    key_name = link_by or "link_by"

    # 1. Every source must expose the key column.
    missing = sorted(s for s, keys in source_keys.items() if keys is None)
    if missing:
        raise RelationValidationError(
            f"link_by={key_name!r} alignment requires a key column for every source, but "
            f"sources {missing} expose none. Provide the key column or drop the link_by spec.",
            code="REL-E005",
        )

    str_keys: dict[str, list[str]] = {
        s: [_key_str(k, source_id=s, row_index=i, key_name=key_name) for i, k in enumerate(keys)]
        for s, keys in source_keys.items()
        if keys is not None
    }

    if require_unique:
        lengths = {source_id: len(keys) for source_id, keys in str_keys.items()}
        if len(set(lengths.values())) > 1:
            raise RelationValidationError(
                f"link_by={key_name!r} keys have different row counts by source: {lengths}. "
                "Positional alignment requires equal lengths; use a relation join for heterogeneous repetitions.",
                code="REL-E017",
            )

    # 2. Uniqueness within each source (1:1 positional correspondence).
    if require_unique:
        for source_id, keys in str_keys.items():
            counts = Counter(keys)
            dupes = sorted(k for k, c in counts.items() if c > 1)
            if dupes:
                raise RelationValidationError(
                    f"link_by={key_name!r} is non-unique in source {source_id!r}: keys {dupes[:5]} "
                    "repeat. Positional alignment needs a unique key per source; declare a "
                    "repetition_spec to treat repeats as repetitions, or deduplicate.",
                    code="REL-E016",
                )

    # 3. Default positional safety requires the same row-by-row key sequence.
    # The relational join path handles shuffled sources; positional concatenation
    # must not silently pair S1 from one source with S3 from another.
    if require_unique:
        reference_source, reference_keys = next(iter(str_keys.items()))
        reference_set = set(reference_keys)
        for source_id, keys in str_keys.items():
            if keys == reference_keys:
                continue
            key_set = set(keys)
            if key_set == reference_set:
                raise RelationValidationError(
                    f"link_by={key_name!r} keys contain the same ids in sources {reference_source!r} and "
                    f"{source_id!r}, but row order differs. Positional alignment would pair different "
                    "physical samples; use a relation join or reorder explicitly.",
                    code="REL-E017",
                )
            only_ref = sorted(reference_set - key_set)
            only_src = sorted(key_set - reference_set)
            raise RelationValidationError(
                f"link_by={key_name!r} keys diverge between sources {reference_source!r} and "
                f"{source_id!r} despite equal lengths: "
                f"{only_ref[:3]} only in {reference_source!r}, {only_src[:3]} only in {source_id!r}. "
                "Sources with divergent ids cannot be aligned positionally; use a relation join.",
                code="REL-E017",
            )
        return

    # Repetition-aware callers can disable uniqueness to audit sample coverage
    # only; cardinality differences are owned by the relation table.
    key_sets: dict[str, set[str]] = {s: set(keys) for s, keys in str_keys.items()}
    reference_source, reference = next(iter(key_sets.items()))
    for source_id, key_set in key_sets.items():
        if key_set != reference:
            only_ref = sorted(reference - key_set)
            only_src = sorted(key_set - reference)
            raise RelationValidationError(
                f"link_by={key_name!r} key coverage diverges between sources {reference_source!r} and "
                f"{source_id!r}: {only_ref[:3]} only in {reference_source!r}, {only_src[:3]} only in "
                f"{source_id!r}. Use a relation join with an explicit missing-source policy.",
                code="REL-E017",
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _key_str(value: Any, *, source_id: str | None = None, row_index: int | None = None, key_name: str = "link_by") -> str:
    """Stringify a join key deterministically (stable physical_sample_id)."""
    if _is_missing_key(value):
        where = []
        if source_id is not None:
            where.append(f"source {source_id!r}")
        if row_index is not None:
            where.append(f"row {row_index}")
        location = " at " + ", ".join(where) if where else ""
        raise RelationValidationError(
            f"Missing {key_name!r} value{location}. Relation joins require a non-null physical sample key "
            "for every observation row.",
            code="REL-E005",
        )
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_missing_key(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, bytes):
        return value.strip() == b""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _available_columns(frame: Any) -> list[str]:
    if hasattr(frame, "columns"):
        return [str(c) for c in frame.columns]
    if isinstance(frame, Mapping):
        return [str(c) for c in frame]
    return []


def _frame_has_column(frame: Any, name: str) -> bool:
    columns = _available_columns(frame)
    if columns:
        return name in columns
    try:
        frame[name]
    except Exception:
        return False
    return True


def _check_column_length(name: str, values: Sequence[Any] | None, n: int, source_id: str) -> None:
    if values is not None and len(values) != n:
        raise RelationValidationError(
            f"Source {source_id!r} column {name} has length {len(values)} but expected {n} "
            "(one value per observation row).",
            code="REL-E015",
        )


def _targets_equal(a: Any, b: Any) -> bool:
    return _values_equal(a, b)


def _values_equal(a: Any, b: Any) -> bool:
    """Equality tolerant of float noise and numpy scalars."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return bool(a == b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)
    return bool(a == b)


def _canonical_record(record: ObservationRecord) -> str:
    """Canonical, order-independent string form of a record for fingerprinting."""
    payload = {
        PHYSICAL_SAMPLE_ID: record.physical_sample_id,
        SOURCE_ID: record.source_id,
        REP_ID: record.rep_id,
        UNIT_LEVEL: record.unit_level.value,
        DERIVED_UNIT_ID: record.derived_unit_id,
        ORIGIN_SAMPLE_ID: record.origin_sample_id,
        PARTITION: record.partition.value,
        TARGET_ID: _fingerprint_value(record.target_id),
        QUALITY_FLAG: record.quality_flag.value,
        COMPONENT_OBSERVATION_IDS: list(record.component_observation_ids),
        "metadata": {k: _fingerprint_value(v) for k, v in sorted(record.metadata.items())},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fingerprint_value(value: Any) -> Any:
    """Normalise a value for stable fingerprinting (floats -> repr)."""
    if isinstance(value, float):
        return repr(round(value, 12))
    if value is None or isinstance(value, (int, str, bool)):
        return value
    return str(value)


__all__ = [
    # column names
    "PHYSICAL_SAMPLE_ID",
    "INTERNAL_SAMPLE_ID",
    "UNIT_LEVEL",
    "UNIT_ID",
    "SOURCE_ID",
    "OBSERVATION_ID",
    "REP_ID",
    "ORIGIN_SAMPLE_ID",
    "DERIVED_UNIT_ID",
    "ROW_ID",
    "PARTITION",
    "TARGET_ID",
    "SAMPLE_INFLUENCE_WEIGHT",
    "QUALITY_FLAG",
    "COMPONENT_OBSERVATION_IDS",
    "RELATION_TABLE_COLUMNS",
    # enums
    "UnitLevel",
    "Partition",
    "RepOrder",
    "MissingSourcePolicy",
    "MissingRepetitionPolicy",
    "QualityFlag",
    # errors
    "RelationValidationError",
    # spec
    "SourceRepetitionSpec",
    "RepetitionSpec",
    # config parsing
    "RelationConfig",
    "parse_relation_config",
    # table
    "ObservationRecord",
    "ComboRecord",
    "NormalizedObservationTable",
    "SourceObservations",
    "make_observation_id",
    "build_relation_table",
    # plan
    "SampleRelationPlan",
    # guardrails
    "REPETITION_MECHANISMS",
    "detect_repetition_mechanisms",
    "check_repetition_exclusivity",
    "audit_source_lengths",
    "audit_link_by_executable",
    "audit_link_by_alignment",
]
