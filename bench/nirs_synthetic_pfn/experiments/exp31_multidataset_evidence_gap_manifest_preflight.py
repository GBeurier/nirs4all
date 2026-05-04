"""exp31 multidataset evidence-gap manifest preflight (Phase M2 next-action #2).

Bench-only preflight that reads a CSV/JSON manifest authored against
``docs/16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md`` and emits per-family
decisions per ``docs/17_M2_MULTIDATASET_PREFLIGHT_DESIGN.md``.

Strict scope:

- no spectral file content is read (no ``X*.csv``/``Y*.csv``/``M*.csv`` reads);
- no nirs4all module is imported or required;
- no statistic, no PCA, no covariance, no quantile/marginal/noise capture,
  no calibration, no ML, no DL is computed;
- no labels, targets, splits, downstream metrics, adversarial AUC, or
  transfer scores are accepted as manifest inputs;
- no aggregate cross-family score, threshold, gate, or parameter is emitted;
- a ``ready`` decision authorizes only writing a per-family Phase M3
  mechanism design document, never a generator/profile/gate/promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

EXP31_AUDIT_SCOPE = "bench_only_phase_m2_multidataset_evidence_gap_manifest_preflight"
COMPARISON_SPACE = "uncalibrated_raw_or_unknown"
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv")

REGIME_FAMILIES: tuple[str, ...] = (
    "plant_leaf_visnir_swir",
    "manure_organic_mineral",
    "liquid_food",
    "mineral_incombustible",
    "wavenumber_domain",
)

FAMILY_TOKENS: dict[str, tuple[str, ...]] = {
    "plant_leaf_visnir_swir": (
        "leaf",
        "leaves",
        "foliage",
        "vegetation",
        "plant",
        "grapevine",
        "vine",
        "grape",
        "alpine",
        "ecosis",
        "neospectra",
        "micronir",
    ),
    "manure_organic_mineral": (
        "manure",
        "slurry",
        "dung",
        "compost",
        "digestate",
        "fertilizer",
        "organic_amendment",
        "livestock_waste",
    ),
    "liquid_food": (
        "beer",
        "wort",
        "brewing",
        "juice",
        "wine",
        "milk",
        "whey",
        "beverage",
        "liquid_food",
    ),
    "mineral_incombustible": (
        "mineral",
        "ore",
        "rock",
        "regolith",
        "sediment",
        "incombustible",
        "ash",
        "inorganic",
        "soil_inorganic",
    ),
    "wavenumber_domain": (
        "wavenumber",
        "cm-1",
        "cm_inv",
        "ftir",
        "mid_ir",
        "mid-infrared",
        "nir_wavenumber",
        "colza_wavenumber",
    ),
}

DIESEL_RESERVED_TOKENS: tuple[str, ...] = (
    "diesel",
    "btex",
    "petro",
    "gasoline",
    "fuel",
    "hydrocarbon",
    "alkane",
    "aromatic",
    "crude_oil",
    "kerosene",
)

REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = (
    "regime_family",
    "source",
    "task",
    "database_name",
    "dataset",
    "axis_unit",
    "axis_min_value",
    "axis_max_value",
    "n_features_after_alignment",
    "n_train_rows",
    "n_test_rows",
)

FAMILY_REQUIRED_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "plant_leaf_visnir_swir": (
        "preprocessing_status_documented_source",
        "preprocessing_status_value",
        "instrument_class",
    ),
    "manure_organic_mineral": (
        "preprocessing_status_documented_source",
        "preprocessing_status_value",
        "acquisition_geometry_documented_source",
        "acquisition_geometry_kind",
    ),
    "liquid_food": (
        "preprocessing_status_documented_source",
        "preprocessing_status_value",
        "pathlength_documented_source",
        "pathlength_mm",
        "temperature_documented_source",
        "temperature_field_value_or_range",
        "batch_documented_source",
        "batch_field_or_descriptor",
    ),
    "mineral_incombustible": (
        "preprocessing_status_documented_source",
        "preprocessing_status_value",
        "acquisition_geometry_documented_source",
        "acquisition_geometry_kind",
        "instrument_class",
    ),
    "wavenumber_domain": (
        "preprocessing_status_documented_source",
        "preprocessing_status_value",
        "axis_unit_documented_source",
        "axis_direction_documented_source",
        "axis_conversion_contract_source",
        "panel_breadth_documented_sources",
    ),
}

MANURE_GEOMETRY_DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "cup_diameter_mm",
    "sample_thickness_mm",
    "presentation_mode",
    "instrument_class",
)

PREPROCESSING_STATUS_ENUM: frozenset[str] = frozenset({
    "raw",
    "absorbance",
    "reflectance",
    "transmittance",
    "derivative",
    "snv",
    "msc",
    "corrected_other",
    "unknown",
})
TARGET_SENTINEL_SEMANTICS_ENUM: frozenset[str] = frozenset({
    "not_measured",
    "out_of_range",
    "quality_flag",
    "placeholder",
    "other",
})
NEGATIVE_X_SEMANTICS_ENUM: frozenset[str] = frozenset({
    "baseline_subtracted",
    "derivative_output",
    "instrument_offset",
    "corrected_other",
    "raw_with_offset",
})
ACQUISITION_GEOMETRY_KIND_ENUM: frozenset[str] = frozenset({
    "row_bound_real_metadata",
    "real_cohort_metadata_header",
    "documented_constant",
    "generic",
})
AXIS_UNIT_ENUM: frozenset[str] = frozenset({"nm", "cm-1", "unknown"})

NUMERIC_GEOMETRY_FIELDS: tuple[str, ...] = (
    "pathlength_mm",
    "cup_diameter_mm",
    "sample_thickness_mm",
)

SINGLE_SOURCE_FIELDS: tuple[str, ...] = (
    "preprocessing_status_documented_source",
    "target_sentinel_value_documented_source",
    "negative_x_semantics_documented_source",
    "acquisition_geometry_documented_source",
    "bulk_packing_documentation_source",
    "pathlength_documented_source",
    "temperature_documented_source",
    "batch_documented_source",
    "axis_unit_documented_source",
    "axis_direction_documented_source",
    "axis_conversion_contract_source",
)
MULTI_SOURCE_FIELDS: tuple[str, ...] = ("panel_breadth_documented_sources",)

REMOTE_IDENTIFIER_PREFIXES: tuple[str, ...] = (
    "http://",
    "https://",
    "ftp://",
    "doi:",
    "arxiv:",
    "urn:",
)

LEAKAGE_FIELDS_REJECTED: tuple[str, ...] = (
    "label",
    "labels",
    "target",
    "targets",
    "target_value",
    "class",
    "class_label",
    "split",
    "splits",
    "fold",
    "metric",
    "metrics",
    "score",
    "auc",
    "auroc",
    "roc_auc",
    "adversarial",
    "adversarial_score",
    "adversarial_metric",
    "adversarial_auc",
    "transfer",
    "transfer_score",
    "transfer_metric",
    "transfer_auc",
    "downstream",
    "downstream_metric",
    "downstream_score",
    "downstream_auc",
    "downstream_feedback",
    "performance_metric",
    "validation_score",
    "test_score",
    "train_score",
    "threshold",
    "gate_threshold",
    "gate",
    "pca",
    "covariance",
    "noise_capture",
    "ml_model",
    "dl_model",
    "calibration",
    "profile",
    "promotion",
)

READY_PREFIX = "ready_for_phase_m3_mechanism_design_"
BLOCKED_PREFIX_HEAD = "blocked_pending_"
BLOCKED_PREFIX_TAIL = "_evidence_no_stats_ml"


@dataclass(frozen=True)
class PreflightRow:
    status: str
    source_kind: str
    manifest_path: str
    row_index: int
    regime_family: str
    source: str
    task: str
    database_name: str
    dataset: str
    axis_unit: str
    axis_min_value: float | None
    axis_max_value: float | None
    n_features_after_alignment: int
    n_train_rows: int
    n_test_rows: int
    preprocessing_status_value: str
    matched_family_identity_tokens: str
    cross_family_tokens: str
    reserved_diesel_tokens: str
    missing_evidence_fields: str
    rejected_leakage_fields: str
    enum_failures: str
    numeric_failures: str
    unresolved_documented_sources: str
    recommendation_signal: str
    audit_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    return "" if value is None else str(value).strip()


def _parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _identity_text(record: dict[str, Any]) -> str:
    parts = [_record_value(record, field) for field in ("source", "task", "database_name", "dataset", "regime_family")]
    return " ".join(part for part in parts if part).casefold()


def _non_regime_family_text(record: dict[str, Any]) -> str:
    parts = [_record_value(record, field) for field in ("source", "task", "database_name", "dataset")]
    return " ".join(part for part in parts if part).casefold()


def _matched_family_tokens(text: str, tokens: Iterable[str]) -> list[str]:
    matched: list[str] = []
    for token in tokens:
        token_folded = token.casefold()
        if token_folded in text:
            matched.append(token_folded)
    return matched


def _rejected_leakage_fields(record: dict[str, Any]) -> list[str]:
    folded_keys = {str(key).strip().casefold(): key for key in record}
    rejected: list[str] = []
    for field in LEAKAGE_FIELDS_REJECTED:
        original = folded_keys.get(field)
        if original is None:
            continue
        if str(record.get(original, "")).strip() == "":
            continue
        rejected.append(field)
    return sorted(set(rejected))


def _missing_required_identity(record: dict[str, Any]) -> list[str]:
    return [field for field in REQUIRED_IDENTITY_FIELDS if not _record_value(record, field)]


def _is_remote_identifier(value: str) -> bool:
    folded = value.strip().casefold()
    return any(folded.startswith(prefix) for prefix in REMOTE_IDENTIFIER_PREFIXES)


def _resolve_source_path(value: str, root: Path) -> bool:
    text = value.strip()
    if not text:
        return True
    if _is_remote_identifier(text):
        return True
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.exists()


def _validate_enum_values(record: dict[str, Any], family: str) -> list[str]:
    failures: list[str] = []
    axis_unit = _record_value(record, "axis_unit").casefold()
    if axis_unit and axis_unit not in AXIS_UNIT_ENUM:
        failures.append(f"axis_unit={axis_unit}")
    pp = _record_value(record, "preprocessing_status_value").casefold()
    if pp and pp not in PREPROCESSING_STATUS_ENUM:
        failures.append(f"preprocessing_status_value={pp}")
    sentinel_value = _record_value(record, "target_sentinel_semantics_value").casefold()
    if sentinel_value and sentinel_value not in TARGET_SENTINEL_SEMANTICS_ENUM:
        failures.append(f"target_sentinel_semantics_value={sentinel_value}")
    neg_value = _record_value(record, "negative_x_semantics_value").casefold()
    if neg_value and neg_value not in NEGATIVE_X_SEMANTICS_ENUM:
        failures.append(f"negative_x_semantics_value={neg_value}")
    if family in {"manure_organic_mineral", "mineral_incombustible"}:
        kind = _record_value(record, "acquisition_geometry_kind").casefold()
        if kind and kind not in ACQUISITION_GEOMETRY_KIND_ENUM:
            failures.append(f"acquisition_geometry_kind={kind}")
    return failures


def _validate_numerics(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    axis_min = _parse_float(record.get("axis_min_value"))
    axis_max = _parse_float(record.get("axis_max_value"))
    n_features = _parse_int(record.get("n_features_after_alignment"))
    n_train = _parse_int(record.get("n_train_rows"))
    n_test = _parse_int(record.get("n_test_rows"))
    if axis_min is None:
        failures.append("axis_min_value_not_numeric")
    if axis_max is None:
        failures.append("axis_max_value_not_numeric")
    if axis_min is not None and axis_max is not None and axis_min >= axis_max:
        failures.append("axis_min_value_not_less_than_axis_max_value")
    if n_features is None or n_features <= 0:
        failures.append("n_features_after_alignment_not_positive_int")
    if n_train is None or n_train < 0:
        failures.append("n_train_rows_not_nonneg_int")
    if n_test is None or n_test < 0:
        failures.append("n_test_rows_not_nonneg_int")
    for field in NUMERIC_GEOMETRY_FIELDS:
        raw = _record_value(record, field)
        if not raw:
            continue
        if _parse_float(raw) is None:
            failures.append(f"{field}_not_numeric")
    return failures


def _validate_documented_sources(record: dict[str, Any], root: Path) -> list[str]:
    unresolved: list[str] = []
    for field in SINGLE_SOURCE_FIELDS:
        value = _record_value(record, field)
        if not value:
            continue
        if not _resolve_source_path(value, root):
            unresolved.append(f"{field}={value}")
    for field in MULTI_SOURCE_FIELDS:
        value = _record_value(record, field)
        if not value:
            continue
        for token in [item.strip() for item in value.split(",") if item.strip()]:
            if not _resolve_source_path(token, root):
                unresolved.append(f"{field}={token}")
    return unresolved


def _per_family_extra_block(record: dict[str, Any], family: str) -> list[str]:
    extras: list[str] = []
    if family == "manure_organic_mineral":
        descriptors_present = [field for field in MANURE_GEOMETRY_DESCRIPTOR_FIELDS if _record_value(record, field)]
        if not descriptors_present:
            extras.append("one_of_" + ";".join(MANURE_GEOMETRY_DESCRIPTOR_FIELDS))
        kind = _record_value(record, "acquisition_geometry_kind").casefold()
        if kind == "documented_constant" and not _record_value(record, "bulk_packing_documentation_source"):
            extras.append("bulk_packing_documentation_source")
    if family in {"manure_organic_mineral", "mineral_incombustible"}:
        kind = _record_value(record, "acquisition_geometry_kind").casefold()
        if kind == "generic":
            extras.append("acquisition_geometry_kind_generic_does_not_close_gap")
    if family == "wavenumber_domain":
        sources = _record_value(record, "panel_breadth_documented_sources")
        tokens = [item.strip() for item in sources.split(",") if item.strip()]
        if len(tokens) < 2:
            extras.append("panel_breadth_documented_sources_lt_2_distinct_sources")
    sentinel_source = _record_value(record, "target_sentinel_value_documented_source")
    sentinel_value = _record_value(record, "target_sentinel_semantics_value")
    if bool(sentinel_source) != bool(sentinel_value):
        extras.append("target_sentinel_pair_inconsistent_source_and_value")
    neg_source = _record_value(record, "negative_x_semantics_documented_source")
    neg_value = _record_value(record, "negative_x_semantics_value")
    if bool(neg_source) != bool(neg_value):
        extras.append("negative_x_pair_inconsistent_source_and_value")
    return extras


def _row_kind(record: dict[str, Any], default: str) -> str:
    value = _record_value(record, "source_kind") or _record_value(record, "manifest_kind")
    return value or default


def _format_blocked_status(family: str) -> str:
    return f"{BLOCKED_PREFIX_HEAD}{family}{BLOCKED_PREFIX_TAIL}"


def _format_ready_status(family: str) -> str:
    return f"{READY_PREFIX}{family}"


def _format_accepted(family: str) -> str:
    return f"accepted_for_{family}"


def _build_row(
    *,
    record: dict[str, Any],
    status: str,
    source_kind: str,
    manifest_path: str,
    row_index: int,
    matched_tokens: Sequence[str] = (),
    cross_family_tokens: Sequence[str] = (),
    reserved_diesel_tokens: Sequence[str] = (),
    missing_evidence_fields: Sequence[str] = (),
    rejected_leakage_fields: Sequence[str] = (),
    enum_failures: Sequence[str] = (),
    numeric_failures: Sequence[str] = (),
    unresolved_documented_sources: Sequence[str] = (),
    recommendation_signal: str,
) -> PreflightRow:
    return PreflightRow(
        status=status,
        source_kind=source_kind,
        manifest_path=manifest_path,
        row_index=row_index,
        regime_family=_record_value(record, "regime_family"),
        source=_record_value(record, "source"),
        task=_record_value(record, "task"),
        database_name=_record_value(record, "database_name"),
        dataset=_record_value(record, "dataset"),
        axis_unit=_record_value(record, "axis_unit"),
        axis_min_value=_parse_float(record.get("axis_min_value")),
        axis_max_value=_parse_float(record.get("axis_max_value")),
        n_features_after_alignment=_parse_int(record.get("n_features_after_alignment")) or 0,
        n_train_rows=_parse_int(record.get("n_train_rows")) or 0,
        n_test_rows=_parse_int(record.get("n_test_rows")) or 0,
        preprocessing_status_value=_record_value(record, "preprocessing_status_value"),
        matched_family_identity_tokens=";".join(matched_tokens),
        cross_family_tokens=";".join(cross_family_tokens),
        reserved_diesel_tokens=";".join(reserved_diesel_tokens),
        missing_evidence_fields=";".join(missing_evidence_fields),
        rejected_leakage_fields=";".join(rejected_leakage_fields),
        enum_failures=";".join(enum_failures),
        numeric_failures=";".join(numeric_failures),
        unresolved_documented_sources=";".join(unresolved_documented_sources),
        recommendation_signal=recommendation_signal,
        audit_scope=EXP31_AUDIT_SCOPE,
    )


def _process_record(
    *,
    record: dict[str, Any],
    root: Path,
    manifest_path: str,
    row_index: int,
    source_kind: str,
    regime_family_filter: frozenset[str],
) -> PreflightRow:
    leakage = _rejected_leakage_fields(record)
    if leakage:
        return _build_row(
            record=record,
            status="rejected_leakage_fields",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            rejected_leakage_fields=leakage,
            recommendation_signal="rejected_no_label_target_split_downstream_adversarial_auc_or_metric_inputs",
        )

    missing_identity = _missing_required_identity(record)
    if missing_identity:
        return _build_row(
            record=record,
            status="blocked_missing_identity",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            missing_evidence_fields=missing_identity,
            recommendation_signal="missing_required_identity_fields",
        )

    family = _record_value(record, "regime_family").casefold()
    if family not in REGIME_FAMILIES:
        return _build_row(
            record=record,
            status="blocked_unknown_regime_family",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            recommendation_signal="unknown_regime_family_value",
        )

    if regime_family_filter and family not in regime_family_filter:
        return _build_row(
            record=record,
            status="filtered_out",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            recommendation_signal=f"filtered_out_not_in_regime_family_filter_{family}",
        )

    identity_text = _identity_text(record)
    matched_own = _matched_family_tokens(identity_text, FAMILY_TOKENS[family])
    if not matched_own:
        return _build_row(
            record=record,
            status="blocked_missing_identity_token_for_family",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            recommendation_signal=f"no_identity_token_match_for_{family}",
        )
    own_token_set = {token.casefold() for token in FAMILY_TOKENS[family]}
    non_family_text = _non_regime_family_text(record)
    cross_tokens: list[str] = []
    for other_family, other_tokens in FAMILY_TOKENS.items():
        if other_family == family:
            continue
        for token in other_tokens:
            folded = token.casefold()
            if folded in own_token_set:
                continue
            if folded in non_family_text:
                cross_tokens.append(f"{other_family}:{folded}")
    if cross_tokens:
        return _build_row(
            record=record,
            status="blocked_cross_family_identity_token",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            cross_family_tokens=sorted(set(cross_tokens)),
            recommendation_signal="cross_family_identity_tokens_present",
        )

    diesel_hits = _matched_family_tokens(identity_text, DIESEL_RESERVED_TOKENS)
    if diesel_hits:
        return _build_row(
            record=record,
            status="blocked_reserved_diesel_fuel_token",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            reserved_diesel_tokens=diesel_hits,
            recommendation_signal="reserved_diesel_fuel_token_use_doc_12_contract",
        )

    missing_evidence = [field for field in FAMILY_REQUIRED_EVIDENCE_FIELDS[family] if not _record_value(record, field)]
    extras = _per_family_extra_block(record, family)
    combined_missing = missing_evidence + extras
    if combined_missing:
        return _build_row(
            record=record,
            status=f"blocked_missing_{family}_evidence_no_stats_ml",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            missing_evidence_fields=combined_missing,
            recommendation_signal=f"missing_{family}_evidence_no_stats_ml",
        )

    enum_failures = _validate_enum_values(record, family)
    if enum_failures:
        return _build_row(
            record=record,
            status="blocked_invalid_enum_value",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            enum_failures=enum_failures,
            recommendation_signal="invalid_enum_value",
        )

    numeric_failures = _validate_numerics(record)
    if numeric_failures:
        return _build_row(
            record=record,
            status="blocked_invalid_numeric_field",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            numeric_failures=numeric_failures,
            recommendation_signal="invalid_numeric_field",
        )

    unresolved = _validate_documented_sources(record, root)
    if unresolved:
        return _build_row(
            record=record,
            status="blocked_documented_source_not_found",
            source_kind=source_kind,
            manifest_path=manifest_path,
            row_index=row_index,
            matched_tokens=matched_own,
            unresolved_documented_sources=unresolved,
            recommendation_signal="documented_source_path_not_resolvable",
        )

    return _build_row(
        record=record,
        status=_format_accepted(family),
        source_kind=source_kind,
        manifest_path=manifest_path,
        row_index=row_index,
        matched_tokens=matched_own,
        recommendation_signal=f"accepted_for_{family}",
    )


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("records", None))
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return [payload]
    return []


def read_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix.casefold() == ".json":
        return _read_json_records(path), "json"
    return _read_csv_records(path), "csv"


def _coerce_filter(values: Iterable[str] | None) -> frozenset[str]:
    if not values:
        return frozenset()
    return frozenset(value.strip().casefold() for value in values if value and value.strip())


def run_preflight(
    root: Path,
    manifest: Path | None = None,
    regime_family_filter: Iterable[str] | None = None,
) -> dict[str, Any]:
    family_filter = _coerce_filter(regime_family_filter)
    unknown_filters = sorted(value for value in family_filter if value not in REGIME_FAMILIES)
    effective_filter = frozenset(value for value in family_filter if value in REGIME_FAMILIES)

    if manifest is None:
        rows: list[PreflightRow] = []
        manifest_path_str = ""
        source_kind = "none"
    else:
        records, source_kind = read_manifest(manifest)
        manifest_path_str = str(manifest)
        rows = [
            _process_record(
                record=record,
                root=root,
                manifest_path=manifest_path_str,
                row_index=index,
                source_kind=source_kind,
                regime_family_filter=effective_filter,
            )
            for index, record in enumerate(records)
        ]

    per_family_decisions: dict[str, str] = {}
    per_family_accepted: dict[str, list[PreflightRow]] = {}
    per_family_blocked_by_status: dict[str, dict[str, int]] = {}
    per_family_missing_field_counts: dict[str, dict[str, int]] = {}
    for family in REGIME_FAMILIES:
        accepted = [row for row in rows if row.status == _format_accepted(family)]
        per_family_accepted[family] = accepted
        if effective_filter and family not in effective_filter:
            per_family_decisions[family] = _format_blocked_status(family)
        elif accepted:
            per_family_decisions[family] = _format_ready_status(family)
        else:
            per_family_decisions[family] = _format_blocked_status(family)
        blocked_status_counts: dict[str, int] = {}
        missing_counts: dict[str, int] = {}
        for row in rows:
            if row.status == _format_accepted(family):
                continue
            if row.regime_family.strip().casefold() != family:
                continue
            blocked_status_counts[row.status] = blocked_status_counts.get(row.status, 0) + 1
            if row.missing_evidence_fields:
                for field in row.missing_evidence_fields.split(";"):
                    if field:
                        missing_counts[field] = missing_counts.get(field, 0) + 1
        per_family_blocked_by_status[family] = blocked_status_counts
        per_family_missing_field_counts[family] = missing_counts

    leakage_rows = [row for row in rows if row.status == "rejected_leakage_fields"]
    return {
        "status": "done",
        "rows": rows,
        "per_family_decisions": per_family_decisions,
        "per_family_accepted": per_family_accepted,
        "per_family_blocked_by_status": per_family_blocked_by_status,
        "per_family_missing_field_counts": per_family_missing_field_counts,
        "leakage_rows": leakage_rows,
        "regime_family_filter": sorted(effective_filter),
        "unknown_regime_family_filter_values": unknown_filters,
        "manifest_source_kind": source_kind,
    }


def write_csv(rows: list[PreflightRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(PreflightRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _fmt_optional(value: float | None, spec: str = "g") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def _top_three_missing(counts: dict[str, int]) -> str:
    if not counts:
        return "n/a"
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return ", ".join(f"{name} ({count})" for name, count in items)


def render_markdown(
    result: dict[str, Any],
    *,
    report_path: Path,
    csv_path: Path | None,
    manifest: Path | None,
) -> str:
    rows = list(result["rows"])
    per_family_decisions: dict[str, str] = dict(result["per_family_decisions"])
    per_family_accepted: dict[str, list[PreflightRow]] = dict(result["per_family_accepted"])
    per_family_blocked_by_status: dict[str, dict[str, int]] = dict(result["per_family_blocked_by_status"])
    per_family_missing_field_counts: dict[str, dict[str, int]] = dict(result["per_family_missing_field_counts"])
    leakage_rows = list(result["leakage_rows"])
    regime_family_filter = list(result["regime_family_filter"])
    unknown_filters = list(result["unknown_regime_family_filter_values"])

    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    manifest_line = f"- manifest: `{manifest}`" if manifest is not None else "- manifest: `none`"
    filter_line = (
        f"- regime_family_filter: `{';'.join(regime_family_filter)}`"
        if regime_family_filter
        else "- regime_family_filter: `none`"
    )
    unknown_filter_line = (
        f"- unknown_regime_family_filter_values: `{';'.join(unknown_filters)}`"
        if unknown_filters
        else "- unknown_regime_family_filter_values: `none`"
    )
    not_inspected = sum(1 for row in rows if row.status == "blocked_missing_identity")

    lines: list[str] = [
        "# exp31 Multidataset Evidence-Gap Manifest Preflight",
        "",
        f"- audit_scope: `{EXP31_AUDIT_SCOPE}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        manifest_line,
        f"- report: `{report_path}`",
        csv_line,
        f"- manifest_source_kind: `{result['manifest_source_kind']}`",
        filter_line,
        unknown_filter_line,
        f"- rows_checked: `{len(rows)}`",
        f"- not_inspected_rows: `{not_inspected}`",
        f"- leakage_rejected_rows: `{len(leakage_rows)}`",
        "",
        "## Per-Family Recommendations",
        "",
    ]
    for family in REGIME_FAMILIES:
        lines.append(f"- `{family}`: `{per_family_decisions[family]}`")

    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Bench-only Phase M2 preflight: descriptive checks against schema 16 only, no realism claim.",
            "- No spectral file content is read. No `nirs4all/` module is required.",
            "- Explicit no statistics, no PCA, no covariance, no quantile/marginal/noise capture, no calibration, no ML, no DL.",
            "- Forbidden inputs: no labels, no targets, no splits, no downstream metrics, no adversarial AUC, no transfer scores.",
            "- No aggregate cross-family score, threshold, gate, or parameter is emitted.",
            "- A `ready_for_phase_m3_mechanism_design_<family>` decision authorizes only writing a per-family Phase M3 mechanism design document, never a generator/profile/gate/promotion.",
            "- `nirs4all/` is not modified or required for this preflight.",
            "",
            "## Per-Family Summary",
            "",
            "| family | accepted rows | blocked-by-status counts | missing-evidence top three | recommendation |",
            "|---|---:|---|---|---|",
        ]
    )
    for family in REGIME_FAMILIES:
        accepted_count = len(per_family_accepted[family])
        blocked_counts = per_family_blocked_by_status[family]
        blocked_label = (
            ", ".join(f"{status} ({count})" for status, count in sorted(blocked_counts.items())) if blocked_counts else "n/a"
        )
        lines.append(
            f"| `{family}` | `{accepted_count}` | {blocked_label} | "
            f"{_top_three_missing(per_family_missing_field_counts[family])} | "
            f"`{per_family_decisions[family]}` |"
        )

    lines.extend(
        [
            "",
            "## Per-Row Detail",
            "",
            "| index | status | regime_family | source/task/database_name/dataset | missing evidence | leakage rejected | unresolved sources |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        identity = f"{row.source}/{row.task}/{row.database_name}/{row.dataset}".strip("/")
        lines.append(
            f"| `{row.row_index}` | `{row.status}` | `{row.regime_family or 'n/a'}` | `{identity or 'n/a'}` | "
            f"`{row.missing_evidence_fields or 'n/a'}` | `{row.rejected_leakage_fields or 'n/a'}` | "
            f"`{row.unresolved_documented_sources or 'n/a'}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp31_multidataset_evidence_gap_manifest_preflight.py \\",
            "  --root . \\",
            f"  --manifest {manifest if manifest is not None else '/path/to/manifest.csv'} \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path if csv_path is not None else 'bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv'}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--regime-family", action="append", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    result = run_preflight(args.root, args.manifest, regime_family_filter=args.regime_family)
    rows = list(result["rows"])
    if args.csv is not None:
        write_csv(rows, args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv, manifest=args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    print(
        {
            "rows": len(rows),
            "status": result["status"],
            "per_family_decisions": result["per_family_decisions"],
        }
    )


if __name__ == "__main__":
    main()
