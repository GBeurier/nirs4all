"""P2-08 data-support manifest preflight.

Bench-only contract check for resuming uncalibrated mechanistic audit design.
This preflight reads optional CSV/JSON manifests, or existing AOM cohort files
when no manifest is supplied, and reports whether the required data support is
present. It does not build spectra, calibrate, fit statistics/PCA/noise,
train ML/DL models, add a profile, or change gates/metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

EXP29_AUDIT_SCOPE = "bench_only_p2_08_data_support_manifest_preflight"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM = 750.0
SUPPORT_HIGH_NM = 1550.0
READY_RECOMMENDATION = "ready_for_mechanistic_audit_design"
BLOCKED_RECOMMENDATION = "blocked_pending_manifest_data_support_no_stats_ml"
DEFAULT_REPORT = Path("/tmp/exp29_data_support_manifest_preflight.md")
DEFAULT_CSV = Path("/tmp/exp29_data_support_manifest_preflight.csv")

FUEL_TOKENS: tuple[str, ...] = (
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
PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("train_path", "train"),
    ("xtrain_path", "train"),
    ("test_path", "test"),
    ("xtest_path", "test"),
    ("x_path", "single"),
    ("spectral_path", "single"),
    ("spectrum_path", "single"),
)
WAVELENGTH_FIELDS: tuple[str, ...] = (
    "wavelengths",
    "wavelength_nm",
    "wavelength_headers",
    "spectral_headers",
    "aligned_wavelengths_nm",
)
WAVELENGTH_MIN_FIELDS: tuple[str, ...] = (
    "wavelength_min",
    "wavelength_min_nm",
    "min_wavelength",
    "min_wavelength_nm",
)
WAVELENGTH_MAX_FIELDS: tuple[str, ...] = (
    "wavelength_max",
    "wavelength_max_nm",
    "max_wavelength",
    "max_wavelength_nm",
)
WAVELENGTH_COUNT_FIELDS: tuple[str, ...] = (
    "n_wavelengths_after_alignment",
    "n_wavelengths",
    "wavelength_count",
    "wavelength_count_after_alignment",
)
SUPPORT_COUNT_FIELDS: tuple[str, ...] = (
    "support_count_after_alignment",
    "support_wavelength_count_after_alignment",
)
OFF_SUPPORT_COUNT_FIELDS: tuple[str, ...] = (
    "off_support_count_after_alignment",
    "off_support_wavelength_count_after_alignment",
)
GEOMETRY_NUMERIC_FIELDS: tuple[str, ...] = (
    "source_detector_distance_mm",
    "source_detector_mm",
    "source_detector_distance_cm",
    "pathlength_mm",
    "path_length_mm",
    "optical_path_mm",
    "collection_angle_deg",
    "illumination_angle_deg",
    "incidence_angle_deg",
)
GEOMETRY_TEXT_FIELDS: tuple[str, ...] = (
    "collection_geometry",
    "measurement_geometry",
    "geometry_description",
)
BINDING_FIELDS_REQUIRED: tuple[str, ...] = (
    "source",
    "task",
    "database_name",
    "dataset",
    "row_binding_key",
    "metadata_source",
)
WIDER_SUPPORT_FIELDS_REQUIRED: tuple[str, ...] = (
    "source",
    "task",
    "database_name",
    "dataset",
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


@dataclass(frozen=True)
class WavelengthSupport:
    wavelength_min: float | None
    wavelength_max: float | None
    n_wavelengths_after_alignment: int
    support_count_after_alignment: int
    off_support_count_after_alignment: int
    extends_outside_750_1550_after_alignment: bool
    spectral_path_fields: str
    evidence: str


@dataclass(frozen=True)
class PreflightRow:
    status: str
    source_kind: str
    manifest_path: str
    row_index: int
    source: str
    task: str
    database_name: str
    dataset: str
    spectral_path_fields: str
    wavelength_min: float | None
    wavelength_max: float | None
    n_wavelengths_after_alignment: int
    support_low_nm: float
    support_high_nm: float
    support_count_after_alignment: int
    off_support_count_after_alignment: int
    extends_outside_750_1550_after_alignment: bool
    row_binding_key: str
    metadata_source: str
    binding_fields_required: str
    binding_fields_missing: str
    row_bound_geometry_metadata_present: bool
    generic_geometry_present: bool
    geometry_metadata_kind: str
    parsed_geometry_fields: str
    rejected_leakage_fields: str
    recommendation_signal: str
    evidence: str
    audit_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contains_fuel_token(record: dict[str, Any]) -> bool:
    haystack = " ".join(str(value) for value in record.values()).casefold()
    return any(token in haystack for token in FUEL_TOKENS)


def _record_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    return "" if value is None else str(value).strip()


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "row_bound", "real_row_bound"}


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


def _parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            parsed = _parse_float(item)
            if parsed is None:
                return []
            out.append(parsed)
        return out
    text = str(value).strip()
    if not text:
        return []
    tokens = [token for token in re.split(r"[;,\t ]+", text) if token]
    out = []
    for token in tokens:
        parsed = _parse_float(token)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def _parse_numeric_header(path: Path) -> list[float]:
    if not path.exists():
        return []
    first = path.open("r", encoding="utf-8", errors="ignore").readline().strip()
    if not first:
        return []
    candidates = [first.split(";"), first.split(","), first.split("\t")]
    tokens = max(candidates, key=len)
    out = []
    for token in tokens:
        parsed = _parse_float(token)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def _support_counts(wavelengths: list[float]) -> tuple[int, int, bool]:
    support = sum(SUPPORT_LOW_NM <= wl <= SUPPORT_HIGH_NM for wl in wavelengths)
    off_support = len(wavelengths) - support
    return support, off_support, off_support > 0


def _first_float(record: dict[str, Any], fields_to_try: tuple[str, ...]) -> tuple[float | None, str]:
    for field in fields_to_try:
        parsed = _parse_float(record.get(field))
        if parsed is not None:
            return parsed, field
    return None, ""


def _first_int(record: dict[str, Any], fields_to_try: tuple[str, ...]) -> tuple[int | None, str]:
    for field in fields_to_try:
        parsed = _parse_int(record.get(field))
        if parsed is not None:
            return parsed, field
    return None, ""


def _support_from_wavelengths(wavelengths: list[float], spectral_path_fields: str, evidence: str) -> WavelengthSupport:
    support_count, off_support_count, extends = _support_counts(wavelengths)
    return WavelengthSupport(
        wavelength_min=min(wavelengths) if wavelengths else None,
        wavelength_max=max(wavelengths) if wavelengths else None,
        n_wavelengths_after_alignment=len(wavelengths),
        support_count_after_alignment=support_count,
        off_support_count_after_alignment=off_support_count,
        extends_outside_750_1550_after_alignment=extends,
        spectral_path_fields=spectral_path_fields,
        evidence=evidence,
    )


def _support_from_min_max_counts(record: dict[str, Any]) -> WavelengthSupport | None:
    wavelength_min, min_field = _first_float(record, WAVELENGTH_MIN_FIELDS)
    wavelength_max, max_field = _first_float(record, WAVELENGTH_MAX_FIELDS)
    n_wavelengths, count_field = _first_int(record, WAVELENGTH_COUNT_FIELDS)
    off_support_count, off_support_field = _first_int(record, OFF_SUPPORT_COUNT_FIELDS)
    support_count, support_field = _first_int(record, SUPPORT_COUNT_FIELDS)
    if wavelength_min is None or wavelength_max is None or n_wavelengths is None:
        return None
    if wavelength_min > wavelength_max or n_wavelengths <= 0:
        return None
    if off_support_count is None and support_count is None:
        return None
    if off_support_count is None and support_count is not None:
        off_support_count = n_wavelengths - support_count
    if support_count is None and off_support_count is not None:
        support_count = n_wavelengths - off_support_count
    if support_count is None or off_support_count is None:
        return None
    if support_count < 0 or off_support_count < 0 or support_count + off_support_count != n_wavelengths:
        return None
    extends = wavelength_min < SUPPORT_LOW_NM or wavelength_max > SUPPORT_HIGH_NM
    if extends != (off_support_count > 0):
        return None
    fields_used = ";".join(field for field in (min_field, max_field, count_field, support_field, off_support_field) if field)
    return WavelengthSupport(
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        n_wavelengths_after_alignment=n_wavelengths,
        support_count_after_alignment=support_count,
        off_support_count_after_alignment=off_support_count,
        extends_outside_750_1550_after_alignment=extends,
        spectral_path_fields=fields_used,
        evidence="manifest_wavelength_min_max_count_fields",
    )


def _aligned_wavelengths_from_record(root: Path, record: dict[str, Any]) -> WavelengthSupport:
    for key in WAVELENGTH_FIELDS:
        wavelengths = _parse_float_list(record.get(key))
        if wavelengths:
            return _support_from_wavelengths(sorted(set(wavelengths)), key, "manifest_wavelength_field")

    summary = _support_from_min_max_counts(record)
    if summary is not None:
        return summary

    by_role: dict[str, list[float]] = {}
    used: list[str] = []
    for key, role in PATH_FIELDS:
        raw_path = _record_value(record, key)
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        wavelengths = _parse_numeric_header(path)
        if wavelengths:
            by_role[role] = wavelengths
            used.append(key)
    if not by_role:
        return _support_from_wavelengths([], "", "no_wavelength_source")
    merged: set[float] = set()
    for wavelengths in by_role.values():
        merged.update(wavelengths)
    return _support_from_wavelengths(sorted(merged), ";".join(used), "merged_train_test_or_single_header")


def _rejected_leakage_fields(record: dict[str, Any]) -> list[str]:
    keys = {str(key).strip().casefold() for key in record}
    return sorted(field for field in LEAKAGE_FIELDS_REJECTED if field in keys and str(record.get(field, "")).strip() != "")


def _binding_missing(record: dict[str, Any]) -> list[str]:
    return [field for field in BINDING_FIELDS_REQUIRED if not _record_value(record, field)]


def _wider_support_identity_missing(record: dict[str, Any]) -> list[str]:
    return [field for field in WIDER_SUPPORT_FIELDS_REQUIRED if not _record_value(record, field)]


def _contains_fuel_identity(record: dict[str, Any]) -> bool:
    haystack = " ".join(_record_value(record, field) for field in (*WIDER_SUPPORT_FIELDS_REQUIRED, "source_kind", "manifest_kind"))
    return any(token in haystack.casefold() for token in FUEL_TOKENS)


def _geometry_summary(record: dict[str, Any], missing_binding: list[str]) -> tuple[bool, bool, str, list[str]]:
    kind = _record_value(record, "geometry_metadata_kind") or _record_value(record, "geometry_scope")
    generic = kind.casefold() in {"generic", "generic_geometry", "generic_synthesis_model", "constant", "builder_constant"}
    generic = generic or _truthy(record.get("generic_geometry", ""))
    row_bound_declared = _truthy(record.get("real_row_bound", ""))
    kind_row_bound = kind.casefold() in {
        "real_row_bound",
        "real_cohort_metadata_header",
        "row_bound_real_metadata",
        "row_bound_real_cohort_metadata",
    }

    parsed_fields = [field for field in GEOMETRY_NUMERIC_FIELDS if _parse_float(record.get(field)) is not None]
    parsed_fields.extend(field for field in GEOMETRY_TEXT_FIELDS if _record_value(record, field))
    row_bound_geometry = bool(parsed_fields) and not generic and not missing_binding and (row_bound_declared or kind_row_bound)
    return row_bound_geometry, generic, kind, parsed_fields


def _source_kind(record: dict[str, Any], default: str) -> str:
    value = _record_value(record, "source_kind") or _record_value(record, "manifest_kind")
    if value:
        return value
    return default


def _preflight_record(root: Path, manifest_path: str, row_index: int, record: dict[str, Any], source_kind: str) -> PreflightRow:
    leakage = _rejected_leakage_fields(record)
    wavelength_support = _aligned_wavelengths_from_record(root, record)
    missing_binding = _binding_missing(record)
    missing_wider_identity = _wider_support_identity_missing(record)
    fuel_identity = _contains_fuel_identity(record)
    row_bound_geometry, generic_geometry, geometry_kind, parsed_geometry = _geometry_summary(record, missing_binding)
    ready_wider_support = (
        wavelength_support.extends_outside_750_1550_after_alignment and not missing_wider_identity and fuel_identity
    )
    ready_row_bound_geometry = row_bound_geometry and fuel_identity

    if leakage:
        status = "rejected_leakage_fields"
        ready_wider_support = False
        ready_row_bound_geometry = False
        row_bound_geometry = False
        signal = "rejected_no_label_target_split_downstream_adversarial_auc_or_metric_inputs"
    elif wavelength_support.extends_outside_750_1550_after_alignment and missing_wider_identity:
        status = "blocked_missing_manifest_identity_for_wider_support"
        signal = "missing_manifest_identity_for_wider_support"
    elif (wavelength_support.extends_outside_750_1550_after_alignment or row_bound_geometry) and not fuel_identity:
        status = "blocked_not_diesel_fuel_manifest_data_support"
        signal = "not_diesel_fuel_manifest_data_support"
        row_bound_geometry = False
    elif not wavelength_support.n_wavelengths_after_alignment and not ready_row_bound_geometry:
        status = "blocked_no_wavelength_or_row_bound_geometry_support"
        signal = "missing_manifest_data_support"
    else:
        status = "checked"
        if ready_wider_support:
            signal = "wider_real_support_available"
        elif ready_row_bound_geometry:
            signal = "row_bound_geometry_available"
        elif generic_geometry:
            signal = "generic_geometry_available_not_row_bound"
        else:
            signal = "inside_750_1550_without_row_bound_geometry"

    return PreflightRow(
        status=status,
        source_kind=source_kind,
        manifest_path=manifest_path,
        row_index=row_index,
        source=_record_value(record, "source"),
        task=_record_value(record, "task"),
        database_name=_record_value(record, "database_name"),
        dataset=_record_value(record, "dataset"),
        spectral_path_fields=wavelength_support.spectral_path_fields,
        wavelength_min=wavelength_support.wavelength_min,
        wavelength_max=wavelength_support.wavelength_max,
        n_wavelengths_after_alignment=wavelength_support.n_wavelengths_after_alignment,
        support_low_nm=SUPPORT_LOW_NM,
        support_high_nm=SUPPORT_HIGH_NM,
        support_count_after_alignment=0 if leakage else wavelength_support.support_count_after_alignment,
        off_support_count_after_alignment=0 if leakage else wavelength_support.off_support_count_after_alignment,
        extends_outside_750_1550_after_alignment=ready_wider_support,
        row_binding_key=_record_value(record, "row_binding_key"),
        metadata_source=_record_value(record, "metadata_source"),
        binding_fields_required=";".join(BINDING_FIELDS_REQUIRED),
        binding_fields_missing=";".join(missing_binding),
        row_bound_geometry_metadata_present=ready_row_bound_geometry,
        generic_geometry_present=generic_geometry,
        geometry_metadata_kind=geometry_kind,
        parsed_geometry_fields=";".join(parsed_geometry),
        rejected_leakage_fields=";".join(leakage),
        recommendation_signal=signal,
        evidence=wavelength_support.evidence,
        audit_scope=EXP29_AUDIT_SCOPE,
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


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".json":
        return _read_json_records(path)
    return _read_csv_records(path)


def _default_aom_records(root: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for path, task in (
        (root / "bench/AOM_v0/benchmarks/cohort_regression.csv", "regression"),
        (root / "bench/AOM_v0/benchmarks/cohort_classification.csv", "classification"),
    ):
        if not path.exists():
            continue
        for record in _read_csv_records(path):
            if not _contains_fuel_token(record):
                continue
            records.append((str(path), {**record, "source": path.stem, "task": task, "source_kind": "existing_aom_cohort"}))
    return records


def run_preflight(root: Path, manifest: Path | None = None) -> dict[str, Any]:
    if manifest is None:
        record_sources = _default_aom_records(root)
    else:
        records = read_manifest(manifest)
        record_sources = [(str(manifest), {**record, "source_kind": _source_kind(record, "manifest")}) for record in records]

    rows = [
        _preflight_record(
            root=root,
            manifest_path=manifest_path,
            row_index=index,
            record=record,
            source_kind=_source_kind(record, "manifest"),
        )
        for index, (manifest_path, record) in enumerate(record_sources)
    ]
    accepted_rows = [row for row in rows if row.status != "rejected_leakage_fields"]
    wider_rows = [row for row in accepted_rows if row.extends_outside_750_1550_after_alignment]
    row_bound_geometry_rows = [row for row in accepted_rows if row.row_bound_geometry_metadata_present]
    generic_geometry_rows = [row for row in accepted_rows if row.generic_geometry_present and not row.row_bound_geometry_metadata_present]
    leakage_rows = [row for row in rows if row.rejected_leakage_fields]
    recommendation = READY_RECOMMENDATION if wider_rows or row_bound_geometry_rows else BLOCKED_RECOMMENDATION
    return {
        "status": "done",
        "rows": rows,
        "accepted_rows": accepted_rows,
        "wider_real_support_rows": wider_rows,
        "row_bound_geometry_rows": row_bound_geometry_rows,
        "generic_geometry_rows": generic_geometry_rows,
        "leakage_rows": leakage_rows,
        "recommendation": recommendation,
    }


def write_csv(rows: list[PreflightRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(PreflightRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None, manifest: Path | None) -> str:
    rows = list(result["rows"])
    wider_rows = list(result["wider_real_support_rows"])
    row_bound_geometry_rows = list(result["row_bound_geometry_rows"])
    generic_geometry_rows = list(result["generic_geometry_rows"])
    leakage_rows = list(result["leakage_rows"])
    recommendation = str(result["recommendation"])
    source = str(manifest) if manifest is not None else "existing AOM cohort files"
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"

    lines = [
        "# P2-08 Data Support Manifest Preflight",
        "",
        f"- audit_scope: `{EXP29_AUDIT_SCOPE}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        f"- source: `{source}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- rows_checked: `{len(rows)}`",
        f"- recommendation: `{recommendation}`",
        "",
        "## Contract",
        "",
        "- Bench-only manifest/cohort preflight; no generator profile, no mechanism, no gate, no promotion, and no metric or threshold change.",
        "- Explicit no stats, no PCA, no covariance, no quantile/marginal/noise capture, no calibration, no ML, no DL, no labels, no targets, and no split/downstream/adversarial/AUC/transfer feedback.",
        "- `nirs4all/` is not modified or required for this preflight.",
        "",
        "## Required Binding Fields",
        "",
        f"`{';'.join(BINDING_FIELDS_REQUIRED)}`",
        "",
        "Row-bound geometry only counts when those fields are complete, at least one geometry field is parseable, and the metadata is not marked generic.",
        "",
        "## Support Checks",
        "",
        f"- Real rows with off-support wavelengths after alignment semantics: `{len(wider_rows)}`.",
        f"- Row-bound geometry metadata rows present and parseable: `{len(row_bound_geometry_rows)}`.",
        f"- Generic geometry rows excluded from readiness: `{len(generic_geometry_rows)}`.",
        f"- Rows rejected for leakage fields: `{len(leakage_rows)}`.",
        "",
    ]
    if wider_rows:
        lines.extend(["| dataset | wavelength range | off-support bins | evidence |", "|---|---:|---:|---|"])
        for row in wider_rows[:20]:
            lines.append(
                f"| `{row.database_name}/{row.dataset}` | `{_fmt_range(row.wavelength_min, row.wavelength_max)}` | "
                f"`{row.off_support_count_after_alignment}` | `{row.evidence}` |"
            )
        lines.append("")
    if row_bound_geometry_rows:
        lines.extend(["| dataset | binding key | metadata source | parsed geometry fields |", "|---|---|---|---|"])
        for row in row_bound_geometry_rows[:20]:
            lines.append(f"| `{row.database_name}/{row.dataset}` | `{row.row_binding_key}` | `{row.metadata_source}` | `{row.parsed_geometry_fields}` |")
        lines.append("")
    if not wider_rows and not row_bound_geometry_rows:
        lines.append(
            "No readiness prerequisite is satisfied: no accepted real row exposes non-empty off-support wavelengths after alignment, "
            "and no accepted row has parseable row-bound source-detector/pathlength/collection geometry metadata."
        )
        lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            f"`{recommendation}`",
            "",
        ]
    )
    if recommendation == READY_RECOMMENDATION:
        lines.append("Write a separate mechanistic audit design before coding any generator/profile change.")
    else:
        lines.append("Remain blocked pending a wider real DIESEL/fuel cohort or row-bound real geometry metadata; do not move to stats/ML/noise capture.")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py \\",
            "  --report bench/nirs_synthetic_pfn/reports/exp29_data_support_manifest_preflight.md",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:g}-{high:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    result = run_preflight(args.root, args.manifest)
    rows = list(result["rows"])
    if args.csv is not None:
        write_csv(rows, args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv, manifest=args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    print({"rows": len(rows), "status": result["status"], "recommendation": result["recommendation"]})


if __name__ == "__main__":
    main()
