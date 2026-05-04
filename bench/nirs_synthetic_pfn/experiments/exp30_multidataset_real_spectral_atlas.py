"""exp30 multidataset real spectral atlas (Phase M0/M1, bench-only).

Read-only file-level inventory of a representative real-data panel from
``bench/tabpfn_paper/data/regression``. The atlas is descriptive only: it
documents what each ``Xtrain``/``Xtest``/``Ytrain``/``Ytest``/``Mtrain``/
``Mtest`` file contains so that later mechanistic regimes can be designed
per data family.

The script does not infer first derivative, SNV, MSC, absorbance, or
reflectance unless documentary evidence is supplied. It does not build
spectra, calibrate, fit statistics/PCA/noise capture, train ML/DL models,
add a profile, change gates/metrics, or use targets/splits as a tuning
oracle. Targets and split policies are recorded as descriptive fields only.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

EXP30_AUDIT_SCOPE = "bench_only_phase_m0_m1_multidataset_real_spectral_atlas"
COMPARISON_SPACE = "uncalibrated_raw_or_unknown"
DEFAULT_PANEL_ROOT = Path("bench/tabpfn_paper/data/regression")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.csv")

WAVENUMBER_THRESHOLD_CM_INV = 4000.0
SAMPLE_X_ROWS = 50
SENTINEL_VALUES: tuple[str, ...] = ("-999", "-9999", "-99")
SPLIT_POLICY_TOKENS: tuple[tuple[str, str], ...] = (
    ("SPXY", "SPXY"),
    ("KS", "KennardStone"),
    ("byCultivar", "byCultivar"),
    ("species", "bySpecies"),
    ("block", "byBlock"),
    ("YbaseSplit", "YbaseSplit"),
    ("strat", "stratified"),
    ("KMeans", "KMeans"),
)
NM_SUFFIX = "_nm"

PANEL: tuple[tuple[str, str, str], ...] = (
    ("MANURE21", "All_manure_MgO_SPXY_strat_Manure_type", "MANURE21/All_manure_MgO_SPXY_strat_Manure_type"),
    ("GRAPEVINE_LeafTraits", "An_spxyG70_30_byCultivar_NeoSpectra", "GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra"),
    ("IncombustibleMaterial", "TIC_spxy70", "IncombustibleMaterial/TIC_spxy70"),
    ("ECOSIS_LeafTraits", "Chla+b_spxyG_species", "ECOSIS_LeafTraits/Chla+b_spxyG_species"),
    ("ALPINE", "ALPINE_P_291_KS", "ALPINE/ALPINE_P_291_KS"),
    ("BEER", "Beer_OriginalExtract_60_YbaseSplit", "BEER/Beer_OriginalExtract_60_YbaseSplit"),
    ("MANURE21", "All_manure_Total_N_SPXY_strat_Manure_type", "MANURE21/All_manure_Total_N_SPXY_strat_Manure_type"),
    ("ECOSIS_LeafTraits", "Chla+b_spxyG_block2deg", "ECOSIS_LeafTraits/Chla+b_spxyG_block2deg"),
    ("COLZA", "N_woOutlier", "COLZA/N_woOutlier"),
    ("GRAPEVINES", "grapevine_chloride_556_KS", "GRAPEVINES/grapevine_chloride_556_KS"),
)

WAVENUMBER_FORCED_DATASETS: frozenset[str] = frozenset({"N_woOutlier"})

DOCUMENTARY_EVIDENCE: dict[tuple[str, str], dict[str, str]] = {
    ("COLZA", "N_woOutlier"): {
        "preprocessing_evidence": "absorbance",
        "documentary_evidence_source": "bench/tabpfn_paper/data/regression/COLZA/README.txt",
        "documentary_evidence_quote": (
            "Absorbance at each wave number within a range of 12489,6 to 3594,9 cm-1 collected by a BRUKER MPA spectrometer."
        ),
    },
}


@dataclass(frozen=True)
class FilePresence:
    Xtrain: bool
    Xtest: bool
    Ytrain: bool
    Ytest: bool
    Mtrain: bool
    Mtest: bool


@dataclass(frozen=True)
class SpectralAxis:
    n_features: int
    axis_type: str
    axis_unit: str
    axis_token_format: str
    axis_direction: str
    axis_min: float | None
    axis_max: float | None
    axis_first: float | None
    axis_last: float | None
    axis_resolution_median: float | None
    axis_separator: str
    axis_forced_reason: str


@dataclass(frozen=True)
class XValueScan:
    rows_sampled: int
    x_min_observed: float | None
    x_max_observed: float | None
    has_negative_values: bool
    has_non_finite_values: bool


@dataclass(frozen=True)
class TargetSummary:
    target_column: str
    train_rows: int
    test_rows: int
    sentinel_values_checked: str
    sentinel_train_rows: int
    sentinel_test_rows: int


@dataclass(frozen=True)
class MetadataSummary:
    has_metadata: bool
    metadata_columns: str
    metadata_columns_count: int


@dataclass(frozen=True)
class AtlasRow:
    status: str
    database_name: str
    dataset: str
    relative_path: str
    files_present: str
    files_missing: str
    train_rows: int
    test_rows: int
    n_features: int
    axis_type: str
    axis_unit: str
    axis_token_format: str
    axis_direction: str
    axis_min: float | None
    axis_max: float | None
    axis_first: float | None
    axis_last: float | None
    axis_resolution_median: float | None
    axis_separator: str
    axis_forced_reason: str
    rows_sampled_for_x_scan: int
    x_min_observed: float | None
    x_max_observed: float | None
    has_negative_values: bool
    has_non_finite_values: bool
    target_column: str
    sentinel_values_checked: str
    sentinel_train_rows: int
    sentinel_test_rows: int
    has_metadata: bool
    metadata_columns_count: int
    metadata_columns: str
    split_policy_inferred: str
    preprocessing_evidence: str
    documentary_evidence_source: str
    documentary_evidence_quote: str
    notes: str = field(default="")
    audit_scope: str = field(default=EXP30_AUDIT_SCOPE)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_separator(line: str) -> str:
    candidates = ((";", line.count(";")), (",", line.count(",")), ("\t", line.count("\t")))
    sep, count = max(candidates, key=lambda item: item[1])
    return sep if count > 0 else ";"


def _strip_token(token: str) -> str:
    cleaned = token.strip()
    if len(cleaned) >= 2 and cleaned[0] in {'"', "'"} and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _classify_token_format(raw_tokens: list[str], stripped_tokens: list[str]) -> str:
    if not raw_tokens:
        return "empty"
    if all(token.casefold().endswith(NM_SUFFIX) for token in stripped_tokens):
        return "nm_suffix"
    quoted = sum(1 for token in raw_tokens if token.strip().startswith('"') or token.strip().startswith("'"))
    if quoted == len(raw_tokens) and all(_is_int_text(_strip_token(token)) for token in raw_tokens):
        return "int_quoted"
    if all(_is_float_text(_strip_token(token)) for token in raw_tokens):
        return "numeric"
    return "mixed"


def _is_float_text(token: str) -> bool:
    if not token:
        return False
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_int_text(token: str) -> bool:
    if not token:
        return False
    try:
        value = float(token)
    except ValueError:
        return False
    return value.is_integer()


def _parse_axis(header_line: str) -> tuple[list[float], str, str]:
    separator = _detect_separator(header_line)
    raw_tokens = [token for token in header_line.rstrip("\n").split(separator) if token.strip()]
    stripped = [_strip_token(token) for token in raw_tokens]
    cleaned: list[float] = []
    for token in stripped:
        text = token
        if text.casefold().endswith(NM_SUFFIX):
            text = text[: -len(NM_SUFFIX)]
        try:
            cleaned.append(float(text))
        except ValueError:
            return [], separator, "unknown"
    token_format = _classify_token_format(raw_tokens, stripped)
    return cleaned, separator, token_format


def _axis_unit_and_type(axis_values: list[float], token_format: str, dataset: str) -> tuple[str, str, str]:
    if not axis_values:
        return "unknown", "unknown", "no_numeric_axis"
    if dataset in WAVENUMBER_FORCED_DATASETS:
        return "wavenumber", "cm-1", "forced_by_panel_rule_N_woOutlier_is_wavenumber"
    if token_format == "nm_suffix":
        return "nm", "nm", "explicit_nm_suffix_in_header_tokens"
    max_value = max(abs(v) for v in axis_values)
    if max_value > WAVENUMBER_THRESHOLD_CM_INV:
        return "wavenumber", "cm-1", "header_max_above_4000_treated_as_cm_inverse"
    if min(axis_values) >= 200.0 and max(axis_values) <= 3500.0:
        return "nm", "nm", "header_range_within_typical_VIS_NIR_SWIR_nanometers"
    return "unknown", "unknown", "header_range_does_not_match_known_axis_class"


def _axis_direction(axis_values: list[float]) -> str:
    if len(axis_values) < 2:
        return "unknown"
    first, last = axis_values[0], axis_values[-1]
    if first < last:
        return "ascending"
    if first > last:
        return "descending"
    return "unknown"


def _axis_resolution(axis_values: list[float]) -> float | None:
    if len(axis_values) < 2:
        return None
    deltas = [abs(b - a) for a, b in zip(axis_values[:-1], axis_values[1:], strict=True) if (b - a) != 0]
    if not deltas:
        return None
    return float(statistics.median(deltas))


def _read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.readline()


def _scan_x_values(path: Path, separator: str, max_rows: int) -> XValueScan:
    if not path.exists():
        return XValueScan(0, None, None, False, False)
    rows_sampled = 0
    x_min: float | None = None
    x_max: float | None = None
    has_neg = False
    has_non_finite = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)
        for _ in range(max_rows):
            line = handle.readline()
            if not line:
                break
            tokens = [_strip_token(tok) for tok in line.rstrip("\n").split(separator) if tok.strip()]
            if not tokens:
                continue
            rows_sampled += 1
            for tok in tokens:
                try:
                    value = float(tok)
                except ValueError:
                    has_non_finite = True
                    continue
                if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
                    has_non_finite = True
                    continue
                if value < 0:
                    has_neg = True
                if x_min is None or value < x_min:
                    x_min = value
                if x_max is None or value > x_max:
                    x_max = value
    return XValueScan(rows_sampled, x_min, x_max, has_neg, has_non_finite)


def _read_target_column_and_sentinels(y_train_path: Path, y_test_path: Path) -> TargetSummary:
    train_rows = 0
    test_rows = 0
    sentinel_train = 0
    sentinel_test = 0
    target_column = ""
    if y_train_path.exists():
        with y_train_path.open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().rstrip("\n")
            target_column = _strip_token(header.split(_detect_separator(header))[0]) if header else ""
            for line in handle:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    continue
                train_rows += 1
                first = _strip_token(stripped.split(",")[0].split(";")[0].split("\t")[0])
                if first in SENTINEL_VALUES:
                    sentinel_train += 1
    if y_test_path.exists():
        with y_test_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.readline()
            for line in handle:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    continue
                test_rows += 1
                first = _strip_token(stripped.split(",")[0].split(";")[0].split("\t")[0])
                if first in SENTINEL_VALUES:
                    sentinel_test += 1
    return TargetSummary(
        target_column=target_column,
        train_rows=train_rows,
        test_rows=test_rows,
        sentinel_values_checked=";".join(SENTINEL_VALUES),
        sentinel_train_rows=sentinel_train,
        sentinel_test_rows=sentinel_test,
    )


def _read_metadata_columns(m_train_path: Path) -> MetadataSummary:
    if not m_train_path.exists():
        return MetadataSummary(False, "", 0)
    header = _read_first_line(m_train_path).rstrip("\n")
    if not header:
        return MetadataSummary(True, "", 0)
    separator = _detect_separator(header)
    columns = [_strip_token(tok) for tok in header.split(separator) if tok.strip()]
    return MetadataSummary(True, ";".join(columns), len(columns))


def _data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if index == 0:
                continue
            if line.strip():
                rows += 1
    return rows


def _file_presence(directory: Path) -> tuple[FilePresence, list[str], list[str]]:
    names = ("Xtrain", "Xtest", "Ytrain", "Ytest", "Mtrain", "Mtest")
    flags: dict[str, bool] = {name: (directory / f"{name}.csv").exists() for name in names}
    presence = FilePresence(**flags)
    present = [name for name in names if flags[name]]
    missing = [name for name in names if not flags[name] and name in {"Xtrain", "Xtest", "Ytrain", "Ytest"}]
    return presence, present, missing


def _split_policy(dataset: str) -> str:
    folded = dataset.casefold()
    found: list[str] = []
    for token, label in SPLIT_POLICY_TOKENS:
        if token.casefold() in folded and label not in found:
            found.append(label)
    return ";".join(found)


def _atlas_row_for_dataset(panel_root: Path, database_name: str, dataset: str, relative_path: str) -> AtlasRow:
    directory = panel_root / relative_path
    if not directory.exists():
        return _missing_directory_row(database_name, dataset, relative_path, "directory_not_found")

    presence, present, missing_required = _file_presence(directory)
    files_present_csv = ";".join(present)
    files_missing_csv = ";".join(missing_required)
    x_train = directory / "Xtrain.csv"
    x_test = directory / "Xtest.csv"
    y_train = directory / "Ytrain.csv"
    y_test = directory / "Ytest.csv"
    m_train = directory / "Mtrain.csv"

    if missing_required:
        return _missing_directory_row(
            database_name,
            dataset,
            relative_path,
            "required_files_missing",
            files_present=files_present_csv,
            files_missing=files_missing_csv,
        )

    header_line = _read_first_line(x_train)
    axis_values, separator, token_format = _parse_axis(header_line)
    axis_type, axis_unit, axis_forced_reason = _axis_unit_and_type(axis_values, token_format, dataset)
    spectral = SpectralAxis(
        n_features=len(axis_values),
        axis_type=axis_type,
        axis_unit=axis_unit,
        axis_token_format=token_format,
        axis_direction=_axis_direction(axis_values),
        axis_min=min(axis_values) if axis_values else None,
        axis_max=max(axis_values) if axis_values else None,
        axis_first=axis_values[0] if axis_values else None,
        axis_last=axis_values[-1] if axis_values else None,
        axis_resolution_median=_axis_resolution(axis_values),
        axis_separator=separator,
        axis_forced_reason=axis_forced_reason,
    )
    x_scan = _scan_x_values(x_train, separator, SAMPLE_X_ROWS)
    targets = _read_target_column_and_sentinels(y_train, y_test)
    metadata_summary = _read_metadata_columns(m_train)

    train_rows = _data_row_count(x_train)
    test_rows = _data_row_count(x_test)

    documentary = DOCUMENTARY_EVIDENCE.get((database_name, dataset), {})
    preprocessing_evidence = documentary.get("preprocessing_evidence", "unknown")
    documentary_source = documentary.get("documentary_evidence_source", "")
    documentary_quote = documentary.get("documentary_evidence_quote", "")

    notes_parts: list[str] = []
    if x_scan.has_negative_values:
        notes_parts.append("negative_x_values_observed_in_sampled_rows")
    if x_scan.has_non_finite_values:
        notes_parts.append("non_finite_x_tokens_observed_in_sampled_rows")
    if targets.sentinel_train_rows or targets.sentinel_test_rows:
        notes_parts.append("target_sentinel_rows_present")
    if presence.Mtrain != presence.Mtest:
        notes_parts.append("metadata_train_test_presence_mismatch")
    if dataset in WAVENUMBER_FORCED_DATASETS:
        notes_parts.append("axis_forced_to_wavenumber_per_panel_rule")

    return AtlasRow(
        status="ok",
        database_name=database_name,
        dataset=dataset,
        relative_path=relative_path,
        files_present=files_present_csv,
        files_missing=files_missing_csv,
        train_rows=train_rows,
        test_rows=test_rows,
        n_features=spectral.n_features,
        axis_type=spectral.axis_type,
        axis_unit=spectral.axis_unit,
        axis_token_format=spectral.axis_token_format,
        axis_direction=spectral.axis_direction,
        axis_min=spectral.axis_min,
        axis_max=spectral.axis_max,
        axis_first=spectral.axis_first,
        axis_last=spectral.axis_last,
        axis_resolution_median=spectral.axis_resolution_median,
        axis_separator=spectral.axis_separator,
        axis_forced_reason=spectral.axis_forced_reason,
        rows_sampled_for_x_scan=x_scan.rows_sampled,
        x_min_observed=x_scan.x_min_observed,
        x_max_observed=x_scan.x_max_observed,
        has_negative_values=x_scan.has_negative_values,
        has_non_finite_values=x_scan.has_non_finite_values,
        target_column=targets.target_column,
        sentinel_values_checked=targets.sentinel_values_checked,
        sentinel_train_rows=targets.sentinel_train_rows,
        sentinel_test_rows=targets.sentinel_test_rows,
        has_metadata=metadata_summary.has_metadata,
        metadata_columns_count=metadata_summary.metadata_columns_count,
        metadata_columns=metadata_summary.metadata_columns,
        split_policy_inferred=_split_policy(dataset),
        preprocessing_evidence=preprocessing_evidence,
        documentary_evidence_source=documentary_source,
        documentary_evidence_quote=documentary_quote,
        notes=";".join(notes_parts),
    )


def _missing_directory_row(
    database_name: str,
    dataset: str,
    relative_path: str,
    status: str,
    *,
    files_present: str = "",
    files_missing: str = "",
) -> AtlasRow:
    return AtlasRow(
        status=status,
        database_name=database_name,
        dataset=dataset,
        relative_path=relative_path,
        files_present=files_present,
        files_missing=files_missing,
        train_rows=0,
        test_rows=0,
        n_features=0,
        axis_type="unknown",
        axis_unit="unknown",
        axis_token_format="empty",
        axis_direction="unknown",
        axis_min=None,
        axis_max=None,
        axis_first=None,
        axis_last=None,
        axis_resolution_median=None,
        axis_separator="",
        axis_forced_reason="not_inspected",
        rows_sampled_for_x_scan=0,
        x_min_observed=None,
        x_max_observed=None,
        has_negative_values=False,
        has_non_finite_values=False,
        target_column="",
        sentinel_values_checked=";".join(SENTINEL_VALUES),
        sentinel_train_rows=0,
        sentinel_test_rows=0,
        has_metadata=False,
        metadata_columns_count=0,
        metadata_columns="",
        split_policy_inferred=_split_policy(dataset),
        preprocessing_evidence="unknown",
        documentary_evidence_source="",
        documentary_evidence_quote="",
        notes=status,
    )


def build_atlas(
    panel_root: Path,
    panel: tuple[tuple[str, str, str], ...] = PANEL,
) -> list[AtlasRow]:
    return [
        _atlas_row_for_dataset(panel_root, database_name, dataset, relative_path)
        for database_name, dataset, relative_path in panel
    ]


def write_csv(rows: list[AtlasRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(AtlasRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _fmt_optional(value: float | None, spec: str = "g") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def render_markdown(
    rows: list[AtlasRow],
    *,
    panel_root: Path,
    report_path: Path,
    csv_path: Path | None,
) -> str:
    ok_rows = [row for row in rows if row.status == "ok"]
    bad_rows = [row for row in rows if row.status != "ok"]
    nm_rows = [row for row in ok_rows if row.axis_type == "nm"]
    wavenumber_rows = [row for row in ok_rows if row.axis_type == "wavenumber"]
    unknown_axis_rows = [row for row in ok_rows if row.axis_type == "unknown"]
    metadata_rich = [row for row in ok_rows if row.has_metadata and row.metadata_columns_count >= 3]
    metadata_poor = [row for row in ok_rows if not row.has_metadata or row.metadata_columns_count < 3]
    negative_x = [row for row in ok_rows if row.has_negative_values]
    sentinel_rows = [row for row in ok_rows if row.sentinel_train_rows or row.sentinel_test_rows]

    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    lines: list[str] = [
        "# Multidataset Real Spectral Atlas (exp30)",
        "",
        f"- audit_scope: `{EXP30_AUDIT_SCOPE}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        f"- panel_root: `{panel_root}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- panel_size: `{len(rows)}`",
        f"- ok_rows: `{len(ok_rows)}`",
        f"- not_inspected_rows: `{len(bad_rows)}`",
        f"- nm_axis_rows: `{len(nm_rows)}`",
        f"- wavenumber_axis_rows: `{len(wavenumber_rows)}`",
        f"- unknown_axis_rows: `{len(unknown_axis_rows)}`",
        f"- rows_with_negative_x_values: `{len(negative_x)}`",
        f"- rows_with_target_sentinels: `{len(sentinel_rows)}`",
        "",
        "## Contract",
        "",
        "- Bench-only Phase M0/M1 inventory: descriptive only, no realism claim, no generator profile, no mechanism, no gate, no promotion, no metric or threshold change.",
        "- Explicit no statistics, no PCA, no covariance, no quantile/marginal/noise capture, no calibration, no ML, no DL, no labels-as-oracle, no targets-as-oracle, no splits-as-oracle, no downstream/adversarial/AUC/transfer feedback for tuning.",
        "- Targets and split policies are listed only as descriptive identity fields, not as tuning oracles.",
        "- Preprocessing status is `unknown` unless a documentary evidence file is cited per row.",
        "- `nirs4all/` is not modified or required for this inventory.",
        "",
        "## Panel Atlas",
        "",
        "| dataset | rows train/test | features | axis | direction | range | resolution (median) | metadata cols | split policy | preprocessing evidence | notes |",
        "|---|---:|---:|---|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        axis_label = f"{row.axis_type} ({row.axis_unit})" if row.axis_unit and row.axis_unit != "unknown" else row.axis_type
        range_label = f"{_fmt_optional(row.axis_min)} → {_fmt_optional(row.axis_max)}" if row.axis_first is not None else "n/a"
        evidence = row.preprocessing_evidence
        if row.preprocessing_evidence != "unknown" and row.documentary_evidence_source:
            evidence = f"{row.preprocessing_evidence} (`{row.documentary_evidence_source}`)"
        notes = row.notes if row.notes else "n/a"
        lines.append(
            "| `"
            f"{row.database_name}/{row.dataset}` | `{row.train_rows}/{row.test_rows}` | `{row.n_features}` | "
            f"{axis_label} | {row.axis_direction} | `{range_label}` | "
            f"`{_fmt_optional(row.axis_resolution_median)}` | `{row.metadata_columns_count}` | "
            f"`{row.split_policy_inferred or 'n/a'}` | {evidence} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Axis Class Map",
            "",
            "| class | count | datasets |",
            "|---|---:|---|",
            f"| nm | `{len(nm_rows)}` | {_dataset_list(nm_rows)} |",
            f"| wavenumber (cm-1) | `{len(wavenumber_rows)}` | {_dataset_list(wavenumber_rows)} |",
            f"| unknown axis | `{len(unknown_axis_rows)}` | {_dataset_list(unknown_axis_rows)} |",
            "",
            "## Value Range And Quality Map",
            "",
            "| dataset | x min sampled | x max sampled | rows sampled | negatives observed | non-finite tokens observed | target sentinels (train/test) |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in ok_rows:
        sentinel_label = f"{row.sentinel_train_rows}/{row.sentinel_test_rows}"
        lines.append(
            f"| `{row.database_name}/{row.dataset}` | `{_fmt_optional(row.x_min_observed)}` | "
            f"`{_fmt_optional(row.x_max_observed)}` | `{row.rows_sampled_for_x_scan}` | "
            f"`{row.has_negative_values}` | `{row.has_non_finite_values}` | `{sentinel_label}` |"
        )

    lines.extend(
        [
            "",
            "## Metadata Map",
            "",
            f"- metadata-rich datasets (>=3 metadata columns): `{len(metadata_rich)}`",
            f"- metadata-poor datasets (no metadata or <3 columns): `{len(metadata_poor)}`",
            "",
            "| dataset | has_metadata | columns_count | columns |",
            "|---|---|---:|---|",
        ]
    )
    for row in ok_rows:
        cols = row.metadata_columns if row.metadata_columns else "n/a"
        lines.append(
            f"| `{row.database_name}/{row.dataset}` | `{row.has_metadata}` | "
            f"`{row.metadata_columns_count}` | `{cols}` |"
        )

    lines.extend(
        [
            "",
            "## Documentary Preprocessing Evidence",
            "",
            "Only entries with explicit on-disk documentation are recorded here. No SNV, MSC, derivative, absorbance, or reflectance status is inferred without a cited source.",
            "",
            "| dataset | preprocessing evidence | source | quote |",
            "|---|---|---|---|",
        ]
    )
    documented = [row for row in ok_rows if row.documentary_evidence_source]
    if documented:
        for row in documented:
            quote = row.documentary_evidence_quote.replace("|", "/")
            lines.append(
                f"| `{row.database_name}/{row.dataset}` | `{row.preprocessing_evidence}` | "
                f"`{row.documentary_evidence_source}` | {quote} |"
            )
    else:
        lines.append("| _none_ | n/a | n/a | n/a |")

    if bad_rows:
        lines.extend(
            [
                "",
                "## Inspection Failures",
                "",
                "| dataset | status | files_present | files_missing |",
                "|---|---|---|---|",
            ]
        )
        for row in bad_rows:
            lines.append(
                f"| `{row.database_name}/{row.dataset}` | `{row.status}` | "
                f"`{row.files_present or 'n/a'}` | `{row.files_missing or 'n/a'}` |"
            )

    lines.extend(
        [
            "",
            "## Phase M1 Distinguishability Checklist",
            "",
            f"- nm vs wavenumber distinguishable: `{bool(nm_rows) and bool(wavenumber_rows)}`",
            f"- raw-positive vs negative-or-processed spectra distinguishable: `{bool(negative_x) and bool([r for r in ok_rows if not r.has_negative_values])}`",
            f"- metadata-rich vs metadata-poor distinguishable: `{bool(metadata_rich) and bool(metadata_poor)}`",
            f"- broad VIS-NIR/SWIR vs narrow instrument supports distinguishable: `{_broad_vs_narrow(ok_rows)}`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py \\",
            "  --panel-root bench/tabpfn_paper/data/regression \\",
            "  --report bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md \\",
            "  --csv bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.csv",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _dataset_list(rows: list[AtlasRow]) -> str:
    if not rows:
        return "n/a"
    return ", ".join(f"`{row.database_name}/{row.dataset}`" for row in rows)


def _broad_vs_narrow(ok_rows: list[AtlasRow]) -> bool:
    broad = False
    narrow = False
    for row in ok_rows:
        if row.axis_min is None or row.axis_max is None:
            continue
        span = row.axis_max - row.axis_min
        if row.axis_type == "nm" and span >= 1500:
            broad = True
        if row.axis_type == "nm" and span <= 1300:
            narrow = True
    return broad and narrow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = build_atlas(args.panel_root)
    if args.csv is not None:
        write_csv(rows, args.csv)
    markdown = render_markdown(
        rows,
        panel_root=args.panel_root,
        report_path=args.report,
        csv_path=args.csv,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    print(
        {
            "panel_size": len(rows),
            "ok_rows": sum(1 for row in rows if row.status == "ok"),
            "axis_classes": sorted({row.axis_type for row in rows if row.status == "ok"}),
        }
    )


if __name__ == "__main__":
    main()
