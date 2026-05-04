"""P2-06 remaining mechanistic evidence/data support inventory.

Read-only inventory for DIESEL/fuel cohort support, available physical
geometry metadata, and predeclared mechanistic laws/constants. This script
does not build synthetic spectra, add a profile, tune constants, run
statistics/PCA/noise capture/ML/DL, or use labels/targets/splits/downstream
feedback.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

EXP28_AUDIT_SCOPE = "bench_only_p2_06_mechanistic_data_support_inventory"
EXP28_DECISION = "blocked_pending_metadata_or_wider_real_cohort_no_stats_ml"
COMPARISON_SPACE = "uncalibrated_raw"
SUPPORT_LOW_NM = 750.0
SUPPORT_HIGH_NM = 1550.0
DEFAULT_REPORT = Path("/tmp/exp28_mechanistic_data_support_inventory.md")
DEFAULT_CSV = Path("/tmp/exp28_mechanistic_data_support_inventory.csv")

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
GEOMETRY_TERMS: tuple[str, ...] = (
    "source_detector",
    "source-detector",
    "source detector",
    "path_length",
    "path length",
    "pathlength",
    "cuvette",
    "illumination_angle",
    "collection_angle",
    "incidence_angle",
    "distance",
    "geometry",
    "optical_path",
    "detector_type",
    "spectral_resolution",
    "stray_light",
    "temperature_sensitivity",
)
LAW_TERMS: tuple[str, ...] = (
    "beer-lambert",
    "kubelka-munk",
    "rayleigh",
    "mie",
    "baseline curvature",
    "edge curvature",
    "detector response",
    "stray light",
    "temperature",
    "cuvette",
    "path_length",
    "path length",
    "optical depth",
    "diesel",
    "gasoline",
    "crude_oil",
    "petrochem_fuels",
    "nirband",
)
TEXT_SUFFIXES = {".csv", ".md", ".py", ".json", ".toml", ".yaml", ".yml", ".txt"}


@dataclass(frozen=True)
class InventoryRow:
    section: str
    status: str
    path: str
    source: str
    task: str
    database_name: str
    dataset: str
    file_kind: str
    in_aom_cohort: bool
    wavelength_min: float | None
    wavelength_max: float | None
    n_wavelengths: int | None
    support_low_nm: float
    support_high_nm: float
    support_count_after_alignment: int | None
    off_support_count_after_alignment: int | None
    extends_outside_750_1550_after_alignment: bool
    geometry_metadata_present: bool
    geometry_metadata_kind: str
    geometry_terms: str
    law_area: str
    law_status: str
    evidence: str
    recommendation_signal: str
    audit_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contains_token(text: str, tokens: tuple[str, ...] = FUEL_TOKENS) -> bool:
    folded = text.casefold()
    return any(token in folded for token in tokens)


def _parse_numeric_header(path: Path) -> list[float]:
    if not path.exists():
        return []
    first = path.open("r", encoding="utf-8", errors="ignore").readline().strip()
    if not first:
        return []
    candidates: list[list[str]] = [
        first.split(";"),
        first.split(","),
        first.split("\t"),
    ]
    tokens = max(candidates, key=len)
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token.strip()))
        except ValueError:
            return []
    return values


def _support_counts(wavelengths: list[float]) -> tuple[int | None, int | None, bool]:
    if not wavelengths:
        return None, None, False
    support = sum(SUPPORT_LOW_NM <= wl <= SUPPORT_HIGH_NM for wl in wavelengths)
    off_support = len(wavelengths) - support
    return support, off_support, off_support > 0


def _merged_wavelength_headers(train_wl: list[float], test_wl: list[float]) -> list[float]:
    if not test_wl or train_wl == test_wl:
        return train_wl
    if not train_wl:
        return test_wl
    return sorted(set(train_wl) | set(test_wl))


def _wavelength_row(
    *,
    section: str,
    status: str,
    root: Path,
    path: Path,
    source: str,
    task: str,
    database_name: str,
    dataset: str,
    file_kind: str,
    in_aom_cohort: bool,
    wavelengths: list[float],
    evidence: str,
    recommendation_signal: str,
) -> InventoryRow:
    support_count, off_support_count, extends = _support_counts(wavelengths)
    return InventoryRow(
        section=section,
        status=status,
        path=_relpath(path, root),
        source=source,
        task=task,
        database_name=database_name,
        dataset=dataset,
        file_kind=file_kind,
        in_aom_cohort=in_aom_cohort,
        wavelength_min=min(wavelengths) if wavelengths else None,
        wavelength_max=max(wavelengths) if wavelengths else None,
        n_wavelengths=len(wavelengths) if wavelengths else None,
        support_low_nm=SUPPORT_LOW_NM,
        support_high_nm=SUPPORT_HIGH_NM,
        support_count_after_alignment=support_count,
        off_support_count_after_alignment=off_support_count,
        extends_outside_750_1550_after_alignment=extends,
        geometry_metadata_present=False,
        geometry_metadata_kind="",
        geometry_terms="",
        law_area="",
        law_status="",
        evidence=evidence,
        recommendation_signal=recommendation_signal,
        audit_scope=EXP28_AUDIT_SCOPE,
    )


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_cohort_rows(root: Path, path: Path, *, source: str, task: str) -> list[InventoryRow]:
    if not path.exists():
        return []
    rows: list[InventoryRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            haystack = " ".join(
                str(record.get(key, ""))
                for key in ("database_name", "dataset", "train_path", "test_path")
            )
            if not _contains_token(haystack):
                continue
            train_path = root / str(record.get("train_path", ""))
            test_path = root / str(record.get("test_path", ""))
            train_wl = _parse_numeric_header(train_path)
            test_wl = _parse_numeric_header(test_path)
            wavelengths = _merged_wavelength_headers(train_wl, test_wl)
            rows.append(
                _wavelength_row(
                    section="aom_real_cohort",
                    status="compared" if wavelengths else "blocked_wavelength_header_unparsed",
                    root=root,
                    path=train_path,
                    source=source,
                    task=task,
                    database_name=str(record.get("database_name", "")),
                    dataset=str(record.get("dataset", "")),
                    file_kind="Xtrain/Xtest",
                    in_aom_cohort=True,
                    wavelengths=wavelengths,
                    evidence="AOM cohort row with numeric spectral header",
                    recommendation_signal=(
                        "wider_real_grid_available"
                        if _support_counts(wavelengths)[2]
                        else "current_real_grid_inside_750_1550"
                    ),
                )
            )
    return rows


def inventory_aom_cohorts(root: Path) -> list[InventoryRow]:
    return [
        * _read_cohort_rows(
            root,
            root / "bench/AOM_v0/benchmarks/cohort_regression.csv",
            source="AOM_regression",
            task="regression",
        ),
        * _read_cohort_rows(
            root,
            root / "bench/AOM_v0/benchmarks/cohort_classification.csv",
            source="AOM_classification",
            task="classification",
        ),
    ]


def inventory_local_fuel_files(root: Path) -> list[InventoryRow]:
    base = root / "bench/tabpfn_paper/data"
    if not base.exists():
        return []
    rows: list[InventoryRow] = []
    seen: set[Path] = set()
    for xtrain in sorted(base.rglob("Xtrain.csv")):
        dataset_dir = xtrain.parent
        haystack = str(dataset_dir.relative_to(base))
        if not _contains_token(haystack):
            continue
        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        xtest = dataset_dir / "Xtest.csv"
        train_wl = _parse_numeric_header(xtrain)
        test_wl = _parse_numeric_header(xtest)
        wavelengths = _merged_wavelength_headers(train_wl, test_wl)
        parts = dataset_dir.relative_to(base).parts
        task = parts[0] if len(parts) > 0 else ""
        database_name = parts[1] if len(parts) > 1 else ""
        dataset = parts[-1]
        rows.append(
            _wavelength_row(
                section="local_fuel_file",
                status="compared" if wavelengths else "blocked_wavelength_header_unparsed",
                root=root,
                path=xtrain,
                source="tabpfn_paper_data",
                task=task,
                database_name=database_name,
                dataset=dataset,
                file_kind="Xtrain/Xtest",
                in_aom_cohort=False,
                wavelengths=wavelengths,
                evidence="local data directory matching fuel/petrochemical tokens",
                recommendation_signal=(
                    "wider_real_grid_available"
                    if _support_counts(wavelengths)[2]
                    else "current_real_grid_inside_750_1550"
                ),
            )
        )
    return rows


def _iter_text_files(root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for relative in relative_roots:
        base = root / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def _matching_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    folded = text.casefold()
    return sorted({term for term in terms if term.casefold() in folded})


def _short_evidence(text: str, terms: list[str]) -> str:
    if not terms:
        return ""
    folded = text.casefold()
    first = min((folded.find(term.casefold()), term) for term in terms if term.casefold() in folded)
    start = max(first[0] - 90, 0)
    end = min(first[0] + 180, len(text))
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet[:260]


def _read_limited_text(path: Path, *, max_chars: int = 2_000_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def inventory_geometry_metadata(root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []

    metadata_files = [
        *sorted((root / "bench").rglob("Mtrain.csv")),
        *sorted((root / "bench").rglob("Mtest.csv")),
        *sorted((root / "bench/AOM_v0/benchmarks").glob("cohort_*.csv")),
    ]
    for path in metadata_files:
        first = path.open("r", encoding="utf-8", errors="ignore").readline()
        terms = _matching_terms(first, GEOMETRY_TERMS)
        if not terms:
            continue
        rows.append(
            _metadata_row(
                root=root,
                path=path,
                kind="real_cohort_metadata_header",
                terms=terms,
                evidence=first[:260],
            )
        )

    for path in _iter_text_files(root, ("bench/nirs_synthetic_pfn", "docs/_internal", "nirs4all/synthesis")):
        text = _read_limited_text(path)
        terms = _matching_terms(text, GEOMETRY_TERMS)
        if not terms:
            continue
        kind = "generic_synthesis_model" if "nirs4all/synthesis" in _relpath(path, root) else "bench_or_internal_doc"
        rows.append(
            _metadata_row(
                root=root,
                path=path,
                kind=kind,
                terms=terms[:12],
                evidence=_short_evidence(text, terms),
            )
        )
    return _dedupe_metadata_rows(rows)


def _metadata_row(*, root: Path, path: Path, kind: str, terms: list[str], evidence: str) -> InventoryRow:
    return InventoryRow(
        section="geometry_metadata",
        status="present",
        path=_relpath(path, root),
        source="",
        task="",
        database_name="",
        dataset="",
        file_kind=path.suffix.lstrip("."),
        in_aom_cohort=False,
        wavelength_min=None,
        wavelength_max=None,
        n_wavelengths=None,
        support_low_nm=SUPPORT_LOW_NM,
        support_high_nm=SUPPORT_HIGH_NM,
        support_count_after_alignment=None,
        off_support_count_after_alignment=None,
        extends_outside_750_1550_after_alignment=False,
        geometry_metadata_present=True,
        geometry_metadata_kind=kind,
        geometry_terms=";".join(terms),
        law_area="",
        law_status="",
        evidence=evidence,
        recommendation_signal=(
            "generic_geometry_available_not_real_row_bound"
            if kind == "generic_synthesis_model"
            else "not_sufficient_for_source_detector_audit"
        ),
        audit_scope=EXP28_AUDIT_SCOPE,
    )


def _dedupe_metadata_rows(rows: list[InventoryRow]) -> list[InventoryRow]:
    seen: set[tuple[str, str]] = set()
    out: list[InventoryRow] = []
    for row in rows:
        key = (row.path, row.geometry_metadata_kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def inventory_mechanistic_laws(root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for path in _iter_text_files(root, ("bench/nirs_synthetic_pfn/docs", "docs/_internal", "nirs4all/synthesis")):
        text = _read_limited_text(path)
        terms = _matching_terms(text, LAW_TERMS)
        if not terms:
            continue
        rel = _relpath(path, root)
        area = _law_area(rel, terms)
        status = _law_status(rel, terms)
        rows.append(
            InventoryRow(
                section="mechanistic_law",
                status="present",
                path=rel,
                source="",
                task="",
                database_name="",
                dataset="",
                file_kind=path.suffix.lstrip("."),
                in_aom_cohort=False,
                wavelength_min=None,
                wavelength_max=None,
                n_wavelengths=None,
                support_low_nm=SUPPORT_LOW_NM,
                support_high_nm=SUPPORT_HIGH_NM,
                support_count_after_alignment=None,
                off_support_count_after_alignment=None,
                extends_outside_750_1550_after_alignment=False,
                geometry_metadata_present=False,
                geometry_metadata_kind="",
                geometry_terms="",
                law_area=area,
                law_status=status,
                evidence=_short_evidence(text, terms),
                recommendation_signal=_law_signal(status),
                audit_scope=EXP28_AUDIT_SCOPE,
            )
        )
    return rows


def _law_area(path: str, terms: list[str]) -> str:
    folded = " ".join([path, *terms]).casefold()
    if "measurement_modes" in folded or "beer-lambert" in folded or "kubelka" in folded:
        return "measurement_mode_pathlength_scattering"
    if "_constants" in folded or "diesel" in folded or "gasoline" in folded or "crude_oil" in folded:
        return "fuel_hydrocarbon_band_constants"
    if "domains.py" in folded or "petrochem_fuels" in folded:
        return "petrochemical_domain_prior"
    if "instrument" in folded or "detector" in folded:
        return "instrument_detector_response"
    if "roadmap" in folded or "decision" in folded:
        return "bench_design_decision"
    return "other_mechanistic_reference"


def _law_status(path: str, terms: list[str]) -> str:
    rel = path.casefold()
    if "10_p2b" in rel:
        return "design_blocked_coupling_law_unapproved"
    if "09_synthetic_realism_roadmap" in rel:
        return "roadmap_hypothesis_already_bounded_or_blocked"
    if "nirs4all/synthesis" in rel:
        return "predeclared_generic_law_not_real_metadata_bound"
    return "doc_reference_not_a_current_bench_test"


def _law_signal(status: str) -> str:
    if status == "predeclared_generic_law_not_real_metadata_bound":
        return "possible_future_design_input_requires_real_metadata_or_explicit_mapping"
    if status == "design_blocked_coupling_law_unapproved":
        return "do_not_code_until_physical_coupling_law_predeclared"
    return "no_immediate_new_uncalibrated_audit_without_more_support"


def run_inventory(root: Path) -> dict[str, Any]:
    aom_rows = inventory_aom_cohorts(root)
    local_rows = inventory_local_fuel_files(root)
    geometry_rows = inventory_geometry_metadata(root)
    law_rows = inventory_mechanistic_laws(root)
    rows = [*aom_rows, *local_rows, *geometry_rows, *law_rows]
    real_grid_rows = [row for row in [*aom_rows, *local_rows] if row.status == "compared"]
    wider_rows = [row for row in real_grid_rows if row.extends_outside_750_1550_after_alignment]
    real_geometry_rows = [
        row
        for row in geometry_rows
        if row.geometry_metadata_kind == "real_cohort_metadata_header"
        and any(term in row.geometry_terms for term in ("source_detector", "collection_angle", "illumination_angle", "path_length"))
    ]
    generic_geometry_rows = [
        row for row in geometry_rows if row.geometry_metadata_kind == "generic_synthesis_model"
    ]
    generic_law_rows = [
        row for row in law_rows if row.law_status == "predeclared_generic_law_not_real_metadata_bound"
    ]
    return {
        "status": "done",
        "rows": rows,
        "aom_fuel_rows": aom_rows,
        "local_fuel_rows": local_rows,
        "real_grid_rows": real_grid_rows,
        "wider_real_grid_rows": wider_rows,
        "geometry_rows": geometry_rows,
        "real_geometry_rows": real_geometry_rows,
        "generic_geometry_rows": generic_geometry_rows,
        "law_rows": law_rows,
        "generic_law_rows": generic_law_rows,
        "recommendation": (
            "blocked_pending_metadata_or_wider_real_cohort_no_stats_ml"
            if not wider_rows and not real_geometry_rows
            else "continue_mechanistic_with_data_supported_audit"
        ),
    }


def write_csv(rows: list[InventoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field.name for field in fields(InventoryRow)],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path) -> str:
    rows = list(result["rows"])
    real_grid_rows = list(result["real_grid_rows"])
    wider_rows = list(result["wider_real_grid_rows"])
    real_geometry_rows = list(result["real_geometry_rows"])
    generic_geometry_rows = list(result["generic_geometry_rows"])
    law_rows = list(result["law_rows"])
    generic_law_rows = list(result["generic_law_rows"])
    aom_rows = list(result["aom_fuel_rows"])
    local_rows = list(result["local_fuel_rows"])
    recommendation = str(result["recommendation"])

    lines = [
        "# P2-06 Mechanistic Data Support Inventory",
        "",
        f"- audit_scope: `{EXP28_AUDIT_SCOPE}`",
        f"- decision: `{EXP28_DECISION}`",
        f"- comparison_space: `{COMPARISON_SPACE}`",
        f"- report: `{report_path}`",
        f"- csv: `{csv_path}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Contract",
        "",
        "- Bench inventory only; no generator profile, gate, promotion, threshold, or metric change.",
        "- No calibration, real-stat capture, PCA/noise capture, ML/DL, labels, targets, splits, or downstream feedback.",
        "- `nirs4all/` is inspected read-only and not modified.",
        "",
        "## Data Support",
        "",
        f"- AOM fuel/DIESEL cohort rows discovered: `{len(aom_rows)}`.",
        f"- Local fuel/DIESEL file directories discovered: `{len(local_rows)}`.",
        f"- Real grid rows with parsed wavelengths: `{len(real_grid_rows)}`.",
        f"- Rows extending outside `750-1550 nm` after real-grid alignment semantics: `{len(wider_rows)}`.",
        "",
    ]
    if wider_rows:
        lines.extend(["| dataset | wavelength range | off-support bins | path |", "|---|---:|---:|---|"])
        for row in wider_rows[:20]:
            lines.append(
                f"| `{row.database_name}/{row.dataset}` | "
                f"`{_fmt_range(row.wavelength_min, row.wavelength_max)}` | "
                f"`{row.off_support_count_after_alignment}` | `{row.path}` |"
            )
    else:
        lines.append(
            "No DIESEL/BTEX/petrochemical real cohort or local fuel file found here provides "
            "post-alignment wavelength support outside `750-1550 nm`; the current structural "
            "P2-03 limitation remains."
        )

    lines.extend(
        [
            "",
            "## Geometry Metadata",
            "",
            f"- Real cohort metadata headers with source-detector/path/angle geometry: `{len(real_geometry_rows)}`.",
            f"- Generic synthesis geometry/instrument definitions found under `nirs4all/synthesis`: `{len(generic_geometry_rows)}`.",
            "",
        ]
    )
    if real_geometry_rows:
        for row in real_geometry_rows[:10]:
            lines.append(f"- `{row.path}`: `{row.geometry_terms}`")
    else:
        lines.append(
            "No source-detector distance, illumination/collection angle, pathlength, or collection geometry "
            "metadata was found in real cohort metadata headers. Generic code-level geometry exists, but it is "
            "not row-bound real metadata for a mechanistic geometry audit."
        )

    lines.extend(
        [
            "",
            "## Mechanistic Laws",
            "",
            f"- Mechanistic law/constant references found: `{len(law_rows)}`.",
            f"- Generic predeclared laws/constants not bound to current real metadata: `{len(generic_law_rows)}`.",
            "",
            "Relevant available inputs include petrochemical domain priors, fuel hydrocarbon band constants, "
            "Beer-Lambert/transmittance pathlength, Kubelka-Munk/reflectance scattering, scattering wavelength "
            "exponents, detector/instrument response fields, edge curvature, and temperature sensitivity. "
            "They are not enough by themselves to approve P2b coupling or a geometry audit because the current "
            "DIESEL real rows do not expose the corresponding physical metadata, and the only explicit P2b "
            "coupling document remains design-blocked pending a physical law.",
            "",
            "## Recommendation",
            "",
            f"`{recommendation}`",
            "",
        ]
    )
    if recommendation == "continue_mechanistic_with_data_supported_audit":
        lines.append(
            "Continue mechanistic work only on the concrete data-supported axis exposed in the CSV."
        )
    else:
        lines.append(
            "Declare the current uncalibrated mechanistic path blocked pending either a wider real DIESEL/fuel "
            "cohort with non-empty off-support wavelengths after alignment, or row-bound source-detector/"
            "pathlength/collection geometry metadata. Do not move to stats/ML yet; that remains a later phase "
            "after a documented mechanistic stop review."
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp28_mechanistic_data_support_inventory.py",
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    result = run_inventory(args.root)
    rows = list(result["rows"])
    write_csv(rows, args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"wrote {args.csv}")
    print({"rows": len(rows), "status": result["status"], "recommendation": result["recommendation"]})


if __name__ == "__main__":
    main()
