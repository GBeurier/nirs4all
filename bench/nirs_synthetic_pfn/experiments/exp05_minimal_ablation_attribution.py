"""Phase B5 report-only attribution over existing B2/B3/B4 reports.

This is not a causal ablation. It consumes CSV/Markdown artifacts already
written by earlier phases and emits grouped attribution summaries only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_B2_CSV = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv")
DEFAULT_B3_CSV = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.csv")
DEFAULT_B3_MD = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.md")
DEFAULT_B4_CSV = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.csv")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv")

SMOKE_AUC_THRESHOLD = 0.85
STRETCH_AUC_THRESHOLD = 0.70
PCA_FAIL_THRESHOLD = 0.0
NAMED_GAP_TOKENS = ("BEER", "DIESEL", "CORN")
GATE_STATUS = "BLOCKED_REPORT_ONLY"

OUTPUT_FIELDS = [
    "category",
    "value",
    "row_count",
    "compared_count",
    "blocked_count",
    "raw_authoritative_count",
    "auc_smoke_fail_count",
    "auc_stretch_fail_count",
    "pca_fail_count",
    "nn_ratio_fail_count",
    "derivative_fail_count",
    "max_auc",
    "median_auc",
    "gate_status",
]


@dataclass(frozen=True)
class B2Row:
    source: str
    task: str
    dataset: str
    preset: str
    comparison_space: str
    status: str
    mapping_strategy: str
    mapping_reason: str
    adversarial_auc: float | None
    pca_overlap: float | None
    provisional_decision: str
    blocked_reason: str

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> B2Row:
        return cls(
            source=_clean(row.get("source")),
            task=_clean(row.get("task")),
            dataset=_clean(row.get("dataset")),
            preset=_clean(row.get("synthetic_preset")) or _clean(row.get("preset")),
            comparison_space=_clean(row.get("comparison_space")),
            status=_clean(row.get("status")),
            mapping_strategy=_clean(row.get("synthetic_mapping_strategy")) or "missing",
            mapping_reason=_clean(row.get("synthetic_mapping_reason")),
            adversarial_auc=_parse_float(_clean(row.get("adversarial_auc"))),
            pca_overlap=_parse_float(_clean(row.get("pca_overlap"))),
            provisional_decision=_clean(row.get("provisional_decision")),
            blocked_reason=_clean(row.get("blocked_reason")),
        )

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.dataset, self.source, self.task, self.preset, self.comparison_space)

    @property
    def is_compared(self) -> bool:
        return self.status == "compared"

    @property
    def is_blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def is_raw_authoritative(self) -> bool:
        return self.comparison_space == "uncalibrated_raw"

    @property
    def is_calibrated_diagnostic(self) -> bool:
        return self.comparison_space == "calibrated_raw_diagnostic"

    @property
    def is_snv_diagnostic(self) -> bool:
        return self.comparison_space == "snv"

    @property
    def companion_failures(self) -> set[str]:
        prefix = "provisional_review:"
        if not self.provisional_decision.startswith(prefix):
            return set()
        return {item for item in self.provisional_decision[len(prefix) :].split(",") if item}


@dataclass(frozen=True)
class B3Row:
    key: tuple[str, str, str, str, str]
    comparison_space: str
    status: str
    adversarial_auc: float | None
    smoke_fail: bool
    stretch_fail: bool
    blocked_class: str

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> B3Row:
        comparison_space = _clean(row.get("comparison_space"))
        return cls(
            key=(
                _clean(row.get("dataset")),
                _clean(row.get("source")),
                _clean(row.get("task")),
                _clean(row.get("preset")),
                comparison_space,
            ),
            comparison_space=comparison_space,
            status=_clean(row.get("status")),
            adversarial_auc=_parse_float(_clean(row.get("adversarial_auc"))),
            smoke_fail=_parse_bool(row.get("smoke_fail")),
            stretch_fail=_parse_bool(row.get("stretch_fail")),
            blocked_class=_clean(row.get("blocked_class")),
        )

    @property
    def raw_authoritative(self) -> bool:
        return self.comparison_space == "uncalibrated_raw"

    @property
    def missing_auc(self) -> bool:
        return self.status == "compared" and self.adversarial_auc is None


@dataclass(frozen=True)
class AttributionRow:
    category: str
    value: str
    row_count: int
    compared_count: int
    blocked_count: int
    raw_authoritative_count: int
    auc_smoke_fail_count: int
    auc_stretch_fail_count: int
    pca_fail_count: int
    nn_ratio_fail_count: int
    derivative_fail_count: int
    max_auc: float | None
    median_auc: float | None
    gate_status: str = GATE_STATUS

    def as_csv_row(self) -> dict[str, str]:
        return {
            "category": self.category,
            "value": self.value,
            "row_count": str(self.row_count),
            "compared_count": str(self.compared_count),
            "blocked_count": str(self.blocked_count),
            "raw_authoritative_count": str(self.raw_authoritative_count),
            "auc_smoke_fail_count": str(self.auc_smoke_fail_count),
            "auc_stretch_fail_count": str(self.auc_stretch_fail_count),
            "pca_fail_count": str(self.pca_fail_count),
            "nn_ratio_fail_count": str(self.nn_ratio_fail_count),
            "derivative_fail_count": str(self.derivative_fail_count),
            "max_auc": _fmt(self.max_auc),
            "median_auc": _fmt(self.median_auc),
            "gate_status": self.gate_status,
        }


@dataclass(frozen=True)
class Summary:
    b2_path: Path
    b3_csv_path: Path
    b3_md_path: Path
    b4_path: Path
    b2_sha256: str
    b3_sha256: str
    b4_sha256: str
    b2_rows: list[B2Row]
    b3_rows: dict[tuple[str, str, str, str, str], B3Row]
    b4_rows: list[dict[str, str]]
    attribution_rows: list[AttributionRow]

    @property
    def raw_compared_count(self) -> int:
        return sum(1 for row in self.b2_rows if row.is_raw_authoritative and row.is_compared)

    @property
    def raw_blocked_count(self) -> int:
        return sum(1 for row in self.b2_rows if row.is_raw_authoritative and row.is_blocked)

    @property
    def b3_no_go(self) -> bool:
        raw_rows = [row for row in self.b3_rows.values() if row.raw_authoritative]
        if not raw_rows:
            return True
        return any(row.smoke_fail or row.blocked_class or row.missing_auc for row in raw_rows)

    @property
    def b2_raw_realism_failed(self) -> bool:
        raw_rows = [row for row in self.b2_rows if row.is_raw_authoritative]
        if not raw_rows:
            return True
        return any(
            row.is_blocked or row.adversarial_auc is None or row.adversarial_auc > SMOKE_AUC_THRESHOLD
            for row in raw_rows
        )

    @property
    def b4_blocked(self) -> bool:
        return any(_clean(row.get("status")).startswith("BLOCKED") for row in self.b4_rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_report(
        b2_csv=args.b2_csv,
        b3_csv=args.b3_csv,
        b3_md=args.b3_md,
        b4_csv=args.b4_csv,
        csv_path=args.csv,
        report_path=args.report,
    )
    print(f"status={GATE_STATUS}")
    print(f"b2_rows={len(summary.b2_rows)}")
    print(f"attribution_rows={len(summary.attribution_rows)}")
    print(f"b2_raw_realism_failed={_bool_text(summary.b2_raw_realism_failed)}")
    print(f"b3_no_go={_bool_text(summary.b3_no_go)}")
    print(f"b4_blocked={_bool_text(summary.b4_blocked)}")
    print(f"csv={args.csv}")
    print(f"report={args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2-csv", type=Path, default=DEFAULT_B2_CSV)
    parser.add_argument("--b3-csv", type=Path, default=DEFAULT_B3_CSV)
    parser.add_argument("--b3-md", type=Path, default=DEFAULT_B3_MD)
    parser.add_argument("--b4-csv", type=Path, default=DEFAULT_B4_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def run_report(
    *,
    b2_csv: Path,
    b3_csv: Path,
    b3_md: Path,
    b4_csv: Path,
    csv_path: Path,
    report_path: Path,
) -> Summary:
    b2_rows = load_b2_rows(b2_csv)
    b3_rows = load_b3_rows(b3_csv)
    b4_rows = load_csv_rows(b4_csv, label="B4 CSV")
    if not b3_md.exists():
        raise FileNotFoundError(f"B3 Markdown report not found: {b3_md}")

    attribution_rows = build_attribution_rows(b2_rows=b2_rows, b3_rows=b3_rows, b4_rows=b4_rows)
    summary = Summary(
        b2_path=b2_csv,
        b3_csv_path=b3_csv,
        b3_md_path=b3_md,
        b4_path=b4_csv,
        b2_sha256=_sha256(b2_csv),
        b3_sha256=_sha256(b3_csv),
        b4_sha256=_sha256(b4_csv),
        b2_rows=b2_rows,
        b3_rows=b3_rows,
        b4_rows=b4_rows,
        attribution_rows=attribution_rows,
    )
    write_attribution_csv(attribution_rows, csv_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary=summary, csv_path=csv_path, report_path=report_path), encoding="utf-8")
    return summary


def load_b2_rows(path: Path) -> list[B2Row]:
    rows = load_csv_rows(path, label="B2 CSV")
    return [B2Row.from_csv(row) for row in rows]


def load_b3_rows(path: Path) -> dict[tuple[str, str, str, str, str], B3Row]:
    rows = load_csv_rows(path, label="B3 CSV")
    parsed = [B3Row.from_csv(row) for row in rows]
    return {row.key: row for row in parsed}


def load_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_attribution_rows(
    *,
    b2_rows: list[B2Row],
    b3_rows: dict[tuple[str, str, str, str, str], B3Row],
    b4_rows: list[dict[str, str]],
) -> list[AttributionRow]:
    groups: dict[tuple[str, str], list[B2Row]] = defaultdict(list)
    for row in b2_rows:
        for category, value in row_categories(row):
            groups[(category, value)].append(row)

    attribution_rows = [
        summarize_group(category=category, value=value, rows=rows, b3_rows=b3_rows)
        for (category, value), rows in groups.items()
    ]
    attribution_rows.extend(b4_gate_rows(b4_rows))
    return sorted(attribution_rows, key=lambda row: (row.category, row.value))


def row_categories(row: B2Row) -> list[tuple[str, str]]:
    provenance = _mapping_provenance(row.mapping_reason)
    blocker = blocked_class(row)
    categories = [
        ("mapping_strategy", row.mapping_strategy or "missing"),
        ("source_override", provenance.source_override),
        ("wavelength_policy", provenance.wavelength_policy),
        ("grid_source", provenance.grid_source),
        ("grid_remap_reason", provenance.grid_remap_reason),
        ("comparison_space", row.comparison_space or "missing"),
    ]
    if blocker:
        categories.append(("blocker_class", blocker))
    for token in named_gap_tokens(row):
        categories.append(("named_gap", token))
    return categories


@dataclass(frozen=True)
class MappingProvenance:
    source_override: str
    wavelength_policy: str
    grid_source: str
    grid_remap_reason: str


def _mapping_provenance(mapping_reason: str) -> MappingProvenance:
    if not mapping_reason:
        return MappingProvenance("missing", "missing", "missing", "missing")
    try:
        metadata = json.loads(mapping_reason)
    except json.JSONDecodeError:
        return MappingProvenance("unparseable_json", "unparseable_json", "unparseable_json", "unparseable_json")

    generation = _dict(metadata.get("generation"))
    source_overrides = _dict(generation.get("source_overrides"))
    wavelength_policy = _dict(generation.get("canonical_wavelength_policy"))
    support_override = _dict(wavelength_policy.get("bench_wavelength_support_override"))
    grid = _dict(generation.get("grid"))
    grid_remap = _dict(metadata.get("grid_remap"))
    return MappingProvenance(
        source_override=_clean(source_overrides.get("reason")) or _enabled_value(source_overrides.get("enabled")),
        wavelength_policy=_clean(support_override.get("reason")) or "default_policy",
        grid_source=_clean(grid.get("grid_source")) or "missing",
        grid_remap_reason=_clean(grid_remap.get("reason")) or "missing",
    )


def summarize_group(
    *,
    category: str,
    value: str,
    rows: list[B2Row],
    b3_rows: dict[tuple[str, str, str, str, str], B3Row],
) -> AttributionRow:
    compared_rows = [row for row in rows if row.is_compared]
    authoritative_rows = [row for row in rows if row.is_raw_authoritative]
    authoritative_compared_rows = [row for row in authoritative_rows if row.is_compared]
    authoritative_aucs = [
        row.adversarial_auc
        for row in authoritative_compared_rows
        if row.adversarial_auc is not None
    ]
    return AttributionRow(
        category=category,
        value=value,
        row_count=len(rows),
        compared_count=len(compared_rows),
        blocked_count=sum(1 for row in rows if row.is_blocked),
        raw_authoritative_count=sum(1 for row in rows if row.is_raw_authoritative),
        auc_smoke_fail_count=sum(
            1
            for row in rows
            if (b3 := b3_rows.get(row.key)) is not None and b3.raw_authoritative and b3.smoke_fail
        ),
        auc_stretch_fail_count=sum(
            1
            for row in rows
            if (b3 := b3_rows.get(row.key)) is not None and b3.raw_authoritative and b3.stretch_fail
        ),
        pca_fail_count=sum(
            1
            for row in authoritative_compared_rows
            if row.pca_overlap is not None and row.pca_overlap <= PCA_FAIL_THRESHOLD
        ),
        nn_ratio_fail_count=sum(
            1 for row in authoritative_compared_rows if "nearest_neighbor_ratio" in row.companion_failures
        ),
        derivative_fail_count=sum(
            1 for row in authoritative_compared_rows if "derivative_gap" in row.companion_failures
        ),
        max_auc=max(authoritative_aucs) if authoritative_aucs else None,
        median_auc=statistics.median(authoritative_aucs) if authoritative_aucs else None,
    )


def b4_gate_rows(b4_rows: list[dict[str, str]]) -> list[AttributionRow]:
    rows: list[AttributionRow] = []
    for row in b4_rows:
        status = _clean(row.get("status")) or "missing"
        blocked = 1 if status.startswith("BLOCKED") else 0
        rows.append(
            AttributionRow(
                category="blocker_class",
                value=f"B4:{status}",
                row_count=1,
                compared_count=0,
                blocked_count=blocked,
                raw_authoritative_count=0,
                auc_smoke_fail_count=_parse_int(row.get("raw_smoke_failures")),
                auc_stretch_fail_count=0,
                pca_fail_count=0,
                nn_ratio_fail_count=0,
                derivative_fail_count=0,
                max_auc=None,
                median_auc=None,
            )
        )
        blocked_reason = _clean(row.get("blocked_reason"))
        if blocked_reason:
            for reason in blocked_reason.split(";"):
                rows.append(
                    AttributionRow(
                        category="blocker_class",
                        value=f"B4:{reason}",
                        row_count=1,
                        compared_count=0,
                        blocked_count=blocked,
                        raw_authoritative_count=0,
                        auc_smoke_fail_count=_parse_int(row.get("raw_smoke_failures")),
                        auc_stretch_fail_count=0,
                        pca_fail_count=0,
                        nn_ratio_fail_count=0,
                        derivative_fail_count=0,
                        max_auc=None,
                        median_auc=None,
                    )
                )
    return rows


def write_attribution_csv(rows: list[AttributionRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def render_markdown(*, summary: Summary, csv_path: Path, report_path: Path) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp05_minimal_ablation_attribution.py"
    )
    return "\n".join(
        [
            "# Minimal Ablation Attribution",
            "",
            "## Objective",
            "",
            "Create a CSV-first, report-only attribution over existing B2/B3/B4 rows. This is not a causal ablation and it does not estimate counterfactual effects.",
            "",
            "## Exact Command",
            "",
            f"`{command}`",
            "",
            "## Gate Status",
            "",
            f"- Status: `{GATE_STATUS}` in B2/B3 `NO-GO` context; B4 blocked: `{_bool_text(summary.b4_blocked)}`.",
            f"- B2 raw realism gate failed reflected: `{_bool_text(summary.b2_raw_realism_failed)}`.",
            f"- B3 NO-GO reflected: `{_bool_text(summary.b3_no_go)}`.",
            "- Authoritative comparison_space is `uncalibrated_raw`; legacy `raw`, `calibrated_raw_diagnostic`, and `snv` rows are diagnostics only and are excluded from the named-gap and authoritative-count summaries.",
            "- No integration readiness is claimed.",
            "- No transfer claim is made.",
            "- No model training is run.",
            "- No synthetic data generation is run.",
            "- No spectra are loaded.",
            "- Existing AUC values and B3 fail flags are read as fixed report inputs.",
            "",
            "## Inputs",
            "",
            f"- B2 CSV: `{summary.b2_path}`; SHA-256 `{summary.b2_sha256}`; rows {len(summary.b2_rows)}.",
            f"- B3 CSV: `{summary.b3_csv_path}`; SHA-256 `{summary.b3_sha256}`; rows {len(summary.b3_rows)}.",
            f"- B3 Markdown: `{summary.b3_md_path}`.",
            f"- B4 CSV: `{summary.b4_path}`; SHA-256 `{summary.b4_sha256}`; rows {len(summary.b4_rows)}.",
            "",
            "## Key Summaries",
            "",
            f"- Raw compared rows: {summary.raw_compared_count}.",
            f"- Raw blocked rows: {summary.raw_blocked_count}.",
            f"- Attribution CSV rows: {len(summary.attribution_rows)}.",
            "",
            "## Top Failing Groups",
            "",
            _top_groups_table(summary.attribution_rows),
            "",
            "## Named Gaps",
            "",
            _named_gap_table(summary.attribution_rows),
            "",
            "## B4 Blocked Gate",
            "",
            _b4_table(summary.attribution_rows),
            "",
            "## Output",
            "",
            f"- CSV attribution: `{csv_path}`",
            f"- Markdown summary: `{report_path}`",
            "",
            "## Next Actions",
            "",
            "- Remediate raw B2/B3 failures before reopening B4.",
            "- Prioritize BEER, DIESEL, and CORN raw gaps where they appear in the attribution CSV.",
            "- Re-run B2/B3 reports after remediation, then regenerate this B5 report from those artifacts.",
            "- Keep threshold changes out of this attribution script; change upstream reports only through their own reviewed phase.",
            "",
        ]
    )


def _top_groups_table(rows: list[AttributionRow]) -> str:
    failing = [
        row
        for row in rows
        if row.category != "comparison_space" and (row.auc_smoke_fail_count or row.blocked_count or row.pca_fail_count or row.nn_ratio_fail_count or row.derivative_fail_count)
    ]
    if not failing:
        return "No failing groups were present in the consumed reports."
    selected = sorted(
        failing,
        key=lambda row: (
            row.auc_smoke_fail_count,
            row.blocked_count,
            row.pca_fail_count + row.nn_ratio_fail_count + row.derivative_fail_count,
            row.row_count,
        ),
        reverse=True,
    )[:15]
    lines = [
        "| category | value | rows | compared | blocked | AUC smoke | AUC stretch | PCA | NN ratio | derivative | max AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row.category}` | `{row.value}` | {row.row_count} | {row.compared_count} | {row.blocked_count} | "
            f"{row.auc_smoke_fail_count} | {row.auc_stretch_fail_count} | {row.pca_fail_count} | "
            f"{row.nn_ratio_fail_count} | {row.derivative_fail_count} | {_fmt(row.max_auc)} |"
        )
    return "\n".join(lines)


def _named_gap_table(rows: list[AttributionRow]) -> str:
    named = [row for row in rows if row.category == "named_gap"]
    if not named:
        return "No BEER, DIESEL, or CORN named gap rows were present."
    lines = [
        "| named gap | rows | compared | blocked | AUC smoke | AUC stretch | max AUC | median AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(named, key=lambda item: item.value):
        lines.append(
            f"| `{row.value}` | {row.row_count} | {row.compared_count} | {row.blocked_count} | "
            f"{row.auc_smoke_fail_count} | {row.auc_stretch_fail_count} | {_fmt(row.max_auc)} | {_fmt(row.median_auc)} |"
        )
    return "\n".join(lines)


def _b4_table(rows: list[AttributionRow]) -> str:
    b4_rows = [row for row in rows if row.category == "blocker_class" and row.value.startswith("B4:")]
    if not b4_rows:
        return "No B4 blocker row was present."
    lines = ["| blocker | blocked | raw smoke failures carried |", "|---|---:|---:|"]
    for row in b4_rows:
        lines.append(f"| `{row.value}` | {row.blocked_count} | {row.auc_smoke_fail_count} |")
    return "\n".join(lines)


def named_gap_tokens(row: B2Row) -> list[str]:
    dataset = row.dataset.upper()
    if not row.is_raw_authoritative:
        return []
    if not row.is_blocked and row.adversarial_auc is not None and row.adversarial_auc <= SMOKE_AUC_THRESHOLD:
        return []
    return [token for token in NAMED_GAP_TOKENS if token in dataset]


def blocked_class(row: B2Row) -> str:
    if not row.is_blocked:
        return ""
    reason = row.blocked_reason.strip()
    return reason.split(":", 1)[0] if reason else "blocked"


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _enabled_value(value: object) -> str:
    if value is True:
        return "enabled"
    if value is False:
        return "not_enabled"
    return "missing"


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int:
    text = _clean(value)
    if text == "":
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _parse_bool(value: str | None) -> bool:
    return _clean(value).lower() == "true"


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_fieldnames() -> Iterable[str]:
    return tuple(OUTPUT_FIELDS)


if __name__ == "__main__":
    main()
