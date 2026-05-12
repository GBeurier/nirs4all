"""Standalone Phase B3 adversarial AUC audit from the existing B2 CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

DEFAULT_B2_CSV = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.csv")
SMOKE_AUC_THRESHOLD = 0.85
STRETCH_AUC_THRESHOLD = 0.70
NAMED_GAP_TOKENS = ("BEER", "DIESEL", "CORN")

OUTPUT_FIELDS = [
    "dataset",
    "source",
    "task",
    "preset",
    "comparison_space",
    "status",
    "adversarial_auc",
    "adversarial_auc_std",
    "smoke_fail",
    "stretch_fail",
    "companion_failures",
    "blocked_class",
    "gate_basis",
    "raw_authoritative",
    "failure_taxonomy",
]


@dataclass(frozen=True)
class B2Row:
    dataset: str
    source: str
    task: str
    preset: str
    comparison_space: str
    status: str
    adversarial_auc: float | None
    adversarial_auc_text: str
    adversarial_auc_std: float | None
    adversarial_auc_std_text: str
    provisional_decision: str
    blocked_reason: str

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> B2Row:
        auc_text = _clean(row.get("adversarial_auc", ""))
        auc_std_text = _clean(row.get("adversarial_auc_std", ""))
        return cls(
            dataset=_clean(row.get("dataset", "")),
            source=_clean(row.get("source", "")),
            task=_clean(row.get("task", "")),
            preset=_clean(row.get("synthetic_preset", "")) or _clean(row.get("preset", "")),
            comparison_space=_clean(row.get("comparison_space", "")),
            status=_clean(row.get("status", "")),
            adversarial_auc=_parse_float(auc_text),
            adversarial_auc_text=auc_text,
            adversarial_auc_std=_parse_float(auc_std_text),
            adversarial_auc_std_text=auc_std_text,
            provisional_decision=_clean(row.get("provisional_decision", "")),
            blocked_reason=_clean(row.get("blocked_reason", "")),
        )


@dataclass(frozen=True)
class B2Provenance:
    path: Path
    sha256: str
    row_count: int
    raw_rows: int
    raw_compared_rows: int
    snv_rows: int
    snv_compared_rows: int
    legacy_raw_rows: int = 0
    calibrated_diagnostic_rows: int = 0


@dataclass(frozen=True)
class AdversarialAucSummary:
    gate_status: str
    rows: list[B2Row]
    provenance: B2Provenance
    raw_compared: list[B2Row]
    raw_blocked: list[B2Row]
    raw_missing_auc: list[B2Row]
    raw_smoke_failures: list[B2Row]
    raw_stretch_failures: list[B2Row]
    snv_compared: list[B2Row]
    snv_smoke_failures: list[B2Row]
    taxonomy_counts: Counter[str]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_report(b2_csv=args.b2_csv, report_path=args.report, csv_path=args.csv)
    print(f"status={summary.gate_status}")
    print(f"raw_compared={len(summary.raw_compared)}")
    print(f"raw_smoke_failures={len(summary.raw_smoke_failures)}")
    print(f"raw_blocked={len(summary.raw_blocked)}")
    print(f"raw_missing_auc={len(summary.raw_missing_auc)}")
    print(f"report={args.report}")
    print(f"csv={args.csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2-csv", type=Path, default=DEFAULT_B2_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser


def run_report(*, b2_csv: Path, report_path: Path, csv_path: Path) -> AdversarialAucSummary:
    rows = load_b2_rows(b2_csv)
    summary = build_summary(rows=rows, b2_csv=b2_csv)
    write_adversarial_csv(summary=summary, csv_path=csv_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary=summary, report_path=report_path, csv_path=csv_path), encoding="utf-8")
    return summary


def load_b2_rows(path: Path) -> list[B2Row]:
    if not path.exists():
        raise FileNotFoundError(f"B2 CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [B2Row.from_csv(dict(row)) for row in reader]


def build_summary(*, rows: list[B2Row], b2_csv: Path) -> AdversarialAucSummary:
    raw_rows = [row for row in rows if row.comparison_space == "uncalibrated_raw"]
    raw_compared = [row for row in raw_rows if row.status == "compared"]
    raw_blocked = [row for row in raw_rows if row.status == "blocked"]
    raw_missing_auc = [row for row in raw_compared if row.adversarial_auc is None]
    raw_smoke_failures = [row for row in raw_compared if row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD]
    raw_stretch_failures = [row for row in raw_compared if row.adversarial_auc is not None and row.adversarial_auc > STRETCH_AUC_THRESHOLD]
    snv_compared = [row for row in rows if row.comparison_space == "snv" and row.status == "compared"]
    snv_smoke_failures = [row for row in snv_compared if row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD]

    if not raw_compared or raw_smoke_failures or raw_missing_auc or raw_blocked:
        gate_status = "NO-GO"
    elif raw_stretch_failures:
        gate_status = "REVIEW"
    else:
        gate_status = "GO_DIAGNOSTIC_ONLY"

    taxonomy_counts = Counter(failure_taxonomy(row) for row in rows)
    provenance = B2Provenance(
        path=b2_csv,
        sha256=_sha256(b2_csv),
        row_count=len(rows),
        raw_rows=len(raw_rows),
        raw_compared_rows=len(raw_compared),
        snv_rows=sum(1 for row in rows if row.comparison_space == "snv"),
        snv_compared_rows=len(snv_compared),
        legacy_raw_rows=sum(1 for row in rows if row.comparison_space == "raw"),
        calibrated_diagnostic_rows=sum(
            1 for row in rows if row.comparison_space == "calibrated_raw_diagnostic"
        ),
    )
    return AdversarialAucSummary(
        gate_status=gate_status,
        rows=rows,
        provenance=provenance,
        raw_compared=raw_compared,
        raw_blocked=raw_blocked,
        raw_missing_auc=raw_missing_auc,
        raw_smoke_failures=raw_smoke_failures,
        raw_stretch_failures=raw_stretch_failures,
        snv_compared=snv_compared,
        snv_smoke_failures=snv_smoke_failures,
        taxonomy_counts=taxonomy_counts,
    )


def write_adversarial_csv(*, summary: AdversarialAucSummary, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(adversarial_csv_row(row))


def adversarial_csv_row(row: B2Row) -> dict[str, str]:
    return {
        "dataset": row.dataset,
        "source": row.source,
        "task": row.task,
        "preset": row.preset,
        "comparison_space": row.comparison_space,
        "status": row.status,
        "adversarial_auc": row.adversarial_auc_text,
        "adversarial_auc_std": row.adversarial_auc_std_text,
        "smoke_fail": _bool_text(row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD),
        "stretch_fail": _bool_text(row.adversarial_auc is not None and row.adversarial_auc > STRETCH_AUC_THRESHOLD),
        "companion_failures": companion_failures(row),
        "blocked_class": blocked_class(row),
        "gate_basis": _gate_basis(row),
        "raw_authoritative": _bool_text(row.comparison_space == "uncalibrated_raw"),
        "failure_taxonomy": failure_taxonomy(row),
    }


def render_markdown(*, summary: AdversarialAucSummary, report_path: Path, csv_path: Path) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp04_adversarial_auc.py "
        f"--b2-csv {summary.provenance.path} --report {report_path} --csv {csv_path}"
    )
    lines = [
        "# Standalone Adversarial AUC Audit",
        "",
        "## Objective",
        "",
        "Consume the existing Phase B2 CSV and emit a bench-only adversarial AUC audit without regenerating datasets, recomputing AUC, changing thresholds, or presenting B2 as passed.",
        "",
        "## Exact Command",
        "",
        f"`{command}`",
        "",
        "## B2 Provenance",
        "",
        f"- Input CSV: `{summary.provenance.path}`",
        f"- SHA-256: `{summary.provenance.sha256}`",
        f"- B2 rows consumed: {summary.provenance.row_count}",
        f"- Uncalibrated_raw rows (authoritative): {summary.provenance.raw_rows}; uncalibrated_raw compared rows: {summary.provenance.raw_compared_rows}",
        f"- Calibrated_raw_diagnostic rows: {summary.provenance.calibrated_diagnostic_rows} (diagnostic only, never authoritative)",
        f"- SNV rows: {summary.provenance.snv_rows}; SNV compared rows: {summary.provenance.snv_compared_rows} (diagnostic only)",
        f"- Legacy `raw` rows present: {summary.provenance.legacy_raw_rows} (never authoritative; ignored by the gate and forces NO-GO when no uncalibrated_raw row is compared).",
        "- The B2 CSV is read-only input for this report.",
        "",
        "## No Integration Or Transfer Claim",
        "",
        "This artifact is bench-only. It does not claim integration readiness, downstream transfer benefit, or a B2 pass. `GO_DIAGNOSTIC_ONLY` means only that this standalone diagnostic has no raw smoke failure or raw evidence gap.",
        "",
        "## Raw Authoritative Gate",
        "",
        f"- Status: `{summary.gate_status}`",
        "- The only authoritative lane is `comparison_space == \"uncalibrated_raw\"`. Legacy `raw` rows, `calibrated_raw_diagnostic` rows, and `snv` rows can never pass the gate on their own.",
        "- If no `uncalibrated_raw` row is compared, the gate fails closed with `NO-GO`, even if legacy `raw` rows look healthy.",
        f"- Smoke threshold: uncalibrated_raw adversarial AUC must be <= {SMOKE_AUC_THRESHOLD}.",
        f"- Stretch threshold: uncalibrated_raw adversarial AUC should be <= {STRETCH_AUC_THRESHOLD}.",
        f"- Uncalibrated_raw smoke failures: {len(summary.raw_smoke_failures)}/{len(summary.raw_compared)} compared uncalibrated_raw rows.",
        f"- Uncalibrated_raw stretch failures: {len(summary.raw_stretch_failures)}/{len(summary.raw_compared)} compared uncalibrated_raw rows.",
        f"- Uncalibrated_raw compared rows with missing adversarial AUC: {len(summary.raw_missing_auc)}.",
        f"- Uncalibrated_raw blocked evidence gaps: {len(summary.raw_blocked)}.",
        _gate_reason(summary),
        "",
        "## SNV Diagnostic",
        "",
        f"- SNV smoke failures: {len(summary.snv_smoke_failures)}/{len(summary.snv_compared)} compared SNV rows.",
        "- SNV is second-order diagnostic evidence only. SNV cannot override raw failures, raw missing AUC values, or raw blocked rows. The same applies to `calibrated_raw_diagnostic`.",
        "",
        "## Grouped Summaries",
        "",
        _grouped_summary_table(summary.rows),
        "",
        "## Failure Taxonomy",
        "",
        _taxonomy_table(summary.taxonomy_counts),
        "",
        "## Named B2 Gaps",
        "",
        _named_gap_section(summary.rows),
        "",
        "## Blocked Rows",
        "",
        _blocked_rows_table(summary.raw_blocked),
        "",
        "## Top Raw AUC Failures",
        "",
        _top_raw_failures_table(summary.raw_smoke_failures),
        "",
        "## Outputs",
        "",
        f"- Row-level CSV: `{csv_path}`",
        f"- Markdown report: `{report_path}`",
        "",
    ]
    return "\n".join(lines)


def failure_taxonomy(row: B2Row) -> str:
    if row.comparison_space == "uncalibrated_raw" and row.status == "blocked":
        return "raw_blocked_evidence_gap"
    if row.status == "compared" and row.adversarial_auc is None:
        if row.comparison_space == "uncalibrated_raw":
            return "raw_missing_auc"
        if row.comparison_space == "snv":
            return "snv_missing_auc"
        return f"{row.comparison_space or 'unknown'}_missing_auc"
    if row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD:
        prefix = "raw" if row.comparison_space == "uncalibrated_raw" else (row.comparison_space or "unknown")
        return f"{prefix}_smoke_high_auc"
    if row.adversarial_auc is not None and row.adversarial_auc > STRETCH_AUC_THRESHOLD:
        prefix = "raw" if row.comparison_space == "uncalibrated_raw" else (row.comparison_space or "unknown")
        return f"{prefix}_stretch_high_auc"
    if row.status != "compared":
        return f"{row.comparison_space or 'unknown'}_{row.status or 'unknown'}"
    return "pass"


def _gate_basis(row: B2Row) -> str:
    if row.comparison_space == "uncalibrated_raw":
        return "raw_authoritative"
    if row.comparison_space == "calibrated_raw_diagnostic":
        return "calibrated_raw_diagnostic_only"
    if row.comparison_space == "snv":
        return "snv_diagnostic_only"
    if row.comparison_space == "raw":
        return "legacy_raw_not_authoritative"
    return "unknown_not_authoritative"


def companion_failures(row: B2Row) -> str:
    prefix = "provisional_review:"
    if not row.provisional_decision.startswith(prefix):
        return ""
    failures = [
        failure
        for failure in row.provisional_decision[len(prefix):].split(",")
        if failure and failure != "adversarial_auc"
    ]
    return ",".join(failures)


def blocked_class(row: B2Row) -> str:
    if row.status != "blocked":
        return ""
    reason = row.blocked_reason.strip()
    return reason.split(":", 1)[0] if reason else "blocked"


def _gate_reason(summary: AdversarialAucSummary) -> str:
    if summary.gate_status == "NO-GO":
        reasons: list[str] = []
        if not summary.raw_compared:
            reasons.append("no uncalibrated_raw compared rows (legacy `raw`, `calibrated_raw_diagnostic`, and `snv` cannot substitute)")
        if summary.raw_smoke_failures:
            reasons.append("one or more uncalibrated_raw AUC values exceed the smoke threshold")
        if summary.raw_missing_auc:
            reasons.append("one or more uncalibrated_raw compared rows have missing AUC")
        if summary.raw_blocked:
            reasons.append("one or more uncalibrated_raw rows are blocked evidence gaps")
        return "- Gate basis: raw authoritative NO-GO because " + "; ".join(reasons) + "."
    if summary.gate_status == "REVIEW":
        return "- Gate basis: uncalibrated_raw smoke passes with no blocked evidence gaps, but at least one uncalibrated_raw row exceeds the stretch threshold."
    return "- Gate basis: uncalibrated_raw smoke passes with no blocked evidence gaps; use only as a diagnostic status."


def _grouped_summary_table(rows: list[B2Row]) -> str:
    raw_rows = [row for row in rows if row.comparison_space == "uncalibrated_raw"]
    groups: dict[tuple[str, str, str], list[B2Row]] = {}
    for row in raw_rows:
        groups.setdefault((row.source, row.preset, row.task), []).append(row)
    if not groups:
        return "No raw rows were present."
    lines = [
        "| source | preset | task | raw rows | compared | blocked | smoke fails | stretch fails | max raw AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (source, preset, task), group_rows in sorted(groups.items()):
        compared = [row for row in group_rows if row.status == "compared"]
        aucs = [row.adversarial_auc for row in compared if row.adversarial_auc is not None]
        lines.append(
            f"| `{source}` | `{preset}` | `{task}` | {len(group_rows)} | {len(compared)} | "
            f"{sum(1 for row in group_rows if row.status == 'blocked')} | "
            f"{sum(1 for row in compared if row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD)} | "
            f"{sum(1 for row in compared if row.adversarial_auc is not None and row.adversarial_auc > STRETCH_AUC_THRESHOLD)} | "
            f"{_fmt(max(aucs) if aucs else None)} |"
        )
    return "\n".join(lines)


def _taxonomy_table(counts: Counter[str]) -> str:
    if not counts:
        return "No rows were available for taxonomy."
    lines = ["| taxonomy | rows |", "|---|---:|"]
    for name, count in sorted(counts.items()):
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _named_gap_section(rows: list[B2Row]) -> str:
    raw_rows = [row for row in rows if row.comparison_space == "uncalibrated_raw"]
    lines: list[str] = []
    for token in NAMED_GAP_TOKENS:
        matches = [
            row
            for row in raw_rows
            if token in row.dataset.upper() and (row.status == "blocked" or row.adversarial_auc is None or (row.adversarial_auc is not None and row.adversarial_auc > SMOKE_AUC_THRESHOLD))
        ]
        if matches:
            datasets = ", ".join(f"`{row.dataset}` ({_fmt(row.adversarial_auc)})" for row in matches)
            lines.append(f"- {token}: {datasets}")
    if not lines:
        return "No failing BEER, DIESEL, or CORN raw rows were present."
    return "\n".join(lines)


def _blocked_rows_table(rows: list[B2Row]) -> str:
    if not rows:
        return "No raw blocked rows."
    lines = [
        "| dataset | source | task | preset | blocked class | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.dataset}` | `{row.source}` | `{row.task}` | `{row.preset}` | "
            f"`{blocked_class(row)}` | {row.blocked_reason} |"
        )
    return "\n".join(lines)


def _top_raw_failures_table(rows: list[B2Row]) -> str:
    if not rows:
        return "No raw smoke failures."
    lines = [
        "| dataset | source | task | preset | adversarial AUC | companion failures |",
        "|---|---|---|---|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: item.adversarial_auc if item.adversarial_auc is not None else -1.0, reverse=True)[:25]:
        lines.append(
            f"| `{row.dataset}` | `{row.source}` | `{row.task}` | `{row.preset}` | "
            f"{_fmt(row.adversarial_auc)} | `{companion_failures(row)}` |"
        )
    return "\n".join(lines)


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clean(value: str | None) -> str:
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


if __name__ == "__main__":
    main()
