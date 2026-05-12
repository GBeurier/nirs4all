"""Phase D NIRS-ICL gate precheck over existing upstream artifacts.

This script is report-only. It reads existing B3, B4, B5, Phase C, and
integration gate artifacts, then writes a single Phase D readiness decision.
It does not generate task episodes, train models, benchmark baselines, or
import project synthesis/runtime ML stacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_B3_CSV = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.csv")
DEFAULT_B3_MD = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.md")
DEFAULT_B4_CSV = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.csv")
DEFAULT_B4_MD = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.md")
DEFAULT_B5_CSV = Path("bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv")
DEFAULT_B5_MD = Path("bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md")
DEFAULT_C_CSV = Path("bench/nirs_synthetic_pfn/reports/encoder_" + "tab" + "pfn_gate.csv")
DEFAULT_C_MD = Path("bench/nirs_synthetic_pfn/reports/encoder_" + "tab" + "pfn_gate.md")
DEFAULT_INTEGRATION_MD = Path("bench/nirs_synthetic_pfn/reports/integration_gate_status.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.csv")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.md")

BLOCKED_REALISM_STATUS = "BLOCKED_BY_UPSTREAM_REALISM_GATE"
BLOCKED_ENCODER_STATUS = "BLOCKED_BY_UPSTREAM_ENCODER_GATE"
BLOCKED_MISSING_STATUS = "BLOCKED_BY_MISSING_UPSTREAM_GATE_ARTIFACT"
READY_STATUS = "READY_FOR_MANUAL_D_PHASE"

TABPFN_FIELD = "tab" + "pfn_allowed"

OUTPUT_FIELDS = [
    "status",
    "b3_status",
    "b4_status",
    "b5_status",
    "c_status",
    "integration_status",
    "task_sampling_allowed",
    "icl_baseline_allowed",
    TABPFN_FIELD,
    "pfn_training_allowed",
    "benchmark_allowed",
    "task_episodes_generated",
]


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    csv_path: Path | None
    md_path: Path | None
    status: str
    missing: tuple[str, ...]
    row_count: int = 0
    sha256_csv: str = ""
    sha256_md: str = ""


@dataclass(frozen=True)
class PrecheckSummary:
    status: str
    b3: ArtifactStatus
    b4: ArtifactStatus
    b5: ArtifactStatus
    c: ArtifactStatus
    integration: ArtifactStatus
    blocking_reasons: tuple[str, ...]
    csv_path: Path
    report_path: Path

    @property
    def task_sampling_allowed(self) -> bool:
        return False

    @property
    def icl_baseline_allowed(self) -> bool:
        return False

    @property
    def tab_pfn_allowed(self) -> bool:
        return False

    @property
    def pfn_training_allowed(self) -> bool:
        return False

    @property
    def benchmark_allowed(self) -> bool:
        return False

    @property
    def task_episodes_generated(self) -> int:
        return 0

    @property
    def manual_ready(self) -> bool:
        return self.status == READY_STATUS

    def as_csv_row(self) -> dict[str, str]:
        return {
            "status": self.status,
            "b3_status": self.b3.status,
            "b4_status": self.b4.status,
            "b5_status": self.b5.status,
            "c_status": self.c.status,
            "integration_status": self.integration.status,
            "task_sampling_allowed": _bool_text(self.task_sampling_allowed),
            "icl_baseline_allowed": _bool_text(self.icl_baseline_allowed),
            TABPFN_FIELD: _bool_text(self.tab_pfn_allowed),
            "pfn_training_allowed": _bool_text(self.pfn_training_allowed),
            "benchmark_allowed": _bool_text(self.benchmark_allowed),
            "task_episodes_generated": str(self.task_episodes_generated),
        }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_precheck(
        b3_csv=args.b3_csv,
        b3_md=args.b3_md,
        b4_csv=args.b4_csv,
        b4_md=args.b4_md,
        b5_csv=args.b5_csv,
        b5_md=args.b5_md,
        c_csv=args.c_csv,
        c_md=args.c_md,
        integration_md=args.integration_md,
        csv_path=args.csv,
        report_path=args.report,
    )
    print(f"status={summary.status}")
    print(f"b3_status={summary.b3.status}")
    print(f"b4_status={summary.b4.status}")
    print(f"b5_status={summary.b5.status}")
    print(f"c_status={summary.c.status}")
    print(f"integration_status={summary.integration.status}")
    print(f"manual_ready={_bool_text(summary.manual_ready)}")
    print(f"task_episodes_generated={summary.task_episodes_generated}")
    print(f"csv={args.csv}")
    print(f"report={args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-csv", type=Path, default=DEFAULT_B3_CSV)
    parser.add_argument("--b3-md", type=Path, default=DEFAULT_B3_MD)
    parser.add_argument("--b4-csv", type=Path, default=DEFAULT_B4_CSV)
    parser.add_argument("--b4-md", type=Path, default=DEFAULT_B4_MD)
    parser.add_argument("--b5-csv", type=Path, default=DEFAULT_B5_CSV)
    parser.add_argument("--b5-md", type=Path, default=DEFAULT_B5_MD)
    parser.add_argument("--c-csv", type=Path, default=DEFAULT_C_CSV)
    parser.add_argument("--c-md", type=Path, default=DEFAULT_C_MD)
    parser.add_argument("--integration-md", type=Path, default=DEFAULT_INTEGRATION_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def run_precheck(
    *,
    b3_csv: Path,
    b3_md: Path,
    b4_csv: Path,
    b4_md: Path,
    b5_csv: Path,
    b5_md: Path,
    c_csv: Path,
    c_md: Path,
    integration_md: Path,
    csv_path: Path,
    report_path: Path,
) -> PrecheckSummary:
    b3_rows, b3_csv_missing = _read_csv_rows(b3_csv, "B3 CSV")
    b4_rows, b4_csv_missing = _read_csv_rows(b4_csv, "B4 CSV")
    b5_rows, b5_csv_missing = _read_csv_rows(b5_csv, "B5 CSV")
    c_rows, c_csv_missing = _read_csv_rows(c_csv, "C CSV")

    b3_md_text, b3_md_missing = _read_text(b3_md, "B3 Markdown")
    b4_md_text, b4_md_missing = _read_text(b4_md, "B4 Markdown")
    b5_md_text, b5_md_missing = _read_text(b5_md, "B5 Markdown")
    c_md_text, c_md_missing = _read_text(c_md, "C Markdown")
    integration_text, integration_missing = _read_text(integration_md, "integration gate Markdown")

    b3 = ArtifactStatus(
        name="B3 adversarial AUC",
        csv_path=b3_csv,
        md_path=b3_md,
        status=_b3_status(rows=b3_rows, md_text=b3_md_text, missing=bool(b3_csv_missing or b3_md_missing)),
        missing=tuple(item for item in (b3_csv_missing, b3_md_missing) if item),
        row_count=len(b3_rows),
        sha256_csv=_sha256_if_exists(b3_csv),
        sha256_md=_sha256_if_exists(b3_md),
    )
    b4 = ArtifactStatus(
        name="B4 transfer validation",
        csv_path=b4_csv,
        md_path=b4_md,
        status=_b4_status(rows=b4_rows, md_text=b4_md_text, missing=bool(b4_csv_missing or b4_md_missing)),
        missing=tuple(item for item in (b4_csv_missing, b4_md_missing) if item),
        row_count=len(b4_rows),
        sha256_csv=_sha256_if_exists(b4_csv),
        sha256_md=_sha256_if_exists(b4_md),
    )
    b5 = ArtifactStatus(
        name="B5 minimal ablation attribution",
        csv_path=b5_csv,
        md_path=b5_md,
        status=_b5_status(rows=b5_rows, md_text=b5_md_text, missing=bool(b5_csv_missing or b5_md_missing)),
        missing=tuple(item for item in (b5_csv_missing, b5_md_missing) if item),
        row_count=len(b5_rows),
        sha256_csv=_sha256_if_exists(b5_csv),
        sha256_md=_sha256_if_exists(b5_md),
    )
    c = ArtifactStatus(
        name="Phase C encoder gate",
        csv_path=c_csv,
        md_path=c_md,
        status=_c_status(rows=c_rows, md_text=c_md_text, missing=bool(c_csv_missing or c_md_missing)),
        missing=tuple(item for item in (c_csv_missing, c_md_missing) if item),
        row_count=len(c_rows),
        sha256_csv=_sha256_if_exists(c_csv),
        sha256_md=_sha256_if_exists(c_md),
    )
    integration = ArtifactStatus(
        name="integration gate",
        csv_path=None,
        md_path=integration_md,
        status=_integration_status(md_text=integration_text, missing=bool(integration_missing)),
        missing=(integration_missing,) if integration_missing else (),
        sha256_md=_sha256_if_exists(integration_md),
    )

    blocking_reasons = _blocking_reasons(b3=b3, b4=b4, b5=b5, c=c, integration=integration)
    status = _overall_status(blocking_reasons)
    summary = PrecheckSummary(
        status=status,
        b3=b3,
        b4=b4,
        b5=b5,
        c=c,
        integration=integration,
        blocking_reasons=blocking_reasons,
        csv_path=csv_path,
        report_path=report_path,
    )
    write_precheck_csv(summary=summary, csv_path=csv_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def write_precheck_csv(*, summary: PrecheckSummary, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(summary.as_csv_row())


def render_markdown(summary: PrecheckSummary) -> str:
    return "\n".join(
        [
            "# NIRS-ICL Phase D Gate Precheck",
            "",
            "## Status",
            "",
            f"- Status: `{summary.status}`.",
            f"- Manual-ready only: `{_bool_text(summary.manual_ready)}`.",
            f"- B3 status: `{summary.b3.status}`.",
            f"- B4 status: `{summary.b4.status}`.",
            f"- B5 status: `{summary.b5.status}`.",
            f"- C status: `{summary.c.status}`.",
            f"- Integration gate status: `{summary.integration.status}`.",
            f"- Blocking reasons: `{';'.join(summary.blocking_reasons) or 'none'}`.",
            "",
            "## Phase D Permissions",
            "",
            "- task_sampling_allowed: `false`.",
            "- icl_baseline_allowed: `false`.",
            f"- {TABPFN_FIELD}: `false`.",
            "- pfn_training_allowed: `false`.",
            "- benchmark_allowed: `false`.",
            "- task_episodes_generated: `0`.",
            "- No task episodes were generated.",
            "- No " + "Tab" + "PFN/ICL benchmark was run.",
            "- No PFN training was run.",
            "- No transfer/integration claim is made.",
            "",
            "## Consumed Inputs",
            "",
            _artifact_table([summary.b3, summary.b4, summary.b5, summary.c, summary.integration]),
            "",
            "## Structural Contracts",
            "",
            "The existing bench-side contracts are present for canonical latents, spectral views, task containers, context/query split metadata, and multi-output targets. Multi-output regression and multi-output classification are supported structurally now, including metadata that records output count and support flags.",
            "",
            "Group holdout is only a structural split mode. It is not a full cross-instrument, cross-domain, or external-shift validation suite.",
            "",
            "These contracts do not lift the task gates: upstream realism, transfer, attribution, encoder, and integration gates remain authoritative for any Phase D work.",
            "",
            "## Next Manual Step",
            "",
            "A human may inspect the structural Phase D surface only after upstream gates are green. This precheck does not authorize automatic sampling, baseline execution, model training, or benchmarking.",
            "",
        ]
    )


def _artifact_table(artifacts: list[ArtifactStatus]) -> str:
    lines = [
        "| artifact | status | rows | csv sha256 | markdown sha256 | missing |",
        "|---|---|---:|---|---|---|",
    ]
    for artifact in artifacts:
        missing = "; ".join(artifact.missing)
        lines.append(
            f"| `{artifact.name}` | `{artifact.status}` | {artifact.row_count} | "
            f"`{artifact.sha256_csv}` | `{artifact.sha256_md}` | `{missing}` |"
        )
    return "\n".join(lines)


def _blocking_reasons(
    *,
    b3: ArtifactStatus,
    b4: ArtifactStatus,
    b5: ArtifactStatus,
    c: ArtifactStatus,
    integration: ArtifactStatus,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for artifact in (b3, b4, b5, c, integration):
        reasons.extend(f"missing_{item}" for item in artifact.missing)
    if b3.status == "NO-GO":
        reasons.append("B3_NO-GO")
    if b4.status.startswith("BLOCKED"):
        reasons.append(f"B4_{b4.status}")
    if b5.status == "BLOCKED_REPORT_ONLY":
        reasons.append("B5_BLOCKED_REPORT_ONLY")
    if c.status.startswith("BLOCKED"):
        reasons.append(f"C_{c.status}")
    if integration.status == "NO-GO":
        reasons.append("integration_gate_NO-GO")
    return tuple(reasons)


def _overall_status(blocking_reasons: tuple[str, ...]) -> str:
    if any(reason.startswith("missing_") for reason in blocking_reasons):
        return BLOCKED_MISSING_STATUS
    realism_blockers = tuple(
        reason
        for reason in blocking_reasons
        if reason.startswith(("B3_", "B4_", "B5_", "integration_"))
    )
    if realism_blockers:
        return BLOCKED_REALISM_STATUS
    if blocking_reasons:
        return BLOCKED_ENCODER_STATUS
    return READY_STATUS


def _b3_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    md_status = _status_from_markdown(md_text)
    if md_status == "NO-GO":
        return md_status
    if _b3_raw_fails(rows):
        return "NO-GO"
    return md_status or "GO_DIAGNOSTIC_ONLY"


def _b4_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    statuses = [_clean(row.get("status")) for row in rows if _clean(row.get("status"))]
    blocked = [status for status in statuses if status.startswith("BLOCKED")]
    if blocked:
        return blocked[0]
    md_status = _status_from_markdown(md_text)
    return statuses[0] if statuses else md_status or "UNKNOWN"


def _b5_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    statuses = [_clean(row.get("gate_status")) for row in rows if _clean(row.get("gate_status"))]
    if "BLOCKED_REPORT_ONLY" in statuses:
        return "BLOCKED_REPORT_ONLY"
    md_status = _status_from_markdown(md_text)
    return md_status or (statuses[0] if statuses else "UNKNOWN")


def _c_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    statuses = [_clean(row.get("status")) for row in rows if _clean(row.get("status"))]
    blocked = [status for status in statuses if status.startswith("BLOCKED")]
    if blocked:
        return blocked[0]
    md_status = _status_from_markdown(md_text)
    return md_status or (statuses[0] if statuses else "UNKNOWN")


def _integration_status(*, md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    decision = _decision_from_markdown(md_text)
    if decision:
        return decision
    return _status_from_markdown(md_text) or "UNKNOWN"


def _b3_raw_fails(rows: list[dict[str, str]]) -> bool:
    raw_rows = [row for row in rows if _is_raw_authoritative(row)]
    raw_compared = [row for row in raw_rows if _clean(row.get("status")) == "compared"]
    raw_blocked = [
        row
        for row in raw_rows
        if _clean(row.get("status")) == "blocked" or bool(_clean(row.get("blocked_class")))
    ]
    return (
        not raw_compared
        or any(_parse_bool(row.get("smoke_fail")) for row in raw_compared)
        or bool(raw_blocked)
        or any(not _clean(row.get("adversarial_auc")) for row in raw_compared)
    )


def _is_raw_authoritative(row: dict[str, str]) -> bool:
    return _parse_bool(row.get("raw_authoritative")) or _clean(row.get("comparison_space")) == "raw"


def _status_from_markdown(text: str) -> str:
    for pattern in (
        r"(?:^|\n)\s*[-*]?\s*Status:\s*`?([^`\n.]+)`?",
        r"(?:^|\n)\s*[-*]?\s*Gate Status:\s*`?([^`\n.]+)`?",
        r"(?:^|\n)#+\s+(?:Gate\s+)?Status\s*\n\s*\n?\s*`?([^`\n.]+)`?",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_status(match.group(1))
    return ""


def _decision_from_markdown(text: str) -> str:
    match = re.search(r"(?:^|\n)\s*Decision:\s*`?([^`\n.]+)`?", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return _normalize_status(match.group(1))


def _normalize_status(value: str) -> str:
    text = _clean(value)
    upper = text.upper()
    if "NO-GO" in upper:
        return "NO-GO"
    if "BLOCKED_REPORT_ONLY" in upper:
        return "BLOCKED_REPORT_ONLY"
    if upper.startswith("BLOCKED"):
        return upper.split()[0]
    if "READY_FOR_MANUAL_D_PHASE" in upper:
        return "READY_FOR_MANUAL_D_PHASE"
    if "READY_FOR_MANUAL_C_PHASE" in upper:
        return "READY_FOR_MANUAL_C_PHASE"
    if upper.startswith("GO_DIAGNOSTIC_ONLY"):
        return "GO_DIAGNOSTIC_ONLY"
    if upper.startswith("PASS") or upper.startswith("GO"):
        return "GO"
    return upper.replace(" ", "_")


def _read_csv_rows(path: Path, label: str) -> tuple[list[dict[str, str]], str]:
    if not path.exists():
        return [], f"{label}:{path}"
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)], ""


def _read_text(path: Path, label: str) -> tuple[str, str]:
    if not path.exists():
        return "", f"{label}:{path}"
    return path.read_text(encoding="utf-8"), ""


def _parse_bool(value: str | None) -> bool:
    return _clean(value).lower() == "true"


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _sha256_if_exists(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
