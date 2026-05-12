"""Phase C encoder/TabPFN precheck over existing upstream gate artifacts.

This script is intentionally report-only. It reads CSV/Markdown gate outputs
from B3/B4/B5 plus the integration gate memo, then writes a single blocking
decision for Phase C manual review. It does not train, benchmark, import
TabPFN, or write checkpoints.
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
DEFAULT_INTEGRATION_MD = Path("bench/nirs_synthetic_pfn/reports/integration_gate_status.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.csv")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.md")

BLOCKED_UPSTREAM_STATUS = "BLOCKED_BY_UPSTREAM_REALISM_GATE"
BLOCKED_MISSING_STATUS = "BLOCKED_BY_MISSING_UPSTREAM_GATE_ARTIFACT"
READY_STATUS = "READY_FOR_MANUAL_C_PHASE"

OUTPUT_FIELDS = [
    "status",
    "b3_status",
    "b4_status",
    "b5_status",
    "integration_status",
    "raw_compared",
    "raw_smoke_failures",
    "raw_blocked",
    "raw_missing_auc",
    "train_allowed",
    "tabpfn_allowed",
    "checkpoint_allowed",
    "blocking_reasons",
]


@dataclass(frozen=True)
class RawGateCounts:
    raw_compared: int
    raw_smoke_failures: int
    raw_blocked: int
    raw_missing_auc: int

    @classmethod
    def empty(cls) -> RawGateCounts:
        return cls(raw_compared=0, raw_smoke_failures=0, raw_blocked=0, raw_missing_auc=0)


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

    @property
    def is_missing(self) -> bool:
        return bool(self.missing)


@dataclass(frozen=True)
class PrecheckSummary:
    status: str
    b3: ArtifactStatus
    b4: ArtifactStatus
    b5: ArtifactStatus
    integration: ArtifactStatus
    counts: RawGateCounts
    blocking_reasons: tuple[str, ...]
    csv_path: Path
    report_path: Path

    @property
    def train_allowed(self) -> bool:
        return False

    @property
    def tabpfn_allowed(self) -> bool:
        return False

    @property
    def checkpoint_allowed(self) -> bool:
        return False

    def as_csv_row(self) -> dict[str, str]:
        return {
            "status": self.status,
            "b3_status": self.b3.status,
            "b4_status": self.b4.status,
            "b5_status": self.b5.status,
            "integration_status": self.integration.status,
            "raw_compared": str(self.counts.raw_compared),
            "raw_smoke_failures": str(self.counts.raw_smoke_failures),
            "raw_blocked": str(self.counts.raw_blocked),
            "raw_missing_auc": str(self.counts.raw_missing_auc),
            "train_allowed": _bool_text(self.train_allowed),
            "tabpfn_allowed": _bool_text(self.tabpfn_allowed),
            "checkpoint_allowed": _bool_text(self.checkpoint_allowed),
            "blocking_reasons": ";".join(self.blocking_reasons),
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
        integration_md=args.integration_md,
        csv_path=args.csv,
        report_path=args.report,
    )
    print(f"status={summary.status}")
    print(f"b3_status={summary.b3.status}")
    print(f"b4_status={summary.b4.status}")
    print(f"b5_status={summary.b5.status}")
    print(f"integration_status={summary.integration.status}")
    print(f"raw_compared={summary.counts.raw_compared}")
    print(f"raw_smoke_failures={summary.counts.raw_smoke_failures}")
    print(f"raw_blocked={summary.counts.raw_blocked}")
    print(f"raw_missing_auc={summary.counts.raw_missing_auc}")
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
    integration_md: Path,
    csv_path: Path,
    report_path: Path,
) -> PrecheckSummary:
    b3_rows, b3_csv_missing = _read_csv_rows(b3_csv, "B3 CSV")
    b4_rows, b4_csv_missing = _read_csv_rows(b4_csv, "B4 CSV")
    b5_rows, b5_csv_missing = _read_csv_rows(b5_csv, "B5 CSV")

    b3_md_text, b3_md_missing = _read_text(b3_md, "B3 Markdown")
    b4_md_text, b4_md_missing = _read_text(b4_md, "B4 Markdown")
    b5_md_text, b5_md_missing = _read_text(b5_md, "B5 Markdown")
    integration_text, integration_missing = _read_text(integration_md, "integration gate Markdown")

    counts = _raw_counts_from_b4(b4_rows) or _raw_counts_from_b3(b3_rows)
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
    integration = ArtifactStatus(
        name="integration gate",
        csv_path=None,
        md_path=integration_md,
        status=_integration_status(md_text=integration_text, missing=bool(integration_missing)),
        missing=(integration_missing,) if integration_missing else (),
        sha256_md=_sha256_if_exists(integration_md),
    )

    blocking_reasons = _blocking_reasons(b3=b3, b4=b4, b5=b5, integration=integration)
    status = _overall_status(blocking_reasons)
    summary = PrecheckSummary(
        status=status,
        b3=b3,
        b4=b4,
        b5=b5,
        integration=integration,
        counts=counts,
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
            "# Encoder TabPFN Phase C Gate Precheck",
            "",
            "## Status",
            "",
            f"- Status: `{summary.status}`.",
            f"- B3 status: `{summary.b3.status}`.",
            f"- B4 status: `{summary.b4.status}`.",
            f"- B5 status: `{summary.b5.status}`.",
            f"- Integration gate status: `{summary.integration.status}`.",
            f"- Blocking reasons: `{';'.join(summary.blocking_reasons) or 'none'}`.",
            "",
            "## Raw Gate Counters",
            "",
            f"- raw_compared: {summary.counts.raw_compared}",
            f"- raw_smoke_failures: {summary.counts.raw_smoke_failures}",
            f"- raw_blocked: {summary.counts.raw_blocked}",
            f"- raw_missing_auc: {summary.counts.raw_missing_auc}",
            "",
            "## Phase C Permissions",
            "",
            "- train_allowed: `false`.",
            "- tabpfn_allowed: `false`.",
            "- checkpoint_allowed: `false`.",
            "- No encoder training.",
            "- No InfoNCE.",
            "- No CNN/patch transformer benchmark.",
            "- No TabPFN/PCA benchmark.",
            "- No checkpoint.",
            "- No transfer/integration claim.",
            "",
            "## Consumed Inputs",
            "",
            _artifact_table([summary.b3, summary.b4, summary.b5, summary.integration]),
            "",
            "## Safe Structural Contracts Only",
            "",
            "Existing safe structural contracts remain bench-side only: canonical latents, spectral view batches, same-latent multiview construction, prior task containers, context/query sampling, and multi-target task containers. These contracts do not lift the B3/B4/B5 or integration gates.",
            "",
            "## Next Manual Step",
            "",
            "Phase C can only be reopened by a human after upstream realism, transfer, attribution, and integration gate artifacts are reviewed and regenerated with passing evidence. This precheck itself never starts training or benchmarking.",
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
    integration: ArtifactStatus,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for artifact in (b3, b4, b5, integration):
        reasons.extend(f"missing_{item}" for item in artifact.missing)
    if b3.status == "NO-GO":
        reasons.append("B3_NO-GO")
    if b4.status.startswith("BLOCKED"):
        reasons.append(f"B4_{b4.status}")
    if b5.status == "BLOCKED_REPORT_ONLY":
        reasons.append("B5_BLOCKED_REPORT_ONLY")
    if integration.status == "NO-GO":
        reasons.append("integration_gate_NO-GO")
    return tuple(reasons)


def _overall_status(blocking_reasons: tuple[str, ...]) -> str:
    if any(reason.startswith("missing_") for reason in blocking_reasons):
        return BLOCKED_MISSING_STATUS
    if blocking_reasons:
        return BLOCKED_UPSTREAM_STATUS
    return READY_STATUS


def _b3_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    md_status = _status_from_markdown(md_text)
    if md_status == "NO-GO":
        return md_status
    counts = _raw_counts_from_b3(rows)
    if not counts.raw_compared or counts.raw_smoke_failures or counts.raw_blocked or counts.raw_missing_auc:
        return "NO-GO"
    if md_status:
        return md_status
    if any(_is_raw_authoritative(row) and _parse_bool(row.get("stretch_fail")) for row in rows):
        return "REVIEW"
    return "GO_DIAGNOSTIC_ONLY"


def _b4_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    statuses = [_clean(row.get("status")) for row in rows if _clean(row.get("status"))]
    blocked = [status for status in statuses if status.startswith("BLOCKED")]
    if blocked:
        return blocked[0]
    md_status = _status_from_markdown(md_text)
    if md_status:
        return md_status
    return statuses[0] if statuses else "UNKNOWN"


def _b5_status(*, rows: list[dict[str, str]], md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    statuses = [_clean(row.get("gate_status")) for row in rows if _clean(row.get("gate_status"))]
    if "BLOCKED_REPORT_ONLY" in statuses:
        return "BLOCKED_REPORT_ONLY"
    md_status = _status_from_markdown(md_text)
    if md_status:
        return md_status
    return statuses[0] if statuses else "UNKNOWN"


def _integration_status(*, md_text: str, missing: bool) -> str:
    if missing:
        return "MISSING"
    decision = _decision_from_markdown(md_text)
    if decision:
        return decision
    return _status_from_markdown(md_text) or "UNKNOWN"


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
    if "READY_FOR_MANUAL_C_PHASE" in upper:
        return "READY_FOR_MANUAL_C_PHASE"
    if upper.startswith("GO_DIAGNOSTIC_ONLY"):
        return "GO_DIAGNOSTIC_ONLY"
    if upper.startswith("PASS") or upper == "GO":
        return "GO"
    return upper.replace(" ", "_")


def _raw_counts_from_b4(rows: list[dict[str, str]]) -> RawGateCounts | None:
    for row in rows:
        if all(field in row for field in ("raw_compared", "raw_smoke_failures", "raw_blocked", "raw_missing_auc")):
            return RawGateCounts(
                raw_compared=_parse_int(row.get("raw_compared")),
                raw_smoke_failures=_parse_int(row.get("raw_smoke_failures")),
                raw_blocked=_parse_int(row.get("raw_blocked")),
                raw_missing_auc=_parse_int(row.get("raw_missing_auc")),
            )
    return None


def _raw_counts_from_b3(rows: list[dict[str, str]]) -> RawGateCounts:
    raw_rows = [row for row in rows if _is_raw_authoritative(row)]
    raw_compared = [row for row in raw_rows if _clean(row.get("status")) == "compared"]
    return RawGateCounts(
        raw_compared=len(raw_compared),
        raw_smoke_failures=sum(1 for row in raw_compared if _parse_bool(row.get("smoke_fail"))),
        raw_blocked=sum(1 for row in raw_rows if _clean(row.get("status")) == "blocked" or bool(_clean(row.get("blocked_class")))),
        raw_missing_auc=sum(1 for row in raw_compared if not _clean(row.get("adversarial_auc"))),
    )


def _is_raw_authoritative(row: dict[str, str]) -> bool:
    return _parse_bool(row.get("raw_authoritative")) or _clean(row.get("comparison_space")) == "raw"


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


def _parse_int(value: str | None) -> int:
    text = _clean(value)
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


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
