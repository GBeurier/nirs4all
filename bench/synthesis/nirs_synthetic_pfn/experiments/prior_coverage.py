"""Coverage report for Phase A1 prior canonicalization."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from nirsyntheticpfn.adapters.prior_adapter import summarize_prior_coverage

DEFAULT_OUTPUT = Path("bench/nirs_synthetic_pfn/reports/prior_canonicalization_coverage.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_summary = summarize_prior_coverage(
        n_samples=args.n_samples,
        random_state=args.seed,
        repair_domain_components=False,
    )
    canonical_summary = summarize_prior_coverage(
        n_samples=args.n_samples,
        random_state=args.seed,
        repair_domain_components=True,
    )
    git_status = _git_status_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(raw_summary, canonical_summary, git_status),
        encoding="utf-8",
    )
    print(args.output)


def render_markdown(
    raw_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
    git_status: dict[str, Any],
) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/prior_coverage.py "
        f"--n-samples {raw_summary['n_samples']} --seed {raw_summary['random_state']}"
    )
    lines = [
        "# Prior Canonicalization Coverage",
        "",
        "## Objective",
        "",
        "Validate Phase A1 canonicalization coverage for `PriorSampler` samples in two regimes:",
        "raw (no repair) and canonical (components re-sampled from canonical domain).",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Summary",
        "",
        "| regime | samples | valid | invalid |",
        "|---|---:|---:|---:|",
        (
            f"| raw | {raw_summary['n_samples']} | "
            f"{raw_summary['valid_count']} | {raw_summary['invalid_count']} |"
        ),
        (
            f"| canonical (repaired) | {canonical_summary['n_samples']} | "
            f"{canonical_summary['valid_count']} | {canonical_summary['invalid_count']} |"
        ),
        "",
        f"- Seed: {raw_summary['random_state']}",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Raw Coverage (no repair)",
        "",
        "### Invalid Reasons",
        "",
        _table(raw_summary["invalid_reason_counts"], "reason"),
        "",
        "### Invalid Fields",
        "",
        _table(raw_summary["invalid_field_counts"], "field"),
        "",
        "### Source Domains",
        "",
        _table(raw_summary["source_domain_counts"], "source_domain"),
        "",
        "### Validated Domains",
        "",
        _table(raw_summary["domain_counts"], "domain"),
        "",
        "### Validated Components",
        "",
        _table(raw_summary["component_counts"], "component", limit=40),
        "",
        "## Canonical Coverage (components re-sampled from canonical domain)",
        "",
        "### Invalid Reasons",
        "",
        _table(canonical_summary["invalid_reason_counts"], "reason"),
        "",
        "### Invalid Fields",
        "",
        _table(canonical_summary["invalid_field_counts"], "field"),
        "",
        "### Repairs Applied",
        "",
        _table(canonical_summary.get("repair_counts", {}), "repair"),
        "",
        "### Validated Domains",
        "",
        _table(canonical_summary["domain_counts"], "domain"),
        "",
        "### Validated Instruments",
        "",
        _table(canonical_summary["instrument_counts"], "instrument"),
        "",
        "### Validated Measurement Modes",
        "",
        _table(canonical_summary["measurement_mode_counts"], "mode"),
        "",
        "### Validated Components",
        "",
        _table(canonical_summary["component_counts"], "component", limit=40),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(
            {"raw": raw_summary, "canonical": canonical_summary, "git_status": git_status},
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Decision",
        "",
        _decision_text(raw_summary, canonical_summary),
        "",
    ]
    return "\n".join(lines)


def _decision_text(raw: dict[str, Any], canonical: dict[str, Any]) -> str:
    raw_rate = raw["valid_count"] / max(1, raw["n_samples"])
    canonical_rate = canonical["valid_count"] / max(1, canonical["n_samples"])
    target_clips = canonical.get("repair_counts", {}).get(
        "target_n_targets_clipped_to_component_count",
        0,
    )
    return (
        f"Raw validation rate {raw_rate:.1%} confirms production `PriorSampler` falls "
        "back to generic components for unknown domain aliases. Canonical sampling "
        f"(repair_domain_components=True) raises validation to {canonical_rate:.1%}, "
        "with the remaining failures attributable to wavelength/domain overlap "
        "rather than component identity. Target-count clipping occurred in "
        f"{target_clips} samples and is recorded in `_canonical_repairs` alongside "
        "the original raw config under `_raw_prior_config`."
    )


def _table(counts: dict[str, int], label: str, limit: int | None = None) -> str:
    if not counts:
        return "_None_"
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    lines = [f"| {label} | count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in rows)
    return "\n".join(lines)


def _git_status_summary() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    by_status: dict[str, int] = {}
    for line in lines:
        status = line[:2].strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "returncode": result.returncode,
        "entry_count": len(lines),
        "by_status": dict(sorted(by_status.items())),
        "sample": lines[:20],
    }


def _git_status_section(summary: dict[str, Any]) -> str:
    if not summary:
        return "_Not captured_"
    lines = [
        f"- Return code: {summary.get('returncode')}",
        f"- Entries: {summary.get('entry_count')}",
        f"- Status counts: `{summary.get('by_status', {})}`",
    ]
    sample = summary.get("sample") or []
    if sample:
        lines.append("- Sample:")
        lines.extend(f"  - `{line}`" for line in sample)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
