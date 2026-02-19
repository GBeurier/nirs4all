#!/usr/bin/env python3
"""Fail when unused-symbol lint debt (F401/F841) increases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = ["nirs4all/pipeline", "nirs4all/data"]
DEFAULT_BASELINE = ROOT / ".github/quality/ruff_unused_budget.json"
TARGET_RULES = ["F401", "F841"]

def _normalize_path(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    try:
        rel = p.relative_to(ROOT)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")

def _run_ruff(scope: list[str]) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *scope,
        "--select",
        ",".join(TARGET_RULES),
        "--isolated",
        "--output-format",
        "json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr or proc.stdout)
        raise RuntimeError(f"ruff invocation failed with exit code {proc.returncode}")
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return []
    return json.loads(stdout)

def _build_snapshot(findings: list[dict[str, Any]], scope: list[str]) -> dict[str, Any]:
    by_rule: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    for item in findings:
        code = str(item.get("code", ""))
        filename = _normalize_path(str(item.get("filename", "")))
        by_rule[code] += 1
        by_file[filename] += 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": scope,
        "rules": TARGET_RULES,
        "total": int(len(findings)),
        "by_rule": dict(sorted(by_rule.items())),
        "by_file": dict(sorted(by_file.items())),
    }

def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    current_total = int(current.get("total", 0))
    baseline_total = int(baseline.get("total", 0))
    if current_total > baseline_total:
        messages.append(
            f"total violations increased: {current_total} > {baseline_total}"
        )

    current_by_file = current.get("by_file", {})
    baseline_by_file = baseline.get("by_file", {})
    for file_path in sorted(current_by_file):
        current_count = int(current_by_file[file_path])
        baseline_count = int(baseline_by_file.get(file_path, 0))
        if current_count > baseline_count:
            messages.append(
                f"{file_path}: {current_count} > {baseline_count}"
            )

    current_by_rule = current.get("by_rule", {})
    baseline_by_rule = baseline.get("by_rule", {})
    for rule in TARGET_RULES:
        cur = int(current_by_rule.get(rule, 0))
        base = int(baseline_by_rule.get(rule, 0))
        if cur > base:
            messages.append(f"rule {rule} increased: {cur} > {base}")

    return messages

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to baseline JSON file.",
    )
    parser.add_argument(
        "--scope",
        nargs="+",
        default=DEFAULT_SCOPE,
        help="Paths to lint.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current counts to baseline file and exit.",
    )
    args = parser.parse_args()

    findings = _run_ruff(args.scope)
    snapshot = _build_snapshot(findings, args.scope)

    if args.write_baseline:
        _write_json(args.baseline, snapshot)
        print(
            f"Wrote unused-symbol baseline to {args.baseline} "
            f"(total={snapshot['total']})."
        )
        return 0

    baseline = _load_baseline(args.baseline)
    regressions = _compare(snapshot, baseline)

    print(
        f"Unused-symbol budget: current={snapshot['total']} "
        f"baseline={baseline.get('total', 0)}"
    )
    if regressions:
        print("Budget regressions detected:")
        for msg in regressions:
            print(f"- {msg}")
        return 1

    print("Unused-symbol budget check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
