"""bench/scenarios/validate_registry.py.

Smoke check that every entry in `bench/scenarios/model_registry.yaml`
resolves to an importable Python class. Intended for CI: returns
non-zero when any locked / strict-preset entry fails to import.

Usage:

    PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge \
    python bench/scenarios/validate_registry.py

Or, more conveniently (the script auto-prepends the standard local
PYTHONPATH if the imports would otherwise fail and the candidate paths
exist on disk):

    python bench/scenarios/validate_registry.py [--strict]

Flags:

    --strict       exit non-zero on any failure, including
                   `exhaustive_research`-only entries (default: only
                   locked / strict-preset entries are gating).
    --json         emit a JSON summary to stdout instead of human text.

Categories emitted per entry:

    OK             import + class lookup succeeded.
    SKIPPED        `not_runnable_in_production: true` — never imported.
    IMPORT_ERROR   `__import__(module)` raised.
    MISSING_CLASS  module imported but `getattr(module, model_class)` is None.

Exit status:
    0  every gating entry resolved (or only non-gating entries failed
       and `--strict` is off).
    1  at least one gating entry failed.

DECISION_PENDING_CODEX_REVIEW (sub-decision under D-C-006).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BENCH = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BENCH / "scenarios" / "model_registry.yaml"
LOCAL_PYTHONPATH_HINTS: tuple[Path, ...] = (
    BENCH / "AOM_v0" / "Ridge",
    BENCH / "AOM_v0",
    BENCH / "AOM_v0" / "multiview",
    BENCH.parent,                          # project root for `bench.X.Y.Z` namespace imports
)
GATING_PRESETS: frozenset[str] = frozenset({"fast_reliable", "strong_practical", "best_current"})


@dataclass(frozen=True)
class CheckResult:
    canonical_name: str
    module: str
    model_class: str
    maturity: str
    not_runnable_in_production: bool
    in_gating_preset: bool
    status: str  # OK / SKIPPED / IMPORT_ERROR / MISSING_CLASS
    error_message: str = ""


def ensure_local_pythonpath() -> None:
    """Prepend bench/AOM_v0/ + bench/AOM_v0/Ridge/ to sys.path if absent.

    The registry uses short-form imports (`aomridge.*`, `aompls.*`); they
    only resolve when those directories are on PYTHONPATH. This helper
    is best-effort — if the user already configured PYTHONPATH it is a
    no-op.
    """
    for hint in LOCAL_PYTHONPATH_HINTS:
        if hint.exists() and str(hint) not in sys.path:
            sys.path.insert(0, str(hint))


def gating_membership(presets: dict[str, Any]) -> dict[str, set[str]]:
    """Map preset -> set of canonical names that are members."""
    out: dict[str, set[str]] = {}
    for name, spec in (presets or {}).items():
        members = spec.get("members") if isinstance(spec, dict) else None
        if isinstance(members, list):
            out[name] = {str(m) for m in members}
    return out


def check_entry(entry: dict[str, Any], gating_names: set[str]) -> CheckResult:
    canonical_name = str(entry.get("canonical_name", ""))
    module = str(entry.get("module", ""))
    model_class = str(entry.get("model_class", ""))
    maturity = str(entry.get("maturity", "exploratory"))
    nrip = bool(entry.get("not_runnable_in_production", False))
    in_gating = canonical_name in gating_names

    if nrip:
        return CheckResult(
            canonical_name=canonical_name,
            module=module,
            model_class=model_class,
            maturity=maturity,
            not_runnable_in_production=True,
            in_gating_preset=in_gating,
            status="SKIPPED",
            error_message="not_runnable_in_production: import skipped",
        )

    try:
        mod = __import__(module, fromlist=[model_class])
    except Exception as exc:  # noqa: BLE001 - registry boundary
        return CheckResult(
            canonical_name=canonical_name,
            module=module,
            model_class=model_class,
            maturity=maturity,
            not_runnable_in_production=False,
            in_gating_preset=in_gating,
            status="IMPORT_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    cls = getattr(mod, model_class, None)
    if cls is None:
        return CheckResult(
            canonical_name=canonical_name,
            module=module,
            model_class=model_class,
            maturity=maturity,
            not_runnable_in_production=False,
            in_gating_preset=in_gating,
            status="MISSING_CLASS",
            error_message=f"module {module} has no attribute {model_class}",
        )

    return CheckResult(
        canonical_name=canonical_name,
        module=module,
        model_class=model_class,
        maturity=maturity,
        not_runnable_in_production=False,
        in_gating_preset=in_gating,
        status="OK",
    )


def fmt_table(results: list[CheckResult]) -> str:
    by_status: dict[str, list[CheckResult]] = {"OK": [], "SKIPPED": [], "IMPORT_ERROR": [], "MISSING_CLASS": []}
    for r in results:
        by_status.setdefault(r.status, []).append(r)
    lines: list[str] = []
    lines.append(f"Registry: {REGISTRY_PATH.relative_to(BENCH.parent)}")
    lines.append(f"Total entries: {len(results)}")
    for tag in ("OK", "SKIPPED", "IMPORT_ERROR", "MISSING_CLASS"):
        bucket = by_status.get(tag, [])
        lines.append(f"  {tag}: {len(bucket)}")
    lines.append("")
    failures = [r for r in results if r.status in {"IMPORT_ERROR", "MISSING_CLASS"}]
    if failures:
        lines.append("Failures:")
        for r in failures:
            mark = "GATING" if r.in_gating_preset else "non-gating"
            lines.append(
                f"  [{r.status:13s}] [{mark:10s}] {r.canonical_name:48s} "
                f"{r.module}:{r.model_class} — {r.error_message}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench/scenarios/validate_registry.py",
        description="Smoke-check registry imports.",
    )
    parser.add_argument("--strict", action="store_true", help="Treat any failure as gating, not just locked / strict-preset entries.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON summary to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_local_pythonpath()

    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        print(f"FATAL: registry {REGISTRY_PATH} has no 'models' key", file=sys.stderr)
        return 2

    presets = data.get("presets") or {}
    gating_names: set[str] = set()
    for name in GATING_PRESETS:
        gating_names.update(gating_membership(presets).get(name, set()))

    results: list[CheckResult] = [check_entry(entry, gating_names) for entry in data["models"]]

    failures_gating = [
        r for r in results if r.status in {"IMPORT_ERROR", "MISSING_CLASS"} and r.in_gating_preset
    ]
    failures_other = [
        r for r in results if r.status in {"IMPORT_ERROR", "MISSING_CLASS"} and not r.in_gating_preset
    ]

    exit_code = 1 if failures_gating or (args.strict and failures_other) else 0

    if args.json:
        payload = {
            "registry": str(REGISTRY_PATH.relative_to(BENCH.parent)),
            "total": len(results),
            "ok": sum(1 for r in results if r.status == "OK"),
            "skipped": sum(1 for r in results if r.status == "SKIPPED"),
            "failures_gating": len(failures_gating),
            "failures_other": len(failures_other),
            "exit_code": exit_code,
            "results": [
                {
                    "canonical_name": r.canonical_name,
                    "module": r.module,
                    "model_class": r.model_class,
                    "maturity": r.maturity,
                    "not_runnable_in_production": r.not_runnable_in_production,
                    "in_gating_preset": r.in_gating_preset,
                    "status": r.status,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(fmt_table(results))
        if failures_gating:
            print(f"\nGATING failures: {len(failures_gating)}", file=sys.stderr)
        if failures_other and not args.strict:
            print(f"Non-gating failures (informational): {len(failures_other)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
