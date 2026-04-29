"""Phase B1 prior predictive checks for the A2 smoke preset datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from exp00_smoke_prior_dataset import PRESETS, _preset_source
from nirsyntheticpfn.adapters.builder_adapter import (
    PriorDatasetAdapterError,
    SyntheticDatasetRun,
    build_synthetic_dataset_run,
)
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_prior_config
from nirsyntheticpfn.evaluation.prior_checks import (
    PHASE_A_GATE_OVERRIDE,
    PriorPredictiveValidation,
    validate_prior_predictive_run,
)

DEFAULT_OUTPUT = Path("bench/nirs_synthetic_pfn/reports/prior_predictive_checks.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = run_prior_predictive_checks(n_samples=args.n_samples, seed=args.seed)
    git_status = _git_status_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(results, args.n_samples, args.seed, git_status),
        encoding="utf-8",
    )
    print(args.output)


def run_prior_predictive_checks(*, n_samples: int, seed: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, preset in enumerate(PRESETS):
        domain_alias, target_type, target_size = preset
        run_seed = seed + idx
        source = _preset_source(
            domain_alias,
            target_type=target_type,
            target_size=target_size,
            seed=run_seed,
        )
        try:
            record = canonicalize_prior_config(source)
            run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=run_seed)
            validation = validate_prior_predictive_run(run, preset=domain_alias)
            results.append(_success_result(domain_alias, run, validation))
        except PriorDatasetAdapterError as exc:
            results.append(_failure_result(domain_alias, "a2_dataset_adapter", exc.validation_summary))
        except Exception as exc:
            results.append(_failure_result(
                domain_alias,
                "experiment_exception",
                {
                    "status": "failed",
                    "failures": [
                        {
                            "reason": "experiment_exception",
                            "field": "experiment",
                            "message": str(exc),
                        }
                    ],
                },
            ))
    return results


def render_markdown(
    results: list[dict[str, Any]],
    n_samples: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp01_prior_predictive_checks.py "
        f"--n-samples {n_samples} --seed {seed}"
    )
    passed = sum(1 for result in results if result["validation_status"] == "passed")
    blocked = len(results) - passed
    lines = [
        "# Prior Predictive Checks",
        "",
        "## Objective",
        "",
        "Run B1 prior predictive checks on the 10 A2 smoke preset datasets and emit an explicit downstream training status.",
        "",
        "## Phase A Gate Override",
        "",
        f"- `phase_a_gate_override`: `{PHASE_A_GATE_OVERRIDE}`",
        "- A3 fitted-only real-fit gate remains scientifically failed/blocked as documented in `real_fit_adapter_smoke.md`; B1 is continued by explicit user instruction.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Config",
        "",
        f"- Seed base: {seed}",
        f"- Samples per dataset: {n_samples}",
        "- Presets: A2 smoke presets from `exp00_smoke_prior_dataset.py`",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Preset Report Table",
        "",
        "| preset | domain | target | X shape | B1 status | downstream training | blocking checks | key spectral metrics |",
        "|---|---|---|---:|---|---|---|---|",
    ]

    for result in results:
        metrics = result.get("key_metrics", {})
        spectral = (
            f"SNR={_fmt(metrics.get('median_snr'))}; "
            f"d1={_fmt(metrics.get('median_first_derivative_std'))}; "
            f"curv={_fmt(metrics.get('median_baseline_curvature'))}; "
            f"peaks={_fmt(metrics.get('median_peak_density_per_100nm'))}"
        )
        lines.append(
            f"| `{result['preset']}` | `{result.get('domain', '')}` | `{result.get('target_type', '')}` | "
            f"`{result.get('X_shape', '')}` | `{result['validation_status']}` | "
            f"`{result['downstream_training_status']}` | "
            f"`{', '.join(result.get('blocking_checks', [])) or 'none'}` | {spectral} |"
        )

    lines.extend([
        "",
        "## Check Coverage",
        "",
        "| preset | concentrations | target | nonlinear | wavelengths/mode | SNR | derivative | baseline | peaks | unsupported |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for result in results:
        checks = result.get("check_statuses", {})
        unsupported = result.get("unsupported_checks", [])
        lines.append(
            f"| `{result['preset']}` | `{checks.get('concentration_sums_and_ranges', 'missing')}` | "
            f"`{checks.get('target_distribution', 'missing')}` | "
            f"`{checks.get('nonlinear_target_behavior', 'missing')}` | "
            f"`{checks.get('wavelengths_and_mode', 'missing')}` | "
            f"`{checks.get('spectral_snr', 'missing')}` | "
            f"`{checks.get('derivative_statistics', 'missing')}` | "
            f"`{checks.get('baseline_curvature', 'missing')}` | "
            f"`{checks.get('peak_density', 'missing')}` | "
            f"`{', '.join(unsupported) or 'none'}` |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Passed: {passed} / {len(results)}",
        f"- Blocked: {blocked} / {len(results)}",
        "- Training rule: any hard `failed` or `unsupported` check sets `downstream_training_status=blocked`.",
        "",
        "## Failing-Case Examples",
        "",
    ])
    failures = [result for result in results if result["validation_status"] != "passed"]
    if failures:
        for result in failures:
            lines.append(
                f"- `{result['preset']}` blocks training because: "
                f"`{', '.join(result.get('blocking_checks', []))}`"
            )
    else:
        lines.append("- None in this 10-preset smoke run.")
        lines.append(
            "- Synthetic blocked examples are covered by tests: bad concentration sums, "
            "imbalanced or non-integer classification labels, missing concentration metadata, "
            "and invalid empty wavelength grids."
        )

    lines.extend([
        "",
        "## Unsupported Checks",
        "",
        _unsupported_section(results),
        "",
        "## Residual Risks",
        "",
        "- Thresholds are B1 smoke guardrails, not calibrated real/synthetic realism thresholds.",
        "- A2 row-normalizes concentrations; B1 checks normalized mixture support and declared prior ranges separately.",
        "- `measurement_mode` compatibility is checked against prior weights and instrument category, but mode-specific optical physics remains a documented A2/A3 risk.",
        "- A3 fitted-only real-fit scientific gate is still failed/blocked; this report intentionally carries the override note above.",
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(_to_builtin({"results": results, "git_status": git_status}), indent=2, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        (
            "Pass B1 prior predictive smoke gate for these presets; downstream training is allowed for all rows."
            if blocked == 0
            else "B1 blocks downstream training for one or more presets; inspect blocking checks before training."
        ),
        "",
    ])
    return "\n".join(lines)


def _success_result(
    preset: str,
    run: SyntheticDatasetRun,
    validation: PriorPredictiveValidation,
) -> dict[str, Any]:
    checks = [check.to_dict() for check in validation.checks]
    return {
        "preset": preset,
        "validation_status": validation.validation_status,
        "downstream_training_status": validation.downstream_training_status,
        "phase_a_gate_override": validation.phase_a_gate_override,
        "domain": validation.summary["domain"],
        "instrument": validation.summary["instrument"],
        "mode": validation.summary["measurement_mode"],
        "target_type": validation.summary["target_type"],
        "X_shape": validation.summary["X_shape"],
        "y_shape": validation.summary["y_shape"],
        "blocking_checks": validation.failed_or_blocking_checks,
        "unsupported_checks": validation.summary["unsupported_checks"],
        "check_statuses": {check["name"]: check["status"] for check in checks},
        "key_metrics": _key_metrics(checks),
        "validation": validation.to_dict(),
    }


def _failure_result(
    preset: str,
    reason: str,
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "preset": preset,
        "validation_status": "blocked",
        "downstream_training_status": "blocked",
        "phase_a_gate_override": PHASE_A_GATE_OVERRIDE,
        "domain": "",
        "target_type": "",
        "X_shape": "",
        "blocking_checks": [reason],
        "unsupported_checks": [],
        "check_statuses": {},
        "key_metrics": {},
        "validation": validation_summary,
    }


def _key_metrics(checks: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for check in checks:
        metrics.update(check.get("metrics") or {})
    return metrics


def _unsupported_section(results: list[dict[str, Any]]) -> str:
    unsupported_rows = [
        f"- `{result['preset']}`: `{', '.join(result.get('unsupported_checks', []))}`"
        for result in results
        if result.get("unsupported_checks")
    ]
    if not unsupported_rows:
        return "No hard unsupported checks in this smoke run. Nonlinear target behavior is marked `not_applicable` for classification and linear-regression presets."
    return "\n".join(unsupported_rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


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


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
