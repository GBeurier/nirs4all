"""Phase B3 bounded transfer-validation smoke on local regression cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from exp00_smoke_prior_dataset import _preset_source
from nirsyntheticpfn.adapters.builder_adapter import build_synthetic_dataset_run
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_prior_config
from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE, validate_prior_predictive_run
from nirsyntheticpfn.evaluation.realism import (
    PROVISIONAL_THRESHOLDS,
    CohortInventory,
    RealDataset,
    discover_local_real_datasets,
)
from nirsyntheticpfn.evaluation.transfer import (
    B2_REALISM_RISK,
    TransferRow,
    downsample_regression_cohort,
    evaluate_regression_transfer_smoke,
    load_regression_cohort,
    write_transfer_csv,
)

DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.csv")
DEFAULT_B2_CSV = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-real-datasets", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=180)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--n-synthetic-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--b2-csv", type=Path, default=DEFAULT_B2_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    root = _repo_root()
    result = run_transfer_validation(
        root=root,
        max_real_datasets=args.max_real_datasets,
        max_samples=args.max_samples,
        n_splits=args.n_splits,
        test_fraction=args.test_fraction,
        n_synthetic_samples=args.n_synthetic_samples,
        seed=args.seed,
        b2_csv=args.b2_csv,
    )
    write_transfer_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            b2_csv=args.b2_csv,
            max_real_datasets=args.max_real_datasets,
            max_samples=args.max_samples,
            n_splits=args.n_splits,
            test_fraction=args.test_fraction,
            n_synthetic_samples=args.n_synthetic_samples,
            seed=args.seed,
            git_status=_git_status_summary(root),
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.csv)


def run_transfer_validation(
    *,
    root: Path,
    max_real_datasets: int,
    max_samples: int,
    n_splits: int,
    test_fraction: float,
    n_synthetic_samples: int,
    seed: int,
    b2_csv: Path,
) -> dict[str, Any]:
    real_datasets, inventories = discover_local_real_datasets(root)
    regression_datasets = [dataset for dataset in real_datasets if dataset.task == "regression"]
    selected = regression_datasets if max_real_datasets <= 0 else regression_datasets[:max_real_datasets]
    synthetic_preset, synthetic_run, b1_validation = _build_validated_synthetic_run(
        n_samples=n_synthetic_samples,
        seed=seed,
    )
    b2_summary = inspect_b2_realism_status(root / b2_csv)
    rows: list[TransferRow] = []
    load_failures: list[dict[str, Any]] = []

    for idx, dataset in enumerate(selected):
        try:
            cohort = load_regression_cohort(dataset, root=root)
            cohort = downsample_regression_cohort(
                cohort,
                max_samples=max_samples,
                random_state=seed + idx,
            )
            rows.extend(evaluate_regression_transfer_smoke(
                cohort=cohort,
                synthetic_X=synthetic_run.X,
                synthetic_wavelengths=synthetic_run.wavelengths,
                synthetic_preset=synthetic_preset,
                n_splits=n_splits,
                test_fraction=test_fraction,
                random_state=seed + idx * 100,
                b1_downstream_training_status=b1_validation.downstream_training_status,
                b2_realism_failed=bool(b2_summary["b2_realism_failed"]),
            ))
        except Exception as exc:
            load_failures.append({
                "source": dataset.source,
                "task": dataset.task,
                "dataset": f"{dataset.database_name}/{dataset.dataset}",
                "failure_class": exc.__class__.__name__,
                "reason": str(exc),
                "paths": {
                    "train_path": dataset.train_path,
                    "test_path": dataset.test_path,
                    "ytrain_path": dataset.ytrain_path,
                    "ytest_path": dataset.ytest_path,
                },
            })

    completed = sum(1 for row in rows if row.status == "completed")
    status = "done" if completed else "blocked_no_completed_transfer_rows"
    return {
        "status": status,
        "rows": rows,
        "inventories": inventories,
        "real_runnable_count": len(real_datasets),
        "regression_runnable_count": len(regression_datasets),
        "real_selected_count": len(selected),
        "load_failures": load_failures,
        "synthetic_preset": synthetic_preset,
        "synthetic_summary": {
            "n_samples": int(synthetic_run.X.shape[0]),
            "n_features": int(synthetic_run.X.shape[1]),
            "wavelength_min": float(synthetic_run.wavelengths[0]),
            "wavelength_max": float(synthetic_run.wavelengths[-1]),
        },
        "b1_validation": b1_validation.to_dict(),
        "b2_summary": b2_summary,
    }


def inspect_b2_realism_status(path: Path) -> dict[str, Any]:
    """Summarize the prior B2 CSV and conservatively flag failed realism."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "compared_rows": 0,
            "adversarial_auc_failures": 0,
            "pca_overlap_failures": 0,
            "b2_realism_failed": True,
            "reason": "missing_B2_csv",
        }
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    compared = [row for row in rows if row.get("status") == "compared"]
    auc_failures = 0
    pca_failures = 0
    for row in compared:
        auc = _optional_float(row.get("adversarial_auc"))
        pca = _optional_float(row.get("pca_overlap"))
        if auc is None or auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]:
            auc_failures += 1
        if pca is None or pca < PROVISIONAL_THRESHOLDS["pca_overlap_min"]:
            pca_failures += 1
    failed = not compared or auc_failures > 0 or pca_failures > 0
    return {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "compared_rows": len(compared),
        "adversarial_auc_failures": auc_failures,
        "pca_overlap_failures": pca_failures,
        "b2_realism_failed": failed,
        "reason": "B2_realism_failed" if failed else "B2_realism_smoke_not_failed",
    }


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    b2_csv: Path,
    max_real_datasets: int,
    max_samples: int,
    n_splits: int,
    test_fraction: float,
    n_synthetic_samples: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    rows: list[TransferRow] = result["rows"]
    inventories: list[CohortInventory] = result["inventories"]
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py "
        f"--max-real-datasets {max_real_datasets} "
        f"--max-samples {max_samples} "
        f"--n-splits {n_splits} "
        f"--test-fraction {test_fraction} "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--seed {seed}"
    )
    completed = [row for row in rows if row.status == "completed"]
    blocked = [row for row in rows if row.status == "blocked"]
    lines = [
        "# Transfer Validation",
        "",
        "## Objective",
        "",
        "Implement a minimal B3 transfer-validation route on local regression cohorts with real-only baselines and a synthetic PCA diagnostic.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Outputs",
        "",
        f"- Markdown: `{report_path}`",
        f"- CSV metrics summary: `{csv_path}`",
        "",
        "## Gate Flags",
        "",
        f"- `phase_a_gate_override`: `{PHASE_A_GATE_OVERRIDE}`",
        f"- `b2_realism_risk`: `{B2_REALISM_RISK}`",
        f"- B2 CSV inspected: `{b2_csv}`",
        f"- B2 compared rows: {result['b2_summary']['compared_rows']}",
        f"- B2 adversarial AUC failures: {result['b2_summary']['adversarial_auc_failures']}",
        f"- B2 PCA overlap failures: {result['b2_summary']['pca_overlap_failures']}",
        "- Synthetic transfer usefulness claims are blocked unless real-data baselines and realism evidence support them.",
        "",
        "## Config",
        "",
        f"- Seed: {seed}",
        f"- Selected real regression rows: {result['real_selected_count']}",
        f"- Repeated splits per selected row: {n_splits}",
        f"- Max real samples per row: {max_samples}",
        f"- Test fraction: {test_fraction}",
        f"- Synthetic preset: `{result['synthetic_preset']}`",
        f"- Synthetic samples: {result['synthetic_summary']['n_samples']}",
        f"- B1 downstream training status: `{result['b1_validation']['downstream_training_status']}`",
        "",
        "## Seed Policy",
        "",
        f"- Primary seed: {seed}",
        "- Real-row downsampling uses `seed + dataset_index`.",
        "- Repeated splits use `seed + dataset_index * 100 + split_index`.",
        "- The A2 synthetic diagnostic run uses the primary seed.",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Real Cohort Inventory",
        "",
        "| source | cohort path | exists | total rows | status ok rows | runnable rows | rows with missing paths |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for inventory in inventories:
        lines.append(
            f"| `{inventory.source}` | `{inventory.path}` | `{inventory.exists}` | {inventory.total_rows} | {inventory.ok_rows} | {inventory.runnable_rows} | {inventory.missing_rows} |"
        )
    lines.extend([
        "",
        "## Runnable Counts",
        "",
        f"- Runnable local real rows discovered: {result['real_runnable_count']}",
        f"- Runnable regression rows discovered: {result['regression_runnable_count']}",
        f"- Selected/runnable regression rows attempted: {result['real_selected_count']}/{result['regression_runnable_count']}",
        f"- Completed metric rows: {len(completed)}",
        f"- Blocked/unsupported rows: {len(blocked)}",
        f"- Dataset load/evaluation failures before row emission: {len(result['load_failures'])}",
        "",
        "## Contract Checks",
        "",
        _contract_checks_section(result["b1_validation"]),
        "",
        "## Repeated Split Metrics",
        "",
        "| route | model | datasets | splits | RMSE mean | RMSE std | MAE mean | R2 mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for aggregate in _aggregate_completed(completed):
        lines.append(
            f"| `{aggregate['route']}` | `{aggregate['model']}` | {aggregate['datasets']} | {aggregate['splits']} | "
            f"{_fmt(aggregate['rmse_mean'])} | {_fmt(aggregate['rmse_std'])} | {_fmt(aggregate['mae_mean'])} | {_fmt(aggregate['r2_mean'])} |"
        )
    if not completed:
        lines.append("| _none_ | _none_ | 0 | 0 | n/a | n/a | n/a | n/a |")
    lines.extend([
        "",
        "## Blocked Transfer Routes",
        "",
        _blocked_section(blocked),
        "",
        "## Ablation Report",
        "",
        "- `without_instruments`: deferred; no defensible synthetic-transfer claim while B2 realism is failed.",
        "- `without_scatter`: deferred; no defensible synthetic-transfer claim while B2 realism is failed.",
        "- `without_products_or_aggregates`: deferred; no matched real/synthetic target route in this smoke.",
        "- `without_procedural_diversity`: deferred; needs a successful realism/domain-matching gate first.",
        "",
        "## Scientific Decision",
        "",
        _decision_text(result),
        "",
        "## Load Failures",
        "",
        _load_failures_section(result["load_failures"]),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(_to_builtin({
            "status": result["status"],
            "b1_validation": result["b1_validation"],
            "b2_summary": result["b2_summary"],
            "inventories": [inventory.__dict__ for inventory in inventories],
            "load_failures": result["load_failures"],
            "row_count": len(rows),
            "git_status": git_status,
        }), indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def _build_validated_synthetic_run(*, n_samples: int, seed: int) -> tuple[str, Any, Any]:
    preset = "grain"
    source = _preset_source(preset, target_type="regression", target_size=1, seed=seed)
    record = canonicalize_prior_config(source)
    run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)
    validation = validate_prior_predictive_run(run, preset=preset)
    return preset, run, validation


def _aggregate_completed(rows: list[TransferRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[TransferRow]] = {}
    for row in rows:
        groups.setdefault((row.route, row.model), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (route, model), group in sorted(groups.items()):
        rmses = np.asarray([row.rmse for row in group if row.rmse is not None], dtype=float)
        maes = np.asarray([row.mae for row in group if row.mae is not None], dtype=float)
        r2s = np.asarray([row.r2 for row in group if row.r2 is not None], dtype=float)
        aggregates.append({
            "route": route,
            "model": model,
            "datasets": len({row.dataset for row in group}),
            "splits": len(group),
            "rmse_mean": float(np.mean(rmses)) if rmses.size else None,
            "rmse_std": float(np.std(rmses)) if rmses.size else None,
            "mae_mean": float(np.mean(maes)) if maes.size else None,
            "r2_mean": float(np.mean(r2s)) if r2s.size else None,
        })
    return aggregates


def _decision_text(result: dict[str, Any]) -> str:
    if result["status"] != "done":
        return "Blocked: no completed real-data baseline rows were produced."
    if result["b2_summary"]["b2_realism_failed"]:
        return (
            "Route runnable: real-only baselines and a synthetic PCA diagnostic were produced. "
            "Synthetic usefulness remains blocked because B2 realism failed and supervised TSTR target matching is not available."
        )
    return "Route runnable: real-only baselines and synthetic diagnostics are available for review, but this smoke remains provisional."


def _blocked_section(rows: list[TransferRow]) -> str:
    if not rows:
        return "None."
    lines = []
    for row in rows[:30]:
        lines.append(f"- `{row.dataset}` split `{row.split_index}` `{row.route}/{row.model}`: {row.blocked_reason}")
    if len(rows) > 30:
        lines.append(f"- _truncated, {len(rows) - 30} more blocked rows in CSV_")
    return "\n".join(lines)


def _load_failures_section(load_failures: list[dict[str, Any]]) -> str:
    if not load_failures:
        return "None."
    return "\n".join(
        f"- `{failure['source']}` `{failure['dataset']}` [{failure['failure_class']}]: {failure['reason']}"
        for failure in load_failures
    )


def _contract_checks_section(b1_validation: dict[str, Any]) -> str:
    checks = list(b1_validation.get("checks", []))
    hard_checks = [check for check in checks if check.get("severity") == "hard"]
    failed_hard = [check for check in hard_checks if check.get("status") != "passed"]
    lines = [
        f"- A2 synthetic validation status: `{b1_validation.get('validation_status', 'unknown')}`",
        f"- Downstream training status: `{b1_validation.get('downstream_training_status', 'unknown')}`",
        f"- Hard checks passed: {len(hard_checks) - len(failed_hard)}/{len(hard_checks)}",
    ]
    if failed_hard:
        lines.append("- Failed hard checks:")
        lines.extend(f"  - `{check.get('name', 'unknown')}`: `{check.get('status', 'unknown')}`" for check in failed_hard)
    else:
        lines.append("- Failed hard checks: none.")
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_status_summary(root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {
        "returncode": result.returncode,
        "line_count": len(lines),
        "lines": lines[:80],
        "truncated": len(lines) > 80,
    }


def _git_status_section(git_status: dict[str, Any]) -> str:
    if git_status["line_count"] == 0:
        return "Clean working tree."
    lines = [
        f"- `git status --short` lines: {git_status['line_count']}",
        "- First entries:",
    ]
    lines.extend(f"  - `{line}`" for line in git_status["lines"])
    if git_status.get("truncated"):
        lines.append("  - _truncated_")
    return "\n".join(lines)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
