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
    REALISM_GATE_BLOCKED_STATUS,
    TransferRow,
    downsample_regression_cohort,
    evaluate_regression_transfer_smoke,
    load_regression_cohort,
    write_transfer_csv,
)

DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/transfer_validation.csv")
DEFAULT_B2_CSV = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv")
DEFAULT_ADVERSARIAL_CSV = Path("bench/nirs_synthetic_pfn/reports/adversarial_auc.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-real-datasets", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=180)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--n-synthetic-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--b2-csv", type=Path, default=DEFAULT_B2_CSV)
    parser.add_argument("--adversarial-csv", type=Path, default=DEFAULT_ADVERSARIAL_CSV)
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
        adversarial_csv=args.adversarial_csv,
    )
    write_transfer_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            b2_csv=args.b2_csv,
            adversarial_csv=args.adversarial_csv,
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
    adversarial_csv: Path = DEFAULT_ADVERSARIAL_CSV,
) -> dict[str, Any]:
    b2_path = root / b2_csv
    adversarial_path = root / adversarial_csv
    gate_summary = inspect_realism_gate_status(b2_path=b2_path, adversarial_path=adversarial_path)
    if gate_summary["blocked"]:
        return _blocked_by_realism_gate_result(gate_summary)

    real_datasets, inventories = discover_local_real_datasets(root)
    regression_datasets = [dataset for dataset in real_datasets if dataset.task == "regression"]
    selected = regression_datasets if max_real_datasets <= 0 else regression_datasets[:max_real_datasets]
    synthetic_preset, synthetic_run, b1_validation = _build_validated_synthetic_run(
        n_samples=n_synthetic_samples,
        seed=seed,
    )
    b2_summary = gate_summary["b2_summary"]
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
        "adversarial_summary": gate_summary["adversarial_summary"],
        "gate_summary": gate_summary,
    }


def inspect_realism_gate_status(*, b2_path: Path, adversarial_path: Path) -> dict[str, Any]:
    """Read B2/B3 realism reports before any transfer data generation or model fit."""
    adversarial_summary = inspect_adversarial_auc_status(adversarial_path)
    b2_summary = inspect_b2_realism_status(b2_path)
    blocking_reasons: list[str] = []

    if adversarial_summary["exists"]:
        if adversarial_summary["gate_status"] == "NO-GO":
            blocking_reasons.append("adversarial_auc_raw_gate_NO-GO")
    elif b2_summary["b2_realism_failed"]:
        blocking_reasons.append("missing_adversarial_auc_csv_with_B2_raw_gate_failure")
    else:
        blocking_reasons.append("missing_adversarial_auc_csv")

    if b2_summary["b2_realism_failed"]:
        blocking_reasons.append(str(b2_summary["reason"]))

    return {
        "status": REALISM_GATE_BLOCKED_STATUS if blocking_reasons else "gate_passed",
        "blocked": bool(blocking_reasons),
        "blocking_reasons": blocking_reasons,
        "b2_summary": b2_summary,
        "adversarial_summary": adversarial_summary,
        "raw_authoritative": True,
        "snv_can_pass_gate": False,
    }


def inspect_adversarial_auc_status(path: Path) -> dict[str, Any]:
    """Summarize B3 adversarial AUC gate with uncalibrated_raw rows as the only authority."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "row_count": 0,
            "raw_rows": 0,
            "raw_compared": 0,
            "raw_smoke_failures": 0,
            "raw_blocked": 0,
            "raw_missing_auc": 0,
            "gate_status": "MISSING",
            "reason": "missing_adversarial_auc_csv",
        }

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    raw_rows = [row for row in rows if _is_raw_authoritative_row(row)]
    raw_compared = [row for row in raw_rows if row.get("status") == "compared"]
    raw_blocked = [row for row in raw_rows if row.get("status") == "blocked"]
    raw_missing_auc = [row for row in raw_compared if _optional_float(row.get("adversarial_auc")) is None]
    raw_smoke_failures = [
        row
        for row in raw_compared
        if _truthy(row.get("smoke_fail"))
        or (
            (auc := _optional_float(row.get("adversarial_auc"))) is not None
            and auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    ]
    gate_failed = (
        not raw_compared
        or bool(raw_smoke_failures)
        or bool(raw_blocked)
        or bool(raw_missing_auc)
    )
    reason_parts = []
    if not raw_compared:
        reason_parts.append("no_raw_compared_rows")
    if raw_smoke_failures:
        reason_parts.append("raw_smoke_failures")
    if raw_blocked:
        reason_parts.append("raw_blocked_evidence_gaps")
    if raw_missing_auc:
        reason_parts.append("raw_missing_auc")
    return {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "raw_rows": len(raw_rows),
        "raw_compared": len(raw_compared),
        "raw_smoke_failures": len(raw_smoke_failures),
        "raw_blocked": len(raw_blocked),
        "raw_missing_auc": len(raw_missing_auc),
        "gate_status": "NO-GO" if gate_failed else "GO_DIAGNOSTIC_ONLY",
        "reason": ";".join(reason_parts) if reason_parts else "raw_gate_no_smoke_failure",
    }


def inspect_b2_realism_status(path: Path) -> dict[str, Any]:
    """Summarize the prior B2 CSV and conservatively flag failed uncalibrated_raw realism."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "row_count": 0,
            "raw_rows": 0,
            "raw_compared": 0,
            "raw_smoke_failures": 0,
            "raw_blocked": 0,
            "raw_missing_auc": 0,
            "compared_rows": 0,
            "adversarial_auc_failures": 0,
            "pca_overlap_failures": 0,
            "b2_realism_failed": True,
            "reason": "missing_B2_csv",
        }
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    raw_rows = [row for row in rows if _is_raw_authoritative_row(row)]
    compared = [row for row in raw_rows if row.get("status") == "compared"]
    raw_blocked = [row for row in raw_rows if row.get("status") == "blocked"]
    raw_missing_auc = 0
    raw_smoke_failures = 0
    pca_failures = 0
    for row in compared:
        auc = _optional_float(row.get("adversarial_auc"))
        pca = _optional_float(row.get("pca_overlap"))
        if auc is None:
            raw_missing_auc += 1
        elif auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]:
            raw_smoke_failures += 1
        if pca is None or pca < PROVISIONAL_THRESHOLDS["pca_overlap_min"]:
            pca_failures += 1
    failed = not compared or raw_smoke_failures > 0 or bool(raw_blocked) or raw_missing_auc > 0
    return {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "raw_rows": len(raw_rows),
        "raw_compared": len(compared),
        "raw_smoke_failures": raw_smoke_failures,
        "raw_blocked": len(raw_blocked),
        "raw_missing_auc": raw_missing_auc,
        "compared_rows": len(compared),
        "adversarial_auc_failures": raw_smoke_failures,
        "pca_overlap_failures": pca_failures,
        "b2_realism_failed": failed,
        "reason": "B2_raw_realism_gate_failed" if failed else "B2_raw_realism_gate_not_failed",
    }


def _blocked_by_realism_gate_result(gate_summary: dict[str, Any]) -> dict[str, Any]:
    evidence = _primary_raw_gate_evidence(gate_summary)
    row = TransferRow(
        status=REALISM_GATE_BLOCKED_STATUS,
        source="B2_B3_realism_gate",
        task="regression",
        dataset="all",
        split_index=None,
        route="gate_first",
        model="none",
        n_train=0,
        n_test=0,
        n_features=0,
        n_synthetic=0,
        synthetic_preset="none",
        rmse=None,
        mae=None,
        r2=None,
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        b1_downstream_training_status="not_run_gate_blocked",
        b2_realism_risk=B2_REALISM_RISK,
        synthetic_transfer_claim_status="blocked_no_transfer_claim",
        blocked_reason=";".join(gate_summary["blocking_reasons"]),
        raw_compared=int(evidence["raw_compared"]),
        raw_smoke_failures=int(evidence["raw_smoke_failures"]),
        raw_blocked=int(evidence["raw_blocked"]),
        raw_missing_auc=int(evidence["raw_missing_auc"]),
    )
    return {
        "status": REALISM_GATE_BLOCKED_STATUS,
        "rows": [row],
        "inventories": [],
        "real_runnable_count": 0,
        "regression_runnable_count": 0,
        "real_selected_count": 0,
        "load_failures": [],
        "synthetic_preset": "none",
        "synthetic_summary": {
            "n_samples": 0,
            "n_features": 0,
            "wavelength_min": None,
            "wavelength_max": None,
        },
        "b1_validation": {
            "validation_status": "not_run_gate_blocked",
            "downstream_training_status": "not_run_gate_blocked",
            "checks": [],
        },
        "b2_summary": gate_summary["b2_summary"],
        "adversarial_summary": gate_summary["adversarial_summary"],
        "gate_summary": gate_summary,
    }


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    b2_csv: Path,
    adversarial_csv: Path,
    max_real_datasets: int,
    max_samples: int,
    n_splits: int,
    test_fraction: float,
    n_synthetic_samples: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    if result["status"] == REALISM_GATE_BLOCKED_STATUS:
        return _render_blocked_markdown(
            result=result,
            report_path=report_path,
            csv_path=csv_path,
            b2_csv=b2_csv,
            adversarial_csv=adversarial_csv,
            max_real_datasets=max_real_datasets,
            max_samples=max_samples,
            n_splits=n_splits,
            test_fraction=test_fraction,
            n_synthetic_samples=n_synthetic_samples,
            seed=seed,
            git_status=git_status,
        )

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


def _render_blocked_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    b2_csv: Path,
    adversarial_csv: Path,
    max_real_datasets: int,
    max_samples: int,
    n_splits: int,
    test_fraction: float,
    n_synthetic_samples: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    gate_summary = result["gate_summary"]
    b2_summary = result["b2_summary"]
    adversarial_summary = result["adversarial_summary"]
    evidence = _primary_raw_gate_evidence(gate_summary)
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
    lines = [
        "# Transfer Validation",
        "",
        "## Status",
        "",
        f"`{REALISM_GATE_BLOCKED_STATUS}`",
        "",
        "Phase B4 transfer validation is gate-first. The B2/B3 realism evidence was read before any synthetic build, real-only fit, TSTR route, or RTSR diagnostic.",
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
        "## B2/B3 Provenance",
        "",
        f"- B2 scorecards CSV inspected: `{b2_csv}`",
        f"- B2 exists: `{b2_summary['exists']}`",
        f"- B2 raw compared rows: {b2_summary['raw_compared']}",
        f"- B2 raw smoke failures: {b2_summary['raw_smoke_failures']}",
        f"- B2 raw blocked rows: {b2_summary['raw_blocked']}",
        f"- B2 raw missing AUC rows: {b2_summary['raw_missing_auc']}",
        f"- B3 adversarial AUC CSV inspected: `{adversarial_csv}`",
        f"- B3 exists: `{adversarial_summary['exists']}`",
        f"- B3 raw gate status: `{adversarial_summary['gate_status']}`",
        f"- B3 raw compared rows: {adversarial_summary['raw_compared']}",
        f"- B3 raw smoke failures: {adversarial_summary['raw_smoke_failures']}",
        f"- B3 raw blocked evidence gaps: {adversarial_summary['raw_blocked']}",
        f"- B3 raw missing AUC rows: {adversarial_summary['raw_missing_auc']}",
        "",
        "## Raw Authoritative Gate",
        "",
        "- Raw evidence is authoritative for this gate.",
        "- SNV evidence is diagnostic only and cannot pass or override the gate.",
        f"- CSV row raw_compared: {evidence['raw_compared']}",
        f"- CSV row raw_smoke_failures: {evidence['raw_smoke_failures']}",
        f"- CSV row raw_blocked: {evidence['raw_blocked']}",
        f"- CSV row raw_missing_auc: {evidence['raw_missing_auc']}",
        f"- Blocking reasons: `{';'.join(gate_summary['blocking_reasons'])}`",
        "",
        "## No Integration Or Transfer Claim",
        "",
        "- No integration readiness is claimed.",
        "- No transfer claim is made.",
        "- Synthetic generation count: 0.",
        "- Fitted model count: 0.",
        "- Real-only baseline fit count: 0.",
        "- TSTR/RTSR route count: 0.",
        "- exp02 was not launched by this experiment.",
        "",
        "## Next Actions",
        "",
        "- Remediate B2 raw failures and evidence gaps before rerunning transfer validation.",
        "- Prioritize named B2 gaps: BEER, DIESEL, and CORN.",
        "- Re-run the B2 scorecards and B3 adversarial AUC audit after remediation.",
        "- Only revisit transfer validation after the raw authoritative realism gate passes.",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(
            _to_builtin({
                "status": result["status"],
                "gate_summary": gate_summary,
                "row_count": len(result["rows"]),
                "git_status": git_status,
            }),
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
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


def _primary_raw_gate_evidence(gate_summary: dict[str, Any]) -> dict[str, int]:
    adversarial_summary = gate_summary["adversarial_summary"]
    if adversarial_summary["exists"]:
        source = adversarial_summary
    else:
        source = gate_summary["b2_summary"]
    return {
        "raw_compared": int(source.get("raw_compared", 0)),
        "raw_smoke_failures": int(source.get("raw_smoke_failures", 0)),
        "raw_blocked": int(source.get("raw_blocked", 0)),
        "raw_missing_auc": int(source.get("raw_missing_auc", 0)),
    }


def _is_raw_authoritative_row(row: dict[str, str]) -> bool:
    # Authoritative lane is `uncalibrated_raw`. Legacy `raw` rows are ignored when
    # `comparison_space` is present; fall back to explicit raw_authoritative flags only
    # when the column is missing entirely.
    comparison_space = row.get("comparison_space")
    if comparison_space is not None and comparison_space.strip() != "":
        return comparison_space.strip().lower() == "uncalibrated_raw"
    return _truthy(row.get("raw_authoritative")) or row.get("gate_basis") == "raw_authoritative"


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
