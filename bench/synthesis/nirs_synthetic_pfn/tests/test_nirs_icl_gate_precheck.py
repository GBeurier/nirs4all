from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_exp07_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments/exp07_nirs_icl_gate_precheck.py"
    spec = importlib.util.spec_from_file_location("exp07_nirs_icl_gate_precheck", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp07_nirs_icl_gate_precheck"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_b3(path: Path, *, status: str = "GO_DIAGNOSTIC_ONLY", smoke_fail: str = "false") -> None:
    _write_csv(
        path,
        [
            "dataset",
            "source",
            "task",
            "preset",
            "comparison_space",
            "status",
            "adversarial_auc",
            "smoke_fail",
            "stretch_fail",
            "blocked_class",
            "raw_authoritative",
        ],
        [
            {
                "dataset": "UNIT/Dataset",
                "source": "unit",
                "task": "regression",
                "preset": "grain",
                "comparison_space": "raw",
                "status": "compared",
                "adversarial_auc": "0.91" if smoke_fail == "true" else "0.50",
                "smoke_fail": smoke_fail,
                "stretch_fail": smoke_fail,
                "blocked_class": "",
                "raw_authoritative": "true",
            }
        ],
    )
    path.with_suffix(".md").write_text(f"# B3\n\n## Raw Authoritative Gate\n\n- Status: `{status}`.\n", encoding="utf-8")


def _write_b4(path: Path, *, status: str = "done") -> None:
    _write_csv(
        path,
        [
            "status",
            "source",
            "task",
            "dataset",
            "raw_compared",
            "raw_smoke_failures",
            "raw_blocked",
            "raw_missing_auc",
        ],
        [
            {
                "status": status,
                "source": "B2_B3_realism_gate",
                "task": "regression",
                "dataset": "all",
                "raw_compared": "1",
                "raw_smoke_failures": "0",
                "raw_blocked": "0",
                "raw_missing_auc": "0",
            }
        ],
    )
    path.with_suffix(".md").write_text(f"# B4\n\n## Status\n\n`{status}`\n", encoding="utf-8")


def _write_b5(path: Path, *, status: str = "GO") -> None:
    _write_csv(
        path,
        ["category", "value", "row_count", "gate_status"],
        [{"category": "summary", "value": "all", "row_count": "1", "gate_status": status}],
    )
    path.with_suffix(".md").write_text(f"# B5\n\n## Gate Status\n\n- Status: `{status}`.\n", encoding="utf-8")


def _write_c(path: Path, *, status: str = "READY_FOR_MANUAL_C_PHASE") -> None:
    _write_csv(
        path,
        [
            "status",
            "b3_status",
            "b4_status",
            "b5_status",
            "integration_status",
            "train_allowed",
            "tabpfn_allowed",
            "checkpoint_allowed",
        ],
        [
            {
                "status": status,
                "b3_status": "GO_DIAGNOSTIC_ONLY",
                "b4_status": "done",
                "b5_status": "GO",
                "integration_status": "GO",
                "train_allowed": "false",
                "tabpfn_allowed": "false",
                "checkpoint_allowed": "false",
            }
        ],
    )
    path.with_suffix(".md").write_text(f"# C\n\n## Status\n\n- Status: `{status}`.\n", encoding="utf-8")


def _write_integration(path: Path, *, status: str = "GO") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Integration Gate Status\n\nDecision: `{status} for integration`\n", encoding="utf-8")


def _write_all_pass_inputs(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv")
    _write_b4(tmp_path / "transfer_validation.csv")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_c(tmp_path / "encoder_tabpfn_gate.csv")
    _write_integration(tmp_path / "integration_gate_status.md")


def _run_with_inputs(tmp_path: Path) -> tuple[ModuleType, object, Path, Path]:
    module = _load_exp07_module()
    b3_csv = tmp_path / "adversarial_auc.csv"
    b4_csv = tmp_path / "transfer_validation.csv"
    b5_csv = tmp_path / "minimal_ablation_attribution.csv"
    c_csv = tmp_path / "encoder_tabpfn_gate.csv"
    integration_md = tmp_path / "integration_gate_status.md"
    out_csv = tmp_path / "nirs_icl_gate_precheck.csv"
    report = tmp_path / "nirs_icl_gate_precheck.md"

    summary = module.run_precheck(
        b3_csv=b3_csv,
        b3_md=b3_csv.with_suffix(".md"),
        b4_csv=b4_csv,
        b4_md=b4_csv.with_suffix(".md"),
        b5_csv=b5_csv,
        b5_md=b5_csv.with_suffix(".md"),
        c_csv=c_csv,
        c_md=c_csv.with_suffix(".md"),
        integration_md=integration_md,
        csv_path=out_csv,
        report_path=report,
    )
    return module, summary, out_csv, report


def test_fake_b3_no_go_blocks(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    _write_b3(tmp_path / "adversarial_auc.csv", status="NO-GO", smoke_fail="true")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["status"] == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["b3_status"] == "NO-GO"
    assert row["task_episodes_generated"] == "0"


def test_b4_blocked_blocks(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    _write_b4(tmp_path / "transfer_validation.csv", status="BLOCKED_BY_REALISM_GATE")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["b4_status"] == "BLOCKED_BY_REALISM_GATE"


def test_b5_report_only_blocks(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    _write_b5(tmp_path / "minimal_ablation_attribution.csv", status="BLOCKED_REPORT_ONLY")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["b5_status"] == "BLOCKED_REPORT_ONLY"


def test_c_blocked_blocks_encoder_gate_after_upstream_passes(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    _write_c(tmp_path / "encoder_tabpfn_gate.csv", status="BLOCKED_BY_UPSTREAM_REALISM_GATE")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_UPSTREAM_ENCODER_GATE"
    assert row["c_status"] == "BLOCKED_BY_UPSTREAM_REALISM_GATE"


def test_integration_no_go_blocks(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    _write_integration(tmp_path / "integration_gate_status.md", status="NO-GO")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["integration_status"] == "NO-GO"


def test_missing_artifact_blocks(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)
    (tmp_path / "encoder_tabpfn_gate.csv").unlink()

    _, summary, out_csv, report = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    assert summary.status == "BLOCKED_BY_MISSING_UPSTREAM_GATE_ARTIFACT"
    assert row["status"] == "BLOCKED_BY_MISSING_UPSTREAM_GATE_ARTIFACT"
    assert row["c_status"] == "MISSING"
    assert "encoder_tabpfn_gate.csv" in report.read_text(encoding="utf-8")


def test_all_fake_pass_ready_manual_only_no_auto_permissions(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)

    _, summary, out_csv, report = _run_with_inputs(tmp_path)

    row = _read_single_row(out_csv)
    text = report.read_text(encoding="utf-8")
    assert summary.status == "READY_FOR_MANUAL_D_PHASE"
    assert summary.manual_ready is True
    assert row == {
        "status": "READY_FOR_MANUAL_D_PHASE",
        "b3_status": "GO_DIAGNOSTIC_ONLY",
        "b4_status": "done",
        "b5_status": "GO",
        "c_status": "READY_FOR_MANUAL_C_PHASE",
        "integration_status": "GO",
        "task_sampling_allowed": "false",
        "icl_baseline_allowed": "false",
        "tabpfn_allowed": "false",
        "pfn_training_allowed": "false",
        "benchmark_allowed": "false",
        "task_episodes_generated": "0",
    }
    assert "Manual-ready only: `true`." in text


def test_source_has_no_disallowed_imports_or_calls() -> None:
    source = (Path(__file__).resolve().parents[1] / "experiments/exp07_nirs_icl_gate_precheck.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "import torch",
        "import tabpfn",
        "import sklearn",
        "nirs4all.synthesis",
        "sample_nirs_prior_task",
        "sample_context_query_split",
        "build_synthetic",
    ]
    assert not any(phrase in source for phrase in forbidden)


def test_report_has_no_task_generation_training_benchmark_claims_and_corrects_multi_output(tmp_path: Path) -> None:
    _write_all_pass_inputs(tmp_path)

    _, _, _, report = _run_with_inputs(tmp_path)
    text = report.read_text(encoding="utf-8")
    lower = text.lower()

    assert "No task episodes were generated." in text
    assert "No TabPFN/ICL benchmark was run." in text
    assert "No PFN training was run." in text
    assert "No transfer/integration claim is made." in text
    assert "Multi-output regression and multi-output classification are supported structurally now" in text
    assert "Group holdout is only a structural split mode" in text
    forbidden = [
        "task episodes generated successfully",
        "training completed",
        "benchmark completed",
        "transfer benefit is established",
        "integration readiness achieved",
        "automatic sampling is allowed",
    ]
    assert not any(phrase in lower for phrase in forbidden)


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]
