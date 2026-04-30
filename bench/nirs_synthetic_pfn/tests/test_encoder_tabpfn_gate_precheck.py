from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_exp06_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments/exp06_encoder_tabpfn_gate_precheck.py"
    spec = importlib.util.spec_from_file_location("exp06_encoder_tabpfn_gate_precheck", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp06_encoder_tabpfn_gate_precheck"] = module
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


def _write_integration(path: Path, *, status: str = "GO") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Integration Gate Status\n\nDecision: `{status} for integration`\n", encoding="utf-8")


def _run_with_inputs(tmp_path: Path) -> tuple[ModuleType, object, Path, Path]:
    module = _load_exp06_module()
    b3_csv = tmp_path / "adversarial_auc.csv"
    b4_csv = tmp_path / "transfer_validation.csv"
    b5_csv = tmp_path / "minimal_ablation_attribution.csv"
    integration_md = tmp_path / "integration_gate_status.md"
    out_csv = tmp_path / "encoder_tabpfn_gate.csv"
    report = tmp_path / "encoder_tabpfn_gate.md"

    summary = module.run_precheck(
        b3_csv=b3_csv,
        b3_md=b3_csv.with_suffix(".md"),
        b4_csv=b4_csv,
        b4_md=b4_csv.with_suffix(".md"),
        b5_csv=b5_csv,
        b5_md=b5_csv.with_suffix(".md"),
        integration_md=integration_md,
        csv_path=out_csv,
        report_path=report,
    )
    return module, summary, out_csv, report


def test_blocked_with_fake_b3_no_go(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv", status="NO-GO", smoke_fail="true")
    _write_b4(tmp_path / "transfer_validation.csv")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert summary.b3.status == "NO-GO"
    assert summary.counts.raw_compared == 1
    row = _read_single_row(out_csv)
    assert row["status"] == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert row["train_allowed"] == "false"
    assert row["tabpfn_allowed"] == "false"
    assert row["checkpoint_allowed"] == "false"


def test_b3_raw_failures_override_false_positive_markdown(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv", status="GO_DIAGNOSTIC_ONLY", smoke_fail="true")
    _write_b4(tmp_path / "transfer_validation.csv")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert summary.b3.status == "NO-GO"
    row = _read_single_row(out_csv)
    assert "B3_NO-GO" in row["blocking_reasons"]
    assert row["train_allowed"] == "false"


def test_blocked_with_b4_transfer_blocked(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv")
    _write_b4(tmp_path / "transfer_validation.csv", status="BLOCKED_BY_REALISM_GATE")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert summary.b4.status == "BLOCKED_BY_REALISM_GATE"
    assert "B4_BLOCKED_BY_REALISM_GATE" in _read_single_row(out_csv)["blocking_reasons"]


def test_b4_markdown_status_heading_is_parsed(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv")
    b4_csv = tmp_path / "transfer_validation.csv"
    _write_csv(
        b4_csv,
        ["source", "raw_compared", "raw_smoke_failures", "raw_blocked", "raw_missing_auc"],
        [
            {
                "source": "B2_B3_realism_gate",
                "raw_compared": "1",
                "raw_smoke_failures": "0",
                "raw_blocked": "0",
                "raw_missing_auc": "0",
            }
        ],
    )
    b4_csv.with_suffix(".md").write_text("# B4\n\n## Status\n\n`BLOCKED_BY_REALISM_GATE`\n", encoding="utf-8")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, _ = _run_with_inputs(tmp_path)

    assert summary.status == "BLOCKED_BY_UPSTREAM_REALISM_GATE"
    assert summary.b4.status == "BLOCKED_BY_REALISM_GATE"
    assert "B4_BLOCKED_BY_REALISM_GATE" in _read_single_row(out_csv)["blocking_reasons"]


def test_missing_gate_artifact_blocks_clearly(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv")
    _write_b4(tmp_path / "transfer_validation.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, report = _run_with_inputs(tmp_path)

    assert summary.status == "BLOCKED_BY_MISSING_UPSTREAM_GATE_ARTIFACT"
    assert summary.b5.status == "MISSING"
    row = _read_single_row(out_csv)
    assert "missing_B5 CSV" in row["blocking_reasons"]
    assert "minimal_ablation_attribution.csv" in report.read_text(encoding="utf-8")


def test_no_positive_training_benchmark_or_integration_claims_in_md(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv", status="NO-GO", smoke_fail="true")
    _write_b4(tmp_path / "transfer_validation.csv", status="BLOCKED_BY_REALISM_GATE")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv", status="BLOCKED_REPORT_ONLY")
    _write_integration(tmp_path / "integration_gate_status.md", status="NO-GO")

    _, _, _, report = _run_with_inputs(tmp_path)
    text = report.read_text(encoding="utf-8")
    lower = text.lower()

    assert "No encoder training." in text
    assert "No InfoNCE." in text
    assert "No CNN/patch transformer benchmark." in text
    assert "No TabPFN/PCA benchmark." in text
    assert "No checkpoint." in text
    assert "No transfer/integration claim." in text
    forbidden = [
        "encoder training is allowed",
        "infonce training ran",
        "cnn benchmark completed",
        "patch transformer benchmark completed",
        "tabpfn benchmark completed",
        "pca benchmark completed",
        "checkpoint generated",
        "transfer benefit is established",
        "integration readiness achieved",
    ]
    assert not any(phrase in lower for phrase in forbidden)


def test_no_cli_threshold_mutation() -> None:
    module = _load_exp06_module()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--smoke-threshold", "0.99"])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--stretch-threshold", "0.99"])


def test_all_fake_gates_pass_ready_for_manual_c_phase_without_training(tmp_path: Path) -> None:
    _write_b3(tmp_path / "adversarial_auc.csv")
    _write_b4(tmp_path / "transfer_validation.csv")
    _write_b5(tmp_path / "minimal_ablation_attribution.csv")
    _write_integration(tmp_path / "integration_gate_status.md")

    _, summary, out_csv, report = _run_with_inputs(tmp_path)
    row = _read_single_row(out_csv)
    text = report.read_text(encoding="utf-8")

    assert summary.status == "READY_FOR_MANUAL_C_PHASE"
    assert row["train_allowed"] == "false"
    assert row["tabpfn_allowed"] == "false"
    assert row["checkpoint_allowed"] == "false"
    assert "This precheck itself never starts training or benchmarking." in text


def test_script_has_no_ml_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / "experiments/exp06_encoder_tabpfn_gate_precheck.py").read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import tabpfn" not in source
    assert "import sklearn" not in source


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]
