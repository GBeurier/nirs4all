from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_exp04_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments/exp04_adversarial_auc.py"
    spec = importlib.util.spec_from_file_location("exp04_adversarial_auc", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp04_adversarial_auc"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_b2_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "source",
        "task",
        "dataset",
        "synthetic_preset",
        "comparison_space",
        "adversarial_auc",
        "adversarial_auc_std",
        "provisional_decision",
        "blocked_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    dataset: str = "UNIT/Dataset",
    source: str = "unit",
    task: str = "regression",
    preset: str = "grain",
    space: str = "uncalibrated_raw",
    status: str = "compared",
    auc: str = "0.60",
    decision: str = "provisional_pass",
    blocked_reason: str = "",
) -> dict[str, str]:
    return {
        "status": status,
        "source": source,
        "task": task,
        "dataset": dataset,
        "synthetic_preset": preset,
        "comparison_space": space,
        "adversarial_auc": auc,
        "adversarial_auc_std": "0.01" if auc else "",
        "provisional_decision": decision,
        "blocked_reason": blocked_reason,
    }


def test_parses_and_generates_reports_from_sample_rows(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "adversarial_auc.md"
    out_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_csv(b2_csv, [_row(), _row(space="snv", auc="0.62")])

    summary = module.run_report(b2_csv=b2_csv, report_path=report, csv_path=out_csv)

    assert summary.gate_status == "GO_DIAGNOSTIC_ONLY"
    assert report.exists()
    assert out_csv.exists()
    assert "## Objective" in report.read_text(encoding="utf-8")
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset"] == "UNIT/Dataset"
    assert rows[0]["smoke_fail"] == "false"
    assert rows[0]["gate_basis"] == "raw_authoritative"


def test_high_auc_taxonomy_is_reported(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    _write_b2_csv(b2_csv, [_row(auc="0.91", decision="provisional_review:adversarial_auc")])

    summary = module.build_summary(rows=module.load_b2_rows(b2_csv), b2_csv=b2_csv)

    assert summary.gate_status == "NO-GO"
    assert summary.taxonomy_counts["raw_smoke_high_auc"] == 1


def test_blocked_rows_force_no_go(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    _write_b2_csv(
        b2_csv,
        [
            _row(auc="0.55"),
            _row(dataset="UNIT/Blocked", status="blocked", auc="", blocked_reason="wavelength_grid_unknown: no physical grid"),
        ],
    )

    summary = module.build_summary(rows=module.load_b2_rows(b2_csv), b2_csv=b2_csv)

    assert summary.gate_status == "NO-GO"
    assert summary.raw_blocked[0].dataset == "UNIT/Blocked"
    assert module.blocked_class(summary.raw_blocked[0]) == "wavelength_grid_unknown"


def test_raw_missing_auc_forces_no_go(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    _write_b2_csv(b2_csv, [_row(dataset="UNIT/MissingRawAuc", auc="")])

    summary = module.build_summary(rows=module.load_b2_rows(b2_csv), b2_csv=b2_csv)

    assert summary.gate_status == "NO-GO"
    assert summary.raw_missing_auc[0].dataset == "UNIT/MissingRawAuc"
    assert summary.taxonomy_counts["raw_missing_auc"] == 1


def test_missing_b2_csv_error(tmp_path: Path) -> None:
    module = _load_exp04_module()

    with pytest.raises(FileNotFoundError, match="B2 CSV not found"):
        module.run_report(b2_csv=tmp_path / "missing.csv", report_path=tmp_path / "report.md", csv_path=tmp_path / "out.csv")


def test_generated_md_has_no_positive_integration_or_transfer_claim(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(b2_csv, [_row()])

    module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")
    text = report.read_text(encoding="utf-8")

    assert "does not claim integration readiness" in text
    assert "downstream transfer benefit" in text
    assert "integration success" not in text.lower()
    assert "transfer benefit is established" not in text.lower()


def test_thresholds_are_fixed_and_not_mutable_by_cli() -> None:
    module = _load_exp04_module()

    assert module.SMOKE_AUC_THRESHOLD == 0.85
    assert module.STRETCH_AUC_THRESHOLD == 0.70
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--smoke-threshold", "0.99"])


def test_snv_pass_does_not_override_uncalibrated_raw_fail(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _row(dataset="UNIT/RawFail", space="uncalibrated_raw", auc="0.90", decision="provisional_review:adversarial_auc"),
            _row(dataset="UNIT/RawFail", space="snv", auc="0.50"),
        ],
    )

    summary = module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")

    assert summary.gate_status == "NO-GO"
    assert "SNV cannot override raw failures" in report.read_text(encoding="utf-8")


def test_snv_fail_does_not_override_clean_uncalibrated_raw(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _row(dataset="UNIT/SnvFail", space="uncalibrated_raw", auc="0.50"),
            _row(dataset="UNIT/SnvFail", space="snv", auc="0.99", decision="provisional_review:adversarial_auc"),
        ],
    )

    summary = module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")

    assert summary.gate_status == "GO_DIAGNOSTIC_ONLY"
    assert summary.snv_smoke_failures[0].dataset == "UNIT/SnvFail"


def test_calibrated_raw_diagnostic_fail_does_not_override_clean_uncalibrated_raw(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _row(dataset="UNIT/CalDiag", space="uncalibrated_raw", auc="0.50"),
            _row(
                dataset="UNIT/CalDiag",
                space="calibrated_raw_diagnostic",
                auc="0.95",
                decision="provisional_review:adversarial_auc",
            ),
            _row(dataset="UNIT/CalDiag", space="snv", auc="0.95", decision="provisional_review:adversarial_auc"),
        ],
    )

    summary = module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")

    assert summary.gate_status == "GO_DIAGNOSTIC_ONLY"
    text = report.read_text(encoding="utf-8")
    assert "calibrated_raw_diagnostic" in text


def test_legacy_raw_only_fails_closed(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _row(dataset="UNIT/LegacyRaw", space="raw", auc="0.40"),
            _row(dataset="UNIT/LegacyRaw", space="snv", auc="0.40"),
        ],
    )

    summary = module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")

    assert summary.gate_status == "NO-GO"
    assert summary.raw_compared == []
    assert summary.provenance.legacy_raw_rows == 1
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    legacy_row = next(row for row in rows if row["dataset"] == "UNIT/LegacyRaw" and row["comparison_space"] == "raw")
    assert legacy_row["raw_authoritative"] == "false"
    assert legacy_row["gate_basis"] == "legacy_raw_not_authoritative"


def test_raw_stretch_only_sets_review(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    _write_b2_csv(b2_csv, [_row(dataset="UNIT/StretchOnly", auc="0.80")])

    summary = module.build_summary(rows=module.load_b2_rows(b2_csv), b2_csv=b2_csv)

    assert summary.gate_status == "REVIEW"
    assert summary.taxonomy_counts["raw_stretch_high_auc"] == 1


def test_named_beer_diesel_corn_gaps_appear_when_failing(tmp_path: Path) -> None:
    module = _load_exp04_module()
    b2_csv = tmp_path / "b2.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _row(dataset="BEER/Beer_OriginalExtract", preset="wine", auc="1.0", decision="provisional_review:adversarial_auc"),
            _row(dataset="DIESEL/DIESEL_bp50", preset="fuel", auc="0.99", decision="provisional_review:adversarial_auc"),
            _row(dataset="CORN/Corn_Oil", preset="grain", auc="0.95", decision="provisional_review:adversarial_auc"),
        ],
    )

    module.run_report(b2_csv=b2_csv, report_path=report, csv_path=tmp_path / "out.csv")
    text = report.read_text(encoding="utf-8")

    assert "BEER:" in text
    assert "DIESEL:" in text
    assert "CORN:" in text
