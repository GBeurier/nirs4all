from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_exp05_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments/exp05_minimal_ablation_attribution.py"
    spec = importlib.util.spec_from_file_location("exp05_minimal_ablation_attribution", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp05_minimal_ablation_attribution"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_b2_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "status",
        "source",
        "task",
        "dataset",
        "synthetic_preset",
        "comparison_space",
        "synthetic_mapping_strategy",
        "synthetic_mapping_reason",
        "pca_overlap",
        "adversarial_auc",
        "provisional_decision",
        "blocked_reason",
    ]
    _write_csv(path, fields, rows)


def _write_b3_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
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
    ]
    _write_csv(path, fields, rows)


def _write_b4_csv(path: Path, *, status: str = "BLOCKED_BY_REALISM_GATE") -> None:
    fields = [
        "status",
        "source",
        "task",
        "dataset",
        "blocked_reason",
        "raw_compared",
        "raw_smoke_failures",
        "raw_blocked",
        "raw_missing_auc",
    ]
    _write_csv(
        path,
        fields,
        [
            {
                "status": status,
                "source": "B2_B3_realism_gate",
                "task": "regression",
                "dataset": "all",
                "blocked_reason": "adversarial_auc_raw_gate_NO-GO;B2_raw_realism_gate_failed",
                "raw_compared": "2",
                "raw_smoke_failures": "1",
                "raw_blocked": "1",
                "raw_missing_auc": "0",
            }
        ],
    )


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mapping_reason(
    *,
    source_override: str = "no source override",
    wavelength_policy: str = "not_requested",
    grid_source: str = "real_grid",
    grid_remap_reason: str = "no_remap_needed",
) -> str:
    return (
        '{"generation":{"source_overrides":{"reason":"'
        + source_override
        + '"},"canonical_wavelength_policy":{"bench_wavelength_support_override":{"reason":"'
        + wavelength_policy
        + '"}},"grid":{"grid_source":"'
        + grid_source
        + '"}},"grid_remap":{"reason":"'
        + grid_remap_reason
        + '"}}'
    )


def _b2_row(
    *,
    dataset: str = "UNIT/Dataset",
    preset: str = "grain",
    space: str = "uncalibrated_raw",
    status: str = "compared",
    auc: str = "0.60",
    pca: str = "0.20",
    decision: str = "provisional_pass",
    blocked_reason: str = "",
    mapping_strategy: str = "matrix_first_dataset",
    mapping_reason: str | None = None,
) -> dict[str, str]:
    return {
        "status": status,
        "source": "unit",
        "task": "regression",
        "dataset": dataset,
        "synthetic_preset": preset,
        "comparison_space": space,
        "synthetic_mapping_strategy": mapping_strategy,
        "synthetic_mapping_reason": mapping_reason if mapping_reason is not None else _mapping_reason(),
        "pca_overlap": pca,
        "adversarial_auc": auc,
        "provisional_decision": decision,
        "blocked_reason": blocked_reason,
    }


def _b3_row(
    *,
    dataset: str = "UNIT/Dataset",
    preset: str = "grain",
    space: str = "uncalibrated_raw",
    status: str = "compared",
    auc: str = "0.60",
    smoke_fail: str = "false",
    stretch_fail: str = "false",
    blocked_class: str = "",
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "source": "unit",
        "task": "regression",
        "preset": preset,
        "comparison_space": space,
        "status": status,
        "adversarial_auc": auc,
        "smoke_fail": smoke_fail,
        "stretch_fail": stretch_fail,
        "blocked_class": blocked_class,
        "raw_authoritative": "true" if space == "uncalibrated_raw" else "false",
    }


def _run_sample(tmp_path: Path) -> tuple[ModuleType, object, Path, Path]:
    module = _load_exp05_module()
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    b3_csv = tmp_path / "adversarial_auc.csv"
    b3_md = tmp_path / "adversarial_auc.md"
    b4_csv = tmp_path / "transfer_validation.csv"
    out_csv = tmp_path / "minimal_ablation_attribution.csv"
    report = tmp_path / "minimal_ablation_attribution.md"
    _write_b2_csv(
        b2_csv,
        [
            _b2_row(
                dataset="BEER/Beer_OriginalExtract",
                preset="wine",
                auc="0.91",
                pca="0.0",
                decision="provisional_review:adversarial_auc,pca_overlap,nearest_neighbor_ratio",
                mapping_reason=_mapping_reason(
                    source_override="preset wine defaults",
                    wavelength_policy="beer/wine wavelength support override requested for real grid",
                    grid_source="real_grid_clipped_to_supported_overlap",
                    grid_remap_reason="grid_compatible_fallback",
                ),
            ),
            _b2_row(dataset="BEER/Beer_OriginalExtract", preset="wine", space="snv", auc="0.50"),
            _b2_row(
                dataset="DIESEL/DIESEL_bp50",
                preset="fuel",
                auc="1.0",
                decision="provisional_review:adversarial_auc,derivative_gap",
            ),
            _b2_row(
                dataset="CORN/Corn_Oil",
                preset="grain",
                status="blocked",
                auc="",
                pca="",
                blocked_reason="wavelength_grid_unknown: no physical grid",
                mapping_reason=_mapping_reason(grid_source="", grid_remap_reason="wavelength_grid_unknown"),
            ),
        ],
    )
    _write_b3_csv(
        b3_csv,
        [
            _b3_row(dataset="BEER/Beer_OriginalExtract", preset="wine", smoke_fail="true", stretch_fail="true", auc="0.91"),
            _b3_row(dataset="BEER/Beer_OriginalExtract", preset="wine", space="snv", auc="0.50"),
            _b3_row(dataset="DIESEL/DIESEL_bp50", preset="fuel", smoke_fail="true", stretch_fail="true", auc="1.0"),
            _b3_row(
                dataset="CORN/Corn_Oil",
                preset="grain",
                status="blocked",
                auc="",
                blocked_class="wavelength_grid_unknown",
            ),
        ],
    )
    b3_md.write_text("# B3\nStatus: NO-GO\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    summary = module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )
    return module, summary, out_csv, report


def _read_output(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sample_csv_grouping_works(tmp_path: Path) -> None:
    _, summary, out_csv, _ = _run_sample(tmp_path)

    rows = _read_output(out_csv)
    mapping = next(row for row in rows if row["category"] == "mapping_strategy" and row["value"] == "matrix_first_dataset")
    assert mapping["row_count"] == "4"
    assert mapping["compared_count"] == "3"
    assert mapping["blocked_count"] == "1"
    assert mapping["auc_smoke_fail_count"] == "2"
    assert mapping["pca_fail_count"] == "1"
    assert mapping["nn_ratio_fail_count"] == "1"
    assert mapping["derivative_fail_count"] == "1"
    assert mapping["gate_status"] == "BLOCKED_REPORT_ONLY"
    assert len(summary.attribution_rows) == len(rows)


def test_no_positive_generation_or_training_claims(tmp_path: Path) -> None:
    _, _, _, report = _run_sample(tmp_path)

    text = report.read_text(encoding="utf-8").lower()

    assert "no model training is run" in text
    assert "no synthetic data generation is run" in text
    assert "training succeeded" not in text
    assert "generation succeeded" not in text
    assert "performance improvement" not in text
    assert "integration readiness achieved" not in text
    assert "transfer benefit is established" not in text


def test_missing_input_fails_clearly(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b3_csv = tmp_path / "adversarial_auc.csv"
    b3_md = tmp_path / "adversarial_auc.md"
    b4_csv = tmp_path / "transfer_validation.csv"
    _write_b3_csv(b3_csv, [_b3_row()])
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    with pytest.raises(FileNotFoundError, match="B2 CSV not found"):
        module.run_report(
            b2_csv=tmp_path / "missing.csv",
            b3_csv=b3_csv,
            b3_md=b3_md,
            b4_csv=b4_csv,
            csv_path=tmp_path / "out.csv",
            report_path=tmp_path / "out.md",
        )


def test_thresholds_fixed_and_not_cli_mutable() -> None:
    module = _load_exp05_module()

    assert module.SMOKE_AUC_THRESHOLD == 0.85
    assert module.STRETCH_AUC_THRESHOLD == 0.70
    assert module.PCA_FAIL_THRESHOLD == 0.0
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--smoke-threshold", "0.99"])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--stretch-threshold", "0.99"])


def test_uncalibrated_raw_is_only_authoritative_lane(tmp_path: Path) -> None:
    _, _, out_csv, report = _run_sample(tmp_path)

    rows = _read_output(out_csv)
    uncalibrated = next(
        row
        for row in rows
        if row["category"] == "comparison_space" and row["value"] == "uncalibrated_raw"
    )
    snv = next(row for row in rows if row["category"] == "comparison_space" and row["value"] == "snv")

    assert uncalibrated["raw_authoritative_count"] == "3"
    assert snv["raw_authoritative_count"] == "0"
    text = report.read_text(encoding="utf-8")
    assert "Existing AUC values and B3 fail flags are read as fixed report inputs." in text
    assert "uncalibrated_raw" in text


def test_legacy_raw_rows_do_not_count_as_authoritative(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b2_csv = tmp_path / "b2.csv"
    b3_csv = tmp_path / "b3.csv"
    b3_md = tmp_path / "b3.md"
    b4_csv = tmp_path / "b4.csv"
    out_csv = tmp_path / "out.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _b2_row(dataset="BEER/Beer_OriginalExtract", preset="wine", space="raw", auc="0.91"),
            _b2_row(dataset="BEER/Beer_OriginalExtract", preset="wine", space="snv", auc="0.50"),
        ],
    )
    _write_b3_csv(
        b3_csv,
        [_b3_row(dataset="BEER/Beer_OriginalExtract", preset="wine", space="raw", auc="0.91")],
    )
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    summary = module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )

    rows = _read_output(out_csv)
    legacy = next(
        (row for row in rows if row["category"] == "comparison_space" and row["value"] == "raw"),
        None,
    )
    assert legacy is not None
    assert legacy["raw_authoritative_count"] == "0"
    named = {(row["category"], row["value"]) for row in rows}
    assert ("named_gap", "BEER") not in named
    # Without any authoritative `uncalibrated_raw` rows, the gate must fail closed.
    assert summary.b2_raw_realism_failed is True
    assert summary.b3_no_go is True
    assert summary.raw_compared_count == 0


def test_b2_b3_no_go_reflected(tmp_path: Path) -> None:
    _, summary, _, report = _run_sample(tmp_path)

    text = report.read_text(encoding="utf-8")

    assert summary.b2_raw_realism_failed is True
    assert summary.b3_no_go is True
    assert "B2 raw realism gate failed reflected: `true`" in text
    assert "B3 NO-GO reflected: `true`" in text


def test_authoritative_b2_row_missing_auc_fails_closed(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b2_csv = tmp_path / "b2.csv"
    b3_csv = tmp_path / "b3.csv"
    b3_md = tmp_path / "b3.md"
    b4_csv = tmp_path / "b4.csv"
    out_csv = tmp_path / "out.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [
            _b2_row(
                dataset="UNIT/MissingAuc",
                preset="grain",
                space="uncalibrated_raw",
                status="compared",
                auc="",
            ),
        ],
    )
    _write_b3_csv(
        b3_csv,
        [
            _b3_row(
                dataset="UNIT/MissingAuc",
                preset="grain",
                space="uncalibrated_raw",
                auc="",
                smoke_fail="false",
                stretch_fail="false",
            ),
        ],
    )
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    summary = module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )

    assert summary.b2_raw_realism_failed is True
    assert summary.b3_no_go is True
    assert summary.raw_compared_count == 1
    assert "B2 raw realism gate failed reflected: `true`" in report.read_text(encoding="utf-8")
    assert "B3 NO-GO reflected: `true`" in report.read_text(encoding="utf-8")


def test_legacy_raw_b3_authoritative_flag_is_ignored(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b2_csv = tmp_path / "b2.csv"
    b3_csv = tmp_path / "b3.csv"
    b3_md = tmp_path / "b3.md"
    b4_csv = tmp_path / "b4.csv"
    out_csv = tmp_path / "out.csv"
    report = tmp_path / "report.md"
    _write_b2_csv(
        b2_csv,
        [_b2_row(dataset="UNIT/Legacy", preset="grain", space="raw", auc="0.40")],
    )
    raw_b3 = _b3_row(dataset="UNIT/Legacy", preset="grain", space="raw", auc="0.40")
    raw_b3["raw_authoritative"] = "true"
    _write_b3_csv(b3_csv, [raw_b3])
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    summary = module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )

    rows = _read_output(out_csv)
    legacy = next(row for row in rows if row["category"] == "comparison_space" and row["value"] == "raw")
    assert legacy["raw_authoritative_count"] == "0"
    assert legacy["auc_smoke_fail_count"] == "0"
    assert summary.b2_raw_realism_failed is True
    assert summary.b3_no_go is True


def test_named_gaps_appear(tmp_path: Path) -> None:
    _, _, out_csv, report = _run_sample(tmp_path)

    rows = _read_output(out_csv)
    named = {(row["category"], row["value"]) for row in rows}

    assert ("named_gap", "BEER") in named
    assert ("named_gap", "DIESEL") in named
    assert ("named_gap", "CORN") in named
    text = report.read_text(encoding="utf-8")
    assert "`BEER`" in text
    assert "`DIESEL`" in text
    assert "`CORN`" in text


def test_b4_blocked_gate_reflected(tmp_path: Path) -> None:
    _, summary, out_csv, report = _run_sample(tmp_path)

    rows = _read_output(out_csv)
    b4 = next(row for row in rows if row["category"] == "blocker_class" and row["value"] == "B4:BLOCKED_BY_REALISM_GATE")

    assert summary.b4_blocked is True
    assert b4["blocked_count"] == "1"
    assert b4["auc_smoke_fail_count"] == "1"
    assert "B4 blocked: `true`" in report.read_text(encoding="utf-8")


def test_fail_closed_when_no_authoritative_b2_or_b3_rows(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b2_csv = tmp_path / "b2.csv"
    b3_csv = tmp_path / "b3.csv"
    b3_md = tmp_path / "b3.md"
    b4_csv = tmp_path / "b4.csv"
    out_csv = tmp_path / "out.csv"
    report = tmp_path / "report.md"
    # Only diagnostic lanes are present (snv + calibrated_raw_diagnostic) — no authoritative rows.
    _write_b2_csv(
        b2_csv,
        [
            _b2_row(dataset="UNIT/Diag", preset="grain", space="snv", auc="0.50"),
            _b2_row(
                dataset="UNIT/Diag",
                preset="grain",
                space="calibrated_raw_diagnostic",
                auc="0.55",
            ),
        ],
    )
    _write_b3_csv(
        b3_csv,
        [
            _b3_row(dataset="UNIT/Diag", preset="grain", space="snv", auc="0.50"),
            _b3_row(
                dataset="UNIT/Diag",
                preset="grain",
                space="calibrated_raw_diagnostic",
                auc="0.55",
            ),
        ],
    )
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    summary = module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )

    # Fail closed: no authoritative `uncalibrated_raw` B2 rows and no authoritative B3 rows.
    assert summary.b2_raw_realism_failed is True
    assert summary.b3_no_go is True
    text = report.read_text(encoding="utf-8")
    assert "B2 raw realism gate failed reflected: `true`" in text
    assert "B3 NO-GO reflected: `true`" in text


def test_diagnostic_smoke_fail_not_counted_in_authoritative_attribution(tmp_path: Path) -> None:
    module = _load_exp05_module()
    b2_csv = tmp_path / "b2.csv"
    b3_csv = tmp_path / "b3.csv"
    b3_md = tmp_path / "b3.md"
    b4_csv = tmp_path / "b4.csv"
    out_csv = tmp_path / "out.csv"
    report = tmp_path / "report.md"
    # Authoritative row passes; diagnostic lanes have smoke/stretch fails that must NOT
    # bleed into generic attribution counts.
    _write_b2_csv(
        b2_csv,
        [
            _b2_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="uncalibrated_raw",
                auc="0.60",
            ),
            _b2_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="calibrated_raw_diagnostic",
                auc="0.95",
                decision="provisional_review:adversarial_auc",
            ),
            _b2_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="snv",
                auc="0.92",
                decision="provisional_review:adversarial_auc",
            ),
        ],
    )
    _write_b3_csv(
        b3_csv,
        [
            _b3_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="uncalibrated_raw",
                auc="0.60",
                smoke_fail="false",
                stretch_fail="false",
            ),
            _b3_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="calibrated_raw_diagnostic",
                auc="0.95",
                smoke_fail="true",
                stretch_fail="true",
            ),
            _b3_row(
                dataset="UNIT/Mix",
                preset="grain",
                space="snv",
                auc="0.92",
                smoke_fail="true",
                stretch_fail="true",
            ),
        ],
    )
    b3_md.write_text("# B3\n", encoding="utf-8")
    _write_b4_csv(b4_csv)

    module.run_report(
        b2_csv=b2_csv,
        b3_csv=b3_csv,
        b3_md=b3_md,
        b4_csv=b4_csv,
        csv_path=out_csv,
        report_path=report,
    )

    rows = _read_output(out_csv)
    # The mapping_strategy group covers all three B2 rows. Only the authoritative row's
    # B3 flags may count; the two diagnostic rows must not contribute.
    mapping = next(
        row for row in rows if row["category"] == "mapping_strategy" and row["value"] == "matrix_first_dataset"
    )
    assert mapping["row_count"] == "3"
    assert mapping["raw_authoritative_count"] == "1"
    assert mapping["auc_smoke_fail_count"] == "0"
    assert mapping["auc_stretch_fail_count"] == "0"

    # Diagnostic comparison_space groups must not advertise authoritative smoke/stretch failures.
    snv = next(row for row in rows if row["category"] == "comparison_space" and row["value"] == "snv")
    diag = next(
        row
        for row in rows
        if row["category"] == "comparison_space" and row["value"] == "calibrated_raw_diagnostic"
    )
    assert snv["raw_authoritative_count"] == "0"
    assert snv["auc_smoke_fail_count"] == "0"
    assert snv["auc_stretch_fail_count"] == "0"
    assert diag["raw_authoritative_count"] == "0"
    assert diag["auc_smoke_fail_count"] == "0"
    assert diag["auc_stretch_fail_count"] == "0"
