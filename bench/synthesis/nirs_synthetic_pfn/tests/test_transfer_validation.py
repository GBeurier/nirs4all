from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE
from nirsyntheticpfn.evaluation.realism import RealDataset
from nirsyntheticpfn.evaluation.transfer import (
    REAL_ONLY_CLAIM_STATUS,
    REALISM_GATE_BLOCKED_STATUS,
    SYNTHETIC_DIAGNOSTIC_STATUS,
    RegressionCohort,
    evaluate_regression_transfer_smoke,
    load_regression_cohort,
    write_transfer_csv,
)

EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "exp03_transfer_validation.py"
sys.path.insert(0, str(EXPERIMENT_PATH.parent))
spec = importlib.util.spec_from_file_location("exp03_transfer_validation", EXPERIMENT_PATH)
assert spec is not None
assert spec.loader is not None
exp03 = importlib.util.module_from_spec(spec)
sys.modules["exp03_transfer_validation"] = exp03
spec.loader.exec_module(exp03)


def test_load_regression_cohort_combines_real_train_test_labels(tmp_path: Path) -> None:
    dataset = _write_regression_dataset(tmp_path, n_train=5, n_test=3, n_features=6)

    cohort = load_regression_cohort(dataset, root=tmp_path)

    assert cohort.name == "DB/DS"
    assert cohort.X.shape == (8, 6)
    assert cohort.y.shape == (8,)
    assert cohort.wavelengths.tolist() == [1000.0, 1002.0, 1004.0, 1006.0, 1008.0, 1010.0]


def test_transfer_smoke_writes_real_baselines_synthetic_diagnostic_when_gate_passed() -> None:
    wavelengths = np.linspace(1000.0, 1030.0, 16)
    x_axis = np.linspace(-1.0, 1.0, 16)
    X = np.vstack([x_axis + idx * 0.02 for idx in range(30)])
    y = X[:, 3] * 2.0 + X[:, 9] * -0.5
    cohort = RegressionCohort(
        dataset=RealDataset(
            source="unit",
            task="regression",
            database_name="DB",
            dataset="DS",
            train_path="unused",
            test_path="unused",
            ytrain_path="unused",
            ytest_path="unused",
            n_train_declared=None,
            n_test_declared=None,
            p_declared=None,
        ),
        X=X,
        y=y,
        wavelengths=wavelengths,
    )
    synthetic_X = X[:20] + 0.01

    rows = evaluate_regression_transfer_smoke(
        cohort=cohort,
        synthetic_X=synthetic_X,
        synthetic_wavelengths=wavelengths,
        synthetic_preset="grain",
        n_splits=2,
        test_fraction=0.25,
        random_state=11,
        b1_downstream_training_status="allowed",
        b2_realism_failed=False,
    )

    assert len(rows) == 10
    completed = [row for row in rows if row.status == "completed"]
    blocked = [row for row in rows if row.status == "blocked"]
    assert {row.model for row in completed} == {"ridge", "pls", "pca_ridge", "synthetic_pca_ridge"}
    assert len(blocked) == 2
    assert all(row.route == "TSTR" for row in blocked)
    assert all(row.phase_a_gate_override == PHASE_A_GATE_OVERRIDE for row in rows)
    assert all(row.b2_realism_risk == "B2_realism_not_failed" for row in rows)
    assert all(
        row.synthetic_transfer_claim_status == REAL_ONLY_CLAIM_STATUS
        for row in completed
        if row.route == "real_only"
    )
    assert all(
        row.synthetic_transfer_claim_status == SYNTHETIC_DIAGNOSTIC_STATUS
        for row in completed
        if row.route == "RTSR_diagnostic"
    )


def test_transfer_smoke_blocks_synthetic_pca_when_b1_blocks_training() -> None:
    wavelengths = np.linspace(1000.0, 1020.0, 8)
    X = np.vstack([np.sin(wavelengths / 100.0) + idx * 0.01 for idx in range(12)])
    y = np.arange(12, dtype=float)
    cohort = RegressionCohort(
        dataset=RealDataset(
            source="unit",
            task="regression",
            database_name="DB",
            dataset="DS",
            train_path="unused",
            test_path="unused",
            ytrain_path="unused",
            ytest_path="unused",
            n_train_declared=None,
            n_test_declared=None,
            p_declared=None,
        ),
        X=X,
        y=y,
        wavelengths=wavelengths,
    )

    rows = evaluate_regression_transfer_smoke(
        cohort=cohort,
        synthetic_X=X.copy(),
        synthetic_wavelengths=wavelengths,
        synthetic_preset="grain",
        n_splits=1,
        test_fraction=0.25,
        random_state=5,
        b1_downstream_training_status="blocked",
        b2_realism_failed=False,
    )

    synthetic_pca = next(row for row in rows if row.model == "synthetic_pca_ridge")
    assert synthetic_pca.status == "blocked"
    assert synthetic_pca.blocked_reason == "blocked_by_B1_prior_predictive_status"


def test_write_transfer_csv_preserves_gate_flags(tmp_path: Path) -> None:
    wavelengths = np.linspace(1000.0, 1010.0, 6)
    X = np.vstack([np.linspace(0.0, 1.0, 6) + idx for idx in range(10)])
    cohort = RegressionCohort(
        dataset=RealDataset(
            source="unit",
            task="regression",
            database_name="DB",
            dataset="DS",
            train_path="unused",
            test_path="unused",
            ytrain_path="unused",
            ytest_path="unused",
            n_train_declared=None,
            n_test_declared=None,
            p_declared=None,
        ),
        X=X,
        y=np.arange(10, dtype=float),
        wavelengths=wavelengths,
    )
    rows = evaluate_regression_transfer_smoke(
        cohort=cohort,
        synthetic_X=X.copy(),
        synthetic_wavelengths=wavelengths,
        synthetic_preset="grain",
        n_splits=1,
        test_fraction=0.3,
        random_state=3,
        b1_downstream_training_status="allowed",
        b2_realism_failed=False,
    )
    path = tmp_path / "transfer.csv"

    write_transfer_csv(rows, path)

    text = path.read_text(encoding="utf-8")
    assert "phase_a_gate_override,b1_downstream_training_status,b2_realism_risk" in text
    assert PHASE_A_GATE_OVERRIDE in text


def test_gate_blocks_before_synthetic_build_or_transfer_fit(tmp_path: Path, monkeypatch: Any) -> None:
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_adversarial_gate_csv(
        adversarial_csv,
        raw_rows=[
            {"status": "compared", "adversarial_auc": "0.91", "smoke_fail": "true"},
            {"status": "compared", "adversarial_auc": "", "smoke_fail": "false"},
            {"status": "blocked", "adversarial_auc": "", "smoke_fail": "false"},
        ],
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("gate-first block must not call synthetic build or transfer fit")

    monkeypatch.setattr(exp03, "_build_validated_synthetic_run", fail_if_called)
    monkeypatch.setattr(exp03, "evaluate_regression_transfer_smoke", fail_if_called)

    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    assert result["status"] == REALISM_GATE_BLOCKED_STATUS
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row.status == REALISM_GATE_BLOCKED_STATUS
    assert row.model == "none"
    assert row.n_train == 0
    assert row.n_test == 0
    assert row.n_synthetic == 0
    assert row.raw_compared == 2
    assert row.raw_smoke_failures == 1
    assert row.raw_blocked == 1
    assert row.raw_missing_auc == 1


def test_blocked_outputs_report_no_transfer_claim_and_raw_gate_stats(tmp_path: Path) -> None:
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    output_csv = tmp_path / "transfer_validation.csv"
    report = tmp_path / "transfer_validation.md"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_adversarial_gate_csv(
        adversarial_csv,
        raw_rows=[
            {"status": "compared", "adversarial_auc": "0.91", "smoke_fail": "true"},
            {"status": "blocked", "adversarial_auc": "", "smoke_fail": "false"},
        ],
    )
    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    write_transfer_csv(result["rows"], output_csv)
    report.write_text(
        exp03.render_markdown(
            result=result,
            report_path=report,
            csv_path=output_csv,
            b2_csv=b2_csv,
            adversarial_csv=adversarial_csv,
            max_real_datasets=3,
            max_samples=180,
            n_splits=2,
            test_fraction=0.25,
            n_synthetic_samples=80,
            seed=20260429,
            git_status={"line_count": 0, "lines": [], "truncated": False},
        ),
        encoding="utf-8",
    )

    csv_rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert csv_rows[0]["status"] == REALISM_GATE_BLOCKED_STATUS
    assert csv_rows[0]["model"] == "none"
    assert csv_rows[0]["n_train"] == "0"
    assert csv_rows[0]["n_synthetic"] == "0"
    assert csv_rows[0]["raw_compared"] == "1"
    assert csv_rows[0]["raw_smoke_failures"] == "1"
    assert csv_rows[0]["raw_blocked"] == "1"
    assert csv_rows[0]["raw_missing_auc"] == "0"
    markdown = report.read_text(encoding="utf-8")
    assert REALISM_GATE_BLOCKED_STATUS in markdown
    assert "No transfer claim is made." in markdown
    assert "Fitted model count: 0." in markdown
    assert "TSTR/RTSR route count: 0." in markdown
    assert "B3 raw gate status: `NO-GO`" in markdown
    assert "Route runnable" not in markdown


def test_missing_adversarial_report_blocks_with_b2_raw_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "missing_adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.93], include_blocked=True)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing adversarial report must not bypass the realism gate")

    monkeypatch.setattr(exp03, "_build_validated_synthetic_run", fail_if_called)
    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    row = result["rows"][0]
    assert result["status"] == REALISM_GATE_BLOCKED_STATUS
    assert "missing_adversarial_auc_csv" in row.blocked_reason
    assert row.raw_compared == 1
    assert row.raw_smoke_failures == 1
    assert row.raw_blocked == 1
    assert row.raw_missing_auc == 0


def test_snv_only_adversarial_rows_cannot_pass_gate(tmp_path: Path, monkeypatch: Any) -> None:
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_snv_only_adversarial_gate_csv(adversarial_csv)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("SNV-only rows must not pass the raw authoritative gate")

    monkeypatch.setattr(exp03, "_build_validated_synthetic_run", fail_if_called)
    monkeypatch.setattr(exp03, "evaluate_regression_transfer_smoke", fail_if_called)

    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    assert result["status"] == REALISM_GATE_BLOCKED_STATUS
    assert result["gate_summary"]["raw_authoritative"] is True
    assert result["gate_summary"]["snv_can_pass_gate"] is False
    assert result["adversarial_summary"]["raw_compared"] == 0
    assert result["adversarial_summary"]["gate_status"] == "NO-GO"
    row = result["rows"][0]
    assert row.raw_compared == 0
    assert "adversarial_auc_raw_gate_NO-GO" in row.blocked_reason


def test_legacy_raw_only_adversarial_rows_fail_closed(tmp_path: Path, monkeypatch: Any) -> None:
    """Legacy `comparison_space=raw` rows must not satisfy the uncalibrated_raw authoritative gate."""
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_legacy_raw_only_adversarial_gate_csv(adversarial_csv)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw rows must not pass the uncalibrated_raw authoritative gate")

    monkeypatch.setattr(exp03, "_build_validated_synthetic_run", fail_if_called)
    monkeypatch.setattr(exp03, "evaluate_regression_transfer_smoke", fail_if_called)

    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    assert result["status"] == REALISM_GATE_BLOCKED_STATUS
    assert result["adversarial_summary"]["raw_compared"] == 0
    assert result["adversarial_summary"]["raw_rows"] == 0
    assert result["adversarial_summary"]["gate_status"] == "NO-GO"
    row = result["rows"][0]
    assert row.raw_compared == 0
    assert "adversarial_auc_raw_gate_NO-GO" in row.blocked_reason


def test_uncalibrated_raw_b3_rows_drive_blocked_row_counters(tmp_path: Path, monkeypatch: Any) -> None:
    """B3 `uncalibrated_raw` rows are counted as authoritative even when legacy `raw` rows exist."""
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_mixed_lane_adversarial_gate_csv(adversarial_csv)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("blocked gate must not invoke synthetic build or transfer fit")

    monkeypatch.setattr(exp03, "_build_validated_synthetic_run", fail_if_called)
    monkeypatch.setattr(exp03, "evaluate_regression_transfer_smoke", fail_if_called)

    result = exp03.run_transfer_validation(
        root=tmp_path,
        max_real_datasets=3,
        max_samples=180,
        n_splits=2,
        test_fraction=0.25,
        n_synthetic_samples=80,
        seed=20260429,
        b2_csv=b2_csv,
        adversarial_csv=adversarial_csv,
    )

    assert result["status"] == REALISM_GATE_BLOCKED_STATUS
    adv = result["adversarial_summary"]
    assert adv["raw_rows"] == 2
    assert adv["raw_compared"] == 1
    assert adv["raw_smoke_failures"] == 1
    assert adv["raw_blocked"] == 1
    assert adv["raw_missing_auc"] == 0
    row = result["rows"][0]
    assert row.raw_compared == 1
    assert row.raw_smoke_failures == 1
    assert row.raw_blocked == 1
    assert row.raw_missing_auc == 0


def test_snv_failure_does_not_block_when_raw_gate_passes(tmp_path: Path) -> None:
    b2_csv = tmp_path / "real_synthetic_scorecards.csv"
    adversarial_csv = tmp_path / "adversarial_auc.csv"
    _write_b2_gate_csv(b2_csv, auc_values=[0.4])
    _write_raw_pass_snv_fail_adversarial_gate_csv(adversarial_csv)

    gate_summary = exp03.inspect_realism_gate_status(b2_path=b2_csv, adversarial_path=adversarial_csv)

    assert gate_summary["status"] == "gate_passed"
    assert gate_summary["blocked"] is False
    assert gate_summary["adversarial_summary"]["raw_compared"] == 1
    assert gate_summary["adversarial_summary"]["raw_smoke_failures"] == 0
    assert gate_summary["snv_can_pass_gate"] is False


def _write_regression_dataset(
    root: Path,
    *,
    n_train: int,
    n_test: int,
    n_features: int,
) -> RealDataset:
    data_dir = root / "bench/tabpfn_paper/data/regression/DB/DS"
    data_dir.mkdir(parents=True)
    wavelengths = [1000.0 + idx * 2.0 for idx in range(n_features)]
    X_train = np.arange(n_train * n_features, dtype=float).reshape(n_train, n_features)
    X_test = np.arange(n_test * n_features, dtype=float).reshape(n_test, n_features) + 100.0
    y_train = np.arange(n_train, dtype=float)
    y_test = np.arange(n_test, dtype=float) + 10.0
    _write_matrix(data_dir / "Xtrain.csv", X_train, wavelengths)
    _write_matrix(data_dir / "Xtest.csv", X_test, wavelengths)
    _write_vector(data_dir / "Ytrain.csv", y_train)
    _write_vector(data_dir / "Ytest.csv", y_test)
    return RealDataset(
        source="unit",
        task="regression",
        database_name="DB",
        dataset="DS",
        train_path="bench/tabpfn_paper/data/regression/DB/DS/Xtrain.csv",
        test_path="bench/tabpfn_paper/data/regression/DB/DS/Xtest.csv",
        ytrain_path="bench/tabpfn_paper/data/regression/DB/DS/Ytrain.csv",
        ytest_path="bench/tabpfn_paper/data/regression/DB/DS/Ytest.csv",
        n_train_declared=n_train,
        n_test_declared=n_test,
        p_declared=n_features,
    )


def _write_matrix(path: Path, X: np.ndarray, wavelengths: list[float]) -> None:
    header = ";".join(f'"{wavelength:g}"' for wavelength in wavelengths)
    rows = [";".join(f"{value:g}" for value in row) for row in X]
    path.write_text("\n".join([header, *rows]), encoding="utf-8")


def _write_vector(path: Path, y: np.ndarray) -> None:
    rows = [f"{value:g}" for value in y]
    path.write_text("\n".join(['"x"', *rows]), encoding="utf-8")


def _write_b2_gate_csv(path: Path, *, auc_values: list[float], include_blocked: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "status",
            "source",
            "task",
            "dataset",
            "synthetic_preset",
            "comparison_space",
            "pca_overlap",
            "adversarial_auc",
        ],
    ]
    for idx, auc in enumerate(auc_values):
        rows.append(["compared", "unit", "regression", f"DB/DS{idx}", "grain", "uncalibrated_raw", "0.5", f"{auc:g}"])
    if include_blocked:
        rows.append(["blocked", "unit", "regression", "DB/BLOCKED", "grain", "uncalibrated_raw", "", ""])
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def _write_adversarial_gate_csv(path: Path, *, raw_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset",
        "source",
        "task",
        "preset",
        "comparison_space",
        "status",
        "adversarial_auc",
        "adversarial_auc_std",
        "smoke_fail",
        "stretch_fail",
        "companion_failures",
        "blocked_class",
        "gate_basis",
        "raw_authoritative",
        "failure_taxonomy",
    ]
    rows = [header]
    for idx, row in enumerate(raw_rows):
        rows.append([
            f"DB/DS{idx}",
            "unit",
            "regression",
            "grain",
            "uncalibrated_raw",
            row["status"],
            row["adversarial_auc"],
            "",
            row["smoke_fail"],
            "false",
            "",
            "wavelength_grid_unknown" if row["status"] == "blocked" else "",
            "raw_authoritative",
            "true",
            "raw_smoke_high_auc" if row["smoke_fail"] == "true" else "pass",
        ])
    rows.append([
        "DB/SNV",
        "unit",
        "regression",
        "grain",
        "snv",
        "compared",
        "0.1",
        "",
        "false",
        "false",
        "",
        "",
        "snv_diagnostic_only",
        "false",
        "pass",
    ])
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def _write_snv_only_adversarial_gate_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "dataset",
            "source",
            "task",
            "preset",
            "comparison_space",
            "status",
            "adversarial_auc",
            "adversarial_auc_std",
            "smoke_fail",
            "stretch_fail",
            "companion_failures",
            "blocked_class",
            "gate_basis",
            "raw_authoritative",
            "failure_taxonomy",
        ],
        [
            "DB/SNV_ONLY",
            "unit",
            "regression",
            "grain",
            "snv",
            "compared",
            "0.1",
            "",
            "false",
            "false",
            "",
            "",
            "raw_authoritative",
            "true",
            "pass",
        ],
    ]
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def _write_legacy_raw_only_adversarial_gate_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "dataset",
            "source",
            "task",
            "preset",
            "comparison_space",
            "status",
            "adversarial_auc",
            "adversarial_auc_std",
            "smoke_fail",
            "stretch_fail",
            "companion_failures",
            "blocked_class",
            "gate_basis",
            "raw_authoritative",
            "failure_taxonomy",
        ],
        [
            "DB/LEGACY_RAW",
            "unit",
            "regression",
            "grain",
            "raw",
            "compared",
            "0.4",
            "",
            "false",
            "false",
            "",
            "",
            "raw_authoritative",
            "true",
            "pass",
        ],
    ]
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def _write_mixed_lane_adversarial_gate_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset",
        "source",
        "task",
        "preset",
        "comparison_space",
        "status",
        "adversarial_auc",
        "adversarial_auc_std",
        "smoke_fail",
        "stretch_fail",
        "companion_failures",
        "blocked_class",
        "gate_basis",
        "raw_authoritative",
        "failure_taxonomy",
    ]
    rows = [
        header,
        [
            "DB/UR0",
            "unit",
            "regression",
            "grain",
            "uncalibrated_raw",
            "compared",
            "0.92",
            "",
            "true",
            "false",
            "",
            "",
            "raw_authoritative",
            "true",
            "raw_smoke_high_auc",
        ],
        [
            "DB/UR1",
            "unit",
            "regression",
            "grain",
            "uncalibrated_raw",
            "blocked",
            "",
            "",
            "false",
            "false",
            "",
            "wavelength_grid_unknown",
            "raw_authoritative",
            "true",
            "blocked_evidence_gap",
        ],
        [
            "DB/LEGACY",
            "unit",
            "regression",
            "grain",
            "raw",
            "compared",
            "0.30",
            "",
            "false",
            "false",
            "",
            "",
            "raw_authoritative",
            "true",
            "pass",
        ],
        [
            "DB/CALIB",
            "unit",
            "regression",
            "grain",
            "calibrated_raw_diagnostic",
            "compared",
            "0.40",
            "",
            "false",
            "false",
            "",
            "",
            "calibrated_raw_diagnostic_only",
            "false",
            "pass",
        ],
    ]
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def _write_raw_pass_snv_fail_adversarial_gate_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "dataset",
            "source",
            "task",
            "preset",
            "comparison_space",
            "status",
            "adversarial_auc",
            "adversarial_auc_std",
            "smoke_fail",
            "stretch_fail",
            "companion_failures",
            "blocked_class",
            "gate_basis",
            "raw_authoritative",
            "failure_taxonomy",
        ],
        [
            "DB/RAW",
            "unit",
            "regression",
            "grain",
            "uncalibrated_raw",
            "compared",
            "0.4",
            "",
            "false",
            "false",
            "",
            "",
            "raw_authoritative",
            "true",
            "pass",
        ],
        [
            "DB/SNV",
            "unit",
            "regression",
            "grain",
            "snv",
            "compared",
            "1.0",
            "",
            "true",
            "true",
            "",
            "",
            "snv_diagnostic_only",
            "false",
            "snv_smoke_high_auc",
        ],
    ]
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")
