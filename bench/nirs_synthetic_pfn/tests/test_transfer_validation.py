from __future__ import annotations

from pathlib import Path

import numpy as np
from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE
from nirsyntheticpfn.evaluation.realism import RealDataset
from nirsyntheticpfn.evaluation.transfer import (
    B2_REALISM_RISK,
    REAL_ONLY_CLAIM_STATUS,
    SYNTHETIC_DIAGNOSTIC_BLOCKED_BY_B2_STATUS,
    RegressionCohort,
    evaluate_regression_transfer_smoke,
    load_regression_cohort,
    write_transfer_csv,
)


def test_load_regression_cohort_combines_real_train_test_labels(tmp_path: Path) -> None:
    dataset = _write_regression_dataset(tmp_path, n_train=5, n_test=3, n_features=6)

    cohort = load_regression_cohort(dataset, root=tmp_path)

    assert cohort.name == "DB/DS"
    assert cohort.X.shape == (8, 6)
    assert cohort.y.shape == (8,)
    assert cohort.wavelengths.tolist() == [1000.0, 1002.0, 1004.0, 1006.0, 1008.0, 1010.0]


def test_transfer_smoke_writes_real_baselines_synthetic_diagnostic_and_blocked_tstr() -> None:
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
        b2_realism_failed=True,
    )

    assert len(rows) == 10
    completed = [row for row in rows if row.status == "completed"]
    blocked = [row for row in rows if row.status == "blocked"]
    assert {row.model for row in completed} == {"ridge", "pls", "pca_ridge", "synthetic_pca_ridge"}
    assert len(blocked) == 2
    assert all(row.route == "TSTR" for row in blocked)
    assert all(row.phase_a_gate_override == PHASE_A_GATE_OVERRIDE for row in rows)
    assert all(row.b2_realism_risk == B2_REALISM_RISK for row in rows)
    assert all(
        row.synthetic_transfer_claim_status == REAL_ONLY_CLAIM_STATUS
        for row in completed
        if row.route == "real_only"
    )
    assert all(
        row.synthetic_transfer_claim_status == SYNTHETIC_DIAGNOSTIC_BLOCKED_BY_B2_STATUS
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
        b2_realism_failed=True,
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
        b2_realism_failed=True,
    )
    path = tmp_path / "transfer.csv"

    write_transfer_csv(rows, path)

    text = path.read_text(encoding="utf-8")
    assert "phase_a_gate_override,b1_downstream_training_status,b2_realism_risk" in text
    assert PHASE_A_GATE_OVERRIDE in text
    assert B2_REALISM_RISK in text


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
