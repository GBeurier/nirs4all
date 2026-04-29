from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType

import numpy as np
from nirsyntheticpfn.evaluation.realism import (
    RealDataset,
    align_to_real_grid,
    apply_real_marginal_calibration,
    blocked_scorecard_row,
    compare_real_synthetic,
    compute_adversarial_auc,
    compute_nearest_neighbor_ratio,
    compute_pca_overlap,
    discover_local_real_datasets,
    fit_real_marginal_calibration,
    summarize_spectra,
    synthetic_only_row,
    write_scorecard_csv,
)


def _load_exp02_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments/exp02_real_synthetic_scorecards.py"
    experiments_dir = str(path.parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    spec = importlib.util.spec_from_file_location("exp02_real_synthetic_scorecards", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_spectra_contains_required_b2_metrics() -> None:
    wavelengths = np.linspace(900.0, 1700.0, 64)
    base = np.sin(wavelengths / 120.0)
    X = np.vstack([base + idx * 0.01 for idx in range(12)])

    summary = summarize_spectra(X, wavelengths)

    assert summary.n_samples == 12
    assert summary.n_wavelengths == 64
    assert summary.wavelength_min == 900.0
    assert summary.wavelength_max == 1700.0
    assert np.isfinite(summary.spectral_mean_mean)
    assert np.isfinite(summary.derivative_variance_median)
    assert np.isfinite(summary.correlation_length_median)
    assert np.isfinite(summary.snr_median)
    assert np.isfinite(summary.baseline_curvature_median)
    assert np.isfinite(summary.peak_density_median)


def test_align_to_real_grid_interpolates_synthetic_to_real_overlap() -> None:
    real_wavelengths = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    synthetic_wavelengths = np.array([900.0, 1050.0, 1200.0, 1350.0])
    real_X = np.arange(8, dtype=float).reshape(2, 4)
    synthetic_X = np.vstack([
        synthetic_wavelengths / 1000.0,
        synthetic_wavelengths / 900.0,
    ])

    real_aligned, synthetic_aligned, wavelengths = align_to_real_grid(
        real_X,
        real_wavelengths,
        synthetic_X,
        synthetic_wavelengths,
    )

    assert wavelengths.tolist() == [1000.0, 1100.0, 1200.0, 1300.0]
    assert real_aligned.shape == (2, 4)
    assert synthetic_aligned.shape == (2, 4)
    assert np.allclose(synthetic_aligned[0], [1.0, 1.1, 1.2, 1.3])


def test_compare_real_synthetic_returns_standardized_row() -> None:
    wavelengths = np.linspace(1000.0, 1800.0, 48)
    base = np.sin(wavelengths / 100.0)
    real_X = np.vstack([base + idx * 0.002 for idx in range(16)])
    synthetic_X = np.vstack([base + idx * 0.0025 for idx in range(16)])

    row = compare_real_synthetic(
        real_X=real_X,
        real_wavelengths=wavelengths,
        synthetic_X=synthetic_X,
        synthetic_wavelengths=wavelengths,
        dataset="unit/test",
        source="unit",
        task="regression",
        synthetic_preset="grain",
        comparison_space="raw",
        random_state=7,
    )

    assert row.status == "compared"
    assert row.comparison_space == "raw"
    assert row.dataset == "unit/test"
    assert row.n_real_samples == 16
    assert row.n_synthetic_samples == 16
    assert row.real_spectral_mean_mean is not None
    assert row.synthetic_spectral_variance_mean is not None
    assert row.derivative_log10_gap is not None
    assert row.provisional_decision


def test_compare_real_synthetic_snv_is_valid_and_does_not_modify_inputs() -> None:
    wavelengths = np.linspace(1000.0, 1800.0, 48)
    base = np.sin(wavelengths / 100.0)
    real_X = np.vstack([base + idx * 0.002 for idx in range(16)])
    synthetic_X = np.vstack([base + idx * 0.0025 for idx in range(16)])
    real_before = real_X.copy()
    synthetic_before = synthetic_X.copy()

    row = compare_real_synthetic(
        real_X=real_X,
        real_wavelengths=wavelengths,
        synthetic_X=synthetic_X,
        synthetic_wavelengths=wavelengths,
        dataset="unit/test",
        source="unit",
        task="regression",
        synthetic_preset="grain",
        comparison_space="snv",
        random_state=11,
    )

    assert row.status == "compared"
    assert row.comparison_space == "snv"
    assert row.n_real_samples == 16
    assert row.n_synthetic_samples == 16
    assert row.real_spectral_mean_mean is not None
    assert np.allclose(real_X, real_before)
    assert np.allclose(synthetic_X, synthetic_before)


def test_real_marginal_calibration_is_non_oracle_deterministic_and_copy_safe() -> None:
    wavelengths = np.linspace(1000.0, 1800.0, 48)
    base = np.sin(wavelengths / 100.0)
    real_X = np.vstack([base + idx * 0.003 for idx in range(18)])
    synthetic_X = np.vstack([base * 0.5 + 2.0 + idx * 0.001 for idx in range(18)])
    real_before = real_X.copy()
    synthetic_before = synthetic_X.copy()

    signature = inspect.signature(fit_real_marginal_calibration)
    assert "y" not in signature.parameters
    assert "labels" not in signature.parameters

    calibration = fit_real_marginal_calibration(real_X, wavelengths)
    first, first_metadata = apply_real_marginal_calibration(
        synthetic_X,
        wavelengths,
        calibration,
    )
    second, second_metadata = apply_real_marginal_calibration(
        synthetic_X,
        wavelengths,
        calibration,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(real_X, real_before)
    np.testing.assert_allclose(synthetic_X, synthetic_before)
    assert first.shape == synthetic_X.shape
    assert first_metadata == second_metadata
    assert first_metadata["oracle"] is False
    assert first_metadata["label_inputs_used"] is False
    assert first_metadata["target_inputs_used"] is False
    assert first_metadata["split_inputs_used"] is False
    assert first_metadata["source_oracle_used"] is False
    assert first_metadata["thresholds_modified"] is False
    assert first_metadata["metrics_modified"] is False
    assert first_metadata["replays_real_rows"] is False
    assert first_metadata["status"] == "provisional"
    assert first_metadata["strength"] == "strong"
    assert "Strong provisional marginal calibration" in first_metadata["warning"]
    assert calibration.metadata["source"] == "real_X_and_real_wavelengths_only"


def test_real_marginal_calibration_does_not_replay_real_rows() -> None:
    rng = np.random.default_rng(42)
    wavelengths = np.linspace(1000.0, 1800.0, 32)
    real_X = rng.normal(loc=0.5, scale=0.2, size=(20, 32))
    synthetic_X = rng.normal(loc=-1.0, scale=0.7, size=(20, 32))

    calibration = fit_real_marginal_calibration(real_X, wavelengths)
    calibrated, metadata = apply_real_marginal_calibration(
        synthetic_X,
        wavelengths,
        calibration,
    )

    row_matches = np.isclose(
        calibrated[:, np.newaxis, :],
        real_X[np.newaxis, :, :],
        rtol=0.0,
        atol=1e-12,
    ).all(axis=2)
    assert metadata["replays_real_rows"] is False
    assert not row_matches.any()


def test_pca_adversarial_and_nn_metrics_have_valid_shapes_without_label_leakage() -> None:
    rng = np.random.default_rng(123)
    real_X = rng.normal(size=(24, 32))
    synthetic_X = real_X.copy()

    pca_overlap = compute_pca_overlap(real_X, synthetic_X, random_state=3)
    adversarial_auc, adversarial_auc_std = compute_adversarial_auc(
        real_X,
        synthetic_X,
        random_state=3,
    )
    nn_ratio = compute_nearest_neighbor_ratio(real_X, synthetic_X)

    assert pca_overlap is not None
    assert 0.0 <= pca_overlap <= 1.0
    assert adversarial_auc is not None
    assert adversarial_auc_std is not None
    assert 0.0 <= adversarial_auc <= 1.0
    assert adversarial_auc < 0.75
    assert adversarial_auc_std >= 0.0
    assert nn_ratio is not None
    assert nn_ratio >= 0.0


def test_metric_helpers_return_none_when_shapes_are_too_small() -> None:
    real_X = np.ones((2, 4))
    synthetic_X = np.ones((2, 4))

    assert compute_pca_overlap(real_X, synthetic_X, random_state=1) is None
    assert compute_adversarial_auc(real_X, synthetic_X, random_state=1) == (None, None)


def test_synthetic_only_row_is_explicitly_blocked() -> None:
    wavelengths = np.linspace(1000.0, 1300.0, 16)
    X = np.vstack([np.sin(wavelengths / 100.0), np.cos(wavelengths / 100.0)])

    row = synthetic_only_row(
        synthetic_X=X,
        synthetic_wavelengths=wavelengths,
        synthetic_preset="grain",
        blocked_reason="blocked_no_real_data: no rows",
    )

    assert row.status == "synthetic_only"
    assert row.comparison_space == "raw"
    assert row.n_real_samples == 0
    assert row.provisional_decision == "blocked_no_real_data"
    assert "blocked_no_real_data" in row.blocked_reason


def test_blocked_scorecard_row_preserves_selected_real_failure_accounting() -> None:
    row = blocked_scorecard_row(
        source="AOM_regression",
        task="regression",
        dataset="DB/DS",
        synthetic_preset="grain",
        blocked_reason="wavelength_grid_overlap: fewer than three overlapping points",
    )

    assert row.status == "blocked"
    assert row.comparison_space == "raw"
    assert row.source == "AOM_regression"
    assert row.dataset == "DB/DS"
    assert row.n_real_samples == 0
    assert row.n_synthetic_samples == 0
    assert row.adversarial_auc is None
    assert row.provisional_decision == "blocked_score_failure"
    assert row.blocked_reason.startswith("wavelength_grid_overlap:")


def test_scorecard_row_to_dict_includes_mapping_and_space_fields() -> None:
    row = blocked_scorecard_row(
        source="AOM_regression",
        task="regression",
        dataset="DB/DS",
        synthetic_preset="grain",
        blocked_reason="blocked",
        comparison_space="raw",
        synthetic_mapping_strategy="dataset_aware_token",
        synthetic_mapping_reason="matched token 'corn' -> grain",
    )

    data = row.to_dict()

    assert data["comparison_space"] == "raw"
    assert data["synthetic_mapping_strategy"] == "dataset_aware_token"
    assert data["synthetic_mapping_reason"] == "matched token 'corn' -> grain"


def test_select_synthetic_run_for_dataset_is_dataset_aware_and_stable() -> None:
    exp02 = _load_exp02_module()
    synthetic_runs = [
        ("grain", object()),
        ("dairy", object()),
        ("fuel", object()),
        ("tablets", object()),
        ("meat", object()),
    ]

    milk = RealDataset("src", "regression", "MILK", "Milk_Fat_1224_KS", "", "", "", "", None, None, None)
    diesel = RealDataset("src", "regression", "DIESEL", "DIESEL_bp50_246_b-a", "", "", "", "", None, None, None)
    tablet = RealDataset("src", "regression", "TABLET", "Escitalopramt_310_Zhao", "", "", "", "", None, None, None)
    fallback = RealDataset("src", "regression", "UNKNOWN", "NoKnownToken", "", "", "", "", None, None, None)

    assert exp02.select_synthetic_run_for_dataset(milk, synthetic_runs)[0] == "dairy"
    assert exp02.select_synthetic_run_for_dataset(diesel, synthetic_runs)[0] == "fuel"
    assert exp02.select_synthetic_run_for_dataset(tablet, synthetic_runs)[0] == "tablets"
    first = exp02.select_synthetic_run_for_dataset(fallback, synthetic_runs)
    second = exp02.select_synthetic_run_for_dataset(fallback, synthetic_runs)
    reordered = exp02.select_synthetic_run_for_dataset(fallback, list(reversed(synthetic_runs)))
    assert first[0] == second[0]
    assert first[0] == reordered[0]
    assert first[2] == "stable_hash_fallback"


def test_on_demand_synthetic_source_uses_real_grid_request() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.arange(1100.0, 1142.0, 2.0)

    source, metadata = exp02.synthetic_source_for_real_grid(
        preset="grain",
        target_type="regression",
        target_size=1,
        seed=123,
        real_wavelengths=real_wavelengths,
    )

    assert source["wavelength_range"] == (1100.0, 1140.0)
    assert source["spectral_resolution"] == 2.0
    assert metadata["grid_source"] == "real_grid"
    assert metadata["wavelength_min"] == 1100.0
    assert metadata["wavelength_max"] == 1140.0
    assert metadata["median_step"] == 2.0


def test_on_demand_synthetic_source_clamps_tiny_real_grid_step() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.arange(1100.0, 1110.0, 0.1)

    source, metadata = exp02.synthetic_source_for_real_grid(
        preset="grain",
        target_type="regression",
        target_size=1,
        seed=123,
        real_wavelengths=real_wavelengths,
    )

    assert metadata["median_step"] < 1.0
    assert source["spectral_resolution"] == 2.0
    assert metadata["source_spectral_resolution"] == 2.0
    assert metadata["min_source_spectral_resolution"] == 2.0


def test_on_demand_synthetic_source_invalid_grid_fallback_is_explicit_and_stable() -> None:
    exp02 = _load_exp02_module()
    invalid_wavelengths = np.array([1000.0, np.nan, 1002.0])

    first_source, first_metadata = exp02.synthetic_source_for_real_grid(
        preset="grain",
        target_type="regression",
        target_size=1,
        seed=123,
        real_wavelengths=invalid_wavelengths,
    )
    second_source, second_metadata = exp02.synthetic_source_for_real_grid(
        preset="grain",
        target_type="regression",
        target_size=1,
        seed=123,
        real_wavelengths=invalid_wavelengths,
    )

    assert first_source["wavelength_range"] == second_source["wavelength_range"]
    assert first_source["spectral_resolution"] == second_source["spectral_resolution"]
    assert first_metadata == second_metadata
    assert first_metadata["grid_source"] == "preset_default_fallback"
    assert "fallback_reason" in first_metadata


def test_discover_local_real_datasets_counts_runnable_and_missing_paths(tmp_path: Path) -> None:
    cohort_dir = tmp_path / "bench/AOM_v0/benchmarks"
    data_dir = tmp_path / "bench/tabpfn_paper/data/regression/DB/DS"
    cohort_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    for filename in ["Xtrain.csv", "Xtest.csv", "Ytrain.csv", "Ytest.csv"]:
        (data_dir / filename).write_text('"1000";"1001"\n1;2\n', encoding="utf-8")
    (cohort_dir / "cohort_regression.csv").write_text(
        "\n".join([
            "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path",
            "DB,DS,ok,,1,1,2,bench/tabpfn_paper/data/regression/DB/DS/Xtrain.csv,"
            "bench/tabpfn_paper/data/regression/DB/DS/Xtest.csv,"
            "bench/tabpfn_paper/data/regression/DB/DS/Ytrain.csv,"
            "bench/tabpfn_paper/data/regression/DB/DS/Ytest.csv",
            "DB,MISSING,ok,,1,1,2,missing/Xtrain.csv,missing/Xtest.csv,missing/Ytrain.csv,missing/Ytest.csv",
        ]),
        encoding="utf-8",
    )
    (cohort_dir / "cohort_classification.csv").write_text(
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n",
        encoding="utf-8",
    )

    datasets, inventories = discover_local_real_datasets(tmp_path)

    assert len(datasets) == 1
    regression_inventory = next(item for item in inventories if item.source == "AOM_regression")
    assert regression_inventory.total_rows == 2
    assert regression_inventory.ok_rows == 2
    assert regression_inventory.runnable_rows == 1
    assert regression_inventory.missing_rows == 1
    assert "missing/Xtrain.csv" in regression_inventory.missing_paths


def test_scorecard_csv_and_report_expose_b2_audit_fields(tmp_path: Path) -> None:
    exp02 = _load_exp02_module()
    wavelengths = np.linspace(1000.0, 1800.0, 48)
    base = np.sin(wavelengths / 100.0)
    real_X = np.vstack([base + idx * 0.002 for idx in range(16)])
    synthetic_X = np.vstack([base + idx * 0.0025 for idx in range(16)])
    calibration_reason = json.dumps({
        "calibration": {
            "grid_strategy": "same_grid",
            "warning": "Strong provisional marginal calibration for B2 diagnostics only",
        }
    })
    row = compare_real_synthetic(
        real_X=real_X,
        real_wavelengths=wavelengths,
        synthetic_X=synthetic_X,
        synthetic_wavelengths=wavelengths,
        dataset="unit/test",
        source="unit",
        task="regression",
        synthetic_preset="grain",
        comparison_space="raw",
        synthetic_mapping_strategy="dataset_aware_token",
        synthetic_mapping_reason=calibration_reason,
        random_state=7,
    )

    csv_path = tmp_path / "scorecards.csv"
    write_scorecard_csv([row], csv_path)
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_row = next(DictReader(file))
    assert csv_row["comparison_space"] == "raw"
    assert csv_row["synthetic_mapping_strategy"] == "dataset_aware_token"
    assert "calibration" in csv_row["synthetic_mapping_reason"]

    report = exp02.render_markdown(
        result={
            "status": "done",
            "rows": [row],
            "inventories": [],
            "real_runnable_count": 1,
            "real_selected_count": 1,
            "synthetic_run_count": 1,
            "load_failures": [],
        },
        report_path=tmp_path / "scorecards.md",
        csv_path=csv_path,
        n_synthetic_samples=16,
        max_real_samples=16,
        max_real_datasets=1,
        seed=7,
        git_status={
            "returncode": 0,
            "line_count": 0,
            "lines": [],
            "truncated": False,
        },
    )
    assert "## Real Marginal Calibration" in report
    assert "Strong provisional marginal calibration" in report
    assert "Thresholds are not changed" in report
    assert 'comparison_space == "raw"' in report
