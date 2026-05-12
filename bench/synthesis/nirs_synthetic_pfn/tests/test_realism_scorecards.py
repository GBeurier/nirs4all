from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from csv import DictReader
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
from nirsyntheticpfn.evaluation.realism import (
    RealDataset,
    align_to_real_grid,
    apply_covariance_calibration,
    apply_real_marginal_calibration,
    blocked_scorecard_row,
    compare_real_synthetic,
    compute_adversarial_auc,
    compute_nearest_neighbor_ratio,
    compute_pca_overlap,
    discover_local_real_datasets,
    fit_real_marginal_calibration,
    is_index_fallback_grid,
    sanitize_finite_spectra,
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


def _write_tmp_real_dataset(
    root: Path,
    *,
    database_name: str,
    dataset: str,
    wavelengths: np.ndarray,
    header: list[str] | None = None,
    n_train: int = 8,
    n_test: int = 8,
) -> None:
    cohort_dir = root / "bench/AOM_v0/benchmarks"
    data_dir = root / f"bench/tabpfn_paper/data/regression/{database_name}/{dataset}"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    header_items = header if header is not None else [f"{float(wavelength):.8g}" for wavelength in wavelengths]
    base = np.sin(np.arange(len(header_items), dtype=float) / 7.0)

    def write_matrix(path: Path, *, start: int, n_rows: int) -> None:
        lines = [";".join(f'"{item}"' for item in header_items)]
        for row_idx in range(start, start + n_rows):
            row = base + row_idx * 0.01
            lines.append(";".join(f"{value:.8f}" for value in row))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_matrix(data_dir / "Xtrain.csv", start=0, n_rows=n_train)
    write_matrix(data_dir / "Xtest.csv", start=n_train, n_rows=n_test)
    (data_dir / "Ytrain.csv").write_text("y\n" + "\n".join(["0"] * n_train) + "\n", encoding="utf-8")
    (data_dir / "Ytest.csv").write_text("y\n" + "\n".join(["0"] * n_test) + "\n", encoding="utf-8")
    (cohort_dir / "cohort_regression.csv").write_text(
        "\n".join([
            "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path",
            (
                f"{database_name},{dataset},ok,,{n_train},{n_test},{len(header_items)},"
                f"bench/tabpfn_paper/data/regression/{database_name}/{dataset}/Xtrain.csv,"
                f"bench/tabpfn_paper/data/regression/{database_name}/{dataset}/Xtest.csv,"
                f"bench/tabpfn_paper/data/regression/{database_name}/{dataset}/Ytrain.csv,"
                f"bench/tabpfn_paper/data/regression/{database_name}/{dataset}/Ytest.csv"
            ),
        ])
        + "\n",
        encoding="utf-8",
    )
    (cohort_dir / "cohort_classification.csv").write_text(
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n",
        encoding="utf-8",
    )


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
    assert first_metadata["imputed"] is False
    assert first_metadata["replays_real_rows"] is False
    assert first_metadata["status"] == "provisional"
    assert first_metadata["strength"] == "strong"
    assert "Strong provisional marginal calibration" in first_metadata["warning"]
    assert calibration.metadata["source"] == "real_X_and_real_wavelengths_only"
    assert calibration.metadata["imputed"] is False


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


def test_select_synthetic_run_for_dataset_is_matrix_first_non_oracle_and_stable() -> None:
    exp02 = _load_exp02_module()
    synthetic_runs = [
        ("grain", object()),
        ("forage", object()),
        ("oilseeds", object()),
        ("wine", object()),
        ("juice", object()),
        ("dairy", object()),
        ("fuel", object()),
        ("tablets", object()),
        ("meat", object()),
        ("baking", object()),
        ("powders", object()),
        ("soil", object()),
    ]

    milk = RealDataset("src", "regression", "MILK", "Milk_Fat_1224_KS", "", "", "", "", None, None, None)
    diesel = RealDataset("src", "regression", "DIESEL", "DIESEL_bp50_246_b-a", "", "", "", "", None, None, None)
    tablet = RealDataset("src", "regression", "TABLET", "Escitalopramt_310_Zhao", "", "", "", "", None, None, None)
    beer = RealDataset("src", "regression", "BEER", "Beer_OriginalExtract_60_KS", "", "", "", "", None, None, None)
    fruit_puree = RealDataset("src", "regression", "FruitPuree", "Brix_juice_context", "", "", "", "", None, None, None)
    berry = RealDataset("src", "regression", "BERRY", "ph_groupSampleID_stratDateVar_balRows", "", "", "", "", None, None, None)
    cassava = RealDataset("src", "regression", "CASSAVA", "Cassava_Starch_150", "", "", "", "", None, None, None)
    corn_oil = RealDataset("src", "regression", "CORN", "Corn_Oil_80_ZhengChenPelegYbaseSplit", "", "", "", "", None, None, None)
    corn_starch = RealDataset("src", "regression", "CORN", "Corn_Starch_80_ZhengChenPelegYbaseSplit", "", "", "", "", None, None, None)
    leaf = RealDataset("src", "regression", "GRAPEVINE_LeafTraits", "An_spxyG70_30_byCultivar_NeoSpectra", "", "", "", "", None, None, None)
    dark_resp = RealDataset("src", "regression", "DarkResp", "Rd25_XSBNtestSite", "", "", "", "", None, None, None)
    ecosis = RealDataset("src", "regression", "ECOSIS", "Arabidopsis_leaf_reflectance", "", "", "", "", None, None, None)
    phosphorus = RealDataset("src", "regression", "PHOSPHORUS", "LUCAS_P_Quartz_Incombustible", "", "", "", "", None, None, None)
    fallback = RealDataset("src", "regression", "UNKNOWN", "NoKnownToken", "", "", "", "", None, None, None)

    assert exp02.select_synthetic_run_for_dataset(milk, synthetic_runs)[0] == "dairy"
    assert exp02.select_synthetic_run_for_dataset(diesel, synthetic_runs)[0] == "fuel"
    assert exp02.select_synthetic_run_for_dataset(tablet, synthetic_runs)[0] == "tablets"
    beer_selected = exp02.select_synthetic_run_for_dataset(beer, synthetic_runs)
    assert beer_selected[0] == "wine"
    assert beer_selected[0] != "baking"
    assert beer_selected[2] == "matrix_first_dataset"
    assert "y_labels_splits_targets_not_used=true" in beer_selected[3]
    assert exp02.select_synthetic_run_for_dataset(fruit_puree, synthetic_runs)[0] == "juice"
    assert exp02.select_synthetic_run_for_dataset(berry, synthetic_runs)[0] == "juice"
    assert exp02.select_synthetic_run_for_dataset(cassava, synthetic_runs)[0] == "grain"
    assert exp02.select_synthetic_run_for_dataset(corn_oil, synthetic_runs)[0] == "grain"
    assert exp02.select_synthetic_run_for_dataset(corn_oil, synthetic_runs)[0] != "oilseeds"
    assert exp02.select_synthetic_run_for_dataset(corn_starch, synthetic_runs)[0] == "grain"
    assert exp02.select_synthetic_run_for_dataset(leaf, synthetic_runs)[0] == "forage"
    assert exp02.select_synthetic_run_for_dataset(leaf, synthetic_runs)[0] != "fruit"
    assert exp02.select_synthetic_run_for_dataset(dark_resp, synthetic_runs)[0] == "forage"
    assert exp02.select_synthetic_run_for_dataset(ecosis, synthetic_runs)[0] == "forage"
    assert exp02.select_synthetic_run_for_dataset(phosphorus, synthetic_runs)[0] == "soil"
    assert exp02.select_synthetic_run_for_dataset(phosphorus, synthetic_runs)[0] != "powders"
    first = exp02.select_synthetic_run_for_dataset(fallback, synthetic_runs)
    second = exp02.select_synthetic_run_for_dataset(fallback, synthetic_runs)
    reordered = exp02.select_synthetic_run_for_dataset(fallback, list(reversed(synthetic_runs)))
    assert first[0] == second[0]
    assert first[0] == reordered[0]
    assert first[2] == "stable_hash_fallback"


def test_source_overrides_are_audited_for_liquid_emulsion_and_powder_presets() -> None:
    exp02 = _load_exp02_module()
    beer = RealDataset("src", "regression", "BEER", "Beer_OriginalExtract_60_KS", "", "", "", "", None, None, None)
    puree = RealDataset("src", "regression", "FruitPuree", "Brix_juice_context", "", "", "", "", None, None, None)
    soil = RealDataset("src", "regression", "LUCAS", "PHOSPHORUS_QUARTZ", "", "", "", "", None, None, None)

    wine_overrides, wine_audit = exp02.source_overrides_for_dataset(dataset=beer, preset="wine")
    juice_overrides, juice_audit = exp02.source_overrides_for_dataset(dataset=puree, preset="juice")
    soil_overrides, soil_audit = exp02.source_overrides_for_dataset(dataset=soil, preset="soil")

    assert wine_overrides["matrix_type"] == "liquid"
    assert wine_overrides["measurement_mode"] == "transmittance"
    assert juice_overrides["matrix_type"] == "emulsion"
    assert juice_overrides["measurement_mode"] == "transmittance"
    assert soil_overrides["matrix_type"] == "powder"
    for audit in (wine_audit, juice_audit, soil_audit):
        assert audit["enabled"] is True
        assert audit["oracle"] is False
        assert audit["label_inputs_used"] is False
        assert audit["target_inputs_used"] is False
        assert audit["split_inputs_used"] is False
        assert audit["thresholds_modified"] is False
        assert audit["metrics_modified"] is False
        assert audit["imputed"] is False


def test_b2_source_overrides_use_matrix_presets_not_target_names() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.linspace(1100.0, 2250.0, 64)
    milk_urea = RealDataset("src", "regression", "MILK", "Milk_Urea_1224_KS", "", "", "", "", None, None, None)
    diesel = RealDataset("src", "regression", "DIESEL", "DIESEL_bp50_246_b-a", "", "", "", "", None, None, None)
    corn_oil = RealDataset("src", "regression", "CORN", "Corn_Oil_80_ZhengChenPelegYbaseSplit", "", "", "", "", None, None, None)
    beer = RealDataset("src", "regression", "BEER", "Beer_OriginalExtract_60_KS", "", "", "", "", None, None, None)

    milk_overrides, milk_audit = exp02.source_overrides_for_dataset(
        dataset=milk_urea,
        preset="dairy",
        real_wavelengths=real_wavelengths,
    )
    diesel_overrides, diesel_audit = exp02.source_overrides_for_dataset(
        dataset=diesel,
        preset="fuel",
        real_wavelengths=np.linspace(900.0, 1700.0, 64),
    )
    corn_overrides, corn_audit = exp02.source_overrides_for_dataset(
        dataset=corn_oil,
        preset="grain",
        real_wavelengths=real_wavelengths,
    )
    beer_overrides, beer_audit = exp02.source_overrides_for_dataset(
        dataset=beer,
        preset="wine",
        real_wavelengths=real_wavelengths,
    )

    assert milk_overrides["matrix_type"] in {"emulsion", "liquid"}
    assert milk_overrides["particle_size"] == 7.5
    assert milk_overrides["components"] == ["water", "lactose", "casein", "lipid"]
    assert "urea" not in milk_overrides["components"]

    assert diesel_overrides["matrix_type"] == "liquid"
    assert diesel_overrides["particle_size"] == 2.0
    assert diesel_overrides["components"] == ["alkane", "aromatic", "oil", "methanol"]

    assert corn_overrides["matrix_type"] == "granular"
    assert corn_overrides["particle_size"] == 250.0
    assert corn_overrides["components"] == ["starch", "protein", "moisture", "lipid"]
    assert corn_overrides["components"] != ["oil"]

    assert beer_overrides["matrix_type"] == "liquid"
    assert beer_overrides["measurement_mode"] == "transmittance"
    assert beer_overrides["_bench_wavelength_support_override"]["enabled"] is True
    assert beer_overrides["_bench_wavelength_support_override"]["domain_range"] == (1100.0, 2250.0)
    for audit in (milk_audit, diesel_audit, corn_audit, beer_audit):
        assert audit["non_oracle"] is True
        assert audit["no_target_or_label"] is True
        assert audit["oracle"] is False
        assert audit["label_inputs_used"] is False
        assert audit["target_inputs_used"] is False
        assert audit["split_inputs_used"] is False
        assert audit["imputed"] is False
        assert audit["rules"]


def test_instrument_token_overrides_are_canonical_and_audited() -> None:
    exp02 = _load_exp02_module()
    micronir = RealDataset("src", "regression", "MILK", "Milk_Fat_MicroNIR", "", "", "", "", None, None, None)
    neospectra = RealDataset("src", "regression", "GRAPEVINE_LeafTraits", "An_NeoSpectra", "", "", "", "", None, None, None)

    micronir_overrides, micronir_audit = exp02.source_overrides_for_dataset(dataset=micronir, preset="dairy")
    neospectra_overrides, neospectra_audit = exp02.source_overrides_for_dataset(
        dataset=neospectra,
        preset="forage",
    )

    assert micronir_overrides["instrument"] == "viavi_micronir"
    assert micronir_audit["instrument"]["applied"] is True
    assert micronir_audit["instrument"]["non_oracle"] is True
    assert neospectra_overrides["instrument"] == "siware_neoscanner"
    assert neospectra_audit["instrument"]["applied"] is True
    assert neospectra_audit["instrument"]["no_target_or_label"] is True


def test_scorecard_mapping_and_override_audits_do_not_use_labels_targets_or_splits() -> None:
    exp02 = _load_exp02_module()
    synthetic_runs = [("dairy", object()), ("grain", object()), ("fuel", object()), ("wine", object())]
    corn_oil = RealDataset("src", "regression", "CORN", "Corn_Oil_80_ZhengChenPelegYbaseSplit", "", "", "", "", None, None, None)

    selected = exp02.select_synthetic_run_for_dataset(corn_oil, synthetic_runs)
    signature = inspect.signature(exp02.source_overrides_for_dataset)
    overrides, audit = exp02.source_overrides_for_dataset(dataset=corn_oil, preset=selected[0])

    assert selected[0] == "grain"
    assert selected[2] == "matrix_first_dataset"
    assert "non_oracle=true" in selected[3]
    assert "y_labels_splits_targets_not_used=true" in selected[3]
    assert set(signature.parameters) == {"dataset", "preset", "real_wavelengths"}
    assert overrides["components"] == ["starch", "protein", "moisture", "lipid"]
    assert audit["non_oracle"] is True
    assert audit["label_inputs_used"] is False
    assert audit["target_inputs_used"] is False
    assert audit["split_inputs_used"] is False


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


def test_sanitize_finite_spectra_no_op_when_all_finite() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(20, 16))
    wavelengths = np.linspace(1000.0, 2400.0, 16)

    cleaned, cleaned_wl, audit, blocked = sanitize_finite_spectra(
        X, wavelengths, side="real"
    )

    assert blocked is None
    assert cleaned is not None and cleaned_wl is not None
    assert cleaned.shape == X.shape
    assert audit["action"] == "no_op"
    assert audit["dropped_rows"] == 0
    assert audit["dropped_cols"] == 0
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False
    assert audit["imputed"] is False
    assert audit["finite_policy"] == "drop_nonfinite_no_imputation"


def test_sanitize_finite_spectra_drop_rows_when_few_rows_have_non_finite() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 16))
    X[1, 4] = np.nan
    X[15, 9] = np.inf
    wavelengths = np.linspace(1000.0, 2400.0, 16)

    cleaned, cleaned_wl, audit, blocked = sanitize_finite_spectra(
        X, wavelengths, side="real"
    )

    assert blocked is None
    assert cleaned is not None and cleaned_wl is not None
    assert cleaned.shape == (18, 16)
    assert np.isfinite(cleaned).all()
    assert cleaned_wl.shape == (16,)
    assert audit["action"] == "drop_rows"
    assert audit["dropped_rows"] == 2
    assert audit["dropped_cols"] == 0
    assert audit["side"] == "real"
    assert audit["n_rows_after"] == 18
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False


def test_sanitize_finite_spectra_drop_columns_when_every_row_has_non_finite() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(20, 16))
    X[:, 3] = np.nan
    X[:, 11] = np.nan
    wavelengths = np.linspace(1000.0, 2400.0, 16)

    cleaned, cleaned_wl, audit, blocked = sanitize_finite_spectra(
        X, wavelengths, side="synthetic"
    )

    assert blocked is None
    assert cleaned is not None and cleaned_wl is not None
    assert cleaned.shape == (20, 14)
    assert np.isfinite(cleaned).all()
    assert cleaned_wl.shape == (14,)
    assert audit["action"] == "drop_columns"
    assert audit["dropped_rows"] == 0
    assert audit["dropped_cols"] == 2
    assert audit["side"] == "synthetic"


def test_sanitize_finite_spectra_mixed_drop_recovers_finite_block() -> None:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(10, 6))
    X[0:4, 0] = np.nan
    X[0, 1] = np.nan
    X[0, 2] = np.nan
    X[1, 2] = np.nan
    X[1, 3] = np.nan
    wavelengths = np.linspace(1000.0, 1500.0, 6)

    cleaned, cleaned_wl, audit, blocked = sanitize_finite_spectra(
        X, wavelengths, side="real"
    )

    assert blocked is None
    assert cleaned is not None and cleaned_wl is not None
    assert cleaned.shape == (8, 5)
    assert cleaned_wl.shape == (5,)
    assert np.isfinite(cleaned).all()
    assert audit["action"] == "drop_columns_then_rows"
    assert audit["dropped_rows"] == 2
    assert audit["dropped_cols"] == 1


def test_sanitize_finite_spectra_blocks_when_retention_below_threshold() -> None:
    X = np.full((10, 10), np.nan)
    X[0, 0] = 1.0
    wavelengths = np.linspace(1000.0, 1100.0, 10)

    cleaned, cleaned_wl, audit, blocked = sanitize_finite_spectra(
        X, wavelengths, side="real"
    )

    assert blocked is not None and "non_finite_retention_below_threshold" in blocked
    assert cleaned is None
    assert cleaned_wl is None
    assert audit["action"] == "blocked"
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False


def test_failure_class_marks_strict_non_finite_validation_as_non_finite_spectra() -> None:
    exp02 = _load_exp02_module()

    failure_class = exp02._failure_class(ValueError("spectra contain non-finite values"))

    assert failure_class == "non_finite_spectra"


def test_exp02_covariance_calibration_metadata_is_disabled_by_default() -> None:
    exp02 = _load_exp02_module()

    metadata = exp02._covariance_disabled_metadata()

    assert metadata["enabled"] is False
    assert metadata["reason"] == "disabled_by_default_auc_regression"
    assert metadata["oracle"] is False
    assert metadata["thresholds_modified"] is False
    assert metadata["metrics_modified"] is False
    assert metadata["imputed"] is False


def test_apply_covariance_calibration_is_deterministic_non_oracle_and_no_replay() -> None:
    rng = np.random.default_rng(2026)
    real_X = rng.normal(loc=0.5, scale=0.3, size=(40, 32))
    synthetic_X = rng.normal(loc=-1.0, scale=1.5, size=(40, 32))

    first, first_meta = apply_covariance_calibration(real_X, synthetic_X)
    second, second_meta = apply_covariance_calibration(real_X, synthetic_X)

    np.testing.assert_allclose(first, second)
    assert first.shape == synthetic_X.shape
    assert first_meta == second_meta
    assert first_meta["enabled"] is True
    assert first_meta["rank"] == min(8, real_X.shape[0] - 2, real_X.shape[1])
    assert first_meta["oracle"] is False
    assert first_meta["label_inputs_used"] is False
    assert first_meta["target_inputs_used"] is False
    assert first_meta["split_inputs_used"] is False
    assert first_meta["source_oracle_used"] is False
    assert first_meta["thresholds_modified"] is False
    assert first_meta["metrics_modified"] is False
    assert first_meta["imputed"] is False
    assert first_meta["replays_real_rows"] is False
    assert first_meta["status"] == "provisional"
    assert first_meta["strength"] == "strong"
    assert "covariance calibration" in first_meta["warning"].lower()
    row_matches = np.isclose(
        first[:, np.newaxis, :], real_X[np.newaxis, :, :], rtol=0.0, atol=1e-12
    ).all(axis=2)
    assert not row_matches.any()


def test_apply_covariance_calibration_disabled_when_rank_cap_below_two() -> None:
    real_X = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]])
    synthetic_X = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])

    out, meta = apply_covariance_calibration(real_X, synthetic_X)

    np.testing.assert_allclose(out, synthetic_X)
    assert meta["enabled"] is False
    assert meta["reason"] == "rank_cap_below_two"
    assert meta["oracle"] is False


def test_header_wavelengths_parses_bom_quotes_and_unit_suffixes(tmp_path: Path) -> None:
    from nirsyntheticpfn.evaluation.realism import _load_semicolon_matrix

    bom = "﻿"
    csv_path = tmp_path / "bom.csv"
    csv_path.write_text(
        f"{bom}\"339_nm\";\"340 nm\";\"341\"\n0.1;0.2;0.3\n0.11;0.21;0.31\n",
        encoding="utf-8",
    )

    X, wavelengths = _load_semicolon_matrix(csv_path)

    assert X.shape == (2, 3)
    assert np.allclose(wavelengths, [339.0, 340.0, 341.0])
    assert is_index_fallback_grid(wavelengths) is False


def test_is_index_fallback_grid_distinguishes_numeric_grids_from_arange_index() -> None:
    assert is_index_fallback_grid(np.arange(10)) is True
    assert is_index_fallback_grid(np.linspace(1100.0, 2400.0, 10)) is False
    assert is_index_fallback_grid(np.array([0.0, 1.0, np.nan, 3.0])) is True


def test_grid_compatible_preset_fallback_remaps_numeric_grid_when_original_blocked() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.linspace(1100.0, 2298.0, 600)

    selected, audit = exp02.grid_compatible_preset_fallback(
        original_preset="fruit",
        real_wavelengths=real_wavelengths,
        available_presets=["grain", "fruit", "dairy", "fuel", "tablets"],
    )

    assert selected != "fruit"
    assert audit["original_preset"] == "fruit"
    assert audit["selected_preset"] == selected
    assert audit["reason"] == "grid_compatible_fallback"
    assert audit["is_index_fallback"] is False
    assert audit["imputed"] is False
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False


def test_grid_compatible_preset_fallback_preserves_preset_when_overlap_exists() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.linspace(1100.0, 2400.0, 600)

    selected, audit = exp02.grid_compatible_preset_fallback(
        original_preset="grain",
        real_wavelengths=real_wavelengths,
        available_presets=["grain", "fruit", "dairy"],
    )

    assert selected == "grain"
    assert audit["selected_preset"] == "grain"
    assert audit["reason"] == "no_remap_needed"


def test_grid_compatible_preset_fallback_skips_index_fallback_grids() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.arange(1003, dtype=float)

    selected, audit = exp02.grid_compatible_preset_fallback(
        original_preset="dairy",
        real_wavelengths=real_wavelengths,
        available_presets=["grain", "fruit", "dairy"],
    )

    assert selected == "dairy"
    assert audit["is_index_fallback"] is True
    assert audit["reason"] == "wavelength_grid_unknown"


def test_run_scorecards_blocks_index_fallback_grid_without_scoring(tmp_path: Path) -> None:
    exp02 = _load_exp02_module()
    _write_tmp_real_dataset(
        tmp_path,
        database_name="UNKNOWN",
        dataset="HeaderFallback",
        wavelengths=np.arange(12, dtype=float),
        header=[f"feature_{idx}" for idx in range(12)],
    )

    result = exp02.run_scorecards(
        root=tmp_path,
        n_synthetic_samples=12,
        max_real_samples=12,
        max_real_datasets=0,
        seed=20260429,
    )

    assert result["synthetic_run_count"] == 0
    assert result["load_failures"][0]["failure_class"] == "wavelength_grid_unknown"
    row = result["rows"][0]
    assert row.status == "blocked"
    assert row.blocked_reason.startswith("wavelength_grid_unknown:")
    reason = json.loads(row.synthetic_mapping_reason)
    assert reason["generation"]["skipped"] is True
    assert reason["grid_remap"]["reason"] == "wavelength_grid_unknown"
    assert reason["grid_remap"]["is_index_fallback"] is True
    assert "clipped" not in row.blocked_reason


def test_run_scorecards_blocks_index_fallback_before_nonfinite_column_drop(tmp_path: Path) -> None:
    exp02 = _load_exp02_module()
    _write_tmp_real_dataset(
        tmp_path,
        database_name="UNKNOWN",
        dataset="HeaderFallbackWithNanColumn",
        wavelengths=np.arange(12, dtype=float),
        header=[f"feature_{idx}" for idx in range(12)],
    )
    data_dir = tmp_path / "bench/tabpfn_paper/data/regression/UNKNOWN/HeaderFallbackWithNanColumn"
    for matrix_path in (data_dir / "Xtrain.csv", data_dir / "Xtest.csv"):
        lines = matrix_path.read_text(encoding="utf-8").splitlines()
        rewritten = [lines[0]]
        for line in lines[1:]:
            cells = line.split(";")
            cells[0] = "nan"
            rewritten.append(";".join(cells))
        matrix_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    result = exp02.run_scorecards(
        root=tmp_path,
        n_synthetic_samples=12,
        max_real_samples=12,
        max_real_datasets=0,
        seed=20260429,
    )

    assert result["synthetic_run_count"] == 0
    assert result["load_failures"][0]["failure_class"] == "wavelength_grid_unknown"
    row = result["rows"][0]
    assert row.status == "blocked"
    assert row.blocked_reason.startswith("wavelength_grid_unknown:")
    reason = json.loads(row.synthetic_mapping_reason)
    assert reason["grid_remap"]["is_index_fallback"] is True
    assert reason["sanitation"]["real"]["action"] == "skipped_due_to_wavelength_grid_unknown"


def test_matrix_first_leaf_mapping_scores_as_forage_not_fruit_or_hash(tmp_path: Path) -> None:
    exp02 = _load_exp02_module()
    _write_tmp_real_dataset(
        tmp_path,
        database_name="GRAPEVINE_LeafTraits",
        dataset="An_spxyG70_30_byCultivar_NeoSpectra",
        wavelengths=np.linspace(1100.0, 2298.0, 64),
    )

    result = exp02.run_scorecards(
        root=tmp_path,
        n_synthetic_samples=12,
        max_real_samples=12,
        max_real_datasets=0,
        seed=20260429,
    )

    assert result["synthetic_run_count"] == 1
    assert result["load_failures"] == []
    row = result["rows"][0]
    assert row.status == "compared"
    assert row.synthetic_preset == "forage"
    assert row.synthetic_preset != "fruit"
    assert row.synthetic_mapping_strategy == "matrix_first_dataset"
    reason = json.loads(row.synthetic_mapping_reason)
    assert "leaf_plant_matrix" in reason["mapping_reason"]
    assert reason["grid_remap"]["reason"] == "no_remap_needed"
    assert reason["grid_remap"]["original_preset"] == "forage"
    assert reason["grid_remap"]["selected_preset"] == "forage"


def test_grid_compatible_preset_fallback_blocks_semantic_cross_domain_fallback() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.linspace(1100.0, 2298.0, 600)

    selected, audit = exp02.grid_compatible_preset_fallback(
        original_preset="fruit",
        real_wavelengths=real_wavelengths,
        available_presets=["grain", "fruit", "dairy", "fuel", "tablets"],
        allow_cross_domain_fallback=False,
    )

    assert selected == "fruit"
    assert audit["selected_preset"] == "fruit"
    assert audit["reason"] == "domain_wavelength_support"
    assert audit["allow_cross_domain_fallback"] is False


def test_grid_compatible_preset_fallback_returns_original_when_no_alternative_supports_grid() -> None:
    exp02 = _load_exp02_module()
    real_wavelengths = np.linspace(7000.0, 10000.0, 300)

    selected, audit = exp02.grid_compatible_preset_fallback(
        original_preset="tablets",
        real_wavelengths=real_wavelengths,
        available_presets=["grain", "fruit", "dairy", "tablets"],
    )

    assert selected == "tablets"
    assert audit["reason"] == "no_grid_compatible_alternative"


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

    beer_reason = json.dumps({
        "generation": {
            "source_overrides": {
                "enabled": True,
                "rules": ["beer_wine_liquid_transmittance", "beer_wine_real_grid_support"],
                "overrides": {
                    "_bench_wavelength_support_override": {
                        "enabled": True,
                        "oracle": False,
                        "label_inputs_used": False,
                        "target_inputs_used": False,
                        "split_inputs_used": False,
                    },
                },
            },
            "canonical_wavelength_policy": {
                "bench_wavelength_support_override": {
                    "enabled": True,
                    "applied": True,
                    "oracle": False,
                    "label_inputs_used": False,
                    "target_inputs_used": False,
                    "split_inputs_used": False,
                },
            },
        },
        "calibration": {
            "marginal": {"grid_strategy": "same_grid"},
            "covariance": {"enabled": False},
        },
        "grid_remap": {"reason": "no_remap_needed"},
    })
    beer_row = replace(
        row,
        dataset="BEER/Beer_OriginalExtract_60_KS",
        synthetic_preset="wine",
        synthetic_mapping_reason=beer_reason,
        adversarial_auc=1.0,
        pca_overlap=0.0,
        provisional_decision="provisional_review:pca_overlap,adversarial_auc",
    )
    report = exp02.render_markdown(
        result={
            "status": "done",
            "rows": [row, beer_row],
            "inventories": [],
            "real_runnable_count": 2,
            "real_selected_count": 2,
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
    assert "## Source Overrides" in report
    assert "Wavelength support override applied rows: 1" in report
    assert "## R9 Gap Summary" in report
    assert "R9 is a partial diagnostic improvement, not a B2 pass." in report
    assert "`BEER/Beer_OriginalExtract_60_KS`" in report
