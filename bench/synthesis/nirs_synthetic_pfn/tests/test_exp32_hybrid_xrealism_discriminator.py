"""Tests for the exp32 hybrid x-realism adversarial discriminator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_exp32() -> ModuleType:
    return _load_module(
        "exp32_hybrid_xrealism_discriminator",
        "exp32_hybrid_xrealism_discriminator.py",
    )


def _make_synthetic_real_distribution(
    *,
    n_samples: int = 200,
    n_features: int = 80,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth-spectra real-like distribution: low-rank linear combinations + noise."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(450.0, 2400.0, n_features)
    basis = np.stack(
        [
            np.exp(-((axis - 970.0) ** 2) / (2 * 60.0**2)),
            np.exp(-((axis - 1450.0) ** 2) / (2 * 80.0**2)),
            np.exp(-((axis - 1940.0) ** 2) / (2 * 120.0**2)),
            np.exp(-((axis - 1700.0) ** 2) / (2 * 50.0**2)),
        ],
        axis=0,
    )
    coeffs = rng.normal(0.5, 0.3, (n_samples, basis.shape[0]))
    baseline = 0.4 + 0.0001 * (axis - axis.mean())
    spectra = baseline + coeffs @ basis + rng.normal(0.0, 0.005, (n_samples, n_features))
    return spectra.astype(np.float64), axis.astype(np.float64)


def test_load_dataset_x_only_handles_quoted_int_headers(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    directory = tmp_path / "fake/quoted_int"
    directory.mkdir(parents=True)
    (directory / "Xtrain.csv").write_text(
        '"1100";"1102";"1104"\n'
        "0.1;0.2;0.3\n"
        "0.15;0.25;0.35\n",
        encoding="utf-8",
    )
    (directory / "Xtest.csv").write_text(
        '"1100";"1102";"1104"\n'
        "0.12;0.22;0.32\n",
        encoding="utf-8",
    )

    x, axis = exp32.load_dataset_x_only(directory)

    assert x.shape == (3, 3)
    assert axis.tolist() == [1100.0, 1102.0, 1104.0]


def test_load_dataset_x_only_handles_nm_suffix_headers(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    directory = tmp_path / "fake/nm_suffix"
    directory.mkdir(parents=True)
    (directory / "Xtrain.csv").write_text(
        "852.78_nm;853.34_nm;853.9_nm\n"
        "0.5;0.6;0.7\n",
        encoding="utf-8",
    )

    x, axis = exp32.load_dataset_x_only(directory)

    assert x.shape == (1, 3)
    assert axis.tolist() == [852.78, 853.34, 853.9]


def test_load_dataset_x_only_rejects_train_test_axis_mismatch(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    directory = tmp_path / "fake/mismatch"
    directory.mkdir(parents=True)
    (directory / "Xtrain.csv").write_text("450;460\n0.1;0.2\n", encoding="utf-8")
    (directory / "Xtest.csv").write_text("470;480\n0.3;0.4\n", encoding="utf-8")

    try:
        exp32.load_dataset_x_only(directory)
    except ValueError as exc:
        assert "axis mismatch" in str(exc)
    else:
        raise AssertionError("expected axis mismatch error")


def test_hybrid_generator_fits_and_samples_correct_shape() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=120, n_features=64, seed=11)
    config = exp32.HybridConfig(baseline_degree=2, max_peaks=8, n_pca_components=10, seed=42)

    gen = exp32.HybridGenerator(config).fit(spectra, axis)
    rng = np.random.default_rng(42)
    sampled = gen.sample(50, rng)

    assert sampled.shape == (50, 64)
    assert gen.peak_centers_ is not None and gen.peak_centers_.size > 0
    assert gen.n_pca_actual_ == 10
    assert gen.mechanistic_mean_ is not None and gen.mechanistic_mean_.shape == (64,)


def test_hybrid_generator_zero_pca_uses_only_mechanistic_plus_noise() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=80, n_features=48, seed=5)
    config = exp32.HybridConfig(baseline_degree=2, max_peaks=4, n_pca_components=0, seed=99)

    gen = exp32.HybridGenerator(config).fit(spectra, axis)
    sampled = gen.sample(20, np.random.default_rng(99))

    assert gen.pca_ is None
    assert gen.score_std_.size == 0
    assert sampled.shape == (20, 48)


def test_adversarial_auc_near_half_when_two_real_halves_compared() -> None:
    """Two disjoint random halves of the same distribution should be near AUC 0.5."""
    exp32 = _load_exp32()
    spectra, _ = _make_synthetic_real_distribution(n_samples=400, n_features=48, seed=3)
    rng = np.random.default_rng(3)
    perm = rng.permutation(spectra.shape[0])
    half_a = spectra[perm[:200]]
    half_b = spectra[perm[200:]]
    rf = exp32.adversarial_auc(half_a, half_b, classifier="rf", n_splits=3, seed=3, n_estimators=100)
    lr = exp32.adversarial_auc(half_a, half_b, classifier="lr", n_splits=3, seed=3)

    assert 0.35 <= rf.mean_auc <= 0.65, rf
    assert 0.35 <= lr.mean_auc <= 0.65, lr


def test_adversarial_auc_high_when_synthetic_is_constant() -> None:
    exp32 = _load_exp32()
    spectra, _ = _make_synthetic_real_distribution(n_samples=200, n_features=48, seed=21)
    constant = np.tile(spectra.mean(axis=0), (spectra.shape[0], 1))

    rf = exp32.adversarial_auc(spectra, constant, classifier="rf", n_splits=3, seed=21, n_estimators=100)

    assert rf.mean_auc >= 0.85, rf


def test_evaluate_dataset_returns_per_pca_rank_rows(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=160, n_features=48, seed=13)
    directory = tmp_path / "fake/eval_dataset"
    directory.mkdir(parents=True)
    header = ";".join(f"{value:g}" for value in axis)
    train_lines = [header] + [";".join(f"{v:g}" for v in row) for row in spectra[:120]]
    test_lines = [header] + [";".join(f"{v:g}" for v in row) for row in spectra[120:]]
    (directory / "Xtrain.csv").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (directory / "Xtest.csv").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    result = exp32.evaluate_dataset(
        directory,
        pca_range=(0, 5, 20),
        n_splits=3,
        n_estimators=80,
        seed=13,
    )

    assert result["status"] == "done"
    assert result["n_real"] == 160
    assert result["n_features"] == 48
    assert len(result["rows"]) == 3
    assert result["best_rf"] is not None
    assert all(0.0 <= row.auc_rf_mean <= 1.0 for row in result["rows"])


def test_evaluate_dataset_higher_pca_rank_reduces_rf_auc(tmp_path: Path) -> None:
    """Adding PCA components should non-trivially reduce discriminator AUC vs the rank-0 baseline."""
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=600, n_features=64, seed=17)
    directory = tmp_path / "fake/rank_trend"
    directory.mkdir(parents=True)
    header = ";".join(f"{value:g}" for value in axis)
    lines = [header] + [";".join(f"{v:g}" for v in row) for row in spectra]
    (directory / "Xtrain.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = exp32.evaluate_dataset(
        directory,
        pca_range=(0, 4, 16),
        n_splits=3,
        n_estimators=100,
        seed=17,
        classifiers=("rf",),
    )

    aucs = {row.n_pca_components_requested: row.auc_rf_mean for row in result["rows"]}
    assert aucs[16] < aucs[0] - 0.05, aucs
    assert aucs[16] < 0.95, aucs


def test_render_markdown_contains_strategy_anchors(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=100, n_features=32, seed=2)
    directory = tmp_path / "fake/markdown_dataset"
    directory.mkdir(parents=True)
    header = ";".join(f"{value:g}" for value in axis)
    lines = [header] + [";".join(f"{v:g}" for v in row) for row in spectra]
    (directory / "Xtrain.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = exp32.evaluate_dataset(directory, pca_range=(0, 5), n_splits=2, n_estimators=50, seed=2)
    markdown = exp32.render_markdown(result, report_path=tmp_path / "r.md", csv_path=tmp_path / "r.csv")

    for anchor in (
        "Adversarial AUC IS the tuning oracle",
        "no labels",
        "no targets",
        "no splits-as-oracle",
        "nirs4all/` import",
        "Per-PCA-Rank Sweep",
    ):
        assert anchor in markdown, anchor


def test_empirical_and_joint_bootstrap_sampling_modes_run_and_produce_correct_shape() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=120, n_features=48, seed=8)
    for mode in ("empirical", "joint_bootstrap"):
        config = exp32.HybridConfig(
            baseline_degree=2,
            max_peaks=4,
            n_pca_components=8,
            score_sampling_mode=mode,
            noise_sampling_mode=mode,
            empirical_jitter_fraction=0.1,
            seed=8,
        )
        gen = exp32.HybridGenerator(config).fit(spectra, axis)
        sampled = gen.sample(40, np.random.default_rng(8))
        assert sampled.shape == (40, 48), (mode, sampled.shape)
        assert np.all(np.isfinite(sampled)), mode


def test_gmm_score_sampling_mode_runs_and_produces_correct_shape() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=160, n_features=64, seed=9)
    config = exp32.HybridConfig(
        baseline_degree=2,
        max_peaks=4,
        n_pca_components=10,
        score_sampling_mode="gmm",
        score_gmm_components=4,
        score_gmm_covariance_type="full",
        seed=9,
    )
    gen = exp32.HybridGenerator(config).fit(spectra, axis)
    assert gen.score_gmm_ is not None
    sampled = gen.sample(50, np.random.default_rng(9))
    assert sampled.shape == (50, 64)
    assert np.all(np.isfinite(sampled))


def test_knn_mixup_score_sampling_mode_runs_and_produces_correct_shape() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=160, n_features=64, seed=12)
    config = exp32.HybridConfig(
        baseline_degree=2,
        max_peaks=4,
        n_pca_components=10,
        score_sampling_mode="knn_mixup",
        score_knn_mixup_k=5,
        score_knn_mixup_dirichlet_alpha=1.0,
        seed=12,
    )
    gen = exp32.HybridGenerator(config).fit(spectra, axis)
    assert gen.score_knn_indices_ is not None
    assert gen.score_knn_indices_.shape == (160, 5)
    sampled = gen.sample(50, np.random.default_rng(12))
    assert sampled.shape == (50, 64)
    assert np.all(np.isfinite(sampled))


def test_copula_score_sampling_mode_runs_and_produces_correct_shape() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=160, n_features=64, seed=10)
    config = exp32.HybridConfig(
        baseline_degree=2,
        max_peaks=4,
        n_pca_components=8,
        score_sampling_mode="copula",
        seed=10,
    )
    gen = exp32.HybridGenerator(config).fit(spectra, axis)
    assert gen.score_copula_chol_ is not None
    assert gen.score_copula_sorted_ is not None
    sampled = gen.sample(50, np.random.default_rng(10))
    assert sampled.shape == (50, 64)
    assert np.all(np.isfinite(sampled))


def test_multiplicative_scattering_and_baseline_shift_apply_during_sampling() -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=120, n_features=48, seed=11)
    config_no_aug = exp32.HybridConfig(
        baseline_degree=2,
        max_peaks=4,
        n_pca_components=4,
        seed=11,
    )
    config_with_aug = exp32.HybridConfig(
        baseline_degree=2,
        max_peaks=4,
        n_pca_components=4,
        multiplicative_scattering_degree=2,
        additive_baseline_shift_std=0.05,
        seed=11,
    )
    gen_no = exp32.HybridGenerator(config_no_aug).fit(spectra, axis)
    gen_yes = exp32.HybridGenerator(config_with_aug).fit(spectra, axis)
    rng = np.random.default_rng(11)
    sampled_no = gen_no.sample(20, rng)
    sampled_yes = gen_yes.sample(20, np.random.default_rng(11))
    assert sampled_no.shape == sampled_yes.shape == (20, 48)
    assert not np.allclose(sampled_no, sampled_yes), "augmentations should produce different samples"


def test_write_csv_round_trip(tmp_path: Path) -> None:
    exp32 = _load_exp32()
    spectra, axis = _make_synthetic_real_distribution(n_samples=80, n_features=24, seed=4)
    directory = tmp_path / "fake/csv_dataset"
    directory.mkdir(parents=True)
    header = ";".join(f"{value:g}" for value in axis)
    lines = [header] + [";".join(f"{v:g}" for v in row) for row in spectra]
    (directory / "Xtrain.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = exp32.evaluate_dataset(directory, pca_range=(0, 4), n_splits=2, n_estimators=30, seed=4)
    csv_path = tmp_path / "out.csv"
    exp32.write_csv(list(result["rows"]), csv_path)
    contents = csv_path.read_text(encoding="utf-8")

    assert "n_pca_components_requested" in contents
    assert "auc_rf_mean" in contents
    assert "auc_lr_mean" in contents
    assert "audit_scope" in contents
