"""Phase B2 real/synthetic spectral scorecards."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from nirs4all.synthesis.validation import (
    compute_baseline_curvature,
    compute_correlation_length,
    compute_derivative_statistics,
    compute_distribution_overlap,
    compute_peak_density,
    compute_snr,
)

ScorecardStatus = Literal["compared", "synthetic_only", "blocked", "skipped"]
ComparisonSpace = Literal["raw", "snv"]

PROVISIONAL_THRESHOLDS: dict[str, float] = {
    "adversarial_auc_smoke": 0.85,
    "adversarial_auc_stretch": 0.70,
    "derivative_order_of_magnitude_gap": 1.0,
    "pca_overlap_min": 0.01,
    "nearest_neighbor_ratio_max": 2.0,
}


@dataclass(frozen=True)
class CohortInventory:
    """Local benchmark file availability summary."""

    source: str
    path: str
    exists: bool
    total_rows: int
    ok_rows: int
    runnable_rows: int
    missing_rows: int
    missing_paths: tuple[str, ...]


@dataclass(frozen=True)
class RealDataset:
    """A locally loadable real spectral benchmark split."""

    source: str
    task: str
    database_name: str
    dataset: str
    train_path: str
    test_path: str
    ytrain_path: str
    ytest_path: str
    n_train_declared: int | None
    n_test_declared: int | None
    p_declared: int | None

    @property
    def key(self) -> str:
        return f"{self.source}:{self.database_name}/{self.dataset}"


@dataclass(frozen=True)
class SpectralSummary:
    """Compact spectral distribution summary for CSV/report output."""

    n_samples: int
    n_wavelengths: int
    wavelength_min: float
    wavelength_max: float
    spectral_mean_mean: float
    spectral_variance_mean: float
    derivative_mean_median: float
    derivative_variance_median: float
    correlation_length_median: float
    snr_median: float
    baseline_curvature_median: float
    peak_density_median: float


@dataclass(frozen=True)
class RealMarginalCalibration:
    """Non-oracle marginal calibration fitted from real spectra only."""

    wavelengths: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    quantile_probabilities: np.ndarray
    quantile_values: np.ndarray
    highpass_scale: float
    smooth_window: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScorecardRow:
    """One standardized B2 scorecard row."""

    status: ScorecardStatus
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    comparison_space: ComparisonSpace
    synthetic_mapping_strategy: str
    synthetic_mapping_reason: str
    n_real_samples: int
    n_synthetic_samples: int
    n_wavelengths: int
    wavelength_min: float | None
    wavelength_max: float | None
    real_spectral_mean_mean: float | None
    synthetic_spectral_mean_mean: float | None
    real_spectral_variance_mean: float | None
    synthetic_spectral_variance_mean: float | None
    real_derivative_mean_median: float | None
    synthetic_derivative_mean_median: float | None
    real_derivative_variance_median: float | None
    synthetic_derivative_variance_median: float | None
    real_correlation_length_median: float | None
    synthetic_correlation_length_median: float | None
    real_snr_median: float | None
    synthetic_snr_median: float | None
    real_baseline_curvature_median: float | None
    synthetic_baseline_curvature_median: float | None
    real_peak_density_median: float | None
    synthetic_peak_density_median: float | None
    derivative_log10_gap: float | None
    snr_iqr_overlap: float | None
    pca_overlap: float | None
    adversarial_auc: float | None
    adversarial_auc_std: float | None
    nearest_neighbor_ratio: float | None
    provisional_decision: str
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


def discover_local_real_datasets(root: Path) -> tuple[list[RealDataset], list[CohortInventory]]:
    """Discover runnable AOM/TabPFN cohort rows without assuming a fixed count."""
    cohort_specs = [
        (
            "AOM_regression",
            "regression",
            root / "bench/AOM_v0/benchmarks/cohort_regression.csv",
        ),
        (
            "AOM_classification",
            "classification",
            root / "bench/AOM_v0/benchmarks/cohort_classification.csv",
        ),
    ]
    datasets: list[RealDataset] = []
    inventories: list[CohortInventory] = []
    for source, task, path in cohort_specs:
        found, inventory = _read_cohort(root, path, source=source, task=task)
        datasets.extend(found)
        inventories.append(inventory)
    return datasets, inventories


def load_real_spectra(dataset: RealDataset, *, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load train+test spectra and infer wavelengths from numeric CSV headers."""
    train_path = root / dataset.train_path
    test_path = root / dataset.test_path
    X_train, wavelengths = _load_semicolon_matrix(train_path)
    X_test, test_wavelengths = _load_semicolon_matrix(test_path)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"train/test feature mismatch for {dataset.key}: "
            f"{X_train.shape[1]} != {X_test.shape[1]}"
        )
    if not np.allclose(wavelengths, test_wavelengths):
        wavelengths = np.arange(X_train.shape[1], dtype=float)
    return np.vstack([X_train, X_test]), wavelengths


def summarize_spectra(X: np.ndarray, wavelengths: np.ndarray) -> SpectralSummary:
    """Compute required B2 single-distribution spectral metrics."""
    X = _as_2d_float(X)
    wavelengths = _wavelengths_or_index(wavelengths, X.shape[1])
    derivative_means, derivative_stds = compute_derivative_statistics(X, wavelengths, order=1)
    correlation_lengths = compute_correlation_length(X)
    snr = compute_snr(X)
    curvature = compute_baseline_curvature(X)
    peak_density = compute_peak_density(X, wavelengths)
    return SpectralSummary(
        n_samples=int(X.shape[0]),
        n_wavelengths=int(X.shape[1]),
        wavelength_min=float(wavelengths[0]),
        wavelength_max=float(wavelengths[-1]),
        spectral_mean_mean=float(np.mean(np.mean(X, axis=0))),
        spectral_variance_mean=float(np.mean(np.var(X, axis=0))),
        derivative_mean_median=float(np.median(derivative_means)),
        derivative_variance_median=float(np.median(derivative_stds**2)),
        correlation_length_median=float(np.median(correlation_lengths)),
        snr_median=float(np.median(snr)),
        baseline_curvature_median=float(np.median(curvature)),
        peak_density_median=float(np.median(peak_density)),
    )


def fit_real_marginal_calibration(
    real_X: np.ndarray,
    real_wavelengths: np.ndarray,
) -> RealMarginalCalibration:
    """Fit finite robust raw-spectrum nuisance statistics from real X only.

    The fit deliberately has no target/label argument. It stores robust
    per-wavelength location/scale and a low-frequency residual scale so the
    synthetic raw spectra can be calibrated before normal B2 scoring.
    """
    real = _as_2d_float(real_X).copy()
    wavelengths = _wavelengths_or_index(real_wavelengths, real.shape[1]).copy()
    location, scale = _robust_column_location_scale(real)
    quantile_probabilities = np.linspace(0.05, 0.95, 19, dtype=float)
    quantile_values = np.quantile(real, quantile_probabilities, axis=0)
    smooth_window = _smooth_window(real.shape[1])
    residual = real - _moving_average_rows(real, smooth_window)
    highpass_scale = _robust_1d_scale(residual.ravel())
    metadata = {
        "method": "robust_per_wavelength_quantile_affine_plus_highpass_scale",
        "source": "real_X_and_real_wavelengths_only",
        "status": "provisional",
        "strength": "strong",
        "warning": (
            "Strong provisional marginal calibration for B2 diagnostics only; "
            "not a calibrated domain gate or transfer-benefit claim."
        ),
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "n_real_samples": int(real.shape[0]),
        "n_wavelengths": int(real.shape[1]),
        "wavelength_min": float(wavelengths[0]),
        "wavelength_max": float(wavelengths[-1]),
        "smooth_window": int(smooth_window),
        "location_statistic": "per_wavelength_median",
        "scale_statistic": "per_wavelength_iqr_over_1.349_with_mad_std_fallback",
        "quantile_mapping": {
            "enabled": True,
            "probability_min": float(quantile_probabilities[0]),
            "probability_max": float(quantile_probabilities[-1]),
            "n_probabilities": int(quantile_probabilities.size),
            "replays_real_rows": False,
        },
        "location_median": float(np.median(location)),
        "scale_median": float(np.median(scale)),
        "highpass_scale": float(highpass_scale),
    }
    return RealMarginalCalibration(
        wavelengths=wavelengths,
        location=location,
        scale=scale,
        quantile_probabilities=quantile_probabilities,
        quantile_values=np.asarray(quantile_values, dtype=float),
        highpass_scale=float(highpass_scale),
        smooth_window=int(smooth_window),
        metadata=metadata,
    )


def apply_real_marginal_calibration(
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
    calibration: RealMarginalCalibration,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a fitted non-oracle real marginal calibration to synthetic X."""
    synthetic = _as_2d_float(synthetic_X).copy()
    wavelengths = _wavelengths_or_index(synthetic_wavelengths, synthetic.shape[1]).copy()
    target_location, target_scale, target_quantiles, grid_strategy = _calibration_target_on_grid(
        calibration,
        wavelengths,
    )
    synthetic_location, synthetic_scale = _robust_column_location_scale(synthetic)
    calibrated = (
        (synthetic - synthetic_location[np.newaxis, :])
        / synthetic_scale[np.newaxis, :]
        * target_scale[np.newaxis, :]
        + target_location[np.newaxis, :]
    )
    calibrated = _quantile_map_columns(
        calibrated,
        target_probabilities=calibration.quantile_probabilities,
        target_quantiles=target_quantiles,
    )

    smooth_window = min(calibration.smooth_window, _smooth_window(calibrated.shape[1]))
    if smooth_window >= 3:
        smooth = _moving_average_rows(calibrated, smooth_window)
        residual = calibrated - smooth
        synthetic_highpass_scale = _robust_1d_scale(residual.ravel())
        if synthetic_highpass_scale > 1e-12 and calibration.highpass_scale > 1e-12:
            ratio = float(np.clip(calibration.highpass_scale / synthetic_highpass_scale, 0.25, 4.0))
            calibrated = smooth + residual * ratio
        else:
            ratio = 1.0
    else:
        synthetic_highpass_scale = 0.0
        ratio = 1.0

    metadata = {
        "method": calibration.metadata["method"],
        "source": calibration.metadata["source"],
        "status": calibration.metadata["status"],
        "strength": calibration.metadata["strength"],
        "warning": calibration.metadata["warning"],
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "replays_real_rows": False,
        "deterministic": True,
        "grid_strategy": grid_strategy,
        "n_synthetic_samples": int(synthetic.shape[0]),
        "n_wavelengths": int(synthetic.shape[1]),
        "synthetic_wavelength_min": float(wavelengths[0]),
        "synthetic_wavelength_max": float(wavelengths[-1]),
        "synthetic_scale_median_before": float(np.median(synthetic_scale)),
        "target_scale_median": float(np.median(target_scale)),
        "synthetic_highpass_scale_before": float(synthetic_highpass_scale),
        "target_highpass_scale": float(calibration.highpass_scale),
        "highpass_scale_ratio": float(ratio),
        "fit": calibration.metadata,
    }
    return np.asarray(calibrated, dtype=float), metadata


def compare_real_synthetic(
    *,
    real_X: np.ndarray,
    real_wavelengths: np.ndarray,
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
    dataset: str,
    source: str,
    task: str,
    synthetic_preset: str,
    comparison_space: ComparisonSpace = "raw",
    synthetic_mapping_strategy: str = "not_recorded",
    synthetic_mapping_reason: str = "mapping metadata not provided",
    random_state: int,
) -> ScorecardRow:
    """Compute a standardized scorecard row for aligned real/synthetic spectra."""
    real_aligned, synthetic_aligned, wavelengths = align_to_real_grid(
        real_X,
        real_wavelengths,
        synthetic_X,
        synthetic_wavelengths,
    )
    if comparison_space == "snv":
        real_aligned = _samplewise_snv(real_aligned)
        synthetic_aligned = _samplewise_snv(synthetic_aligned)
    real_summary = summarize_spectra(real_aligned, wavelengths)
    synthetic_summary = summarize_spectra(synthetic_aligned, wavelengths)
    real_derivative = _distribution_bundle(real_aligned, wavelengths)
    synthetic_derivative = _distribution_bundle(synthetic_aligned, wavelengths)
    pca_overlap = compute_pca_overlap(real_aligned, synthetic_aligned, random_state=random_state)
    adversarial_auc, adversarial_auc_std = compute_adversarial_auc(
        real_aligned,
        synthetic_aligned,
        random_state=random_state,
    )
    nearest_ratio = compute_nearest_neighbor_ratio(real_aligned, synthetic_aligned)
    derivative_gap = _log10_gap(
        real_summary.derivative_variance_median,
        synthetic_summary.derivative_variance_median,
    )
    snr_overlap = compute_distribution_overlap(real_derivative["snr"], synthetic_derivative["snr"])
    decision = _provisional_decision(
        derivative_gap=derivative_gap,
        pca_overlap=pca_overlap,
        adversarial_auc=adversarial_auc,
        nearest_neighbor_ratio=nearest_ratio,
    )
    return ScorecardRow(
        status="compared",
        source=source,
        task=task,
        dataset=dataset,
        synthetic_preset=synthetic_preset,
        comparison_space=comparison_space,
        synthetic_mapping_strategy=synthetic_mapping_strategy,
        synthetic_mapping_reason=synthetic_mapping_reason,
        n_real_samples=real_summary.n_samples,
        n_synthetic_samples=synthetic_summary.n_samples,
        n_wavelengths=real_summary.n_wavelengths,
        wavelength_min=real_summary.wavelength_min,
        wavelength_max=real_summary.wavelength_max,
        real_spectral_mean_mean=real_summary.spectral_mean_mean,
        synthetic_spectral_mean_mean=synthetic_summary.spectral_mean_mean,
        real_spectral_variance_mean=real_summary.spectral_variance_mean,
        synthetic_spectral_variance_mean=synthetic_summary.spectral_variance_mean,
        real_derivative_mean_median=real_summary.derivative_mean_median,
        synthetic_derivative_mean_median=synthetic_summary.derivative_mean_median,
        real_derivative_variance_median=real_summary.derivative_variance_median,
        synthetic_derivative_variance_median=synthetic_summary.derivative_variance_median,
        real_correlation_length_median=real_summary.correlation_length_median,
        synthetic_correlation_length_median=synthetic_summary.correlation_length_median,
        real_snr_median=real_summary.snr_median,
        synthetic_snr_median=synthetic_summary.snr_median,
        real_baseline_curvature_median=real_summary.baseline_curvature_median,
        synthetic_baseline_curvature_median=synthetic_summary.baseline_curvature_median,
        real_peak_density_median=real_summary.peak_density_median,
        synthetic_peak_density_median=synthetic_summary.peak_density_median,
        derivative_log10_gap=derivative_gap,
        snr_iqr_overlap=snr_overlap,
        pca_overlap=pca_overlap,
        adversarial_auc=adversarial_auc,
        adversarial_auc_std=adversarial_auc_std,
        nearest_neighbor_ratio=nearest_ratio,
        provisional_decision=decision,
        blocked_reason="",
    )


def synthetic_only_row(
    *,
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
    synthetic_preset: str,
    blocked_reason: str,
    comparison_space: ComparisonSpace = "raw",
    synthetic_mapping_strategy: str = "synthetic_only",
    synthetic_mapping_reason: str = "no real dataset was selected",
) -> ScorecardRow:
    """Create an explicitly blocked synthetic-only dry-run scorecard row."""
    summary = summarize_spectra(synthetic_X, synthetic_wavelengths)
    return ScorecardRow(
        status="synthetic_only",
        source="A2_synthetic",
        task="dry_run",
        dataset=f"synthetic_only_{synthetic_preset}",
        synthetic_preset=synthetic_preset,
        comparison_space=comparison_space,
        synthetic_mapping_strategy=synthetic_mapping_strategy,
        synthetic_mapping_reason=synthetic_mapping_reason,
        n_real_samples=0,
        n_synthetic_samples=summary.n_samples,
        n_wavelengths=summary.n_wavelengths,
        wavelength_min=summary.wavelength_min,
        wavelength_max=summary.wavelength_max,
        real_spectral_mean_mean=None,
        synthetic_spectral_mean_mean=summary.spectral_mean_mean,
        real_spectral_variance_mean=None,
        synthetic_spectral_variance_mean=summary.spectral_variance_mean,
        real_derivative_mean_median=None,
        synthetic_derivative_mean_median=summary.derivative_mean_median,
        real_derivative_variance_median=None,
        synthetic_derivative_variance_median=summary.derivative_variance_median,
        real_correlation_length_median=None,
        synthetic_correlation_length_median=summary.correlation_length_median,
        real_snr_median=None,
        synthetic_snr_median=summary.snr_median,
        real_baseline_curvature_median=None,
        synthetic_baseline_curvature_median=summary.baseline_curvature_median,
        real_peak_density_median=None,
        synthetic_peak_density_median=summary.peak_density_median,
        derivative_log10_gap=None,
        snr_iqr_overlap=None,
        pca_overlap=None,
        adversarial_auc=None,
        adversarial_auc_std=None,
        nearest_neighbor_ratio=None,
        provisional_decision="blocked_no_real_data",
        blocked_reason=blocked_reason,
    )


def blocked_scorecard_row(
    *,
    source: str,
    task: str,
    dataset: str,
    synthetic_preset: str,
    blocked_reason: str,
    comparison_space: ComparisonSpace = "raw",
    synthetic_mapping_strategy: str = "not_recorded",
    synthetic_mapping_reason: str = "mapping metadata not provided",
) -> ScorecardRow:
    """Create an explicit row for a selected real dataset that could not be scored."""
    return ScorecardRow(
        status="blocked",
        source=source,
        task=task,
        dataset=dataset,
        synthetic_preset=synthetic_preset,
        comparison_space=comparison_space,
        synthetic_mapping_strategy=synthetic_mapping_strategy,
        synthetic_mapping_reason=synthetic_mapping_reason,
        n_real_samples=0,
        n_synthetic_samples=0,
        n_wavelengths=0,
        wavelength_min=None,
        wavelength_max=None,
        real_spectral_mean_mean=None,
        synthetic_spectral_mean_mean=None,
        real_spectral_variance_mean=None,
        synthetic_spectral_variance_mean=None,
        real_derivative_mean_median=None,
        synthetic_derivative_mean_median=None,
        real_derivative_variance_median=None,
        synthetic_derivative_variance_median=None,
        real_correlation_length_median=None,
        synthetic_correlation_length_median=None,
        real_snr_median=None,
        synthetic_snr_median=None,
        real_baseline_curvature_median=None,
        synthetic_baseline_curvature_median=None,
        real_peak_density_median=None,
        synthetic_peak_density_median=None,
        derivative_log10_gap=None,
        snr_iqr_overlap=None,
        pca_overlap=None,
        adversarial_auc=None,
        adversarial_auc_std=None,
        nearest_neighbor_ratio=None,
        provisional_decision="blocked_score_failure",
        blocked_reason=blocked_reason,
    )


def align_to_real_grid(
    real_X: np.ndarray,
    real_wavelengths: np.ndarray,
    synthetic_X: np.ndarray,
    synthetic_wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate synthetic spectra onto the real wavelength/index grid."""
    real_X = _as_2d_float(real_X)
    synthetic_X = _as_2d_float(synthetic_X)
    real_wavelengths = _wavelengths_or_index(real_wavelengths, real_X.shape[1])
    synthetic_wavelengths = _wavelengths_or_index(synthetic_wavelengths, synthetic_X.shape[1])
    if real_X.shape[1] == synthetic_X.shape[1] and np.allclose(real_wavelengths, synthetic_wavelengths):
        return real_X, synthetic_X, real_wavelengths
    overlap = (
        max(float(real_wavelengths[0]), float(synthetic_wavelengths[0])),
        min(float(real_wavelengths[-1]), float(synthetic_wavelengths[-1])),
    )
    mask = (real_wavelengths >= overlap[0]) & (real_wavelengths <= overlap[1])
    if int(mask.sum()) < 3:
        raise ValueError(
            "real/synthetic wavelength grids have fewer than three overlapping points"
        )
    target_wavelengths = real_wavelengths[mask]
    synthetic_interp = np.vstack([
        np.interp(target_wavelengths, synthetic_wavelengths, spectrum)
        for spectrum in synthetic_X
    ])
    return real_X[:, mask], synthetic_interp, target_wavelengths


def downsample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    """Deterministically cap rows for bounded scorecard runtime."""
    X = _as_2d_float(X)
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return X
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(X.shape[0], size=max_rows, replace=False))
    return X[indices]


def compute_pca_overlap(
    real_X: np.ndarray,
    synthetic_X: np.ndarray,
    *,
    random_state: int,
) -> float | None:
    """Compute a fast two-dimensional PCA histogram overlap if sklearn exists."""
    if min(real_X.shape[0], synthetic_X.shape[0]) < 3 or real_X.shape[1] < 2:
        return None
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    X = np.vstack([real_X, synthetic_X])
    X = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=random_state).fit_transform(X)
    real_coords = coords[: real_X.shape[0]]
    synthetic_coords = coords[real_X.shape[0] :]
    return _histogram_overlap_2d(real_coords, synthetic_coords)


def compute_adversarial_auc(
    real_X: np.ndarray,
    synthetic_X: np.ndarray,
    *,
    random_state: int,
) -> tuple[float | None, float | None]:
    """Compute real-vs-synthetic adversarial AUC when sample counts permit."""
    min_class = min(real_X.shape[0], synthetic_X.shape[0])
    if min_class < 8:
        return None, None
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None, None
    X = np.vstack([real_X, synthetic_X])
    y = np.concatenate([np.ones(real_X.shape[0]), np.zeros(synthetic_X.shape[0])])
    folds = min(5, min_class)
    components = max(2, min(12, X.shape[0] - 2, X.shape[1]))
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=random_state),
        LogisticRegression(max_iter=1000, C=0.2, random_state=random_state),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    except Exception:
        return None, None
    return float(np.mean(scores)), float(np.std(scores))


def compute_nearest_neighbor_ratio(real_X: np.ndarray, synthetic_X: np.ndarray) -> float | None:
    """Mean real-to-synthetic NN distance divided by real-to-real NN distance."""
    if real_X.shape[0] < 2 or synthetic_X.shape[0] < 1:
        return None
    try:
        from sklearn.metrics import pairwise_distances
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    X = StandardScaler().fit_transform(np.vstack([real_X, synthetic_X]))
    real_scaled = X[: real_X.shape[0]]
    synthetic_scaled = X[real_X.shape[0] :]
    real_to_synthetic = pairwise_distances(real_scaled, synthetic_scaled)
    real_to_real = pairwise_distances(real_scaled, real_scaled)
    np.fill_diagonal(real_to_real, np.inf)
    within = float(np.mean(np.min(real_to_real, axis=1)))
    across = float(np.mean(np.min(real_to_synthetic, axis=1)))
    if within <= 1e-12:
        return None
    return across / within


def write_scorecard_csv(rows: list[ScorecardRow], path: Path) -> None:
    """Write scorecard rows as a flat CSV metrics summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = [row.to_dict() for row in rows]
    if not dict_rows:
        dict_rows = [synthetic_only_row(
            synthetic_X=np.zeros((2, 3), dtype=float),
            synthetic_wavelengths=np.arange(3, dtype=float),
            synthetic_preset="none",
            blocked_reason="no_rows",
        ).to_dict()]
    fieldnames = list(dict_rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def _read_cohort(
    root: Path,
    path: Path,
    *,
    source: str,
    task: str,
) -> tuple[list[RealDataset], CohortInventory]:
    if not path.exists():
        return [], CohortInventory(
            source=source,
            path=str(path),
            exists=False,
            total_rows=0,
            ok_rows=0,
            runnable_rows=0,
            missing_rows=0,
            missing_paths=(str(path),),
        )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    found: list[RealDataset] = []
    missing: list[str] = []
    ok_rows = 0
    missing_rows = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        ok_rows += 1
        required = [
            row.get("train_path", ""),
            row.get("test_path", ""),
            row.get("ytrain_path", ""),
            row.get("ytest_path", ""),
        ]
        missing_for_row = [item for item in required if not item or not (root / item).exists()]
        if missing_for_row:
            missing_rows += 1
            missing.extend(missing_for_row)
            continue
        found.append(RealDataset(
            source=source,
            task=task,
            database_name=str(row.get("database_name", "")),
            dataset=str(row.get("dataset", "")),
            train_path=str(row["train_path"]),
            test_path=str(row["test_path"]),
            ytrain_path=str(row["ytrain_path"]),
            ytest_path=str(row["ytest_path"]),
            n_train_declared=_optional_int(row.get("n_train")),
            n_test_declared=_optional_int(row.get("n_test")),
            p_declared=_optional_int(row.get("p")),
        ))
    return found, CohortInventory(
        source=source,
        path=str(path),
        exists=True,
        total_rows=len(rows),
        ok_rows=ok_rows,
        runnable_rows=len(found),
        missing_rows=missing_rows,
        missing_paths=tuple(sorted(set(missing))),
    )


def _load_semicolon_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as file:
        header = file.readline().strip().split(";")
    X = np.genfromtxt(path, delimiter=";", skip_header=1, dtype=float)
    X = np.atleast_2d(X)
    wavelengths = _header_wavelengths(header, X.shape[1])
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite spectra in {path}")
    return X, wavelengths


def _header_wavelengths(header: list[str], n_features: int) -> np.ndarray:
    if len(header) != n_features:
        return np.arange(n_features, dtype=float)
    try:
        wavelengths = np.asarray([float(item.strip().strip('"')) for item in header], dtype=float)
    except ValueError:
        return np.arange(n_features, dtype=float)
    if wavelengths.size < 2 or not np.all(np.diff(wavelengths) > 0):
        return np.arange(n_features, dtype=float)
    return wavelengths


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected 2D spectra, got shape {X.shape}")
    if X.shape[0] < 1 or X.shape[1] < 3:
        raise ValueError(f"spectra need at least one row and three wavelengths, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("spectra contain non-finite values")
    return X


def _samplewise_snv(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = _as_2d_float(X)
    means = np.mean(X, axis=1, keepdims=True)
    stds = np.std(X, axis=1, keepdims=True)
    safe_stds = np.where(stds <= eps, 1.0, stds)
    normalized = np.asarray((X.copy() - means) / safe_stds, dtype=float)
    return normalized


def _robust_column_location_scale(X: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    mad = np.median(np.abs(X - location[np.newaxis, :]), axis=0) * 1.4826
    std = np.std(X, axis=0)
    scale = np.where(scale > eps, scale, mad)
    scale = np.where(scale > eps, scale, std)
    global_scale = _robust_1d_scale(X.ravel(), eps=eps)
    scale = np.where(scale > eps, scale, global_scale)
    scale = np.maximum(scale, eps)
    return np.asarray(location, dtype=float), np.asarray(scale, dtype=float)


def _robust_1d_scale(values: np.ndarray, *, eps: float = 1e-12) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    q75, q25 = np.percentile(finite, [75.0, 25.0])
    scale = float((q75 - q25) / 1.349)
    if scale > eps:
        return scale
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)) * 1.4826)
    if mad > eps:
        return mad
    std = float(np.std(finite))
    if std > eps:
        return std
    return 1.0


def _smooth_window(n_features: int) -> int:
    if n_features < 5:
        return 0
    window = max(5, int(round(n_features * 0.03)))
    if window % 2 == 0:
        window += 1
    if window >= n_features:
        window = n_features - 1 if n_features % 2 == 0 else n_features
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def _moving_average_rows(X: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return X.copy()
    half = window // 2
    kernel = np.full(window, 1.0 / window, dtype=float)
    padded = np.pad(X, ((0, 0), (half, half)), mode="edge")
    return np.vstack([
        np.convolve(row, kernel, mode="valid")
        for row in padded
    ])


def _calibration_target_on_grid(
    calibration: RealMarginalCalibration,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if (
        wavelengths.shape == calibration.wavelengths.shape
        and np.allclose(wavelengths, calibration.wavelengths)
    ):
        return (
            calibration.location.copy(),
            calibration.scale.copy(),
            calibration.quantile_values.copy(),
            "same_grid",
        )
    location = np.interp(wavelengths, calibration.wavelengths, calibration.location)
    scale = np.interp(wavelengths, calibration.wavelengths, calibration.scale)
    scale = np.maximum(scale, 1e-12)
    quantiles = np.vstack([
        np.interp(wavelengths, calibration.wavelengths, quantile_row)
        for quantile_row in calibration.quantile_values
    ])
    return location, scale, quantiles, "interpolated_real_calibration_to_synthetic_grid"


def _quantile_map_columns(
    X: np.ndarray,
    *,
    target_probabilities: np.ndarray,
    target_quantiles: np.ndarray,
) -> np.ndarray:
    mapped = np.empty_like(X, dtype=float)
    n_rows, n_cols = X.shape
    source_probabilities = (np.arange(n_rows, dtype=float) + 0.5) / n_rows
    for col_idx in range(n_cols):
        values = X[:, col_idx]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        empirical = np.interp(
            values,
            sorted_values,
            source_probabilities,
            left=source_probabilities[0],
            right=source_probabilities[-1],
        )
        mapped[:, col_idx] = np.interp(
            empirical,
            target_probabilities,
            target_quantiles[:, col_idx],
            left=target_quantiles[0, col_idx],
            right=target_quantiles[-1, col_idx],
        )
    return mapped


def _wavelengths_or_index(wavelengths: np.ndarray, n_features: int) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size != n_features:
        wavelengths = np.arange(n_features, dtype=float)
    if wavelengths.size < 2 or not np.all(np.diff(wavelengths) > 0):
        wavelengths = np.arange(n_features, dtype=float)
    return wavelengths


def _distribution_bundle(X: np.ndarray, wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    _, derivative_stds = compute_derivative_statistics(X, wavelengths, order=1)
    return {
        "derivative_variance": derivative_stds**2,
        "snr": compute_snr(X),
        "curvature": compute_baseline_curvature(X),
        "peak_density": compute_peak_density(X, wavelengths),
    }


def _histogram_overlap_2d(real_coords: np.ndarray, synthetic_coords: np.ndarray) -> float:
    all_coords = np.vstack([real_coords, synthetic_coords])
    mins = np.min(all_coords, axis=0)
    maxs = np.max(all_coords, axis=0)
    if np.any(maxs - mins <= 1e-12):
        return 1.0
    bins = [
        np.linspace(float(mins[0]), float(maxs[0]), 16),
        np.linspace(float(mins[1]), float(maxs[1]), 16),
    ]
    real_hist, _ = np.histogramdd(real_coords, bins=bins)
    synthetic_hist, _ = np.histogramdd(synthetic_coords, bins=bins)
    real_sum = real_hist.sum()
    synthetic_sum = synthetic_hist.sum()
    if real_sum == 0 or synthetic_sum == 0:
        return 0.0
    real_hist = real_hist / real_sum
    synthetic_hist = synthetic_hist / synthetic_sum
    return float(np.minimum(real_hist, synthetic_hist).sum())


def _log10_gap(real_value: float, synthetic_value: float) -> float:
    return abs(float(np.log10(abs(real_value) + 1e-12) - np.log10(abs(synthetic_value) + 1e-12)))


def _provisional_decision(
    *,
    derivative_gap: float | None,
    pca_overlap: float | None,
    adversarial_auc: float | None,
    nearest_neighbor_ratio: float | None,
) -> str:
    failures: list[str] = []
    if derivative_gap is not None and derivative_gap > PROVISIONAL_THRESHOLDS["derivative_order_of_magnitude_gap"]:
        failures.append("derivative_gap")
    if pca_overlap is not None and pca_overlap < PROVISIONAL_THRESHOLDS["pca_overlap_min"]:
        failures.append("pca_overlap")
    if adversarial_auc is not None and adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]:
        failures.append("adversarial_auc")
    if (
        nearest_neighbor_ratio is not None
        and nearest_neighbor_ratio > PROVISIONAL_THRESHOLDS["nearest_neighbor_ratio_max"]
    ):
        failures.append("nearest_neighbor_ratio")
    return "provisional_pass" if not failures else "provisional_review:" + ",".join(failures)


def _optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
