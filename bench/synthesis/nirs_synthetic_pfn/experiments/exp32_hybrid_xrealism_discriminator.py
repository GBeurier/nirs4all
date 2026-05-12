"""exp32 hybrid x-realism + adversarial discriminator (single dataset).

Bench-only experiment for the X-realism strategy in
``docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md``.

Goal: produce synthetic spectra distributionally indistinguishable from
a real dataset's spectra, evaluated by an adversarial classifier on the
spectra alone. AUC near 0.5 is the win condition.

The generator is hybrid:

- mechanistic skeleton: low-degree polynomial baseline plus a small bank
  of parametric absorption peaks fit to the dataset mean spectrum;
- statistical residual: low-rank PCA of (real - mechanistic mean), with
  scores sampled from per-component Gaussians whose variances are
  fitted to the real PCA scores;
- per-channel residual noise: Gaussian with per-channel std fit to the
  PCA reconstruction tail.

The discriminator is RandomForest (canonical) plus LogisticRegression
(linear sanity baseline). Scores are reported per PCA rank in a sweep.

No Y file is read. The dataset's official train/test split is not used
as an oracle. ``nirs4all/`` is not imported.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EXP32_AUDIT_SCOPE = "bench_only_phase_r0_hybrid_xrealism_adversarial_discriminator"
DEFAULT_PCA_RANGE: tuple[int, ...] = (0, 2, 5, 10, 20, 40, 80)
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/xrealism_single_dataset.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/xrealism_single_dataset.csv")


# --------------------------------------------------------------------------
# Data loading: X spectra and axis only, no Y, no metadata, no nirs4all.
# --------------------------------------------------------------------------


def _detect_separator(line: str) -> str:
    candidates = [(";", line.count(";")), (",", line.count(",")), ("\t", line.count("\t"))]
    sep, count = max(candidates, key=lambda item: item[1])
    return sep if count > 0 else ";"


def _strip_token(token: str) -> str:
    cleaned = token.strip()
    if len(cleaned) >= 2 and cleaned[0] in {'"', "'"} and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _parse_axis_token(token: str) -> float | None:
    text = _strip_token(token)
    if text.casefold().endswith("_nm"):
        text = text[:-3]
    try:
        return float(text)
    except ValueError:
        return None


def _read_data_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis, X) for one Xtrain/Xtest file."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    sep = _detect_separator(header)
    axis_tokens = [_parse_axis_token(t) for t in header.rstrip("\n").split(sep) if t.strip()]
    if any(token is None for token in axis_tokens):
        raise ValueError(f"non-numeric axis token in header of {path}")
    axis = np.asarray(axis_tokens, dtype=np.float64)

    def _stripped_lines() -> Any:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.readline()
            for line in handle:
                cleaned = line.replace('"', "").replace("'", "")
                if cleaned.strip():
                    yield cleaned

    try:
        x = np.loadtxt(_stripped_lines(), delimiter=sep, dtype=np.float64)
    except (ValueError, OSError) as exc:
        raise ValueError(f"failed to parse data rows in {path}: {exc}") from exc
    if x.ndim == 1:
        x = x.reshape(1, -1) if axis.shape[0] == x.shape[0] else x.reshape(-1, 1)
    if x.shape[1] != axis.shape[0]:
        raise ValueError(f"feature count mismatch axis={axis.shape[0]} x={x.shape[1]} in {path}")
    return axis, x


def load_dataset_x_only(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load X spectra (union of Xtrain and Xtest) and axis from a tabpfn_paper-style directory."""
    xtrain_path = directory / "Xtrain.csv"
    xtest_path = directory / "Xtest.csv"
    if not xtrain_path.exists():
        raise FileNotFoundError(f"missing Xtrain.csv in {directory}")
    axis, train = _read_data_file(xtrain_path)
    if xtest_path.exists():
        axis_test, test = _read_data_file(xtest_path)
        if axis_test.shape != axis.shape or not np.allclose(axis_test, axis):
            raise ValueError(f"train/test axis mismatch in {directory}")
        x = np.vstack([train, test])
    else:
        x = train
    if x.shape[0] == 0:
        raise ValueError(f"no data rows in {directory}")
    return x, axis


# --------------------------------------------------------------------------
# Hybrid mechanistic + statistical generator.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridConfig:
    baseline_degree: int = 3
    max_peaks: int = 16
    peak_prominence_factor: float = 0.5
    n_pca_components: int = 20
    add_per_channel_noise: bool = True
    # Sampling modes for PCA scores: "gaussian", "empirical", "joint_bootstrap", "gmm", "copula"
    score_sampling_mode: str = "gaussian"
    # Sampling modes for per-channel noise tail: "gaussian", "empirical", "joint_bootstrap"
    noise_sampling_mode: str = "gaussian"
    empirical_jitter_fraction: float = 0.05
    score_gmm_components: int = 8
    score_gmm_covariance_type: str = "full"  # "full", "tied", "diag", "spherical"
    score_knn_mixup_k: int = 5
    score_knn_mixup_dirichlet_alpha: float = 1.0
    # Per-spectrum mechanistic augmentations applied during sampling.
    multiplicative_scattering_degree: int = 0  # 0 disables; otherwise polynomial degree (e.g. 2)
    additive_baseline_shift_std: float = 0.0  # 0 disables; std for additive baseline shift, in spectrum units
    seed: int = 20260501


class HybridGenerator:
    """Mechanistic baseline + parametric peaks + PCA residual + per-channel noise."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.axis_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.baseline_coefs_: np.ndarray | None = None
        self.peak_centers_: np.ndarray | None = None
        self.peak_amplitudes_: np.ndarray | None = None
        self.peak_widths_sigma_: np.ndarray | None = None
        self.mechanistic_mean_: np.ndarray | None = None
        self.pca_: PCA | None = None
        self.score_std_: np.ndarray | None = None
        self.score_samples_: np.ndarray | None = None
        self.score_gmm_: GaussianMixture | None = None
        self.score_copula_chol_: np.ndarray | None = None
        self.score_copula_sorted_: np.ndarray | None = None
        self.score_knn_indices_: np.ndarray | None = None
        self.noise_std_: np.ndarray | None = None
        self.noise_samples_: np.ndarray | None = None
        self.scattering_coefs_mean_: np.ndarray | None = None
        self.scattering_coefs_std_: np.ndarray | None = None
        self.baseline_shift_std_: float = 0.0
        self.n_pca_actual_: int = 0

    def fit(self, x: np.ndarray, axis: np.ndarray) -> HybridGenerator:
        x = np.asarray(x, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        n_samples, n_features = x.shape
        if axis.shape != (n_features,):
            raise ValueError(f"axis shape {axis.shape} does not match feature count {n_features}")
        if n_samples < 2:
            raise ValueError("need at least 2 spectra to fit hybrid generator")

        self.axis_ = axis
        self.mean_ = x.mean(axis=0)

        baseline_degree = max(0, min(self.config.baseline_degree, n_features - 1))
        self.baseline_coefs_ = np.polyfit(axis, self.mean_, baseline_degree)
        baseline = np.polyval(self.baseline_coefs_, axis)

        residual_for_peaks = self.mean_ - baseline
        prominence_threshold = float(np.std(residual_for_peaks)) * self.config.peak_prominence_factor
        prominence_threshold = max(prominence_threshold, 1e-12)
        abs_residual = np.abs(residual_for_peaks)
        peak_idx, props = find_peaks(abs_residual, prominence=prominence_threshold)
        if len(peak_idx) > self.config.max_peaks:
            order = np.argsort(props["prominences"])[::-1][: self.config.max_peaks]
            peak_idx = peak_idx[order]
        self.peak_centers_ = axis[peak_idx]
        self.peak_amplitudes_ = residual_for_peaks[peak_idx]
        if len(peak_idx) > 0:
            widths_result = peak_widths(abs_residual, peak_idx, rel_height=0.5)
            axis_step = float(np.median(np.abs(np.diff(axis)))) if n_features > 1 else 1.0
            fwhm_axis_units = widths_result[0] * axis_step
            sigma = fwhm_axis_units / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma = np.where(sigma > 0, sigma, axis_step)
            self.peak_widths_sigma_ = sigma
        else:
            self.peak_widths_sigma_ = np.array([], dtype=np.float64)

        peaks_signal = np.zeros_like(axis)
        for center, amplitude, width in zip(
            self.peak_centers_, self.peak_amplitudes_, self.peak_widths_sigma_, strict=True
        ):
            if width <= 0:
                continue
            peaks_signal = peaks_signal + amplitude * np.exp(-((axis - center) ** 2) / (2.0 * width**2))
        self.mechanistic_mean_ = baseline + peaks_signal

        residuals = x - self.mechanistic_mean_
        n_pca = min(max(0, self.config.n_pca_components), n_samples - 1, n_features)
        self.n_pca_actual_ = n_pca
        if n_pca > 0:
            self.pca_ = PCA(n_components=n_pca)
            scores = self.pca_.fit_transform(residuals)
            self.score_std_ = scores.std(axis=0)
            self.score_samples_ = scores.copy()
            reconstructed = self.pca_.inverse_transform(scores)
            tail = residuals - reconstructed
            self._fit_gmm(scores)
            self._fit_copula(scores)
            self._fit_knn(scores)
        else:
            self.pca_ = None
            self.score_std_ = np.array([], dtype=np.float64)
            self.score_samples_ = None
            self.score_gmm_ = None
            self.score_copula_chol_ = None
            self.score_copula_sorted_ = None
            self.score_knn_indices_ = None
            tail = residuals

        if self.config.add_per_channel_noise:
            self.noise_std_ = tail.std(axis=0)
            self.noise_samples_ = tail.copy()
        else:
            self.noise_std_ = np.zeros(n_features, dtype=np.float64)
            self.noise_samples_ = None

        self._fit_scattering_prior(x, axis)

        return self

    def _fit_gmm(self, scores: np.ndarray) -> None:
        if self.config.score_sampling_mode != "gmm":
            self.score_gmm_ = None
            return
        n_samples = scores.shape[0]
        n_components = max(1, min(self.config.score_gmm_components, n_samples))
        self.score_gmm_ = GaussianMixture(
            n_components=n_components,
            covariance_type=self.config.score_gmm_covariance_type,
            random_state=self.config.seed,
            reg_covar=1e-6,
            max_iter=200,
        ).fit(scores)

    def _fit_copula(self, scores: np.ndarray) -> None:
        if self.config.score_sampling_mode != "copula":
            self.score_copula_chol_ = None
            self.score_copula_sorted_ = None
            return
        n_samples, n_components = scores.shape
        sorted_scores = np.sort(scores, axis=0)
        # Empirical rank CDF in (0, 1) avoiding 0 and 1 to keep ppf finite.
        ranks = (np.argsort(np.argsort(scores, axis=0), axis=0) + 1) / (n_samples + 1)
        from scipy.stats import norm

        z = norm.ppf(ranks)
        corr = np.corrcoef(z, rowvar=False)
        if not np.all(np.isfinite(corr)):
            corr = np.eye(n_components)
        regularized = corr + 1e-6 * np.eye(n_components)
        try:
            chol = np.linalg.cholesky(regularized)
        except np.linalg.LinAlgError:
            chol = np.eye(n_components)
        self.score_copula_chol_ = chol
        self.score_copula_sorted_ = sorted_scores

    def _fit_knn(self, scores: np.ndarray) -> None:
        if self.config.score_sampling_mode != "knn_mixup":
            self.score_knn_indices_ = None
            return
        n_samples = scores.shape[0]
        k = max(2, min(self.config.score_knn_mixup_k, n_samples))
        nn = NearestNeighbors(n_neighbors=k).fit(scores)
        self.score_knn_indices_ = np.asarray(nn.kneighbors(scores, return_distance=False), dtype=np.int64)

    def _fit_scattering_prior(self, x: np.ndarray, axis: np.ndarray) -> None:
        degree = max(0, self.config.multiplicative_scattering_degree)
        if degree <= 0 and self.config.additive_baseline_shift_std <= 0.0:
            self.scattering_coefs_mean_ = None
            self.scattering_coefs_std_ = None
            self.baseline_shift_std_ = 0.0
            return

        if degree > 0 and self.mean_ is not None:
            mean_safe = np.where(np.abs(self.mean_) < 1e-8, 1e-8, self.mean_)
            ratios = x / mean_safe
            normalized = (axis - axis.mean()) / max(np.abs(axis.max() - axis.min()) / 2.0, 1e-8)
            design = np.stack([normalized**k for k in range(degree + 1)], axis=1)
            try:
                coefs, *_ = np.linalg.lstsq(design, ratios.T, rcond=None)
                coefs = coefs.T
                self.scattering_coefs_mean_ = coefs.mean(axis=0)
                self.scattering_coefs_std_ = coefs.std(axis=0)
            except np.linalg.LinAlgError:
                self.scattering_coefs_mean_ = None
                self.scattering_coefs_std_ = None
        else:
            self.scattering_coefs_mean_ = None
            self.scattering_coefs_std_ = None

        if self.config.additive_baseline_shift_std > 0.0:
            self.baseline_shift_std_ = float(self.config.additive_baseline_shift_std)
        else:
            self.baseline_shift_std_ = 0.0

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.mechanistic_mean_ is None or self.noise_std_ is None:
            raise RuntimeError("HybridGenerator must be fitted before sampling")
        if n_samples <= 0:
            return np.empty((0, self.mechanistic_mean_.shape[0]), dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng(self.config.seed)
        n_features = self.mechanistic_mean_.shape[0]
        out = np.tile(self.mechanistic_mean_, (n_samples, 1))
        if self.pca_ is not None and self.score_std_ is not None and self.score_std_.size:
            scores = self._sample_scores(n_samples, rng)
            out = out + self.pca_.mean_ + scores @ self.pca_.components_
        if np.any(self.noise_std_ > 0):
            noise = self._sample_noise(n_samples, n_features, rng)
            out = out + noise
        out = self._apply_scattering(out, rng)
        out = self._apply_baseline_shift(out, rng)
        return out

    def _apply_scattering(self, spectra: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if (
            self.scattering_coefs_mean_ is None
            or self.scattering_coefs_std_ is None
            or self.axis_ is None
        ):
            return spectra
        degree = self.scattering_coefs_mean_.shape[0] - 1
        if degree < 0:
            return spectra
        normalized = (self.axis_ - self.axis_.mean()) / max(
            np.abs(self.axis_.max() - self.axis_.min()) / 2.0,
            1e-8,
        )
        design = np.stack([normalized**k for k in range(degree + 1)], axis=1)
        coefs = self.scattering_coefs_mean_ + rng.standard_normal((spectra.shape[0], degree + 1)) * self.scattering_coefs_std_
        scattering = coefs @ design.T  # (n_samples, n_features)
        return np.asarray(spectra * scattering, dtype=np.float64)

    def _apply_baseline_shift(self, spectra: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.baseline_shift_std_ <= 0.0:
            return spectra
        shift = rng.standard_normal((spectra.shape[0], 1)) * self.baseline_shift_std_
        return spectra + shift

    def _sample_scores(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        assert self.pca_ is not None and self.score_std_ is not None
        n_components = self.pca_.n_components_
        mode = self.config.score_sampling_mode
        if mode == "empirical" and self.score_samples_ is not None and self.score_samples_.shape[0] > 0:
            n_real = self.score_samples_.shape[0]
            scores = np.empty((n_samples, n_components), dtype=np.float64)
            for component in range(n_components):
                indices = rng.integers(0, n_real, n_samples)
                scores[:, component] = self.score_samples_[indices, component]
            jitter = self.config.empirical_jitter_fraction
            if jitter > 0:
                scores = scores + rng.standard_normal(scores.shape) * (self.score_std_ * jitter)
            return scores
        if mode == "joint_bootstrap" and self.score_samples_ is not None and self.score_samples_.shape[0] > 0:
            n_real = self.score_samples_.shape[0]
            indices = rng.integers(0, n_real, n_samples)
            scores = self.score_samples_[indices].copy()
            jitter = self.config.empirical_jitter_fraction
            if jitter > 0:
                scores = scores + rng.standard_normal(scores.shape) * (self.score_std_ * jitter)
            return scores
        if mode == "gmm" and self.score_gmm_ is not None:
            sampled, _ = self.score_gmm_.sample(n_samples)
            sampled = np.asarray(sampled, dtype=np.float64)
            rng.shuffle(sampled, axis=0)
            return sampled
        if mode == "copula" and self.score_copula_chol_ is not None and self.score_copula_sorted_ is not None:
            return self._sample_copula_scores(n_samples, rng)
        if mode == "knn_mixup" and self.score_knn_indices_ is not None and self.score_samples_ is not None:
            n_real, k = self.score_knn_indices_.shape
            seed_indices = rng.integers(0, n_real, n_samples)
            neighbor_groups = self.score_knn_indices_[seed_indices]
            alpha_param = max(self.config.score_knn_mixup_dirichlet_alpha, 1e-4)
            weights = rng.dirichlet([alpha_param] * k, size=n_samples)
            neighbor_scores = self.score_samples_[neighbor_groups]
            mixed = np.einsum("ij,ijk->ik", weights, neighbor_scores)
            return np.asarray(mixed, dtype=np.float64)
        return np.asarray(rng.standard_normal((n_samples, n_components)) * self.score_std_, dtype=np.float64)

    def _sample_copula_scores(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        assert self.score_copula_chol_ is not None and self.score_copula_sorted_ is not None
        from scipy.stats import norm

        n_components = self.score_copula_sorted_.shape[1]
        n_real = self.score_copula_sorted_.shape[0]
        z = rng.standard_normal((n_samples, n_components)) @ self.score_copula_chol_.T
        u = norm.cdf(z)
        u = np.clip(u, 1.0 / (n_real + 1), n_real / (n_real + 1))
        positions = u * (n_real + 1) - 1
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, n_real - 1)
        lower = np.clip(lower, 0, n_real - 1)
        frac = positions - lower
        scores = np.empty((n_samples, n_components), dtype=np.float64)
        for component in range(n_components):
            sorted_col = self.score_copula_sorted_[:, component]
            scores[:, component] = sorted_col[lower[:, component]] * (1.0 - frac[:, component]) + sorted_col[upper[:, component]] * frac[:, component]
        return scores

    def _sample_noise(self, n_samples: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
        assert self.noise_std_ is not None
        mode = self.config.noise_sampling_mode
        if mode == "empirical" and self.noise_samples_ is not None and self.noise_samples_.shape[0] > 0:
            n_real = self.noise_samples_.shape[0]
            noise = np.empty((n_samples, n_features), dtype=np.float64)
            for channel in range(n_features):
                indices = rng.integers(0, n_real, n_samples)
                noise[:, channel] = self.noise_samples_[indices, channel]
            jitter = self.config.empirical_jitter_fraction
            if jitter > 0:
                noise = noise + rng.standard_normal(noise.shape) * (self.noise_std_ * jitter)
            return noise
        if mode == "joint_bootstrap" and self.noise_samples_ is not None and self.noise_samples_.shape[0] > 0:
            n_real = self.noise_samples_.shape[0]
            indices = rng.integers(0, n_real, n_samples)
            noise = self.noise_samples_[indices].copy()
            jitter = self.config.empirical_jitter_fraction
            if jitter > 0:
                noise = noise + rng.standard_normal(noise.shape) * (self.noise_std_ * jitter)
            return noise
        return np.asarray(rng.standard_normal((n_samples, n_features)) * self.noise_std_, dtype=np.float64)


# --------------------------------------------------------------------------
# Adversarial AUC harness.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialAUC:
    classifier: str
    n_splits: int
    test_size: float
    mean_auc: float
    std_auc: float
    min_auc: float
    max_auc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adversarial_auc(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    classifier: str = "rf",
    n_splits: int = 5,
    test_size: float = 0.3,
    seed: int = 20260501,
    n_estimators: int = 200,
) -> AdversarialAUC:
    real = np.asarray(real, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2:
        raise ValueError("real and synthetic must be 2D arrays")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError(f"feature mismatch real={real.shape[1]} synthetic={synthetic.shape[1]}")
    if real.shape[0] < 2 or synthetic.shape[0] < 2:
        raise ValueError("need at least 2 samples per pool")

    x = np.vstack([real, synthetic])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synthetic))])
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in splitter.split(x, y):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if classifier == "rf":
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=1,
            )
            clf.fit(x_tr, y_tr)
            prob = clf.predict_proba(x_te)[:, 1]
        elif classifier == "lr":
            scaler = StandardScaler().fit(x_tr)
            x_tr_s = scaler.transform(x_tr)
            x_te_s = scaler.transform(x_te)
            clf = LogisticRegression(max_iter=2000, random_state=seed)
            clf.fit(x_tr_s, y_tr)
            prob = clf.predict_proba(x_te_s)[:, 1]
        else:
            raise ValueError(f"unknown classifier {classifier}")
        aucs.append(float(roc_auc_score(y_te, prob)))
    arr = np.asarray(aucs, dtype=np.float64)
    return AdversarialAUC(
        classifier=classifier,
        n_splits=n_splits,
        test_size=test_size,
        mean_auc=float(arr.mean()),
        std_auc=float(arr.std()),
        min_auc=float(arr.min()),
        max_auc=float(arr.max()),
    )


# --------------------------------------------------------------------------
# Single-dataset evaluator.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRow:
    dataset: str
    n_real: int
    n_synthetic: int
    n_features: int
    axis_min: float
    axis_max: float
    n_pca_components_requested: int
    n_pca_components_actual: int
    baseline_degree: int
    n_peaks_fitted: int
    auc_rf_mean: float
    auc_rf_std: float
    auc_rf_min: float
    auc_rf_max: float
    auc_lr_mean: float
    auc_lr_std: float
    auc_lr_min: float
    auc_lr_max: float
    audit_scope: str = EXP32_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_dataset(
    directory: Path,
    *,
    pca_range: tuple[int, ...] = DEFAULT_PCA_RANGE,
    n_synthetic_factor: float = 1.0,
    n_splits: int = 5,
    test_size: float = 0.3,
    seed: int = 20260501,
    baseline_degree: int = 3,
    max_peaks: int = 16,
    n_estimators: int = 200,
    classifiers: tuple[str, ...] = ("rf", "lr"),
    score_sampling_mode: str = "gaussian",
    noise_sampling_mode: str = "gaussian",
    empirical_jitter_fraction: float = 0.05,
    score_gmm_components: int = 8,
    score_gmm_covariance_type: str = "full",
    score_knn_mixup_k: int = 5,
    score_knn_mixup_dirichlet_alpha: float = 1.0,
    multiplicative_scattering_degree: int = 0,
    additive_baseline_shift_std: float = 0.0,
    subsample_rows: int | None = None,
) -> dict[str, Any]:
    x, axis = load_dataset_x_only(directory)
    if subsample_rows is not None and subsample_rows > 0 and subsample_rows < x.shape[0]:
        rng_subsample = np.random.default_rng(seed)
        indices = rng_subsample.choice(x.shape[0], size=subsample_rows, replace=False)
        x = x[indices]
    n_real = int(x.shape[0])
    n_synthetic = max(2, int(round(n_real * n_synthetic_factor)))
    rng = np.random.default_rng(seed)

    rows: list[EvaluationRow] = []
    for n_pca in pca_range:
        config = HybridConfig(
            baseline_degree=baseline_degree,
            max_peaks=max_peaks,
            n_pca_components=n_pca,
            add_per_channel_noise=True,
            score_sampling_mode=score_sampling_mode,
            noise_sampling_mode=noise_sampling_mode,
            empirical_jitter_fraction=empirical_jitter_fraction,
            score_gmm_components=score_gmm_components,
            score_gmm_covariance_type=score_gmm_covariance_type,
            score_knn_mixup_k=score_knn_mixup_k,
            score_knn_mixup_dirichlet_alpha=score_knn_mixup_dirichlet_alpha,
            multiplicative_scattering_degree=multiplicative_scattering_degree,
            additive_baseline_shift_std=additive_baseline_shift_std,
            seed=seed,
        )
        gen = HybridGenerator(config).fit(x, axis)
        synthetic = gen.sample(n_synthetic, rng)

        rf_result = (
            adversarial_auc(x, synthetic, classifier="rf", n_splits=n_splits, test_size=test_size, seed=seed, n_estimators=n_estimators)
            if "rf" in classifiers
            else AdversarialAUC("rf_skipped", 0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"))
        )
        lr_result = (
            adversarial_auc(x, synthetic, classifier="lr", n_splits=n_splits, test_size=test_size, seed=seed)
            if "lr" in classifiers
            else AdversarialAUC("lr_skipped", 0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"))
        )

        rows.append(
            EvaluationRow(
                dataset=str(directory),
                n_real=n_real,
                n_synthetic=n_synthetic,
                n_features=int(x.shape[1]),
                axis_min=float(axis.min()),
                axis_max=float(axis.max()),
                n_pca_components_requested=int(n_pca),
                n_pca_components_actual=int(gen.n_pca_actual_),
                baseline_degree=int(baseline_degree),
                n_peaks_fitted=int(0 if gen.peak_centers_ is None else gen.peak_centers_.size),
                auc_rf_mean=float(rf_result.mean_auc),
                auc_rf_std=float(rf_result.std_auc),
                auc_rf_min=float(rf_result.min_auc),
                auc_rf_max=float(rf_result.max_auc),
                auc_lr_mean=float(lr_result.mean_auc),
                auc_lr_std=float(lr_result.std_auc),
                auc_lr_min=float(lr_result.min_auc),
                auc_lr_max=float(lr_result.max_auc),
            )
        )

    return {
        "status": "done",
        "dataset": str(directory),
        "n_real": n_real,
        "n_features": int(x.shape[1]),
        "axis_min": float(axis.min()),
        "axis_max": float(axis.max()),
        "rows": rows,
        "best_rf": min((row for row in rows if not np.isnan(row.auc_rf_mean)), key=lambda row: abs(row.auc_rf_mean - 0.5), default=None),
        "best_lr": min((row for row in rows if not np.isnan(row.auc_lr_mean)), key=lambda row: abs(row.auc_lr_mean - 0.5), default=None),
    }


# --------------------------------------------------------------------------
# Output: CSV + Markdown.
# --------------------------------------------------------------------------


def write_csv(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(EvaluationRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(
    result: dict[str, Any],
    *,
    report_path: Path,
    csv_path: Path | None,
) -> str:
    rows: list[EvaluationRow] = list(result["rows"])
    best_rf: EvaluationRow | None = result.get("best_rf")
    best_lr: EvaluationRow | None = result.get("best_lr")
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"

    lines: list[str] = [
        "# exp32 Hybrid X-Realism Adversarial Discriminator (single dataset)",
        "",
        f"- audit_scope: `{EXP32_AUDIT_SCOPE}`",
        f"- dataset: `{result['dataset']}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- n_real: `{result['n_real']}`",
        f"- n_features: `{result['n_features']}`",
        f"- axis_range: `{result['axis_min']:g} - {result['axis_max']:g}`",
        "",
        "## Goal",
        "",
        "- Produce synthetic spectra distributionally indistinguishable from this dataset.",
        "- Win condition: RandomForest AUC near 0.5 across CV splits, with LogisticRegression AUC also near 0.5.",
        "- Forbidden inputs (kept from prior doctrine): no labels, no targets, no splits-as-oracle, no downstream metrics, no transfer scores. Adversarial AUC IS the tuning oracle (per `docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`).",
        "- No `nirs4all/` import; no Y file is read.",
        "",
        "## Per-PCA-Rank Sweep",
        "",
        "| n_pca_requested | n_pca_actual | n_peaks | RF AUC mean +/- std (min/max) | LR AUC mean +/- std (min/max) |",
        "|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.n_pca_components_requested}` | `{row.n_pca_components_actual}` | `{row.n_peaks_fitted}` | "
            f"`{row.auc_rf_mean:.4f} +/- {row.auc_rf_std:.4f} ({row.auc_rf_min:.4f}/{row.auc_rf_max:.4f})` | "
            f"`{row.auc_lr_mean:.4f} +/- {row.auc_lr_std:.4f} ({row.auc_lr_min:.4f}/{row.auc_lr_max:.4f})` |"
        )

    lines.extend(["", "## Best Per Discriminator (closest to 0.5)", ""])
    if best_rf is not None:
        lines.append(
            f"- RF best: n_pca_requested=`{best_rf.n_pca_components_requested}`, "
            f"AUC mean=`{best_rf.auc_rf_mean:.4f}`, std=`{best_rf.auc_rf_std:.4f}`."
        )
    else:
        lines.append("- RF best: `n/a`.")
    if best_lr is not None:
        lines.append(
            f"- LR best: n_pca_requested=`{best_lr.n_pca_components_requested}`, "
            f"AUC mean=`{best_lr.auc_lr_mean:.4f}`, std=`{best_lr.auc_lr_std:.4f}`."
        )
    else:
        lines.append("- LR best: `n/a`.")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp32_hybrid_xrealism_discriminator.py \\",
            f"  --dataset {result['dataset']} \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path if csv_path is not None else 'bench/nirs_synthetic_pfn/reports/xrealism_single_dataset.csv'}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _parse_pca_range(spec: str) -> tuple[int, ...]:
    if not spec:
        return DEFAULT_PCA_RANGE
    return tuple(int(token) for token in spec.split(",") if token.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the dataset directory containing Xtrain.csv (and optionally Xtest.csv).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pca-range", type=str, default="", help=f"Comma-separated PCA rank values (default: {','.join(str(v) for v in DEFAULT_PCA_RANGE)}).")
    parser.add_argument("--n-synthetic-factor", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--baseline-degree", type=int, default=3)
    parser.add_argument("--max-peaks", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--classifiers", type=str, default="rf,lr", help="Comma-separated subset of {rf, lr}.")
    parser.add_argument("--score-sampling-mode", type=str, default="gaussian", choices=("gaussian", "empirical", "joint_bootstrap", "gmm", "copula", "knn_mixup"))
    parser.add_argument("--noise-sampling-mode", type=str, default="gaussian", choices=("gaussian", "empirical", "joint_bootstrap"))
    parser.add_argument("--empirical-jitter-fraction", type=float, default=0.05)
    parser.add_argument("--score-gmm-components", type=int, default=8)
    parser.add_argument("--score-gmm-covariance-type", type=str, default="full", choices=("full", "tied", "diag", "spherical"))
    parser.add_argument("--score-knn-mixup-k", type=int, default=5)
    parser.add_argument("--score-knn-mixup-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--multiplicative-scattering-degree", type=int, default=0)
    parser.add_argument("--additive-baseline-shift-std", type=float, default=0.0)
    parser.add_argument("--subsample-rows", type=int, default=None)
    args = parser.parse_args()

    pca_range = _parse_pca_range(args.pca_range)
    classifiers = tuple(token.strip() for token in args.classifiers.split(",") if token.strip())

    result = evaluate_dataset(
        args.dataset,
        pca_range=pca_range,
        n_synthetic_factor=args.n_synthetic_factor,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        baseline_degree=args.baseline_degree,
        max_peaks=args.max_peaks,
        n_estimators=args.n_estimators,
        classifiers=classifiers,
        score_sampling_mode=args.score_sampling_mode,
        noise_sampling_mode=args.noise_sampling_mode,
        empirical_jitter_fraction=args.empirical_jitter_fraction,
        score_gmm_components=args.score_gmm_components,
        score_gmm_covariance_type=args.score_gmm_covariance_type,
        score_knn_mixup_k=args.score_knn_mixup_k,
        score_knn_mixup_dirichlet_alpha=args.score_knn_mixup_dirichlet_alpha,
        multiplicative_scattering_degree=args.multiplicative_scattering_degree,
        additive_baseline_shift_std=args.additive_baseline_shift_std,
        subsample_rows=args.subsample_rows,
    )
    if args.csv is not None:
        write_csv(list(result["rows"]), args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    summary = {
        "dataset": result["dataset"],
        "n_real": result["n_real"],
        "n_features": result["n_features"],
        "best_rf_auc_mean": result["best_rf"].auc_rf_mean if result["best_rf"] is not None else None,
        "best_rf_n_pca": result["best_rf"].n_pca_components_requested if result["best_rf"] is not None else None,
        "best_lr_auc_mean": result["best_lr"].auc_lr_mean if result["best_lr"] is not None else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
