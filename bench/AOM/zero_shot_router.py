"""Zero-Shot Preprocessing Router for PLS.

CHANGELOG:
    v1: 4 hardcoded heuristic thresholds, hard routing to 1 pipeline
    v2: 10+ spectral features, soft scoring, 2-split holdout validation, 10% threshold
    v3: Proper 3-fold CV validation (much more robust than 2-split holdout),
        15% improvement threshold to override raw (stricter to avoid regression).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor


def _extract_spectral_features(X):
    """Extract comprehensive spectral features for intelligent routing.

    Returns a dict of named features describing the spectral characteristics.
    """
    n, p = X.shape

    # Per-sample statistics
    row_means = np.mean(X, axis=1)
    row_stds = np.std(X, axis=1)

    # Baseline / drift indicators
    dx = np.diff(X, axis=1)
    var_x = np.var(X, axis=1)
    var_dx = np.var(dx, axis=1)
    baseline_ratio = np.mean(var_x / (var_dx + 1e-10))

    # Noise level: ratio of high-frequency to low-frequency variance
    d2x = np.diff(dx, axis=1)
    var_d2x = np.var(d2x, axis=1)
    noise_ratio = np.mean(var_d2x / (var_dx + 1e-10))

    # Scatter indicators
    scatter_cv = np.mean(row_stds / (np.abs(row_means) + 1e-10))  # coefficient of variation
    mean_range = np.max(row_means) - np.min(row_means)
    std_range = np.max(row_stds) - np.min(row_stds)
    scatter_severity = std_range / (np.mean(row_stds) + 1e-10)

    # Skewness / asymmetry
    centered = X - np.mean(X, axis=1, keepdims=True)
    skew = np.mean(np.mean(centered ** 3, axis=1) / (row_stds ** 3 + 1e-10))

    # Correlation between samples (high = structured signal, low = noisy)
    if n > 5:
        sample_corr = np.corrcoef(X[:min(50, n)])[np.triu_indices(min(50, n), k=1)]
        mean_corr = np.mean(sample_corr)
    else:
        mean_corr = 0.5

    # Spectral smoothness: autocorrelation lag-1 of mean spectrum
    mean_spec = np.mean(X, axis=0)
    ac = np.corrcoef(mean_spec[:-1], mean_spec[1:])[0, 1]

    # Peak-to-noise ratio
    spec_range = np.max(mean_spec) - np.min(mean_spec)
    noise_est = np.std(d2x) / np.sqrt(6)  # Donoho-Johnstone noise estimate
    pnr = spec_range / (noise_est + 1e-10)

    return {
        'baseline_ratio': baseline_ratio,
        'noise_ratio': noise_ratio,
        'scatter_cv': scatter_cv,
        'scatter_severity': scatter_severity,
        'skewness': abs(skew),
        'mean_corr': mean_corr,
        'smoothness': ac,
        'pnr': pnr,
        'n_samples': n,
        'n_features': p,
    }


def _build_candidate_pipelines():
    """Build candidate preprocessing pipelines.

    Focus on NON-LINEAR transforms that AOM-PLS cannot do natively (SNV, MSC,
    WaveletDenoise). AOM-PLS already handles linear operators (SG, Detrend)
    internally, so the router's value-add is in applying non-linear preprocessing
    BEFORE AOM-PLS.
    """
    from nirs4all.operators.transforms import (
        StandardNormalVariate as SNV,
        MultiplicativeScatterCorrection as MSC,
        SavitzkyGolay as SG,
        Detrend,
        WaveletDenoise,
    )

    return {
        # Always include raw - lets AOM-PLS handle everything with its linear bank
        "raw": [],
        # Non-linear scatter corrections (AOM-PLS can't do these)
        "snv": [SNV()],
        "msc": [MSC()],
        # Non-linear + linear combinations
        "snv_d1": [SNV(), SG(window_length=21, polyorder=2, deriv=1)],
        "snv_detrend": [SNV(), Detrend()],
        "msc_d1": [MSC(), SG(window_length=21, polyorder=2, deriv=1)],
        "detrend_snv": [Detrend(), SNV()],
        # Wavelet denoising (non-linear thresholding)
        "wavelet": [WaveletDenoise(wavelet='db4', level=4)],
        "wavelet_snv": [WaveletDenoise(wavelet='db4', level=4), SNV()],
    }


def _compute_pipeline_scores(features):
    """Score each pipeline based on spectral features (soft gating).

    Returns dict of pipeline_name -> score (unnormalized, higher = better match).
    Must match the pipeline names from _build_candidate_pipelines().
    """
    br = features['baseline_ratio']
    nr = features['noise_ratio']
    sc = features['scatter_cv']
    ss = features['scatter_severity']
    sk = features['skewness']
    mc = features['mean_corr']
    sm = features['smoothness']
    pnr = features['pnr']

    scores = {}

    # Raw: best when data is clean (let AOM-PLS handle internally)
    scores['raw'] = 0.5 * min(pnr / 100, 1.0) + 0.3 * (1 - min(sc, 1.0)) + 0.2 * (1 - nr / 10)

    # SNV: strong scatter indicator
    scores['snv'] = 0.4 * min(sc, 2.0) + 0.3 * min(ss, 2.0) + 0.3 * (1 - nr / 10)

    # MSC: multiplicative scatter
    scores['msc'] = 0.3 * min(sc, 2.0) + 0.3 * sk + 0.2 * min(ss, 2.0) + 0.2 * (1 - nr / 10)

    # SNV + d1: scatter + baseline (most universal NIRS pipeline)
    scores['snv_d1'] = (0.25 * min(sc, 2.0) + 0.25 * min(br / 500, 2.0)
                        + 0.25 * mc + 0.25 * sm)

    # SNV + Detrend
    scores['snv_detrend'] = 0.3 * min(sc, 2.0) + 0.3 * min(br / 500, 1.0) + 0.4 * ss

    # MSC + d1
    scores['msc_d1'] = (0.3 * min(sc, 2.0) + 0.25 * min(br / 500, 1.0)
                        + 0.25 * sk + 0.2 * mc)

    # Detrend + SNV: strong baseline + scatter
    scores['detrend_snv'] = (0.4 * min(br / 1000, 2.0) + 0.3 * min(sc, 2.0)
                             + 0.3 * (1 - sm))

    # Wavelet: high noise
    scores['wavelet'] = 0.5 * min(nr / 5, 1.0) + 0.3 * (1 - pnr / 200) + 0.2 * (1 - mc)

    # Wavelet + SNV: high noise + scatter
    scores['wavelet_snv'] = (0.4 * min(nr / 5, 1.0) + 0.3 * min(sc, 2.0)
                             + 0.3 * (1 - pnr / 200))

    return scores


class ZeroShotRouterPLSRegressor(BaseEstimator, RegressorMixin):
    """Soft-gating zero-shot router: scores 8 candidate pipelines using spectral
    features, applies the top-3 in parallel, and picks the best by internal
    cross-validation RMSE.

    v2: 10+ spectral features, 8 diverse pipelines, soft scoring + validation.
    """
    def __init__(self, n_components=15):
        self.n_components = n_components

    def fit(self, X, y):
        # Extract spectral features
        features = _extract_spectral_features(X)
        self.features_ = features

        # Score pipelines (for diagnostics) and prepare validation
        raw_scores = _compute_pipeline_scores(features)
        pipelines = _build_candidate_pipelines()
        self.pipeline_scores_ = raw_scores

        # Use heuristic scores to pre-filter to top candidates, then validate
        sorted_names = sorted(raw_scores, key=raw_scores.get, reverse=True)
        candidates = sorted_names[:4]
        if "raw" not in candidates:
            candidates.append("raw")
        pipeline_rmses = {}

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        folds = list(kf.split(X))

        # Track per-fold RMSEs for pairwise comparison against raw
        pipeline_fold_rmses = {}

        for name in candidates:
            try:
                fold_rmses = []
                for train_idx, val_idx in folds:
                    fresh_ops = _build_candidate_pipelines()[name]
                    X_tr = X[train_idx].copy()
                    for op in fresh_ops:
                        if hasattr(op, 'fit'):
                            op.fit(X_tr)
                        X_tr = op.apply(X_tr) if hasattr(op, 'apply') else op.transform(X_tr)

                    X_va = X[val_idx].copy()
                    for op in fresh_ops:
                        X_va = op.apply(X_va) if hasattr(op, 'apply') else op.transform(X_va)

                    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
                        break

                    pls = AOMPLSRegressor(n_components=self.n_components)
                    pls.fit(X_tr, y[train_idx])
                    y_pred = pls.predict(X_va).flatten()
                    fold_rmses.append(np.sqrt(np.mean((y[val_idx] - y_pred) ** 2)))

                if len(fold_rmses) == 3:
                    pipeline_rmses[name] = np.mean(fold_rmses)
                    pipeline_fold_rmses[name] = fold_rmses
            except Exception:
                continue

        self.pipeline_rmses_ = pipeline_rmses

        # Very conservative: must beat raw on ALL folds AND by >25% on average.
        # High threshold because AOM-PLS already handles linear operators internally,
        # so preprocessing rarely provides >25% lift unless there's severe scatter/noise.
        raw_rmse = pipeline_rmses.get("raw", np.inf)
        raw_folds = pipeline_fold_rmses.get("raw", [np.inf] * 3)
        best_name = "raw"

        if pipeline_rmses:
            best_candidate = min(pipeline_rmses, key=pipeline_rmses.get)
            best_rmse = pipeline_rmses[best_candidate]

            if best_candidate != "raw" and best_rmse < raw_rmse * 0.75:
                cand_folds = pipeline_fold_rmses[best_candidate]
                all_better = all(cf < rf for cf, rf in zip(cand_folds, raw_folds))
                if all_better:
                    best_name = best_candidate

        self.selected_pipeline_ = best_name
        self.cv_rmse_ = pipeline_rmses.get(best_name, np.inf)

        # Always fit raw AOM-PLS as safety baseline
        self.pls_raw_ = AOMPLSRegressor(n_components=self.n_components)
        self.pls_raw_.fit(X, y)

        if best_name == "raw":
            self.prep_ = []
            self.pls_ = self.pls_raw_
            self.blend_weight_ = 0.0
        else:
            # Fit preprocessed model
            fresh_ops = _build_candidate_pipelines()[best_name]
            X_pre = X.copy()
            self.prep_ = []
            for op in fresh_ops:
                if hasattr(op, 'fit'):
                    op.fit(X_pre)
                self.prep_.append(op)
                X_pre = op.apply(X_pre) if hasattr(op, 'apply') else op.transform(X_pre)

            self.pls_ = AOMPLSRegressor(n_components=self.n_components)
            self.pls_.fit(X_pre, y)

            # Blend weight based on CV improvement confidence
            improvement = 1 - best_rmse / raw_rmse  # e.g., 0.30 means 30% better
            self.blend_weight_ = min(improvement * 2, 0.8)  # cap at 80% preprocessed

        return self

    def predict(self, X):
        if self.blend_weight_ == 0.0:
            return self.pls_raw_.predict(X).flatten()

        # Blend: preprocessed prediction + raw prediction
        X_pre = X.copy()
        for op in self.prep_:
            X_pre = op.apply(X_pre) if hasattr(op, 'apply') else op.transform(X_pre)
        pred_pre = self.pls_.predict(X_pre).flatten()
        pred_raw = self.pls_raw_.predict(X).flatten()

        w = self.blend_weight_
        return w * pred_pre + (1 - w) * pred_raw
