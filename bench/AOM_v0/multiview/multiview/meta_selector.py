"""Meta-learner for per-dataset multi-view variant selection.

Given the per-dataset RMSEP matrix from full-57.csv, train a small classifier
that predicts the best multi-view variant for a new dataset based on simple
features extracted from (X, y).

Features per dataset:
- n_train, p, log(n), log(p), p/n ratio
- spectral mean / std / kurtosis / skewness
- block-variance ratio (variance within K=3 equal-width blocks vs total)
- mean abs first-derivative (smoothness proxy)
- y std / range
- y skewness, kurtosis

Classifier: leave-one-dataset-out cross-validation.
- Multinomial logistic regression (small capacity)
- RandomForestClassifier (bigger capacity, baseline)

Uses scikit-learn only — no fitting of multi-view variants here, just
selection from precomputed RMSEP matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(X: np.ndarray, y: np.ndarray, K: int = 3) -> Dict[str, float]:
    """Compute meta-learning features from a (X, y) regression dataset.

    Returns a dict of scalar features. Stable across different X shapes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    feats: Dict[str, float] = {
        "n": float(n),
        "p": float(p),
        "log_n": float(np.log(max(n, 1))),
        "log_p": float(np.log(max(p, 1))),
        "p_over_n": float(p / max(n, 1)),
    }

    # Spectral statistics
    spec_mean = float(X.mean())
    spec_std = float(X.std())
    spec_kurt = float(stats.kurtosis(X, axis=None, fisher=True, bias=False))
    spec_skew = float(stats.skew(X, axis=None, bias=False))
    feats.update({
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "spec_kurt": spec_kurt,
        "spec_skew": spec_skew,
    })

    # Block-variance ratio: how much variance is in each equal-width block
    block_edges = np.linspace(0, p, K + 1).astype(int)
    block_vars = []
    total_var = float(X.var())
    for k in range(K):
        s, e = block_edges[k], block_edges[k + 1]
        block_vars.append(float(X[:, s:e].var()))
    feats["block_var_max_ratio"] = float(max(block_vars) / max(total_var, 1e-12))
    feats["block_var_min_ratio"] = float(min(block_vars) / max(total_var, 1e-12))
    feats["block_var_std_ratio"] = float(np.std(block_vars) / max(total_var, 1e-12))

    # Smoothness via mean abs first derivative
    if p >= 2:
        diffs = np.abs(np.diff(X, axis=1))
        feats["mean_abs_diff"] = float(diffs.mean())
        feats["std_abs_diff"] = float(diffs.std())
    else:
        feats["mean_abs_diff"] = 0.0
        feats["std_abs_diff"] = 0.0

    # Cross-cov skew between blocks (captures whether y signal lives in a
    # specific block — block-aware methods should help when this is high).
    # This is the only feature that retained signal across feature-set
    # ablations; FFT bands and PCA eigenvalue ratios added dimensions
    # without help on a 58-row training set (curse of dimensionality).
    try:
        Xc = X - X.mean(axis=0, keepdims=True)
        S = (Xc.T @ (y - y.mean())) / max(n - 1, 1)  # (p,)
        block_S2 = []
        for k in range(K):
            s, e = block_edges[k], block_edges[k + 1]
            block_S2.append(float((S[s:e] ** 2).sum()))
        block_S2 = np.array(block_S2)
        feats["xcov_block_max_ratio"] = float(block_S2.max() / (block_S2.sum() + 1e-12))
    except Exception:
        feats["xcov_block_max_ratio"] = 0.0

    # y statistics
    y_std = float(np.std(y))
    feats.update({
        "y_std": y_std,
        "y_range": float(y.max() - y.min()),
        "y_kurt": float(stats.kurtosis(y, fisher=True, bias=False)),
        "y_skew": float(stats.skew(y, bias=False)),
    })

    return feats


# ---------------------------------------------------------------------------
# Meta-classifier evaluation via leave-one-out
# ---------------------------------------------------------------------------


def leave_one_out_select(
    feature_matrix: pd.DataFrame,
    rmsep_matrix: pd.DataFrame,
    variants: List[str],
    classifier="logreg",
    random_state: int = 0,
) -> pd.Series:
    """Predict the best variant per dataset via leave-one-out classification.

    Args:
        feature_matrix: index=dataset, columns=feature names. All numeric.
        rmsep_matrix: index=dataset, columns=variant names. Per-cell RMSEP.
        variants: list of variant names to choose between (must be in rmsep_matrix.columns).
        classifier: "logreg" or "rf".
    Returns:
        pandas Series indexed by dataset with the predicted variant name.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    common = feature_matrix.index.intersection(rmsep_matrix.index)
    feature_matrix = feature_matrix.loc[common].copy()
    rmsep_matrix = rmsep_matrix.loc[common, variants].copy()

    # Drop rows where all variant RMSEPs are NaN
    rmsep_matrix = rmsep_matrix.dropna(how="all")
    feature_matrix = feature_matrix.loc[rmsep_matrix.index]

    # Best variant per dataset = label
    labels = rmsep_matrix.idxmin(axis=1)

    predictions = pd.Series(index=labels.index, dtype=object)
    feat_arr = feature_matrix.to_numpy(dtype=float)
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=1e6, neginf=-1e6)
    for i, ds in enumerate(labels.index):
        train_idx = np.array([j for j in range(len(labels)) if j != i])
        X_tr = feat_arr[train_idx]
        y_tr = labels.iloc[train_idx].to_numpy()
        X_te = feat_arr[i:i + 1]
        # Skip if a class only appears once in train (LogReg needs ≥2)
        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            if classifier == "logreg":
                clf = LogisticRegression(
                    max_iter=2000, multi_class="multinomial",
                    class_weight="balanced", random_state=random_state,
                )
            elif classifier == "rf":
                clf = RandomForestClassifier(
                    n_estimators=200, max_depth=4, min_samples_leaf=2,
                    class_weight="balanced", random_state=random_state, n_jobs=1,
                )
            else:
                raise ValueError(f"unknown classifier: {classifier}")
            clf.fit(X_tr_s, y_tr)
            predictions.iloc[i] = clf.predict(X_te_s)[0]
        except Exception:
            # Fallback: pick the variant with lowest median RMSEP across train
            train_med = rmsep_matrix.iloc[train_idx].median(axis=0)
            predictions.iloc[i] = train_med.idxmin()
    return predictions


def selector_rmsep(
    predictions: pd.Series,
    rmsep_matrix: pd.DataFrame,
) -> pd.Series:
    """Look up the predicted-variant RMSEP per dataset."""
    out = pd.Series(index=predictions.index, dtype=float)
    for ds, var in predictions.items():
        if var in rmsep_matrix.columns:
            out.loc[ds] = rmsep_matrix.loc[ds, var]
        else:
            out.loc[ds] = float("nan")
    return out
