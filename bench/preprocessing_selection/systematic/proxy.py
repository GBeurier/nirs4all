"""Proxy model evaluation for preprocessing selection."""

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def evaluate_with_proxies(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    is_classification: bool = False,
) -> Dict[str, float]:
    """Evaluate preprocessing with Ridge, KNN, and XGBoost proxy models.

    Args:
        X: Transformed feature matrix (n_samples, n_features).
        y: Target values.
        cv_folds: Number of cross-validation folds.
        is_classification: Whether this is a classification task.

    Returns:
        Dictionary with ridge_r2, knn_score, xgb_score, and composite proxy_score.
    """
    results = {}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle high-dimensional data
    if X_scaled.shape[1] > X_scaled.shape[0]:
        pca = PCA(n_components=min(50, X_scaled.shape[0] - 1))
        X_scaled = pca.fit_transform(X_scaled)

    # Ridge
    try:
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        scores = cross_val_score(ridge, X_scaled, y, cv=cv_folds, scoring="r2")
        results["ridge_r2"] = float(np.mean(scores))
    except Exception:
        results["ridge_r2"] = 0.0

    # KNN
    try:
        if is_classification:
            knn = KNeighborsClassifier(n_neighbors=5)
            scoring = "accuracy"
        else:
            knn = KNeighborsRegressor(n_neighbors=5)
            scoring = "r2"
        scores = cross_val_score(knn, X_scaled, y, cv=cv_folds, scoring=scoring)
        results["knn_score"] = float(np.mean(scores))
    except Exception:
        results["knn_score"] = 0.0

    # XGBoost (minimal config for speed)
    if HAS_XGBOOST:
        try:
            if is_classification:
                xgb_model = XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=0,
                    n_jobs=1,
                    random_state=42,
                )
                scoring = "accuracy"
            else:
                xgb_model = XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=0,
                    n_jobs=1,
                    random_state=42,
                )
                scoring = "r2"
            scores = cross_val_score(xgb_model, X_scaled, y, cv=cv_folds, scoring=scoring)
            results["xgb_score"] = float(np.mean(scores))
        except Exception:
            results["xgb_score"] = 0.0
    else:
        results["xgb_score"] = 0.0

    # Composite score (weighted average of all three proxies)
    results["proxy_score"] = (
        0.4 * results["ridge_r2"]
        + 0.3 * results["knn_score"]
        + 0.3 * results["xgb_score"]
    )

    return results
