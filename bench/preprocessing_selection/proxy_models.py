"""
Proxy Models for Preprocessing Selection
=========================================

This module provides quick proxy models for final ranking of top preprocessing
candidates. These are Stage C methods that provide more accurate ranking than
Stage B metrics but are still much faster than full model training.

Stage C: Proxy Models (Final Selection)
- Ridge Regression: Fast linear model with regularization
- KNN: Simple non-parametric model
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def ridge_proxy(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    alphas: list = None
) -> dict:
    """
    Quick Ridge regression evaluation.

    Fast Ridge regression with minimal CV (2-3 folds) for final ranking
    of preprocessing candidates.

    Args:
        X_preprocessed: Transformed spectra (n_samples, n_features)
        y: Target values (n_samples,)
        cv_folds: Number of CV folds (default: 3)
        alphas: Ridge regularization values to try

    Returns:
        dict with 'ridge_r2', 'best_alpha', 'ridge_std'
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    try:
        ridge = RidgeCV(alphas=alphas, cv=cv_folds)
        ridge.fit(X_preprocessed, y)

        # Get score with best alpha using cross-validation
        scores = cross_val_score(ridge, X_preprocessed, y, cv=cv_folds, scoring='r2')

        return {
            'ridge_r2': float(np.mean(scores)),
            'ridge_std': float(np.std(scores)),
            'best_alpha': float(ridge.alpha_)
        }
    except Exception as e:
        return {
            'ridge_r2': 0.0,
            'ridge_std': 0.0,
            'best_alpha': None,
            'error': str(e)
        }


def knn_proxy(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    cv_folds: int = 3,
    task: str = 'auto'
) -> dict:
    """
    Quick KNN evaluation.

    Fast KNN with k=3-5 for quick preprocessing evaluation.
    Automatically detects regression vs classification based on target.

    Args:
        X_preprocessed: Transformed spectra (n_samples, n_features)
        y: Target values (n_samples,)
        n_neighbors: Number of neighbors (default: 3)
        cv_folds: Number of CV folds (default: 3)
        task: 'regression', 'classification', or 'auto'

    Returns:
        dict with 'knn_score', 'knn_std', 'task'
    """
    if task == 'auto':
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'

    try:
        if task == 'regression':
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            scoring = 'r2'
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            scoring = 'accuracy'

        scores = cross_val_score(knn, X_preprocessed, y, cv=cv_folds, scoring=scoring)

        return {
            'knn_score': float(np.mean(scores)),
            'knn_std': float(np.std(scores)),
            'task': task,
            'n_neighbors': n_neighbors
        }
    except Exception as e:
        return {
            'knn_score': 0.0,
            'knn_std': 0.0,
            'task': task,
            'n_neighbors': n_neighbors,
            'error': str(e)
        }


def evaluate_proxies(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    ridge_alphas: list = None,
    knn_neighbors: int = 3,
    task: str = 'auto'
) -> dict:
    """
    Run all Stage C proxy models on a preprocessed dataset.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        cv_folds: Number of CV folds
        ridge_alphas: Ridge regularization values
        knn_neighbors: Number of neighbors for KNN
        task: 'regression', 'classification', or 'auto'

    Returns:
        dict with results from all proxy models and composite score
    """
    results = {}

    # Determine task type
    if task == 'auto':
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'

    # Ridge proxy
    ridge_result = ridge_proxy(
        X_preprocessed,
        y,
        cv_folds=cv_folds,
        alphas=ridge_alphas
    )
    results['ridge'] = ridge_result

    # KNN proxy
    knn_result = knn_proxy(
        X_preprocessed,
        y,
        n_neighbors=knn_neighbors,
        cv_folds=cv_folds,
        task=task
    )
    results['knn'] = knn_result

    # Composite score (average of both proxies)
    scores = []
    if 'error' not in ridge_result:
        scores.append(max(0, ridge_result['ridge_r2']))
    if 'error' not in knn_result:
        scores.append(max(0, knn_result['knn_score']))

    results['composite_score'] = float(np.mean(scores)) if scores else 0.0
    results['task'] = task

    return results
