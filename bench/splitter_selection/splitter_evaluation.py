"""
Splitter Evaluation Module
==========================

This module provides functions for evaluating splitting strategies using
baseline models. It trains models on each fold and evaluates performance
on both validation and test sets.

Metrics computed:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Generalization gap (CV vs Test performance)
- Fold stability (variance across folds)
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None

from splitter_strategies import SplitResult


@dataclass
class FoldResult:
    """Results from training on a single fold."""
    fold_idx: int
    n_train: int
    n_val: int
    train_rmse: float
    train_mae: float
    train_r2: float
    val_rmse: float
    val_mae: float
    val_r2: float
    train_time: float
    y_val_true: np.ndarray
    y_val_pred: np.ndarray

@dataclass
class StrategyResult:
    """Aggregated results for a splitting strategy."""
    strategy_key: str
    strategy_name: str
    category: str
    fold_results: list[FoldResult]
    test_rmse: float
    test_mae: float
    test_r2: float
    n_test: int
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    generalization_gap: float
    strategy_info: dict[str, Any]

class BaselineModel:
    """Wrapper for baseline regression models."""

    def __init__(self, model_type: str = 'ridge', **kwargs):
        """
        Initialize baseline model.

        Args:
            model_type: One of 'ridge', 'pls', 'knn'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self):
        """Create a fresh model instance."""
        if self.model_type == 'ridge':
            alphas = self.kwargs.get('alphas', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            return RidgeCV(alphas=alphas)
        elif self.model_type == 'pls':
            n_components = self.kwargs.get('n_components', 10)
            return PLSRegression(n_components=n_components)
        elif self.model_type == 'knn':
            n_neighbors = self.kwargs.get('n_neighbors', 5)
            return KNeighborsRegressor(n_neighbors=n_neighbors)
        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
            n_estimators = self.kwargs.get('n_estimators', 100)
            max_depth = self.kwargs.get('max_depth', 3)
            learning_rate = self.kwargs.get('learning_rate', 0.1)
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.kwargs.get('random_state', 42),
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model."""
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return pred.ravel() if len(pred.shape) > 1 else pred

def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_idx: int,
    model_type: str = 'ridge',
    **model_kwargs
) -> FoldResult:
    """
    Train model on a single fold and evaluate.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        fold_idx: Fold index
        model_type: Type of baseline model
        **model_kwargs: Model-specific parameters

    Returns:
        FoldResult with metrics and predictions
    """
    model = BaselineModel(model_type, **model_kwargs)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    return FoldResult(
        fold_idx=fold_idx,
        n_train=len(y_train),
        n_val=len(y_val),
        train_rmse=train_rmse,
        train_mae=train_mae,
        train_r2=train_r2,
        val_rmse=val_rmse,
        val_mae=val_mae,
        val_r2=val_r2,
        train_time=train_time,
        y_val_true=y_val,
        y_val_pred=y_val_pred
    )

def evaluate_strategy(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    split_result: SplitResult,
    strategy_key: str,
    strategy_name: str,
    category: str,
    model_type: str = 'ridge',
    verbose: bool = True,
    **model_kwargs
) -> StrategyResult:
    """
    Evaluate a splitting strategy by training on all folds and testing.

    Args:
        X: Full feature matrix
        y: Full target vector
        sample_ids: Sample IDs for each observation
        split_result: SplitResult from a splitter
        strategy_key: Short key for the strategy
        strategy_name: Display name for the strategy
        category: Category of the strategy
        model_type: Baseline model type
        verbose: Print progress
        **model_kwargs: Model parameters

    Returns:
        StrategyResult with all metrics
    """
    fold_df = split_result.fold_assignments

    # Get test data
    test_ids = split_result.test_ids
    test_mask = np.isin(sample_ids, test_ids)
    X_test = X[test_mask]
    y_test = y[test_mask]

    if verbose:
        print(f"\n  Test set: {len(y_test)} samples from {len(test_ids)} unique IDs")

    # Train on each fold
    n_folds = int(fold_df[fold_df['split'] != 'test']['fold'].max() + 1)
    fold_results = []
    fold_models = []

    for fold_idx in range(n_folds):
        # Get train and validation IDs for this fold
        train_ids_fold = fold_df[
            (fold_df['fold'] != fold_idx) & (fold_df['split'] == 'train')
        ]['ID'].values

        val_ids_fold = fold_df[
            (fold_df['fold'] == fold_idx) & (fold_df['split'] == 'train')
        ]['ID'].values

        # Get data masks
        train_mask = np.isin(sample_ids, train_ids_fold)
        val_mask = np.isin(sample_ids, val_ids_fold)

        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]

        # Train and evaluate
        fold_result = train_fold(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            fold_idx,
            model_type,
            **model_kwargs
        )

        fold_results.append(fold_result)

        # Keep model for test evaluation
        model = BaselineModel(model_type, **model_kwargs)
        model.fit(X_train_fold, y_train_fold)
        fold_models.append(model)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: "
                  f"Val RMSE={fold_result.val_rmse:.3f}, "
                  f"Val R²={fold_result.val_r2:.3f}")

    # Ensemble prediction on test set
    test_predictions = np.mean([m.predict(X_test) for m in fold_models], axis=0)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Aggregate CV metrics
    val_rmses = [r.val_rmse for r in fold_results]
    val_r2s = [r.val_r2 for r in fold_results]

    cv_rmse_mean = np.mean(val_rmses)
    cv_rmse_std = np.std(val_rmses)
    cv_r2_mean = np.mean(val_r2s)
    cv_r2_std = np.std(val_r2s)

    generalization_gap = test_rmse - cv_rmse_mean

    if verbose:
        print("\n  📊 Summary:")
        print(f"     CV RMSE: {cv_rmse_mean:.3f} ± {cv_rmse_std:.3f}")
        print(f"     Test RMSE: {test_rmse:.3f}")
        print(f"     Gap (CV→Test): {generalization_gap:+.3f}")

    return StrategyResult(
        strategy_key=strategy_key,
        strategy_name=strategy_name,
        category=category,
        fold_results=fold_results,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,
        n_test=len(y_test),
        y_test_true=y_test,
        y_test_pred=test_predictions,
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        generalization_gap=generalization_gap,
        strategy_info=split_result.strategy_info
    )

def compare_strategies(
    results: list[StrategyResult]
) -> pd.DataFrame:
    """
    Create comparison DataFrame from strategy results.

    Args:
        results: List of StrategyResult objects

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for r in results:
        comparison_data.append({
            'strategy_key': r.strategy_key,
            'strategy': r.strategy_name,
            'category': r.category,
            'cv_rmse_mean': r.cv_rmse_mean,
            'cv_rmse_std': r.cv_rmse_std,
            'cv_r2_mean': r.cv_r2_mean,
            'cv_r2_std': r.cv_r2_std,
            'test_rmse': r.test_rmse,
            'test_mae': r.test_mae,
            'test_r2': r.test_r2,
            'n_test': r.n_test,
            'generalization_gap': r.generalization_gap,
            'cv_stability': r.cv_rmse_std / r.cv_rmse_mean if r.cv_rmse_mean > 0 else 0,
            'overfitting': r.cv_rmse_mean - np.mean([f.train_rmse for f in r.fold_results])
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('test_rmse')

    return df

def identify_best_strategies(
    comparison_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """
    Identify best strategies by different criteria.

    Args:
        comparison_df: Comparison DataFrame

    Returns:
        Dictionary with best strategies by criterion
    """
    best = {}

    # Best test performance
    best_idx = comparison_df['test_rmse'].idxmin()
    best['test_performance'] = {
        'strategy': comparison_df.loc[best_idx, 'strategy'],
        'test_rmse': comparison_df.loc[best_idx, 'test_rmse'],
        'test_r2': comparison_df.loc[best_idx, 'test_r2']
    }

    # Most stable CV
    best_idx = comparison_df['cv_stability'].idxmin()
    best['cv_stability'] = {
        'strategy': comparison_df.loc[best_idx, 'strategy'],
        'cv_stability': comparison_df.loc[best_idx, 'cv_stability'],
        'cv_rmse_std': comparison_df.loc[best_idx, 'cv_rmse_std']
    }

    # Best generalization (smallest gap)
    best_idx = comparison_df['generalization_gap'].abs().idxmin()
    best['generalization'] = {
        'strategy': comparison_df.loc[best_idx, 'strategy'],
        'generalization_gap': comparison_df.loc[best_idx, 'generalization_gap']
    }

    # Best R²
    best_idx = comparison_df['test_r2'].idxmax()
    best['r2'] = {
        'strategy': comparison_df.loc[best_idx, 'strategy'],
        'test_r2': comparison_df.loc[best_idx, 'test_r2']
    }

    return best

def compute_statistical_tests(
    results: list[StrategyResult],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute pairwise statistical tests between strategies.

    Args:
        results: List of StrategyResult objects
        alpha: Significance level

    Returns:
        DataFrame with pairwise test results
    """
    from scipy.stats import mannwhitneyu, ttest_ind

    # Sort by test performance
    results_sorted = sorted(results, key=lambda r: r.test_rmse)

    pairwise_results = []

    for i in range(len(results_sorted)):
        for j in range(i + 1, len(results_sorted)):
            r1 = results_sorted[i]
            r2 = results_sorted[j]

            rmse1 = [f.val_rmse for f in r1.fold_results]
            rmse2 = [f.val_rmse for f in r2.fold_results]

            # T-test
            t_stat, t_pval = ttest_ind(rmse1, rmse2)

            # Mann-Whitney U
            u_stat, u_pval = mannwhitneyu(rmse1, rmse2, alternative='two-sided')

            pairwise_results.append({
                'strategy_1': r1.strategy_name,
                'strategy_2': r2.strategy_name,
                'rmse_diff': np.mean(rmse1) - np.mean(rmse2),
                't_statistic': t_stat,
                't_pvalue': t_pval,
                't_significant': t_pval < alpha,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'u_significant': u_pval < alpha
            })

    return pd.DataFrame(pairwise_results)
