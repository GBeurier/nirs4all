"""
Enhanced Splitter Evaluation Module
====================================

This module provides comprehensive evaluation of splitting strategies with:
- Multiple models (Ridge, PLS, XGBoost, GBR, SVR, MLP)
- Repeated cross-validation (multiple random seeds)
- Bootstrap confidence intervals
- Representativeness metrics (spectral coverage, target coverage, leverage)

Metrics computed:
- RMSE, MAE, R² with bootstrap 95% CIs
- Generalization gap
- Fold stability
- Spectral coverage (PCA hull volume ratio)
- Target coverage (Wasserstein distance)
- Leverage analysis (Hotelling's T²)
"""

import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None

from splitter_strategies import SplitResult

warnings.filterwarnings('ignore')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelResult:
    """Results from a single model training."""
    model_name: str
    train_rmse: float
    train_mae: float
    train_r2: float
    val_rmse: float
    val_mae: float
    val_r2: float
    train_time: float
    y_val_pred: np.ndarray


@dataclass
class FoldResult:
    """Results from training all models on a single fold."""
    fold_idx: int
    repeat_idx: int
    n_train: int
    n_val: int
    model_results: Dict[str, ModelResult]
    ensemble_val_rmse: float
    ensemble_val_mae: float
    ensemble_val_r2: float
    y_val_true: np.ndarray
    y_val_pred_ensemble: np.ndarray


@dataclass
class BootstrapMetrics:
    """Bootstrap confidence intervals for metrics."""
    rmse_mean: float
    rmse_ci_lower: float
    rmse_ci_upper: float
    mae_mean: float
    mae_ci_lower: float
    mae_ci_upper: float
    r2_mean: float
    r2_ci_lower: float
    r2_ci_upper: float


@dataclass
class RepresentativenessMetrics:
    """Metrics for evaluating split representativeness."""
    spectral_coverage: float  # Ratio of test spectral space covered by train
    target_wasserstein: float  # Wasserstein distance between train/test targets
    target_kl_divergence: float  # KL divergence between train/test targets
    leverage_mean: float  # Mean leverage of test samples
    leverage_max: float  # Max leverage (potential extrapolation)
    n_high_leverage: int  # Number of high-leverage test samples
    hotelling_t2_mean: float  # Mean Hotelling's T² for test samples


@dataclass
class EnhancedStrategyResult:
    """Comprehensive results for a splitting strategy."""
    strategy_key: str
    strategy_name: str
    category: str

    # Fold results (all repeats)
    fold_results: List[FoldResult]
    n_repeats: int
    n_folds: int

    # Per-model test results
    model_test_results: Dict[str, Dict[str, float]]

    # Ensemble test results
    test_rmse: float
    test_mae: float
    test_r2: float
    n_test: int
    y_test_true: np.ndarray
    y_test_pred: np.ndarray

    # CV aggregated metrics
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    generalization_gap: float

    # Bootstrap CIs
    bootstrap_metrics: BootstrapMetrics

    # Representativeness
    representativeness: RepresentativenessMetrics

    # Strategy info
    strategy_info: Dict[str, Any]

    # Timing
    total_time: float


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_model_suite() -> Dict[str, Any]:
    """
    Get the full suite of models with more complex/deep parameters.

    Returns:
        Dictionary of model configurations
    """
    models = {
        'ridge': {
            'class': RidgeCV,
            'params': {
                'alphas': np.logspace(-4, 4, 20)
            },
            'category': 'linear'
        },
        'elasticnet': {
            'class': ElasticNetCV,
            'params': {
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'alphas': np.logspace(-4, 2, 15),
                'max_iter': 1000
            },
            'category': 'linear'
        },
        'pls': {
            'class': PLSRegression,
            'params': {
                'n_components': 15,  # More components
                'max_iter': 500
            },
            'category': 'linear'
        },
        'svr': {
            'class': SVR,
            'params': {
                'kernel': 'rbf',
                'C': 100.0,  # Higher regularization
                'gamma': 'scale',
                'epsilon': 0.01
            },
            'category': 'kernel'
        },
        # 'gbr': {
        #     'class': GradientBoostingRegressor,
        #     'params': {
        #         'n_estimators': 300,  # More trees
        #         'max_depth': 5,  # Deeper
        #         'learning_rate': 0.05,  # Slower learning
        #         'subsample': 0.8,
        #         'min_samples_split': 5,
        #         'min_samples_leaf': 3,
        #         'random_state': 42
        #     },
        #     'category': 'ensemble'
        # },
        'mlp': {
            'class': MLPRegressor,
            'params': {
                'hidden_layer_sizes': (256, 128, 64),  # Deep network
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,  # L2 regularization
                'batch_size': 32,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 100,
                'early_stopping': True,
                'validation_fraction': 0.15,
                'n_iter_no_change': 20,
                'random_state': 42
            },
            'category': 'deep'
        }
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        models['xgboost'] = {
            'class': XGBRegressor,
            'params': {
                'n_estimators': 200,  # More trees
                'max_depth': 6,  # Deeper
                'learning_rate': 0.03,  # Slower learning
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': 1
            },
            'category': 'ensemble'
        }

    return models


class ModelWrapper:
    """Wrapper for different model types with scaling."""

    def __init__(self, model_name: str, model_config: Dict[str, Any], random_state: int = 42):
        self.model_name = model_name
        self.model_config = model_config
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self):
        """Create a fresh model instance."""
        params = self.model_config['params'].copy()

        # Update random state if applicable
        if 'random_state' in params:
            params['random_state'] = self.random_state

        return self.model_config['class'](**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with scaling."""
        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the fitted model."""
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return pred.ravel() if len(pred.shape) > 1 else pred


# ============================================================================
# REPRESENTATIVENESS METRICS
# ============================================================================

def compute_spectral_coverage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 10
) -> float:
    """
    Compute spectral coverage using PCA convex hull volume ratio.

    Args:
        X_train: Training spectra
        X_test: Test spectra
        n_components: PCA components for hull computation

    Returns:
        Coverage ratio (0-1, higher is better)
    """
    # Fit PCA on combined data
    X_all = np.vstack([X_train, X_test])
    n_comp = min(n_components, X_all.shape[0] - 1, X_all.shape[1])

    pca = PCA(n_components=n_comp)
    pca.fit(X_all)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Use first 3 components for hull (higher dims can fail)
    if n_comp >= 3:
        X_train_hull = X_train_pca[:, :3]
        X_test_hull = X_test_pca[:, :3]
    else:
        X_train_hull = X_train_pca
        X_test_hull = X_test_pca

    try:
        # Check if test points are inside train hull
        train_hull = ConvexHull(X_train_hull)

        # Count test points inside train hull (approximate)
        from scipy.spatial import Delaunay
        try:
            train_delaunay = Delaunay(X_train_hull)
            inside = train_delaunay.find_simplex(X_test_hull) >= 0
            coverage = np.mean(inside)
        except:
            # Fallback: use distance-based coverage
            distances = cdist(X_test_hull, X_train_hull, 'euclidean')
            min_distances = np.min(distances, axis=1)
            train_radius = np.max(cdist(X_train_hull, X_train_hull.mean(axis=0, keepdims=True)))
            coverage = np.mean(min_distances < train_radius)

    except Exception:
        # Fallback for small/degenerate datasets
        distances = cdist(X_test_pca, X_train_pca, 'euclidean')
        min_distances = np.min(distances, axis=1)
        max_train_dist = np.max(cdist(X_train_pca, X_train_pca.mean(axis=0, keepdims=True)))
        coverage = np.mean(min_distances < max_train_dist)

    return float(coverage)


def compute_target_coverage(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 50
) -> Tuple[float, float]:
    """
    Compute target distribution coverage metrics.

    Args:
        y_train: Training targets
        y_test: Test targets
        n_bins: Number of bins for histogram

    Returns:
        Tuple of (Wasserstein distance, KL divergence)
    """
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein = stats.wasserstein_distance(y_train, y_test)

    # KL divergence (need to discretize)
    y_min = min(y_train.min(), y_test.min())
    y_max = max(y_train.max(), y_test.max())
    bins = np.linspace(y_min - 1e-6, y_max + 1e-6, n_bins + 1)

    train_hist, _ = np.histogram(y_train, bins=bins, density=True)
    test_hist, _ = np.histogram(y_test, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    train_hist = train_hist + eps
    test_hist = test_hist + eps

    # Normalize
    train_hist = train_hist / train_hist.sum()
    test_hist = test_hist / test_hist.sum()

    # KL divergence: KL(test || train)
    kl_div = stats.entropy(test_hist, train_hist)

    return float(wasserstein), float(kl_div)


def compute_leverage_metrics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 10
) -> Tuple[float, float, int, float]:
    """
    Compute leverage analysis for test samples.

    Leverage indicates how "unusual" a test sample is relative to training.
    High leverage samples may lead to extrapolation.

    Args:
        X_train: Training features
        X_test: Test features
        n_components: PCA components

    Returns:
        Tuple of (mean_leverage, max_leverage, n_high_leverage, mean_t2)
    """
    # Fit PCA on training
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Compute covariance of training data in PCA space
    cov_train = np.cov(X_train_pca.T)
    if cov_train.ndim == 0:
        cov_train = np.array([[cov_train]])

    try:
        cov_inv = np.linalg.inv(cov_train)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_train)

    # Center test data relative to training
    train_mean = X_train_pca.mean(axis=0)
    X_test_centered = X_test_pca - train_mean

    # Hotelling's T² for each test sample
    t2_scores = np.array([
        x @ cov_inv @ x.T for x in X_test_centered
    ])

    # Leverage: diagonal of hat matrix H = X(X'X)^{-1}X'
    # For PCA-transformed data
    n = X_train_pca.shape[0]
    leverage = 1/n + t2_scores / (n - 1)

    # High leverage threshold (common rule: 2 * (p+1) / n)
    p = n_comp
    high_leverage_threshold = 2 * (p + 1) / n
    n_high = int(np.sum(leverage > high_leverage_threshold))

    return (
        float(np.mean(leverage)),
        float(np.max(leverage)),
        n_high,
        float(np.mean(t2_scores))
    )


def compute_representativeness(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_pca_components: int = 10
) -> RepresentativenessMetrics:
    """
    Compute all representativeness metrics.

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        n_pca_components: PCA components for analysis

    Returns:
        RepresentativenessMetrics dataclass
    """
    spectral_coverage = compute_spectral_coverage(X_train, X_test, n_pca_components)
    wasserstein, kl_div = compute_target_coverage(y_train, y_test)
    lev_mean, lev_max, n_high, t2_mean = compute_leverage_metrics(
        X_train, X_test, n_pca_components
    )

    return RepresentativenessMetrics(
        spectral_coverage=spectral_coverage,
        target_wasserstein=wasserstein,
        target_kl_divergence=kl_div,
        leverage_mean=lev_mean,
        leverage_max=lev_max,
        n_high_leverage=n_high,
        hotelling_t2_mean=t2_mean
    )


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> BootstrapMetrics:
    """
    Compute bootstrap confidence intervals for regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        BootstrapMetrics with CIs
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    rmse_samples = []
    mae_samples = []
    r2_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        rmse_samples.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        mae_samples.append(mean_absolute_error(y_true_boot, y_pred_boot))

        # R² can be undefined if all same value
        try:
            r2_samples.append(r2_score(y_true_boot, y_pred_boot))
        except:
            r2_samples.append(np.nan)

    alpha = 1 - confidence

    rmse_samples = np.array(rmse_samples)
    mae_samples = np.array(mae_samples)
    r2_samples = np.array([r for r in r2_samples if not np.isnan(r)])

    return BootstrapMetrics(
        rmse_mean=float(np.mean(rmse_samples)),
        rmse_ci_lower=float(np.percentile(rmse_samples, 100 * alpha / 2)),
        rmse_ci_upper=float(np.percentile(rmse_samples, 100 * (1 - alpha / 2))),
        mae_mean=float(np.mean(mae_samples)),
        mae_ci_lower=float(np.percentile(mae_samples, 100 * alpha / 2)),
        mae_ci_upper=float(np.percentile(mae_samples, 100 * (1 - alpha / 2))),
        r2_mean=float(np.mean(r2_samples)) if len(r2_samples) > 0 else 0.0,
        r2_ci_lower=float(np.percentile(r2_samples, 100 * alpha / 2)) if len(r2_samples) > 0 else 0.0,
        r2_ci_upper=float(np.percentile(r2_samples, 100 * (1 - alpha / 2))) if len(r2_samples) > 0 else 0.0
    )


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_models_on_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    models: Dict[str, Any],
    random_state: int = 42
) -> Tuple[Dict[str, ModelResult], Dict[str, ModelWrapper]]:
    """
    Train all models on a single fold.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        models: Model configurations
        random_state: Random seed

    Returns:
        Tuple of (model results dict, fitted models dict)
    """
    results = {}
    fitted_models = {}

    for model_name, model_config in models.items():
        wrapper = ModelWrapper(model_name, model_config, random_state)

        start_time = time.time()
        try:
            wrapper.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_train_pred = wrapper.predict(X_train)
            y_val_pred = wrapper.predict(X_val)

            results[model_name] = ModelResult(
                model_name=model_name,
                train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
                train_mae=mean_absolute_error(y_train, y_train_pred),
                train_r2=r2_score(y_train, y_train_pred),
                val_rmse=np.sqrt(mean_squared_error(y_val, y_val_pred)),
                val_mae=mean_absolute_error(y_val, y_val_pred),
                val_r2=r2_score(y_val, y_val_pred),
                train_time=train_time,
                y_val_pred=y_val_pred
            )
            fitted_models[model_name] = wrapper

        except Exception as e:
            print(f"    ⚠ Model {model_name} failed: {str(e)[:50]}")
            continue

    return results, fitted_models


def evaluate_strategy_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    split_result: SplitResult,
    strategy_key: str,
    strategy_name: str,
    category: str,
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    verbose: bool = True
) -> EnhancedStrategyResult:
    """
    Comprehensive evaluation of a splitting strategy.

    Args:
        X: Full feature matrix
        y: Full target vector
        sample_ids: Sample IDs
        split_result: Split configuration
        strategy_key: Strategy identifier
        strategy_name: Display name
        category: Strategy category
        n_repeats: Number of CV repetitions
        n_bootstrap: Bootstrap samples for CIs
        verbose: Print progress

    Returns:
        EnhancedStrategyResult with comprehensive metrics
    """
    start_time = time.time()

    fold_df = split_result.fold_assignments
    models = get_model_suite()

    # Get test data
    test_ids = split_result.test_ids
    test_mask = np.isin(sample_ids, test_ids)
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Get train data for representativeness
    train_ids = split_result.train_ids
    train_mask = np.isin(sample_ids, train_ids)
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]

    if verbose:
        print(f"\n  Test set: {len(y_test)} samples from {len(test_ids)} unique IDs")
        print(f"  Training with {len(models)} models x {n_repeats} repeats")

    n_folds = int(fold_df[fold_df['split'] != 'test']['fold'].max() + 1)
    all_fold_results = []
    all_fold_models = {model_name: [] for model_name in models.keys()}

    base_seeds = [42, 123, 456]  # Different seeds for repeats

    for repeat_idx in range(n_repeats):
        repeat_seed = base_seeds[repeat_idx % len(base_seeds)]

        if verbose:
            print(f"\n  Repeat {repeat_idx + 1}/{n_repeats} (seed={repeat_seed}):")

        for fold_idx in range(n_folds):
            # Get train and validation IDs for this fold
            train_ids_fold = fold_df[
                (fold_df['fold'] != fold_idx) & (fold_df['split'] == 'train')
            ]['ID'].values

            val_ids_fold = fold_df[
                (fold_df['fold'] == fold_idx) & (fold_df['split'] == 'train')
            ]['ID'].values

            # Get data
            train_mask_fold = np.isin(sample_ids, train_ids_fold)
            val_mask_fold = np.isin(sample_ids, val_ids_fold)

            X_train_fold = X[train_mask_fold]
            y_train_fold = y[train_mask_fold]
            X_val_fold = X[val_mask_fold]
            y_val_fold = y[val_mask_fold]

            # Train all models
            model_results, fitted_models = train_models_on_fold(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                models,
                random_state=repeat_seed + fold_idx
            )

            # Store fitted models for test prediction
            for model_name, wrapper in fitted_models.items():
                all_fold_models[model_name].append(wrapper)

            # Compute ensemble prediction (mean of all models)
            val_preds = [r.y_val_pred for r in model_results.values()]
            if val_preds:
                ensemble_pred = np.mean(val_preds, axis=0)
                ensemble_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred))
                ensemble_mae = mean_absolute_error(y_val_fold, ensemble_pred)
                ensemble_r2 = r2_score(y_val_fold, ensemble_pred)
            else:
                ensemble_pred = np.zeros_like(y_val_fold)
                ensemble_rmse = ensemble_mae = ensemble_r2 = np.nan

            fold_result = FoldResult(
                fold_idx=fold_idx,
                repeat_idx=repeat_idx,
                n_train=len(y_train_fold),
                n_val=len(y_val_fold),
                model_results=model_results,
                ensemble_val_rmse=ensemble_rmse,
                ensemble_val_mae=ensemble_mae,
                ensemble_val_r2=ensemble_r2,
                y_val_true=y_val_fold,
                y_val_pred_ensemble=ensemble_pred
            )
            all_fold_results.append(fold_result)

            if verbose:
                model_str = " | ".join([
                    f"{m[:3]}:{r.val_rmse:.3f}"
                    for m, r in list(model_results.items())[:4]
                ])
                print(f"    Fold {fold_idx + 1}: Ens={ensemble_rmse:.3f} | {model_str}")

    # Aggregate test predictions from all fold models
    if verbose:
        print(f"\n  Computing test predictions...")

    model_test_results = {}
    all_test_preds = []

    for model_name, wrappers in all_fold_models.items():
        if not wrappers:
            continue

        # Ensemble prediction from all folds x repeats
        preds = [w.predict(X_test) for w in wrappers]
        mean_pred = np.mean(preds, axis=0)

        test_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        test_mae = mean_absolute_error(y_test, mean_pred)
        test_r2 = r2_score(y_test, mean_pred)

        model_test_results[model_name] = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2
        }
        all_test_preds.append(mean_pred)

    # Final ensemble prediction (mean of all models)
    if all_test_preds:
        final_ensemble_pred = np.mean(all_test_preds, axis=0)
    else:
        final_ensemble_pred = np.zeros_like(y_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, final_ensemble_pred))
    test_mae = mean_absolute_error(y_test, final_ensemble_pred)
    test_r2 = r2_score(y_test, final_ensemble_pred)

    if verbose:
        print(f"\n  📊 Per-model test performance:")
        for model_name, results in model_test_results.items():
            print(f"     {model_name:12s}: RMSE={results['rmse']:.4f}, R²={results['r2']:.4f}")
        print(f"     {'ENSEMBLE':12s}: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

    # Aggregate CV metrics
    cv_rmses = [r.ensemble_val_rmse for r in all_fold_results if not np.isnan(r.ensemble_val_rmse)]
    cv_r2s = [r.ensemble_val_r2 for r in all_fold_results if not np.isnan(r.ensemble_val_r2)]

    cv_rmse_mean = np.mean(cv_rmses) if cv_rmses else np.nan
    cv_rmse_std = np.std(cv_rmses) if cv_rmses else np.nan
    cv_r2_mean = np.mean(cv_r2s) if cv_r2s else np.nan
    cv_r2_std = np.std(cv_r2s) if cv_r2s else np.nan

    generalization_gap = test_rmse - cv_rmse_mean

    # Bootstrap confidence intervals
    if verbose:
        print(f"\n  Computing bootstrap CIs ({n_bootstrap} samples)...")

    boot_metrics = bootstrap_metrics(
        y_test, final_ensemble_pred,
        n_bootstrap=n_bootstrap,
        random_state=42
    )

    # Representativeness metrics
    if verbose:
        print(f"  Computing representativeness metrics...")

    repr_metrics = compute_representativeness(
        X_train_full, X_test,
        y_train_full, y_test,
        n_pca_components=10
    )

    total_time = time.time() - start_time

    if verbose:
        print(f"\n  ✅ Evaluation complete in {total_time:.1f}s")
        print(f"     CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
        print(f"     Test RMSE: {test_rmse:.4f} [{boot_metrics.rmse_ci_lower:.4f}, {boot_metrics.rmse_ci_upper:.4f}]")
        print(f"     Test R²: {test_r2:.4f} [{boot_metrics.r2_ci_lower:.4f}, {boot_metrics.r2_ci_upper:.4f}]")
        print(f"     Gap (CV→Test): {generalization_gap:+.4f}")
        print(f"     Spectral coverage: {repr_metrics.spectral_coverage:.2%}")
        print(f"     Target Wasserstein: {repr_metrics.target_wasserstein:.4f}")
        print(f"     High leverage samples: {repr_metrics.n_high_leverage}")

    return EnhancedStrategyResult(
        strategy_key=strategy_key,
        strategy_name=strategy_name,
        category=category,
        fold_results=all_fold_results,
        n_repeats=n_repeats,
        n_folds=n_folds,
        model_test_results=model_test_results,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,
        n_test=len(y_test),
        y_test_true=y_test,
        y_test_pred=final_ensemble_pred,
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        generalization_gap=generalization_gap,
        bootstrap_metrics=boot_metrics,
        representativeness=repr_metrics,
        strategy_info=split_result.strategy_info,
        total_time=total_time
    )


# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_strategies_enhanced(
    results: List[EnhancedStrategyResult]
) -> pd.DataFrame:
    """
    Create comprehensive comparison DataFrame.

    Args:
        results: List of EnhancedStrategyResult

    Returns:
        DataFrame with all metrics
    """
    comparison_data = []

    for r in results:
        row = {
            'strategy_key': r.strategy_key,
            'strategy': r.strategy_name,
            'category': r.category,

            # CV metrics
            'cv_rmse_mean': r.cv_rmse_mean,
            'cv_rmse_std': r.cv_rmse_std,
            'cv_r2_mean': r.cv_r2_mean,
            'cv_r2_std': r.cv_r2_std,

            # Test metrics
            'test_rmse': r.test_rmse,
            'test_rmse_ci_lower': r.bootstrap_metrics.rmse_ci_lower,
            'test_rmse_ci_upper': r.bootstrap_metrics.rmse_ci_upper,
            'test_mae': r.test_mae,
            'test_r2': r.test_r2,
            'test_r2_ci_lower': r.bootstrap_metrics.r2_ci_lower,
            'test_r2_ci_upper': r.bootstrap_metrics.r2_ci_upper,

            # Generalization
            'n_test': r.n_test,
            'generalization_gap': r.generalization_gap,
            'cv_stability': r.cv_rmse_std / r.cv_rmse_mean if r.cv_rmse_mean > 0 else 0,

            # Representativeness
            'spectral_coverage': r.representativeness.spectral_coverage,
            'target_wasserstein': r.representativeness.target_wasserstein,
            'target_kl_divergence': r.representativeness.target_kl_divergence,
            'leverage_mean': r.representativeness.leverage_mean,
            'leverage_max': r.representativeness.leverage_max,
            'n_high_leverage': r.representativeness.n_high_leverage,
            'hotelling_t2_mean': r.representativeness.hotelling_t2_mean,

            # Timing
            'total_time_sec': r.total_time
        }

        # Add per-model results
        for model_name, model_res in r.model_test_results.items():
            row[f'{model_name}_rmse'] = model_res['rmse']
            row[f'{model_name}_r2'] = model_res['r2']

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('test_rmse')

    return df


def identify_best_strategies_enhanced(
    comparison_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Identify best strategies by multiple criteria.

    Args:
        comparison_df: Comparison DataFrame

    Returns:
        Dictionary with best strategies
    """
    best = {}

    # Best test RMSE
    idx = comparison_df['test_rmse'].idxmin()
    best['test_rmse'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'test_rmse'],
        'ci': f"[{comparison_df.loc[idx, 'test_rmse_ci_lower']:.4f}, {comparison_df.loc[idx, 'test_rmse_ci_upper']:.4f}]"
    }

    # Best test R²
    idx = comparison_df['test_r2'].idxmax()
    best['test_r2'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'test_r2'],
        'ci': f"[{comparison_df.loc[idx, 'test_r2_ci_lower']:.4f}, {comparison_df.loc[idx, 'test_r2_ci_upper']:.4f}]"
    }

    # Most stable CV
    idx = comparison_df['cv_stability'].idxmin()
    best['cv_stability'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'cv_stability']
    }

    # Best generalization
    idx = comparison_df['generalization_gap'].abs().idxmin()
    best['generalization'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'generalization_gap']
    }

    # Best spectral coverage
    idx = comparison_df['spectral_coverage'].idxmax()
    best['spectral_coverage'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'spectral_coverage']
    }

    # Best target coverage (lowest Wasserstein)
    idx = comparison_df['target_wasserstein'].idxmin()
    best['target_coverage'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'target_wasserstein']
    }

    # Lowest leverage (least extrapolation)
    idx = comparison_df['n_high_leverage'].idxmin()
    best['low_extrapolation'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'n_high_leverage']
    }

    return best


def compute_statistical_tests_enhanced(
    results: List[EnhancedStrategyResult],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute pairwise statistical tests between strategies.

    Uses all CV fold results across repeats for more statistical power.

    Args:
        results: List of EnhancedStrategyResult
        alpha: Significance level

    Returns:
        DataFrame with test results
    """
    from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

    results_sorted = sorted(results, key=lambda r: r.test_rmse)
    pairwise_results = []

    for i in range(len(results_sorted)):
        for j in range(i + 1, len(results_sorted)):
            r1 = results_sorted[i]
            r2 = results_sorted[j]

            # Get all CV RMSEs across repeats
            rmse1 = [f.ensemble_val_rmse for f in r1.fold_results if not np.isnan(f.ensemble_val_rmse)]
            rmse2 = [f.ensemble_val_rmse for f in r2.fold_results if not np.isnan(f.ensemble_val_rmse)]

            if len(rmse1) < 2 or len(rmse2) < 2:
                continue

            # T-test
            t_stat, t_pval = ttest_ind(rmse1, rmse2)

            # Mann-Whitney U
            u_stat, u_pval = mannwhitneyu(rmse1, rmse2, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(rmse1) + np.var(rmse2)) / 2)
            cohens_d = (np.mean(rmse1) - np.mean(rmse2)) / pooled_std if pooled_std > 0 else 0

            pairwise_results.append({
                'strategy_1': r1.strategy_name,
                'strategy_2': r2.strategy_name,
                'rmse_1_mean': np.mean(rmse1),
                'rmse_2_mean': np.mean(rmse2),
                'rmse_diff': np.mean(rmse1) - np.mean(rmse2),
                'cohens_d': cohens_d,
                't_pvalue': t_pval,
                't_significant': t_pval < alpha,
                'u_pvalue': u_pval,
                'u_significant': u_pval < alpha
            })

    return pd.DataFrame(pairwise_results)
