"""
Classification Splitter Evaluation Module
==========================================

This module provides comprehensive evaluation of splitting strategies for
classification tasks with:
- Multiple models (LogisticRegression, SVC, RF, XGBoost, MLP)
- StratifiedGroupKFold cross-validation (to avoid repetition leakage)
- Bootstrap confidence intervals
- Representativeness metrics (spectral coverage, class balance)

Metrics computed:
- Accuracy, F1 (macro/weighted), Precision, Recall with bootstrap 95% CIs
- Generalization gap
- Fold stability
- Spectral coverage (PCA hull volume ratio)
- Class balance analysis
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

from splitter_strategies import SplitResult

warnings.filterwarnings('ignore')

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ClassificationModelResult:
    """Results from a single model training for classification."""
    model_name: str
    train_accuracy: float
    train_f1_macro: float
    train_f1_weighted: float
    val_accuracy: float
    val_f1_macro: float
    val_f1_weighted: float
    val_precision_macro: float
    val_recall_macro: float
    val_balanced_accuracy: float
    train_time: float
    y_val_pred: np.ndarray
    y_val_proba: np.ndarray | None

@dataclass
class ClassificationFoldResult:
    """Results from training all models on a single fold."""
    fold_idx: int
    repeat_idx: int
    n_train: int
    n_val: int
    model_results: dict[str, ClassificationModelResult]
    ensemble_val_accuracy: float
    ensemble_val_f1_macro: float
    ensemble_val_f1_weighted: float
    y_val_true: np.ndarray
    y_val_pred_ensemble: np.ndarray

@dataclass
class ClassificationBootstrapMetrics:
    """Bootstrap confidence intervals for classification metrics."""
    accuracy_mean: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    f1_macro_mean: float
    f1_macro_ci_lower: float
    f1_macro_ci_upper: float
    f1_weighted_mean: float
    f1_weighted_ci_lower: float
    f1_weighted_ci_upper: float
    balanced_accuracy_mean: float
    balanced_accuracy_ci_lower: float
    balanced_accuracy_ci_upper: float

@dataclass
class ClassificationRepresentativenessMetrics:
    """Metrics for evaluating split representativeness for classification."""
    spectral_coverage: float  # Ratio of test spectral space covered by train
    train_class_distribution: dict[int, float]  # Class proportions in train
    test_class_distribution: dict[int, float]  # Class proportions in test
    class_balance_similarity: float  # How similar train/test class distributions are
    leverage_mean: float
    leverage_max: float
    n_high_leverage: int
    hotelling_t2_mean: float

@dataclass
class ClassificationStrategyResult:
    """Comprehensive results for a splitting strategy (classification)."""
    strategy_key: str
    strategy_name: str
    category: str

    # Fold results (all repeats)
    fold_results: list[ClassificationFoldResult]
    n_repeats: int
    n_folds: int

    # Per-model test results
    model_test_results: dict[str, dict[str, float]]

    # Ensemble test results
    test_accuracy: float
    test_f1_macro: float
    test_f1_weighted: float
    test_precision_macro: float
    test_recall_macro: float
    test_balanced_accuracy: float
    n_test: int
    n_classes: int
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    confusion_matrix: np.ndarray

    # CV aggregated metrics
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_macro_mean: float
    cv_f1_macro_std: float
    generalization_gap: float

    # Bootstrap CIs
    bootstrap_metrics: ClassificationBootstrapMetrics

    # Representativeness
    representativeness: ClassificationRepresentativenessMetrics

    # Strategy info
    strategy_info: dict[str, Any]

    # Timing
    total_time: float

    # Split data (for export)
    train_indices: np.ndarray = None
    test_indices: np.ndarray = None

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_classification_model_suite(n_classes: int = 2) -> dict[str, Any]:
    """
    Get the full suite of classification models.

    Args:
        n_classes: Number of classes

    Returns:
        Dictionary of model configurations
    """
    models = {
        'logistic': {
            'class': LogisticRegression,
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'multi_class': 'multinomial' if n_classes > 2 else 'auto',
                'solver': 'lbfgs',
                'random_state': 42
            },
            'category': 'linear'
        },
        'svc': {
            'class': SVC,
            'params': {
                'kernel': 'rbf',
                'C': 10.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'category': 'kernel'
        },
        'knn': {
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean'
            },
            'category': 'instance'
        },
        'rf': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'category': 'ensemble'
        },
        'mlp': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (256, 128, 64),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 32,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 300,
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
            'class': XGBClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
                'n_jobs': 1
            },
            'category': 'ensemble'
        }

    return models

class ClassificationModelWrapper:
    """Wrapper for different classification model types with scaling."""

    def __init__(self, model_name: str, model_config: dict[str, Any], random_state: int = 42):
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
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict probabilities if available."""
        if hasattr(self.model, 'predict_proba'):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        return None

# ============================================================================
# REPRESENTATIVENESS METRICS (CLASSIFICATION)
# ============================================================================

def compute_spectral_coverage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 10
) -> float:
    """
    Compute spectral coverage using PCA convex hull volume ratio.
    """
    X_all = np.vstack([X_train, X_test])
    n_comp = min(n_components, X_all.shape[0] - 1, X_all.shape[1])

    pca = PCA(n_components=n_comp)
    pca.fit(X_all)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    if n_comp >= 3:
        X_train_hull = X_train_pca[:, :3]
        X_test_hull = X_test_pca[:, :3]
    else:
        X_train_hull = X_train_pca
        X_test_hull = X_test_pca

    try:
        from scipy.spatial import Delaunay
        try:
            train_delaunay = Delaunay(X_train_hull)
            inside = train_delaunay.find_simplex(X_test_hull) >= 0
            coverage = np.mean(inside)
        except Exception:
            distances = cdist(X_test_hull, X_train_hull, 'euclidean')
            min_distances = np.min(distances, axis=1)
            train_radius = np.max(cdist(X_train_hull, X_train_hull.mean(axis=0, keepdims=True)))
            coverage = np.mean(min_distances < train_radius)

    except Exception:
        distances = cdist(X_test_pca, X_train_pca, 'euclidean')
        min_distances = np.min(distances, axis=1)
        max_train_dist = np.max(cdist(X_train_pca, X_train_pca.mean(axis=0, keepdims=True)))
        coverage = np.mean(min_distances < max_train_dist)

    return float(coverage)

def compute_class_distribution(y: np.ndarray) -> dict[int, float]:
    """Compute class distribution as proportions."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(c): float(cnt / total) for c, cnt in zip(classes, counts, strict=False)}

def compute_class_balance_similarity(
    y_train: np.ndarray,
    y_test: np.ndarray
) -> float:
    """
    Compute similarity between train and test class distributions.

    Uses 1 - total variation distance (higher is better, max 1.0).
    """
    train_dist = compute_class_distribution(y_train)
    test_dist = compute_class_distribution(y_test)

    all_classes = set(train_dist.keys()) | set(test_dist.keys())

    total_variation = 0.0
    for c in all_classes:
        p_train = train_dist.get(c, 0.0)
        p_test = test_dist.get(c, 0.0)
        total_variation += abs(p_train - p_test)

    # Total variation distance is in [0, 2], normalize to [0, 1]
    tv_normalized = total_variation / 2.0

    return 1.0 - tv_normalized

def compute_leverage_metrics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 10
) -> tuple[float, float, int, float]:
    """Compute leverage analysis for test samples."""
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    cov_train = np.cov(X_train_pca.T)
    if cov_train.ndim == 0:
        cov_train = np.array([[cov_train]])

    try:
        cov_inv = np.linalg.inv(cov_train)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_train)

    train_mean = X_train_pca.mean(axis=0)
    X_test_centered = X_test_pca - train_mean

    t2_scores = np.array([
        x @ cov_inv @ x.T for x in X_test_centered
    ])

    n = X_train_pca.shape[0]
    leverage = 1/n + t2_scores / (n - 1)

    p = n_comp
    high_leverage_threshold = 2 * (p + 1) / n
    n_high = int(np.sum(leverage > high_leverage_threshold))

    return (
        float(np.mean(leverage)),
        float(np.max(leverage)),
        n_high,
        float(np.mean(t2_scores))
    )

def compute_representativeness_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_pca_components: int = 10
) -> ClassificationRepresentativenessMetrics:
    """Compute all representativeness metrics for classification."""
    spectral_coverage = compute_spectral_coverage(X_train, X_test, n_pca_components)
    train_dist = compute_class_distribution(y_train)
    test_dist = compute_class_distribution(y_test)
    balance_sim = compute_class_balance_similarity(y_train, y_test)
    lev_mean, lev_max, n_high, t2_mean = compute_leverage_metrics(
        X_train, X_test, n_pca_components
    )

    return ClassificationRepresentativenessMetrics(
        spectral_coverage=spectral_coverage,
        train_class_distribution=train_dist,
        test_class_distribution=test_dist,
        class_balance_similarity=balance_sim,
        leverage_mean=lev_mean,
        leverage_max=lev_max,
        n_high_leverage=n_high,
        hotelling_t2_mean=t2_mean
    )

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> ClassificationBootstrapMetrics:
    """Compute bootstrap confidence intervals for classification metrics."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    accuracy_samples = []
    f1_macro_samples = []
    f1_weighted_samples = []
    balanced_accuracy_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Ensure all classes are represented
        if len(np.unique(y_true_boot)) < 2:
            continue

        accuracy_samples.append(accuracy_score(y_true_boot, y_pred_boot))
        f1_macro_samples.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        f1_weighted_samples.append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        balanced_accuracy_samples.append(balanced_accuracy_score(y_true_boot, y_pred_boot))

    alpha = 1 - confidence

    def get_ci(samples):
        if len(samples) == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(samples)),
            float(np.percentile(samples, 100 * alpha / 2)),
            float(np.percentile(samples, 100 * (1 - alpha / 2)))
        )

    acc_mean, acc_lo, acc_hi = get_ci(accuracy_samples)
    f1m_mean, f1m_lo, f1m_hi = get_ci(f1_macro_samples)
    f1w_mean, f1w_lo, f1w_hi = get_ci(f1_weighted_samples)
    ba_mean, ba_lo, ba_hi = get_ci(balanced_accuracy_samples)

    return ClassificationBootstrapMetrics(
        accuracy_mean=acc_mean,
        accuracy_ci_lower=acc_lo,
        accuracy_ci_upper=acc_hi,
        f1_macro_mean=f1m_mean,
        f1_macro_ci_lower=f1m_lo,
        f1_macro_ci_upper=f1m_hi,
        f1_weighted_mean=f1w_mean,
        f1_weighted_ci_lower=f1w_lo,
        f1_weighted_ci_upper=f1w_hi,
        balanced_accuracy_mean=ba_mean,
        balanced_accuracy_ci_lower=ba_lo,
        balanced_accuracy_ci_upper=ba_hi
    )

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_classification_models_on_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    models: dict[str, Any],
    random_state: int = 42
) -> tuple[dict[str, ClassificationModelResult], dict[str, ClassificationModelWrapper]]:
    """Train all classification models on a single fold."""
    results = {}
    fitted_models = {}

    for model_name, model_config in models.items():
        wrapper = ClassificationModelWrapper(model_name, model_config, random_state)

        start_time = time.time()
        try:
            wrapper.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_train_pred = wrapper.predict(X_train)
            y_val_pred = wrapper.predict(X_val)
            y_val_proba = wrapper.predict_proba(X_val)

            results[model_name] = ClassificationModelResult(
                model_name=model_name,
                train_accuracy=accuracy_score(y_train, y_train_pred),
                train_f1_macro=f1_score(y_train, y_train_pred, average='macro', zero_division=0),
                train_f1_weighted=f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                val_accuracy=accuracy_score(y_val, y_val_pred),
                val_f1_macro=f1_score(y_val, y_val_pred, average='macro', zero_division=0),
                val_f1_weighted=f1_score(y_val, y_val_pred, average='weighted', zero_division=0),
                val_precision_macro=precision_score(y_val, y_val_pred, average='macro', zero_division=0),
                val_recall_macro=recall_score(y_val, y_val_pred, average='macro', zero_division=0),
                val_balanced_accuracy=balanced_accuracy_score(y_val, y_val_pred),
                train_time=train_time,
                y_val_pred=y_val_pred,
                y_val_proba=y_val_proba
            )
            fitted_models[model_name] = wrapper

        except Exception as e:
            print(f"    ⚠ Model {model_name} failed: {str(e)[:50]}")
            continue

    return results, fitted_models

def evaluate_classification_strategy(
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
) -> ClassificationStrategyResult:
    """
    Comprehensive evaluation of a splitting strategy for classification.

    Uses StratifiedGroupKFold internally to prevent repetition leakage.
    """
    start_time = time.time()

    fold_df = split_result.fold_assignments
    n_classes = len(np.unique(y))
    models = get_classification_model_suite(n_classes)

    # Get test data
    test_ids = split_result.test_ids
    test_mask = np.isin(sample_ids, test_ids)
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_indices = np.where(test_mask)[0]

    # Get train data
    train_ids = split_result.train_ids
    train_mask = np.isin(sample_ids, train_ids)
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    sample_ids_train = sample_ids[train_mask]
    train_indices = np.where(train_mask)[0]

    if verbose:
        print(f"\n  Test set: {len(y_test)} samples from {len(test_ids)} unique IDs")
        print(f"  Train set: {len(y_train_full)} samples from {len(train_ids)} unique IDs")
        print(f"  Classes: {n_classes} ({np.unique(y)})")
        print(f"  Training with {len(models)} models x {n_repeats} repeats")

    n_folds = split_result.strategy_info.get('n_folds', 3)
    all_fold_results = []
    all_fold_models = {model_name: [] for model_name in models}

    base_seeds = [42, 123, 456, 789, 1011]

    for repeat_idx in range(n_repeats):
        repeat_seed = base_seeds[repeat_idx % len(base_seeds)]

        if verbose:
            print(f"\n  Repeat {repeat_idx + 1}/{n_repeats} (seed={repeat_seed}):")

        # Use StratifiedGroupKFold for CV within training set
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_train_full, y_train_full, sample_ids_train)):
            X_train_fold = X_train_full[train_idx]
            y_train_fold = y_train_full[train_idx]
            X_val_fold = X_train_full[val_idx]
            y_val_fold = y_train_full[val_idx]

            # Train all models
            model_results, fitted_models = train_classification_models_on_fold(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                models,
                random_state=repeat_seed + fold_idx
            )

            # Store fitted models
            for model_name, wrapper in fitted_models.items():
                all_fold_models[model_name].append(wrapper)

            # Compute ensemble prediction (majority vote)
            val_preds = [r.y_val_pred for r in model_results.values()]
            if val_preds:
                # Stack predictions and take mode (majority vote)
                stacked_preds = np.stack(val_preds, axis=0)
                ensemble_pred = stats.mode(stacked_preds, axis=0, keepdims=False)[0]
                ensemble_accuracy = accuracy_score(y_val_fold, ensemble_pred)
                ensemble_f1_macro = f1_score(y_val_fold, ensemble_pred, average='macro', zero_division=0)
                ensemble_f1_weighted = f1_score(y_val_fold, ensemble_pred, average='weighted', zero_division=0)
            else:
                ensemble_pred = np.zeros_like(y_val_fold)
                ensemble_accuracy = ensemble_f1_macro = ensemble_f1_weighted = np.nan

            fold_result = ClassificationFoldResult(
                fold_idx=fold_idx,
                repeat_idx=repeat_idx,
                n_train=len(y_train_fold),
                n_val=len(y_val_fold),
                model_results=model_results,
                ensemble_val_accuracy=ensemble_accuracy,
                ensemble_val_f1_macro=ensemble_f1_macro,
                ensemble_val_f1_weighted=ensemble_f1_weighted,
                y_val_true=y_val_fold,
                y_val_pred_ensemble=ensemble_pred
            )
            all_fold_results.append(fold_result)

            if verbose:
                model_str = " | ".join([
                    f"{m[:3]}:{r.val_accuracy:.3f}"
                    for m, r in list(model_results.items())[:4]
                ])
                print(f"    Fold {fold_idx + 1}: Ens={ensemble_accuracy:.3f} | {model_str}")


    # Aggregate test predictions
    if verbose:
        print("\n  Computing test predictions...")

    model_test_results = {}
    all_test_preds = []

    for model_name, wrappers in all_fold_models.items():
        if not wrappers:
            continue

        # Collect predictions from all models (majority vote)
        preds = np.stack([w.predict(X_test) for w in wrappers], axis=0)
        mean_pred = stats.mode(preds, axis=0, keepdims=False)[0]

        test_accuracy = accuracy_score(y_test, mean_pred)
        test_f1_macro = f1_score(y_test, mean_pred, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_test, mean_pred, average='weighted', zero_division=0)
        test_balanced_accuracy = balanced_accuracy_score(y_test, mean_pred)

        model_test_results[model_name] = {
            'accuracy': test_accuracy,
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted,
            'balanced_accuracy': test_balanced_accuracy
        }
        all_test_preds.append(mean_pred)

    # Final ensemble prediction
    if all_test_preds:
        stacked_test = np.stack(all_test_preds, axis=0)
        final_ensemble_pred = stats.mode(stacked_test, axis=0, keepdims=False)[0]
    else:
        final_ensemble_pred = np.zeros_like(y_test)

    test_accuracy = accuracy_score(y_test, final_ensemble_pred)
    test_f1_macro = f1_score(y_test, final_ensemble_pred, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_test, final_ensemble_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_test, final_ensemble_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test, final_ensemble_pred, average='macro', zero_division=0)
    test_balanced_accuracy = balanced_accuracy_score(y_test, final_ensemble_pred)
    conf_matrix = confusion_matrix(y_test, final_ensemble_pred)

    if verbose:
        print("\n  📊 Per-model test performance:")
        for model_name, results in model_test_results.items():
            print(f"     {model_name:12s}: Acc={results['accuracy']:.4f}, F1m={results['f1_macro']:.4f}")
        print(f"     {'ENSEMBLE':12s}: Acc={test_accuracy:.4f}, F1m={test_f1_macro:.4f}")

    # Aggregate CV metrics
    cv_accuracies = [r.ensemble_val_accuracy for r in all_fold_results if not np.isnan(r.ensemble_val_accuracy)]
    cv_f1_macros = [r.ensemble_val_f1_macro for r in all_fold_results if not np.isnan(r.ensemble_val_f1_macro)]

    cv_accuracy_mean = np.mean(cv_accuracies) if cv_accuracies else np.nan
    cv_accuracy_std = np.std(cv_accuracies) if cv_accuracies else np.nan
    cv_f1_macro_mean = np.mean(cv_f1_macros) if cv_f1_macros else np.nan
    cv_f1_macro_std = np.std(cv_f1_macros) if cv_f1_macros else np.nan

    generalization_gap = test_accuracy - cv_accuracy_mean

    # Bootstrap confidence intervals
    if verbose:
        print(f"\n  Computing bootstrap CIs ({n_bootstrap} samples)...")

    boot_metrics = bootstrap_classification_metrics(
        y_test, final_ensemble_pred,
        n_bootstrap=n_bootstrap,
        random_state=42
    )

    # Representativeness metrics
    if verbose:
        print("  Computing representativeness metrics...")

    repr_metrics = compute_representativeness_classification(
        X_train_full, X_test,
        y_train_full, y_test,
        n_pca_components=10
    )

    total_time = time.time() - start_time

    if verbose:
        print(f"\n  ✅ Evaluation complete in {total_time:.1f}s")
        print(f"     CV Accuracy: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
        print(f"     Test Accuracy: {test_accuracy:.4f} [{boot_metrics.accuracy_ci_lower:.4f}, {boot_metrics.accuracy_ci_upper:.4f}]")
        print(f"     Test F1 (macro): {test_f1_macro:.4f} [{boot_metrics.f1_macro_ci_lower:.4f}, {boot_metrics.f1_macro_ci_upper:.4f}]")
        print(f"     Gap (CV→Test): {generalization_gap:+.4f}")
        print(f"     Spectral coverage: {repr_metrics.spectral_coverage:.2%}")
        print(f"     Class balance similarity: {repr_metrics.class_balance_similarity:.2%}")

    return ClassificationStrategyResult(
        strategy_key=strategy_key,
        strategy_name=strategy_name,
        category=category,
        fold_results=all_fold_results,
        n_repeats=n_repeats,
        n_folds=n_folds,
        model_test_results=model_test_results,
        test_accuracy=test_accuracy,
        test_f1_macro=test_f1_macro,
        test_f1_weighted=test_f1_weighted,
        test_precision_macro=test_precision_macro,
        test_recall_macro=test_recall_macro,
        test_balanced_accuracy=test_balanced_accuracy,
        n_test=len(y_test),
        n_classes=n_classes,
        y_test_true=y_test,
        y_test_pred=final_ensemble_pred,
        confusion_matrix=conf_matrix,
        cv_accuracy_mean=cv_accuracy_mean,
        cv_accuracy_std=cv_accuracy_std,
        cv_f1_macro_mean=cv_f1_macro_mean,
        cv_f1_macro_std=cv_f1_macro_std,
        generalization_gap=generalization_gap,
        bootstrap_metrics=boot_metrics,
        representativeness=repr_metrics,
        strategy_info=split_result.strategy_info,
        total_time=total_time,
        train_indices=train_indices,
        test_indices=test_indices
    )

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_classification_strategies(
    results: list[ClassificationStrategyResult]
) -> pd.DataFrame:
    """Create comprehensive comparison DataFrame for classification."""
    comparison_data = []

    for r in results:
        row = {
            'strategy_key': r.strategy_key,
            'strategy': r.strategy_name,
            'category': r.category,

            # CV metrics
            'cv_accuracy_mean': r.cv_accuracy_mean,
            'cv_accuracy_std': r.cv_accuracy_std,
            'cv_f1_macro_mean': r.cv_f1_macro_mean,
            'cv_f1_macro_std': r.cv_f1_macro_std,

            # Test metrics
            'test_accuracy': r.test_accuracy,
            'test_accuracy_ci_lower': r.bootstrap_metrics.accuracy_ci_lower,
            'test_accuracy_ci_upper': r.bootstrap_metrics.accuracy_ci_upper,
            'test_f1_macro': r.test_f1_macro,
            'test_f1_macro_ci_lower': r.bootstrap_metrics.f1_macro_ci_lower,
            'test_f1_macro_ci_upper': r.bootstrap_metrics.f1_macro_ci_upper,
            'test_f1_weighted': r.test_f1_weighted,
            'test_precision_macro': r.test_precision_macro,
            'test_recall_macro': r.test_recall_macro,
            'test_balanced_accuracy': r.test_balanced_accuracy,

            # Generalization
            'n_test': r.n_test,
            'n_classes': r.n_classes,
            'generalization_gap': r.generalization_gap,
            'cv_stability': r.cv_accuracy_std / r.cv_accuracy_mean if r.cv_accuracy_mean > 0 else 0,

            # Representativeness
            'spectral_coverage': r.representativeness.spectral_coverage,
            'class_balance_similarity': r.representativeness.class_balance_similarity,
            'leverage_mean': r.representativeness.leverage_mean,
            'leverage_max': r.representativeness.leverage_max,
            'n_high_leverage': r.representativeness.n_high_leverage,
            'hotelling_t2_mean': r.representativeness.hotelling_t2_mean,

            # Timing
            'total_time_sec': r.total_time
        }

        # Add per-model results
        for model_name, model_res in r.model_test_results.items():
            row[f'{model_name}_accuracy'] = model_res['accuracy']
            row[f'{model_name}_f1_macro'] = model_res['f1_macro']

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    # Sort by test accuracy (descending for classification)
    df = df.sort_values('test_accuracy', ascending=False)

    return df

def identify_best_classification_strategies(
    comparison_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Identify best strategies by multiple criteria for classification."""
    best = {}

    # Best test accuracy
    idx = comparison_df['test_accuracy'].idxmax()
    best['test_accuracy'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'test_accuracy'],
        'ci': f"[{comparison_df.loc[idx, 'test_accuracy_ci_lower']:.4f}, {comparison_df.loc[idx, 'test_accuracy_ci_upper']:.4f}]"
    }

    # Best test F1 macro
    idx = comparison_df['test_f1_macro'].idxmax()
    best['test_f1_macro'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'test_f1_macro'],
        'ci': f"[{comparison_df.loc[idx, 'test_f1_macro_ci_lower']:.4f}, {comparison_df.loc[idx, 'test_f1_macro_ci_upper']:.4f}]"
    }

    # Best balanced accuracy
    idx = comparison_df['test_balanced_accuracy'].idxmax()
    best['balanced_accuracy'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'test_balanced_accuracy']
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

    # Best class balance
    idx = comparison_df['class_balance_similarity'].idxmax()
    best['class_balance'] = {
        'strategy': comparison_df.loc[idx, 'strategy'],
        'value': comparison_df.loc[idx, 'class_balance_similarity']
    }

    return best
