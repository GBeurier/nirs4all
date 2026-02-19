"""
Evaluator module - Generic functions for calculating metrics

This module provides:
- eval(y_true, y_pred, metric): Calculate a specific metric
- eval_multi(y_true, y_pred, task_type): Calculate all metrics for a task type

Supports regression, binary classification, and multiclass classification metrics
using sklearn, scipy, and other standard libraries.
"""

import contextlib
import warnings

import numpy as np

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn import metrics as sklearn_metrics
    from sklearn.metrics import (
        # Classification metrics
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        explained_variance_score,
        f1_score,
        hamming_loss,
        jaccard_score,
        log_loss,
        matthews_corrcoef,
        max_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        # Regression metrics
        mean_squared_error,
        median_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        # Multi-label/multi-class specific
        top_k_accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Metric abbreviation mapping: full name -> abbreviated name
METRIC_ABBREVIATIONS = {
    # Regression metrics
    'mean_squared_error': 'MSE',
    'mse': 'MSE',
    'root_mean_squared_error': 'RMSE',
    'rmse': 'RMSE',
    'mean_absolute_error': 'MAE',
    'mae': 'MAE',
    'mean_absolute_percentage_error': 'MAPE',
    'mape': 'MAPE',
    'r2_score': 'R²',
    'r2': 'R²',
    'explained_variance': 'ExpVar',
    'explained_variance_score': 'ExpVar',
    'max_error': 'MaxErr',
    'median_absolute_error': 'MedAE',
    'median_ae': 'MedAE',
    'bias': 'Bias',
    'sep': 'SEP',
    'rpd': 'RPD',
    'consistency': 'Cons',
    'nrmse': 'NRMSE',
    'nmse': 'NMSE',
    'nmae': 'NMAE',
    'pearson_r': 'Pearson',
    'spearman_r': 'Spearman',
    # Classification metrics
    'accuracy': 'Acc',
    'balanced_accuracy': 'BalAcc',
    'precision': 'Prec',
    'balanced_precision': 'BalPrec',
    'recall': 'Rec',
    'balanced_recall': 'BalRec',
    'f1': 'F1',
    'f1_score': 'F1',
    'f1_micro': 'F1µ',
    'f1_macro': 'F1M',
    'precision_micro': 'Precµ',
    'precision_macro': 'PrecM',
    'recall_micro': 'Recµ',
    'recall_macro': 'RecM',
    'specificity': 'Spec',
    'roc_auc': 'AUC',
    'auc': 'AUC',
    'log_loss': 'LogLoss',
    'matthews_corrcoef': 'MCC',
    'mcc': 'MCC',
    'cohen_kappa': 'Kappa',
    'jaccard': 'Jaccard',
    'jaccard_score': 'Jaccard',
    'hamming_loss': 'Hamming',
}

def abbreviate_metric(metric: str) -> str:
    """Convert metric name to abbreviated form.

    Args:
        metric: Full metric name (e.g., 'balanced_accuracy').

    Returns:
        Abbreviated metric name (e.g., 'BalAcc').
    """
    return METRIC_ABBREVIATIONS.get(metric.lower(), metric)

def eval(y_true: np.ndarray, y_pred: np.ndarray, metric: str | list[str]) -> float | dict[str, float]:
    """
    Calculate a specific metric for given predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        metric: Metric name (e.g., 'mse', 'accuracy', 'f1', 'r2'), or list of metric names

    Returns:
        float: Calculated metric value (single metric), or dict of metric values (list of metrics)

    Raises:
        ValueError: If metric is not supported or calculation fails
    """
    if isinstance(metric, list):
        return {m: _eval_single(y_true, y_pred, m) for m in metric}
    return _eval_single(y_true, y_pred, metric)


def _eval_single(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Calculate a single metric for given predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        metric: Metric name (e.g., 'mse', 'accuracy', 'f1', 'r2')

    Returns:
        Calculated metric value

    Raises:
        ValueError: If metric is not supported or calculation fails
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for metric calculations")

    # Ensure arrays are numpy arrays and flattened
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0 or len(y_pred) == 0:
        return float('nan')

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")

    metric = metric.lower()

    try:
        # Regression metrics
        if metric in ['mse', 'mean_squared_error']:
            return float(mean_squared_error(y_true, y_pred))
        elif metric in ['rmse', 'root_mean_squared_error']:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric in ['mae', 'mean_absolute_error']:
            return float(mean_absolute_error(y_true, y_pred))
        elif metric in ['mape', 'mean_absolute_percentage_error']:
            return float(mean_absolute_percentage_error(y_true, y_pred))
        elif metric in ['r2', 'r2_score']:
            return float(r2_score(y_true, y_pred))
        elif metric in ['explained_variance', 'explained_variance_score']:
            return float(explained_variance_score(y_true, y_pred))
        elif metric in ['max_error']:
            return float(max_error(y_true, y_pred))
        elif metric in ['median_ae', 'median_absolute_error']:
            return float(median_absolute_error(y_true, y_pred))

        # Classification metrics
        elif metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_score',
                        'precision_micro', 'recall_micro', 'f1_micro',
                        'precision_macro', 'recall_macro', 'f1_macro',
                        'balanced_accuracy', 'balanced_precision', 'balanced_recall',
                        'matthews_corrcoef', 'mcc',
                        'cohen_kappa', 'jaccard', 'jaccard_score', 'hamming_loss', 'specificity']:

            y_pred_labels = y_pred
            # Auto-convert probabilities to labels for binary classification
            if len(np.unique(y_true)) == 2 and np.issubdtype(y_pred.dtype, np.floating):
                 unique_vals = np.unique(y_pred)
                 if np.min(y_pred) >= 0 and np.max(y_pred) <= 1 and \
                   not (len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0]))):
                    y_pred_labels = (y_pred > 0.5).astype(int)

            if metric in ['accuracy']:
                return float(accuracy_score(y_true, y_pred_labels))
            elif metric in ['precision']:
                return float(precision_score(y_true, y_pred_labels, average='weighted', zero_division=0))
            elif metric in ['recall']:
                return float(recall_score(y_true, y_pred_labels, average='weighted', zero_division=0))
            elif metric in ['f1', 'f1_score']:
                return float(f1_score(y_true, y_pred_labels, average='weighted', zero_division=0))
            elif metric in ['precision_micro']:
                return float(precision_score(y_true, y_pred_labels, average='micro', zero_division=0))
            elif metric in ['recall_micro']:
                return float(recall_score(y_true, y_pred_labels, average='micro', zero_division=0))
            elif metric in ['f1_micro']:
                return float(f1_score(y_true, y_pred_labels, average='micro', zero_division=0))
            elif metric in ['precision_macro', 'balanced_precision']:
                return float(precision_score(y_true, y_pred_labels, average='macro', zero_division=0))
            elif metric in ['recall_macro', 'balanced_recall']:
                return float(recall_score(y_true, y_pred_labels, average='macro', zero_division=0))
            elif metric in ['f1_macro']:
                return float(f1_score(y_true, y_pred_labels, average='macro', zero_division=0))
            elif metric in ['balanced_accuracy']:
                return float(balanced_accuracy_score(y_true, y_pred_labels))
            elif metric in ['matthews_corrcoef', 'mcc']:
                return float(matthews_corrcoef(y_true, y_pred_labels))
            elif metric in ['cohen_kappa']:
                return float(cohen_kappa_score(y_true, y_pred_labels))
            elif metric in ['jaccard', 'jaccard_score']:
                return float(jaccard_score(y_true, y_pred_labels, average='weighted', zero_division=0))
            elif metric in ['hamming_loss']:
                return float(hamming_loss(y_true, y_pred_labels))
            else:  # metric == 'specificity'
                if len(np.unique(y_true)) == 2:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
                    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    # For multiclass, calculate macro-averaged specificity
                    cm = confusion_matrix(y_true, y_pred_labels)
                    specificities = []
                    for i in range(cm.shape[0]):
                        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                        fp = np.sum(cm[:, i]) - cm[i, i]
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        specificities.append(specificity)
                    return float(np.mean(specificities))

        elif metric in ['roc_auc', 'auc']:
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                return float(roc_auc_score(y_true, y_pred))
            else:
                return float(roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted'))
        elif metric in ['log_loss']:
            # Convert to probabilities if needed
            if np.all(np.isin(y_pred, [0, 1])):
                # Binary predictions, convert to probabilities
                y_pred_proba = np.column_stack([1 - y_pred, y_pred])
                return float(log_loss(y_true, y_pred_proba))
            else:
                return float(log_loss(y_true, y_pred))

        # Additional regression metrics with scipy
        elif metric == 'pearson_r' and SCIPY_AVAILABLE:
            correlation, _ = stats.pearsonr(y_true, y_pred)
            return float(correlation)
        elif metric == 'spearman_r' and SCIPY_AVAILABLE:
            correlation, _ = stats.spearmanr(y_true, y_pred)
            return float(correlation)

        # Custom metrics
        elif metric == 'bias':
            return float(np.mean(y_pred - y_true))
        elif metric == 'sep':  # Standard Error of Prediction
            return float(np.std(y_pred - y_true))
        elif metric == 'rpd':  # Ratio of Performance to Deviation
            sep = float(np.std(y_pred - y_true))
            sd = float(np.std(y_true))
            return sd / sep if sep != 0 else float('inf')
        elif metric == 'consistency':
            # Consistency: 1 - (RMSE / std(y_true))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            sd = float(np.std(y_true))
            return 1 - (rmse / sd) if sd != 0 else 0.0
        elif metric == 'nrmse':
            # Normalized RMSE: RMSE / (max - min)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            y_range = float(np.max(y_true) - np.min(y_true))
            return rmse / y_range if y_range != 0 else float('inf')
        elif metric == 'nmse':
            # Normalized MSE: MSE / var(y_true)
            mse = float(mean_squared_error(y_true, y_pred))
            var = float(np.var(y_true))
            return mse / var if var != 0 else float('inf')
        elif metric == 'nmae':
            # Normalized MAE: MAE / (max - min)
            mae = float(mean_absolute_error(y_true, y_pred))
            y_range = float(np.max(y_true) - np.min(y_true))
            return mae / y_range if y_range != 0 else float('inf')
        elif metric == 'specificity':
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                # For multiclass, calculate macro-averaged specificity
                cm = confusion_matrix(y_true, y_pred)
                specificities = []
                for i in range(cm.shape[0]):
                    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    specificities.append(specificity)
                return float(np.mean(specificities))

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    except Exception as e:
        raise ValueError(f"Error calculating {metric}: {str(e)}") from e

def eval_multi(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    """
    Calculate all relevant metrics for a given task type.

    Args:
        y_true: True target values
        y_pred: Predicted values
        task_type: Type of task ('regression', 'binary_classification', 'multiclass_classification')

    Returns:
        Dict[str, float]: Dictionary of metric names and their values

    Raises:
        ValueError: If task_type is not supported
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for metric calculations")

    # Ensure arrays are numpy arrays and flattened
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")

    task_type = task_type.lower()
    metrics = {}

    try:
        if task_type == 'regression':
            # Core regression metrics
            metrics['mse'] = _eval_single(y_true, y_pred, 'mse')
            metrics['rmse'] = _eval_single(y_true, y_pred, 'rmse')
            metrics['mae'] = _eval_single(y_true, y_pred, 'mae')
            metrics['r2'] = _eval_single(y_true, y_pred, 'r2')

            # Additional regression metrics
            with contextlib.suppress(Exception):
                metrics['mape'] = _eval_single(y_true, y_pred, 'mape')

            with contextlib.suppress(Exception):
                metrics['explained_variance'] = _eval_single(y_true, y_pred, 'explained_variance')

            with contextlib.suppress(Exception):
                metrics['max_error'] = _eval_single(y_true, y_pred, 'max_error')

            with contextlib.suppress(Exception):
                metrics['median_ae'] = _eval_single(y_true, y_pred, 'median_ae')

            # Custom regression metrics
            try:
                metrics['bias'] = _eval_single(y_true, y_pred, 'bias')
                metrics['sep'] = _eval_single(y_true, y_pred, 'sep')
                metrics['rpd'] = _eval_single(y_true, y_pred, 'rpd')
            except Exception:
                pass

            # Correlation metrics (if scipy available)
            if SCIPY_AVAILABLE:
                try:
                    metrics['pearson_r'] = _eval_single(y_true, y_pred, 'pearson_r')
                    metrics['spearman_r'] = _eval_single(y_true, y_pred, 'spearman_r')
                except Exception:
                    pass

        elif task_type == 'binary_classification':
            # Check if predictions are probabilities (continuous in [0,1])
            # and convert to labels for metrics that require discrete classes
            y_pred_labels = y_pred
            if np.issubdtype(y_pred.dtype, np.floating):
                # Check if values are probabilities (0-1) but not just 0.0 and 1.0
                unique_vals = np.unique(y_pred)
                if np.min(y_pred) >= 0 and np.max(y_pred) <= 1 and \
                   not (len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0]))):
                    y_pred_labels = (y_pred > 0.5).astype(int)

            # Core classification metrics
            metrics['accuracy'] = _eval_single(y_true, y_pred_labels, 'accuracy')
            metrics['balanced_accuracy'] = _eval_single(y_true, y_pred_labels, 'balanced_accuracy')
            metrics['precision'] = _eval_single(y_true, y_pred_labels, 'precision')
            metrics['balanced_precision'] = _eval_single(y_true, y_pred_labels, 'balanced_precision')
            metrics['recall'] = _eval_single(y_true, y_pred_labels, 'recall')
            metrics['balanced_recall'] = _eval_single(y_true, y_pred_labels, 'balanced_recall')
            metrics['f1'] = _eval_single(y_true, y_pred_labels, 'f1')
            metrics['specificity'] = _eval_single(y_true, y_pred_labels, 'specificity')

            # Binary-specific metrics
            with contextlib.suppress(Exception):
                metrics['roc_auc'] = _eval_single(y_true, y_pred, 'roc_auc')

            with contextlib.suppress(Exception):
                metrics['matthews_corrcoef'] = _eval_single(y_true, y_pred_labels, 'matthews_corrcoef')

            with contextlib.suppress(Exception):
                metrics['cohen_kappa'] = _eval_single(y_true, y_pred_labels, 'cohen_kappa')

            with contextlib.suppress(Exception):
                metrics['jaccard'] = _eval_single(y_true, y_pred_labels, 'jaccard')

        elif task_type == 'multiclass_classification':
            # Core classification metrics
            metrics['accuracy'] = _eval_single(y_true, y_pred, 'accuracy')
            metrics['balanced_accuracy'] = _eval_single(y_true, y_pred, 'balanced_accuracy')

            # Weighted averages (default for multiclass)
            metrics['precision'] = _eval_single(y_true, y_pred, 'precision')
            metrics['balanced_precision'] = _eval_single(y_true, y_pred, 'balanced_precision')
            metrics['recall'] = _eval_single(y_true, y_pred, 'recall')
            metrics['balanced_recall'] = _eval_single(y_true, y_pred, 'balanced_recall')
            metrics['f1'] = _eval_single(y_true, y_pred, 'f1')
            metrics['specificity'] = _eval_single(y_true, y_pred, 'specificity')

            # Micro averages
            try:
                metrics['precision_micro'] = _eval_single(y_true, y_pred, 'precision_micro')
                metrics['recall_micro'] = _eval_single(y_true, y_pred, 'recall_micro')
                metrics['f1_micro'] = _eval_single(y_true, y_pred, 'f1_micro')
            except Exception:
                pass

            # Macro averages
            try:
                metrics['precision_macro'] = _eval_single(y_true, y_pred, 'precision_macro')
                metrics['recall_macro'] = _eval_single(y_true, y_pred, 'recall_macro')
                metrics['f1_macro'] = _eval_single(y_true, y_pred, 'f1_macro')
            except Exception:
                pass

            # Multiclass-specific metrics
            with contextlib.suppress(Exception):
                metrics['roc_auc'] = _eval_single(y_true, y_pred, 'roc_auc')

            with contextlib.suppress(Exception):
                metrics['matthews_corrcoef'] = _eval_single(y_true, y_pred, 'matthews_corrcoef')

            with contextlib.suppress(Exception):
                metrics['cohen_kappa'] = _eval_single(y_true, y_pred, 'cohen_kappa')

            with contextlib.suppress(Exception):
                metrics['jaccard'] = _eval_single(y_true, y_pred, 'jaccard')

            with contextlib.suppress(Exception):
                metrics['hamming_loss'] = _eval_single(y_true, y_pred, 'hamming_loss')

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression', 'binary_classification', or 'multiclass_classification'")

    except Exception as e:
        raise ValueError(f"Error calculating metrics for {task_type}: {str(e)}") from e

    return metrics

def get_stats(y: np.ndarray) -> dict[str, float]:
    """
    Calculate descriptive statistics for target values.

    Args:
        y: Target values

    Returns:
        Dict[str, float]: Dictionary of statistical measures

    Example:
        stats = get_stats(y_true)
        # Returns: {'nsample': 100, 'mean': 2.5, 'median': 2.4, 'min': 0.1, 'max': 5.0, 'sd': 1.2, 'cv': 0.48}
    """
    y = np.asarray(y).flatten()
    y_clean = y[~np.isnan(y)]  # Remove NaN values

    if len(y_clean) == 0:
        return {
            'nsample': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sd': 0.0,
            'cv': 0.0
        }

    result_stats = {
        'nsample': len(y_clean),
        'mean': float(np.mean(y_clean)),
        'median': float(np.median(y_clean)),
        'min': float(np.min(y_clean)),
        'max': float(np.max(y_clean)),
        'sd': float(np.std(y_clean)),
    }

    # Calculate coefficient of variation
    if result_stats['mean'] != 0:
        result_stats['cv'] = result_stats['sd'] / result_stats['mean']
    else:
        result_stats['cv'] = 0.0

    return result_stats

def eval_list(y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str]) -> list[float | None]:
    """
    Calculate multiple metrics and return their scores as a list.

    Args:
        y_true: True target values
        y_pred: Predicted values
        metrics: List of metric names to calculate

    Returns:
        list: List of calculated metric values in the same order as input metrics

    Example:
        scores = eval_list(y_true, y_pred, ['mse', 'r2', 'mae'])
        # Returns: [0.022, 0.989, 0.14]
    """
    if not isinstance(metrics, (list, tuple)):
        raise ValueError("metrics must be a list or tuple of metric names")

    scores: list[float | None] = []
    for metric in metrics:
        try:
            score = _eval_single(y_true, y_pred, metric)
            scores.append(score)
        except Exception as e:
            # Handle individual metric failures gracefully
            logger.warning(f"Failed to calculate {metric}: {str(e)}")
            scores.append(None)

    return scores

def get_available_metrics(task_type: str) -> list:
    """
    Get list of available metrics for a given task type.

    Args:
        task_type: Type of task ('regression', 'binary_classification', 'multiclass_classification')

    Returns:
        List of available metric names
    """
    if task_type.lower() == 'regression':
        metrics = ['mse', 'rmse', 'mae', 'r2', 'mape', 'explained_variance',
                  'max_error', 'median_ae', 'bias', 'sep', 'rpd']
        if SCIPY_AVAILABLE:
            metrics.extend(['pearson_r', 'spearman_r'])
        return metrics

    elif task_type.lower() == 'binary_classification':
        return ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision',
                'recall', 'balanced_recall', 'f1', 'specificity', 'roc_auc',
                'matthews_corrcoef', 'cohen_kappa', 'jaccard']

    elif task_type.lower() == 'multiclass_classification':
        return ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision',
                'recall', 'balanced_recall', 'f1', 'specificity',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_macro', 'recall_macro', 'f1_macro',
                'roc_auc', 'matthews_corrcoef',
                'cohen_kappa', 'jaccard', 'hamming_loss']

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

def get_default_metrics(task_type: str) -> list:
    """
    Get list of default/essential metrics for a given task type.
    This is a subset of available metrics, focusing on the most commonly used ones.

    Args:
        task_type: Type of task ('regression', 'binary_classification', 'multiclass_classification')

    Returns:
        List of default metric names
    """
    if task_type.lower() == 'regression':
        return ['r2', 'rmse', 'mse', 'sep', 'mae', 'rpd', 'bias', 'consistency', 'nrmse', 'nmse', 'nmae', 'pearson_r', 'spearman_r']

    elif task_type.lower() == 'binary_classification':
        return ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity', 'roc_auc', 'jaccard']

    elif task_type.lower() == 'multiclass_classification':
        return ['accuracy', 'balanced_accuracy', 'precision', 'balanced_precision', 'recall', 'balanced_recall', 'f1', 'specificity']

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

