"""
Evaluator module - Generic functions for calculating metrics

This module provides:
- eval(y_true, y_pred, metric): Calculate a specific metric
- eval_multi(y_true, y_pred, task_type): Calculate all metrics for a task type

Supports regression, binary classification, and multiclass classification metrics
using sklearn, scipy, and other standard libraries.
"""

from typing import Dict, Any, Union, Optional, List
import numpy as np
import warnings

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn import metrics as sklearn_metrics
    from sklearn.metrics import (
        # Regression metrics
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error, explained_variance_score,
        max_error, median_absolute_error,

        # Classification metrics
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss,
        confusion_matrix, classification_report,
        balanced_accuracy_score, matthews_corrcoef,
        cohen_kappa_score, hamming_loss, jaccard_score,

        # Multi-label/multi-class specific
        top_k_accuracy_score
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


def eval(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, List[str]]) -> Union[float, Dict[str, float]]:
    """
    Calculate a specific metric for given predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        metric: Metric name (e.g., 'mse', 'accuracy', 'f1', 'r2')

    Returns:
        float: Calculated metric value

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
            return mean_squared_error(y_true, y_pred)
        elif metric in ['rmse', 'root_mean_squared_error']:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric in ['mae', 'mean_absolute_error']:
            return mean_absolute_error(y_true, y_pred)
        elif metric in ['mape', 'mean_absolute_percentage_error']:
            return mean_absolute_percentage_error(y_true, y_pred)
        elif metric in ['r2', 'r2_score']:
            return r2_score(y_true, y_pred)
        elif metric in ['explained_variance', 'explained_variance_score']:
            return explained_variance_score(y_true, y_pred)
        elif metric in ['max_error']:
            return max_error(y_true, y_pred)
        elif metric in ['median_ae', 'median_absolute_error']:
            return median_absolute_error(y_true, y_pred)

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
                return accuracy_score(y_true, y_pred_labels)
            elif metric in ['precision']:
                return precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            elif metric in ['recall']:
                return recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            elif metric in ['f1', 'f1_score']:
                return f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            elif metric in ['precision_micro']:
                return precision_score(y_true, y_pred_labels, average='micro', zero_division=0)
            elif metric in ['recall_micro']:
                return recall_score(y_true, y_pred_labels, average='micro', zero_division=0)
            elif metric in ['f1_micro']:
                return f1_score(y_true, y_pred_labels, average='micro', zero_division=0)
            elif metric in ['precision_macro', 'balanced_precision']:
                return precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
            elif metric in ['recall_macro', 'balanced_recall']:
                return recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
            elif metric in ['f1_macro']:
                return f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
            elif metric in ['balanced_accuracy']:
                return balanced_accuracy_score(y_true, y_pred_labels)
            elif metric in ['matthews_corrcoef', 'mcc']:
                return matthews_corrcoef(y_true, y_pred_labels)
            elif metric in ['cohen_kappa']:
                return cohen_kappa_score(y_true, y_pred_labels)
            elif metric in ['jaccard', 'jaccard_score']:
                return jaccard_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            elif metric in ['hamming_loss']:
                return hamming_loss(y_true, y_pred_labels)
            elif metric == 'specificity':
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
                    return np.mean(specificities)

        elif metric in ['roc_auc', 'auc']:
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                return roc_auc_score(y_true, y_pred)
            else:
                return roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')
        elif metric in ['log_loss']:
            # Convert to probabilities if needed
            if np.all(np.isin(y_pred, [0, 1])):
                # Binary predictions, convert to probabilities
                y_pred_proba = np.column_stack([1 - y_pred, y_pred])
                return log_loss(y_true, y_pred_proba)
            else:
                return log_loss(y_true, y_pred)

        # Additional regression metrics with scipy
        elif metric == 'pearson_r' and SCIPY_AVAILABLE:
            correlation, _ = stats.pearsonr(y_true, y_pred)
            return correlation
        elif metric == 'spearman_r' and SCIPY_AVAILABLE:
            correlation, _ = stats.spearmanr(y_true, y_pred)
            return correlation

        # Custom metrics
        elif metric == 'bias':
            return np.mean(y_pred - y_true)
        elif metric == 'sep':  # Standard Error of Prediction
            return np.std(y_pred - y_true)
        elif metric == 'rpd':  # Ratio of Performance to Deviation
            sep = np.std(y_pred - y_true)
            sd = np.std(y_true)
            return sd / sep if sep != 0 else float('inf')
        elif metric == 'consistency':
            # Consistency: 1 - (RMSE / std(y_true))
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            sd = np.std(y_true)
            return 1 - (rmse / sd) if sd != 0 else 0.0
        elif metric == 'nrmse':
            # Normalized RMSE: RMSE / (max - min)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            y_range = np.max(y_true) - np.min(y_true)
            return rmse / y_range if y_range != 0 else float('inf')
        elif metric == 'nmse':
            # Normalized MSE: MSE / var(y_true)
            mse = mean_squared_error(y_true, y_pred)
            var = np.var(y_true)
            return mse / var if var != 0 else float('inf')
        elif metric == 'nmae':
            # Normalized MAE: MAE / (max - min)
            mae = mean_absolute_error(y_true, y_pred)
            y_range = np.max(y_true) - np.min(y_true)
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
                return np.mean(specificities)

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    except Exception as e:
        raise ValueError(f"Error calculating {metric}: {str(e)}")


def eval_multi(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
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
            metrics['mse'] = eval(y_true, y_pred, 'mse')
            metrics['rmse'] = eval(y_true, y_pred, 'rmse')
            metrics['mae'] = eval(y_true, y_pred, 'mae')
            metrics['r2'] = eval(y_true, y_pred, 'r2')

            # Additional regression metrics
            try:
                metrics['mape'] = eval(y_true, y_pred, 'mape')
            except:
                pass

            try:
                metrics['explained_variance'] = eval(y_true, y_pred, 'explained_variance')
            except:
                pass

            try:
                metrics['max_error'] = eval(y_true, y_pred, 'max_error')
            except:
                pass

            try:
                metrics['median_ae'] = eval(y_true, y_pred, 'median_ae')
            except:
                pass

            # Custom regression metrics
            try:
                metrics['bias'] = eval(y_true, y_pred, 'bias')
                metrics['sep'] = eval(y_true, y_pred, 'sep')
                metrics['rpd'] = eval(y_true, y_pred, 'rpd')
            except:
                pass

            # Correlation metrics (if scipy available)
            if SCIPY_AVAILABLE:
                try:
                    metrics['pearson_r'] = eval(y_true, y_pred, 'pearson_r')
                    metrics['spearman_r'] = eval(y_true, y_pred, 'spearman_r')
                except:
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
            metrics['accuracy'] = eval(y_true, y_pred_labels, 'accuracy')
            metrics['balanced_accuracy'] = eval(y_true, y_pred_labels, 'balanced_accuracy')
            metrics['precision'] = eval(y_true, y_pred_labels, 'precision')
            metrics['balanced_precision'] = eval(y_true, y_pred_labels, 'balanced_precision')
            metrics['recall'] = eval(y_true, y_pred_labels, 'recall')
            metrics['balanced_recall'] = eval(y_true, y_pred_labels, 'balanced_recall')
            metrics['f1'] = eval(y_true, y_pred_labels, 'f1')
            metrics['specificity'] = eval(y_true, y_pred_labels, 'specificity')

            # Binary-specific metrics
            try:
                metrics['roc_auc'] = eval(y_true, y_pred, 'roc_auc')
            except:
                pass

            try:
                metrics['matthews_corrcoef'] = eval(y_true, y_pred_labels, 'matthews_corrcoef')
            except:
                pass

            try:
                metrics['cohen_kappa'] = eval(y_true, y_pred_labels, 'cohen_kappa')
            except:
                pass

            try:
                metrics['jaccard'] = eval(y_true, y_pred_labels, 'jaccard')
            except:
                pass

        elif task_type == 'multiclass_classification':
            # Core classification metrics
            metrics['accuracy'] = eval(y_true, y_pred, 'accuracy')
            metrics['balanced_accuracy'] = eval(y_true, y_pred, 'balanced_accuracy')

            # Weighted averages (default for multiclass)
            metrics['precision'] = eval(y_true, y_pred, 'precision')
            metrics['balanced_precision'] = eval(y_true, y_pred, 'balanced_precision')
            metrics['recall'] = eval(y_true, y_pred, 'recall')
            metrics['balanced_recall'] = eval(y_true, y_pred, 'balanced_recall')
            metrics['f1'] = eval(y_true, y_pred, 'f1')
            metrics['specificity'] = eval(y_true, y_pred, 'specificity')

            # Micro averages
            try:
                metrics['precision_micro'] = eval(y_true, y_pred, 'precision_micro')
                metrics['recall_micro'] = eval(y_true, y_pred, 'recall_micro')
                metrics['f1_micro'] = eval(y_true, y_pred, 'f1_micro')
            except:
                pass

            # Macro averages
            try:
                metrics['precision_macro'] = eval(y_true, y_pred, 'precision_macro')
                metrics['recall_macro'] = eval(y_true, y_pred, 'recall_macro')
                metrics['f1_macro'] = eval(y_true, y_pred, 'f1_macro')
            except:
                pass

            # Multiclass-specific metrics
            try:
                metrics['roc_auc'] = eval(y_true, y_pred, 'roc_auc')
            except:
                pass

            try:
                metrics['matthews_corrcoef'] = eval(y_true, y_pred, 'matthews_corrcoef')
            except:
                pass

            try:
                metrics['cohen_kappa'] = eval(y_true, y_pred, 'cohen_kappa')
            except:
                pass

            try:
                metrics['jaccard'] = eval(y_true, y_pred, 'jaccard')
            except:
                pass

            try:
                metrics['hamming_loss'] = eval(y_true, y_pred, 'hamming_loss')
            except:
                pass

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression', 'binary_classification', or 'multiclass_classification'")

    except Exception as e:
        raise ValueError(f"Error calculating metrics for {task_type}: {str(e)}")

    return metrics

def get_stats(y: np.ndarray) -> Dict[str, float]:
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


def eval_list(y_true: np.ndarray, y_pred: np.ndarray, metrics: list) -> list:
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

    scores = []
    for metric in metrics:
        try:
            score = eval(y_true, y_pred, metric)
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

