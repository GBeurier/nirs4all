"""Naming conventions for reports and visualization.

This module provides configurable naming conventions for metrics and scores
to support different user communities (NIRS/chemometrics vs ML/DL).
"""

from typing import Literal

NamingMode = Literal["nirs", "ml", "auto"]


NAMING_CONVENTIONS = {
    "nirs": {
        # Regression metrics (chemometrics standard)
        "cv_score": "RMSECV",
        "test_score": "RMSEP",
        "mean_fold_test": "Mean_Fold_RMSEP",
        "wmean_fold_test": "W_Mean_Fold_RMSEP",
        "selection_score": "Selection_Score",
        # Classification metrics
        "cv_classification": "CV_{metric}",  # e.g., CV_BalAcc
        "test_classification": "Test_{metric}",  # e.g., Test_BalAcc
        "mean_fold_classification": "Mean_Fold_{metric}",
        "wmean_fold_classification": "W_Mean_Fold_{metric}",
    },
    "ml": {
        # Regression metrics (ML/DL convention)
        "cv_score": "CV_Score",
        "test_score": "Test_Score",
        "mean_fold_test": "Mean_Fold_Test",
        "wmean_fold_test": "W_Mean_Fold_Test",
        "selection_score": "Selection_Score",
        # Classification metrics
        "cv_classification": "CV_Score",
        "test_classification": "Test_Score",
        "mean_fold_classification": "Mean_Fold_Score",
        "wmean_fold_classification": "W_Mean_Fold_Score",
    },
}


def get_metric_names(
    mode: NamingMode,
    task_type: str = "regression",
    metric: str = "rmse",
) -> dict[str, str]:
    """Get metric names for a given naming mode and task type.

    Args:
        mode: Naming convention mode ("nirs", "ml", or "auto").
        task_type: Task type ("regression" or "classification").
        metric: Metric name (e.g., "rmse", "balanced_accuracy").

    Returns:
        Dictionary mapping internal names to display names.

    Examples:
        >>> get_metric_names("nirs", "regression")
        {'cv_score': 'RMSECV', 'test_score': 'RMSEP', ...}
        >>> get_metric_names("ml", "classification", "balanced_accuracy")
        {'cv_score': 'CV_Score', 'test_score': 'Test_Score', ...}
    """
    # Auto mode defaults to NIRS for consistency with existing reports
    if mode == "auto":
        mode = "nirs"

    convention = NAMING_CONVENTIONS[mode]

    if task_type == "regression":
        return {
            "cv_score": convention["cv_score"],
            "test_score": convention["test_score"],
            "mean_fold_test": convention["mean_fold_test"],
            "wmean_fold_test": convention["wmean_fold_test"],
            "selection_score": convention["selection_score"],
        }
    else:  # classification
        # Format classification templates with metric name
        metric_display = _format_metric_display(metric)
        return {
            "cv_score": convention["cv_classification"].format(metric=metric_display),
            "test_score": convention["test_classification"].format(metric=metric_display),
            "mean_fold_test": convention["mean_fold_classification"].format(metric=metric_display),
            "wmean_fold_test": convention["wmean_fold_classification"].format(metric=metric_display),
            "selection_score": convention["selection_score"],
        }


def _format_metric_display(metric: str) -> str:
    """Format a metric name for display.

    Args:
        metric: Internal metric name (e.g., "balanced_accuracy", "rmse").

    Returns:
        Formatted display name (e.g., "BalAcc", "RMSE").

    Examples:
        >>> _format_metric_display("balanced_accuracy")
        'BalAcc'
        >>> _format_metric_display("rmse")
        'RMSE'
        >>> _format_metric_display("accuracy")
        'Accuracy'
    """
    # Common abbreviations
    abbreviations = {
        "balanced_accuracy": "BalAcc",
        "accuracy": "Accuracy",
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC_AUC",
        "rmse": "RMSE",
        "mse": "MSE",
        "mae": "MAE",
        "r2": "R2",
    }

    return abbreviations.get(metric.lower(), metric.replace("_", " ").title())
