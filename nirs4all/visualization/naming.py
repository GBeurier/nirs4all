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
        "selection_score": "Selection_Score",
        # CV-phase ensemble and fold-level metrics
        "ens_test": "Ens_Test",
        "w_ens_test": "W_Ens_Test",
        "mean_fold_cv": "MF_Val",
        # Classification metrics
        "cv_classification": "CV_{metric}",  # e.g., CV_BalAcc
        "test_classification": "Test_{metric}",  # e.g., Test_BalAcc
        "ens_test_classification": "Ens_Test_{metric}",
        "w_ens_test_classification": "W_Ens_Test_{metric}",
        "mean_fold_cv_classification": "MF_Val_{metric}",
    },
    "ml": {
        # Regression metrics (ML/DL convention)
        "cv_score": "CV_Score",
        "test_score": "Test_Score",
        "selection_score": "Selection_Score",
        # CV-phase ensemble and fold-level metrics
        "ens_test": "Ens_Test_Score",
        "w_ens_test": "W_Ens_Test_Score",
        "mean_fold_cv": "MF_CV",
        # Classification metrics
        "cv_classification": "CV_Score",
        "test_classification": "Test_Score",
        "ens_test_classification": "Ens_Test_Score",
        "w_ens_test_classification": "W_Ens_Test_Score",
        "mean_fold_cv_classification": "MF_CV_Score",
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
            "selection_score": convention["selection_score"],
            "ens_test": convention["ens_test"],
            "w_ens_test": convention["w_ens_test"],
            "mean_fold_cv": convention["mean_fold_cv"],
        }
    else:  # classification
        # Format classification templates with metric name
        metric_display = _format_metric_display(metric)
        return {
            "cv_score": convention["cv_classification"].format(metric=metric_display),
            "test_score": convention["test_classification"].format(metric=metric_display),
            "selection_score": convention["selection_score"],
            "ens_test": convention["ens_test_classification"].format(metric=metric_display),
            "w_ens_test": convention["w_ens_test_classification"].format(metric=metric_display),
            "mean_fold_cv": convention["mean_fold_cv_classification"].format(metric=metric_display),
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
