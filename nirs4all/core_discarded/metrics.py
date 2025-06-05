import importlib
from sklearn import metrics as sklearn_metrics
from tensorflow.keras import metrics as keras_metrics

# Expanded list with their classes and mappings
metric_mappings = [
    # Sklearn and Keras mappings
    ("explained_variance", None, None, None, sklearn_metrics.explained_variance_score),
    ("r2", "R2Score", "r2", keras_metrics.R2Score, sklearn_metrics.r2_score),
    ("max_error", None, None, None, sklearn_metrics.max_error),
    ("matthews_corrcoef", None, None, None, sklearn_metrics.matthews_corrcoef),
    ("neg_median_absolute_error", None, None, None, sklearn_metrics.median_absolute_error),
    ("neg_mean_absolute_error", "MeanAbsoluteError", "mae", keras_metrics.MeanAbsoluteError, sklearn_metrics.mean_absolute_error),
    ("neg_mean_absolute_percentage_error", "MeanAbsolutePercentageError", "mape", keras_metrics.MeanAbsolutePercentageError, sklearn_metrics.mean_absolute_percentage_error),
    ("neg_mean_squared_error", "MeanSquaredError", "mse", keras_metrics.MeanSquaredError, sklearn_metrics.mean_squared_error),
    ("neg_mean_squared_log_error", "MeanSquaredLogarithmicError", "msle", keras_metrics.MeanSquaredLogarithmicError, sklearn_metrics.mean_squared_log_error),
    ("neg_root_mean_squared_error", "RootMeanSquaredError", "rmse", keras_metrics.RootMeanSquaredError, None),
    ("neg_root_mean_squared_log_error", None, None, None, None),
    ("neg_mean_poisson_deviance", None, None, None, sklearn_metrics.mean_poisson_deviance),
    ("neg_mean_gamma_deviance", None, None, None, sklearn_metrics.mean_gamma_deviance),
    ("d2_absolute_error_score", None, None, None, sklearn_metrics.d2_absolute_error_score),
    ("accuracy", "Accuracy", "acc", keras_metrics.Accuracy, sklearn_metrics.accuracy_score),
    ("top_k_accuracy", "TopKCategoricalAccuracy", "top_k_acc", keras_metrics.TopKCategoricalAccuracy, None),
    ("roc_auc", "AUC", "auc", keras_metrics.AUC, sklearn_metrics.roc_auc_score),
    ("roc_auc_ovr", None, None, None, sklearn_metrics.roc_auc_score),
    ("roc_auc_ovo", None, None, None, sklearn_metrics.roc_auc_score),
    ("roc_auc_ovr_weighted", None, None, None, sklearn_metrics.roc_auc_score),
    ("roc_auc_ovo_weighted", None, None, None, sklearn_metrics.roc_auc_score),
    ("balanced_accuracy", None, None, None, sklearn_metrics.balanced_accuracy_score),
    ("average_precision", None, None, None, sklearn_metrics.average_precision_score),
    ("neg_log_loss", None, None, None, sklearn_metrics.log_loss),
    ("neg_brier_score", None, None, None, sklearn_metrics.brier_score_loss),
    ("adjusted_rand_score", None, None, None, sklearn_metrics.adjusted_rand_score),
    ("rand_score", None, None, None, sklearn_metrics.rand_score),
    ("homogeneity_score", None, None, None, sklearn_metrics.homogeneity_score),
    ("completeness_score", None, None, None, sklearn_metrics.completeness_score),
    ("v_measure_score", None, None, None, sklearn_metrics.v_measure_score),
    ("mutual_info_score", None, None, None, sklearn_metrics.mutual_info_score),
    ("adjusted_mutual_info_score", None, None, None, sklearn_metrics.adjusted_mutual_info_score),
    ("normalized_mutual_info_score", None, None, None, sklearn_metrics.normalized_mutual_info_score),
    ("fowlkes_mallows_score", None, None, None, sklearn_metrics.fowlkes_mallows_score),

    # Tensorflow/Keras metrics that are not in sklearn
    (None, "FalseNegatives", None, keras_metrics.FalseNegatives, None),
    (None, "FalsePositives", None, keras_metrics.FalsePositives, None),
    (None, "Precision", "prec", keras_metrics.Precision, None),
    (None, "PrecisionAtRecall", None, keras_metrics.PrecisionAtRecall, None),
    (None, "Recall", "recall", keras_metrics.Recall, None),
    (None, "RecallAtPrecision", None, keras_metrics.RecallAtPrecision, None),
    (None, "SensitivityAtSpecificity", None, keras_metrics.SensitivityAtSpecificity, None),
    (None, "SpecificityAtSensitivity", None, keras_metrics.SpecificityAtSensitivity, None),
    (None, "TrueNegatives", None, keras_metrics.TrueNegatives, None),
    (None, "TruePositives", None, keras_metrics.TruePositives, None),
    (None, "Hinge", None, keras_metrics.Hinge, None),
    (None, "SquaredHinge", None, keras_metrics.SquaredHinge, None),
    (None, "CategoricalHinge", None, keras_metrics.CategoricalHinge, None),
    (None, "KLDivergence", None, keras_metrics.KLDivergence, None),
    (None, "Poisson", None, keras_metrics.Poisson, None),
    (None, "BinaryCrossentropy", None, keras_metrics.BinaryCrossentropy, None),
    (None, "CategoricalCrossentropy", None, keras_metrics.CategoricalCrossentropy, None),
    (None, "SparseCategoricalCrossentropy", None, keras_metrics.SparseCategoricalCrossentropy, None),
    (None, "BinaryAccuracy", None, keras_metrics.BinaryAccuracy, None),
    (None, "CategoricalAccuracy", None, keras_metrics.CategoricalAccuracy, None),
    (None, "SparseCategoricalAccuracy", None, keras_metrics.SparseCategoricalAccuracy, None),
    (None, "SparseTopKCategoricalAccuracy", None, keras_metrics.SparseTopKCategoricalAccuracy, None),
    (None, "F1Score", "f1", keras_metrics.MeanIoU, None),  # No direct F1 in Keras
    (None, "FBetaScore", None, keras_metrics.FBetaScore, None),
    (None, "IoU", None, keras_metrics.IoU, None),
    (None, "BinaryIoU", None, keras_metrics.BinaryIoU, None),
    (None, "MeanIoU", None, keras_metrics.MeanIoU, None),
    (None, "OneHotIoU", None, keras_metrics.OneHotIoU, None),
    (None, "OneHotMeanIoU", None, keras_metrics.OneHotMeanIoU, None),
]

# Function to retrieve the metric class/function


def get_metric(metric, framework=None):
    # If a class is provided, return the class
    if isinstance(metric, type):
        return metric

    # If a string is provided, attempt to map it
    for sk_name, tf_name, abbreviation, tf_class, sk_function in metric_mappings:
        if metric in [sk_name, tf_name, abbreviation]:
            if framework == 'tensorflow' or framework == 'keras':
                return tf_class
            elif framework == 'sklearn':
                return sk_function
            else:
                return tf_class if tf_class is not None else sk_function

    # Try to import if a full path string is provided
    try:
        module_path, class_name = metric.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        raise ValueError(f"Metric '{metric}' not found in sklearn or tensorflow/keras.")
