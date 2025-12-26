"""
NIRS4All sklearn Integration Module.

This module provides sklearn-compatible wrappers for nirs4all pipelines,
enabling integration with scikit-learn tools like cross_validate,
GridSearchCV, and SHAP explainers.

Classes:
    NIRSPipeline: sklearn-compatible regressor wrapper for trained pipelines.
    NIRSPipelineClassifier: Classification variant of NIRSPipeline.

Important:
    NIRSPipeline is a PREDICTION wrapper, not a training estimator.
    Training is done via nirs4all.run(), then the result can be wrapped
    for sklearn compatibility using NIRSPipeline.from_result() or
    NIRSPipeline.from_bundle().

Example:
    >>> import nirs4all
    >>> from nirs4all.sklearn import NIRSPipeline
    >>> import shap
    >>>
    >>> # Train with nirs4all
    >>> result = nirs4all.run(pipeline, dataset)
    >>>
    >>> # Wrap for sklearn/SHAP compatibility
    >>> pipe = NIRSPipeline.from_result(result)
    >>> explainer = shap.Explainer(pipe.predict, X_background)
    >>> shap_values = explainer(X_test)
    >>>
    >>> # Or from exported bundle
    >>> pipe = NIRSPipeline.from_bundle("exports/model.n4a")
    >>> y_pred = pipe.predict(X_new)
    >>> print(f"R²: {pipe.score(X_test, y_test):.4f}")
"""

from .pipeline import NIRSPipeline
from .classifier import NIRSPipelineClassifier

__all__ = [
    "NIRSPipeline",
    "NIRSPipelineClassifier",
]
