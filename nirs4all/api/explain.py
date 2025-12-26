"""
Module-level explain() function for nirs4all.

This module provides a simple interface for generating SHAP explanations
for trained nirs4all models. It wraps PipelineRunner.explain() with
ergonomic defaults and returns a structured ExplainResult.

Example:
    >>> import nirs4all
    >>> result = nirs4all.explain(
    ...     model="exports/best_model.n4a",
    ...     data=X_test,
    ...     verbose=1
    ... )
    >>> print(f"Top features: {result.top_features[:5]}")
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset

from .result import ExplainResult
from .session import Session


# Type aliases for clarity
ModelSpec = Union[
    Dict[str, Any],               # Prediction dict from previous run
    str,                          # Path to bundle (.n4a) or config
    Path                          # Path to bundle or config
]

DataSpec = Union[
    str,                          # Path to data folder
    Path,                         # Path to data folder
    np.ndarray,                   # X array
    Dict[str, Any],               # Dict with X key
    SpectroDataset,               # Direct SpectroDataset instance
    DatasetConfigs                # Backward compat
]


def explain(
    model: ModelSpec,
    data: DataSpec,
    *,
    name: str = "explain_dataset",
    session: Optional[Session] = None,
    verbose: int = 1,
    plots_visible: bool = True,
    # SHAP-specific parameters
    n_samples: Optional[int] = None,
    explainer_type: str = "auto",
    **shap_params: Any
) -> ExplainResult:
    """Generate SHAP explanations for a trained model.

    This function provides a simple interface for computing SHAP values
    to explain model predictions. It supports various SHAP explainer types
    and generates visualizations.

    Args:
        model: Trained model specification. Can be:
            - Prediction dict from ``result.best`` or ``result.top()``
            - Path to exported bundle: ``"exports/model.n4a"``
            - Path to pipeline config directory

        data: Data to explain. Can be:
            - Path to data folder: ``"test_data/"``
            - Numpy array: ``X_test`` (n_samples, n_features)
            - Dict: ``{"X": X, "metadata": meta}``
            - SpectroDataset instance

        name: Name for the explanation dataset (for logging).
            Default: "explain_dataset"

        session: Optional Session for resource reuse.
            If provided, uses the session's runner.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug).
            Default: 1

        plots_visible: Whether to display plots interactively.
            Default: True

        n_samples: Number of background samples for SHAP.
            If None, uses default (typically 100-200).

        explainer_type: SHAP explainer type. Options:
            - "auto": Automatically select best explainer
            - "tree": TreeExplainer (for tree-based models)
            - "kernel": KernelExplainer (model-agnostic)
            - "deep": DeepExplainer (for neural networks)
            - "linear": LinearExplainer (for linear models)
            Default: "auto"

        **shap_params: Additional SHAP configuration parameters.
            Common options:
            - feature_names: List of feature names
            - background_samples: Number of background samples
            - max_display: Max features to show in plots

    Returns:
        ExplainResult containing:
            - shap_values: SHAP values array or Explanation object
            - feature_names: Names/labels of features
            - base_value: Expected value (baseline prediction)
            - visualizations: Paths to generated plots
            - mean_abs_shap: Mean absolute SHAP per feature
            - top_features: Features sorted by importance

        Use ``result.get_feature_importance()`` for importance ranking,
        or ``result.to_dataframe()`` for pandas DataFrame output.

    Raises:
        ValueError: If model specification is invalid.
        FileNotFoundError: If model bundle or data path doesn't exist.
        ImportError: If SHAP is not installed.

    Examples:
        Explain an exported model:

        >>> import nirs4all
        >>>
        >>> result = nirs4all.explain(
        ...     model="exports/wheat_model.n4a",
        ...     data=X_test
        ... )
        >>> print(f"Top 5 features: {result.top_features[:5]}")
        >>> importance = result.get_feature_importance(top_n=10)

        Explain using a result from a previous run:

        >>> # Training
        >>> train_result = nirs4all.run(pipeline, train_data)
        >>>
        >>> # Explain best model
        >>> explain_result = nirs4all.explain(
        ...     model=train_result.best,
        ...     data=X_test,
        ...     explainer_type="kernel"
        ... )

        Get SHAP values as DataFrame:

        >>> result = nirs4all.explain(model, data)
        >>> df = result.to_dataframe()
        >>> df.to_csv("shap_values.csv")

        Get per-sample explanations:

        >>> result = nirs4all.explain(model, data)
        >>> sample_0_shap = result.get_sample_explanation(0)
        >>> for feature, value in list(sample_0_shap.items())[:5]:
        ...     print(f"{feature}: {value:.4f}")

    See Also:
        - :func:`nirs4all.run`: Train a pipeline
        - :func:`nirs4all.predict`: Make predictions
        - :class:`nirs4all.api.result.ExplainResult`: Result class
    """
    # Build SHAP params dict
    full_shap_params = dict(shap_params)
    if n_samples is not None:
        full_shap_params["n_samples"] = n_samples
    if explainer_type != "auto":
        full_shap_params["explainer_type"] = explainer_type

    # Use session runner if provided, otherwise create new
    if session is not None:
        runner = session.runner
    else:
        runner = PipelineRunner(
            mode="explain",
            verbose=verbose,
            plots_visible=plots_visible
        )

    # Convert Path to str for compatibility with type hints
    model_arg = str(model) if isinstance(model, Path) else model
    data_arg = str(data) if isinstance(data, Path) else data

    # Call the runner's explain method
    shap_results, output_dir = runner.explain(
        prediction_obj=model_arg,
        dataset=data_arg,
        dataset_name=name,
        shap_params=full_shap_params if full_shap_params else None,
        verbose=verbose,
        plots_visible=plots_visible
    )

    # Extract SHAP values from results
    shap_values = shap_results.get("shap_values")
    feature_names = shap_results.get("feature_names")
    base_value = shap_results.get("expected_value") or shap_results.get("base_value")

    # Build visualizations dict from output directory
    visualizations = {}
    if output_dir:
        output_path = Path(output_dir)
        if output_path.exists():
            for f in output_path.glob("*.png"):
                visualizations[f.stem] = f
            for f in output_path.glob("*.html"):
                visualizations[f.stem] = f

    # Determine explainer type from results
    actual_explainer = shap_results.get("explainer_type", explainer_type)

    # Get model name
    model_name = ""
    if isinstance(model, dict):
        model_name = model.get("model_name", "")

    # Count samples
    n_explained = 0
    if shap_values is not None:
        if hasattr(shap_values, 'values'):
            n_explained = len(shap_values.values)
        elif isinstance(shap_values, np.ndarray):
            n_explained = len(shap_values)

    return ExplainResult(
        shap_values=shap_values,
        feature_names=feature_names,
        base_value=base_value,
        visualizations=visualizations,
        explainer_type=actual_explainer,
        model_name=model_name,
        n_samples=n_explained
    )
