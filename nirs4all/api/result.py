"""
Result classes for nirs4all API.

These dataclasses wrap the outputs from pipeline execution, prediction,
and explanation operations, providing convenient accessor methods.

Classes:
    RunResult: Result from nirs4all.run()
    PredictResult: Result from nirs4all.predict()
    ExplainResult: Result from nirs4all.explain()

Phase 1 Implementation (v0.6.0):
    - RunResult: Full implementation with best, best_score, top(), export()
    - PredictResult: Full implementation with values, to_dataframe()
    - ExplainResult: Full implementation with values, feature attributions
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline import PipelineRunner
    from nirs4all.data.predictions import Predictions


@dataclass
class RunResult:
    """Result from nirs4all.run().

    Provides convenient access to predictions, best model, and artifacts.
    Wraps the raw (predictions, per_dataset) tuple returned by PipelineRunner.run().

    Attributes:
        predictions: Predictions object containing all pipeline results.
        per_dataset: Dictionary with per-dataset execution details.

    Properties:
        best: Best prediction entry by default ranking.
        best_score: Best model's primary test score.
        best_rmse: Best model's RMSE (regression).
        best_r2: Best model's R² (regression).
        best_accuracy: Best model's accuracy (classification).
        artifacts_path: Path to run artifacts directory.
        num_predictions: Total number of predictions stored.

    Methods:
        top(n): Get top N predictions by ranking.
        export(path): Export best model to .n4a bundle.
        filter(**kwargs): Filter predictions by criteria.
        get_datasets(): Get list of unique dataset names.
        get_models(): Get list of unique model names.

    Example:
        >>> result = nirs4all.run(pipeline, dataset)
        >>> print(f"Best RMSE: {result.best_rmse:.4f}")
        >>> print(f"Best R²: {result.best_r2:.4f}")
        >>> result.export("exports/best_model.n4a")
    """

    predictions: "Predictions"
    per_dataset: Dict[str, Any]
    _runner: Optional["PipelineRunner"] = field(default=None, repr=False)

    # --- Primary accessors ---

    @property
    def best(self) -> Dict[str, Any]:
        """Get best prediction entry by default ranking.

        Returns:
            Dictionary containing best model's metrics, name, and configuration.
            Empty dict if no predictions available.
        """
        top = self.predictions.top(n=1)
        return top[0] if top else {}

    @property
    def best_score(self) -> float:
        """Get best model's primary test score.

        Returns:
            The test_score value from best prediction, or NaN if unavailable.
        """
        return self.best.get('test_score', float('nan'))

    @property
    def best_rmse(self) -> float:
        """Get best model's RMSE score.

        Looks for 'rmse' in scores dict, then falls back to computing from y arrays.

        Returns:
            RMSE value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        # Try scores dict first
        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'rmse' in test_scores:
                return test_scores['rmse']

        # Fall back to test_score if metric is rmse-like
        metric = best.get('metric', '')
        if metric in ('rmse', 'mse'):
            return best.get('test_score', float('nan'))

        return float('nan')

    @property
    def best_r2(self) -> float:
        """Get best model's R² score.

        Looks for 'r2' in scores dict.

        Returns:
            R² value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'r2' in test_scores:
                return test_scores['r2']

        return float('nan')

    @property
    def best_accuracy(self) -> float:
        """Get best model's accuracy score (for classification).

        Returns:
            Accuracy value or NaN if unavailable.
        """
        best = self.best
        if not best:
            return float('nan')

        scores = best.get('scores', {})
        if isinstance(scores, dict):
            test_scores = scores.get('test', {})
            if 'accuracy' in test_scores:
                return test_scores['accuracy']

        # Fall back to test_score if metric is accuracy
        metric = best.get('metric', '')
        if metric == 'accuracy':
            return best.get('test_score', float('nan'))

        return float('nan')

    # --- Metadata accessors ---

    @property
    def artifacts_path(self) -> Optional[Path]:
        """Get path to run artifacts directory.

        Returns:
            Path to the current run directory, or None if not available.
        """
        if self._runner and hasattr(self._runner, 'current_run_dir'):
            return self._runner.current_run_dir
        return None

    @property
    def num_predictions(self) -> int:
        """Get total number of predictions stored.

        Returns:
            Number of prediction entries.
        """
        return self.predictions.num_predictions

    # --- Query methods ---

    def top(self, n: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Get top N predictions by ranking.

        Args:
            n: Number of top predictions to return.
            **kwargs: Additional arguments passed to predictions.top().
                Supported kwargs include:
                - rank_metric: Metric to rank by (default: uses record's metric)
                - rank_partition: Partition to rank on (default: "val")
                - display_partition: Partition for display metrics (default: "test")
                - aggregate_partitions: If True, include train/val/test data
                - ascending: Sort order (None = infer from metric)
                - group_by: Group predictions, keep best per group

        Returns:
            List of prediction dictionaries, ranked by score.
        """
        return self.predictions.top(n=n, **kwargs)

    def filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Filter predictions by criteria.

        Args:
            **kwargs: Filter criteria passed to predictions.filter_predictions().
                Supported kwargs include:
                - dataset_name: Filter by dataset name
                - model_name: Filter by model name
                - partition: Filter by partition ('train', 'val', 'test')
                - fold_id: Filter by fold ID
                - step_idx: Filter by pipeline step index
                - branch_id: Filter by branch ID
                - load_arrays: If True, load actual arrays (default: True)

        Returns:
            List of matching prediction dictionaries.
        """
        return self.predictions.filter_predictions(**kwargs)

    def get_datasets(self) -> List[str]:
        """Get list of unique dataset names.

        Returns:
            List of dataset names in predictions.
        """
        return self.predictions.get_datasets()

    def get_models(self) -> List[str]:
        """Get list of unique model names.

        Returns:
            List of model names in predictions.
        """
        return self.predictions.get_models()

    # --- Export methods ---

    def export(
        self,
        output_path: Union[str, Path],
        format: str = "n4a",
        source: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export a model to bundle.

        Args:
            output_path: Path for the exported bundle file.
            format: Export format ('n4a' or 'n4a.py').
            source: Prediction dict to export. If None, exports best model.

        Returns:
            Path to the exported bundle file.

        Raises:
            RuntimeError: If runner reference is not available.
            ValueError: If no predictions available and source not provided.
        """
        if self._runner is None:
            raise RuntimeError("Cannot export: runner reference not available")

        if source is None:
            source = self.best
            if not source:
                raise ValueError("No predictions available to export")

        return self._runner.export(
            source=source,
            output_path=output_path,
            format=format
        )

    def export_model(
        self,
        output_path: Union[str, Path],
        source: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        fold: Optional[int] = None
    ) -> Path:
        """Export only the model artifact (lightweight).

        Unlike export() which creates a full bundle, this exports just the model.

        Args:
            output_path: Path for the output model file.
            source: Prediction dict to export. If None, exports best model.
            format: Model format (inferred from extension if None).
            fold: Fold index to export (default: fold 0).

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If runner reference is not available.
        """
        if self._runner is None:
            raise RuntimeError("Cannot export: runner reference not available")

        if source is None:
            source = self.best
            if not source:
                raise ValueError("No predictions available to export")

        return self._runner.export_model(
            source=source,
            output_path=output_path,
            format=format,
            fold=fold
        )

    # --- Utility methods ---

    def summary(self) -> str:
        """Get a summary string of the run result.

        Returns:
            Multi-line summary string with key metrics.
        """
        lines = []
        lines.append(f"RunResult: {self.num_predictions} predictions")

        if self.artifacts_path:
            lines.append(f"  Artifacts: {self.artifacts_path}")

        datasets = self.get_datasets()
        if datasets:
            lines.append(f"  Datasets: {', '.join(datasets)}")

        models = self.get_models()
        if models:
            lines.append(f"  Models: {', '.join(models[:5])}" +
                        (f" (+{len(models)-5} more)" if len(models) > 5 else ""))

        best = self.best
        if best:
            lines.append(f"  Best: {best.get('model_name', 'unknown')}")
            lines.append(f"    test_score: {self.best_score:.4f}")
            if not np.isnan(self.best_rmse):
                lines.append(f"    rmse: {self.best_rmse:.4f}")
            if not np.isnan(self.best_r2):
                lines.append(f"    r2: {self.best_r2:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"RunResult(predictions={self.num_predictions}, best_score={self.best_score:.4f})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.summary()

    def validate(
        self,
        check_nan_metrics: bool = True,
        check_empty: bool = True,
        raise_on_failure: bool = True,
        nan_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Validate the run result for common issues.

        Checks for NaN values in metrics, empty predictions, and other issues
        that might indicate problems with the pipeline execution.

        Args:
            check_nan_metrics: If True, check for NaN values in metrics.
            check_empty: If True, check for empty predictions.
            raise_on_failure: If True, raise ValueError on validation failure.
            nan_threshold: Maximum allowed ratio of predictions with NaN metrics (0.0 = none allowed).

        Returns:
            Dictionary with validation results:
                - valid: True if all checks passed.
                - issues: List of issue descriptions.
                - nan_count: Number of predictions with NaN metrics.
                - total_count: Total number of predictions.

        Raises:
            ValueError: If raise_on_failure=True and validation fails.

        Example:
            >>> result = nirs4all.run(pipeline, dataset)
            >>> result.validate()  # Raises if issues found
            >>> # Or check without raising
            >>> report = result.validate(raise_on_failure=False)
            >>> if not report['valid']:
            ...     print(f"Issues: {report['issues']}")
        """
        issues = []
        nan_count = 0
        total_count = self.num_predictions

        # Check for empty predictions
        if check_empty and total_count == 0:
            issues.append("No predictions found")

        # Check for NaN metrics
        if check_nan_metrics and total_count > 0:
            all_preds = self.predictions.top(n=total_count)
            for pred in all_preds:
                has_nan = False
                # Check common metrics
                for metric in ['rmse', 'r2', 'accuracy', 'mse', 'mae']:
                    value = pred.get(metric)
                    if value is not None and np.isnan(value):
                        has_nan = True
                        break

                # Check scores dict
                if not has_nan:
                    scores = pred.get('scores', {})
                    if isinstance(scores, dict):
                        for partition_scores in scores.values():
                            if isinstance(partition_scores, dict):
                                for val in partition_scores.values():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        has_nan = True
                                        break

                # Check test_score
                if not has_nan:
                    test_score = pred.get('test_score')
                    if test_score is not None and np.isnan(test_score):
                        has_nan = True

                if has_nan:
                    nan_count += 1
                    model_name = pred.get('model_name', 'unknown')
                    if nan_count <= 5:  # Only report first 5
                        issues.append(f"NaN metrics found in prediction: {model_name}")

            if nan_count > 5:
                issues.append(f"... and {nan_count - 5} more predictions with NaN metrics")

            # Check threshold
            nan_ratio = nan_count / total_count if total_count > 0 else 0
            if nan_ratio > nan_threshold:
                issues.append(
                    f"NaN ratio ({nan_ratio:.1%}) exceeds threshold ({nan_threshold:.1%})"
                )

        valid = len(issues) == 0

        report = {
            'valid': valid,
            'issues': issues,
            'nan_count': nan_count,
            'total_count': total_count,
        }

        if raise_on_failure and not valid:
            raise ValueError(
                f"RunResult validation failed:\n" +
                "\n".join(f"  - {issue}" for issue in issues)
            )

        return report


@dataclass
class PredictResult:
    """Result from nirs4all.predict().

    Wraps prediction outputs with convenient accessors and conversion methods.

    Attributes:
        y_pred: Predicted values array (n_samples,) or (n_samples, n_outputs).
        metadata: Additional prediction metadata (uncertainty, timing, etc.).
        sample_indices: Optional indices of predicted samples.
        model_name: Name of the model used for prediction.
        preprocessing_steps: List of preprocessing steps applied.

    Properties:
        values: Alias for y_pred (for consistency).
        shape: Shape of prediction array.
        is_multioutput: True if predictions have multiple outputs.

    Methods:
        to_numpy(): Get predictions as numpy array.
        to_list(): Get predictions as Python list.
        to_dataframe(): Get predictions as pandas DataFrame.
        flatten(): Get flattened 1D predictions.

    Example:
        >>> result = nirs4all.predict(model, X_new)
        >>> print(f"Predictions shape: {result.shape}")
        >>> df = result.to_dataframe()
    """

    y_pred: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_indices: Optional[np.ndarray] = None
    model_name: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure y_pred is a numpy array."""
        if self.y_pred is not None and not isinstance(self.y_pred, np.ndarray):
            self.y_pred = np.asarray(self.y_pred)

    @property
    def values(self) -> np.ndarray:
        """Get prediction values (alias for y_pred)."""
        return self.y_pred

    @property
    def shape(self) -> tuple:
        """Get shape of prediction array."""
        if self.y_pred is None:
            return (0,)
        return self.y_pred.shape

    @property
    def is_multioutput(self) -> bool:
        """Check if predictions have multiple outputs."""
        return len(self.shape) > 1 and self.shape[1] > 1

    def __len__(self) -> int:
        """Return number of predictions."""
        if self.y_pred is None:
            return 0
        return len(self.y_pred)

    def to_numpy(self) -> np.ndarray:
        """Get predictions as numpy array.

        Returns:
            Numpy array of predictions.
        """
        return self.y_pred

    def to_list(self) -> List[float]:
        """Get predictions as Python list.

        Returns:
            List of prediction values (flattened if 2D).
        """
        if self.y_pred is None:
            return []
        return self.y_pred.flatten().tolist()

    def to_dataframe(self, include_indices: bool = True):
        """Get predictions as pandas DataFrame.

        Args:
            include_indices: If True and sample_indices available, include as column.

        Returns:
            pandas DataFrame with predictions.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        data = {}

        if include_indices and self.sample_indices is not None:
            data['sample_index'] = self.sample_indices

        if self.is_multioutput:
            for i in range(self.shape[1]):
                data[f'y_pred_{i}'] = self.y_pred[:, i]
        else:
            data['y_pred'] = self.y_pred.flatten()

        return pd.DataFrame(data)

    def flatten(self) -> np.ndarray:
        """Get flattened 1D predictions.

        Returns:
            1D numpy array of predictions.
        """
        if self.y_pred is None:
            return np.array([])
        return self.y_pred.flatten()

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictResult(shape={self.shape}, model='{self.model_name}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"PredictResult: {len(self)} predictions"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        if self.preprocessing_steps:
            lines.append(f"  Preprocessing: {' -> '.join(self.preprocessing_steps)}")
        lines.append(f"  Shape: {self.shape}")
        return "\n".join(lines)


@dataclass
class ExplainResult:
    """Result from nirs4all.explain().

    Wraps SHAP explanation outputs with visualization helpers and accessors.

    Attributes:
        shap_values: SHAP values array or Explanation object.
        feature_names: Names/labels of features explained.
        base_value: Expected value (baseline prediction).
        visualizations: Paths to generated visualization files.
        explainer_type: Type of SHAP explainer used.
        model_name: Name of the explained model.
        n_samples: Number of samples explained.

    Properties:
        values: Raw SHAP values array.
        shape: Shape of SHAP values array.
        mean_abs_shap: Mean absolute SHAP values per feature.
        top_features: Feature names sorted by importance.

    Methods:
        get_feature_importance(): Get feature importance ranking.
        get_sample_explanation(idx): Get explanation for a single sample.
        to_dataframe(): Get SHAP values as DataFrame.

    Example:
        >>> result = nirs4all.explain(model, X_test)
        >>> print(f"Top features: {result.top_features[:5]}")
        >>> importance = result.get_feature_importance()
    """

    shap_values: Any  # shap.Explanation or np.ndarray
    feature_names: Optional[List[str]] = None
    base_value: Optional[Union[float, np.ndarray]] = None
    visualizations: Dict[str, Path] = field(default_factory=dict)
    explainer_type: str = "auto"
    model_name: str = ""
    n_samples: int = 0

    def __post_init__(self):
        """Extract metadata from shap_values if available."""
        if hasattr(self.shap_values, 'values'):
            # It's a shap.Explanation object
            if self.feature_names is None and hasattr(self.shap_values, 'feature_names'):
                self.feature_names = list(self.shap_values.feature_names)
            if self.base_value is None and hasattr(self.shap_values, 'base_values'):
                self.base_value = self.shap_values.base_values
            if self.n_samples == 0:
                self.n_samples = len(self.shap_values.values)

    @property
    def values(self) -> np.ndarray:
        """Get raw SHAP values array.

        Returns:
            Numpy array of SHAP values (n_samples, n_features).
        """
        if hasattr(self.shap_values, 'values'):
            return self.shap_values.values
        return np.asarray(self.shap_values)

    @property
    def shape(self) -> tuple:
        """Get shape of SHAP values array."""
        return self.values.shape

    @property
    def mean_abs_shap(self) -> np.ndarray:
        """Get mean absolute SHAP values per feature.

        Returns:
            1D array of mean |SHAP| values, one per feature.
        """
        vals = self.values
        if vals.ndim == 1:
            return np.abs(vals)
        return np.mean(np.abs(vals), axis=0)

    @property
    def top_features(self) -> List[str]:
        """Get feature names sorted by importance (descending).

        Returns:
            List of feature names, most important first.
            Returns indices as strings if feature_names not available.
        """
        importance = self.mean_abs_shap
        sorted_indices = np.argsort(importance)[::-1]

        if self.feature_names:
            return [self.feature_names[i] for i in sorted_indices]
        return [str(i) for i in sorted_indices]

    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
        normalize: bool = False
    ) -> Dict[str, float]:
        """Get feature importance ranking.

        Args:
            top_n: If provided, return only top N features.
            normalize: If True, normalize values to sum to 1.

        Returns:
            Dictionary mapping feature names to importance values.
        """
        importance = self.mean_abs_shap

        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()

        sorted_indices = np.argsort(importance)[::-1]

        if top_n:
            sorted_indices = sorted_indices[:top_n]

        result = {}
        for idx in sorted_indices:
            name = self.feature_names[idx] if self.feature_names else str(idx)
            result[name] = float(importance[idx])

        return result

    def get_sample_explanation(
        self,
        idx: int
    ) -> Dict[str, float]:
        """Get SHAP explanation for a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary mapping feature names to SHAP values for that sample.
        """
        vals = self.values
        if idx >= len(vals):
            raise IndexError(f"Sample index {idx} out of range (n_samples={len(vals)})")

        sample_shap = vals[idx] if vals.ndim > 1 else vals

        result = {}
        for i, val in enumerate(sample_shap):
            name = self.feature_names[i] if self.feature_names else str(i)
            result[name] = float(val)

        return result

    def to_dataframe(self, include_feature_names: bool = True):
        """Get SHAP values as pandas DataFrame.

        Args:
            include_feature_names: If True, use feature names as columns.

        Returns:
            pandas DataFrame with SHAP values.

        Raises:
            ImportError: If pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        vals = self.values

        if include_feature_names and self.feature_names:
            columns = self.feature_names
        else:
            columns = [f"feature_{i}" for i in range(vals.shape[-1])]

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        return pd.DataFrame(vals, columns=columns)

    def __repr__(self) -> str:
        """String representation."""
        return f"ExplainResult(shape={self.shape}, explainer='{self.explainer_type}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [f"ExplainResult: {self.n_samples} samples explained"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name}")
        lines.append(f"  Explainer: {self.explainer_type}")
        lines.append(f"  Shape: {self.shape}")
        if self.feature_names:
            lines.append(f"  Features: {len(self.feature_names)}")

        # Show top 5 features
        top = self.top_features[:5]
        if top:
            lines.append(f"  Top features: {', '.join(top)}")

        if self.visualizations:
            lines.append(f"  Visualizations: {list(self.visualizations.keys())}")

        return "\n".join(lines)
