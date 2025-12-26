"""
sklearn-compatible pipeline wrapper for nirs4all.

NIRSPipeline wraps a trained nirs4all pipeline to provide sklearn's
BaseEstimator interface, enabling use with sklearn tools and SHAP.

Important Design Decision:
    NIRSPipeline is a PREDICTION-ONLY wrapper. It does NOT implement fit()
    for training. This is because:
    1. nirs4all's CV creates N models per fold - no single "fitted" model
    2. Generator syntax expansion happens at config time, not fit time
    3. Branching pipelines have multiple output paths

    Training should be done via nirs4all.run(), then wrapped with from_result().

Example:
    >>> # Train with nirs4all
    >>> result = nirs4all.run(pipeline, dataset)
    >>>
    >>> # Wrap for sklearn compatibility
    >>> pipe = NIRSPipeline.from_result(result)
    >>>
    >>> # Use with SHAP
    >>> explainer = shap.Explainer(pipe.predict, X_background)
    >>> shap_values = explainer(X_test)
    >>>
    >>> # Or from exported bundle
    >>> pipe = NIRSPipeline.from_bundle("exports/model.n4a")
"""

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import logging

import numpy as np

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult
    from nirs4all.pipeline.bundle import BundleLoader

logger = logging.getLogger(__name__)


class NIRSPipeline:
    """sklearn-compatible wrapper for trained nirs4all pipelines.

    This class wraps a trained nirs4all pipeline to provide sklearn's
    BaseEstimator interface. It is designed for PREDICTION and EXPLANATION,
    not for training (use nirs4all.run() for training).

    Construction:
        Use class methods to create instances:
        - NIRSPipeline.from_result(result): From a RunResult
        - NIRSPipeline.from_bundle(path): From an exported .n4a bundle

    Attributes:
        is_fitted_: Always True for wrapped pipelines.
        model_: The underlying model (fold 0) for SHAP access.
        bundle_loader_: BundleLoader instance (if created from bundle).
        preprocessing_chain: String summary of preprocessing steps.
        model_step_index: Index of the model step in the pipeline.
        fold_weights: Dictionary of fold weights for CV ensemble.

    Methods:
        predict(X): Make predictions on new data.
        score(X, y): Compute R² score.
        transform(X): Apply preprocessing steps (without model).

    sklearn Compatibility:
        - Implements BaseEstimator interface (get_params, set_params)
        - Implements RegressorMixin (score method)
        - Works with SHAP explainers
        - Works with sklearn.model_selection.cross_val_predict (predict only)

    Example:
        >>> result = nirs4all.run(pipeline, dataset)
        >>> pipe = NIRSPipeline.from_result(result)
        >>> y_pred = pipe.predict(X_new)
        >>> print(f"R²: {pipe.score(X_test, y_test):.4f}")
    """

    def __init__(self) -> None:
        """Private constructor - use from_result() or from_bundle() instead."""
        # Core state
        self._bundle_loader: Optional["BundleLoader"] = None
        self._runner: Optional[Any] = None
        self._prediction_source: Optional[Dict[str, Any]] = None
        self._is_fitted: bool = False
        self._fold: int = 0

        # Cached model for SHAP access
        self._cached_model: Optional[Any] = None
        self._cached_transformers: Optional[List[Any]] = None

        # Metadata
        self._preprocessing_chain: str = ""
        self._model_step_index: Optional[int] = None
        self._fold_weights: Dict[int, float] = {}
        self._model_name: str = ""
        self._source_path: Optional[Path] = None

    @classmethod
    def from_result(
        cls,
        result: "RunResult",
        source: Optional[Dict[str, Any]] = None,
        fold: int = 0
    ) -> "NIRSPipeline":
        """Create NIRSPipeline from a RunResult.

        This exports the best (or specified) model from the RunResult to a
        temporary bundle, then loads it for prediction. This ensures consistent
        prediction behavior between direct bundle loading and result-based
        creation.

        Args:
            result: RunResult from nirs4all.run().
            source: Optional prediction dict to wrap. If None, uses best model.
            fold: Which fold's model to use for `model_` property (default: 0).

        Returns:
            NIRSPipeline instance ready for prediction.

        Raises:
            ValueError: If no predictions available in result.
            RuntimeError: If export fails.

        Example:
            >>> result = nirs4all.run(pipeline, dataset)
            >>> pipe = NIRSPipeline.from_result(result)
            >>> y_pred = pipe.predict(X_new)
        """
        import tempfile
        from nirs4all.pipeline.bundle import BundleLoader

        # Get source prediction
        if source is None:
            source = result.best
            if not source:
                raise ValueError(
                    "No predictions available in result. "
                    "Ensure nirs4all.run() completed successfully."
                )

        # Export to temporary bundle
        temp_dir = tempfile.mkdtemp(prefix="nirs4all_sklearn_")
        bundle_path = Path(temp_dir) / "model.n4a"

        try:
            result.export(bundle_path, source=source)
        except Exception as e:
            raise RuntimeError(f"Failed to export model to bundle: {e}") from e

        # Create instance from bundle
        instance = cls._from_bundle_internal(bundle_path, fold=fold)
        instance._runner = result._runner
        instance._prediction_source = source

        return instance

    @classmethod
    def from_bundle(cls, bundle_path: Union[str, Path], fold: int = 0) -> "NIRSPipeline":
        """Create NIRSPipeline from an exported .n4a bundle.

        Args:
            bundle_path: Path to the exported .n4a bundle file.
            fold: Which fold's model to use for `model_` property (default: 0).

        Returns:
            NIRSPipeline instance ready for prediction.

        Raises:
            FileNotFoundError: If bundle file doesn't exist.
            ValueError: If bundle is invalid or corrupted.

        Example:
            >>> pipe = NIRSPipeline.from_bundle("exports/model.n4a")
            >>> y_pred = pipe.predict(X_new)
        """
        return cls._from_bundle_internal(bundle_path, fold=fold)

    @classmethod
    def _from_bundle_internal(
        cls,
        bundle_path: Union[str, Path],
        fold: int = 0
    ) -> "NIRSPipeline":
        """Internal method to create NIRSPipeline from bundle.

        Args:
            bundle_path: Path to the .n4a bundle file.
            fold: Which fold's model to use.

        Returns:
            NIRSPipeline instance.
        """
        from nirs4all.pipeline.bundle import BundleLoader

        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        # Load the bundle
        loader = BundleLoader(bundle_path)

        # Create instance
        instance = cls()
        instance._bundle_loader = loader
        instance._is_fitted = True
        instance._fold = fold
        instance._source_path = bundle_path

        # Extract metadata
        if loader.metadata:
            instance._preprocessing_chain = loader.metadata.preprocessing_chain
            instance._model_step_index = loader.metadata.model_step_index
            instance._model_name = loader.metadata.original_manifest.get("name", "")

        instance._fold_weights = loader.fold_weights.copy()

        logger.debug(f"Created NIRSPipeline from bundle: {bundle_path}")

        return instance

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "NIRSPipeline":
        """Fit is not supported - use nirs4all.run() for training.

        NIRSPipeline is a prediction wrapper, not a training estimator.
        Training should be done with nirs4all.run(), then wrapped.

        Args:
            X: Ignored.
            y: Ignored.
            **fit_params: Ignored.

        Raises:
            NotImplementedError: Always, by design.

        Example:
            >>> # Correct workflow:
            >>> result = nirs4all.run(pipeline, dataset)  # Training
            >>> pipe = NIRSPipeline.from_result(result)   # Wrapping
            >>> y_pred = pipe.predict(X_new)              # Prediction
        """
        raise NotImplementedError(
            "NIRSPipeline.fit() is not supported. "
            "Training should be done with nirs4all.run(), then use "
            "NIRSPipeline.from_result(result) or NIRSPipeline.from_bundle(path) "
            "to wrap for sklearn compatibility."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature matrix (n_samples, n_features) as numpy array.

        Returns:
            Predicted values array (n_samples,).

        Raises:
            RuntimeError: If pipeline is not properly initialized.

        Example:
            >>> pipe = NIRSPipeline.from_bundle("model.n4a")
            >>> y_pred = pipe.predict(X_test)
        """
        self._check_is_fitted()

        # Ensure X is numpy array
        X = np.asarray(X)

        # Use bundle loader for prediction
        if self._bundle_loader is not None:
            return self._bundle_loader.predict(X)

        raise RuntimeError(
            "NIRSPipeline not properly initialized. "
            "Use NIRSPipeline.from_result() or NIRSPipeline.from_bundle()."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to data (without model prediction).

        This applies all preprocessing transformers but stops before the
        model step. Useful for getting base model predictions in stacking
        or for debugging preprocessing.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Transformed features (n_samples, n_transformed_features).

        Raises:
            RuntimeError: If pipeline is not properly initialized.
        """
        self._check_is_fitted()

        X = np.asarray(X)

        if self._bundle_loader is not None:
            # Apply transformers from bundle
            X_current = X.copy()

            if self._bundle_loader.trace:
                # Use trace for accurate step order
                steps = self._bundle_loader.trace.get_steps_up_to_model()
                for step in steps:
                    if step.operator_type in ("model", "meta_model"):
                        break
                    if step.operator_type == "y_processing":
                        continue

                    step_idx = step.step_index
                    if step.operator_type == "feature_augmentation":
                        X_current = self._bundle_loader._transform_feature_augmentation(
                            X_current, step_idx
                        )
                    else:
                        X_current = self._bundle_loader._transform_step(
                            X_current, step_idx
                        )
            else:
                # Fallback: apply all non-model artifacts
                model_step = self._model_step_index or 0
                for key in sorted(self._bundle_loader._artifact_index.keys()):
                    if key.startswith("step_"):
                        parts = key.split("_")
                        if len(parts) >= 2:
                            try:
                                step_idx = int(parts[1])
                                if step_idx < model_step:
                                    X_current = self._bundle_loader._transform_step(
                                        X_current, step_idx
                                    )
                            except ValueError:
                                pass

            return X_current

        raise RuntimeError("Pipeline not properly initialized")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score on test data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: True target values (n_samples,).

        Returns:
            R² score (coefficient of determination).

        Example:
            >>> pipe = NIRSPipeline.from_bundle("model.n4a")
            >>> r2 = pipe.score(X_test, y_test)
            >>> print(f"R²: {r2:.4f}")
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def _check_is_fitted(self) -> None:
        """Check if pipeline is fitted/initialized.

        Raises:
            RuntimeError: If pipeline is not properly initialized.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "NIRSPipeline is not fitted. Use NIRSPipeline.from_result() "
                "or NIRSPipeline.from_bundle() to create a fitted instance."
            )

    @property
    def is_fitted_(self) -> bool:
        """Whether the pipeline is fitted (always True for wrapped pipelines)."""
        return self._is_fitted

    @property
    def model_(self) -> Any:
        """Get the underlying model for SHAP access.

        Returns the model from the specified fold (default: fold 0).
        For tree-based models, this enables TreeExplainer.
        For neural networks, enables DeepExplainer.

        Returns:
            The fitted model object.

        Raises:
            RuntimeError: If model cannot be accessed.

        Example:
            >>> pipe = NIRSPipeline.from_bundle("model.n4a")
            >>> model = pipe.model_
            >>> explainer = shap.TreeExplainer(model)  # If tree-based
        """
        if self._cached_model is not None:
            return self._cached_model

        self._check_is_fitted()

        if self._bundle_loader is not None and self._bundle_loader.artifact_provider is not None:
            model_step = self._model_step_index

            if model_step is not None:
                # Try to get fold-specific model
                fold_artifacts = self._bundle_loader.artifact_provider.get_fold_artifacts(
                    model_step
                )
                if fold_artifacts:
                    # Find the requested fold
                    for fold_id, model in fold_artifacts:
                        if fold_id == self._fold:
                            self._cached_model = model
                            return model
                    # Fall back to first available fold
                    _, model = fold_artifacts[0]
                    self._cached_model = model
                    return model

                # Try single model (no CV)
                artifacts = self._bundle_loader.artifact_provider.get_artifacts_for_step(
                    model_step
                )
                if artifacts:
                    _, model = artifacts[0]
                    self._cached_model = model
                    return model

        raise RuntimeError(
            "Could not access underlying model. "
            "This may happen if the bundle doesn't contain model artifacts."
        )

    @property
    def shap_model(self) -> Any:
        """Alias for model_ for SHAP compatibility.

        Returns:
            The fitted model object.
        """
        return self.model_

    @property
    def bundle_loader_(self) -> Optional["BundleLoader"]:
        """Get the underlying BundleLoader (if created from bundle).

        Returns:
            BundleLoader instance or None.
        """
        return self._bundle_loader

    @property
    def preprocessing_chain(self) -> str:
        """Get string summary of preprocessing steps.

        Returns:
            Preprocessing chain description.
        """
        return self._preprocessing_chain

    @property
    def model_step_index(self) -> Optional[int]:
        """Get the index of the model step in the pipeline.

        Returns:
            Model step index or None.
        """
        return self._model_step_index

    @property
    def fold_weights(self) -> Dict[int, float]:
        """Get fold weights for CV ensemble.

        Returns:
            Dictionary mapping fold_id to weight.
        """
        return self._fold_weights.copy()

    @property
    def n_folds(self) -> int:
        """Get number of CV folds (0 if no CV).

        Returns:
            Number of folds.
        """
        return len(self._fold_weights)

    @property
    def model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name string.
        """
        return self._model_name

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn interface).

        Args:
            deep: If True, return nested parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "fold": self._fold,
        }

    def set_params(self, **params: Any) -> "NIRSPipeline":
        """Set parameters for this estimator (sklearn interface).

        Args:
            **params: Parameters to set. Only 'fold' is supported.

        Returns:
            self
        """
        if "fold" in params:
            self._fold = params["fold"]
            self._cached_model = None  # Invalidate cache
        return self

    def get_transformers(self) -> List[Tuple[str, Any]]:
        """Get list of preprocessing transformers.

        Returns:
            List of (name, transformer) tuples.

        Example:
            >>> pipe = NIRSPipeline.from_bundle("model.n4a")
            >>> for name, transformer in pipe.get_transformers():
            ...     print(f"{name}: {type(transformer).__name__}")
        """
        if self._cached_transformers is not None:
            return [(t.__class__.__name__, t) for t in self._cached_transformers]

        transformers = []

        if self._bundle_loader is not None and self._bundle_loader.artifact_provider is not None:
            model_step = self._model_step_index or 0

            # Collect transformers from steps before model
            for key in sorted(self._bundle_loader._artifact_index.keys()):
                if key.startswith("step_"):
                    parts = key.split("_")
                    if len(parts) >= 2:
                        try:
                            step_idx = int(parts[1])
                            if step_idx < model_step:
                                artifacts = self._bundle_loader.artifact_provider.get_artifacts_for_step(
                                    step_idx
                                )
                                for name, obj in artifacts:
                                    if hasattr(obj, "transform"):
                                        transformers.append((name, obj))
                        except ValueError:
                            pass

        return transformers

    def __repr__(self) -> str:
        """Return string representation."""
        if self._is_fitted:
            info_parts = ["fitted"]
            if self._model_name:
                info_parts.append(f"model='{self._model_name}'")
            if self._preprocessing_chain:
                info_parts.append(f"chain='{self._preprocessing_chain}'")
            return f"NIRSPipeline({', '.join(info_parts)})"
        return "NIRSPipeline(not fitted)"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        lines = ["NIRSPipeline"]
        if self._is_fitted:
            lines.append(f"  Status: fitted")
            if self._model_name:
                lines.append(f"  Model: {self._model_name}")
            if self._preprocessing_chain:
                lines.append(f"  Preprocessing: {self._preprocessing_chain}")
            if self._fold_weights:
                lines.append(f"  CV Folds: {len(self._fold_weights)}")
            if self._source_path:
                lines.append(f"  Source: {self._source_path}")
        else:
            lines.append("  Status: not fitted")
        return "\n".join(lines)
