"""
sklearn-compatible classification pipeline wrapper for nirs4all.

NIRSPipelineClassifier is the classification variant of NIRSPipeline,
providing ClassifierMixin compatibility for sklearn tools.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
import logging

import numpy as np

if TYPE_CHECKING:
    from nirs4all.api.result import RunResult

from .pipeline import NIRSPipeline

logger = logging.getLogger(__name__)


class NIRSPipelineClassifier(NIRSPipeline):
    """sklearn-compatible classifier wrapper for trained nirs4all pipelines.

    This is the classification variant of NIRSPipeline, providing
    ClassifierMixin compatibility (predict_proba, classes_).

    Construction:
        Use class methods to create instances:
        - NIRSPipelineClassifier.from_result(result): From a RunResult
        - NIRSPipelineClassifier.from_bundle(path): From an exported .n4a bundle

    Additional Attributes:
        classes_: Array of class labels.

    Additional Methods:
        predict_proba(X): Predict class probabilities.

    Example:
        >>> result = nirs4all.run(classification_pipeline, dataset)
        >>> clf = NIRSPipelineClassifier.from_result(result)
        >>> proba = clf.predict_proba(X_new)
        >>> print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
    """

    def __init__(self) -> None:
        """Private constructor - use from_result() or from_bundle() instead."""
        super().__init__()
        self._classes: Optional[np.ndarray] = None
        self._label_encoder: Optional[Any] = None

    @classmethod
    def from_result(
        cls,
        result: "RunResult",
        source: Optional[Dict[str, Any]] = None,
        fold: int = 0
    ) -> "NIRSPipelineClassifier":
        """Create NIRSPipelineClassifier from a RunResult.

        Args:
            result: RunResult from nirs4all.run() with a classification pipeline.
            source: Optional prediction dict to wrap. If None, uses best model.
            fold: Which fold's model to use (default: 0).

        Returns:
            NIRSPipelineClassifier instance ready for prediction.

        Example:
            >>> result = nirs4all.run(classification_pipeline, dataset)
            >>> clf = NIRSPipelineClassifier.from_result(result)
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
        temp_dir = tempfile.mkdtemp(prefix="nirs4all_sklearn_clf_")
        bundle_path = Path(temp_dir) / "model.n4a"

        try:
            result.export(bundle_path, source=source)
        except Exception as e:
            raise RuntimeError(f"Failed to export model to bundle: {e}") from e

        # Create instance from bundle
        instance = cls._from_bundle_internal_classifier(bundle_path, fold=fold)
        instance._runner = result._runner
        instance._prediction_source = source

        # Try to extract classes from prediction
        if "classes" in source:
            instance._classes = np.asarray(source["classes"])

        return instance

    @classmethod
    def from_bundle(
        cls,
        bundle_path: Union[str, Path],
        fold: int = 0
    ) -> "NIRSPipelineClassifier":
        """Create NIRSPipelineClassifier from an exported .n4a bundle.

        Args:
            bundle_path: Path to the exported .n4a bundle file.
            fold: Which fold's model to use (default: 0).

        Returns:
            NIRSPipelineClassifier instance ready for prediction.

        Example:
            >>> clf = NIRSPipelineClassifier.from_bundle("exports/classifier.n4a")
            >>> proba = clf.predict_proba(X_new)
        """
        return cls._from_bundle_internal_classifier(bundle_path, fold=fold)

    @classmethod
    def _from_bundle_internal_classifier(
        cls,
        bundle_path: Union[str, Path],
        fold: int = 0
    ) -> "NIRSPipelineClassifier":
        """Internal method to create NIRSPipelineClassifier from bundle.

        Args:
            bundle_path: Path to the .n4a bundle file.
            fold: Which fold's model to use.

        Returns:
            NIRSPipelineClassifier instance.
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

        # Try to get classes from the model
        instance._extract_classes()

        logger.debug(f"Created NIRSPipelineClassifier from bundle: {bundle_path}")

        return instance

    def _extract_classes(self) -> None:
        """Extract class labels from the underlying model."""
        try:
            model = self.model_
            if hasattr(model, 'classes_'):
                self._classes = np.asarray(model.classes_)
            elif hasattr(model, 'classes'):
                self._classes = np.asarray(model.classes)
        except (RuntimeError, AttributeError):
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predicted class labels (n_samples,).

        Example:
            >>> clf = NIRSPipelineClassifier.from_bundle("model.n4a")
            >>> y_pred = clf.predict(X_test)
        """
        self._check_is_fitted()

        X = np.asarray(X)

        # Use bundle loader for prediction
        if self._bundle_loader is not None:
            y_pred = self._bundle_loader.predict(X)

            # For classification, predictions might be probabilities
            # If so, convert to class labels
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-class probabilities - take argmax
                y_pred = np.argmax(y_pred, axis=1)
                if self._classes is not None:
                    y_pred = self._classes[y_pred]

            return y_pred

        raise RuntimeError(
            "NIRSPipelineClassifier not properly initialized. "
            "Use NIRSPipelineClassifier.from_result() or "
            "NIRSPipelineClassifier.from_bundle()."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Class probability matrix (n_samples, n_classes).

        Raises:
            RuntimeError: If model doesn't support predict_proba.

        Example:
            >>> clf = NIRSPipelineClassifier.from_bundle("model.n4a")
            >>> proba = clf.predict_proba(X_test)
            >>> print(f"Probability of class 0: {proba[:, 0]}")
        """
        self._check_is_fitted()

        X = np.asarray(X)

        # Try to get predict_proba from underlying model
        try:
            model = self.model_

            # First transform X through preprocessing
            X_transformed = self.transform(X)

            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_transformed)
            elif hasattr(model, 'predict_log_proba'):
                return np.exp(model.predict_log_proba(X_transformed))
            else:
                # Fall back to regular predict and convert to pseudo-probabilities
                y_pred = self._bundle_loader.predict(X)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Already probabilities
                    return y_pred
                else:
                    # Convert to one-hot style (not true probabilities)
                    logger.warning(
                        "Model doesn't support predict_proba. "
                        "Returning pseudo-probabilities based on predictions."
                    )
                    n_classes = len(self._classes) if self._classes is not None else 2
                    proba = np.zeros((len(X), n_classes))
                    for i, pred in enumerate(y_pred):
                        if self._classes is not None:
                            idx = np.where(self._classes == pred)[0]
                            if len(idx) > 0:
                                proba[i, idx[0]] = 1.0
                        else:
                            proba[i, int(pred)] = 1.0
                    return proba

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute class probabilities: {e}. "
                "The underlying model may not support probability predictions."
            ) from e

    @property
    def classes_(self) -> np.ndarray:
        """Get array of class labels.

        Returns:
            Array of unique class labels.

        Raises:
            RuntimeError: If classes cannot be determined.
        """
        if self._classes is not None:
            return self._classes

        # Try to get from model
        self._extract_classes()
        if self._classes is not None:
            return self._classes

        raise RuntimeError(
            "Could not determine class labels. "
            "Try setting classes manually via clf._classes = np.array([...])"
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score on test data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: True class labels (n_samples,).

        Returns:
            Accuracy score (fraction correctly classified).

        Example:
            >>> clf = NIRSPipelineClassifier.from_bundle("model.n4a")
            >>> accuracy = clf.score(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.4f}")
        """
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._is_fitted:
            info_parts = ["fitted"]
            if self._model_name:
                info_parts.append(f"model='{self._model_name}'")
            if self._classes is not None:
                info_parts.append(f"n_classes={len(self._classes)}")
            return f"NIRSPipelineClassifier({', '.join(info_parts)})"
        return "NIRSPipelineClassifier(not fitted)"
