"""
Classification Support for Meta-Model Stacking.

Phase 5 Implementation - Provides utilities for:
1. Detecting classification vs regression task types from predictions
2. Extracting probability features for classification stacking
3. Handling binary and multiclass classification scenarios
4. Generating meaningful feature names with class information

Key components:
- ClassificationFeatureExtractor: Extracts probability features from predictions
- TaskTypeDetector: Detects task type from prediction metadata
- FeatureNameGenerator: Creates descriptive feature names for meta-features
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import warnings

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext


class StackingTaskType(Enum):
    """Task type for stacking.

    Attributes:
        REGRESSION: Regression task using y_pred as features.
        BINARY_CLASSIFICATION: Binary classification (2 classes).
        MULTICLASS_CLASSIFICATION: Multi-class classification (>2 classes).
        UNKNOWN: Could not determine task type.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    UNKNOWN = "unknown"

    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task type."""
        return self in (
            StackingTaskType.BINARY_CLASSIFICATION,
            StackingTaskType.MULTICLASS_CLASSIFICATION
        )

    @property
    def n_classes(self) -> Optional[int]:
        """Return expected number of classes or None for regression."""
        if self == StackingTaskType.BINARY_CLASSIFICATION:
            return 2
        elif self == StackingTaskType.MULTICLASS_CLASSIFICATION:
            return None  # Variable
        return None


@dataclass
class ClassificationInfo:
    """Information about classification task detected from predictions.

    Attributes:
        task_type: Detected task type (regression/binary/multiclass).
        n_classes: Number of classes if classification, else None.
        class_labels: Optional class labels if available.
        has_probabilities: Whether y_proba is available in predictions.
        proba_shape: Shape of probability arrays if available.
    """

    task_type: StackingTaskType
    n_classes: Optional[int] = None
    class_labels: Optional[List[Any]] = None
    has_probabilities: bool = False
    proba_shape: Optional[Tuple[int, ...]] = None

    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task."""
        return self.task_type.is_classification

    @property
    def is_binary(self) -> bool:
        """Check if this is binary classification."""
        return self.task_type == StackingTaskType.BINARY_CLASSIFICATION

    @property
    def is_multiclass(self) -> bool:
        """Check if this is multiclass classification."""
        return self.task_type == StackingTaskType.MULTICLASS_CLASSIFICATION

    def get_n_features_per_model(self, use_proba: bool = False) -> int:
        """Get number of features per source model.

        Args:
            use_proba: Whether probability features are requested.

        Returns:
            Number of feature columns per source model.
            - Regression: 1 (y_pred)
            - Binary + use_proba: 1 (positive class probability)
            - Multiclass + use_proba: n_classes (all class probabilities)
            - Classification without use_proba: 1 (y_pred)
        """
        if not use_proba or not self.is_classification:
            return 1

        if self.is_binary:
            return 1  # Only positive class probability

        if self.is_multiclass and self.n_classes:
            return self.n_classes

        return 1  # Fallback


class TaskTypeDetector:
    """Detects task type from prediction metadata.

    Uses prediction store metadata and y_proba presence to determine
    whether the stacking involves regression or classification.
    """

    def __init__(self, prediction_store: 'Predictions'):
        """Initialize detector.

        Args:
            prediction_store: Predictions storage with metadata.
        """
        self.prediction_store = prediction_store

    def detect(
        self,
        source_model_names: List[str],
        context: 'ExecutionContext'
    ) -> ClassificationInfo:
        """Detect task type from source model predictions.

        Examines predictions from source models to determine task type
        and gather classification metadata.

        Args:
            source_model_names: List of source model names to examine.
            context: Execution context with branch info.

        Returns:
            ClassificationInfo with detected task type and metadata.
        """
        branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        # Check each source model for task type info
        task_types_found = []
        n_classes_found = []
        has_proba = False
        proba_shape = None

        for model_name in source_model_names:
            info = self._get_model_task_info(
                model_name, branch_id, current_step
            )
            if info:
                task_types_found.append(info['task_type'])
                if info.get('n_classes'):
                    n_classes_found.append(info['n_classes'])
                if info.get('has_proba'):
                    has_proba = True
                    if info.get('proba_shape'):
                        proba_shape = info['proba_shape']

        # Determine overall task type
        task_type = self._resolve_task_type(task_types_found)

        # Determine number of classes
        n_classes = None
        if n_classes_found:
            n_classes = max(n_classes_found)  # Use max to handle any inconsistency

        return ClassificationInfo(
            task_type=task_type,
            n_classes=n_classes,
            class_labels=None,  # Could be extracted from metadata if available
            has_probabilities=has_proba,
            proba_shape=proba_shape
        )

    def _get_model_task_info(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int
    ) -> Optional[Dict[str, Any]]:
        """Get task type info for a single model.

        Args:
            model_name: Name of the source model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).

        Returns:
            Dictionary with task_type, n_classes, has_proba, proba_shape
            or None if no predictions found.
        """
        filter_kwargs = {
            'model_name': model_name,
            'partition': 'val',  # Check validation predictions
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        predictions = [p for p in predictions if p.get('step_idx', 0) < max_step]

        # Filter out averaged predictions
        predictions = [
            p for p in predictions
            if str(p.get('fold_id', '')) not in {'avg', 'w_avg'}
        ]

        if not predictions:
            return None

        # Take first prediction for task type info
        pred = predictions[0]

        task_type_str = pred.get('task_type', 'regression')
        task_type = self._string_to_task_type(task_type_str)

        # Check for probabilities
        y_proba = pred.get('y_proba')
        has_proba = y_proba is not None and (
            hasattr(y_proba, 'size') and y_proba.size > 0
        )

        proba_shape = None
        n_classes = None

        if has_proba:
            y_proba = np.asarray(y_proba)
            proba_shape = y_proba.shape
            if y_proba.ndim == 2:
                n_classes = y_proba.shape[1]
            elif y_proba.ndim == 1:
                n_classes = 2  # Binary with single probability column
        elif task_type.is_classification:
            # Try to infer n_classes from y_true/y_pred
            y_true = pred.get('y_true')
            if y_true is not None:
                y_true = np.asarray(y_true)
                n_classes = len(np.unique(y_true))

        return {
            'task_type': task_type,
            'n_classes': n_classes,
            'has_proba': has_proba,
            'proba_shape': proba_shape
        }

    def _string_to_task_type(self, task_type_str: str) -> StackingTaskType:
        """Convert task type string to StackingTaskType enum.

        Args:
            task_type_str: Task type string from predictions.

        Returns:
            StackingTaskType enum value.
        """
        task_type_str = task_type_str.lower()

        if 'binary' in task_type_str:
            return StackingTaskType.BINARY_CLASSIFICATION
        elif 'multiclass' in task_type_str or 'classification' in task_type_str:
            # Need to check more carefully
            if 'binary' not in task_type_str:
                return StackingTaskType.MULTICLASS_CLASSIFICATION
            return StackingTaskType.BINARY_CLASSIFICATION
        elif 'regression' in task_type_str:
            return StackingTaskType.REGRESSION

        return StackingTaskType.UNKNOWN

    def _resolve_task_type(
        self,
        task_types: List[StackingTaskType]
    ) -> StackingTaskType:
        """Resolve conflicting task types from multiple models.

        Args:
            task_types: List of task types from different models.

        Returns:
            Resolved task type.

        Note:
            All source models should have the same task type.
            If mixed, we warn and use the most common.
        """
        if not task_types:
            return StackingTaskType.UNKNOWN

        # Filter out unknown
        known_types = [t for t in task_types if t != StackingTaskType.UNKNOWN]

        if not known_types:
            return StackingTaskType.UNKNOWN

        # Check for consistency
        unique_types = set(known_types)

        if len(unique_types) == 1:
            return known_types[0]

        # Mixed types - warn and use most common
        from collections import Counter
        counter = Counter(known_types)
        most_common = counter.most_common(1)[0][0]

        warnings.warn(
            f"Mixed task types detected in source models: {unique_types}. "
            f"Using most common: {most_common.value}. "
            f"All source models should have the same task type for proper stacking."
        )

        return most_common


class ClassificationFeatureExtractor:
    """Extracts classification features from predictions.

    Handles extraction of probability features for binary and multiclass
    classification, with proper handling of different array shapes.
    """

    def __init__(
        self,
        classification_info: ClassificationInfo,
        use_proba: bool = False
    ):
        """Initialize extractor.

        Args:
            classification_info: Classification metadata.
            use_proba: Whether to extract probability features.
        """
        self.classification_info = classification_info
        self.use_proba = use_proba

    def extract_features(
        self,
        pred: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Extract features from a single prediction entry.

        Args:
            pred: Prediction dictionary with y_pred and optionally y_proba.
            n_samples: Expected number of samples.

        Returns:
            Feature array of shape (n_samples,) or (n_samples, n_features).
        """
        if self.use_proba and self.classification_info.is_classification:
            return self._extract_proba_features(pred, n_samples)
        else:
            return self._extract_pred_features(pred, n_samples)

    def _extract_pred_features(
        self,
        pred: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Extract y_pred as features.

        Args:
            pred: Prediction dictionary.
            n_samples: Expected number of samples.

        Returns:
            1D array of predictions.
        """
        y_pred = pred.get('y_pred', [])
        y_pred = np.asarray(y_pred).flatten()

        if len(y_pred) != n_samples:
            # Pad or truncate
            result = np.full(n_samples, np.nan)
            result[:min(len(y_pred), n_samples)] = y_pred[:n_samples]
            return result

        return y_pred

    def _extract_proba_features(
        self,
        pred: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Extract probability features.

        For binary classification: returns positive class probability (1 column).
        For multiclass: returns all class probabilities (n_classes columns).

        Args:
            pred: Prediction dictionary.
            n_samples: Expected number of samples.

        Returns:
            Array of shape (n_samples,) for binary or (n_samples, n_classes) for multiclass.
        """
        y_proba = pred.get('y_proba')

        # Fallback to y_pred if no probabilities
        if y_proba is None or (hasattr(y_proba, 'size') and y_proba.size == 0):
            warnings.warn(
                f"use_proba=True but no y_proba available for model "
                f"{pred.get('model_name', 'unknown')}. Falling back to y_pred."
            )
            return self._extract_pred_features(pred, n_samples)

        y_proba = np.asarray(y_proba)

        if self.classification_info.is_binary:
            return self._extract_binary_proba(y_proba, n_samples)
        else:
            return self._extract_multiclass_proba(y_proba, n_samples)

    def _extract_binary_proba(
        self,
        y_proba: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Extract binary classification probability.

        Returns probability of the positive class (class 1).

        Args:
            y_proba: Probability array.
            n_samples: Expected number of samples.

        Returns:
            1D array of positive class probabilities.
        """
        if y_proba.ndim == 1:
            # Already 1D - assume it's positive class probability
            proba_1d = y_proba
        elif y_proba.ndim == 2:
            if y_proba.shape[1] == 2:
                # Standard (n_samples, 2) shape - take positive class
                proba_1d = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                # Single column - treat as positive class
                proba_1d = y_proba[:, 0]
            else:
                # More than 2 classes - should be multiclass, take positive
                warnings.warn(
                    f"Expected binary probabilities but got shape {y_proba.shape}. "
                    f"Using column 1 as positive class."
                )
                proba_1d = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            raise ValueError(f"Unexpected y_proba shape: {y_proba.shape}")

        # Handle size mismatch
        if len(proba_1d) != n_samples:
            result = np.full(n_samples, np.nan)
            result[:min(len(proba_1d), n_samples)] = proba_1d[:n_samples]
            return result

        return proba_1d

    def _extract_multiclass_proba(
        self,
        y_proba: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Extract multiclass classification probabilities.

        Returns all class probabilities as separate features.

        Args:
            y_proba: Probability array.
            n_samples: Expected number of samples.

        Returns:
            2D array of shape (n_samples, n_classes).
        """
        if y_proba.ndim == 1:
            # Convert 1D to 2D for consistency
            # Assume binary if 1D
            proba_2d = np.column_stack([1 - y_proba, y_proba])
        elif y_proba.ndim == 2:
            proba_2d = y_proba
        else:
            raise ValueError(f"Unexpected y_proba shape: {y_proba.shape}")

        # Handle size mismatch
        if proba_2d.shape[0] != n_samples:
            n_classes = proba_2d.shape[1]
            result = np.full((n_samples, n_classes), np.nan)
            n_copy = min(proba_2d.shape[0], n_samples)
            result[:n_copy, :] = proba_2d[:n_copy, :]
            return result

        return proba_2d

    def get_n_features(self) -> int:
        """Get number of features that will be extracted per model.

        Returns:
            Number of feature columns per source model.
        """
        return self.classification_info.get_n_features_per_model(self.use_proba)


class FeatureNameGenerator:
    """Generates meaningful feature names for meta-model.

    Creates descriptive feature names that include model name and,
    for classification with probabilities, class information.
    """

    def __init__(
        self,
        classification_info: ClassificationInfo,
        use_proba: bool = False,
        pattern: str = "{model_name}_pred"
    ):
        """Initialize generator.

        Args:
            classification_info: Classification metadata.
            use_proba: Whether probability features are used.
            pattern: Base pattern for feature names.
        """
        self.classification_info = classification_info
        self.use_proba = use_proba
        self.pattern = pattern

    def generate_names(
        self,
        source_model_names: List[str]
    ) -> List[str]:
        """Generate feature names for all source models.

        Args:
            source_model_names: List of source model names.

        Returns:
            List of feature column names.
        """
        names = []

        for model_name in source_model_names:
            model_names = self._generate_model_names(model_name)
            names.extend(model_names)

        return names

    def _generate_model_names(self, model_name: str) -> List[str]:
        """Generate feature names for a single source model.

        Args:
            model_name: Source model name.

        Returns:
            List of feature names (1 for regression, may be more for classification).
        """
        if not self.use_proba or not self.classification_info.is_classification:
            # Single prediction feature
            return [self._format_name(model_name, suffix="_pred")]

        if self.classification_info.is_binary:
            # Single probability feature (positive class)
            return [self._format_name(model_name, suffix="_proba_1")]

        # Multiclass - one feature per class
        n_classes = self.classification_info.n_classes or 2
        names = []
        for class_idx in range(n_classes):
            names.append(
                self._format_name(model_name, suffix=f"_proba_{class_idx}")
            )
        return names

    def _format_name(self, model_name: str, suffix: str = "") -> str:
        """Format a single feature name.

        Args:
            model_name: Source model name.
            suffix: Suffix to append (only used with default pattern).

        Returns:
            Formatted feature name.
        """
        # Use simple format if pattern is default
        default_pattern = "{model_name}_pred"
        if self.pattern == default_pattern:
            return f"{model_name}{suffix}"

        # Custom pattern: use pattern as-is, only add class suffix for multiclass
        try:
            base = self.pattern.format(model_name=model_name)
            # Only append class suffix for multiclass proba (e.g., _proba_0, _proba_1)
            if suffix.startswith("_proba_") and suffix != "_proba_1":
                return f"{base}{suffix}"
            return base
        except KeyError:
            return f"{model_name}{suffix}"

    def get_feature_importance_mapping(
        self,
        source_model_names: List[str]
    ) -> Dict[str, List[str]]:
        """Get mapping from source models to their feature names.

        Useful for feature importance analysis.

        Args:
            source_model_names: List of source model names.

        Returns:
            Dictionary mapping model name to list of feature names.
        """
        mapping = {}

        for model_name in source_model_names:
            feature_names = self._generate_model_names(model_name)
            mapping[model_name] = feature_names

        return mapping


@dataclass
class MetaFeatureInfo:
    """Information about generated meta-features.

    Used for tracking feature importance and providing interpretable results.

    Attributes:
        feature_names: List of all feature column names.
        source_models: List of source model names.
        feature_to_model: Mapping from feature name to source model.
        classification_info: Classification metadata.
        n_features_per_model: Number of features from each model.
    """

    feature_names: List[str]
    source_models: List[str]
    feature_to_model: Dict[str, str]
    classification_info: ClassificationInfo
    n_features_per_model: Dict[str, int] = field(default_factory=dict)

    def get_model_for_feature(self, feature_name: str) -> Optional[str]:
        """Get source model name for a feature.

        Args:
            feature_name: Feature column name.

        Returns:
            Source model name or None if not found.
        """
        return self.feature_to_model.get(feature_name)

    def aggregate_importance_by_model(
        self,
        feature_importances: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate feature importances by source model.

        Sums importance scores for all features from the same source model.

        Args:
            feature_importances: Mapping from feature name to importance score.

        Returns:
            Mapping from model name to aggregated importance.
        """
        model_importance = {model: 0.0 for model in self.source_models}

        for feature_name, importance in feature_importances.items():
            model_name = self.get_model_for_feature(feature_name)
            if model_name is not None:
                model_importance[model_name] += importance

        return model_importance


def build_meta_feature_info(
    source_model_names: List[str],
    classification_info: ClassificationInfo,
    use_proba: bool = False,
    name_pattern: str = "{model_name}_pred"
) -> MetaFeatureInfo:
    """Build MetaFeatureInfo from source models and classification info.

    Args:
        source_model_names: List of source model names.
        classification_info: Classification metadata.
        use_proba: Whether probability features are used.
        name_pattern: Pattern for feature names.

    Returns:
        MetaFeatureInfo with all mappings populated.
    """
    generator = FeatureNameGenerator(
        classification_info=classification_info,
        use_proba=use_proba,
        pattern=name_pattern
    )

    feature_names = generator.generate_names(source_model_names)
    feature_to_model_mapping = generator.get_feature_importance_mapping(source_model_names)

    # Invert to get feature->model mapping
    feature_to_model = {}
    n_features_per_model = {}

    for model_name, features in feature_to_model_mapping.items():
        n_features_per_model[model_name] = len(features)
        for feature in features:
            feature_to_model[feature] = model_name

    return MetaFeatureInfo(
        feature_names=feature_names,
        source_models=source_model_names,
        feature_to_model=feature_to_model,
        classification_info=classification_info,
        n_features_per_model=n_features_per_model
    )
