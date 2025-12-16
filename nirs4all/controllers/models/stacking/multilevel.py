"""
Multi-Level Stacking Validator (Phase 7).

This module provides validation and level detection for multi-level stacking,
where meta-models can use predictions from other meta-models as sources.

The validator ensures:
1. No circular dependencies exist in the stacking hierarchy
2. Stacking levels don't exceed configured maximum
3. Level detection works correctly for AUTO mode
4. Source models from appropriate levels are selected

Stacking Hierarchy:
    Level 0: Base models (PLS, RF, XGBoost, Neural Networks, etc.)
    Level 1: First meta-models (stack on Level 0 only)
    Level 2: Second meta-models (stack on Level 0 + Level 1)
    Level 3: Third meta-models (stack on Level 0 + Level 1 + Level 2)

Example:
    >>> validator = MultiLevelValidator(prediction_store)
    >>> result = validator.validate_sources(
    ...     meta_model_name="FinalMeta",
    ...     source_candidates=candidates,
    ...     context=context
    ... )
    >>> if not result.is_valid:
    ...     raise CircularDependencyError(...)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import warnings

from .exceptions import (
    CircularDependencyError,
    MaxStackingLevelExceededError,
    InconsistentLevelError,
)

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.operators.models.selection import ModelCandidate


@dataclass
class ModelLevelInfo:
    """Information about a model's stacking level.

    Attributes:
        model_name: Name of the model.
        level: Stacking level (0 for base models, 1+ for meta-models).
        is_meta_model: Whether this is a meta-model.
        source_models: List of source model names (for meta-models).
        step_idx: Pipeline step index.
    """
    model_name: str
    level: int
    is_meta_model: bool
    source_models: List[str] = field(default_factory=list)
    step_idx: int = 0


@dataclass
class LevelValidationResult:
    """Result of multi-level stacking validation.

    Attributes:
        is_valid: Whether the validation passed.
        detected_level: The detected stacking level for the meta-model.
        source_levels: Dict mapping source model names to their levels.
        circular_dependencies: List of detected circular dependencies.
        warnings: List of warning messages.
        errors: List of error messages.
    """
    is_valid: bool = True
    detected_level: int = 1
    source_levels: Dict[str, int] = field(default_factory=dict)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False


class MultiLevelValidator:
    """Validates multi-level stacking configurations.

    Ensures that stacking hierarchies are valid, detects circular dependencies,
    and computes appropriate stacking levels for meta-models.

    Attributes:
        prediction_store: Predictions storage for analyzing model metadata.
        max_level: Maximum allowed stacking level.
        log_warnings: Whether to emit Python warnings.
    """

    # Meta-model class name patterns to detect
    META_MODEL_PATTERNS = {'MetaModel', 'StackingRegressor', 'StackingClassifier'}

    def __init__(
        self,
        prediction_store: 'Predictions',
        max_level: int = 3,
        log_warnings: bool = True
    ):
        """Initialize multi-level validator.

        Args:
            prediction_store: Predictions storage.
            max_level: Maximum allowed stacking level (default 3).
            log_warnings: Whether to emit Python warnings.
        """
        self.prediction_store = prediction_store
        self.max_level = max_level
        self.log_warnings = log_warnings

        # Cache for model level info
        self._level_cache: Dict[str, ModelLevelInfo] = {}

    def validate_sources(
        self,
        meta_model_name: str,
        source_candidates: List['ModelCandidate'],
        context: 'ExecutionContext',
        allow_meta_sources: bool = True
    ) -> LevelValidationResult:
        """Validate source models for a meta-model.

        Checks for circular dependencies and computes the appropriate
        stacking level based on source model levels.

        Args:
            meta_model_name: Name of the meta-model being validated.
            source_candidates: List of candidate source models.
            context: Execution context.
            allow_meta_sources: Whether to allow other meta-models as sources.

        Returns:
            LevelValidationResult with validation status and detected level.
        """
        result = LevelValidationResult()

        # Get unique source model names
        source_names = list(dict.fromkeys(c.model_name for c in source_candidates))

        if not source_names:
            result.add_warning("No source models provided for multi-level validation")
            return result

        # Build level info for all sources
        for name in source_names:
            level_info = self._get_model_level_info(name, context)
            result.source_levels[name] = level_info.level

            # Check if meta-model sources are allowed
            if level_info.is_meta_model and not allow_meta_sources:
                result.add_error(
                    f"Source model '{name}' is a meta-model but allow_meta_sources=False"
                )
                continue

        # Check for circular dependencies
        for name in source_names:
            cycle = self._detect_circular_dependency(
                meta_model_name, name, context, visited=set()
            )
            if cycle:
                result.circular_dependencies.append(cycle)
                result.add_error(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )

        if not result.is_valid:
            return result

        # Compute detected level
        max_source_level = max(result.source_levels.values()) if result.source_levels else 0
        result.detected_level = max_source_level + 1

        # Check against max level
        if result.detected_level > self.max_level:
            result.add_error(
                f"Detected level {result.detected_level} exceeds maximum {self.max_level}"
            )
            return result

        # Add info about detected level
        if max_source_level > 0:
            meta_sources = [n for n, l in result.source_levels.items() if l > 0]
            result.add_warning(
                f"Multi-level stacking detected (level {result.detected_level}). "
                f"Using meta-model sources: {meta_sources}"
            )

        # Emit warnings if configured
        if self.log_warnings:
            for warning in result.warnings:
                warnings.warn(warning)

        return result

    def detect_level(
        self,
        source_candidates: List['ModelCandidate'],
        context: 'ExecutionContext'
    ) -> int:
        """Detect the appropriate stacking level based on source models.

        Args:
            source_candidates: List of candidate source models.
            context: Execution context.

        Returns:
            Detected stacking level (1 if no meta-model sources, 2+ otherwise).
        """
        if not source_candidates:
            return 1

        max_level = 0
        for candidate in source_candidates:
            level_info = self._get_model_level_info(candidate.model_name, context)
            max_level = max(max_level, level_info.level)

        return max_level + 1

    def filter_by_level(
        self,
        candidates: List['ModelCandidate'],
        context: 'ExecutionContext',
        max_source_level: Optional[int] = None,
        exclude_meta_models: bool = False
    ) -> List['ModelCandidate']:
        """Filter source candidates by stacking level.

        Args:
            candidates: List of candidate source models.
            context: Execution context.
            max_source_level: Maximum allowed source level (None = no limit).
            exclude_meta_models: If True, exclude all meta-models from sources.

        Returns:
            Filtered list of candidates.
        """
        filtered = []

        for candidate in candidates:
            level_info = self._get_model_level_info(candidate.model_name, context)

            # Check meta-model exclusion
            if exclude_meta_models and level_info.is_meta_model:
                continue

            # Check level limit
            if max_source_level is not None and level_info.level > max_source_level:
                continue

            filtered.append(candidate)

        return filtered

    def _get_model_level_info(
        self,
        model_name: str,
        context: 'ExecutionContext'
    ) -> ModelLevelInfo:
        """Get or compute level info for a model.

        Args:
            model_name: Name of the model.
            context: Execution context.

        Returns:
            ModelLevelInfo with level and metadata.
        """
        # Check cache
        if model_name in self._level_cache:
            return self._level_cache[model_name]

        # Get model predictions to determine type
        preds = self.prediction_store.filter_predictions(
            model_name=model_name,
            load_arrays=False
        )

        if not preds:
            # Unknown model - assume base level
            info = ModelLevelInfo(
                model_name=model_name,
                level=0,
                is_meta_model=False,
                source_models=[],
                step_idx=0
            )
            self._level_cache[model_name] = info
            return info

        # Check if this is a meta-model based on class name
        first_pred = preds[0]
        classname = first_pred.get('model_classname', '')
        step_idx = first_pred.get('step_idx', 0)

        is_meta = any(
            pattern in classname
            for pattern in self.META_MODEL_PATTERNS
        )

        # Also check model name for MetaModel prefix
        if 'MetaModel' in model_name or 'Meta_' in model_name:
            is_meta = True

        if not is_meta:
            # Base model - level 0
            info = ModelLevelInfo(
                model_name=model_name,
                level=0,
                is_meta_model=False,
                source_models=[],
                step_idx=step_idx
            )
            self._level_cache[model_name] = info
            return info

        # Meta-model - need to find its source models
        source_models = self._find_source_models(model_name, step_idx, context)

        # Compute level based on source model levels
        if not source_models:
            level = 1  # Meta-model with unknown sources
        else:
            max_source_level = 0
            for source_name in source_models:
                source_info = self._get_model_level_info(source_name, context)
                max_source_level = max(max_source_level, source_info.level)
            level = max_source_level + 1

        info = ModelLevelInfo(
            model_name=model_name,
            level=level,
            is_meta_model=True,
            source_models=source_models,
            step_idx=step_idx
        )
        self._level_cache[model_name] = info
        return info

    def _find_source_models(
        self,
        meta_model_name: str,
        meta_step_idx: int,
        context: 'ExecutionContext'
    ) -> List[str]:
        """Find source models for a meta-model.

        Attempts to identify which models were used as sources for a meta-model
        by looking at predictions from earlier steps in the same branch.

        Args:
            meta_model_name: Name of the meta-model.
            meta_step_idx: Step index of the meta-model.
            context: Execution context.

        Returns:
            List of source model names.
        """
        branch_id = getattr(context.selector, 'branch_id', None)

        # Get all models from earlier steps
        all_preds = self.prediction_store.filter_predictions(
            load_arrays=False,
            branch_id=branch_id
        )

        # Filter to earlier steps
        earlier_preds = [
            p for p in all_preds
            if p.get('step_idx', 0) < meta_step_idx
        ]

        # Get unique model names
        source_models = list(dict.fromkeys(
            p.get('model_name', '')
            for p in earlier_preds
            if p.get('model_name')
        ))

        return source_models

    def _detect_circular_dependency(
        self,
        meta_model_name: str,
        source_name: str,
        context: 'ExecutionContext',
        visited: Set[str],
        path: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """Detect circular dependencies in the stacking hierarchy.

        Uses DFS to detect cycles in the dependency graph.

        Args:
            meta_model_name: Name of the meta-model being built.
            source_name: Name of the source model being checked.
            context: Execution context.
            visited: Set of already visited model names.
            path: Current path in the dependency graph.

        Returns:
            List of model names forming a cycle, or None if no cycle.
        """
        if path is None:
            path = [meta_model_name]

        # Direct self-reference
        if source_name == meta_model_name:
            return path + [source_name]

        # Already visited in this path
        if source_name in visited:
            return None  # Not a cycle for us

        # Check if source is in current path (cycle)
        if source_name in path:
            cycle_start = path.index(source_name)
            return path[cycle_start:] + [source_name]

        # Get source model info
        source_info = self._get_model_level_info(source_name, context)

        # If not a meta-model, no further dependencies
        if not source_info.is_meta_model:
            return None

        # Check source's sources recursively
        new_path = path + [source_name]
        new_visited = visited | {source_name}

        for sub_source in source_info.source_models:
            cycle = self._detect_circular_dependency(
                meta_model_name, sub_source, context, new_visited, new_path
            )
            if cycle:
                return cycle

        return None

    def clear_cache(self) -> None:
        """Clear the level info cache."""
        self._level_cache.clear()

    def get_all_levels(
        self,
        context: 'ExecutionContext'
    ) -> Dict[str, int]:
        """Get levels for all models in the prediction store.

        Args:
            context: Execution context.

        Returns:
            Dict mapping model names to their stacking levels.
        """
        all_preds = self.prediction_store.filter_predictions(load_arrays=False)
        model_names = list(dict.fromkeys(
            p.get('model_name', '') for p in all_preds if p.get('model_name')
        ))

        levels = {}
        for name in model_names:
            info = self._get_model_level_info(name, context)
            levels[name] = info.level

        return levels


def validate_multi_level_stacking(
    prediction_store: 'Predictions',
    meta_model_name: str,
    source_candidates: List['ModelCandidate'],
    context: 'ExecutionContext',
    max_level: int = 3,
    allow_meta_sources: bool = True
) -> LevelValidationResult:
    """Convenience function for validating multi-level stacking.

    Args:
        prediction_store: Predictions storage.
        meta_model_name: Name of the meta-model.
        source_candidates: List of candidate source models.
        context: Execution context.
        max_level: Maximum allowed stacking level.
        allow_meta_sources: Whether to allow meta-model sources.

    Returns:
        LevelValidationResult with validation status.
    """
    validator = MultiLevelValidator(
        prediction_store=prediction_store,
        max_level=max_level,
        log_warnings=True
    )
    return validator.validate_sources(
        meta_model_name=meta_model_name,
        source_candidates=source_candidates,
        context=context,
        allow_meta_sources=allow_meta_sources
    )


def detect_stacking_level(
    prediction_store: 'Predictions',
    source_candidates: List['ModelCandidate'],
    context: 'ExecutionContext'
) -> int:
    """Convenience function for detecting stacking level.

    Args:
        prediction_store: Predictions storage.
        source_candidates: List of candidate source models.
        context: Execution context.

    Returns:
        Detected stacking level.
    """
    validator = MultiLevelValidator(
        prediction_store=prediction_store,
        log_warnings=False
    )
    return validator.detect_level(source_candidates, context)
