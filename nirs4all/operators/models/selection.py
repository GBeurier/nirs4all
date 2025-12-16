"""Source model selection strategies for meta-model stacking.

This module provides flexible strategies for selecting which models
to use as sources in a stacking ensemble.

Available selectors:
    - AllPreviousModelsSelector: Use all models from previous steps (default)
    - ExplicitModelSelector: Use explicitly named models
    - TopKByMetricSelector: Use top K models by validation metric
    - DiversitySelector: Select diverse models by class type

Example:
    >>> from nirs4all.operators.models.selection import (
    ...     AllPreviousModelsSelector,
    ...     TopKByMetricSelector,
    ...     SelectorFactory
    ... )
    >>>
    >>> # Default: all previous models
    >>> selector = AllPreviousModelsSelector()
    >>>
    >>> # Top 3 by RMSE
    >>> selector = TopKByMetricSelector(k=3, metric="rmse")
    >>>
    >>> # Using factory
    >>> selector = SelectorFactory.create("top_k", k=5, metric="r2")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext


@dataclass
class ModelCandidate:
    """Information about a candidate source model.

    Contains metadata and optionally predictions for a model
    that may be selected as a source for stacking.

    Attributes:
        model_name: Name of the model.
        model_classname: Class name of the model (e.g., "PLSRegression").
        step_idx: Pipeline step index where the model was trained.
        fold_id: Fold identifier (or "avg"/"w_avg" for averaged models).
        branch_id: Branch identifier if in a branched pipeline.
        branch_name: Human-readable branch name.
        val_score: Validation score for the model.
        metric: Metric used for scoring.
        predictions: Optional dictionary with predictions data.
    """

    model_name: str
    model_classname: str
    step_idx: int
    fold_id: Optional[str] = None
    branch_id: Optional[int] = None
    branch_name: Optional[str] = None
    val_score: Optional[float] = None
    metric: Optional[str] = None
    predictions: Optional[Dict[str, np.ndarray]] = None


class SourceModelSelector(ABC):
    """Abstract base class for source model selection strategies.

    Defines the interface for selecting which models to include
    as sources in a stacking ensemble.

    Subclasses must implement the select() method to define
    their selection logic.

    Example:
        >>> class CustomSelector(SourceModelSelector):
        ...     def select(self, candidates, context, prediction_store):
        ...         # Custom selection logic
        ...         return [c for c in candidates if c.val_score > 0.9]
    """

    @abstractmethod
    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Select source models from candidates.

        Args:
            candidates: List of candidate models to select from.
            context: Execution context with current step and branch info.
            prediction_store: Predictions store for accessing model data.

        Returns:
            List of selected ModelCandidate objects in the order they should
            be used as features (determines column order in meta-features).
        """
        pass

    def validate(
        self,
        selected: List[ModelCandidate],
        context: 'ExecutionContext'
    ) -> None:
        """Validate the selection (optional override).

        Can raise ValueError if selection is invalid for the context.

        Args:
            selected: List of selected model candidates.
            context: Execution context for validation.

        Raises:
            ValueError: If selection is invalid.
        """
        if not selected:
            raise ValueError(
                "No source models selected for stacking. "
                "Ensure previous models exist in the pipeline."
            )


class AllPreviousModelsSelector(SourceModelSelector):
    """Select all models from previous steps in current branch.

    This is the default selector that includes all models trained
    before the meta-model step within the same branch context.

    Attributes:
        include_averaged: If True, include fold-averaged models.
            Default False (uses individual fold models).
        exclude_classnames: Set of model class names to exclude.

    Example:
        >>> selector = AllPreviousModelsSelector(include_averaged=True)
    """

    def __init__(
        self,
        include_averaged: bool = False,
        exclude_classnames: Optional[Set[str]] = None
    ):
        """Initialize selector.

        Args:
            include_averaged: If True, include 'avg' and 'w_avg' fold entries.
                Default False to use individual fold predictions for OOF.
            exclude_classnames: Optional set of model class names to exclude
                from selection.
        """
        self.include_averaged = include_averaged
        self.exclude_classnames = exclude_classnames or set()

    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Select all previous models in current branch.

        Args:
            candidates: List of candidate models.
            context: Execution context with step and branch info.
            prediction_store: Predictions store (unused in this selector).

        Returns:
            Filtered list of candidates ordered by step index.
        """
        current_step = context.state.step_number
        current_branch_id = getattr(context.selector, 'branch_id', None)

        selected = []
        for candidate in candidates:
            # Skip models from current or later steps
            if candidate.step_idx >= current_step:
                continue

            # Filter by branch if in a branch context
            if current_branch_id is not None:
                # Include models from same branch OR from before the branch (branch_id is None)
                if candidate.branch_id is not None and candidate.branch_id != current_branch_id:
                    continue

            # Skip averaged models if not requested
            if not self.include_averaged:
                if candidate.fold_id in ('avg', 'w_avg'):
                    continue

            # Skip excluded class names
            if candidate.model_classname in self.exclude_classnames:
                continue

            selected.append(candidate)

        # Sort by step index for consistent feature ordering
        selected.sort(key=lambda c: (c.step_idx, c.model_name, c.fold_id or ''))

        return selected


class ExplicitModelSelector(SourceModelSelector):
    """Select explicitly named models.

    Uses a predefined list of model names to select sources.
    Model names must match exactly (case-sensitive).

    Attributes:
        model_names: List of model names to select.
        strict: If True, raise error if any named model is not found.

    Example:
        >>> selector = ExplicitModelSelector(
        ...     model_names=["PLS", "RandomForest", "XGBoost"],
        ...     strict=True
        ... )
    """

    def __init__(
        self,
        model_names: List[str],
        strict: bool = True
    ):
        """Initialize selector with explicit model names.

        Args:
            model_names: List of model names to select.
            strict: If True, raise ValueError if any model is not found
                among candidates. Default True.
        """
        if not model_names:
            raise ValueError("model_names cannot be empty")
        self.model_names = model_names
        self.strict = strict

    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Select models matching the specified names.

        Args:
            candidates: List of candidate models.
            context: Execution context.
            prediction_store: Predictions store (unused).

        Returns:
            List of candidates matching specified names, in the order
            specified by model_names.

        Raises:
            ValueError: If strict=True and any model name is not found.
        """
        current_step = context.state.step_number
        current_branch_id = getattr(context.selector, 'branch_id', None)

        # Build a map of model_name -> candidates
        name_to_candidates: Dict[str, List[ModelCandidate]] = {}
        for candidate in candidates:
            # Skip models from current or later steps
            if candidate.step_idx >= current_step:
                continue

            # Filter by branch
            if current_branch_id is not None:
                if candidate.branch_id != current_branch_id:
                    continue

            if candidate.model_name not in name_to_candidates:
                name_to_candidates[candidate.model_name] = []
            name_to_candidates[candidate.model_name].append(candidate)

        # Select in order specified
        selected = []
        missing = []

        for name in self.model_names:
            if name in name_to_candidates:
                # Add all candidates with this name (all folds)
                candidates_for_name = name_to_candidates[name]
                # Sort by fold for consistency
                candidates_for_name.sort(key=lambda c: c.fold_id or '')
                selected.extend(candidates_for_name)
            else:
                missing.append(name)

        if self.strict and missing:
            raise ValueError(
                f"Source models not found in prediction store: {missing}. "
                f"Available models: {list(name_to_candidates.keys())}"
            )

        return selected


class TopKByMetricSelector(SourceModelSelector):
    """Select top K models by a validation metric.

    Ranks models by their validation score and selects the top K performers.

    Attributes:
        k: Number of top models to select.
        metric: Metric to rank by (e.g., "rmse", "r2", "accuracy").
        ascending: Sort direction. If None, inferred from metric.
        per_class: If True, select top K per model class (for diversity).

    Example:
        >>> selector = TopKByMetricSelector(k=3, metric="rmse", ascending=True)
    """

    # Metrics where lower is better
    _LOWER_IS_BETTER = {'rmse', 'mse', 'mae', 'mape', 'loss'}

    def __init__(
        self,
        k: int,
        metric: str = "val_score",
        ascending: Optional[bool] = None,
        per_class: bool = False
    ):
        """Initialize top-K selector.

        Args:
            k: Number of top models to select.
            metric: Metric name to rank by. If "val_score", uses the stored
                validation score.
            ascending: Sort order. If True, lower scores rank higher.
                If None, inferred from metric name.
            per_class: If True, select top K per unique model class name.
                This ensures diversity in the ensemble.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        self.k = k
        self.metric = metric
        self.ascending = ascending
        self.per_class = per_class

    def _infer_ascending(self, metric: str) -> bool:
        """Infer sort direction from metric name.

        Args:
            metric: Metric name.

        Returns:
            True if lower is better, False if higher is better.
        """
        metric_lower = metric.lower()
        return any(m in metric_lower for m in self._LOWER_IS_BETTER)

    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Select top K models by metric.

        Args:
            candidates: List of candidate models.
            context: Execution context.
            prediction_store: Predictions store (unused).

        Returns:
            Top K models sorted by metric.
        """
        current_step = context.state.step_number
        current_branch_id = getattr(context.selector, 'branch_id', None)

        # Filter candidates
        valid_candidates = []
        for candidate in candidates:
            if candidate.step_idx >= current_step:
                continue
            if current_branch_id is not None and candidate.branch_id != current_branch_id:
                continue
            # Skip averaged models for individual fold handling
            if candidate.fold_id in ('avg', 'w_avg'):
                continue
            if candidate.val_score is not None:
                valid_candidates.append(candidate)

        # Determine sort direction
        ascending = self.ascending
        if ascending is None:
            ascending = self._infer_ascending(self.metric)

        if self.per_class:
            # Select top K per class
            class_groups: Dict[str, List[ModelCandidate]] = {}
            for candidate in valid_candidates:
                classname = candidate.model_classname
                if classname not in class_groups:
                    class_groups[classname] = []
                class_groups[classname].append(candidate)

            selected = []
            for classname, group in class_groups.items():
                # Sort group by score
                group.sort(key=lambda c: c.val_score or float('inf'), reverse=not ascending)
                selected.extend(group[:self.k])

            # Final sort by score
            selected.sort(key=lambda c: c.val_score or float('inf'), reverse=not ascending)
        else:
            # Sort all candidates by score
            valid_candidates.sort(
                key=lambda c: c.val_score or float('inf'),
                reverse=not ascending
            )
            selected = valid_candidates[:self.k]

        return selected


class DiversitySelector(SourceModelSelector):
    """Select diverse models by class type to maximize ensemble diversity.

    Ensures the stacking ensemble includes different types of models
    rather than multiple similar models.

    Attributes:
        max_per_class: Maximum models per class type.
        preferred_classes: Optional list of preferred class names.

    Example:
        >>> selector = DiversitySelector(
        ...     max_per_class=2,
        ...     preferred_classes=["PLSRegression", "RandomForestRegressor"]
        ... )
    """

    def __init__(
        self,
        max_per_class: int = 1,
        preferred_classes: Optional[List[str]] = None
    ):
        """Initialize diversity selector.

        Args:
            max_per_class: Maximum number of models per class type.
            preferred_classes: Optional list of preferred model class names.
                These are selected first if available.
        """
        if max_per_class < 1:
            raise ValueError(f"max_per_class must be at least 1, got {max_per_class}")

        self.max_per_class = max_per_class
        self.preferred_classes = preferred_classes or []

    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Select diverse models by class type.

        Args:
            candidates: List of candidate models.
            context: Execution context.
            prediction_store: Predictions store (unused).

        Returns:
            List of diverse models with at most max_per_class per type.
        """
        current_step = context.state.step_number
        current_branch_id = getattr(context.selector, 'branch_id', None)

        # Filter and group by class
        class_groups: Dict[str, List[ModelCandidate]] = {}
        for candidate in candidates:
            if candidate.step_idx >= current_step:
                continue
            if current_branch_id is not None and candidate.branch_id != current_branch_id:
                continue
            if candidate.fold_id in ('avg', 'w_avg'):
                continue

            classname = candidate.model_classname
            if classname not in class_groups:
                class_groups[classname] = []
            class_groups[classname].append(candidate)

        selected = []

        # First, add preferred classes
        for classname in self.preferred_classes:
            if classname in class_groups:
                group = class_groups.pop(classname)
                # Sort by score (best first) and take up to max_per_class
                group.sort(key=lambda c: c.val_score or float('inf'))
                selected.extend(group[:self.max_per_class])

        # Then add remaining classes
        for classname, group in sorted(class_groups.items()):
            group.sort(key=lambda c: c.val_score or float('inf'))
            selected.extend(group[:self.max_per_class])

        # Sort final selection by step index for consistent feature ordering
        selected.sort(key=lambda c: (c.step_idx, c.model_name))

        return selected


class SelectorFactory:
    """Factory for creating source model selectors.

    Provides a convenient way to instantiate selectors by name.

    Example:
        >>> selector = SelectorFactory.create("all")
        >>> selector = SelectorFactory.create("explicit", model_names=["PLS", "RF"])
        >>> selector = SelectorFactory.create("top_k", k=5, metric="rmse")
    """

    _REGISTRY: Dict[str, type] = {
        'all': AllPreviousModelsSelector,
        'all_previous': AllPreviousModelsSelector,
        'explicit': ExplicitModelSelector,
        'top_k': TopKByMetricSelector,
        'topk': TopKByMetricSelector,
        'diversity': DiversitySelector,
        'diverse': DiversitySelector,
    }

    @classmethod
    def create(cls, selector_type: str, **kwargs) -> SourceModelSelector:
        """Create a selector by type name.

        Args:
            selector_type: Type name (e.g., "all", "explicit", "top_k", "diversity").
            **kwargs: Arguments passed to the selector constructor.

        Returns:
            SourceModelSelector instance.

        Raises:
            ValueError: If selector_type is not recognized.
        """
        selector_type_lower = selector_type.lower()
        if selector_type_lower not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown selector type: {selector_type}. "
                f"Available: {available}"
            )

        selector_class = cls._REGISTRY[selector_type_lower]
        return selector_class(**kwargs)

    @classmethod
    def register(cls, name: str, selector_class: type) -> None:
        """Register a custom selector type.

        Args:
            name: Name to register under.
            selector_class: Selector class (must inherit from SourceModelSelector).

        Raises:
            TypeError: If selector_class doesn't inherit from SourceModelSelector.
        """
        if not issubclass(selector_class, SourceModelSelector):
            raise TypeError(
                f"Selector class must inherit from SourceModelSelector, "
                f"got {selector_class.__name__}"
            )
        cls._REGISTRY[name.lower()] = selector_class
