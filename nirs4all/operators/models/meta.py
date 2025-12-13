"""Meta-model operator for stacking ensemble.

This module provides the MetaModel operator for building stacking ensembles
that use predictions from previously trained models as input features.

The meta-model trains on out-of-fold (OOF) predictions from base models
to prevent data leakage and overfitting.

Example:
    >>> from nirs4all.operators.models import MetaModel
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> pipeline = [
    ...     MinMaxScaler(),
    ...     KFold(n_splits=5),
    ...     PLSRegression(n_components=10),
    ...     RandomForestRegressor(n_estimators=100),
    ...     {"model": MetaModel(model=Ridge(), source_models="all")},
    ... ]
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .base import BaseModelOperator


class CoverageStrategy(Enum):
    """Strategy for handling partial coverage in OOF reconstruction.

    When some samples are missing predictions (e.g., from sample partitioning),
    this determines how to handle them.

    Attributes:
        STRICT: Raise error if any sample is missing predictions (default).
        DROP_INCOMPLETE: Drop samples missing any source model predictions.
        IMPUTE_ZERO: Fill missing predictions with zeros.
        IMPUTE_MEAN: Fill missing predictions with mean of available predictions.
        IMPUTE_FOLD_MEAN: Fill with mean from the same fold.
    """

    STRICT = "strict"
    DROP_INCOMPLETE = "drop_incomplete"
    IMPUTE_ZERO = "impute_zero"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_FOLD_MEAN = "impute_fold_mean"


class TestAggregation(Enum):
    """Strategy for aggregating test predictions from multiple folds.

    When base models are trained with cross-validation, each fold produces
    predictions for the test set. This determines how to combine them.

    Attributes:
        MEAN: Simple average across folds (default).
        WEIGHTED_MEAN: Weighted average by validation scores.
        BEST_FOLD: Use prediction from best-scoring fold only.
    """

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted"
    BEST_FOLD = "best"


class BranchScope(Enum):
    """Which branches to include as source models.

    Controls which branches' predictions are used for stacking when
    the pipeline contains branching.

    Attributes:
        CURRENT_ONLY: Only use models from the current branch (default).
        ALL_BRANCHES: Use models from all branches (requires compatible samples).
        SPECIFIED: Use explicit list from source_models parameter.
    """

    CURRENT_ONLY = "current_only"
    ALL_BRANCHES = "all_branches"
    SPECIFIED = "specified"


class StackingLevel(Enum):
    """Level of stacking in multi-level stacking architecture.

    Indicates where this meta-model sits in a stacking hierarchy.
    Used for validation and dependency tracking.

    Attributes:
        AUTO: Automatically detect level based on source models (default).
        LEVEL_1: First meta-level (stacks on base models only).
        LEVEL_2: Second meta-level (can stack on LEVEL_1 meta-models).
        LEVEL_3: Third meta-level (can stack on LEVEL_1 and LEVEL_2).
    """

    AUTO = "auto"
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


@dataclass
class StackingConfig:
    """Configuration for meta-model training set reconstruction.

    Controls how out-of-fold predictions are collected and processed
    to build the training features for the meta-model.

    Attributes:
        coverage_strategy: How to handle samples with missing predictions.
        test_aggregation: How to aggregate test predictions across folds.
        branch_scope: Which branches to include as source models.
        allow_no_cv: If True, allow stacking without cross-validation (with warning).
        min_coverage_ratio: Minimum ratio of source models required per sample.
        level: Stacking level for multi-level stacking (AUTO, LEVEL_1, LEVEL_2, LEVEL_3).
        allow_meta_sources: If True, allow other MetaModels as source models.
        max_level: Maximum allowed stacking level (for validation).

    Example:
        >>> config = StackingConfig(
        ...     coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
        ...     test_aggregation=TestAggregation.WEIGHTED_MEAN,
        ...     min_coverage_ratio=0.5,
        ...     level=StackingLevel.AUTO,
        ...     allow_meta_sources=True
        ... )
    """

    coverage_strategy: CoverageStrategy = CoverageStrategy.STRICT
    test_aggregation: TestAggregation = TestAggregation.MEAN
    branch_scope: BranchScope = BranchScope.CURRENT_ONLY
    allow_no_cv: bool = False
    min_coverage_ratio: float = 1.0
    level: StackingLevel = StackingLevel.AUTO
    allow_meta_sources: bool = True
    max_level: int = 3

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.min_coverage_ratio <= 1.0:
            raise ValueError(
                f"min_coverage_ratio must be between 0 and 1, got {self.min_coverage_ratio}"
            )

        if self.max_level < 1 or self.max_level > 10:
            raise ValueError(
                f"max_level must be between 1 and 10, got {self.max_level}"
            )

        # Convert string values to enums if needed
        if isinstance(self.coverage_strategy, str):
            self.coverage_strategy = CoverageStrategy(self.coverage_strategy)
        if isinstance(self.test_aggregation, str):
            self.test_aggregation = TestAggregation(self.test_aggregation)
        if isinstance(self.branch_scope, str):
            self.branch_scope = BranchScope(self.branch_scope)
        if isinstance(self.level, str):
            if self.level == "auto":
                self.level = StackingLevel.AUTO
            else:
                self.level = StackingLevel(int(self.level))
        if isinstance(self.level, int):
            self.level = StackingLevel(self.level)


class MetaModel(BaseModelOperator):
    """Wrapper for meta-model stacking using pipeline predictions.

    Creates a meta-learner that uses predictions from previously trained
    models in the pipeline as input features. Implements stacked generalization
    with proper out-of-fold prediction handling to prevent data leakage.

    The meta-model:
    1. Collects out-of-fold (OOF) predictions from specified source models
    2. Constructs training features from these predictions
    3. Trains on these features using the provided sklearn-compatible model
    4. For test data, aggregates source model predictions across folds

    Multi-Level Stacking (Phase 7):
    MetaModel supports multi-level stacking where meta-models can use predictions
    from other meta-models as sources. This enables hierarchical ensemble architectures:
    - Level 0: Base models (PLS, RF, XGBoost, etc.)
    - Level 1: First meta-models (stack on Level 0)
    - Level 2: Second meta-models (stack on Level 0 + Level 1)
    - Level 3: Third meta-models (stack on all previous levels)

    The level is auto-detected by default but can be explicitly set via
    stacking_config.level. Circular dependencies are automatically prevented.

    Attributes:
        model: Sklearn-compatible model to use as meta-learner.
        source_models: Which models to use as sources ("all" or list of names).
        use_proba: For classification, use probabilities instead of class predictions.
        stacking_config: Configuration for OOF reconstruction and multi-level stacking.
        selector: Optional custom source model selector.
        finetune_space: Optional hyperparameter search space for Optuna finetuning.

    Example:
        >>> # Basic usage - stack all previous models
        >>> MetaModel(model=Ridge())
        >>>
        >>> # Explicit source selection
        >>> MetaModel(
        ...     model=Ridge(),
        ...     source_models=["PLS", "RandomForest", "XGBoost"]
        ... )
        >>>
        >>> # Multi-level stacking
        >>> pipeline = [
        ...     KFold(n_splits=5),
        ...     PLSRegression(n_components=5),         # Level 0
        ...     RandomForestRegressor(),               # Level 0
        ...     {"model": MetaModel(model=Ridge())},   # Level 1 (auto-detected)
        ...     {"model": MetaModel(                   # Level 2 (uses Level 0 + Level 1)
        ...         model=Lasso(),
        ...         stacking_config=StackingConfig(level=StackingLevel.LEVEL_2)
        ...     )},
        ... ]
        >>>
        >>> # With probability features for classification
        >>> MetaModel(
        ...     model=LogisticRegression(),
        ...     use_proba=True
        ... )
        >>>
        >>> # With Optuna hyperparameter tuning
        >>> MetaModel(
        ...     model=Ridge(),
        ...     finetune_space={"model__alpha": (0.001, 100.0)}
        ... )

    Notes:
        - Source models must be from earlier steps in the pipeline
        - In branched pipelines, only models from the current branch are used by default
        - For sample_partitioner branches, stacking is done within each partition
        - Multi-level stacking supports up to 3 levels by default (configurable)
        - Circular dependencies are automatically detected and prevented
    """

    def __init__(
        self,
        model: Any,
        source_models: Union[str, List[str]] = "all",
        use_proba: bool = False,
        stacking_config: Optional[StackingConfig] = None,
        selector: Optional[Any] = None,
        name: Optional[str] = None,
        finetune_space: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MetaModel operator.

        Args:
            model: Sklearn-compatible model to use as meta-learner.
                Must implement fit() and predict() methods.
            source_models: Specifies which models to use as sources:
                - "all": Use all previous models in the pipeline (default)
                - List[str]: Explicit list of model names to use
            use_proba: For classification tasks, if True use class probabilities
                as features instead of class predictions. For binary classification,
                uses probability of positive class. For multiclass, uses all
                class probabilities. Default False.
            stacking_config: Configuration for OOF reconstruction and multi-level
                stacking. If None, uses default StackingConfig.
            selector: Optional SourceModelSelector instance for custom selection.
                If provided, overrides source_models parameter.
            name: Optional name for the meta-model. If None, uses model class name.
            finetune_space: Optional hyperparameter search space for Optuna.
                Keys should use 'model__param' format for meta-learner params.
                Example: {"model__alpha": (0.001, 100.0)}

        Raises:
            ValueError: If model doesn't have required fit/predict methods.
            ValueError: If source_models is not "all" or a list of strings.
        """
        # Validate model has required methods
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise ValueError(
                f"Model must have fit() and predict() methods, "
                f"got {type(model).__name__}"
            )

        # Validate source_models
        if source_models != "all" and not isinstance(source_models, list):
            raise ValueError(
                f"source_models must be 'all' or a list of model names, "
                f"got {type(source_models).__name__}"
            )

        if isinstance(source_models, list):
            if not all(isinstance(s, str) for s in source_models):
                raise ValueError("All source_models entries must be strings")

        self.model = model
        self.source_models = source_models
        self.use_proba = use_proba
        self.stacking_config = stacking_config or StackingConfig()
        self.selector = selector
        self._name = name
        self.finetune_space = finetune_space

        # Track detected level (will be set during execution)
        self._detected_level: Optional[int] = None

    def get_controller_type(self) -> str:
        """Return the type of controller that handles this operator.

        Returns:
            str: "meta" to indicate MetaModelController should handle this.
        """
        return "meta"

    @property
    def level(self) -> int:
        """Get the stacking level of this meta-model.

        Returns the detected level if AUTO, otherwise the configured level.

        Returns:
            int: Stacking level (1, 2, or 3).
        """
        if self.stacking_config.level == StackingLevel.AUTO:
            return self._detected_level or 1
        return self.stacking_config.level.value

    def get_finetune_params(self) -> Optional[Dict[str, Any]]:
        """Get finetuning parameters for Optuna optimization.

        Returns the finetune_space with proper formatting for the Optuna manager.

        Returns:
            Dict with finetune configuration or None if no finetuning configured.
        """
        if not self.finetune_space:
            return None

        return {
            'model_params': self.finetune_space,
            'n_trials': self.finetune_space.get('n_trials', 50),
            'approach': self.finetune_space.get('approach', 'grouped'),
            'eval_mode': self.finetune_space.get('eval_mode', 'best'),
            'verbose': self.finetune_space.get('verbose', 0),
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this operator.

        Parameters:
            deep: If True, returns nested parameters from the model.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            'model': self.model,
            'source_models': self.source_models,
            'use_proba': self.use_proba,
            'stacking_config': self.stacking_config,
            'selector': self.selector,
            'name': self._name,
            'finetune_space': self.finetune_space,
        }

        if deep and hasattr(self.model, 'get_params'):
            model_params = self.model.get_params(deep=True)
            for key, value in model_params.items():
                params[f'model__{key}'] = value

        return params

    def set_params(self, **params) -> 'MetaModel':
        """Set the parameters of this operator.

        Parameters:
            **params: Operator parameters. Supports nested parameters
                for the model using 'model__param_name' syntax.

        Returns:
            self: MetaModel instance.
        """
        # Separate model params from operator params
        model_params = {}
        operator_params = {}

        for key, value in params.items():
            if key.startswith('model__'):
                model_params[key[7:]] = value  # Strip 'model__' prefix
            else:
                operator_params[key] = value

        # Set operator params
        for key, value in operator_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == 'name':
                self._name = value
            else:
                raise ValueError(f"Unknown parameter: {key}")

        # Set model params
        if model_params and hasattr(self.model, 'set_params'):
            self.model.set_params(**model_params)

        return self

    @property
    def name(self) -> str:
        """Get the display name for this meta-model.

        Returns:
            str: User-provided name or 'MetaModel_<model_class>'.
        """
        if self._name:
            return self._name
        return f"MetaModel_{type(self.model).__name__}"

    def __repr__(self) -> str:
        """Return string representation."""
        source_str = (
            self.source_models if isinstance(self.source_models, str)
            else f"[{', '.join(self.source_models[:3])}{'...' if len(self.source_models) > 3 else ''}]"
        )
        return (
            f"MetaModel(model={type(self.model).__name__}, "
            f"source_models={source_str}, "
            f"use_proba={self.use_proba})"
        )
