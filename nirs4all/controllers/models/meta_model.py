"""
MetaModel Controller - Controller for meta-model stacking.

This controller handles MetaModel operators by:
1. Collecting out-of-fold (OOF) predictions from source models
2. Constructing training features from these predictions
3. Training the meta-learner on these features
4. Storing predictions with proper metadata for serialization

The controller prevents data leakage by using only validation partition
predictions from each fold to construct the training set.

Phase 2 Enhancement: Delegates OOF reconstruction to TrainingSetReconstructor
for cleaner separation of concerns and more robust coverage handling.

Phase 3 Enhancement: Implements prediction mode with dependency resolution
and meta-model artifact persistence with source model references.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import warnings

from .base_model import BaseModelController
from .sklearn_model import SklearnModelController
from .stacking import (
    TrainingSetReconstructor,
    ReconstructionResult,
    MetaModelArtifact,
    MetaModelSerializer,
    SourceModelReference,
    stacking_config_to_dict,
    stacking_config_from_dict,
    # Phase 4 - Branch Validation
    BranchValidator,
    BranchType,
    BranchValidationResult,
    StackingCompatibility,
    detect_branch_type,
    is_stacking_compatible,
    # Phase 7 - Multi-Level Stacking
    MultiLevelValidator,
    ModelLevelInfo,
    LevelValidationResult,
    # Phase 7 - Cross-Branch Stacking
    CrossBranchValidator,
    CrossBranchValidationResult,
    CrossBranchCompatibility,
    BranchPredictionInfo,
    # Exceptions
    MetaModelPredictionError,
    MissingSourceModelError,
    SourcePredictionError,
    FeatureOrderMismatchError,
    BranchMismatchError,
    NoSourcePredictionsError,
    # Phase 4 - Branching Exceptions
    IncompatibleBranchTypeError,
    CrossPartitionStackingError,
    NestedBranchStackingError,
    DisjointSampleSetsError,
    # Phase 7 - Multi-Level Stacking Exceptions
    CircularDependencyError,
    MaxStackingLevelExceededError,
    InconsistentLevelError,
    # Phase 7 - Cross-Branch Stacking Exceptions
    IncompatibleBranchSamplesError,
    BranchFeatureAlignmentError,
)
from .stacking.config import ReconstructorConfig
from nirs4all.controllers.registry import register_controller
from nirs4all.operators.models.meta import (
    MetaModel,
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
    StackingLevel,
)
from nirs4all.operators.models.selection import (
    SourceModelSelector,
    AllPreviousModelsSelector,
    ExplicitModelSelector,
    ModelCandidate,
    SelectorFactory,
)
from nirs4all.utils.emoji import WARNING, CHECK, SEARCH
from nirs4all.pipeline.storage.artifacts.types import ArtifactType, MetaModelConfig

# Define STACK emoji if not available
STACK = "📚"

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRecord


@register_controller
class MetaModelController(SklearnModelController):
    """Controller for meta-model stacking using pipeline predictions.

    This controller handles MetaModel operators, constructing training features
    from out-of-fold predictions of previous models. It extends SklearnModelController
    since the meta-learner is always sklearn-compatible.

    The key difference from regular model controllers is that get_xy() returns
    features constructed from predictions rather than the original dataset features.

    Attributes:
        priority: Controller priority (5) - higher than SklearnModelController (6)
            to ensure MetaModel operators are handled by this controller.
    """

    priority = 5  # Higher priority than SklearnModelController (6)

    # Enable Phase 2 TrainingSetReconstructor (set to True to use new implementation)
    use_reconstructor: bool = True

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match MetaModel operators.

        Args:
            step: Pipeline step configuration.
            operator: Instantiated operator object.
            keyword: Pipeline keyword (unused).

        Returns:
            True if the operator is a MetaModel instance.
        """
        # Check operator first (most common case)
        if isinstance(operator, MetaModel):
            return True

        # Check step dict for MetaModel
        if isinstance(step, dict):
            model = step.get('model')
            if isinstance(model, MetaModel):
                return True

        return False

    def _get_meta_operator(self, step: Any, operator: Any) -> MetaModel:
        """Extract MetaModel operator from step configuration.

        Args:
            step: Pipeline step configuration.
            operator: Instantiated operator object.

        Returns:
            MetaModel operator instance.

        Raises:
            ValueError: If MetaModel cannot be extracted.
        """
        if isinstance(operator, MetaModel):
            return operator
        if isinstance(step, dict) and isinstance(step.get('model'), MetaModel):
            return step['model']
        raise ValueError("Could not extract MetaModel from step configuration")

    def _get_model_instance(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        force_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get the underlying meta-learner model instance.

        Extracts the sklearn-compatible model from the MetaModel wrapper.

        Args:
            dataset: SpectroDataset for context.
            model_config: Model configuration dictionary.
            force_params: Optional parameters to override.

        Returns:
            The underlying sklearn-compatible model.
        """
        # Extract MetaModel operator
        meta_operator = model_config.get('model_instance')
        if isinstance(meta_operator, MetaModel):
            model = meta_operator.model
        elif isinstance(model_config.get('model'), MetaModel):
            model = model_config['model'].model
        else:
            model = meta_operator

        # Apply force_params if provided and model supports it
        if force_params and model is not None and hasattr(model, 'set_params'):
            model.set_params(**force_params)

        return model

    def get_xy(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext'
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Extract train/test splits using meta-features from predictions.

        Instead of using the original dataset features, this constructs
        features from out-of-fold predictions of source models.

        For training:
            - X_train: OOF predictions from source models (n_train_samples, n_source_models)
            - y_train: Original target values

        For test:
            - X_test: Aggregated source model test predictions
            - y_test: Original target values

        Args:
            dataset: SpectroDataset with partitioned data.
            context: Execution context with partition and branch info.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled)
            where X_train and X_test are meta-features from predictions.
        """
        # Get mode from context
        mode = context.state.mode

        # Get original y values from parent implementation
        # We need y but will replace X with meta-features
        # Use explicit parent class call to satisfy type checker
        (
            _X_train_orig, y_train,
            _X_test_orig, y_test,
            y_train_unscaled, y_test_unscaled
        ) = SklearnModelController.get_xy(self, dataset, context)

        # In prediction/explain mode, we need to handle this differently
        # The prediction_store should have predictions from source models run in this session
        if mode in ("predict", "explain"):
            # For prediction mode, we'll build meta-features from current session predictions
            X_train_meta = np.array([]).reshape(0, 1)  # Empty for prediction mode
            X_test_meta = self._build_test_features_from_predictions(dataset, context)
        else:
            # Training mode: use TrainingSetReconstructor if enabled
            if self.use_reconstructor:
                result = self._reconstruct_with_reconstructor(
                    dataset, context, y_train, y_test
                )
                if result is not None:
                    # Store reconstruction result for later use
                    context.custom['_reconstruction_result'] = result
                    return (
                        result.X_train_meta, result.y_train,
                        result.X_test_meta, result.y_test,
                        y_train_unscaled, y_test_unscaled
                    )

            # Fallback to Phase 1 implementation
            X_train_meta = self._build_oof_features_from_predictions(dataset, context)
            X_test_meta = self._build_test_features_from_predictions(dataset, context)

        return X_train_meta, y_train, X_test_meta, y_test, y_train_unscaled, y_test_unscaled

    def _reconstruct_with_reconstructor(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Optional[ReconstructionResult]:
        """Use TrainingSetReconstructor for OOF feature construction.

        This is the Phase 2 implementation that provides cleaner separation
        of concerns and more robust coverage/validation handling.

        Phase 4 Enhancement: Includes comprehensive branch validation for
        preprocessing branches, sample partitioner, outlier excluder, and
        generator syntax.

        Phase 7 Enhancement: Includes multi-level stacking validation (circular
        dependency detection, level consistency) and cross-branch stacking
        validation (sample alignment, feature alignment).

        Args:
            dataset: SpectroDataset for sample indices.
            context: Execution context.
            y_train: Pre-computed training targets.
            y_test: Pre-computed test targets.

        Returns:
            ReconstructionResult with meta-features, or None if fallback needed.

        Raises:
            IncompatibleBranchTypeError: If branch type is not compatible.
            CrossPartitionStackingError: If cross-partition stacking attempted.
            NestedBranchStackingError: If nested branching too deep.
            CircularDependencyError: If circular dependencies detected.
            MaxStackingLevelExceededError: If stacking level exceeds max_level.
        """
        prediction_store = context.custom.get('_prediction_store')
        if prediction_store is None:
            runtime_context = context.custom.get('_runtime_context')
            if runtime_context is not None:
                prediction_store = getattr(runtime_context, 'prediction_store', None)

        if prediction_store is None:
            return None

        try:
            meta_operator = self._get_meta_operator_from_context(context)
        except ValueError:
            return None

        # Get source model names
        source_models = self._get_source_models(meta_operator, context, prediction_store)
        if not source_models:
            return None

        unique_source_names = list(dict.fromkeys(m.model_name for m in source_models))
        verbose = getattr(self, 'verbose', 0)
        stacking_config = meta_operator.stacking_config

        # Phase 7: Multi-Level Stacking Validation
        self._validate_multi_level_stacking(
            prediction_store, source_models, meta_operator, context, stacking_config, verbose
        )

        # Phase 7: Cross-Branch Stacking Validation (for ALL_BRANCHES scope)
        self._validate_cross_branch_stacking(
            prediction_store, source_models, context, stacking_config, verbose
        )

        # Phase 4: Branch Validation
        self._validate_branch_context(
            prediction_store, dataset, unique_source_names, context, verbose
        )

        # Create and run reconstructor
        return self._run_reconstructor(
            prediction_store, dataset, unique_source_names,
            meta_operator, context, y_train, y_test, verbose
        )

    def _validate_multi_level_stacking(
        self,
        prediction_store: 'Predictions',
        source_models: List['ModelCandidate'],
        meta_operator: 'MetaModel',
        context: 'ExecutionContext',
        stacking_config: 'StackingConfig',
        verbose: int
    ) -> None:
        """Validate multi-level stacking configuration.

        Args:
            prediction_store: Predictions storage.
            source_models: List of source model candidates.
            meta_operator: MetaModel operator for name extraction.
            context: Execution context.
            stacking_config: Stacking configuration.
            verbose: Verbosity level.

        Raises:
            CircularDependencyError: If circular dependencies detected.
            MaxStackingLevelExceededError: If level exceeds max.
            InconsistentLevelError: If levels are inconsistent.
        """
        multi_level_validator = MultiLevelValidator(
            prediction_store=prediction_store,
            max_level=stacking_config.max_level,
        )

        # Get meta-model name from the operator
        meta_model_name = getattr(meta_operator, 'name', None) or 'MetaModel'

        level_result = multi_level_validator.validate_sources(
            meta_model_name=meta_model_name,
            source_candidates=source_models,
            context=context,
            allow_meta_sources=stacking_config.allow_meta_sources,
        )

        unique_source_names = list(dict.fromkeys(m.model_name for m in source_models))

        if not level_result.is_valid:
            self._raise_multi_level_error(level_result, stacking_config, unique_source_names)

        context.custom['_detected_stacking_level'] = level_result.detected_level

        if verbose > 0 and level_result.detected_level > 1:
            print(f"  {STACK} Multi-level stacking detected: level {level_result.detected_level}")

    def _raise_multi_level_error(
        self,
        level_result: 'LevelValidationResult',
        stacking_config: 'StackingConfig',
        unique_source_names: List[str]
    ) -> None:
        """Raise appropriate error based on multi-level validation result."""
        for error_msg in level_result.errors:
            if "circular" in error_msg.lower():
                raise CircularDependencyError(
                    dependency_chain=level_result.circular_dependencies or [unique_source_names[0]],
                )
            elif "level" in error_msg.lower() and "exceeded" in error_msg.lower():
                raise MaxStackingLevelExceededError(
                    current_level=level_result.detected_level,
                    max_level=stacking_config.max_level,
                    model_name=unique_source_names[0] if unique_source_names else "unknown",
                )
            elif "inconsistent" in error_msg.lower():
                expected = stacking_config.level.value if stacking_config.level != StackingLevel.AUTO else level_result.detected_level
                raise InconsistentLevelError(
                    model_name=unique_source_names[0] if unique_source_names else "unknown",
                    expected_level=expected,
                    actual_level=level_result.detected_level,
                )

    def _validate_cross_branch_stacking(
        self,
        prediction_store: 'Predictions',
        source_candidates: List['ModelCandidate'],
        context: 'ExecutionContext',
        stacking_config: 'StackingConfig',
        verbose: int
    ) -> None:
        """Validate cross-branch stacking for ALL_BRANCHES scope.

        Args:
            prediction_store: Predictions storage.
            source_candidates: List of source model candidates.
            context: Execution context.
            stacking_config: Stacking configuration.
            verbose: Verbosity level.

        Raises:
            IncompatibleBranchSamplesError: If branches have incompatible samples.
            DisjointSampleSetsError: If folds are incompatible.
            BranchFeatureAlignmentError: If features can't be aligned.
        """
        if stacking_config.branch_scope != BranchScope.ALL_BRANCHES:
            return

        cross_branch_validator = CrossBranchValidator(
            prediction_store=prediction_store,
        )

        cross_branch_result = cross_branch_validator.validate_cross_branch_stacking(
            source_candidates=source_candidates,
            context=context,
        )

        if not cross_branch_result.is_compatible:
            self._raise_cross_branch_error(cross_branch_result)

        if cross_branch_result.compatibility == CrossBranchCompatibility.COMPATIBLE_WITH_ALIGNMENT:
            if verbose > 0:
                print(f"  {WARNING} Cross-branch stacking requires feature alignment")

        # Store aligned sources for later use
        aligned_sources = cross_branch_validator.get_cross_branch_sources(
            source_candidates=source_candidates,
            context=context,
        )
        context.custom['_cross_branch_sources'] = aligned_sources

    def _raise_cross_branch_error(self, cross_branch_result: 'CrossBranchValidationResult') -> None:
        """Raise appropriate error based on cross-branch validation result."""
        if cross_branch_result.compatibility == CrossBranchCompatibility.INCOMPATIBLE_SAMPLES:
            raise IncompatibleBranchSamplesError(
                branches=list(cross_branch_result.branch_info.keys()),
                sample_overlap=cross_branch_result.sample_overlap_ratio,
            )
        elif cross_branch_result.compatibility == CrossBranchCompatibility.INCOMPATIBLE_FOLDS:
            raise DisjointSampleSetsError(
                branch_a=str(cross_branch_result.branch_info.get('branch_a', {}).get('branch_id', 'A')),
                branch_b=str(cross_branch_result.branch_info.get('branch_b', {}).get('branch_id', 'B')),
                sample_ids_a=set(),
                sample_ids_b=set(),
            )
        elif cross_branch_result.compatibility == CrossBranchCompatibility.INCOMPATIBLE_FEATURES:
            raise BranchFeatureAlignmentError(
                branches=list(cross_branch_result.branch_info.keys()),
                feature_counts={k: v.n_features for k, v in cross_branch_result.branch_info.items()},
            )

    def _validate_branch_context(
        self,
        prediction_store: 'Predictions',
        dataset: 'SpectroDataset',
        unique_source_names: List[str],
        context: 'ExecutionContext',
        verbose: int
    ) -> None:
        """Validate branch context for stacking compatibility.

        Args:
            prediction_store: Predictions storage.
            dataset: SpectroDataset for sample indices.
            unique_source_names: List of unique source model names.
            context: Execution context.
            verbose: Verbosity level.

        Raises:
            CrossPartitionStackingError: If cross-partition stacking attempted.
            NestedBranchStackingError: If nested branching too deep.
            IncompatibleBranchTypeError: If branch type is not compatible.
        """
        branch_type = detect_branch_type(context)

        if branch_type == BranchType.NONE:
            return

        branch_validator = BranchValidator(
            prediction_store=prediction_store,
            log_warnings=True
        )

        branch_result = branch_validator.validate(
            context=context,
            source_model_names=unique_source_names,
            dataset=dataset
        )

        if verbose > 0:
            print(f"  {SEARCH} Branch type: {branch_type.value}, "
                  f"compatibility: {branch_result.compatibility.value}")

        if not branch_result.is_valid:
            self._raise_branch_error(branch_result, branch_type)

        for warning in branch_result.warnings:
            if verbose > 0:
                print(f"  {WARNING} {warning}")
            warnings.warn(warning)

        if branch_type == BranchType.SAMPLE_PARTITIONER:
            self._validate_sample_alignment(
                branch_validator, dataset, unique_source_names, context
            )

    def _raise_branch_error(
        self,
        branch_result: 'BranchValidationResult',
        branch_type: 'BranchType'
    ) -> None:
        """Raise appropriate error based on branch validation result."""
        for error in branch_result.errors:
            if "cross-partition" in error.lower() or "disjoint" in error.lower():
                raise CrossPartitionStackingError(
                    partition_a=branch_result.branch_info.branch_name or "current",
                    partition_b="other",
                    n_samples_a=branch_result.branch_info.n_samples or 0,
                    n_samples_b=0
                )
            elif "nested" in error.lower() or "depth" in error.lower():
                raise NestedBranchStackingError(
                    branch_depth=branch_result.branch_info.nesting_depth,
                    branch_path=branch_result.branch_info.branch_path,
                    reason=error
                )
            else:
                raise IncompatibleBranchTypeError(
                    branch_type=branch_type.value,
                    reason=error,
                    suggestions=["Check branch configuration",
                                 "Use explicit source_models parameter"]
                )

    def _validate_sample_alignment(
        self,
        branch_validator: 'BranchValidator',
        dataset: 'SpectroDataset',
        unique_source_names: List[str],
        context: 'ExecutionContext'
    ) -> None:
        """Validate sample alignment for sample_partitioner branches."""
        train_context = context.with_partition('train')
        train_sample_ids = dataset._indexer.x_indices(
            train_context.selector,
            include_augmented=True,
            include_excluded=False
        )
        sample_result = branch_validator.validate_sample_alignment(
            source_model_names=unique_source_names,
            expected_sample_indices=list(train_sample_ids),
            context=context
        )
        if not sample_result.is_valid:
            for error in sample_result.errors:
                warnings.warn(f"Sample alignment warning: {error}")

    def _run_reconstructor(
        self,
        prediction_store: 'Predictions',
        dataset: 'SpectroDataset',
        unique_source_names: List[str],
        meta_operator: MetaModel,
        context: 'ExecutionContext',
        y_train: np.ndarray,
        y_test: np.ndarray,
        verbose: int
    ) -> ReconstructionResult:
        """Run the TrainingSetReconstructor.

        Args:
            prediction_store: Predictions storage.
            dataset: SpectroDataset for sample indices.
            unique_source_names: List of unique source model names.
            meta_operator: MetaModel operator.
            context: Execution context.
            y_train: Pre-computed training targets.
            y_test: Pre-computed test targets.
            verbose: Verbosity level.

        Returns:
            ReconstructionResult with meta-features.
        """
        reconstructor_config = ReconstructorConfig(
            validate_fold_alignment=True,
            validate_sample_coverage=True,
            log_warnings=True,
        )

        reconstructor = TrainingSetReconstructor(
            prediction_store=prediction_store,
            source_model_names=unique_source_names,
            stacking_config=meta_operator.stacking_config,
            reconstructor_config=reconstructor_config,
        )

        branch_validation = reconstructor.validate_branch_compatibility(context)
        for warning in branch_validation.warnings:
            if reconstructor_config.log_warnings:
                warnings.warn(f"[{warning.code}] {warning.message}")

        result = reconstructor.reconstruct(
            dataset=dataset,
            context=context,
            y_train=y_train,
            y_test=y_test,
            use_proba=meta_operator.use_proba,
        )

        if verbose > 0:
            print(f"  {CHECK} OOF reconstruction: {result.n_folds} folds, "
                  f"{len(result.source_models)} sources, "
                  f"coverage={result.coverage_ratio:.1%}")

        if not result.validation_result.is_valid:
            for error in result.validation_result.errors:
                warnings.warn(f"[{error.code}] {error.message}")

        return result

    def _build_oof_features_from_predictions(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext'
    ) -> np.ndarray:
        """Build training features from out-of-fold predictions.

        Collects validation partition predictions from each fold to construct
        a training set where each sample's feature comes from a fold where
        it was NOT used for training.

        Args:
            dataset: SpectroDataset for context.
            context: Execution context with prediction_store access.

        Returns:
            np.ndarray of shape (n_train_samples, n_source_models * n_outputs)
        """
        prediction_store = context.custom.get('_prediction_store')
        if prediction_store is None:
            # Try to get from runtime context
            runtime_context = context.custom.get('_runtime_context')
            if runtime_context is not None:
                prediction_store = getattr(runtime_context, 'prediction_store', None)

        if prediction_store is None:
            raise ValueError(
                "prediction_store not available in context. "
                "MetaModel requires access to predictions from previous steps."
            )

        # Get MetaModel configuration from context
        meta_operator = self._get_meta_operator_from_context(context)
        stacking_config = meta_operator.stacking_config

        # Get source models using selector
        source_models = self._get_source_models(meta_operator, context, prediction_store)

        if not source_models:
            raise ValueError(
                "No source models found for stacking. "
                "Ensure previous model steps exist before MetaModel."
            )

        # Get training sample indices
        train_context = context.with_partition('train')
        train_sample_ids = dataset._indexer.x_indices(
            train_context.selector, include_augmented=True, include_excluded=False
        )
        n_samples = len(train_sample_ids)

        # Initialize feature matrix
        # Group source models by name to determine feature count
        unique_source_names = list(dict.fromkeys(m.model_name for m in source_models))
        n_features = len(unique_source_names)

        X_meta = np.full((n_samples, n_features), np.nan)

        # Build sample_id to position mapping
        id_to_pos = {int(sid): pos for pos, sid in enumerate(train_sample_ids)}

        # Collect OOF predictions for each source model
        for feat_idx, model_name in enumerate(unique_source_names):
            oof_preds = self._collect_oof_predictions_for_model(
                model_name, prediction_store, context, id_to_pos, n_samples,
                meta_operator.use_proba
            )
            X_meta[:, feat_idx] = oof_preds

        # Handle coverage based on stacking config
        X_meta = self._handle_coverage(X_meta, stacking_config, n_samples)

        return X_meta

    def _collect_oof_predictions_for_model(
        self,
        model_name: str,
        prediction_store: 'Predictions',
        context: 'ExecutionContext',
        id_to_pos: Dict[int, int],
        n_samples: int,
        use_proba: bool = False
    ) -> np.ndarray:
        """Collect out-of-fold predictions for a single source model.

        Args:
            model_name: Name of the source model.
            prediction_store: Predictions storage.
            context: Execution context.
            id_to_pos: Mapping from sample ID to position.
            n_samples: Total number of samples.
            use_proba: If True, use probability predictions.

        Returns:
            np.ndarray of OOF predictions for this model.
        """
        oof_preds = np.full(n_samples, np.nan)

        # Get current branch info
        branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        # Filter predictions for this model from validation partition
        filter_kwargs = {
            'model_name': model_name,
            'partition': 'val',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        val_predictions = prediction_store.filter_predictions(**filter_kwargs)

        # Filter to only include predictions from steps before current
        val_predictions = [
            p for p in val_predictions
            if p.get('step_idx', 0) < current_step
        ]

        # Collect predictions from each fold
        for pred in val_predictions:
            fold_id = pred.get('fold_id')
            # Skip averaged predictions
            if fold_id in ('avg', 'w_avg'):
                continue

            sample_indices = pred.get('sample_indices', [])
            if use_proba and 'y_proba' in pred and pred['y_proba'] is not None:
                y_vals = pred['y_proba']
                # For binary, use positive class probability
                if y_vals.ndim > 1 and y_vals.shape[1] == 2:
                    y_vals = y_vals[:, 1]
                elif y_vals.ndim > 1:
                    # For multiclass, use first class (or could average)
                    y_vals = y_vals[:, 0]
            else:
                y_vals = pred.get('y_pred', [])

            # Flatten if needed
            if hasattr(y_vals, 'flatten'):
                y_vals = y_vals.flatten()

            # Place predictions at correct positions
            for i, sample_idx in enumerate(sample_indices):
                if i < len(y_vals):
                    pos = id_to_pos.get(int(sample_idx))
                    if pos is not None:
                        oof_preds[pos] = y_vals[i]

        return oof_preds

    def _build_test_features_from_predictions(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext'
    ) -> np.ndarray:
        """Build test features from aggregated source model predictions.

        For test data, aggregates predictions across folds according to
        the configured TestAggregation strategy.

        Args:
            dataset: SpectroDataset for context.
            context: Execution context.

        Returns:
            np.ndarray of shape (n_test_samples, n_source_models * n_outputs)
        """
        prediction_store = context.custom.get('_prediction_store')
        if prediction_store is None:
            runtime_context = context.custom.get('_runtime_context')
            if runtime_context is not None:
                prediction_store = getattr(runtime_context, 'prediction_store', None)

        if prediction_store is None:
            raise ValueError("prediction_store not available for test feature construction")

        meta_operator = self._get_meta_operator_from_context(context)

        # Get source models
        source_models = self._get_source_models(meta_operator, context, prediction_store)
        unique_source_names = list(dict.fromkeys(m.model_name for m in source_models))

        if not unique_source_names:
            raise ValueError("No source models found for test feature construction")

        # Get test sample count from predictions
        branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        # Get a sample prediction to determine test size
        filter_kwargs = {
            'model_name': unique_source_names[0],
            'partition': 'test',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        test_preds = prediction_store.filter_predictions(**filter_kwargs)
        test_preds = [p for p in test_preds if p.get('step_idx', 0) < current_step]

        if not test_preds:
            # Return empty array if no test predictions
            return np.array([]).reshape(0, len(unique_source_names))

        # Get n_test from first prediction
        first_pred = test_preds[0]
        n_test = len(first_pred.get('y_pred', []))

        # Initialize feature matrix
        X_test_meta = np.zeros((n_test, len(unique_source_names)))

        # Collect test predictions for each source model
        for feat_idx, model_name in enumerate(unique_source_names):
            test_pred = self._aggregate_test_predictions_for_model(
                model_name, prediction_store, context, n_test,
                meta_operator.use_proba, meta_operator.stacking_config.test_aggregation
            )
            X_test_meta[:, feat_idx] = test_pred

        return X_test_meta

    def _aggregate_test_predictions_for_model(
        self,
        model_name: str,
        prediction_store: 'Predictions',
        context: 'ExecutionContext',
        n_samples: int,
        use_proba: bool,
        aggregation: TestAggregation
    ) -> np.ndarray:
        """Aggregate test predictions for a source model across folds.

        Args:
            model_name: Name of the source model.
            prediction_store: Predictions storage.
            context: Execution context.
            n_samples: Number of test samples.
            use_proba: Use probability predictions if available.
            aggregation: Aggregation strategy.

        Returns:
            np.ndarray of aggregated test predictions.
        """
        branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        filter_kwargs = {
            'model_name': model_name,
            'partition': 'test',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        test_preds = prediction_store.filter_predictions(**filter_kwargs)
        test_preds = [p for p in test_preds if p.get('step_idx', 0) < current_step]

        # Handle averaged predictions if using MEAN/WEIGHTED_MEAN
        if aggregation in (TestAggregation.MEAN, TestAggregation.WEIGHTED_MEAN):
            # Look for pre-computed averages
            avg_preds = [p for p in test_preds if p.get('fold_id') == 'avg']
            if avg_preds and aggregation == TestAggregation.MEAN:
                pred = avg_preds[0]
                y_vals = pred.get('y_pred', np.zeros(n_samples))
                return np.asarray(y_vals).flatten()

            w_avg_preds = [p for p in test_preds if p.get('fold_id') == 'w_avg']
            if w_avg_preds and aggregation == TestAggregation.WEIGHTED_MEAN:
                pred = w_avg_preds[0]
                y_vals = pred.get('y_pred', np.zeros(n_samples))
                return np.asarray(y_vals).flatten()

        # Filter to individual folds only
        fold_preds = [
            p for p in test_preds
            if p.get('fold_id') not in ('avg', 'w_avg', None)
        ]

        if not fold_preds:
            return np.zeros(n_samples)

        # Collect all fold predictions
        all_preds = []
        all_scores = []
        for pred in fold_preds:
            if use_proba and 'y_proba' in pred and pred['y_proba'] is not None:
                y_vals = pred['y_proba']
                if y_vals.ndim > 1 and y_vals.shape[1] == 2:
                    y_vals = y_vals[:, 1]
                elif y_vals.ndim > 1:
                    y_vals = y_vals[:, 0]
            else:
                y_vals = pred.get('y_pred', [])

            y_vals = np.asarray(y_vals).flatten()
            if len(y_vals) == n_samples:
                all_preds.append(y_vals)
                all_scores.append(pred.get('val_score', 0.0))

        if not all_preds:
            return np.zeros(n_samples)

        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        if aggregation == TestAggregation.BEST_FOLD:
            # Use predictions from best fold
            best_idx = np.argmax(all_scores) if np.any(all_scores) else 0
            return all_preds[best_idx]
        elif aggregation == TestAggregation.WEIGHTED_MEAN:
            # Weighted average by validation scores
            weights = np.clip(all_scores, 0, None)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(all_preds)) / len(all_preds)
            return np.average(all_preds, axis=0, weights=weights)
        else:
            # Simple mean (default)
            return np.mean(all_preds, axis=0)

    def _get_source_models(
        self,
        meta_operator: MetaModel,
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Get list of source models using the configured selector.

        Args:
            meta_operator: MetaModel operator with configuration.
            context: Execution context.
            prediction_store: Predictions storage.

        Returns:
            List of ModelCandidate objects for source models.
        """
        # Build candidates from prediction store
        candidates = self._build_candidates_from_predictions(context, prediction_store)

        # Get or create selector
        if meta_operator.selector is not None:
            selector = meta_operator.selector
        elif meta_operator.source_models == "all":
            selector = AllPreviousModelsSelector(include_averaged=False)
        elif isinstance(meta_operator.source_models, list):
            selector = ExplicitModelSelector(
                model_names=meta_operator.source_models,
                strict=True
            )
        else:
            # Fallback to all models
            selector = AllPreviousModelsSelector(include_averaged=False)

        # Apply selection
        selected = selector.select(candidates, context, prediction_store)
        selector.validate(selected, context)

        return selected

    def _build_candidates_from_predictions(
        self,
        context: 'ExecutionContext',
        prediction_store: 'Predictions'
    ) -> List[ModelCandidate]:
        """Build ModelCandidate list from prediction store.

        Args:
            context: Execution context.
            prediction_store: Predictions storage.

        Returns:
            List of ModelCandidate objects.
        """
        candidates = []

        # Get all predictions (metadata only for speed)
        all_preds = prediction_store.filter_predictions(load_arrays=False)

        seen = set()  # Track unique model instances

        for pred in all_preds:
            # Create a unique key for this model instance
            key = (
                pred.get('model_name'),
                pred.get('step_idx'),
                pred.get('fold_id'),
                pred.get('branch_id'),
            )
            if key in seen:
                continue
            seen.add(key)

            candidate = ModelCandidate(
                model_name=pred.get('model_name', ''),
                model_classname=pred.get('model_classname', ''),
                step_idx=pred.get('step_idx', 0),
                fold_id=str(pred.get('fold_id')) if pred.get('fold_id') is not None else None,
                branch_id=pred.get('branch_id'),
                branch_name=pred.get('branch_name'),
                val_score=pred.get('val_score'),
                metric=pred.get('metric'),
            )
            candidates.append(candidate)

        return candidates

    def _get_meta_operator_from_context(self, context: 'ExecutionContext') -> MetaModel:
        """Get MetaModel operator from execution context.

        Args:
            context: Execution context.

        Returns:
            MetaModel operator.

        Raises:
            ValueError: If MetaModel cannot be found in context.
        """
        # Try to get from custom context (preferred)
        meta_op = context.custom.get('meta_operator')
        if isinstance(meta_op, MetaModel):
            return meta_op

        # Try to get from context's step info
        step_info = context.custom.get('_step_info')
        if step_info is not None:
            operator = getattr(step_info, 'operator', None)
            if isinstance(operator, MetaModel):
                return operator
            step = getattr(step_info, 'original_step', None)
            if isinstance(step, dict) and isinstance(step.get('model'), MetaModel):
                return step['model']

        raise ValueError("Could not find MetaModel operator in context")

    def _handle_coverage(
        self,
        X_meta: np.ndarray,
        stacking_config: StackingConfig,
        n_samples: int
    ) -> np.ndarray:
        """Handle missing predictions based on coverage strategy.

        Args:
            X_meta: Feature matrix with potential NaN values.
            stacking_config: Stacking configuration.
            n_samples: Total number of samples.

        Returns:
            Processed feature matrix.

        Raises:
            ValueError: If coverage is insufficient for STRICT mode.
        """
        nan_mask = np.isnan(X_meta)
        coverage = 1.0 - (nan_mask.sum() / X_meta.size)

        strategy = stacking_config.coverage_strategy

        if strategy == CoverageStrategy.STRICT:
            if nan_mask.any():
                n_missing = nan_mask.any(axis=1).sum()
                raise ValueError(
                    f"Incomplete coverage: {n_missing}/{n_samples} samples missing predictions. "
                    f"Use CoverageStrategy.DROP_INCOMPLETE or IMPUTE_* to handle this."
                )
            return X_meta

        elif strategy == CoverageStrategy.DROP_INCOMPLETE:
            # Mark incomplete samples (will be handled by caller)
            complete_mask = ~nan_mask.any(axis=1)
            if not complete_mask.all():
                n_dropped = (~complete_mask).sum()
                warnings.warn(
                    f"Dropping {n_dropped}/{n_samples} samples with incomplete predictions"
                )
            # Replace NaN with 0 for now (samples will be masked later)
            X_meta = np.nan_to_num(X_meta, nan=0.0)
            return X_meta

        elif strategy == CoverageStrategy.IMPUTE_ZERO:
            return np.nan_to_num(X_meta, nan=0.0)

        elif strategy == CoverageStrategy.IMPUTE_MEAN:
            # Impute with column means
            col_means = np.nanmean(X_meta, axis=0)
            for col in range(X_meta.shape[1]):
                mask = nan_mask[:, col]
                X_meta[mask, col] = col_means[col]
            return X_meta

        elif strategy == CoverageStrategy.IMPUTE_FOLD_MEAN:
            # Same as IMPUTE_MEAN for now (fold-level imputation requires more context)
            col_means = np.nanmean(X_meta, axis=0)
            for col in range(X_meta.shape[1]):
                mask = nan_mask[:, col]
                X_meta[mask, col] = col_means[col]
            return X_meta

        return X_meta

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute meta-model controller.

        Stores MetaModel operator and prediction_store in context for use by get_xy().
        Also stores source models for artifact persistence in Phase 3.

        Args:
            step_info: Parsed step with MetaModel operator.
            dataset: SpectroDataset.
            context: Execution context.
            runtime_context: Runtime context.
            source: Data source index.
            mode: Execution mode.
            loaded_binaries: Pre-loaded model binaries.
            prediction_store: Predictions store.

        Returns:
            Tuple of (updated_context, list_of_binaries).
        """
        # Store references for get_xy() to access using custom dict (proper pattern)
        context.custom['_prediction_store'] = prediction_store
        context.custom['_runtime_context'] = runtime_context
        context.custom['_step_info'] = step_info

        # Store context in runtime_context for _persist_model access
        runtime_context.current_context = context

        # Extract and store meta operator in custom context
        operator = step_info.operator
        if isinstance(operator, MetaModel):
            context.custom['meta_operator'] = operator

            # Pre-compute source models for artifact persistence (Phase 3)
            if mode == "train" and prediction_store is not None:
                try:
                    source_models = self._get_source_models(
                        operator, context, prediction_store
                    )
                    context.custom['_source_models'] = source_models
                except Exception:
                    # Source models may not be available yet in early steps
                    pass

        verbose = getattr(self, 'verbose', 0)
        if verbose > 0:
            print(f"{STACK} MetaModel stacking step {context.state.step_number}")

        # Set layout preference
        context = context.with_layout("2d")

        # Call parent execute (SklearnModelController.execute)
        return SklearnModelController.execute(
            self, step_info, dataset, context, runtime_context,
            source, mode, loaded_binaries, prediction_store
        )

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from MetaModel step.

        Extracts configuration including finetune_space for Optuna integration.

        Args:
            step: Pipeline step configuration.
            operator: MetaModel operator.

        Returns:
            Configuration dict with model_instance set to MetaModel and
            finetune_params if configured.
        """
        if isinstance(operator, MetaModel):
            config = {
                'model_instance': operator,
                'name': operator.name,
            }
            # Phase 7: Extract finetune parameters from MetaModel
            finetune_params = operator.get_finetune_params()
            if finetune_params:
                config['finetune_params'] = finetune_params
            if isinstance(step, dict):
                config.update({k: v for k, v in step.items() if k != 'model'})
            return config

        if isinstance(step, dict):
            config = step.copy()
            if isinstance(step.get('model'), MetaModel):
                config['model_instance'] = step['model']
                config['name'] = step['model'].name
                # Phase 7: Extract finetune parameters
                finetune_params = step['model'].get_finetune_params()
                if finetune_params:
                    config['finetune_params'] = finetune_params
            return config

        return {'model_instance': step}

    # =========================================================================
    # Phase 3: Prediction Mode Implementation
    # =========================================================================

    def _execute_predict(
        self,
        dataset,
        model_config,
        context,
        runtime_context,
        prediction_store,
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_unscaled,
        y_test_unscaled,
        folds,
        loaded_binaries,
        mode
    ):
        """Execute meta-model prediction with dependency resolution.

        In prediction mode, this method:
        1. Loads the meta-model artifact configuration
        2. Validates branch context matches training
        3. Collects predictions from source models (already in prediction_store)
        4. Constructs meta-features from source predictions
        5. Loads meta-model binary and generates predictions

        Args:
            dataset: SpectroDataset instance.
            model_config: Model configuration dictionary.
            context: Execution context.
            runtime_context: Runtime context.
            prediction_store: Predictions storage.
            X_train: Training features (empty in prediction mode).
            y_train: Training targets (empty in prediction mode).
            X_test: Test features.
            y_test: Test targets.
            y_train_unscaled: Unscaled training targets.
            y_test_unscaled: Unscaled test targets.
            folds: Cross-validation folds (unused in prediction mode).
            loaded_binaries: Pre-loaded model binaries.
            mode: Execution mode ('predict' or 'explain').

        Returns:
            Tuple of (context, binaries).

        Raises:
            MetaModelPredictionError: If prediction fails.
        """
        verbose = model_config.get('train_params', {}).get('verbose', 0)

        if verbose > 0:
            print(f"{STACK} MetaModel prediction mode")

        # Get meta operator from context
        try:
            meta_operator = self._get_meta_operator_from_context(context)
        except ValueError:
            # Fallback to extracting from model_config
            meta_operator = model_config.get('model_instance')
            if not isinstance(meta_operator, MetaModel):
                raise MetaModelPredictionError(
                    "Cannot find MetaModel operator for prediction"
                )

        # Try to load meta-model configuration from artifacts
        meta_artifact = self._load_meta_artifact_config(
            runtime_context, context, loaded_binaries
        )

        if meta_artifact is not None:
            # Validate branch context
            self._validate_prediction_branch_context(meta_artifact, context)

        # Build meta-features from current session predictions
        X_test_meta = self._build_predict_features(
            dataset, context, meta_operator, meta_artifact, prediction_store
        )

        if verbose > 0:
            print(f"  {CHECK} Built meta-features: shape={X_test_meta.shape}")

        # Update X_test with meta-features (training data is empty in prediction mode)
        # Store original for reference
        context.custom['_X_test_meta'] = X_test_meta

        # Call parent's prediction execution with meta-features
        return SklearnModelController._execute_predict(
            self, dataset, model_config, context, runtime_context, prediction_store,
            X_train, y_train, X_test_meta, y_test, y_train_unscaled, y_test_unscaled,
            folds, loaded_binaries, mode
        )

    def _load_meta_artifact_config(
        self,
        runtime_context: 'RuntimeContext',
        context: 'ExecutionContext',
        loaded_binaries: Optional[List[Tuple[str, Any]]]
    ) -> Optional[MetaModelArtifact]:
        """Load meta-model artifact configuration.

        Attempts to load the MetaModelArtifact from:
        1. Artifact registry (if available)
        2. Target model metadata (from runtime_context)

        Args:
            runtime_context: Runtime context with artifact registry.
            context: Execution context.
            loaded_binaries: Loaded binaries (may contain config).

        Returns:
            MetaModelArtifact or None if not found.
        """
        # Try to get from artifact registry
        if hasattr(runtime_context, 'artifact_registry') and runtime_context.artifact_registry:
            registry = runtime_context.artifact_registry
            step_index = context.state.step_number
            branch_path = getattr(context.selector, 'branch_path', [])

            # Look for meta-model artifact at this step
            pipeline_id = getattr(runtime_context.saver, 'pipeline_id', None) if runtime_context.saver else None
            if pipeline_id:
                artifacts = registry.get_artifacts_for_step(
                    pipeline_id=pipeline_id,
                    step_index=step_index,
                    branch_path=branch_path or None
                )
                for record in artifacts:
                    if record.artifact_type == ArtifactType.META_MODEL and record.meta_config:
                        # Convert MetaModelConfig to MetaModelArtifact
                        return self._record_to_artifact(record)

        # Try to get from target_model metadata
        if hasattr(runtime_context, 'target_model') and runtime_context.target_model:
            meta_config_data = runtime_context.target_model.get('meta_config')
            if meta_config_data:
                return MetaModelArtifact.from_dict(meta_config_data)

        return None

    def _record_to_artifact(self, record: 'ArtifactRecord') -> MetaModelArtifact:
        """Convert ArtifactRecord to MetaModelArtifact.

        Args:
            record: ArtifactRecord with meta_config.

        Returns:
            MetaModelArtifact instance.
        """
        meta_config = record.meta_config

        # Build source model references from meta_config
        source_refs = []
        if meta_config and meta_config.source_models:
            for idx, source_info in enumerate(meta_config.source_models):
                ref = SourceModelReference(
                    model_name=source_info.get('model_name', ''),
                    model_classname='',
                    step_idx=0,
                    artifact_id=source_info.get('artifact_id', ''),
                    feature_index=source_info.get('feature_index', idx),
                )
                source_refs.append(ref)

        feature_columns = meta_config.feature_columns if meta_config else []

        return MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name=record.class_name,
            meta_learner_class=record.class_name,
            source_models=source_refs,
            feature_columns=feature_columns,
            stacking_config={},
            artifact_id=record.artifact_id,
        )

    def _validate_prediction_branch_context(
        self,
        meta_artifact: MetaModelArtifact,
        context: 'ExecutionContext'
    ) -> None:
        """Validate that prediction branch matches training branch.

        Args:
            meta_artifact: Meta-model artifact with training branch info.
            context: Current execution context.

        Raises:
            BranchMismatchError: If branch contexts are incompatible.
        """
        if meta_artifact.branch_context is None:
            return  # No branch context to validate

        training_branch = meta_artifact.branch_context
        prediction_branch = {
            'branch_id': getattr(context.selector, 'branch_id', None),
            'branch_name': getattr(context.selector, 'branch_name', None),
            'branch_path': getattr(context.selector, 'branch_path', None),
        }

        # Check branch_id match (None matches None)
        train_id = training_branch.get('branch_id')
        pred_id = prediction_branch.get('branch_id')

        if train_id != pred_id:
            # Allow if prediction is in a sub-branch of training
            train_path = training_branch.get('branch_path') or []
            pred_path = prediction_branch.get('branch_path') or []

            path_len_ok = len(train_path) <= len(pred_path)
            path_prefix_ok = pred_path[:len(train_path)] == train_path
            is_valid_sub_branch = path_len_ok and path_prefix_ok

            if not is_valid_sub_branch:
                raise BranchMismatchError(
                    training_branch=training_branch,
                    prediction_branch=prediction_branch,
                    meta_model_id=meta_artifact.artifact_id
                )

    def _build_predict_features(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        meta_operator: MetaModel,
        meta_artifact: Optional[MetaModelArtifact],
        prediction_store: 'Predictions'
    ) -> np.ndarray:
        """Build meta-features from source model predictions for prediction mode.

        Collects predictions from source models (stored in prediction_store
        during the current prediction session) and constructs feature matrix.

        Args:
            dataset: SpectroDataset.
            context: Execution context.
            meta_operator: MetaModel operator.
            meta_artifact: Optional loaded artifact for feature ordering.
            prediction_store: Predictions storage.

        Returns:
            Meta-feature matrix for test data.

        Raises:
            NoSourcePredictionsError: If no source predictions found.
            FeatureOrderMismatchError: If feature order doesn't match training.
        """
        # Get source models
        source_models = self._get_source_models(meta_operator, context, prediction_store)

        if not source_models:
            expected = meta_artifact.feature_columns if meta_artifact else []
            raise NoSourcePredictionsError(
                expected_sources=expected,
                meta_model_id=meta_artifact.artifact_id if meta_artifact else "unknown"
            )

        # Get unique source names in order
        unique_source_names = list(dict.fromkeys(m.model_name for m in source_models))

        # If we have artifact config, validate order matches
        if meta_artifact and meta_artifact.feature_columns:
            expected_sources = [
                col.replace('_pred', '') for col in meta_artifact.feature_columns
            ]
            # Reorder to match training order
            if set(unique_source_names) == set(expected_sources):
                unique_source_names = expected_sources
            elif len(unique_source_names) != len(expected_sources):
                raise FeatureOrderMismatchError(
                    expected_columns=meta_artifact.feature_columns,
                    actual_columns=[f"{n}_pred" for n in unique_source_names],
                    meta_model_id=meta_artifact.artifact_id
                )

        # Build test features
        return self._build_test_features_from_predictions(dataset, context)

    # =========================================================================
    # Phase 3: Meta-Model Artifact Persistence
    # =========================================================================

    def _persist_meta_model(
        self,
        runtime_context: 'RuntimeContext',
        model: Any,
        model_id: str,
        meta_operator: MetaModel,
        source_models: List[ModelCandidate],
        reconstruction_result: Optional[ReconstructionResult],
        context: 'ExecutionContext',
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None,
    ) -> Any:
        """Persist meta-model with source model references.

        Extends standard model persistence to include:
        - Source model artifact references (ordered)
        - Feature column mapping
        - Stacking configuration
        - Branch context for validation

        Args:
            runtime_context: Runtime context with artifact registry.
            model: Trained meta-model.
            model_id: Model identifier.
            meta_operator: MetaModel operator.
            source_models: List of source model candidates used.
            reconstruction_result: Result from OOF reconstruction.
            context: Execution context.
            branch_id: Branch identifier.
            branch_name: Branch name.
            branch_path: Full branch path.
            fold_id: Fold identifier.

        Returns:
            ArtifactMeta or ArtifactRecord for the persisted meta-model.
        """
        # Build MetaModelArtifact
        serializer = MetaModelSerializer()

        # Generate artifact ID
        pipeline_id = runtime_context.saver.pipeline_id if runtime_context.saver else "unknown"
        step_index = runtime_context.step_number
        bp = branch_path or ([branch_id] if branch_id is not None else [])

        if runtime_context.artifact_registry is not None:
            artifact_id = runtime_context.artifact_registry.generate_id(
                pipeline_id=pipeline_id,
                branch_path=bp,
                step_index=step_index,
                fold_id=fold_id
            )
        else:
            # Fallback artifact ID
            artifact_id = f"{pipeline_id}:{step_index}:{fold_id or 'all'}"

        meta_artifact = serializer.build_artifact(
            meta_operator=meta_operator,
            source_models=source_models,
            reconstruction_result=reconstruction_result,
            context=context,
            artifact_id=artifact_id,
        )

        # Validate artifact
        validation_errors = serializer.validate_artifact(meta_artifact)
        if validation_errors:
            warnings.warn(
                f"Meta-model artifact validation warnings: {validation_errors}"
            )

        # Use artifact registry if available
        if runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry

            # Get source model artifact IDs for dependencies
            source_artifact_ids = []
            for ref in meta_artifact.source_models:
                if ref.artifact_id:
                    source_artifact_ids.append(ref.artifact_id)

            # Create MetaModelConfig for registry
            meta_config = serializer.to_meta_model_config(meta_artifact)

            # Register meta-model with dependencies
            record = registry.register(
                obj=model,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.META_MODEL,
                depends_on=source_artifact_ids,
                params=meta_operator.model.get_params() if hasattr(meta_operator.model, 'get_params') else {},
                meta_config=meta_config,
                format_hint='sklearn'
            )

            return record

        # Fallback to legacy persistence with extended metadata
        artifact_meta = runtime_context.saver.persist_artifact(
            step_number=runtime_context.step_number,
            name=f"{model_id}.pkl",
            obj=model,
            format_hint='sklearn',
            branch_id=branch_id,
            branch_name=branch_name
        )

        # Store meta_config in the artifact metadata for later retrieval
        if hasattr(artifact_meta, '__setitem__'):
            artifact_meta['meta_config'] = meta_artifact.to_dict()

        return artifact_meta

    def _get_source_artifact_ids(
        self,
        source_models: List[ModelCandidate],
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext'
    ) -> List[str]:
        """Get artifact IDs for source models.

        Attempts to find the artifact IDs for each source model in the
        artifact registry.

        Args:
            source_models: List of source model candidates.
            context: Execution context.
            runtime_context: Runtime context with registry.

        Returns:
            List of artifact IDs (may be empty if registry unavailable).
        """
        artifact_ids = []

        if not hasattr(runtime_context, 'artifact_registry') or not runtime_context.artifact_registry:
            return artifact_ids

        registry = runtime_context.artifact_registry
        pipeline_id = runtime_context.saver.pipeline_id if runtime_context.saver else None

        if not pipeline_id:
            return artifact_ids

        # Deduplicate by model name
        seen = set()
        for candidate in source_models:
            if candidate.model_name in seen:
                continue
            seen.add(candidate.model_name)

            # Find artifact for this source model
            branch_path = []
            if candidate.branch_id is not None:
                branch_path = [candidate.branch_id]

            # Try to find the artifact
            artifacts = registry.get_artifacts_for_step(
                pipeline_id=pipeline_id,
                step_index=candidate.step_idx,
                branch_path=branch_path or None
            )

            # Look for a model artifact at this step
            for record in artifacts:
                if record.artifact_type in (ArtifactType.MODEL, ArtifactType.META_MODEL):
                    artifact_ids.append(record.artifact_id)
                    break

        return artifact_ids

    def _persist_model(
        self,
        runtime_context: 'RuntimeContext',
        model: Any,
        model_id: str,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Override to persist meta-model with source references.

        Extends parent _persist_model to include source model dependencies
        and stacking configuration in the artifact.

        Args:
            runtime_context: Runtime context with saver/registry instances.
            model: Trained meta-model.
            model_id: Unique identifier for the model.
            branch_id: Optional branch identifier for branched pipelines.
            branch_name: Optional branch name for branched pipelines.
            branch_path: Optional list of branch indices for nested branching.
            fold_id: Optional fold identifier for CV artifacts.
            params: Optional model parameters for inspection.

        Returns:
            ArtifactMeta or ArtifactRecord for persisted model.
        """
        # Get context from runtime_context custom storage if available
        context = getattr(runtime_context, 'current_context', None)
        if context is None:
            # Fallback to parent implementation if no context
            return super()._persist_model(
                runtime_context, model, model_id,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=branch_path,
                fold_id=fold_id,
                params=params
            )

        # Extract meta_operator from context
        meta_operator = context.custom.get('meta_operator')

        # If not a MetaModel or no operator found, fall back to parent
        if not isinstance(meta_operator, MetaModel):
            return super()._persist_model(
                runtime_context, model, model_id,
                branch_id=branch_id,
                branch_name=branch_name,
                branch_path=branch_path,
                fold_id=fold_id,
                params=params
            )

        # Get source models from context (stored during execute)
        source_models = context.custom.get('_source_models', [])
        prediction_store = context.custom.get('_prediction_store')

        # If source_models weren't pre-computed, try to get them now
        if not source_models and prediction_store is not None:
            try:
                source_models = self._get_source_models(
                    meta_operator, context, prediction_store
                )
            except Exception:
                source_models = []

        # Get reconstruction result if available
        reconstruction_result = context.custom.get('_reconstruction_result')

        # Use the passed model_id (already generated by base class)
        # Branch info already passed as parameters

        # Use meta-model specific persistence
        return self._persist_meta_model(
            runtime_context=runtime_context,
            model=model,
            model_id=model_id,
            meta_operator=meta_operator,
            source_models=source_models,
            reconstruction_result=reconstruction_result,
            context=context,
            branch_id=branch_id,
            branch_name=branch_name,
            branch_path=branch_path,
            fold_id=fold_id
        )

