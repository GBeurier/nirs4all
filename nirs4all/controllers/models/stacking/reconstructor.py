"""
Training Set Reconstructor for Meta-Model Stacking.

This module provides the TrainingSetReconstructor class that builds meta-model
training features from out-of-fold (OOF) predictions of source models.

The key principle is that each training sample's meta-feature comes from a fold
where that sample was NOT used for training, preventing data leakage.

Classes:
    TrainingSetReconstructor: Main class for OOF reconstruction.
    FoldAlignmentValidator: Validates fold structure consistency.
    ValidationResult: Container for validation errors and warnings.
    ReconstructionResult: Container for reconstructed data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import warnings
import numpy as np

from nirs4all.operators.models.meta import (
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
)
from nirs4all.operators.models.selection import ModelCandidate
from .config import ReconstructorConfig

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from .classification import ClassificationInfo, MetaFeatureInfo


@dataclass
class ValidationError:
    """A single validation error."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationWarning:
    """A single validation warning."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Container for validation errors and warnings.

    Accumulates validation issues during fold alignment and coverage checks.

    Attributes:
        errors: List of validation errors (critical issues).
        warnings: List of validation warnings (non-critical issues).
        is_valid: True if no errors (warnings are allowed).

    Example:
        >>> result = ValidationResult()
        >>> result.add_error("FOLD_MISMATCH", "Folds don't align")
        >>> result.add_warning("PARTIAL_COVERAGE", "80% coverage")
        >>> if not result.is_valid:
        ...     raise ValueError(result.format_errors())
    """

    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def add_error(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(code, message, details))

    def add_warning(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationWarning(code, message, details))

    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def format_errors(self) -> str:
        """Format all errors as a human-readable string."""
        if not self.errors:
            return "No errors"
        lines = [f"[{e.code}] {e.message}" for e in self.errors]
        return "\n".join(lines)

    def format_warnings(self) -> str:
        """Format all warnings as a human-readable string."""
        if not self.warnings:
            return "No warnings"
        lines = [f"[{w.code}] {w.message}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class ReconstructionResult:
    """Container for reconstructed training set data.

    Holds the meta-feature matrices for training and test sets,
    along with metadata about the reconstruction process.

    Attributes:
        X_train_meta: Training meta-features (n_train_samples, n_features).
        X_test_meta: Test meta-features (n_test_samples, n_features).
        y_train: Training targets (n_train_samples,).
        y_test: Test targets (n_test_samples,).
        feature_names: List of feature column names.
        source_models: List of source model names used.
        valid_train_mask: Boolean mask of valid training samples (after coverage handling).
        valid_test_mask: Boolean mask of valid test samples.
        validation_result: Validation result from fold alignment.
        n_folds: Number of folds detected.
        coverage_ratio: Ratio of samples with complete predictions.
        meta_feature_info: Optional MetaFeatureInfo for feature importance tracking.
        classification_info: Optional ClassificationInfo for task type metadata.

    Example:
        >>> result = reconstructor.reconstruct(dataset, context)
        >>> X_train = result.X_train_meta[result.valid_train_mask]
        >>> y_train = result.y_train[result.valid_train_mask]
        >>> # For feature importance tracking
        >>> if result.meta_feature_info:
        ...     model_importance = result.meta_feature_info.aggregate_importance_by_model(
        ...         feature_importances
        ...     )
    """

    X_train_meta: np.ndarray
    X_test_meta: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    source_models: List[str]
    valid_train_mask: np.ndarray
    valid_test_mask: np.ndarray
    validation_result: ValidationResult
    n_folds: int
    coverage_ratio: float
    meta_feature_info: Optional[Any] = None  # MetaFeatureInfo from classification module
    classification_info: Optional[Any] = None  # ClassificationInfo from classification module


class FoldAlignmentValidator:
    """Validates fold structure consistency across source models.

    Ensures that all source models have compatible fold structures
    for proper out-of-fold reconstruction.

    Checks performed:
    1. All models have the same number of folds.
    2. Fold indices are sequential (0, 1, 2, ..., K-1).
    3. No sample appears in multiple validation sets within a model.
    4. Sample indices are consistent across folds.

    Attributes:
        prediction_store: Predictions storage for accessing fold data.
        config: Reconstructor configuration.
    """

    def __init__(
        self,
        prediction_store: 'Predictions',
        config: Optional[ReconstructorConfig] = None
    ):
        """Initialize fold alignment validator.

        Args:
            prediction_store: Predictions storage.
            config: Optional reconstructor configuration.
        """
        self.prediction_store = prediction_store
        self.config = config or ReconstructorConfig()

    def validate(
        self,
        source_model_names: List[str],
        context: 'ExecutionContext',
        branch_id_override: Optional[int] = -1
    ) -> ValidationResult:
        """Validate fold alignment across source models.

        Args:
            source_model_names: List of source model names to validate.
            context: Execution context with branch info.
            branch_id_override: Optional branch_id override. If -1 (default),
                use context's branch_id. If None, don't filter by branch
                (for ALL_BRANCHES scope).

        Returns:
            ValidationResult with any errors or warnings.
        """
        result = ValidationResult()

        if not source_model_names:
            result.add_error(
                "NO_SOURCE_MODELS",
                "No source models provided for fold alignment validation"
            )
            return result

        # Get branch context - allow override for ALL_BRANCHES scope
        if branch_id_override == -1:
            branch_id = getattr(context.selector, 'branch_id', None)
        else:
            branch_id = branch_id_override
        current_step = context.state.step_number

        # Collect fold info per model
        model_fold_info: Dict[str, Dict[str, Any]] = {}

        for model_name in source_model_names:
            fold_info = self._get_model_fold_info(
                model_name, branch_id, current_step
            )
            if fold_info:
                model_fold_info[model_name] = fold_info

        if not model_fold_info:
            result.add_error(
                "NO_FOLD_DATA",
                "No fold data found for any source model"
            )
            return result

        # Check 1: Same fold count
        fold_counts = {
            name: info['n_folds']
            for name, info in model_fold_info.items()
        }
        unique_counts = set(fold_counts.values())

        if len(unique_counts) > 1:
            result.add_error(
                "FOLD_COUNT_MISMATCH",
                f"Source models have different fold counts: {fold_counts}. "
                f"All source models must use the same cross-validation splitter."
            )

        # Check 2: Sequential fold indices
        for model_name, info in model_fold_info.items():
            fold_ids = sorted(info['fold_ids'])
            expected = list(range(len(fold_ids)))
            # Convert fold_ids to integers for comparison
            int_fold_ids = []
            for fid in fold_ids:
                try:
                    int_fold_ids.append(int(fid))
                except (ValueError, TypeError):
                    continue

            if int_fold_ids and sorted(int_fold_ids) != expected[:len(int_fold_ids)]:
                result.add_warning(
                    "NON_SEQUENTIAL_FOLDS",
                    f"Model {model_name} has non-sequential fold IDs: {fold_ids}. "
                    f"Expected sequential IDs starting from 0."
                )

        # Check 3: No duplicate samples in validation sets
        for model_name, info in model_fold_info.items():
            all_val_samples = info.get('all_val_samples', [])
            unique_samples = set(all_val_samples)

            if len(all_val_samples) != len(unique_samples):
                duplicates = [
                    s for s in unique_samples
                    if all_val_samples.count(s) > 1
                ]
                result.add_warning(
                    "DUPLICATE_VAL_SAMPLES",
                    f"Model {model_name} has samples appearing in multiple "
                    f"validation sets: {duplicates[:5]}... This may indicate "
                    f"an issue with the cross-validation splitter."
                )

        return result

    def _get_model_fold_info(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int
    ) -> Optional[Dict[str, Any]]:
        """Get fold information for a single model.

        Args:
            model_name: Name of the model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).

        Returns:
            Dictionary with fold info or None if no data found.
        """
        filter_kwargs = {
            'model_name': model_name,
            'partition': 'val',
            'load_arrays': False,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        val_preds = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        val_preds = [p for p in val_preds if p.get('step_idx', 0) < max_step]

        # If no predictions found with branch filter, try without branch filter
        # This handles models trained before any branch was created
        if not val_preds and branch_id is not None:
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': False,
            }
            val_preds = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            val_preds = [p for p in val_preds if p.get('step_idx', 0) < max_step]
            # Only keep predictions that don't have a branch_id (pre-branch models)
            val_preds = [p for p in val_preds if p.get('branch_id') is None]

        # Filter out averaged predictions
        val_preds = [
            p for p in val_preds
            if str(p.get('fold_id', '')) not in self.config.excluded_fold_ids
        ]

        if not val_preds:
            return None

        # Collect fold IDs and sample indices
        fold_ids = set()
        all_val_samples = []

        for pred in val_preds:
            fold_id = pred.get('fold_id')
            if fold_id is not None:
                fold_ids.add(str(fold_id))

            # Collect sample indices if available
            sample_indices = pred.get('sample_indices', [])
            if sample_indices is not None:
                if hasattr(sample_indices, 'tolist'):
                    sample_indices = sample_indices.tolist()
                all_val_samples.extend(sample_indices)

        return {
            'n_folds': len(fold_ids),
            'fold_ids': list(fold_ids),
            'all_val_samples': all_val_samples,
            'n_val_predictions': len(val_preds),
        }


class TrainingSetReconstructor:
    """Reconstructs meta-model training set from out-of-fold predictions.

    This is the core class for Phase 2 of the meta-model stacking implementation.
    It handles the critical task of collecting OOF predictions from source models
    and constructing feature matrices for the meta-learner.

    The fundamental invariant is: **No sample sees predictions from a model
    trained on that sample**. This prevents data leakage.

    Attributes:
        prediction_store: Predictions storage for accessing source predictions.
        source_model_names: List of source model names to use.
        stacking_config: Configuration for coverage and aggregation strategies.
        reconstructor_config: Internal configuration for reconstruction.
        fold_validator: Validator for fold alignment.

    Example:
        >>> reconstructor = TrainingSetReconstructor(
        ...     prediction_store=predictions,
        ...     source_model_names=["PLS", "RF", "XGB"],
        ...     stacking_config=StackingConfig(
        ...         coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
        ...         test_aggregation=TestAggregation.MEAN
        ...     )
        ... )
        >>> result = reconstructor.reconstruct(dataset, context)
        >>> print(f"Coverage: {result.coverage_ratio:.1%}")
        >>> print(f"Features: {result.feature_names}")
    """

    def __init__(
        self,
        prediction_store: 'Predictions',
        source_model_names: List[str],
        stacking_config: Optional[StackingConfig] = None,
        reconstructor_config: Optional[ReconstructorConfig] = None,
        source_model_branch_map: Optional[Dict[str, Optional[int]]] = None
    ):
        """Initialize TrainingSetReconstructor.

        Args:
            prediction_store: Predictions storage for accessing source predictions.
            source_model_names: List of source model names to use as features.
                Order determines feature column order. For cross-branch stacking,
                these may be unique names (e.g., "Ridge_MetaModel_br0") that include
                branch suffix.
            stacking_config: User-facing configuration for coverage and aggregation.
            reconstructor_config: Internal configuration for reconstruction behavior.
            source_model_branch_map: Mapping from unique source name to
                (original_model_name, branch_id). Used for cross-branch stacking
                where same model name appears in multiple branches. Keys are the
                unique names in source_model_names, values are (name, branch_id)
                tuples. If None, uses source_model_names as-is with context branch.
        """
        self.prediction_store = prediction_store
        self.source_model_names = source_model_names
        self.stacking_config = stacking_config or StackingConfig()
        self.reconstructor_config = reconstructor_config or ReconstructorConfig()
        self.source_model_branch_map = source_model_branch_map or {}
        self.fold_validator = FoldAlignmentValidator(
            prediction_store, self.reconstructor_config
        )

    def _resolve_model_lookup(
        self,
        unique_name: str,
        default_branch_id: Optional[int]
    ) -> Tuple[str, Optional[int]]:
        """Resolve unique name to actual model name and branch_id for lookup.

        For cross-branch stacking, unique_name may include branch suffix
        (e.g., "Ridge_MetaModel_br0"). This method extracts the original
        model name and branch_id for prediction lookup.

        Args:
            unique_name: The unique source name (may include branch suffix).
            default_branch_id: Default branch_id from context if not in map.

        Returns:
            Tuple of (actual_model_name, branch_id_for_lookup).
        """
        if unique_name in self.source_model_branch_map:
            actual_name, branch_id = self.source_model_branch_map[unique_name]
            return actual_name, branch_id
        # Not in map - use unique_name as-is with default branch
        return unique_name, default_branch_id

    def reconstruct(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        use_proba: bool = False
    ) -> ReconstructionResult:
        """Reconstruct meta-model training and test sets from predictions.

        Collects out-of-fold predictions for training samples and aggregated
        predictions for test samples.

        Phase 5 Enhancement: Supports classification tasks with probability
        features for binary and multiclass classification.

        Args:
            dataset: SpectroDataset for sample indices.
            context: Execution context with partition and branch info.
            y_train: Optional pre-computed training targets.
            y_test: Optional pre-computed test targets.
            use_proba: If True, use probability predictions for classification.

        Returns:
            ReconstructionResult containing meta-feature matrices and metadata.

        Raises:
            ValueError: If no source models found or critical validation fails.
        """
        # Validate inputs
        if not self.source_model_names:
            raise ValueError("No source model names provided for reconstruction")

        # Get branch context - for ALL_BRANCHES scope, don't filter by branch
        branch_scope = self.stacking_config.branch_scope
        if branch_scope == BranchScope.ALL_BRANCHES:
            # ALL_BRANCHES: Collect predictions from all branches
            branch_id = None
        else:
            # CURRENT_ONLY or SPECIFIED: Filter by current branch
            branch_id = getattr(context.selector, 'branch_id', None)
        current_step = context.state.step_number

        # Validate fold alignment if configured
        validation_result = ValidationResult()
        if self.reconstructor_config.validate_fold_alignment:
            # Pass branch_id_override to respect ALL_BRANCHES scope
            branch_id_override = None if branch_scope == BranchScope.ALL_BRANCHES else -1
            validation_result = self.fold_validator.validate(
                self.source_model_names, context, branch_id_override=branch_id_override
            )
            if not validation_result.is_valid:
                # Log warnings but continue for non-critical issues
                if self.reconstructor_config.log_warnings:
                    for warning in validation_result.warnings:
                        warnings.warn(f"[{warning.code}] {warning.message}")

        # Phase 5: Detect classification task type
        classification_info = self._detect_classification_info(context, use_proba)
        meta_feature_info = None

        # Get sample indices
        train_sample_ids, test_sample_ids = self._get_sample_indices(dataset, context)
        n_train = len(train_sample_ids)
        n_test = len(test_sample_ids)

        # Calculate number of features based on classification info
        n_features_per_model = classification_info.get_n_features_per_model(use_proba)
        n_total_features = len(self.source_model_names) * n_features_per_model

        # Build ID to position mappings
        train_id_to_pos = {int(sid): pos for pos, sid in enumerate(train_sample_ids)}
        test_id_to_pos = {int(sid): pos for pos, sid in enumerate(test_sample_ids)}

        # Initialize feature matrices with NaN
        X_train_meta = np.full((n_train, n_total_features), np.nan)
        X_test_meta = np.full((n_test, n_total_features), np.nan)

        # Collect predictions for each source model
        n_folds = 0
        feat_col = 0  # Current feature column index

        for unique_name in self.source_model_names:
            # Resolve unique name to actual model name and branch for lookup
            actual_model_name, model_branch_id = self._resolve_model_lookup(
                unique_name, branch_id
            )

            # OOF predictions for training
            oof_features, model_n_folds = self._collect_oof_predictions_with_proba(
                model_name=actual_model_name,
                branch_id=model_branch_id,
                max_step=current_step,
                id_to_pos=train_id_to_pos,
                n_samples=n_train,
                use_proba=use_proba,
                classification_info=classification_info
            )
            n_folds = max(n_folds, model_n_folds)

            # Aggregated test predictions
            test_features = self._collect_test_predictions_with_proba(
                model_name=actual_model_name,
                branch_id=model_branch_id,
                max_step=current_step,
                id_to_pos=test_id_to_pos,
                n_samples=n_test,
                use_proba=use_proba,
                classification_info=classification_info
            )

            # Assign features to correct columns
            n_cols = oof_features.shape[1] if oof_features.ndim > 1 else 1
            if oof_features.ndim == 1:
                X_train_meta[:, feat_col] = oof_features
                X_test_meta[:, feat_col] = test_features
            else:
                X_train_meta[:, feat_col:feat_col + n_cols] = oof_features
                X_test_meta[:, feat_col:feat_col + n_cols] = test_features

            feat_col += n_cols

        # Trim to actual features used (in case of inconsistent feature counts)
        if feat_col < n_total_features:
            X_train_meta = X_train_meta[:, :feat_col]
            X_test_meta = X_test_meta[:, :feat_col]

        # Compute coverage
        nan_mask_train = np.isnan(X_train_meta)
        coverage_ratio = 1.0 - (nan_mask_train.sum() / max(X_train_meta.size, 1))

        # Apply coverage strategy
        X_train_meta, valid_train_mask = self._apply_coverage_strategy(
            X_train_meta, n_train
        )

        # For test, use simpler handling (just fill NaN with 0)
        nan_mask_test = np.isnan(X_test_meta)
        X_test_meta = np.nan_to_num(X_test_meta, nan=0.0)
        valid_test_mask = ~nan_mask_test.any(axis=1)

        # Generate feature names with classification support
        feature_names = self._generate_feature_names_with_classification(
            classification_info, use_proba
        )
        # Trim feature names to match actual columns
        feature_names = feature_names[:feat_col]

        # Build meta feature info for feature importance tracking
        meta_feature_info = self._build_meta_feature_info(
            classification_info, use_proba, feature_names
        )

        # CRITICAL: Always get UNSCALED y values for meta-model training.
        # The OOF predictions (X_train_meta) are stored in unscaled (original y) space,
        # so the meta-model must train on unscaled y to avoid scale mismatch.
        # If y_processing was applied, the passed-in y_train/y_test are scaled,
        # but we need unscaled values to match the prediction space.
        #
        # Additionally handle sample_augmentation: predictions are stored with
        # original sample indices only, so we need to match sample counts.
        y_train = self._get_y_values_for_samples(dataset, context, train_sample_ids)
        y_test = self._get_y_values_for_samples(dataset, context, test_sample_ids)

        return ReconstructionResult(
            X_train_meta=X_train_meta,
            X_test_meta=X_test_meta,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            source_models=self.source_model_names.copy(),
            valid_train_mask=valid_train_mask,
            valid_test_mask=valid_test_mask,
            validation_result=validation_result,
            n_folds=n_folds,
            coverage_ratio=coverage_ratio,
            meta_feature_info=meta_feature_info,
            classification_info=classification_info,
        )

    def _get_sample_indices(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get training and test sample indices.

        For OOF reconstruction, we need to match sample indices with what
        was stored in the prediction_store. Models store predictions using
        ORIGINAL sample indices (not augmented), so we must use
        include_augmented=False to get the correct sample IDs.

        Args:
            dataset: SpectroDataset.
            context: Execution context.

        Returns:
            Tuple of (train_sample_ids, test_sample_ids).
        """
        train_context = context.with_partition('train')
        test_context = context.with_partition('test')

        # IMPORTANT: Use include_augmented=False because predictions are stored
        # with original sample indices only. Augmented samples are synthetic
        # copies that share the same sample ID as their source samples.
        train_ids = dataset._indexer.x_indices(
            train_context.selector,
            include_augmented=False,
            include_excluded=False
        )
        test_ids = dataset._indexer.x_indices(
            test_context.selector,
            include_augmented=False,
            include_excluded=False
        )

        return np.asarray(train_ids), np.asarray(test_ids)

    def _collect_oof_predictions(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int,
        id_to_pos: Dict[int, int],
        n_samples: int,
        use_proba: bool = False
    ) -> Tuple[np.ndarray, int]:
        """Collect out-of-fold predictions for a source model.

        For each fold, collects validation partition predictions and places
        them at the correct sample positions.

        Args:
            model_name: Name of the source model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).
            id_to_pos: Mapping from sample ID to array position.
            n_samples: Total number of training samples.
            use_proba: If True, use probability predictions.

        Returns:
            Tuple of (oof_predictions_array, n_folds_found).
        """
        oof_preds = np.full(n_samples, np.nan)

        filter_kwargs = {
            'model_name': model_name,
            'partition': 'val',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        val_predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        val_predictions = [p for p in val_predictions if p.get('step_idx', 0) < max_step]

        # If no predictions found with branch filter, try without branch filter
        # This handles models trained before any branch was created
        if not val_predictions and branch_id is not None:
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': True,
            }
            val_predictions = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            val_predictions = [p for p in val_predictions if p.get('step_idx', 0) < max_step]
            # Only keep predictions that don't have a branch_id (pre-branch models)
            val_predictions = [p for p in val_predictions if p.get('branch_id') is None]

        # Filter out averaged predictions
        val_predictions = [
            p for p in val_predictions
            if str(p.get('fold_id', '')) not in self.reconstructor_config.excluded_fold_ids
        ]

        n_folds = len(val_predictions)

        for pred in val_predictions:
            sample_indices = pred.get('sample_indices', [])
            if sample_indices is None:
                continue

            # Get prediction values
            if use_proba and 'y_proba' in pred and pred['y_proba'] is not None:
                y_vals = pred['y_proba']
                # For binary classification, use positive class probability
                if hasattr(y_vals, 'ndim') and y_vals.ndim > 1:
                    if y_vals.shape[1] == 2:
                        y_vals = y_vals[:, 1]
                    else:
                        y_vals = y_vals[:, 0]  # Use first class for multiclass
            else:
                y_vals = pred.get('y_pred', [])

            # Flatten if needed
            if hasattr(y_vals, 'flatten'):
                y_vals = y_vals.flatten()

            # Convert sample_indices to list if needed
            if hasattr(sample_indices, 'tolist'):
                sample_indices = sample_indices.tolist()

            # Place predictions at correct positions
            for i, sample_idx in enumerate(sample_indices):
                if i < len(y_vals):
                    pos = id_to_pos.get(int(sample_idx))
                    if pos is not None:
                        oof_preds[pos] = y_vals[i]

        return oof_preds, n_folds

    def _collect_test_predictions(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int,
        id_to_pos: Dict[int, int],
        n_samples: int,
        use_proba: bool = False
    ) -> np.ndarray:
        """Collect and aggregate test predictions for a source model.

        Aggregates predictions across folds according to the configured
        TestAggregation strategy.

        Args:
            model_name: Name of the source model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).
            id_to_pos: Mapping from sample ID to array position.
            n_samples: Total number of test samples.
            use_proba: If True, use probability predictions.

        Returns:
            Aggregated test predictions array.
        """
        filter_kwargs = {
            'model_name': model_name,
            'partition': 'test',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        test_predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        test_predictions = [p for p in test_predictions if p.get('step_idx', 0) < max_step]

        # If no predictions found with branch filter, try without branch filter
        # This handles models trained before any branch was created
        if not test_predictions and branch_id is not None:
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'test',
                'load_arrays': True,
            }
            test_predictions = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            test_predictions = [p for p in test_predictions if p.get('step_idx', 0) < max_step]
            # Only keep predictions that don't have a branch_id (pre-branch models)
            test_predictions = [p for p in test_predictions if p.get('branch_id') is None]

        if not test_predictions:
            return np.zeros(n_samples)

        aggregation = self.stacking_config.test_aggregation

        # Handle pre-computed averages
        if aggregation in (TestAggregation.MEAN, TestAggregation.WEIGHTED_MEAN):
            avg_preds = [p for p in test_predictions if p.get('fold_id') == 'avg']
            if avg_preds and aggregation == TestAggregation.MEAN:
                y_vals = avg_preds[0].get('y_pred', np.zeros(n_samples))
                aligned = self._align_predictions_to_positions(
                    y_vals, avg_preds[0].get('sample_indices'), id_to_pos, n_samples
                )
                return aligned if aligned is not None else np.zeros(n_samples)

            w_avg_preds = [p for p in test_predictions if p.get('fold_id') == 'w_avg']
            if w_avg_preds and aggregation == TestAggregation.WEIGHTED_MEAN:
                y_vals = w_avg_preds[0].get('y_pred', np.zeros(n_samples))
                aligned = self._align_predictions_to_positions(
                    y_vals, w_avg_preds[0].get('sample_indices'), id_to_pos, n_samples
                )
                return aligned if aligned is not None else np.zeros(n_samples)

        # Filter to individual folds
        fold_predictions = [
            p for p in test_predictions
            if (str(p.get('fold_id', '')) not in self.reconstructor_config.excluded_fold_ids
                and p.get('fold_id') is not None)
        ]

        if not fold_predictions:
            return np.zeros(n_samples)

        # Collect all fold predictions
        all_preds_list = []
        all_scores = []

        for pred in fold_predictions:
            if use_proba and 'y_proba' in pred and pred['y_proba'] is not None:
                y_vals = pred['y_proba']
                if hasattr(y_vals, 'ndim') and y_vals.ndim > 1:
                    if y_vals.shape[1] == 2:
                        y_vals = y_vals[:, 1]
                    else:
                        y_vals = y_vals[:, 0]
            else:
                y_vals = pred.get('y_pred', [])

            aligned = self._align_predictions_to_positions(
                y_vals, pred.get('sample_indices'), id_to_pos, n_samples
            )
            if aligned is not None:
                all_preds_list.append(aligned)
                all_scores.append(pred.get('val_score', 0.0) or 0.0)

        if not all_preds_list:
            return np.zeros(n_samples)

        all_preds = np.array(all_preds_list)
        all_scores = np.array(all_scores)

        # Apply aggregation strategy
        if aggregation == TestAggregation.BEST_FOLD:
            best_idx = np.argmax(all_scores) if np.any(all_scores > 0) else 0
            return all_preds[best_idx]
        elif aggregation == TestAggregation.WEIGHTED_MEAN:
            weights = np.clip(all_scores, 0, None)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(all_preds)) / len(all_preds)
            return np.average(all_preds, axis=0, weights=weights)
        else:
            # Default: MEAN
            return np.mean(all_preds, axis=0)

    def _align_predictions_to_positions(
        self,
        y_vals: Any,
        sample_indices: Any,
        id_to_pos: Dict[int, int],
        n_samples: int
    ) -> Optional[np.ndarray]:
        """Align predictions to correct array positions using sample indices.

        Args:
            y_vals: Prediction values.
            sample_indices: Sample indices for predictions.
            id_to_pos: Mapping from sample ID to array position.
            n_samples: Total number of samples.

        Returns:
            Aligned predictions array or None if alignment fails.
        """
        y_vals = np.asarray(y_vals).flatten()

        if sample_indices is None:
            # If no sample indices, assume direct correspondence
            if len(y_vals) == n_samples:
                return y_vals
            return None

        # Convert sample_indices if needed
        if hasattr(sample_indices, 'tolist'):
            sample_indices = sample_indices.tolist()

        aligned = np.zeros(n_samples)
        for i, sample_idx in enumerate(sample_indices):
            if i < len(y_vals):
                pos = id_to_pos.get(int(sample_idx))
                if pos is not None:
                    aligned[pos] = y_vals[i]

        return aligned

    def _apply_coverage_strategy(
        self,
        X_meta: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply coverage strategy to handle missing predictions.

        Args:
            X_meta: Feature matrix with potential NaN values.
            n_samples: Total number of samples.

        Returns:
            Tuple of (processed_X_meta, valid_sample_mask).

        Raises:
            ValueError: If coverage is insufficient for STRICT mode.
        """
        nan_mask = np.isnan(X_meta)
        strategy = self.stacking_config.coverage_strategy

        if strategy == CoverageStrategy.STRICT:
            if nan_mask.any():
                n_missing = nan_mask.any(axis=1).sum()
                n_features_missing = nan_mask.sum()
                raise ValueError(
                    f"Incomplete OOF coverage: {n_missing}/{n_samples} samples "
                    f"missing predictions ({n_features_missing} total missing values). "
                    f"This may indicate:\n"
                    f"  - Mismatched sample indices between model steps\n"
                    f"  - Different splitters used for different models\n"
                    f"  - Sample filtering that excludes some samples\n"
                    f"Use coverage_strategy=CoverageStrategy.DROP_INCOMPLETE or "
                    f"IMPUTE_* to handle this."
                )
            return X_meta, np.ones(n_samples, dtype=bool)

        elif strategy == CoverageStrategy.DROP_INCOMPLETE:
            complete_mask = ~nan_mask.any(axis=1)
            n_dropped = (~complete_mask).sum()

            if n_dropped > 0 and self.reconstructor_config.log_warnings:
                warnings.warn(
                    f"Dropping {n_dropped}/{n_samples} samples with incomplete "
                    f"OOF predictions (coverage_strategy=DROP_INCOMPLETE)"
                )

            # Check minimum coverage ratio
            coverage_ratio = complete_mask.sum() / n_samples
            if coverage_ratio < self.stacking_config.min_coverage_ratio:
                raise ValueError(
                    f"Coverage ratio {coverage_ratio:.1%} is below minimum "
                    f"required {self.stacking_config.min_coverage_ratio:.1%}"
                )

            # Replace NaN with 0 (samples will be masked)
            X_meta = np.nan_to_num(X_meta, nan=0.0)
            return X_meta, complete_mask

        elif strategy == CoverageStrategy.IMPUTE_ZERO:
            if nan_mask.any() and self.reconstructor_config.log_warnings:
                n_imputed = nan_mask.sum()
                warnings.warn(
                    f"Imputing {n_imputed} missing values with zeros "
                    f"(coverage_strategy=IMPUTE_ZERO)"
                )
            return np.nan_to_num(X_meta, nan=0.0), np.ones(n_samples, dtype=bool)

        elif strategy == CoverageStrategy.IMPUTE_MEAN:
            if nan_mask.any():
                if self.reconstructor_config.log_warnings:
                    n_imputed = nan_mask.sum()
                    warnings.warn(
                        f"Imputing {n_imputed} missing values with column means "
                        f"(coverage_strategy=IMPUTE_MEAN)"
                    )
                # Impute with column means
                col_means = np.nanmean(X_meta, axis=0)
                for col in range(X_meta.shape[1]):
                    mask = nan_mask[:, col]
                    X_meta[mask, col] = col_means[col] if not np.isnan(col_means[col]) else 0.0
            return X_meta, np.ones(n_samples, dtype=bool)

        elif strategy == CoverageStrategy.IMPUTE_FOLD_MEAN:
            # For IMPUTE_FOLD_MEAN, we'd need fold info per sample
            # Fall back to regular mean imputation for now
            if nan_mask.any():
                if self.reconstructor_config.log_warnings:
                    n_imputed = nan_mask.sum()
                    warnings.warn(
                        f"Imputing {n_imputed} missing values with column means "
                        f"(coverage_strategy=IMPUTE_FOLD_MEAN, falling back to mean)"
                    )
                col_means = np.nanmean(X_meta, axis=0)
                for col in range(X_meta.shape[1]):
                    mask = nan_mask[:, col]
                    X_meta[mask, col] = col_means[col] if not np.isnan(col_means[col]) else 0.0
            return X_meta, np.ones(n_samples, dtype=bool)

        # Default: just fill NaN with 0
        return np.nan_to_num(X_meta, nan=0.0), np.ones(n_samples, dtype=bool)

    def _generate_feature_names(self) -> List[str]:
        """Generate feature column names based on configuration.

        Returns:
            List of feature names.
        """
        pattern = self.reconstructor_config.feature_name_pattern
        names = []

        for model_name in self.source_model_names:
            name = pattern.format(
                model_name=model_name,
                classname="",  # Could be added if we store this info
                step_idx="",
            )
            names.append(name)

        return names

    def _get_y_values(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        partition: str
    ) -> np.ndarray:
        """Get target values for a partition.

        Args:
            dataset: SpectroDataset.
            context: Execution context.
            partition: 'train' or 'test'.

        Returns:
            Target values array.
        """
        partition_context = context.with_partition(partition)
        return dataset.y(partition_context.selector)

    def _get_y_values_for_samples(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        sample_ids: np.ndarray
    ) -> np.ndarray:
        """Get target values for specific sample IDs.

        This method is used when sample_augmentation is active and we need
        to get y values only for the original (non-augmented) samples that
        have corresponding OOF predictions.

        Args:
            dataset: SpectroDataset.
            context: Execution context.
            sample_ids: Array of sample IDs to get targets for.

        Returns:
            Target values array matching the sample_ids.
        """
        # Get all y values from the dataset for the partition
        # We need to figure out which partition based on the sample IDs
        # Try train first (most common for OOF)
        train_context = context.with_partition('train')
        all_train_ids = dataset._indexer.x_indices(
            train_context.selector,
            include_augmented=False,
            include_excluded=False
        )

        # Check if sample_ids match train or test
        sample_set = set(sample_ids)
        train_set = set(all_train_ids)

        if sample_set.issubset(train_set) or sample_set == train_set:
            # Get y values from train partition without augmented samples
            # Create a selector that explicitly selects non-augmented samples
            all_y = dataset.y(train_context.selector, include_augmented=False)
            if len(all_y) == len(sample_ids):
                return all_y

            # If lengths don't match, manually select by sample ID mapping
            all_ids = np.asarray(all_train_ids)
            id_to_pos = {int(sid): i for i, sid in enumerate(all_ids)}
            y_selected = np.zeros(len(sample_ids))
            for i, sid in enumerate(sample_ids):
                pos = id_to_pos.get(int(sid))
                if pos is not None and pos < len(all_y):
                    y_selected[i] = all_y[pos]
            return y_selected
        else:
            # Assume test partition
            test_context = context.with_partition('test')
            return dataset.y(test_context.selector, include_augmented=False)

    def validate_branch_compatibility(
        self,
        context: 'ExecutionContext'
    ) -> ValidationResult:
        """Validate branch compatibility for stacking.

        Checks that the current branch context is compatible with stacking
        based on the configured BranchScope.

        Args:
            context: Execution context with branch info.

        Returns:
            ValidationResult with any errors or warnings.
        """
        result = ValidationResult()
        branch_scope = self.stacking_config.branch_scope

        branch_id = getattr(context.selector, 'branch_id', None)
        branch_name = getattr(context.selector, 'branch_name', None)

        if branch_scope == BranchScope.CURRENT_ONLY:
            # Check that we have predictions in current branch
            if branch_id is not None:
                filter_kwargs = {'branch_id': branch_id, 'load_arrays': False}
                branch_preds = self.prediction_store.filter_predictions(**filter_kwargs)
                if not branch_preds:
                    result.add_warning(
                        "NO_BRANCH_PREDICTIONS",
                        f"No predictions found for branch {branch_id} ({branch_name}). "
                        f"Stacking will look for predictions without branch filter."
                    )

        elif branch_scope == BranchScope.ALL_BRANCHES:
            # Validate cross-branch stacking is possible
            # ALL_BRANCHES collects predictions from all branches
            result.add_warning(
                "ALL_BRANCHES_INFO",
                "BranchScope.ALL_BRANCHES is active - collecting predictions from all branches. "
                "Note: This may not work correctly with sample partitioning."
            )

        return result

    # =========================================================================
    # Phase 5: Classification Support Methods
    # =========================================================================

    def _detect_classification_info(
        self,
        context: 'ExecutionContext',
        use_proba: bool
    ) -> 'ClassificationInfo':
        """Detect classification task type from source model predictions.

        Args:
            context: Execution context.
            use_proba: Whether probability features are requested.

        Returns:
            ClassificationInfo with detected task type and metadata.
        """
        from .classification import TaskTypeDetector, ClassificationInfo, StackingTaskType

        detector = TaskTypeDetector(self.prediction_store)
        try:
            classification_info = detector.detect(
                source_model_names=self.source_model_names,
                context=context
            )
        except Exception as e:
            # Fallback to regression if detection fails
            if self.reconstructor_config.log_warnings:
                warnings.warn(
                    f"Failed to detect classification info: {e}. "
                    f"Falling back to regression task type."
                )
            classification_info = ClassificationInfo(
                task_type=StackingTaskType.REGRESSION,
                n_classes=None,
                has_probabilities=False
            )

        return classification_info

    def _collect_oof_predictions_with_proba(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int,
        id_to_pos: Dict[int, int],
        n_samples: int,
        use_proba: bool,
        classification_info: 'ClassificationInfo'
    ) -> Tuple[np.ndarray, int]:
        """Collect out-of-fold predictions with classification support.

        For classification with use_proba=True:
        - Binary: returns (n_samples,) with positive class probability
        - Multiclass: returns (n_samples, n_classes) with all class probabilities

        Args:
            model_name: Name of the source model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).
            id_to_pos: Mapping from sample ID to array position.
            n_samples: Total number of training samples.
            use_proba: Whether to extract probability features.
            classification_info: Classification metadata.

        Returns:
            Tuple of (features_array, n_folds_found).
        """
        from .classification import ClassificationFeatureExtractor

        # Determine output shape
        n_features = classification_info.get_n_features_per_model(use_proba)

        if n_features == 1:
            oof_preds = np.full(n_samples, np.nan)
        else:
            oof_preds = np.full((n_samples, n_features), np.nan)

        filter_kwargs = {
            'model_name': model_name,
            'partition': 'val',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        val_predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        val_predictions = [p for p in val_predictions if p.get('step_idx', 0) < max_step]

        # If no predictions found with branch filter, try without branch filter
        # This handles models trained before any branch was created
        if not val_predictions and branch_id is not None:
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'val',
                'load_arrays': True,
            }
            val_predictions = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            val_predictions = [p for p in val_predictions if p.get('step_idx', 0) < max_step]
            # Only keep predictions that don't have a branch_id (pre-branch models)
            val_predictions = [p for p in val_predictions if p.get('branch_id') is None]

        # Filter out averaged predictions
        val_predictions = [
            p for p in val_predictions
            if str(p.get('fold_id', '')) not in self.reconstructor_config.excluded_fold_ids
        ]

        n_folds = len(val_predictions)

        # Create feature extractor
        extractor = ClassificationFeatureExtractor(
            classification_info=classification_info,
            use_proba=use_proba
        )

        for pred in val_predictions:
            sample_indices = pred.get('sample_indices', [])
            if sample_indices is None:
                continue

            # Extract features using the extractor
            features = extractor.extract_features(pred, len(sample_indices))

            # Convert sample_indices to list if needed
            if hasattr(sample_indices, 'tolist'):
                sample_indices = sample_indices.tolist()

            # Place features at correct positions
            for i, sample_idx in enumerate(sample_indices):
                if i < len(features):
                    pos = id_to_pos.get(int(sample_idx))
                    if pos is not None:
                        if n_features == 1:
                            oof_preds[pos] = features[i] if features.ndim == 1 else features[i, 0]
                        else:
                            if features.ndim == 1:
                                oof_preds[pos, 0] = features[i]
                            else:
                                oof_preds[pos, :] = features[i, :]

        return oof_preds, n_folds

    def _collect_test_predictions_with_proba(
        self,
        model_name: str,
        branch_id: Optional[int],
        max_step: int,
        id_to_pos: Dict[int, int],
        n_samples: int,
        use_proba: bool,
        classification_info: 'ClassificationInfo'
    ) -> np.ndarray:
        """Collect and aggregate test predictions with classification support.

        Args:
            model_name: Name of the source model.
            branch_id: Branch ID filter.
            max_step: Maximum step index (exclusive).
            id_to_pos: Mapping from sample ID to array position.
            n_samples: Total number of test samples.
            use_proba: Whether to extract probability features.
            classification_info: Classification metadata.

        Returns:
            Aggregated test features array.
        """
        from .classification import ClassificationFeatureExtractor

        # Determine output shape
        n_features = classification_info.get_n_features_per_model(use_proba)

        filter_kwargs = {
            'model_name': model_name,
            'partition': 'test',
            'load_arrays': True,
        }
        if branch_id is not None:
            filter_kwargs['branch_id'] = branch_id

        test_predictions = self.prediction_store.filter_predictions(**filter_kwargs)

        # Filter by step
        test_predictions = [p for p in test_predictions if p.get('step_idx', 0) < max_step]

        # If no predictions found with branch filter, try without branch filter
        # This handles models trained before any branch was created
        if not test_predictions and branch_id is not None:
            filter_kwargs_no_branch = {
                'model_name': model_name,
                'partition': 'test',
                'load_arrays': True,
            }
            test_predictions = self.prediction_store.filter_predictions(**filter_kwargs_no_branch)
            test_predictions = [p for p in test_predictions if p.get('step_idx', 0) < max_step]
            # Only keep predictions that don't have a branch_id (pre-branch models)
            test_predictions = [p for p in test_predictions if p.get('branch_id') is None]

        if not test_predictions:
            if n_features == 1:
                return np.zeros(n_samples)
            else:
                return np.zeros((n_samples, n_features))

        aggregation = self.stacking_config.test_aggregation

        # Handle pre-computed averages (only for single-feature case)
        if n_features == 1 and aggregation in (TestAggregation.MEAN, TestAggregation.WEIGHTED_MEAN):
            avg_preds = [p for p in test_predictions if p.get('fold_id') == 'avg']
            if avg_preds and aggregation == TestAggregation.MEAN:
                y_vals = avg_preds[0].get('y_pred', np.zeros(n_samples))
                aligned = self._align_predictions_to_positions(
                    y_vals, avg_preds[0].get('sample_indices'), id_to_pos, n_samples
                )
                return aligned if aligned is not None else np.zeros(n_samples)

        # Filter to individual folds
        fold_predictions = [
            p for p in test_predictions
            if (str(p.get('fold_id', '')) not in self.reconstructor_config.excluded_fold_ids
                and p.get('fold_id') is not None)
        ]

        if not fold_predictions:
            if n_features == 1:
                return np.zeros(n_samples)
            else:
                return np.zeros((n_samples, n_features))

        # Create feature extractor
        extractor = ClassificationFeatureExtractor(
            classification_info=classification_info,
            use_proba=use_proba
        )

        # Collect all fold predictions
        all_preds_list = []
        all_scores = []

        for pred in fold_predictions:
            features = extractor.extract_features(pred, n_samples)

            # Align features to positions
            if n_features == 1:
                aligned = self._align_predictions_to_positions(
                    features, pred.get('sample_indices'), id_to_pos, n_samples
                )
                if aligned is not None:
                    all_preds_list.append(aligned)
                    all_scores.append(pred.get('val_score', 0.0) or 0.0)
            else:
                # For multiclass, align each column
                sample_indices = pred.get('sample_indices')
                if sample_indices is not None:
                    if hasattr(sample_indices, 'tolist'):
                        sample_indices = sample_indices.tolist()

                    aligned = np.zeros((n_samples, n_features))
                    for i, sample_idx in enumerate(sample_indices):
                        if i < len(features):
                            pos = id_to_pos.get(int(sample_idx))
                            if pos is not None:
                                if features.ndim == 1:
                                    aligned[pos, 0] = features[i]
                                else:
                                    aligned[pos, :] = features[i, :] if i < features.shape[0] else 0.0

                    all_preds_list.append(aligned)
                    all_scores.append(pred.get('val_score', 0.0) or 0.0)

        if not all_preds_list:
            if n_features == 1:
                return np.zeros(n_samples)
            else:
                return np.zeros((n_samples, n_features))

        all_preds = np.array(all_preds_list)
        all_scores = np.array(all_scores)

        # Apply aggregation strategy
        if aggregation == TestAggregation.BEST_FOLD:
            best_idx = np.argmax(all_scores) if np.any(all_scores > 0) else 0
            return all_preds[best_idx]
        elif aggregation == TestAggregation.WEIGHTED_MEAN:
            weights = np.clip(all_scores, 0, None)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(all_preds)) / len(all_preds)
            return np.average(all_preds, axis=0, weights=weights)
        else:
            # Default: MEAN
            return np.mean(all_preds, axis=0)

    def _generate_feature_names_with_classification(
        self,
        classification_info: 'ClassificationInfo',
        use_proba: bool
    ) -> List[str]:
        """Generate feature names with classification support.

        Args:
            classification_info: Classification metadata.
            use_proba: Whether probability features are used.

        Returns:
            List of feature column names.
        """
        from .classification import FeatureNameGenerator

        generator = FeatureNameGenerator(
            classification_info=classification_info,
            use_proba=use_proba,
            pattern=self.reconstructor_config.feature_name_pattern
        )

        return generator.generate_names(self.source_model_names)

    def _build_meta_feature_info(
        self,
        classification_info: 'ClassificationInfo',
        use_proba: bool,
        feature_names: List[str]
    ) -> 'MetaFeatureInfo':
        """Build MetaFeatureInfo for feature importance tracking.

        Args:
            classification_info: Classification metadata.
            use_proba: Whether probability features are used.
            feature_names: Generated feature names.

        Returns:
            MetaFeatureInfo with all mappings populated.
        """
        from .classification import build_meta_feature_info

        return build_meta_feature_info(
            source_model_names=self.source_model_names,
            classification_info=classification_info,
            use_proba=use_proba,
            name_pattern=self.reconstructor_config.feature_name_pattern
        )
