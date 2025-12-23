"""
Meta-Model Serialization - Artifact persistence for meta-model stacking.

This module provides dataclasses and utilities for persisting meta-model
artifacts with complete source model dependency tracking.

Phase 3 Implementation - Key components:
1. SourceModelReference: Reference to a source model with feature mapping
2. MetaModelArtifact: Complete artifact for meta-model persistence
3. MetaModelSerializer: Handles serialization/deserialization

The meta-model serialization captures:
- The trained meta-learner itself (via artifact_registry)
- Ordered references to source models (for feature column alignment)
- Stacking configuration (coverage strategy, aggregation, etc.)
- Branch context (for validation during prediction)
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import warnings

from nirs4all.operators.models.meta import (
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
)

if TYPE_CHECKING:
    from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord, MetaModelConfig
    from nirs4all.operators.models.meta import MetaModel
    from nirs4all.operators.models.selection import ModelCandidate
    from nirs4all.pipeline.config.context import ExecutionContext
    from .reconstructor import ReconstructionResult


@dataclass
class SourceModelReference:
    """Reference to a source model used in stacking.

    Stores all information needed to locate and validate a source model
    during prediction mode.

    Attributes:
        model_name: Display name of the model (e.g., "PLSRegression").
        model_classname: Full class name (e.g., "sklearn.cross_decomposition.PLSRegression").
        step_idx: Pipeline step index where the model was trained.
        artifact_id: Unique artifact ID for loading the model binary.
        feature_index: Column index in meta-features matrix.
        fold_id: Optional fold ID if fold-specific reference.
        branch_id: Branch ID where model was trained.
        branch_name: Branch name where model was trained.
        branch_path: Full branch path for nested branches.
        val_score: Validation score for weighted averaging.
        metric: Metric used for scoring (e.g., "r2", "rmse").

    Example:
        >>> ref = SourceModelReference(
        ...     model_name="PLSRegression",
        ...     model_classname="sklearn.cross_decomposition.PLSRegression",
        ...     step_idx=3,
        ...     artifact_id="0001:3:all",
        ...     feature_index=0,
        ...     branch_id=None,
        ...     val_score=0.92,
        ...     metric="r2"
        ... )
    """

    model_name: str
    model_classname: str
    step_idx: int
    artifact_id: str
    feature_index: int
    fold_id: Optional[str] = None
    branch_id: Optional[int] = None
    branch_name: Optional[str] = None
    branch_path: Optional[List[int]] = None
    val_score: Optional[float] = None
    metric: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML serialization."""
        return {
            "model_name": self.model_name,
            "model_classname": self.model_classname,
            "step_idx": self.step_idx,
            "artifact_id": self.artifact_id,
            "feature_index": self.feature_index,
            "fold_id": self.fold_id,
            "branch_id": self.branch_id,
            "branch_name": self.branch_name,
            "branch_path": self.branch_path,
            "val_score": self.val_score,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceModelReference":
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", ""),
            model_classname=data.get("model_classname", ""),
            step_idx=data.get("step_idx", 0),
            artifact_id=data.get("artifact_id", ""),
            feature_index=data.get("feature_index", 0),
            fold_id=data.get("fold_id"),
            branch_id=data.get("branch_id"),
            branch_name=data.get("branch_name"),
            branch_path=data.get("branch_path"),
            val_score=data.get("val_score"),
            metric=data.get("metric"),
        )


@dataclass
class MetaModelArtifact:
    """Complete artifact for meta-model persistence.

    Contains all information needed to:
    - Reload the meta-model and its dependencies
    - Reconstruct feature columns in the correct order
    - Validate branch context during prediction
    - Apply the same stacking configuration

    Attributes:
        meta_model_type: Type identifier ("MetaModel").
        meta_model_name: Display name of the meta-model.
        meta_learner_class: Class name of the meta-learner (e.g., "Ridge").
        source_models: Ordered list of source model references.
        feature_columns: Feature column names in order.
        stacking_config: Serialized stacking configuration.
        selector_config: Configuration of the model selector used.
        branch_context: Branch context during training.
        use_proba: Whether probability features were used.
        n_folds: Number of cross-validation folds.
        coverage_ratio: OOF coverage ratio achieved during training.
        artifact_id: The artifact ID for the meta-model itself.
        training_timestamp: ISO timestamp of training.

    Example:
        >>> artifact = MetaModelArtifact(
        ...     meta_model_type="MetaModel",
        ...     meta_model_name="MetaModel_Ridge",
        ...     meta_learner_class="Ridge",
        ...     source_models=[ref1, ref2],
        ...     feature_columns=["PLS_pred", "RF_pred"],
        ...     stacking_config=stacking_config_dict,
        ...     branch_context={"branch_id": None},
        ...     use_proba=False,
        ...     n_folds=5,
        ...     coverage_ratio=1.0,
        ...     artifact_id="0001:5:all",
        ...     training_timestamp="2024-12-12T14:30:22Z"
        ... )
    """

    meta_model_type: str
    meta_model_name: str
    meta_learner_class: str
    source_models: List[SourceModelReference]
    feature_columns: List[str]
    stacking_config: Dict[str, Any]
    selector_config: Optional[Dict[str, Any]] = None
    branch_context: Optional[Dict[str, Any]] = None
    use_proba: bool = False
    n_folds: int = 0
    coverage_ratio: float = 1.0
    artifact_id: str = ""
    training_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Phase 5: Classification support fields
    task_type: str = "regression"  # "regression", "binary_classification", "multiclass_classification"
    n_classes: Optional[int] = None  # Number of classes for classification tasks
    feature_to_model_mapping: Optional[Dict[str, str]] = None  # Feature name -> source model name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML serialization."""
        return {
            "meta_model_type": self.meta_model_type,
            "meta_model_name": self.meta_model_name,
            "meta_learner_class": self.meta_learner_class,
            "source_models": [ref.to_dict() for ref in self.source_models],
            "feature_columns": self.feature_columns,
            "stacking_config": self.stacking_config,
            "selector_config": self.selector_config,
            "branch_context": self.branch_context,
            "use_proba": self.use_proba,
            "n_folds": self.n_folds,
            "coverage_ratio": self.coverage_ratio,
            "artifact_id": self.artifact_id,
            "training_timestamp": self.training_timestamp,
            # Phase 5: Classification support
            "task_type": self.task_type,
            "n_classes": self.n_classes,
            "feature_to_model_mapping": self.feature_to_model_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaModelArtifact":
        """Create from dictionary."""
        source_models = [
            SourceModelReference.from_dict(ref)
            for ref in data.get("source_models", [])
        ]
        return cls(
            meta_model_type=data.get("meta_model_type", "MetaModel"),
            meta_model_name=data.get("meta_model_name", ""),
            meta_learner_class=data.get("meta_learner_class", ""),
            source_models=source_models,
            feature_columns=data.get("feature_columns", []),
            stacking_config=data.get("stacking_config", {}),
            selector_config=data.get("selector_config"),
            branch_context=data.get("branch_context"),
            use_proba=data.get("use_proba", False),
            n_folds=data.get("n_folds", 0),
            coverage_ratio=data.get("coverage_ratio", 1.0),
            artifact_id=data.get("artifact_id", ""),
            training_timestamp=data.get("training_timestamp", ""),
            # Phase 5: Classification support
            task_type=data.get("task_type", "regression"),
            n_classes=data.get("n_classes"),
            feature_to_model_mapping=data.get("feature_to_model_mapping"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MetaModelArtifact":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_source_artifact_ids(self) -> List[str]:
        """Get ordered list of source model artifact IDs.

        Returns:
            List of artifact IDs in feature column order.
        """
        return [ref.artifact_id for ref in self.source_models]

    def get_source_by_index(self, index: int) -> Optional[SourceModelReference]:
        """Get source model reference by feature index.

        Args:
            index: Feature column index.

        Returns:
            SourceModelReference or None if index out of range.
        """
        for ref in self.source_models:
            if ref.feature_index == index:
                return ref
        return None

    def validate_feature_alignment(self) -> bool:
        """Validate that feature columns match source models.

        Returns:
            True if alignment is valid.
        """
        if len(self.feature_columns) != len(self.source_models):
            return False

        # Check feature indices are sequential and match
        for idx, ref in enumerate(self.source_models):
            if ref.feature_index != idx:
                return False

        return True


def stacking_config_to_dict(config: StackingConfig) -> Dict[str, Any]:
    """Convert StackingConfig to serializable dictionary.

    Args:
        config: StackingConfig instance.

    Returns:
        Dictionary with string enum values.
    """
    return {
        "coverage_strategy": config.coverage_strategy.value,
        "test_aggregation": config.test_aggregation.value,
        "branch_scope": config.branch_scope.value,
        "allow_no_cv": config.allow_no_cv,
        "min_coverage_ratio": config.min_coverage_ratio,
    }


def stacking_config_from_dict(data: Dict[str, Any]) -> StackingConfig:
    """Create StackingConfig from dictionary.

    Args:
        data: Dictionary with config values.

    Returns:
        StackingConfig instance.
    """
    return StackingConfig(
        coverage_strategy=CoverageStrategy(data.get("coverage_strategy", "strict")),
        test_aggregation=TestAggregation(data.get("test_aggregation", "mean")),
        branch_scope=BranchScope(data.get("branch_scope", "current_only")),
        allow_no_cv=data.get("allow_no_cv", False),
        min_coverage_ratio=data.get("min_coverage_ratio", 1.0),
    )


class MetaModelSerializer:
    """Handles serialization and deserialization of meta-model artifacts.

    Provides methods to:
    - Build MetaModelArtifact from training context
    - Convert to/from MetaModelConfig for artifact registry
    - Validate artifact completeness

    Example:
        >>> serializer = MetaModelSerializer()
        >>> artifact = serializer.build_artifact(
        ...     meta_operator=meta_model_op,
        ...     source_models=selected_sources,
        ...     reconstruction_result=result,
        ...     context=execution_context
        ... )
        >>> config = serializer.to_meta_model_config(artifact)
    """

    def build_artifact(
        self,
        meta_operator: 'MetaModel',
        source_models: List['ModelCandidate'],
        reconstruction_result: Optional['ReconstructionResult'] = None,
        context: Optional['ExecutionContext'] = None,
        artifact_id: str = "",
    ) -> MetaModelArtifact:
        """Build MetaModelArtifact from training context.

        Args:
            meta_operator: The MetaModel operator being trained.
            source_models: List of selected source model candidates.
            reconstruction_result: Optional result from TrainingSetReconstructor.
            context: Optional execution context for branch info.
            artifact_id: The artifact ID for this meta-model.

        Returns:
            MetaModelArtifact ready for persistence.
        """
        # Import here to avoid circular imports
        from nirs4all.operators.models.meta import MetaModel
        from nirs4all.operators.models.selection import ModelCandidate

        # Build source model references
        source_refs = []
        feature_columns = []

        # Build unique names for models, handling cross-branch duplicates
        # Count unique branches per model_name
        name_branch_pairs = set()
        for candidate in source_models:
            name_branch_pairs.add((candidate.model_name, candidate.branch_id))

        branch_count_per_name: Dict[str, int] = {}
        for name, branch_id in name_branch_pairs:
            branch_count_per_name[name] = branch_count_per_name.get(name, 0) + 1

        # Models needing branch suffix are those appearing in multiple branches
        needs_branch_suffix = {
            name for name, count in branch_count_per_name.items() if count > 1
        }

        # Build unique source list with branch-aware deduplication
        seen_unique = set()
        unique_sources: List[Tuple['ModelCandidate', str]] = []  # (candidate, unique_name)
        for candidate in source_models:
            model_name = candidate.model_name
            branch_id = candidate.branch_id

            if model_name in needs_branch_suffix:
                if branch_id is not None:
                    unique_name = f"{model_name}_br{branch_id}"
                else:
                    unique_name = f"{model_name}_br_none"
            else:
                unique_name = model_name

            if unique_name not in seen_unique:
                seen_unique.add(unique_name)
                unique_sources.append((candidate, unique_name))

        for idx, (candidate, unique_name) in enumerate(unique_sources):
            ref = SourceModelReference(
                model_name=unique_name,  # Use unique name for persistence
                model_classname=candidate.model_classname,
                step_idx=candidate.step_idx,
                artifact_id=self._generate_source_artifact_id(candidate, context),
                feature_index=idx,
                fold_id=candidate.fold_id,
                branch_id=candidate.branch_id,
                branch_name=candidate.branch_name,
                val_score=candidate.val_score,
                metric=candidate.metric,
            )
            source_refs.append(ref)
            feature_columns.append(f"{unique_name}_pred")

        # Get stacking config
        stacking_config_dict = stacking_config_to_dict(meta_operator.stacking_config)

        # Get selector config
        selector_config = None
        if meta_operator.selector is not None:
            selector_class = meta_operator.selector.__class__.__name__
            selector_config = {
                "type": selector_class,
                "params": getattr(meta_operator.selector, 'get_params', lambda: {})()
            }
        elif meta_operator.source_models == "all":
            selector_config = {"type": "AllPreviousModelsSelector", "params": {}}
        elif isinstance(meta_operator.source_models, list):
            selector_config = {
                "type": "ExplicitModelSelector",
                "params": {"model_names": meta_operator.source_models}
            }

        # Get branch context
        branch_context = None
        if context is not None:
            branch_context = {
                "branch_id": getattr(context.selector, 'branch_id', None),
                "branch_name": getattr(context.selector, 'branch_name', None),
                "branch_path": getattr(context.selector, 'branch_path', None),
            }

        # Get reconstruction info and classification info
        n_folds = 0
        coverage_ratio = 1.0
        task_type = "regression"
        n_classes = None
        feature_to_model_mapping = None

        if reconstruction_result is not None:
            n_folds = reconstruction_result.n_folds
            coverage_ratio = reconstruction_result.coverage_ratio

            # Phase 5: Extract classification info from reconstruction result
            classification_info = getattr(reconstruction_result, 'classification_info', None)
            if classification_info is not None:
                task_type = classification_info.task_type.value
                n_classes = classification_info.n_classes

            # Extract feature to model mapping
            meta_feature_info = getattr(reconstruction_result, 'meta_feature_info', None)
            if meta_feature_info is not None:
                feature_to_model_mapping = meta_feature_info.feature_to_model
                # Update feature_columns from reconstruction result
                feature_columns = reconstruction_result.feature_names

        return MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name=meta_operator.name,
            meta_learner_class=type(meta_operator.model).__name__,
            source_models=source_refs,
            feature_columns=feature_columns,
            stacking_config=stacking_config_dict,
            selector_config=selector_config,
            branch_context=branch_context,
            use_proba=meta_operator.use_proba,
            n_folds=n_folds,
            coverage_ratio=coverage_ratio,
            artifact_id=artifact_id,
            # Phase 5: Classification support
            task_type=task_type,
            n_classes=n_classes,
            feature_to_model_mapping=feature_to_model_mapping,
        )

    def _generate_source_artifact_id(
        self,
        candidate: 'ModelCandidate',
        context: Optional['ExecutionContext'] = None
    ) -> str:
        """Get or generate artifact ID for a source model.

        First tries to look up the actual artifact ID from the registry.
        Falls back to generating a V2-style ID for compatibility.

        Args:
            candidate: Source model candidate.
            context: Optional execution context for pipeline info.

        Returns:
            Artifact ID string (V3 format if found in registry, V2 fallback otherwise).
        """
        # Get pipeline_id from context if available
        pipeline_id = "pipeline"
        runtime_context = None
        if context is not None:
            # Try to get from custom context
            runtime_context = context.custom.get('_runtime_context')
            if runtime_context is not None and hasattr(runtime_context, 'saver'):
                saver = runtime_context.saver
                if saver is not None:
                    pipeline_id = getattr(saver, 'pipeline_id', 'pipeline')

        # Try to look up actual artifact ID from registry (V3 approach)
        if runtime_context is not None and runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry
            # Build branch_path from candidate
            branch_path = [candidate.branch_id] if candidate.branch_id is not None else []
            # Parse fold_id (may be "avg", "w_avg", or numeric)
            fold_id = None
            if candidate.fold_id is not None:
                if isinstance(candidate.fold_id, int):
                    fold_id = candidate.fold_id
                elif isinstance(candidate.fold_id, str) and candidate.fold_id.isdigit():
                    fold_id = int(candidate.fold_id)
                # "avg" and "w_avg" remain as None fold_id

            # Look up artifacts for this step
            step_artifacts = registry.get_artifacts_for_step(
                pipeline_id=pipeline_id,
                step_index=candidate.step_idx
            )
            for record in step_artifacts:
                # Match by branch and fold first
                record_branch = record.branch_path or []
                if record_branch != branch_path or record.fold_id != fold_id:
                    continue

                # Match by class name OR custom_name (model_name)
                # Custom name match handles MetaModel case where:
                # - record.class_name = "Ridge" (underlying meta-learner)
                # - candidate.model_classname = "MetaModel" (operator class)
                # - record.custom_name = "Ridge_MetaModel" (model_name)
                # - candidate.model_name = "Ridge_MetaModel" (from predictions)
                class_match = record.class_name == candidate.model_classname
                name_match = record.custom_name and record.custom_name == candidate.model_name
                if class_match or name_match:
                    return record.artifact_id

        # Fallback: generate V2-style ID for compatibility
        branch_path = []
        if candidate.branch_id is not None:
            branch_path = [candidate.branch_id]

        # Simple format: pipeline_id:step:fold
        branch_str = ':'.join(str(b) for b in branch_path) if branch_path else ""
        fold_str = str(candidate.fold_id) if candidate.fold_id else "all"

        if branch_str:
            return f"{pipeline_id}:{branch_str}:{candidate.step_idx}:{fold_str}"
        else:
            return f"{pipeline_id}:{candidate.step_idx}:{fold_str}"

    def to_meta_model_config(self, artifact: MetaModelArtifact) -> 'MetaModelConfig':
        """Convert MetaModelArtifact to MetaModelConfig for registry.

        The ArtifactRegistry uses MetaModelConfig to track source model
        dependencies. This method creates the appropriate config.

        Args:
            artifact: MetaModelArtifact to convert.

        Returns:
            MetaModelConfig for artifact registry.
        """
        from nirs4all.pipeline.storage.artifacts.types import MetaModelConfig

        source_models = [
            {
                "artifact_id": ref.artifact_id,
                "feature_index": ref.feature_index,
                "model_name": ref.model_name,
            }
            for ref in artifact.source_models
        ]

        return MetaModelConfig(
            source_models=source_models,
            feature_columns=artifact.feature_columns
        )

    def validate_artifact(self, artifact: MetaModelArtifact) -> List[str]:
        """Validate artifact completeness and consistency.

        Args:
            artifact: MetaModelArtifact to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check required fields
        if not artifact.meta_model_name:
            errors.append("Missing meta_model_name")
        if not artifact.meta_learner_class:
            errors.append("Missing meta_learner_class")
        if not artifact.source_models:
            errors.append("No source models defined")
        if not artifact.feature_columns:
            errors.append("No feature columns defined")

        # Check feature alignment
        if len(artifact.feature_columns) != len(artifact.source_models):
            errors.append(
                f"Feature column count ({len(artifact.feature_columns)}) "
                f"doesn't match source model count ({len(artifact.source_models)})"
            )

        # Check feature indices are sequential
        if artifact.source_models:
            indices = [ref.feature_index for ref in artifact.source_models]
            expected = list(range(len(indices)))
            if sorted(indices) != expected:
                errors.append(f"Non-sequential feature indices: {indices}")

        # Check all source models have artifact IDs
        for ref in artifact.source_models:
            if not ref.artifact_id:
                errors.append(f"Source model {ref.model_name} missing artifact_id")

        return errors
