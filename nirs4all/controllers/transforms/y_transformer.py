from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.storage.artifacts.types import ArtifactType

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep

import numpy as np


def _is_transformer_like(obj: Any) -> bool:
    """Check if object is a TransformerMixin instance or class."""
    # Instance check
    if isinstance(obj, TransformerMixin):
        return True
    # Class check (e.g., StandardScaler without parentheses)
    if isinstance(obj, type) and issubclass(obj, TransformerMixin):
        return True
    # Fallback for edge cases
    if hasattr(obj, '__class__') and hasattr(obj.__class__, '__mro__'):
        return TransformerMixin in obj.__class__.__mro__
    return False


@register_controller
class YTransformerMixinController(OperatorController):
    """
    Controller for applying sklearn TransformerMixin operators to targets (y) instead of features (X).

    Triggered by the "y_processing" keyword and applies transformations to target data,
    fitting on train targets and transforming all target data.

    Supports both single transformers and chained transformers (list syntax):
        - Single: {"y_processing": StandardScaler()}
        - Chained: {"y_processing": [StandardScaler, QuantileTransformer(n_quantiles=30)]}

    When using chained transformers, each transformer is applied sequentially,
    with proper ancestry tracking and individual artifact persistence for prediction mode.
    """
    priority = 5

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match if keyword is 'y_processing' and operator is TransformerMixin or list thereof.

        Args:
            step: Original step configuration
            operator: Parsed operator (TransformerMixin instance, class, or list)
            keyword: Step keyword

        Returns:
            True if this controller should handle the step
        """
        if keyword != "y_processing":
            return False

        # Single transformer (instance or class)
        if _is_transformer_like(operator):
            return True

        # List of transformers
        if isinstance(operator, (list, tuple)) and len(operator) > 0:
            return all(_is_transformer_like(t) for t in operator)

        return False

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False  # Target processing doesn't depend on multiple sources

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Y transformers should not execute during prediction mode."""
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ) -> Tuple[ExecutionContext, List[Any]]:
        """
        Execute transformer(s) on dataset targets, fitting on train targets and transforming all targets.

        Supports both single transformers and chained transformers (list).
        Each transformer is applied sequentially, with proper ancestry tracking.

        Args:
            step_info: Parsed step containing operator and metadata
            dataset: Dataset containing targets to transform
            context: Pipeline context with partition information
            runtime_context: Runtime context containing infrastructure components
            source: Source index (not used for target processing)
            mode: Execution mode ("train", "predict", or "explain")
            loaded_binaries: Pre-loaded fitted transformers for predict/explain mode
            prediction_store: Not used for y_processing

        Returns:
            Tuple of (updated_context, fitted_transformers_list)
        """
        operator = step_info.operator

        # Normalize to list and instantiate class types
        operators = self._normalize_operators(operator)

        # Convert loaded_binaries to dict for easy lookup by name
        binaries_dict = {}
        if loaded_binaries:
            binaries_dict = {name: obj for name, obj in loaded_binaries}

        # Execute each transformer sequentially
        current_context = context
        all_artifacts = []

        for idx, op in enumerate(operators):
            current_context, artifacts = self._execute_single_transformer(
                transformer=op,
                transformer_index=idx,
                dataset=dataset,
                context=current_context,
                runtime_context=runtime_context,
                mode=mode,
                binaries_dict=binaries_dict
            )
            all_artifacts.extend(artifacts)

        return current_context, all_artifacts

    def _normalize_operators(self, operator: Any) -> List[TransformerMixin]:
        """Normalize operator(s) to a list of instantiated TransformerMixin objects.

        Args:
            operator: Single transformer, class, or list thereof

        Returns:
            List of instantiated TransformerMixin objects
        """
        # Handle single transformer
        if not isinstance(operator, (list, tuple)):
            operators = [operator]
        else:
            operators = list(operator)

        # Instantiate class types (e.g., StandardScaler vs StandardScaler())
        instantiated = []
        for op in operators:
            if isinstance(op, type) and issubclass(op, TransformerMixin):
                instantiated.append(op())
            else:
                instantiated.append(op)

        return instantiated

    def _execute_single_transformer(
        self,
        transformer: TransformerMixin,
        transformer_index: int,
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        mode: str,
        binaries_dict: Dict[str, Any]
    ) -> Tuple[ExecutionContext, List[Any]]:
        """Execute a single transformer on dataset targets.

        Args:
            transformer: The transformer to apply
            transformer_index: Index in the chain (for naming)
            dataset: Dataset containing targets
            context: Current execution context
            runtime_context: Runtime context for artifact persistence
            mode: Execution mode
            binaries_dict: Pre-loaded binaries dict for predict/explain mode

        Returns:
            Tuple of (updated_context, artifacts_list)
        """
        from sklearn.base import clone

        # Generate unique name for this transformer
        operator_name = transformer.__class__.__name__
        op_id = runtime_context.next_op()
        current_y_processing = context.state.y_processing
        new_processing_name = f"{current_y_processing}_{operator_name}{op_id}"

        # Artifact name for saving/loading (includes index for ordering in chained transformers)
        artifact_name = f"y_{operator_name}_{op_id}"

        # Handle prediction/explain mode: load pre-fitted transformer
        if mode in ("predict", "explain"):
            # Try to find the transformer in loaded binaries
            fitted_transformer = binaries_dict.get(artifact_name)

            if fitted_transformer is None:
                # Fallback: try without index suffix for backward compatibility
                legacy_name = f"y_{operator_name}"
                fitted_transformer = binaries_dict.get(legacy_name)

            if fitted_transformer is None:
                # Fallback: search by class name pattern (handles v2 artifact system)
                # Look for any key matching y_{class_name}_{number}
                import re
                pattern = re.compile(rf'^y_{re.escape(operator_name)}_(\d+)$')
                for key, obj in binaries_dict.items():
                    if pattern.match(key):
                        fitted_transformer = obj
                        break

            if fitted_transformer is not None:
                dataset._targets.add_processed_targets(
                    processing_name=new_processing_name,
                    targets=np.array([]),
                    ancestor=current_y_processing,
                    transformer=fitted_transformer,
                    mode=mode
                )
                updated_context = context.with_y(new_processing_name)
                return updated_context, []
            else:
                # No pre-fitted transformer found - this shouldn't happen in proper predict mode
                raise ValueError(
                    f"No fitted transformer found for '{artifact_name}' in loaded binaries. "
                    f"Available: {list(binaries_dict.keys())}"
                )

        # Training mode: fit and transform

        # Get train targets for fitting (excluding filtered samples)
        train_context = context.with_partition("train")
        train_y_selector = dict(train_context.selector)
        train_y_selector['y'] = current_y_processing
        train_data = dataset.y(train_y_selector, include_excluded=False)

        # Get all targets for transformation (INCLUDING excluded samples)
        # This is necessary because add_processed_targets expects targets for ALL samples
        all_y_selector = dict(context.selector)
        all_y_selector['y'] = current_y_processing
        all_data = dataset.y(all_y_selector, include_excluded=True)

        # Clone, fit, and transform
        fitted_transformer = clone(transformer)
        fitted_transformer.fit(train_data)
        transformed_targets = fitted_transformer.transform(all_data)

        # Add processed targets to dataset with proper ancestry
        dataset.add_processed_targets(
            processing_name=new_processing_name,
            targets=transformed_targets,
            ancestor_processing=current_y_processing,
            transformer=fitted_transformer
        )

        # Update context to use new processing
        updated_context = context.with_y(new_processing_name)

        # Persist fitted transformer
        artifacts = []
        if mode == "train":
            artifact = self._persist_y_transformer(
                runtime_context=runtime_context,
                transformer=fitted_transformer,
                name=artifact_name,
                context=context
            )
            artifacts.append(artifact)

        return updated_context, artifacts

    def _persist_y_transformer(
        self,
        runtime_context: 'RuntimeContext',
        transformer: Any,
        name: str,
        context: ExecutionContext
    ) -> Any:
        """Persist fitted Y transformer using artifact registry or legacy saver.

        Uses artifact_registry.register() if available (v2 system), otherwise
        falls back to saver.persist_artifact() (legacy).

        Args:
            runtime_context: Runtime context with saver/registry instances.
            transformer: Fitted transformer to persist.
            name: Artifact name for the transformer.
            context: Execution context with branch information.

        Returns:
            ArtifactRecord (v2) or ArtifactMeta (legacy) with persistence metadata.
        """
        # Use artifact registry if available (v2 system)
        if runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry
            pipeline_id = runtime_context.saver.pipeline_id if runtime_context.saver else "unknown"
            step_index = runtime_context.step_number
            branch_path = context.selector.branch_path or []

            # Generate deterministic artifact ID
            artifact_id = registry.generate_id(
                pipeline_id=pipeline_id,
                branch_path=branch_path,
                step_index=step_index,
                fold_id=None  # Y transformers are shared across folds
            )

            # Register artifact with registry (use ENCODER type for y transformers)
            return registry.register(
                obj=transformer,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.ENCODER,
                format_hint='sklearn'
            )

        # Fallback to legacy saver.persist_artifact()
        return runtime_context.saver.persist_artifact(
            step_number=runtime_context.step_number,
            name=name,
            obj=transformer,
            format_hint='sklearn',
            branch_id=context.selector.branch_id,
            branch_name=context.selector.branch_name
        )

