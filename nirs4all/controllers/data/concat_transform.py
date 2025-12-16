"""
Concat Augmentation Controller.

This module provides the ConcatAugmentationController for concatenating
multiple transformer outputs horizontally. It can either:
- REPLACE each processing with concatenated versions (top-level usage)
- ADD a new processing with concatenated output (inside feature_augmentation)
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from sklearn.base import clone, TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.config.component_serialization import deserialize_component

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class ConcatAugmentationController(OperatorController):
    """
    Controller that concatenates multiple transformer outputs.

    Semantics:
    - Top-level (add_feature=False): REPLACES each processing with concatenated version
    - Inside feature_augmentation (add_feature=True): ADDS one new processing

    Supports:
    - Single transformers: PCA(50)
    - Chained transformers: [Wavelet(), PCA(50)] → sequential application
    - Mixed: [PCA(50), [Wavelet(), SVD(30)], LocalStats()]

    Examples:
        Top-level replacement:
        >>> pipeline = [{"concat_transform": [PCA(50), SVD(50)]}]
        # Before: (500, 3, 500) with ["raw", "snv", "savgol"]
        # After:  (500, 3, 100) with ["raw_concat_PCA_SVD", "snv_concat_PCA_SVD", ...]

        Nested inside feature_augmentation:
        >>> pipeline = [{
        ...     "feature_augmentation": [
        ...         SNV(),
        ...         {"concat_transform": [PCA(50), SVD(50)]}
        ...     ]
        ... }]
        # Before: (500, 1, 500) with ["raw"]
        # After:  (500, 3, 500) with ["raw", "snv", "concat_PCA_SVD"] (padded)
    """

    priority = 10

    @staticmethod
    def normalize_generator_spec(spec: Any) -> Any:
        """Normalize generator spec for concat_transform context.

        In concat_transform context, multi-selection should use combinations
        by default since the order of concatenated features doesn't matter.
        Translates legacy 'size' to 'pick' for explicit semantics.

        Args:
            spec: Generator specification (may contain _or_, size, pick, arrange).

        Returns:
            Normalized spec with 'size' converted to 'pick' if needed.
        """
        if not isinstance(spec, dict):
            return spec

        # If explicit pick/arrange specified, honor it
        if "pick" in spec or "arrange" in spec:
            return spec

        # Convert legacy size to pick (combinations) for concat_transform
        if "size" in spec and "_or_" in spec:
            result = dict(spec)
            result["pick"] = result.pop("size")
            return result

        return spec

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if step is a concat_transform operation."""
        return keyword == "concat_transform"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Supports prediction mode for applying saved transformers."""
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """
        Execute concat augmentation.

        Args:
            step_info: Parsed step containing the concat_transform config
            dataset: SpectroDataset to operate on
            context: Execution context with selector and metadata
            runtime_context: Runtime infrastructure (saver, step_number, etc.)
            source: Source index (-1 for all sources)
            mode: Execution mode ("train", "predict", "explain")
            loaded_binaries: Pre-fitted transformers for predict/explain mode
            prediction_store: Not used by this controller

        Returns:
            Tuple of (updated_context, list_of_artifacts)
        """
        config = self._parse_config(step_info.original_step["concat_transform"])
        operations = config["operations"]
        output_name_base = config.get("name")
        source_processing_filter = config.get("source_processing")

        if not operations:
            return context, []

        # Determine mode: replace or add
        is_add_mode = context.metadata.add_feature

        all_artifacts = []
        n_sources = dataset.features_sources()

        for sd_idx in range(n_sources):
            processing_ids = list(dataset.features_processings(sd_idx))

            # Determine which processings to operate on
            if source_processing_filter:
                target_processings = [p for p in processing_ids if p == source_processing_filter]
            else:
                # In add mode, only apply to the first processing (or raw)
                # In replace mode, apply to all processings
                target_processings = [processing_ids[0]] if is_add_mode else list(processing_ids)

            # Get data
            train_context = context.with_partition("train")
            train_data = dataset.x(train_context.selector, "3d", concat_source=False)
            all_data = dataset.x(context.selector, "3d", concat_source=False)

            if not isinstance(train_data, list):
                train_data = [train_data]
            if not isinstance(all_data, list):
                all_data = [all_data]

            source_train = train_data[sd_idx]
            source_all = all_data[sd_idx]

            # Track new processing names for context update
            replaced_processings = []
            new_processing_names_for_source = []

            # Process each target processing
            for proc_name in target_processings:
                proc_idx = processing_ids.index(proc_name)
                train_2d = source_train[:, proc_idx, :]
                all_2d = source_all[:, proc_idx, :]

                # Apply all operations and concatenate
                concat_blocks = []
                for op_idx, operation in enumerate(operations):
                    # Handle None operations - pass through unchanged
                    if operation is None:
                        concat_blocks.append(all_2d)
                        continue

                    op_name_base = self._get_operation_name(operation, op_idx)
                    binary_key = f"{proc_name}_{op_name_base}"

                    # Handle nested concat_transform dict
                    if isinstance(operation, dict) and "concat_transform" in operation:
                        transformed, nested_artifacts = self._execute_nested_concat(
                            operation["concat_transform"], train_2d, all_2d, binary_key,
                            mode, loaded_binaries, runtime_context, context
                        )
                        all_artifacts.extend(nested_artifacts)
                    # Handle chain vs single transformer
                    elif isinstance(operation, list):
                        # Chain: [A, B, C] → C(B(A(X)))
                        transformed, chain_artifacts = self._execute_chain(
                            operation, train_2d, all_2d, binary_key,
                            mode, loaded_binaries, runtime_context, context
                        )
                        all_artifacts.extend(chain_artifacts)
                    else:
                        # Single transformer
                        transformed, artifact = self._execute_single(
                            operation, train_2d, all_2d, binary_key,
                            mode, loaded_binaries, runtime_context, context
                        )
                        if artifact:
                            all_artifacts.append(artifact)

                    concat_blocks.append(transformed)

                # Concatenate all blocks horizontally
                concatenated = np.hstack(concat_blocks)

                # Generate output processing name
                output_name = self._generate_output_name(
                    proc_name, operations, output_name_base, is_add_mode
                )

                # Apply result to dataset
                if is_add_mode:
                    # ADD mode: add as new processing (will be padded by Features)
                    dataset.update_features(
                        source_processings=[""],  # Empty string means add new
                        features=[concatenated],
                        processings=[output_name],
                        source=sd_idx
                    )
                    new_processing_names_for_source.append(output_name)
                    break  # Only add once in add mode
                else:
                    # REPLACE mode: replace this processing
                    dataset.replace_features(
                        source_processings=[proc_name],
                        features=[concatenated],
                        processings=[output_name],
                        source=sd_idx
                    )
                    replaced_processings.append(proc_name)
                    new_processing_names_for_source.append(output_name)

        # Update context with new processing names
        new_processing = []
        for sd_idx in range(n_sources):
            src_processing = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processing)
        context = context.with_processing(new_processing)

        return context, all_artifacts

    def _parse_config(self, config: Any) -> Dict[str, Any]:
        """
        Parse concat_transform configuration.

        Supports:
        - List format: [PCA(50), SVD(50)]
        - Dict format: {"operations": [...], "name": "...", "source_processing": "..."}

        Args:
            config: The concat_transform configuration

        Returns:
            Normalized config dict with 'operations', 'name', 'source_processing' keys

        Raises:
            ValueError: If config format is invalid or generator syntax is not expanded
        """
        if isinstance(config, list):
            # Handle generator pick output:
            # - pick=1 with single transformer: [PCA(50)] -> one operation: PCA(50)
            # - pick=1 with chain: [[Wavelet, PCA]] -> one operation: chain [Wavelet, PCA]
            # - pick=2: [[op1], [op2]] or [[chain1], [chain2]] -> two operations
            #
            # When picking a single item (len=1), the item itself is the operation.
            # If that item is a list, it's a chain (sequential transformers).
            # We should NOT unwrap chains - they should remain as one operation.
            #
            # Only unwrap when we have multiple picked items that each need
            # to be treated as separate operations to concatenate.
            if len(config) == 1 and isinstance(config[0], list):
                # Single selection that is a list - this is ONE chain operation
                # Keep as-is, _deserialize_operations will handle it
                pass  # Don't unwrap
            operations = self._deserialize_operations(config)
            return {"operations": operations, "name": None, "source_processing": None}
        elif isinstance(config, dict):
            if "operations" in config:
                operations = self._deserialize_operations(config["operations"])
                return {
                    "operations": operations,
                    "name": config.get("name"),
                    "source_processing": config.get("source_processing")
                }
            elif "_or_" in config:
                # Generator syntax - should be expanded by generator before reaching here
                raise ValueError(
                    "Generator syntax not expanded - should be handled by pipeline generator"
                )
            else:
                # Maybe a dict with transformer objects as values?
                operations = self._deserialize_operations(list(config.values()))
                return {"operations": operations, "name": None, "source_processing": None}
        else:
            raise ValueError(f"Invalid concat_transform config: {type(config)}")

    def _deserialize_operations(self, operations: List[Any]) -> List[Any]:
        """
        Deserialize operations that may be serialized as dicts or strings.

        Args:
            operations: List of operations (may be serialized or instances)

        Returns:
            List of deserialized transformer instances or nested concat_transform dicts
        """
        deserialized = []
        for op in operations:
            if isinstance(op, dict) and "concat_transform" in op:
                # Nested concat_transform - keep as dict, will be handled during execution
                deserialized.append(op)
            elif isinstance(op, list):
                # Chain of transformers
                chain = [self._deserialize_single_operation(item) for item in op]
                deserialized.append(chain)
            else:
                deserialized.append(self._deserialize_single_operation(op))
        return deserialized

    def _deserialize_single_operation(self, op: Any) -> Any:
        """
        Deserialize a single operation if needed.

        Args:
            op: Single operation (may be serialized or instance)

        Returns:
            Deserialized transformer instance or chain of instances
        """
        # Already a transformer instance
        if hasattr(op, 'fit') and hasattr(op, 'transform'):
            return op

        # Nested concat_transform dict - preserve as-is
        if isinstance(op, dict) and "concat_transform" in op:
            return op

        # Serialized as dict or string
        if isinstance(op, (dict, str)):
            return deserialize_component(op)

        # Handle lists (chains of transformers) - recursively deserialize
        if isinstance(op, list):
            return [self._deserialize_single_operation(item) for item in op]

        return op

    def _get_operation_name(self, operation: Any, index: int) -> str:
        """
        Get a name for an operation (single transformer, chain, or nested concat).

        Args:
            operation: Single transformer, list of transformers (chain),
                       or dict with concat_transform key
            index: Operation index in the operations list

        Returns:
            String name for the operation
        """
        if isinstance(operation, dict) and "concat_transform" in operation:
            # Nested concat_transform
            return f"nested_concat_{index}"
        elif isinstance(operation, list):
            # Chain: use last transformer's class name
            if operation:
                last_name = operation[-1].__class__.__name__
                return f"chain_{last_name}_{index}"
            return f"chain_empty_{index}"
        else:
            return f"{operation.__class__.__name__}_{index}"

    def _generate_output_name(
        self,
        proc_name: str,
        operations: List[Any],
        output_name_base: Optional[str],
        is_add_mode: bool
    ) -> str:
        """
        Generate the output processing name.

        Args:
            proc_name: Source processing name
            operations: List of operations being concatenated
            output_name_base: Optional custom name override
            is_add_mode: Whether we're in add mode (feature_augmentation)

        Returns:
            Generated processing name
        """
        if output_name_base:
            if is_add_mode:
                return output_name_base
            else:
                return f"{proc_name}_{output_name_base}"

        # Generate name from operation names
        op_names = []
        for op in operations:
            if op is None:
                op_names.append("raw")
            elif isinstance(op, list) and op:
                # Chain: use last transformer name
                op_names.append(op[-1].__class__.__name__)
            elif hasattr(op, '__class__'):
                op_names.append(op.__class__.__name__)

        # Truncate if too many operations
        if len(op_names) > 3:
            suffix = "concat_" + "_".join(op_names[:3]) + f"_+{len(op_names) - 3}"
        else:
            suffix = "concat_" + "_".join(op_names) if op_names else "concat"

        if is_add_mode:
            return suffix
        else:
            return f"{proc_name}_{suffix}"

    def _execute_single(
        self,
        transformer: TransformerMixin,
        train_data: np.ndarray,
        all_data: np.ndarray,
        binary_key: str,
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        runtime_context: 'RuntimeContext',
        context: Optional['ExecutionContext'] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Execute a single transformer.

        Args:
            transformer: sklearn-compatible transformer
            train_data: 2D training data for fitting
            all_data: 2D data to transform
            binary_key: Key for saving/loading the fitted transformer
            mode: Execution mode
            loaded_binaries: Pre-loaded binaries for predict mode
            runtime_context: Runtime infrastructure
            context: Optional execution context for branch info

        Returns:
            Tuple of (transformed_data, artifact_metadata)
        """
        if mode in ["predict", "explain"]:
            fitted = None

            # V3: Use artifact_provider for chain-based loading
            if runtime_context.artifact_provider is not None:
                step_index = runtime_context.step_number
                step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                    step_index,
                    branch_path=context.selector.branch_path if context else None
                )
                # Find artifact by key matching or by index
                for artifact_id, obj in step_artifacts:
                    if binary_key in artifact_id:
                        fitted = obj
                        break

            if fitted is None:
                raise ValueError(f"Transformer '{binary_key}' not found at step {runtime_context.step_number}")
        else:
            # Fit new transformer
            fitted = clone(transformer)
            fitted.fit(train_data)

        transformed = fitted.transform(all_data)

        artifact = None
        if mode == "train" and runtime_context.saver is not None:
            branch_id = context.selector.branch_id if context else None
            branch_name = context.selector.branch_name if context else None
            artifact = runtime_context.saver.persist_artifact(
                step_number=runtime_context.step_number,
                name=binary_key,
                obj=fitted,
                format_hint='sklearn',
                branch_id=branch_id,
                branch_name=branch_name
            )

        return transformed, artifact

    def _execute_chain(
        self,
        chain: List[TransformerMixin],
        train_data: np.ndarray,
        all_data: np.ndarray,
        binary_key_base: str,
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        runtime_context: 'RuntimeContext',
        context: Optional['ExecutionContext'] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Execute a chain of transformers sequentially: [A, B, C] → C(B(A(X))).

        Args:
            chain: List of sklearn-compatible transformers
            train_data: 2D training data for fitting
            all_data: 2D data to transform
            binary_key_base: Base key for saving/loading transformers
            mode: Execution mode
            loaded_binaries: Pre-loaded binaries for predict mode
            runtime_context: Runtime infrastructure
            context: Optional execution context for branch info

        Returns:
            Tuple of (final_transformed_data, list_of_artifact_metadata)
        """
        artifacts = []
        current_train = train_data
        current_all = all_data

        for i, transformer in enumerate(chain):
            binary_key = f"{binary_key_base}_chain{i}"

            if mode in ["predict", "explain"]:
                fitted = None

                # V3: Use artifact_provider for chain-based loading
                if runtime_context.artifact_provider is not None:
                    step_index = runtime_context.step_number
                    step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                        step_index,
                        branch_path=context.selector.branch_path if context else None
                    )
                    # Find artifact by key matching
                    for artifact_id, obj in step_artifacts:
                        if binary_key in artifact_id:
                            fitted = obj
                            break

                if fitted is None:
                    raise ValueError(f"Transformer '{binary_key}' not found at step {runtime_context.step_number}")
            else:
                fitted = clone(transformer)
                fitted.fit(current_train)

            current_train = fitted.transform(current_train)
            current_all = fitted.transform(current_all)

            if mode == "train" and runtime_context.saver is not None:
                branch_id = context.selector.branch_id if context else None
                branch_name = context.selector.branch_name if context else None
                artifact = runtime_context.saver.persist_artifact(
                    step_number=runtime_context.step_number,
                    name=binary_key,
                    obj=fitted,
                    format_hint='sklearn',
                    branch_id=branch_id,
                    branch_name=branch_name
                )
                artifacts.append(artifact)

        return current_all, artifacts

    def _execute_nested_concat(
        self,
        nested_config: Any,
        train_data: np.ndarray,
        all_data: np.ndarray,
        binary_key_base: str,
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        runtime_context: 'RuntimeContext',
        context: Optional['ExecutionContext'] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Execute a nested concat_transform: concatenate multiple pipelines.

        This allows nesting concat_transform within concat_transform, enabling
        complex preprocessing structures like:
        concat_transform: [
            PCA(50),
            {"concat_transform": [[SNV(), PCA(50)], [SG(), PCA(50)]]}
        ]

        Args:
            nested_config: The nested concat_transform configuration (list of operations)
            train_data: 2D training data for fitting
            all_data: 2D data to transform
            binary_key_base: Base key for saving/loading transformers
            mode: Execution mode
            loaded_binaries: Pre-loaded binaries for predict mode
            runtime_context: Runtime infrastructure
            context: Optional execution context for branch info

        Returns:
            Tuple of (concatenated_data, list_of_artifact_metadata)
        """
        # Parse the nested config to get operations
        config = self._parse_config(nested_config)
        operations = config["operations"]

        if not operations:
            return all_data, []

        artifacts = []
        concat_blocks = []

        for op_idx, operation in enumerate(operations):
            # Handle None operations - pass through unchanged
            if operation is None:
                concat_blocks.append(all_data)
                continue

            op_name_base = self._get_operation_name(operation, op_idx)
            binary_key = f"{binary_key_base}_nested{op_idx}_{op_name_base}"

            # Recursively handle nested concat_transform
            if isinstance(operation, dict) and "concat_transform" in operation:
                transformed, nested_artifacts = self._execute_nested_concat(
                    operation["concat_transform"], train_data, all_data, binary_key,
                    mode, loaded_binaries, runtime_context, context
                )
                artifacts.extend(nested_artifacts)
            # Handle chain vs single transformer
            elif isinstance(operation, list):
                # Chain: [A, B, C] → C(B(A(X)))
                transformed, chain_artifacts = self._execute_chain(
                    operation, train_data, all_data, binary_key,
                    mode, loaded_binaries, runtime_context, context
                )
                artifacts.extend(chain_artifacts)
            else:
                # Single transformer
                transformed, artifact = self._execute_single(
                    operation, train_data, all_data, binary_key,
                    mode, loaded_binaries, runtime_context, context
                )
                if artifact:
                    artifacts.append(artifact)

            concat_blocks.append(transformed)

        # Concatenate all blocks horizontally
        concatenated = np.hstack(concat_blocks)
        return concatenated, artifacts
