from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING, Union

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.storage.artifacts.types import ArtifactType

if TYPE_CHECKING:
    from nirs4all.spectra.spectra_dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep

import numpy as np
from sklearn.base import clone
import pickle
## TODO add parrallel support for multi-source datasets and multi-processing datasets


@register_controller
class TransformerMixinController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match TransformerMixin objects."""
        # Get the actual model object
        model_obj = None
        if isinstance(step, dict) and 'model' in step:
            model_obj = step['model']
        elif operator is not None:
            model_obj = operator
        else:
            model_obj = step

        # Check if it's a TransformerMixin
        return (isinstance(model_obj, TransformerMixin) or
                (hasattr(model_obj, '__class__') and issubclass(model_obj.__class__, TransformerMixin)))

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """TransformerMixin controllers support prediction mode."""
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ):
        """Execute transformer - handles normal, feature augmentation, and sample augmentation modes.

        Supports optional `fit_on_all` parameter in step configuration to fit the transformer
        on all data instead of just training data. This is useful for unsupervised preprocessing
        where you want the transformation to capture the full data distribution.

        Step format:
            # Standard (fit on train, transform all):
            StandardScaler()

            # Fit on ALL data (unsupervised preprocessing):
            {"preprocessing": StandardScaler(), "fit_on_all": True}
        """
        op = step_info.operator

        # Extract fit_on_all option from step configuration
        fit_on_all = False
        if isinstance(step_info.original_step, dict):
            fit_on_all = step_info.original_step.get("fit_on_all", False)

        # Check if we're in sample augmentation mode
        if context.metadata.augment_sample and mode not in ["predict", "explain"]:
            return self._execute_for_sample_augmentation(
                op, dataset, context, runtime_context, mode, loaded_binaries, prediction_store,
                fit_on_all=fit_on_all
            )

        # Normal or feature augmentation execution (existing code)
        operator_name = op.__class__.__name__

        # Get all data (always needed for transform)
        # IMPORTANT: Include excluded samples to maintain consistent array shapes
        # when replacing features. Excluded samples are filtered at query time, not transform time.
        all_data = dataset.x(context.selector, "3d", concat_source=False, include_excluded=True)

        # Get fitting data based on fit_on_all option
        # Note: Fitting should EXCLUDE filtered samples to prevent outlier influence
        if fit_on_all:
            # Fit on all data (unsupervised preprocessing) but exclude filtered samples
            fit_data = dataset.x(context.selector, "3d", concat_source=False, include_excluded=False)
        else:
            # Standard: fit on train data only (excluding filtered samples)
            train_context = context.with_partition("train")
            fit_data = dataset.x(train_context.selector, "3d", concat_source=False, include_excluded=False)

        # Ensure data is in list format
        if not isinstance(fit_data, list):
            fit_data = [fit_data]
        if not isinstance(all_data, list):
            all_data = [all_data]

        fitted_transformers = []
        transformed_features_list = []
        new_processing_names = []
        processing_names = []

        # Note: We use runtime_context.next_processing_index() to track processing counter
        # across all sources for unique artifact IDs. This ensures each (source, processing)
        # pair gets a unique substep_index even across feature_augmentation sub-operations.

        # Loop through each data source
        for sd_idx, (fit_x, all_x) in enumerate(zip(fit_data, all_data)):
            # print(f"Processing source {sd_idx}: fit shape {fit_x.shape}, all shape {all_x.shape}")

            # Get processing names for this source
            processing_ids = dataset.features_processings(sd_idx)
            source_processings = processing_ids
            # print("🔹 Processing source", sd_idx, "with processings:", source_processings)
            if context.selector.processing:
                # Handle case where processing list has fewer entries than sources
                # (e.g., after source merge, only source 0 has processings)
                if sd_idx < len(context.selector.processing):
                    source_processings = context.selector.processing[sd_idx]
                else:
                    # Skip this source - it was merged into source 0
                    continue

            source_transformed_features = []
            source_new_processing_names = []
            source_processing_names = []

            # Loop through each processing in the 3D data (samples, processings, features)
            for processing_idx in range(fit_x.shape[1]):
                processing_name = processing_ids[processing_idx]
                # print(f" Processing {processing_name} (idx {processing_idx})")
                # print(processing_name, processing_name in source_processings)
                if processing_name not in source_processings:
                    continue
                fit_2d = fit_x[:, processing_idx, :]      # Data for fitting
                all_2d = all_x[:, processing_idx, :]      # All data to transform

                # print(f" Processing {processing_name} (idx {processing_idx}): fit {fit_2d.shape}, all {all_2d.shape}")

                if mode == "predict" or mode == "explain":
                    transformer = None
                    loaded_artifact_name = None

                    # V3: Use artifact_provider for chain-based loading
                    if runtime_context.artifact_provider is not None:
                        step_index = runtime_context.step_number
                        # Load all artifacts for this source, then pick by global index
                        # The global index persists across feature_augmentation sub-operations
                        step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                            step_index,
                            branch_path=context.selector.branch_path,
                            source_index=sd_idx,
                            substep_index=None  # Load all artifacts for this source
                        )
                        if step_artifacts:
                            artifacts_list = list(step_artifacts)
                            # Use global artifact load index for this source to handle
                            # feature_augmentation sub-operations correctly
                            artifact_idx = runtime_context.next_artifact_load_index(sd_idx)
                            if artifact_idx < len(artifacts_list):
                                loaded_artifact_name, transformer = artifacts_list[artifact_idx]

                    # Use the artifact name from what was actually loaded, not next_op()
                    if loaded_artifact_name:
                        new_operator_name = loaded_artifact_name
                    else:
                        # Fallback: generate name for error message
                        new_operator_name = f"{operator_name}_{runtime_context.next_op()}"

                    if transformer is None:
                        available = []
                        if runtime_context.artifact_provider is not None:
                            step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                                runtime_context.step_number,
                                branch_path=context.selector.branch_path
                            )
                            available = [name for name, _ in step_artifacts] if step_artifacts else []
                        raise ValueError(
                            f"Transformer for {operator_name} not found at step {runtime_context.step_number} "
                            f"(branch_path={context.selector.branch_path}, source={sd_idx}, "
                            f"artifact_idx={artifact_idx if 'artifact_idx' in dir() else 'N/A'}). "
                            f"Available artifacts: {available}"
                        )
                else:
                    new_operator_name = f"{operator_name}_{runtime_context.next_op()}"
                    transformer = clone(op)
                    transformer.fit(fit_2d)

                transformed_2d = transformer.transform(all_2d)

                # print("  Transformed shape:", transformed_2d.shape)

                # Store results
                source_transformed_features.append(transformed_2d)
                new_processing_name = f"{processing_name}_{new_operator_name}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)

                # Persist fitted transformer using artifact registry
                if mode == "train":
                    artifact = self._persist_transformer(
                        runtime_context=runtime_context,
                        transformer=transformer,
                        name=new_operator_name,
                        context=context,
                        source_index=sd_idx,
                        processing_index=runtime_context.next_processing_index()
                    )
                    fitted_transformers.append(artifact)

            # print("🔹 Finished processing source", sd_idx, len(fitted_transformers))
            # ("🔹 New processing names:", source_new_processing_names)
            transformed_features_list.append(source_transformed_features)
            new_processing_names.append(source_new_processing_names)
            processing_names.append(source_processing_names)

        for sd_idx, (source_features, src_new_processing_names) in enumerate(zip(transformed_features_list, new_processing_names)):
            if context.metadata.add_feature:
                dataset.add_features(source_features, src_new_processing_names, source=sd_idx)
                # Update processing in context (requires creating new list)
                new_processing = list(context.selector.processing)
                new_processing[sd_idx] = src_new_processing_names
                context = context.with_processing(new_processing)
            else:
                dataset.replace_features(
                    source_processings=processing_names[sd_idx],
                    features=source_features,
                    processings=src_new_processing_names,
                    source=sd_idx
                )
                # Update processing in context (requires creating new list)
                new_processing = list(context.selector.processing)
                new_processing[sd_idx] = src_new_processing_names
                context = context.with_processing(new_processing)
        context = context.with_metadata(add_feature=False)

        # print(dataset)
        return context, fitted_transformers

    def _execute_for_sample_augmentation(
        self,
        operator: Any,
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        prediction_store: Optional[Any],
        fit_on_all: bool = False
    ) -> Tuple[ExecutionContext, List]:
        """
        Apply transformer to origin samples and add augmented samples.

        Optimized implementation:
        - Batch data fetching: fetches all target samples in one call
        - Single transformer fit: fits transformer once on train/all data, reuses for all samples
        - Batch transform: transforms all samples at once per processing
        - Bulk insert: adds all augmented samples in a loop but with pre-fitted transformer

        Args:
            operator: The transformer operator to apply
            dataset: The dataset to operate on
            context: Execution context
            runtime_context: Runtime context with saver, step info, etc.
            mode: Execution mode ("train", "predict", "explain")
            loaded_binaries: Pre-loaded binaries for predict/explain mode
            prediction_store: Not used
            fit_on_all: If True, fit transformer on all data instead of train only
        """
        target_sample_ids = context.metadata.target_samples
        if not target_sample_ids:
            return context, []

        operator_name = operator.__class__.__name__
        fitted_transformers = []
        n_targets = len(target_sample_ids)

        # Get data for fitting (if not in predict/explain mode) - once for all samples
        fit_data = None
        fitted_transformers_cache = {}  # Cache fitted transformers per source/processing

        if mode not in ["predict", "explain"]:
            if fit_on_all:
                # Fit on all data (unsupervised preprocessing)
                fit_selector = context.selector.with_augmented(False)
            else:
                # Standard: fit on train data only
                train_context = context.with_partition("train")
                fit_selector = train_context.selector.with_augmented(False)
            fit_data = dataset.x(fit_selector, "3d", concat_source=False)
            if not isinstance(fit_data, list):
                fit_data = [fit_data]

        # Batch fetch all target samples at once
        batch_selector = {"sample": list(target_sample_ids)}
        all_origin_data = dataset.x(batch_selector, "3d", concat_source=False, include_augmented=False)

        if not isinstance(all_origin_data, list):
            all_origin_data = [all_origin_data]

        # Determine dimensions - use actual data shape, not target_sample_ids length
        n_sources = len(all_origin_data)
        n_actual_samples = all_origin_data[0].shape[0] if n_sources > 0 else 0
        n_processings = all_origin_data[0].shape[1] if n_sources > 0 else 0

        # Ensure we have the expected number of samples
        if n_actual_samples != n_targets:
            # If mismatch, fallback to original sample-by-sample approach
            # This can happen if some target_sample_ids don't exist or are filtered out
            return self._execute_for_sample_augmentation_sequential(
                operator, dataset, context, runtime_context, mode, loaded_binaries, prediction_store,
                fit_on_all=fit_on_all
            )

        # Pre-fit and cache transformers for each source/processing combination (once!)
        if mode not in ["predict", "explain"] and fit_data:
            for source_idx in range(n_sources):
                for proc_idx in range(n_processings):
                    cache_key = (source_idx, proc_idx)
                    transformer = clone(operator)
                    fit_proc_data = fit_data[source_idx][:, proc_idx, :]
                    transformer.fit(fit_proc_data)
                    fitted_transformers_cache[cache_key] = transformer

                    # Save a single transformer binary per source/processing (not per sample)
                    if mode == "train":
                        artifact = self._persist_transformer(
                            runtime_context=runtime_context,
                            transformer=transformer,
                            name=f"{operator_name}_{source_idx}_{proc_idx}",
                            context=context,
                            source_index=source_idx
                        )
                        fitted_transformers.append(artifact)

        # Batch transform all samples per source/processing
        # all_origin_data[source_idx] shape: (n_samples, n_processings, n_features)
        all_transformed = []  # List[List[ndarray]]: [source][processing] -> (n_samples, n_features)

        for source_idx in range(n_sources):
            source_transformed = []
            source_data = all_origin_data[source_idx]  # (n_samples, n_processings, n_features)

            for proc_idx in range(n_processings):
                proc_data = source_data[:, proc_idx, :]  # (n_samples, n_features)

                if mode in ["predict", "explain"]:
                    transformer = None
                    artifact_key = f"{operator_name}_{source_idx}_{proc_idx}"

                    # V3: Use artifact_provider for chain-based loading
                    if runtime_context.artifact_provider is not None:
                        step_index = runtime_context.step_number
                        step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                            step_index,
                            branch_path=context.selector.branch_path,
                            source_index=source_idx
                        )
                        if step_artifacts:
                            artifacts_dict = dict(step_artifacts)
                            transformer = artifacts_dict.get(artifact_key)
                            # Also try matching by proc_idx position if name doesn't match
                            if transformer is None:
                                artifacts_list = list(step_artifacts)
                                if proc_idx < len(artifacts_list):
                                    _, transformer = artifacts_list[proc_idx]

                    if transformer is None:
                        raise ValueError(f"Transformer for {artifact_key} not found at step {runtime_context.step_number}")
                else:
                    # Use pre-fitted transformer from cache
                    cache_key = (source_idx, proc_idx)
                    transformer = fitted_transformers_cache[cache_key]

                # Batch transform all samples at once
                transformed_data = transformer.transform(proc_data)  # (n_samples, n_features)
                source_transformed.append(transformed_data)

            all_transformed.append(source_transformed)

        # OPTIMIZED: Collect all augmented samples, then batch insert
        # Build 3D arrays for batch insertion: (n_samples, n_processings, n_features)
        if n_sources == 1:
            # Single source: stack transformed data into 3D array
            # all_transformed[0] is list of (n_samples, n_features) arrays, one per processing
            batch_data = np.stack(all_transformed[0], axis=1)  # (n_samples, n_processings, n_features)
        else:
            # Multi-source: create list of 3D arrays
            batch_data = []
            for source_idx in range(n_sources):
                source_3d = np.stack(all_transformed[source_idx], axis=1)
                batch_data.append(source_3d)

        # Build index dictionaries for all samples
        indexes_list = [
            {
                "partition": "train",
                "origin": sample_id,
                "augmentation": operator_name
            }
            for sample_id in target_sample_ids
        ]

        # Single batch insert - O(N) instead of O(N²)
        dataset.add_samples_batch(data=batch_data, indexes_list=indexes_list)

        return context, fitted_transformers

    def _execute_for_sample_augmentation_sequential(
        self,
        operator: Any,
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        prediction_store: Optional[Any],
        fit_on_all: bool = False
    ) -> Tuple[ExecutionContext, List]:
        """
        Fallback sequential implementation for sample augmentation.
        Used when batch processing is not possible due to data shape mismatches.

        Args:
            operator: The transformer operator to apply
            dataset: The dataset to operate on
            context: Execution context
            runtime_context: Runtime context with saver, step info, etc.
            mode: Execution mode ("train", "predict", "explain")
            loaded_binaries: Pre-loaded binaries for predict/explain mode
            prediction_store: Not used
            fit_on_all: If True, fit transformer on all data instead of train only
        """
        target_sample_ids = context.metadata.target_samples
        if not target_sample_ids:
            return context, []

        operator_name = operator.__class__.__name__
        fitted_transformers = []
        fitted_transformers_cache = {}

        # Get data for fitting (if not in predict/explain mode)
        fit_data = None
        if mode not in ["predict", "explain"]:
            if fit_on_all:
                # Fit on all data (unsupervised preprocessing)
                fit_selector = context.selector.with_augmented(False)
            else:
                # Standard: fit on train data only
                train_context = context.with_partition("train")
                fit_selector = train_context.selector.with_augmented(False)
            fit_data = dataset.x(fit_selector, "3d", concat_source=False)
            if not isinstance(fit_data, list):
                fit_data = [fit_data]

        # Process each target sample
        for sample_id in target_sample_ids:
            # Get origin sample data (all sources, base samples only)
            origin_selector = {"sample": [sample_id]}
            origin_data = dataset.x(origin_selector, "3d", concat_source=False, include_augmented=False)

            if not isinstance(origin_data, list):
                origin_data = [origin_data]

            # Transform each source
            transformed_sources = []

            for source_idx, source_data in enumerate(origin_data):
                source_2d_list = []

                for proc_idx in range(source_data.shape[1]):
                    proc_data = source_data[:, proc_idx, :]

                    cache_key = (source_idx, proc_idx)

                    if mode in ["predict", "explain"]:
                        transformer = None
                        artifact_key = f"{operator_name}_{source_idx}_{proc_idx}"

                        # V3: Use artifact_provider for chain-based loading
                        if runtime_context.artifact_provider is not None:
                            step_index = runtime_context.step_number
                            step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                                step_index,
                                branch_path=context.selector.branch_path,
                                source_index=source_idx
                            )
                            if step_artifacts:
                                artifacts_dict = dict(step_artifacts)
                                transformer = artifacts_dict.get(artifact_key)
                                # Also try matching by proc_idx position if name doesn't match
                                if transformer is None:
                                    artifacts_list = list(step_artifacts)
                                    if proc_idx < len(artifacts_list):
                                        _, transformer = artifacts_list[proc_idx]

                        if transformer is None:
                            raise ValueError(f"Transformer for {artifact_key} not found at step {runtime_context.step_number}")
                    elif cache_key in fitted_transformers_cache:
                        # Reuse already fitted transformer
                        transformer = fitted_transformers_cache[cache_key]
                    else:
                        transformer = clone(operator)
                        if fit_data:
                            fit_proc_data = fit_data[source_idx][:, proc_idx, :]
                            transformer.fit(fit_proc_data)
                        fitted_transformers_cache[cache_key] = transformer

                        # Save transformer binary once
                        if mode == "train":
                            artifact = self._persist_transformer(
                                runtime_context=runtime_context,
                                transformer=transformer,
                                name=f"{operator_name}_{source_idx}_{proc_idx}",
                                context=context,
                                source_index=source_idx
                            )
                            fitted_transformers.append(artifact)

                    transformed_data = transformer.transform(proc_data)
                    source_2d_list.append(transformed_data)

                source_3d = np.stack(source_2d_list, axis=1)
                transformed_sources.append(source_3d)

            # Build index dictionary for the augmented sample
            index_dict = {
                "partition": "train",
                "origin": sample_id,
                "augmentation": operator_name
            }

            if len(transformed_sources) == 1:
                data_to_add = transformed_sources[0][0, :, :]
            else:
                data_to_add = [src[0, :, :] for src in transformed_sources]

            dataset.add_samples(data=data_to_add, indexes=index_dict)

        return context, fitted_transformers

    def _persist_transformer(
        self,
        runtime_context: 'RuntimeContext',
        transformer: Any,
        name: str,
        context: ExecutionContext,
        source_index: Optional[int] = None,
        processing_index: Optional[int] = None
    ) -> Any:
        """Persist fitted transformer using V3 chain-based artifact registry.

        Uses artifact_registry.register() with V3 chain-based identification
        for complete execution path tracking, including multi-source support.

        Args:
            runtime_context: Runtime context with saver/registry instances.
            transformer: Fitted transformer to persist.
            name: Operator name for the transformer (e.g., "StandardScaler_3").
            context: Execution context with branch information.
            source_index: Source index for multi-source transformers.
            processing_index: Index of processing within source (for multi-processing steps).

        Returns:
            ArtifactRecord with V3 chain-based metadata.
        """
        # Use artifact registry (V3 system)
        if runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry
            pipeline_id = runtime_context.saver.pipeline_id if runtime_context.saver else "unknown"
            step_index = runtime_context.step_number
            branch_path = context.selector.branch_path or []

            # Use processing_index for substep_index to ensure unique artifact IDs
            # for each processing within a source. This is critical for multi-source
            # pipelines with feature augmentation where multiple transformers are
            # fit per source. Falls back to substep_number for branch contexts.
            if processing_index is not None:
                substep_index = processing_index
            elif runtime_context.substep_number >= 0:
                substep_index = runtime_context.substep_number
            else:
                substep_index = None

            # V3: Build operator chain for this artifact
            from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorNode, OperatorChain

            # Get the current chain from trace recorder or build new one
            if runtime_context.trace_recorder is not None:
                current_chain = runtime_context.trace_recorder.current_chain()
            else:
                current_chain = OperatorChain(pipeline_id=pipeline_id)

            # Create node for this transformer with source_index for multi-source
            transformer_node = OperatorNode(
                step_index=step_index,
                operator_class=transformer.__class__.__name__,
                branch_path=branch_path,
                source_index=source_index,
                fold_id=None,  # Transformers are shared across folds
                substep_index=substep_index,
            )

            # Build chain path for this artifact
            artifact_chain = current_chain.append(transformer_node)
            chain_path = artifact_chain.to_path()

            # Generate V3 artifact ID using chain
            artifact_id = registry.generate_id(chain_path, None, pipeline_id)

            # Register artifact with V3 chain tracking
            record = registry.register(
                obj=transformer,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.TRANSFORMER,
                format_hint='sklearn',
                chain_path=chain_path,
                source_index=source_index,
            )

            # Record artifact in execution trace with V3 chain info
            runtime_context.record_step_artifact(
                artifact_id=artifact_id,
                is_primary=False,  # Transformers are not primary artifacts
                fold_id=None,
                chain_path=chain_path,
                branch_path=branch_path,
                source_index=source_index,
                metadata={"class_name": transformer.__class__.__name__, "name": name}
            )

            return record

        # No registry available - skip persistence (for unit tests)
        # In production, artifact_registry should always be set by the runner
        return None
