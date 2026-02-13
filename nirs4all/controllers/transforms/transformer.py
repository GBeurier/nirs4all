from typing import Any, List, Tuple, Optional, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.exceptions import NAError
from nirs4all.operators.base import SpectraTransformerMixin
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.storage.artifacts.types import ArtifactType

if TYPE_CHECKING:
    from nirs4all.spectra.spectra_dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep

import warnings
import numpy as np
from sklearn.base import clone

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


@register_controller
class TransformerMixinController(OperatorController):
    priority = 10

    @staticmethod
    def _needs_wavelengths(operator: Any) -> bool:
        """Check if the operator needs wavelengths passed.

        Args:
            operator: The operator to check.

        Returns:
            True if the operator is a SpectraTransformerMixin with
            _requires_wavelengths set to True or "optional".
        """
        return (
            isinstance(operator, SpectraTransformerMixin) and
            getattr(operator, '_requires_wavelengths', False)
        )

    @staticmethod
    def _requires_y(operator: Any) -> bool:
        """Check if the operator requires y during fit (supervised transform).

        Args:
            operator: The operator to check.

        Returns:
            True if the operator declares requires_y=True in its _more_tags().
        """
        # Check _more_tags() method (sklearn convention)
        if hasattr(operator, '_more_tags'):
            tags = operator._more_tags()
            if tags.get('requires_y', False):
                return True

        # Fallback: check _tags attribute directly
        if hasattr(operator, '_tags'):
            tags = operator._tags
            if isinstance(tags, dict) and tags.get('requires_y', False):
                return True

        return False

    @staticmethod
    def _extract_wavelengths(
        dataset: 'SpectroDataset',
        source_index: int,
        operator_name: str
    ) -> np.ndarray:
        """Extract wavelengths from dataset for a given source.

        Attempts to get wavelengths using dataset.wavelengths_nm(). If that fails,
        falls back to dataset.float_headers() as a legacy fallback.

        Args:
            dataset: The SpectroDataset to extract wavelengths from.
            source_index: The source index for multi-source datasets.
            operator_name: Name of the operator (for error messages).

        Returns:
            Wavelength array in nm.

        Raises:
            ValueError: If wavelengths cannot be extracted from the dataset.
        """
        try:
            wavelengths = dataset.wavelengths_nm(source_index)
            return wavelengths
        except (ValueError, AttributeError):
            pass

        # Fall back to inferring from headers
        try:
            wavelengths = dataset.float_headers(source_index)
            if wavelengths is not None and len(wavelengths) > 0:
                return wavelengths
        except (ValueError, AttributeError):
            pass

        raise ValueError(
            f"Operator {operator_name} requires wavelengths but dataset has no "
            f"wavelength information for source {source_index}. Ensure the dataset "
            f"has wavelength headers (nm or cm⁻¹)."
        )

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

    @classmethod
    def supports_step_cache(cls) -> bool:
        """Preprocessing transforms benefit from cross-variant step caching."""
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

        # Extract step-level options from step configuration
        fit_on_all = False
        na_policy = None
        fill_value = 0
        if isinstance(step_info.original_step, dict):
            fit_on_all = step_info.original_step.get("fit_on_all", False)
            na_policy = step_info.original_step.get("na_policy")
            fill_value = step_info.original_step.get("fill_value", 0)

        # Check if we're in sample augmentation mode
        if context.metadata.augment_sample and mode not in ["predict", "explain"]:
            return self._execute_for_sample_augmentation(
                op, dataset, context, runtime_context, mode, loaded_binaries, prediction_store,
                fit_on_all=fit_on_all
            )

        # Normal or feature augmentation execution (existing code)
        operator_name = op.__class__.__name__

        # Generate operator name with parameters for preprocessing chain display
        from nirs4all.utils.operator_formatting import format_operator_with_params
        operator_name_with_params = format_operator_with_params(op)

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

        # Inflight memory check: warn if transient peak (all_data + fit_data) is large
        inflight_bytes = sum(a.nbytes for a in all_data) + sum(a.nbytes for a in fit_data)
        if inflight_bytes > 1024 ** 3:  # > 1 GB
            from nirs4all.utils.memory import format_bytes
            logger.warning(
                f"Transformer {operator_name}: inflight data {format_bytes(inflight_bytes)} "
                f"(all_data + fit_data). Consider reducing dataset size or processings."
            )

        fitted_transformers = []
        transformed_features_list = []
        new_processing_names = []
        processing_names = []

        # Note: We use runtime_context.next_processing_index() to track processing counter
        # across all sources for unique artifact IDs. This ensures each (source, processing)
        # pair gets a unique substep_index even across feature_augmentation sub-operations.

        # Check if operator needs wavelengths (once, outside source loop)
        needs_wavelengths = self._needs_wavelengths(op)

        # Check if operator requires y (supervised transform like OSC, EPO)
        requires_y = self._requires_y(op)
        y_fit = None
        if requires_y:
            # Get target values for fitting
            if fit_on_all:
                y_fit = dataset.y(context.selector, include_excluded=False)
            else:
                train_context = context.with_partition("train")
                y_fit = dataset.y(train_context.selector, include_excluded=False)
            # Handle multi-target datasets: use first target column
            if y_fit.ndim > 1:
                y_fit = y_fit[:, 0]
            y_fit = y_fit.ravel()

        # Loop through each data source
        for sd_idx, (fit_x, all_x) in enumerate(zip(fit_data, all_data)):
            # print(f"Processing source {sd_idx}: fit shape {fit_x.shape}, all shape {all_x.shape}")

            # Extract wavelengths for this source if needed
            wavelengths = None
            if needs_wavelengths:
                wavelengths = self._extract_wavelengths(dataset, sd_idx, operator_name)

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

                # --- Pre-transform NA guard ---
                had_nan_before = False
                if dataset._may_contain_nan and np.any(np.isnan(all_2d)):
                    had_nan_before = True
                    allow_nan = getattr(op, '_tags', {}).get('allow_nan', False)
                    if allow_nan:
                        pass  # Operator natively handles NaN
                    elif na_policy == "replace":
                        all_2d = np.where(np.isnan(all_2d), fill_value, all_2d)
                        fit_2d = np.where(np.isnan(fit_2d), fill_value, fit_2d)
                    elif na_policy == "ignore":
                        pass  # NaN samples will be handled below after transform
                    else:
                        raise NAError(
                            f"Transform '{operator_name}' received NaN input. "
                            f"Set na_policy on this step or handle NAs upstream."
                        )

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

                    # --- Check-before-fit: reuse cached fitted transformer ---
                    cached_transformer = self._try_cache_lookup(
                        runtime_context=runtime_context,
                        context=context,
                        dataset=dataset,
                        operator_name=operator_name,
                        source_index=sd_idx,
                        operator=op,
                    )
                    if cached_transformer is not None:
                        transformer = cached_transformer
                    else:
                        transformer = clone(op)
                        if needs_wavelengths and requires_y:
                            transformer.fit(fit_2d, y_fit, wavelengths=wavelengths)
                        elif needs_wavelengths:
                            transformer.fit(fit_2d, wavelengths=wavelengths)
                        elif requires_y:
                            transformer.fit(fit_2d, y_fit)
                        else:
                            transformer.fit(fit_2d)

                if needs_wavelengths:
                    transformed_2d = transformer.transform(all_2d, wavelengths=wavelengths)
                else:
                    transformed_2d = transformer.transform(all_2d)

                # --- Post-transform NaN detection ---
                if not had_nan_before and np.any(np.isnan(transformed_2d)):
                    n_new_nan = int(np.isnan(transformed_2d).sum())
                    m_samples = int(np.isnan(transformed_2d).any(axis=1).sum())
                    warnings.warn(
                        f"Transform '{operator_name}' introduced "
                        f"{n_new_nan} NaN values in {m_samples} samples.",
                        UserWarning,
                    )
                    dataset._may_contain_nan = True

                # Store results
                source_transformed_features.append(transformed_2d)
                # Use operator name with parameters for better readability
                # Extract the operator counter from new_operator_name (e.g., "SNV_3" -> "3")
                op_counter = new_operator_name.split('_')[-1]
                new_processing_name = f"{processing_name}_{operator_name_with_params}_{op_counter}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)

                # Persist fitted transformer using artifact registry
                if mode == "train":
                    if cached_transformer is not None:
                        input_data_hash = None
                    elif self._is_stateless(op):
                        input_data_hash = self._compute_operator_params_hash(op)
                    else:
                        input_data_hash = dataset.content_hash()
                    artifact = self._persist_transformer(
                        runtime_context=runtime_context,
                        transformer=transformer,
                        name=new_operator_name,
                        context=context,
                        source_index=sd_idx,
                        processing_index=runtime_context.next_processing_index(),
                        input_data_hash=input_data_hash,
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

        # Check if operator needs wavelengths
        needs_wavelengths = self._needs_wavelengths(operator)
        wavelengths_cache = {}  # Cache wavelengths per source

        # Check if operator requires y (supervised transform)
        requires_y = self._requires_y(operator)

        # Get data for fitting (if not in predict/explain mode) - once for all samples
        fit_data = None
        y_fit = None
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

            # Get target values if required
            if requires_y:
                y_fit = dataset.y(fit_selector, include_excluded=False)
                # Handle multi-target datasets: use first target column
                if y_fit.ndim > 1:
                    y_fit = y_fit[:, 0]
                y_fit = y_fit.ravel()

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
                # Extract wavelengths for this source if needed
                wavelengths = None
                if needs_wavelengths:
                    if source_idx not in wavelengths_cache:
                        wavelengths_cache[source_idx] = self._extract_wavelengths(
                            dataset, source_idx, operator_name
                        )
                    wavelengths = wavelengths_cache[source_idx]

                for proc_idx in range(n_processings):
                    cache_key = (source_idx, proc_idx)
                    transformer = clone(operator)
                    fit_proc_data = fit_data[source_idx][:, proc_idx, :]
                    if needs_wavelengths and requires_y:
                        transformer.fit(fit_proc_data, y_fit, wavelengths=wavelengths)
                    elif needs_wavelengths:
                        transformer.fit(fit_proc_data, wavelengths=wavelengths)
                    elif requires_y:
                        transformer.fit(fit_proc_data, y_fit)
                    else:
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

            # Extract wavelengths for this source if needed (for predict/explain mode)
            wavelengths = None
            if needs_wavelengths:
                if source_idx not in wavelengths_cache:
                    wavelengths_cache[source_idx] = self._extract_wavelengths(
                        dataset, source_idx, operator_name
                    )
                wavelengths = wavelengths_cache[source_idx]

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
                if needs_wavelengths:
                    transformed_data = transformer.transform(proc_data, wavelengths=wavelengths)
                else:
                    transformed_data = transformer.transform(proc_data)
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
        wavelengths_cache = {}  # Cache wavelengths per source

        # Check if operator needs wavelengths
        needs_wavelengths = self._needs_wavelengths(operator)

        # Check if operator requires y (supervised transform)
        requires_y = self._requires_y(operator)

        # Get data for fitting (if not in predict/explain mode)
        fit_data = None
        y_fit = None
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

            # Get target values if required
            if requires_y:
                y_fit = dataset.y(fit_selector, include_excluded=False)
                # Handle multi-target datasets: use first target column
                if y_fit.ndim > 1:
                    y_fit = y_fit[:, 0]
                y_fit = y_fit.ravel()

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

                # Extract wavelengths for this source if needed
                wavelengths = None
                if needs_wavelengths:
                    if source_idx not in wavelengths_cache:
                        wavelengths_cache[source_idx] = self._extract_wavelengths(
                            dataset, source_idx, operator_name
                        )
                    wavelengths = wavelengths_cache[source_idx]

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
                            if needs_wavelengths and requires_y:
                                transformer.fit(fit_proc_data, y_fit, wavelengths=wavelengths)
                            elif needs_wavelengths:
                                transformer.fit(fit_proc_data, wavelengths=wavelengths)
                            elif requires_y:
                                transformer.fit(fit_proc_data, y_fit)
                            else:
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

                    if needs_wavelengths:
                        transformed_data = transformer.transform(proc_data, wavelengths=wavelengths)
                    else:
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

    @staticmethod
    def _is_stateless(operator: Any) -> bool:
        """Check if the operator is stateless (fit produces no data-dependent state).

        An operator is stateless when its output depends only on the input data
        and fixed constructor parameters, not on any state learned during fit().
        For such operators, the cache can skip the data hash entirely and use
        only the chain path hash + operator params hash.

        Args:
            operator: The operator to check.

        Returns:
            True if the operator declares ``_stateless = True``.
        """
        return getattr(operator, '_stateless', False) is True

    @staticmethod
    def _compute_operator_params_hash(operator: Any) -> str:
        """Compute a deterministic hash of an operator's constructor parameters.

        Uses sklearn's ``get_params(deep=False)`` to obtain the parameter
        dict, then hashes a sorted canonical string representation.

        Args:
            operator: A sklearn-compatible estimator.

        Returns:
            Hex string hash of the operator parameters.
        """
        import hashlib
        params = {}
        if hasattr(operator, 'get_params'):
            params = operator.get_params(deep=False)
        # Build a canonical sorted string of (key, repr(value)) pairs
        canonical = ";".join(f"{k}={repr(v)}" for k, v in sorted(params.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _try_cache_lookup(
        self,
        runtime_context: 'RuntimeContext',
        context: ExecutionContext,
        dataset: 'SpectroDataset',
        operator_name: str,
        source_index: int,
        operator: Any = None,
    ) -> Optional[Any]:
        """Check the artifact registry for a previously fitted transformer with the same chain and data.

        Builds the chain path that ``_persist_transformer`` would produce
        and queries the registry's (chain_path, data_hash) index.  If a
        matching artifact is found, the fitted transformer is loaded and
        returned.

        For stateless operators (``_stateless = True``), the lookup uses
        ``(chain_path_hash, operator_params_hash)`` instead of
        ``(chain_path_hash, data_hash)`` since the fitted state is always
        identical regardless of training data.

        Args:
            runtime_context: Runtime context with artifact registry.
            context: Execution context with branch information.
            dataset: Current dataset (used to compute content hash).
            operator_name: Class name of the operator.
            source_index: Source index for multi-source transformers.
            operator: The operator instance (used for stateless param hashing).

        Returns:
            The loaded fitted transformer if a cache hit occurs, or ``None``.
        """
        registry = runtime_context.artifact_registry
        if registry is None:
            return None

        # Build the chain path that _persist_transformer would create
        # (mirrors the logic there, without actually registering)
        step_index = runtime_context.step_number
        branch_path = context.selector.branch_path or []

        # Peek at the processing counter without incrementing
        substep_index = runtime_context.processing_counter

        from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorNode, OperatorChain

        if runtime_context.trace_recorder is not None:
            current_chain = runtime_context.trace_recorder.current_chain()
        else:
            pipeline_id = runtime_context.pipeline_name or "unknown"
            current_chain = OperatorChain(pipeline_id=pipeline_id)

        transformer_node = OperatorNode(
            step_index=step_index,
            operator_class=operator_name,
            branch_path=branch_path,
            source_index=source_index,
            fold_id=None,
            substep_index=substep_index,
        )

        artifact_chain = current_chain.append(transformer_node)
        chain_path = artifact_chain.to_path()

        # For stateless operators, use params hash instead of data hash
        stateless = operator is not None and self._is_stateless(operator)
        if stateless:
            lookup_hash = self._compute_operator_params_hash(operator)
        else:
            lookup_hash = dataset.content_hash()

        # Lookup
        record = registry.get_by_chain_and_data(chain_path, lookup_hash)
        if record is None:
            return None

        # Verify the record is for the same operator class
        if record.class_name != operator_name:
            return None

        # Load the fitted transformer
        try:
            transformer = registry.load_artifact(record)
        except Exception as exc:
            logger.debug("Cache hit but failed to load artifact %s: %s", record.artifact_id, exc)
            return None

        cache_type = "stateless params" if stateless else "data hash"
        logger.info(
            "Cache hit: reusing fitted %s from chain %s (%s match)",
            operator_name,
            chain_path,
            cache_type,
        )
        return transformer

    def _persist_transformer(
        self,
        runtime_context: 'RuntimeContext',
        transformer: Any,
        name: str,
        context: ExecutionContext,
        source_index: Optional[int] = None,
        processing_index: Optional[int] = None,
        input_data_hash: Optional[str] = None,
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
            input_data_hash: Optional hash of the input data for cache-key lookups.
                When provided, enables check-before-fit cache hits on subsequent
                pipelines that share the same preprocessing prefix.

        Returns:
            ArtifactRecord with V3 chain-based metadata.
        """
        # Use artifact registry (V3 system)
        if runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry
            pipeline_id = runtime_context.pipeline_name or "unknown"
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

            # Populate cache-key index for check-before-fit lookups
            if input_data_hash:
                from nirs4all.pipeline.storage.artifacts.operator_chain import compute_chain_hash
                chain_path_hash = compute_chain_hash(chain_path)
                registry._by_chain_and_data[(chain_path_hash, input_data_hash)] = artifact_id

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
