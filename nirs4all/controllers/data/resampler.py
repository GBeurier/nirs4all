"""
Controller for wavelength resampling operations.

This controller handles the Resampler operator, extracting wavelengths from
dataset headers and managing the resampling process across multiple sources.
"""

from typing import Any, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.transforms.resampler import Resampler

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class ResamplerController(OperatorController):
    """
    Controller for Resampler operators.

    This controller:
    1. Extracts wavelengths from dataset headers
    2. Validates that headers are convertible to float (wavelengths in cm-1)
    3. Fits the resampler with original wavelengths
    4. Transforms all data to the target wavelength grid
    5. Updates dataset with new features and headers
    6. Supports multi-source datasets with per-source or shared parameters
    """

    priority = 5  # Higher priority than TransformerMixin (10) to match Resampler first

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match Resampler objects."""
        # Get the actual model object
        model_obj = None
        if isinstance(step, dict) and 'model' in step:
            model_obj = step['model']
        elif operator is not None:
            model_obj = operator
        else:
            model_obj = step

        # Check if it's a Resampler
        is_resampler_class = hasattr(model_obj, '__class__') and model_obj.__class__.__name__ == 'Resampler'
        return isinstance(model_obj, Resampler) or is_resampler_class

    @classmethod
    def use_multi_source(cls) -> bool:
        """Resampler supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Resampler supports prediction mode."""
        return True

    def _extract_wavelengths(self, dataset: 'SpectroDataset', source_idx: int) -> np.ndarray:
        """
        Extract and validate wavelengths from dataset headers.

        Args:
            dataset: The spectroscopic dataset
            source_idx: Index of the data source

        Returns:
            Array of wavelengths in cm-1 units

        Raises:
            ValueError: If headers cannot be converted to wavelengths
        """
        # Check the header unit
        header_unit = dataset.header_unit(source_idx)

        # Resampler requires actual wavelength data (not text, indices, or none)
        if header_unit in ["text", "none", "index"]:
            headers = dataset.headers(source_idx)
            raise ValueError(
                f"Cannot resample data with header_unit='{header_unit}' for source {source_idx}. "
                f"Resampler requires numeric wavelength headers (cm-1 or nm). "
                f"Got headers: {headers[:5]}..."
            )

        # Use the dataset's wavelength conversion methods
        try:
            wavelengths = dataset.wavelengths_cm1(source_idx)
            return wavelengths
        except (ValueError, TypeError) as e:
            # Provide helpful error message
            headers = dataset.headers(source_idx)
            raise ValueError(
                f"Failed to extract wavelengths from headers for source {source_idx}. "
                f"Header unit: {header_unit}. Headers: {headers[:5]}... "
                f"Error: {str(e)}"
            ) from e

    def _get_target_wavelengths_for_source(
        self,
        operator: Resampler,
        source_idx: int,
        n_sources: int
    ) -> np.ndarray:
        """
        Get target wavelengths for a specific source.

        If target_wavelengths is a list of arrays, use per-source targets.
        Otherwise, use the same targets for all sources.

        Args:
            operator: The Resampler instance
            source_idx: Current source index
            n_sources: Total number of sources

        Returns:
            Target wavelengths for this source
        """
        target_wl = operator.target_wavelengths

        # Check if it's a list of arrays (per-source targets)
        if isinstance(target_wl, list):
            if len(target_wl) != n_sources:
                raise ValueError(
                    f"If target_wavelengths is a list, it must have {n_sources} elements "
                    f"(one per source), but got {len(target_wl)} elements"
                )
            return np.asarray(target_wl[source_idx])
        else:
            # Same targets for all sources
            return np.asarray(target_wl)

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
    ) -> Tuple['ExecutionContext', List]:
        """
        Execute resampling operation.

        Args:
            step_info: Pipeline step configuration
            dataset: Dataset to operate on
            context: Pipeline execution context
            runtime_context: Runtime context
            source: Data source index (-1 for all sources)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store (unused)

        Returns:
            Tuple of (updated_context, fitted_resamplers)
        """
        op = step_info.operator
        operator_name = op.__class__.__name__

        # Get train and all data as lists of 3D arrays (one per source)
        train_context = context.with_partition("train")

        train_data = dataset.x(train_context.selector, "3d", concat_source=False)
        all_data = dataset.x(context.selector, "3d", concat_source=False)

        # Ensure data is in list format
        if not isinstance(train_data, list):
            train_data = [train_data]
        if not isinstance(all_data, list):
            all_data = [all_data]

        n_sources = len(train_data)
        fitted_resamplers = []
        transformed_features_list = []
        new_processing_names = []
        processing_names = []
        new_headers_list = []

        # Loop through each data source
        for sd_idx, (train_x, all_x) in enumerate(zip(train_data, all_data)):
            # Get processing names for this source
            processing_ids = dataset.features_processings(sd_idx)
            source_processings = processing_ids

            if context.selector.processing:
                source_processings = context.selector.processing[sd_idx]

            # Extract wavelengths for this source
            original_wavelengths = self._extract_wavelengths(dataset, sd_idx)

            # Get target wavelengths for this source
            target_wavelengths = self._get_target_wavelengths_for_source(
                op, sd_idx, n_sources
            )

            source_transformed_features = []
            source_new_processing_names = []
            source_processing_names = []
            source_resamplers = []  # Track resamplers to determine final wavelengths

            # Loop through each processing in the 3D data (samples, processings, features)
            for processing_idx in range(train_x.shape[1]):
                processing_name = processing_ids[processing_idx]

                if processing_name not in source_processings:
                    continue

                train_2d = train_x[:, processing_idx, :]  # Training data
                all_2d = all_x[:, processing_idx, :]      # All data to transform

                new_operator_name = f"{operator_name}_{runtime_context.next_op()}"

                if mode == "predict" or mode == "explain":
                    resampler = None

                    # V3: Use artifact_provider for chain-based loading
                    if runtime_context.artifact_provider is not None:
                        step_index = runtime_context.step_number
                        step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                            step_index,
                            branch_path=context.selector.branch_path,
                            source_index=sd_idx
                        )
                        # Find artifact by name matching
                        for artifact_id, obj in step_artifacts:
                            if new_operator_name in artifact_id:
                                resampler = obj
                                break

                    if resampler is None:
                        raise ValueError(
                            f"Resampler {new_operator_name} not found at step {runtime_context.step_number}"
                        )
                else:
                    # Create new resampler with target wavelengths for this source
                    from sklearn.base import clone
                    resampler = clone(op)
                    resampler.target_wavelengths = target_wavelengths

                    # Fit the resampler with original wavelengths
                    resampler.fit(train_2d, wavelengths=original_wavelengths)

                # Transform all data
                transformed_2d = resampler.transform(all_2d)

                # Apply cropping if needed based on processing type
                # Raw data: crop features directly using the stored crop mask
                # Preprocessed data: padding with 0 is already handled by fill_value in interpolation
                is_raw = processing_name.lower() == "raw" or processing_name.startswith("raw")
                if is_raw and hasattr(resampler, 'crop_mask_') and resampler.crop_mask_ is not None:
                    # Apply the crop mask to remove features outside the target range
                    from nirs4all.operators.transforms.features import CropTransformer
                    crop_indices = np.where(resampler.crop_mask_)[0]
                    if len(crop_indices) > 0:
                        crop_start = crop_indices[0]
                        crop_end = crop_indices[-1] + 1
                        cropper = CropTransformer(start=crop_start, end=crop_end)
                        transformed_2d = cropper.transform(transformed_2d)

                # Store results
                source_transformed_features.append(transformed_2d)
                new_processing_name = f"{processing_name}_{new_operator_name}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)
                source_resamplers.append(resampler)

                # Persist fitted resampler using new serializer
                if mode == "train":
                    artifact = runtime_context.saver.persist_artifact(
                        step_number=runtime_context.step_number,
                        name=new_operator_name,
                        obj=resampler,
                        format_hint='sklearn',
                        branch_id=context.selector.branch_id,
                        branch_name=context.selector.branch_name
                    )
                    fitted_resamplers.append(artifact)

            # Determine final wavelengths for headers
            # Use the OUTPUT wavelengths (target_wavelengths from interpolator_params_)
            # NOT the input wavelengths (wavelengths_after_crop_)
            final_wavelengths = target_wavelengths
            for resampler in source_resamplers:
                if hasattr(resampler, 'interpolator_params_') and resampler.interpolator_params_ is not None:
                    final_wavelengths = resampler.interpolator_params_['target_wavelengths']
                    break

            new_headers = [f"{wl:.2f}" for wl in final_wavelengths]
            new_headers_list.append(new_headers)

            transformed_features_list.append(source_transformed_features)
            new_processing_names.append(source_new_processing_names)
            processing_names.append(source_processing_names)

        # Update dataset with resampled features
        new_processing_list = list(context.selector.processing)
        for sd_idx, (source_features, src_new_processing_names, new_headers) in enumerate(
            zip(transformed_features_list, new_processing_names, new_headers_list)
        ):
            # Replace features first (resampling changes the wavelength grid)
            # Note: When feature count changes, the dataset system will handle it properly
            dataset.replace_features(
                source_processings=processing_names[sd_idx],
                features=source_features,
                processings=src_new_processing_names,
                source=sd_idx
            )
            new_processing_list[sd_idx] = src_new_processing_names

            # Update headers AFTER replacing features (so they don't get reset)
            # Resampler always outputs wavelengths in cm-1
            dataset._features.sources[sd_idx].set_headers(new_headers, unit="cm-1")  # noqa: SLF001

            if runtime_context.saver.save_artifacts:
                logger.debug(f"Exporting resampled features for dataset '{dataset.name}', source {sd_idx} to CSV...")
                logger.debug(dataset.features_processings(sd_idx))
                train_context = context.with_partition("train")
                train_x_full = dataset.x(train_context.selector, "2d", concat_source=True)
                test_context = context.with_partition("test")
                test_x_full = dataset.x(test_context.selector, "2d", concat_source=True)
                # save train and test features to CSV for debugging, create folder if needed
                import os
                root_path = runtime_context.saver.base_path
                os.makedirs(f"{root_path}/{dataset.name}", exist_ok=True)
                np.savetxt(f"{root_path}/{dataset.name}/Export_X_train.csv", train_x_full, delimiter=",")
                np.savetxt(f"{root_path}/{dataset.name}/Export_X_test.csv", test_x_full, delimiter=",")

        context = context.with_processing(new_processing_list)
        context = context.with_metadata(add_feature=False)

        return context, fitted_resamplers
