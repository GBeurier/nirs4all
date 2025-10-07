"""
Controller for wavelength resampling operations.

This controller handles the Resampler operator, extracting wavelengths from
dataset headers and managing the resampling process across multiple sources.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import pickle

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.operators.transformations.resampler import Resampler

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset


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
            if isinstance(model_obj, dict) and '_runtime_instance' in model_obj:
                model_obj = model_obj['_runtime_instance']
        elif operator is not None:
            model_obj = operator
        else:
            model_obj = step

        # Check if it's a Resampler
        return (isinstance(model_obj, Resampler) or
                (hasattr(model_obj, '__class__') and
                 model_obj.__class__.__name__ == 'Resampler'))

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
            Array of wavelengths as floats

        Raises:
            ValueError: If headers cannot be converted to float
        """
        headers = dataset.headers(source_idx)

        try:
            wavelengths = np.array([float(h) for h in headers])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert headers to wavelengths for source {source_idx}. "
                f"Headers must be numeric values (wavelengths in cm-1). "
                f"Got headers: {headers[:5]}... "
                f"Error: {str(e)}"
            )

        return wavelengths

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
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ):
        """
        Execute resampling operation.

        Args:
            step: Pipeline step configuration
            operator: The Resampler instance
            dataset: Dataset to operate on
            context: Pipeline execution context
            runner: Pipeline runner instance
            source: Data source index (-1 for all sources)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode
            prediction_store: External prediction store (unused)

        Returns:
            Tuple of (updated_context, fitted_resamplers)
        """
        operator_name = operator.__class__.__name__

        # Get train and all data as lists of 3D arrays (one per source)
        train_context = context.copy()
        train_context["partition"] = "train"

        train_data = dataset.x(train_context, "3d", concat_source=False)
        all_data = dataset.x(context, "3d", concat_source=False)

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

            if "processing" in context:
                source_processings = context["processing"][sd_idx]

            # Extract wavelengths for this source
            original_wavelengths = self._extract_wavelengths(dataset, sd_idx)

            # Get target wavelengths for this source
            target_wavelengths = self._get_target_wavelengths_for_source(
                operator, sd_idx, n_sources
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

                new_operator_name = f"{operator_name}_{runner.next_op()}"

                if loaded_binaries and (mode == "predict" or mode == "explain"):
                    # Load pre-fitted resampler from binaries
                    resampler = dict(loaded_binaries).get(f"{new_operator_name}")
                    if resampler is None:
                        raise ValueError(
                            f"Binary for {new_operator_name} not found in loaded_binaries"
                        )
                else:
                    # Create new resampler with target wavelengths for this source
                    from sklearn.base import clone
                    resampler = clone(operator)
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
                    from nirs4all.operators.transformations.features import CropTransformer
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

                # Serialize fitted resampler
                resampler_binary = pickle.dumps(resampler)
                fitted_resamplers.append((f"{new_operator_name}.pkl", resampler_binary))

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
            context["processing"][sd_idx] = src_new_processing_names

            # Update headers AFTER replacing features (so they don't get reset)
            dataset._features.sources[sd_idx].set_headers(new_headers)

            if runner.save_files:
                print(f"Exporting resampled features for dataset '{dataset.name}', source {sd_idx} to CSV...")
                print(dataset.features_processings(sd_idx))
                train_context = {"partition": "train"}
                train_x_full = dataset.x(train_context, "2d", concat_source=True)
                test_context = {"partition": "test"}
                test_x_full = dataset.x(test_context, "2d", concat_source=True)
                # save train and test features to CSV for debugging, create folder if needed
                import os
                root_path = runner.saver.base_path
                os.makedirs(f"{root_path}/{dataset.name}", exist_ok=True)
                np.savetxt(f"{root_path}/{dataset.name}/Export_X_train.csv", train_x_full, delimiter=",")
                np.savetxt(f"{root_path}/{dataset.name}/Export_X_test.csv", test_x_full, delimiter=",")

        context["add_feature"] = False



        return context, fitted_resamplers
