"""
Controller for feature selection operations (CARS, MC-UVE).

This controller handles feature selection operators, extracting wavelengths from
dataset headers and managing the selection process across multiple sources and
preprocessings.
"""

from typing import Any, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.transforms.feature_selection import CARS, MCUVE

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.steps.runtime import RuntimeContext


@register_controller
class FeatureSelectionController(OperatorController):
    """
    Controller for feature selection operators (CARS, MC-UVE).

    This controller:
    1. Extracts wavelengths from dataset headers
    2. Fits the selector on training data with target values
    3. Transforms all data to keep only selected wavelengths
    4. Updates dataset with new features and headers
    5. Supports multi-source datasets with per-source selection
    """

    priority = 5  # Higher priority than TransformerMixin (10) to match first

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match CARS and MCUVE objects."""
        # Get the actual model object
        model_obj = None
        if isinstance(step, dict) and 'model' in step:
            model_obj = step['model']
        elif operator is not None:
            model_obj = operator
        else:
            model_obj = step

        # Check if it's a feature selection operator
        is_cars = hasattr(model_obj, '__class__') and model_obj.__class__.__name__ == 'CARS'
        is_mcuve = hasattr(model_obj, '__class__') and model_obj.__class__.__name__ == 'MCUVE'

        return isinstance(model_obj, (CARS, MCUVE)) or is_cars or is_mcuve

    @classmethod
    def use_multi_source(cls) -> bool:
        """Feature selection supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Feature selection supports prediction mode."""
        return True

    def _extract_wavelengths(self, dataset: 'SpectroDataset', source_idx: int) -> Optional[np.ndarray]:
        """
        Extract wavelengths from dataset headers if available.

        Args:
            dataset: The spectroscopic dataset
            source_idx: Index of the data source

        Returns:
            Array of wavelengths or None if headers are not numeric
        """
        try:
            header_unit = dataset.header_unit(source_idx)

            # Feature selection can work without wavelengths (just indices)
            if header_unit in ["text", "none", "index"]:
                return None

            # Try to get wavelengths
            wavelengths = dataset.wavelengths_cm1(source_idx)
            return wavelengths

        except (ValueError, TypeError):
            return None

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
        Execute feature selection operation.

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
            Tuple of (updated_context, fitted_selectors)
        """
        op = step_info.operator
        operator_name = op.__class__.__name__

        # Get train and all data as lists of 3D arrays (one per source)
        train_context = context.with_partition("train")

        train_data = dataset.x(train_context.selector, "3d", concat_source=False)
        all_data = dataset.x(context.selector, "3d", concat_source=False)

        # Get target values for fitting
        y_train = dataset.y(train_context.selector).ravel()

        # Ensure data is in list format
        if not isinstance(train_data, list):
            train_data = [train_data]
        if not isinstance(all_data, list):
            all_data = [all_data]

        fitted_selectors = []
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

            # Extract wavelengths for this source (may be None if not numeric)
            original_wavelengths = self._extract_wavelengths(dataset, sd_idx)

            source_transformed_features = []
            source_new_processing_names = []
            source_processing_names = []
            source_selectors = []

            # For feature selection, we fit ONE selector and apply to all preprocessings
            # This ensures all preprocessings have the same selected features
            master_selector = None

            # Loop through each processing in the 3D data (samples, processings, features)
            for processing_idx in range(train_x.shape[1]):
                processing_name = processing_ids[processing_idx]

                if processing_name not in source_processings:
                    continue

                train_2d = train_x[:, processing_idx, :]  # Training data
                all_2d = all_x[:, processing_idx, :]      # All data to transform

                new_operator_name = f"{operator_name}_{runtime_context.next_op()}"

                if mode == "predict" or mode == "explain":
                    selector = None

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
                                selector = obj
                                break

                    if selector is None:
                        raise ValueError(
                            f"Feature selector {new_operator_name} not found at step {runtime_context.step_number}"
                        )
                elif master_selector is None:
                    # First preprocessing: fit the master selector
                    from sklearn.base import clone
                    master_selector = clone(op)

                    # Fit the selector with training data and target values
                    master_selector.fit(train_2d, y_train, wavelengths=original_wavelengths)
                    selector = master_selector

                    if runtime_context.step_runner.verbose > 0:
                        logger.info(f"   {operator_name}: Selected {selector.n_features_out_} "
                              f"from {selector.n_features_in_} features (applied to all preprocessings)")
                else:
                    # Use master selector for subsequent preprocessings
                    selector = master_selector

                # Transform all data
                transformed_2d = selector.transform(all_2d)

                # Store results
                source_transformed_features.append(transformed_2d)
                new_processing_name = f"{processing_name}_{new_operator_name}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)
                source_selectors.append(selector)

                # Persist fitted selector using serializer
                if mode == "train":
                    artifact = runtime_context.saver.persist_artifact(
                        step_number=runtime_context.step_number,
                        name=new_operator_name,
                        obj=selector,
                        format_hint='sklearn',
                        branch_id=context.selector.branch_id,
                        branch_name=context.selector.branch_name
                    )
                    fitted_selectors.append(artifact)

            # Determine new headers based on selected indices
            if original_wavelengths is not None and len(source_selectors) > 0:
                # Use first selector's indices (all should select same features)
                selected_wl = original_wavelengths[source_selectors[0].selected_indices_]
                new_headers = [f"{wl:.2f}" for wl in selected_wl]
            else:
                # Use indices if no wavelengths
                if len(source_selectors) > 0:
                    new_headers = [str(i) for i in source_selectors[0].selected_indices_]
                else:
                    new_headers = None

            new_headers_list.append(new_headers)
            transformed_features_list.append(source_transformed_features)
            new_processing_names.append(source_new_processing_names)
            processing_names.append(source_processing_names)

        # Update dataset with selected features
        new_processing_list = list(context.selector.processing)
        for sd_idx, (source_features, src_new_processing_names, new_headers) in enumerate(
            zip(transformed_features_list, new_processing_names, new_headers_list)
        ):
            # Replace features (selection changes the feature count)
            dataset.replace_features(
                source_processings=processing_names[sd_idx],
                features=source_features,
                processings=src_new_processing_names,
                source=sd_idx
            )
            new_processing_list[sd_idx] = src_new_processing_names

            # Update headers AFTER replacing features
            if new_headers is not None:
                header_unit = dataset.header_unit(sd_idx)
                dataset._features.sources[sd_idx].set_headers(new_headers, unit=header_unit)  # noqa: SLF001

            if runtime_context.step_runner.verbose > 0:
                n_features = source_features[0].shape[1] if source_features else 0
                logger.info(f"   Source {sd_idx}: Updated to {n_features} features")

        context = context.with_processing(new_processing_list)
        context = context.with_metadata(add_feature=False)

        return context, fitted_selectors
