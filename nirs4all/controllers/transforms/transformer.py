from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
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
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ):
        """Execute transformer - handles normal, feature augmentation, and sample augmentation modes."""
        op = step_info.operator

        # Check if we're in sample augmentation mode
        if context.get("augment_sample", False) and mode not in ["predict", "explain"]:
            return self._execute_for_sample_augmentation(
                op, dataset, context, runner, mode, loaded_binaries, prediction_store
            )

        # Normal or feature augmentation execution (existing code)
        operator_name = op.__class__.__name__

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

        fitted_transformers = []
        transformed_features_list = []
        new_processing_names = []
        processing_names = []

        # Loop through each data source
        for sd_idx, (train_x, all_x) in enumerate(zip(train_data, all_data)):
            # print(f"Processing source {sd_idx}: train shape {train_x.shape}, all shape {all_x.shape}")

            # Get processing names for this source
            processing_ids = dataset.features_processings(sd_idx)
            source_processings = processing_ids
            # print("🔹 Processing source", sd_idx, "with processings:", source_processings)
            if "processing" in context:
                source_processings = context["processing"][sd_idx]

            source_transformed_features = []
            source_new_processing_names = []
            source_processing_names = []

            # Loop through each processing in the 3D data (samples, processings, features)
            for processing_idx in range(train_x.shape[1]):
                processing_name = processing_ids[processing_idx]
                # print(f" Processing {processing_name} (idx {processing_idx})")
                # print(processing_name, processing_name in source_processings)
                if processing_name not in source_processings:
                    continue
                train_2d = train_x[:, processing_idx, :]  # Training data
                all_2d = all_x[:, processing_idx, :]      # All data to transform

                # print(f" Processing {processing_name} (idx {processing_idx}): train {train_2d.shape}, all {all_2d.shape}")
                new_operator_name = f"{operator_name}_{runner.next_op()}"

                if loaded_binaries and (mode == "predict" or mode == "explain"):
                    transformer = dict(loaded_binaries).get(f"{new_operator_name}")
                    if transformer is None:
                        raise ValueError(f"Binary for {new_operator_name} not found in loaded_binaries")
                else:
                    transformer = clone(op)
                    transformer.fit(train_2d)

                transformed_2d = transformer.transform(all_2d)

                # print("  Transformed shape:", transformed_2d.shape)

                # Store results
                source_transformed_features.append(transformed_2d)
                new_processing_name = f"{processing_name}_{new_operator_name}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)

                # Persist fitted transformer using new serializer
                if mode == "train":
                    artifact = runner.saver.persist_artifact(
                        step_number=runner.step_number,
                        name=new_operator_name,
                        obj=transformer,
                        format_hint='sklearn'
                    )
                    fitted_transformers.append(artifact)

            # print("🔹 Finished processing source", sd_idx, len(fitted_transformers))
            # ("🔹 New processing names:", source_new_processing_names)
            transformed_features_list.append(source_transformed_features)
            new_processing_names.append(source_new_processing_names)
            processing_names.append(source_processing_names)

        for sd_idx, (source_features, src_new_processing_names) in enumerate(zip(transformed_features_list, new_processing_names)):
            if "add_feature" in context and context["add_feature"]:
                dataset.add_features(source_features, src_new_processing_names, source=sd_idx)
                context["processing"][sd_idx] = src_new_processing_names
            else:
                dataset.replace_features(
                    source_processings=processing_names[sd_idx],
                    features=source_features,
                    processings=src_new_processing_names,
                    source=sd_idx
                )
                context["processing"][sd_idx] = src_new_processing_names
        context["add_feature"] = False

        # print(dataset)
        return context, fitted_transformers

    def _execute_for_sample_augmentation(
        self,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        prediction_store: Optional[Any]
    ) -> Tuple[Dict[str, Any], List]:
        """
        Apply transformer to origin samples and add augmented samples.

        Context contains:
            - augment_sample: True flag (like add_feature)
            - target_samples: list of sample_ids to augment
            - partition: "train" (filtering context)
        """
        target_sample_ids = context.get("target_samples", [])
        if not target_sample_ids:
            return context, []

        operator_name = operator.__class__.__name__
        fitted_transformers = []

        # Get train data for fitting (if not in predict/explain mode)
        if mode not in ["predict", "explain"]:
            train_context = context.copy()
            train_context["partition"] = "train"
            train_data = dataset.x(train_context, "3d", concat_source=False, include_augmented=False)
            if not isinstance(train_data, list):
                train_data = [train_data]

        # Process each target sample
        for sample_id in target_sample_ids:
            # Get origin sample data (all sources, base samples only)
            origin_selector = {"sample": [sample_id]}
            origin_data = dataset.x(origin_selector, "3d", concat_source=False, include_augmented=False)

            # Ensure list format for multi-source
            if not isinstance(origin_data, list):
                origin_data = [origin_data]

            # Transform each source
            transformed_sources = []

            for source_idx, source_data in enumerate(origin_data):
                # source_data shape: (1, n_processings, n_features) for single sample
                source_2d_list = []

                for proc_idx in range(source_data.shape[1]):
                    proc_data = source_data[:, proc_idx, :]  # (1, n_features)

                    # Apply transformer
                    if loaded_binaries and mode in ["predict", "explain"]:
                        transformer = dict(loaded_binaries).get(f"{operator_name}_{source_idx}_{proc_idx}_{sample_id}")
                        if transformer is None:
                            raise ValueError(f"Binary for {operator_name} not found")
                    else:
                        transformer = clone(operator)
                        # Fit on train data for this source/processing
                        train_proc_data = train_data[source_idx][:, proc_idx, :]
                        transformer.fit(train_proc_data)

                    transformed_data = transformer.transform(proc_data)
                    source_2d_list.append(transformed_data)

                    # Save transformer binary
                    transformer_binary = pickle.dumps(transformer)
                    fitted_transformers.append(
                        (f"{operator_name}_{source_idx}_{proc_idx}_{sample_id}.pkl", transformer_binary)
                    )

                # Stack back to 3D (1, n_processings, n_features)
                source_3d = np.stack(source_2d_list, axis=1)
                transformed_sources.append(source_3d)

            # Add augmented sample to dataset using add_samples with proper indexing
            # Use only the transformer name for augmentation column (like processings)

            # Build index dictionary for the augmented sample
            index_dict = {
                "partition": "train",
                "origin": sample_id,  # Track origin
                "augmentation": operator_name  # Transformer name only
            }

            # Note: Metadata copying is handled by the indexer, not here
            # The indexer's add_samples_dict will set origin and augmentation columns

            # Add augmented sample (transformed_sources is list of 3D arrays, one per source)
            # Need to convert to format expected by add_samples
            if len(transformed_sources) == 1:
                # Single source: pass 2D array (squeeze out sample dimension)
                data_to_add = transformed_sources[0][0, :, :]  # (n_processings, n_features)
            else:
                # Multi-source: pass list of 2D arrays
                data_to_add = [src[0, :, :] for src in transformed_sources]

            dataset.add_samples(
                data=data_to_add,
                indexes=index_dict
            )

        return context, fitted_transformers
