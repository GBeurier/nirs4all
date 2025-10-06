from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset

import numpy as np
import pickle
from sklearn.base import clone
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
            if isinstance(model_obj, dict) and '_runtime_instance' in model_obj:
                model_obj = model_obj['_runtime_instance']
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
                    transformer = clone(operator)
                    transformer.fit(train_2d)

                transformed_2d = transformer.transform(all_2d)

                # print("  Transformed shape:", transformed_2d.shape)

                # Store results
                source_transformed_features.append(transformed_2d)
                new_processing_name = f"{processing_name}_{new_operator_name}"
                source_new_processing_names.append(new_processing_name)
                source_processing_names.append(processing_name)

                # Serialize fitted transformer
                transformer_binary = pickle.dumps(transformer)
                fitted_transformers.append((f"{new_operator_name}.pkl", transformer_binary))

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
