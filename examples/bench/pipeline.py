from typing import Any, Dict, List, Callable, Union, Optional, Tuple
import numpy as np

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
import tensorflow as tf

from spectradataset import SpectraDataset


class PipelineRunner:
    def __init__(self):
        self.selector = {"branch": 0}

    def run_pipeline(self, pipeline: List[Any], data: SpectraDataset, prefix: str = ""):
        """Point d'entrée principal pour exécuter un pipeline."""
        print("Running pipeline")

        for step in pipeline:
            self._run_step(step, data, prefix=prefix)

        print("Pipeline finished.")

    def _run_step(self, step: Any, data: SpectraDataset, prefix: str = ""):
        def p(msg: str, end: str = "\n"):
            print(prefix + msg, end=end)

        # Étapes string simples
        if isinstance(step, str):
            self._handle_string_step(step, data, p)
            return

        # Étapes de contrôle (dict avec une seule clé)
        if isinstance(step, dict) and len(step) == 1:
            key, spec = next(iter(step.items()))
            self._handle_control_step(key, spec, data, prefix)
            return

        # Étapes complexes (dict avec clés multiples)
        if isinstance(step, dict) and len(step) > 1:
            self._handle_complex_step(step, data, prefix)
            return

        # Listes (sous-pipelines)
        if isinstance(step, list):
            self._handle_sub_pipeline(step, data, prefix)
            return

        # Instances d'objets
        self._handle_instance_step(step, data, prefix)

    def _handle_string_step(self, step: str, data: SpectraDataset, p_func):
        """Gère les étapes string simples."""
        if step == "uncluster":
            self.context.update(group_filter=False)
            p_func(f"Control step: {step}")
        else:
            # Étapes de plotting ou non implémentées
            p_func(f"Unknown string step: {step}")

    def _handle_control_step(self, key: str, spec: Any, data: SpectraDataset, prefix: str):
        """Gère les étapes de contrôle (clustering, augmentation, branching, etc.)."""

        def p(msg: str, end: str = "\n"):
            print(prefix + msg, end=end)

        p(f"Control step: {key}")

        if key == "cluster":
            self._run_clustering(spec, data, prefix + "  ")

        elif key in ["sample_augmentation", "samples", "S"]:
            self._run_sample_augmentation(spec, data, prefix + "  ")

        elif key in ["feature_augmentation", "features", "F"]:
            self._run_feature_augmentation(spec, data, prefix + "  ")

        elif key == "branch":
            self._run_branching(spec, data, prefix + "  ")

        elif key == "stack":
            self._run_stacking(spec, data, prefix + "  ")

        else:
            p(f"Unknown control step: {key}")

    def _handle_complex_step(self, step: Dict[str, Any], data: SpectraDataset, prefix: str):
        """Gère les étapes complexes (modèles, etc.)."""

        if "model" in step:
            self._run_model(step, data, prefix)
        elif "cluster" in step:
            self._run_clustering(step["cluster"], data, prefix)
        else:
            print(f"{prefix}Unknown complex step: {step}")

    def _handle_sub_pipeline(self, steps: List[Any], data: SpectraDataset, prefix: str):
        """Gère les sous-pipelines (listes d'étapes)."""
        print(f"{prefix}Sub-pipeline")
        for sub_step in steps:
            self._run_step(sub_step, data, prefix + "  ")

    def _handle_instance_step(self, step: Any, data: SpectraDataset, prefix: str):
        """Gère les instances d'objets (transformers, splitters, etc.)."""

        def p(msg: str, end: str = "\n"):
            print(prefix + msg, end=end)

        if hasattr(step, "transform") and issubclass(step.__class__, TransformerMixin):
            data._run_transformation(selector=self.selector, transformer=step)

        elif hasattr(step, "split"):
            self._run_splitting(step, data, prefix)

        elif hasattr(step, "fit") and issubclass(step.__class__, ClusterMixin):
            self._run_clustering(step, data, prefix)

        elif issubclass(step.__class__, BaseEstimator):
            self._run_model({"model": step}, data, prefix)

        elif issubclass(step.__class__, tf.keras.Model):
            p(f"tensorflow > {step.__class__.__name__}")
            self._run_model({"model": step}, data, prefix)

        # elif issubclass(step.__class__, nn.Module):
        #     p(f"pytorch > {step.__class__.__name__}")
        #     self._run_model({"model": step}, data, prefix)

        else:
            p(f"Unknown step: {step}")




        # Update the spectra with transformed data


        #
        # print(f"{prefix}  Filters: {filters}")

        # # Récupère les données d'entraînement pour le fit
        # X_train = data.X(set="train", **filters)
        # if len(X_train) == 0:
        #     print(f"{prefix}  Warning: No training data found for transformation")
        #     return

        # # Fit sur les données d'entraînement
        # transformer.fit(X_train)

        # # Transform all data (train + test) corresponding to filters
        # for dataset_name in ["train", "test"]:
        #     X_set = data.X(set=dataset_name, **filters)
        #     if len(X_set) > 0:
        #         X_transformed = transformer.transform(X_set)

        #         # Update the spectra for this specific set - replace existing data
        #         set_filters = {**filters, "set": dataset_name}
        #         data.change_spectra(X_transformed, **set_filters)
        #         print(f"{prefix}  Transformed {len(X_transformed)} {dataset_name} samples")

        # # Update context with new processing hash
        # new_processing = self._hash_transformation(transformer, self.context.processing)
        # self.context.update(processing=new_processing)
        # print(f"{prefix}  Updated processing: {self.context.processing} -> {new_processing}")

    def _run_splitting(self, splitter, data: SpectraDataset, prefix: str):
        """Gère le splitting des données."""
        print(f"{prefix}Splitting with {splitter.__class__.__name__}")

        # filters = self.context.get_filters()

        # # For the first split, we need to work with all data (currently all marked as train)
        # if not self.context.train_test_split_done:
        #     X = data.X(**filters)
        #     y = data.y(**filters)

        #     if len(X) == 0:
        #         print(f"{prefix}  Warning: No data found for splitting")
        #         return

        #     try:
        #         # Get the first split (train/test)
        #         splits = list(splitter.split(X, y))
        #         if len(splits) == 0:
        #             print(f"{prefix}  Warning: No splits generated")
        #             return

        #         train_idx, test_idx = splits[0]

        #         # Get all source_ids that match current filters
        #         # We need to get the source_ids in the same order as X and y
        #         result_df = data.features

        #         # Apply feature-level filters
        #         feat_cols = set(data.features.columns)
        #         label_cols = set(data.labels.columns) if data.labels is not None else set()

        #         feat_filters = {k: v for k, v in filters.items() if k in feat_cols}
        #         label_filters = {k: v for k, v in filters.items() if k in label_cols}

        #         if feat_filters:
        #             result_df = result_df.filter(data._mask(**feat_filters))

        #         if label_filters and data.labels is not None:
        #             filtered_labels = data.labels.filter(data._mask(**label_filters))
        #             result_df = result_df.join(filtered_labels, on="source_id", how="inner")
        #         elif data.labels is not None:
        #             result_df = result_df.join(data.labels, on="source_id", how="inner")

        #         source_ids = result_df["source_id"].to_list()

        #         # Set train/test tags based on split
        #         train_source_ids = [source_ids[i] for i in train_idx]
        #         test_source_ids = [source_ids[i] for i in test_idx]

        #         # Update the set tags
        #         data.set_tag("set", "train", source_id=train_source_ids)
        #         data.set_tag("set", "test", source_id=test_source_ids)

        #         self.context.update(train_test_split_done=True)
        #         print(f"{prefix}  Train/Test split: {len(train_idx)}/{len(test_idx)}")

        #     except Exception as e:
        #         print(f"{prefix}  Error in splitting: {e}")

        # # Splits suivants = cross-validation
        # else:
        #     if isinstance(splitter, BaseCrossValidator):
        #         print(f"{prefix}  Setting up cross-validation with {splitter.__class__.__name__}")
        #         # La CV sera gérée lors de l'entraînement des modèles

    def _run_clustering(self, clusterer, data: SpectraDataset, prefix: str):
        """Gère le clustering des données."""
        print(f"{prefix}Clustering with {clusterer.__class__.__name__ if hasattr(clusterer, '__class__') else str(clusterer)}")
        # """Applique le clustering sur les données."""
        # if hasattr(clusterer, "fit") and issubclass(clusterer.__class__, ClusterMixin):
        #     print(f"{prefix}Clustering with {clusterer.__class__.__name__}")

        #     filters = self.context.get_filters()
        #     filters["set"] = "train"  # Cluster seulement sur train

        #     X_train = data.X(**filters)
        #     if len(X_train) > 0:
        #         cluster_labels = clusterer.fit_predict(X_train)

        #         # Get the source_ids for the training data
        #         result_df = data.features

        #         # Apply filters to get matching features
        #         feat_cols = set(data.features.columns)
        #         label_cols = set(data.labels.columns) if data.labels is not None else set()

        #         feat_filters = {k: v for k, v in filters.items() if k in feat_cols}
        #         label_filters = {k: v for k, v in filters.items() if k in label_cols}

        #         if feat_filters:
        #             result_df = result_df.filter(data._mask(**feat_filters))

        #         if label_filters and data.labels is not None:
        #             filtered_labels = data.labels.filter(data._mask(**label_filters))
        #             result_df = result_df.join(filtered_labels, on="source_id", how="inner")
        #         elif data.labels is not None:
        #             result_df = result_df.join(data.labels, on="source_id", how="inner")

        #         source_ids = result_df["source_id"].unique().to_list()

        #         # Add cluster labels - need to match source_ids to cluster labels
        #         if len(source_ids) == len(cluster_labels):
        #             data.add_tag("group", cluster_labels.tolist(), source_id=source_ids)
        #             self.context.update(group_filter=True)
        #             print(f"{prefix}  Created {len(np.unique(cluster_labels))} clusters")
        #         else:
        #             print(f"{prefix}  Warning: Mismatch between source_ids ({len(source_ids)}) and cluster labels ({len(cluster_labels)})")
        #     else:
        #         print(f"{prefix}  Warning: No training data for clustering")
        # else:
        #     print(f"{prefix}Unknown clustering step: {clusterer}")

    def _run_sample_augmentation(self, augmenters: List[Any], data: SpectraDataset, prefix: str):
        print(f"{prefix}Sample augmentation with {len(augmenters)} augmenter(s)")
        # """Applique l'augmentation d'échantillons - ajoute de nouveaux échantillons."""
        for augmenter in augmenters:
            if augmenter is None:
                print(f"{prefix}Identity (no sample augmentation)")
                continue

            print(f"{prefix}Sample augmentation with {augmenter}")

            filters = self.context.get_filters()
            filters["set"] = "train"  # Augmente seulement le train

        #     X_train = data.X(**filters)
        #     y_train = data.y(**filters)

        #     if len(X_train) > 0 and hasattr(augmenter, "fit_transform"):
        #         X_augmented = augmenter.fit_transform(X_train)

        #         # Get original source_ids for tracking
        #         result_df = data.features
        #         feat_cols = set(data.features.columns)
        #         label_cols = set(data.labels.columns) if data.labels is not None else set()

        #         feat_filters = {k: v for k, v in filters.items() if k in feat_cols}
        #         label_filters = {k: v for k, v in filters.items() if k in label_cols}

        #         if feat_filters:
        #             result_df = result_df.filter(data._mask(**feat_filters))

        #         if label_filters and data.labels is not None:
        #             filtered_labels = data.labels.filter(data._mask(**label_filters))
        #             result_df = result_df.join(filtered_labels, on="source_id", how="inner")
        #         elif data.labels is not None:
        #             result_df = result_df.join(data.labels, on="source_id", how="inner")

        #         original_source_ids = result_df["source_id"].to_list()

        #         # Ajoute les échantillons augmentés avec nouveaux source_ids mais garde l'origine
        #         aug_id = self._hash_transformation(augmenter, "sample_aug")
        #         for i, (spec, target) in enumerate(zip(X_augmented, y_train)):
        #             original_id = original_source_ids[i % len(original_source_ids)]
        #             data.add_spectra(
        #                 [spec], target,
        #                 set="train",
        #                 branch=self.context.branch,  # Keep current branch
        #                 processing=self.context.processing,  # Keep current processing
        #                 augmentation=aug_id,
        #                 origin=original_id  # Track original source
        #             )

        #         print(f"{prefix}  Added {len(X_augmented)} augmented samples")

    def _run_feature_augmentation(self, augmenters: List[Any], data: SpectraDataset, prefix: str):
        print(f"{prefix}Feature augmentation with {len(augmenters)} augmenter(s)")
        # """Applique l'augmentation de features - ajoute de nouvelles versions des échantillons."""

        for i, augmenter in enumerate(augmenters):
            print(f"{prefix}Feature augmentation {i+1}/{len(augmenters)}")

            # Get all data from current context
            filters = self.context.get_filters()

            for set_name in ["train", "test"]:
                set_filters = {**filters, "set": set_name}
                # X_set = data.X(**set_filters)
        #         y_set = data.y(**set_filters)

        #         if len(X_set) == 0:
        #             continue

                if augmenter is None:
                    print(f"{prefix}  Identity feature augmentation for {set_name}")
        #             # For identity, add copies with different augmentation ID
        #             aug_id = "identity"

        #             # Get original source_ids
        #             result_df = data.features
        #             feat_cols = set(data.features.columns)
        #             label_cols = set(data.labels.columns) if data.labels is not None else set()

        #             feat_filters = {k: v for k, v in set_filters.items() if k in feat_cols}
        #             label_filters = {k: v for k, v in set_filters.items() if k in label_cols}

        #             if feat_filters:
        #                 result_df = result_df.filter(data._mask(**feat_filters))

        #             if label_filters and data.labels is not None:
        #                 filtered_labels = data.labels.filter(data._mask(**label_filters))
        #                 result_df = result_df.join(filtered_labels, on="source_id", how="inner")
        #             elif data.labels is not None:
        #                 result_df = result_df.join(data.labels, on="source_id", how="inner")

        #             original_source_ids = result_df["source_id"].to_list()

        #             # Add identity copies with same source_id but different augmentation
        #             for spec, target, orig_id in zip(X_set, y_set, original_source_ids):
        #                 data.add_spectra(
        #                     [spec], target,
        #                     set=set_name,
        #                     branch=self.context.branch,
        #                     processing=self.context.processing,
        #                     augmentation=aug_id,
        #                     sample=orig_id  # Keep original sample reference
        #                 )

                else:
                    print(f"{prefix}  Applying {augmenter} to {set_name}")

        #             # Fit only on train, transform on current set
        #             if set_name == "train":
        #                 augmenter.fit(X_set)
        #                 X_augmented = augmenter.transform(X_set)
        #             else:
        #                 X_augmented = augmenter.transform(X_set)

        #             # Get original source_ids
        #             result_df = data.features
        #             feat_cols = set(data.features.columns)
        #             label_cols = set(data.labels.columns) if data.labels is not None else set()

        #             feat_filters = {k: v for k, v in set_filters.items() if k in feat_cols}
        #             label_filters = {k: v for k, v in set_filters.items() if k in label_cols}

        #             if feat_filters:
        #                 result_df = result_df.filter(data._mask(**feat_filters))

        #             if label_filters and data.labels is not None:
        #                 filtered_labels = data.labels.filter(data._mask(**label_filters))
        #                 result_df = result_df.join(filtered_labels, on="source_id", how="inner")
        #             elif data.labels is not None:
        #                 result_df = result_df.join(data.labels, on="source_id", how="inner")

        #             original_source_ids = result_df["source_id"].to_list()

        #             # Create augmentation ID
        #             aug_id = self._hash_transformation(augmenter, f"feature_aug_{i}")

        #             # Add augmented features with same source_id but different augmentation
        #             for spec, target, orig_id in zip(X_augmented, y_set, original_source_ids):
        #                 data.add_spectra(
        #                     [spec], target,
        #                     set=set_name,
        #                     branch=self.context.branch,
        #                     processing=self.context.processing,
        #                     augmentation=aug_id,
        #                     sample=orig_id  # Keep original sample reference
        #                 )

        #         print(f"{prefix}    Added {len(X_set)} {set_name} feature-augmented samples")

    def _run_branching(self, branches: List[Dict[str, Any]], data: SpectraDataset, prefix: str):
        print(f"{prefix}Branching with {len(branches)} branches")

        for i, branch_config in enumerate(branches):
            branch_number = i + 1  # Branch 0 is original, branches start at 1
            print(f"{prefix}  Branch {branch_number}")

            # First, copy all current data to the new branch
        #     self._copy_data_to_new_branch(data, branch_number, prefix + "    ")

            # Create new context for this branch
            branch_context = self.context.copy()
            branch_context.update(branch=branch_number)

            # Save current context and switch to branch context
            old_context = self.context
            self.context = branch_context

            try:
                # Execute the branch pipeline
                self._run_step(branch_config, data, prefix + "    ")
            finally:
                # Restore original context
                self.context = old_context

    # def _copy_data_to_new_branch(self, data: SpectraDataset, new_branch: int, prefix: str):
    #     """Copy all data from current branch to new branch."""
    #     current_filters = self.context.get_filters()

    #     for set_name in ["train", "test"]:
    #         set_filters = {**current_filters, "set": set_name}
    #         X_source = data.X(**set_filters)
    #         y_source = data.y(**set_filters)

    #         if len(X_source) > 0:
    #             # Get source metadata for copying
    #             result_df = data.features
    #             feat_cols = set(data.features.columns)
    #             label_cols = set(data.labels.columns) if data.labels is not None else set()

    #             feat_filters = {k: v for k, v in set_filters.items() if k in feat_cols}
    #             label_filters = {k: v for k, v in set_filters.items() if k in label_cols}

    #             if feat_filters:
    #                 result_df = result_df.filter(data._mask(**feat_filters))

    #             if label_filters and data.labels is not None:
    #                 filtered_labels = data.labels.filter(data._mask(**label_filters))
    #                 result_df = result_df.join(filtered_labels, on="source_id", how="inner")
    #             elif data.labels is not None:
    #                 result_df = result_df.join(data.labels, on="source_id", how="inner")

    #             # Copy each sample to new branch
    #             for j, (spec, target) in enumerate(zip(X_source, y_source)):
    #                 original_row = result_df.row(j, named=True)

    #                 # Copy with new branch but preserve other metadata
    #                 metadata = {
    #                     "set": set_name,
    #                     "branch": new_branch,
    #                     "processing": self.context.processing,
    #                     "augmentation": original_row.get("augmentation", "raw"),
    #                 }

    #                 # Copy other metadata if present
    #                 for key in ["origin", "sample", "type"]:
    #                     if key in original_row:
    #                         metadata[key] = original_row[key]

    #                 data.add_spectra([spec], target, **metadata)

    #             print(f"{prefix}Copied {len(X_source)} {set_name} samples to branch {new_branch}")

    def _run_stacking(self, stack_config: Dict[str, Any], data: SpectraDataset, prefix: str):
        # """Gère le stacking de modèles."""
        print(f"{prefix}Stacking setup")

        # Entraîne d'abord les base learners
        if "base_learners" in stack_config:
            print(f"{prefix}  Training base learners:")
            for i, base_config in enumerate(stack_config["base_learners"]):
                print(f"{prefix}    Base learner {i + 1}:")
                self._run_model(base_config, data, prefix + "      ")

        # # Puis le meta-modèle
        print(f"{prefix}  Training meta-model:")
        meta_config = {k: v for k, v in stack_config.items() if k != "base_learners"}
        self._run_model(meta_config, data, prefix + "    ")

    def _run_model(self, model_config: Dict[str, Any], data: SpectraDataset, prefix: str):
        """Entraîne et évalue un modèle."""
        print(f"{prefix}Model step:")

        model = model_config["model"]
        print(f"{prefix}  {model.__class__.__name__}")

        # Applique le y_pipeline si présent
        if "y_pipeline" in model_config and model_config["y_pipeline"] is not None:
            self._run_step(model_config["y_pipeline"], data, prefix + "  ")

        # # Récupère les données pour l'entraînement
        # filters = self.context.get_filters()
        # X_train = data.X(set="train", **filters)
        # y_train = data.y(set="train", **filters)
        # X_test = data.X(set="test", **filters)
        # y_test = data.y(set="test", **filters)

        # if len(X_train) == 0:
        #     print(f"{prefix}  Warning: No training data found")
        #     return

        # # Gère le fine-tuning si présent
        # if "finetune_params" in model_config:
        #     print(f"{prefix}  Finetuning: {model_config['finetune_params']}")
        #     # TODO: Implémenter grid search / random search

        # try:
        #     # Entraînement
        #     model.fit(X_train, y_train)

        #     # Prédictions
        #     if len(X_test) > 0:
        #         y_pred = model.predict(X_test)

        #         # Sauvegarde les prédictions avec le contexte actuel
        #         data.add_prediction(
        #             model=model.__class__.__name__,
        #             fold=self.context.current_fold or 0,
        #             seed=self.context.current_seed or 42,
        #             preds=y_pred,
        #             set="test",
        #             **filters
        #         )

        #         print(f"{prefix}  Model trained and predictions saved")
        #     else:
        #         print(f"{prefix}  Warning: No test data found")

        # except Exception as e:
        #     print(f"{prefix}  Error training model: {e}")

    # def _hash_transformation(self, transformer, previous_hash: str) -> str:
    #     """Crée un hash unique pour une transformation."""
    #     transformer_str = str(transformer)
    #     combined = f"{previous_hash}_{transformer_str}"
    #     return hashlib.md5(combined.encode()).hexdigest()[:8]