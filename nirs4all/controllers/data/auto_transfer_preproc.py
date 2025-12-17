"""
Auto Transfer Preprocessing Controller.

This module provides the AutoTransferPreprocessingController which automatically
selects optimal preprocessing for transfer learning scenarios. It uses the
TransferPreprocessingSelector to analyze source and target data and select
preprocessing that minimizes distributional distance while preserving signal.

Usage in pipeline:
    # Standalone operator
    pipeline = [
        {"auto_transfer_preproc": {"preset": "balanced"}},
        "PLSRegressor",
    ]

    # With explicit configuration
    pipeline = [
        {
            "auto_transfer_preproc": {
                "preset": "thorough",
                "source_partition": "train",
                "target_partition": "test",
                "apply_recommendation": True,
            }
        },
        {"model": "PLSRegressor"},
    ]
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from copy import deepcopy

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class AutoTransferPreprocessingController(OperatorController):
    """
    Controller for automatic transfer-optimized preprocessing selection.

    This controller analyzes the distributional distance between source and
    target datasets and automatically selects preprocessing that best aligns
    them while preserving predictive information.

    Configuration options:
        preset: Preset configuration for the selector.
            - "fast" (default): Quick evaluation of single preprocessings only
            - "balanced": Includes stacking evaluation
            - "thorough": Includes stacking and augmentation
            - "full": All stages including supervised validation
            - "exhaustive": Deep analysis for research/benchmarking

        source_partition: Partition to use as source data ("train" or "test").
            Default is "train".

        target_partition: Partition to use as target data ("train" or "test").
            Default is "test".

        apply_recommendation: Whether to apply the best preprocessing to the
            dataset. If False, only stores the recommendation in context.
            Default is True.

        top_k: Number of top recommendations to apply if using augmentation.
            Default is 1 (best single preprocessing).

        use_augmentation: If top_k > 1, whether to use feature augmentation
            to concatenate outputs. Default is False.

        n_components: Number of PCA components for metric computation.
            Default is 10.

        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            Default is 1.

        # Stage-specific options (override preset)
        run_stage2: Enable stacking evaluation.
        stage2_top_k: Number of top candidates for stacking.
        run_stage3: Enable augmentation evaluation.
        run_stage4: Enable supervised validation.

    Example pipeline configurations:
        # Simple - use defaults
        {"auto_transfer_preproc": {}}

        # With preset
        {"auto_transfer_preproc": {"preset": "balanced"}}

        # Full configuration
        {
            "auto_transfer_preproc": {
                "preset": "thorough",
                "source_partition": "train",
                "target_partition": "test",
                "apply_recommendation": True,
                "top_k": 1,
                "verbose": 2,
            }
        }

        # Multi-source with augmentation
        {
            "auto_transfer_preproc": {
                "preset": "balanced",
                "top_k": 3,
                "use_augmentation": True,
            }
        }
    """

    priority = 9  # Higher priority than feature_augmentation (10)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if step is an auto_transfer_preproc operation."""
        return keyword == "auto_transfer_preproc"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """
        Supports prediction mode for applying saved recommendations.

        In prediction mode, the controller loads the previously computed
        preprocessing recommendation and applies it to the new data.
        """
        return True

    def execute(
        self,
        step_info: "ParsedStep",
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None,
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Execute auto transfer preprocessing selection.

        In train mode:
            1. Extract source and target data from the dataset
            2. Run TransferPreprocessingSelector to find best preprocessing
            3. Apply the recommended preprocessing if configured
            4. Store the recommendation as an artifact

        In predict mode:
            1. Load the saved preprocessing recommendation
            2. Apply it to the incoming data

        Args:
            step_info: Parsed step containing the auto_transfer_preproc config
            dataset: SpectroDataset to operate on
            context: Execution context with selector and metadata
            runtime_context: Runtime infrastructure (saver, step_number, etc.)
            source: Source index (-1 for all sources)
            mode: Execution mode ("train", "predict", "explain")
            loaded_binaries: Pre-loaded artifacts for predict/explain mode
            prediction_store: Not used by this controller

        Returns:
            Tuple of (updated_context, list_of_artifacts)
        """
        config = self._parse_config(step_info.original_step.get("auto_transfer_preproc", {}))

        if mode in ["predict", "explain"]:
            return self._execute_predict_mode(
                config, dataset, context, runtime_context, source, loaded_binaries
            )

        # Train mode: run transfer selection
        return self._execute_train_mode(
            config, dataset, context, runtime_context, source
        )

    def _parse_config(self, config: Any) -> Dict[str, Any]:
        """
        Parse and normalize the auto_transfer_preproc configuration.

        Args:
            config: Configuration from the pipeline step (dict, None, or empty).

        Returns:
            Normalized configuration dictionary with defaults.
        """
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            config = {}

        defaults = {
            "preset": "fast",
            "source_partition": "train",
            "target_partition": "test",
            "apply_recommendation": True,
            "top_k": 1,
            "use_augmentation": False,
            "n_components": 10,
            "verbose": 1,
            # Stage overrides (None means use preset defaults)
            "run_stage2": None,
            "stage2_top_k": None,
            "stage2_max_depth": None,
            "run_stage3": None,
            "stage3_top_k": None,
            "run_stage4": None,
            "stage4_top_k": None,
            # Metric weights (None means use defaults)
            "metric_weights": None,
            # Generator spec (optional)
            "preprocessing_spec": None,
        }

        # Merge with defaults
        result = {**defaults, **config}
        return result

    def _execute_train_mode(
        self,
        config: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Execute in train mode: run selection and apply recommendation.

        Args:
            config: Parsed configuration.
            dataset: SpectroDataset to operate on.
            context: Execution context.
            runtime_context: Runtime infrastructure.
            source: Source index.

        Returns:
            Tuple of (updated_context, artifacts).
        """
        from nirs4all.analysis import TransferPreprocessingSelector

        verbose = config["verbose"]
        artifacts = []

        # Extract source and target data
        X_source, y_source = self._extract_partition_data(
            dataset, context, config["source_partition"], source
        )
        X_target, y_target = self._extract_partition_data(
            dataset, context, config["target_partition"], source
        )

        if verbose >= 1:
            logger.info("Auto Transfer Preprocessing Selection")
            logger.info(f"    Source: {X_source.shape[0]} samples from '{config['source_partition']}' partition")
            logger.info(f"    Target: {X_target.shape[0]} samples from '{config['target_partition']}' partition")

        # Build selector kwargs from config
        selector_kwargs = self._build_selector_kwargs(config)

        # Run transfer preprocessing selection
        selector = TransferPreprocessingSelector(**selector_kwargs)
        results = selector.fit(X_source, X_target, y_source, y_target)

        # Get recommendation
        top_k = config["top_k"]
        use_augmentation = config["use_augmentation"]
        pipeline_spec = results.to_pipeline_spec(
            top_k=top_k,
            use_augmentation=use_augmentation,
        )

        if verbose >= 1:
            best = results.best
            logger.success(f"Best recommendation: {best.name}")
            logger.info(f"    Transfer score: {best.transfer_score:.4f}")
            logger.info(f"    Improvement: {best.improvement_pct:.1f}%")
            if top_k > 1:
                logger.info(f"    Pipeline spec (top {top_k}): {pipeline_spec}")

        # Store recommendation as artifact
        recommendation_data = {
            "pipeline_spec": pipeline_spec,
            "best_name": results.best.name,
            "transfer_score": results.best.transfer_score,
            "improvement_pct": results.best.improvement_pct,
            "top_k": top_k,
            "use_augmentation": use_augmentation,
            "ranking": [r.to_dict() for r in results.top_k(min(5, len(results.ranking)))],
        }

        if runtime_context.saver is not None:
            artifact = runtime_context.saver.persist_artifact(
                step_number=runtime_context.step_number,
                name="transfer_preprocessing_recommendation",
                obj=recommendation_data,
                format_hint="json",
                branch_id=context.selector.branch_id,
                branch_name=context.selector.branch_name,
            )
            artifacts.append(artifact)

        # Store full results in context metadata for later use
        context = context.with_metadata(
            transfer_preprocessing_results=results,
            transfer_preprocessing_recommendation=recommendation_data,
        )

        # Apply recommendation if configured
        if config["apply_recommendation"]:
            context, apply_artifacts = self._apply_recommendation(
                pipeline_spec, dataset, context, runtime_context, source
            )
            artifacts.extend(apply_artifacts)

        return context, artifacts

    def _execute_predict_mode(
        self,
        config: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Execute in predict mode: load and apply saved recommendation.

        Args:
            config: Parsed configuration.
            dataset: SpectroDataset to operate on.
            context: Execution context.
            runtime_context: Runtime infrastructure.
            source: Source index.
            loaded_binaries: Pre-loaded artifacts (deprecated, use artifact_provider).

        Returns:
            Tuple of (updated_context, artifacts).
        """
        verbose = config["verbose"]

        recommendation_data = None

        # V3: Try artifact_provider first
        if runtime_context.artifact_provider is not None:
            step_index = runtime_context.step_number
            step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                step_index,
                branch_path=context.selector.branch_path
            )
            if step_artifacts:
                artifacts_dict = dict(step_artifacts)
                recommendation_data = artifacts_dict.get("transfer_preprocessing_recommendation")

        if recommendation_data is None:
            raise ValueError(
                "transfer_preprocessing_recommendation not found. "
                "Ensure the model was trained with auto_transfer_preproc."
            )

        pipeline_spec = recommendation_data["pipeline_spec"]

        if verbose >= 1:
            logger.info("Loading saved transfer preprocessing recommendation")
            logger.info(f"    Best: {recommendation_data['best_name']}")
            logger.info(f"    Pipeline spec: {pipeline_spec}")

        # Apply recommendation
        if config["apply_recommendation"]:
            context, artifacts = self._apply_recommendation(
                pipeline_spec, dataset, context, runtime_context, source,
                mode="predict"
            )
        else:
            artifacts = []

        return context, artifacts

    def _build_selector_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build kwargs for TransferPreprocessingSelector from config.

        Args:
            config: Parsed configuration.

        Returns:
            Dictionary of kwargs for the selector.
        """
        kwargs = {
            "preset": config["preset"],
            "n_components": config["n_components"],
            "verbose": config["verbose"],
        }

        # Add stage overrides if specified
        stage_params = [
            "run_stage2", "stage2_top_k", "stage2_max_depth",
            "run_stage3", "stage3_top_k",
            "run_stage4", "stage4_top_k",
        ]

        for param in stage_params:
            if config.get(param) is not None:
                kwargs[param] = config[param]

        # Add metric weights if specified
        if config.get("metric_weights") is not None:
            kwargs["metric_weights"] = config["metric_weights"]

        # Add generator spec if specified
        if config.get("preprocessing_spec") is not None:
            kwargs["preprocessing_spec"] = config["preprocessing_spec"]

        return kwargs

    def _extract_partition_data(
        self,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        partition: str,
        source: int = -1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract X and y data from a specific partition.

        Args:
            dataset: SpectroDataset to extract from.
            context: Execution context.
            partition: Partition name ("train" or "test").
            source: Source index (-1 for first/all sources).

        Returns:
            Tuple of (X, y) where y may be None.
        """
        # Create partition-specific selector
        partition_context = context.with_partition(partition)
        selector = partition_context.selector

        # Get X data (2D array for the specified source)
        X = dataset.x(selector, layout="2d", concat_source=True)
        if isinstance(X, list) and len(X) > 0:
            # If multiple sources, use first or specified source
            src_idx = 0 if source < 0 else source
            if src_idx < len(X):
                X = X[src_idx]
            else:
                X = X[0]

        # Ensure 2D
        X = np.atleast_2d(X)
        if X.ndim == 3:
            # Flatten processings if 3D: (samples, processings, features) -> (samples, processings*features)
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)

        # Get y data
        y = dataset.y(selector)
        if y is not None:
            y = np.asarray(y)
            if y.ndim > 1:
                y = y.flatten()

        return X, y

    def _apply_recommendation(
        self,
        pipeline_spec: Any,
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train"
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Apply the recommended preprocessing to the dataset.

        Delegates to the appropriate controller based on the recommendation type:
        - Single preprocessing (string): Apply via preprocessing step
        - List of preprocessings: Apply each sequentially
        - Augmentation dict: Apply via feature_augmentation

        Args:
            pipeline_spec: The preprocessing specification to apply.
            dataset: SpectroDataset to operate on.
            context: Execution context.
            runtime_context: Runtime infrastructure.
            source: Source index.
            mode: Execution mode.

        Returns:
            Tuple of (updated_context, artifacts).
        """
        from nirs4all.analysis import get_base_preprocessings

        artifacts = []
        verbose = getattr(context.metadata, "verbose", 1) if hasattr(context.metadata, "verbose") else 1

        # Get preprocessing transforms
        preprocessings = get_base_preprocessings()

        if isinstance(pipeline_spec, str):
            # Single preprocessing or stacked (e.g., "snv" or "snv>d1")
            if verbose >= 1:
                logger.info(f"    Applying preprocessing: {pipeline_spec}")

            context, apply_artifacts = self._apply_stacked_preprocessing(
                pipeline_spec, preprocessings, dataset, context, runtime_context, source, mode
            )
            artifacts.extend(apply_artifacts)

        elif isinstance(pipeline_spec, list):
            # List of preprocessings to apply sequentially
            for pp_name in pipeline_spec:
                if verbose >= 2:
                    logger.debug(f"    Applying preprocessing: {pp_name}")

                context, apply_artifacts = self._apply_stacked_preprocessing(
                    pp_name, preprocessings, dataset, context, runtime_context, source, mode
                )
                artifacts.extend(apply_artifacts)

        elif isinstance(pipeline_spec, dict) and "feature_augmentation" in pipeline_spec:
            # Feature augmentation - delegate to feature_augmentation controller
            if verbose >= 1:
                pp_list = pipeline_spec["feature_augmentation"]
                logger.info(f"    Applying feature augmentation: {pp_list}")

            context, apply_artifacts = self._apply_feature_augmentation(
                pipeline_spec["feature_augmentation"],
                preprocessings,
                dataset,
                context,
                runtime_context,
                source,
                mode,
            )
            artifacts.extend(apply_artifacts)

        return context, artifacts

    def _apply_stacked_preprocessing(
        self,
        pp_name: str,
        preprocessings: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Apply a stacked preprocessing (e.g., "snv>d1") to the dataset.

        Args:
            pp_name: Preprocessing name (may include ">" for stacking).
            preprocessings: Dictionary of available transforms.
            dataset: SpectroDataset to operate on.
            context: Execution context.
            runtime_context: Runtime infrastructure.
            source: Source index.
            mode: Execution mode.

        Returns:
            Tuple of (updated_context, artifacts).
        """
        artifacts = []
        components = pp_name.split(">")

        # Get all source indices to process
        n_sources = dataset.features_sources()
        source_indices = [source] if source >= 0 else list(range(n_sources))

        for sd_idx in source_indices:
            # Get current processing names
            processing_ids = list(dataset.features_processings(sd_idx))

            # Get data for this source
            train_context = context.with_partition("train")
            train_data = dataset.x(train_context.selector, "3d", concat_source=False)
            all_data = dataset.x(context.selector, "3d", concat_source=False)

            if isinstance(train_data, list):
                train_data = train_data[sd_idx]
            if isinstance(all_data, list):
                all_data = all_data[sd_idx]

            # Process each current processing
            for proc_idx, proc_name in enumerate(processing_ids):
                # Extract 2D slice for this processing
                train_2d = train_data[:, proc_idx, :]
                all_2d = all_data[:, proc_idx, :]

                # Apply stacked preprocessing
                current_train = train_2d
                current_all = all_2d

                for comp_idx, comp_name in enumerate(components):
                    if comp_name not in preprocessings:
                        raise ValueError(f"Unknown preprocessing: {comp_name}")

                    transform = deepcopy(preprocessings[comp_name])
                    transform.fit(current_train)

                    current_train = transform.transform(current_train)
                    current_all = transform.transform(current_all)

                    # Save artifact in train mode
                    if mode == "train" and runtime_context.saver is not None:
                        binary_key = f"transfer_pp_{sd_idx}_{proc_name}_{comp_idx}_{comp_name}"
                        artifact = runtime_context.saver.persist_artifact(
                            step_number=runtime_context.step_number,
                            name=binary_key,
                            obj=transform,
                            format_hint="sklearn",
                            branch_id=context.selector.branch_id,
                            branch_name=context.selector.branch_name,
                        )
                        artifacts.append(artifact)

                # Update dataset with transformed features
                new_proc_name = f"{proc_name}_{pp_name.replace('>', '_')}"
                dataset.replace_features(
                    source_processings=[proc_name],
                    features=[current_all],
                    processings=[new_proc_name],
                    source=sd_idx,
                )

        # Update context with new processing names
        new_processing = []
        for sd_idx in range(dataset.features_sources()):
            src_processing = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processing)
        context = context.with_processing(new_processing)

        return context, artifacts

    def _apply_feature_augmentation(
        self,
        pp_list: List[str],
        preprocessings: Dict[str, Any],
        dataset: "SpectroDataset",
        context: "ExecutionContext",
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
    ) -> Tuple["ExecutionContext", List[Tuple[str, Any]]]:
        """
        Apply feature augmentation (concatenate multiple preprocessing outputs).

        This creates new feature processings by applying each preprocessing
        and concatenating the results horizontally.

        Args:
            pp_list: List of preprocessing names to apply and concatenate.
            preprocessings: Dictionary of available transforms.
            dataset: SpectroDataset to operate on.
            context: Execution context.
            runtime_context: Runtime infrastructure.
            source: Source index.
            mode: Execution mode.

        Returns:
            Tuple of (updated_context, artifacts).
        """
        artifacts = []

        # Set add_feature mode in context
        context = context.with_metadata(add_feature=True)

        # Get all source indices to process
        n_sources = dataset.features_sources()
        source_indices = [source] if source >= 0 else list(range(n_sources))

        for sd_idx in source_indices:
            # Get data for this source
            train_context = context.with_partition("train")
            train_data = dataset.x(train_context.selector, "3d", concat_source=False)
            all_data = dataset.x(context.selector, "3d", concat_source=False)

            if isinstance(train_data, list):
                train_data = train_data[sd_idx]
            if isinstance(all_data, list):
                all_data = all_data[sd_idx]

            # Get base 2D data (first processing)
            base_train_2d = train_data[:, 0, :]
            base_all_2d = all_data[:, 0, :]

            # Apply each preprocessing and add as new feature processing
            for pp_name in pp_list:
                components = pp_name.split(">")

                current_train = base_train_2d
                current_all = base_all_2d

                for comp_idx, comp_name in enumerate(components):
                    if comp_name not in preprocessings:
                        raise ValueError(f"Unknown preprocessing: {comp_name}")

                    transform = deepcopy(preprocessings[comp_name])
                    transform.fit(current_train)

                    current_train = transform.transform(current_train)
                    current_all = transform.transform(current_all)

                    # Save artifact in train mode
                    if mode == "train" and runtime_context.saver is not None:
                        binary_key = f"transfer_aug_{sd_idx}_{pp_name}_{comp_idx}_{comp_name}"
                        artifact = runtime_context.saver.persist_artifact(
                            step_number=runtime_context.step_number,
                            name=binary_key,
                            obj=transform,
                            format_hint="sklearn",
                            branch_id=context.selector.branch_id,
                            branch_name=context.selector.branch_name,
                        )
                        artifacts.append(artifact)

                # Add as new processing
                new_proc_name = pp_name.replace(">", "_")
                dataset.update_features(
                    source_processings=[""],  # Empty string means add new
                    features=[current_all],
                    processings=[new_proc_name],
                    source=sd_idx,
                )

        # Update context with new processing names
        new_processing = []
        for sd_idx in range(dataset.features_sources()):
            src_processing = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processing)
        context = context.with_processing(new_processing)

        return context, artifacts
