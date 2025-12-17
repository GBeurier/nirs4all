"""
Transfer Preprocessing Selector.

Main class for transfer-optimized preprocessing selection. Evaluates
preprocessings and their combinations to find those that best align
source and target datasets while preserving predictive information.

Supports two modes for preprocessing generation:
1. Combinatoric mode (default): Uses simple permutations from base preprocessings
2. Generator mode: Uses nirs4all's generator DSL for flexible, constraint-based specification
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nirs4all.analysis.presets import PRESETS
from nirs4all.analysis.results import TransferResult, TransferSelectionResults
from nirs4all.analysis.transfer_metrics import (
    TransferMetrics,
    TransferMetricsComputer,
    compute_transfer_score,
)
from nirs4all.analysis.transfer_utils import (
    apply_augmentation,
    apply_pipeline,
    apply_preprocessing_objects,
    apply_stacked_pipeline,
    generate_augmentation_combinations,
    generate_top_k_stacked_pipelines,
    get_base_preprocessings,
    get_transform_name,
    get_transform_signature,
    normalize_preprocessing,
    validate_datasets,
)
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class TransferPreprocessingSelector:
    """
    Select preprocessing for optimal transfer between datasets.

    This class evaluates preprocessing methods to find those that best
    minimize distributional distance between source and target datasets
    while preserving predictive information.

    Supports two modes for preprocessing generation:

    1. **Combinatoric mode** (default): Uses simple permutations from base
       preprocessings. Stage 1 evaluates all singles, Stage 2 generates
       permutations from top-K candidates.

    2. **Generator mode**: Uses nirs4all's generator DSL for flexible,
       constraint-based specification. Enable by providing `preprocessing_spec`.

    Stages:
        1. Single Preprocessing (required): Evaluate all base preprocessings
        1b. Generator Stacked (optional): If using generator with stacked specs
        2. Stacking (optional): Evaluate depth-2+ combinations of top-K
        2b. Generator Augmented (optional): If using generator with augmentation specs
        3. Augmentation (optional): Evaluate feature concatenation
        4. Validation (optional): Supervised validation with proxy models

    Args:
        preset: Preset configuration ('fast', 'balanced', 'thorough', 'full',
            'exhaustive') or None for manual configuration. Default is 'fast'.
        preprocessings: Custom preprocessings dict or None for base set.
        n_components: PCA components for metric computation.
        k_neighbors: Neighbors for trustworthiness metric.

        Stage 2 (stacking):
            run_stage2: Enable stacking evaluation.
            stage2_top_k: Number of top candidates for stacking.
            stage2_max_depth: Maximum stacking depth.

        Stage 3 (augmentation):
            run_stage3: Enable augmentation evaluation.
            stage3_top_k: Number of top candidates for augmentation.
            stage3_max_order: Maximum augmentation order (2 or 3).

        Stage 4 (validation):
            run_stage4: Enable supervised validation.
            stage4_top_k: Number of candidates to validate.
            stage4_cv_folds: Cross-validation folds.

        Generator integration:
            preprocessing_spec: Generator specification dict for flexible
                preprocessing definition. Uses nirs4all.pipeline.config.generator.
                Supports keywords like `_or_`, `arrange`, `pick`, `_mutex_`, etc.
            use_generator: Enable generator mode. Auto-detected if preprocessing_spec
                is provided. Set to False to disable even with preprocessing_spec.

        Parallelization:
            n_jobs: Number of parallel jobs for preprocessing evaluation.
                - n_jobs=-1: Use all available CPU cores (default)
                - n_jobs=1: Sequential execution (useful for debugging)
                - n_jobs=N: Use N cores

        Other:
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            random_state: Random seed for reproducibility.

    Example:
        >>> # Quick usage with default fast preset
        >>> selector = TransferPreprocessingSelector()
        >>> results = selector.fit(X_source, X_target)
        >>> print(results.best.name)
        'snv'

        >>> # With balanced preset for stacking
        >>> selector = TransferPreprocessingSelector(preset='balanced')
        >>> results = selector.fit(X_source, X_target)
        >>> print(results.to_pipeline_spec())
        'snv>d1'

        >>> # Generator mode: constrained stacking
        >>> selector = TransferPreprocessingSelector(
        ...     preprocessing_spec={
        ...         "_or_": ["snv", "msc", "d1", "d2", "savgol"],
        ...         "arrange": 2,
        ...         "_mutex_": [["d1", "d2"]],  # Don't stack derivatives
        ...     },
        ... )
        >>> results = selector.fit(X_source, X_target)

        >>> # Custom configuration
        >>> selector = TransferPreprocessingSelector(
        ...     preset=None,
        ...     run_stage2=True,
        ...     stage2_top_k=10,
        ...     stage2_max_depth=2,
        ...     n_components=20,
        ... )
    """

    def __init__(
        self,
        preset: Optional[str] = "fast",
        preprocessings: Optional[Dict[str, Any]] = None,
        n_components: int = 10,
        k_neighbors: int = 10,
        # Stage 2
        run_stage2: bool = False,
        stage2_top_k: Optional[int] = 5,
        stage2_max_depth: int = 2,
        stage2_exhaustive: bool = False,
        # Stage 3
        run_stage3: bool = False,
        stage3_top_k: int = 5,
        stage3_max_order: int = 2,
        # Stage 4
        run_stage4: bool = False,
        stage4_top_k: int = 10,
        stage4_cv_folds: int = 3,
        stage4_models: Optional[List[str]] = None,
        # Metric weights
        metric_weights: Optional[Dict[str, float]] = None,
        # Generator integration
        preprocessing_spec: Optional[Dict[str, Any]] = None,
        use_generator: Optional[bool] = None,
        # Parallelization
        n_jobs: int = -1,
        # Other
        verbose: int = 1,
        random_state: int = 0,
    ):
        # Apply preset if specified
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}"
                )
            preset_config = PRESETS[preset]
            # Override with preset values
            n_components = preset_config.get("n_components", n_components)
            run_stage2 = preset_config.get("run_stage2", run_stage2)
            stage2_top_k = preset_config.get("stage2_top_k", stage2_top_k)
            stage2_max_depth = preset_config.get("stage2_max_depth", stage2_max_depth)
            stage2_exhaustive = preset_config.get("stage2_exhaustive", stage2_exhaustive)
            run_stage3 = preset_config.get("run_stage3", run_stage3)
            stage3_top_k = preset_config.get("stage3_top_k", stage3_top_k)
            stage3_max_order = preset_config.get("stage3_max_order", stage3_max_order)
            run_stage4 = preset_config.get("run_stage4", run_stage4)
            stage4_top_k = preset_config.get("stage4_top_k", stage4_top_k)
            stage4_cv_folds = preset_config.get("stage4_cv_folds", stage4_cv_folds)

        self.preset = preset
        self.preprocessings = preprocessings or get_base_preprocessings()
        self.n_components = n_components
        self.k_neighbors = k_neighbors

        # Stage configs
        self.run_stage2 = run_stage2
        self.stage2_top_k = stage2_top_k
        self.stage2_max_depth = stage2_max_depth
        self.stage2_exhaustive = stage2_exhaustive

        self.run_stage3 = run_stage3
        self.stage3_top_k = stage3_top_k
        self.stage3_max_order = stage3_max_order

        self.run_stage4 = run_stage4
        self.stage4_top_k = stage4_top_k
        self.stage4_cv_folds = stage4_cv_folds
        self.stage4_models = stage4_models or ["ridge", "pls"]

        self.metric_weights = metric_weights
        self.verbose = verbose
        self.random_state = random_state

        # Parallelization
        self.n_jobs = n_jobs
        self._effective_n_jobs = self._compute_effective_n_jobs(n_jobs)

        # Generator integration
        self.preprocessing_spec = preprocessing_spec
        # Auto-detect generator mode if preprocessing_spec is provided
        self.use_generator = (
            use_generator if use_generator is not None
            else (preprocessing_spec is not None)
        )

        # Expanded preprocessing lists (populated when use_generator=True)
        # Now stores objects/transforms instead of just names
        self._expanded_singles: Optional[List[Any]] = None
        self._expanded_stacked: Optional[List[List[Any]]] = None
        self._expanded_augmented: Optional[List[List[Any]]] = None

        # Initialize metrics computer
        self.metrics_computer = TransferMetricsComputer(
            n_components=n_components,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

        # Results storage
        self.results_: Optional[TransferSelectionResults] = None
        self.raw_metrics_: Optional[TransferMetrics] = None

    def _log(self, msg: str, level: int = 1) -> None:
        """Log message if verbosity level is sufficient."""
        if self.verbose >= level:
            if level >= 2:
                logger.debug(msg)
            else:
                logger.info(msg)

    @staticmethod
    def _compute_effective_n_jobs(n_jobs: int) -> int:
        """
        Compute effective number of parallel jobs.

        Args:
            n_jobs: User-specified n_jobs value.
                -1 means use all CPUs, positive integer specifies count.

        Returns:
            Effective number of workers to use.
        """
        import os
        if n_jobs == -1:
            return os.cpu_count() or 1
        elif n_jobs < 1:
            return 1
        else:
            return n_jobs

    def _parallel_evaluate_preprocessings(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        pp_items: List[Tuple[str, Any]],
        pipeline_type: str = "single",
    ) -> List[TransferResult]:
        """
        Evaluate preprocessings in parallel using ThreadPoolExecutor.

        Uses ThreadPoolExecutor instead of ProcessPoolExecutor because:
        1. NumPy/scipy operations release the GIL, enabling true parallelism
        2. No need to serialize/pickle sklearn transformers
        3. Shared memory avoids copying large datasets to workers

        Supports both object-based and string-based preprocessings:
        - Objects: Applied directly via apply_preprocessing_objects
        - Strings: Resolved via preprocessings dict (legacy support)

        Args:
            X_source: Source dataset.
            X_target: Target dataset.
            pp_items: List of (name, transform_or_transforms) tuples.
                For object-based: (display_name, transform_object_or_list)
                For string-based: (name, name_or_list_of_names)
            pipeline_type: Type of pipeline ('single', 'stacked', 'augmented').

        Returns:
            List of TransferResult for each preprocessing.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from typing import Optional as Opt

        def evaluate_single(args: Tuple[str, Any]) -> Opt[TransferResult]:
            """Worker function to evaluate a single preprocessing."""
            pp_name, pp_item = args
            try:
                # Track the actual transforms used for storing in result
                actual_transforms: List[Any] = []

                # Determine how to apply the preprocessing based on item type
                if pipeline_type == "augmented":
                    # pp_item is list of components (objects or strings)
                    components = pp_item if isinstance(pp_item, list) else [pp_item]
                    X_src_pp = apply_augmentation(
                        X_source, components, self.preprocessings
                    )
                    X_tgt_pp = apply_augmentation(
                        X_target, components, self.preprocessings
                    )
                    # Component names for result
                    comp_names = [
                        get_transform_name(c) if not isinstance(c, str) else c
                        for c in components
                    ]
                    # Store actual transforms
                    actual_transforms = [
                        c if not isinstance(c, str) else self.preprocessings.get(c)
                        for c in components
                    ]
                elif pipeline_type == "stacked":
                    # Check if pp_item is object-based (list of transforms) or string
                    if isinstance(pp_item, list) and len(pp_item) > 0 and not isinstance(pp_item[0], str):
                        # Object-based stacked pipeline
                        X_src_pp = apply_preprocessing_objects(X_source, pp_item)
                        X_tgt_pp = apply_preprocessing_objects(X_target, pp_item)
                        comp_names = [get_transform_name(t) for t in pp_item]
                        actual_transforms = list(pp_item)
                    elif isinstance(pp_item, str) and ">" in pp_item:
                        # String-based stacked pipeline
                        X_src_pp = apply_stacked_pipeline(
                            X_source, pp_item, self.preprocessings
                        )
                        X_tgt_pp = apply_stacked_pipeline(
                            X_target, pp_item, self.preprocessings
                        )
                        comp_names = pp_item.split(">")
                        actual_transforms = [self.preprocessings[n] for n in comp_names]
                    else:
                        # Single transform passed as stacked
                        transforms = [pp_item] if not isinstance(pp_item, list) else pp_item
                        X_src_pp = apply_preprocessing_objects(X_source, transforms)
                        X_tgt_pp = apply_preprocessing_objects(X_target, transforms)
                        comp_names = [get_transform_name(t) for t in transforms]
                        actual_transforms = list(transforms)
                else:
                    # Single preprocessing
                    if isinstance(pp_item, str):
                        # String name - resolve from dict
                        transforms = [self.preprocessings[pp_item]]
                        actual_transforms = transforms
                    elif isinstance(pp_item, list):
                        transforms = pp_item
                        actual_transforms = transforms
                    else:
                        transforms = [pp_item]
                        actual_transforms = transforms
                    X_src_pp = apply_pipeline(X_source, transforms)
                    X_tgt_pp = apply_pipeline(X_target, transforms)
                    comp_names = [pp_name]

                # Compute transfer metrics
                metrics = self.metrics_computer.compute(
                    X_src_pp, X_tgt_pp, compute_trust=False
                )

                # Compute transfer score
                score = compute_transfer_score(
                    metrics,
                    raw_metrics=self.raw_metrics_,
                    weights=self.metric_weights,
                )

                # Compute improvement percentage
                raw_centroid = self.raw_metrics_.centroid_distance
                improvement_pct = (
                    (raw_centroid - metrics.centroid_distance)
                    / (raw_centroid + 1e-10)
                ) * 100

                return TransferResult(
                    name=pp_name,
                    pipeline_type=pipeline_type,
                    components=comp_names,
                    transfer_score=score,
                    metrics=metrics.to_dict(),
                    improvement_pct=improvement_pct,
                    transforms=actual_transforms if actual_transforms else None,
                )

            except Exception as e:
                self._log(f"  Warning: Failed to evaluate {pp_name}: {e}", level=2)
                return None

        results: List[TransferResult] = []

        # Use sequential execution if n_jobs=1
        if self._effective_n_jobs == 1:
            for pp_item in pp_items:
                self._log(f"  Evaluating: {pp_item[0]}", level=2)
                result = evaluate_single(pp_item)
                if result is not None:
                    results.append(result)
        else:
            # Parallel execution with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self._effective_n_jobs) as executor:
                futures = {
                    executor.submit(evaluate_single, pp_item): pp_item[0]
                    for pp_item in pp_items
                }
                for future in as_completed(futures):
                    pp_name = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        self._log(
                            f"  Warning: Failed to evaluate {pp_name}: {e}",
                            level=2
                        )

        return results

    def _expand_preprocessing_spec(
        self,
    ) -> Tuple[List[Any], List[List[Any]], List[List[Any]]]:
        """
        Expand preprocessing_spec using the nirs4all generator.

        Handles both object-based and string-based preprocessing specs.
        Objects are used directly; strings are resolved via the preprocessings dict.

        Returns:
            Tuple of:
            - single_preprocessings: List of single transforms [SNV(), D1()]
            - stacked_pipelines: List of stacked transforms [[SNV(), D1()], ...]
            - augmented_combinations: List of augmentation combos [[SNV()], [D1()], ...]
        """
        from nirs4all.pipeline.config.generator import (
            ARRANGE_KEYWORD,
            PICK_KEYWORD,
            expand_spec,
        )

        if self.preprocessing_spec is None:
            return [], [], []

        expanded = expand_spec(self.preprocessing_spec, seed=self.random_state)

        # Determine if the spec used pick (augmentation) or arrange (stacking)
        spec = self.preprocessing_spec
        uses_pick = PICK_KEYWORD in spec if isinstance(spec, dict) else False
        uses_arrange = ARRANGE_KEYWORD in spec if isinstance(spec, dict) else False

        singles: List[Any] = []
        stacked: List[List[Any]] = []
        augmented: List[List[Any]] = []

        # Helper to normalize an item (object or string) to a transform object
        def to_transform(item: Any) -> Any:
            """Convert item to transform object."""
            if item is None:
                return None
            if isinstance(item, str):
                if item == "None":
                    return None
                # Try to resolve from preprocessings dict
                return normalize_preprocessing(item, self.preprocessings)
            # Already an object
            return item

        for item in expanded:
            if isinstance(item, list):
                # Convert each element to a transform
                transforms = [to_transform(x) for x in item]
                # Filter out None values
                transforms = [t for t in transforms if t is not None]

                # Skip empty combinations (all None)
                if len(transforms) == 0:
                    continue

                if len(transforms) == 1:
                    singles.append(transforms[0])
                elif uses_pick and not uses_arrange:
                    # Augmentation: keep as separate transforms
                    augmented.append(transforms)
                else:
                    # Stacking: keep as ordered sequence
                    stacked.append(transforms)

            else:
                # Single item (could be object or string)
                transform = to_transform(item)
                if transform is not None:
                    singles.append(transform)

        # Remove duplicates while preserving order (using signature for identity)
        seen_singles: set = set()
        unique_singles: List[Any] = []
        for t in singles:
            sig = get_transform_signature(t)
            if sig not in seen_singles:
                seen_singles.add(sig)
                unique_singles.append(t)
        singles = unique_singles

        seen_stacked: set = set()
        unique_stacked: List[List[Any]] = []
        for pipeline in stacked:
            sig = ">".join(get_transform_signature(t) for t in pipeline)
            if sig not in seen_stacked:
                seen_stacked.add(sig)
                unique_stacked.append(pipeline)
        stacked = unique_stacked

        # For augmented, use order-independent key for deduplication
        seen_aug: set = set()
        unique_augmented: List[List[Any]] = []
        for combo in augmented:
            key = tuple(sorted(get_transform_signature(t) for t in combo))
            if key not in seen_aug:
                seen_aug.add(key)
                unique_augmented.append(combo)
        augmented = unique_augmented

        self._log(
            f"  Generator expanded: {len(singles)} singles, "
            f"{len(stacked)} stacked, {len(augmented)} augmented",
            level=1
        )

        return singles, stacked, augmented

    def _prepare_generator_preprocessings(self) -> None:
        """
        Prepare preprocessing lists from generator spec.

        Expands the preprocessing_spec and populates:
        - self._expanded_singles
        - self._expanded_stacked
        - self._expanded_augmented
        """
        if not self.use_generator or self.preprocessing_spec is None:
            self._expanded_singles = None
            self._expanded_stacked = None
            self._expanded_augmented = None
            return

        self._log("Expanding preprocessing specification...")
        (
            self._expanded_singles,
            self._expanded_stacked,
            self._expanded_augmented,
        ) = self._expand_preprocessing_spec()

    def fit(
        self,
        X_source_or_config,
        X_target: Optional[np.ndarray] = None,
        y_source: Optional[np.ndarray] = None,
        y_target: Optional[np.ndarray] = None,
    ) -> TransferSelectionResults:
        """
        Run transfer-optimized preprocessing selection.

        Supports two calling conventions:

        1. **Raw arrays** (original API):
            selector.fit(X_source, X_target, y_source, y_target)

        2. **DatasetConfigs** (nirs4all-native API):
            selector.fit(dataset_config)
            - Single dataset: Uses train as source, test as target
            - Multiple datasets: Combines X("all") from all datasets

        Args:
            X_source_or_config: Either:
                - np.ndarray: Source dataset (n_samples_src, n_features)
                - DatasetConfigs: nirs4all dataset configuration
            X_target: Target dataset (required if X_source_or_config is array).
            y_source: Optional source targets for supervised validation.
            y_target: Optional target labels for supervised validation.

        Returns:
            TransferSelectionResults with ranked recommendations.

        Example:
            >>> # Using DatasetConfigs (recommended nirs4all way)
            >>> selector = TransferPreprocessingSelector(preset="balanced")
            >>> results = selector.fit(DatasetConfigs(data_path))
            >>> pp_list = results.to_preprocessing_list(top_k=10)

            >>> # Using raw arrays
            >>> results = selector.fit(X_train, X_test, y_train)
        """
        # Determine calling convention
        if X_target is None:
            # DatasetConfigs mode
            X_source, X_target, y_source, y_target = self._extract_data_from_dataset_configs(
                X_source_or_config
            )
        else:
            # Raw arrays mode
            X_source = X_source_or_config

        # Validate inputs
        X_source, X_target = validate_datasets(X_source, X_target)

        timing: Dict[str, float] = {}
        all_results: List[TransferResult] = []

        # Prepare generator-based preprocessings if enabled
        if self.use_generator:
            t0 = time.time()
            self._prepare_generator_preprocessings()
            timing["generator"] = time.time() - t0

        # Compute raw baseline metrics
        self._log("Computing baseline (raw) metrics...")
        t0 = time.time()
        self.raw_metrics_ = self.metrics_computer.compute(
            X_source, X_target, compute_trust=True
        )
        raw_metrics_dict = self.raw_metrics_.to_dict()
        timing["baseline"] = time.time() - t0

        # Stage 1: Single preprocessing evaluation
        self._log("\n=== Stage 1: Single Preprocessing Evaluation ===")
        t0 = time.time()
        stage1_results = self._stage1_single_evaluation(X_source, X_target)
        timing["stage1"] = time.time() - t0
        all_results.extend(stage1_results)
        self._log(
            f"Stage 1 complete: {len(stage1_results)} preprocessings "
            f"in {timing['stage1']:.2f}s"
        )

        # Generator mode: Evaluate pre-generated stacked pipelines
        gen_stacked_results: List[TransferResult] = []
        if self.use_generator and self._expanded_stacked:
            self._log("\n=== Stage 1b: Generator-Based Stacked Evaluation ===")
            t0 = time.time()
            gen_stacked_results = self._evaluate_generator_stacked(
                X_source, X_target
            )
            timing["stage1b_gen_stacked"] = time.time() - t0
            all_results.extend(gen_stacked_results)
            self._log(
                f"Stage 1b complete: {len(gen_stacked_results)} generator-stacked "
                f"in {timing['stage1b_gen_stacked']:.2f}s"
            )

        # Stage 2: Stacking evaluation
        if self.run_stage2:
            if not (self.use_generator and self._expanded_stacked):
                # Combinatoric mode
                self._log("\n=== Stage 2: Stacking Evaluation ===")
                t0 = time.time()
                stage2_results = self._stage2_stacking_evaluation(
                    X_source, X_target, stage1_results
                )
                timing["stage2"] = time.time() - t0
                all_results.extend(stage2_results)
                self._log(
                    f"Stage 2 complete: {len(stage2_results)} stacked "
                    f"in {timing['stage2']:.2f}s"
                )
            else:
                # Generator mode with stage2 enabled: extend with top-K
                self._log("\n=== Stage 2: Extended Stacking Evaluation ===")
                t0 = time.time()
                stage2_results = self._stage2_stacking_evaluation(
                    X_source, X_target, stage1_results + gen_stacked_results
                )
                timing["stage2"] = time.time() - t0
                all_results.extend(stage2_results)
                self._log(
                    f"Stage 2 complete: {len(stage2_results)} additional stacked "
                    f"in {timing['stage2']:.2f}s"
                )

        # Generator mode: Evaluate pre-generated augmentation combinations
        if self.use_generator and self._expanded_augmented:
            self._log("\n=== Stage 2b: Generator-Based Augmentation Evaluation ===")
            t0 = time.time()
            gen_aug_results = self._evaluate_generator_augmented(
                X_source, X_target
            )
            timing["stage2b_gen_aug"] = time.time() - t0
            all_results.extend(gen_aug_results)
            self._log(
                f"Stage 2b complete: {len(gen_aug_results)} generator-augmented "
                f"in {timing['stage2b_gen_aug']:.2f}s"
            )

        # Stage 3: Augmentation evaluation
        if self.run_stage3:
            self._log("\n=== Stage 3: Augmentation Evaluation ===")
            t0 = time.time()
            stage3_results = self._stage3_augmentation_evaluation(
                X_source, X_target, all_results
            )
            timing["stage3"] = time.time() - t0
            all_results.extend(stage3_results)
            self._log(
                f"Stage 3 complete: {len(stage3_results)} augmented "
                f"in {timing['stage3']:.2f}s"
            )

        # Stage 4: Supervised validation
        if self.run_stage4 and y_source is not None:
            self._log("\n=== Stage 4: Supervised Validation ===")
            t0 = time.time()
            all_results = self._stage4_supervised_validation(
                X_source, y_source, all_results
            )
            timing["stage4"] = time.time() - t0
            self._log(f"Stage 4 complete in {timing['stage4']:.2f}s")

        # Filter out results with NaN transfer scores (invalid preprocessing results)
        valid_results = [
            r for r in all_results
            if r.transfer_score is not None and not np.isnan(r.transfer_score)
        ]
        invalid_count = len(all_results) - len(valid_results)
        if invalid_count > 0:
            self._log(
                f"  Filtered {invalid_count} results with invalid transfer scores"
            )
        all_results = valid_results

        # Sort by transfer score (higher is better)
        all_results.sort(key=lambda r: r.transfer_score, reverse=True)

        # Create results object
        self.results_ = TransferSelectionResults(
            ranking=all_results,
            raw_metrics=raw_metrics_dict,
            timing=timing,
        )

        # Print summary
        if self.verbose >= 1:
            self._log("\n" + self.results_.summary())

        return self.results_

    def _stage1_single_evaluation(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> List[TransferResult]:
        """
        Stage 1: Evaluate all single preprocessings.

        Args:
            X_source: Source dataset.
            X_target: Target dataset.

        Returns:
            List of TransferResult for each preprocessing.
        """
        # Determine which preprocessings to evaluate
        if self.use_generator and self._expanded_singles:
            # Object-based: use expanded transforms directly
            pp_items = [
                (get_transform_name(t), t)
                for t in self._expanded_singles
            ]
            self._log(
                f"  Using generator-specified preprocessings: {len(pp_items)} items"
            )
        else:
            pp_items = list(self.preprocessings.items())

        # Log parallelization info
        if self._effective_n_jobs > 1:
            self._log(f"  Parallel evaluation with {self._effective_n_jobs} workers...")

        # Use parallel evaluation helper
        return self._parallel_evaluate_preprocessings(
            X_source, X_target, pp_items, pipeline_type="single"
        )

    def _evaluate_generator_stacked(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> List[TransferResult]:
        """
        Evaluate stacked pipelines from generator specification.

        Uses parallel evaluation when n_jobs != 1 for improved performance.

        Args:
            X_source: Source dataset.
            X_target: Target dataset.

        Returns:
            List of TransferResult for each stacked pipeline.
        """
        if not self._expanded_stacked:
            return []

        self._log(
            f"  Evaluating {len(self._expanded_stacked)} generator-stacked pipelines..."
        )

        # Log parallelization info
        if self._effective_n_jobs > 1:
            self._log(f"  Parallel evaluation with {self._effective_n_jobs} workers...")

        # Prepare items for parallel evaluation: (name, transforms_list)
        # Name is derived from transform types for display
        pp_items = [
            (">".join(get_transform_name(t) for t in transforms), transforms)
            for transforms in self._expanded_stacked
        ]

        return self._parallel_evaluate_preprocessings(
            X_source, X_target, pp_items, pipeline_type="stacked"
        )

    def _evaluate_generator_augmented(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> List[TransferResult]:
        """
        Evaluate augmentation combinations from generator specification.

        Uses parallel evaluation when n_jobs != 1 for improved performance.

        Args:
            X_source: Source dataset.
            X_target: Target dataset.

        Returns:
            List of TransferResult for each augmented combination.
        """
        if not self._expanded_augmented:
            return []

        self._log(
            f"  Evaluating {len(self._expanded_augmented)} generator-augmented "
            "combinations..."
        )

        # Log parallelization info
        if self._effective_n_jobs > 1:
            self._log(f"  Parallel evaluation with {self._effective_n_jobs} workers...")

        # Prepare items for parallel evaluation: (name, transforms_list)
        pp_items = [
            ("+".join(get_transform_name(t) for t in transforms), transforms)
            for transforms in self._expanded_augmented
        ]

        return self._parallel_evaluate_preprocessings(
            X_source, X_target, pp_items, pipeline_type="augmented"
        )

    def _stage2_stacking_evaluation(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        stage1_results: List[TransferResult],
    ) -> List[TransferResult]:
        """
        Stage 2: Evaluate stacked pipeline combinations.

        Uses parallel evaluation when n_jobs != 1 for improved performance.

        Args:
            X_source: Source dataset.
            X_target: Target dataset.
            stage1_results: Results from Stage 1.

        Returns:
            List of TransferResult for stacked pipelines.
        """
        # Select top-K from Stage 1
        stage1_sorted = sorted(
            stage1_results, key=lambda r: r.transfer_score, reverse=True
        )

        if self.stage2_top_k is not None:
            top_k_results = stage1_sorted[:self.stage2_top_k]
        else:
            top_k_results = stage1_sorted

        top_k_names = [r.name for r in top_k_results]

        self._log(
            f"  Generating stacked combinations from top-{len(top_k_names)}: "
            f"{top_k_names[:5]}..."
        )

        # Generate stacked pipelines
        stacked_pipelines = generate_top_k_stacked_pipelines(
            top_k_names,
            self.preprocessings,
            max_depth=self.stage2_max_depth,
        )

        self._log(f"  Evaluating {len(stacked_pipelines)} stacked pipelines...")

        # Log parallelization info
        if self._effective_n_jobs > 1:
            self._log(f"  Parallel evaluation with {self._effective_n_jobs} workers...")

        # Prepare items for parallel evaluation: (name, name) for stacked lookup
        pp_items = [(pp_name, pp_name) for pp_name, _, _ in stacked_pipelines]

        return self._parallel_evaluate_preprocessings(
            X_source, X_target, pp_items, pipeline_type="stacked"
        )

    def _stage3_augmentation_evaluation(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        previous_results: List[TransferResult],
    ) -> List[TransferResult]:
        """
        Stage 3: Evaluate feature augmentation combinations.

        Uses parallel evaluation when n_jobs != 1 for improved performance.

        Args:
            X_source: Source dataset.
            X_target: Target dataset.
            previous_results: Results from Stage 1 and 2.

        Returns:
            List of TransferResult for augmented pipelines.
        """
        # Select top-K diverse candidates for augmentation
        sorted_results = sorted(
            previous_results, key=lambda r: r.transfer_score, reverse=True
        )
        top_k_results = sorted_results[:self.stage3_top_k]
        top_k_names = [r.name for r in top_k_results]

        self._log(
            f"  Generating augmentation combinations from top-{len(top_k_names)}..."
        )

        # Generate augmentation combinations
        aug_combinations = generate_augmentation_combinations(
            top_k_names,
            max_order=self.stage3_max_order,
        )

        self._log(f"  Evaluating {len(aug_combinations)} augmented pipelines...")

        # Log parallelization info
        if self._effective_n_jobs > 1:
            self._log(f"  Parallel evaluation with {self._effective_n_jobs} workers...")

        # Prepare items for parallel evaluation: (name, components_list)
        pp_items = [(aug_name, components) for aug_name, components in aug_combinations]

        return self._parallel_evaluate_preprocessings(
            X_source, X_target, pp_items, pipeline_type="augmented"
        )

    def _stage4_supervised_validation(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        previous_results: List[TransferResult],
    ) -> List[TransferResult]:
        """
        Stage 4: Supervised validation with proxy models.

        Args:
            X_source: Source dataset.
            y_source: Source targets.
            previous_results: Results from previous stages.

        Returns:
            List of TransferResult with signal_score populated.
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Select top-K candidates for validation
        sorted_results = sorted(
            previous_results, key=lambda r: r.transfer_score, reverse=True
        )
        top_k = min(self.stage4_top_k, len(sorted_results))
        candidates = sorted_results[:top_k]

        self._log(f"  Validating top-{top_k} candidates with proxy models...")

        # Determine task type
        n_unique = len(np.unique(y_source))
        is_classification = n_unique < 10

        validated_results = []

        for result in candidates:
            self._log(f"  Validating: {result.name}", level=2)

            try:
                # Apply preprocessing to source data
                if result.pipeline_type == "augmented":
                    X_pp = apply_augmentation(
                        X_source, result.components, self.preprocessings
                    )
                else:
                    X_pp = apply_stacked_pipeline(
                        X_source, result.name, self.preprocessings
                    )

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pp)

                # Reduce dimensionality if needed
                if X_scaled.shape[1] > X_scaled.shape[0]:
                    n_comp = min(50, X_scaled.shape[0] - 1)
                    pca = PCA(n_components=n_comp, random_state=self.random_state)
                    X_scaled = pca.fit_transform(X_scaled)

                # Evaluate with proxy models
                scores = []

                # Ridge regression
                if "ridge" in self.stage4_models:
                    try:
                        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                        cv_scores = cross_val_score(
                            ridge, X_scaled, y_source,
                            cv=self.stage4_cv_folds, scoring="r2"
                        )
                        scores.append(float(np.clip(np.mean(cv_scores), 0, 1)))
                    except Exception:
                        pass

                # PLS regression
                if "pls" in self.stage4_models:
                    try:
                        n_components = min(
                            10, X_scaled.shape[1], X_scaled.shape[0] - 1
                        )
                        pls = PLSRegression(n_components=n_components)
                        cv_scores = cross_val_score(
                            pls, X_scaled, y_source,
                            cv=self.stage4_cv_folds, scoring="r2"
                        )
                        scores.append(float(np.clip(np.mean(cv_scores), 0, 1)))
                    except Exception:
                        pass

                # KNN (optional)
                if "knn" in self.stage4_models:
                    try:
                        from sklearn.neighbors import (
                            KNeighborsClassifier,
                            KNeighborsRegressor,
                        )
                        if is_classification:
                            knn = KNeighborsClassifier(n_neighbors=5)
                            scoring = "accuracy"
                        else:
                            knn = KNeighborsRegressor(n_neighbors=5)
                            scoring = "r2"
                        cv_scores = cross_val_score(
                            knn, X_scaled, y_source,
                            cv=self.stage4_cv_folds, scoring=scoring
                        )
                        scores.append(float(np.clip(np.mean(cv_scores), 0, 1)))
                    except Exception:
                        pass

                # XGBoost (optional)
                if "xgb" in self.stage4_models:
                    try:
                        from xgboost import XGBClassifier, XGBRegressor
                        if is_classification:
                            xgb = XGBClassifier(
                                n_estimators=50, max_depth=3,
                                verbosity=0, n_jobs=1,
                                random_state=self.random_state
                            )
                            scoring = "accuracy"
                        else:
                            xgb = XGBRegressor(
                                n_estimators=50, max_depth=3,
                                verbosity=0, n_jobs=1,
                                random_state=self.random_state
                            )
                            scoring = "r2"
                        cv_scores = cross_val_score(
                            xgb, X_scaled, y_source,
                            cv=self.stage4_cv_folds, scoring=scoring
                        )
                        scores.append(float(np.clip(np.mean(cv_scores), 0, 1)))
                    except ImportError:
                        pass
                    except Exception:
                        pass

                # Compute composite signal score
                signal_score = float(np.mean(scores)) if scores else 0.0

                # Create updated result with signal score
                validated_result = TransferResult(
                    name=result.name,
                    pipeline_type=result.pipeline_type,
                    components=result.components,
                    transfer_score=result.transfer_score,
                    metrics=result.metrics,
                    improvement_pct=result.improvement_pct,
                    signal_score=signal_score,
                )
                validated_results.append(validated_result)

            except Exception as e:
                self._log(
                    f"  Warning: Failed to validate {result.name}: {e}",
                    level=2
                )
                validated_results.append(result)

        # Add remaining non-validated results
        non_validated = sorted_results[top_k:]
        validated_results.extend(non_validated)

        return validated_results

    def fit_from_configs(
        self,
        config_source,
        config_target,
        partition: str = "train",
    ) -> TransferSelectionResults:
        """
        Fit from DatasetConfigs or SpectroDataset.

        Args:
            config_source: DatasetConfigs or SpectroDataset for source dataset.
            config_target: DatasetConfigs or SpectroDataset for target dataset.
            partition: Which partition to use ('train' or 'test').

        Returns:
            TransferSelectionResults with ranked recommendations.

        Example:
            >>> from nirs4all.data.config import DatasetConfigs
            >>> config_src = DatasetConfigs("path/to/source.json")
            >>> config_tgt = DatasetConfigs("path/to/target.json")
            >>> selector = TransferPreprocessingSelector()
            >>> results = selector.fit_from_configs(config_src, config_tgt)
        """
        X_source, y_source = self._extract_data_from_config(config_source, partition)
        X_target, y_target = self._extract_data_from_config(config_target, partition)

        return self.fit(X_source, X_target, y_source, y_target)

    def _extract_data_from_config(
        self,
        config,
        partition: str = "train",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract X, y from various nirs4all data structures.

        Args:
            config: Data configuration object.
            partition: Partition to extract ('train' or 'test').

        Returns:
            Tuple of (X, y) where y may be None.
        """
        # Handle SpectroDataset (uses x() and y() methods with selector dict)
        if hasattr(config, "x") and hasattr(config, "y"):
            try:
                samples = config.x({"partition": partition})
                if samples is not None:
                    X = samples[0] if isinstance(samples, tuple) else np.asarray(samples)
                else:
                    raise ValueError("Could not extract samples from SpectroDataset")
            except Exception as e:
                raise ValueError(f"Could not extract samples from SpectroDataset: {e}")

            try:
                y = config.y({"partition": partition})
                y = np.asarray(y).ravel() if y is not None else None
            except Exception:
                y = None

            return np.asarray(X), y

        # Handle DatasetConfigs
        if hasattr(config, "get_datasets") or hasattr(config, "iter_datasets"):
            if hasattr(config, "get_dataset_at"):
                dataset = config.get_dataset_at(0)
            elif hasattr(config, "get_datasets"):
                datasets = config.get_datasets()
                dataset = datasets[0] if datasets else None
            else:
                dataset = next(config.iter_datasets())

            if dataset is None:
                raise ValueError("Could not extract dataset from DatasetConfigs")

            return self._extract_data_from_config(dataset, partition)

        # Handle dict-like config
        if isinstance(config, dict):
            X = None
            for key in ["X", "x", "samples"]:
                if key in config and config[key] is not None:
                    X = config[key]
                    break

            y = None
            for key in ["y", "Y", "targets"]:
                if key in config and config[key] is not None:
                    y = config[key]
                    break

            if X is None:
                raise ValueError("Dict config must have 'X' or 'samples' key")

            return np.asarray(X), np.asarray(y) if y is not None else None

        # Handle raw numpy array
        if isinstance(config, np.ndarray):
            return config, None

        raise ValueError(
            f"Unsupported config type: {type(config)}. "
            "Expected DatasetConfigs, SpectroDataset, dict, or numpy array."
        )

    def _extract_data_from_dataset_configs(
        self,
        config,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract X_source, X_target, y_source, y_target from DatasetConfigs.

        Handles two cases:
        1. Single dataset: Uses train partition as source, test as target
        2. Multiple datasets: Combines all samples from each dataset

        Args:
            config: DatasetConfigs or similar nirs4all data structure.

        Returns:
            Tuple of (X_source, X_target, y_source, y_target).
        """
        # Handle DatasetConfigs
        if hasattr(config, "get_datasets") or hasattr(config, "iter_datasets"):
            if hasattr(config, "get_datasets"):
                datasets = config.get_datasets()
            else:
                datasets = list(config.iter_datasets())

            if not datasets:
                raise ValueError("DatasetConfigs contains no datasets")

            if len(datasets) == 1:
                # Single dataset: train as source, test as target
                dataset = datasets[0]
                X_source, y_source = self._extract_samples_from_dataset(
                    dataset, partition="train"
                )
                X_target, y_target = self._extract_samples_from_dataset(
                    dataset, partition="test"
                )
                return X_source, X_target, y_source, y_target
            else:
                # Multiple datasets: combine all samples from each
                all_X = []
                all_y = []
                for ds in datasets:
                    X_all, y_all = self._extract_samples_from_dataset(ds, partition="all")
                    all_X.append(X_all)
                    if y_all is not None:
                        all_y.append(y_all)

                # Use first dataset as source, rest combined as target
                X_source = all_X[0]
                y_source = all_y[0] if all_y else None
                X_target = np.vstack(all_X[1:])
                y_target = np.concatenate(all_y[1:]) if len(all_y) > 1 else None

                return X_source, X_target, y_source, y_target

        # Handle single SpectroDataset
        if hasattr(config, "get_samples"):
            X_source, y_source = self._extract_samples_from_dataset(
                config, partition="train"
            )
            X_target, y_target = self._extract_samples_from_dataset(
                config, partition="test"
            )
            return X_source, X_target, y_source, y_target

        raise ValueError(
            f"Cannot extract data from {type(config)}. "
            "Expected DatasetConfigs or SpectroDataset."
        )

    def _extract_samples_from_dataset(
        self,
        dataset,
        partition: str = "train",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract X, y from a SpectroDataset for a given partition.

        Args:
            dataset: SpectroDataset instance.
            partition: Partition to extract ('train', 'test', or 'all').

        Returns:
            Tuple of (X, y) where y may be None.
        """
        # Get samples using SpectroDataset.x() method
        if partition == "all":
            # Combine train and test
            try:
                X_train = dataset.x({"partition": "train"})
            except Exception:
                X_train = None

            try:
                X_test = dataset.x({"partition": "test"})
            except Exception:
                X_test = None

            if X_train is not None and X_test is not None:
                # Handle tuple returns (multi-source)
                if isinstance(X_train, tuple):
                    X_train = X_train[0]
                if isinstance(X_test, tuple):
                    X_test = X_test[0]
                X = np.vstack([np.asarray(X_train), np.asarray(X_test)])
            elif X_train is not None:
                X = np.asarray(X_train[0] if isinstance(X_train, tuple) else X_train)
            elif X_test is not None:
                X = np.asarray(X_test[0] if isinstance(X_test, tuple) else X_test)
            else:
                raise ValueError("Dataset has no samples")
        else:
            try:
                samples = dataset.x({"partition": partition})
            except Exception:
                raise ValueError(f"No samples found for partition '{partition}'")
            if samples is None:
                raise ValueError(f"No samples found for partition '{partition}'")
            X = samples[0] if isinstance(samples, tuple) else np.asarray(samples)

        # Get targets using SpectroDataset.y() method
        y = None
        try:
            if partition == "all":
                y_train = dataset.y({"partition": "train"})
                y_test = dataset.y({"partition": "test"})
                if y_train is not None and y_test is not None:
                    y = np.concatenate([
                        np.asarray(y_train).ravel(),
                        np.asarray(y_test).ravel()
                    ])
                elif y_train is not None:
                    y = np.asarray(y_train).ravel()
                elif y_test is not None:
                    y = np.asarray(y_test).ravel()
            else:
                y_data = dataset.y({"partition": partition})
                if y_data is not None:
                    y = np.asarray(y_data).ravel()
        except Exception:
            pass

        return np.asarray(X), y

    def get_preprocessing_by_name(self, name: str) -> Any:
        """
        Get a preprocessing transform by name.

        Args:
            name: Preprocessing name (e.g., "snv", "snv>d1").

        Returns:
            Transformer or list of transformers for stacked pipelines.
        """
        if ">" in name:
            components = name.split(">")
            return [self.preprocessings[c] for c in components]
        else:
            return self.preprocessings.get(name)
