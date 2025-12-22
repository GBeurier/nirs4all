from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import Counter
import numpy as np  # noqa: F401

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.controllers.data.balancing import BalancingCalculator
from nirs4all.data.binning import BinningCalculator  # noqa: F401 - used in _execute_balanced
from nirs4all.pipeline.config.component_serialization import deserialize_component

logger = get_logger(__name__)

try:
    import joblib  # noqa: F401 - used to check availability
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class SampleAugmentationController(OperatorController):
    """
    Sample Augmentation Controller with delegation pattern.

    This controller orchestrates sample augmentation by:
    1. Calculating augmentation distribution (standard or balanced mode)
    2. Creating transformer→samples mapping
    3. Emitting ONE run_step per transformer with target samples

    The actual augmentation work is delegated to TransformerMixinController.
    """
    priority = 10

    @staticmethod
    def normalize_generator_spec(spec: Any) -> Any:
        """Normalize generator spec for sample_augmentation context.

        In sample_augmentation context, multi-selection should use combinations
        by default since the order of transformers doesn't matter.
        Translates legacy 'size' to 'pick' for explicit semantics.

        Args:
            spec: Generator specification (may contain _or_, size, pick, arrange).

        Returns:
            Normalized spec with 'size' converted to 'pick' if needed.
        """
        if not isinstance(spec, dict):
            return spec

        # If explicit pick/arrange specified, honor it
        if "pick" in spec or "arrange" in spec:
            return spec

        # Convert legacy size to pick (combinations) for sample_augmentation
        if "size" in spec and "_or_" in spec:
            result = dict(spec)
            result["pick"] = result.pop("size")
            return result

        return spec

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "sample_augmentation"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Sample augmentation only runs during training."""
        return False

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[Any] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List]:
        """
        Execute sample augmentation with standard or balanced mode.

        Step format for standard mode:
            {
                "sample_augmentation": {
                    "transformers": [transformer1, transformer2, ...],
                    "count": int,
                    "selection": "random" or "all",  # Default "random"
                    "random_state": int  # Optional
                }
            }

        Step format for balanced mode (choose one balancing strategy):
            Mode 1 - Fixed target size per class:
            {
                "sample_augmentation": {
                    "transformers": [...],
                    "balance": "y" or "metadata_column",  # Default "y"
                    "target_size": int,  # Fixed target samples per class
                    "selection": "random" or "all",
                    "random_state": int
                }
            }

            Mode 2 - Multiplier for augmentation:
            {
                "sample_augmentation": {
                    "transformers": [...],
                    "balance": "y" or "metadata_column",
                    "max_factor": float,  # Multiplier (e.g., 3 means class grows 3x)
                    "selection": "random" or "all",
                    "random_state": int
                }
            }

            Mode 3 - Percentage of majority class:
            {
                "sample_augmentation": {
                    "transformers": [...],
                    "balance": "y" or "metadata_column",
                    "ref_percentage": float,  # Target as % of majority (0.0-1.0)
                    "selection": "random" or "all",
                    "random_state": int
                }
            }

        Binning for regression (automatic when balance="y" and task is regression):
            {
                "sample_augmentation": {
                    "transformers": [...],
                    "balance": "y",
                    "bins": int,  # Number of virtual classes (default: 10)
                    "binning_strategy": "equal_width" or "quantile",  # Default: "equal_width"
                    "max_factor": float,  # Choose one balancing mode
                    "selection": "random" or "all",
                    "random_state": int
                }
            }
        """
        # Extract step config for compatibility
        step = step_info.original_step

        config = step["sample_augmentation"]
        transformers_raw = config.get("transformers", [])

        if not transformers_raw:
            raise ValueError("sample_augmentation requires at least one transformer")

        # Deserialize transformers (they may be stored as serialized class paths)
        transformers = [deserialize_component(t) for t in transformers_raw]

        # Determine mode
        is_balanced = "balance" in config

        if is_balanced:
            return self._execute_balanced(config, transformers, dataset, context, runtime_context, loaded_binaries)
        else:
            return self._execute_standard(config, transformers, dataset, context, runtime_context, loaded_binaries)

    def _execute_standard(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Optional[Any]
    ) -> Tuple['ExecutionContext', List]:
        """Execute standard count-based augmentation."""
        count = config.get("count", 1)
        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)

        # Get train samples (base only, no augmented)
        train_context = context.with_partition("train")

        # Get base samples only (exclude augmented)
        base_samples_idx = dataset._indexer.x_indices(train_context.selector, include_augmented=False)  # noqa: SLF001
        base_samples = base_samples_idx.tolist() if hasattr(base_samples_idx, 'tolist') else list(base_samples_idx)

        if not base_samples:
            return context, []

        # Create augmentation plan: sample_id → number of augmentations
        augmentation_counts = {sample_id: count for sample_id in base_samples}

        # Build transformer distribution: sample_id → list of transformer indices
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:  # "all"
            transformer_map = self._cycle_transformers(transformers, augmentation_counts)

        # Invert map: transformer_idx → list of sample_ids
        transformer_to_samples = self._invert_transformer_map(transformer_map, len(transformers))

        # Emit ONE run_step per transformer
        self._emit_augmentation_steps(
            transformer_to_samples, transformers, context, dataset, runtime_context, loaded_binaries
        )

        return context, []

    def _execute_balanced(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Optional[Any]
    ) -> Tuple['ExecutionContext', List]:
        """Execute balanced class-aware augmentation."""
        balance_source = config.get("balance", "y")
        target_size = config.get("target_size", None)
        max_factor = config.get("max_factor", None)
        ref_percentage = config.get("ref_percentage", None)
        if target_size is None and ref_percentage is None and max_factor is None:
            ref_percentage = 1.0  # Default to ref_percentage=1.0 if none specified

        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)
        bin_balancing = config.get("bin_balancing", "sample")  # NEW: "sample" or "value"

        # Get train samples ONLY (ensure we're in train partition)
        train_context = context.with_partition("train")
        # train_context.pop("train_indices", None)  # Remove any existing indices
        # train_context.pop("test_indices", None)

        # Get ALL TRAIN samples (base + augmented)
        all_train_samples = dataset._indexer.x_indices(train_context.selector, include_augmented=True)  # noqa: SLF001
        # Get only BASE TRAIN samples (these have actual data to augment)
        base_train_samples = dataset._indexer.x_indices(train_context.selector, include_augmented=False)  # noqa: SLF001

        if len(base_train_samples) == 0:
            return context, []

        # Get labels for ALL TRAIN samples (to calculate target size)
        if balance_source == "y":
            labels_all_train = dataset.y(train_context.selector, include_augmented=True)
            # Flatten if necessary
            labels_all_train = labels_all_train.flatten() if labels_all_train.ndim > 1 else labels_all_train

            # Store original values before binning (needed for value-aware balancing)
            original_values_all = labels_all_train.copy()

            # Apply binning for regression tasks
            if dataset.is_regression:
                bins = config.get("bins", 10)
                strategy = config.get("binning_strategy", "equal_width")
                labels_all_train, _ = BinningCalculator.bin_continuous_targets(
                    labels_all_train, bins=bins, strategy=strategy
                )
        else:
            # Metadata column - map augmented samples to origins using y_indices
            if not isinstance(balance_source, str):
                raise ValueError(f"balance source must be 'y' or a metadata column name, got {balance_source}")
            # Get origin indices for all train samples (including augmented mapped to origins)
            origin_indices = dataset._indexer.y_indices(train_context.selector, include_augmented=True)  # noqa: SLF001
            # Get base metadata and index into it using origin indices
            base_metadata = dataset._metadata.get_column(balance_source)  # noqa: SLF001
            labels_all_train = base_metadata[origin_indices]
            original_values_all = None

        # Get labels for BASE TRAIN samples only (for calculating augmentation per base sample)
        labels_base_train = labels_all_train[:len(base_train_samples)]
        if original_values_all is not None:
            original_values_base = original_values_all[:len(base_train_samples)]
        else:
            original_values_base = None

        # Calculate augmentation counts per BASE TRAIN sample using specified mode
        if bin_balancing == "value" and dataset.is_regression and original_values_base is not None:
            # Use value-aware balancing for regression with binning
            augmentation_counts = BalancingCalculator.calculate_balanced_counts_value_aware(
                labels_base_train,
                base_train_samples,
                original_values_base,
                labels_all_train,
                all_train_samples,
                target_size=target_size,
                max_factor=max_factor,
                ref_percentage=ref_percentage,
                random_state=random_state
            )
        else:
            # Use standard sample-aware balancing
            augmentation_counts = BalancingCalculator.calculate_balanced_counts(
                labels_base_train,
                base_train_samples,
                labels_all_train,
                all_train_samples,
                target_size=target_size,
                max_factor=max_factor,
                ref_percentage=ref_percentage,
                random_state=random_state
            )

        # --- Debug Print ---
        logger.debug("--- Sample Augmentation Class Distribution ---")
        logger.debug("Before Augmentation:")
        before_counts = Counter(labels_all_train)
        for label, count in sorted(before_counts.items()):
            logger.debug(f"  Class {label}: {count}")

        logger.debug("Planned Augmentation:")
        sample_to_label = {sid: lbl for sid, lbl in zip(base_train_samples, labels_base_train)}
        added_counts = Counter()
        for sample_id, count in augmentation_counts.items():
            if count > 0:
                lbl = sample_to_label.get(sample_id)
                if lbl is not None:
                    added_counts[lbl] += count

        logger.debug("After Augmentation (Expected):")
        all_labels = set(before_counts.keys()) | set(added_counts.keys())
        for label in sorted(all_labels):
            before = before_counts[label]
            added = added_counts[label]
            total = before + added
            logger.debug(f"  Class {label}: {before} + {added} = {total}")
        logger.debug("----------------------------------------------")
        # -------------------

        # Check if any augmentation is needed
        if sum(augmentation_counts.values()) == 0:
            # All classes already balanced, no augmentation needed
            return context, []

        # Build transformer distribution
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:
            transformer_map = self._cycle_transformers(transformers, augmentation_counts)

        # Invert map: transformer_idx → list of sample_ids
        transformer_to_samples = self._invert_transformer_map(transformer_map, len(transformers))

        # Emit ONE run_step per transformer-
        self._emit_augmentation_steps(
            transformer_to_samples, transformers, context, dataset, runtime_context, loaded_binaries
        )

        return context, []

    def _invert_transformer_map(
        self,
        transformer_map: Dict[int, List[int]],
        n_transformers: int
    ) -> Dict[int, List[int]]:
        """
        Invert sample→transformer map to transformer→samples map.

        Args:
            transformer_map: {sample_id: [trans_idx1, trans_idx2, ...]}
            n_transformers: Total number of transformers

        Returns:
            {trans_idx: [sample_id1, sample_id2, ...]}
        """
        inverted = {i: [] for i in range(n_transformers)}

        for sample_id, trans_indices in transformer_map.items():
            for trans_idx in trans_indices:
                inverted[trans_idx].append(sample_id)

        return inverted

    def _emit_augmentation_steps(
        self,
        transformer_to_samples: Dict[int, List[int]],
        transformers: List,
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Optional[Any]
    ):
        """
        Execute transformers and add augmented samples to dataset.

        This method supports two modes:
        1. Parallel mode (when joblib available and n_jobs > 1): Execute transformers in parallel,
           collect all augmented data, then batch insert. Much faster for many transformers.
        2. Sequential mode: Execute transformers one by one (fallback).

        TransformerMixinController will:
        1. Detect augment_sample action
        2. Transform all target samples in batch
        3. Return augmented data OR add to dataset directly
        """
        # Check if parallel execution is possible and beneficial
        active_transformers = [(idx, samples) for idx, samples in transformer_to_samples.items()
                               if samples and len(samples) > 0]

        n_transformers = len(active_transformers)
        if n_transformers == 0:
            return

        # Use parallel execution if joblib available and multiple transformers
        use_parallel = JOBLIB_AVAILABLE and n_transformers > 1

        if use_parallel:
            self._emit_augmentation_steps_parallel(
                active_transformers, transformers, context, dataset, runtime_context, loaded_binaries
            )
        else:
            self._emit_augmentation_steps_sequential(
                active_transformers, transformers, context, dataset, runtime_context, loaded_binaries
            )

    def _emit_augmentation_steps_sequential(
        self,
        active_transformers: List[Tuple[int, List[int]]],
        transformers: List,
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Optional[Any]
    ):
        """Sequential execution of transformers (original implementation)."""
        for trans_idx, sample_ids in active_transformers:
            transformer = transformers[trans_idx]

            # Create context for this transformer's augmentation
            local_context = context.with_metadata(
                augment_sample=True,
                target_samples=sample_ids
            ).with_partition("train")

            # ONE run_step per transformer - it handles all target samples
            if runtime_context.step_runner:
                runtime_context.substep_number += 1
                _ = runtime_context.step_runner.execute(
                    transformer,
                    dataset,
                    local_context,
                    runtime_context,
                    loaded_binaries=loaded_binaries,
                    prediction_store=None
                )

    def _emit_augmentation_steps_parallel(
        self,
        active_transformers: List[Tuple[int, List[int]]],
        transformers: List,
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Optional[Any]
    ):
        """
        Parallel execution of transformers using joblib.

        Flow:
        1. Fetch train data once (for fitting) and all origin data (for transform)
        2. Execute all transformers in parallel, each returning augmented data
        3. Collect all results and batch insert into dataset
        """
        from sklearn.base import clone
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Get train data for fitting (once for all transformers)
        train_context = context.with_partition("train")
        train_selector = train_context.selector.with_augmented(False)
        train_data = dataset.x(train_selector, "3d", concat_source=False)
        if not isinstance(train_data, list):
            train_data = [train_data]

        n_sources = len(train_data)
        n_processings = train_data[0].shape[1] if n_sources > 0 else 0

        # Collect all unique sample IDs across all transformers
        all_sample_ids = set()
        for _, sample_ids in active_transformers:
            all_sample_ids.update(sample_ids)
        all_sample_ids_list = sorted(all_sample_ids)

        # Batch fetch all origin samples once
        batch_selector = {"sample": all_sample_ids_list}
        all_origin_data = dataset.x(batch_selector, "3d", concat_source=False, include_augmented=False)
        if not isinstance(all_origin_data, list):
            all_origin_data = [all_origin_data]

        # Create sample_id to index mapping for efficient lookup
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(all_sample_ids_list)}

        # Pre-fit all transformer × source × processing combinations
        # This can be done in parallel too, but keep it simple for now
        all_fitted = {}  # (trans_idx, source_idx, proc_idx) -> fitted transformer
        for trans_idx, _ in active_transformers:
            transformer = transformers[trans_idx]
            # Check if transformer is actual object or string reference
            if isinstance(transformer, str):
                raise ValueError(f"Transformer at index {trans_idx} is a string '{transformer}' instead of an object. "
                                 "Ensure transformers are instantiated before passing to sample_augmentation.")
            for source_idx in range(n_sources):
                for proc_idx in range(n_processings):
                    cloned = clone(transformer)
                    train_proc = train_data[source_idx][:, proc_idx, :]
                    cloned.fit(train_proc)
                    all_fitted[(trans_idx, source_idx, proc_idx)] = cloned

        def process_transformer(args):
            """Process a single transformer and return augmented data + index info."""
            trans_idx, sample_ids = args
            transformer = transformers[trans_idx]
            operator_name = transformer.__class__.__name__

            # Get indices for this transformer's samples
            local_indices = [sample_id_to_idx[sid] for sid in sample_ids]

            # Transform all samples for this transformer
            transformed_per_source = []
            for source_idx in range(n_sources):
                source_origin = all_origin_data[source_idx]  # (all_samples, procs, feats)
                local_source_data = source_origin[local_indices]  # (n_local, procs, feats)

                transformed_procs = []
                for proc_idx in range(n_processings):
                    proc_data = local_source_data[:, proc_idx, :]  # (n_local, feats)
                    fitted = all_fitted[(trans_idx, source_idx, proc_idx)]
                    transformed = fitted.transform(proc_data)  # (n_local, feats)
                    transformed_procs.append(transformed)

                # Stack processings: (n_local, n_processings, n_features)
                source_3d = np.stack(transformed_procs, axis=1)
                transformed_per_source.append(source_3d)

            # Prepare output data
            # For multi-source, return list of arrays (one per source)
            # For single source, return single array
            if n_sources == 1:
                batch_data = transformed_per_source[0]  # (n_local, n_procs, n_feats)
            else:
                # For multi-source, return list of arrays
                batch_data = transformed_per_source  # List of (n_local, n_procs, n_feats)

            # Build index dictionaries
            indexes_list = [
                {"partition": "train", "origin": sid, "augmentation": operator_name}
                for sid in sample_ids
            ]

            return batch_data, indexes_list

        # Execute in parallel using ThreadPoolExecutor (no pickling issues)
        all_batch_data = []
        all_indexes = []

        max_workers = min(len(active_transformers), 16)  # Cap at 16 threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_transformer, args): args
                       for args in active_transformers}

            for future in as_completed(futures):
                batch_data, indexes_list = future.result()
                all_batch_data.append(batch_data)
                all_indexes.extend(indexes_list)

        if not all_batch_data:
            return

        # Concatenate all augmented data
        # Handle both single-source (arrays) and multi-source (list of arrays)
        if n_sources == 1:
            # Single source: all_batch_data is list of arrays
            combined_data = np.concatenate(all_batch_data, axis=0)
        else:
            # Multi-source: all_batch_data is list of lists of arrays
            # Need to concatenate per-source, then return as list
            combined_data = []
            for source_idx in range(n_sources):
                source_arrays = [batch[source_idx] for batch in all_batch_data]
                combined_source = np.concatenate(source_arrays, axis=0)
                combined_data.append(combined_source)

        # Single batch insert for ALL augmented samples from ALL transformers
        dataset.add_samples_batch(data=combined_data, indexes_list=all_indexes)

    def _cycle_transformers(
        self,
        transformers: List,
        augmentation_counts: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """Cycle through transformers for 'all' selection mode."""
        transformer_map = {}
        for sample_id, count in augmentation_counts.items():
            if count > 0:
                transformer_map[sample_id] = [i % len(transformers) for i in range(count)]
            else:
                transformer_map[sample_id] = []
        return transformer_map
