from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

import numpy as np  # noqa: F401

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.data.balancing import BalancingCalculator
from nirs4all.controllers.registry import register_controller
from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.core.logging import get_logger
from nirs4all.data.binning import BinningCalculator  # noqa: F401 - used in _execute_balanced
from nirs4all.pipeline.config.component_serialization import deserialize_component

logger = get_logger(__name__)

try:
    import joblib  # noqa: F401 - used to check availability
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.runner import PipelineRunner
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

        Args:
            spec: Generator specification (may contain _or_, pick, arrange).

        Returns:
            Normalized spec.
        """
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
        loaded_binaries: Any | None = None,
        prediction_store: Any | None = None
    ) -> tuple['ExecutionContext', list]:
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
        # For dict-style specs, extract the transformer object and keep the spec for variation_scope parsing
        transformers = []
        for t in transformers_raw:
            if isinstance(t, dict) and "transformer" in t:
                transformers.append(deserialize_component(t["transformer"]))
            else:
                transformers.append(deserialize_component(t))

        # Parse variation_scope per transformer (parallel to transformers list)
        variation_scopes = [
            self._parse_variation_scope(config, transformer_spec=raw)
            for raw in transformers_raw
        ]

        # Determine mode
        is_balanced = "balance" in config

        if is_balanced:
            return self._execute_balanced(config, transformers, variation_scopes, dataset, context, runtime_context, loaded_binaries)
        else:
            return self._execute_standard(config, transformers, variation_scopes, dataset, context, runtime_context, loaded_binaries)

    def _execute_standard(
        self,
        config: dict,
        transformers: list,
        variation_scopes: list[str],
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Any | None
    ) -> tuple['ExecutionContext', list]:
        """Execute standard count-based augmentation."""
        count = config.get("count", 1)
        selection = config.get("selection", "random")
        random_state = config.get("random_state")

        # Get train samples (base only, no augmented)
        train_context = context.with_partition("train")

        # Get base samples only (exclude augmented)
        base_samples_idx = dataset._indexer.x_indices(train_context.selector, include_augmented=False)  # noqa: SLF001
        base_samples = base_samples_idx.tolist() if hasattr(base_samples_idx, 'tolist') else list(base_samples_idx)

        if not base_samples:
            return context, []

        # Create augmentation plan: sample_id → number of augmentations
        augmentation_counts = dict.fromkeys(base_samples, count)

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
            transformer_to_samples, transformers, variation_scopes, context, dataset, runtime_context, loaded_binaries
        )

        return context, []

    def _execute_balanced(
        self,
        config: dict,
        transformers: list,
        variation_scopes: list[str],
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Any | None
    ) -> tuple['ExecutionContext', list]:
        """Execute balanced class-aware augmentation."""
        balance_source = config.get("balance", "y")
        target_size = config.get("target_size")
        max_factor = config.get("max_factor")
        ref_percentage = config.get("ref_percentage")
        if target_size is None and ref_percentage is None and max_factor is None:
            ref_percentage = 1.0  # Default to ref_percentage=1.0 if none specified

        selection = config.get("selection", "random")
        random_state = config.get("random_state")
        bin_balancing = config.get("bin_balancing", "sample")  # "sample" or "value"

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
        original_values_base = original_values_all[:len(base_train_samples)] if original_values_all is not None else None

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
        sample_to_label = dict(zip(base_train_samples, labels_base_train, strict=False))
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
        transformer_map = BalancingCalculator.apply_random_transformer_selection(transformers, augmentation_counts, random_state) if selection == "random" else self._cycle_transformers(transformers, augmentation_counts)

        # Invert map: transformer_idx → list of sample_ids
        transformer_to_samples = self._invert_transformer_map(transformer_map, len(transformers))

        # Emit ONE run_step per transformer
        self._emit_augmentation_steps(
            transformer_to_samples, transformers, variation_scopes, context, dataset, runtime_context, loaded_binaries
        )

        return context, []

    def _invert_transformer_map(
        self,
        transformer_map: dict[int, list[int]],
        n_transformers: int
    ) -> dict[int, list[int]]:
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
        transformer_to_samples: dict[int, list[int]],
        transformers: list,
        variation_scopes: list[str],
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Any | None
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
                active_transformers, transformers, variation_scopes, context, dataset, runtime_context, loaded_binaries
            )
        else:
            self._emit_augmentation_steps_sequential(
                active_transformers, transformers, variation_scopes, context, dataset, runtime_context, loaded_binaries
            )

    def _emit_augmentation_steps_sequential(
        self,
        active_transformers: list[tuple[int, list[int]]],
        transformers: list,
        variation_scopes: list[str],
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Any | None
    ):
        """Sequential execution of transformers with variation_scope support.

        For operators with ``_supports_variation_scope``:
            Sets ``variation_scope`` on a cloned operator and delegates to
            step_runner in a single call with all target samples.

        For operators without internal support and scope=="sample":
            Executes step_runner once per sample, each with a cloned
            transformer carrying a unique random seed.

        For operators without internal support and scope=="batch" (or default):
            Single step_runner call with all samples (original behavior).
        """
        from sklearn.base import clone

        for trans_idx, sample_ids in active_transformers:
            transformer = transformers[trans_idx]
            scope = variation_scopes[trans_idx]

            if self._supports_internal_variation(transformer):
                # Performance path: operator handles variation_scope internally
                transformer_to_use = clone(transformer)
                transformer_to_use.set_params(variation_scope=scope)

                local_context = context.with_metadata(
                    augment_sample=True,
                    target_samples=sample_ids
                ).with_partition("train")

                if runtime_context.step_runner:
                    runtime_context.substep_number += 1
                    _ = runtime_context.step_runner.execute(
                        transformer_to_use,
                        dataset,
                        local_context,
                        runtime_context,
                        loaded_binaries=loaded_binaries,
                        prediction_store=None
                    )

            elif scope == "sample":
                # Per-sample execution: each sample gets a uniquely seeded clone
                base_seed = getattr(transformer, 'random_state', None)
                for i, sample_id in enumerate(sample_ids):
                    cloned = clone(transformer)
                    if base_seed is not None:
                        cloned.set_params(random_state=base_seed + i)

                    local_context = context.with_metadata(
                        augment_sample=True,
                        target_samples=[sample_id]
                    ).with_partition("train")

                    if runtime_context.step_runner:
                        runtime_context.substep_number += 1
                        _ = runtime_context.step_runner.execute(
                            cloned,
                            dataset,
                            local_context,
                            runtime_context,
                            loaded_binaries=loaded_binaries,
                            prediction_store=None
                        )

            else:
                # Batch scope (default for non-stochastic or explicit batch):
                # single step_runner call with all samples
                local_context = context.with_metadata(
                    augment_sample=True,
                    target_samples=sample_ids
                ).with_partition("train")

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
        active_transformers: list[tuple[int, list[int]]],
        transformers: list,
        variation_scopes: list[str],
        context: 'ExecutionContext',
        dataset: 'SpectroDataset',
        runtime_context: 'RuntimeContext',
        loaded_binaries: Any | None
    ):
        """
        Parallel execution of transformers using ThreadPoolExecutor.

        Supports variation_scope:
        - Operators with ``_supports_variation_scope``: sets scope on clone,
          single fit, single batch transform (performance path).
        - Operators without internal support and scope=="batch": single clone,
          single fit, single batch transform (same as previous behavior).
        - Operators without internal support and scope=="sample": per-sample
          clones with unique random seeds, individual transforms.

        Wavelength-aware operators (SpectraTransformerMixin subclasses) receive
        wavelengths via kwargs on both fit() and transform() calls. Wavelengths
        are cached per source to avoid redundant lookups.

        Flow:
        1. Fetch train data once (for fitting) and all origin data (for transform)
        2. Execute all transformers in parallel, each returning augmented data
        3. Collect all results and batch insert into dataset
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from sklearn.base import clone

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

        # --- Wavelength caching ---
        _MISSING = object()  # Sentinel for "no wavelengths needed/available"
        wavelengths_cache = {}  # source_idx -> wavelengths array or _MISSING

        def get_wavelengths(source_idx, operator):
            """Get wavelengths for a source, with caching."""
            if source_idx in wavelengths_cache:
                wl = wavelengths_cache[source_idx]
                return None if wl is _MISSING else wl

            if not TransformerMixinController._needs_wavelengths(operator):
                wavelengths_cache[source_idx] = _MISSING
                return None

            try:
                wl = TransformerMixinController._extract_wavelengths(
                    dataset, source_idx, operator.__class__.__name__
                )
                wavelengths_cache[source_idx] = wl
                return wl
            except (ValueError, AttributeError):
                req = getattr(operator, '_requires_wavelengths', False)
                if req is True:
                    raise
                wavelengths_cache[source_idx] = _MISSING
                return None

        # Pre-fit all transformer x source x processing combinations
        all_fitted = {}  # (trans_idx, source_idx, proc_idx) -> fitted transformer
        for trans_idx, _ in active_transformers:
            transformer = transformers[trans_idx]
            scope = variation_scopes[trans_idx]
            if isinstance(transformer, str):
                raise ValueError(f"Transformer at index {trans_idx} is a string '{transformer}' instead of an object. "
                                 "Ensure transformers are instantiated before passing to sample_augmentation.")
            for source_idx in range(n_sources):
                for proc_idx in range(n_processings):
                    cloned = clone(transformer)
                    if self._supports_internal_variation(transformer):
                        cloned.set_params(variation_scope=scope)
                    train_proc = train_data[source_idx][:, proc_idx, :]

                    wl = get_wavelengths(source_idx, cloned)
                    if wl is not None:
                        cloned.fit(train_proc, wavelengths=wl)
                    else:
                        cloned.fit(train_proc)

                    all_fitted[(trans_idx, source_idx, proc_idx)] = cloned

        def process_transformer(args):
            """Process a single transformer and return augmented data + index info."""
            trans_idx, sample_ids = args
            transformer = transformers[trans_idx]
            scope = variation_scopes[trans_idx]
            operator_name = transformer.__class__.__name__
            # Generate operator name with parameters for augmentation labels
            from nirs4all.utils.operator_formatting import format_operator_with_params
            operator_name_with_params = format_operator_with_params(transformer)

            # Get indices for this transformer's samples
            local_indices = [sample_id_to_idx[sid] for sid in sample_ids]

            if self._supports_internal_variation(transformer) or scope == "batch":
                # Performance path (internal variation support) or batch scope:
                # single fit already done, single batch transform
                transformed_per_source = []
                for source_idx in range(n_sources):
                    source_origin = all_origin_data[source_idx]
                    local_source_data = source_origin[local_indices]

                    transformed_procs = []
                    for proc_idx in range(n_processings):
                        proc_data = local_source_data[:, proc_idx, :]
                        fitted = all_fitted[(trans_idx, source_idx, proc_idx)]

                        wl = get_wavelengths(source_idx, fitted)
                        transformed = fitted.transform(proc_data, wavelengths=wl) if wl is not None else fitted.transform(proc_data)

                        transformed_procs.append(transformed)

                    source_3d = np.stack(transformed_procs, axis=1)
                    transformed_per_source.append(source_3d)

                batch_data = transformed_per_source[0] if n_sources == 1 else transformed_per_source

                indexes_list = [
                    {"partition": "train", "origin": sid, "augmentation": operator_name_with_params}
                    for sid in sample_ids
                ]
                return batch_data, indexes_list

            else:
                # scope == "sample" without internal support:
                # Per-sample clones with unique seeds
                base_seed = getattr(transformer, 'random_state', None)
                per_sample_sources = [[] for _ in range(n_sources)]

                for i, sample_id in enumerate(sample_ids):
                    local_idx = sample_id_to_idx[sample_id]
                    sample_seed = (base_seed + i) if base_seed is not None else None

                    for source_idx in range(n_sources):
                        source_origin = all_origin_data[source_idx]
                        sample_data = source_origin[local_idx:local_idx + 1]  # (1, procs, feats)

                        transformed_procs = []
                        for proc_idx in range(n_processings):
                            sample_proc_data = sample_data[:, proc_idx, :]  # (1, feats)
                            cloned = clone(transformer)
                            if sample_seed is not None:
                                cloned.set_params(random_state=sample_seed)
                            # Fit on train data (reuse train_data from outer scope)
                            train_proc = train_data[source_idx][:, proc_idx, :]

                            wl = get_wavelengths(source_idx, cloned)
                            if wl is not None:
                                cloned.fit(train_proc, wavelengths=wl)
                                transformed = cloned.transform(sample_proc_data, wavelengths=wl)
                            else:
                                cloned.fit(train_proc)
                                transformed = cloned.transform(sample_proc_data)

                            transformed_procs.append(transformed)

                        sample_3d = np.stack(transformed_procs, axis=1)  # (1, procs, feats)
                        per_sample_sources[source_idx].append(sample_3d)

                # Concatenate per-sample results for each source
                transformed_per_source = []
                for source_idx in range(n_sources):
                    source_3d = np.concatenate(per_sample_sources[source_idx], axis=0)
                    transformed_per_source.append(source_3d)

                batch_data = transformed_per_source[0] if n_sources == 1 else transformed_per_source

                indexes_list = [
                    {"partition": "train", "origin": sid, "augmentation": operator_name_with_params}
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

    def _parse_variation_scope(self, config: dict, transformer_spec=None) -> str:
        """Get variation_scope for a transformer, with inheritance.

        Resolution order:
        1. Per-transformer override (dict spec with 'variation_scope' key)
        2. Step-level default from config
        3. Global default: "sample"

        Args:
            config: The sample_augmentation step config dict.
            transformer_spec: Original transformer specification (may be a dict
                with 'transformer' and 'variation_scope' keys).

        Returns:
            The resolved variation_scope string ("sample" or "batch").
        """
        step_scope = config.get("variation_scope", "sample")
        if isinstance(transformer_spec, dict):
            return transformer_spec.get("variation_scope", step_scope)
        return step_scope

    def _supports_internal_variation(self, transformer) -> bool:
        """Check if transformer can handle variation_scope internally.

        Operators with ``_supports_variation_scope = True`` accept a
        ``variation_scope`` parameter and produce the correct per-sample
        or per-batch noise pattern themselves, avoiding the need for
        per-sample cloning in the controller.

        Args:
            transformer: The transformer instance to check.

        Returns:
            True if the transformer handles variation_scope internally.
        """
        return getattr(transformer, '_supports_variation_scope', False)

    def _cycle_transformers(
        self,
        transformers: list,
        augmentation_counts: dict[int, int]
    ) -> dict[int, list[int]]:
        """Cycle through transformers for 'all' selection mode."""
        transformer_map = {}
        for sample_id, count in augmentation_counts.items():
            if count > 0:
                transformer_map[sample_id] = [i % len(transformers) for i in range(count)]
            else:
                transformer_map[sample_id] = []
        return transformer_map
