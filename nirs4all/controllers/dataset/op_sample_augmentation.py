from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import copy
import numpy as np  # noqa: F401

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.balancing import BalancingCalculator
from nirs4all.utils.binning import BinningCalculator  # noqa: F401 - used in _execute_balanced

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


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
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[Any] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List]:
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
                    "balance": "y" or "metadata_column",
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
        config = step["sample_augmentation"]
        transformers = config.get("transformers", [])

        if not transformers:
            raise ValueError("sample_augmentation requires at least one transformer")

        # Determine mode
        is_balanced = "balance" in config

        if is_balanced:
            return self._execute_balanced(config, transformers, dataset, context, runner, loaded_binaries)
        else:
            return self._execute_standard(config, transformers, dataset, context, runner, loaded_binaries)

    def _execute_standard(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ) -> Tuple[Dict[str, Any], List]:
        """Execute standard count-based augmentation."""
        count = config.get("count", 1)
        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)

        # Get train samples (base only, no augmented)
        train_context = copy.deepcopy(context)
        train_context["partition"] = "train"

        # Get base samples only (exclude augmented)
        base_samples_idx = dataset._indexer.x_indices(train_context, include_augmented=False)  # noqa: SLF001
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
            transformer_to_samples, transformers, context, dataset, runner, loaded_binaries
        )

        return context, []

    def _execute_balanced(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ) -> Tuple[Dict[str, Any], List]:
        """Execute balanced class-aware augmentation."""
        balance_source = config.get("balance")
        target_size = config.get("target_size", None)
        max_factor = config.get("max_factor", None)
        ref_percentage = config.get("ref_percentage", None)
        if target_size is None and ref_percentage is None and max_factor is None:
            ref_percentage = 1.0  # Default to ref_percentage=1.0 if none specified

        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)
        bin_balancing = config.get("bin_balancing", "sample")  # NEW: "sample" or "value"

        # Get train samples ONLY (ensure we're in train partition)
        train_context = copy.deepcopy(context)
        train_context["partition"] = "train"
        train_context.pop("train_indices", None)  # Remove any existing indices
        train_context.pop("test_indices", None)

        # Get ALL TRAIN samples (base + augmented)
        all_train_samples = dataset._indexer.x_indices(train_context, include_augmented=True)  # noqa: SLF001
        # Get only BASE TRAIN samples (these have actual data to augment)
        base_train_samples = dataset._indexer.x_indices(train_context, include_augmented=False)  # noqa: SLF001

        if len(base_train_samples) == 0:
            return context, []

        # Get labels for ALL TRAIN samples (to calculate target size)
        if balance_source == "y":
            labels_all_train = dataset.y(train_context, include_augmented=True)
            # Flatten if necessary
            labels_all_train = labels_all_train.flatten() if labels_all_train.ndim > 1 else labels_all_train

            # Store original values before binning (needed for value-aware balancing)
            original_values_all = labels_all_train.copy()

            # Apply binning for regression tasks
            if dataset.is_regression():
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
            origin_indices = dataset._indexer.y_indices(train_context, include_augmented=True)  # noqa: SLF001
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
        if bin_balancing == "value" and dataset.is_regression() and original_values_base is not None:
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
            transformer_to_samples, transformers, context, dataset, runner, loaded_binaries
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
        context: Dict[str, Any],
        dataset: 'SpectroDataset',
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ):
        """
        Emit ONE run_step per transformer with the list of target sample indices.

        TransformerMixinController will:
        1. Detect augment_sample action
        2. For each sample in the list:
           - Get origin data (all sources)
           - Transform it
           - Call dataset.add_samples() (or augment_samples with count=1)
        """
        for trans_idx, sample_ids in transformer_to_samples.items():
            if not sample_ids or len(sample_ids) == 0:
                continue

            transformer = transformers[trans_idx]

            # print(f"Applying transformer {transformer} to {sample_ids} samples")

            # Create context for this transformer's augmentation
            local_context = copy.deepcopy(context)
            local_context["augment_sample"] = True  # Signal action (like add_feature)
            local_context["target_samples"] = sample_ids  # Indices to augment
            local_context["partition"] = "train"

            # ONE run_step per transformer - it handles all target samples
            runner.run_step(
                transformer,
                dataset,
                local_context,
                prediction_store=None,
                is_substep=True,
                propagated_binaries=loaded_binaries
            )

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
