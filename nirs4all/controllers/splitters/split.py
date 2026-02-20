from __future__ import annotations

import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.operators.splitters import GroupedSplitterWrapper
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep

# Native group-aware splitter class names (sklearn and nirs4all)
# These splitters have built-in group support and handle groups directly
_NATIVE_GROUP_SPLITTERS = frozenset({
    "GroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "StratifiedGroupKFold",
    "SPXYGFold",
    "BinnedStratifiedGroupKFold",
})

def compute_effective_groups(
    dataset: SpectroDataset,
    group_by: str | list[str] | None = None,
    ignore_repetition: bool = False,
    context: Any | None = None,
    include_augmented: bool = False
) -> np.ndarray | None:
    """Compute final group labels for splitting.

    Combines repetition column (if defined and not ignored) with
    additional group_by columns into tuple-based group identifiers.
    This ensures that all spectra from the same physical sample
    (and optionally the same metadata groups) stay together in folds.

    Parameters
    ----------
    dataset : SpectroDataset
        The dataset containing metadata columns for grouping.
    group_by : str or List[str], optional
        Additional column(s) to group by. Combined with repetition
        column if repetition is defined and not ignored.
    ignore_repetition : bool, default=False
        If True, ignore the dataset's repetition column and only use
        group_by columns for grouping. Use for specific experiments
        where you want to split within repetitions.
    context : Any, optional
        Execution context for selecting samples (e.g., partition filter).
        If None, uses all samples.
    include_augmented : bool, default=False
        If True, include augmented samples. Typically False for splitting
        to avoid data leakage.

    Returns
    -------
    np.ndarray or None
        Array of group labels (one per sample), or None if no grouping needed.
        - For single column: array of column values
        - For multiple columns: array of tuples (e.g., (Sample_ID, Year))

    Examples
    --------
    >>> # Repetition only (auto from dataset)
    >>> groups = compute_effective_groups(dataset)
    >>> # groups = ['S1', 'S1', 'S2', 'S2', ...]

    >>> # Repetition + additional grouping
    >>> groups = compute_effective_groups(dataset, group_by=['Year'])
    >>> # groups = [('S1', 2020), ('S1', 2020), ('S2', 2021), ...]

    >>> # Explicit grouping without repetition
    >>> groups = compute_effective_groups(dataset, group_by='Batch', ignore_repetition=True)
    >>> # groups = ['B1', 'B1', 'B2', 'B2', ...]
    """
    columns_to_use: list[str] = []

    # Add repetition column first (unless ignored)
    if not ignore_repetition and dataset.repetition:
        columns_to_use.append(dataset.repetition)

    # Add group_by columns (deduplicate if already in repetition)
    if group_by:
        if isinstance(group_by, str):
            if group_by not in columns_to_use:
                columns_to_use.append(group_by)
        else:
            for col in group_by:
                if col not in columns_to_use:
                    columns_to_use.append(col)

    if not columns_to_use:
        return None

    # Validate columns exist in metadata
    available_columns = dataset.metadata_columns
    for col in columns_to_use:
        if col not in available_columns:
            raise ValueError(
                f"Grouping column '{col}' not found in metadata.\n"
                f"Available columns: {available_columns}"
            )

    # Extract column values
    if len(columns_to_use) == 1:
        # Single column: return simple array
        return dataset.metadata_column(
            columns_to_use[0],
            context,
            include_augmented=include_augmented
        )
    else:
        # Multiple columns: create tuple identifiers
        arrays = [
            dataset.metadata_column(col, context, include_augmented=include_augmented)
            for col in columns_to_use
        ]
        # Create array of tuples for multi-column grouping
        return np.array([tuple(row) for row in zip(*arrays, strict=False)], dtype=object)

def _is_native_group_splitter(splitter: Any) -> bool:
    """Check if splitter has native group support.

    Returns True if the splitter is a known group-aware splitter that
    properly handles the 'groups' parameter directly.
    """
    return splitter.__class__.__name__ in _NATIVE_GROUP_SPLITTERS

def _needs(splitter: Any) -> tuple[bool, bool]:
    """Return booleans *(needs_y, needs_groups)* for the given splitter.

    Introspects the signature of ``split`` *plus* estimator tags (when
    available) so it works for *any* class respecting the sklearn contract.
    """
    split_fn = getattr(splitter, "split", None)
    if not callable(split_fn):
        # No split method → cannot be a valid splitter
        return False, False

    sig = inspect.signature(split_fn)
    params = sig.parameters

    needs_y = "y" in params # and params["y"].default is inspect._empty
    # Check if 'groups' parameter exists - sklearn group splitters have groups=None default
    # but still require the parameter to be provided for proper operation
    needs_g = "groups" in params

    # Honour estimator tags (sklearn >=1.3)
    if hasattr(splitter, "_get_tags"):
        tags = splitter._get_tags()
        needs_y = needs_y or tags.get("requires_y", False)

    return needs_y, needs_g

@register_controller
class CrossValidatorController(OperatorController):
    """Controller for **any** sklearn‑compatible splitter (native or custom)."""

    priority = 10  # processed early but after mandatory pre‑processing steps

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:  # noqa: D401
        """Return *True* if *operator* behaves like a splitter.

        **Criteria** – must expose a callable ``split`` whose first positional
        argument is named *X*.  Optional presence of ``get_n_splits`` is a plus
        but not mandatory, so user‑defined simple splitters are still accepted.

        Also matches on the 'split' keyword for group-aware splitting syntax.
        """
        # Priority 1: Match on 'split' keyword (explicit workflow operator)
        if keyword == "split":
            return True

        # Priority 2: Match dict with 'split' key
        if isinstance(step, dict) and "split" in step:
            return True

        # Priority 3: Match objects with split() method (existing behavior)
        if operator is None:
            return False

        split_fn = getattr(operator, "split", None)
        if not callable(split_fn):
            return False
        try:
            sig = inspect.signature(split_fn)
        except (TypeError, ValueError):  # edge‑cases: C‑extensions or cythonised
            return True  # accept – we can still attempt runtime call
        params: list[inspect.Parameter] = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return bool(params) and params[0].name == "X"

    @classmethod
    def use_multi_source(cls) -> bool:  # noqa: D401
        """Cross‑validators themselves are single‑source operators."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Cross-validators should not execute during prediction mode."""
        return True

    def execute(
        self,
        step_info: ParsedStep,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ):
        """Run ``operator.split`` and store the resulting folds on *dataset*.

        * Smartly supplies ``y`` / ``groups`` only if required.
        * Extracts groups from metadata if specified.
        * Supports ``group_by`` parameter and dataset repetition for group-aware splitting.
        * Maps local indices back to the global index space.
        * Stores the list of folds into the dataset for subsequent steps.

        Parameters
        ----------
        step_info : ParsedStep
            Parsed step containing the operator and original step configuration.
        dataset : SpectroDataset
            The dataset to split.
        context : ExecutionContext
            Current execution context.
        runtime_context : RuntimeContext
            Runtime context with global settings.
        source : int
            Source index (-1 for combined sources).
        mode : str
            Execution mode ("train", "predict", or "explain").
        loaded_binaries : Any
            Pre-loaded binary data (not used).
        prediction_store : Any
            Store for predictions (not used).
        """
        from nirs4all.pipeline.execution.result import StepOutput

        op = step_info.operator

        # Extract grouping parameters from step dict
        group_by = None
        ignore_repetition = False
        aggregation = "mean"
        y_aggregation = None

        if isinstance(step_info.original_step, dict):
            group_by = step_info.original_step.get("group_by")
            ignore_repetition = step_info.original_step.get("ignore_repetition", False)
            aggregation = step_info.original_step.get("aggregation", "mean")
            y_aggregation = step_info.original_step.get("y_aggregation")

        # In predict/explain mode, skip fold splitting entirely
        if mode == "predict" or mode == "explain":
            # Don't filter by partition - prediction data may be in "test" partition
            local_context = context.copy()
            local_context.selector.partition = None
            needs_y, needs_g = _needs(op)
            X_raw = dataset.x(local_context, layout="2d", concat_source=True)
            X = np.asarray(X_raw) if isinstance(X_raw, list) else X_raw
            n_samples = X.shape[0]

            # Build minimal kwargs for get_n_splits
            predict_kwargs: dict[str, Any] = {}
            if needs_y:
                predict_y = dataset.y(local_context)
                if predict_y is not None:
                    predict_kwargs["y"] = predict_y

            n_folds = op.get_n_splits(**predict_kwargs) if hasattr(op, "get_n_splits") else 1
            dataset.set_folds([(list(range(n_samples)), [])] * n_folds)
            return context, StepOutput()

        # Extract group column specification from step dict (train mode only)
        group_column = None
        if isinstance(step_info.original_step, dict) and "group" in step_info.original_step:
            group_column = step_info.original_step["group"]
            if not isinstance(group_column, str):
                raise TypeError(
                    f"Group column must be a string, got {type(group_column).__name__}"
                )

            # Warn if 'group' is used with a non-native-group splitter
            # These splitters will silently ignore the groups parameter
            if not _is_native_group_splitter(op):
                splitter_name = op.__class__.__name__
                warnings.warn(
                    f"'group' parameter specified with {splitter_name}, which does not "
                    f"natively support groups. The 'group' parameter will be ignored.\n"
                    f"Use 'repetition' parameter in DatasetConfigs for automatic grouping, "
                    f"or 'group_by' for explicit grouping:\n"
                    f"    DatasetConfigs(folder, repetition='{group_column}')\n"
                    f"    {{'split': {splitter_name}(...), 'group_by': '{group_column}'}}",
                    UserWarning,
                    stacklevel=2
                )

        use_effective_groups = False  # Track if we should use compute_effective_groups

        # Check if we should compute effective groups (from repetition + group_by)
        has_repetition_or_group_by = (
            (dataset.repetition and not ignore_repetition) or
            group_by is not None
        )

        if has_repetition_or_group_by:
            use_effective_groups = True

        local_context = context.with_partition("train")
        needs_y, needs_g = _needs(op)
        # IMPORTANT: Only split on base samples (exclude augmented) to prevent data leakage
        X_raw_train = dataset.x(local_context, layout="2d", concat_source=True, include_augmented=False)
        X = np.asarray(X_raw_train) if isinstance(X_raw_train, list) else X_raw_train

        # Get the actual sample IDs from the indexer - these will be used to store folds
        # with absolute sample IDs instead of positional indices, so folds remain valid
        # even if samples are excluded later by sample_filter
        base_sample_ids = dataset._indexer.x_indices(  # noqa: SLF001
            local_context.selector, include_augmented=False, include_excluded=False
        )

        y: np.ndarray | None = None
        if needs_y or use_effective_groups:
            # Get y for splitters that need it, or for effective_groups wrapper
            y = dataset.y(local_context, include_augmented=False)

        # Get groups from metadata if available
        # Priority: use_effective_groups > group (legacy)
        groups = None

        if use_effective_groups:
            # NEW Phase 2: Use compute_effective_groups to combine repetition + group_by
            # This is the primary path for automatic grouping
            groups = compute_effective_groups(
                dataset=dataset,
                group_by=group_by,
                ignore_repetition=ignore_repetition,
                context=local_context,
                include_augmented=False
            )
            if groups is not None and len(groups) != X.shape[0]:
                raise ValueError(
                    f"Effective groups array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
                )

        elif needs_g and (group_column is not None or _is_native_group_splitter(op)):
            # Legacy 'group' parameter for native group splitters
            # Only extract groups if:
            # 1. Explicit group column specified (user requested grouping), OR
            # 2. Splitter is a native group splitter (GroupKFold, etc.) that requires groups
            if group_column is not None:
                # Explicit group column specified - validate and extract
                if not hasattr(dataset, 'metadata_columns') or not dataset.metadata_columns:
                    raise ValueError(
                        f"Group column '{group_column}' specified but dataset has no metadata columns."
                    )
                if group_column not in dataset.metadata_columns:
                    raise ValueError(
                        f"Group column '{group_column}' not found in metadata.\n"
                        f"Available columns: {dataset.metadata_columns}"
                    )
                # Extract groups from specified column (base samples only)
                try:
                    groups = dataset.metadata_column(group_column, local_context, include_augmented=False)
                    if len(groups) != X.shape[0]:
                        raise ValueError(
                            f"Group array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to extract groups from metadata column '{group_column}': {e}"
                    ) from e
            elif _is_native_group_splitter(op):
                # Native group splitter without explicit group column.
                # Auto-detect from first metadata column if available.
                if hasattr(dataset, 'metadata_columns') and dataset.metadata_columns:
                    group_column = dataset.metadata_columns[0]
                    groups = dataset.metadata_column(group_column, local_context, include_augmented=False)
                    if len(groups) != X.shape[0]:
                        raise ValueError(
                            f"Group array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
                        )

        # Wrap splitter with GroupedSplitterWrapper if groups exist and splitter
        # is NOT a native group splitter (e.g., GroupKFold, StratifiedGroupKFold)
        # Native group splitters handle groups directly via the groups parameter
        requires_wrapper = groups is not None and not _is_native_group_splitter(op)

        if requires_wrapper:
            logger.debug(
                f"Wrapping {op.__class__.__name__} with GroupedSplitterWrapper "
                f"(groups from: {'repetition' if dataset.repetition else 'group_by'})"
            )
            op = GroupedSplitterWrapper(
                splitter=op,
                aggregation=aggregation,
                y_aggregation=y_aggregation
            )
            # Update needs_y and needs_g for the wrapped splitter
            needs_y, needs_g = _needs(op)

        n_samples = X.shape[0]

        # Build kwargs for split()
        kwargs: dict[str, Any] = {}
        if needs_y or use_effective_groups:
            # Provide y for splitters that need it, or for effective_groups wrapper
            if needs_y and y is None:
                raise ValueError(
                    f"{op.__class__.__name__} requires y but dataset.y returned None"
                )
            if y is not None:
                kwargs["y"] = y
        if groups is not None:
            # Provide groups for:
            # 1. Native group splitters (needs_g is True)
            # 2. Effective_groups wrapped splitters (wrapper needs groups)
            kwargs["groups"] = groups

        # Train mode: perform actual fold splitting
        folds = list(op.split(X, **kwargs))  # Convert to list to avoid iterator consumption

        # Convert positional indices to absolute sample IDs
        # This ensures folds remain valid even if samples are excluded later by sample_filter
        sample_id_folds = [
            (base_sample_ids[train_idx].tolist(), base_sample_ids[val_idx].tolist())
            for train_idx, val_idx in folds
        ]

        # If no test partition exists and this is a single-fold split,
        # use the validation set as test partition (not as fold)
        # This is expected behavior for single-fold splitters (e.g., SPXYGFold with n_splits=1)
        # which are designed to create train/test splits, not cross-validation folds
        X_test_raw = dataset.x({"partition": "test"})
        X_test = np.asarray(X_test_raw) if isinstance(X_test_raw, list) else X_test_raw
        if X_test.shape[0] == 0 and len(sample_id_folds) == 1:
            fold_1 = sample_id_folds[0]
            if len(fold_1[1]) > 0:  # Only if there are validation samples
                # Move validation samples to test partition using sample IDs
                dataset._indexer.update_by_indices(
                    fold_1[1], {"partition": "test"}
                )

                # Keep train sample IDs, clear validation (they're now in test partition)
                sample_id_folds = [(fold_1[0], [])]

        # Store the folds in the dataset (using sample IDs, not positional indices)
        dataset.set_folds(sample_id_folds)

        # Leakage warning: Check if groups are split across train/val folds
        # This only applies when groups exist and is a safety check
        if groups is not None and len(sample_id_folds) > 0:
            self._check_group_leakage(groups, folds, base_sample_ids, dataset.repetition)

        # Generate binary output with fold information (using sample IDs)
        headers = [f"fold_{i}" for i in range(len(sample_id_folds))]
        binary = ",".join(headers).encode("utf-8") + b"\n"
        max_train_samples = max(len(train_idx) for train_idx, _ in sample_id_folds)

        for row_idx in range(max_train_samples):
            row_values = []
            for _fold_idx, (train_idx, _val_idx) in enumerate(sample_id_folds):
                if row_idx < len(train_idx):
                    row_values.append(str(train_idx[row_idx]))
                else:
                    row_values.append("")  # Empty cell if this fold has fewer samples
            binary += ",".join(row_values).encode("utf-8") + b"\n"

        # Filename includes group column if used
        # For effective_groups, use the inner splitter's name
        if use_effective_groups and requires_wrapper:
            # Phase 2: effective groups from repetition + group_by
            inner_splitter = op.splitter  # GroupedSplitterWrapper stores inner splitter
            folds_name = f"folds_{inner_splitter.__class__.__name__}"
            # Build group info string
            group_info_parts = []
            if dataset.repetition and not ignore_repetition:
                group_info_parts.append(f"rep-{dataset.repetition}")
            if group_by:
                if isinstance(group_by, str):
                    group_info_parts.append(group_by)
                else:
                    group_info_parts.extend(group_by)
            if group_info_parts:
                folds_name += f"_groups-{'+'.join(group_info_parts)}"
            if aggregation != "mean":
                folds_name += f"_{aggregation}"
            if hasattr(inner_splitter, "random_state"):
                seed = inner_splitter.random_state
                if seed is not None:
                    folds_name += f"_seed{seed}"
        elif use_effective_groups:
            # Native group splitter with effective groups (no wrapper)
            folds_name = f"folds_{op.__class__.__name__}"
            # Build group info string
            group_info_parts = []
            if dataset.repetition and not ignore_repetition:
                group_info_parts.append(f"rep-{dataset.repetition}")
            if group_by:
                if isinstance(group_by, str):
                    group_info_parts.append(group_by)
                else:
                    group_info_parts.extend(group_by)
            if group_info_parts:
                folds_name += f"_groups-{'+'.join(group_info_parts)}"
            if hasattr(op, "random_state"):
                seed = op.random_state
                if seed is not None:
                    folds_name += f"_seed{seed}"
        else:
            folds_name = f"folds_{op.__class__.__name__}"
            if group_column:
                folds_name += f"_group-{group_column}"
            if hasattr(op, "random_state"):
                seed = op.random_state
                if seed is not None:
                    folds_name += f"_seed{seed}"
        # folds_name += ".csv" # Extension handled by StepOutput tuple

        # print(f"Generated {len(folds)} folds.")

        # Create StepOutput with the CSV
        step_output = StepOutput(
            outputs=[(binary, folds_name, "csv")]
        )

        return context, step_output
        # else:
        #     n_folds = operator.get_n_splits(**kwargs) if hasattr(operator, "get_n_splits") else 1
        #     dataset.set_folds([(list(range(n_samples)), [])] * n_folds)
        #     return context, []

    def _check_group_leakage(
        self,
        groups: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        base_sample_ids: np.ndarray,
        repetition_col: str | None = None,
    ) -> None:
        """Check for group leakage across train/val splits and issue warning if detected.

        Group leakage occurs when the same group (e.g., same physical sample)
        appears in both training and validation sets within a fold. This can
        happen if the splitter doesn't properly respect groups.

        Parameters
        ----------
        groups : np.ndarray
            Array of group labels (one per sample).
        folds : list of (train_idx, val_idx) tuples
            The fold indices (positional, not sample IDs).
        base_sample_ids : np.ndarray
            Array mapping positional indices to sample IDs.
        repetition_col : str, optional
            Name of repetition column for warning message.
        """
        leaked_folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if len(val_idx) == 0:
                continue

            # Get groups for train and val samples
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])

            # Check for overlap
            overlap = train_groups & val_groups
            if overlap:
                leaked_folds.append((fold_idx, len(overlap)))

        if leaked_folds:
            col_name = repetition_col or "grouping column"
            fold_info = ", ".join(f"fold {f}: {n} groups" for f, n in leaked_folds[:3])
            if len(leaked_folds) > 3:
                fold_info += f", ... ({len(leaked_folds)} folds total)"

            warnings.warn(
                f"⚠️ Group leakage detected: same groups appear in both train and validation sets.\n"
                f"   Affected folds: {fold_info}\n"
                f"   This may indicate that the splitter is not respecting {col_name} grouping.\n"
                f"   Consider using a group-aware splitter (GroupKFold, StratifiedGroupKFold) "
                f"or check your grouping configuration.",
                UserWarning,
                stacklevel=3,
            )

