from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Tuple, TYPE_CHECKING, List, Union
import copy
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.operators.splitters import GroupedSplitterWrapper

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from nirs4all.data.dataset import SpectroDataset


# Native group-aware splitter class names (sklearn and nirs4all)
# These splitters have built-in group support and don't need force_group
_NATIVE_GROUP_SPLITTERS = frozenset({
    "GroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "StratifiedGroupKFold",
    "SPXYGFold",
    "BinnedStratifiedGroupKFold",
})


def _is_native_group_splitter(splitter: Any) -> bool:
    """Check if splitter has native group support.

    Returns True if the splitter is a known group-aware splitter that
    properly handles the 'groups' parameter without needing force_group.
    """
    return splitter.__class__.__name__ in _NATIVE_GROUP_SPLITTERS


def _needs(splitter: Any) -> Tuple[bool, bool]:
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
        params: List[inspect.Parameter] = [
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

    def execute(  # type: ignore[override]
        self,
        step_info: 'ParsedStep',
        dataset: "SpectroDataset",
        context: ExecutionContext,
        runtime_context: "RuntimeContext",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ):
        """Run ``operator.split`` and store the resulting folds on *dataset*.

        * Smartly supplies ``y`` / ``groups`` only if required.
        * Extracts groups from metadata if specified.
        * Supports ``force_group`` parameter to wrap any splitter with group-awareness.
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

        Notes
        -----
        The ``force_group`` parameter enables any sklearn-compatible splitter
        to work with grouped samples by wrapping it with ``GroupedSplitterWrapper``.
        This aggregates samples by group, passes "virtual samples" to the splitter,
        and expands fold indices back to the original dataset.

        Example usage::

            {"split": KFold(n_splits=5), "force_group": "Sample_ID"}
            {"split": ShuffleSplit(test_size=0.2), "force_group": "ID", "aggregation": "median"}
        """
        from nirs4all.pipeline.execution.result import StepOutput

        op = step_info.operator

        # Extract force_group and aggregation parameters from step dict
        force_group = None
        aggregation = "mean"
        y_aggregation = None

        if isinstance(step_info.original_step, dict):
            force_group = step_info.original_step.get("force_group")
            aggregation = step_info.original_step.get("aggregation", "mean")
            y_aggregation = step_info.original_step.get("y_aggregation")

        # In predict/explain mode, skip fold splitting entirely
        if mode == "predict" or mode == "explain":
            # Don't filter by partition - prediction data may be in "test" partition
            local_context = context.with_partition(None)
            needs_y, needs_g = _needs(op)
            X = dataset.x(local_context, layout="2d", concat_source=True)
            n_samples = X.shape[0]

            # Build minimal kwargs for get_n_splits
            kwargs: Dict[str, Any] = {}
            if needs_y:
                y = dataset.y(local_context)
                if y is not None:
                    kwargs["y"] = y

            n_folds = op.get_n_splits(**kwargs) if hasattr(op, "get_n_splits") else 1
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
            # Suggest using 'force_group' instead for universal group support
            if not _is_native_group_splitter(op):
                splitter_name = op.__class__.__name__
                warnings.warn(
                    f"⚠️ 'group' parameter specified with {splitter_name}, which does not "
                    f"natively support groups. The 'group' parameter will be ignored.\n"
                    f"💡 Use 'force_group' instead to enable group-aware splitting with any splitter:\n"
                    f"    {{'split': {splitter_name}(...), 'force_group': '{group_column}'}}\n"
                    f"This will ensure all samples from the same group stay together in train/test.",
                    UserWarning,
                    stacklevel=2
                )

        # Handle force_group: wrap the splitter with GroupedSplitterWrapper
        # This enables any sklearn-compatible splitter to work with groups
        force_group_column = None
        force_group_is_y = False  # Track if force_group uses y-binning
        n_bins = 5  # Default bins for y-binning

        if force_group is not None:
            if not isinstance(force_group, str):
                raise TypeError(
                    f"force_group must be a string column name or 'y', got {type(force_group).__name__}"
                )

            # Check if force_group is "y" (special case: use binned y values as groups)
            if force_group.lower() == "y":
                force_group_is_y = True
                # Extract n_bins from step dict if provided
                if isinstance(step_info.original_step, dict):
                    n_bins = step_info.original_step.get("n_bins", 5)
                    if not isinstance(n_bins, int) or n_bins < 2:
                        raise ValueError(
                            f"n_bins must be an integer >= 2, got {n_bins}"
                        )
            else:
                force_group_column = force_group

            # Validate aggregation parameter
            valid_aggregations = ("mean", "median", "first")
            if aggregation not in valid_aggregations:
                raise ValueError(
                    f"aggregation must be one of {valid_aggregations}, got '{aggregation}'"
                )

            # Validate y_aggregation parameter if provided
            valid_y_aggregations = ("mean", "mode", "first", None)
            if y_aggregation not in valid_y_aggregations:
                raise ValueError(
                    f"y_aggregation must be one of {valid_y_aggregations}, got '{y_aggregation}'"
                )

            # Wrap the splitter with GroupedSplitterWrapper
            op = GroupedSplitterWrapper(
                splitter=op,
                aggregation=aggregation,
                y_aggregation=y_aggregation
            )

        local_context = context.with_partition("train")
        needs_y, needs_g = _needs(op)
        # IMPORTANT: Only split on base samples (exclude augmented) to prevent data leakage
        X = dataset.x(local_context, layout="2d", concat_source=True, include_augmented=False)

        # Get the actual sample IDs from the indexer - these will be used to store folds
        # with absolute sample IDs instead of positional indices, so folds remain valid
        # even if samples are excluded later by sample_filter
        base_sample_ids = dataset._indexer.x_indices(  # noqa: SLF001
            local_context.selector, include_augmented=False, include_excluded=False
        )

        y = None
        if needs_y or force_group_column is not None or force_group_is_y:
            # Get y for splitters that need it, or for force_group (wrapper may need it)
            y = dataset.y(local_context, include_augmented=False)

        # Get groups from metadata if available
        # For force_group: always extract groups (wrapper requires them)
        # For native group splitters: extract if needs_g is True
        groups = None
        effective_group_column = force_group_column or group_column

        if force_group_is_y:
            # force_group: "y" - use binned y values as groups
            # This enables stratification on continuous targets
            if y is None:
                raise ValueError(
                    "force_group='y' specified but dataset.y returned None"
                )
            # Bin y values into n_bins quantile bins for group-aware splitting
            # Each bin becomes a "group" that won't be split across train/test
            groups = self._bin_y_for_groups(y, n_bins)
        elif force_group_column is not None:
            # force_group: always extract groups for the wrapper
            if not hasattr(dataset, 'metadata_columns') or not dataset.metadata_columns:
                raise ValueError(
                    f"force_group='{force_group_column}' specified but dataset has no metadata columns."
                )
            if force_group_column not in dataset.metadata_columns:
                raise ValueError(
                    f"force_group column '{force_group_column}' not found in metadata.\n"
                    f"Available columns: {dataset.metadata_columns}"
                )
            try:
                groups = dataset.metadata_column(force_group_column, local_context, include_augmented=False)
                if len(groups) != X.shape[0]:
                    raise ValueError(
                        f"Group array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to extract groups from force_group column '{force_group_column}': {e}"
                ) from e
        elif needs_g and (group_column is not None or _is_native_group_splitter(op)):
            # Only extract groups if:
            # 1. Explicit group column specified (user requested grouping), OR
            # 2. Splitter is a native group splitter (GroupKFold, etc.) that requires groups
            # Note: Many sklearn splitters (KFold, ShuffleSplit, etc.) have 'groups' parameter
            # for API compatibility, but don't require it. We should NOT auto-assign groups for those.
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
            elif hasattr(dataset, 'metadata_columns') and dataset.metadata_columns:
                # No explicit group column, but metadata available - use first column as default
                group_column = dataset.metadata_columns[0]
                logger.warning(
                    f"{op.__class__.__name__} has 'groups' parameter but no 'group' specified. "
                    f"Using default: '{group_column}'"
                )
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
            # else: No group column specified and no metadata available
            # Leave groups=None and let the splitter handle it
            # (will work for splitters that don't require groups, will fail for those that do)

        n_samples = X.shape[0]

        # Build kwargs for split()
        kwargs: Dict[str, Any] = {}
        if needs_y or force_group_column is not None or force_group_is_y:
            # Provide y for splitters that need it, or for force_group wrapper
            if needs_y and y is None:
                raise ValueError(
                    f"{op.__class__.__name__} requires y but dataset.y returned None"
                )
            if y is not None:
                # Special case: force_group="y" with StratifiedKFold
                # Pass bin labels (groups) as y for stratification on continuous targets
                if force_group_is_y and "Stratified" in step_info.operator.__class__.__name__:
                    # Use bin labels for stratification instead of continuous y
                    kwargs["y"] = groups.astype(int)
                else:
                    kwargs["y"] = y
        if groups is not None:
            # Provide groups for:
            # 1. Native group splitters (needs_g is True)
            # 2. force_group wrapped splitters (wrapper needs groups)
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
        if dataset.x({"partition": "test"}).shape[0] == 0 and len(sample_id_folds) == 1:
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

        # Generate binary output with fold information (using sample IDs)
        headers = [f"fold_{i}" for i in range(len(sample_id_folds))]
        binary = ",".join(headers).encode("utf-8") + b"\n"
        max_train_samples = max(len(train_idx) for train_idx, _ in sample_id_folds)

        for row_idx in range(max_train_samples):
            row_values = []
            for fold_idx, (train_idx, val_idx) in enumerate(sample_id_folds):
                if row_idx < len(train_idx):
                    row_values.append(str(train_idx[row_idx]))
                else:
                    row_values.append("")  # Empty cell if this fold has fewer samples
            binary += ",".join(row_values).encode("utf-8") + b"\n"

        # Filename includes group column if used
        # For force_group, use the inner splitter's name
        if force_group_column is not None or force_group_is_y:
            inner_splitter = op.splitter  # GroupedSplitterWrapper stores inner splitter
            folds_name = f"folds_{inner_splitter.__class__.__name__}"
            if force_group_is_y:
                folds_name += f"_force_group-y_bins{n_bins}"
            else:
                folds_name += f"_force_group-{force_group_column}"
            if aggregation != "mean":
                folds_name += f"_{aggregation}"
            if hasattr(inner_splitter, "random_state"):
                seed = getattr(inner_splitter, "random_state")
                if seed is not None:
                    folds_name += f"_seed{seed}"
        else:
            folds_name = f"folds_{op.__class__.__name__}"
            if group_column:
                folds_name += f"_group-{group_column}"
            if hasattr(op, "random_state"):
                seed = getattr(op, "random_state")
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

    def _bin_y_for_groups(self, y, n_bins: int = 5):
        """Bin continuous y values into quantile-based groups for stratified splitting.

        This method enables stratification on continuous targets by binning y values
        into quantiles. Each bin becomes a "pseudo-group" that ensures samples with
        similar y values are kept together during splitting, enabling balanced
        distribution of target values across folds.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Continuous target values to bin.
        n_bins : int, default=5
            Number of quantile bins to create. More bins = finer stratification
            but may fail with small datasets. Recommended: 3-10 bins.

        Returns
        -------
        groups : ndarray of shape (n_samples,)
            Integer bin labels (0 to n_bins-1) for each sample.

        Notes
        -----
        Uses quantile-based binning (pd.qcut equivalent) to ensure approximately
        equal number of samples per bin, regardless of y value distribution.

        If there are fewer unique y values than n_bins, reduces to unique value
        binning (each unique value = one bin).

        Examples
        --------
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> groups = self._bin_y_for_groups(y, n_bins=5)
        >>> # groups will be [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] (approximately)
        """
        import numpy as np

        y = np.asarray(y).ravel()
        n_samples = len(y)

        # Handle edge cases
        n_unique = len(np.unique(y))
        if n_unique <= n_bins:
            # Fewer unique values than bins - use unique values as groups
            _, groups = np.unique(y, return_inverse=True)
            return groups

        # Quantile-based binning for balanced bin sizes
        # This ensures each bin has approximately equal number of samples
        try:
            # Compute quantile edges
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(y, quantiles)

            # Make edges unique to avoid empty bins
            bin_edges = np.unique(bin_edges)

            # If we lost bins due to non-unique edges, adjust
            actual_bins = len(bin_edges) - 1
            if actual_bins < 2:
                # Fall back to unique value binning
                _, groups = np.unique(y, return_inverse=True)
                return groups

            # Digitize y values into bins (1-indexed, so subtract 1)
            # np.digitize returns indices 1 to n_bins, we want 0 to n_bins-1
            groups = np.digitize(y, bin_edges[1:-1], right=False)

            return groups

        except Exception:
            # Fallback: equal-width binning
            y_min, y_max = y.min(), y.max()
            if y_min == y_max:
                # All values identical
                return np.zeros(n_samples, dtype=int)

            bin_width = (y_max - y_min) / n_bins
            groups = ((y - y_min) / bin_width).astype(int)
            groups = np.clip(groups, 0, n_bins - 1)  # Handle edge case of y == y_max

            return groups

