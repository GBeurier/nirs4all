from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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

SplitGroupHandling = Literal["native", "wrapper"]


@dataclass(frozen=True)
class SplitGroupingCapability:
    group_required: bool
    group_handling: SplitGroupHandling


@dataclass(frozen=True)
class ResolvedSplitGroups:
    capability: SplitGroupingCapability
    group_by: str | list[str] | None
    effective_groups: np.ndarray | None
    uses_repetition: bool
    uses_group_by: bool
    satisfied_by_repetition_only: bool

    @property
    def requires_wrapper(self) -> bool:
        return (
            self.effective_groups is not None
            and self.capability.group_handling == "wrapper"
        )


_NATIVE_REQUIRED_GROUP_CAPABILITY = SplitGroupingCapability(
    group_required=True,
    group_handling="native",
)
_OPTIONAL_WRAPPER_GROUP_CAPABILITY = SplitGroupingCapability(
    group_required=False,
    group_handling="wrapper",
)

# Source of truth for how splitters consume effective groups.
_SPLITTER_GROUPING_CAPABILITIES: dict[str, SplitGroupingCapability] = {
    "GroupKFold": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "GroupShuffleSplit": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "LeaveOneGroupOut": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "LeavePGroupsOut": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "StratifiedGroupKFold": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "BinnedStratifiedGroupKFold": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "SPXYGFold": _NATIVE_REQUIRED_GROUP_CAPABILITY,
    "KFold": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "RepeatedKFold": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "ShuffleSplit": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "StratifiedKFold": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "StratifiedShuffleSplit": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "RepeatedStratifiedKFold": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "LeaveOneOut": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "TimeSeriesSplit": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "KennardStoneSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "SPXYSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "KMeansSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "SPlitSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "KBinsStratifiedSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    "SystematicCircularSplitter": _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
}


def get_split_grouping_capability(splitter: Any) -> SplitGroupingCapability:
    """Return how a splitter consumes effective groups."""
    return _SPLITTER_GROUPING_CAPABILITIES.get(
        splitter.__class__.__name__,
        _OPTIONAL_WRAPPER_GROUP_CAPABILITY,
    )


def _freeze_group_value(value: Any) -> Any:
    """Normalize a group label into a stable, hashable key."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return ("__missing__", "none")

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ("__missing__", "nan")

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return tuple(_freeze_group_value(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_freeze_group_value(item) for item in value)

    return value


def _build_group_constraint(
    dataset: SpectroDataset,
    columns: list[str],
    context: Any | None,
    include_augmented: bool,
) -> np.ndarray:
    """Return one grouping constraint from one or more metadata columns."""
    if len(columns) == 1:
        return dataset.metadata_column(
            columns[0],
            context,
            include_augmented=include_augmented,
        )

    arrays = [
        dataset.metadata_column(col, context, include_augmented=include_augmented)
        for col in columns
    ]
    tuple_groups = [tuple(row) for row in zip(*arrays, strict=False)]
    groups = np.empty(len(tuple_groups), dtype=object)
    groups[:] = tuple_groups
    return groups


def _compute_connected_group_components(
    constraints: list[np.ndarray],
) -> np.ndarray:
    """Resolve multiple grouping constraints into connected component labels."""
    if not constraints:
        return np.array([], dtype=int)

    n_samples = len(constraints[0])
    if any(len(constraint) != n_samples for constraint in constraints):
        raise ValueError("All grouping constraints must have the same length.")

    parent = list(range(n_samples))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for constraint in constraints:
        first_seen_by_value: dict[Any, int] = {}
        for sample_idx, raw_value in enumerate(constraint):
            value = _freeze_group_value(raw_value)
            first_idx = first_seen_by_value.get(value)
            if first_idx is None:
                first_seen_by_value[value] = sample_idx
            else:
                union(first_idx, sample_idx)

    component_ids = np.empty(n_samples, dtype=int)
    root_to_component: dict[int, int] = {}
    next_component_id = 0

    for sample_idx in range(n_samples):
        root = find(sample_idx)
        component_id = root_to_component.get(root)
        if component_id is None:
            component_id = next_component_id
            root_to_component[root] = component_id
            next_component_id += 1
        component_ids[sample_idx] = component_id

    return component_ids

def compute_effective_groups(
    dataset: SpectroDataset,
    group_by: str | list[str] | None = None,
    ignore_repetition: bool = False,
    context: Any | None = None,
    include_augmented: bool = False
) -> np.ndarray | None:
    """Compute final group labels for splitting.

    Applies repetition and explicit ``group_by`` columns as split constraints.
    When both are present, the effective groups are the connected components
    induced by:
    - same repetition value
    - same explicit ``group_by`` value (or tuple of values for multi-column
      ``group_by``)

    This is stricter than a plain tuple of ``(repetition, group_by)`` because
    it guarantees that raw ``group_by`` values cannot leak across folds when
    the dataset repetition column is also enforced.

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
        - For multi-column explicit ``group_by`` without repetition:
          array of tuples (e.g., ``(Site, Year)``)
        - For repetition + explicit ``group_by``:
          connected-component labels enforcing both constraints

    Examples
    --------
    >>> # Repetition only (auto from dataset)
    >>> groups = compute_effective_groups(dataset)
    >>> # groups = ['S1', 'S1', 'S2', 'S2', ...]

    >>> # Repetition + additional grouping
    >>> groups = compute_effective_groups(dataset, group_by=['Year'])
    >>> # groups = connected-component labels enforcing both sample_id and Year

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

    if len(columns_to_use) == 1:
        return dataset.metadata_column(
            columns_to_use[0],
            context,
            include_augmented=include_augmented,
        )

    repetition_column = dataset.repetition if not ignore_repetition else None
    if repetition_column and columns_to_use[0] == repetition_column:
        repetition_constraint = dataset.metadata_column(
            repetition_column,
            context,
            include_augmented=include_augmented,
        )
        explicit_group_columns = columns_to_use[1:]
        explicit_group_constraint = _build_group_constraint(
            dataset,
            explicit_group_columns,
            context,
            include_augmented,
        )
        return _compute_connected_group_components(
            [repetition_constraint, explicit_group_constraint],
        )

    return _build_group_constraint(
        dataset,
        columns_to_use,
        context,
        include_augmented,
    )

def _is_native_group_splitter(splitter: Any) -> bool:
    """Check if splitter has native group support.

    Returns True if the splitter is a known group-aware splitter that
    properly handles the 'groups' parameter directly.
    """
    return get_split_grouping_capability(splitter).group_handling == "native"


def _normalize_group_columns(
    group_by: Any,
) -> str | list[str] | None:
    """Validate and normalize explicit group column input."""
    if group_by is None:
        return None

    if isinstance(group_by, str):
        return group_by or None

    if isinstance(group_by, (list, tuple)):
        normalized: list[str] = []
        for index, column in enumerate(group_by):
            if not isinstance(column, str):
                raise TypeError(
                    "group_by must be a string or a list of strings. "
                    f"Entry {index} has type {type(column).__name__}."
                )
            if column:
                normalized.append(column)
        return normalized or None

    raise TypeError(
        "group_by must be a string or a list of strings, "
        f"got {type(group_by).__name__}."
    )


def _normalize_group_alias(
    group_by: Any,
    legacy_group: Any = None,
) -> str | list[str] | None:
    """Normalize legacy ``group`` onto ``group_by`` with deprecation warning."""
    normalized_group_by = _normalize_group_columns(group_by)
    normalized_legacy_group = _normalize_group_columns(legacy_group)

    if normalized_legacy_group is None:
        return normalized_group_by

    if normalized_group_by is not None and normalized_group_by != normalized_legacy_group:
        raise ValueError(
            "Use only 'group_by'. The legacy 'group' alias cannot be combined with "
            "a different 'group_by' value."
        )

    warnings.warn(
        "The legacy 'group' alias is deprecated and will be removed in a future "
        "release. Use 'group_by' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return normalized_group_by if normalized_group_by is not None else normalized_legacy_group


def resolve_split_groups(
    dataset: SpectroDataset,
    splitter: Any,
    group_by: Any = None,
    *,
    legacy_group: Any = None,
    group_required: bool | None = None,
    ignore_repetition: bool = False,
    context: Any | None = None,
    include_augmented: bool = False,
) -> ResolvedSplitGroups:
    """Resolve effective split groups from repetition + explicit grouping."""
    capability = get_split_grouping_capability(splitter)
    normalized_group_by = _normalize_group_alias(group_by, legacy_group)
    uses_repetition = bool(dataset.repetition and not ignore_repetition)
    uses_group_by = normalized_group_by is not None
    effective_group_required = (
        capability.group_required if group_required is None else group_required
    )

    if effective_group_required and not uses_repetition and not uses_group_by:
        raise ValueError(
            f"{splitter.__class__.__name__} requires an effective group, but neither "
            "dataset repetition nor 'group_by' was provided."
        )

    effective_groups = compute_effective_groups(
        dataset=dataset,
        group_by=normalized_group_by,
        ignore_repetition=ignore_repetition,
        context=context,
        include_augmented=include_augmented,
    )

    satisfied_by_repetition_only = effective_group_required and uses_repetition and not uses_group_by
    if satisfied_by_repetition_only:
        warnings.warn(
            f"{splitter.__class__.__name__} requires an effective group. No explicit "
            f"'group_by' was provided, so the split will use only the configured dataset "
            f"repetition '{dataset.repetition}'.",
            UserWarning,
            stacklevel=3,
        )

    return ResolvedSplitGroups(
        capability=capability,
        group_by=normalized_group_by,
        effective_groups=effective_groups,
        uses_repetition=uses_repetition,
        uses_group_by=uses_group_by,
        satisfied_by_repetition_only=satisfied_by_repetition_only,
    )


def _group_info_parts(
    resolved_groups: ResolvedSplitGroups,
    repetition_column: str | None,
) -> list[str]:
    """Return filename-friendly group labels for the resolved configuration."""
    parts: list[str] = []

    if resolved_groups.uses_repetition and repetition_column:
        parts.append(f"rep-{repetition_column}")

    if isinstance(resolved_groups.group_by, str):
        parts.append(resolved_groups.group_by)
    elif resolved_groups.group_by:
        parts.extend(resolved_groups.group_by)

    return parts

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
        legacy_group = None
        ignore_repetition = False
        aggregation = "mean"
        y_aggregation = None

        if isinstance(step_info.original_step, dict):
            group_by = step_info.original_step.get("group_by")
            legacy_group = step_info.original_step.get("group")
            ignore_repetition = step_info.original_step.get("ignore_repetition", False)
            aggregation = step_info.original_step.get("aggregation", "mean")
            y_aggregation = step_info.original_step.get("y_aggregation")

        # In predict/explain mode, skip fold splitting entirely
        if mode == "predict" or mode == "explain":
            # Don't filter by partition - prediction data may be in "test" partition
            local_context = context.copy()
            local_context.selector.partition = None
            needs_y, _ = _needs(op)
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

        local_context = context.with_partition("train")
        resolved_groups = resolve_split_groups(
            dataset=dataset,
            splitter=op,
            group_by=group_by,
            legacy_group=legacy_group,
            ignore_repetition=ignore_repetition,
            context=local_context,
            include_augmented=False,
        )

        needs_y, _ = _needs(op)
        # IMPORTANT: Only split on base samples (exclude augmented) to prevent data leakage
        X_raw_train = dataset.x(local_context, layout="2d", concat_source=True, include_augmented=False)
        X = np.asarray(X_raw_train) if isinstance(X_raw_train, list) else X_raw_train

        # Get the actual sample IDs from the indexer - these will be used to store folds
        # with absolute sample IDs instead of positional indices, so folds remain valid
        # even if samples are excluded later by sample_filter
        base_sample_ids = dataset._indexer.x_indices(  # noqa: SLF001
            local_context.selector, include_augmented=False, include_excluded=False
        )

        groups = resolved_groups.effective_groups
        if groups is not None and len(groups) != X.shape[0]:
            raise ValueError(
                f"Effective groups array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
            )

        y: np.ndarray | None = None
        if needs_y or groups is not None:
            # Get y for splitters that need it, or for grouped wrapper/native grouped flow.
            y = dataset.y(local_context, include_augmented=False)

        # Wrap splitter with GroupedSplitterWrapper if groups exist and splitter
        # is NOT a native group splitter (e.g., GroupKFold, StratifiedGroupKFold)
        # Native group splitters handle groups directly via the groups parameter
        requires_wrapper = resolved_groups.requires_wrapper

        if requires_wrapper:
            logger.debug(
                f"Wrapping {op.__class__.__name__} with GroupedSplitterWrapper "
                f"(group handling: {resolved_groups.capability.group_handling})"
            )
            op = GroupedSplitterWrapper(
                splitter=op,
                aggregation=aggregation,
                y_aggregation=y_aggregation
            )

        # Build kwargs for split()
        kwargs: dict[str, Any] = {}
        if needs_y:
            if needs_y and y is None:
                raise ValueError(
                    f"{op.__class__.__name__} requires y but dataset.y returned None"
                )
            kwargs["y"] = y
        elif groups is not None and y is not None:
            kwargs["y"] = y
        if groups is not None:
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

        base_splitter = op.splitter if requires_wrapper else op
        folds_name = f"folds_{base_splitter.__class__.__name__}"
        if groups is not None:
            group_info_parts = _group_info_parts(resolved_groups, dataset.repetition)
            if group_info_parts:
                folds_name += f"_groups-{'+'.join(group_info_parts)}"
            if requires_wrapper and aggregation != "mean":
                folds_name += f"_{aggregation}"
        if hasattr(base_splitter, "random_state"):
            seed = base_splitter.random_state
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

