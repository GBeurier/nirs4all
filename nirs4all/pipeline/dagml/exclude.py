"""Host-side ``exclude`` and ``tag`` resolution for the dag-ml backend.

Run nirs4all's real SampleFilters in-process on the CV train pool to compute the excluded /
tagged sample ints (mirroring ExcludeController / TagController), so the dag-ml engine consumes the
result as IDENTITY (sample-int sets) instead of marking the indexer. Augmented children cascade out
with their origin (the origin-boundary invariant).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nirs4all.operators.filters.base import filter_targets
from nirs4all.operators.filters.metadata import MetadataFilter

from .detect import _is_exclude_step
from .steps import _taggers_from_step


class FoldLocalExclusion(set[int]):
    """The ``keep_in_oof=True`` exclusion: members are the full-train (refit) exclusion.

    Each CV fold refits the exclude steps on its own train rows (:meth:`apply`), so validation
    targets never decide which rows a fold trains on; the members mark the envelope for lineage
    and the refit, and the fold set declares ``train_exclusion: "fold_local"`` so DAG-ML trains
    each fold on its host train list. Truthy even when empty: a fold may exclude rows that the
    full-train fit keeps.
    """

    steps: tuple[Any, ...]
    cascade_to_augmented: bool
    children_by_origin: dict[int, list[int]]

    def __init__(self, members: set[int], steps: list[Any], cascade_to_augmented: bool, children_by_origin: dict[int, list[int]]) -> None:
        super().__init__(members)
        self.steps = tuple(steps)
        self.cascade_to_augmented = cascade_to_augmented
        self.children_by_origin = children_by_origin

    def __bool__(self) -> bool:
        return True

    def apply(self, spectro: Any, folds: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
        """``folds`` with each train list minus the exclusion fitted on that train list alone."""
        from .folds import FoldLocalFolds

        trimmed = FoldLocalFolds()
        for train, validation in folds:
            dropped = _sequential_exclusion(list(self.steps), spectro, list(train), self.cascade_to_augmented, self.children_by_origin)
            trimmed.append(([sample for sample in train if sample not in dropped], list(validation)))
        return trimmed


def _base_pool_ints(spectro: Any, pool_ints: list[int]) -> list[int]:
    """The non-augmented (base/origin) subset of ``pool_ints``, preserving order.

    A base sample self-references its ``origin`` (``origin == sample``); an augmented child has
    ``origin != sample`` (indexer.py:310-312). The legacy ExcludeController fits its SampleFilters on
    BASE samples only (``include_augmented=False`` at exclude.py:135-137,146-149), so the fitting pool
    here must drop the augmented children — their exclusion is inherited from their origin (cascade).
    """
    origin_of = {int(s): int(o) for s, o in zip(spectro.index_column("sample", {}), spectro.index_column("origin", {}), strict=True)}
    return [int(sample_int) for sample_int in pool_ints if origin_of.get(int(sample_int), int(sample_int)) == int(sample_int)]


def _filter_data_for_pool(spectro: Any, base_ints: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
    """X/y for SampleFilter fitting, aligned exactly to ``base_ints`` order.

    ``base_ints`` MUST be non-augmented (base/origin) sample ints — see :func:`_base_pool_ints`. The
    ``include_augmented=False`` X/y rows are base-grain, so requesting only base ints keeps x_pool,
    y_pool, ``stored`` and ``order`` consistent (a child id in the request would have no base row to
    map to and crash the re-key).
    """
    x_pool = np.asarray(spectro.x({"sample": list(base_ints)}, layout="2d", concat_source=True, include_augmented=False))
    y_values = spectro.y({"sample": list(base_ints)}, include_augmented=False)
    y_pool = None if y_values is None else np.asarray(y_values)

    # `spectro.x/y({"sample": ids})` returns ascending storage order, not request order; re-key so
    # masks align to `base_ints` exactly (the storage-vs-request trap the resolver also guards against).
    stored = spectro.index_column("sample", {"sample": list(base_ints)})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    order = [row_of[int(sample_int)] for sample_int in base_ints]
    x_pool = x_pool[order]
    if y_pool is not None and y_pool.size:
        y_pool = filter_targets(y_pool[order])
    return x_pool, y_pool


def _filter_mask_for_pool(filter_obj: Any, spectro: Any, pool: list[int], X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
    """Apply a filter with metadata in the same requested row order as X/y."""
    if isinstance(filter_obj, MetadataFilter):
        selector = {"sample": list(pool)}
        metadata = spectro.metadata(selector, include_augmented=False)
        stored = spectro.index_column("sample", selector)
        row_of = {int(sample): row for row, sample in enumerate(stored)}
        metadata = metadata[[row_of[int(sample)] for sample in pool]]
        return np.asarray(filter_obj.get_mask(X, y, metadata=metadata))
    return np.asarray(filter_obj.get_mask(X, y))


def _excluded_from_pool(exclude_step: dict[str, Any], spectro: Any, pool_ints: list[int]) -> set[int]:
    """Excluded BASE sample ints from ``pool_ints`` for one ``exclude_step``, mirroring ExcludeController.

    Fits each :class:`~nirs4all.operators.filters.base.SampleFilter` on the CURRENT kept pool's BASE
    X/y (``include_augmented=False``) and combines the per-filter keep-masks by ``mode`` — exactly the
    legacy :class:`~nirs4all.controllers.data.exclude.ExcludeController` mask logic:

    * ``mode="any"`` → exclude if ANY filter flags = ``np.all`` of the keep-masks (exclude.py:193);
    * ``mode="all"`` → exclude only if ALL filters flag = ``np.any`` (exclude.py:196).

    The filters fit on the BASE (origin) rows ONLY — :func:`_base_pool_ints` drops any augmented child
    ids from ``pool_ints`` first, matching legacy (exclude.py:135-137,146-149 select base samples via
    ``include_augmented=False``). The returned set is therefore base origin ints; the augmented children
    of a flagged origin are cascaded out by the caller (:func:`_resolve_exclude`), never flagged here on
    their own — the origin-boundary invariant.

    Two legacy edge behaviors are replicated:

    * **A filter that cannot be applied fails the step** (``ExcludeController``): its ``ValueError``
      is re-raised with the filter name, never replaced by a keep-all mask.
    * **All-excluded guard** (exclude.py:213-222): if the COMBINED keep-mask would exclude every row,
      keep the first sample so exclusion never empties the pool.

    The engine consumes the result as identity (a sample-int set) instead of marking the indexer.
    """
    from nirs4all.controllers.data.exclude import ExcludeController

    controller = ExcludeController()
    filters, filter_mode, _cascade = controller._parse_config(exclude_step)  # noqa: SLF001 - reuse legacy parsing
    if not filters:
        raise ValueError("exclude keyword requires at least one filter")
    base_ints = _base_pool_ints(spectro, pool_ints)
    if not base_ints:
        return set()

    x_pool, y_pool = _filter_data_for_pool(spectro, base_ints)
    if y_pool is None or y_pool.size == 0:
        return set()

    masks: list[np.ndarray] = []
    for filter_obj in filters:
        try:
            filter_obj.fit(x_pool, y_pool)
            masks.append(_filter_mask_for_pool(filter_obj, spectro, base_ints, x_pool, y_pool))
        except ValueError as error:
            raise ValueError(f"{filter_obj.__class__.__name__} could not be applied: {error}") from error

    if len(masks) == 1:
        keep_mask = masks[0].copy()
    else:
        stacked = np.stack(masks, axis=0)
        keep_mask = np.all(stacked, axis=0) if filter_mode == "any" else np.any(stacked, axis=0)

    # exclude.py:213-222 — never empty the pool: if all rows would be excluded, keep the first.
    if not keep_mask.any():
        keep_mask[0] = True

    return {int(sample_int) for sample_int, keep in zip(base_ints, keep_mask, strict=True) if not keep}


def _sequential_exclusion(
    exclude_steps: list[Any],
    spectro: Any,
    pool: list[int],
    cascade_to_augmented: bool,
    children_by_origin: dict[int, list[int]],
    chart_stages: list[tuple[list[int], str, bool]] | None = None,
) -> set[int]:
    """Sample ints of ``pool`` excluded by ``exclude_steps`` fitted on ``pool`` alone.

    Steps apply SEQUENTIALLY, exactly as legacy: each step fits on the rows the earlier steps kept
    (base origins still kept AND their children not already cascaded out). Flagged origins cascade
    to their augmented children when ``cascade_to_augmented``.
    """

    def _cascade(origins: set[int]) -> set[int]:
        if not cascade_to_augmented:
            return origins
        return origins | {child for origin in origins for child in children_by_origin.get(origin, [])}

    excluded_origins: set[int] = set()
    for step in exclude_steps:
        cascaded = _cascade(excluded_origins)
        current_pool = [sample_int for sample_int in pool if sample_int not in cascaded]
        newly_excluded = _excluded_from_pool(step, spectro, current_pool)
        excluded_origins |= newly_excluded
        if chart_stages is not None:
            from nirs4all.controllers.data.exclude import ExcludeController

            controller = ExcludeController()
            filters, filter_mode, _ = controller._parse_config(step)  # noqa: SLF001 - legacy reason contract
            names = [controller._get_filter_name(item) for item in filters]  # noqa: SLF001
            reason = filters[0].exclusion_reason if len(filters) == 1 else f"exclude({filter_mode}:{','.join(names)})"
            chart_stages.append((sorted(newly_excluded), reason, cascade_to_augmented))
    return _cascade(excluded_origins)


def _resolve_exclude(pipeline: list[Any], spectro: Any) -> tuple[list[Any], list[int], set[int]]:
    """Consume ALL ``exclude`` steps and return ``(pipeline_without_exclude, cv_pool, excluded)``.

    * **No exclude step** → ``(pipeline, full_train, set())``.
    * **``keep_in_oof=False`` (default = legacy parity)** → dataset cleaning before CV: the filters
      fit on the full train and the CV pool is the train universe MINUS the excluded ints, so
      excluded samples are absent from the folds (train AND validation) and the envelope, matching
      legacy (the splitter runs over ``include_excluded=False``).
    * **``keep_in_oof=True`` (opt-in)** → the CV pool is the FULL train universe and ``excluded`` is
      a :class:`FoldLocalExclusion`: each fold refits the filters on its own train rows and drops
      what they flag from that train only (validation keeps every sample, predicted in the OOF);
      the full-train fit marks the envelope and drives the refit.

    Multiple ``exclude`` steps apply SEQUENTIALLY (see :func:`_sequential_exclusion`). The
    ``keep_in_oof`` and ``cascade_to_augmented`` flags are honored from any exclude step (consistent
    across steps is the caller's contract). All ``exclude`` steps are removed from the remaining
    pipeline — none is lowered to a dag-ml node.

    AUGMENTED CHILDREN — the origin-boundary invariant. Filters fit on BASE samples only (origins);
    a flagged origin then CASCADES to its augmented children, exactly as legacy ``mark_excluded(...,
    cascade_to_augmented=True)`` (exclude.py:230-234). A child is therefore never excluded without
    its origin, and an excluded origin never keeps a child in the pool.
    """
    train_ints = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    exclude_steps = [step for step in pipeline if _is_exclude_step(step)]
    if not exclude_steps:
        return pipeline, train_ints, set()

    # {origin_int: [child_int, ...]} over the train universe — base rows self-reference origin
    # (origin == sample), so only augmented children (origin != sample) populate the map.
    children_by_origin: dict[int, list[int]] = {}
    origin_col = [int(o) for o in spectro.index_column("origin", {"partition": "train"})]
    for sample_int, origin_int in zip(train_ints, origin_col, strict=True):
        if sample_int != origin_int:
            children_by_origin.setdefault(origin_int, []).append(sample_int)

    keep_in_oof = any(bool(step.get("keep_in_oof", False)) for step in exclude_steps)
    cascade_to_augmented = any(bool(step.get("cascade_to_augmented", True)) for step in exclude_steps)
    capture_charts = getattr(spectro, "_dagml_capture_exclusion_charts", False)
    chart_stages: list[tuple[list[int], str, bool]] | None = [] if capture_charts else None
    excluded = _sequential_exclusion(exclude_steps, spectro, train_ints, cascade_to_augmented, children_by_origin, chart_stages)
    if capture_charts:
        spectro._dagml_exclusion_chart_stages = chart_stages

    remaining = [step for step in pipeline if not _is_exclude_step(step)]
    if keep_in_oof:
        return remaining, train_ints, FoldLocalExclusion(excluded, exclude_steps, cascade_to_augmented, children_by_origin)
    # Default (legacy): drop excluded from the CV universe entirely; envelope marks nothing excluded.
    pool = [sample_int for sample_int in train_ints if sample_int not in excluded]
    return remaining, pool, set()


def _resolve_tags(pipeline: list[Any], spectro: Any, pool: list[int]) -> tuple[list[Any], dict[int, list[str]] | None]:
    """Consume handled ``tag`` steps and return ``(pipeline_without_tag, tags_by_sample)``.

    Each tag filter is fit once on the CV train pool only (leakage-safe, matching TagController's
    train-context fit) and marks the samples it flags, i.e. ``SampleFilter.get_mask`` false values.
    Unlike ``exclude``, this never changes the CV universe: the same ``pool`` is returned to the
    splitter and model training path, with tag labels carried only on the envelope relations.
    """
    parsed_steps: list[tuple[int, list[tuple[str, Any]]]] = []
    for index, step in enumerate(pipeline):
        taggers = _taggers_from_step(step)
        if taggers is not None:
            parsed_steps.append((index, taggers))
    if not parsed_steps:
        return pipeline, None

    consumed = {index for index, _ in parsed_steps}
    remaining = [step for index, step in enumerate(pipeline) if index not in consumed]
    if not pool:
        return remaining, None

    x_pool, y_pool = _filter_data_for_pool(spectro, pool)
    tags_by_sample: dict[int, list[str]] = {}
    for _, taggers in parsed_steps:
        for tag_name, filter_obj in taggers:
            try:
                filter_obj.fit(x_pool, y_pool)
                mask = np.asarray(_filter_mask_for_pool(filter_obj, spectro, pool, x_pool, y_pool), dtype=bool)
            except ValueError as error:
                raise ValueError(f"{filter_obj.__class__.__name__} could not be applied: {error}") from error
            if mask.shape[0] != len(pool):
                raise ValueError(f"tag filter {filter_obj.__class__.__name__} returned {mask.shape[0]} masks for {len(pool)} samples")
            for sample_int, keep in zip(pool, mask, strict=True):
                if keep:
                    continue
                labels = tags_by_sample.setdefault(int(sample_int), [])
                if tag_name not in labels:
                    labels.append(str(tag_name))

    return remaining, tags_by_sample or None
