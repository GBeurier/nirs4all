"""Host-side ``exclude`` and ``tag`` resolution for the dag-ml backend.

Run nirs4all's real SampleFilters in-process on the CV train pool to compute the excluded /
tagged sample ints (mirroring ExcludeController / TagController), so the dag-ml engine consumes the
result as IDENTITY (sample-int sets) instead of marking the indexer. Augmented children cascade out
with their origin (the origin-boundary invariant).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .detect import _is_exclude_step
from .steps import _taggers_from_step


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
        y_pool = y_pool[order]
        if y_pool.ndim > 1:
            y_pool = y_pool.flatten()
    return x_pool, y_pool


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

    * **Per-filter ``ValueError`` → neutral keep-all** (exclude.py:175-184): a filter that fails to
      fit/mask (e.g. insufficient data) contributes a keep-all mask rather than propagating.
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
            masks.append(filter_obj.get_mask(x_pool, y_pool))
        except ValueError:
            # exclude.py:175-184 — a filter that can't be applied contributes a neutral keep-all mask.
            masks.append(np.ones(len(base_ints), dtype=bool))

    if len(masks) == 1:
        keep_mask = masks[0].copy()
    else:
        stacked = np.stack(masks, axis=0)
        keep_mask = np.all(stacked, axis=0) if filter_mode == "any" else np.any(stacked, axis=0)

    # exclude.py:213-222 — never empty the pool: if all rows would be excluded, keep the first.
    if not keep_mask.any():
        keep_mask[0] = True

    return {int(sample_int) for sample_int, keep in zip(base_ints, keep_mask, strict=True) if not keep}


def _resolve_exclude(pipeline: list[Any], spectro: Any) -> tuple[list[Any], list[int], set[int]]:
    """Consume ALL ``exclude`` steps and return ``(pipeline_without_exclude, cv_pool, excluded)``.

    Mirrors the verified legacy + opt-in semantics:

    * **No exclude step** → ``(pipeline, full_train, set())``.
    * **``keep_in_oof=False`` (default = legacy parity)** → the CV pool is the train universe MINUS
      the excluded ints; excluded samples are absent from the folds AND the envelope (removed from
      the CV universe entirely, matching legacy: the splitter runs over ``include_excluded=False``).
      The native ``excluded`` bit is unused (``excluded`` set is empty for the envelope).
    * **``keep_in_oof=True`` (opt-in, leakage-pure)** → the CV pool is the FULL train universe; the
      excluded ints are marked in the envelope so Phase 1's native bit drops them from each fold's
      TRAIN while keeping them in validation/OOF.

    Multiple ``exclude`` steps are applied SEQUENTIALLY, exactly as legacy: each step's filter fits on
    the CURRENT kept train (``include_excluded=False``), i.e. the pool after the earlier steps'
    exclusions (exclude.py:135-137 reads ``include_excluded=False``), so the excluded set is built
    progressively. The ``keep_in_oof`` flag is honored from any exclude step (consistent across steps
    is the caller's contract). All ``exclude`` steps are removed from the remaining pipeline — none is
    lowered to a dag-ml node (the bridge still raises ``NotImplementedError`` for a raw ``exclude``).

    AUGMENTED CHILDREN — the origin-boundary invariant. Filters fit on BASE samples only (origins);
    a flagged origin then CASCADES to its augmented children, exactly as legacy ``mark_excluded(...,
    cascade_to_augmented=True)`` removes a base sample AND its children from the ``include_excluded=
    False`` train universe (exclude.py:230-234, default ``cascade_to_augmented=True`` at exclude.py:278).
    A child is therefore never excluded without its origin, and an excluded origin never keeps a child
    in the pool. ``cascade_to_augmented`` is honored from the exclude steps (caller's contract, like
    ``keep_in_oof``); cascade is a no-op on a dataset with no augmented children.
    """
    train_ints = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    exclude_steps = [step for step in pipeline if _is_exclude_step(step)]
    if not exclude_steps:
        return pipeline, train_ints, set()

    # {origin_int: [child_int, ...]} over the train universe — base rows self-reference origin
    # (origin == sample), so only augmented children (origin != sample) populate the map.
    children_by_origin: dict[int, list[int]] = {}
    sample_col = [int(s) for s in spectro.index_column("sample", {"partition": "train"})]
    origin_col = [int(o) for o in spectro.index_column("origin", {"partition": "train"})]
    for sample_int, origin_int in zip(sample_col, origin_col, strict=True):
        if sample_int != origin_int:
            children_by_origin.setdefault(origin_int, []).append(sample_int)

    def _cascade(origins: set[int]) -> set[int]:
        return origins | {child for origin in origins for child in children_by_origin.get(origin, [])}

    keep_in_oof = any(bool(step.get("keep_in_oof", False)) for step in exclude_steps)
    cascade_to_augmented = any(bool(step.get("cascade_to_augmented", True)) for step in exclude_steps)
    excluded_origins: set[int] = set()
    for step in exclude_steps:
        # Each step fits on the CURRENT kept train: base origins still kept AND their children that an
        # earlier step's cascade has not already removed (mirrors legacy include_excluded=False).
        cascaded = _cascade(excluded_origins) if cascade_to_augmented else excluded_origins
        current_pool = [sample_int for sample_int in train_ints if sample_int not in cascaded]
        excluded_origins |= _excluded_from_pool(step, spectro, current_pool)

    excluded = _cascade(excluded_origins) if cascade_to_augmented else excluded_origins
    remaining = [step for step in pipeline if not _is_exclude_step(step)]
    if keep_in_oof:
        # Opt-in: keep excluded in the CV universe; mark them excluded in the envelope (native bit)
        # and (host-side) drop them from each fold's TRAIN below so the OOF is leakage-pure.
        return remaining, train_ints, excluded
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
                mask = np.asarray(filter_obj.get_mask(x_pool, y_pool), dtype=bool)
            except ValueError:
                # TagController skips a filter that cannot be applied (e.g. insufficient data).
                continue
            if mask.shape[0] != len(pool):
                raise ValueError(f"tag filter {filter_obj.__class__.__name__} returned {mask.shape[0]} masks for {len(pool)} samples")
            for sample_int, keep in zip(pool, mask, strict=True):
                if keep:
                    continue
                labels = tags_by_sample.setdefault(int(sample_int), [])
                if tag_name not in labels:
                    labels.append(str(tag_name))

    return remaining, tags_by_sample or None
