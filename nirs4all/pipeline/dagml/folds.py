"""Host-side CV split + fold construction for the dag-ml backend.

dag-ml has no runtime splitter, so the host owns the split: fetch the real X/y for the CV pool
(identity-keyed, in request order), call the splitter, and materialize the FoldSet's sample-int
folds — including group-aware folds for repetition datasets and the exclude-aware fold variant.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _pool_features(spectro: Any, pool: list[int]) -> np.ndarray:
    """Real X for the CV pool, in ``pool`` order — what a distance/clustering splitter needs.

    ``x_rows`` addresses each stored row by its own sample int and returns rows in request order
    (identity-keyed, never positional), so the matrix aligns to ``pool`` exactly — the same float32
    spectra the legacy :class:`CrossValidatorController` feeds ``splitter.split``.
    """
    return np.asarray(spectro.x_rows(pool, layout="2d"))


def _pool_targets(spectro: Any, pool: list[int]) -> np.ndarray:
    """Real y for the CV pool, in ``pool`` order — what a supervised splitter (SPXY, KBins) needs.

    ``spectro.y({"sample": ids})`` returns ASCENDING STORAGE order, not request order, so the block is
    re-keyed to ``pool`` order via ``index_column`` (the storage-vs-request trap the resolver guards
    against). Returned 1D to match the legacy controller's ``dataset.y`` feed.
    """
    block = np.asarray(spectro.y({"sample": pool}, include_augmented=False))
    stored = spectro.index_column("sample", {"sample": pool})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    order = [row_of[int(sample_int)] for sample_int in pool]
    block = block[order]
    return block.ravel() if block.ndim > 1 and block.shape[1] == 1 else block


def _split_pool(splitter: Any, spectro: Any, pool: list[int]) -> list[tuple[Any, Any]]:
    """Call ``splitter.split`` with the REAL X (+ y when supervised), mirroring the legacy controller.

    The host owns the split — dag-ml has no runtime splitter — so this is where a distance/supervised
    NIRS calibration splitter (KennardStone, SPXY, KMeans, SPXYFold, KBinsStratified) actually sees the
    spectra/targets it partitions on. Index-only splitters (KFold/ShuffleSplit) use only ``len(X)`` and
    ignore y, so passing the real X (same row count) and y is a no-op for them — folds are identical to
    splitting on a bare index list, which the parity gate verifies.

    ``y`` is supplied whenever the splitter's ``split`` signature declares it (``_needs`` is true for
    every sklearn-contract splitter) and the dataset has targets — exactly the legacy feed. ``groups``
    are NOT resolved here: the dag-ml engine path carries no ``group_by``/repetition group source yet
    (backlog #21), so a splitter that genuinely REQUIRES a group (``group_required`` capability, e.g.
    ``BinnedStratifiedGroupKFold``) is rejected loud rather than split without its group constraint.
    Optional-group splitters (``SPXYGFold``, a ``GroupedSplitterWrapper``) run group-free here — they
    degrade to their non-grouped behavior (matching the legacy controller when no group source exists)
    and gain the group constraint once #21 wires it.
    """
    from nirs4all.controllers.splitters.split import _needs, get_split_grouping_capability

    if get_split_grouping_capability(splitter).group_required:
        raise NotImplementedError(
            f"engine='dag-ml' does not yet wire group constraints (group_by/repetition) into the split; "
            f"{splitter.__class__.__name__} requires a group (backlog #21)."
        )

    features = _pool_features(spectro, pool)
    needs_y, _ = _needs(splitter)
    kwargs: dict[str, Any] = {}
    if needs_y:
        targets = _pool_targets(spectro, pool)
        if targets.size:
            kwargs["y"] = targets
    return list(splitter.split(features, **kwargs))


def _is_repetition_dataset(spectro: Any) -> bool:
    """True when the dataset declares a repetition column (sample-grain grouping of replicate rows)."""
    return bool(getattr(spectro, "repetition", None))


def _repetition_groups_for_pool(spectro: Any, pool: list[int]) -> np.ndarray:
    """Per-row group value for ``pool`` (the repetition column), aligned to ``pool`` order.

    ``compute_effective_groups`` returns the group vector in ASCENDING STORAGE order (the
    metadata column order); re-key it to ``pool`` request order so it aligns 1:1 with the
    ``_pool_features``/``_pool_targets`` matrices the splitter sees — the storage-vs-request
    trap the resolver guards against. The result is the ``groups`` vector a group-aware
    splitter (or ``GroupedSplitterWrapper``) consumes so all replicates of a sample stay
    together on one fold side.
    """
    from nirs4all.controllers.splitters.split import compute_effective_groups

    groups_all = compute_effective_groups(spectro)
    if groups_all is None:
        raise ValueError("repetition dataset has no effective groups (no repetition/group_by column)")
    stored = spectro.index_column("sample", {})
    group_of_sample = {int(sample_int): groups_all[row] for row, sample_int in enumerate(stored)}
    return np.array([group_of_sample[sample_int] for sample_int in pool], dtype=object)


def _repetition_grain(spectro: Any, pool: list[int]) -> dict[int, str]:
    """``{sample_int: group_value}`` for ``pool`` — the ``group_id`` emitted onto the relations.

    The group value is stringified so it is a stable dag-ml-data id token; dag-ml-data then
    refuses any fold that splits a group across train/validation (native group-leakage check).
    """
    groups = _repetition_groups_for_pool(spectro, pool)
    return {sample_int: str(group) for sample_int, group in zip(pool, groups, strict=True)}


def _build_group_folds(splitter: Any, spectro: Any, pool: list[int]) -> list[tuple[list[int], list[int]]]:
    """Group-aware folds over ``pool``: all repetitions of a sample land on the SAME fold side.

    Mirrors the legacy :class:`CrossValidatorController` fold construction for a repetition /
    group-by dataset: resolve the per-row group vector, wrap an index-only splitter with
    :class:`GroupedSplitterWrapper` (a native group splitter — ``GroupKFold`` etc. — consumes
    ``groups`` directly), split the REAL X/y (in ``pool`` order, so a distance/supervised
    splitter still partitions on real data), and map the positional fold indices back to ``pool``
    sample ints. Each repetition ROW is its own sample int in the resulting folds; because a
    group is never split, every rep row is validated exactly once → a clean OOF partition that
    dag-ml-data accepts, with the group constraint enforced by the emitted ``group_id``.
    """
    from nirs4all.controllers.splitters.split import _needs, get_split_grouping_capability
    from nirs4all.operators.splitters import GroupedSplitterWrapper

    groups = _repetition_groups_for_pool(spectro, pool)
    features = _pool_features(spectro, pool)
    needs_y, _ = _needs(splitter)

    op = splitter
    if get_split_grouping_capability(splitter).group_handling == "wrapper":
        op = GroupedSplitterWrapper(splitter=splitter)

    kwargs: dict[str, Any] = {"groups": groups}
    if needs_y:
        targets = _pool_targets(spectro, pool)
        if targets.size:
            kwargs["y"] = targets

    return [
        ([pool[i] for i in train_idx], [pool[i] for i in val_idx])
        for train_idx, val_idx in op.split(features, **kwargs)
    ]


def _build_folds(splitter: Any, spectro: Any, pool: list[int], excluded: set[int]) -> list[tuple[list[int], list[int]]]:
    """Split ``pool`` over the REAL X/y and drop ``excluded`` from each fold's TRAIN, keeping it in VALIDATION.

    The split runs on the actual spectra (and targets, for supervised splitters) fetched from
    ``spectro`` — the legacy ``CrossValidatorController`` feed — so distance/supervised NIRS splitters
    partition on real data instead of the sample-int list. Index returned by ``split`` is positional
    into ``pool`` and mapped back to absolute sample ints, exactly as the legacy controller maps
    ``base_sample_ids[train_idx]``.

    In legacy mode ``excluded`` is empty (excluded samples are already absent from ``pool``), so this
    is a plain split. In the opt-in (``keep_in_oof=True``) mode ``pool`` is the full train and
    ``excluded`` is non-empty: excluded samples stay in each fold's validation (predicted in OOF) but
    are removed from its train pool — the leakage-pure semantic, materialized in the host FoldSet (the
    adapter owns the split; dag-ml has no runtime splitter, so the FoldSet's ``train_sample_ids`` are
    authoritative for what the node trains on). The envelope still marks them ``excluded`` for lineage.
    """
    return [
        ([pool[i] for i in train_idx if pool[i] not in excluded], [pool[i] for i in val_idx])
        for train_idx, val_idx in _split_pool(splitter, spectro, pool)
    ]
