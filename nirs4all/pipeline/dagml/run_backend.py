"""Run a nirs4all pipeline on the **dag-ml** engine and return a ``RunResult`` (ADR-17 backend).

This is the operational seam for ``engine="dag-ml"``: it assembles the executable compat DSL,
drives ``dag-ml-cli`` through the nirs4all process adapter, and maps dag-ml's **native**
``bundle.scores`` — per-fold validation RMSE/R², the cross-fold OOF average (``cv_best_score``)
and the final-test score (``best_rmse``), all computed in Rust — into an in-memory
:class:`~nirs4all.data.predictions.Predictions`, wrapped in a :class:`~nirs4all.api.result.RunResult`.

No workspace is created and no scoring happens Python-side: the numbers are dag-ml's. Supports the
vertical-slice shape (feature transforms + one model + an OOF/KFold-style splitter). Non-partition
CV (e.g. ``ShuffleSplit``) is not yet supported by the dag-ml ``FoldSet`` (see migration notes).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle
from .envelope import build_envelope
from .identity import mint_identity

_DEFAULT_CLI = Path(__file__).resolve().parents[4] / "dag-ml" / "target" / "release" / "dag-ml-cli"


def _split_pipeline(pipeline: list[Any]) -> tuple[list[Any], Any]:
    """Separate the cross-validator step (the object exposing ``.split``) from the operator steps."""
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    steps = [step for step in pipeline if step is not splitter]
    return steps, splitter


def _materialize_dataset(dataset: Any) -> Any:
    """Materialize the host ``SpectroDataset`` from ANY input ``nirs4all.run()`` accepts.

    ``run_via_dagml`` must accept the SAME dataset arg as legacy ``run()``: a path / config dict /
    JSON-YAML file (via :class:`DatasetConfigs`), an already-built :class:`DatasetConfigs`, a live
    :class:`SpectroDataset` (e.g. ``nirs4all.generate.regression(...)``), a single feature array, or
    an ``(X, y[, partition_info])`` tuple. ``DatasetConfigs`` SILENTLY SKIPS the in-memory inputs
    (its normalizer returns ``None`` for a ``SpectroDataset`` / ``ndarray`` / ``tuple``), leaving
    ``configs`` empty so ``get_dataset_at(0)`` raises a misleading ``IndexError`` — so those inputs
    are wrapped HERE with the SAME normalization the legacy orchestrator uses
    (:meth:`PipelineOrchestrator._wrap_dataset`), never handed to ``DatasetConfigs`` raw.

    A list of datasets is rejected loud: the dag-ml backend runs ONE dataset (the cartesian product
    over datasets stays a legacy-orchestrator concern).
    """
    from nirs4all.data.dataset import SpectroDataset

    if isinstance(dataset, DatasetConfigs):
        return dataset.get_dataset_at(0)
    if isinstance(dataset, SpectroDataset):
        return dataset
    if isinstance(dataset, list):
        raise NotImplementedError("engine='dag-ml' runs a single dataset; pass one dataset, not a list of datasets")
    if isinstance(dataset, (np.ndarray, tuple)):
        return _wrap_in_memory_arrays(dataset)
    return DatasetConfigs(dataset).get_dataset_at(0)


def _wrap_in_memory_arrays(dataset: np.ndarray | tuple) -> Any:
    """Build a ``SpectroDataset`` from a feature array or ``(X, y[, partition_info])`` tuple.

    Mirrors the legacy :meth:`PipelineOrchestrator._wrap_dataset` array/tuple branch exactly (public
    ``add_samples`` / ``add_targets``), so the dag-ml backend materializes the IDENTICAL dataset the
    legacy engine would for the same input. A bare array goes to the ``test`` partition (prediction
    shape); an ``(X, y)`` tuple goes to ``train`` with targets; an optional third element is a
    partition map (``{"train": 80}`` / explicit slices / index lists).
    """
    from nirs4all.data.dataset import SpectroDataset

    spectro = SpectroDataset(name="array_dataset")
    if isinstance(dataset, np.ndarray):
        spectro.add_samples(dataset, indexes={"partition": "test"})
        return spectro

    x = dataset[0]
    y = dataset[1] if len(dataset) > 1 else None
    partition_info = dataset[2] if len(dataset) > 2 else None
    if partition_info is None:
        spectro.add_samples(x, indexes={"partition": "train"})
        if y is not None:
            spectro.add_targets(y)
    else:
        _split_and_add_in_memory(spectro, x, y, partition_info)
    return spectro


def _split_and_add_in_memory(spectro: Any, x: np.ndarray, y: np.ndarray | None, partition_info: dict[str, Any]) -> None:
    """Add ``(X, y)`` to ``spectro`` per ``partition_info`` — the legacy ``_split_and_add_data`` rules.

    ``partition_info`` values: an int (``{"train": 80}`` = first N samples), a ``slice``/list/array of
    indices. If only ``train`` is given, the remaining rows become ``test``. Mirrors
    :meth:`PipelineOrchestrator._split_and_add_data` so the materialized dataset is identical.
    """
    n_samples = x.shape[0]
    partition_indices: dict[str, slice | list[Any] | np.ndarray] = {}
    for partition_name, partition_spec in partition_info.items():
        if isinstance(partition_spec, int):
            partition_indices[partition_name] = slice(0, partition_spec)
        elif isinstance(partition_spec, (slice, list, np.ndarray)):
            partition_indices[partition_name] = partition_spec
        else:
            raise ValueError(f"Invalid partition spec for '{partition_name}': {partition_spec}")

    if "train" in partition_indices and "test" not in partition_indices:
        train_spec = partition_indices["train"]
        if isinstance(train_spec, slice):
            train_end = train_spec.stop if train_spec.stop is not None else train_spec.start
        else:
            train_array = np.array(train_spec)
            train_end = int(train_array.max()) + 1 if len(train_array) > 0 else 0
        if train_end < n_samples:
            partition_indices["test"] = slice(train_end, n_samples)

    for partition_name, indices_spec in partition_indices.items():
        x_partition = x[indices_spec]
        y_partition = y[indices_spec] if y is not None else None
        if len(x_partition) > 0:
            spectro.add_samples(x_partition, indexes={"partition": partition_name})
            if y_partition is not None:
                spectro.add_targets(y_partition)


def _dataset_inputs(dataset: Any, spectro: Any, base_dir: Path) -> tuple[str, str | None]:
    """Resolve how the adapter materializes ``spectro``: ``(dataset_path, dataset_pickle)``.

    The adapter (in a subprocess) must materialize the EXACT dataset the host built the
    envelope/identity from — the wire ids are ``f"{content_hash()}.s{sample_int}"``, so any drift in
    the re-loaded dataset's content hash or sample order makes the run wrong (host-audit H-P1-1).

    Prefer the cheap path re-load when it is provably faithful: for a file-path / config input that
    exposes a reloadable path, re-load ``DatasetConfigs(path).get_dataset_at(0)`` IN-PROCESS and keep
    the path ONLY if its identity fingerprint matches the host's. If the input is in-memory (a
    ``SpectroDataset`` / array / tuple, with no reloadable path) OR the path re-load diverges, PICKLE
    the host ``spectro`` and ship it via the existing ``N4A_DAGML_DATASET_PICKLE`` channel — the
    adapter then loads the byte-identical dataset (the same mechanism augmentation/rep-fusion use).
    """
    path = _reloadable_path(dataset)
    if path is not None:
        try:
            reloaded = DatasetConfigs(path).get_dataset_at(0)
            if mint_identity(reloaded).fingerprint == mint_identity(spectro).fingerprint:
                return path, None
        except (IndexError, ValueError, OSError):
            pass  # path not faithfully reloadable — fall through to pickle

    base_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = base_dir / "host_dataset.pkl"
    import pickle

    pickle_path.write_bytes(pickle.dumps(spectro))
    return (path or str(pickle_path)), str(pickle_path)


def _reloadable_path(dataset: Any) -> str | None:
    """A filesystem path the adapter can reload via ``DatasetConfigs(path)``, or ``None``.

    A raw path string / ``Path`` is reloadable as-is; an already-built :class:`DatasetConfigs` (e.g.
    ``DatasetConfigs(folder, repetition="col")``) exposes its source files in ``configs[0]`` — the
    folder-style dataset reloads from the common parent directory of the ``train_x`` file. In-memory
    inputs (``SpectroDataset`` / array / tuple) have no reloadable path and return ``None``.
    """
    if isinstance(dataset, (str, Path)):
        return str(dataset)
    if isinstance(dataset, DatasetConfigs):
        config_dict = dataset.configs[0][0]
        train_x = config_dict.get("train_x")
        if isinstance(train_x, str) and train_x:
            return str(Path(train_x).parent)
    return None


def _is_exclude_step(step: Any) -> bool:
    return isinstance(step, dict) and "exclude" in step


def _taggers_from_step(step: Any) -> list[tuple[str, Any]] | None:
    """Parse a handled ``{"tag": SampleFilter}`` step, else return ``None`` for bridge fail-loud."""
    if not isinstance(step, dict) or "tag" not in step:
        return None

    from nirs4all.controllers.data.tag import TagController

    try:
        taggers = TagController()._parse_taggers(step.get("tag", {}))  # noqa: SLF001 - reuse production parsing/name rules
    except (TypeError, ValueError):
        return None
    return taggers or None


def _is_tag_step(step: Any) -> bool:
    return _taggers_from_step(step) is not None


def _is_augmentation_step(step: Any) -> bool:
    return isinstance(step, dict) and "sample_augmentation" in step


# `rep_to_sources` / `rep_to_pp` are one-time HOST dataset RESHAPES (RepToSourcesController /
# RepToPPController, priority 3 — applied BEFORE the CV splitter). `rep_to_sources` turns each
# replicate of a physical sample into a separate feature SOURCE (N reps → N sources × n_unique
# samples), and `rep_to_pp` stacks each replicate into the PROCESSING axis (N reps → n_pp×N
# processing layers × n_unique samples). After the reshape the unit of analysis is the physical
# SAMPLE (not the rep row), so folds/OOF are sample-grain — distinct from a PLAIN repetition
# dataset (#21, which keeps the rep rows and scores at the rep grain).
_REP_FUSION_KEYS = ("rep_to_sources", "rep_to_pp")


def _is_rep_fusion_step(step: Any) -> bool:
    return isinstance(step, dict) and any(key in step for key in _REP_FUSION_KEYS)


def _detect_rep_fusion(pipeline: list[Any]) -> dict[str, Any] | None:
    """The single ``rep_to_sources`` / ``rep_to_pp`` reshape step, else ``None`` (fail-loud elsewhere).

    Returns the reshape step only for the EXACTLY-supported shape — one reshape step plus the
    ordinary ``transform* + splitter + model`` body. More than one reshape, or a reshape combined
    with a branch / exclude / sample_augmentation (compositions the reshaped sample-grain folds
    cannot honor here), returns ``None`` so the bridge's generic fail-loud path names #31.
    """
    rep_steps: list[dict[str, Any]] = [step for step in pipeline if _is_rep_fusion_step(step)]
    if len(rep_steps) != 1:
        return None
    if any(_is_augmentation_step(step) or _is_exclude_step(step) or (isinstance(step, dict) and "branch" in step) for step in pipeline):
        return None
    return rep_steps[0]


def _filter_data_for_pool(spectro: Any, pool_ints: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
    """X/y for SampleFilter fitting, aligned exactly to ``pool_ints`` order."""
    x_pool = np.asarray(spectro.x({"sample": list(pool_ints)}, layout="2d", concat_source=True, include_augmented=False))
    y_values = spectro.y({"sample": list(pool_ints)}, include_augmented=False)
    y_pool = None if y_values is None else np.asarray(y_values)

    # `spectro.x/y({"sample": ids})` returns ascending storage order, not request order; re-key so
    # masks align to `pool_ints` exactly (the storage-vs-request trap the resolver also guards against).
    stored = spectro.index_column("sample", {"sample": list(pool_ints)})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    order = [row_of[int(sample_int)] for sample_int in pool_ints]
    x_pool = x_pool[order]
    if y_pool is not None and y_pool.size:
        y_pool = y_pool[order]
        if y_pool.ndim > 1:
            y_pool = y_pool.flatten()
    return x_pool, y_pool


def _excluded_from_pool(exclude_step: dict[str, Any], spectro: Any, pool_ints: list[int]) -> set[int]:
    """Excluded sample ints from ``pool_ints`` for one ``exclude_step``, mirroring ExcludeController.

    Fits each :class:`~nirs4all.operators.filters.base.SampleFilter` on the CURRENT kept pool's X/y
    (``include_augmented=False``) and combines the per-filter keep-masks by ``mode`` — exactly the
    legacy :class:`~nirs4all.controllers.data.exclude.ExcludeController` mask logic:

    * ``mode="any"`` → exclude if ANY filter flags = ``np.all`` of the keep-masks (exclude.py:193);
    * ``mode="all"`` → exclude only if ALL filters flag = ``np.any`` (exclude.py:196).

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
    if not pool_ints:
        return set()

    x_pool, y_pool = _filter_data_for_pool(spectro, pool_ints)
    if y_pool is None or y_pool.size == 0:
        return set()

    masks: list[np.ndarray] = []
    for filter_obj in filters:
        try:
            filter_obj.fit(x_pool, y_pool)
            masks.append(filter_obj.get_mask(x_pool, y_pool))
        except ValueError:
            # exclude.py:175-184 — a filter that can't be applied contributes a neutral keep-all mask.
            masks.append(np.ones(len(pool_ints), dtype=bool))

    if len(masks) == 1:
        keep_mask = masks[0].copy()
    else:
        stacked = np.stack(masks, axis=0)
        keep_mask = np.all(stacked, axis=0) if filter_mode == "any" else np.any(stacked, axis=0)

    # exclude.py:213-222 — never empty the pool: if all rows would be excluded, keep the first.
    if not keep_mask.any():
        keep_mask[0] = True

    return {int(sample_int) for sample_int, keep in zip(pool_ints, keep_mask, strict=True) if not keep}


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
    """
    train_ints = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    exclude_steps = [step for step in pipeline if _is_exclude_step(step)]
    if not exclude_steps:
        return pipeline, train_ints, set()

    keep_in_oof = any(bool(step.get("keep_in_oof", False)) for step in exclude_steps)
    excluded: set[int] = set()
    for step in exclude_steps:
        current_pool = [sample_int for sample_int in train_ints if sample_int not in excluded]
        excluded |= _excluded_from_pool(step, spectro, current_pool)

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


# Keys on a step dict that are NOT model hyperparameters (mirrors StepParser.RESERVED_KEYWORDS).
_RESERVED_STEP_KEYS = frozenset({"model", "params", "metadata", "steps", "name", "finetune_params", "train_params", "refit_params", "fit_on_all", "force_layout", "na_policy", "fill_value", "y_processing"})


def _apply_model_params(steps: list[Any]) -> list[Any]:
    """Apply sibling hyperparameters to the model (e.g. `{"model": PLS(), "n_components": 9}`).

    Generators expand a param sweep into `{"model": M, "<param>": value}` steps; nirs4all applies the
    non-reserved siblings to the model via set_params. We do the same on a clone so concurrent
    variants do not share mutated state.
    """
    from sklearn.base import clone

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            params = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS}
            if params:
                model = step["model"]
                # A class-model (e.g. ``PLSRegression`` rather than ``PLSRegression()``) must be
                # instantiated before clone — ``clone`` rejects a class. The expansion path normally
                # instantiates it (:func:`_expand_operator_generators`); this guards the remaining
                # bare-class shape rather than crashing on ``clone(<class>)``.
                if isinstance(model, type):
                    model = model()
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**params)
                step = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS}
                step["model"] = model
        out.append(step)
    return out


def _flatten_steps(steps: list[Any]) -> list[Any]:
    """Flatten nested-list pipeline steps into a single top-level step list.

    Operator-level generators expand a stage into a SUB-PIPELINE of steps held inside a list
    (``_cartesian_`` builds ``[[A, B], splitter, model]`` and ``_or_``+``pick`` builds
    ``[[A, B], …]``). The legacy executor flattens such nested lists into consecutive steps; the
    dag-ml bridge lowers one step at a time and would otherwise lower the inner list to a
    ``builtins.list`` node (→ ``make_pipeline([], model)`` crash), so flatten here first.
    """
    out: list[Any] = []
    for step in steps:
        if isinstance(step, list):
            out.extend(_flatten_steps(step))
        else:
            out.append(step)
    return out


# Step keywords whose value is the operator a top-level param-keyed sweep targets (mirrors the
# generator core's ``_OPERATOR_WRAPPER_KEYS``). When the step carries a top-level generator keyword
# (``_range_``/``_grid_``/…) over such an operator, the operator must be a ``{"class": …}`` dict for
# ``expand_spec``'s ``_normalize_param_sweep``/``_normalize_param_grid`` to expand it.
_OPERATOR_WRAPPER_KEYS = ("model", "y_processing")


def _wrap_param_keyed_operator(step: Any) -> Any:
    """Wrap a bare-string operator into a ``{"class": …}`` dict for a top-level param-keyed sweep.

    The canonical generator dialect places the generator keyword at the TOP level of the step dict
    beside ``param`` (the attribute) and ``model`` (the operator), e.g.
    ``{"_range_": [5, 25, 5], "param": "n_components", "model": PLSRegression}``. After
    ``serialize_component`` the operator is a bare string (``"sklearn…PLSRegression"``), but
    ``expand_spec``'s sweep/grid normalizers only fire on a ``{"class": …}`` dict (or a top-level
    ``class`` key) — a bare string is left unexpanded (1 variant), then ``clone`` crashes on the
    class. Wrapping the string in ``{"class": …}`` routes the sweep through the same nested-form
    expansion the list-form already uses. Steps without a top-level generator are returned unchanged.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS

    if not isinstance(step, dict) or not (GENERATION_KEYWORDS & set(step)):
        return step
    wrapped = dict(step)
    for opkw in _OPERATOR_WRAPPER_KEYS:
        operator = wrapped.get(opkw)
        if isinstance(operator, str):
            wrapped[opkw] = {"class": operator}
    return wrapped


def _expand_operator_generators(pipeline: list[Any]) -> list[list[Any]]:
    """Expand operator-level generators into concrete, flat pipelines of live operator instances.

    Mirrors nirs4all's production expansion (``PipelineConfigs``): ``serialize_component`` the
    pipeline to its canonical form FIRST so a bare-class/instance operator and the param-keyed sweep
    dialect (``_range_``/``_log_range_``/``_grid_``/``_zip_``/``_sample_`` beside ``param``/``model``)
    normalize and expand identically to legacy; then ``deserialize_component`` each variant back to
    live instances and flatten any nested sub-pipeline lists (``_cartesian_``/``_or_``+``pick``). The
    result is what ``_run_concrete`` expects — a flat ``[transform…, splitter, model]`` list with no
    nested lists and no bare classes — so the bridge lowers it cleanly instead of crashing on a
    ``clone(<class>)`` or a ``builtins.list`` intermediate step.

    A generator-free pipeline is returned unchanged (one variant of the original live operators) —
    the serialize/deserialize round-trip is reserved for the generator path so the common
    transform+model shape keeps its exact operator instances.
    """
    from nirs4all.pipeline.config._generator.keywords import has_nested_generator_keywords
    from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
    from nirs4all.pipeline.config.generator import expand_spec

    if not has_nested_generator_keywords(pipeline):
        return [pipeline]

    serialized = serialize_component(pipeline)
    normalized = [_wrap_param_keyed_operator(step) for step in serialized]
    return [_flatten_steps(deserialize_component(variant)) for variant in expand_spec(normalized)]


def _model_name(steps: list[Any]) -> str:
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            return type(step["model"]).__name__
    return "model"


# Step keywords whose presence forces the Python path even alongside a model param sweep: Optuna
# finetune / per-model train kwargs are not part of the native generation+SELECT contract, so a
# pipeline carrying them must NOT be mistaken for a clean param-sweep-only pipeline.
_FORCE_PYTHON_STEP_KEYS = frozenset({"finetune_params", "train_params"})


def _generation_kind(pipeline: list[Any]) -> str:
    """Classify a pipeline's generators: ``"none"``, ``"param_model"`` (native), or ``"operator"``.

    CONSERVATIVE by design — native (``"param_model"``) is returned ONLY when the pipeline is a clean
    model-param-sweep, i.e. ALL of:

    (a) at least one ``{"model": ...}`` step carries a natively-lowerable param sweep
        (:func:`~nirs4all.pipeline.dagml_bridge.is_param_generator_spec` — the exact ``_range_`` /
        ``_log_range_`` list forms), AND
    (b) NO other generator exists ANYWHERE — no generator keyword on a non-model step, no
        generator-valued model (multi-model ``{"model": {"_or_": ...}}``), no generator-shaped model
        sibling that is not natively lowerable (``_grid_``, dict-form, modifier-bearing), AND
    (c) NO step carries ``finetune_params`` or ``train_params``.

    Any other generator (or finetune/train_params) → ``"operator"`` (the correct Python ``expand_spec``
    path). ``"none"`` means no generators at all. When in doubt, this never returns ``"param_model"``.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS, has_nested_generator_keywords
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    has_param_model = False
    has_other = False
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            has_other = True  # finetune/train_params are not in the native contract
        if "model" in step:
            # A generator-valued model (multi-model) is operator-level, not a clean param sweep.
            if has_nested_generator_keywords(step["model"]):
                has_other = True
            for key, value in step.items():
                if key in _RESERVED_STEP_KEYS:
                    continue
                if is_param_generator_spec(value):
                    has_param_model = True
                elif has_nested_generator_keywords(value):
                    # A generator-shaped sibling we cannot lower natively (e.g. `_grid_`, dict-form,
                    # or a modifier-bearing range) — Python expand owns it.
                    has_other = True
        elif GENERATION_KEYWORDS & set(step) or has_nested_generator_keywords(step):
            # Any generator on a non-model step (bare `_or_`/`_range_`/... or a nested one).
            has_other = True
    if has_other:
        return "operator"
    return "param_model" if has_param_model else "none"


def _is_separation_branch_step(step: Any) -> bool:
    """True for a separation branch by metadata/tag: ``{"branch": {"by_metadata"|"by_tag": ...}}``."""
    return isinstance(step, dict) and isinstance(step.get("branch"), dict) and bool({"by_metadata", "by_tag"} & set(step["branch"]))


def _is_concat_merge_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("merge") == "concat"


# Keys recognised inside a separation-branch criterion dict. `run_backend` honors ONLY the criterion
# (by_metadata/by_tag) + the shared `steps` body; `values` (explicit grouping), `min_samples`
# (cardinality drop) and per-branch selectors are NOT honored, so a branch carrying them must fall
# through to the loud bridge error rather than be silently run with default behavior.
_HANDLED_BRANCH_KEYS = frozenset({"by_metadata", "by_tag", "steps"})


def _detect_separation_branch(pipeline: list[Any]) -> tuple[dict[str, Any], list[Any]] | None:
    """Detect the EXACT handled shape, else return ``None`` (fail-loud via the bridge).

    Admits ONLY a pipeline that is exactly: the splitter + ONE by_metadata/by_tag separation branch
    (a single shared ``steps`` body containing the model) + ONE ``{"merge": "concat"}`` — nothing
    that ``_run_separation_branch`` does not actually honor. Returns ``(branch_step, branch_body)``
    when matched. ANY deviation returns ``None`` so the bridge's raw-branch ``NotImplementedError``
    fires (the coverage-boundary fail-loud guarantee), never a silent-wrong run. Specifically REJECTED:

    * a top-level operator/transform/``tag``/``y_processing`` step beside the branch (only the branch
      body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * an ``exclude`` step anywhere (the folds are built over the full pool with no excluded bit, so the
      exclusion would be silently lost) — exclude+branch is a follow-up slice;
    * an unhandled branch option (``values`` / ``min_samples`` / a per-branch ``selector`` / any key
      outside ``by_metadata``/``by_tag``/``steps``) — those grouping semantics are not honored;
    * a per-value dict ``steps`` (different sub-pipeline per partition), a missing model in the body,
      a model after the merge, a non-concat merge, or more than one branch/merge.
    """
    branch_steps = [step for step in pipeline if _is_separation_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    if len(branch_steps) != 1 or len(merge_steps) != 1:
        return None
    branch_step, merge_step = branch_steps[0], merge_steps[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
    # transform / tag / y_processing / exclude / model would be silently ignored (only the branch body
    # is lowered), so its presence rejects the match → fail-loud.
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    criterion = branch_step["branch"]
    # Only the criterion (by_metadata/by_tag) + the shared `steps` body are honored. Any other branch
    # option (values/min_samples/per-branch selector/...) is not → reject.
    if set(criterion) - _HANDLED_BRANCH_KEYS:
        return None

    body = criterion.get("steps")
    # Only the shared-body LIST form (one sub-pipeline applied per partition) with a model inside is
    # supported. The per-value dict form and a body without a model fall through to the bridge error.
    if not isinstance(body, list) or not any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    return branch_step, body


# Keys recognised inside a by_source branch criterion dict. `run_backend` honors ONLY the
# `by_source` flag + the shared `steps` body (one sub-pipeline applied per source). A per-source
# dict body (`{"src0": [...], "src1": [...]}`), `values`/`min_samples`, or any other option falls
# through to the loud bridge error rather than being silently run with default behavior.
_HANDLED_BY_SOURCE_KEYS = frozenset({"by_source", "steps"})


def _is_by_source_branch_step(step: Any) -> bool:
    """True for a by_source separation branch: ``{"branch": {"by_source": True|"auto", ...}}``.

    LATE fusion BY SOURCE: a branch PER feature source, each branch's model fed ONLY that source's
    block (a feature-axis selection — all samples, one source's columns), distinct from by_metadata
    (a SAMPLE partition) and duplication (every branch sees the FULL data).
    """
    if not isinstance(step, dict) or not isinstance(step.get("branch"), dict):
        return False
    return step["branch"].get("by_source") in (True, "auto")


def _detect_by_source_branch(pipeline: list[Any], n_sources: int) -> list[Any] | None:
    """Detect the EXACT handled by_source shape, else ``None`` (fail-loud via the bridge).

    Admits ONLY: the splitter + ONE ``{"branch": {"by_source": True|"auto", "steps": [...model...]}}``
    (a single shared body LIST containing the model, applied per source) + ONE avg/mean fusion merge
    (:func:`_fusion_merge_aggregate`) on a MULTI-source dataset (``n_sources >= 2``). Returns the shared
    branch body (the model sub-pipeline) when matched. ANY deviation returns ``None`` so the bridge's
    raw-branch ``NotImplementedError`` fires — never a silent-wrong run. Specifically REJECTED:

    * a single-source dataset (by_source on one source is a no-op — there is nothing to fuse);
    * the per-source DICT body (``{"src0": [...], "src1": [...]}`` — different model per source) — a
      later slice; only the shared body is honored here;
    * an unhandled branch option (``values`` / ``min_samples`` / any key outside ``by_source``/``steps``);
    * a body without a model (late fusion averages MODEL predictions);
    * a non-fusion merge (``concat`` / ``predictions`` stacking), a top-level step beside the branch,
      a model after the merge, or more than one branch/merge.
    """
    branch_steps = [step for step in pipeline if _is_by_source_branch_step(step)]
    merge_aggregates = [(step, agg) for step in pipeline if (agg := _fusion_merge_aggregate(step)) is not None]
    if len(branch_steps) != 1 or len(merge_aggregates) != 1 or n_sources < 2:
        return None
    branch_step = branch_steps[0]
    merge_step = merge_aggregates[0][0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps (a top-level
    # transform / tag / y_processing / exclude / model would be silently dropped, since only the branch
    # body is lowered per source).
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    criterion = branch_step["branch"]
    if set(criterion) - _HANDLED_BY_SOURCE_KEYS:
        return None
    body = criterion.get("steps")
    # Only the shared-body LIST form (one model sub-pipeline applied per source) is supported here.
    if not isinstance(body, list) or not any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    return body


def _is_duplication_branch_step(step: Any) -> bool:
    """True for a DUPLICATION branch: ``{"branch": [[A], [B], ...]}`` (the list-of-lists form).

    Legacy nirs4all (``BranchController._detect_branch_mode``) treats *list* branch syntax as ALWAYS
    duplication — N parallel sub-pipelines, each seeing the FULL data (no sample partitioning). The
    dict form (``{"by_metadata": ...}``/named branches) is separation/other and is NOT matched here.
    Each inner element must itself be a list (a sub-pipeline of steps).
    """
    if not isinstance(step, dict):
        return False
    branch = step.get("branch")
    return isinstance(branch, list) and len(branch) >= 2 and all(isinstance(sub, list) for sub in branch)


# The cross-branch fusion (avg / proba-mean) merge tokens this backend maps to dag-ml's native fusion
# merge handler. Simple-string ``"mean"``/``"average"`` (a NEW token — legacy MergeConfigParser rejects
# it, so there is no collision) average the branches' held-out OOF per sample into ONE final prediction;
# the explicit-aggregation dict form reuses nirs4all's established aggregation vocabulary
# (``AggregationStrategy.MEAN``/``PROBA_MEAN``). A STACKING merge (``{"merge": "predictions"}`` →
# MetaModel, backlog #10) is deliberately NOT a fusion token and falls through to the loud bridge error.
_FUSION_MERGE_STRINGS = frozenset({"mean", "average"})


def _fusion_merge_aggregate(step: Any) -> str | None:
    """The fusion aggregation if ``step`` is a handled avg/mean fusion merge, else ``None``.

    Returns ``"mean"`` (value average → dag-ml ``merge_mode: "fusion"``) or ``"proba_mean"``
    (class-probability average → ``"fusion_proba_mean"``) for the two recognized spellings:

    * ``{"merge": "mean"}`` / ``{"merge": "average"}`` → ``"mean"``;
    * ``{"merge": {"predictions": "all", "aggregate": "mean"|"proba_mean"}}`` — the explicit
      aggregation-vocabulary form (``predictions`` collection + an ``aggregate`` reducer), with NO
      other keys (no per-branch ``select``/``metric``, no ``features``, no downstream model implied).

    Everything else (``"predictions"`` stacking, ``"concat"``, ``"features"``, a per-branch config,
    ``weighted_mean``, ``separate``) returns ``None`` so the bridge fails loud.
    """
    if not isinstance(step, dict) or "merge" not in step:
        return None
    spec = step["merge"]
    if isinstance(spec, str):
        return "mean" if spec in _FUSION_MERGE_STRINGS else None
    if isinstance(spec, dict):
        # Only the exact {"predictions": "all", "aggregate": <mean|proba_mean>} shape — nothing else.
        if set(spec) != {"predictions", "aggregate"} or spec.get("predictions") not in ("all", True):
            return None
        aggregate = spec.get("aggregate")
        return aggregate if aggregate in ("mean", "proba_mean") else None
    return None


def _is_stacking_merge_step(step: Any) -> bool:
    """True for a STACKING merge (``{"merge": "predictions"}`` or a per-branch predictions config).

    Stacking turns the branch OOF into meta-features for a downstream meta-model — a separate, larger
    subsystem (backlog #10). It is detected only to fail LOUD with a clear #10 message, never run.
    """
    if not isinstance(step, dict) or "merge" not in step:
        return False
    spec = step["merge"]
    if spec == "predictions":
        return True
    return isinstance(spec, dict) and ("predictions" in spec) and _fusion_merge_aggregate(step) is None


def _detect_duplication_branch(pipeline: list[Any]) -> tuple[list[list[Any]], str] | None:
    """Detect the EXACT duplication-branch + avg/mean fusion-merge shape, else ``None`` (fail-loud).

    Admits ONLY a pipeline that is exactly: the splitter + ONE duplication branch
    (``{"branch": [[A], [B], ...]}`` with N≥2 sub-pipelines, each containing a model) + ONE avg/mean
    fusion merge (:func:`_fusion_merge_aggregate`). Returns ``(branches, aggregate)`` when matched
    (``aggregate`` is ``"mean"`` or ``"proba_mean"``). ANY deviation returns ``None`` so the bridge's
    raw-branch / raw-merge ``NotImplementedError`` fires. Specifically REJECTED (fall through to loud):

    * a STACKING merge (``{"merge": "predictions"}`` / a per-branch predictions config → a meta-model,
      backlog #10) — raised loud naming #10 by the caller, never silently averaged;
    * a separation (dict-form) branch — handled by :func:`_detect_separation_branch`, not here;
    * a sub-pipeline without a model (fusion averages MODEL predictions);
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * a model after the merge, more than one branch/merge, or any unrecognized merge spelling.
    """
    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_aggregates = [(step, agg) for step in pipeline if (agg := _fusion_merge_aggregate(step)) is not None]
    if len(branch_steps) != 1 or len(merge_aggregates) != 1:
        return None
    branch_step = branch_steps[0]
    merge_step, aggregate = merge_aggregates[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
    # transform / tag / y_processing / exclude / model would be silently ignored (each branch body is
    # lowered, not the top level), so its presence rejects the match → fail-loud.
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    branches = branch_step["branch"]
    # Every sub-pipeline must contain a model — fusion averages MODEL predictions; a modelless branch
    # (features only) is not the supported shape.
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    return branches, aggregate


def _is_simple_predictions_merge_step(step: Any) -> bool:
    """True ONLY for the exact ``{"merge": "predictions"}`` stacking-merge string — nothing richer.

    A per-branch predictions config (``{"merge": {"predictions": [{"branch": 0, "select": "best"}]}}``)
    carries model-selection/aggregation semantics this slice does NOT honor, so it is rejected (it stays
    on the loud #10 path). Only the plain ``"predictions"`` collect-all merge is the supported stacking shape.
    """
    return isinstance(step, dict) and step.get("merge") == "predictions"


def _is_default_except_level(config: Any) -> bool:
    """True iff ``config`` is a ``StackingConfig`` equal to the default in EVERY field except ``level``.

    A MetaModel may carry only the stacking options this slice actually HONORS. ``level`` is the one
    permitted deviation (AUTO / LEVEL_1 → a single base→meta level, which the dag-ml lowering produces);
    every other field (``coverage_strategy``, ``test_aggregation``, ``branch_scope``, ``allow_no_cv``,
    ``min_coverage_ratio``, ``allow_meta_sources``, ``max_level``, ``relation_profile``) is SILENTLY
    IGNORED by the lowering — notably ``test_aggregation``, which has no effect because this slice cannot
    score test meta-features at all (best_rmse is NaN). So a non-default value of any of those must reject
    the stacking shape (fail loud) rather than run with the option dropped. Comparison is field-exhaustive
    by construction: clone the config with ``level`` reset to the default and compare to a fresh default,
    so any future ``StackingConfig`` field is covered without enumerating them here.
    """
    import dataclasses

    from nirs4all.operators.models.meta import StackingConfig

    if not isinstance(config, StackingConfig):
        return False
    normalized = dataclasses.replace(config, level=StackingConfig().level)
    return normalized == StackingConfig()


def _meta_learner(model_step: dict[str, Any]) -> Any | None:
    """The sklearn meta-learner estimator from a downstream ``{"model": …}`` stacking step, else ``None``.

    Two equivalent nirs4all spellings (per ``MergeController``'s own docstring): a ``MetaModel`` wrapper
    (``{"model": MetaModel(Ridge())}`` — the meta-learner is its wrapped ``.model``) or a plain sklearn
    estimator (``{"model": Ridge()}`` after ``{"merge": "predictions"}``). Either way we return the bare
    sklearn estimator that fits on the meta-feature matrix.

    Returns ``None`` (→ fail loud, never run wrong) for any MetaModel option this slice does not honor:
    a non-default ``source_models`` list, ``use_proba``, a custom ``selector``, a ``finetune_space``, a
    non-AUTO/non-1 stacking ``level``, OR any OTHER non-default ``stacking_config`` field
    (``test_aggregation``, ``coverage_strategy``, … — silently ignored by the lowering; see
    :func:`_is_default_except_level`).
    """
    from nirs4all.operators.models.meta import MetaModel, StackingLevel

    model = model_step.get("model")
    if isinstance(model, MetaModel):
        config = model.stacking_config
        if (
            model.source_models != "all"
            or model.use_proba
            or model.selector is not None
            or model.finetune_space is not None
            or config.level not in (StackingLevel.AUTO, StackingLevel.LEVEL_1)
            or not _is_default_except_level(config)
        ):
            return None
        return model.model
    # A plain sklearn estimator step (no other generator/reserved sibling that would change its meaning).
    if model is not None and hasattr(model, "fit") and hasattr(model, "predict"):
        return model
    return None


def _detect_stacking_branch(pipeline: list[Any]) -> tuple[list[list[Any]], Any] | None:
    """Detect the EXACT duplication-branch + ``{"merge": "predictions"}`` + meta-model shape, else ``None``.

    Admits ONLY: the splitter + ONE duplication branch (``{"branch": [[A], [B], …]}``, N≥2 sub-pipelines
    each with a model) + ONE ``{"merge": "predictions"}`` + ONE downstream ``{"model": M}`` whose M is a
    handled meta-learner (a default ``MetaModel`` wrapper or a plain sklearn estimator; see
    :func:`_meta_learner`). Returns ``(branches, meta_learner)`` (the bare sklearn estimator) when matched.
    ANY deviation returns ``None`` so the bridge / the loud #10 path fires — never a silent-wrong run.
    Specifically REJECTED (fall through to loud):

    * a fusion/avg merge or a concat merge (those are the duplication-fusion / separation paths);
    * a per-branch predictions config (model-selection/aggregation semantics not honored — stays #10);
    * a missing downstream model, more than one branch/merge/model, or a model BEFORE the merge;
    * a top-level operator/transform/``tag``/``y_processing``/``exclude`` beside the branch (only each
      branch body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * a sub-pipeline without a model (the base level needs a model to produce OOF);
    * a MetaModel carrying unhandled options (non-default source_models/use_proba/selector/finetune/config);
    * a meta-model step carrying a sibling param (``{"model": Ridge(), "alpha": 0.2}``) or a generator
      (``{"model": Ridge(), "alpha": {"_range_": [...]}}``): the meta-model node is lowered as a bare
      estimator, so ``_apply_model_params`` / native generation never run for it — the param/sweep would
      be silently ignored. A tuned/swept meta-model is a later slice.
    """
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    branch_steps = [step for step in pipeline if _is_duplication_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_simple_predictions_merge_step(step)]
    model_steps = [step for step in pipeline if isinstance(step, dict) and "model" in step]
    if len(branch_steps) != 1 or len(merge_steps) != 1 or len(model_steps) != 1:
        return None
    branch_step, merge_step, model_step = branch_steps[0], merge_steps[0], model_steps[0]

    # The meta-model step must be a BARE {"model": <estimator>} (plus harmless reserved keys like name):
    # any extra non-reserved sibling param OR a param-generator on the meta step is silently dropped by the
    # bare-estimator lowering, so reject it (fail loud) rather than run the meta-model with the option lost.
    if any(key not in _RESERVED_STEP_KEYS or is_param_generator_spec(value) for key, value in model_step.items() if key != "model"):
        return None

    # The merge must come BEFORE the model (the model is the meta-learner over the merged OOF), and the
    # pipeline must be EXACTLY {splitter, branch, merge, model} — no other top-level steps (a top-level
    # transform / tag / y_processing / exclude would be silently dropped, since only branch bodies are
    # lowered). Order + membership are both enforced.
    order = [step for step in pipeline if step is branch_step or step is merge_step or step is model_step]
    if order != [branch_step, merge_step, model_step]:
        return None
    for step in pipeline:
        if step is branch_step or step is merge_step or step is model_step or hasattr(step, "split"):
            continue
        return None

    branches = branch_step["branch"]
    if not all(any(isinstance(sub, dict) and "model" in sub for sub in branch) for branch in branches):
        return None
    meta_learner = _meta_learner(model_step)
    if meta_learner is None:
        return None
    return branches, meta_learner


def run_via_dagml(
    pipeline: Any,
    dataset: Any,
    *,
    dagml_cli: str | Path | None = None,
    venv_python: str | None = None,
    workdir: str | Path | None = None,
) -> RunResult:
    """Execute ``pipeline`` on ``dataset`` via dag-ml-cli; return a RunResult of dag-ml's native scores.

    Args:
        pipeline: nirs4all pipeline (feature transforms, one ``{"model": ...}`` step, and a splitter).
        dataset: Anything :class:`~nirs4all.data.config.DatasetConfigs` accepts (path/config).
        dagml_cli: Path to the ``dag-ml-cli`` binary (defaults to the sibling ``dag-ml`` build).
        venv_python: Python interpreter the process adapter re-execs under (defaults to the current).
        workdir: Scratch directory for the run inputs/outputs (defaults to a temp dir).

    Returns:
        A :class:`~nirs4all.api.result.RunResult` whose ``best_rmse`` is the native final-test score
        and ``cv_best_score`` is the native cross-fold OOF average.
    """
    cli = str(dagml_cli or _DEFAULT_CLI)
    if not Path(cli).exists():
        raise FileNotFoundError(f"dag-ml-cli binary not found at {cli}; build it (cargo build -p dag-ml-cli --release)")

    from nirs4all.core import detect_task_type

    # Materialize the host dataset from ANY input legacy `run()` accepts (path / config /
    # DatasetConfigs / live SpectroDataset / (X, y) tuple / array) — DatasetConfigs alone silently
    # skips the in-memory ones, so `_materialize_dataset` wraps them with the legacy normalization.
    spectro = _materialize_dataset(dataset)
    base_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="n4a_dagml_"))
    # `dataset_arg` is the reloadable path (clean file-path datasets, no pickle — fast); `host_pickle`
    # is set only when the adapter cannot faithfully reload from a path (in-memory inputs, or a path
    # whose re-load diverges from the host identity), and ships the byte-identical host dataset.
    dataset_arg, host_pickle = _dataset_inputs(dataset, spectro, base_dir / "host")

    is_classification = "classif" in str(detect_task_type(np.asarray(spectro.y({"partition": "train"}))))
    metric = "accuracy" if is_classification else "rmse"
    task_type = "classification" if is_classification else "regression"

    # Detect the special-composition steps UP FRONT so the repetition guard below can reject an
    # unsupported combination BEFORE any non-group dispatch path (branch/augmentation/exclude) runs.
    detected = _detect_separation_branch(list(pipeline))
    detected_duplication = _detect_duplication_branch(list(pipeline))
    detected_stacking = _detect_stacking_branch(list(pipeline))
    detected_by_source = _detect_by_source_branch(list(pipeline), spectro.features_sources())
    detected_rep_fusion = _detect_rep_fusion(list(pipeline))
    augmentation_steps = [step for step in pipeline if _is_augmentation_step(step)]

    # REP FUSION (`rep_to_sources` / `rep_to_pp`, #31): a one-time HOST RESHAPE that turns each replicate
    # of a physical sample into a feature SOURCE (→ MULTI-SOURCE early fusion S3 / MB-PLS S5) or a
    # PROCESSING layer (→ the feature-axis concat S6). After the reshape the unit of analysis is the
    # physical SAMPLE (folds/OOF sample-grain — distinct from the plain repetition rep-grain path #21,
    # below). Detected BEFORE the repetition guard because the reshape CONSUMES the rep grouping (the
    # reshaped dataset is no longer a repetition dataset); the reshape feeds the already-native
    # multi-source / feature-concat materialization, pickled for the adapter (the on-disk dataset has no
    # such structure). A reshape combined with branch/exclude/augmentation is rejected by `_detect_rep_fusion`
    # (returns None) and falls through to the bridge's fail-loud path naming #31.
    if detected_rep_fusion is not None:
        return _run_rep_fusion(list(pipeline), detected_rep_fusion, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "rep_fusion", metric, task_type)

    # REPETITIONS (sample-grain grouping): when the dataset declares a repetition column, several stored
    # rows share one physical sample. The split must be GROUP-aware — all replicates of a sample land on
    # the SAME fold side — and each rep row is scored individually (the repetition grain), which is what
    # nirs4all's `cv_best_score`/`best_rmse` report (the sample-level `_agg` aggregation is a separate twin
    # entry, NOT those scores). Folds are over the rep ROWS, group-grouped (a clean OOF partition), and the
    # envelope emits `group_id` so dag-ml-data refuses any fold that splits a group. The first slice handles
    # the supported transform+model+splitter shape only.
    #
    # This guard runs BEFORE the branch/augmentation/exclude dispatch below: those paths build folds
    # WITHOUT the group constraint, so a repetition dataset reaching them could split a sample's reps
    # across train/val (silent leakage). An unhandled composition therefore fails LOUD here (naming #21)
    # rather than taking a non-group path and running wrong.
    if _is_repetition_dataset(spectro):
        if augmentation_steps or detected is not None or detected_duplication is not None or detected_stacking is not None or detected_by_source is not None or any(_is_exclude_step(step) for step in pipeline):
            raise NotImplementedError(
                "engine='dag-ml' does not yet support a repetition dataset combined with "
                "exclude/branch/sample_augmentation (the group constraint would be lost); backlog #21."
            )
        return _run_repetition(list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "repetition", metric, task_type, dataset_pickle=host_pickle)

    # Separation branch (by_metadata/by_tag) + concat merge → ONE native fan-out run: dag-ml fans the
    # branch into one model node per partition value (discovered from the envelope metadata/tags),
    # runs per-partition FIT_CV, and the native concat-merge handler reassembles a full-universe OOF.
    # Detected on the ORIGINAL pipeline (before exclude consumption) so an exclude step beside the
    # branch is still visible — exclude+branch is rejected (out of scope) rather than silently dropped.
    if detected is not None:
        branch_step, branch_body = detected
        return _run_separation_branch(list(pipeline), branch_step, branch_body, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "branch", metric, task_type, dataset_pickle=host_pickle)

    # Duplication branch (`{"branch": [[A], [B], …]}`) + avg/mean fusion merge → ONE native run: each
    # branch is a full-data model node (NO fan-out / NO branch_view); dag-ml's native fusion merge handler
    # averages the branches' held-out Validation OOF per sample (leakage-safe) into one full-universe OOF.
    if detected_duplication is not None:
        branches, aggregate = detected_duplication
        return _run_duplication_branch(list(pipeline), branches, aggregate, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "duplication", metric, task_type, dataset_pickle=host_pickle)

    # by_source separation branch (`{"branch": {"by_source": True, "steps": [...model...]}}`) + avg/mean
    # fusion merge on a MULTI-source dataset → ONE native run: dag-ml fans the shared body into one
    # per-source model node (each bound to its source's block via metadata.source_index — LATE fusion
    # by source), and the native fusion merge handler averages the per-source held-out Validation OOF
    # per sample into one full-universe OOF. Each branch sees ALL samples but only ITS source's columns
    # (a feature-axis selection, not a sample partition like by_metadata).
    if detected_by_source is not None:
        return _run_by_source_branch(list(pipeline), detected_by_source, spectro.features_sources(), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "by_source", metric, task_type, dataset_pickle=host_pickle)

    # STACKING (backlog #10): a duplication branch (`{"branch": [[A], [B], …]}`) + `{"merge": "predictions"}`
    # + a downstream meta-model (`{"model": MetaModel(Ridge())}` or a plain `{"model": Ridge()}`) → ONE
    # native dag-ml run: each base branch model is FIT_CV on the full fold-train and predicts the full
    # fold-validation (held-out Validation OOF); the meta-node consumes those branches' Validation OOF
    # (via requires_oof+requires_fold_alignment edges, leakage-safe — train predictions are refused), fits
    # the meta-learner on the per-fold OOF meta-feature matrix and emits its own scored OOF.
    if detected_stacking is not None:
        branches, meta_learner = detected_stacking
        return _run_stacking_branch(list(pipeline), branches, meta_learner, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "stacking", metric, task_type, dataset_pickle=host_pickle)

    # A STACKING merge that is NOT the handled shape above (a per-branch predictions config, a missing /
    # mis-ordered meta-model, a MetaModel carrying unhandled options) must fail LOUD here, naming #10,
    # rather than reach the bridge's generic raw-merge error — so the deferral stays explicit.
    if any(_is_stacking_merge_step(step) for step in pipeline) and any(_is_duplication_branch_step(step) for step in pipeline):
        raise NotImplementedError(
            "engine='dag-ml' supports STACKING only as a duplication branch + {'merge': 'predictions'} + "
            "a downstream meta-model ({'model': MetaModel(Ridge())} or {'model': Ridge()}) with default "
            "options; this richer stacking shape is not yet wired (backlog #10). Use {'merge': 'mean'} for "
            "an averaging (fusion) ensemble instead."
        )

    # `sample_augmentation` → run nirs4all's REAL augmentation machinery to create the synthetic TRAIN
    # rows in the dataset, then run ONE native dag-ml CV+refit: base-grain folds (the synthetic children
    # never reach a holdout) + a CV-universe envelope carrying the children's origin/augmentation grain.
    # The model trains on base + its augmented children (host-side expansion); OOF is over base val only.
    # Detected on the ORIGINAL pipeline so it composes only with the supported transform+model+splitter
    # shape — a branch/exclude beside it is out of scope (the bridge fails loud below).
    #
    # Both leakage regimes run natively (`_run_augmentation` picks the path): a STATELESS augmenter is
    # augmented ONCE globally (#8, children shared across folds); a STATEFUL/SUPERVISED/BALANCED augmenter
    # is augmented FOLD-LOCALLY (#32, fit inside each fold's train only + a full-train refit pass), so it
    # never sees a fold's validation rows. A single augmentation step of either kind is supported here; an
    # unsupported richer shape still falls through to the bridge's raw `sample_augmentation` error.
    if augmentation_steps:
        return _run_augmentation(list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "augment", metric, task_type)

    # Consume the `exclude` step (if any) BEFORE generator handling: run the SampleFilter operator(s)
    # in Python on the full CV train pool to get the excluded sample ints, then choose the CV universe
    # per the `keep_in_oof` mode. `cv_pool` is the sample-int universe the splitter runs over;
    # `excluded` is non-empty only in the opt-in (keep_in_oof=True) leakage-pure mode.
    pipeline, cv_pool, excluded = _resolve_exclude(list(pipeline), spectro)
    # Consume handled `tag` steps AFTER the CV universe is known: tags fit on that train pool and are
    # emitted onto relations, but do not remove samples from the splitter/model pool.
    pipeline, tags_by_sample = _resolve_tags(list(pipeline), spectro, cv_pool)

    # Param-level model sweeps (`_range_`/`_log_range_`/`_grid_` on a model step) run as ONE native
    # dag-ml run: the bridge lowers them to native `generators`, the compiler expands variants, and
    # dag-ml generates + scores + SELECTs + refits the best (no Python expand). Operator-level
    # generators (`_or_`/`_cartesian_`, multi-model) stay on the Python `expand_spec` path below.
    if _generation_kind(list(pipeline)) == "param_model":
        return _run_native_generation(
            list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "native", metric, task_type, cv_pool, excluded, tags_by_sample, dataset_pickle=host_pickle
        )

    # Expand operator-level generators (_or_/_cartesian_/param-keyed _range_/_grid_/...) into concrete,
    # flat pipelines of live operator instances (nirs4all's own serialize → expand → deserialize +
    # flatten), run each through the verified single-variant dag-ml path, then select the best by its
    # CV score — mirroring nirs4all selecting the best variant by its metric.
    variants = _expand_operator_generators(list(pipeline))
    results = [
        _run_concrete(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}", metric, task_type, cv_pool, excluded, tags_by_sample, dataset_pickle=host_pickle)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN (no CV score) ranks last
            return float("inf")
        return -score if is_classification else score  # maximize accuracy, minimize RMSE

    return min(results, key=_cv_rank)


def _run_native_generation(
    pipeline: list[Any],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
) -> RunResult:
    """Run a param-level model sweep as ONE native dag-ml generation + SELECT + refit run.

    The model step keeps its generator dict so the bridge lowers it to native ``generators``; we
    apply only the plain (non-generator) sibling params to the model, never the sweep. dag-ml
    expands the variants, scores each by its cross-fold OOF ``metric``, and refits the winner —
    ``bundle.scores`` is the selected variant's, mapped to a RunResult exactly like the single path.

    ``cv_pool`` is the CV sample-int universe (the de-excluded pool in legacy mode, the full train
    in opt-in mode); ``excluded`` is marked in the envelope only in opt-in (``keep_in_oof=True``).
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_plain_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml engine run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


def _apply_plain_model_params(steps: list[Any]) -> list[Any]:
    """Apply only the PLAIN (non-generator) sibling hyperparameters to the model, keeping sweeps.

    The native path lowers param-level sweeps (``_range_``/``_log_range_``/``_grid_``) to dag-ml
    ``generators``, so they must stay on the step dict; plain siblings (e.g. ``scale=False``) are
    set on a model clone, exactly like ``_apply_model_params`` but leaving the generator dicts in
    place for the bridge to lower.
    """
    from sklearn.base import clone

    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            plain = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS and not is_param_generator_spec(value)}
            if plain:
                model = step["model"]
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**plain)
                kept = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS or is_param_generator_spec(value)}
                kept["model"] = model
                step = kept
        out.append(step)
    return out


def _run_concrete(
    pipeline: Any,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str = "rmse",
    task_type: str = "regression",
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
) -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores.

    ``cv_pool`` is the CV sample-int universe (de-excluded pool in legacy mode, full train in opt-in
    mode); ``excluded`` is marked in the envelope only in the opt-in (``keep_in_oof=True``) mode.
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml engine run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


def _run_repetition(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """Run a REPETITION (sample-grain grouped) pipeline as ONE native dag-ml CV+refit run.

    The CV universe is the repetition ROWS of the train partition (each stored row is its own
    sample int / sample_id — repetitions are NOT collapsed). Folds are GROUP-aware
    (:func:`_build_group_folds`): all replicates of a sample land on the same fold side, so every
    rep row is validated exactly once (a clean OOF partition) while a group is never split. The
    envelope carries each row's ``group_id`` (the repetition column value), and dag-ml-data's
    ``validate_fold_set_against_sample_relations`` refuses any fold that splits a group — the
    native group-leakage guarantee. Each rep row is scored individually (the repetition grain),
    which is exactly what nirs4all's ``cv_best_score``/``best_rmse`` report; the sample-level
    aggregation (the ``_agg`` twin) is a separate score nirs4all does not surface on RunResult,
    so no aggregation reducer is needed.

    Generators are expanded in Python (operator-level via ``expand_spec``; a param-level model
    sweep also goes through ``expand_spec`` here for simplicity) and each concrete variant runs
    through the group-aware path, selecting the best by its CV score — mirroring nirs4all.
    """
    from nirs4all.pipeline.config.generator import expand_spec

    variants = expand_spec(pipeline)
    results = [
        _run_repetition_concrete(variant, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, task_type, dataset_pickle=dataset_pickle)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]
    maximize = metric in ("accuracy", "r2")

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if maximize else score

    return min(results, key=_cv_rank)


def _run_repetition_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """One concrete repetition variant: group-aware folds + a ``group_id``-carrying envelope."""
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_group_folds(splitter, spectro, pool)
    group_by_sample = _repetition_grain(spectro, pool)
    envelope = build_envelope(spectro, identity, sample_ints=pool, group_by_sample=group_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml repetition run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


def _reshape_for_rep_fusion(rep_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL repetition reshape (``rep_to_sources`` / ``rep_to_pp``) in place.

    Reuses the production operator (:class:`RepetitionConfig`) + the dataset's own reshape methods
    (``reshape_reps_to_sources`` / ``reshape_reps_to_preprocessings``) — no reshape logic is
    reimplemented here. The reshape collapses the N replicate ROWS of a physical sample into either
    N sample-aligned feature SOURCES (``rep_to_sources``) or N stacked PROCESSING layers
    (``rep_to_pp``), reducing the dataset to ``n_unique`` physical samples.

    The ``repetition`` flag is cleared afterwards: it named the rep grouping of the ORIGINAL rows,
    which no longer exists once the replicates have become sources/processings — the reshaped
    dataset's unit of analysis is the physical sample, so the downstream folds must be sample-grain
    (the plain-repetition group-fold path, #21, must NOT fire on the reshaped dataset).
    """
    from nirs4all.operators.data.repetition import RepetitionConfig

    if "rep_to_sources" in rep_step:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_sources"])
        spectro.reshape_reps_to_sources(config)
    else:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_pp"])
        spectro.reshape_reps_to_preprocessings(config)
    spectro._repetition = None  # noqa: SLF001 - the rep grouping was consumed by the reshape


def _run_rep_fusion(pipeline: list[Any], rep_step: dict[str, Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str) -> RunResult:
    """Run a ``rep_to_sources`` / ``rep_to_pp`` pipeline as ONE native dag-ml CV+refit run (S7).

    REP FUSION is a one-time host RESHAPE feeding an already-native multimodal path:

    * ``rep_to_sources`` — each replicate becomes a feature SOURCE, so the reshaped dataset is
      MULTI-SOURCE; the envelope auto-emits a ``feature_block_set`` (S3 early fusion), and an MB-PLS
      model takes the per-source block list (S5 intermediate fusion) — both already native.
    * ``rep_to_pp`` — each replicate becomes a PROCESSING layer, so the reshaped dataset is
      single-source with N stacked layers; the FLAT_2D materialization hstacks them by processing
      order (the feature-axis concat S6 already does), which the legacy ``tabular_numeric`` path runs.

    The reshaped dataset lives only in host memory (the on-disk dataset has no such structure), so it
    is PICKLED for the adapter (the same mechanism ``sample_augmentation`` uses) — the adapter resolves
    the exact reshaped sources/processings, identity-keyed by the physical sample_id.

    LEAKAGE: after the reshape the unit of analysis is the physical SAMPLE (N reps → N sources/layers
    of ONE sample), so folds/OOF are over SAMPLES — distinct from a plain repetition dataset (#21,
    rep-grain). A sample's N source-blocks (or N processing layers) all ride ONE row and therefore the
    SAME fold side by construction; per-source / per-layer preprocessing fits on fold-train only (the
    per-block X-chain, like the multi-source path). No cross-sample mixing.

    Generators are expanded in Python (operator-level via ``expand_spec``), each concrete variant runs
    through the reshaped early-fusion path, and the best is selected by its CV score — mirroring nirs4all.
    """
    import pickle

    from nirs4all.pipeline.config.generator import expand_spec

    body = [step for step in pipeline if not _is_rep_fusion_step(step)]
    variants = expand_spec(body)
    run_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_rep_fusion_concrete(variant, rep_step, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, task_type, pickle)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]
    maximize = metric in ("accuracy", "r2")

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if maximize else score

    return min(results, key=_cv_rank)


def _run_rep_fusion_concrete(body: Any, rep_step: dict[str, Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, pickle: Any) -> RunResult:
    """One concrete rep-fusion variant: reshape a fresh dataset copy, then the sample-grain CV+refit."""
    import copy

    steps, splitter = _split_pipeline(body)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    # Reshape a FRESH copy per variant so each variant's pickled dataset is independent (and the
    # caller's spectro stays the original rep dataset, unmutated, for any later use).
    reshaped = copy.deepcopy(spectro)
    _reshape_for_rep_fusion(rep_step, reshaped)

    identity = mint_identity(reshaped)
    # The reshape leaves every physical sample in `partition: train` (the splitter follows the reshape),
    # so the CV universe is the full reshaped sample set. Folds are sample-grain — a sample's N
    # source-blocks / processing-layers ride ONE row, so they cannot split across the fold boundary.
    pool = reshaped.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, reshaped, pool, set())
    envelope = build_envelope(reshaped, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "reshaped_dataset.pkl"
    pickle_path.write_bytes(pickle.dumps(reshaped))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path)
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml rep-fusion run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    rep_key = "rep_to_sources" if "rep_to_sources" in rep_step else "rep_to_pp"
    return _scores_to_run_result(bundle.get("scores"), reshaped.name, f"{rep_key}_{_model_name(steps)}", metric, task_type)


def _apply_sample_augmentation(aug_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL ``SampleAugmentationController`` to add synthetic train rows in place.

    Reuses the production machinery (the controller delegates to ``TransformerMixinController``, which
    fits the augmentation transformer on fold-train data and inserts each synthetic row via
    ``dataset.add_samples_batch`` with index ``{partition: train, origin: <base sample int>,
    augmentation: <op>}``) — no augmentation logic is reimplemented here. A minimal real ``StepRunner``
    + ``RuntimeContext`` drive it (no workspace/artifacts), exactly as the orchestrator would. The
    children land in ``partition: train`` with their ``origin`` set to the base sample, which is what
    the base→child fold expansion and the envelope's augmentation grain key off.
    """
    from nirs4all.controllers.data.sample_augmentation import SampleAugmentationController
    from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, RuntimeContext, StepMetadata
    from nirs4all.pipeline.steps.parser import ParsedStep, StepType
    from nirs4all.pipeline.steps.step_runner import StepRunner

    step_info = ParsedStep(operator=None, keyword="sample_augmentation", step_type=StepType.DIRECT, original_step=aug_step, metadata={})
    context = ExecutionContext(selector=DataSelector(partition="train", processing=[["raw"]]), state=PipelineState(), metadata=StepMetadata())
    runtime_context = RuntimeContext()
    runtime_context.step_runner = StepRunner(verbose=0, mode="train")
    runtime_context.save_artifacts = False
    runtime_context.save_charts = False
    SampleAugmentationController().execute(step_info, spectro, context, runtime_context, mode="train")


def _augment_fold_train(aug_step: dict[str, Any], spectro: Any, fold_train: list[int]) -> list[tuple[int, np.ndarray]]:
    """Augment a fold's TRAIN only and return the synthetic children as ``[(origin_int, child_X(1,F)), ...]``.

    A FRESH copy of ``spectro`` is restricted so ``partition: train`` is exactly ``fold_train`` (the
    rest held out), then nirs4all's real augmentation machinery runs — so a STATEFUL/SUPERVISED/balanced
    augmenter fits inside this fold's train ONLY (its neighbors / global mean / class balance never see
    the fold's validation rows). The created children (origin in ``fold_train``) are read back as flat
    feature rows, in creation order — one tuple per child, so an origin augmented multiple times (``count``
    > 1 / a balancing factor) yields several. The caller inserts them into the master dataset and records
    the fold→children map; the fold copy is discarded. Leakage-safe by construction: each fold's children
    come only from its train.
    """
    import copy

    fold_ds = copy.deepcopy(spectro)
    fold_ds._indexer.update_by_filter({"partition": "train"}, {"partition": "hold"})  # noqa: SLF001
    fold_ds._indexer.update_by_indices(list(fold_train), {"partition": "train"})  # noqa: SLF001
    fold_ds._invalidate_content_hash()  # noqa: SLF001

    before = {int(s) for s in fold_ds.index_column("sample", {})}
    _apply_sample_augmentation(aug_step, fold_ds)
    samples = [int(s) for s in fold_ds.index_column("sample", {})]
    origins = [int(o) for o in fold_ds.index_column("origin", {})]
    children: list[tuple[int, np.ndarray]] = []
    for sample_int, origin_int in zip(samples, origins, strict=True):
        if sample_int not in before and sample_int != origin_int:
            children.append((origin_int, np.asarray(fold_ds.x_rows([sample_int], layout="2d"), dtype=float).reshape(1, -1)))
    return children


def _build_fold_local_children(aug_step: dict[str, Any], spectro: Any, base_folds: list[tuple[list[int], list[int]]], base_train: list[int]) -> tuple[dict[str, dict[int, list[int]]], dict[int, str]]:
    """Augment fold-by-fold + a full-train refit pass; insert all children into ``spectro`` in place.

    For each fold (key ``"fold{i}"``, matching :func:`build_fold_set`'s fold ids) and the full-train
    refit (key ``"refit"``) the augmenter is fit inside that train pool only (:func:`_augment_fold_train`),
    and the resulting children are appended to the master ``spectro`` as ``partition: train`` rows with
    their ``origin``. Returns ``(fold_children, augmentation_by_sample)`` where ``fold_children`` is
    ``{fold_label: {origin_int: [child_int, ...]}}`` (the resolver's fold-local expansion map) and
    ``augmentation_by_sample`` tags every inserted child for the envelope's augmentation metadata.

    All folds' children coexist in one dataset and one base-grain envelope (each child shares its
    origin's ``sample_id``, a base train id in the fold set → the origin-boundary contract holds), but
    they are kept fold-distinct host-side via the returned map — a fold's children only ever join that
    fold's fit-train (see :meth:`MaterializationResolver.expand_with_augmented_children`).
    """
    transform_label = _augmentation_label(aug_step)
    passes: list[tuple[str, list[int]]] = [(f"fold{index}", train_ints) for index, (train_ints, _val) in enumerate(base_folds)]
    passes.append(("refit", base_train))

    fold_children: dict[str, dict[int, list[int]]] = {}
    augmentation_by_sample: dict[int, str] = {}
    for fold_label, fold_train in passes:
        children = _augment_fold_train(aug_step, spectro, fold_train)
        if not children:
            fold_children[fold_label] = {}
            continue
        rows = np.stack([child_x for _origin, child_x in children])  # (n, 1, F), one row per child
        indexes = [{"partition": "train", "origin": origin_int, "augmentation": f"{transform_label}|{fold_label}"} for origin_int, _x in children]
        before = {int(s) for s in spectro.index_column("sample", {})}
        spectro.add_samples_batch(data=rows, indexes_list=indexes)
        samples = [int(s) for s in spectro.index_column("sample", {})]
        origins = [int(o) for o in spectro.index_column("origin", {})]
        by_origin: dict[int, list[int]] = {}
        for sample_int, origin_int in zip(samples, origins, strict=True):
            if sample_int not in before and sample_int != origin_int:
                by_origin.setdefault(origin_int, []).append(sample_int)
                augmentation_by_sample[sample_int] = transform_label
        fold_children[fold_label] = by_origin
    return fold_children, augmentation_by_sample


def _augmentation_grain(spectro: Any, transform_label: str) -> tuple[list[int], dict[int, str]]:
    """Post-augmentation grain: ``(augmented_ints, augmentation_by_sample)``.

    ``augmented_ints`` is every augmented child (``sample != origin``); ``augmentation_by_sample`` tags
    each with the augmentation transform id for the envelope's structured ``augmentation`` metadata. The
    fit-time base→child expansion is recomputed by the resolver from the minted identity, not here.
    """
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    augmentation_by_sample = dict.fromkeys(augmented_ints, transform_label)
    return augmented_ints, augmentation_by_sample


def _augmentation_label(aug_step: dict[str, Any]) -> str:
    """A stable transform label for the augmentation step's structured envelope metadata."""
    transformers = _augmentation_transformers(aug_step)
    return "+".join(type(transformer).__name__ for transformer in transformers) or "sample_augmentation"


def _augmentation_transformers(aug_step: dict[str, Any]) -> list[Any]:
    """The deserialized transformer instances of an augmentation step (mirrors the controller).

    A transformer may be given bare or wrapped in a ``{"transformer": op, ...}`` dict (per-transformer
    variation_scope); both forms resolve to the instance via ``deserialize_component``, exactly as
    :meth:`SampleAugmentationController.execute` parses them.
    """
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    raw = aug_step["sample_augmentation"].get("transformers", [])
    return [deserialize_component(t["transformer"] if isinstance(t, dict) and "transformer" in t else t) for t in raw]


def _operator_is_stateless(operator: Any) -> bool:
    """Whether an augmentation transformer learns NO data-dependent state in ``fit``.

    The first augmentation slice augments ONCE globally (before folds exist), so a transformer that
    fits on the whole train partition would see future fold-validation rows. That is leakage-free ONLY
    for STATELESS per-sample augmenters (Gaussian/multiplicative noise, scatter, baseline, spline,
    wavelength warps — ``fit`` only seeds an RNG); a STATEFUL/SUPERVISED one (mixup storing neighbors,
    a global-mean reference, a supervised transform) leaks. There is no declared marker on these
    operators, so the signal is twofold and conservative:

    * **supervised** — a ``requires_y`` tag (``_more_tags()``/``_tags``) means ``fit`` consumes y; reject.
    * **fit-learned data state** — fit the transformer on two differently-distributed dummy datasets and
      compare its post-fit attributes (sklearn fitted attrs ``*_`` + any ndarray, excluding the seeded
      RNG). A transformer whose state varies with the fit data carries learned state (``X_fit_``,
      ``global_mean_``, …) → stateful → reject. A clone is fit each time so no instance state is shared;
      any error during the probe is treated as NOT stateless (fail closed).
    """
    from nirs4all.controllers.transforms.transformer import TransformerMixinController

    if TransformerMixinController._requires_y(operator):  # noqa: SLF001 - reuse the supervised-tag check
        return False
    try:
        from sklearn.base import clone

        rng = np.random.default_rng(0)
        probe_a = rng.normal(size=(24, 32))
        probe_b = rng.normal(loc=8.0, scale=4.0, size=(24, 32))

        def _fit_state(data: np.ndarray) -> dict[str, Any]:
            fitted = clone(operator).fit(data)
            return {name: value for name, value in vars(fitted).items() if (name.endswith("_") or isinstance(value, np.ndarray)) and not name.startswith("_")}

        state_a, state_b = _fit_state(probe_a), _fit_state(probe_b)
        for name in set(state_a) | set(state_b):
            left, right = state_a.get(name), state_b.get(name)
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                if not (left is not None and right is not None and np.array_equal(np.asarray(left), np.asarray(right))):
                    return False
            elif left != right:
                return False
    except Exception:  # noqa: BLE001 - any probe failure ⇒ cannot prove stateless ⇒ fail closed
        return False
    return True


def _augmentation_is_leakage_free(aug_step: dict[str, Any]) -> bool:
    """Whether a ``sample_augmentation`` step is safe to run as ONE GLOBAL pre-fold augmentation.

    True (→ the GLOBAL path, #8) ONLY when neither leakage vector applies to global augmentation:

    * the **balanced/supervised mode** (a ``balance`` key) is NOT used — it fits class/bin targets on
      the whole train y, so the synthetic counts depend on the (future-fold-val-inclusive) label set;
    * EVERY transformer is stateless (:func:`_operator_is_stateless`).

    False (→ the FOLD-LOCAL path, #32) for a stateful/supervised/balanced augmentation: it is fit inside
    each fold's train only (and a full-train refit pass), so it never sees a fold's validation rows. Both
    paths are leakage-safe; this only routes which one :func:`_run_augmentation` uses.
    """
    config = aug_step["sample_augmentation"]
    if "balance" in config:
        return False
    transformers = _augmentation_transformers(aug_step)
    return bool(transformers) and all(_operator_is_stateless(transformer) for transformer in transformers)


def _run_augmentation(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str) -> RunResult:
    """Run a ``sample_augmentation`` pipeline as ONE native dag-ml CV+refit on augmented train.

    Adds the synthetic train rows (real augmentation machinery), builds BASE-grain folds (each base
    val sample validated once — a clean OOF partition that dag-ml/dag-ml-data accept) and a CV-universe
    envelope (base + augmented children; observation-grain relations carrying each child's origin +
    augmentation, deduped to the origin's sample grain in the schema). The fold-train views stay
    base-grain + ``include_augmented_train`` so the host expands each base id to base + its children at
    fit time — the children TRAIN, the OOF/validation/test never see them. The augmented dataset is
    pickled for the adapter (augmentation is stochastic, not reproducible cross-process).

    Two augmentation regimes, picked by :func:`_augmentation_is_leakage_free`:

    * **GLOBAL** (stateless per-sample augmenter) — augment ONCE on the whole train; the children are
      shared across folds. Leakage-free because the augmenter learns no data state (#8).
    * **FOLD-LOCAL** (stateful / supervised / balanced augmenter) — augment inside EACH fold's train
      (plus a full-train refit pass) separately, so each fold has its own children fit only on that
      fold's train. A ``fold_children`` map keys the resolver's per-fold expansion; it is pickled with
      the dataset so the adapter expands the right children per fold (#32). This is what makes the
      stateful case (mixup neighbors, global-mean scatter, class balancing) leakage-safe.

    Only the supported ``transform* + sample_augmentation + splitter + model`` shape runs here; the
    remaining steps are lowered through the bridge (a raw ``sample_augmentation`` still raises, keeping
    the coverage boundary). A branch / exclude / generator beside it is out of scope and fails loud.
    """
    import pickle

    aug_steps = [step for step in pipeline if _is_augmentation_step(step)]
    if len(aug_steps) != 1:
        raise NotImplementedError("engine='dag-ml' supports exactly one sample_augmentation step")
    aug_step = aug_steps[0]
    rest = [step for step in pipeline if not _is_augmentation_step(step)]
    steps, splitter = _split_pipeline(rest)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    base_train = [int(s) for s in spectro.index_column("sample", {"partition": "train"})]
    fold_local = not _augmentation_is_leakage_free(aug_step)

    # BASE-grain folds: split the base train pool only; train = base-train, val = base-val. The children
    # are NEVER listed in a fold (the FoldSet stays a clean base-grain OOF partition); they are pulled
    # into fit-train by the host expansion keyed on the origin's fold side. The split runs over the BASE
    # rows only (before any child exists), so the fold partition is identical for both augmentation paths.
    base_folds = [([base_train[i] for i in train_idx], [base_train[i] for i in val_idx]) for train_idx, val_idx in _split_pool(splitter, spectro, base_train)]

    # GLOBAL stateless augmentation (#8): fit once on the whole train (leakage-free only for stateless
    # per-sample augmenters), children shared across all folds (resolver discovers them from identity).
    # FOLD-LOCAL augmentation (#32): a STATEFUL/SUPERVISED/balanced augmenter is fit inside EACH fold's
    # train only, so each fold (+ the full-train refit) has its OWN children — leakage-safe for the
    # stateful case. `fold_children` keys the per-fold expansion; it is pickled for the adapter's resolver.
    fold_children: dict[str, dict[int, list[int]]] | None = None
    if fold_local:
        fold_children, augmentation_by_sample_int = _build_fold_local_children(aug_step, spectro, base_folds, base_train)
    else:
        _apply_sample_augmentation(aug_step, spectro)
        _augmented_ints, augmentation_by_sample_int = _augmentation_grain(spectro, _augmentation_label(aug_step))

    # Identity is minted on the AUGMENTED dataset so children get their own observation_id + the origin's
    # sample_id (augmented=True). The CV universe = base train + the augmented children.
    identity = mint_identity(spectro)
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    cv_universe = base_train + augmented_ints

    envelope = build_envelope(spectro, identity, sample_ints=cv_universe, augmentation_by_sample=augmentation_by_sample_int)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, base_folds, dsl_id="nirs4all-augmentation", n_splits=len(base_folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "augmented_dataset.pkl"
    # Fold-local pickles the dataset + the fold→children map (the resolver's per-fold expansion); the
    # global path pickles the bare dataset (the resolver discovers dataset-global children from identity).
    pickle_path.write_bytes(pickle.dumps({"dataset": spectro, "fold_children": fold_children} if fold_local else spectro))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path)
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml augmentation run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


_MERGE_NODE_ID = "merge:concat"


def _run_separation_branch(pipeline: list[Any], branch_step: dict[str, Any], branch_body: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """Run a by_metadata/by_tag separation branch + concat merge as ONE native dag-ml fan-out run.

    Lowers the branch to an ``auto_separate`` template (one branch carrying the criterion + the
    model sub-pipeline) followed by a concat ``PredictionJoin`` merge, builds the envelope with the
    criterion column's per-sample metadata so the **native** ``fan_out_data_aware_branches`` discovers
    the partition values, compiles the fanned graph (one model node per value), and drives dag-ml-cli.
    The native concat-merge handler reassembles the per-partition OOF into one full-universe OOF
    attributed to the merge node — its cross-fold average is ``cv_best_score``.

    The criterion column's metadata is emitted onto the relations; the adapter honors each fanned
    model's ``branch_view`` selector (via the sample→metadata map) so each model fits/predicts only
    its partition. The legacy nirs4all concat-merge reassembly is broken here (MERGE-E003), so this
    native path is a correction — the parity baseline is a direct sklearn-per-partition OOF.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    criterion = branch_step["branch"]
    mode, key = ("by_metadata", criterion["by_metadata"]) if "by_metadata" in criterion else ("by_tag", criterion["by_tag"])

    # The splitter lives at the top level (before the branch); the branch body is the model
    # sub-pipeline applied per partition. Drop the splitter from the body if it slipped in.
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    body_steps = [step for step in branch_body if not hasattr(step, "split")]

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())

    # Per-sample criterion values: the first map seeds the envelope relations (native fan-out reads
    # partition values from it); the second is the adapter's sample_id→metadata map for branch_view.
    metadata_by_sample, sample_metadata = _branch_metadata(spectro, identity, mode, key)
    envelope = build_envelope(spectro, identity, sample_ints=pool, metadata_by_sample=metadata_by_sample)

    # Compat auto_separate template: ONE branch (the model sub-pipeline) carrying the criterion +
    # mode, marked auto_separate; the native fan-out expands it into N per-partition branches.
    template = {"id": "per_partition", "steps": [_branch_compat_step(step) for step in body_steps]}
    # Always by_metadata mode: the criterion (whether nirs4all by_metadata or by_tag) is emitted as a
    # metadata column on the relations, so the native fan-out discovers its values from there.
    compat_dsl = {
        "id": "nirs4all-separation-branch",
        "pipeline": [
            {"branch": {"branches": [template]}, "mode": "by_metadata", "selector": {"metadata_key": key}, "metadata": {"auto_separate": True}},
            {"merge": "concat", "output_as": "predictions", "id": _MERGE_NODE_ID},
        ],
    }

    # NATIVE fan-out (no Python suffix replication): dag-ml reads the partition values from the
    # envelope relations and expands one branch per sorted value, owning the node-id suffixing.
    fanned_dsl = dag_ml.fan_out_data_aware_branches(compat_dsl, envelope).to_dict()
    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(fanned_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if not model_ids:
        raise RuntimeError("separation-branch fan-out produced no per-partition model nodes")

    # Per-partition data_bindings (one per fanned model node) + the materialized fold set. The CLI's
    # own fan-out is a no-op on the already-fanned DSL (the auto_separate marker is consumed).
    fanned_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    fanned_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=fanned_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, sample_metadata=sample_metadata, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml separation-branch run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    # The concat-merge producer's reports carry both the full-universe cross-fold OOF average
    # (`cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`): dag-ml's native
    # off-fold merge handler reassembles each per-partition refit model's held-out TEST prediction
    # (the node runner emits it with `fold_id=None`) into one full-universe test block under the merge
    # node. Both scores are the separation branch's, surfaced by `_scores_to_run_result`.
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(body_steps), metric, task_type, producer=_MERGE_NODE_ID)


def _branch_compat_step(step: Any) -> dict[str, Any]:
    """Lower one branch-body step (transform / {"model": M} / {"y_processing": op}) to compat DSL."""
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    return _step_to_dsl(step)


def _branch_metadata(spectro: Any, identity: Any, mode: str, key: str) -> tuple[dict[str, dict[int, Any]], dict[str, dict[str, Any]]]:
    """Build the criterion's per-sample metadata: ``({col: {sample_int: value}}, {wire_id: {col: value}})``.

    The first map seeds the envelope relations (native fan-out reads partition values from it); the
    second is the adapter's ``sample_id → metadata`` map for honoring each branch's ``branch_view``.
    A ``by_tag`` criterion is represented as a metadata column (the tag value per sample), matching
    the envelope's metadata-carried relations.
    """
    sample_ints = [int(value) for value in spectro.index_column("sample", {})]
    values = spectro.metadata_column(key, {}) if mode == "by_metadata" else spectro.get_tag(key, {})
    by_int = {sample_int: (str(value) if value is not None else None) for sample_int, value in zip(sample_ints, values, strict=True)}
    metadata_by_sample = {key: dict(by_int)}
    sample_metadata = {identity.to_wire(sample_int): {key: value} for sample_int, value in by_int.items()}
    return metadata_by_sample, sample_metadata


_FUSION_MERGE_NODE_ID = "merge:fusion"


def _canonical_branch_step(step: Any, node_id: str) -> dict[str, Any]:
    """Lower one branch-body step to a CANONICAL pipeline-DSL step (``kind`` + ``operator.class``).

    The duplication-fusion path emits a *canonical* ``steps`` DSL (not the compat ``pipeline`` form):
    dag-ml's nirs4all-compat importer whitelists merge modes (``concat``/``predictions``/… only) and
    REFUSES ``fusion`` — but the canonical ``PipelineDslMergeStep.merge_mode`` is a free-form string the
    runtime reads verbatim, so the fusion merge must be expressed canonically. The branch bodies must
    therefore also be canonical. This reuses the verified compat lowering
    (:func:`~nirs4all.pipeline.dagml_bridge._step_to_dsl` — operator FQN + JSON-safe params +
    native param-generator entries) and re-keys it into the canonical ``{"kind", "id", "operator":
    {"class": …}, "params": …}`` shape per node kind (transform / y_transform / model), assigning the
    explicit ``node_id`` so each branch's nodes are uniquely named.
    """
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    compat = _step_to_dsl(step)
    if "model" in compat:
        out: dict[str, Any] = {"kind": "model", "id": node_id, "operator": {"class": compat["model"]}, "params": compat.get("params", {})}
        if "generators" in compat:
            out["generators"] = compat["generators"]
        return out
    if "y_processing" in compat:
        inner = compat["y_processing"]
        return {"kind": "y_transform", "id": node_id, "operator": {"class": inner["class"]}, "params": inner.get("params", {})}
    # Bare transform: compat is {"class": FQN, "params": {...}}.
    return {"kind": "transform", "id": node_id, "operator": {"class": compat["class"]}, "params": compat.get("params", {})}


def _canonical_branch(branch_body: list[Any], branch_index: int) -> dict[str, Any]:
    """Lower one duplication sub-pipeline (a list of steps) to a canonical branch with unique node ids."""
    steps = [step for step in branch_body if not hasattr(step, "split")]
    return {
        "id": f"branch_{branch_index}",
        "steps": [_canonical_branch_step(step, f"branch:{branch_index}.node:{node_index}") for node_index, step in enumerate(steps)],
    }


def _canonical_source_branch(branch_body: list[Any], source_index: int) -> dict[str, Any]:
    """Lower the shared by_source body to a canonical branch BOUND to one source (S4).

    Same lowering as :func:`_canonical_branch` (the shared model sub-pipeline, unique node ids per
    source), but every MODEL node carries ``metadata.source_index`` so the node runner materializes
    ONLY that source's feature block — late fusion by source. The branch index IS the source index
    (one branch per source), so a fold view stays full-sample (all branches see all samples) while
    each branch's model sees a different source's columns.
    """
    branch = _canonical_branch(branch_body, source_index)
    for node in branch["steps"]:
        if node["kind"] == "model":
            node["metadata"] = {**node.get("metadata", {}), "source_index": source_index}
    return branch


def _run_by_source_branch(pipeline: list[Any], branch_body: list[Any], n_sources: int, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """Run a by_source separation branch + avg/mean fusion merge as ONE native dag-ml run (S4).

    LATE fusion BY SOURCE: fans the shared body into one canonical branch PER feature source
    (:func:`_canonical_source_branch`), each MODEL node bound to its source via ``metadata.source_index``
    so the node runner feeds it ONLY that source's block (all samples, that source's columns). The fold
    set, OOF, and merge are sample-keyed and identical to the duplication-fusion path — the ONLY
    difference from duplication is the feature-axis (per-source) restriction each branch's model sees.

    dag-ml runs ONE native CV+refit: each per-source branch model is FIT_CV on the full fold-train
    (its source's columns) and predicts the full fold-validation (held-out OOF); the native fusion
    merge handler averages the branches' per-sample Validation OOF (leakage-safe — train predictions
    never enter the average) into one full-universe OOF attributed to the merge node, whose cross-fold
    average is ``cv_best_score``. ``best_rmse`` (final test) is also native: each branch's REFIT predicts
    the held-out TEST set (its source's columns, ``fold_id=None``) and dag-ml's off-fold fusion handler
    averages those per sample into one scored ``(test, fold_id=None)`` block under the merge node.

    LEAKAGE: folds/OOF over SAMPLES (unchanged — all branches see all samples, just different source
    columns); a sample's blocks all land on the same fold side (the source restriction is a feature-axis
    selection, never a sample partition); per-source per-fold preprocessing fits on fold-train only
    (the per-branch X-chain is fit inside the fold's train materialization, like the single-block path);
    the fusion merge is OOF-safe. No cross-sample mixing.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication-mode branch with N per-source sub-pipelines (each the shared body
    # bound to its source via metadata.source_index) + a fusion merge. The branch is `duplication` mode
    # because every branch sees the FULL fold sample view (no fan-out / no branch_view / no
    # sample_metadata) — the per-source restriction is applied host-side at materialization, not by a
    # sample-partition branch_view.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-by-source-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_source_branch(branch_body, source_index) for source_index in range(n_sources)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": "fusion", "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) != n_sources:
        raise RuntimeError(f"by_source compile produced {len(model_ids)} model nodes, expected {n_sources}")

    # One data_binding per per-source model node (each binds its `x` to the full source set; the node
    # runner selects its own source's block by metadata.source_index) + the materialized fold set.
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml by_source run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    model_label = f"by_source_{_model_name(branch_body)}x{n_sources}"
    return _scores_to_run_result(bundle.get("scores"), spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID)


def _run_duplication_branch(pipeline: list[Any], branches: list[list[Any]], aggregate: str, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """Run a duplication branch (``[[A], [B], …]``) + avg/mean fusion merge as ONE native dag-ml run.

    Lowers each inner sub-pipeline to a canonical branch (``mode: "duplication"`` — every branch model
    node gets the FULL fold data view: NO fan-out, NO ``auto_separate``, NO ``branch_view``,
    NO ``sample_metadata``) and a fusion merge node (``merge_mode: "fusion"`` for the value mean,
    ``output_as: "predictions"``). dag-ml runs ONE native CV+refit: each branch model is FIT_CV on the
    full fold-train and predicts the full fold-validation (held-out OOF); the native fusion merge handler
    averages the branches' per-sample Validation OOF (leakage-safe — train predictions never enter the
    average) into one full-universe OOF attributed to the merge node, whose cross-fold average is
    ``cv_best_score``.

    ``best_rmse`` (final test) is also native: each branch's REFIT predicts the held-out TEST set
    (the node runner emits it with ``fold_id=None``), and dag-ml's off-fold fusion handler
    (``reassemble_branch_merge_off_fold``) averages those base test predictions per sample into one
    scored ``(test, fold_id=None)`` block under the merge node — the same average as ``cv_best_score``
    but over the held-out test set.

    Classification (``fusion_proba_mean``) is NOT wired: the node runner emits scalar value predictions,
    not per-class probability rows, so a probability-mean fusion has no proba blocks to average — it
    fails loud rather than averaging class labels (which is not what ``proba_mean`` means).
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    if aggregate == "proba_mean":
        raise NotImplementedError(
            "engine='dag-ml' does not yet support proba-mean fusion (classification): the process adapter "
            "emits class-label predictions, not per-class probability rows; backlog #20-avg (proba)."
        )
    merge_mode = "fusion"

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N sub-pipelines (each on the FULL data) + a fusion merge.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-duplication-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": merge_mode, "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) < 2:
        raise RuntimeError("duplication-fusion compile produced fewer than two model nodes")

    # One data_binding per branch model node (each binds its `x` to the full source) + the materialized
    # fold set. Every model node sees the full fold view — no branch_view/sample_metadata (duplication).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml duplication-fusion run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    # The fusion-merge producer's reports carry the full-universe cross-fold OOF average (the fused
    # ensemble's `cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`, the
    # branches' test predictions averaged per sample). Both are surfaced by `_scores_to_run_result`.
    model_label = "+".join(_model_name(branch) for branch in branches)
    return _scores_to_run_result(bundle.get("scores"), spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID)


_META_NODE_ID = "merge:stack"


def _run_stacking_branch(pipeline: list[Any], branches: list[list[Any]], meta_learner: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None) -> RunResult:
    """Run a duplication branch + ``{"merge": "predictions"}`` + meta-model as ONE native dag-ml run (#10).

    Lowers each inner sub-pipeline to a canonical duplication branch (``mode: "duplication"`` — each base
    model node gets the FULL fold data view) + a ``merge_model`` meta-node carrying the meta-learner (its
    FQN as ``operator.class`` so the node runner instantiates it) bound to ``controller:nirs4all.meta_model``
    (which declares ``consumes_oof_predictions``, so dag-ml's planner permits the base→meta ``requires_oof``
    edges). dag-ml runs ONE native CV+refit:

    * each base branch model is FIT_CV on the full fold-train and predicts the full fold-validation
      (held-out Validation OOF);
    * the meta-node receives the base branches' **Validation OOF** per fold (Option A: the runtime delivers
      ``prediction_inputs[*].values`` aligned by sample_id, sourced ONLY from Validation blocks — the
      ``requires_oof`` edge refuses any train block), builds the per-fold meta-feature matrix (columns in
      deterministic producer order), fits the meta-learner and emits its own scored Validation OOF.

    The meta producer's cross-fold OOF average is ``cv_best_score`` — the stacking ensemble's CV score.
    ``best_rmse`` (final test) is also native: in REFIT dag-ml delivers each base producer's held-out
    TEST prediction to the meta-node as a SEPARATE off-fold input keyed ``…oof:refit`` (partition Test,
    ``fold_id=None``), alongside the full Validation OOF the meta fits on. The refit meta-model predicts
    the test set from those base TEST meta-features and emits a scored ``(test, fold_id=None)`` block.

    LEAKAGE INVARIANT: the meta-learner is fit on Validation OOF ONLY (the ``requires_oof`` edge +
    ``collect_oof_prediction_input`` enforce Validation-only); the TEST meta-features come from the base
    models' TEST predictions (the ``:refit`` off-fold input, phase-gated to REFIT), never their OOF/train,
    and never enter the FIT_CV meta-features.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for
    from nirs4all.pipeline.dagml_bridge import _META_MODEL_CONTROLLER_ID, _META_MODEL_REF, _json_safe_params, _qualname

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N base sub-pipelines (each on the FULL data) + a
    # merge_model meta-node. The meta-node carries the bare sklearn meta-learner (FQN + params) and the
    # _META_MODEL_REF (so its dedicated manifest is not a generic model-kind catch-all) and binds to the
    # meta-model controller via metadata.controller_id.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-stacking",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {
                "kind": "merge_model",
                "id": _META_NODE_ID,
                "operator": {"class": _qualname(meta_learner), "ref": _META_MODEL_REF},
                "params": _json_safe_params(meta_learner),
                "metadata": {"controller_id": _META_MODEL_CONTROLLER_ID},
            },
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    base_model_ids = [model_id for model_id in model_ids if model_id != _META_NODE_ID]
    if len(base_model_ids) < 2:
        raise RuntimeError("stacking compile produced fewer than two base model nodes")
    if _META_NODE_ID not in model_ids:
        raise RuntimeError("stacking compile produced no meta-model node")

    # One data_binding per BASE model node (each binds its `x` to the full source). The meta-node has NO
    # data binding: its features are the base branches' OOF, delivered as prediction_inputs (not data).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(base_model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml stacking run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    # The meta-node producer's reports carry the full-universe cross-fold OOF average (the stacking
    # ensemble's `cv_best_score`) AND a `(test, fold_id=None)` block (`best_rmse`): the refit meta-model
    # predicting the held-out test from the base producers' REFIT-test predictions (`…oof:refit`).
    model_label = f"MetaModel_{type(meta_learner).__name__}"
    return _scores_to_run_result(bundle.get("scores"), spectro.name, model_label, metric, task_type, producer=_META_NODE_ID)


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str, metric: str = "rmse", task_type: str = "regression", producer: str | None = None) -> RunResult:
    """Map a dag-ml ScoreSet into a RunResult, mirroring nirs4all's entry shape.

    ``producer`` filters to one ``producer_node`` — e.g. a separation/duplication/stacking merge node,
    whose reports carry the full-universe OOF average (``cv_best_score``) and — since dag-ml routes the
    base branches' REFIT-test predictions to the merge node (``reassemble_branch_merge_off_fold`` for
    concat/fusion; the ``:refit`` off-fold input for stacking) — a reassembled ``(test, fold_id=None)``
    block (``best_rmse``); ``None`` keeps all reports (the single-model path, where exactly one producer
    scores).

    dag-ml emits one report per (partition, fold). nirs4all's RunResult expects per-fold validation
    entries + a single combined **refit/final** entry that carries val (the CV score), test and train
    scores together — that combined entry is what `best`/`best_rmse`/`best_final` resolve. We build
    exactly that: an `avg` CV entry (`cv_best_score`), and one `final` entry (`fold_id="final"`,
    `partition="val"`) with val_score=OOF-average, test_score from the producer's `(test, None)` block,
    train_score=final-train, and a combined `scores` dict.
    """
    reports = [report for report in (scores or {}).get("reports", []) if producer is None or report.get("producer_node") == producer]
    by_key = {(report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in reports}

    def add(fold_id: str | None, partition: str, scores_map: dict[str, dict[str, float]], *, val: float | None = None, test: float | None = None, train: float | None = None) -> None:
        predictions.add_prediction(dataset_name=dataset_name, model_name=model_name, fold_id=fold_id, partition=partition, metric=metric, task_type=task_type, scores=scores_map, val_score=val, test_score=test, train_score=train)

    # Two entries: the `avg` CV entry (cv_best_score) and the combined refit `final` entry that holds
    # val + test + train scores (best/best_rmse/best_accuracy resolve from it). Per-fold val entries
    # are omitted — they would let get_best(score_scope="all") rank a single fold first.
    #
    # _rank_candidates ranks an is_final entry by its OWN partition's metric, so the final entry uses
    # partition="val" (ranks on the CV metric, same axis as the avg) with the ranking value nudged a
    # negligible epsilon in the better direction (maximize accuracy, minimize rmse). That makes the
    # refit entry win the get_best tie over the avg for BOTH task directions; reported scores come
    # from the unmodified `scores` dict, so best_rmse/best_accuracy read the true final-test value.
    maximize = metric in ("accuracy", "r2")
    predictions = Predictions()
    # The CV score is the cross-fold OOF average ("avg") when dag-ml concatenated multiple folds. A
    # single-fold splitter (a train/validation split, e.g. KennardStone/SPXY with n_splits=1) emits no
    # "avg" — just the one validation fold — so fall back to that single fold's report as the CV score.
    avg = by_key.get(("validation", "avg"))
    if avg is None:
        validation_folds = [value for (partition, fold_id), value in by_key.items() if partition == "validation"]
        if len(validation_folds) == 1:
            avg = validation_folds[0]
    # The held-out TEST score (best_rmse) is the producer's `(test, fold_id=None)` block: the off-fold
    # convention the node runner emits for a model's REFIT test prediction AND that dag-ml's native
    # concat/fusion/stacking merge handlers reassemble under the merge producer. `producer` already
    # scopes `by_key` to the right node (the merge node for a merge pipeline; the single model otherwise).
    test = by_key.get(("test", None))
    train = by_key.get(("final", None))
    if avg is not None:
        add("avg", "val", {"val": avg}, val=avg.get(metric))
    if test is not None or train is not None:
        rank_val = dict(avg) if avg is not None else {}
        if metric in rank_val:
            rank_val[metric] += 1e-9 if maximize else -1e-9
        combined: dict[str, dict[str, float]] = {"val": rank_val}
        if train is not None:
            combined["train"] = train
        if test is not None:
            combined["test"] = test
        add("final", "val", combined, val=(avg or {}).get(metric), test=(test or {}).get(metric), train=(train or {}).get(metric))

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})
