"""Materialization resolver — the mechanism-agnostic real-data fetch.

dag-ml's control core never sees feature matrices; it asks a host for the rows of a
*view*, keyed by stable identity. This resolver is the single object both execution
mechanisms call (the CLI process-adapter deserializes wire ids from a ``NodeTask``; an
in-process C-ABI callback hands them as a Python list) — it takes only JSON-friendly
identity refs and returns plain Python values, so it carries zero FFI.

It is the real ``sample_id → X/y`` fetch that the shipped conformance adapters *fake*
(they synthesize features by hashing sample ids). Two correctness invariants, both
verified against the live ``SpectroDataset``:

* **Order is the caller's request, by identity.** Features are fetched with
  :meth:`SpectroDataset.x_rows`, which addresses each stored row by its own ``sample`` int
  (base *or* augmented) and returns rows in request order — identity-keyed, never
  positional. (The base-keyed ``x`` path expands base→augmented and returns ascending
  storage order, dropping augmented-only ids; ``x_rows`` is the observation-grain read.)
* **Real spectra, not a hash.** Values come from ``SpectroDataset.x_rows``/``.y``.

Augmented rows are observation-grain: a train view's ``observation_ids`` may include
augmented children; a validation/predict view (``include_augmented=False``) must not — the
resolver refuses an augmented child in such a view (the origin-boundary leakage guard at the
host boundary, mirroring exclude's ``include_excluded`` discipline). A child's *target* is
its origin's y, so ``resolve_targets`` is keyed by the origin's ``sample_id`` (a base id).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

    from .identity import IdentityMap


class MaterializationResolver:
    """Resolve dag-ml view identity refs to real ``SpectroDataset`` X/y, in request order.

    ``fold_children`` (``{fold_label: {origin_sample_int: [child_sample_int, ...]}}``) carries
    **fold-local** augmentation: a stateful/supervised/balanced augmenter fits inside each fold's
    train only, so each fold has its OWN synthetic children (different per fold). When supplied, the
    fit-view expansion is keyed by the task's ``fold_id`` (the ``"refit"`` key for the full-train
    refit), so a fold's children only ever join that fold's fit-train and never another fold's
    holdout. Omitted (``None``) for the GLOBAL stateless slice (#8): the children are dataset-global
    and discovered from the identity grain (every augmented row, regardless of fold).
    """

    def __init__(
        self,
        dataset: SpectroDataset,
        identity: IdentityMap,
        fold_children: dict[str, dict[int, list[int]]] | None = None,
    ) -> None:
        self._dataset = dataset
        self._identity = identity
        self._augmented_observation_ids = frozenset(
            sample.observation_id for sample in identity.identities if sample.augmented
        )
        # origin sample_int → the augmented children's observation ids, in dataset order. Lets a
        # fit (training) view expand a base id to base + its augmented children, the host-side
        # equivalent of dag-ml delivering base ids + include_augmented_train=true (the core stays
        # base-grain so the FoldSet validates; the children only ever appear in a TRAIN materialization).
        self._children_of_origin: dict[int, list[str]] = {}
        self._augmented_order: list[str] = []
        self._origin_of_observation: dict[str, int] = {}
        # observation_id → the sample_id of its target grain (an augmented child's origin sample_id;
        # a base row's own sample_id). resolve_targets is keyed by sample_id, so a fit view whose ids
        # are observation_ids (after augmented expansion) maps each to its target's sample_id here.
        self._sample_id_of_observation: dict[str, str] = {}
        for sample in identity.identities:
            self._sample_id_of_observation[sample.observation_id] = sample.sample_id
            self._origin_of_observation[sample.observation_id] = sample.origin_int
            if sample.augmented:
                self._augmented_order.append(sample.observation_id)
                self._children_of_origin.setdefault(sample.origin_int, []).append(sample.observation_id)
        # Fold-local children: per (fold_label, origin) → child observation ids (the fold's own
        # synthetic rows). Translated from sample ints to wire observation ids once here.
        self._fold_children: dict[str, dict[int, list[str]]] | None = None
        self._fold_children_order: dict[str, list[str]] | None = None
        if fold_children is not None:
            self._fold_children = {
                fold_label: {origin_int: [self._identity.to_wire(child_int) for child_int in child_ints] for origin_int, child_ints in by_origin.items()}
                for fold_label, by_origin in fold_children.items()
            }
            self._fold_children_order = {
                fold_label: [self._identity.to_wire(child_int) for child_ints in by_origin.values() for child_int in child_ints]
                for fold_label, by_origin in fold_children.items()
            }

    def is_multi_source(self) -> bool:
        """Whether the backing dataset has more than one feature source (>1 ``FeatureSource``).

        The node runner consults this to decide intermediate fusion: a multi-block model (MB-PLS) on a
        multi-source dataset receives the per-source LIST (:meth:`resolve_feature_blocks`); a single-source
        dataset always keeps the concat path (the one-block list would be identical to early fusion).
        """
        return self._dataset.features_sources() > 1

    def target_sample_ids(self, observation_ids: list[str]) -> list[str]:
        """Map observation ids to their target grain's ``sample_id`` (an augmented child → its origin).

        ``resolve_targets`` is keyed by ``sample_id`` (the y grain). A fit materialization that has been
        expanded with augmented children carries child *observation* ids, whose target is the origin's y;
        this collapses each to the origin's ``sample_id`` so ``resolve_targets`` fetches the right row.
        """
        return [self._sample_id_of_observation[observation_id] for observation_id in observation_ids]

    def expand_with_augmented_children(self, observation_ids: list[str], fold_label: str | None = None) -> list[str]:
        """Append augmented children after the base ids, preserving dataset order.

        The fit (training) materialization for a base-grain fold/full-train view: dag-ml keeps the
        view base-grain (so the FoldSet is a clean OOF partition) and signals ``include_augmented=true``;
        the host expands it. A no-op when no augmentation ran (the children map is empty). Already-augmented
        ids pass through unchanged (they carry no further children).

        With **fold-local** augmentation, ``fold_label`` (the task's ``fold_id``, or ``"refit"`` for the
        full-train refit) selects that fold's OWN children — each fold has different children because the
        augmenter was fit inside that fold's train only. A fold with no entry expands to no children (a
        base-only view); a global (non-fold-local) augmentation ignores ``fold_label`` and uses the
        dataset-wide children map. Required when fold-local: a base id requested for a fold with no fold
        children map is an internal error (the run never built one).
        """
        origin_ints = {self._identity.to_int(observation_id) for observation_id in observation_ids}
        if self._fold_children is not None:
            by_origin = self._fold_children.get(fold_label or "", {})
            fold_order = (self._fold_children_order or {}).get(fold_label or "", [])
            children = [
                child_id
                for child_id in fold_order
                if self._origin_of_observation.get(child_id) in origin_ints
            ]
            if not children and by_origin:
                children = [child_id for origin_int in origin_ints for child_id in by_origin.get(origin_int, [])]
            return [*observation_ids, *children]
        if not self._children_of_origin:
            return list(observation_ids)
        children = [
            child_id
            for child_id in self._augmented_order
            if self._origin_of_observation.get(child_id) in origin_ints
        ]
        return [*observation_ids, *children]

    def partition_wire_ids(self, partition: str) -> list[str]:
        """Wire sample ids for a dataset partition (e.g. ``"test"``), empty if none.

        Lets the adapter predict a held-out partition the CV fold set does not cover (the final
        model's test predictions): dag-ml only scope-checks ``validation`` predictions, so a
        ``test``/``final`` block for these ids is accepted and scored natively.
        """
        return [self._identity.to_wire(sample_int) for sample_int in self._dataset.index_column("sample", {"partition": partition})]

    def resolve_features(
        self,
        observation_ids: list[str],
        *,
        include_augmented: bool = True,
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{feature_set_id, observation_ids, values}`` for the requested view, in order.

        Each observation id addresses its own stored row (base or augmented) via
        :meth:`SpectroDataset.x_rows`, so rows come back in request order with no
        positional re-keying. When ``include_augmented`` is ``False`` (a validation/predict
        view) an augmented observation id in the request is refused — an augmented child must
        never reach a validation/OOF view (the origin-boundary leakage guard).
        """
        self._guard_origin_boundary(observation_ids, include_augmented)
        sample_ints = [self._identity.to_int(observation_id) for observation_id in observation_ids]
        # Return the ndarray as-is (no .tolist()): the host must fit on the dataset's NATIVE storage
        # dtype (x_rows preserves it — float32 for the legacy SpectroDataset contract), not a float64
        # widening. A .tolist() round-trips to Python doubles, so the estimator would see float64 while
        # legacy feeds float32; on a fixed-seed tree ensemble that ~1e-7 input shift tips split
        # thresholds and the fitted trees diverge. Keeping the array preserves byte-level parity.
        block = np.asarray(self._dataset.x_rows(sample_ints, layout="2d"))
        return {
            "feature_set_id": "features",
            "observation_ids": list(observation_ids),
            "values": block,
        }

    def resolve_feature_blocks(
        self,
        observation_ids: list[str],
        *,
        include_augmented: bool = True,
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{feature_set_id, observation_ids, blocks}`` — the per-source feature blocks (S5).

        INTERMEDIATE fusion: a multi-block model (MB-PLS) must receive its inputs as a LIST of
        per-source matrices ``[X_src0, X_src1, …]`` (``concat_source=False``), NOT the early-fusion
        feature-axis concatenation :meth:`resolve_features` delivers — the model fuses the blocks
        itself. Each block is one source's rows, sample-aligned to ``observation_ids`` in request
        order (``x_rows`` addresses each stored row by its own sample int, never positionally), so
        block ``b``'s row ``i`` and block ``c``'s row ``i`` are the SAME sample — the identity-keyed
        alignment fusion relies on. A single-source dataset yields a one-element list (the degenerate
        block == the early-fusion matrix). The same origin-boundary leakage guard as
        :meth:`resolve_features` applies: an augmented child is refused in a non-augmented view.
        """
        self._guard_origin_boundary(observation_ids, include_augmented)
        sample_ints = [self._identity.to_int(observation_id) for observation_id in observation_ids]
        per_source = self._dataset.x_rows(sample_ints, layout="2d", concat_source=False)
        # x_rows(concat_source=False) returns a list of per-source 2D arrays for a multi-source
        # dataset, or a single 2D array for a single source — normalize to a list either way.
        blocks = per_source if isinstance(per_source, list) else [per_source]
        # Preserve each source block's NATIVE storage dtype (no .tolist() widening to float64) — same
        # parity reason as resolve_features: the host fits on what legacy dataset.x() returns (float32).
        return {
            "feature_set_id": "features",
            "observation_ids": list(observation_ids),
            "blocks": [np.asarray(block) for block in blocks],
        }

    def resolve_source_block(
        self,
        observation_ids: list[str],
        source_index: int,
        *,
        include_augmented: bool = True,
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{feature_set_id, observation_ids, values}`` for ONE source's block (S4 by_source).

        LATE fusion BY SOURCE: a per-source branch model sees only its own source's features — a
        feature-axis selection, NOT a sample partition (all samples are present, only the source's
        columns are kept). This selects block ``source_index`` from the per-source blocks
        (:meth:`resolve_feature_blocks`), so block ``b``'s rows are one source's rows, sample-aligned
        to ``observation_ids`` in request order (``x_rows`` addresses each stored row by its own sample
        int, never positionally). The same origin-boundary leakage guard as :meth:`resolve_features`
        applies: an augmented child is refused in a non-augmented (validation/predict) view.
        """
        blocks = self.resolve_feature_blocks(observation_ids, include_augmented=include_augmented, include_excluded=include_excluded)["blocks"]
        if not 0 <= source_index < len(blocks):
            raise ValueError(f"by_source block index {source_index} out of range for {len(blocks)} source(s)")
        return {
            "feature_set_id": "features",
            "observation_ids": list(observation_ids),
            "values": blocks[source_index],
        }

    def _guard_origin_boundary(self, observation_ids: list[str], include_augmented: bool) -> None:
        """Refuse an augmented child in a non-augmented (validation/predict) view — the leakage guard."""
        if include_augmented:
            return
        leaked = [oid for oid in observation_ids if oid in self._augmented_observation_ids]
        if leaked:
            raise ValueError(
                "augmented observation ids in a non-augmented (validation/predict) view "
                f"would leak across the origin boundary: {leaked}"
            )

    def resolve_targets(
        self,
        sample_ids: list[str],
        *,
        target_id: str = "y",
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{target_id, sample_ids, values}`` for the requested samples, in order.

        Keyed by ``sample_id`` (the origin's id for an augmented child), so a child's target
        is its origin's y. Base sample ids fetch one stored row each; the request order is
        restored by re-keying the storage-ordered block.

        ``values`` shape follows the target width: a **single-target** dataset returns a flat
        ``[y0, y1, …]`` list (byte-identical to the legacy ``.ravel()``); a **multi-target**
        dataset (``num_targets>1``) returns the ``(n, n_targets)`` block as a list-of-rows so the
        per-target columns ride inside one sample-keyed block (S0 multi-target emit). The fold/OOF
        partition is over SAMPLES, so per-target columns stay leakage-safe by construction.
        """
        sample_ints = [self._identity.to_int(sample_id) for sample_id in sample_ids]
        uniq = list(dict.fromkeys(sample_ints))
        returned = self._dataset.index_column("sample", {"sample": uniq})
        block = np.asarray(self._dataset.y({"sample": uniq}, include_augmented=False, include_excluded=include_excluded)).reshape(len(returned), -1)
        row_of = {sample_int: row for row, sample_int in enumerate(returned)}
        rows = [row_of[sample_int] for sample_int in sample_ints]
        ordered = block[rows]
        values = ordered.ravel().tolist() if ordered.shape[1] == 1 else ordered.tolist()
        return {
            "target_id": target_id,
            "sample_ids": list(sample_ids),
            "values": values,
        }
