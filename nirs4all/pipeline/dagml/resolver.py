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
    """Resolve dag-ml view identity refs to real ``SpectroDataset`` X/y, in request order."""

    def __init__(self, dataset: SpectroDataset, identity: IdentityMap) -> None:
        self._dataset = dataset
        self._identity = identity
        self._augmented_observation_ids = frozenset(
            sample.observation_id for sample in identity.identities if sample.augmented
        )

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
        if not include_augmented:
            leaked = [oid for oid in observation_ids if oid in self._augmented_observation_ids]
            if leaked:
                raise ValueError(
                    "augmented observation ids in a non-augmented (validation/predict) view "
                    f"would leak across the origin boundary: {leaked}"
                )
        sample_ints = [self._identity.to_int(observation_id) for observation_id in observation_ids]
        block = np.asarray(self._dataset.x_rows(sample_ints, layout="2d"))
        return {
            "feature_set_id": "features",
            "observation_ids": list(observation_ids),
            "values": block.tolist(),
        }

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
        """
        sample_ints = [self._identity.to_int(sample_id) for sample_id in sample_ids]
        uniq = list(dict.fromkeys(sample_ints))
        returned = self._dataset.index_column("sample", {"sample": uniq})
        block = np.asarray(self._dataset.y({"sample": uniq}, include_augmented=False, include_excluded=include_excluded)).reshape(len(returned), -1)
        row_of = {sample_int: row for row, sample_int in enumerate(returned)}
        rows = [row_of[sample_int] for sample_int in sample_ints]
        return {
            "target_id": target_id,
            "sample_ids": list(sample_ids),
            "values": block[rows].ravel().tolist(),
        }
