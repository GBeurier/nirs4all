"""Materialization resolver — the mechanism-agnostic real-data fetch.

dag-ml's control core never sees feature matrices; it asks a host for the rows of a
*view*, keyed by stable identity. This resolver is the single object both execution
mechanisms call (the CLI process-adapter deserializes wire ids from a ``NodeTask``; an
in-process C-ABI callback hands them as a Python list) — it takes only JSON-friendly
identity refs and returns plain Python values, so it carries zero FFI.

It is the real ``sample_id → X/y`` fetch that the shipped conformance adapters *fake*
(they synthesize features by hashing sample ids). Two correctness invariants, both
verified against the live ``SpectroDataset``:

* **Order is restored to the caller's request.** ``SpectroDataset.x({"sample": ids})``
  returns rows in ascending storage order, *not* request order, so a naive pass-through
  would silently misalign dag-ml's identity join. The resolver re-keys the returned
  block by the authoritative fetch order (``index_column("sample", …)``) and re-emits
  rows in the requested order — identity-keyed, never positional.
* **Real spectra, not a hash.** Values come from ``SpectroDataset.x``/``.y``.

Scope: single-source / no-repetition baseline. A request whose stored-row count differs
from its unique-sample count (augmentation / repetitions) raises rather than guessing.
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

    def partition_wire_ids(self, partition: str) -> list[str]:
        """Wire sample ids for a dataset partition (e.g. ``"test"``), empty if none.

        Lets the adapter predict a held-out partition the CV fold set does not cover (the final
        model's test predictions): dag-ml only scope-checks ``validation`` predictions, so a
        ``test``/``final`` block for these ids is accepted and scored natively.
        """
        return [self._identity.to_wire(sample_int) for sample_int in self._dataset.index_column("sample", {"partition": partition})]

    def _ordered_rows(self, sample_ints: list[int], block: np.ndarray, returned: list[int]) -> list[int]:
        """Row index in ``block`` for each requested sample int, restoring request order."""
        if len(returned) != block.shape[0]:
            raise NotImplementedError(
                "resolver baseline supports one stored row per sample (single-source, no augmentation/repetition); "
                f"got {block.shape[0]} rows for {len(returned)} samples"
            )
        row_of = {sample_int: row for row, sample_int in enumerate(returned)}
        return [row_of[sample_int] for sample_int in sample_ints]

    def resolve_features(
        self,
        observation_ids: list[str],
        *,
        include_augmented: bool = True,
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{feature_set_id, observation_ids, values}`` for the requested view, in order."""
        sample_ints = [self._identity.to_int(observation_id) for observation_id in observation_ids]
        uniq = list(dict.fromkeys(sample_ints))
        returned = self._dataset.index_column("sample", {"sample": uniq})
        block = np.asarray(self._dataset.x({"sample": uniq}, layout="2d", include_augmented=include_augmented, include_excluded=include_excluded))
        rows = self._ordered_rows(sample_ints, block, returned)
        return {
            "feature_set_id": "features",
            "observation_ids": list(observation_ids),
            "values": block[rows].tolist(),
        }

    def resolve_targets(
        self,
        sample_ids: list[str],
        *,
        target_id: str = "y",
        include_excluded: bool = False,
    ) -> dict[str, Any]:
        """Return ``{target_id, sample_ids, values}`` for the requested samples, in order."""
        sample_ints = [self._identity.to_int(sample_id) for sample_id in sample_ids]
        uniq = list(dict.fromkeys(sample_ints))
        returned = self._dataset.index_column("sample", {"sample": uniq})
        block = np.asarray(self._dataset.y({"sample": uniq}, include_augmented=False, include_excluded=include_excluded)).reshape(len(returned), -1)
        rows = self._ordered_rows(sample_ints, block, returned)
        return {
            "target_id": target_id,
            "sample_ids": list(sample_ids),
            "values": block[rows].ravel().tolist(),
        }
