"""Stable sample identity for the nirs4all → dag-ml(-data) bridge.

``SpectroDataset``'s only identity is the integer ``sample`` column. It equals the
positional row at load time and is renumbered ``0..N-1`` on every re-parse, so it is the
correct *in-memory* join key but an unacceptable *cross-run wire id* — dag-ml is
identity-keyed and refuses row-position joins. This module mints a content-derived
stable wire id per sample and keeps the bidirectional map the resolver uses to answer a
dag-ml view (which speaks wire ids) with the right ``SpectroDataset`` rows.

Wire ids use ``.`` separators: dag-ml-data ids validate as ``[A-Za-z0-9_.-]`` (≤128
bytes) and **reject** ``:`` (that is dag-ml's graph-id style, not a data id).

The two grains diverge for augmented rows: an augmented child gets its own
``observation_id`` (feature-level, one per stored row) while its ``sample_id`` stays the
origin's grouping key, so the origin-boundary check sees the child grouped with its base.
For a base row the grains coincide (``origin == sample``). ``mint_identity`` covers
augmented rows when run on a dataset that already holds them. Repetitions are the same
divergence (several stored rows share one ``sample_id``) and reuse this machinery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")


def validate_data_id(wire: str) -> str:
    """Return ``wire`` if it is a legal dag-ml-data id, else raise ``ValueError``."""
    if not _ID_RE.match(wire):
        raise ValueError(f"invalid dag-ml-data id {wire!r}: must match {_ID_RE.pattern} (alnum + '_.-', no ':', ≤128 bytes)")
    return wire


@dataclass(frozen=True)
class SampleIdentity:
    """One sample's identity grains.

    ``observation_id`` is feature-level (one per stored row); ``sample_id`` is
    target-level (origin-keyed, deduped over repetitions). For the baseline they are
    equal because ``origin == sample`` and there are no repetitions.
    """

    sample_int: int
    origin_int: int
    observation_id: str
    sample_id: str
    augmented: bool


@dataclass(frozen=True)
class IdentityMap:
    """Bidirectional map between SpectroDataset ``sample`` ints and stable wire ids."""

    fingerprint: str
    identities: tuple[SampleIdentity, ...]
    _to_int: dict[str, int] = field(repr=False)
    _to_wire: dict[int, str] = field(repr=False)

    def to_int(self, observation_id: str) -> int:
        """Map a wire ``observation_id`` back to its in-memory ``sample`` int."""
        try:
            return self._to_int[observation_id]
        except KeyError as exc:
            raise KeyError(f"unknown observation_id {observation_id!r}") from exc

    def to_wire(self, sample_int: int) -> str:
        """Map a ``sample`` int to its stable wire ``observation_id``."""
        try:
            return self._to_wire[sample_int]
        except KeyError as exc:
            raise KeyError(f"unknown sample int {sample_int!r}") from exc

    def observation_ids(self) -> list[str]:
        """All wire observation ids, in dataset order."""
        return [identity.observation_id for identity in self.identities]


def mint_identity(dataset: SpectroDataset) -> IdentityMap:
    """Mint the stable identity map for ``dataset``.

    The per-dataset ``content_hash()`` (64-hex, stable across calls, invalidated on
    feature mutation) anchors every wire id, so ids are reproducible from content and
    never from row position.
    """
    fingerprint = dataset.content_hash()
    samples = dataset.index_column("sample", {})
    origins = dataset.index_column("origin", {})
    identities: list[SampleIdentity] = []
    for sample_int, origin_int in zip(samples, origins, strict=True):
        observation_id = validate_data_id(f"{fingerprint}.s{sample_int}")
        sample_id = validate_data_id(f"{fingerprint}.s{origin_int}")
        identities.append(SampleIdentity(sample_int, origin_int, observation_id, sample_id, sample_int != origin_int))
    return IdentityMap(
        fingerprint=fingerprint,
        identities=tuple(identities),
        _to_int={identity.observation_id: identity.sample_int for identity in identities},
        _to_wire={identity.sample_int: identity.observation_id for identity in identities},
    )
