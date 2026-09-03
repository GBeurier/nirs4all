"""Native nested-stacking admission tests."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.identity import IdentityMap, SampleIdentity
from nirs4all.pipeline.dagml.run_paths import _require_partitioned_outer_stacking


def _identity(sample_count: int = 4) -> IdentityMap:
    identities = tuple(
        SampleIdentity(
            sample_int=index,
            origin_int=index,
            observation_id=f"fixture.s{index}",
            sample_id=f"fixture.s{index}",
            augmented=False,
        )
        for index in range(sample_count)
    )
    return IdentityMap(
        fingerprint="fixture",
        identities=identities,
        _to_int={identity.observation_id: identity.sample_int for identity in identities},
        _to_wire={identity.sample_int: identity.observation_id for identity in identities},
    )


def test_nested_stacking_accepts_partitioned_outer_folds() -> None:
    _require_partitioned_outer_stacking(
        _identity(),
        [([2, 3], [0, 1]), ([0, 1], [2, 3])],
    )


def test_nested_stacking_refuses_resampled_outer_folds_before_execution() -> None:
    with pytest.raises(DagMlUnsupported, match="requires an outer CV partition"):
        _require_partitioned_outer_stacking(
            _identity(),
            [([2, 3], [0, 1]), ([0, 3], [1, 2])],
        )
