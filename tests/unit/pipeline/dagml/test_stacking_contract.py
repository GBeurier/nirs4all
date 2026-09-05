"""Native nested-stacking admission tests."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml.envelope import build_fold_set
from nirs4all.pipeline.dagml.identity import IdentityMap, SampleIdentity


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
    folds = build_fold_set(
        _identity(),
        [([2, 3], [0, 1]), ([0, 1], [2, 3])],
    )
    assert "partition_mode" not in folds  # Frozen KFold wire default stays byte-identical.


def test_nested_stacking_keeps_resampled_outer_evidence_explicit() -> None:
    # Real nested execution/REFIT/poison gates live in test_resampled_dag_stacking.
    # Admission must preserve repeated validation, not relabel it a partition.
    folds = build_fold_set(_identity(), [([2, 3], [0, 1]), ([0, 3], [1, 2])])
    assert folds["partition_mode"] == "resampled"
    assert folds["folds"][0]["validation_sample_ids"] == ["fixture.s0", "fixture.s1"]
    assert folds["folds"][1]["validation_sample_ids"] == ["fixture.s1", "fixture.s2"]


def test_nested_stacking_fold_identity_cannot_invent_samples() -> None:
    with pytest.raises(KeyError):
        build_fold_set(_identity(), [([2, 3], [0, 99])])
