"""Tests for the loader-level relation materialisation seam (roadmap N2).

``materialize_relation_table`` is the additive capability that lets the loader
join heterogeneous per-source frames *by key* into a validated relation table,
without changing the legacy positional return contract.
"""

from __future__ import annotations

import pytest

from nirs4all.data.loaders.loader import handle_data, materialize_relation_table
from nirs4all.data.relations import RelationValidationError


def _config():
    return {
        "experimental_relation_pipeline": True,
        "repetition_spec": {
            "sample_id": "sid",
            "link_by": "sid",
            "sources": {"MIR": {"expected": 2}, "RAMAN": {"expected": 3}},
        },
    }


def test_materialize_joins_shuffled_sources_by_key():
    frames = {
        "MIR": {"sid": ["S2", "S1", "S1", "S2"], "y": [20.0, 10.0, 10.0, 20.0]},
        "RAMAN": {"sid": ["S1", "S2", "S1", "S2", "S1", "S2"]},
    }
    table = materialize_relation_table(_config(), frames, target_col="y")
    assert table is not None
    assert table.physical_sample_ids == ["S1", "S2"]
    assert table.source_ids == ["MIR", "RAMAN"]
    cards = table.cardinalities()
    assert cards[("S1", "MIR")] == 2
    assert cards[("S1", "RAMAN")] == 3
    assert cards[("S2", "MIR")] == 2
    assert cards[("S2", "RAMAN")] == 3


def test_materialize_shares_target_via_link_by():
    # The target lives only on MIR; the join shares it to the physical sample.
    frames = {
        "MIR": {"sid": ["S1", "S1", "S2", "S2"], "y": [10.0, 10.0, 20.0, 20.0]},
        "RAMAN": {"sid": ["S1", "S2", "S1", "S2", "S1", "S2"]},
    }
    table = materialize_relation_table(_config(), frames, target_col="y")
    assert table is not None
    assert table.targets_by_sample() == {"S1": 10.0, "S2": 20.0}


def test_materialize_returns_none_without_relational_spec():
    frames = {"MIR": {"sid": ["S1"]}, "RAMAN": {"sid": ["S1"]}}
    assert materialize_relation_table({"train_x": "a.csv"}, frames) is None


def test_materialize_missing_join_key_column_raises_relation_error():
    frames = {
        "MIR": {"sid": ["S1", "S1"]},
        "RAMAN": {"other_id": ["S1", "S1", "S1"]},
    }
    with pytest.raises(RelationValidationError) as exc:
        materialize_relation_table(_config(), frames)
    assert exc.value.code == "REL-E005"
    assert "missing required column" in str(exc.value)


def test_materialize_missing_source_strict_refused():
    frames = {
        "MIR": {"sid": ["S1", "S2"]},
        "RAMAN": {"sid": ["S1"]},  # S2 missing RAMAN, strict default
    }
    with pytest.raises(RelationValidationError) as exc:
        materialize_relation_table(_config(), frames)
    assert exc.value.code == "REL-E007"


def test_materialize_missing_declared_source_frame_refused():
    frames = {"MIR": {"sid": ["S1", "S1"]}}
    with pytest.raises(RelationValidationError) as exc:
        materialize_relation_table(_config(), frames)
    assert exc.value.code == "REL-E007"


def test_handle_data_refuses_experimental_relation_sources_config():
    # The legacy loader cannot materialise the experimental relation pipeline: it
    # must fail loudly instead of returning an empty/positional dataset.
    config = {
        **_config(),
        "train_x": ["MIR_train.csv", "RAMAN_train.csv"],
        "_sources": [{"name": "MIR", "link_by": "sid"}, {"name": "RAMAN", "link_by": "sid"}],
    }
    with pytest.raises(RelationValidationError) as exc:
        handle_data(config, "train")
    assert exc.value.code == "REL-E023"
    assert "rep_fusion" in str(exc.value)


def test_handle_data_allows_plain_legacy_config():
    # A plain legacy config without relation fields is not affected by the guard.
    import numpy as np

    config = {"train_x": np.zeros((3, 2)), "train_y": np.zeros((3, 1))}
    x, _y, _m, _xh, _mh, _unit, _sig = handle_data(config, "train")
    assert x.shape == (3, 2)
