"""Unit tests for the relational identity model (roadmap N1).

Covers the :class:`NormalizedObservationTable` and the ``link_by`` join builder:
nominal heterogeneous cardinalities (A=2/B=3/C=2), shuffled source order,
duplicate observation keys, contradictory targets / metadata, declared
cardinality mismatch, default influence weight, deterministic fingerprint and
combo lineage.
"""

from __future__ import annotations

import pytest

from nirs4all.data.relations import (
    MissingSourcePolicy,
    NormalizedObservationTable,
    Partition,
    RelationValidationError,
    RepetitionSpec,
    SampleRelationPlan,
    SourceObservations,
    SourceRepetitionSpec,
    UnitLevel,
    build_relation_table,
)


def _nominal_sources(*, targets=True):
    """A=2, B=3, C=2 repetitions for two physical samples S1, S2."""
    t = {"S1": 10.0, "S2": 20.0}
    a = SourceObservations(
        source_id="A",
        sample_ids=["S1", "S1", "S2", "S2"],
        targets=[t["S1"], t["S1"], t["S2"], t["S2"]] if targets else None,
    )
    b = SourceObservations(
        source_id="B",
        sample_ids=["S1", "S1", "S1", "S2", "S2", "S2"],
        targets=[t["S1"], t["S1"], t["S1"], t["S2"], t["S2"], t["S2"]] if targets else None,
    )
    c = SourceObservations(
        source_id="C",
        sample_ids=["S1", "S1", "S2", "S2"],
        targets=[t["S1"], t["S1"], t["S2"], t["S2"]] if targets else None,
    )
    return [a, b, c]


def _spec(**kwargs) -> RepetitionSpec:
    base = {"sample_id": "sample_id", "link_by": "sample_id"}
    base.update(kwargs)
    return RepetitionSpec(**base)


# ---------------------------------------------------------------------------
# Nominal heterogeneous table
# ---------------------------------------------------------------------------


def test_nominal_a2_b3_c2_table():
    table = build_relation_table(_spec(), _nominal_sources())
    # 2 samples * (2 + 3 + 2) observations = 14 rows
    assert len(table) == 14
    assert table.physical_sample_ids == ["S1", "S2"]
    assert table.source_ids == ["A", "B", "C"]

    cards = table.cardinalities()
    assert cards[("S1", "A")] == 2
    assert cards[("S1", "B")] == 3
    assert cards[("S1", "C")] == 2
    assert cards[("S2", "A")] == 2
    assert cards[("S2", "B")] == 3
    assert cards[("S2", "C")] == 2

    # All raw observations.
    assert all(r.unit_level is UnitLevel.OBSERVATION for r in table.records)
    # Targets are sample-level.
    assert table.targets_by_sample() == {"S1": 10.0, "S2": 20.0}


def test_internal_ids_are_deterministic_and_stable():
    table = build_relation_table(_spec(), _nominal_sources())
    mapping = table.internal_id_map()
    # Sorted by physical_sample_id -> S1 = 0, S2 = 1.
    assert mapping == {"S1": 0, "S2": 1}


def test_observation_ids_and_rep_ids_assigned_positionally():
    table = build_relation_table(_spec(), _nominal_sources())
    b_reps_s1 = sorted(r.rep_id for r in table.records if r.source_id == "B" and r.physical_sample_id == "S1")
    assert b_reps_s1 == [0, 1, 2]
    ids = {r.observation_id for r in table.records if r.source_id == "B" and r.physical_sample_id == "S1"}
    assert ids == {"B:S1:0", "B:S1:1", "B:S1:2"}


def test_sample_influence_weight_default_is_one():
    table = build_relation_table(_spec(), _nominal_sources())
    assert all(r.sample_influence_weight == 1.0 for r in table.records)


def test_equal_sample_influence_derivation():
    table = build_relation_table(_spec(), _nominal_sources())
    equalised = table.with_equal_sample_influence()
    # S1 has 7 observations -> each weight 1/7; total influence per sample == 1.
    s1_weights = [r.sample_influence_weight for r in equalised.records if r.physical_sample_id == "S1"]
    assert len(s1_weights) == 7
    assert all(abs(w - 1.0 / 7.0) < 1e-12 for w in s1_weights)
    assert abs(sum(s1_weights) - 1.0) < 1e-12
    # Original table is untouched.
    assert all(r.sample_influence_weight == 1.0 for r in table.records)


# ---------------------------------------------------------------------------
# Shuffled source order / rows
# ---------------------------------------------------------------------------


def test_shuffled_source_order_same_fingerprint():
    a, b, c = _nominal_sources()
    fp1 = build_relation_table(_spec(), [a, b, c]).fingerprint()
    fp2 = build_relation_table(_spec(), [c, a, b]).fingerprint()
    assert fp1 == fp2


def test_shuffled_rows_same_fingerprint():
    table_ref = build_relation_table(_spec(), _nominal_sources())
    # Permute B's rows within the sources (order must not matter for exchangeable).
    a, b, c = _nominal_sources()
    b_shuffled = SourceObservations(
        source_id="B",
        sample_ids=["S2", "S1", "S2", "S1", "S2", "S1"],
        targets=[20.0, 10.0, 20.0, 10.0, 20.0, 10.0],
    )
    table_shuf = build_relation_table(_spec(), [a, b_shuffled, c])
    assert table_ref.fingerprint() == table_shuf.fingerprint()


def test_fingerprint_changes_with_data():
    table = build_relation_table(_spec(), _nominal_sources())
    a, b, c = _nominal_sources(targets=False)
    # Add an extra repetition to source A of S1 -> different structure.
    a2 = SourceObservations(source_id="A", sample_ids=["S1", "S1", "S1", "S2", "S2"])
    table2 = build_relation_table(
        _spec(missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED),
        [a2, b, c],
    )
    assert table.fingerprint() != table2.fingerprint()


# ---------------------------------------------------------------------------
# Validation: duplicates, contradictions, cardinalities
# ---------------------------------------------------------------------------


def test_duplicate_observation_key_raises():
    a = SourceObservations(
        source_id="A",
        sample_ids=["S1", "S1"],
        rep_ids=[0, 0],  # duplicate (S1, A, 0)
    )
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(
            _spec(sources={"A": SourceRepetitionSpec()}, missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED),
            [a],
        )
    assert exc.value.code == "REL-E001"


def test_contradictory_target_raises():
    a = SourceObservations(source_id="A", sample_ids=["S1", "S1"], targets=[10.0, 10.0])
    b = SourceObservations(source_id="B", sample_ids=["S1"], targets=[11.0])  # different target
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [a, b])
    assert exc.value.code == "REL-E002"


def test_contradictory_metadata_raises():
    a = SourceObservations(
        source_id="A",
        sample_ids=["S1", "S1"],
        metadata={"site": ["north", "north"]},
    )
    b = SourceObservations(
        source_id="B",
        sample_ids=["S1"],
        metadata={"site": ["south"]},  # contradictory sample-level metadata
    )
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [a, b])
    assert exc.value.code == "REL-E003"


def test_declared_cardinality_mismatch_raises():
    # A declares expected=2 but S2 has 3 observations -> strict mismatch.
    a = SourceObservations(source_id="A", sample_ids=["S1", "S1", "S2", "S2", "S2"])
    spec = _spec(
        sources={"A": SourceRepetitionSpec(expected=2)},
        strict_cardinality=True,
        missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED,
    )
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(spec, [a])
    assert exc.value.code == "REL-E004"


def test_strict_cardinality_counts_zero_observation_missing_sources():
    spec = RepetitionSpec.from_config(
        {
            "sample_id": "sample_id",
            "strict_cardinality": True,
            "missing_source_policy": "drop_incomplete",
            "sources": {"A": 2, "B": 2},
        }
    )
    a = SourceObservations(source_id="A", sample_ids=["S1", "S1", "S2", "S2"])
    b = SourceObservations(source_id="B", sample_ids=["S1", "S1"])  # S2 has zero B observations
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(spec, [a, b])
    assert exc.value.code == "REL-E004"
    assert "S2" in str(exc.value)


def test_declared_cardinality_non_strict_tolerated():
    a = SourceObservations(source_id="A", sample_ids=["S1", "S1", "S2", "S2", "S2"])
    spec = _spec(
        sources={"A": SourceRepetitionSpec(expected=2)},
        strict_cardinality=False,
        missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED,
    )
    table = build_relation_table(spec, [a])  # no raise
    assert table.cardinalities()[("S2", "A")] == 3


def test_missing_source_strict_raises():
    a = SourceObservations(source_id="A", sample_ids=["S1", "S2"])
    b = SourceObservations(source_id="B", sample_ids=["S1"])  # S2 missing source B
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [a, b])
    assert exc.value.code == "REL-E007"


def test_positional_join_refused_without_key():
    a = SourceObservations(source_id="A", sample_ids=None, rep_ids=[0, 1])
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [a])
    assert exc.value.code == "REL-E005"


@pytest.mark.parametrize("bad_key", [None, float("nan"), ""])
def test_missing_or_null_sample_key_rejected(bad_key):
    a = SourceObservations(source_id="A", sample_ids=["S1", bad_key])
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [a])
    assert exc.value.code == "REL-E005"
    assert "Missing" in str(exc.value)


def test_non_integer_rep_id_raises():
    a = SourceObservations(source_id="A", sample_ids=["S1", "S1"], rep_ids=["x", "y"])
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(
            _spec(missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED),
            [a],
        )
    assert exc.value.code == "REL-E006"


# ---------------------------------------------------------------------------
# Combo lineage
# ---------------------------------------------------------------------------


def test_combo_lineage_full_cartesian():
    table = build_relation_table(_spec(), _nominal_sources())
    combos = table.enumerate_combos("S1")
    # 2 * 3 * 2 = 12 combos, each with one component per source.
    assert len(combos) == 12
    for combo in combos:
        assert combo.physical_sample_id == "S1"
        assert combo.origin_sample_id == "S1"
        assert len(combo.component_observation_ids) == 3
        assert set(combo.rep_ids_by_source) == {"A", "B", "C"}
    # Combo ids are unique and deterministic.
    ids = [c.derived_unit_id for c in combos]
    assert len(set(ids)) == 12
    assert ids == sorted(ids)


def test_combo_lineage_requires_all_sources():
    a = SourceObservations(source_id="A", sample_ids=["S1", "S2"])
    b = SourceObservations(source_id="B", sample_ids=["S1"])
    table = build_relation_table(_spec(missing_source_policy=MissingSourcePolicy.DROP_INCOMPLETE), [a, b])

    assert len(table.enumerate_combos("S1")) == 1
    assert table.enumerate_combos("S2") == []


# ---------------------------------------------------------------------------
# SampleRelationPlan
# ---------------------------------------------------------------------------


def test_sample_relation_plan_fingerprint_and_mapping():
    plan = SampleRelationPlan.from_sources(_spec(), _nominal_sources(), partition=Partition.TRAIN)
    assert plan.physical_to_internal == {"S1": 0, "S2": 1}
    # Fingerprint is deterministic and combines spec + table.
    plan2 = SampleRelationPlan.from_sources(_spec(), _nominal_sources(), partition=Partition.TRAIN)
    assert plan.fingerprint() == plan2.fingerprint()


def test_to_columns_has_canonical_schema():
    from nirs4all.data.relations import RELATION_TABLE_COLUMNS

    table = build_relation_table(_spec(), _nominal_sources())
    cols = table.to_columns()
    assert set(cols) == set(RELATION_TABLE_COLUMNS)
    assert all(len(v) == len(table) for v in cols.values())
