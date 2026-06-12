"""Unit tests for the N3 ``RawMultiSourceDataset`` staging object.

Covers source-specific staging with heterogeneous cardinalities, the
deterministic source/row mapping, the order-independent content fingerprint, the
explicit-materialisation requirement (no implicit ragged -> rectangular
coercion) and the minimal-safe ``per_source_aggregate`` representation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.raw_multisource import (
    CARTESIAN_REPRESENTATIONS,
    AlignedMaterialization,
    CombinationPlan,
    RawMultiSourceDataset,
    RepresentationPlan,
    replay_materialization,
)
from nirs4all.data.relations import (
    NormalizedObservationTable,
    RelationValidationError,
    RepetitionSpec,
    SourceObservations,
    build_relation_table,
)


def _spec(**kwargs) -> RepetitionSpec:
    base = {"sample_id": "sid", "link_by": "sid"}
    base.update(kwargs)
    return RepetitionSpec(**base)


def _heterogeneous_dataset():
    """MIR=2 (2 feats), RAMAN=3 (1 feat) for two physical samples S1, S2."""
    X = {
        "MIR": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        "RAMAN": np.array([[10.0], [20.0], [11.0], [21.0], [12.0], [22.0]]),
    }
    keys = {
        "MIR": ["S1", "S1", "S2", "S2"],
        "RAMAN": ["S1", "S2", "S1", "S2", "S1", "S2"],
    }
    targets = {"MIR": [10.0, 10.0, 20.0, 20.0]}
    return RawMultiSourceDataset.from_sources(_spec(), X, keys, targets_by_source=targets)


# ---------------------------------------------------------------------------
# Construction / cardinalities
# ---------------------------------------------------------------------------


def test_staging_cardinalities_and_shapes():
    ds = _heterogeneous_dataset()
    assert ds.n_samples == 2
    assert ds.n_observations == 10  # 2*(2 + 3)
    assert ds.source_ids == ["MIR", "RAMAN"]
    assert ds.physical_sample_ids == ["S1", "S2"]
    assert ds.feature_dims() == {"MIR": 2, "RAMAN": 1}
    cards = ds.cardinalities()
    assert cards[("S1", "MIR")] == 2
    assert cards[("S1", "RAMAN")] == 3
    assert cards[("S2", "MIR")] == 2
    assert cards[("S2", "RAMAN")] == 3
    assert ds.targets_by_sample() == {"S1": 10.0, "S2": 20.0}


def test_from_sources_rejects_key_length_mismatch():
    X = {"A": np.zeros((3, 2))}
    keys = {"A": ["S1", "S2"]}  # 2 keys for 3 rows
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset.from_sources(_spec(), X, keys)
    assert exc.value.code == "REL-E021"


def test_from_sources_rejects_source_set_mismatch():
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset.from_sources(
            _spec(), {"A": np.zeros((2, 2))}, {"A": ["S1", "S2"], "B": ["S1", "S2"]}
        )
    assert exc.value.code == "REL-E021"


@pytest.mark.parametrize(
    ("mapping_name", "kwargs"),
    [
        ("rep_by_source", {"rep_by_source": {"missing": [0, 1]}}),
        ("targets_by_source", {"targets_by_source": {"missing": [1.0, 2.0]}}),
        ("metadata_by_source", {"metadata_by_source": {"missing": {"site": ["a", "b"]}}}),
    ],
)
def test_from_sources_rejects_unknown_auxiliary_mapping_source(mapping_name, kwargs):
    X = {"A": np.zeros((2, 2))}
    keys = {"A": ["S1", "S2"]}
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset.from_sources(_spec(), X, keys, **kwargs)
    assert exc.value.code == "REL-E021"
    assert mapping_name in str(exc.value)


def test_header_width_mismatch_rejected():
    X = {"A": np.zeros((2, 3))}
    keys = {"A": ["S1", "S2"]}
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset.from_sources(_spec(), X, keys, headers_by_source={"A": ["only", "two"]})
    assert exc.value.code == "REL-E021"


def test_from_sources_rejects_non_2d_feature_block():
    X = {"A": np.zeros(2)}
    keys = {"A": ["S1", "S2"]}
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset.from_sources(_spec(), X, keys)
    assert exc.value.code == "REL-E021"
    assert "2D array" in str(exc.value)


def test_constructor_rejects_unreferenced_source_block():
    table = build_relation_table(_spec(), [SourceObservations("A", ["S1"])])
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset(table, {"A": np.zeros((1, 1)), "B": np.zeros((1, 1))})
    assert exc.value.code == "REL-E021"
    assert "not referenced" in str(exc.value)


def test_constructor_rejects_unreferenced_source_rows():
    table = build_relation_table(_spec(), [SourceObservations("A", ["S1", "S2"])])
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset(table, {"A": np.zeros((3, 1))})
    assert exc.value.code == "REL-E021"
    assert "unreferenced" in str(exc.value)


def test_constructor_rejects_duplicate_source_row_mapping():
    table = build_relation_table(_spec(), [SourceObservations("A", ["S1", "S2"])])
    records = list(table.records)
    records[1].source_row = records[0].source_row
    with pytest.raises(RelationValidationError) as exc:
        RawMultiSourceDataset(NormalizedObservationTable(records), {"A": np.zeros((2, 1))})
    assert exc.value.code == "REL-E021"
    assert "multiple observations" in str(exc.value)


# ---------------------------------------------------------------------------
# Deterministic source/row mapping
# ---------------------------------------------------------------------------


def test_aligned_row_order_is_canonical_and_independent_of_input_order():
    ds = _heterogeneous_dataset()
    order = ds.aligned_row_order()
    # Sorted by (sample, source, rep): all S1 before S2, MIR before RAMAN.
    samples_sources = [(s, src) for (s, src, _row) in order]
    assert samples_sources == sorted(samples_sources)
    assert samples_sources[0] == ("S1", "MIR")
    # The same logical data fed with shuffled rows yields the same canonical order.
    X = {
        "MIR": np.array([[3.0, 3.0], [1.0, 1.0], [4.0, 4.0], [2.0, 2.0]]),
        "RAMAN": np.array([[12.0], [22.0], [10.0], [20.0], [11.0], [21.0]]),
    }
    keys = {"MIR": ["S2", "S1", "S2", "S1"], "RAMAN": ["S1", "S2", "S1", "S2", "S1", "S2"]}
    targets = {"MIR": [20.0, 10.0, 20.0, 10.0]}
    ds_shuf = RawMultiSourceDataset.from_sources(_spec(), X, keys, targets_by_source=targets)
    order_pairs = [(s, src) for (s, src, _row) in ds_shuf.aligned_row_order()]
    assert order_pairs == samples_sources


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic():
    assert _heterogeneous_dataset().fingerprint() == _heterogeneous_dataset().fingerprint()


def test_fingerprint_invariant_under_source_and_row_shuffle():
    ds = _heterogeneous_dataset()
    # Reorder sources and rows; bind features to the same identities.
    X = {
        "RAMAN": np.array([[12.0], [22.0], [10.0], [20.0], [11.0], [21.0]]),
        "MIR": np.array([[3.0, 3.0], [1.0, 1.0], [4.0, 4.0], [2.0, 2.0]]),
    }
    keys = {"RAMAN": ["S1", "S2", "S1", "S2", "S1", "S2"], "MIR": ["S2", "S1", "S2", "S1"]}
    targets = {"MIR": [20.0, 10.0, 20.0, 10.0]}
    ds_shuf = RawMultiSourceDataset.from_sources(_spec(), X, keys, targets_by_source=targets)
    assert ds.fingerprint() == ds_shuf.fingerprint()


def test_fingerprint_changes_with_feature_content():
    ds = _heterogeneous_dataset()
    X = {
        "MIR": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]) + 0.5,
        "RAMAN": np.array([[10.0], [20.0], [11.0], [21.0], [12.0], [22.0]]),
    }
    keys = {"MIR": ["S1", "S1", "S2", "S2"], "RAMAN": ["S1", "S2", "S1", "S2", "S1", "S2"]}
    ds2 = RawMultiSourceDataset.from_sources(_spec(), X, keys, targets_by_source={"MIR": [10.0, 10.0, 20.0, 20.0]})
    assert ds.fingerprint() != ds2.fingerprint()


def test_ordered_fingerprint_changes_with_row_order():
    spec = _spec(rep_order="ordered")
    ds = RawMultiSourceDataset.from_sources(spec, {"A": np.array([[1.0], [2.0]])}, {"A": ["S1", "S1"]})
    ds_shuf = RawMultiSourceDataset.from_sources(spec, {"A": np.array([[2.0], [1.0]])}, {"A": ["S1", "S1"]})
    assert ds.fingerprint() != ds_shuf.fingerprint()


# ---------------------------------------------------------------------------
# Explicit materialisation requirement + per_source_aggregate
# ---------------------------------------------------------------------------


def test_materialize_requires_explicit_representation():
    ds = _heterogeneous_dataset()
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize()
    assert exc.value.code == "REL-E019"
    assert "explicit representation" in str(exc.value)


def test_cartesian_representations_are_executable():
    ds = _heterogeneous_dataset()
    for representation in sorted(CARTESIAN_REPRESENTATIONS):
        plan = RepresentationPlan(representation, max_combos_per_sample=1) if representation == "cartesian_mc" else RepresentationPlan(representation)
        mat = ds.materialize(plan)
        assert mat.representation == representation
        assert mat.unit_ids is not None
        assert mat.lineage is not None


def test_sample_aggregate_is_executable_with_explicit_plan():
    ds = _heterogeneous_dataset()
    mat = ds.materialize(RepresentationPlan("sample_aggregate"))
    assert mat.representation == "sample_aggregate"
    assert mat.representation_plan is not None
    assert mat.representation_plan.stage == "sample_aggregate"
    np.testing.assert_allclose(mat.X, ds.materialize("per_source_aggregate").X)


def test_materialize_refuses_unknown_representation():
    ds = _heterogeneous_dataset()
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize("not_a_real_representation")
    assert exc.value.code == "REL-E019"


def test_per_source_aggregate_is_rectangular_and_aligned():
    ds = _heterogeneous_dataset()
    mat = ds.materialize("per_source_aggregate")
    assert isinstance(mat, AlignedMaterialization)
    assert mat.representation == "per_source_aggregate"
    assert mat.sample_ids == ["S1", "S2"]
    # 2 MIR feats + 1 RAMAN feat = 3 columns; one row per physical sample.
    assert mat.X.shape == (2, 3)
    # S1: MIR mean of [1,1],[2,2] = [1.5,1.5]; RAMAN mean of [10,11,12] = 11.
    np.testing.assert_allclose(mat.X[0], [1.5, 1.5, 11.0])
    # S2: MIR mean of [3,3],[4,4] = [3.5,3.5]; RAMAN mean of [20,21,22] = 21.
    np.testing.assert_allclose(mat.X[1], [3.5, 3.5, 21.0])
    assert mat.headers == ["MIR:MIR_f0", "MIR:MIR_f1", "RAMAN:RAMAN_f0"]
    assert mat.targets == [10.0, 20.0]
    assert mat.representation_plan is not None
    assert mat.to_manifest()["representation_plan"]["representation"] == "per_source_aggregate"


def test_per_source_aggregate_is_invariant_to_input_shuffle():
    ds = _heterogeneous_dataset()
    ref = ds.materialize("per_source_aggregate")
    X = {
        "MIR": np.array([[4.0, 4.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]),
        "RAMAN": np.array([[22.0], [12.0], [21.0], [11.0], [20.0], [10.0]]),
    }
    keys = {"MIR": ["S2", "S1", "S2", "S1"], "RAMAN": ["S2", "S1", "S2", "S1", "S2", "S1"]}
    ds_shuf = RawMultiSourceDataset.from_sources(_spec(), X, keys, targets_by_source={"MIR": [20.0, 10.0, 20.0, 10.0]})
    mat = ds_shuf.materialize("per_source_aggregate")
    assert mat.sample_ids == ref.sample_ids
    np.testing.assert_allclose(mat.X, ref.X)


def test_per_source_aggregate_missing_source_refused():
    # DROP_INCOMPLETE lets a partial-coverage table build, but materialisation
    # to an aligned matrix still needs every source present per sample.
    from nirs4all.data.relations import MissingSourcePolicy

    X = {"A": np.array([[1.0], [2.0]]), "B": np.array([[3.0]])}
    keys = {"A": ["S1", "S2"], "B": ["S1"]}
    ds = RawMultiSourceDataset.from_sources(
        _spec(missing_source_policy=MissingSourcePolicy.DROP_INCOMPLETE), X, keys
    )
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize("per_source_aggregate")
    assert exc.value.code == "REL-E007"


def test_representation_plan_round_trips_and_fingerprints():
    plan = RepresentationPlan(
        "stack_padded_masked",
        missing_source_policy="nan",
        missing_repetition_policy="pad_masked",
        max_total_rows=10,
        memory_budget="1MB",
        random_state=123,
    )
    restored = RepresentationPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.fingerprint() == plan.fingerprint()
    assert restored.unit_level == "stack"
    assert restored.stage == "stack"


def test_cartesian_representation_plan_carries_combination_plan():
    plan = RepresentationPlan(
        "cartesian_mc",
        max_combos_per_sample=2,
        max_total_combos=4,
        max_total_rows=4,
        memory_budget="1MB",
        random_state=7,
    )
    assert plan.combo_selection == "random_seeded"
    assert plan.combination_plan == CombinationPlan(
        combo_selection="random_seeded",
        max_combos_per_sample=2,
        max_total_combos=4,
        max_total_rows=4,
        memory_budget="1MB",
        random_state=7,
    )
    restored = RepresentationPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.fingerprint() == plan.fingerprint()


def test_materialization_manifest_replays_exactly():
    ds = _heterogeneous_dataset()
    plan = RepresentationPlan("per_source_aggregate", max_total_rows=2)
    mat = ds.materialize(plan)
    replayed = replay_materialization(ds, mat.to_manifest(), validate_fingerprint=True)
    assert replayed.fingerprint == mat.fingerprint
    np.testing.assert_allclose(replayed.X, mat.X)


def test_per_source_observation_is_sparse_rectangular_with_mask():
    mat = _heterogeneous_dataset().materialize("per_source_observation")
    assert mat.representation == "per_source_observation"
    assert mat.X.shape == (10, 3)
    assert mat.feature_mask is not None
    assert mat.sample_ids[:2] == ["S1", "S1"]
    assert mat.source_ids[:2] == ["MIR", "MIR"]
    np.testing.assert_allclose(mat.X[0], [1.0, 1.0, np.nan], equal_nan=True)
    assert mat.feature_mask[0].tolist() == [True, True, False]
    assert mat.unit_ids[0] == "S1|MIR|rep0"


def test_stack_fixed_flattens_fixed_repetition_slots():
    mat = _heterogeneous_dataset().materialize("stack_fixed")
    assert mat.X.shape == (2, 7)
    assert mat.headers == [
        "MIR:rep0:MIR_f0",
        "MIR:rep0:MIR_f1",
        "MIR:rep1:MIR_f0",
        "MIR:rep1:MIR_f1",
        "RAMAN:rep0:RAMAN_f0",
        "RAMAN:rep1:RAMAN_f0",
        "RAMAN:rep2:RAMAN_f0",
    ]
    np.testing.assert_allclose(mat.X[0], [1.0, 1.0, 2.0, 2.0, 10.0, 11.0, 12.0])


def test_stack_fixed_refuses_variable_or_missing_cardinalities():
    X = {"A": np.array([[1.0], [2.0], [3.0]])}
    keys = {"A": ["S1", "S1", "S2"]}
    ds = RawMultiSourceDataset.from_sources(_spec(), X, keys)
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize("stack_fixed")
    assert exc.value.code == "REL-E019"


def test_stack_padded_masked_accepts_missing_source_with_mask():
    from nirs4all.data.relations import MissingSourcePolicy

    X = {"A": np.array([[1.0], [2.0]]), "B": np.array([[3.0]])}
    keys = {"A": ["S1", "S2"], "B": ["S1"]}
    ds = RawMultiSourceDataset.from_sources(
        _spec(missing_source_policy=MissingSourcePolicy.DROP_INCOMPLETE), X, keys
    )
    mat = ds.materialize(RepresentationPlan("stack_padded_masked", missing_source_policy="nan"))
    assert mat.X.shape == (2, 2)
    assert mat.feature_mask is not None
    np.testing.assert_allclose(mat.X[1], [2.0, np.nan], equal_nan=True)
    assert mat.feature_mask[1].tolist() == [True, False]


def test_cartesian_full_materializes_combo_rows_with_lineage():
    mat = _heterogeneous_dataset().materialize("cartesian_full")
    assert mat.representation == "cartesian_full"
    assert mat.X.shape == (12, 3)
    assert mat.sample_ids[:6] == ["S1"] * 6
    assert mat.sample_ids[6:] == ["S2"] * 6
    assert mat.headers == ["MIR:MIR_f0", "MIR:MIR_f1", "RAMAN:RAMAN_f0"]
    assert mat.unit_ids is not None
    assert mat.unit_ids[0] == "S1::MIR0xRAMAN0"
    np.testing.assert_allclose(mat.X[0], [1.0, 1.0, 10.0])
    assert mat.lineage is not None
    assert mat.lineage[0]["unit_level"] == "combo"
    assert mat.lineage[0]["origin_sample_id"] == "S1"
    assert mat.lineage[0]["component_observation_ids"] == ["MIR:S1:0", "RAMAN:S1:0"]
    assert mat.lineage[0]["combination_plan"]["combo_selection"] == "deterministic_all"


def test_cartesian_mc_is_seeded_and_replayable():
    ds = _heterogeneous_dataset()
    plan = RepresentationPlan("cartesian_mc", max_combos_per_sample=2, random_state=42)
    mat = ds.materialize(plan)
    replayed = replay_materialization(ds, mat.to_manifest(), validate_fingerprint=True)

    assert mat.X.shape == (4, 3)
    assert mat.unit_ids == replayed.unit_ids
    np.testing.assert_allclose(mat.X, replayed.X)


def test_cartesian_full_refuses_per_sample_cap_overflow():
    ds = _heterogeneous_dataset()
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize(RepresentationPlan("cartesian_full", max_combos_per_sample=5))
    assert exc.value.code == "REL-E019"
    assert "max_combos_per_sample" in str(exc.value)


def test_cartesian_full_refuses_global_combo_cap_overflow():
    ds = _heterogeneous_dataset()
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize(RepresentationPlan("cartesian_full", max_total_combos=10))
    assert exc.value.code == "REL-E019"
    assert "max_total_combos" in str(exc.value)


def test_cartesian_augmentation_preserves_origin_lineage():
    mat = _heterogeneous_dataset().materialize(RepresentationPlan("cartesian_augmentation"))

    assert mat.lineage is not None
    assert mat.representation_plan is not None
    assert mat.representation_plan.combination_plan is not None
    assert mat.representation_plan.combination_plan.train_only is True
    assert mat.lineage[0]["augmentation"] == "cartesian_augmentation"
    assert mat.lineage[0]["origin_sample_id"] == mat.lineage[0]["physical_sample_id"]


def test_representation_caps_are_enforced_before_allocation():
    ds = _heterogeneous_dataset()
    with pytest.raises(RelationValidationError) as exc:
        ds.materialize(RepresentationPlan("per_source_observation", max_total_rows=2))
    assert exc.value.code == "REL-E019"


# ---------------------------------------------------------------------------
# Manifest serialisability
# ---------------------------------------------------------------------------


def test_manifest_is_json_serialisable_and_complete():
    import json

    ds = _heterogeneous_dataset()
    manifest = ds.to_manifest()
    # Round-trips through JSON without error.
    restored = json.loads(json.dumps(manifest))
    assert restored["source_ids"] == ["MIR", "RAMAN"]
    assert restored["feature_dims"] == {"MIR": 2, "RAMAN": 1}
    assert restored["fingerprint"] == ds.fingerprint()
    assert restored["cardinalities"]["S1|RAMAN"] == 3
    assert restored["targets_by_sample"] == {"S1": 10.0, "S2": 20.0}


def test_manifest_serialises_numpy_scalar_targets():
    import json

    ds = RawMultiSourceDataset.from_sources(
        _spec(),
        {"A": np.ones((2, 1))},
        {"A": ["S1", "S2"]},
        targets_by_source={"A": list(np.array([1, 2], dtype=np.int64))},
    )
    restored = json.loads(json.dumps(ds.to_manifest()))
    assert restored["targets_by_sample"] == {"S1": 1, "S2": 2}
