"""Unit tests for RepetitionSpec parsing and the repetition exclusivity matrix.

Roadmap N0/N2: ``RepetitionSpec`` config parsing/validation, the exclusivity
matrix across ``repetition`` / ``rep_to_sources`` / ``rep_to_pp`` / ``rep_fusion``,
the multi-source length audit and the ``link_by`` executability guard.
"""

from __future__ import annotations

import pytest

from nirs4all.data.relations import (
    MissingRepetitionPolicy,
    MissingSourcePolicy,
    RelationValidationError,
    RepetitionSpec,
    RepOrder,
    SourceRepetitionSpec,
    audit_link_by_executable,
    audit_source_lengths,
    check_repetition_exclusivity,
    detect_repetition_mechanisms,
)

# ---------------------------------------------------------------------------
# RepetitionSpec.from_config
# ---------------------------------------------------------------------------


def test_from_config_full():
    cfg = {
        "sample_id": "sample_id",
        "link_by": "sample_id",
        "rep_order": "ordered",
        "strict_cardinality": True,
        "missing_repetition_policy": "pad",
        "missing_source_policy": "drop_incomplete",
        "sources": {
            "MIR": {"rep_col": "rep", "expected": 2},
            "RAMAN": {"expected": 3},
            "NIRS": None,
        },
    }
    spec = RepetitionSpec.from_config(cfg)
    assert spec.sample_id == "sample_id"
    assert spec.join_key == "sample_id"
    assert spec.rep_order is RepOrder.ORDERED
    assert spec.strict_cardinality is True
    assert spec.missing_repetition_policy is MissingRepetitionPolicy.PAD
    assert spec.missing_source_policy is MissingSourcePolicy.DROP_INCOMPLETE
    assert spec.source_spec("MIR") == SourceRepetitionSpec(rep_col="rep", expected=2)
    assert spec.source_spec("RAMAN") == SourceRepetitionSpec(expected=3)
    assert spec.source_spec("NIRS") == SourceRepetitionSpec()


def test_from_config_int_shorthand_for_expected():
    spec = RepetitionSpec.from_config({"sample_id": "id", "sources": {"A": 2, "B": 3}})
    assert spec.source_spec("A").expected == 2
    assert spec.source_spec("B").expected == 3


def test_join_key_defaults_to_sample_id():
    spec = RepetitionSpec(sample_id="my_id")
    assert spec.join_key == "my_id"


def test_distinct_link_by_is_rejected_until_two_column_identity_is_executable():
    with pytest.raises(RelationValidationError) as exc:
        RepetitionSpec(sample_id="physical_sample", link_by="vial_id")
    assert exc.value.code == "REL-E022"
    assert "Distinct link_by" in str(exc.value)


def test_from_config_requires_sample_id():
    with pytest.raises(RelationValidationError) as exc:
        RepetitionSpec.from_config({"sources": {"A": 2}})
    assert exc.value.code == "REL-E012"


def test_invalid_target_level_rejected():
    with pytest.raises(RelationValidationError) as exc:
        RepetitionSpec(sample_id="id", target_level="observation")
    assert exc.value.code == "REL-E011"


def test_negative_expected_rejected():
    with pytest.raises(RelationValidationError) as exc:
        SourceRepetitionSpec(expected=0)
    assert exc.value.code == "REL-E010"


def test_spec_round_trip_to_dict():
    spec = RepetitionSpec.from_config(
        {"sample_id": "id", "link_by": "id", "sources": {"A": {"expected": 2}}}
    )
    d = spec.to_dict()
    assert d["sample_id"] == "id"
    assert d["link_by"] == "id"
    assert d["sources"]["A"] == {"expected": 2}


# ---------------------------------------------------------------------------
# Exclusivity matrix
# ---------------------------------------------------------------------------


def test_detect_mechanisms():
    steps = [{"rep_to_sources": "Sample_ID"}, {"model": object()}]
    assert detect_repetition_mechanisms(steps) == {"rep_to_sources"}
    assert detect_repetition_mechanisms(steps, has_global_repetition=True) == {"rep_to_sources", "repetition"}


def test_exclusivity_single_mechanism_ok():
    check_repetition_exclusivity([{"rep_to_sources": "Sample_ID"}])  # no raise
    check_repetition_exclusivity([], has_global_repetition=True)  # no raise


def test_exclusivity_rep_fusion_with_rep_to_sources_rejected():
    steps = [{"rep_to_sources": "Sample_ID"}, {"rep_fusion": {"mode": "per_source_aggregate"}}]
    with pytest.raises(RelationValidationError) as exc:
        check_repetition_exclusivity(steps)
    assert exc.value.code == "REL-E008"


def test_exclusivity_global_repetition_with_rep_to_pp_rejected():
    with pytest.raises(RelationValidationError) as exc:
        check_repetition_exclusivity([{"rep_to_pp": "Sample_ID"}], has_global_repetition=True)
    assert exc.value.code == "REL-E008"


def test_exclusivity_rep_to_sources_and_rep_to_pp_rejected():
    steps = [{"rep_to_sources": "id"}, {"rep_to_pp": "id"}]
    with pytest.raises(RelationValidationError) as exc:
        check_repetition_exclusivity(steps)
    assert exc.value.code == "REL-E008"


# ---------------------------------------------------------------------------
# Multi-source length audit
# ---------------------------------------------------------------------------


def test_audit_equal_lengths_ok():
    audit_source_lengths([100, 100, 100])  # no raise


def test_audit_single_source_ok():
    audit_source_lengths([100])  # no raise


def test_audit_unequal_lengths_without_relation_raises():
    with pytest.raises(RelationValidationError) as exc:
        audit_source_lengths([200, 300, 200])
    assert exc.value.code == "REL-E009"


def test_audit_unequal_lengths_message_mentions_link_by():
    with pytest.raises(RelationValidationError) as exc:
        audit_source_lengths([200, 300], link_by="sample_id")
    assert "link_by" in str(exc.value)


def test_audit_unequal_lengths_in_relation_mode_defers():
    # When a relational plan owns alignment, the positional audit defers.
    audit_source_lengths([200, 300, 200], relation_mode=True)  # no raise


# ---------------------------------------------------------------------------
# link_by executability guard
# ---------------------------------------------------------------------------


def test_link_by_executable_ok_when_column_present():
    audit_link_by_executable(link_by="sample_id", relation_mode=True, available_columns=["sample_id", "x"])


def test_link_by_not_executable_raises():
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_executable(link_by="sample_id", relation_mode=True, available_columns=["x", "y"])
    assert exc.value.code == "REL-E005"


def test_link_by_ignored_outside_relation_mode():
    # Legacy profile: link_by is not enforced as a join (non-regression).
    audit_link_by_executable(link_by="sample_id", relation_mode=False, available_columns=["x"])  # no raise
