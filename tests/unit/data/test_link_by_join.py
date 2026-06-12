"""Unit tests for the executable N2 ``link_by`` join and relation-config parsing.

Complements ``test_relation_table.py`` (which covers the table internals) and
``test_repetition_spec.py`` (which covers spec parsing). Here the focus is the
N2 deliverables that make ``link_by`` a *real validated join*:

* parsing source-aware ``repetition_spec`` / ``relations`` / per-source
  ``link_by`` out of a legacy-shaped dataset dict (:func:`parse_relation_config`);
* the positional-alignment guardrail (:func:`audit_link_by_alignment`) rejecting
  divergent same-length ids, non-unique ids and missing key columns;
* the relational join honouring shuffled rows, sharing targets via ``link_by``
  and refusing accidental concatenation of heterogeneous sources.
"""

from __future__ import annotations

import pytest

from nirs4all.data.relations import (
    MissingSourcePolicy,
    RelationConfig,
    RelationValidationError,
    RepetitionSpec,
    RepOrder,
    SourceObservations,
    audit_link_by_alignment,
    build_relation_table,
    parse_relation_config,
)

# ---------------------------------------------------------------------------
# parse_relation_config
# ---------------------------------------------------------------------------


def test_parse_relation_config_from_repetition_spec():
    config = {
        "experimental_relation_pipeline": True,
        "repetition_spec": {
            "sample_id": "sid",
            "link_by": "sid",
            "sources": {"MIR": {"expected": 2}, "RAMAN": {"expected": 3}},
        },
    }
    rel = parse_relation_config(config)
    assert isinstance(rel, RelationConfig)
    assert rel.enabled is True
    assert rel.is_relational is True
    assert rel.link_by == "sid"
    assert rel.spec is not None
    assert sorted(rel.spec.sources) == ["MIR", "RAMAN"]


def test_parse_relation_config_link_by_from_sources_only():
    # No repetition_spec, but per-source link_by is declared -> link_by resolves.
    config = {"_sources": [{"name": "MIR", "link_by": "sid"}, {"name": "RAMAN", "link_by": "sid"}]}
    rel = parse_relation_config(config)
    assert rel is not None
    assert rel.spec is None
    assert rel.enabled is False
    assert rel.link_by == "sid"
    assert rel.is_relational is True


def test_parse_relation_config_relations_block_inline_spec():
    config = {"relations": {"sample_id": "sid", "sources": {"A": 2}}}
    rel = parse_relation_config(config)
    assert rel is not None
    assert rel.spec is not None
    assert rel.spec.sample_id == "sid"
    assert rel.link_by == "sid"


def test_parse_relation_config_returns_none_for_legacy_dict():
    assert parse_relation_config({"train_x": "a.csv", "train_y": "y.csv"}) is None
    assert parse_relation_config("not-a-mapping") is None  # type: ignore[arg-type]


def test_parse_relation_config_malformed_spec_raises():
    with pytest.raises(RelationValidationError) as exc:
        parse_relation_config({"repetition_spec": {"sources": {"A": 2}}})  # missing sample_id
    assert exc.value.code == "REL-E012"


def test_parse_relation_config_non_mapping_relations_raises():
    with pytest.raises(RelationValidationError) as exc:
        parse_relation_config({"relations": ["not", "a", "mapping"]})
    assert exc.value.code == "REL-E018"


def test_parse_relation_config_from_dataset_schema_legacy_dict():
    # End-to-end: the schema emits the experimental relational fields into the
    # legacy dict, and parse_relation_config reads them into a validated spec.
    from nirs4all.data.schema.config import DatasetConfigSchema

    schema = DatasetConfigSchema(
        sources=[
            {"name": "MIR", "train_x": "mir.csv", "link_by": "sid"},
            {"name": "RAMAN", "train_x": "raman.csv", "link_by": "sid"},
        ],
        experimental_relation_pipeline=True,
        repetition_spec={
            "sample_id": "sid",
            "link_by": "sid",
            "sources": {"MIR": {"expected": 2}, "RAMAN": {"expected": 3}},
        },
    )
    legacy = schema.to_legacy_format()
    rel = parse_relation_config(legacy)
    assert rel is not None
    assert rel.enabled is True
    assert rel.link_by == "sid"
    assert rel.spec is not None
    assert sorted(rel.spec.sources) == ["MIR", "RAMAN"]
    assert rel.spec.source_spec("RAMAN").expected == 3


def test_parse_relation_config_rejects_conflicting_source_link_by():
    config = {"_sources": [{"name": "MIR", "link_by": "sid"}, {"name": "RAMAN", "link_by": "other_id"}]}
    with pytest.raises(RelationValidationError) as exc:
        parse_relation_config(config)
    assert exc.value.code == "REL-E017"
    assert "disagree" in str(exc.value)


def test_parse_relation_config_rejects_spec_and_source_link_by_conflict():
    config = {
        "repetition_spec": {"sample_id": "sid", "link_by": "sid", "sources": {"MIR": 2}},
        "_sources": [{"name": "MIR", "link_by": "other_id"}],
    }
    with pytest.raises(RelationValidationError) as exc:
        parse_relation_config(config)
    assert exc.value.code == "REL-E017"
    assert "Conflicting link_by" in str(exc.value)


# ---------------------------------------------------------------------------
# audit_link_by_alignment -- positional safety guardrail
# ---------------------------------------------------------------------------


def test_alignment_ok_when_keys_match_positionally():
    audit_link_by_alignment(
        {"A": ["S1", "S2", "S3"], "B": ["S1", "S2", "S3"]},
        link_by="sid",
    )


def test_alignment_rejects_same_key_set_but_shuffled_order():
    # Same unique ids in a different order are safe for a relation join, but not
    # for legacy positional concatenation.
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_alignment(
            {"A": ["S1", "S2", "S3"], "B": ["S3", "S1", "S2"]},
            link_by="sid",
        )
    assert exc.value.code == "REL-E017"
    assert "row order differs" in str(exc.value)


def test_alignment_single_source_is_noop():
    audit_link_by_alignment({"A": ["S1", "S1"]}, link_by="sid")  # no raise


def test_alignment_divergent_same_length_ids_rejected():
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_alignment({"A": ["S1", "S2"], "B": ["S3", "S4"]}, link_by="sid")
    assert exc.value.code == "REL-E017"
    assert "diverge" in str(exc.value)


def test_alignment_different_lengths_message_is_explicit():
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_alignment({"A": ["S1", "S2"], "B": ["S1", "S2", "S3"]}, link_by="sid")
    assert exc.value.code == "REL-E017"
    assert "different row counts" in str(exc.value)
    assert "despite equal lengths" not in str(exc.value)


def test_alignment_non_unique_link_by_rejected():
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_alignment({"A": ["S1", "S1"], "B": ["S1", "S2"]}, link_by="sid")
    assert exc.value.code == "REL-E016"


def test_alignment_missing_key_column_rejected():
    with pytest.raises(RelationValidationError) as exc:
        audit_link_by_alignment({"A": None, "B": ["S1", "S2"]}, link_by="sid")
    assert exc.value.code == "REL-E005"


def test_alignment_non_unique_tolerated_when_not_required():
    # When 1:1 uniqueness is not required, repeats are tolerated as long as the
    # key *sets* match -- this is the repetition (relational join) regime.
    audit_link_by_alignment(
        {"A": ["S1", "S1", "S2"], "B": ["S2", "S1", "S1"]},
        link_by="sid",
        require_unique=False,
    )


# ---------------------------------------------------------------------------
# Relational join semantics (shuffled / shared targets / divergence)
# ---------------------------------------------------------------------------


def _spec(**kwargs) -> RepetitionSpec:
    base = {"sample_id": "sid", "link_by": "sid"}
    base.update(kwargs)
    return RepetitionSpec(**base)


def test_join_is_by_key_not_position_under_shuffle():
    # Heterogeneous MIR=2 / RAMAN=3, with both sources shuffled. The join is by
    # key, so the table is identical to the unshuffled layout.
    mir = SourceObservations("MIR", ["S1", "S1", "S2", "S2"])
    raman = SourceObservations("RAMAN", ["S1", "S2", "S1", "S2", "S1", "S2"])
    ref = build_relation_table(_spec(), [mir, raman])

    mir_shuf = SourceObservations("MIR", ["S2", "S1", "S2", "S1"])
    raman_shuf = SourceObservations("RAMAN", ["S2", "S2", "S1", "S1", "S2", "S1"])
    shuffled = build_relation_table(_spec(), [raman_shuf, mir_shuf])

    assert ref.fingerprint() == shuffled.fingerprint()
    assert ref.cardinalities() == shuffled.cardinalities()


def test_shared_targets_via_link_by():
    # The target is declared on a single source; the join shares it across the
    # physical sample (and thus its other sources). Targets stay sample-level.
    mir = SourceObservations("MIR", ["S1", "S1", "S2", "S2"], targets=[10.0, 10.0, 20.0, 20.0])
    raman = SourceObservations("RAMAN", ["S2", "S1", "S1", "S2", "S1", "S2"])  # no targets
    table = build_relation_table(_spec(), [mir, raman])

    assert table.targets_by_sample() == {"S1": 10.0, "S2": 20.0}
    # RAMAN observations carry no per-row target; the value is sample-level only.
    raman_targets = {r.target_id for r in table.records if r.source_id == "RAMAN"}
    assert raman_targets == {None}


def test_shared_target_conflict_via_link_by_rejected():
    # Two sources disagree on the shared sample target -> contradiction.
    mir = SourceObservations("MIR", ["S1", "S1"], targets=[10.0, 10.0])
    raman = SourceObservations("RAMAN", ["S1"], targets=[99.0])
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [mir, raman])
    assert exc.value.code == "REL-E002"


def test_missing_source_strict_is_default_refusal():
    # S2 has no RAMAN -> strict (default) refuses; this is the divergent-id case
    # surfaced by the real join rather than by positional concatenation.
    mir = SourceObservations("MIR", ["S1", "S2"])
    raman = SourceObservations("RAMAN", ["S1"])
    with pytest.raises(RelationValidationError) as exc:
        build_relation_table(_spec(), [mir, raman])
    assert exc.value.code == "REL-E007"


def test_missing_source_drop_incomplete_allows_partial_coverage():
    mir = SourceObservations("MIR", ["S1", "S2"])
    raman = SourceObservations("RAMAN", ["S1"])
    table = build_relation_table(
        _spec(missing_source_policy=MissingSourcePolicy.DROP_INCOMPLETE),
        [mir, raman],
    )
    # Both samples still materialise as observations; coverage handling is later.
    assert table.physical_sample_ids == ["S1", "S2"]


def test_explicit_rep_ids_are_honoured():
    spec = _spec(
        rep_order=RepOrder.ORDERED,
        missing_source_policy=MissingSourcePolicy.IMPUTE_DECLARED,
    )
    src = SourceObservations("A", ["S1", "S1", "S1"], rep_ids=[2, 0, 1])
    table = build_relation_table(spec, [src])
    reps = sorted(r.rep_id for r in table.records if r.physical_sample_id == "S1")
    assert reps == [0, 1, 2]
