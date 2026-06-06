"""Tests for nirs4all.data.repetition_detection."""

from __future__ import annotations

import pytest

from nirs4all.data.repetition_detection import (
    RepetitionGroups,
    auto_detect_repetition_column,
    detect_repetition_groups,
)


class TestAutoDetectRepetitionColumn:
    def test_prefers_bio_sample_column(self):
        metadata = {
            "set": ["train", "train", "test", "test"],
            "bio_sample": ["A", "A", "B", "B"],
            "note": ["x", "y", "z", "w"],
        }
        assert auto_detect_repetition_column(metadata) == "bio_sample"

    def test_skips_partition_fold_and_repeat_index_columns(self):
        metadata = {
            "fold": [1, 1, 2, 2],
            "replicate": [1, 2, 1, 2],
            "partition": ["a", "a", "b", "b"],
        }
        assert auto_detect_repetition_column(metadata) is None

    def test_requires_repeated_values(self):
        metadata = {"bio_sample": ["A", "B", "C", "D"]}
        assert auto_detect_repetition_column(metadata) is None

    def test_sample_group_variant_qualifies(self):
        metadata = {"Sample Group": ["g1", "g1", "g2", "g2"]}
        assert auto_detect_repetition_column(metadata) == "Sample Group"

    def test_ignores_empty_and_none_values(self):
        metadata = {"bio_sample": [None, "", "A", "A"]}
        assert auto_detect_repetition_column(metadata) == "bio_sample"


class TestDetectRepetitionGroups:
    def test_rep_suffix_convention(self):
        ids = ["s1_rep1", "s1_rep2", "s2_rep1", "s2_rep2"]
        result = detect_repetition_groups(ids)
        assert result.has_repetitions
        assert result.groups == {"s1": [0, 1], "s2": [2, 3]}
        assert result.pattern is not None

    def test_numeric_suffix_convention(self):
        ids = ["wheat_1", "wheat_2", "corn_1", "corn_2", "corn_3"]
        result = detect_repetition_groups(ids)
        assert result.groups == {"wheat": [0, 1], "corn": [2, 3, 4]}
        assert result.n_repeated == 2

    def test_unique_ids_have_no_repetitions(self):
        result = detect_repetition_groups(["a", "b", "c"])
        assert not result.has_repetitions
        assert result.pattern is None

    def test_explicit_pattern(self):
        ids = ["X-01-a", "X-01-b", "Y-02-a"]
        result = detect_repetition_groups(ids, pattern=r"^([A-Z]-\d+)")
        assert result.groups == {"X-01": [0, 1], "Y-02": [2]}
        assert result.pattern == r"^([A-Z]-\d+)"
        assert result.n_repeated == 1

    def test_explicit_pattern_non_matching_ids_form_own_groups(self):
        ids = ["X-01-a", "weird"]
        result = detect_repetition_groups(ids, pattern=r"^([A-Z]-\d+)")
        assert result.groups == {"X-01": [0], "weird": [1]}

    def test_invalid_explicit_pattern_raises(self):
        with pytest.raises(Exception):
            detect_repetition_groups(["a"], pattern="(")

    def test_dataclass_defaults(self):
        empty = RepetitionGroups()
        assert not empty.has_repetitions
        assert empty.groups == {}
