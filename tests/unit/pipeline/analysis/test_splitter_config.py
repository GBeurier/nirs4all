"""Tests for nirs4all.pipeline.analysis.splitter_config."""

from __future__ import annotations

import json

from nirs4all.pipeline.analysis.splitter_config import (
    extract_splitter_config,
    is_splitter_reference,
    parse_expanded_config_steps,
)


class TestParseExpandedConfigSteps:
    def test_json_string(self):
        steps = parse_expanded_config_steps(json.dumps([{"class": "a.B"}]))
        assert steps == [{"class": "a.B"}]

    def test_dict_with_pipeline_key(self):
        assert parse_expanded_config_steps({"pipeline": [1, 2]}) == [1, 2]

    def test_none_and_scalar(self):
        assert parse_expanded_config_steps(None) == []
        assert parse_expanded_config_steps("not json") == ["not json"]


class TestIsSplitterReference:
    def test_sklearn_splitters(self):
        assert is_splitter_reference("sklearn.model_selection._split.KFold")
        assert is_splitter_reference("KFold")
        assert is_splitter_reference("StratifiedShuffleSplit")

    def test_nirs4all_registry(self):
        assert is_splitter_reference("nirs4all.operators.splitters.KennardStoneSplitter")
        assert is_splitter_reference("SPXYSplitter")

    def test_token_fallback_for_custom(self):
        assert is_splitter_reference("my.lab.CustomHoldoutThing")
        assert is_splitter_reference("some.pkg.WeirdFolder")  # 'fold' token

    def test_non_splitters(self):
        assert not is_splitter_reference("sklearn.preprocessing.StandardScaler")
        assert not is_splitter_reference("sklearn.cross_decomposition.PLSRegression")
        assert not is_splitter_reference("")


class TestExtractSplitterConfig:
    def test_full_kfold(self):
        config = extract_splitter_config([
            {"class": "sklearn.preprocessing._data.MinMaxScaler"},
            {"class": "sklearn.model_selection._split.KFold",
             "params": {"n_splits": 5, "shuffle": True, "random_state": 42}},
            {"model": {"class": "PLSRegression"}},
        ])
        assert config is not None
        assert config.splitter_class == "KFold"
        assert config.n_splits == 5
        assert config.shuffle is True
        assert config.random_state == 42
        assert config.test_size is None

    def test_shuffle_split_test_size_and_groups(self):
        config = extract_splitter_config([
            {"class": "GroupShuffleSplit", "params": {"test_size": 0.2, "group_by": "sample_id"}},
        ])
        assert config is not None
        assert config.test_size == 0.2
        assert config.group_by == "sample_id"

    def test_repr_string_steps_are_normalized(self):
        config = extract_splitter_config([
            {"class": "<sklearn.model_selection._split.KFold object at 0x7f3a2b>"},
        ])
        assert config is not None
        assert config.splitter_class == "KFold"

    def test_no_splitter_returns_none(self):
        assert extract_splitter_config([
            {"class": "sklearn.preprocessing.StandardScaler"},
        ]) is None
        assert extract_splitter_config(None) is None

    def test_json_string_input(self):
        config = extract_splitter_config(json.dumps({
            "pipeline": [{"class": "KFold", "params": {"n_splits": 3}}],
        }))
        assert config is not None
        assert config.n_splits == 3
