"""Full-training candidate expansion shares the public generator contract."""

from __future__ import annotations

import copy

import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.pipeline.config.component_serialization import serialize_component
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.dagml.full_train_variants import expand_full_train_variants
from nirs4all.pipeline.dagml.steps import _apply_model_params, _is_split_step


def test_model_choices_keep_public_order_names_and_independent_instances():
    pipeline = [StandardScaler(), {"model": {"_or_": [Ridge(alpha=0.1), Ridge(alpha=2.0)]}}]
    expected = PipelineConfigs(pipeline, name="comparison")
    variants = expand_full_train_variants(pipeline, name="comparison")
    assert [name for _, name in variants] == expected.names
    assert [steps[-1]["model"].alpha for steps, _ in variants] == [0.1, 2.0]
    assert variants[0][0][0] is not variants[1][0][0]
    assert variants[0][0][0] is not pipeline[0]
    assert all(not any(_is_split_step(step) for step in steps) for steps, _ in variants)


def test_nested_generator_stages_are_flattened_without_mutating_input():
    pipeline = [{"_or_": [[StandardScaler(), MinMaxScaler()], None]}, Ridge()]
    before = serialize_component(pipeline)
    variants = expand_full_train_variants(pipeline)
    assert [len(steps) for steps, _ in variants] == [3, 2]
    assert isinstance(variants[0][0][0], StandardScaler)
    assert isinstance(variants[0][0][1], MinMaxScaler)
    assert variants[1][0][0] is None
    assert serialize_component(pipeline) == before


@pytest.mark.parametrize("canonical", [False, True])
def test_param_keyed_sweep_expands_bare_model_class(canonical):
    pipeline = [{"model": Ridge, "_range_": [1, 3, 1], "param": "alpha"}]
    definition = serialize_component(pipeline) if canonical else pipeline
    before = copy.deepcopy(serialize_component(definition))
    variants = expand_full_train_variants(definition)
    concrete = [_apply_model_params(steps) for steps, _ in variants]
    assert [steps[0]["model"].alpha for steps in concrete] == [1, 2, 3]
    assert serialize_component(definition) == before


def test_canonical_splitter_choice_is_detectable_after_expansion():
    pipeline = [{"_or_": [None, KFold(n_splits=3)]}, Ridge()]
    variants = expand_full_train_variants(serialize_component(pipeline))
    assert [any(_is_split_step(step) for step in steps) for steps, _ in variants] == [False, True]


def test_seeded_choices_are_reproducible():
    pipeline = [{"_or_": [Ridge(alpha=float(i)) for i in range(8)], "count": 3, "_seed_": 42}]
    first = expand_full_train_variants(pipeline)
    second = expand_full_train_variants(pipeline)
    assert [name for _, name in first] == [name for _, name in second]
    assert len(first) == 3
    assert [steps[0].alpha for steps, _ in first] == [steps[0].alpha for steps, _ in second]


def test_public_generation_limit_remains_enforced():
    pipeline = [{"model": Ridge, "_range_": [1, 10001, 1], "param": "alpha"}]
    with pytest.raises(ValueError, match="exceeding the limit"):
        expand_full_train_variants(pipeline)
