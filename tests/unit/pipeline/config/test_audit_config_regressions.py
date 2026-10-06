"""Regression witnesses for the reviewed October configuration audit."""

import pytest
from sklearn.linear_model import Ridge

from nirs4all.optimization.optuna import OptunaManager
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.config._generator.iterator import expand_spec_iter
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.config.generator import count_combinations, expand_spec, expand_spec_with_choices


@pytest.mark.parametrize("spec", [
    {"_grid_": {"x": [{"_range_": [1, 3]}, 10]}},
    {"_or_": [{"_range_": [1, 3]}, "B"], "pick": 2},
    {"_or_": [{"_range_": [1, 3]}, "B"], "arrange": 2},
    {"_zip_": {"x": [], "y": [1]}},
    *({"_or_": [{"w": {"_range_": [5, 7, 2]}}, "snv", "msc"], primary: 1, secondary: 2}
      for primary in ("pick", "arrange") for secondary in ("then_pick", "then_arrange")),
])
def test_nested_counts_match_expanded_variants(spec):
    expanded = expand_spec(spec)
    assert count_combinations(spec) == len(expanded)
    assert "_range_" not in repr(expanded)


@pytest.mark.parametrize("shape", ["list", "steps", "pipeline"])
def test_validator_accepts_loader_pipeline_forms(tmp_path, shape):
    import json

    from nirs4all.config.validator import validate_config_file, validate_pipeline_config

    steps = [{"class": "sklearn.linear_model.Ridge"}]
    config = steps if shape == "list" else {shape: steps}
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config))
    assert validate_pipeline_config(config)[0]
    assert validate_config_file(str(path))[0]
    assert validate_config_file(str(path), config_type="pipeline")[0]


@pytest.mark.parametrize("spec", [[["A", "B"]], [[{"_or_": ["A", "B"]}, "C"]]])
def test_literal_nested_list_remains_a_sequence(spec):
    eager = expand_spec(spec)
    assert all(isinstance(result[0], list) for result in eager)
    assert list(expand_spec_iter(spec)) == eager
    assert [value for value, _ in expand_spec_with_choices(spec)] == eager


def test_augmentation_count_does_not_trigger_generator_expansion():
    step = {"sample_augmentation": {"count": 2, "transformers": []}}
    assert not PipelineConfigs._has_gen_keys([step])


def test_yaml_mapping_uses_pipeline_list(tmp_path):
    text = "pipeline:\n  - class: sklearn.linear_model.Ridge\n"
    assert len(PipelineConfigs(text).steps[0]) == 1
    path = tmp_path / "pipeline.YAML"
    path.write_text(text)
    assert PipelineConfigs(str(path)).steps == PipelineConfigs(text).steps
    with pytest.raises(ValueError, match="Pipeline YAML must contain"):
        PipelineConfigs("configs/my_pipeline")


@pytest.mark.parametrize("value", ["./x.csv", "../ref/y.csv", ".5", "./ckpt"])
def test_relative_parameter_strings_roundtrip(value):
    assert serialize_component(value) == value
    assert deserialize_component(value) == value


def test_seed_is_consumed_in_mixed_or_nodes():
    spec = {"_or_": [{"a": i} for i in range(10)], "count": 3, "_seed_": 17, "b": 5}
    eager = expand_spec(spec)
    assert expand_spec(spec) == eager
    assert list(expand_spec_iter(spec)) == eager
    assert len(eager) == 3
    assert all(set(value) == {"a", "b"} for value in eager)


def test_lazy_and_eager_sibling_sweeps_match():
    spec = {"model": {"class": "sklearn.linear_model.Ridge"}, "_range_": [2, 6, 2], "param": "alpha"}
    assert list(expand_spec_iter(spec)) == expand_spec(spec)
    assert len(expand_spec(spec)) == 3


def test_numeric_tuple_range_does_not_become_endpoint_choices():
    step = {"model": Ridge(), "finetune_params": {"model_params": {"alpha": (1, 5), "max_iter": [1, 5]}}}
    params = PipelineConfigs([step]).steps[0][0]["finetune_params"]["model_params"]
    assert params["alpha"] == {"type": "int", "min": 1, "max": 5}
    assert params["max_iter"] == [1, 5]
    import optuna

    trial = optuna.trial.FixedTrial({"alpha": 3, "max_iter": 5})
    sampled, _ = OptunaManager().sample_hyperparameters(trial, {"model_params": params})
    assert sampled == {"alpha": 3, "max_iter": 5}
