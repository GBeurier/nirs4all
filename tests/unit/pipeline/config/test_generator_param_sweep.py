"""Sibling-form parameter sweep: {operator, <gen_kw>, "param": name} -> nested params sweep.

Regression coverage for the ergonomic step-level form

    {"model": {"class": P}, "_range_": [2, 20, 2], "param": "n_components"}

which must expand to one variant per generated value with the named param injected into the
operator's ``params`` (equivalent to the canonical nested form).
"""

from nirs4all.pipeline.config.generator import (
    count_combinations,
    expand_spec,
    expand_spec_with_choices,
)

P = "sklearn.cross_decomposition.PLSRegression"


def test_sibling_range_sweep_expands_per_value():
    spec = {"model": {"class": P}, "_range_": [2, 20, 2], "param": "n_components"}
    out = expand_spec(spec)
    assert len(out) == 10
    assert out[0] == {"model": {"class": P, "params": {"n_components": 2}}}
    assert out[-1] == {"model": {"class": P, "params": {"n_components": 20}}}


def test_sibling_sweep_count_matches_expand():
    spec = {"model": {"class": P}, "_range_": [2, 20, 2], "param": "n_components"}
    assert count_combinations(spec) == len(expand_spec(spec))


def test_sibling_sweep_equivalent_to_nested():
    sibling = {"model": {"class": P}, "_range_": [2, 20, 2], "param": "n_components"}
    nested = {"model": {"class": P, "params": {"n_components": {"_range_": [2, 20, 2]}}}}
    assert expand_spec(sibling) == expand_spec(nested)


def test_sibling_sweep_tracks_choices():
    spec = {"model": {"class": P}, "_range_": [3, 9, 3], "param": "n_components"}
    choices = [c for _, c in expand_spec_with_choices(spec)]
    assert choices == [[{"_range_": 3}], [{"_range_": 6}], [{"_range_": 9}]]


def test_sibling_sweep_top_level_class_dict():
    spec = {"class": P, "_range_": [3, 9, 3], "param": "n_components"}
    out = expand_spec(spec)
    assert out == [{"class": P, "params": {"n_components": v}} for v in (3, 6, 9)]


def test_sibling_sweep_merges_existing_params():
    spec = {"model": {"class": P, "params": {"scale": False}},
            "_range_": [1, 3], "param": "n_components"}
    out = expand_spec(spec)
    assert out[0]["model"]["params"] == {"scale": False, "n_components": 1}
    assert len(out) == 3


def test_sibling_log_range_sweep():
    spec = {"model": {"class": "sklearn.linear_model.Ridge"},
            "_log_range_": [0.001, 1, 4], "param": "alpha"}
    out = expand_spec(spec)
    assert [v["model"]["params"]["alpha"] for v in out] == [0.001, 0.01, 0.1, 1.0]


def test_param_key_without_generator_is_untouched():
    # No generation keyword -> not a sweep; the literal 'param' key is preserved.
    spec = {"model": {"class": P}, "param": "n_components"}
    assert expand_spec(spec) == [spec]


def test_nested_form_unaffected():
    nested = {"model": {"class": P, "params": {"n_components": {"_range_": [2, 20, 2]}}}}
    assert len(expand_spec(nested)) == 10


def test_or_unaffected():
    assert expand_spec({"_or_": ["A", "B", "C"]}) == ["A", "B", "C"]
