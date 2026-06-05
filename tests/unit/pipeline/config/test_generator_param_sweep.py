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


# ---------------------------------------------------------------------------
# Param-map form: {<_grid_|_zip_>: {param: values, ...}, model: {class}}
# The generator's value *is* the operator's parameter set (no separate `param`).
# ---------------------------------------------------------------------------


def test_grid_over_model_expands_to_cartesian_product():
    spec = {"_grid_": {"n_components": [5, 10, 15], "scale": [True, False]},
            "model": {"class": P}}
    out = expand_spec(spec)
    assert len(out) == 6
    assert count_combinations(spec) == 6
    assert out[0] == {"model": {"class": P, "params": {"n_components": 5, "scale": True}}}
    assert out[-1] == {"model": {"class": P, "params": {"n_components": 15, "scale": False}}}


def test_grid_over_model_equivalent_to_nested():
    sibling = {"_grid_": {"n_components": [5, 10], "scale": [True, False]},
               "model": {"class": P}}
    nested = {"model": {"class": P,
                        "params": {"_grid_": {"n_components": [5, 10], "scale": [True, False]}}}}
    assert expand_spec(sibling) == expand_spec(nested)


def test_zip_over_model_pairs_by_position():
    spec = {"_zip_": {"n_components": [5, 10, 15], "scale": [True, False, True]},
            "model": {"class": P}}
    out = expand_spec(spec)
    assert len(out) == 3
    assert count_combinations(spec) == 3
    assert [(v["model"]["params"]["n_components"], v["model"]["params"]["scale"]) for v in out] == [
        (5, True), (10, False), (15, True)
    ]


def test_grid_over_model_count_matches_expand():
    spec = {"_grid_": {"a": [1, 2], "b": ["x", "y", "z"]}, "model": {"class": P}}
    assert count_combinations(spec) == len(expand_spec(spec)) == 6


def test_grid_over_top_level_class_dict():
    spec = {"_grid_": {"n_components": [3, 6]}, "class": P}
    out = expand_spec(spec)
    assert out == [{"class": P, "params": {"n_components": v}} for v in (3, 6)]


def test_grid_over_model_tracks_choices():
    spec = {"_grid_": {"n_components": [5, 10]}, "model": {"class": P}}
    choices = [c for _, c in expand_spec_with_choices(spec)]
    assert choices == [
        [{"_grid_": {"n_components": 5}}],
        [{"_grid_": {"n_components": 10}}],
    ]


def test_grid_with_static_params_left_unchanged():
    # Static params alongside a param-map is an unsupported shape -> returned as-is.
    spec = {"_grid_": {"n_components": [5, 10]},
            "model": {"class": P, "params": {"scale": False}}}
    assert expand_spec(spec) == [spec]
    assert count_combinations(spec) == 1


def test_pure_grid_node_unaffected():
    # A bare _grid_ node (no operator) still expands to param dicts as before.
    assert expand_spec({"_grid_": {"x": [1, 2], "y": ["A", "B"]}}) == [
        {"x": 1, "y": "A"}, {"x": 1, "y": "B"}, {"x": 2, "y": "A"}, {"x": 2, "y": "B"},
    ]


def test_grid_over_raw_model_class_via_pipelineconfigs():
    # The DOCUMENTED step-level form uses a RAW class object (not a {"class": ...} dict):
    #     {"_grid_": {...}, "model": PLSRegression}
    # PipelineConfigs normalizes the raw class to a {"class": ...} dict *before* generator
    # expansion runs, so _normalize_param_grid still fires and the grid expands to one model
    # per combination. This pins the documented end-to-end path (raw class -> 6 variants),
    # complementing the {"class": ...}-dict unit cases above.
    from sklearn.cross_decomposition import PLSRegression

    from nirs4all.pipeline.config import PipelineConfigs

    spec = [{"_grid_": {"n_components": [5, 10, 15], "scale": [True, False]}, "model": PLSRegression}]
    pc = PipelineConfigs(spec, "grid_raw_class")
    assert len(pc.steps) == 6
