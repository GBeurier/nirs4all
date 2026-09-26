"""The shared n4m recipe names resolve to Methods-backed Python operators."""

import json

import numpy as np
import pytest

from nirs4all.pipeline.config.component_serialization import deserialize_component
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.steps.parser import StepParser

try:
    import n4m  # noqa: F401
    import pls4all  # noqa: F401
except (ImportError, OSError, RuntimeError) as error:
    pytest.skip(f"Methods Python runtime is unavailable: {error}", allow_module_level=True)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("n4m.KennardStone", "KennardStoneSplitter"),
        ("n4m.SNV", "SNV"),
        ("n4m.SavitzkyGolay", "SavitzkyGolay"),
        ("n4m.LSNV", "LSNV"),
        ("n4m.RNV", "RNV"),
        ("n4m.AreaNormalization", "AreaNormalization"),
        ("n4m.Detrend", "Detrend"),
        ("n4m.MSC", "MSC"),
        ("n4m.EMSC", "EMSC"),
        ("n4m.PLS", "PLSRegression"),
        ("n4m.PLSRegression", "PLSRegression"),
        ("n4m.SparsePLSDA", "SparsePLSDAClassifier"),
        ("n4m.Ridge", "Ridge"),
        ("n4m.RidgePLS", "NativeRidgePLSRegressor"),
        ("n4m.RobustPLS", "NativeRobustPLSRegressor"),
        ("n4m.CPPLS", "CPPLSRegression"),
        ("n4m.SparseSIMPLS", "SparseSimplsRegression"),
        ("n4m.ECR", "ECRegression"),
        ("n4m.ContinuumRegression", "NativeContinuumRegressionRegressor"),
        ("n4m.MIRPLS", "MIRPLSRegression"),
        ("n4m.FusedSparsePLS", "FusedSparsePLSRegression"),
        ("n4m.BaggingPLS", "BaggingPLSRegression"),
        ("n4m.BoostingPLS", "BoostingPLSRegression"),
        ("n4m.RandomSubspacePLS", "RandomSubspacePLSRegression"),
    ],
)
def test_portable_alias_resolves_with_methods(alias: str, expected: str) -> None:
    operator = StepParser().parse({"class": alias}).operator
    assert type(operator).__name__ == expected
    assert deserialize_component(alias).__class__.__name__ == expected


def test_portable_pls_preserves_native_r_scaling_defaults() -> None:
    default = StepParser().parse({"model": {"class": "n4m.PLS", "params": {"n_components": 3}}}).operator
    override = StepParser().parse(
        {"model": {"class": "n4m.PLS", "params": {"n_components": 3, "scale_y": False}}}
    ).operator
    assert default.solver == "simpls"
    assert default.scale_y is True
    assert override.scale_y is False


def test_affine_recipe_defaults_match_r_dispatch() -> None:
    ridge_pls = StepParser().parse({"model": {"class": "n4m.RidgePLS"}}).operator
    robust = StepParser().parse({"model": {"class": "n4m.RobustPLS"}}).operator
    assert ridge_pls.ridge_lambda == 1.0
    assert ridge_pls.scale_x is False
    assert robust.max_irls_iter == 20
    assert robust.scale_x is False
    multiblock = StepParser().parse({"model": {"class": "n4m.MBPLS", "params": {
        "n_components": 2, "block_sizes": [4, 4, 4]}}}).operator
    assert multiblock.scale_x is False
    assert multiblock.scale_y is False


def test_group_sparse_alias_resolves_with_explicit_groups() -> None:
    operator = StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": {
        "n_components": 2, "group_assignment": [0, 0, 1, 1]}}}).operator
    assert type(operator).__name__ == "GroupSparsePLS"
    np.testing.assert_array_equal(operator.group_assignment, [0, 0, 1, 1])
    assert operator.group_lambda == 0.05


def test_group_sparse_string_alias_requires_groups() -> None:
    with pytest.raises(ValueError, match="group_assignment"):
        deserialize_component("n4m.GroupSparsePLS")


@pytest.mark.parametrize("extension", ["json", "yaml"])
@pytest.mark.parametrize(("groups", "expected"), [
    ([0] * 4 + [1] * 4 + [2] * 4,
     [1.2560303930849952, 1.8411427818857744, 0.8454051444648407]),
    ([0, 1, 2] * 4,
     [1.3302312403142085, 1.9038372601712736, 0.8103946779875588]),
])
def test_group_sparse_recipe_matches_r_native_heldout(
        tmp_path, extension: str, groups: list[int], expected: list[float]) -> None:
    samples = np.arange(1, 22, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    X_test = X[[1, 7, 16], :] + 0.031
    recipe = {"pipeline": [{"model": {"class": "n4m.GroupSparsePLS", "params": {
        "n_components": 2, "group_assignment": groups, "group_lambda": 0.05}}}]}
    path = tmp_path / f"group_sparse.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        path.write_text("pipeline:\n  - model:\n      class: n4m.GroupSparsePLS\n"
                        "      params:\n        n_components: 2\n"
                        f"        group_assignment: {groups}\n"
                        "        group_lambda: 0.05\n", encoding="utf-8")
    operator = StepParser().parse(PipelineConfigs(str(path)).steps[0][0]).operator
    np.testing.assert_array_equal(operator.group_assignment, groups)
    np.testing.assert_allclose(operator.fit(X, y).predict(X_test),
                               expected, rtol=0, atol=1e-10)


def test_group_sparse_recipe_rejects_feature_width_mismatch() -> None:
    operator = StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": {
        "n_components": 2, "group_assignment": [0, 0, 1], "group_lambda": 0.05}}}).operator
    with pytest.raises(ValueError, match="group_assignment|feature"):
        operator.fit(np.ones((6, 4)), np.arange(6, dtype=float))


def test_group_sparse_recipe_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Invalid parameters"):
        StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": {
            "group_assignment": [0, 1], "unknown_native_parameter": 1}}})


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("n4m.Ridge", [1.119585507454503, 2.2247502591042467, 0.8880639379404299]),
        ("n4m.RidgePLS", [1.0183856539647425, 2.164229001896515, 0.9228238984475714]),
        ("n4m.RobustPLS", [1.2000108169670543, 1.5460522175157725, 0.7548630520848121]),
        ("n4m.CPPLS", [0.9908421322688001, 2.2185030364862115, 0.9364520576202219]),
        ("n4m.SparseSIMPLS", [0.9908421322687999, 2.2185030364862115, 0.9364520576202217]),
        ("n4m.ECR", [1.1953068770625324, 2.0055006340124004, 0.4704252881332379]),
        ("n4m.ContinuumRegression", [1.0601251478760563, 2.285195247718287, 0.919738050799327]),
        ("n4m.MIRPLS", [1.184521178890891, 2.2497913086895096, 0.26128963080778034]),
    ],
)
def test_affine_recipe_held_out_matches_r_n4m(alias: str, expected: list[float]) -> None:
    samples = np.arange(1, 22, dtype=np.float64)[:, None]
    bands = np.arange(1, 9, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    X_test = X[[1, 7, 16], :] + 0.031
    params = {} if alias == "n4m.Ridge" else {"n_components": 2}
    operator = StepParser().parse({"model": {"class": alias, "params": params}}).operator
    np.testing.assert_allclose(operator.fit(X, y).predict(X_test), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("extension", ["json", "yaml"])
@pytest.mark.parametrize(("alias", "params", "expected"), [
    # Reconfirmed independently with R n4m 1.0.21.9004 (Methods d058890d).
    ("n4m.FusedSparsePLS", {"l1_lambda": 0.05, "fusion_lambda": 0.05},
     [1.3373539467794611, 1.9377148952044152, 0.7544322586932715]),
    ("n4m.BaggingPLS", {"n_estimators": 7, "seed": 13},
     [1.340366231116839, 2.072160066869946, 0.8905637569136029]),
    ("n4m.BoostingPLS", {"n_estimators": 7, "learning_rate": 0.3},
     [1.190830194241934, 2.206788158315018, 0.9303362422665358]),
    ("n4m.RandomSubspacePLS", {"n_estimators": 7,
                               "features_per_subspace": 5, "seed": 13},
     [1.39393404029896, 1.847198984652901, 0.8903822546307825]),
])
def test_extra_affine_recipe_matches_r_native_heldout(
        tmp_path, extension: str, alias: str,
        params: dict, expected: list[float]) -> None:
    samples = np.arange(1, 22, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    X_test = X[[1, 7, 16], :] + 0.031
    recipe = {"pipeline": [{"model": {"class": alias,
                                      "params": {"n_components": 2, **params}}}]}
    path = tmp_path / f"extra_affine.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        lines = ["pipeline:", "  - model:", f"      class: {alias}", "      params:",
                 "        n_components: 2"]
        lines.extend(f"        {key}: {value}" for key, value in params.items())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = PipelineConfigs(str(path))
    operator = StepParser().parse(config.steps[0][0]).operator
    np.testing.assert_allclose(operator.fit(X, y).predict(X_test),
                               expected, rtol=0, atol=1e-10)


def test_extra_affine_recipe_rejects_unknown_parameter() -> None:
    with pytest.raises((TypeError, ValueError)):
        StepParser().parse({"model": {"class": "n4m.BaggingPLS",
                                       "params": {"unknown_native_parameter": 1}}})


@pytest.mark.parametrize("bad", [
    {},
    {"group_assignment": []},
    {"group_assignment": [0]},
    {"group_assignment": [0, -1]},
    {"group_assignment": [0, 2**31]},
    {"group_assignment": [0, 1.0]},
    {"group_assignment": [0, True]},
    {"group_assignment": "0,1"},
])
def test_group_sparse_recipe_rejects_invalid_groups(bad: dict) -> None:
    with pytest.raises(ValueError, match="group_assignment"):
        StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": bad}})


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), 10**1000, True, "0.05"])
def test_group_sparse_recipe_rejects_invalid_lambda(bad: object) -> None:
    with pytest.raises(ValueError, match="group_lambda"):
        StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": {
            "group_assignment": [0, 1], "group_lambda": bad}}})


@pytest.mark.parametrize("bad", [0, -1, 2**31, 2.0, True])
def test_group_sparse_recipe_rejects_invalid_components(bad: object) -> None:
    with pytest.raises(ValueError, match="n_components"):
        StepParser().parse({"model": {"class": "n4m.GroupSparsePLS", "params": {
            "n_components": bad, "group_assignment": [0, 1]}}})


@pytest.mark.parametrize("alias,params", [
    ("n4m.BaggingPLS", {"seed": 2**31}),
    ("n4m.RandomSubspacePLS", {"seed": 2**31}),
    ("n4m.BoostingPLS", {"learning_rate": 1.2}),
])
def test_extra_affine_recipe_rejects_out_of_shared_range(alias: str, params: dict) -> None:
    with pytest.raises(ValueError, match="seed|learning_rate"):
        StepParser().parse({"model": {"class": alias, "params": params}})


@pytest.mark.parametrize("extension", ["json", "yaml"])
@pytest.mark.parametrize("multi_target", [False, True])
def test_npls_recipe_matches_r_native_heldout(tmp_path, extension: str,
                                              multi_target: bool) -> None:
    samples = np.arange(1, 22, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    y2 = 0.3 - 0.2 * X[:, 3] + 0.5 * X[:, 9]
    target = np.column_stack((y, y2)) if multi_target else y
    X_test = X[[1, 7, 16], :] + 0.031
    recipe = {"pipeline": [{"model": {"class": "n4m.NPLS", "params": {
        "n_components": 2, "mode_j": 3, "mode_k": 4}}}]}
    path = tmp_path / f"npls.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        path.write_text("pipeline:\n  - model:\n      class: n4m.NPLS\n"
                        "      params:\n        n_components: 2\n        mode_j: 3\n"
                        "        mode_k: 4\n", encoding="utf-8")
    config = PipelineConfigs(str(path))
    operator = StepParser().parse(config.steps[0][0]).operator
    expected = ([[1.227494353285351, 0.28947162598321],
                 [2.01114317927755, 0.2706367639092839],
                 [0.5127342674702883, 1.178039833756177]] if multi_target else
                [1.242673188360166, 1.939312398512768, 0.5941411153991755])
    np.testing.assert_allclose(operator.fit(X, target).predict(X_test),
                               expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_mbpls_recipe_matches_r_native_heldout(tmp_path, extension: str) -> None:
    samples = np.arange(1, 22, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    X_test = X[[1, 7, 16], :] + 0.031
    recipe = {"pipeline": [{"model": {"class": "n4m.MBPLS", "params": {
        "n_components": 2, "block_sizes": [4, 4, 4]}}}]}
    path = tmp_path / f"mbpls.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        path.write_text("pipeline:\n  - model:\n      class: n4m.MBPLS\n"
                        "      params:\n        n_components: 2\n"
                        "        block_sizes: [4, 4, 4]\n", encoding="utf-8")
    operator = StepParser().parse(PipelineConfigs(str(path)).steps[0][0]).operator
    np.testing.assert_allclose(operator.fit(X, y).predict(X_test),
                               [1.3614391588922699, 2.033212108151359,
                                0.7661914180346159], rtol=0, atol=1e-10)


@pytest.mark.parametrize("params", [
    {"n_components": 2, "mode_j": 3},
    {"n_components": 2, "mode_j": 3, "mode_k": 0},
    {"n_components": 2, "mode_j": 3.0, "mode_k": 4},
    {"n_components": 2, "mode_j": True, "mode_k": 4},
    {"n_components": 2, "mode_j": 3, "mode_k": 2**31},
])
def test_npls_recipe_requires_bounded_integer_modes(params: dict) -> None:
    with pytest.raises(ValueError, match="mode_j|mode_k"):
        StepParser().parse({"model": {"class": "n4m.NPLS", "params": params}})


def test_portable_json_envelope_reaches_executable_steps() -> None:
    config = PipelineConfigs(
        {"pipeline": [
            {"class": "n4m.LSNV", "params": {"window": 5}},
            {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
        ]}
    )
    parsed = [StepParser().parse(step).operator for step in config.steps[0]]
    assert type(parsed[0]).__name__ == "LSNV"
    assert type(parsed[1]).__name__ == "PLSRegression"
    assert parsed[1].scale_y is True


def test_unknown_portable_model_still_fails_strictly() -> None:
    with pytest.raises(ValueError, match="Could not deserialize component"):
        StepParser().parse({"model": {"class": "n4m.UnknownPortableModel"}})
