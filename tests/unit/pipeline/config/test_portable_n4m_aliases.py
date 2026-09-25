"""The shared n4m recipe names resolve to Methods-backed Python operators."""

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
