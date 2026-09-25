"""The shared n4m recipe names resolve to Methods-backed Python operators."""

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
