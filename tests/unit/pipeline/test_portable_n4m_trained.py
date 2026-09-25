"""Native trained-pipeline envelope tests independent of an R runtime."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nirs4all.pipeline.portable_n4m_trained import PortableN4MTrainedPipeline


@pytest.fixture
def native_inputs() -> tuple[np.ndarray, np.ndarray]:
    pytest.importorskip("pls4all")
    pytest.importorskip("n4m")
    X = np.fromfunction(
        lambda i, j: np.sin((i + 1) * (j + 1) / 7) + (i + 1) * (j + 1) / 50,
        (24, 8), dtype=float,
    )
    return X, X[:, 1] - 0.3 * X[:, 4]


@pytest.mark.methods
@pytest.mark.parametrize("preprocessing", [
    [],
    [{"class": "n4m.SNV"}, {"class": "n4m.Detrend", "params": {"polyorder": 1}}],
    [{"class": "n4m.MSC"}, {"class": "n4m.EMSC", "params": {"degree": 2}}],
    [{"branch": {
        "msc": [{"class": "n4m.MSC"}],
        "emsc": [{"class": "n4m.EMSC", "params": {"degree": 2}}],
    }}, {"merge": "features"}],
])
def test_native_trained_pipeline_roundtrip(
    native_inputs: tuple[np.ndarray, np.ndarray],
    preprocessing: list[dict[str, object]],
) -> None:
    X, y = native_inputs
    recipe = {"name": "native", "pipeline": [
        *preprocessing,
        {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:17], y[:17]) as fitted:
        expected = fitted.predict(X[17:])
        text = fitted.to_json()
        with PortableN4MTrainedPipeline.from_json(text) as restored:
            np.testing.assert_allclose(restored.predict(X[17:]), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(restored.retrain(X[:17], y[:17]).predict(X[17:]),
                                       expected, rtol=0, atol=1e-8)
        document = json.loads(text)
        document["manifest_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="manifest hash"):
            PortableN4MTrainedPipeline(document)
        document = json.loads(text)
        document["model"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="payload hash"):
            PortableN4MTrainedPipeline(document)


@pytest.mark.methods
def test_native_trained_pipeline_refuses_mismatched_width(
    native_inputs: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = native_inputs
    recipe = {"pipeline": [
        {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:17], y[:17]) as fitted:
        with pytest.raises(ValueError, match="samples-by-features"):
            fitted.predict(X[17:, :-1])
        with pytest.raises(ValueError, match="aligned training"):
            fitted.retrain(X[:17], y[:16])
