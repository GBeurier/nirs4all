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


@pytest.mark.methods
@pytest.mark.parametrize("preprocessing", [[], [{"class": "n4m.SNV"}], [{"class": "n4m.MSC"}]])
def test_sparse_plsda_trained_envelope_roundtrip(
    preprocessing: list[dict[str, object]],
) -> None:
    from sklearn.datasets import load_iris

    iris = load_iris()
    rows = np.asarray([*range(12), *range(50, 62), *range(100, 112)])
    held = np.asarray([*range(12, 15), *range(62, 65), *range(112, 115)])
    X = np.asarray(iris.data, dtype=np.float64)
    labels = np.asarray(iris.target_names[iris.target], dtype=str)
    recipe = {"pipeline": [
        *preprocessing,
        {"model": {"class": "n4m.SparsePLSDA", "params": {
            "n_components": 2, "sparsity_lambda": 0.05,
        }}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[rows], labels[rows]) as fitted:
        assert fitted.task == "classification"
        assert fitted.classes == iris.target_names.tolist()
        expected = fitted.predict(X[held])
        scores = fitted.predict_scores(X[held])
        assert scores.shape == (len(held), 3)
        proba = fitted.predict_proba(X[held])
        np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=0, atol=1e-12)
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as loaded:
            np.testing.assert_array_equal(loaded.predict(X[held]), expected)
            np.testing.assert_allclose(loaded.predict_scores(X[held]), scores, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(loaded.retrain(X[rows], labels[rows]).predict(X[held]), expected)
            with pytest.raises(ValueError, match="classes differ"):
                loaded.retrain(X[rows], np.where(labels[rows] == "setosa", "unknown", labels[rows]))
        document = json.loads(fitted.to_json())
        manifest = json.loads(document["manifest_json"])
        manifest["classes"] = manifest["classes"][:-1]
        document["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
        import hashlib
        document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
        with pytest.raises(ValueError, match="descriptor"):
            PortableN4MTrainedPipeline(document)
