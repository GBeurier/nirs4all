"""Predict-only N4MM envelopes for the shared native affine recipe vocabulary."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from nirs4all.pipeline.portable_n4m_trained import PortableN4MTrainedPipeline

pytest.importorskip("pls4all")
pytest.importorskip("n4m")


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 11) + np.cos(samples / 3 + bands / 7) + samples * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    return X, y


_AFFINE_CASES = [
    ("n4m.Ridge", {"alpha": 0.7}),
    ("n4m.RidgePLS", {"n_components": 2, "ridge_lambda": 0.7}),
    ("n4m.RobustPLS", {"n_components": 2, "huber_k": 1.345, "max_irls_iter": 20}),
    ("n4m.CPPLS", {"n_components": 2, "gamma": 0.5}),
    ("n4m.SparseSIMPLS", {"n_components": 2, "sparsity_lambda": 0.05}),
    ("n4m.ECR", {"n_components": 2, "alpha": 0.5}),
    ("n4m.ContinuumRegression", {"n_components": 2, "tau": 0.5}),
    ("n4m.MIRPLS", {"n_components": 2}),
    ("n4m.FusedSparsePLS", {"n_components": 2, "l1_lambda": 0.05, "fusion_lambda": 0.05}),
    ("n4m.BaggingPLS", {"n_components": 2, "n_estimators": 7, "seed": 13}),
    ("n4m.BoostingPLS", {"n_components": 2, "n_estimators": 7, "learning_rate": 0.3}),
    ("n4m.RandomSubspacePLS", {"n_components": 2, "n_estimators": 7,
                               "features_per_subspace": 5, "seed": 13}),
    ("n4m.NPLS", {"n_components": 2, "mode_j": 3, "mode_k": 4}),
]


@pytest.mark.methods
@pytest.mark.parametrize("name,params", _AFFINE_CASES)
def test_affine_v5_predict_roundtrip_and_refit(name: str, params: dict) -> None:
    X, y = _inputs()
    recipe = {"pipeline": [{"model": {"class": name, "params": params}}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        expected = fitted.predict(X[28:])
        document = json.loads(fitted.to_json())
        assert document["schema"] == "nirs4all.n4m.trained_pipeline.v5"
        manifest = json.loads(document["manifest_json"])
        assert manifest["fit_recipe_assertion"] == {"kind": "affine_recipe", "recipe_class": name}
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as restored:
            np.testing.assert_allclose(restored.predict(X[28:]), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(restored.retrain(X[:28], y[:28]).predict(X[28:]),
                                       expected, rtol=0, atol=1e-10)


@pytest.mark.methods
@pytest.mark.parametrize("preprocessing", [
    [{"class": "n4m.MSC"}],
    [{"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}}],
    [{"class": "n4m.Selector", "params": {"method": "wvc_select", "n_components": 2,
                                            "method_params": {"top_k": 5, "normalize": False}}}],
    [{"branch": {"msc": [{"class": "n4m.MSC"}], "snv": [{"class": "n4m.SNV"}]}},
     {"merge": "features"}],
])
def test_affine_v5_replays_external_preprocessing(preprocessing: list[dict]) -> None:
    X, y = _inputs()
    recipe = {"pipeline": [*preprocessing,
                           {"model": {"class": "n4m.BaggingPLS", "params": {
                               "n_components": 2, "n_estimators": 7, "seed": 13}}}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        expected = fitted.predict(X[28:])
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as restored:
            np.testing.assert_allclose(restored.predict(X[28:]), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(restored.retrain(X[:28], y[:28]).predict(X[28:]),
                                       expected, rtol=0, atol=1e-10)


def _rehashed_manifest(document: dict, manifest: dict) -> dict:
    changed = json.loads(json.dumps(document))
    changed["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
    changed["manifest_sha256"] = hashlib.sha256(changed["manifest_json"].encode()).hexdigest()
    return changed


@pytest.mark.methods
def test_affine_v5_rejects_tampered_recipe_state_and_descriptor() -> None:
    X, y = _inputs()
    recipe = {"pipeline": [{"class": "n4m.MSC"},
                           {"model": {"class": "n4m.BaggingPLS", "params": {
                               "n_components": 2, "n_estimators": 7, "seed": 13}}}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        document = json.loads(fitted.to_json())
    manifest = json.loads(document["manifest_json"])

    changed = json.loads(json.dumps(document))
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        PortableN4MTrainedPipeline(changed)
    changed = json.loads(json.dumps(document))
    changed["model"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload hash"):
        PortableN4MTrainedPipeline(changed)
    claimed = json.loads(json.dumps(manifest))
    claimed["fit_recipe_assertion"]["recipe_class"] = "n4m.BoostingPLS"
    with pytest.raises(ValueError, match="assertion"):
        PortableN4MTrainedPipeline(_rehashed_manifest(document, claimed))
    bad_state = json.loads(json.dumps(manifest))
    bad_state["step_states"][0]["reference"] = bad_state["step_states"][0]["reference"][:-1]
    with pytest.raises(ValueError, match="reference"):
        PortableN4MTrainedPipeline(_rehashed_manifest(document, bad_state))
    bad_width = json.loads(json.dumps(manifest))
    bad_width["input_n_features"] += 1
    with pytest.raises(ValueError, match="reference"):
        PortableN4MTrainedPipeline(_rehashed_manifest(document, bad_width))

    old_recipe = {"pipeline": [{"model": {"class": "n4m.PLS", "params": {"n_components": 2}}}]}
    with PortableN4MTrainedPipeline.fit_recipe(old_recipe, X[:28], y[:28]) as old:
        old_model = json.loads(old.to_json())["model"]
    changed = json.loads(json.dumps(document))
    changed["model"] = old_model
    with pytest.raises(ValueError, match="affine descriptor"):
        PortableN4MTrainedPipeline(changed)


@pytest.mark.methods
@pytest.mark.parametrize("name,params", [
    ("n4m.BaggingPLS", {"n_components": 2, "seed": 2**31}),
    ("n4m.BoostingPLS", {"n_components": 2, "learning_rate": 1.2}),
    ("n4m.RandomSubspacePLS", {"n_components": 2, "n_estimators": 3.5}),
    ("n4m.Ridge", {"n_components": 2}),
    ("n4m.NPLS", {"n_components": 2, "mode_j": 3, "mode_k": 5}),
])
def test_affine_v5_rejects_invalid_shared_recipe(name: str, params: dict) -> None:
    X, y = _inputs()
    with pytest.raises((ValueError, TypeError)):
        PortableN4MTrainedPipeline.fit_recipe(
            {"pipeline": [{"model": {"class": name, "params": params}}]}, X[:28], y[:28])


@pytest.mark.methods
def test_npls_v5_rejects_changed_tensor_modes_and_multitarget_fit() -> None:
    X, y = _inputs()
    recipe = {"pipeline": [{"model": {"class": "n4m.NPLS", "params": {
        "n_components": 2, "mode_j": 3, "mode_k": 4}}}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        document = json.loads(fitted.to_json())
    manifest = json.loads(document["manifest_json"])
    manifest["recipe"]["pipeline"][-1]["model"]["params"]["mode_k"] = 5
    with pytest.raises(ValueError, match="tensor modes"):
        PortableN4MTrainedPipeline(_rehashed_manifest(document, manifest))
    with pytest.raises(ValueError, match="aligned training"):
        PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], np.column_stack((y[:28], y[:28])))
