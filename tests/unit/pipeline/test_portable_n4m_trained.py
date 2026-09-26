"""Native trained-pipeline envelope tests independent of an R runtime."""

from __future__ import annotations

import base64
import hashlib
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
    pytest.importorskip("pls4all")
    pytest.importorskip("n4m")
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


@pytest.mark.methods
@pytest.mark.parametrize("preprocessing", [
    [{"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}}],
    [{"class": "n4m.SNV"},
     {"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}},
     {"class": "n4m.MSC"}],
    [{"branch": {
        "selected": [{"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}}],
        "original": [{"class": "n4m.MSC"}],
    }}, {"merge": "features"}],
])
def test_spa_trained_envelope_uses_fitted_train_only_selection(
    preprocessing: list[dict[str, object]],
) -> None:
    pytest.importorskip("pls4all")
    pytest.importorskip("n4m.feature_selection.wrapper")
    samples = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 11) + np.cos(samples / 3 + bands / 7) + samples * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [*preprocessing, {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        expected = fitted.predict(X[28:])
        document = json.loads(fitted.to_json())
        assert document["schema"] == "nirs4all.n4m.trained_pipeline.v3"
        manifest = json.loads(document["manifest_json"])
        state = manifest["step_states"][0 if "branch" in preprocessing[0] else next(
            index for index, node in enumerate(preprocessing) if node.get("class") == "n4m.SPA")]
        if state["kind"] == "branch":
            state = state["branches"]["selected"][0]
        selected = state["selected_indices"]
        assert state["kind"] == "selector"
        assert len(selected) == 5 and len(set(selected)) == 5
        if "branch" in preprocessing[0]:
            from pls4all import inspect_n4mm

            descriptor = inspect_n4mm(base64.b64decode(document["model"]["payload"]))
            assert descriptor.n_features == 5 + X.shape[1]
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as loaded:
            np.testing.assert_allclose(loaded.predict(X[28:]), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(loaded.retrain(X[:28], y[:28]).predict(X[28:]),
                                       expected, rtol=0, atol=1e-8)


@pytest.mark.methods
def test_spa_trained_envelope_refuses_invalid_selector_state_and_old_schema() -> None:
    pytest.importorskip("pls4all")
    pytest.importorskip("n4m.feature_selection.wrapper")
    samples = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 11) + np.cos(samples / 3 + bands / 7) + samples * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [
        {"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}},
        {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        baseline = json.loads(fitted.to_json())
    for invalid in ([0, 0, 1, 2, 3], [0, 1, 2, 3, 12], [0, 1, 2, 3], [True, 1, 2, 3, 4]):
        document = json.loads(json.dumps(baseline))
        manifest = json.loads(document["manifest_json"])
        manifest["step_states"][0]["selected_indices"] = invalid
        document["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
        document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
        with pytest.raises(ValueError, match="SPA selected_indices"):
            PortableN4MTrainedPipeline(document)
    baseline["schema"] = "nirs4all.n4m.trained_pipeline.v1"
    with pytest.raises(ValueError, match="v3 envelope"):
        PortableN4MTrainedPipeline(baseline)


@pytest.mark.methods
def test_spa_trained_envelope_refuses_malformed_branch_before_config_resolution() -> None:
    pytest.importorskip("pls4all")
    pytest.importorskip("n4m.feature_selection.wrapper")
    samples = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 11) + np.cos(samples / 3 + bands / 7) + samples * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [
        {"branch": {
            "selected": [{"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}}],
            "original": [{"class": "n4m.MSC"}],
        }},
        {"merge": "features"},
        {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        document = json.loads(fitted.to_json())
    manifest = json.loads(document["manifest_json"])
    manifest["recipe"]["pipeline"][0]["branch"] = []
    document["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
    document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
    with pytest.raises(ValueError, match="invalid feature branch"):
        PortableN4MTrainedPipeline(document)


@pytest.mark.methods
@pytest.mark.parametrize("preprocessing", [
    [{"class": "n4m.Selector", "params": {
        "method": "wvc_select", "n_components": 2,
        "method_params": {"top_k": 5, "normalize": False},
    }}],
    [{"class": "n4m.SNV"}, {"class": "n4m.Selector", "params": {
        "method": "interval_select", "n_components": 2,
        "method_params": {"interval_width": 3, "step": 1},
    }}],
    [{"branch": {
        "selected": [{"class": "n4m.Selector", "params": {
            "method": "wvc_select", "n_components": 2,
            "method_params": {"top_k": 5, "normalize": False},
        }}],
        "original": [{"class": "n4m.SNV"}],
    }}, {"merge": "features"}],
])
def test_generic_selector_v4_trained_envelope_replays_native_selection(
    preprocessing: list[dict[str, object]],
) -> None:
    pytest.importorskip("n4m.feature_selection.generic")
    rows = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(rows * bands / 11) + np.cos(rows / 3 + bands / 7) + rows * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [*preprocessing, {"model": {
        "class": "n4m.PLS", "params": {"n_components": 2},
    }}]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        baseline = fitted.predict(X[28:])
        document = json.loads(fitted.to_json())
        assert document["schema"] == "nirs4all.n4m.trained_pipeline.v4"
        manifest = json.loads(document["manifest_json"])
        state = manifest["step_states"][0 if "branch" in preprocessing[0] else
                                        next(i for i, node in enumerate(preprocessing)
                                             if node.get("class") == "n4m.Selector")]
        if state["kind"] == "branch":
            state = state["branches"]["selected"][0]
        assert state["kind"] == "selector"
        assert len(state["selected_indices"]) == len(set(state["selected_indices"]))
        assert all(0 <= index < X.shape[1] for index in state["selected_indices"])
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as loaded:
            np.testing.assert_allclose(loaded.predict(X[28:]), baseline, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                loaded.retrain(X[:28], y[:28]).predict(X[28:]),
                baseline, rtol=0, atol=1e-8,
            )


@pytest.mark.methods
def test_generic_selector_v4_refuses_old_schema_and_malformed_state() -> None:
    pytest.importorskip("n4m.feature_selection.generic")
    rows = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(rows * bands / 11) + np.cos(rows / 3 + bands / 7) + rows * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [
        {"class": "n4m.Selector", "params": {
            "method": "wvc_select", "n_components": 2,
            "method_params": {"top_k": 5, "normalize": False},
        }},
        {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
        baseline = json.loads(fitted.to_json())
    for schema in ("v1", "v2", "v3"):
        document = json.loads(json.dumps(baseline))
        document["schema"] = f"nirs4all.n4m.trained_pipeline.{schema}"
        with pytest.raises(ValueError):
            PortableN4MTrainedPipeline(document)
    for invalid in ([], [0, 0], [12], [True, 1]):
        document = json.loads(json.dumps(baseline))
        manifest = json.loads(document["manifest_json"])
        manifest["step_states"][0]["selected_indices"] = invalid
        document["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
        document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
        with pytest.raises(ValueError, match="generic selected_indices"):
            PortableN4MTrainedPipeline(document)


@pytest.mark.methods
def test_generic_selector_v4_replays_all_native_selector_families() -> None:
    from n4m.feature_selection import SELECTOR_METHODS

    cases = {
        "spa_select": {"top_k": 5},
        "cars_select": {"n_iterations": 8},
        "interval_select": {"interval_width": 3},
        "stability_select": {"top_k": 5},
        "uve_select": {"noise_seed": 7},
        "random_frog_select": {"top_k": 5, "seed": 7, "n_iterations": 8,
                               "initial_size": 6},
        "scars_select": {"seed": 7, "n_iterations": 8},
        "ga_select": {"seed": 7, "n_generations": 5, "population_size": 8},
        "pso_select": {"seed": 7, "n_iterations": 5, "n_swarm": 8},
        "vissa_select": {"seed": 7, "n_iterations": 3, "n_submodels": 8},
        "shaving_select": {},
        "bve_select": {"n_steps": 3, "min_features": 3},
        "t2_select": {"alpha_thresholds": [0.1, 0.3, 0.5]},
        "wvc_select": {"top_k": 5},
        "wvc_threshold_select": {},
        "emcuve_select": {"noise_seed": 7, "n_ensembles": 3},
        "randomization_select": {"randomization_seed": 7},
        "bipls_select": {"interval_width": 3},
        "sipls_select": {"interval_width": 3},
        "rep_select": {},
        "ipw_select": {"top_k": 5},
        "st_select": {"thresholds": [0.1, 0.5, 1.0]},
        "iriv_select": {"seed": 7, "max_rounds": 4},
        "irf_select": {"top_k": 3, "seed": 7, "n_iterations": 8,
                       "window_size": 3, "initial_intervals": 3},
        "vip_spa_select": {"top_k": 5},
    }
    assert set(cases) == set(SELECTOR_METHODS)
    rows = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(rows * bands / 11) + np.cos(rows / 3 + bands / 7) + rows * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    for method, method_params in cases.items():
        recipe = {"pipeline": [
            {"class": "n4m.Selector", "params": {
                "method": method, "n_components": 2,
                "method_params": method_params,
            }},
            {"model": {"class": "n4m.PLS", "params": {"n_components": 1}}},
        ]}
        with PortableN4MTrainedPipeline.fit_recipe(recipe, X[:28], y[:28]) as fitted:
            with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as restored:
                np.testing.assert_allclose(
                    restored.predict(X[28:]), fitted.predict(X[28:]),
                    rtol=0, atol=1e-12, err_msg=f"v4 replay differs for {method}",
                )
