"""Public DAG-ML training and archive replay for built-in neural factories."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytest.importorskip("dag_ml")


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_customizable_decon_run_export_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    pytest.importorskip("tensorflow")

    import nirs4all
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, (12, 128)).astype(np.float32)
    y = rng.uniform(0.1, 0.9, (12, 1)).astype(np.float32)
    pipeline = [KFold(n_splits=2, shuffle=True, random_state=1), {"model": customizable_decon, "model_params": {"dense_units2": 16}, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit identity
    assert fitted._features(x[:3]).shape == (3, 128, 1)  # noqa: SLF001 - model input contract
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)

    archive = result.export(tmp_path / "tensorflow_decon.n4a")
    np.testing.assert_allclose(nirs4all.predict(archive, x[:3]).y_pred, expected, rtol=1e-6, atol=1e-6)
    result.close()


@pytest.mark.jax
@pytest.mark.parity
def test_jax_customizable_decon_run_export_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    import nirs4all
    from nirs4all.operators.models.jax.nicon import customizable_decon

    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0.1, 0.9, (8, 1)).astype(np.float32)
    pipeline = [KFold(n_splits=2, shuffle=True, random_state=1), {"model": customizable_decon, "model_params": {"dense_units2": 16}, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit identity
    assert fitted._features(x[:3]).shape == (3, 64, 1)  # noqa: SLF001 - model input contract
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)

    archive = result.export(tmp_path / "jax_decon.n4a")
    np.testing.assert_allclose(nirs4all.predict(archive, x[:3]).y_pred, expected, rtol=1e-6, atol=1e-6)
    result.close()


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_customizable_decon_multiclass_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    pytest.importorskip("tensorflow")

    import nirs4all
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon_classification

    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, (12, 64)).astype(np.float32)
    y = np.tile(np.arange(3), 4)
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=1),
        {
            "model": customizable_decon_classification,
            "model_params": {"dense_units2": 16},
            "train_params": {
                "epochs": 1,
                "batch_size": 6,
                "loss": "sparse_categorical_crossentropy",
                "metrics": ["accuracy"],
            },
        },
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit identity
    assert fitted.task_type == "multiclass_classification"
    assert fitted.num_classes == 3
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)
    assert set(expected).issubset({0, 1, 2})

    archive = result.export(tmp_path / "tensorflow_classification.n4a")
    np.testing.assert_array_equal(nirs4all.predict(archive, x[:3]).y_pred, expected)
    result.close()


@pytest.mark.jax
@pytest.mark.parity
def test_jax_customizable_decon_multiclass_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    import nirs4all
    from nirs4all.operators.models.jax.nicon import customizable_decon_classification

    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, (12, 64)).astype(np.float32)
    y = np.tile(np.arange(3), 4)
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=1),
        {"model": customizable_decon_classification, "model_params": {"dense_units2": 16}, "train_params": {"epochs": 1, "batch_size": 6}},
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit identity
    assert fitted.task_type == "multiclass_classification"
    assert fitted.num_classes == 3
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)
    assert set(expected).issubset({0, 1, 2})

    archive = result.export(tmp_path / "jax_classification.n4a")
    np.testing.assert_array_equal(nirs4all.predict(archive, x[:3]).y_pred, expected)
    result.close()


@pytest.mark.tensorflow
@pytest.mark.parity
def test_raw_tensorflow_model_export_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    tf = pytest.importorskip("tensorflow")
    import nirs4all

    rng = np.random.default_rng(13)
    x = rng.uniform(0, 1, (8, 32)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    model = tf.keras.Sequential([tf.keras.layers.Input((32, 1)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)])
    pipeline = [KFold(2), {"model": model, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit model
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "tensorflow_raw.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    result.close()


@pytest.mark.jax
@pytest.mark.parity
def test_raw_jax_model_export_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    pytest.importorskip("jax")
    nn = pytest.importorskip("flax.linen")
    import nirs4all
    from nirs4all.operators.models.jax.nicon import DynamicModel, Flatten

    rng = np.random.default_rng(14)
    x = rng.uniform(0, 1, (8, 32)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    model = DynamicModel(layers=[Flatten(), nn.Dense(features=1)])
    pipeline = [KFold(2), {"model": model, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit model
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "jax_raw.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    result.close()
