"""Public DAG-ML parity for the PyTorch models the legacy runner accepts."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytest.importorskip("dag_ml")
torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parity
def test_builtin_customizable_decon_run_export_replay(tmp_path) -> None:
    """A built-in factory trains per fold, refits, and replays its captured model."""
    import nirs4all
    from nirs4all.operators.models.pytorch.nicon import customizable_decon

    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, (12, 128)).astype(np.float32)
    y = rng.uniform(0.1, 0.9, (12, 2)).astype(np.float32)
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=1),
        {"model": customizable_decon, "model_params": {"output_units": 2}, "train_params": {"epochs": 1, "batch_size": 4}},
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - verify captured refit identity
    assert fitted.model_._nirs4all_input_shape == (1, 128)
    expected = np.asarray(fitted.predict(x[:3]))
    assert expected.shape == (3, 2)

    archive = result.export(tmp_path / "decon.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred), expected, rtol=1e-6, atol=1e-6)
    result.close()


@pytest.mark.torch
@pytest.mark.parity
def test_builtin_customizable_decon_multiclass_logits_replay(tmp_path) -> None:
    """Class-index training and archived predictions retain the task contract."""
    import nirs4all
    from nirs4all.operators.models.pytorch.nicon import customizable_decon_classification

    rng = np.random.default_rng(16)
    x = rng.uniform(0, 1, (12, 64)).astype(np.float32)
    y = np.tile(np.arange(3), 4)
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=1),
        {
            "model": customizable_decon_classification,
            "model_params": {"dense_units2": 16},
            "train_params": {"epochs": 1, "batch_size": 4, "loss": "CrossEntropyLoss"},
        },
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - captured refit model
    assert fitted.model_._nirs4all_task_type == "multiclass_classification"
    assert fitted.model_._nirs4all_input_shape == (1, 64)
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)
    assert set(expected).issubset({0, 1, 2})

    archive = result.export(tmp_path / "torch_multiclass.n4a")
    np.testing.assert_array_equal(np.asarray(nirs4all.predict(archive, x[:3]).y_pred).reshape(-1), expected)
    result.close()


@pytest.mark.torch
@pytest.mark.parity
def test_raw_torch_module_keeps_flat_input_and_multiple_regression_outputs(tmp_path) -> None:
    """A supplied module keeps its constructor state and regression output width."""
    import nirs4all

    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 16)).astype(np.float32)
    y = rng.normal(size=(12, 2)).astype(np.float32)
    model = torch.nn.Linear(16, 2)
    pipeline = [KFold(n_splits=2, shuffle=True, random_state=1), {"model": model, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - verify captured refit identity
    assert fitted.input_layout_ == "flat"
    expected = np.asarray(fitted.predict(x[:3]))
    assert expected.shape == (3, 2)

    archive = result.export(tmp_path / "raw_torch.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred), expected)
    result.close()


@pytest.mark.torch
@pytest.mark.parity
def test_raw_torch_conv_module_receives_channel_axis() -> None:
    """A caller-supplied CNN retains the channel layout it used in legacy."""
    import nirs4all

    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 24)).astype(np.float32)
    y = rng.normal(size=(12, 1)).astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 2, kernel_size=3),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(2, 1),
    )
    pipeline = [KFold(n_splits=2, shuffle=True, random_state=1), {"model": model, "train_params": {"epochs": 1, "batch_size": 4}}]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - verify captured refit identity
    assert fitted.input_layout_ == "channels_first"
    assert fitted.model_._nirs4all_input_shape == (1, 24)
    assert np.asarray(fitted.predict(x[:3])).shape == (3, 1)
    result.close()


@pytest.mark.torch
@pytest.mark.parity
def test_builtin_factory_deterministic_finetune_replay(tmp_path) -> None:
    """A legacy model-local grid sets factory parameters on each candidate."""
    import nirs4all
    from nirs4all.operators.models.pytorch.nicon import customizable_decon

    rng = np.random.default_rng(9)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    pipeline = [
        KFold(n_splits=2),
        {
            "model": customizable_decon,
            "train_params": {"epochs": 1, "batch_size": 4},
            "finetune_params": {
                "engine": "dag-ml",
                "approach": "grouped",
                "eval_mode": "mean",
                "model_params": {"filters1": [4, 8]},
            },
        },
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    assert result.num_predictions >= 4  # both candidate architectures emit fold scores
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - selected refit candidate
    assert fitted.factory_params["filters1"] in {4, 8}
    expected = np.asarray(fitted.predict(x[:2]))

    archive = result.export(tmp_path / "decon_finetuned.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected.reshape(-1))
    result.close()


@pytest.mark.torch
@pytest.mark.parity
@pytest.mark.parametrize("approach", ["single", "grouped", "individual"])
def test_builtin_factory_host_optuna_finetune_replay(tmp_path, approach) -> None:
    """The host optimizer selects factory arguments inside DAG training scopes."""
    pytest.importorskip("optuna")
    import nirs4all
    from nirs4all.operators.models.pytorch.nicon import customizable_decon

    rng = np.random.default_rng(10)
    x = rng.uniform(0, 1, (12, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (12, 1)).astype(np.float32)
    pipeline = [
        KFold(n_splits=2),
        {
            "model": customizable_decon,
            "train_params": {"epochs": 1, "batch_size": 4},
            "finetune_params": {
                "engine": "optuna",
                "approach": approach,
                "eval_mode": "best",
                "n_trials": 2,
                "sample": "grid",
                "model_params": {"filters1": [4, 8]},
            },
        },
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - selected refit candidate
    assert fitted.factory_params["filters1"] in {4, 8}
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / f"decon_optuna_{approach}.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected)
    result.close()


@pytest.mark.torch
@pytest.mark.parity
def test_host_optuna_sampled_training_control_is_trial_scoped() -> None:
    """A sampled fit control reaches trials; terminal refit keeps step controls."""
    pytest.importorskip("optuna")
    import nirs4all
    from nirs4all.operators.models.pytorch.nicon import customizable_decon

    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, (10, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (10, 1)).astype(np.float32)
    pipeline = [
        KFold(2),
        {
            "model": customizable_decon,
            "train_params": {"epochs": 1, "batch_size": 4},
            "finetune_params": {
                "engine": "optuna",
                "approach": "single",
                "sample": "grid",
                "n_trials": 1,
                "model_params": {"filters1": [4]},
                "train_params": {"batch_size": [2]},
            },
        },
    ]

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - selected refit candidate
    assert fitted.factory_params["filters1"] == 4
    assert fitted.batch_size == 4
    assert fitted._nirs4all_host_hpo["trials"][0]["effective_model_params"]["batch_size"] == 2  # noqa: SLF001
    assert fitted._nirs4all_host_hpo["effective_selected_model_params"]["batch_size"] == 4  # noqa: SLF001
    result.close()
