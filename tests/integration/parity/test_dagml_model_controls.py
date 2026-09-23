"""Public legacy/DAG-ML parity for neural model layout and TensorFlow controls."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytest.importorskip("dag_ml")
try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class FlatOnlyModule(torch.nn.Module):
        framework = "pytorch"

        def __init__(self) -> None:
            super().__init__()
            # A Conv1d child makes the default PyTorch layout 3D. The model's
            # actual forward contract intentionally requires the 2D override.
            self.conv = torch.nn.Conv1d(1, 1, 1)
            self.linear = torch.nn.Linear(16, 1)
            self.last_shape: tuple[int, ...] | None = None

        def forward(self, features):
            self.last_shape = tuple(features.shape[1:])
            if features.ndim != 2:
                raise ValueError("this model requires force_layout='2d'")
            return self.linear(features)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.torch
@pytest.mark.parity
def test_torch_force_layout_2d_matches_legacy_and_replays(tmp_path, monkeypatch, mechanism: str) -> None:
    pytest.importorskip("torch")
    import nirs4all

    rng = np.random.default_rng(23)
    x = rng.normal(size=(10, 16)).astype(np.float32)
    y = rng.normal(size=(10, 1)).astype(np.float32)
    pipeline = [KFold(2), {"model": FlatOnlyModule(), "force_layout": "2d", "train_params": {"epochs": 1, "batch_size": 4}}]

    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - inspect refit layout
    assert fitted.model_.last_shape == (16,)
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "force_layout_2d.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected)
    result.close()


@pytest.mark.tensorflow
@pytest.mark.parity
def test_tensorflow_nested_compile_fit_controls_match_legacy_and_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    pytest.importorskip("tensorflow")
    import nirs4all
    from nirs4all.operators.models.tensorflow.nicon import customizable_decon

    rng = np.random.default_rng(24)
    x = rng.uniform(0, 1, (8, 64)).astype(np.float32)
    y = rng.uniform(0, 1, (8, 1)).astype(np.float32)
    pipeline = [
        KFold(2),
        {"model": customizable_decon, "train_params": {"compile": {"loss": "mae"}, "fit": {"epochs": 1, "batch_size": 4, "verbose": 0}}},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - inspect refit model
    assert fitted.model_.loss == "mae"
    assert len(fitted.model_.history.epoch) == 1
    expected = np.asarray(fitted.predict(x[:2])).reshape(-1)
    archive = result.export(tmp_path / "tensorflow_nested_controls.n4a")
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:2]).y_pred).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    result.close()
