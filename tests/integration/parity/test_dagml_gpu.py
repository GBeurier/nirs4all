"""Real CUDA qualification for the DAG-ML host execution path."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

import nirs4all

pytestmark = pytest.mark.gpu


class _TorchCudaRegressor(RegressorMixin, BaseEstimator):
    """Tiny importable estimator whose fit performs real linear algebra on CUDA."""

    framework = "pytorch"
    observed_devices: set[str] = set()

    def fit(self, X: np.ndarray, y: np.ndarray) -> _TorchCudaRegressor:
        import torch

        x_tensor = torch.as_tensor(X, dtype=torch.float32, device="cuda")
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device="cuda").reshape(-1, 1)
        ones = torch.ones((x_tensor.shape[0], 1), dtype=x_tensor.dtype, device=x_tensor.device)
        design = torch.cat((x_tensor, ones), dim=1)
        solution = torch.linalg.lstsq(design, y_tensor).solution
        self.coef_ = solution[:-1, 0].cpu().numpy()
        self.intercept_ = float(solution[-1, 0].cpu())
        type(self).observed_devices.add(str(x_tensor.device))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.coef_ + self.intercept_


def test_public_dagml_run_executes_model_fit_on_requested_cuda_device() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("optional GPU environment unavailable: requires a CUDA-capable PyTorch runtime")
    rng = np.random.default_rng(71)
    X = rng.normal(size=(36, 10)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.2).astype(np.float32)
    _TorchCudaRegressor.observed_devices.clear()
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)

    result = nirs4all.run(
        [KFold(n_splits=3, shuffle=False), _TorchCudaRegressor()],
        (X, y),
        engine="dag-ml",
        gpu_devices=["cuda:0"],
        cpu_threads=2,
        save_artifacts=False,
        save_charts=False,
        verbose=0,
    )
    try:
        assert result.execution_engine == "dag-ml"
        assert result.cv_best_score < 1.0e-3
        assert _TorchCudaRegressor.observed_devices == {"cuda:0"}
    finally:
        result.close()
