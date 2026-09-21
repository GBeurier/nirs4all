"""Controller prediction coverage for deprecated legacy Python bundles."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from nirs4all.pipeline.bundle.loader import BundleLoader


class _ArtifactProvider:
    def __init__(self, *, folds: list[tuple[int, Any]] | None = None, single: Any | None = None):
        self.folds = folds or []
        self.single = single

    def get_fold_artifacts(self, step_idx: int, branch_path: list[int] | None = None) -> list[tuple[int, Any]]:
        return self.folds

    def get_artifacts_for_step(self, step_idx: int, branch_path: list[int] | None = None) -> list[tuple[str, Any]]:
        return [] if self.single is None else [("artifact", self.single)]


def _bare_loader(provider: _ArtifactProvider | None = None) -> BundleLoader:
    loader = BundleLoader.__new__(BundleLoader)
    loader.artifact_provider = provider
    loader.fold_weights = {}
    loader.trace = None
    return loader


@pytest.mark.torch
def test_legacy_bundle_routes_torch_artifact_through_framework_controller() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.copy_(torch.tensor([0.5]))

    X = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    actual = _bare_loader()._predict_legacy_model_artifact(model, X)

    np.testing.assert_allclose(actual, np.asarray([[-0.5], [0.5]], dtype=np.float32))


def test_legacy_bundle_preserves_sklearn_prediction_shape() -> None:
    X = np.asarray([[0.0], [1.0], [2.0]])
    model = LinearRegression().fit(X, np.asarray([1.0, 3.0, 5.0]))

    actual = _bare_loader()._predict_legacy_model_artifact(model, X)

    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, model.predict(X))


@pytest.mark.parametrize("artifact_mode", ["refit", "folds", "single"])
def test_all_legacy_model_artifact_modes_use_controller_dispatch(monkeypatch: pytest.MonkeyPatch, artifact_mode: str) -> None:
    model = object()
    provider = _ArtifactProvider(
        folds=[(0, model), (1, model)] if artifact_mode == "folds" else None,
        single=model if artifact_mode == "single" else None,
    )
    loader = _bare_loader(provider)
    monkeypatch.setattr(loader, "_get_refit_model", lambda step_idx: model if artifact_mode == "refit" else None)
    calls: list[Any] = []

    def predict(artifact: Any, X: np.ndarray) -> np.ndarray:
        calls.append(artifact)
        return np.ones((len(X), 1))

    monkeypatch.setattr(loader, "_predict_legacy_model_artifact", predict)
    actual = loader._predict_model_step(np.zeros((3, 2)), 4)

    assert len(calls) == (2 if artifact_mode == "folds" else 1)
    np.testing.assert_array_equal(actual, np.ones((3, 1)))


@pytest.mark.parametrize("artifact_mode", ["folds", "single"])
def test_all_legacy_meta_model_modes_use_controller_dispatch(monkeypatch: pytest.MonkeyPatch, artifact_mode: str) -> None:
    model = object()
    provider = _ArtifactProvider(
        folds=[(0, model), (1, model)] if artifact_mode == "folds" else None,
        single=model if artifact_mode == "single" else None,
    )
    loader = _bare_loader(provider)
    monkeypatch.setattr(loader, "_predict_model_step", lambda X, step_idx, branch_path: np.ones((len(X), 1)))
    calls: list[Any] = []

    def predict(artifact: Any, X: np.ndarray) -> np.ndarray:
        calls.append(artifact)
        return np.full((len(X), 1), 2.0)

    monkeypatch.setattr(loader, "_predict_legacy_model_artifact", predict)
    meta_step = SimpleNamespace(step_index=5, metadata={"source_models": [{"step_index": 4}]})
    actual = loader._predict_meta_model(np.zeros((3, 2)), meta_step)

    assert len(calls) == (2 if artifact_mode == "folds" else 1)
    np.testing.assert_array_equal(actual, np.full((3, 1), 2.0))
