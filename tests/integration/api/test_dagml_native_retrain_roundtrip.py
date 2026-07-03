"""Strict dag-ml public-API roundtrip: run(results_path) → export() → predict() → retrain(mode="full").

RC-D runtime gate evidence (2026-07-02). The V1 native persistence surface is ``run(results_path=...)``
(NOT the legacy ``workspace_path`` workspace, which the strict dag-ml engine refuses by design). This file
pins the verbs the production flip needs on that supported native path, under the DEFAULT engine
(no ``$N4A_ENGINE`` override, ``allow_fallback`` default ``False``):

* run() captures native results (manifest + score_set + predictions.parquet + refit artifact);
* export() builds the native ``.n4a`` from the captured refit artifact — including the ADDITIVE
  ``train_pipeline.json`` replayable training spec (fully-qualified classes + params) for a concrete
  (non-generator) pipeline;
* predict() from that bundle returns finite values of the right shape;
* retrain(mode="full") from that bundle RE-TRAINS the ORIGINAL pipeline structure on new data — the
  regression this file exists for: without ``train_pipeline.json`` the bundle's cosmetic
  ``{"model": {"class": "<label>"}}`` step is not deserializable and retrain crashed with
  "Could not deserialize component".

No exception swallowing: any strict-mode refusal (RtError) or replay failure fails the test.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import nirs4all


@pytest.fixture
def regression_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(0.5, 0.1, size=(80, 50)).astype(np.float64)
    y = x[:, :5].sum(axis=1) + rng.normal(0, 0.05, size=80)
    return x, y


def _pipeline() -> list:
    return [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=0),
        {"model": PLSRegression(n_components=3)},
    ]


def test_native_results_run_export_predict_retrain_roundtrip(regression_xy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = regression_xy

    result = nirs4all.run(pipeline=_pipeline(), dataset=(x, y), verbose=0, results_path=tmp_path, random_state=0)
    assert result.num_predictions > 0

    bundle_path = tmp_path / "model.n4a"
    result.export(bundle_path)
    assert bundle_path.exists()

    # The native bundle carries BOTH the predict artifact and the replayable training spec.
    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert any(n.startswith("artifacts/step_1_foldfinal_") for n in names)
        train_cfg = json.loads(zf.read("train_pipeline.json"))
    steps = train_cfg["steps"]
    assert len(steps) == 3, "the ORIGINAL steps (scaler + splitter + model), not the cosmetic model label"
    assert steps[2]["model"]["class"].endswith("PLSRegression")
    assert steps[2]["model"]["params"] == {"n_components": 3}

    pred = nirs4all.predict(model=str(bundle_path), data=x[:10], verbose=0)
    assert len(pred.y_pred) == 10
    assert np.all(np.isfinite(pred.y_pred))

    captured_explain: dict[str, object] = {}

    class _FakeShapAnalyzer:
        def explain_model(self, **kwargs):
            model = kwargs["model"]
            x_data = kwargs["X"]
            captured_explain["model_class"] = type(model).__name__
            captured_explain["x_shape"] = x_data.shape
            return {
                "shap_values": np.zeros_like(x_data, dtype=float),
                "feature_names": kwargs.get("feature_names"),
                "expected_value": 0.0,
                "explainer_type": kwargs.get("explainer_type", "auto"),
            }

    from nirs4all.visualization.analysis import shap as shap_module

    monkeypatch.setattr(shap_module, "ShapAnalyzer", _FakeShapAnalyzer)

    explanation = nirs4all.explain(
        model=str(bundle_path),
        data=x[:8],
        verbose=0,
        plots_visible=False,
        n_samples=5,
    )
    assert explanation is not None
    assert explanation.n_samples == 8
    assert captured_explain == {
        "model_class": "_DagmlExportedModel",
        "x_shape": (8, 50),
    }

    retrained = nirs4all.retrain(
        source=str(bundle_path),
        data=(x[40:], y[40:]),
        mode="full",
        verbose=0,
        save_artifacts=False,
    )
    assert isinstance(retrained, nirs4all.RunResult)
    assert retrained.num_predictions > 0
    validation = retrained.validate(raise_on_failure=False)
    assert validation["nan_count"] == 0, f"retrain() produced NaN scores: {validation['issues']}"


def test_generator_pipeline_native_bundle_stays_predict_only(regression_xy, tmp_path: Path) -> None:
    """A GENERATOR run exports winner-only; its bundle must NOT carry a training spec (retraining the
    frozen pipeline would re-run the WHOLE sweep, not the exported winner)."""
    x, y = regression_xy
    sweep = [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=0),
        {"model": {"_or_": [PLSRegression(n_components=2), PLSRegression(n_components=4)]}},
    ]

    result = nirs4all.run(pipeline=sweep, dataset=(x, y), verbose=0, results_path=tmp_path, random_state=0)
    bundle_path = tmp_path / "winner.n4a"
    result.export(bundle_path)

    with zipfile.ZipFile(bundle_path) as zf:
        assert "train_pipeline.json" not in zf.namelist()

    pred = nirs4all.predict(model=str(bundle_path), data=x[:5], verbose=0)
    assert np.all(np.isfinite(pred.y_pred))
