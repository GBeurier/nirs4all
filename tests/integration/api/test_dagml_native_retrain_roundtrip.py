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
* explain() keeps exercising the explicit Python SHAP rollback lane until an
  API-005 native/plugin contract is wired (never an implicit fallback);
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
import nirs4all.pipeline as pipeline_module
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.dagml.rt import RtError


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
        engine="legacy",
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

    def _never_runner(*args, **kwargs):
        raise AssertionError(f"native retrain constructed PipelineRunner: {args!r} {kwargs!r}")

    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    retrain_results_path = tmp_path / "retrained-results"
    retrained = nirs4all.retrain(
        source=str(bundle_path),
        data=(x[40:], y[40:]),
        mode="full",
        verbose=0,
        save_artifacts=False,
        results_path=retrain_results_path,
    )
    assert isinstance(retrained, nirs4all.RunResult)
    assert retrained.num_predictions > 0
    assert {entry["engine"] for entry in retrained.per_dataset.values()} == {"dag-ml"}
    assert retrained._dagml_refit_artifacts
    assert Path(retrained._dagml_results_dir).is_dir()
    assert Path(retrained._dagml_results_dir) != Path(result._dagml_results_dir)

    lineage = retrained._retrain_lineage
    assert lineage == next(iter(retrained.per_dataset.values()))["retrain_lineage"]
    assert lineage["operation"] == "retrain"
    assert lineage["mode"] == "full"
    assert lineage["engine"] == "dag-ml"
    assert lineage["source_bundle"] == bundle_path.name
    assert len(lineage["source_bundle_sha256"]) == 64
    assert len(lineage["source_training_spec_sha256"]) == 64
    assert lineage["new_artifact_count"] == len(retrained._dagml_refit_artifacts)

    # The run writer completed before retrain could attach source lineage; do
    # not backpatch or overclaim durability in that immutable result record.
    native_manifest = json.loads(
        (Path(retrained._dagml_results_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert "retrain_lineage" not in native_manifest

    # The existing public bundle export provenance contract persists lineage
    # alongside the newly fitted native artifact. BundleLoader validates and
    # exposes it, and the exported artifact remains predict-capable.
    retrained_bundle = tmp_path / "retrained-model.n4a"
    retrained.export(retrained_bundle)
    loaded_retrained = BundleLoader(retrained_bundle)
    assert loaded_retrained.metadata.retrain_lineage == lineage
    retrained_prediction = nirs4all.predict(retrained_bundle, x[:5], verbose=0)
    assert np.all(np.isfinite(retrained_prediction.y_pred))

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

    with pytest.raises(RtError) as caught:
        nirs4all.retrain(bundle_path, (x, y), verbose=0)
    assert caught.value.unsupported_capability == "dagml_full_retrain_training_spec"
