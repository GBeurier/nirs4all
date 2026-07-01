"""Unit tests for :func:`write_single_model_bundle` — the native single-model ``.n4a`` writer (W13, P3).

These tests are engine-agnostic (no dag-ml run, no CLI): they package an in-hand predict-capable model
into a ``.n4a`` and prove an UNMODIFIED :class:`BundleLoader` reload-predicts it EXACTLY. This pins the
contract the dag-ml NATIVE ``.n4a`` export relies on — the bundle-format counterpart of ``joblib.dump``,
built WITHOUT a legacy refit — independent of the dag-ml backend:

* (1) a wrapped fitted sklearn ``Pipeline`` → bundle reload-predict == ``model.predict`` (bit-exact), and
  the manifest carries the additive provenance (``model_step_index == 1``, ``export_path`` marker).
* (2) a wrapper carrying a fitted y-inverse → the bundle reload-predict is in the ORIGINAL target space
  (the inverse is applied, not a no-op), still bit-exact vs ``model.predict``.
* (3) the ``.n4a`` suffix is enforced when the caller passes a bare path.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.api.result import _DagmlExportedModel
from nirs4all.pipeline.bundle import BundleLoader, write_single_model_bundle


def _fit_pipeline(seed: int = 0) -> tuple[Pipeline, np.ndarray]:
    """A tiny fitted (X-scaler + PLS) pipeline and a held-out test X, both at float32 (SpectroDataset dtype)."""
    rng = np.random.default_rng(seed)
    x_fit = rng.standard_normal((40, 12)).astype(np.float32)
    y_fit = rng.standard_normal((40, 1)).astype(np.float32)
    pipe = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=3))])
    pipe.fit(x_fit, y_fit)
    x_test = rng.standard_normal((7, 12)).astype(np.float32)
    return pipe, x_test


def test_native_bundle_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    """A wrapped fitted Pipeline → the bundle reload-predict reproduces ``model.predict`` bit-exactly, and
    the manifest carries the native provenance + a single model step."""
    pipe, x_test = _fit_pipeline(seed=1)
    model = _DagmlExportedModel(pipe, None)  # the exact wrapper the native export path bundles

    out = write_single_model_bundle(
        model,
        tmp_path / "native_model.n4a",
        model_label="PLSRegression",
        pipeline_uid="run-123",
        provenance={"source_type": "dagml_native", "export_path": "dagml_native", "dagml_run_id": "run-123"},
    )
    assert out.exists() and out.suffix == ".n4a"

    reloaded = BundleLoader(out).predict(x_test)
    expected = model.predict(x_test)
    assert reloaded.shape == expected.shape
    # Bit-exact: the same fitted object crosses the joblib round-trip and the loader calls .predict directly.
    assert np.array_equal(reloaded, expected), "bundle reload-predict IS the wrapped model's predict"

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        pipeline_cfg = json.loads(zf.read("pipeline.json"))
    assert "manifest.json" in names and "pipeline.json" in names
    assert any(n.startswith("artifacts/step_1_foldfinal_") for n in names), "single refit-model artifact"
    assert manifest["bundle_format_version"] == "1.0"
    assert manifest["model_step_index"] == 1
    assert manifest["source_type"] == "dagml_native"
    assert manifest["export_path"] == "dagml_native"
    assert manifest["dagml_run_id"] == "run-123"
    assert pipeline_cfg["model_step_index"] == 1
    assert pipeline_cfg["steps"] == [{"model": {"class": "PLSRegression"}}]


def test_native_bundle_applies_y_inverse(tmp_path: Path) -> None:
    """A wrapper with a fitted y-inverse → the bundle reload-predict is in the original target space (inverse
    applied), bit-exact vs the wrapper; the raw estimator (scaled space) must NOT match (inverse matters)."""
    pipe, x_test = _fit_pipeline(seed=2)
    rng = np.random.default_rng(99)
    y_scaler = MinMaxScaler().fit(rng.standard_normal((40, 1)).astype(np.float32))
    model = _DagmlExportedModel(pipe, y_scaler)

    out = write_single_model_bundle(model, tmp_path / "yinv.n4a", model_label="PLSRegression")

    reloaded = BundleLoader(out).predict(x_test)
    expected = model.predict(x_test)
    assert np.array_equal(reloaded, expected), "the y inverse round-trips through the bundle"

    raw = np.asarray(pipe.predict(x_test), dtype=float).ravel()
    assert not np.allclose(raw, np.asarray(reloaded, dtype=float).ravel(), atol=1e-6), "inverse is not a no-op"


def test_native_bundle_enforces_n4a_suffix(tmp_path: Path) -> None:
    """A bare (suffixless) output path gets the ``.n4a`` suffix, and the result still loads + predicts."""
    pipe, x_test = _fit_pipeline(seed=3)
    model = _DagmlExportedModel(pipe, None)

    out = write_single_model_bundle(model, tmp_path / "bare_name", model_label="PLSRegression")
    assert out.suffix == ".n4a" and out.exists()
    assert BundleLoader(out).predict(x_test).shape[0] == x_test.shape[0]
