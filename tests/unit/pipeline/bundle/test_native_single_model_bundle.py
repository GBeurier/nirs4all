"""Native single-model ``.n4a`` bundle writer coverage."""

from __future__ import annotations

import json
import zipfile

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.api.result import _DagmlExportedModel
from nirs4all.pipeline.bundle import BundleLoader, write_single_model_bundle


def test_native_single_model_bundle_round_trips_without_workspace(tmp_path) -> None:
    rng = np.random.default_rng(4)
    x_train = rng.standard_normal((24, 6)).astype(np.float32)
    y_train = rng.standard_normal((24, 1)).astype(np.float32)
    x_test = rng.standard_normal((5, 6)).astype(np.float32)
    pipeline = Pipeline([("scale", StandardScaler()), ("pls", PLSRegression(n_components=2))]).fit(x_train, y_train)
    model = _DagmlExportedModel(pipeline, None)

    bundle = write_single_model_bundle(
        model,
        tmp_path / "native",
        model_label="PLS / exported",
        pipeline_uid="run-native-4",
        provenance={"source_type": "dagml_native", "export_path": "dagml_native"},
    )

    expected = model.predict(x_test)
    actual = BundleLoader(bundle).predict(x_test)
    assert bundle.suffix == ".n4a"
    assert np.array_equal(actual, expected)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source_type"] == "dagml_native"
        assert manifest["model_step_index"] == 1
        assert any(name.startswith("artifacts/step_1_foldfinal_") for name in archive.namelist())
