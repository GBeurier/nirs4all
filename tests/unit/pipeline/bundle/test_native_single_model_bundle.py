"""Native single-model ``.n4a`` bundle writer coverage."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.api.result import _DagmlExportedModel
from nirs4all.pipeline.bundle import BundleLoader, write_single_model_bundle
from nirs4all.pipeline.dagml.host_container import BUNDLE_MANIFEST_MEMBER, JOBLIB_CONTENT_TYPE, open_n4a_source
from tests.unit.pipeline.dagml.host_container_helpers import reseal_host_member


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
    with open_n4a_source(bundle) as source:
        container = source.require_host()
        manifest = container.bundle_manifest
        assert manifest is not None
        assert manifest["source_type"] == "dagml_native"
        assert manifest["model_step_index"] == 1
        assert container.namelist()[:2] == [BUNDLE_MANIFEST_MEMBER, "pipeline.json"]
        artifact = next(name for name in container.namelist() if name.startswith("artifacts/step_1_foldfinal_"))
        assert container.artifact(artifact)["content_type"] == {"id": JOBLIB_CONTENT_TYPE, "version": 1}
        assert manifest["artifact_integrity"] == {artifact: container.digest(artifact)}


def test_resealed_model_payload_is_refused_by_the_declared_digest_before_deserialization(tmp_path, monkeypatch) -> None:
    import joblib

    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    rng = np.random.default_rng(5)
    pipeline = Pipeline([("scale", StandardScaler()), ("pls", PLSRegression(n_components=2))]).fit(
        rng.standard_normal((24, 6)).astype(np.float32), rng.standard_normal((24, 1)).astype(np.float32)
    )
    bundle = write_single_model_bundle(
        _DagmlExportedModel(pipeline, None), tmp_path / "native", model_label="PLS", provenance={"source_type": "dagml_native", "export_path": "dagml_native"}
    )
    with open_n4a_source(bundle) as source:
        member = next(name for name in source.require_host().namelist() if name.startswith("artifacts/"))
    # A fully valid container (Core accepts it) whose model bytes differ from the manifest's recorded digest.
    resealed = reseal_host_member(bundle, tmp_path / "resealed.n4a", member, b"different model bytes")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("a changed model payload was deserialized"))

    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_general_archive(resealed)
