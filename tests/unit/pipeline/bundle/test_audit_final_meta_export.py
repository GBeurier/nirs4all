"""Final legacy meta artifacts survive cleanup; incomplete exports fail atomically."""

import hashlib
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.bundle import BundleGenerator, BundleLoader
from nirs4all.pipeline.config.context import MapArtifactProvider
from nirs4all.pipeline.resolver import ResolvedPrediction
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _cohort():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(36, 6))
    y = x @ np.arange(1., 7.) + rng.normal(scale=.2, size=36)
    dataset = SpectroDataset("final_meta_export")
    dataset.add_samples(x[:28], {"partition": "train"})
    dataset.add_samples(x[28:], {"partition": "test"})
    dataset.add_targets(y)
    return (dataset, dataset.x({"partition": "train"}), dataset.y({"partition": "train"}).ravel(),
            dataset.x({"partition": "test"}))


@pytest.mark.parametrize("shape", ["sequential", "merged"])
def test_final_meta_refit_artifact_survives_close_and_incomplete_export_refuses(tmp_path, shape):
    dataset, x, y, xt = _cohort()
    base = Ridge(alpha=2.).fit(x, y)
    if shape == "sequential":
        features = base.predict(x).reshape(-1, 1)
        heldout_features = base.predict(xt).reshape(-1, 1)
        steps = [Ridge(alpha=2.), MetaModel(Ridge(alpha=.1))]
    else:
        scaler = StandardScaler().fit(x)
        other = Ridge(alpha=3.).fit(scaler.transform(x), y)
        features = np.column_stack([base.predict(x), other.predict(scaler.transform(x))])
        heldout_features = np.column_stack([base.predict(xt), other.predict(scaler.transform(xt))])
        steps = [{"branch": [[Ridge(alpha=2.)], [StandardScaler(), Ridge(alpha=3.)]]},
                 {"merge": "predictions"}, {"model": Ridge(alpha=.1)}]
    meta = Ridge(alpha=.1).fit(features, y)
    expected = meta.predict(heldout_features)
    workspace = tmp_path / "workspace"
    with nirs4all.run([KFold(3, shuffle=True, random_state=42), *steps], dataset, engine="legacy",
                      refit=True, workspace_path=workspace, save_charts=False, verbose=0) as result:
        rows = result.predictions.filter_predictions(load_arrays=True)
        selected = [row for row in rows if str(row.get("fold_id")) == "final" and (
            row.get("model_classname") == "MetaModel" or (row.get("metadata") or {}).get("stacking_role") == "meta_model")]
        assert selected
        test_rows = [row for row in selected if row.get("partition") == "test"]
        if test_rows:
            np.testing.assert_allclose(np.asarray(test_rows[0]["y_pred"]).ravel(), expected, rtol=1e-6, atol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(selected[0]["y_pred"]).ravel(), meta.predict(features), rtol=1e-6, atol=1e-6)
        source = test_rows[0] if test_rows else selected[0]
        artifact_id = source["model_artifact_id"]
        chain_id = source["chain_id"]
        for fmt in ("n4a", "n4a.py"):
            for existing in (False, True):
                output = tmp_path / f"selected-{existing}.{fmt}"
                if existing:
                    output.write_bytes(b"previous valid destination")
                with pytest.raises(NotImplementedError, match="raw-input dependency closure was not captured"):
                    result.export(output, source=source, format=fmt)
                assert output.read_bytes() == b"previous valid destination" if existing else not output.exists()
                with pytest.raises(NotImplementedError, match="raw-input dependency closure was not captured"):
                    result.export(output, chain_id=chain_id, format=fmt)
                assert output.read_bytes() == b"previous valid destination" if existing else not output.exists()

    # The genuine final fitted model must remain persisted after normal cleanup,
    # runner close, and a new independent store connection.
    store = WorkspaceStore(workspace)
    try:
        fitted = store.load_artifact(artifact_id)
        assert isinstance(fitted, Ridge) and fitted.alpha == .1
        np.testing.assert_allclose(fitted.coef_, meta.coef_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fitted.predict(heldout_features), expected, rtol=1e-6, atol=1e-6)
        chain = store.get_chain(chain_id)
        assert chain is not None and chain["fold_artifacts"]["fold_final"] == artifact_id
        for fmt in ("n4a", "n4a.py"):
            with pytest.raises(NotImplementedError, match="raw-input dependency closure was not captured"):
                BundleGenerator(workspace, store=store).export_from_chain(chain_id, tmp_path / f"chain.{fmt}", fmt=fmt)
            assert not (tmp_path / f"chain.{fmt}").exists()
        artifact_path = store.get_artifact_path(artifact_id)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    finally:
        store.close()
    reopened = WorkspaceStore(workspace)
    try:
        assert hashlib.sha256(reopened.get_artifact_path(artifact_id).read_bytes()).hexdigest() == digest
        np.testing.assert_allclose(reopened.load_artifact(artifact_id).predict(heldout_features), expected, rtol=1e-6, atol=1e-6)
    finally:
        reopened.close()
    arrays = tmp_path / "oracle.npz"
    np.savez(arrays, features=heldout_features, expected=expected)
    cold = subprocess.run([sys.executable, "-c", """
import sys
import numpy as np
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
store = WorkspaceStore(sys.argv[1])
try:
    model = store.load_artifact(sys.argv[2])
    assert model.alpha == .1
    oracle = np.load(sys.argv[3])
    np.testing.assert_allclose(model.predict(oracle['features']), oracle['expected'], rtol=1e-6, atol=1e-6)
finally:
    store.close()
""", str(workspace), artifact_id, str(arrays)], capture_output=True, text=True, timeout=60)
    assert cold.returncode == 0, cold.stdout + cold.stderr


@pytest.mark.parametrize("fmt", ["n4a", "n4a.py"])
def test_exact_selected_artifact_missing_rejects_before_export_write(tmp_path, fmt):
    model = Ridge().fit(np.eye(4), np.arange(4.))
    resolved = ResolvedPrediction(model_step_index=3, target_model={"model_artifact_id": "actual-meta"},
                                  artifact_provider=MapArtifactProvider({2: [("earlier-base", model)], 3: [("wrong-model", model)]}))
    generator = BundleGenerator(tmp_path)
    destination = tmp_path / f"previous.{fmt}"
    destination.write_bytes(b"unchanged")
    with patch.object(generator.resolver, "resolve", return_value=resolved):
        with pytest.raises(ValueError, match="target artifact 'actual-meta' is unavailable"):
            generator.export({"model_artifact_id": "actual-meta"}, destination, format=fmt)
    assert destination.read_bytes() == b"unchanged"


def test_plain_base_cv_and_refit_exports_remain_replayable(tmp_path):
    dataset, x, y, xt = _cohort()
    splitter = KFold(3, shuffle=True, random_state=42)
    expected_cv = np.mean([Ridge(alpha=2.).fit(x[train], y[train]).predict(xt) for train, _ in splitter.split(x)], axis=0)
    expected_final = Ridge(alpha=2.).fit(x, y).predict(xt)
    for refit, expected in ((False, expected_cv), (True, expected_final)):
        with nirs4all.run([KFold(3, shuffle=True, random_state=42), Ridge(alpha=2.)], dataset, engine="legacy",
                          refit=refit, workspace_path=tmp_path / f"base-{refit}", save_charts=False, verbose=0) as result:
            rows = result.predictions.filter_predictions(partition="test", load_arrays=True)
            source = next(row for row in rows if str(row["fold_id"]) == ("final" if refit else "avg"))
            for fmt in ("n4a", "n4a.py"):
                archive = result.export(tmp_path / f"base-{refit}.{fmt}", source=source, format=fmt)
                if fmt == "n4a":
                    np.testing.assert_allclose(BundleLoader(archive).predict(xt).ravel(), expected, rtol=1e-6, atol=1e-6)
                else:
                    namespace = {}
                    exec(compile(archive.read_text(), str(archive), "exec"), namespace)
                    np.testing.assert_allclose(namespace["predict"](xt).ravel(), expected, rtol=1e-6, atol=1e-6)
                chain_archive = result.export(tmp_path / f"base-chain-{refit}.{fmt}", chain_id=source["chain_id"], format=fmt)
                if fmt == "n4a":
                    np.testing.assert_allclose(BundleLoader(chain_archive).predict(xt).ravel(), expected, rtol=1e-6, atol=1e-6)
                else:
                    namespace = {}
                    exec(compile(chain_archive.read_text(), str(chain_archive), "exec"), namespace)
                    np.testing.assert_allclose(namespace["predict"](xt).ravel(), expected, rtol=1e-6, atol=1e-6)
