"""FINAL Python-host bundles retain the fitted raw-input dependency closure."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.bundle import BundleGenerator, BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.mark.parametrize("mixed,branch_meta,scaled_targets,weighted,concat,homonyms", [
    (False, False, False, False, False, False), (True, False, False, False, False, False),
    (True, True, False, False, False, False), (True, False, True, True, False, False),
    (True, False, False, False, True, False), (True, False, False, False, False, True),
])
def test_final_closure_reopens_and_predicts_cold_without_workspace_or_fit(tmp_path, mixed, branch_meta, scaled_targets, weighted, concat, homonyms):
    rng = np.random.default_rng(29)
    x = rng.normal(size=(30, 5)).astype(np.float32)
    y = (x @ np.arange(1., 6.) + rng.normal(scale=.1, size=30)).astype(np.float32)
    data = {"train_x": x[:24], "train_y": y[:24], "test_x": x[24:], "test_y": y[24:]}
    branches = {"second": [StandardScaler(), {"model": Ridge(alpha=.5), "name": "A"}],
                "first": [{"model": Ridge(alpha=2.), "name": "B"}, PCA(n_components=2),
                          {"model": Ridge(alpha=1.), "name": "C"}]}
    if homonyms:
        branches["first"][0]["name"] = "shared"
        branches["first"][2]["name"] = "shared"
    if concat:
        branches["second"].insert(1, {"concat_transform": [PCA(n_components=2), PCA(n_components=1)]})
    if not mixed:
        branches["first"] = [{"model": Ridge(alpha=2.), "name": "B"}]
    merge = {"predictions": [{"branch": 1, "select": "best", "metric": "rmse"}, {"branch": 0}],
             "features": [1], "include_original": True} if mixed else {"predictions": [0, 1], "include_original": True}
    if weighted:
        merge["predictions"][0].update(select="all", aggregate="weighted_mean", weight_metric="rmse")
    steps = [StandardScaler(), KFold(3, shuffle=True, random_state=42), {"branch": branches}]
    if scaled_targets:
        steps.insert(0, {"y_processing": StandardScaler()})
    if branch_meta:
        steps.append({"model": MetaModel(Ridge(alpha=.3)), "name": "BranchMeta"})
    steps += [{"merge": merge}, {"model": Ridge(alpha=.1), "name": "FINAL"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, data, engine="legacy", refit=True, workspace_path=workspace,
                      save_charts=False, save_artifacts=True, verbose=0) as result:
        rows = [row for row in result.predictions.iter_entries()
                if str(row["fold_id"]) == "final" and row["model_name"] == "FINAL"]
        train = next(row for row in rows if row["partition"] == "train")
        test = next((row for row in rows if row["partition"] == "test"), None)
        expected = np.asarray(train["y_pred"]).reshape(-1)
        raw_x = x[:24]
        if test is not None:
            expected = np.concatenate([expected, np.asarray(test["y_pred"]).reshape(-1)])
            raw_x = x
        model_id = train["model_artifact_id"]
    # Reopen after run cleanup: no registry or in-memory graph can fill gaps.
    with WorkspaceStore(workspace) as store:
        chains = store.get_chains_for_pipeline(train["pipeline_uid"])
        chain_id = next(row["chain_id"] for row in chains.iter_rows(named=True)
                        if model_id in json.loads(row["fold_artifacts"]).values())
        np.testing.assert_allclose(store.replay_chain(chain_id, raw_x), expected, rtol=1e-5, atol=1e-5)
        bundle = store.export_chain(chain_id, tmp_path / "final.n4a")
        via_dict = BundleGenerator(workspace, store=store).export(train, tmp_path / "dict.n4a")
        wrong_identity = {**train, "chain_id": chain_id, "model_artifact_id": "different-explicit-fitted-model"}
        destination = tmp_path / "preserved.n4a"
        destination.write_bytes(b"previous valid destination")
        with pytest.raises(ValueError, match="does not match the FINAL raw-input closure"):
            BundleGenerator(workspace, store=store).export(wrong_identity, destination)
        assert destination.read_bytes() == b"previous valid destination"
    with pytest.raises(ValueError, match="does not match the FINAL raw-input closure"):
        nirs4all.predict(wrong_identity, raw_x, engine="legacy", workspace_path=workspace, verbose=0)
    for source in (train, bundle, via_dict):
        predicted = nirs4all.predict(model=source, data=raw_x, engine="legacy", workspace_path=workspace, verbose=0)
        np.testing.assert_allclose(np.asarray(predicted.y_pred).ravel(), expected, rtol=1e-5, atol=1e-5)
    predicted = nirs4all.predict(chain_id=chain_id, data=raw_x, workspace_path=workspace, engine="legacy", verbose=0)
    np.testing.assert_allclose(np.asarray(predicted.y_pred).ravel(), expected, rtol=1e-5, atol=1e-5)
    with BundleLoader(bundle) as loader:
        np.testing.assert_allclose(loader.predict(raw_x), expected, rtol=1e-5, atol=1e-5)
        assert len(loader.train_pipeline_config["steps"]) == len(steps)
        assert any("branch" in step for step in loader.train_pipeline_config["steps"] if isinstance(step, dict))
    if mixed and not branch_meta and not scaled_targets:
        # Real Kernel SHAP must explain the complete raw-input function.
        with nirs4all.session(engine="legacy", workspace_path=workspace, verbose=0) as session:
            for source in (train, bundle):
                explanation = nirs4all.explain(source, raw_x[:4], session=session, engine="legacy", verbose=0,
                                               plots_visible=False, explainer_type="kernel", n_samples=4, visualizations=[])
                assert explanation.values.shape == (4, raw_x.shape[1])
                np.testing.assert_allclose(explanation.values.sum(axis=1) + explanation.base_value, expected[:4], rtol=1e-5, atol=1e-5)
    np.save(tmp_path / "raw.npy", raw_x)
    np.save(tmp_path / "expected.npy", expected)
    probe = r'''
import pathlib, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nirs4all.pipeline.bundle import BundleLoader
blocked = pathlib.Path(sys.argv[1]).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if path == blocked or blocked in path.parents:
            raise AssertionError('Cold prediction accessed its source workspace')
sys.addaudithook(audit)
def no_fit(*args, **kwargs):
    raise AssertionError('Cold prediction attempted to fit')
for cls in (Ridge, StandardScaler, PCA):
    cls.fit = no_fit
with BundleLoader(sys.argv[2]) as loader:
    actual = loader.predict(np.load(sys.argv[3]))
np.testing.assert_allclose(actual, np.load(sys.argv[4]), rtol=1e-5, atol=1e-5)
'''
    completed = subprocess.run([sys.executable, "-c", probe, str(workspace), str(bundle), str(tmp_path / "raw.npy"),
                                str(tmp_path / "expected.npy")], cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
