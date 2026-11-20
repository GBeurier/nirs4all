"""Nested producers retain their exact fitted identity and preprocessing prefix."""

import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models.residual import ResidualModel
from nirs4all.pipeline.bundle import BundleGenerator
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.mark.parametrize("mode", ["train", "predict"])
def test_source_local_feature_reader_keeps_source_and_row_identity_in_each_mode(mode):
    sources = [np.arange(36, dtype=np.float32).reshape(12, 3), np.arange(48, dtype=np.float32).reshape(12, 4) + 100]
    dataset = SpectroDataset("source_reader")
    dataset.add_samples([source[:8] for source in sources], {"partition": "train"})
    dataset.add_samples([source[8:] for source in sources], {"partition": "test"})
    dataset.add_targets(np.arange(12, dtype=np.float32))
    for index, source in enumerate(sources):
        context = ExecutionContext(custom={"_current_source_idx": index})
        context.state.mode = mode
        train_x, train_y, test_x, test_y, _, _ = SklearnModelController().get_xy(dataset, context)
        if mode == "train":
            np.testing.assert_array_equal(train_x, source[:8])
            np.testing.assert_array_equal(test_x, source[8:])
            np.testing.assert_array_equal(train_y, np.arange(8).reshape(-1, 1))
            np.testing.assert_array_equal(test_y, np.arange(8, 12).reshape(-1, 1))
        else:
            assert train_x.shape == (0, source.shape[1])
            np.testing.assert_array_equal(test_x, source)
            np.testing.assert_array_equal(test_y, np.arange(12).reshape(-1, 1))


def test_by_source_fitted_chains_keep_source_prefixes_and_sample_identity(tmp_path):
    rng = np.random.default_rng(51)
    sources = [rng.normal(size=(24, 4)).astype(np.float32), (rng.normal(size=(24, 3)) * 3 + 17).astype(np.float32)]
    for source in sources:
        source[20:] += 1000  # Neither source's scaler may fit held-out rows.
    y = sources[0][:, 0] + 2 * sources[1][:, 1]
    dataset = SpectroDataset("distinct_source_chains")
    dataset.add_samples([source[:20] for source in sources], {"partition": "train"})
    dataset.add_samples([source[20:] for source in sources], {"partition": "test"})
    dataset.add_targets(y)
    steps = [KFold(2), {"branch": {"by_source": True, "steps": [StandardScaler(), {"model": Ridge(alpha=.5)}]}}, {"merge": "auto"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, dataset, engine="legacy", refit=False, workspace_path=workspace,
                      save_artifacts=True, save_charts=False, verbose=0) as result:
        rows = [row for row in result.predictions.iter_entries() if row.get("model_artifact_id")]
        assert {row["branch_id"] for row in rows} == {0, 1}
    with WorkspaceStore(workspace) as store:
        chains = store.get_chains_for_pipeline(rows[0]["pipeline_uid"]).to_dicts()
        assert len(chains) == 2
        for branch in (0, 1):
            chain = next(chain for chain in chains if json.loads(chain["branch_path"]) == [branch])
            shared = json.loads(chain["shared_artifacts"])
            scaler_ids = [artifact for key, artifacts in shared.items() if not key.startswith("_") for artifact in artifacts]
            assert len(scaler_ids) == 1
            scaler = store.load_artifact(scaler_ids[0])
            reference_scaler = StandardScaler().fit(sources[branch][:20])
            np.testing.assert_allclose(scaler.mean_, reference_scaler.mean_)
            np.testing.assert_allclose(scaler.scale_, reference_scaler.scale_)
            fold_ids = json.loads(chain["fold_artifacts"])
            for row in (row for row in rows if row["branch_id"] == branch):
                assert row["model_artifact_id"] == fold_ids[f"fold_{row['fold_id']}"]
                persisted = store.get_prediction(row["id"])
                assert persisted["chain_id"] == chain["chain_id"]
                train, _ = list(KFold(2).split(sources[branch][:20]))[int(row["fold_id"])]
                reference = Ridge(alpha=.5).fit(reference_scaler.transform(sources[branch][train]), y[train])
                indices = np.asarray(row["sample_indices"], dtype=int)
                expected = reference.predict(reference_scaler.transform(sources[branch][indices]))
                np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected, rtol=1e-5, atol=1e-5)
            expected_all = np.mean([
                Ridge(alpha=.5).fit(reference_scaler.transform(sources[branch][train]), y[train]).predict(reference_scaler.transform(sources[branch]))
                for train, _ in KFold(2).split(sources[branch][:20])
            ], axis=0)
            np.testing.assert_allclose(store.replay_chain(chain["chain_id"], sources[branch]), expected_all, rtol=1e-5, atol=1e-5)
            archive = store.export_chain(chain["chain_id"], tmp_path / f"source-{branch}.n4a")
            selected = next(row for row in rows if row["branch_id"] == branch)
            dictionary_archive = BundleGenerator(workspace, store=store).export(selected, tmp_path / f"source-{branch}-dict.n4a")
            weighted_archive = BundleGenerator(workspace, store=store).export(
                {**selected, "fold_weights": {0: 1., 1: 3.}}, tmp_path / f"source-{branch}-weighted.n4a")
            script = tmp_path / f"source-{branch}.n4a.py"
            script.write_bytes(b"previous valid script")
            for export in (
                lambda: store.export_chain(chain["chain_id"], script, format="n4a.py"),
                lambda: BundleGenerator(workspace, store=store).export(selected, script, format="n4a.py"),
            ):
                with pytest.raises(NotImplementedError, match="scripts do not support branching"):
                    export()
                assert script.read_bytes() == b"previous valid script"
            inputs = tmp_path / f"source-{branch}.npz"
            np.savez(inputs, raw_x=sources[branch], expected=expected_all, branch=branch)
            weighted_inputs = tmp_path / f"source-{branch}-weighted.npz"
            weighted_expected = np.average([
                Ridge(alpha=.5).fit(reference_scaler.transform(sources[branch][train]), y[train]).predict(reference_scaler.transform(sources[branch]))
                for train, _ in KFold(2).split(sources[branch][:20])
            ], axis=0, weights=[1., 3.])
            np.savez(weighted_inputs, raw_x=sources[branch], expected=weighted_expected, branch=branch)
            probe = r'''
import pathlib, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
archive, inputs, workspace = sys.argv[1:]
data = np.load(inputs)
blocked = pathlib.Path(workspace).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if path == blocked or blocked in path.parents:
            raise AssertionError('Source-local cold replay accessed its workspace')
sys.addaudithook(audit)
def no_fit(*args, **kwargs):
    raise AssertionError('Source-local cold replay attempted FIT or opened a Store')
WorkspaceStore.__init__ = no_fit
Ridge.fit = StandardScaler.fit = no_fit
with BundleLoader(archive) as loader:
    assert loader._chain_data['branch_path'] == [int(data['branch'])]
    np.testing.assert_allclose(loader.predict(data['raw_x']), data['expected'], rtol=1e-5, atol=1e-5)
'''
            subprocess.run([sys.executable, "-B", "-c", probe, str(archive), str(inputs), str(workspace)], check=True)
            subprocess.run([sys.executable, "-B", "-c", probe, str(dictionary_archive), str(inputs), str(workspace)], check=True)
            subprocess.run([sys.executable, "-B", "-c", probe, str(weighted_archive), str(weighted_inputs), str(workspace)], check=True)


def test_residual_submodels_keep_distinct_fitted_chains_and_shared_prefix(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(24, 5)).astype(np.float32)
    y = x @ np.arange(1., 6.) + rng.normal(scale=.1, size=24)
    data = {"train_x": x[:20], "train_y": y[:20], "test_x": x[20:], "test_y": y[20:]}
    steps = [StandardScaler(), KFold(2), {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(alpha=.5), gate=False)}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, data, engine="legacy", refit=False, workspace_path=workspace,
                      save_artifacts=True, save_charts=False, verbose=0) as result:
        rows = [row for row in result.predictions.iter_entries() if row.get("model_artifact_id")]
        assert {row["model_classname"] for row in rows} == {"PLSRegression", "Ridge"}
    with WorkspaceStore(workspace) as store:
        chains = store.get_chains_for_pipeline(rows[0]["pipeline_uid"]).to_dicts()
        assert len(chains) == 2
        assert {chain["model_class"] for chain in chains} == {"PLSRegression", "Ridge"}
        shared_ids = []
        for chain in chains:
            fold_ids = json.loads(chain["fold_artifacts"])
            shared = json.loads(chain["shared_artifacts"])
            shared_ids.append([artifact for key, artifacts in shared.items() if not key.startswith("_") for artifact in artifacts])
            assert len(fold_ids) == 2
            for row in (row for row in rows if row["model_classname"] == chain["model_class"]):
                assert row["model_artifact_id"] == fold_ids[f"fold_{row['fold_id']}"]
                assert store.get_prediction(row["id"])["chain_id"] == chain["chain_id"]
                fitted = store.load_artifact(row["model_artifact_id"])
                assert type(fitted).__name__ == chain["model_class"]
        assert shared_ids[0] == shared_ids[1] and len(shared_ids[0]) == 1
        scaler = store.load_artifact(shared_ids[0][0])
        np.testing.assert_allclose(scaler.mean_, np.mean(x[:20], axis=0, dtype=np.float64))
