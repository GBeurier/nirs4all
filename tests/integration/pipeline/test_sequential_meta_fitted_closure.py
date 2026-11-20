"""Sequential META CV and standalone FINAL retain their actual raw-input function."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models import MetaModel, StackingConfig, TestAggregation
from nirs4all.pipeline.bundle import BundleGenerator, BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

from ._cold_replay import run_cold_replay


@pytest.mark.parametrize("profile", ["mean", "weighted", "binary", "multiclass"])
def test_sequential_meta_cv_and_final_cold_exact_heldout_function(tmp_path, profile):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    classification = profile in ("binary", "multiclass")
    classes = 3 if profile == "multiclass" else 2
    raw_vocabulary = np.asarray([2., 5., 9.][:classes])
    if classification:
        indices = np.arange(40) % classes
        x += indices[:, None] * np.asarray([.8, -.5, .4, 1., -.4, .2], dtype=np.float32)
        y = raw_vocabulary[indices]
        dataset = SpectroDataset("sequential_classifier")
        dataset.add_samples(x[:32], {"partition": "train"})
        dataset.add_samples(x[32:], {"partition": "test"})
        dataset.add_targets(y)
        dataset.set_task_type("multiclass" if classes == 3 else "binary")
        base, meta = LogisticRegression(C=.5), LogisticRegression(C=.7)
        splitter = StratifiedKFold(2, shuffle=True, random_state=42)
    else:
        y = (x @ np.arange(1., 7.) + rng.normal(scale=.2, size=40)).astype(np.float32)
        dataset = {"train_x": x[:32], "train_y": y[:32], "test_x": x[32:], "test_y": y[32:]}
        base, meta = PLSRegression(n_components=3), Ridge(alpha=1.)
        splitter = KFold(2, shuffle=True, random_state=42)
    aggregation = TestAggregation.WEIGHTED_MEAN if profile == "weighted" else TestAggregation.MEAN
    steps = [MinMaxScaler(), splitter, {"model": base, "name": "BASE"},
             {"model": MetaModel(meta, use_proba=classification, stacking_config=StackingConfig(test_aggregation=aggregation)), "name": "META"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, dataset, engine="legacy", refit=True, workspace_path=workspace,
                      save_artifacts=True, save_charts=False, verbose=0) as result:
        rows = result.predictions.filter_predictions(load_arrays=True)
        selected = [next(row for row in rows if row["model_name"] == "META" and row["partition"] == "test"
                         and str(row["fold_id"]) == fold) for fold in ("0", "final")]
        expected = [np.asarray(row["y_pred"]).copy() for row in selected]
        probabilities = [np.asarray(row["y_proba"]).copy() for row in selected] if classification else []
    # A new store connection cannot recover an omitted graph from a live registry.
    with WorkspaceStore(workspace) as store:
        for index, row in enumerate(selected):
            fitted = store.get_raw_input_predictor(row["chain_id"], fold_id=row["fold_id"], model_artifact_id=row["model_artifact_id"])
            assert fitted is not None and fitted.root.artifact_id == row["model_artifact_id"]
            np.testing.assert_allclose(fitted.predict(x[32:]).ravel(), expected[index].ravel(), rtol=1e-5, atol=1e-8)
            base_rows = [item for item in rows if item["model_name"] == "BASE" and item["partition"] == "train"
                         and item["pipeline_uid"] == row["pipeline_uid"] and str(item["fold_id"]) not in ("avg", "w_avg")]
            assert len(base_rows) == (2 if index == 0 else 1)
            base_chain = store.get_chain(base_rows[0]["chain_id"])
            prefix = store.load_artifact(base_chain["shared_artifacts"]["1"][0])
            base_x = prefix.transform(x[32:])
            base_models = [store.load_artifact(item["model_artifact_id"]) for item in base_rows]
            if classification:
                values = [model.predict_proba(base_x)[:, 1:] if classes == 2 else model.predict_proba(base_x) for model in base_models]
            else:
                values = [model.predict(base_x).reshape(len(base_x), -1) for model in base_models]
            if profile == "weighted" and index == 0:
                from nirs4all.data.ensemble_utils import EnsembleUtils

                scores = np.asarray([item["val_score"] for item in base_rows])
                weights = EnsembleUtils._scores_to_weights(scores, higher_is_better=False)
                independent_features = np.average(np.asarray(values, dtype=float), axis=0, weights=weights)
                captured = fitted.root.inputs[0].inputs[0]
                np.testing.assert_array_equal(captured.state["scores"], scores)
                assert captured.state["metric"] == base_rows[0]["metric"] == "rmse"
            else:
                independent_features = np.mean(np.asarray(values, dtype=float), axis=0)
            actual_meta = store.load_artifact(row["model_artifact_id"])
            np.testing.assert_allclose(actual_meta.predict(independent_features).ravel(), expected[index].ravel(), rtol=1e-5, atol=1e-8)
            if classification:
                np.testing.assert_allclose(actual_meta.predict_proba(independent_features), probabilities[index], rtol=1e-5, atol=1e-8)
            bundle = BundleGenerator(workspace, store=store).export(row, tmp_path / f"{index}.n4a")
            with BundleLoader(bundle) as loader:
                np.testing.assert_allclose(loader.predict(x[32:]).ravel(), expected[index].ravel(), rtol=1e-5, atol=1e-8)
            arrays = {"raw_X": x[32:], "expected": expected[index]}
            if classification:
                np.testing.assert_allclose(fitted.predict_proba(x[32:]), probabilities[index], rtol=1e-5, atol=1e-8)
                np.testing.assert_array_equal(fitted.class_labels_, raw_vocabulary)
                arrays.update(probabilities=probabilities[index], classes=fitted.classes_, raw_classes=raw_vocabulary)
            np.savez(tmp_path / f"{index}.npz", **arrays)
            # Preserve the requested held-out partition while reusing a single captured CV model.
            runner = nirs4all.PipelineRunner(workspace_path=workspace, save_charts=False, verbose=0)
            predicted, produced = runner.predict(row, dataset, verbose=0)
            np.testing.assert_allclose(np.asarray(predicted).ravel(), expected[index].ravel(), rtol=1e-5, atol=1e-8)
            assert any(str(item["fold_id"]) == str(row["fold_id"]) and item["partition"] == "test"
                       for item in produced.filter_predictions(load_arrays=False))
            replayed_test = next(item for item in produced.filter_predictions(load_arrays=True)
                                 if str(item["fold_id"]) == str(row["fold_id"]) and item["partition"] == "test")
            np.testing.assert_array_equal(replayed_test["sample_indices"], row["sample_indices"])
    probe = r'''
import pathlib,sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import MinMaxScaler
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
blocked=pathlib.Path(sys.argv[1]).resolve()
def audit(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
        path=pathlib.Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if path==blocked or blocked in path.parents:raise AssertionError('cold replay accessed producer workspace')
def forbidden(*args,**kwargs):raise AssertionError('cold replay attempted fit or WorkspaceStore access')
sys.addaudithook(audit)
WorkspaceStore.__init__=forbidden
for cls in (PLSRegression,LogisticRegression,Ridge,MinMaxScaler):cls.fit=forbidden
arrays=np.load(sys.argv[3])
with BundleLoader(sys.argv[2]) as loader:
    np.testing.assert_allclose(loader.predict(arrays['raw_X']).ravel(),arrays['expected'].ravel(),rtol=1e-5,atol=1e-8)
    if 'probabilities' in arrays:
        model=loader.artifact_provider.get_artifacts_for_step(1)[0][1]
        np.testing.assert_allclose(model.predict_proba(arrays['raw_X']),arrays['probabilities'],rtol=1e-5,atol=1e-8)
        np.testing.assert_array_equal(model.classes_,arrays['classes'])
        np.testing.assert_array_equal(model.class_labels_,arrays['raw_classes'])
'''
    for index in range(2):
        completed = run_cold_replay(probe, [str(workspace), str(tmp_path / f"{index}.n4a"), str(tmp_path / f"{index}.npz")], tmp_path)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_sequential_meta_without_saved_artifacts_remains_valid_training(tmp_path):
    x = np.arange(120., dtype=np.float32).reshape(30, 4)
    y = x[:, 0] + np.sin(x[:, 1])
    with nirs4all.run([KFold(2), Ridge(), {"model": MetaModel(Ridge())}], (x, y), engine="legacy", refit=True,
                      workspace_path=tmp_path / "workspace", save_artifacts=False, save_charts=False, verbose=0) as result:
        assert result.predictions.num_predictions > 0
