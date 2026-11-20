"""Real binary/multiclass probability stacking preserves cold labels and vocabulary."""

import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("mode", ["global", "per_branch", "meta"])
def test_probability_stacking_final_cold_labels_probabilities_and_vocabulary(tmp_path, n_classes, mode):
    rng = np.random.default_rng(77)
    indices = np.arange(60) % n_classes
    vocabulary = np.asarray([2., 5., 9.][:n_classes])
    y = vocabulary[indices]
    x = (rng.normal(size=(60, 4)) + indices[:, None] * np.asarray([1., -.8, .4, 1.2])).astype(np.float32)
    dataset = SpectroDataset("real_probability_stacking")
    dataset.add_samples(x[:48], {"partition": "train"})
    dataset.add_samples(x[48:], {"partition": "test"})
    dataset.add_targets(y)
    dataset.set_task_type("binary" if n_classes == 2 else "multiclass")
    numeric_y = dataset.y({"partition": "train"}).ravel()
    numeric_vocabulary = np.unique(numeric_y)
    steps = [StandardScaler(), StratifiedKFold(3, shuffle=True, random_state=42),
             {"branch": {"second": [{"model": LogisticRegression(C=.5), "name": "A"}],
                         "first": [{"model": LogisticRegression(C=2.), "name": "B"}]}}]
    if mode == "meta":
        steps.append({"model": MetaModel(LogisticRegression(C=.7), use_proba=True), "name": "BranchMeta"})
    selections = [{"branch": index, "proba": True} for index in (0, 1)]
    if mode == "meta":
        for selection in selections:
            selection["select"] = ["BranchMeta"]
    merge = {"predictions": {"branches": [0, 1], "proba": True}} if mode == "global" else {"predictions": selections}
    steps += [{"merge": merge}, {"model": LogisticRegression(C=.3), "name": "FINAL"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, dataset, engine="legacy", refit=True, workspace_path=workspace,
                      save_charts=False, save_artifacts=True, verbose=0) as result:
        final = next(row for row in result.predictions.filter_predictions(load_arrays=True)
                     if row["model_name"] == "FINAL" and str(row["fold_id"]) == "final" and row["partition"] == "test")
        expected_labels = np.asarray(final["y_pred"]).ravel()
        expected_proba = np.asarray(final["y_proba"])
        assert expected_proba.shape == (12, n_classes)
        assert final["n_features"] == (2 if n_classes == 2 else 6)
    with WorkspaceStore(workspace) as store:
        predictor = store.get_raw_input_predictor(final["chain_id"])
        assert predictor is not None
        np.testing.assert_array_equal(predictor.classes_, numeric_vocabulary)
        np.testing.assert_array_equal(predictor.class_labels_, vocabulary)
        np.testing.assert_array_equal(predictor.predict(x[48:]), expected_labels)
        np.testing.assert_allclose(predictor.predict_proba(x[48:]), expected_proba, rtol=1e-6, atol=1e-6)
        # Independent existing owner fits reproduce the exact train-only feature algorithm.
        prefix = StandardScaler().fit(x[:48])
        train_x, test_x = prefix.transform(x[:48]), prefix.transform(x[48:])
        train_columns, test_columns = [], []
        for strength in (.5, 2.):
            fitted = LogisticRegression(C=strength).fit(train_x, numeric_y)
            train_prob, test_prob = fitted.predict_proba(train_x), fitted.predict_proba(test_x)
            if mode == "meta":
                meta_x = train_prob[:, 1:] if n_classes == 2 else train_prob
                meta_test = test_prob[:, 1:] if n_classes == 2 else test_prob
                branch_meta = LogisticRegression(C=.7).fit(meta_x, numeric_y)
                train_prob, test_prob = branch_meta.predict_proba(meta_x), branch_meta.predict_proba(meta_test)
            train_columns.append(train_prob[:, 1:] if n_classes == 2 else train_prob)
            test_columns.append(test_prob[:, 1:] if n_classes == 2 else test_prob)
        independent = LogisticRegression(C=.3).fit(np.hstack(train_columns).astype(np.float32), numeric_y)
        np.testing.assert_array_equal(independent.predict(np.hstack(test_columns).astype(np.float32)), expected_labels)
        np.testing.assert_allclose(independent.predict_proba(np.hstack(test_columns).astype(np.float32)), expected_proba, rtol=1e-5, atol=1e-5)
        bundle = store.export_chain(final["chain_id"], tmp_path / "final.n4a")
    with BundleLoader(bundle) as loader:
        np.testing.assert_array_equal(loader.predict(x[48:]), expected_labels)
    np.savez(tmp_path / "inputs.npz", raw_X=x[48:], labels=expected_labels, probabilities=expected_proba, classes=numeric_vocabulary, class_labels=vocabulary)
    probe = r'''
import pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
blocked = pathlib.Path(sys.argv[1]).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = pathlib.Path(args[0]).resolve()
        if p == blocked or blocked in p.parents:
            raise AssertionError('cold classification opened its source workspace')
def forbidden(*args, **kwargs):
    raise AssertionError('cold classification attempted a fit or workspace open')
sys.addaudithook(audit)
LogisticRegression.fit = StandardScaler.fit = WorkspaceStore.__init__ = forbidden
arrays = np.load(sys.argv[3])
with BundleLoader(sys.argv[2]) as loader:
    model = loader.artifact_provider.get_artifacts_for_step(1)[0][1]
    np.testing.assert_array_equal(model.classes_, arrays['classes'])
    np.testing.assert_array_equal(model.class_labels_, arrays['class_labels'])
    np.testing.assert_array_equal(loader.predict(arrays['raw_X']), arrays['labels'])
    np.testing.assert_allclose(model.predict_proba(arrays['raw_X']), arrays['probabilities'], rtol=1e-6, atol=1e-6)
'''
    completed = subprocess.run([sys.executable, "-c", probe, str(workspace), str(bundle), str(tmp_path / "inputs.npz")],
                               cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
