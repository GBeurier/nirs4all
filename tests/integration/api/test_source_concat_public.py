"""Historical source-merge spelling preserves source-local fitted transforms."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml.detect import _detect_by_source_distinct_preproc_concat, _detect_separation_branch


@pytest.mark.parametrize("merge", ["concat", {"sources": "concat"}])
def test_source_merge_matches_independent_fold_local_oracle(tmp_path, merge, monkeypatch):
    import nirs4all

    X = np.random.default_rng(82).normal(size=(36, 8)).astype(np.float32)
    y = 2.5 * X[:, 0] - X[:, 7] + np.arange(36) * 0.001
    dataset = SpectroDataset("sources")
    dataset.add_samples([X[:, :3], X[:, 3:]], {"partition": "train"})
    dataset.add_targets(y)
    pipeline = [KFold(3), {"branch": {"by_source": True,
        "steps": {"source_0": [StandardScaler()], "source_1": [MinMaxScaler()]}}}, {"merge": merge}, {"model": Ridge()}]
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.__init__", lambda *a, **k: pytest.fail("implicit legacy execution"))
    result = nirs4all.run(pipeline, dataset, workspace_path=tmp_path, save_charts=False, verbose=0)
    try:
        expected_errors = []
        for train, validation in KFold(3).split(X):
            first = StandardScaler().fit(X[train, :3])
            second = MinMaxScaler().fit(X[train, 3:])
            # The historical source merge replaces source 0 with the merged
            # block while retaining source 1. Preserve that published V1
            # contract here; changing the model design matrix is a separate fix.
            train_second = second.transform(X[train, 3:])
            val_second = second.transform(X[validation, 3:])
            model = Ridge().fit(np.hstack([first.transform(X[train, :3]), train_second, train_second]), y[train])
            predicted = model.predict(np.hstack([first.transform(X[validation, :3]), val_second, val_second]))
            expected_errors.extend((predicted - y[validation]) ** 2)
        assert result.execution_engine == "dag-ml"
        # cv_best's avg entry scores the pooled OOF rows, not a mean of RMSEs.
        np.testing.assert_allclose(result.cv_best_score, np.sqrt(np.mean(expected_errors)), rtol=1e-5, atol=1e-6)
        assert result.num_predictions > 0
        archive = result.export(tmp_path / "sources.n4a")
        first, second = StandardScaler().fit(X[:, :3]), MinMaxScaler().fit(X[:, 3:])
        second_block = second.transform(X[:, 3:])
        final_X = np.hstack([first.transform(X[:, :3]), second_block, second_block])
        final_model = Ridge().fit(final_X, y)
        expected_final = final_model.predict(final_X)
        predicted_final = nirs4all.predict(archive, dataset, verbose=0)
        np.testing.assert_allclose(np.asarray(predicted_final.y_pred).ravel(), expected_final, rtol=1e-5, atol=1e-6)
        unseen_X = np.random.default_rng(83).normal(size=(7, 8)).astype(np.float32)
        unseen_second = second.transform(unseen_X[:, 3:])
        expected_unseen = final_model.predict(np.hstack([first.transform(unseen_X[:, :3]), unseen_second, unseen_second]))
        predicted_unseen = nirs4all.predict(archive, unseen_X, verbose=0)
        np.testing.assert_allclose(np.asarray(predicted_unseen.y_pred).ravel(), expected_unseen, rtol=1e-5, atol=1e-6)
    finally:
        result.close()


def test_source_merge_does_not_admit_extra_unconsumed_options_or_row_separation():
    pipeline = [KFold(3), {"branch": {"by_source": True,
        "steps": {"source_0": [StandardScaler()], "source_1": [MinMaxScaler()]}}},
        {"merge": {"sources": "concat", "predictions": "mean"}}, {"model": Ridge()}]
    assert _detect_by_source_distinct_preproc_concat(pipeline, 2) is None
    row_pipeline = [KFold(3), {"branch": {"by_metadata": "site", "steps": [{"model": Ridge()}]}},
                    {"merge": {"sources": "concat"}}]
    assert _detect_separation_branch(row_pipeline) is None


def test_source_concat_flat_replay_requires_recorded_layout():
    from nirs4all.pipeline.dagml.node_runner import _SourceConcatEstimator

    estimator = _SourceConcatEstimator(Ridge(), shared_chain_template=[StandardScaler()])
    X = np.random.default_rng(91).normal(size=(9, 8))
    estimator.fit([X[:, :3], X[:, 3:]], X[:, 0])
    np.testing.assert_allclose(estimator.predict(X), estimator.predict([X[:, :3], X[:, 3:]]))
    with pytest.raises(ValueError, match="fitted source layout"):
        estimator.predict(X[:, :-1])
    del estimator._source_widths
    with pytest.raises(ValueError, match="fitted source layout"):
        estimator.predict(X)
