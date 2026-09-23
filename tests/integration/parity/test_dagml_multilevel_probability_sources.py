"""Mixed prior-meta and historical-base probability sources replay exactly."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import nirs4all
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.dagml.native_results import read_native_results

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("first_probability", [False, True])
def test_second_named_meta_reuses_base_probability_columns(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, first_probability: bool,
) -> None:
    """A later meta sees prior-meta class-1 and base full probabilities in that order."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    features, targets = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=19,
    )
    pipeline = [
        StratifiedKFold(3, shuffle=True, random_state=11),
        LogisticRegression(max_iter=300),
        RandomForestClassifier(n_estimators=12, random_state=11),
        {"model": MetaModel(LogisticRegression(max_iter=300),
                            source_models=["LogisticRegression", "RandomForestClassifier"],
                            use_proba=first_probability, name="first")},
        {"model": MetaModel(LogisticRegression(max_iter=300),
                            source_models=["first", "RandomForestClassifier"],
                            use_proba=True, name="second")},
    ]
    legacy = nirs4all.run(pipeline, (features, targets), engine="legacy", allow_fallback=False,
                          refit=False, workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, (features, targets), engine="dag-ml", allow_fallback=False,
                          refit=True, workspace_path=tmp_path / "native", save_artifacts=True,
                          save_charts=False, verbose=0)
    try:
        legacy_second = next(row for row in legacy.predictions.filter_predictions()
                             if row["model_name"] == "second" and row["partition"] == "val"
                             and row["fold_id"] == "avg")
        assert legacy_second["val_score"] == pytest.approx(
            0.9166666666666667 if first_probability else 0.8333333333333334,
        )
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        persisted = read_native_results(native._dagml_results_dir)
        assert all(report.get("producer_port", "oof") == "oof"
                   for report in persisted["score_set"]["reports"])
        stages = persisted["manifest"]["stacking_replay"]["stages"]
        assert [source["column_block"] for source in stages[1]["base_producers"]] == [
            "probability_values", "probability_values",
        ]
        assert all(source["column_projection"] == "selected_class" for source in stages[1]["base_producers"])
        assert [source["producer_node"] for source in stages[1]["base_producers"]] == [
            "merge:stack", "branch:1.node:0",
        ]
        assert [source["column_block"] for source in stages[0]["base_producers"]] == [
            "probability_values" if first_probability else "prediction_values",
        ] * 2
        if first_probability:
            assert all(source["column_projection"] == "selected_class" for source in stages[0]["base_producers"])
        artifacts = {artifact["artifact_id"]: artifact["estimator"] for artifact in persisted["artifacts"]}
        x = features[:7]
        base_probs = [
            np.asarray(artifacts[source["artifact_id"]].predict_proba(x))
            for source in stages[0]["base_producers"]
        ]
        first_inputs = [prob[:, 1:2] for prob in base_probs] if first_probability else [
            np.asarray(artifacts[source["artifact_id"]].predict(x)).reshape(len(x), -1)
            for source in stages[0]["base_producers"]
        ]
        first_prob = np.asarray(artifacts[stages[0]["meta_artifact_id"]].predict_proba(
            np.column_stack(first_inputs),
        ))[:, 1:2]
        second_features = np.column_stack([first_prob, base_probs[1][:, 1:2]])
        assert second_features.shape[1] == 2
        assert artifacts[stages[1]["meta_artifact_id"]].n_features_in_ == 2
        expected = artifacts[stages[1]["meta_artifact_id"]].predict(second_features)
        archive = native.export(tmp_path / "multi_source_probability.n4a")
        np.testing.assert_array_equal(np.asarray(nirs4all.predict(archive, x).y_pred).ravel(),
                                      np.asarray(expected).ravel())
    finally:
        native.close()
        legacy.close()
