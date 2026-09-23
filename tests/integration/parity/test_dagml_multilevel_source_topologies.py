"""Named MetaModels can reuse earlier prediction producers at later levels."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.dagml.native_results import read_native_results

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("second_sources", [["first", "Ridge"], ["Ridge", "first"]])
def test_named_metamodel_reuses_base_prediction_at_second_level(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, second_sources: list[str],
) -> None:
    """A diamond source graph retains its declared feature order and replay closure."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    features, targets = make_regression(n_samples=48, n_features=6, noise=0.1, random_state=42)
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2),
        Ridge(alpha=100),
        {"model": MetaModel(Ridge(), source_models=["PLSRegression", "Ridge"], name="first")},
        {"model": MetaModel(Lasso(alpha=0.1), source_models=second_sources, name="second")},
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
            12.489151848289701 if second_sources[0] == "first" else 12.489720546134414, abs=1e-6,
        )
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        persisted = read_native_results(native._dagml_results_dir)
        stages = persisted["manifest"]["stacking_replay"]["stages"]
        assert len(stages) == 2
        source_nodes = {
            "first": "merge:stack", "Ridge": "branch:1.node:0",
        }
        assert [source["producer_node"] for source in stages[1]["base_producers"]] == [
            source_nodes[name] for name in second_sources
        ]
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        first_features = np.column_stack([
            np.asarray(by_id[source["artifact_id"]]["estimator"].predict(features[:7])).reshape(7, -1)
            for source in stages[0]["base_producers"]
        ])
        first_prediction = np.asarray(by_id[stages[0]["meta_artifact_id"]]["estimator"].predict(first_features)).reshape(7, -1)
        second_features = np.column_stack([
            first_prediction if name == "first" else
            np.asarray(by_id[stages[0]["base_producers"][1]["artifact_id"]]["estimator"].predict(features[:7])).reshape(7, -1)
            for name in second_sources
        ])
        expected = by_id[stages[1]["meta_artifact_id"]]["estimator"].predict(second_features)
        archive = native.export(tmp_path / "multi_source_meta.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, features[:7]).y_pred).ravel(),
                                   np.asarray(expected).ravel(), atol=1e-8)
    finally:
        native.close()
        legacy.close()
