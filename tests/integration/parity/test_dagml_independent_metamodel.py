"""Independent named MetaModels remain separate scored and exportable outputs."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.models import MetaModel
from nirs4all.pipeline.dagml.native_results import read_native_results

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_independent_named_metamodels_keep_both_scores_and_archives(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """One graph preserves both terminal outputs and selects by CV, not DSL order."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    features, targets = make_regression(n_samples=60, n_features=6, noise=0.1, random_state=42)
    dataset = SpectroDataset("independent_metamodels")
    dataset.add_samples(features[:48], {"partition": "train"})
    dataset.add_samples(features[48:], {"partition": "test"})
    dataset.add_targets(targets)
    held_out = features[48:].astype(np.float32)
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2),
        Ridge(alpha=100),
        {"model": MetaModel(Ridge(alpha=1), source_models=["PLSRegression"], name="A")},
        {"model": MetaModel(Ridge(alpha=2), source_models=["Ridge"], name="B")},
    ]
    legacy = nirs4all.run(
        pipeline, dataset, engine="legacy", allow_fallback=False, refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    native = nirs4all.run(
        pipeline, dataset, engine="dag-ml", allow_fallback=False, refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        # Legacy attests two finite CV outputs. Its own REFIT of this shape is
        # broken, so numerical CV scores are not required to match nested OOF.
        legacy_meta = {
            row["model_name"]: row
            for row in legacy.predictions.filter_predictions(partition="val", fold_id="avg")
            if row["model_name"] in {"A", "B"}
        }
        assert set(legacy_meta) == {"A", "B"}
        assert all(np.isfinite(row["val_score"]) and row["n_samples"] == 48 for row in legacy_meta.values())

        assert native.execution_engine == "dag-ml"
        assert [child.cv_best["model_name"] for child in native.runs] == ["A", "B"]
        assert all(np.isfinite(child.cv_best_score) and np.isfinite(child.best_rmse) for child in native.runs)
        assert native.runs[0].cv_best_score < native.runs[1].cv_best_score
        assert native._dagml_selection_decision["selected_candidate_id"] == "merge:stack"
        assert native.cv_best["model_name"] == native.best_final["model_name"] == "A"
        assert {report["producer_node"] for report in native._dagml_score_set["reports"]
                if report["partition"] == "validation" and report.get("fold_id") == "avg"} >= {
                    "merge:stack", "merge:stack.level2",
                }
        assert all(report.get("producer_port", "oof") == "oof" for report in native._dagml_score_set["reports"])

        for child in native.runs:
            label = child.cv_best["model_name"]
            archive = native.export(tmp_path / f"{label}.n4a", source=child.best_final)
            persisted = read_native_results(child._dagml_results_dir)
            replay = persisted["manifest"]["stacking_replay"]
            assert replay["producer_node"] == child._dagml_stacking_replay_producer
            assert len(replay["base_producers"]) == 1
            by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
            base = by_id[replay["base_producers"][0]["artifact_id"]]["estimator"]
            meta = by_id[replay["meta_artifact_id"]]["estimator"]
            expected = np.asarray(meta.predict(np.asarray(base.predict(held_out)).reshape(12, -1))).ravel()
            np.testing.assert_allclose(np.asarray(child.best_final["y_pred"]).ravel(), expected, atol=1e-5)
            np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, held_out).y_pred).ravel(), expected, atol=1e-5)

        default_archive = native.export(tmp_path / "selected.n4a")
        np.testing.assert_allclose(
            np.asarray(nirs4all.predict(default_archive, held_out).y_pred).ravel(),
            np.asarray(native.runs[0].best_final["y_pred"]).ravel(), atol=1e-5,
        )
    finally:
        native.close()
        legacy.close()
