"""Public legacy/DAG oracle for per-branch classifier probability aggregation."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import nirs4all
from nirs4all.pipeline.dagml.detect import _detect_proba_mean_stacking_branch


def _pipeline() -> list:
    return [
        StratifiedKFold(3, shuffle=True, random_state=17),
        {"branch": [
            [{"model": LogisticRegression(C=0.3, max_iter=300)},
             {"model": LogisticRegression(C=1.0, max_iter=300)}],
            [{"model": LogisticRegression(C=0.5, max_iter=300)},
             {"model": LogisticRegression(C=2.0, max_iter=300)}],
        ]},
        {"merge": {"predictions": [
            {"branch": 0, "aggregate": "proba_mean"},
            {"branch": 1, "aggregate": "proba_mean"},
        ]}},
        {"model": LogisticRegression(max_iter=300)},
    ]


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_legacy_and_dag_support_per_branch_probability_mean(tmp_path, monkeypatch, mechanism) -> None:
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(731)
    features = rng.normal(size=(60, 6))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)
    pipeline = _pipeline()
    detected = _detect_proba_mean_stacking_branch(pipeline)
    assert detected is not None
    assert [entry["branch"] for entry in detected[2]] == ["branch_0", "branch_1"]

    legacy = nirs4all.run(
        pipeline, (features, labels), engine="legacy", refit=False,
        save_artifacts=False, save_charts=False, verbose=0,
    )
    assert legacy.cv_best_score == pytest.approx(0.9326599326599327)

    native = nirs4all.run(
        pipeline, (features, labels), engine="dag-ml", refit=False,
        workspace_path=tmp_path / "dag", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert np.isfinite(native.cv_best_score)
        assert 0.0 <= native.cv_best_score <= 1.0
        probability_blocks = [
            block for frame in native._dagml_node_results
            for block in frame.get("result", frame).get("predictions", [])
            if str(block.get("producer_node", "")).startswith("branch:")
            and block.get("partition") == "validation"
            and block.get("producer_port") == "proba"
        ]
        assert probability_blocks
        assert all(len(row) == 2 and sum(row) == pytest.approx(1.0)
                   for block in probability_blocks for row in block["values"])
    finally:
        native.close()


def test_score_selection_is_lowered_or_rejected_explicitly() -> None:
    pipeline = _pipeline()
    pipeline[2]["merge"]["predictions"][0]["select"] = "best"
    pipeline[2]["merge"]["predictions"][0]["metric"] = "accuracy"
    detected = _detect_proba_mean_stacking_branch(pipeline)
    assert detected is not None
    assert detected[2][0]["select"] == "best"
    assert detected[2][0]["metric"] == "accuracy"
    pipeline[2]["merge"]["predictions"][0]["metric"] = "f1"
    assert _detect_proba_mean_stacking_branch(pipeline)[2][0]["metric"] == "f1"
    for unavailable_legacy_score in ("auc", "log_loss"):
        pipeline[2]["merge"]["predictions"][0]["metric"] = unavailable_legacy_score
        assert _detect_proba_mean_stacking_branch(pipeline)[2][0]["metric"] == unavailable_legacy_score


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("metric", ["f1", "auc", "log_loss"])
def test_legacy_merge_selection_metrics_complete_native_stacking(tmp_path, monkeypatch, mechanism, metric):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(731)
    features = rng.normal(size=(60, 6))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)
    pipeline = _pipeline()
    pipeline[2]["merge"]["predictions"][0].update({"select": "best", "metric": metric})
    legacy = nirs4all.run(pipeline, (features, labels), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(pipeline, (features, labels), engine="dag-ml", refit=False,
                         allow_fallback=False, workspace_path=tmp_path / metric,
                         save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert np.isfinite(native.cv_best_score)
        reports = native._dagml_score_set["reports"]
        base_reports = [report for report in reports if str(report["producer_node"]).startswith("branch:")
                        and report["partition"] == "validation"]
        assert base_reports
        if metric == "f1":
            assert all(np.isfinite(report["metrics"]["f1"]) for report in base_reports)
        else:
            # Legacy has no ranked validation evidence for these two names in
            # this pipeline; both engines retain first-model fallback.
            assert all(metric not in report["metrics"] for report in base_reports)
    finally:
        native.close()
