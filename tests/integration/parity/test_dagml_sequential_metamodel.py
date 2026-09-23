"""Public oracle for a sequential classifier followed by a MetaModel."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models import MetaModel
from nirs4all.operators.models.meta import StackingConfig
from nirs4all.operators.models.meta import TestAggregation as FoldAggregation

from ._datasets import dataset_path


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("use_proba", [False, True])
def test_sequential_classification_metamodel_uses_native_oof(use_proba, mechanism, monkeypatch):
    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    rng = np.random.default_rng(79)
    features = rng.normal(size=(30, 6))
    targets = (features[:, 0] + features[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        {"model": MetaModel(model=LogisticRegression(max_iter=300), use_proba=use_proba)},
    ]
    legacy = nirs4all.run(pipeline, (features, targets), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    assert legacy.cv_best_score == pytest.approx(0.9333333333333333)

    native = nirs4all.run(pipeline, (features, targets), engine="dag-ml", refit=False,
                         save_artifacts=False, save_charts=False, verbose=0)
    if not use_proba:
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)
    else:
        # Native nested OOF evaluation is stricter than legacy's reuse of the
        # base CV predictions, so the validation score need not be identical.
        assert native.cv_best_score == pytest.approx(0.7)
        if mechanism == "pyo3":
            probability_blocks = [
                block for node in native._dagml_node_results
                for block in node.get("predictions", [])
                if str(block.get("producer_node", "")).startswith("branch:")
                and block.get("partition") == "validation"
            ]
            assert probability_blocks
            assert all(len(row) == 2 and sum(row) == pytest.approx(1.0)
                       for block in probability_blocks for row in block["values"])
    assert native.cv_best["model_name"] == "MetaModel_LogisticRegression"


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_named_probability_source_metamodel_uses_native_oof(mechanism, monkeypatch):
    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    rng = np.random.default_rng(79)
    features = rng.normal(size=(30, 6))
    targets = (features[:, 0] + features[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        {"model": MetaModel(
            model=LogisticRegression(max_iter=300), use_proba=True,
            source_models=["LogisticRegression"],
        )},
    ]
    legacy = nirs4all.run(
        pipeline, (features, targets), engine="legacy", refit=False,
        save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(
        pipeline, (features, targets), engine="dag-ml", refit=False,
        allow_fallback=False, save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert np.isfinite(native.cv_best_score)
    assert native.cv_best["model_name"] == "MetaModel_LogisticRegression"


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("source_models", ["all", ["RandomForestClassifier"]])
def test_multiple_probability_sources_feed_native_metamodel(mechanism, source_models, monkeypatch):
    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    rng = np.random.default_rng(79)
    features = rng.normal(size=(40, 6))
    targets = (features[:, 0] + features[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        RandomForestClassifier(n_estimators=10, random_state=42),
        {"model": MetaModel(
            model=LogisticRegression(max_iter=300), use_proba=True,
            source_models=source_models,
        )},
    ]
    legacy = nirs4all.run(
        pipeline, (features, targets), engine="legacy", refit=False,
        save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(
        pipeline, (features, targets), engine="dag-ml", refit=False,
        allow_fallback=False, save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert np.isfinite(native.cv_best_score)
    assert native.cv_best["model_name"] == "MetaModel_LogisticRegression"


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("test_aggregation", [FoldAggregation.MEAN, FoldAggregation.BEST_FOLD, FoldAggregation.WEIGHTED_MEAN])
def test_named_probability_sources_replay_from_archive(tmp_path, mechanism, test_aggregation, monkeypatch):
    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        RandomForestClassifier(n_estimators=10, random_state=42),
        {"model": MetaModel(
            model=LogisticRegression(max_iter=300), use_proba=True,
            source_models=["RandomForestClassifier", "LogisticRegression"],
            stacking_config=StackingConfig(test_aggregation=test_aggregation),
        )},
    ]
    path = dataset_path("binary")
    native = nirs4all.run(
        pipeline, path, engine="dag-ml", allow_fallback=False,
        workspace_path=tmp_path / "train", save_artifacts=False,
        save_charts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        archive = native.export(tmp_path / "meta.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        assert replay.shape == y_test.shape
        assert np.mean(replay == y_test) == pytest.approx(native.best_accuracy, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("level_count", [2, 3])
def test_named_metamodel_chain_uses_native_nested_oof_and_archive(tmp_path, mechanism, level_count, monkeypatch):
    """Both DAG transports retain each legacy named source and nested OOF scope."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.model_selection import KFold

    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    features, targets = make_regression(
        n_samples=48, n_features=6, noise=0.1, random_state=42,
    )
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2),
        Ridge(alpha=100),
        {"model": MetaModel(Ridge(), source_models=["PLSRegression", "Ridge"], name="first")},
        {"model": MetaModel(Lasso(alpha=0.1), source_models=["first"], name="second")},
    ]
    if level_count == 3:
        pipeline.append({"model": MetaModel(Ridge(alpha=2), source_models=["second"], name="third")})
    legacy = nirs4all.run(
        pipeline, (features, targets), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert np.isfinite(legacy.cv_best_score)
    finally:
        legacy.close()

    native = nirs4all.run(
        pipeline, (features, targets), engine="dag-ml", allow_fallback=False, refit=True,
        workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        from nirs4all.pipeline.dagml.native_results import read_native_results

        persisted = read_native_results(native._dagml_results_dir)
        replay_manifest = persisted["manifest"]["stacking_replay"]
        assert replay_manifest["schema_version"] == 2
        stages = replay_manifest["stages"]
        assert len(stages) == level_count
        first_stage = stages[0]
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        assert set(by_id) == {
            *(producer["artifact_id"] for producer in first_stage["base_producers"]),
            *(stage["meta_artifact_id"] for stage in stages),
        }
        base_features = np.column_stack([
            np.asarray(by_id[producer["artifact_id"]]["estimator"].predict(features[:7])).reshape(7, -1)
            for producer in first_stage["base_producers"]
        ])
        expected = np.asarray(by_id[first_stage["meta_artifact_id"]]["estimator"].predict(base_features)).reshape(7, -1)
        for previous, stage in zip(stages, stages[1:], strict=False):
            assert stage["base_producers"][0]["artifact_id"] == previous["meta_artifact_id"]
            expected = np.asarray(by_id[stage["meta_artifact_id"]]["estimator"].predict(expected)).reshape(7, -1)
        archive = native.export(tmp_path / f"{level_count}-level.n4a")
        replay = np.asarray(nirs4all.predict(archive, features[:7]).y_pred).reshape(-1)
        assert replay.shape == (7,)
        assert np.all(np.isfinite(replay))
        np.testing.assert_allclose(replay, expected.reshape(-1))
    finally:
        native.close()
