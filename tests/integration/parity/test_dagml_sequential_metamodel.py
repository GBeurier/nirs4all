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
@pytest.mark.parametrize("first_sources", ["all", ["RandomForestClassifier", "LogisticRegression"]])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_named_classifier_meta_probability_chain_replays_selected_class(tmp_path, mechanism, first_sources, n_classes, monkeypatch):
    """Downstream use_proba consumes legacy's selected upstream class column."""
    import nirs4all
    from nirs4all.pipeline.dagml.native_results import read_native_results

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    rng = np.random.default_rng(23)
    features = rng.normal(size=(42 if n_classes == 2 else 66, 6))
    targets = (
        (features[:, 0] + features[:, 1] > 0).astype(int)
        if n_classes == 2 else np.argmax(features[:, :3] + rng.normal(scale=0.15, size=(66, 3)), axis=1)
    )
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=1),
        LogisticRegression(max_iter=300),
        RandomForestClassifier(n_estimators=8, random_state=1),
        {"model": MetaModel(LogisticRegression(max_iter=300), source_models=first_sources, use_proba=True, name="first")},
        {"model": MetaModel(LogisticRegression(max_iter=300), source_models=["first"], use_proba=True, name="second")},
    ]
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
        assert np.isfinite(native.cv_best_score)
        # DAG-ML builds fresh nested OOF features at each level; legacy reuses
        # earlier CV predictions, so their CV scores are not equality oracles.
        if mechanism == "pyo3":
            upstream_oof = [
                np.asarray(block["values"])
                for node in native._dagml_node_results for block in node.get("predictions", [])
                if block.get("producer_node") == "merge:stack" and block.get("partition") == "validation"
            ]
            assert upstream_oof and all(block.ndim == 2 and block.shape[1] == 1 for block in upstream_oof)
            assert all(np.all((0 <= block) & (block <= 1)) for block in upstream_oof)
        persisted = read_native_results(native._dagml_results_dir)
        first_stage, second_stage = persisted["manifest"]["stacking_replay"]["stages"]
        assert second_stage["base_producers"][0]["column_block"] == "probability_values"
        assert second_stage["base_producers"][0]["artifact_id"] == first_stage["meta_artifact_id"]
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        base_probabilities = [
            np.asarray(by_id[producer["artifact_id"]]["estimator"].predict_proba(features[:7]))
            for producer in first_stage["base_producers"]
        ]
        groups = first_stage["reduction_groups"]
        assert all(group["proba"] and len(group["members"]) == 1 for group in groups)
        first_features = np.column_stack([base_probabilities[group["members"][0]] for group in groups])
        class_column = 1 if n_classes == 2 else 0
        first_probability = np.asarray(by_id[first_stage["meta_artifact_id"]]["estimator"].predict_proba(first_features))[:, class_column:class_column + 1]
        second_estimator = by_id[second_stage["meta_artifact_id"]]["estimator"]
        assert second_estimator.n_features_in_ == 1
        expected = np.asarray(second_estimator.predict(first_probability)).reshape(-1)
        archive = native.export(tmp_path / "probability_chain.n4a")
        replay = np.asarray(nirs4all.predict(archive, features[:7]).y_pred).reshape(-1)
        np.testing.assert_allclose(replay, expected)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("test_aggregation", [FoldAggregation.BEST_FOLD, FoldAggregation.WEIGHTED_MEAN])
def test_second_named_meta_fold_aggregation_replays_paired_cv_stacks(tmp_path, mechanism, test_aggregation, monkeypatch):
    """Second-stage Test features replay each first meta with its own base fold estimators."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    import nirs4all
    from nirs4all.pipeline.dagml.native_results import read_native_results

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2), Ridge(alpha=10000),
        {"model": MetaModel(Ridge(alpha=1), name="first")},
        {"model": MetaModel(Ridge(alpha=2), name="second", source_models=["first"],
                             stacking_config=StackingConfig(test_aggregation=test_aggregation))},
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
    finally:
        legacy.close()
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True,
                         save_charts=False, verbose=0)
    try:
        assert np.isfinite(native.best_rmse)
        persisted = read_native_results(native._dagml_results_dir)
        stages = persisted["manifest"]["stacking_replay"]["stages"]
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        first_meta = by_id[stages[0]["meta_artifact_id"]]
        second_meta = by_id[stages[1]["meta_artifact_id"]]
        bases = [by_id[producer["artifact_id"]] for producer in stages[0]["base_producers"]]
        folds = first_meta["fold_estimators"]
        assert folds and all(set(base["fold_estimators"]) == set(folds) for base in bases)
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        fold_predictions = {
            fold: np.asarray(meta.predict(np.column_stack([
                np.asarray(base["fold_estimators"][fold].predict(x_test)).reshape(len(x_test), -1)
                for base in bases
            ]))).reshape(-1)
            for fold, meta in folds.items()
        }
        selection = first_meta["fold_selection"]
        if test_aggregation == FoldAggregation.BEST_FOLD:
            first_test = fold_predictions[selection["selected_fold"]]
        else:
            first_test = sum(float(weight) * fold_predictions[fold]
                             for fold, weight in selection["weights"].items())
        expected = np.asarray(second_meta["estimator"].predict(first_test.reshape(-1, 1))).ravel()
        archive = native.export(tmp_path / "selected_meta.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, expected)
        assert np.sqrt(np.mean((y_test - replay) ** 2)) == pytest.approx(native.best_rmse, rel=1e-6, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("test_aggregation", [FoldAggregation.BEST_FOLD, FoldAggregation.WEIGHTED_MEAN])
def test_second_named_classifier_probability_fold_aggregation_replays_paired_stacks(tmp_path, mechanism, test_aggregation, monkeypatch):
    """A fold-selected classifier consumes the first meta's selected probability column."""
    import nirs4all
    from nirs4all.pipeline.dagml.native_results import read_native_results

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300), RandomForestClassifier(n_estimators=10, random_state=42),
        {"model": MetaModel(LogisticRegression(max_iter=300), name="first", use_proba=True)},
        {"model": MetaModel(LogisticRegression(max_iter=300), name="second", source_models=["first"],
                             use_proba=True, stacking_config=StackingConfig(test_aggregation=test_aggregation))},
    ]
    path = dataset_path("binary")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
    finally:
        legacy.close()
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True,
                         save_charts=False, verbose=0)
    try:
        assert np.isfinite(native.best_accuracy)
        persisted = read_native_results(native._dagml_results_dir)
        first_stage, second_stage = persisted["manifest"]["stacking_replay"]["stages"]
        assert second_stage["base_producers"][0]["column_block"] == "probability_values"
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        bases = [by_id[producer["artifact_id"]] for producer in first_stage["base_producers"]]
        first_meta = by_id[first_stage["meta_artifact_id"]]
        second_meta = by_id[second_stage["meta_artifact_id"]]
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        fold_probabilities = {}
        for fold, meta in first_meta["fold_estimators"].items():
            base_probabilities = [np.asarray(base["fold_estimators"][fold].predict_proba(x_test)) for base in bases]
            features = np.column_stack([base_probabilities[group["members"][0]] for group in first_stage["reduction_groups"]])
            fold_probabilities[fold] = np.asarray(meta.predict_proba(features))[:, 1:2]
        selection = first_meta["fold_selection"]
        if test_aggregation == FoldAggregation.BEST_FOLD:
            first_test = fold_probabilities[selection["selected_fold"]]
        else:
            first_test = sum(float(weight) * fold_probabilities[fold]
                             for fold, weight in selection["weights"].items())
        expected = np.asarray(second_meta["estimator"].predict(first_test)).ravel()
        archive = native.export(tmp_path / "selected_classifier.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_array_equal(replay, expected)
        assert np.mean(replay == y_test) == pytest.approx(native.best_accuracy, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("test_aggregation", [FoldAggregation.BEST_FOLD, FoldAggregation.WEIGHTED_MEAN])
def test_multiclass_named_probability_fold_aggregation_scores_and_replays(tmp_path, mechanism, test_aggregation, monkeypatch):
    """Score all class probabilities while downstream stacking reads only the legacy first class."""
    import nirs4all
    from nirs4all.pipeline.dagml.native_results import read_native_results

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    rng = np.random.default_rng(204)
    y_train = np.repeat(np.arange(3), 24)
    y_test = np.repeat(np.arange(3), 6)
    x_train = rng.normal(size=(len(y_train), 6)) + np.eye(3, 6)[y_train] * 0.8
    x_test = rng.normal(size=(len(y_test), 6)) + np.eye(3, 6)[y_test] * 0.8
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split, features, labels in (("train", x_train, y_train), ("test", x_test, y_test)):
        np.savetxt(dataset / f"X{split}.csv", features, delimiter=";", header=";".join(str(1000 + i) for i in range(6)), comments="")
        np.savetxt(dataset / f"Y{split}.csv", labels, delimiter=";", header="class", comments="", fmt="%d")

    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300), RandomForestClassifier(n_estimators=10, random_state=42),
        {"model": MetaModel(LogisticRegression(max_iter=300), name="first", use_proba=True)},
        {"model": MetaModel(LogisticRegression(max_iter=300), name="second", source_models=["first"],
                             use_proba=True, stacking_config=StackingConfig(test_aggregation=test_aggregation))},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
    finally:
        legacy.close()

    native = nirs4all.run(pipeline, dataset, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True,
                         save_charts=False, verbose=0)
    try:
        persisted = read_native_results(native._dagml_results_dir)
        score_set = persisted["score_set"]["reports"]
        first_oof = next(report for report in score_set if report["producer_node"] == "merge:stack"
                         and report["partition"] == "validation" and report["fold_id"] == "avg")
        # The old scalar-score path treated the first class probability as a label
        # and reported about 0.28-0.31; the three-class argmax gives 11/24.
        assert first_oof["metrics"]["accuracy"] == pytest.approx(11 / 24)
        assert first_oof["metrics"]["balanced_accuracy"] == pytest.approx(11 / 24)
        if mechanism == "pyo3":
            distributions = [
                block for result in native._dagml_node_results
                for block in result.get("classification_probabilities", [])
                if block["producer_node"] == "merge:stack"
                and block["partition"] == "validation"
                and block["fold_id"] in {"fold0", "fold1"}
            ]
            assert len(distributions) == 2
            assert all(len(block["class_labels"]) == 3 and all(len(row) == 3 for row in block["values"])
                       for block in distributions)

        first_stage, second_stage = persisted["manifest"]["stacking_replay"]["stages"]
        assert second_stage["base_producers"][0]["column_block"] == "probability_values"
        by_id = {artifact["artifact_id"]: artifact for artifact in persisted["artifacts"]}
        bases = [by_id[producer["artifact_id"]] for producer in first_stage["base_producers"]]
        first_meta = by_id[first_stage["meta_artifact_id"]]
        second_meta = by_id[second_stage["meta_artifact_id"]]
        fold_probabilities = {}
        for fold, meta in first_meta["fold_estimators"].items():
            base_probabilities = [np.asarray(base["fold_estimators"][fold].predict_proba(x_test)) for base in bases]
            features = np.column_stack([base_probabilities[group["members"][0]] for group in first_stage["reduction_groups"]])
            fold_probabilities[fold] = np.asarray(meta.predict_proba(features))[:, :1]
        selection = first_meta["fold_selection"]
        if test_aggregation == FoldAggregation.BEST_FOLD:
            first_test = fold_probabilities[selection["selected_fold"]]
        else:
            first_test = sum(float(weight) * fold_probabilities[fold]
                             for fold, weight in selection["weights"].items())
        expected = np.asarray(second_meta["estimator"].predict(first_test)).ravel()
        archive = native.export(tmp_path / "multiclass_selected.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_array_equal(replay, expected)
        assert np.mean(replay == y_test) == pytest.approx(native.best_accuracy, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("level_count", [2, 3])
@pytest.mark.parametrize("regression_probability_flag", [False, True])
def test_named_metamodel_chain_uses_native_nested_oof_and_archive(tmp_path, mechanism, level_count, regression_probability_flag, monkeypatch):
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
        {"model": MetaModel(Lasso(alpha=0.1), source_models=["first"], use_proba=regression_probability_flag, name="second")},
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
        assert stages[1]["base_producers"][0]["column_block"] == "prediction_values"
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
