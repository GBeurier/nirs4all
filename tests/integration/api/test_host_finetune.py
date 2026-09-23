"""Host optimizer proposals with true DAG evaluation and isolated outer folds."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _data():
    rng = np.random.default_rng(234)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    return X, (X @ np.arange(1.0, 7.0) + rng.normal(scale=0.25, size=40)).astype(np.float32)


def _run(X, y, tmp_path, *, sampler="grid", model_params=None, n_trials=2, approach="single", eval_mode="best", splitter=None):
    import nirs4all

    return nirs4all.run(
        [StandardScaler(), splitter if splitter is not None else ShuffleSplit(2, test_size=0.25, random_state=42),
         {"model": PLSRegression(), "finetune_params": {
             "approach": approach, "eval_mode": eval_mode, "sampler": sampler, "verbose": 0, "seed": 42,
             "n_trials": n_trials, "model_params": model_params or {"n_components": [1, 2]},
         }}],
        (X, y), workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0,
    )


@pytest.mark.parametrize("sampler,space", [("grid", {"n_components": [1, 2]}), ("tpe", {"n_components": ("int", 1, 3)})])
def test_host_search_real_scores_selection_and_export(tmp_path, monkeypatch, sampler, space):
    import nirs4all
    from nirs4all.optimization.optuna import OptunaManager

    monkeypatch.setattr(OptunaManager, "finetune", lambda *args, **kwargs: pytest.fail("legacy optimizer execution"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler"))
    X, y = _data()
    result = _run(X, y, tmp_path, sampler=sampler, model_params=space)
    artifact = result._dagml_refit_artifacts[0]
    fitted = artifact["estimator"]
    evidence = fitted._nirs4all_host_hpo
    assert evidence["profile"] == "host_optimizer_search_v1" and evidence["portable"] is False
    assert evidence["scope"]["phase"] == "REFIT"
    assert len(evidence["trials"]) == 2
    assert evidence["evaluation"] == {"role": "inner_parameter_selection", "outer_validation_used": False, "test_used": False,
                                      "approach": "single", "inner_fold_count": 1, "score_reduction": "native_holdout"}
    train, val = next(ShuffleSplit(1, test_size=0.2, random_state=42).split(X))
    for trial in evidence["trials"]:
        expected = make_pipeline(StandardScaler(), PLSRegression(**trial["params"])).fit(X[train], y[train].astype(float))
        score = np.sqrt(np.mean((expected.predict(X[val]).ravel() - y[val]) ** 2))
        assert trial["score"] == pytest.approx(score, rel=1e-7, abs=1e-7)
        assert trial["scores"]["reports"]
    winner = min(evidence["trials"], key=lambda trial: (trial["score"], trial["trial_index"]))
    assert evidence["selected_params"] == winner["params"]
    assert fitted.steps[-1][1].get_params()["n_components"] == winner["params"]["n_components"]
    expected = fitted.predict(X[:7]).ravel()
    monkeypatch.setattr(PLSRegression, "fit", lambda *args, **kwargs: pytest.fail("export/replay fitted"))
    archive = tmp_path / "tuned.n4a"
    result.export(archive)
    np.testing.assert_array_equal(nirs4all.predict(archive, X[:7]).y_pred, expected)
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    recorded = load_general_archive(archive)["manifest"]["host_hpo"]
    assert recorded["portable"] is False
    assert [search["scope"]["phase"] for search in recorded["searches"]] == ["FIT_CV", "FIT_CV", "REFIT"]
    result.close()


@pytest.mark.parametrize("approach", ["single", "grouped", "individual"])
def test_outer_validation_poison_cannot_change_tuning_or_prediction(tmp_path, monkeypatch, approach):
    from nirs4all.pipeline.dagml import host_finetune

    X, y = _data()
    _, val = next(ShuffleSplit(2, test_size=0.25, random_state=42).split(X))
    original = host_finetune.run_scoped_finetune
    recorded = []

    def capture(*args, **kwargs):
        answer = original(*args, **kwargs)
        if kwargs["scope"]["fold_id"] == "fold0":
            recorded.append(answer)
        return answer

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", capture)
    first = _run(X, y, tmp_path / "first", approach=approach)
    poisoned = y.copy()
    poisoned[val] += 10000
    second = _run(X, poisoned, tmp_path / "second", approach=approach)
    assert len(recorded) == 2
    assert recorded[0][0] == recorded[1][0]
    assert [trial["score"] for trial in recorded[0][1]["trials"]] == [trial["score"] for trial in recorded[1][1]["trials"]]
    rows = [result.predictions.filter_predictions(partition="val", fold_id="0", load_arrays=True)[0] for result in (first, second)]
    np.testing.assert_array_equal(rows[0]["y_pred"], rows[1]["y_pred"])
    first.close()
    second.close()


def test_grid_exhaustion_stops_before_budget_without_duplicate_trials(tmp_path):
    X, y = _data()
    result = _run(X, y, tmp_path, n_trials=5)
    evidence = result._dagml_refit_artifacts[0]["estimator"]._nirs4all_host_hpo
    assert len(evidence["trials"]) == 2
    assert {trial["params"]["n_components"] for trial in evidence["trials"]} == {1, 2}
    result.close()


def test_failed_inner_fit_propagates_once_without_retry(tmp_path, monkeypatch):
    X, y = _data()
    calls = []

    def fail_fit(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("deliberate inner candidate failure")

    monkeypatch.setattr(PLSRegression, "fit", fail_fit)
    with pytest.raises(Exception, match="deliberate inner candidate failure"):
        _run(X, y, tmp_path, n_trials=5)
    assert calls == [1]


@pytest.mark.parametrize("eval_mode", ["best", "mean", "robust_best"])
@pytest.mark.parametrize("splitter", [KFold(3, shuffle=True, random_state=19), ShuffleSplit(2, test_size=0.25, random_state=42)])
def test_grouped_uses_exact_native_fold_reduction_and_inner_policy(tmp_path, monkeypatch, eval_mode, splitter):
    import nirs4all

    X, y = _data()
    result = _run(X, y, tmp_path, approach="grouped", eval_mode=eval_mode, splitter=splitter)
    fitted = result._dagml_refit_artifacts[0]["estimator"]
    evidence = fitted._nirs4all_host_hpo
    expected_folds = list(splitter.split(X, y))
    assert evidence["evaluation"]["inner_fold_count"] == len(expected_folds)
    for observed, (train, val) in zip(evidence["inner_cv"]["folds"], expected_folds, strict=True):
        np.testing.assert_array_equal(observed[0], train)
        np.testing.assert_array_equal(observed[1], val)
    for trial in evidence["trials"]:
        independent = []
        for train, val in expected_folds:
            model = make_pipeline(StandardScaler(), PLSRegression(**trial["params"])).fit(X[train], y[train].astype(float))
            independent.append(float(np.sqrt(np.mean((model.predict(X[val]).ravel() - y[val]) ** 2))))
        np.testing.assert_allclose(list(trial["objective_fold_scores"].values()), independent, rtol=1e-7, atol=1e-7)
        expected_score = np.mean(independent) if eval_mode == "mean" else min(independent)
        assert trial["score"] == pytest.approx(expected_score, rel=1e-7, abs=1e-7)
        assert trial["scores"]["reports"], "native reports remain evidence, not replaced by aggregate objectives"
    for search in fitted._nirs4all_host_hpo_history:
        outer_train = set(search["scope"]["training_sample_ids"])
        for fold in search["inner_cv"]["source_fold_rows"]:
            assert set(fold["train"]).isdisjoint(fold["validation"])
            assert set(fold["train"]) | set(fold["validation"]) <= outer_train
    expected = fitted.predict(X[:5]).ravel()
    monkeypatch.setattr(PLSRegression, "fit", lambda *args, **kwargs: pytest.fail("export cannot refit grouped winner"))
    archive = tmp_path / "grouped.n4a"
    result.export(archive)
    np.testing.assert_array_equal(nirs4all.predict(archive, X[:5]).y_pred, expected)
    result.close()


def test_individual_keeps_separate_fold_studies_and_fresh_refit_search(tmp_path):
    X, y = _data()
    result = _run(X, y, tmp_path, approach="individual", splitter=KFold(3, shuffle=True, random_state=19))
    history = result._dagml_refit_artifacts[0]["estimator"]._nirs4all_host_hpo_history
    assert len(history) == 4
    assert [search["scope"]["phase"] for search in history] == ["FIT_CV"] * 3 + ["REFIT"]
    assert all(search["evaluation"]["approach"] == "individual" for search in history)
    assert all(search["evaluation"]["inner_fold_count"] == 1 for search in history)
    assert len({tuple(search["scope"]["training_sample_ids"]) for search in history}) == 4
    assert all(len(search["trials"]) == 2 for search in history)
    result.close()


def test_grouped_classification_keeps_subjects_disjoint_and_maximizes_native_accuracy(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    import nirs4all
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.dagml.identity import mint_identity

    rng = np.random.default_rng(921)
    groups = np.repeat(np.arange(20), 4)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    y = (groups % 2).astype(float)
    X[:, 0] += y
    dataset = SpectroDataset("grouped-classification")
    dataset.set_task_type("binary_classification")
    dataset.add_samples(X, indexes={"partition": "train"})
    dataset.add_targets(y)
    dataset.add_metadata(groups.reshape(-1, 1), headers=["subject"])
    identity = mint_identity(dataset)
    result = nirs4all.run(
        [StandardScaler(), {"split": GroupKFold(3), "group_by": "subject"},
         {"model": LogisticRegression(max_iter=300), "finetune_params": {
             "approach": "grouped", "sampler": "grid", "n_trials": 2,
             "model_params": {"C": [0.1, 10.0]}, "metric": "balanced_accuracy", "eval_mode": "best",
         }}], dataset, save_artifacts=False, save_charts=False, verbose=0, workspace_path=tmp_path,
    )
    fitted = result._dagml_refit_artifacts[0]["estimator"]
    for search in fitted._nirs4all_host_hpo_history:
        for fold in search["inner_cv"]["source_fold_rows"]:
            train_groups = {groups[identity.to_int(sample)] for sample in fold["train"]}
            val_groups = {groups[identity.to_int(sample)] for sample in fold["validation"]}
            assert train_groups.isdisjoint(val_groups)
    final = fitted._nirs4all_host_hpo
    for trial in final["trials"]:
        fold_scores = []
        for train, val in GroupKFold(3).split(X, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, **trial["params"])).fit(X[train], y[train])
            fold_scores.append(balanced_accuracy_score(y[val], model.predict(X[val])))
        assert trial["score"] == pytest.approx(max(fold_scores))
    winner = max(final["trials"], key=lambda trial: (trial["score"], -trial["trial_index"]))
    assert final["selected_params"] == winner["params"]
    result.close()


def test_private_inner_splitter_context_cannot_be_injected_by_public_config():
    from nirs4all.pipeline.dagml.host_finetune import validate_host_finetune

    with pytest.raises(NotImplementedError, match="__dagml_inner_splitter"):
        validate_host_finetune({"approach": "grouped", "n_trials": 2, "model_params": {"alpha": [1]},
                                "__dagml_inner_splitter": {"operator_json": "untrusted"}})


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("approach", ["single", "grouped", "individual"])
def test_n4m_finetune_params_use_native_scoped_search_and_export(tmp_path, monkeypatch, mechanism, approach):
    pytest.importorskip("n4m")
    import nirs4all
    from nirs4all.optimization.n4m_engine import N4MFinetuneManager

    X, y = _data()
    pipeline = [KFold(2), {"model": PLSRegression(), "finetune_params": {
        "engine": "n4m", "sampler": "random", "seed": 7, "n_trials": 2, "approach": approach,
        "model_params": {"n_components": [1, 2]},
    }}]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    monkeypatch.setattr(N4MFinetuneManager, "finetune", lambda *args, **kwargs: pytest.fail("legacy n4m CV execution"))
    result = nirs4all.run(pipeline, (X, y), engine="dag-ml", save_charts=False)
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001
    history = fitted._nirs4all_host_hpo_history
    assert [search["scope"]["phase"] for search in history] == ["FIT_CV", "FIT_CV", "REFIT"]
    assert all(search["optimizer"]["name"] == "n4m" and len(search["trials"]) == 2 for search in history)
    for search in history:
        assert search["evaluation"]["approach"] == approach
        assert search["evaluation"]["inner_fold_count"] == (2 if approach == "grouped" else 1)
        assert search["selected_params"] in [trial["params"] for trial in search["trials"]]
        assert search["evaluation"]["outer_validation_used"] is False
    expected = fitted.predict(X[:4]).ravel()
    archive = result.export(tmp_path / "n4m_finetuned.n4a")
    np.testing.assert_allclose(nirs4all.predict(archive, X[:4]).y_pred, expected)
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_optuna_storage_keeps_outer_training_scopes_separate(tmp_path, monkeypatch, mechanism):
    pytest.importorskip("optuna")
    import nirs4all

    X, y = _data()
    storage = f"sqlite:///{tmp_path / 'scoped_optuna.sqlite3'}"
    base = {"engine": "optuna", "approach": "grouped", "sampler": "random", "seed": 7,
            "n_trials": 2, "model_params": {"n_components": [1, 2]}, "storage": storage}
    legacy = nirs4all.run(
        [KFold(2), {"model": PLSRegression(), "finetune_params": {**base, "study_name": "legacy"}}],
        (X, y), engine="legacy", save_charts=False,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [KFold(2), {"model": PLSRegression(), "finetune_params": {**base, "study_name": "dag"}}]
    result = nirs4all.run(pipeline, (X, y), engine="dag-ml", save_charts=False)
    history = result._dagml_refit_artifacts[0]["estimator"]._nirs4all_host_hpo_history  # noqa: SLF001
    names = [search["optimizer"]["study_name"] for search in history]
    assert len(names) == len(set(names)) == 3
    assert all(name.startswith("dag:scope:") for name in names)
    assert np.isfinite(result.cv_best_score)
    result.close()

    with pytest.raises(Exception, match="paired native DAG trial checkpoint"):
        nirs4all.run(
            [KFold(2), {"model": PLSRegression(), "finetune_params": {**base, "study_name": "dag", "resume": True}}],
            (X, y), engine="dag-ml", save_charts=False,
        )


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_optuna_phases_follow_native_budgets_and_replay(tmp_path, monkeypatch, mechanism):
    pytest.importorskip("optuna")
    import nirs4all
    from nirs4all.optimization.optuna import OptunaManager

    X, y = _data()
    settings = {"engine": "optuna", "approach": "grouped", "sampler": "grid", "seed": 7,
                "n_trials": 1, "phases": [{"n_trials": 1, "sampler": "random"},
                                          {"n_trials": 2, "sampler": "tpe"}],
                "model_params": {"n_components": ["int", 1, 3]}}
    pipeline = [KFold(2), {"model": PLSRegression(), "finetune_params": settings}]
    legacy = nirs4all.run(pipeline, (X, y), engine="legacy", save_charts=False)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    if mechanism == "subprocess":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    original_sampler = OptunaManager._create_sampler  # noqa: SLF001
    sampler_calls = []

    def record_sampler(self, sampler_type, params, seed=None):
        sampler_calls.append(sampler_type)
        return original_sampler(self, sampler_type, params, seed)

    monkeypatch.setattr(OptunaManager, "_create_sampler", record_sampler)
    result = nirs4all.run(pipeline, (X, y), engine="dag-ml", save_charts=False)
    history = result._dagml_refit_artifacts[0]["estimator"]._nirs4all_host_hpo_history  # noqa: SLF001
    assert len(history) == 3
    assert all(len(search["trials"]) == 3 for search in history)
    assert all(search["optimizer"]["phase_trial_budgets"] == [1, 2] for search in history)
    if mechanism == "in_process":
        assert sampler_calls.count("random") == sampler_calls.count("tpe") == 3
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001
    expected = fitted.predict(X[:4]).ravel()
    archive = result.export(tmp_path / "phased.n4a")
    np.testing.assert_allclose(nirs4all.predict(archive, X[:4]).y_pred, expected)
    result.close()
