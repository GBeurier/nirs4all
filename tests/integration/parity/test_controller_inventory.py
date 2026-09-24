"""Executable legacy-controller coverage boundaries for DAG-ML."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from nirs4all.controllers.registry import CONTROLLER_REGISTRY


def test_coverage_manifest_tracks_every_registered_legacy_controller() -> None:
    manifest = json.loads((Path(__file__).with_name("controller_coverage.json")).read_text(encoding="utf-8"))
    expected = {cls.__name__ for cls in CONTROLLER_REGISTRY}
    assert set(manifest["controllers"]) == expected
    for name, entry in manifest["controllers"].items():
        assert entry["route"] in {"native_graph", "host_operator", "host_preparation", "host_presentation", "gap", "fallback"}, name
        assert entry["coverage"] in {"public", "partial", "none", "not_pipeline"}, name
        for path in entry.get("evidence", []):
            assert (Path(__file__).parents[3] / path).is_file(), (name, path)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_fold_file_loader_imports_sample_ids_into_native_foldset(tmp_path, monkeypatch, mechanism: str) -> None:
    """A saved legacy fold file yields the same CV sample partition in DAG-ML."""
    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    features = np.arange(60, dtype=float).reshape(15, 4) / 20
    features[:, 1] = np.sin(features[:, 1])
    target = 1 + features[:, 0] * 2 + features[:, 1] * 3
    dataset = SpectroDataset("nonpositional_fold_ids")
    headers = [str(column) for column in range(4)]
    dataset.add_samples(features[:3], {"partition": "test"}, headers=headers)
    dataset.add_samples(features[3:], {"partition": "train"}, headers=headers)
    dataset.add_targets(target.reshape(-1, 1))
    train_ids = list(map(int, dataset.index_column("sample", {"partition": "train"})))
    assert train_ids[0] == 3  # file IDs are absolute, not offsets into the train matrix
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps([
        {"train": train_ids[:6], "val": train_ids[6:]},
        {"train": train_ids[6:], "val": train_ids[:6]},
    ]), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, {"model": Ridge(alpha=1.0)}]

    legacy = nirs4all.run(
        pipeline, dataset, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset, engine="dag-ml", refit=False,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("omit_train_rows", [False, True])
@pytest.mark.parity
def test_single_fold_file_becomes_test_holdout_with_native_replay(tmp_path, monkeypatch, mechanism: str, omit_train_rows: bool) -> None:
    """A single file fold is a held-out test, even when it omits train rows."""
    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    features = np.arange(60, dtype=float).reshape(15, 4) / 20
    features[:, 1] = np.sin(features[:, 1])
    target = 1 + features[:, 0] * 2 + features[:, 1] * 3
    dataset = SpectroDataset("single_fold_holdout")
    dataset.add_samples(features, {"partition": "train"}, headers=[str(column) for column in range(4)])
    dataset.add_targets(target.reshape(-1, 1))
    sample_ids = list(map(int, dataset.index_column("sample", {"partition": "train"})))
    train_ids = sample_ids[:8] if omit_train_rows else sample_ids[:10]
    test_ids = sample_ids[10:]
    fold_file = tmp_path / "one_fold.json"
    fold_file.write_text(json.dumps([{"train": train_ids, "val": test_ids}]), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, {"model": Ridge(alpha=1.0)}]

    legacy = nirs4all.run(
        pipeline, dataset, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    legacy_train = legacy.predictions.filter_predictions(partition="train", load_arrays=True)[0]
    legacy_test = legacy.predictions.filter_predictions(partition="test", load_arrays=True)[0]
    assert legacy_train["sample_indices"] == train_ids
    assert legacy_test["sample_indices"] == test_ids
    assert np.isnan(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset, engine="dag-ml", refit=False,
        workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0,
    )
    native_train = native.predictions.filter_predictions(partition="train", load_arrays=True)[0]
    native_test = native.predictions.filter_predictions(partition="test", load_arrays=True)[0]
    assert native.execution_engine == "dag-ml"
    assert native_train["sample_indices"] == train_ids
    assert native_test["sample_indices"] == test_ids
    assert np.isnan(native.cv_best_score)  # no fictitious OOF score
    assert native.best_rmse == pytest.approx(legacy_test["test_score"], abs=1e-6)
    np.testing.assert_allclose(np.asarray(native_test["y_pred"]).ravel(), np.asarray(legacy_test["y_pred"]).ravel(), atol=1e-6)
    assert native.per_dataset[dataset.name]["fold_file_holdout"] is True

    archive = native.export(tmp_path / "single_fold.n4a")
    replay = nirs4all.predict(archive, features[10:])
    np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), np.asarray(native_test["y_pred"]).ravel(), atol=1e-6)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("gate", [False, 0.25])
@pytest.mark.parametrize("preprocessing", [False, True])
def test_residual_model_native_graph_fits_oof_targets_and_fuses_predictions(tmp_path, gate, preprocessing) -> None:
    """The learner fits OOF-derived residuals; DAG-ML fuses held-out predictions."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    import nirs4all
    from nirs4all.operators.models.residual import ResidualModel

    from ._datasets import dataset_path

    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        *([StandardScaler()] if preprocessing else []),
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(alpha=1.0), gate=gate)},
    ]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / f"legacy-residual-{gate}-{preprocessing}", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    assert np.isfinite(legacy.best_rmse)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=True,
        workspace_path=tmp_path / f"native-residual-{gate}-{preprocessing}", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    frames = native._dagml_node_results
    by_node_fold = {
        (frame["node_id"], prediction["partition"], prediction.get("fold_id")): prediction
        for frame in frames for prediction in frame.get("predictions", [])
    }
    fusion_id = "model:residual.learner.residual_fusion"
    fusion_rows = 0
    for (node_id, partition, fold_id), fused in by_node_fold.items():
        if node_id != fusion_id or partition not in {"validation", "test"}:
            continue
        fusion_rows += len(fused["sample_ids"])
        base = by_node_fold[("branch:0.node:0", partition, fold_id)]
        learner = by_node_fold[("model:residual.learner", partition, fold_id)]
        base_rows = dict(zip(base["sample_ids"], base["values"], strict=True))
        learner_rows = dict(zip(learner["sample_ids"], learner["values"], strict=True))
        for sample_id, fused_row in zip(fused["sample_ids"], fused["values"], strict=True):
            expected = np.asarray(base_rows[sample_id]) + float(gate if gate is not False else 1) * np.asarray(learner_rows[sample_id])
            assert fused_row == pytest.approx(expected, abs=1e-8)
    if any(frame.get("node_id") == fusion_id for frame in frames):  # CLI bundles omit merge-node frames.
        assert fusion_rows > 0
    from nirs4all.data import DatasetConfigs

    archive = native.export(tmp_path / f"residual_{gate}_{preprocessing}.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
def test_residual_learner_finetune_search_uses_native_train_scope(tmp_path) -> None:
    """A legacy learner search also executes on the DAG residual train scope."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    import nirs4all
    from nirs4all.operators.models.residual import ResidualModel

    from ._datasets import dataset_path

    search = {"n_trials": 2, "sampler": "grid", "approach": "single", "model_params": {"alpha": [0.01, 1.0]}}
    pipeline = [KFold(2, shuffle=True, random_state=1), {
        "model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=False, finetune_space=search),
    }]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy-residual-search", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native-residual-search", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(native.cv_best_score)
    evidence = [getattr(artifact["estimator"], "_nirs4all_host_hpo", None) for artifact in native._dagml_refit_artifacts]
    searches = [entry for entry in evidence if entry is not None]
    assert len(searches) == 1
    assert {trial["params"]["alpha"] for trial in searches[0]["trials"]} == {0.01, 1.0}
    assert searches[0]["evaluation"]["outer_validation_used"] is False
    from nirs4all.data import DatasetConfigs

    archive = native.export(tmp_path / "residual_search.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    # Float32 Ridge predictions can shift slightly when DAG replay orders the same batch differently.
    assert replay_rmse == pytest.approx(native.best_rmse, rel=1e-7, abs=1e-4)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("preprocessing", [False, True])
def test_residual_auto_gate_calibrates_on_nested_oof_and_replays(tmp_path, monkeypatch, mechanism, preprocessing) -> None:
    """The automatic gate comes from nested learner OOF and survives refit/export."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    import nirs4all
    from nirs4all.data import DatasetConfigs
    from nirs4all.operators.models.residual import ResidualModel

    from ._datasets import dataset_path

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [KFold(2, shuffle=True, random_state=1), *([StandardScaler()] if preprocessing else []), {
        "model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate="auto"),
    }]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy-auto", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native-auto", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    replay_meta = native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]
    gate = replay_meta["gate"]
    assert isinstance(gate, float) and 0 <= gate <= 1
    assert {record["fold_id"] for record in replay_meta["gate_records"]} == {"fold0", "fold1", None}
    if mechanism == "in_process":
        base_folds = {
            prediction.get("fold_id")
            for frame in native._dagml_node_results if frame.get("node_id") == "branch:0.node:0"
            for prediction in frame.get("predictions", []) if prediction.get("partition") == "validation"
        }
        assert "fold0.inner.fold0.inner.fold0" in base_folds  # base OOF nested below learner CV
    archive = native.export(tmp_path / "residual_auto.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("gate", [False, "auto"])
def test_residual_without_splitter_uses_training_only_oof_and_holdout(tmp_path, monkeypatch, mechanism, gate) -> None:
    """A legacy no-split run succeeds; DAG-ML keeps test rows out of OOF fitting."""
    from sklearn.cross_decomposition import PLSRegression

    import nirs4all
    from nirs4all.data import DatasetConfigs
    from nirs4all.operators.models.residual import ResidualModel
    from nirs4all.pipeline.dagml.residual_run import ResidualImplicitCvWarning

    from ._datasets import dataset_path

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [{"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=gate)}]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy-nosplit", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.best_rmse)
    legacy.close()

    with pytest.warns(ResidualImplicitCvWarning, match="training-only CV"):
        native = nirs4all.run(
            pipeline, dataset_path("regression"), engine="dag-ml", refit=False,
            workspace_path=tmp_path / "native-nosplit", save_artifacts=False, save_charts=False, verbose=0,
        )
    assert native.execution_engine == "dag-ml"
    assert np.isfinite(native.best_rmse)
    assert native.per_dataset[next(iter(native.per_dataset))]["residual_replay"]["implicit_training_cv"] is True
    archive = native.export(tmp_path / "residual_nosplit.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("syntax", ["keyword", "operator"])
def test_residual_nested_operator_choices_select_cv_winner_and_replay(tmp_path, monkeypatch, mechanism, syntax) -> None:
    """Legacy's base × learner choices run as independent native residual graphs."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import KFold

    import nirs4all
    from nirs4all.data import DatasetConfigs
    from nirs4all.operators.models.residual import ResidualModel

    from ._datasets import dataset_path

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    base = {"_or_": [PLSRegression(n_components=2), Ridge(alpha=0.5)]}
    learner = {"_or_": [Ridge(alpha=1.0), ElasticNet(alpha=0.1)]}
    residual = (
        {"residual": {"base": base, "learner": learner, "gate": False}}
        if syntax == "keyword"
        else {"model": ResidualModel(base=base, learner=learner, gate=False)}
    )
    pipeline = [KFold(2, shuffle=True, random_state=1), residual]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", name="choices", refit=False,
        workspace_path=tmp_path / "legacy-choices", save_artifacts=False, save_charts=False, verbose=0,
    )
    legacy_names = {
        row["config_name"] for row in legacy.predictions.filter_predictions(load_arrays=False)
        if row.get("config_name")
    }
    assert len(legacy_names) == 4
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", name="choices", refit=True,
        workspace_path=tmp_path / "native-choices", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert len(native.runs) == 4
    assert {run.cv_best["config_name"] for run in native.runs} == legacy_names
    decision = native._dagml_selection_decision
    assert len(decision["ranked_candidates"]) == 4
    winner = native.runs[int(decision["selected_candidate_id"])]
    assert native.cv_best_score == pytest.approx(min(run.cv_best_score for run in native.runs))
    assert native.best["id"] == winner.best["id"]
    archive = native.export(tmp_path / "residual_choices.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(winner.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
def test_residual_nested_choices_without_refit_keep_cv_selection(tmp_path, monkeypatch) -> None:
    """The core CV decision also controls a generated campaign without refit."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    import nirs4all

    from ._datasets import dataset_path

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    pipeline = [KFold(2, shuffle=True, random_state=1), {
        "residual": {
            "base": {"_or_": [PLSRegression(n_components=2), Ridge(alpha=0.5)]},
            "learner": Ridge(alpha=1.0),
            "gate": False,
        },
    }]
    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=False,
        workspace_path=tmp_path / "native-no-refit-choices", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert len(native.runs) == 2
    winner = native.runs[int(native._dagml_selection_decision["selected_candidate_id"])]
    assert native.cv_best_score == pytest.approx(winner.cv_best_score)
    assert native.best["id"] == winner.cv_best["id"]
    native.close()
