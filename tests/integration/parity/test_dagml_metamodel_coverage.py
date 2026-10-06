"""Partial OOF coverage policy is enforced by the native stacking campaign."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.meta import CoverageStrategy, MetaModel, StackingConfig

from ._datasets import dataset_path


def _pipeline(min_ratio: float) -> list:
    return [
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        PLSRegression(2),
        Ridge(alpha=1000),
        {"model": MetaModel(Ridge(), stacking_config=StackingConfig(
            coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
            min_coverage_ratio=min_ratio,
        ))},
    ]


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("coverage_strategy", [CoverageStrategy.IMPUTE_MEAN, CoverageStrategy.IMPUTE_ZERO, CoverageStrategy.IMPUTE_FOLD_MEAN])
def test_imputation_policy_uses_complete_inner_oof_and_replays_with_partial_outer_coverage(tmp_path, monkeypatch, mechanism, coverage_strategy):
    """Legacy fills sparse outer OOF; DAG-ML's nested inner OOF is complete and attested."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    pipeline = _pipeline(0.3)
    pipeline[-1]["model"].stacking_config.coverage_strategy = coverage_strategy
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert np.isfinite(native.cv_best_score)
        assert native.execution_engine == "dag-ml"
        validation = [row for row in native.predictions._buffer
                      if row.get("model_name") == "MetaModel_Ridge" and row.get("partition") == "val"
                      and row.get("fold_id") in {"0", "1", "2"}]
        assert len(validation) == 3
        dataset = DatasetConfigs(path).get_dataset_at(0)
        n_train = len(dataset.index_column("sample", {"partition": "train"}))
        assert len({int(sample) for row in validation for sample in row["sample_indices"]}) < n_train
        final = [row for row in native.predictions._buffer
                 if row.get("partition") == "test" and row.get("fold_id") == "final"]
        assert len(final) == 1
        archive = native.export(tmp_path / f"{coverage_strategy.value}_policy.n4a")
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(final[0]["y_pred"]).ravel(), rtol=1e-5, atol=3e-4)
    finally:
        legacy.close()
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_partial_oof_coverage_ratio_matches_legacy_gate_and_replays_archive(tmp_path, monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    path = dataset_path("regression")
    dataset = DatasetConfigs(path).get_dataset_at(0)
    n_train = len(dataset.index_column("sample", {"partition": "train"}))
    splitter = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    covered = {int(index) for _, validation in splitter.split(range(n_train)) for index in validation}
    assert 0.3 < len(covered) / n_train < 0.8

    legacy = nirs4all.run(_pipeline(0.3), path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(_pipeline(0.3), path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        validation = [row for row in native.predictions._buffer
                      if row.get("model_name") == "MetaModel_Ridge" and row.get("partition") == "val"
                      and row.get("fold_id") in {"0", "1", "2"}]
        assert len(validation) == 3
        assert len({int(index) for row in validation for index in row["sample_indices"]}) == len(covered)

        final_rows = [row for row in native.predictions._buffer
                      if row.get("model_name") == "MetaModel_Ridge" and row.get("partition") == "test"
                      and row.get("fold_id") == "final"]
        assert len(final_rows) == 1
        archive = native.export(tmp_path / "partial_oof.n4a")
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(final_rows[0]["y_pred"]).ravel(), atol=1e-4)
    finally:
        legacy.close()
        native.close()

    with pytest.raises(Exception, match="Coverage ratio .* below minimum required 80.0%"):
        nirs4all.run(_pipeline(0.8), path, engine="legacy", refit=False,
                     workspace_path=tmp_path / "legacy_rejected", save_artifacts=False, save_charts=False, verbose=0)
    with pytest.raises(Exception) as rejected:
        nirs4all.run(_pipeline(0.8), path, engine="dag-ml", allow_fallback=False,
                     workspace_path=tmp_path / "native_rejected", save_artifacts=False, save_charts=False, verbose=0)
    assert ("coverage ratio" if mechanism == "in_process" else "bundle capture failed") in str(rejected.value)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("coverage_strategy", [CoverageStrategy.DROP_INCOMPLETE, CoverageStrategy.IMPUTE_MEAN,
                                               CoverageStrategy.IMPUTE_ZERO, CoverageStrategy.IMPUTE_FOLD_MEAN])
def test_no_split_opt_in_uses_training_only_native_oof_and_replays(tmp_path, monkeypatch, mechanism, coverage_strategy):
    """DAG-ML's no-split opt-in uses implicit train-only folds and replays."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    source = dataset_path("regression")
    pipeline = [
        Ridge(),
        {"model": MetaModel(Ridge(), stacking_config=StackingConfig(
            allow_no_cv=True, coverage_strategy=coverage_strategy, min_coverage_ratio=0.3,
        ))},
    ]
    with nirs4all.run(pipeline, source, engine="legacy", refit=False,
                      workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0) as legacy:
        assert np.isfinite(legacy.cv_best_score)

    with nirs4all.run(pipeline, source, engine="dag-ml", allow_fallback=False, refit=True,
                      workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0) as native:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        dataset = DatasetConfigs(source).get_dataset_at(0)
        train_ids = set(dataset.index_column("sample", {"partition": "train"}))
        test_ids = set(dataset.index_column("sample", {"partition": "test"}))
        validation = [row for row in native.predictions._buffer
                      if row.get("model_name") == "MetaModel_Ridge" and row.get("partition") == "val"
                      and str(row.get("fold_id")) in {"0", "1"}]
        assert validation and train_ids.isdisjoint(test_ids)
        assert all(set(row["sample_indices"]) <= train_ids for row in validation)
        assert sorted(sample for row in validation for sample in row["sample_indices"]) == sorted(train_ids)

        # Independent nested Ridge oracle: implicit outer KFold(2, shuffle,
        # seed0), inner identity-sorted alternating folds, no held-out labels.
        from sklearn.model_selection import KFold

        base_ids = np.asarray(sorted(train_ids))
        x_train = np.asarray(dataset.x_rows(base_ids.tolist(), layout="2d"))
        y_train = np.asarray(dataset.y({"sample": base_ids.tolist()}), dtype=float).ravel()
        expected = np.empty(len(base_ids))
        for outer_train, outer_val in KFold(2, shuffle=True, random_state=0).split(x_train):
            ordered = np.asarray(sorted(outer_train, key=lambda position: f"s{base_ids[position]}"))
            inner_oof = np.empty(len(ordered))
            for fold in range(2):
                held_out = np.arange(len(ordered)) % 2 == fold
                base_model = Ridge().fit(x_train[ordered[~held_out]], y_train[ordered[~held_out]])
                inner_oof[held_out] = base_model.predict(x_train[ordered[held_out]]).ravel()
            outer_model = Ridge().fit(x_train[outer_train], y_train[outer_train])
            outer_features = outer_model.predict(x_train[outer_val]).reshape(-1, 1)
            meta = Ridge().fit(inner_oof.reshape(-1, 1), y_train[ordered])
            expected[outer_val] = meta.predict(outer_features).ravel()
        position_of = {sample: position for position, sample in enumerate(base_ids)}
        for row in validation:
            positions = [position_of[sample] for sample in row["sample_indices"]]
            np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected[positions], rtol=1e-10, atol=1e-8)
        assert native.cv_best_score == pytest.approx(np.sqrt(np.mean((y_train - expected) ** 2)), rel=1e-10, abs=1e-8)
        features = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        final = [row for row in native.predictions._buffer
                 if row.get("model_name") == "MetaModel_Ridge" and row.get("partition") == "test"
                 and row.get("fold_id") == "final"]
        assert len(final) == 1
        replay = np.asarray(nirs4all.predict(native.export(tmp_path / "no_split_meta.n4a"), features).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(final[0]["y_pred"]).ravel(), rtol=1e-5, atol=3e-4)
