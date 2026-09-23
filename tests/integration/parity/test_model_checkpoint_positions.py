"""Public oracles for ordered model checkpoints around pipeline stages."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset

from ._dagml_cli import dagml_cli_path


def _dataset() -> tuple[SpectroDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(804)
    features = rng.normal(size=(34, 8))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=34)
    dataset = SpectroDataset("checkpoint_positions")
    dataset.add_samples(features[:30], {"partition": "train"}, headers=[str(index) for index in range(8)])
    dataset.add_samples(features[30:], {"partition": "test"})
    dataset.add_targets(targets.reshape(-1, 1))
    return dataset, features[30:], targets[30:]


def _splitter() -> KFold:
    return KFold(2, shuffle=True, random_state=1)


def _first_model() -> dict:
    return {"model": PLSRegression(n_components=2)}


def _second_model() -> dict:
    return {"model": Ridge(alpha=0.3)}


def _pipeline(stage: str) -> list:
    if stage == "none":
        return [_splitter(), _first_model(), _second_model()]
    if stage == "preprocessing":
        return [_splitter(), _first_model(), StandardScaler(), _second_model()]
    if stage == "y_processing":
        return [_splitter(), _first_model(), {"y_processing": StandardScaler()}, _second_model()]
    if stage == "feature_augmentation":
        return [_splitter(), _first_model(), {"feature_augmentation": [StandardScaler()], "action": "add"}, _second_model()]
    if stage == "concat_transform":
        return [_splitter(), _first_model(), {"concat_transform": [StandardScaler(), MinMaxScaler()]}, _second_model()]
    if stage == "splitter":
        return [_splitter(), _first_model(), _splitter(), _second_model()]
    if stage == "branch":
        return [_splitter(), _first_model(), {"branch": [[StandardScaler()], [MinMaxScaler()]]}, _second_model()]
    if stage == "merge":
        return [_splitter(), {"branch": [[StandardScaler()], [MinMaxScaler()]]}, _first_model(), {"merge": "features"}, _second_model()]
    raise ValueError(f"unknown checkpoint stage {stage!r}")


def _transport(mechanism: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("stage", ["none", "preprocessing", "y_processing", "feature_augmentation", "concat_transform"])
def test_two_model_checkpoints_preserve_final_predictions_and_archive(tmp_path, monkeypatch, mechanism: str, stage: str) -> None:
    """Both model identities survive; each final prediction agrees with legacy."""
    _transport(mechanism, monkeypatch)
    legacy_data, x_test, _ = _dataset()
    legacy = nirs4all.run(_pipeline(stage), legacy_data, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native_data, _, _ = _dataset()
    native = nirs4all.run(_pipeline(stage), native_data, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        assert len(native.runs) == 2
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 28
        for model in ("PLSRegression", "Ridge"):
            native_final = next(row for row in native_rows if row["model_name"] == model
                                and row["partition"] == "test" and row["fold_id"] == "final")
            legacy_final = next(row for row in legacy_rows if row["model_name"] == model
                                and row["partition"] == "test" and row["fold_id"] == "final")
            assert native_final["id"] != legacy_final["id"]
            assert native_final["test_score"] == pytest.approx(legacy_final["test_score"], abs=1e-5)
            np.testing.assert_allclose(np.asarray(native_final["y_pred"]).ravel(),
                                       np.asarray(legacy_final["y_pred"]).ravel(), atol=1e-5)
        selected_model = native.cv_best["model_name"]
        selected_final = next(row for row in native_rows if row["model_name"] == selected_model
                              and row["partition"] == "test" and row["fold_id"] == "final")
        archive = native.export(tmp_path / "checkpoint.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(selected_final["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_model_checkpoint_before_duplication_branch_replays_selected_native_producer(tmp_path, monkeypatch, mechanism: str) -> None:
    """One DAG retains a pre-branch model and both branch models through REFIT and archive."""
    _transport(mechanism, monkeypatch)
    legacy_data, x_test, _ = _dataset()
    legacy = nirs4all.run(_pipeline("branch"), legacy_data, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native_data, _, _ = _dataset()
    native = nirs4all.run(_pipeline("branch"), native_data, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        assert native.per_dataset["checkpoint_positions"]["checkpoint_producers"] == [
            "model:checkpoint_before_branch", "branch:0.node:1", "branch:1.node:1",
        ]
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        # Legacy redundantly emits the pre-branch REFIT model once per branch;
        # DAG keeps one attested producer/artifact for the same model.
        assert len(native_rows) == 42 and len(legacy_rows) == 44
        for model, branch in (("PLSRegression", None), ("Ridge", 0), ("Ridge", 1)):
            native_final = next(row for row in native_rows if row["model_name"] == model
                                and row["branch_id"] == branch and row["partition"] == "test"
                                and row["fold_id"] == "final")
            legacy_final_rows = [row for row in legacy_rows if row["model_name"] == model
                                 and row["partition"] == "test" and row["fold_id"] == "final"]
            # Legacy drops branch_id on REFIT rows, but emits them in branch order.
            legacy_final = legacy_final_rows[branch or 0]
            assert np.isfinite(native_final["test_score"])
            assert native_final["test_score"] == pytest.approx(legacy_final["test_score"], abs=1e-5)
            np.testing.assert_allclose(np.asarray(native_final["y_pred"]).ravel(),
                                       np.asarray(legacy_final["y_pred"]).ravel(), atol=1e-5)
        winner = native.cv_best
        selected_final = next(row for row in native_rows if row["model_name"] == winner["model_name"]
                              and row["branch_id"] == winner["branch_id"]
                              and row["partition"] == "test" and row["fold_id"] == "final")
        archive = native.export(tmp_path / "checkpoint_branch.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(selected_final["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_model_checkpoint_inside_duplication_feature_merge_preserves_legacy_refits(tmp_path, monkeypatch, mechanism: str) -> None:
    """Branch-local model refits and selected archive agree across the native transports."""
    _transport(mechanism, monkeypatch)
    legacy_data, x_test, _ = _dataset()
    legacy = nirs4all.run(_pipeline("merge"), legacy_data, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native_data, _, _ = _dataset()
    native = nirs4all.run(_pipeline("merge"), native_data, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        assert native.per_dataset["checkpoint_positions"]["checkpoint_producers"] == [
            "branch:0.node:1", "branch:0.node:2", "branch:1.node:1", "branch:1.node:2",
        ]
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        assert len(native_rows) == len(legacy_rows) == 44
        # Legacy's feature merge keeps the two terminal branch models, but
        # emits CV evidence for only the first Ridge.  Both native REFIT
        # artifacts must still reproduce their exact legacy test predictions.
        for model in ("PLSRegression", "Ridge"):
            native_final = [row for row in native_rows if row["model_name"] == model
                            and row["partition"] == "test" and row["fold_id"] == "final"]
            legacy_final = [row for row in legacy_rows if row["model_name"] == model
                            and row["partition"] == "test" and row["fold_id"] == "final"]
            assert len(native_final) == len(legacy_final) == 2
            for actual, expected in zip(native_final, legacy_final, strict=True):
                assert actual["test_score"] == pytest.approx(expected["test_score"], abs=1e-5)
                np.testing.assert_allclose(np.asarray(actual["y_pred"]).ravel(),
                                           np.asarray(expected["y_pred"]).ravel(), atol=1e-5)
        winner = native.cv_best
        selected_final = next(row for row in native_rows if row["model_name"] == winner["model_name"]
                              and row["partition"] == "test" and row["fold_id"] == "final")
        archive = native.export(tmp_path / "checkpoint_merge.n4a")
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(selected_final["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()
