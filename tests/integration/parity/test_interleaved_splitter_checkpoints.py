"""Public checkpoints with a new splitter between two models."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.dataset import SpectroDataset

from ._dagml_cli import dagml_cli_path


def _dataset() -> tuple[SpectroDataset, np.ndarray]:
    rng = np.random.default_rng(804)
    features = rng.normal(size=(34, 8))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=34)
    dataset = SpectroDataset("interleaved_splitters")
    dataset.add_samples(features[:30], {"partition": "train"}, headers=[str(index) for index in range(8)])
    dataset.add_samples(features[30:], {"partition": "test"})
    dataset.add_targets(targets.reshape(-1, 1))
    return dataset, features[30:]


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("second_seed", [1, 7])
def test_new_splitter_uses_independent_native_fold_sets_and_refits(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, second_seed: int,
) -> None:
    """Both checkpoints retain CV outputs; native REFIT is a real full-train fit."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")
    pipeline = [
        KFold(2, shuffle=True, random_state=1), {"model": PLSRegression(n_components=2)},
        KFold(2, shuffle=True, random_state=second_seed), {"model": Ridge(alpha=0.3)},
    ]
    legacy_data, _ = _dataset()
    legacy = nirs4all.run(pipeline, legacy_data, engine="legacy", allow_fallback=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native_data, x_test = _dataset()
    native = nirs4all.run(pipeline, native_data, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert native.get_models() == legacy.get_models() == ["PLSRegression", "Ridge"]
        assert len(native.runs) == 2
        native_rows = native.predictions.filter_predictions()
        legacy_rows = legacy.predictions.filter_predictions()
        for model in ("PLSRegression", "Ridge"):
            for partition, fold_id in (("val", 0), ("val", 1), ("test", 0), ("test", 1),
                                       ("test", "avg"), ("test", "w_avg")):
                native_row = next(row for row in native_rows if row["model_name"] == model
                                  and row["partition"] == partition and str(row["fold_id"]) == str(fold_id))
                legacy_row = next(row for row in legacy_rows if row["model_name"] == model
                                  and row["partition"] == partition and str(row["fold_id"]) == str(fold_id))
                assert list(native_row["sample_indices"]) == list(legacy_row["sample_indices"])
                np.testing.assert_allclose(np.asarray(native_row["y_pred"]).ravel(),
                                           np.asarray(legacy_row["y_pred"]).ravel(), atol=1e-5)
                score_key = "val_score" if partition == "val" else "test_score"
                assert native_row[score_key] == pytest.approx(legacy_row[score_key], abs=1e-5)
        # Legacy's four Ridge rows labeled `final` duplicate its fold/mean
        # TEST predictions. The native run retains those four source rows and
        # additionally emits a genuine full-train REFIT with its own artifact.
        legacy_aliases = [row for row in legacy_rows if row["model_name"] == "Ridge"
                          and row["partition"] == "test" and row["fold_id"] == "final"]
        assert len(legacy_aliases) == 4
        for alias, fold_id in zip(legacy_aliases, (0, 1, "avg", "w_avg"), strict=True):
            source = next(row for row in legacy_rows if row["model_name"] == "Ridge"
                          and row["partition"] == "test" and str(row["fold_id"]) == str(fold_id))
            np.testing.assert_array_equal(alias["y_pred"], source["y_pred"])
        native_refit = next(row for row in native_rows if row["model_name"] == "Ridge"
                            and row["partition"] == "test" and row["fold_id"] == "final")
        assert not np.allclose(np.asarray(native_refit["y_pred"]).ravel(),
                               np.asarray(legacy_aliases[-1]["y_pred"]).ravel(), atol=1e-5)
        selected = native.cv_best["model_name"]
        selected_final = next(row for row in native_rows if row["model_name"] == selected
                              and row["partition"] == "test" and row["fold_id"] == "final")
        archive = native.export(tmp_path / "splitter_checkpoints.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel(),
                                   np.asarray(selected_final["y_pred"]).ravel(), atol=1e-4)
    finally:
        native.close()
        legacy.close()
