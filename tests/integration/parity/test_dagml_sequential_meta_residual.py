"""Legacy residual checkpoints are independent of preceding meta-models."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.models import MetaModel
from nirs4all.operators.models.residual import ResidualModel


@pytest.mark.parity
def test_legacy_residual_checkpoint_is_independent_of_preceding_meta(tmp_path) -> None:
    rng = np.random.default_rng(44)
    features = rng.normal(size=(24, 5))
    targets = features[:, 0] * 2 + features[:, 1] + rng.normal(scale=0.1, size=24)
    snapshots = []
    for alpha in (0.1, 100.0):
        pipeline = [
            KFold(2, shuffle=True, random_state=1),
            Ridge(),
            {"model": MetaModel(model=Ridge(alpha=alpha))},
            {"model": ResidualModel(base=Ridge(), learner=Ridge(), gate=False)},
        ]
        result = nirs4all.run(
            pipeline, (features, targets), engine="legacy", refit=False,
            workspace_path=tmp_path / f"legacy_{alpha}", save_artifacts=False,
            save_charts=False, verbose=0,
        )
        residual_rows = [
            row for row in result.predictions.filter_predictions(load_arrays=True)
            if row["step_idx"] == 4 and row["fold_id"] == "avg" and row["partition"] == "val"
        ]
        assert residual_rows
        snapshots.append([(row["val_score"], np.asarray(row["y_pred"]).copy()) for row in residual_rows])
        result.close()

    assert len(snapshots[0]) == len(snapshots[1])
    for left, right in zip(*snapshots, strict=True):
        assert left[0] == pytest.approx(right[0], abs=1e-12)
        np.testing.assert_allclose(left[1], right[1], atol=1e-12, rtol=0)


@pytest.mark.parity
def test_legacy_residual_cannot_supply_a_named_meta_source(tmp_path) -> None:
    """The source predictions exist, but legacy lacks their artifact dependency."""
    rng = np.random.default_rng(44)
    features = rng.normal(size=(32, 5))
    targets = features[:, 0] * 2 + features[:, 1] + rng.normal(scale=0.1, size=32)
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        Ridge(),
        {"model": ResidualModel(base=Ridge(), learner=Ridge(), gate=False)},
        {"model": MetaModel(model=Ridge(), source_models=["Residual_Ridge+Ridge"], name="second")},
    ]
    with pytest.raises(RuntimeError, match="missing source model dependencies"):
        nirs4all.run(
            pipeline, (features, targets), engine="legacy", refit=False,
            workspace_path=tmp_path / "legacy", save_artifacts=False,
            save_charts=False, verbose=0,
        )


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_sequential_meta_then_residual_runs_with_native_nested_oof(tmp_path, monkeypatch, mechanism: str) -> None:
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    rng = np.random.default_rng(44)
    features = rng.normal(size=(24, 5))
    targets = features[:, 0] * 2 + features[:, 1] + rng.normal(scale=0.1, size=24)
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        Ridge(),
        {"model": MetaModel(model=Ridge())},
        {"model": ResidualModel(base=Ridge(), learner=Ridge(), gate=False)},
    ]
    legacy = nirs4all.run(
        pipeline, (features, targets), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, (features, targets), engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert len(native.runs) == 3
    assert all(np.isfinite(checkpoint.cv_best_score) for checkpoint in native.runs)
    assert np.isfinite(native.cv_best_score)
    assert np.all(np.isfinite(nirs4all.predict(native.export(tmp_path / "meta_residual.n4a"), features).y_pred))
    native.close()
