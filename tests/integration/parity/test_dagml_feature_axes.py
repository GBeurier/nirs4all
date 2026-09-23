"""Source-local feature coordinates reach wavelength-aware operators through DAG-ML."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import Resampler

from ._dagml_cli import dagml_cli_path

pytestmark = pytest.mark.parity


def _dataset() -> tuple[SpectroDataset, np.ndarray]:
    x = np.random.default_rng(3).normal(size=(18, 12))
    y = 2 * x[:, 1] - x[:, 5] + 0.1 * x[:, 8]
    headers = [str(value) for value in np.linspace(1000, 1200, 12)]
    dataset = SpectroDataset("wavelength_oracle")
    dataset.add_samples(x[:14], {"partition": "train"}, headers=headers, header_unit="cm-1")
    dataset.add_samples(x[14:], {"partition": "test"}, headers=headers, header_unit="cm-1")
    dataset.add_targets(y)
    return dataset, x[14:]


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_resampler_matches_legacy_and_archive(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out = _dataset()
    pipeline = [
        Resampler(target_wavelengths=[np.linspace(1010, 1190, 8)]),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.cv_best_score == pytest.approx(3.382102416, abs=1e-6)
    assert legacy.best_rmse == pytest.approx(1.209796885, abs=1e-6)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    archive = native.export(tmp_path / "resampler.n4a")
    replay = nirs4all.predict(archive, held_out)
    assert np.asarray(replay.y_pred).shape == (4,)
    assert np.isfinite(replay.y_pred).all()


def test_resampler_chain_uses_first_target_as_second_input_axis():
    from nirs4all.pipeline.dagml.node_runner import _coordinate_chain

    x = np.random.default_rng(3).normal(size=(18, 12))
    first = Resampler(target_wavelengths=[np.linspace(1010, 1190, 8)])
    second = Resampler(target_wavelengths=[np.linspace(1020, 1180, 6)])
    steps = _coordinate_chain(
        [first, second], [tuple(str(value) for value in np.linspace(1000, 1200, 12))]
    )
    for step in steps:
        x = step.fit_transform(x)
    assert x.shape == (18, 6)
    np.testing.assert_allclose(
        steps[1].transformer.original_wavelengths_,
        [float(f"{value:.2f}") for value in np.linspace(1010, 1190, 8)],
    )


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_resampler_keeps_distinct_source_axes(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    rng = np.random.default_rng(23)
    first = rng.normal(size=(18, 12))
    second = rng.normal(size=(18, 9))
    y = 2 * first[:, 1] - first[:, 5] + 0.5 * second[:, 3]
    headers = [
        [str(value) for value in np.linspace(1000, 1200, 12)],
        [str(value) for value in np.linspace(800, 1600, 9)],
    ]
    dataset = SpectroDataset("source_wavelength_oracle")
    dataset.add_samples([first[:14], second[:14]], {"partition": "train"}, headers=headers, header_unit=["cm-1", "cm-1"])
    dataset.add_samples([first[14:], second[14:]], {"partition": "test"}, headers=headers, header_unit=["cm-1", "cm-1"])
    dataset.add_targets(y)
    pipeline = [
        Resampler(target_wavelengths=[np.linspace(1010, 1190, 8), np.linspace(850, 1550, 7)]),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    archive = native.export(tmp_path / "source_resampler.n4a")
    replay = nirs4all.predict(archive, np.hstack([first[14:], second[14:]]))
    assert np.asarray(replay.y_pred).shape == (4,)


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_chained_resamplers_match_legacy(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out = _dataset()
    pipeline = [
        Resampler(target_wavelengths=[np.linspace(1010, 1190, 8)]),
        Resampler(target_wavelengths=[np.linspace(1020, 1180, 6)]),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    archive = native.export(tmp_path / "chained_resamplers.n4a")
    assert np.asarray(nirs4all.predict(archive, held_out).y_pred).shape == (4,)


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_resampler_converts_nm_headers_like_legacy(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    x = np.random.default_rng(3).normal(size=(18, 12))
    y = 2 * x[:, 1] - x[:, 5] + 0.1 * x[:, 8]
    headers = [str(10_000_000 / value) for value in np.linspace(1000, 1200, 12)]
    dataset = SpectroDataset("nm_wavelength_oracle")
    dataset.add_samples(x[:14], {"partition": "train"}, headers=headers, header_unit="nm")
    dataset.add_samples(x[14:], {"partition": "test"}, headers=headers, header_unit="nm")
    dataset.add_targets(y)
    pipeline = [Resampler(target_wavelengths=[np.linspace(1010, 1190, 8)]), KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    archive = native.export(tmp_path / "nm_resampler.n4a")
    replay = nirs4all.predict(archive, x[14:])
    assert np.asarray(replay.y_pred).shape == (4,)
    assert np.isfinite(replay.y_pred).all()


@pytest.mark.parametrize("method", ["nearest", "cubic", "quadratic", "slinear", "zero"])
@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_resampler_interpolation_methods_match_legacy_and_replay(tmp_path, monkeypatch, method, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out = _dataset()
    pipeline = [
        Resampler(target_wavelengths=[np.linspace(1010, 1190, 8)], method=method),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    replay = nirs4all.predict(native.export(tmp_path / f"resampler_{method}.n4a"), held_out)
    assert np.asarray(replay.y_pred).shape == (4,)
    assert np.isfinite(replay.y_pred).all()


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_resampler_outside_fill_matches_legacy(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out = _dataset()
    pipeline = [
        Resampler(target_wavelengths=[np.linspace(990, 1210, 8)], fill_value=-7.0),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-6)
    replay = nirs4all.predict(native.export(tmp_path / "resampler_fill_value.n4a"), held_out)
    assert np.asarray(replay.y_pred).shape == (4,)
    assert np.isfinite(replay.y_pred).all()
