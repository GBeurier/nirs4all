"""A selected feature axis feeds later wavelength-aware operators per source."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import CARS, MCUVE, Resampler

from ._dagml_cli import dagml_cli_path

pytestmark = pytest.mark.parity


def _dataset(multi_source):
    rng = np.random.default_rng(44)
    first = rng.normal(size=(30, 20))
    second = rng.normal(size=(30, 16)) if multi_source else None
    y = 3 * first[:, 2] - first[:, 9] + (0.5 * second[:, 3] if second is not None else 0) + rng.normal(size=30) * 0.1
    first_headers = [str(value) for value in np.linspace(1000, 1200, 20)]
    dataset = SpectroDataset(f"selected_axis_{multi_source}")
    if second is None:
        dataset.add_samples(first[:24], {"partition": "train"}, headers=first_headers, header_unit="cm-1")
        dataset.add_samples(first[24:], {"partition": "test"}, headers=first_headers, header_unit="cm-1")
        held_out = first[24:]
        targets = [np.linspace(1020, 1180, 7)]
    else:
        second_headers = [str(value) for value in np.linspace(800, 1500, 16)]
        headers = [first_headers, second_headers]
        dataset.add_samples([first[:24], second[:24]], {"partition": "train"}, headers=headers, header_unit=["cm-1", "cm-1"])
        dataset.add_samples([first[24:], second[24:]], {"partition": "test"}, headers=headers, header_unit=["cm-1", "cm-1"])
        held_out = np.hstack([first[24:], second[24:]])
        targets = [np.linspace(1020, 1180, 7), np.linspace(850, 1450, 7)]
    dataset.add_targets(y)
    return dataset, held_out, targets


@pytest.mark.parametrize("selector", ["cars", "mcuve"])
@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_selection_then_resampling_preserves_source_axis(tmp_path, monkeypatch, selector, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out, targets = _dataset(False)
    selection = (
        CARS(n_components=2, n_sampling_runs=10, random_state=42)
        if selector == "cars" else MCUVE(n_components=2, n_iterations=20, random_state=42)
    )
    pipeline = [selection, Resampler(target_wavelengths=targets), KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    # Legacy selects once before CV; DAG-ML selects independently inside each
    # fold. The held-out refit uses the same full training rows in both paths.
    assert np.isfinite(native.cv_best_score)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / f"selected_axis_{selector}.n4a")
    assert np.asarray(nirs4all.predict(archive, held_out).y_pred).shape == (6,)


@pytest.mark.parametrize("in_process", [True, False], ids=["pyo3", "cli"])
def test_selection_then_resampling_uses_each_source_axis(tmp_path, monkeypatch, in_process):
    if not in_process:
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if in_process else "0")
    dataset, held_out, targets = _dataset(True)
    pipeline = [
        CARS(n_components=2, n_sampling_runs=10, random_state=42),
        Resampler(target_wavelengths=targets),
        KFold(2),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert np.isfinite(native.cv_best_score)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / "selected_source_axes.n4a")
    assert np.asarray(nirs4all.predict(archive, held_out).y_pred).shape == (6,)


def test_selection_then_resampling_without_cv_keeps_fitted_axis(tmp_path, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    dataset, held_out, targets = _dataset(False)
    pipeline = [
        CARS(n_components=2, n_sampling_runs=10, random_state=42),
        Resampler(target_wavelengths=targets),
        {"model": Ridge()},
    ]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / "full_train_selected_axis.n4a")
    assert np.asarray(nirs4all.predict(archive, held_out).y_pred).shape == (6,)


def test_selection_accepts_mixed_numeric_and_index_source_axes(tmp_path, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    rng = np.random.default_rng(44)
    first = rng.normal(size=(30, 20))
    second = rng.normal(size=(30, 16))
    y = 3 * first[:, 2] - first[:, 9] + 0.5 * second[:, 3] + rng.normal(size=30) * 0.1
    headers = [
        [str(value) for value in np.linspace(1000, 1200, 20)],
        [str(value) for value in range(16)],
    ]
    dataset = SpectroDataset("mixed_selected_axes")
    dataset.add_samples([first[:24], second[:24]], {"partition": "train"}, headers=headers, header_unit=["cm-1", "index"])
    dataset.add_samples([first[24:], second[24:]], {"partition": "test"}, headers=headers, header_unit=["cm-1", "index"])
    dataset.add_targets(y)
    pipeline = [CARS(n_components=2, n_sampling_runs=10, random_state=42), KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / "mixed_selected_axes.n4a")
    assert np.asarray(nirs4all.predict(archive, np.hstack([first[24:], second[24:]])).y_pred).shape == (6,)
