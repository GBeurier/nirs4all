"""A width-changing transform invalidates wavelengths before optional selectors."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import CARS, MCUVE, Resampler
from nirs4all.operators.transforms.features import CropTransformer, ResampleTransformer

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("width_step", ["crop", "resample"])
@pytest.mark.parametrize("selector", ["cars", "mcuve"])
def test_width_change_before_optional_selector_uses_legacy_index_axis(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str, width_step: str, selector: str,
) -> None:
    """Both mechanisms keep the legacy-successful fit and fitted replay."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    x = np.random.default_rng(2).normal(size=(18, 12))
    y = x[:, 2] - x[:, 5]
    headers = [str(value) for value in np.linspace(1000, 1200, 12)]

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("changed_feature_axis")
        samples.add_samples(x[:14], {"partition": "train"}, headers=headers, header_unit="cm-1")
        samples.add_samples(x[14:], {"partition": "test"})
        samples.add_targets(y)
        return samples

    width_transform = CropTransformer(start=2, end=10) if width_step == "crop" else ResampleTransformer(num_samples=8)
    selection = (CARS(n_components=2, n_sampling_runs=10, random_state=42) if selector == "cars"
                 else MCUVE(n_components=2, n_iterations=20, random_state=42))
    pipeline = [width_transform, selection, KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_artifacts=False, verbose=0)
    assert np.isfinite(native.cv_best_score)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / f"{width_step}_{selector}_{mechanism}.n4a")
    assert np.asarray(nirs4all.predict(archive, x[14:]).y_pred).shape == (4,)
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_resampler_crop_range_keeps_target_grid_for_legacy_and_dag(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str,
) -> None:
    """The input crop precedes interpolation and never crops output columns."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    rng = np.random.default_rng(54)
    x = rng.normal(size=(18, 12))
    y = x[:, 2] - x[:, 5]
    wavelengths = np.linspace(1000, 1200, 12)
    targets = np.linspace(1060, 1140, 12)

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("cropped_resampler")
        samples.add_samples(x[:14], {"partition": "train"}, headers=[str(value) for value in wavelengths], header_unit="cm-1")
        samples.add_samples(x[14:], {"partition": "test"})
        samples.add_targets(y)
        return samples

    operator = Resampler(target_wavelengths=[targets], crop_range=(1040, 1160))
    direct = Resampler(target_wavelengths=targets, crop_range=(1040, 1160)).fit(x[:14], wavelengths=wavelengths)
    assert direct.transform(x).shape == (18, len(targets))
    pipeline = [operator, KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_artifacts=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / f"cropped_resampler_{mechanism}.n4a")
    assert np.asarray(nirs4all.predict(archive, x[14:]).y_pred).shape == (4,)
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_each_source_drops_invalidated_wavelengths_independently(tmp_path, monkeypatch: pytest.MonkeyPatch, mechanism: str) -> None:
    """A mixed source-width crop keeps each selector's own index space."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    rng = np.random.default_rng(53)
    left = rng.normal(size=(18, 20))
    right = rng.normal(size=(18, 16))
    y = left[:, 2] + right[:, 4]
    headers = [
        [str(value) for value in np.linspace(1000, 1200, 20)],
        [str(value) for value in np.linspace(800, 1500, 16)],
    ]

    def dataset() -> SpectroDataset:
        samples = SpectroDataset("multi_source_changed_axis")
        samples.add_samples([left[:14], right[:14]], {"partition": "train"}, headers=headers, header_unit=["cm-1", "cm-1"])
        samples.add_samples([left[14:], right[14:]], {"partition": "test"})
        samples.add_targets(y)
        return samples

    pipeline = [CropTransformer(start=2, end=14), CARS(n_components=2, n_sampling_runs=10, random_state=42), KFold(2), {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, dataset(), engine="legacy", workspace_path=tmp_path / "legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset(), engine="dag-ml", workspace_path=tmp_path / mechanism, save_artifacts=False, verbose=0)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-5)
    archive = native.export(tmp_path / f"multisource_{mechanism}.n4a")
    assert np.asarray(nirs4all.predict(archive, np.hstack([left[14:], right[14:]])).y_pred).shape == (4,)
    legacy.close()
    native.close()
