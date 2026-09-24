import numpy as np
import pytest
from scipy.interpolate import interp1d
from sklearn.base import clone

from nirs4all.controllers.data.resampler import ResamplerController
from nirs4all.operators.transforms import Resampler


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic", "quadratic", "slinear", "zero"])
def test_point_count_resampling_matches_per_spectrum_reference(method):
    wavelengths = np.linspace(2500, 1000, 30)
    spectra = np.random.default_rng(54).normal(size=(12, 30))
    operator = clone(Resampler(n_points=17, method=method))
    actual = operator.fit_transform(spectra, wavelengths=wavelengths)
    target = np.linspace(2500, 1000, 17)
    expected = np.stack([interp1d(wavelengths, row, kind=method)(target) for row in spectra])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.shape == (12, 17)


def test_controller_preserves_none_and_accepts_flat_json_wavelength_lists():
    controller = ResamplerController()
    assert controller._get_target_wavelengths_for_source(Resampler(n_points=17), 0, 1) is None
    np.testing.assert_array_equal(
        controller._get_target_wavelengths_for_source(Resampler(target_wavelengths=[1000, 1200, 1400]), 0, 1),
        [1000, 1200, 1400],
    )


@pytest.mark.parametrize("n_points", [0, 1, 2.5, True])
def test_invalid_point_count_is_rejected(n_points):
    with pytest.raises(ValueError, match="n_points"):
        Resampler(n_points=n_points).fit(np.ones((5, 4)), wavelengths=[1, 2, 3, 4])


@pytest.mark.parametrize("operator", [
    Resampler(n_points=17), Resampler(),
    Resampler(target_wavelengths=[1200, 1400, 1600, 1800], crop_range=(1100, 2000)),
])
def test_resampling_runs_through_pipeline_controller_and_persists_predictions(tmp_path, operator):
    from sklearn.linear_model import Ridge

    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    rng = np.random.default_rng(55)
    spectra = rng.normal(size=(24, 30))
    targets = spectra[:, 0] + spectra[:, 2] * .1
    dataset = SpectroDataset("resampling")
    dataset.add_samples(spectra, {"partition": "train"},
                        headers=[str(value) for value in np.linspace(1000, 2500, 30)], header_unit="cm-1")
    dataset.add_targets(targets)
    with nirs4all.run([operator, Ridge()], dataset, workspace_path=tmp_path, engine="legacy",
                     verbose=0, save_charts=False, refit=False) as result:
        rows = result.predictions.filter_predictions(partition="train", load_arrays=True)
        assert rows and np.isfinite(rows[0]["y_pred"]).all()
        assert len(rows[0]["y_pred"]) == len(targets)
