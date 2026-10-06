"""Export and diagnostic contract regressions, including local fitter seeds."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nirs4all.synthesis import validation
from nirs4all.synthesis.builder import SyntheticDatasetBuilder
from nirs4all.synthesis.exporter import DatasetExporter, ExportConfig
from nirs4all.synthesis.fitter import FittedParameters, RealBandFitter, RealDataFitter, _compute_km_linearity
from nirs4all.synthesis.procedural import ProceduralComponentConfig, ProceduralComponentGenerator
from nirs4all.synthesis.targets import TargetGenerator


@pytest.mark.parametrize("compression,suffix", [(None, ""), ("gzip", ".gz"), ("zip", ".zip")])
@pytest.mark.parametrize("headers", [False, True])
@pytest.mark.parametrize("format", ["standard", "single", "fragmented"])
def test_csv_exports_honor_headers_compression_and_exact_axis(tmp_path, compression, suffix, headers, format):
    exporter = DatasetExporter(ExportConfig(format=format, include_headers=headers, compression=compression))
    axis = np.array([1100., 1100.5, 1101.125])
    x, y = np.arange(120).reshape(40, 3), np.arange(40)
    folder = exporter.to_folder(tmp_path / "data", x, y, wavelengths=axis)
    files = list(folder.rglob("*" + exporter.config.file_extension + suffix))
    assert files
    for path in files:
        frame = pd.read_csv(path, sep=";", header=0 if headers else None)
        assert len(frame) > 0
        if headers and (path.name.startswith("X") or path.name.startswith("data")):
            for wl in axis:
                assert str(float(wl)) in frame.columns
    path = exporter.to_csv(tmp_path / "all.csv", x, y, wavelengths=axis)
    assert str(path).endswith(".csv" + suffix) and path.exists()
    frame = pd.read_csv(path, sep=";", header=0 if headers else None)
    assert len(frame) == 40
    if headers:
        assert list(frame.columns[:3]) == [str(float(wl)) for wl in axis]


@pytest.mark.parametrize("function,metric", [
    ("compute_correlation_length", validation.RealismMetric.CORRELATION_LENGTH),
    ("compute_derivative_statistics", validation.RealismMetric.DERIVATIVE_STATISTICS),
    ("compute_peak_density", validation.RealismMetric.PEAK_DENSITY),
    ("compute_baseline_curvature", validation.RealismMetric.BASELINE_CURVATURE),
    ("compute_snr", validation.RealismMetric.SNR_DISTRIBUTION),
    ("compute_adversarial_validation_auc", validation.RealismMetric.ADVERSARIAL_AUC),
])
def test_unevaluated_realism_metric_is_visible_and_fails_gate(monkeypatch, function, metric):
    x = np.random.default_rng(4).normal(size=(24, 32))
    def fail(*args, **kwargs):
        raise RuntimeError("unevaluated gate")
    monkeypatch.setattr(validation, function, fail)
    score = validation.compute_spectral_realism_scorecard(x, x, random_state=4, include_adversarial=function.endswith("auc"))
    results = [r for r in score.metric_results if r.metric == metric]
    assert len(results) == 1 and not results[0].passed
    assert "unevaluated gate" in results[0].details["error"]
    assert not score.overall_pass and score.warnings


def test_benchmark_transfer_failure_is_reported():
    x = np.random.default_rng(4).normal(size=(24, 32))
    result = validation.validate_against_benchmark(x, x, "test", synthetic_targets=np.ones((24, 2)), benchmark_targets=np.ones(24), random_state=4)
    assert any("TSTR/TRTS evaluation failed" in message for message in result.realism_score.warnings)
    assert not result.realism_score.overall_pass


def test_noise_region_fraction_changes_snr_estimation():
    rng = np.random.default_rng(4)
    x = np.tile(np.sin(np.linspace(0, 8, 100)), (8, 1))
    x[:, :50] += rng.normal(0, .01, (8, 50))
    x[:, 50:] += rng.normal(0, .2, (8, 50))
    quiet = validation.compute_snr(x, .2)
    all_regions = validation.compute_snr(x, 1.)
    assert np.all(quiet > all_regions)
    with pytest.raises(ValueError, match="fraction"):
        validation.compute_snr(x, 0)


def test_procedural_library_honors_explicit_config(monkeypatch):
    generator = ProceduralComponentGenerator(random_state=4)
    config = ProceduralComponentConfig(n_fundamental_bands=3, h_bond_strength=.6)
    original = generator.generate_component
    applied = []
    def capture(*args, **kwargs):
        applied.append(kwargs["config"])
        return original(*args, **kwargs)
    monkeypatch.setattr(generator, "generate_component", capture)
    library = generator.generate_library(4, config=config)
    assert library.n_components == 4
    assert all(c.n_fundamental_bands == 3 and c.h_bond_strength == .6 for c in applied)


def test_target_range_describes_base_not_clipped_noisy_targets():
    y = TargetGenerator(random_state=4).regression(1000, range=(5, 50), correlation=.1, noise=3.)
    assert np.min(y) < 5 or np.max(y) > 50


def test_fitted_component_detection_propagates_to_generator_and_builder(monkeypatch):
    parameters = FittedParameters(detected_components=["water", "protein"], measurement_mode="reflectance")
    assert parameters.to_generator_kwargs()["component_library"].component_names == ["water", "protein"]
    original = RealDataFitter.fit
    def fit(self, *args, **kwargs):
        original(self, *args, **kwargs)
        return parameters
    monkeypatch.setattr(RealDataFitter, "fit", fit)
    builder = SyntheticDatasetBuilder(12, random_state=4).fit_to(np.random.default_rng(4).uniform(.1, 1, (24, 16)), wavelengths=np.linspace(1100, 2400, 16))
    assert builder._create_generator().library.component_names == ["water", "protein"]
    assert builder.state.measurement_mode == "reflectance"


def test_km_diagnostic_seed_is_reproducible_and_global_rng_is_untouched():
    x = np.random.default_rng(4).uniform(.1, 1, (300, 32))
    before = np.random.get_state()
    first = _compute_km_linearity(x, random_state=4)
    second = _compute_km_linearity(x, random_state=4)
    assert first == second
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    fitters = [RealDataFitter(random_state=4) for _ in range(2)]
    for fitter in fitters:
        fitter.fit(x, wavelengths=np.linspace(1100, 2400, 32))
    assert fitters[0].source_properties.kubelka_munk_linearity == fitters[1].source_properties.kubelka_munk_linearity


def test_band_fitter_restarts_use_local_seed_and_preserve_global_rng(monkeypatch):
    starts = []
    def minimize(objective, x0, **kwargs):
        starts.append(x0.copy())
        return SimpleNamespace(x=x0.copy())
    monkeypatch.setattr("scipy.optimize.minimize", minimize)
    axis = np.linspace(1100, 2400, 32)
    spectrum = np.sin(axis / 40) + .5
    before = np.random.get_state()
    for _ in range(2):
        RealBandFitter(max_bands=4, n_iterations=3, target_r2=2, random_state=4).fit(spectrum, axis)
    assert len(starts) == 6
    for first, second in zip(starts[:3], starts[3:], strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.array_equal(starts[0], starts[1])
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
