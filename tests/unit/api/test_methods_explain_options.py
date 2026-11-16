"""Option validation, backend routing and envelope shape of the Methods explain backend."""

import numpy as np
import pytest

from nirs4all.api.explain import explain
from nirs4all.api.explanation_envelope import build_explanation_v1, describe_groups, feature_index_axis, recorded_axis
from nirs4all.api.methods_explain import parse_methods_options
from nirs4all.api.result import ExplainResult


@pytest.mark.parametrize(("explainer_type", "options", "message"), [
    ("deep", {}, "not available with explainer_backend='methods'"),
    ("tree", {"baseline": "mean"}, "uses no baseline"),
    ("linear", {"baseline": "interpolate"}, "requires a model-agnostic"),
    ("permutation", {"baseline": "median"}, "baseline must be one of"),
    ("kernel", {"n_permutations": 4}, "n_permutations applies"),
    ("permutation", {"n_coalitions": 50}, "n_coalitions applies"),
    ("auto", {"n_coalitions": 50}, "n_coalitions applies"),
    ("permutation", {"n_permutations": 0}, "n_permutations must be an integer"),
    ("permutation", {"seed": -1}, "seed must be an integer"),
    ("permutation", {"seed": 2**53}, "smaller than 2"),
    ("permutation", {"group_width": 5, "group_ranges": [(0, 4)]}, "mutually exclusive"),
    ("permutation", {"group_width": 0}, "group_width must be an integer"),
    ("permutation", {"group_ranges": []}, "group_ranges must be"),
    ("permutation", {"group_ranges": [(0, 4, 1)]}, "group_ranges must be"),
    ("permutation", {"group_ranges": [(0.0, 4)]}, "group_ranges must be"),
    ("permutation", {"attribution_space": "latent"}, "attribution_space must be one of"),
])
def test_methods_options_fail_closed(explainer_type, options, message):
    with pytest.raises(ValueError, match=message):
        parse_methods_options(explainer_type, options)


def test_methods_options_normalize_valid_requests():
    settings = parse_methods_options("permutation", {"baseline": "interpolate", "group_ranges": [[0, 10], (20, 30)], "n_permutations": 4, "seed": 3,
                                                     "attribution_space": "transformed_features"})
    assert settings.group_ranges == ((0, 10), (20, 30))
    assert settings.baseline == "interpolate"
    assert settings.n_permutations == 4
    assert settings.seed == 3
    assert settings.attribution_space == "transformed_features"
    default = parse_methods_options("auto", {})
    assert (default.baseline, default.group_width, default.seed, default.attribution_space) == (None, None, 0, "raw_input")


def test_general_lane_validates_backend_before_loading_model(monkeypatch):
    """Invalid selections fail before the (missing) model or data is touched."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_EXPLAIN_PLUGIN", raising=False)
    with pytest.raises(ValueError, match="explainer_backend must be"):
        explain("missing.n4a", np.zeros((2, 2)), explainer_backend="onnx", plots_visible=False)
    with pytest.raises(TypeError, match="require explainer_backend='methods'"):
        explain("missing.n4a", np.zeros((2, 2)), group_width=5, plots_visible=False)
    with pytest.raises(ValueError, match="not available with explainer_backend='methods'"):
        explain("missing.n4a", np.zeros((2, 2)), explainer_backend="methods", explainer_type="deep", plots_visible=False)


@pytest.mark.parametrize("options", [{"explainer_backend": "methods"}, {"group_ranges": [(0, 2)]}])
def test_legacy_lane_refuses_methods_backend(options):
    with pytest.raises(ValueError, match="only on the general explain lane"):
        explain({"model_name": "PLS"}, np.zeros((2, 2)), engine="legacy", plots_visible=False, **options)


def test_recorded_axis_never_invents_units():
    from nirs4all.data.dataset import SpectroDataset

    X = np.ones((2, 3), dtype=np.float32)
    nm = SpectroDataset("nm")
    nm.add_samples(X, headers=["1100", "1102", "1104"], header_unit="nm", indexes={"partition": "test"})
    assert recorded_axis(nm, 3) == {"values": [1100.0, 1102.0, 1104.0], "unit": "nm", "kind": "wavelength"}
    text = SpectroDataset("text")
    text.add_samples(X, headers=["a", "b", "c"], header_unit="text", indexes={"partition": "test"})
    assert recorded_axis(text, 3) == feature_index_axis(3)
    assert feature_index_axis(2) == {"values": [0.0, 1.0], "unit": None, "kind": "feature_index"}


def test_envelope_records_groups_rows_and_selected_output_summary():
    names = ["1100", "1102", "1104", "1106", "1108"]
    axis = {"values": [1100.0, 1102.0, 1104.0, 1106.0, 1108.0], "unit": "nm", "kind": "wavelength"}
    groups = describe_groups(names, axis, np.array([1, 1, 0, 0, 2]), 3, ["ranges", "remainder", "remainder"])
    assert [(group["feature_start"], group["feature_end"], group["source"]) for group in groups] == [(2, 4, "ranges"), (0, 2, "remainder"), (4, 5, "remainder")]
    assert groups[0]["label"] == "1104–1106"
    assert (groups[0]["axis_start"], groups[0]["axis_end"]) == (1104.0, 1106.0)
    attributions = np.arange(12, dtype=float).reshape(2, 3, 2)
    std_error = np.full_like(attributions, np.nan)
    envelope = build_explanation_v1(
        kind="permutation_shap", engine={"name": "nirs4all-methods", "version": "1.3.17", "abi": "2.23.0"}, exactness="sampled", additive=True,
        group_aggregation="group_game", attribution_space="raw_input", output={"index": 1, "label": None, "space": "target"}, axis=axis,
        feature_names=names, groups=groups, sample_indices=[7, 3], attributions=attributions, std_error=std_error,
        expected=np.zeros((2, 2)), prediction=attributions.sum(axis=1), additivity_max_abs_error=0.0, settings={"seed": 0},
    )
    assert envelope["schema"] == "nirs4all.explanation.v1"
    assert envelope["rows"] == {"count": 2, "sample_indices": [7, 3], "source": "dataset_rows"}
    assert np.asarray(envelope["attributions"]).shape == (2, 3, 2)
    assert envelope["std_error"][0][0] == [None, None]
    assert envelope["summary"]["output_index"] == 1
    np.testing.assert_allclose(envelope["summary"]["mean_signed"], attributions[:, :, 1].mean(axis=0))
    result = ExplainResult(shap_values=attributions[:, :, 1], explanation=envelope)
    exported = result.to_explanation_v1()
    exported["rows"]["sample_indices"].append(99)
    assert result.explanation["rows"]["sample_indices"] == [7, 3]
    with pytest.raises(ValueError, match="no nirs4all.explanation.v1 envelope"):
        ExplainResult(shap_values=attributions[:, :, 0]).to_explanation_v1()


def test_recorded_axis_ignores_the_implicit_cm1_default():
    """Numeric headers without a declared unit are indices, not wavenumbers."""
    from nirs4all.data.dataset import SpectroDataset

    X = np.ones((2, 3), dtype=np.float32)
    implicit = SpectroDataset("implicit")
    implicit.add_samples(X, headers=["0", "1", "2"], indexes={"partition": "test"})
    assert implicit.header_unit(0) == "cm-1"  # unchanged default for other callers
    assert not implicit.header_unit_recorded(0)
    assert recorded_axis(implicit, 3) == feature_index_axis(3)
    declared = SpectroDataset("declared")
    declared.add_samples(X, headers=["4000", "4002", "4004"], header_unit="cm-1", indexes={"partition": "test"})
    declared.add_samples(X, headers=["4000", "4002", "4004"], indexes={"partition": "test"})  # appending keeps the declaration
    assert declared.header_unit_recorded(0)
    assert recorded_axis(declared, 3)["kind"] == "wavenumber"


def _fitted_inverse(inverse):
    return inverse.fit(np.array([[1.0, -3.0], [4.0, 2.0], [9.0, 5.0]]))


@pytest.mark.parametrize("inverse", [
    "standard", "standard_no_mean", "minmax", "maxabs", "pipeline", "identity",
])
def test_affine_target_inverses_match_inverse_transform(inverse):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler

    from nirs4all.api.methods_explain import _affine_inverse

    transformer = {
        "standard": StandardScaler(), "standard_no_mean": StandardScaler(with_mean=False), "minmax": MinMaxScaler(feature_range=(-2, 3)),
        "maxabs": MaxAbsScaler(), "pipeline": make_pipeline(StandardScaler(), MinMaxScaler()), "identity": FunctionTransformer(),
    }[inverse]
    _fitted_inverse(transformer)
    scale, offset = _affine_inverse(transformer)
    P = np.array([[0.3, -1.2], [2.0, 0.5], [-0.7, 4.0]])
    np.testing.assert_allclose(P * scale + offset, transformer.inverse_transform(P), rtol=1e-12, atol=1e-12)


def test_nonlinear_target_inverse_refuses_linear_and_tree_shap():
    """Agreement at zero and basis probes does not prove affinity; exp is never folded."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer

    from nirs4all.api.methods_explain import _affine_inverse, _decompose, _select_kind

    exp_inverse = FunctionTransformer(func=np.log, inverse_func=np.exp).fit(np.ones((2, 1)))
    assert _affine_inverse(exp_inverse) is None
    parts = _decompose(LinearRegression().fit(np.array([[0.0], [2.0]]), np.array([0.0, 2.0])), exp_inverse)
    with pytest.raises(ValueError, match="not a recognized affine transform"):
        _select_kind(parse_methods_options("linear", {}), parts, False, False)
    with pytest.raises(ValueError, match="not a recognized affine transform"):
        _select_kind(parse_methods_options("tree", {}), parts, False, False)
    assert _select_kind(parse_methods_options("auto", {}), parts, False, False) == "permutation"


@pytest.mark.parametrize(("options", "message"), [
    ({"explainer_type": "occlusion", "visualizations": ["waterfall"]}, "not additive"),
    ({"explainer_type": "lime", "visualizations": ["force"]}, "not additive"),
    ({"explainer_type": "permutation", "baseline": "interpolate", "visualizations": ["waterfall"]}, "one expected value"),
])
def test_decomposition_figures_refuse_nonadditive_methods_before_loading(monkeypatch, options, message):
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_EXPLAIN_PLUGIN", raising=False)
    with pytest.raises(ValueError, match=message):
        explain("missing.n4a", np.zeros((2, 2)), explainer_backend="methods", plots_visible=False, **options)


def test_waterfall_refuses_explicit_overlapping_or_averaged_bins(monkeypatch):
    from nirs4all.api.general_explain import _require_additive_waterfall_options

    _require_additive_waterfall_options({"bin_size": 20, "bin_stride": 10, "bin_aggregation": "mean"})  # shared values configure other figures
    _require_additive_waterfall_options({"bin_size": {"waterfall": 30}, "bin_stride": {"waterfall": 30}, "bin_aggregation": {"waterfall": "sum"}})
    with pytest.raises(ValueError, match="disjoint"):
        _require_additive_waterfall_options({"bin_size": {"waterfall": 20}, "bin_stride": {"waterfall": 10}})
    with pytest.raises(ValueError, match="disjoint"):
        _require_additive_waterfall_options({"bin_aggregation": {"waterfall": "mean"}})
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_EXPLAIN_PLUGIN", raising=False)
    with pytest.raises(ValueError, match="disjoint"):
        explain("missing.n4a", np.zeros((2, 2)), visualizations=["waterfall"], bin_stride={"waterfall": 5}, plots_visible=False)


def test_disjoint_waterfall_bins_sum_to_the_total_attribution():
    pytest.importorskip("shap")
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    analyzer = ShapAnalyzer()
    analyzer.bin_size = analyzer.bin_stride = 20
    analyzer.bin_aggregation = "sum"
    values = np.ones(30)
    binned, labels = analyzer._aggregate_shap_bins(values, [f"w{index}" for index in range(30)])
    assert binned.sum() == 30.0  # overlapping 20/10 windows would total 50
    assert labels == ["w0–w19", "w20–w29"]


def test_spectral_figure_uses_supplied_axis_and_keeps_legacy_label():
    pytest.importorskip("shap")
    import matplotlib

    matplotlib.use("Agg")
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    analyzer = ShapAnalyzer()
    figures = []
    analyzer._finalize_figure = lambda fig=None, *, plots_visible: figures.append(fig)
    analyzer.shap_values = np.random.default_rng(0).normal(size=(4, 6))
    analyzer.data = np.ones((4, 6))
    analyzer.bin_size, analyzer.bin_stride, analyzer.bin_aggregation = 2, 2, "sum"
    analyzer.plot_spectral_importance(feature_names=[str(1100 + 2 * index) for index in range(6)], plots_visible=False,
                                      axis_values=[1100.0 + 2 * index for index in range(6)], axis_label="Wavelength (nm)")
    analyzer.plot_spectral_importance(feature_names=[f"transformed_{index}" for index in range(6)], plots_visible=False,
                                      axis_values=list(range(6)), axis_label="Transformed feature index")
    analyzer.plot_spectral_importance(feature_names=[str(4000 + index) for index in range(6)], plots_visible=False)
    assert [figure.axes[0].get_xlabel() for figure in figures] == ["Wavelength (nm)", "Transformed feature index", "Wavelength (cm-1)"]
