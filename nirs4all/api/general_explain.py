"""SHAP over a captured scientific predictor, without replaying a legacy runner."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np

from nirs4all.pipeline.dagml.dataset import _materialize_dataset
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage

from .explanation_envelope import build_explanation_v1, describe_groups, feature_index_axis, recorded_axis
from .result import ExplainResult, RunResult


class _CapturedPredictor:
    """Expose one original-target output of an already fitted model to SHAP."""

    def __init__(self, predictor: Any, output_index: int):
        self.predictor = predictor
        self.output_index = output_index

    def predict(self, X: Any) -> np.ndarray:
        values = np.asarray(self.predictor(X))
        matrix = values.reshape(len(X), -1)
        if self.output_index >= matrix.shape[1]:
            raise ValueError("SHAP output_index exceeds the captured predictor's output width")
        return matrix[:, self.output_index]


def _is_affine_transform(operator: Any) -> bool:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

    if operator is None or isinstance(operator, str) and operator == "passthrough":
        return True
    if not type(operator).__module__.startswith("sklearn.") or isinstance(operator, MinMaxScaler) and operator.clip:
        return False
    return isinstance(operator, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PCA))


def _is_affine_predictor(operator: Any) -> bool:
    """Certify the fitted structure before extracting raw-input coefficients."""
    from sklearn.base import is_classifier
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model._base import LinearModel
    from sklearn.pipeline import Pipeline

    from nirs4all.api.result import _DagmlExportedModel

    if type(operator) is Pipeline:
        return all(_is_affine_transform(step) for _, step in operator.steps[:-1]) and _is_affine_predictor(operator.steps[-1][1])
    if type(operator).__module__.startswith("sklearn.") and isinstance(operator, (LinearModel, PLSRegression)):
        return not is_classifier(operator)
    # Captured DAG predictors retain their complete fitted X-chain and target
    # inverse. Both must be affine; a class label or a local numerical probe is
    # never evidence that an arbitrary nonlinear pipeline is globally linear.
    if type(operator) is _DagmlExportedModel:
        return _is_affine_predictor(operator.estimator) and _is_affine_transform(getattr(operator, "y_transform", None))
    return False


def _raw_affine_parameters(predictor: _CapturedPredictor, X: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Extract the certified affine callable in batches, without fitting."""
    width = X.shape[1]
    anchor = np.mean(X, axis=0, dtype=float)
    steps = np.maximum(np.ptp(X, axis=0), 1.0)
    baseline = float(predictor.predict(anchor[None])[0])
    coefficients = np.empty(width, dtype=float)
    for start in range(0, width, 128):
        end = min(start + 128, width)
        basis = np.tile(anchor, (end - start, 1))
        basis[np.arange(end - start), np.arange(start, end)] += steps[start:end]
        coefficients[start:end] = (predictor.predict(basis) - baseline) / steps[start:end]
    intercept = baseline - float(anchor @ coefficients)
    # Admission is relative to observed variation, not a potentially enormous
    # target offset that could conceal cancellation in coefficient extraction.
    expected = predictor.predict(X)
    affine_replay_atol, affine_replay_rtol = 1e-8, 1e-6
    tolerance = affine_replay_atol + affine_replay_rtol * float(np.ptp(expected))
    if not np.all(np.abs((X @ coefficients + intercept) - expected) <= tolerance):
        return None
    return coefficients, intercept


def _raw_tree_estimator(captured: Any, target_inverse: Any) -> Any:
    """Expose an unchanged scalar tree predictor only across identity wrappers."""
    from sklearn.pipeline import Pipeline

    from nirs4all.api.result import _DagmlExportedModel

    if target_inverse is not None:
        return None
    if type(captured) is _DagmlExportedModel:
        if captured.y_transform is not None:
            return None
        captured = captured.estimator
    elif not type(captured).__module__.startswith("sklearn."):
        return None
    if type(captured) is Pipeline:
        if len(captured.steps) != 1:
            return None
        captured = captured.steps[0][1]
    return captured


def _load_captured_source(model: Any, session: Any, workspace_path: Any = None) -> dict[str, Any]:
    if isinstance(model, (str, Path)):
        expected = getattr(session, "_general_archive_fingerprint", None)
        return load_general_archive(model, expected_archive_fingerprint=expected)
    if isinstance(model, dict) and model.get("chain_id"):
        root = workspace_path or model.get("workspace_path") or getattr(session, "workspace_path", None)
        if root is not None:
            from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain

            workspace_source = load_general_workspace_chain(root, model["chain_id"])
            if workspace_source is not None:
                workspace_source["model_name"] = workspace_source["chain"].get("model_name") or ""
                workspace_source["source_provenance"] = {"source_type": "workspace", **workspace_source["metadata"]}
                return workspace_source
    result = model if isinstance(model, RunResult) else getattr(session, "_last_result", None)
    if (
        not isinstance(result, RunResult)
        or not getattr(result, "_dagml_refit_artifacts", None)
        or result.execution_engine != "dag-ml"
    ):
        raise ValueError("General explain requires a captured .n4a archive or a trained DAG Session/result")
    if isinstance(model, dict):
        if hasattr(result, "_source_run"):
            result = result._source_run(model)
        identifier = model.get("id") or model.get("prediction_id")
        if identifier != (result.best.get("id") or result.best.get("prediction_id")):
            raise ValueError("This source does not identify the captured final predictor; fold artifacts are not available")
    with TemporaryDirectory(prefix="n4a-explain-model-") as temporary:
        archive = result.export(Path(temporary) / "captured.n4a")
        return load_general_archive(archive)


_SHAP_KINDS = {"KernelExplainer": ("kernel_shap", "sampled"), "TreeExplainer": ("tree_shap", "exact"), "LinearExplainer": ("linear_shap", "exact"), "PermutationExplainer": ("permutation_shap", "sampled")}


def _explained_rows(sample_indices: Any, n_rows: int) -> list[int]:
    if sample_indices is None:
        return list(range(n_rows))
    if isinstance(sample_indices, (str, bytes)) or not isinstance(sample_indices, Sequence | np.ndarray) or not len(sample_indices):
        raise ValueError("sample_indices must be a nonempty sequence of row indices")
    rows = [int(index) for index in sample_indices if isinstance(index, int | np.integer) and not isinstance(index, bool)]
    if len(rows) != len(sample_indices) or any(index < 0 or index >= n_rows for index in rows):
        raise ValueError(f"sample_indices must be integers in [0, {n_rows})")
    return rows


#: Background subsampling seed of the exact linear path (the Methods backend default).
LINEAR_BACKGROUND_SEED = 0


def _affine_shap(predictor: _CapturedPredictor, X: np.ndarray, rows: list[int], background: int) -> tuple[np.ndarray, float]:
    """Exact interventional linear SHAP for a captured predictor verified to be affine in its raw inputs.

    The affine form is identified from one batched prediction at the background
    mean and one per feature, then checked against the predictor on the
    background and explained rows. A nonlinear chain (for example SNV or a
    nonlinear model) is refused instead of silently using another explainer.
    """
    # Seeded like the Methods backend default so reference, base value and attributions are reproducible.
    take = np.sort(np.random.default_rng(LINEAR_BACKGROUND_SEED).choice(X.shape[0], size=min(background, X.shape[0]), replace=False))
    reference = X[take].mean(axis=0)
    parameters = _raw_affine_parameters(predictor, X)
    if parameters is None:
        raise ValueError("Raw linear SHAP reconstruction is numerically unstable; use auto or kernel.")
    weights, intercept = parameters
    return (X[rows] - reference) * weights, float(reference @ weights + intercept)


def explain_general(
    model: Any, data: Any, *, name: str, session: Any, verbose: int,
    plots_visible: bool, n_samples: int | None, explainer_type: str, options: dict[str, Any],
) -> ExplainResult:
    """Explain the complete frozen raw-input predictor with the selected attribution backend.

    Preprocessing and target inverses stay inside the captured predictor. SHAP
    therefore explains original input columns and original target units, not a
    terminal estimator with an incorrectly labelled raw feature axis. Multi-output
    requests expose a labelled selected output, preserving the 2D ExplainResult.
    ``explainer_backend="shap"`` (default) uses the ``shap`` package;
    ``explainer_backend="methods"`` uses the nirs4all-methods kernels (see
    :mod:`nirs4all.api.methods_explain`). Both record a
    ``nirs4all.explanation.v1`` envelope in ``ExplainResult.explanation``.
    """
    from .methods_explain import METHODS_OPTIONS, NON_ADDITIVE_EXPLAINER_TYPES, parse_methods_options

    supported = {"feature_names", "background_samples", "max_display", "visualizations", "output_dir", "workspace_path",
                 "bin_size", "bin_stride", "bin_aggregation", "sample_indices", "output_index", "explainer_backend"} | METHODS_OPTIONS
    if set(options) - supported:
        raise TypeError(f"Unsupported general SHAP options: {sorted(set(options) - supported)}")
    backend = options.get("explainer_backend", "shap")
    if backend not in ("shap", "methods"):
        raise ValueError("explainer_backend must be 'shap' or 'methods'")
    methods_settings = None
    if backend == "methods":
        methods_settings = parse_methods_options(explainer_type, options)
    elif set(options) & METHODS_OPTIONS:
        raise TypeError(f"Options {sorted(set(options) & METHODS_OPTIONS)} require explainer_backend='methods'")
    output_index = options.get("output_index", 0)
    if type(output_index) is not int or output_index < 0:
        raise ValueError("output_index must be a nonnegative integer")
    background = n_samples if n_samples is not None else options.get("background_samples", 100)
    if type(background) is not int or background < 1:
        raise ValueError("SHAP background sample count must be a positive integer")
    visualizations = options.get("visualizations", ["spectral", "summary"])
    allowed_visualizations = {"spectral", "summary", "waterfall", "force", "beeswarm"}
    if not isinstance(visualizations, list) or any(item not in allowed_visualizations for item in visualizations):
        raise ValueError("Unsupported SHAP visualization selection")
    decompositions = sorted({"waterfall", "force"} & set(visualizations))
    if methods_settings is not None and decompositions:
        if methods_settings.explainer_type in NON_ADDITIVE_EXPLAINER_TYPES:
            raise ValueError(f"{decompositions} figures decompose a prediction additively, but {methods_settings.explainer_type} attributions are not additive; "
                             "select spectral, summary or beeswarm")
        if methods_settings.baseline == "interpolate":
            raise ValueError(f"{decompositions} figures need one expected value; baseline='interpolate' has one per row. Select spectral, summary or beeswarm")
    if "waterfall" in visualizations:
        _require_additive_waterfall_options(options)
    if session is not None:
        session._ensure_open()
    loaded = _load_captured_source(model, session, options.get("workspace_path"))
    dataset = _materialize_dataset(data)
    X = np.asarray(dataset.x({}, layout="2d"))
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("SHAP requires a nonempty finite 2D feature matrix")
    rows = _explained_rows(options.get("sample_indices"), X.shape[0])
    supplied_names = options.get("feature_names")
    relation_manifest = getattr(dataset, "_relation_materialization_manifest", None)
    if relation_manifest is None:
        manifest = loaded.get("manifest", {})
        relation_manifest = manifest.get("relation_materialization_manifest") or manifest.get("relation_replay_manifest")
    relation = derive_relation_explain_lineage(relation_manifest, feature_names=supplied_names, n_features=X.shape[1])
    recorded_names = relation.feature_names if relation is not None else dataset.headers(0)
    default_names = recorded_names if recorded_names is not None and len(recorded_names) == X.shape[1] else [f"feature_{index}" for index in range(X.shape[1])]
    names = list(supplied_names) if supplied_names is not None else list(default_names)
    if len(names) != X.shape[1] or any(not isinstance(item, str) for item in names) or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique strings matching the raw input columns")
    # Relation aggregates are derived features: their headers are names, not coordinates.
    axis = recorded_axis(dataset, X.shape[1]) if relation is None else feature_index_axis(X.shape[1])
    captured = loaded["artifact"]["estimator"]
    # General exports retain the complete X-chain and optional target inverse.
    # Classification probabilities are read from that fitted X-chain only when
    # no target inverse changes its meaning; no class is trained or inferred here.
    estimator = getattr(captured, "estimator", captured)
    target_inverse = loaded["artifact"].get("y_transform")
    probabilities = callable(getattr(estimator, "predict_proba", None)) and getattr(captured, "y_transform", None) is None and target_inverse is None

    def function(values: Any) -> Any:
        if probabilities:
            return estimator.predict_proba(values)
        predicted = np.asarray(captured.predict(values))
        if target_inverse is None:
            return predicted
        restored = target_inverse.inverse_transform(predicted.reshape(len(values), -1))
        return np.asarray(restored)

    predictor = _CapturedPredictor(function, output_index)
    predictor.predict(X[:1])  # Validate output selection before lengthy SHAP work.
    classes = getattr(estimator, "classes_", None) if probabilities else None
    output_label = str(classes[output_index]) if classes is not None and output_index < len(classes) else None
    unit = "class probability" if probabilities else "original target"
    level = relation.explanation_level if relation is not None else "raw_observation"
    lineage = relation.feature_lineage if relation is not None else {feature: {"representation": "raw_input"} for feature in names}

    analyzer: Any = None
    if methods_settings is not None:
        from .methods_explain import explain_with_methods

        computed = explain_with_methods(
            function=function, probabilities=probabilities, captured=captured, target_inverse=target_inverse, X=X, rows=rows, names=names, axis=axis,
            output_index=output_index, output_label=output_label, background_samples=background, settings=methods_settings,
        )
        values = computed.values
        base_value: Any = computed.base_value
        result_names, explained_rows, explained_names = computed.labels, computed.explained_rows, computed.explained_names
        actual_explainer = computed.kind
        envelope = computed.envelope
        engine = envelope["engine"]
        if computed.attribution_space == "transformed_features":
            level = "transformed_feature"
            lineage = {feature: {"representation": "transformed_feature", "preprocessing": computed.preprocessing} for feature in explained_names}
        if len(result_names) != len(explained_names):
            lineage = {group["label"]: {"representation": "feature_group", "attribution_space": computed.attribution_space, "feature_start": group["feature_start"],
                                        "feature_end": group["feature_end"], "group_source": group["source"],
                                        "member_lineage": [lineage[feature] for feature in explained_names[group["feature_start"]:group["feature_end"]]]}
                       for group in envelope["groups"]}
        summary = (f"{actual_explainer} attributions from {engine['name']} {engine['version']} for captured REFIT predictor, {computed.attribution_space} space, "
                   f"{level} features, {unit} output {output_index}; {len(values)} rows, {len(result_names)} groups. No training performed.")
    else:
        import shap

        affine = not probabilities and _is_affine_predictor(captured) and _is_affine_transform(target_inverse)
        use_linear = explainer_type == "linear" or explainer_type == "auto" and affine and _raw_affine_parameters(predictor, X) is not None
        if use_linear:
            if not affine:
                raise ValueError("Linear SHAP requires a pipeline that is linear in its raw inputs; use auto or kernel.")
            # shap.LinearExplainer cannot read a complete captured chain; explain
            # its verified raw-input affine form exactly instead of sampling.
            values, base_value = _affine_shap(predictor, X, rows, background)
            explained_rows, actual_explainer = X[rows], "LinearExplainer"
            if visualizations:
                analyzer = _loaded_figure_analyzer(values, explained_rows, base_value, names, options)
        else:
            from nirs4all.visualization.analysis.shap import ShapAnalyzer

            selected_explainer = "permutation" if explainer_type == "auto" else explainer_type
            explained_model: Any = predictor
            if explainer_type == "tree" and not probabilities and output_index == 0:
                tree = _raw_tree_estimator(captured, target_inverse)
                if tree is None or np.asarray(tree.predict(X[:1])).size != 1:
                    raise ValueError("Tree SHAP requires an unchanged single-output tree predictor; use auto for a preprocessing pipeline.")
                explained_model = tree
            elif explainer_type in {"tree", "deep"}:
                raise ValueError(f"{explainer_type} SHAP cannot explain this complete captured pipeline; use auto or kernel.")
            analyzer = ShapAnalyzer()
            analyzed = analyzer.explain_model(
                explained_model, X, feature_names=names, sample_indices=rows if options.get("sample_indices") is not None else None,
                n_background=background, explainer_type=selected_explainer, visualizations=[], plots_visible=False,
                bin_size=options.get("bin_size", 20), bin_stride=options.get("bin_stride", 10),
                bin_aggregation=options.get("bin_aggregation", "sum"),
            )
            values = np.asarray(analyzed["shap_values"])
            base_value = analyzed["base_value"]
            explained_rows, actual_explainer = analyzer.data, analyzed["explainer_type"]
        result_names, explained_names = names, names
        kind, exactness = _SHAP_KINDS[actual_explainer]
        expected = np.full((len(values), 1), float(np.asarray(base_value).reshape(-1)[0]))
        prediction = predictor.predict(explained_rows).reshape(-1, 1)
        envelope = build_explanation_v1(
            kind=kind, engine={"name": "python-shap", "version": str(shap.__version__)}, exactness=exactness, additive=True, group_aggregation="none",
            attribution_space="raw_input", output={"index": 0, "label": output_label, "space": "class_probability" if probabilities else "target"},
            axis=axis, feature_names=names, groups=describe_groups(names, axis, None, len(names), ["per_feature"] * len(names)), sample_indices=rows,
            attributions=values[:, :, None], std_error=None, expected=expected, prediction=prediction,
            additivity_max_abs_error=float(np.max(np.abs(values.sum(axis=1)[:, None] + expected - prediction))),
            settings={"baseline": "background", "n_background": min(background, X.shape[0]),
                      **({"background_selection": "seeded_without_replacement", "seed": LINEAR_BACKGROUND_SEED} if actual_explainer == "LinearExplainer"
                         else {"background_selection": "seeded_without_replacement", "seed": 42}),
                      "explainer": actual_explainer, "model_output_index": output_index, "grouping": "per_feature"},
        )
        summary = f"SHAP for captured REFIT predictor, {level} input columns, {unit} output {output_index}; {len(values)} rows. No training performed."
    if relation is not None and relation.lineage_warning:
        summary += " " + relation.lineage_warning
    lineage = {feature: {**entry, "output_index": output_index, "output_space": unit} for feature, entry in lineage.items()}
    result = ExplainResult(
        shap_values=values, feature_names=list(result_names), base_value=base_value,
        explainer_type=actual_explainer, model_name=loaded["model_name"], n_samples=len(values),
        explanation_level=level, lineage_warning=summary, feature_lineage=lineage, explanation=envelope,
    )
    if verbose or plots_visible:
        print(summary)
        print(result.get_feature_importance(top_n=options.get("max_display", 20)))
    if visualizations or options.get("output_dir") is not None:
        workspace = Path(options.get("workspace_path", "workspace"))
        output = Path(options["output_dir"]) if options.get("output_dir") is not None else workspace / "explanations" / uuid4().hex
        output.mkdir(parents=True, exist_ok=True)
        result.to_dataframe().to_csv(output / "shap_values.csv", index=False)
        import pandas as pd

        pd.DataFrame(explained_rows, columns=explained_names).to_csv(output / "input_features.csv", index=False)
        (output / "explanation_v1.json").write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")
        source_provenance = loaded.get("source_provenance") or {
            "source_type": "archive", "archive_fingerprint": loaded["archive_fingerprint"],
            "artifact_integrity_verified": loaded["artifact_integrity_verified"],
        }
        provenance = {"contract": "nirs4all.python.shap.v1", "training_performed": False, "dataset_name": name,
                      **source_provenance, "explainer_backend": backend, "explainer_type": actual_explainer,
                      "attribution_space": envelope["attribution_space"], "explanation_schema": envelope["schema"],
                      "explanation_level": level, "feature_lineage": lineage,
                      "output_index": output_index, "output_space": unit, "base_value": np.asarray(result.base_value).tolist()}
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        table = result.to_dataframe().to_html(index=False, escape=True)
        (output / "summary.html").write_text(f"<!doctype html><html lang='en'><title>SHAP explanation</title><h1>SHAP explanation</h1><p>{html.escape(summary)}</p>{table}</html>", encoding="utf-8")
        result.visualizations["summary_table"] = output / "summary.html"
        result.visualizations["explanation_v1"] = output / "explanation_v1.json"
        if visualizations and analyzer is None:
            analyzer = _methods_figure_analyzer(computed, options)
        # Figures use the envelope axis: recorded coordinates and unit, or feature indices.
        axis_record = envelope["axis"]
        axis_label = {"wavelength": f"Wavelength ({axis_record['unit']})", "wavenumber": f"Wavenumber ({axis_record['unit']})"}.get(
            axis_record["kind"], "Transformed feature index" if envelope["attribution_space"] == "transformed_features" else "Feature index")
        axis_centres = [(group["axis_start"] + group["axis_end"]) / 2 for group in envelope["groups"]]
        # Publish the text and exact values before rendering any optional figure.
        for kind in visualizations:
            path = output / ("force.html" if kind == "force" else f"{kind}.png")
            # A short spectrum still forms one bin; rendering must not fail
            # merely because the historical default window has 20 columns.
            analyzer.bin_size = min(analyzer.bin_size_dict.get(kind, 20), len(result_names))
            analyzer.bin_stride = analyzer.bin_stride_dict.get(kind, 10)
            analyzer.bin_aggregation = analyzer.bin_aggregation_dict.get(kind, "sum")
            if kind == "waterfall":
                # A waterfall decomposes one prediction: disjoint bins with signed
                # sums cover every column, and the hidden bins form one remainder.
                analyzer.bin_stride, analyzer.bin_aggregation = analyzer.bin_size, "sum"
            kwargs: dict[str, Any] = {"output_path": str(path), "plots_visible": plots_visible}
            if kind in {"spectral", "summary", "force"}:
                kwargs["feature_names"] = list(result_names)
            if kind == "spectral":
                kwargs.update(axis_values=axis_centres, axis_label=axis_label)
            if kind in {"waterfall", "beeswarm"}:
                kwargs["feature_labels"] = list(result_names)
            if kind == "summary":
                kwargs["max_display"] = options.get("max_display", 20)
            methods: dict[str, Callable[..., Any]] = {"spectral": analyzer.plot_spectral_importance, "summary": analyzer.plot_summary,
                      "waterfall": analyzer.plot_waterfall_binned, "force": analyzer.plot_force,
                      "beeswarm": analyzer.plot_beeswarm_binned}
            methods[kind](**kwargs)
            result.visualizations[kind] = path
    return result


def _require_additive_waterfall_options(options: dict[str, Any]) -> None:
    """Refuse waterfall-specific binning that would break its additive decomposition.

    Shared ``bin_stride``/``bin_aggregation`` values configure the overlapping
    spectral and beeswarm summaries; the waterfall always uses disjoint bins and
    signed sums. Only an explicit waterfall entry that contradicts this is refused.
    """
    stride, size, aggregation = (options.get(name) for name in ("bin_stride", "bin_size", "bin_aggregation"))
    waterfall_stride = stride.get("waterfall") if isinstance(stride, dict) else None
    waterfall_size = size.get("waterfall") if isinstance(size, dict) else size
    waterfall_aggregation = aggregation.get("waterfall") if isinstance(aggregation, dict) else None
    if (waterfall_stride is not None and waterfall_stride != waterfall_size) or waterfall_aggregation not in (None, "sum"):
        raise ValueError("A waterfall decomposes one prediction additively: its bins must be disjoint (waterfall bin_stride equal to its bin_size) and summed "
                         "(bin_aggregation 'sum')")


def _loaded_figure_analyzer(values: np.ndarray, rows: np.ndarray, base_value: float, names: list[str], options: dict[str, Any]) -> Any:
    """Load precomputed per-feature attributions into the shared SHAP figure helpers."""
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    analyzer = ShapAnalyzer()
    analyzer.shap_values, analyzer.data, analyzer.base_value, analyzer.feature_names = values, rows, base_value, names
    for option, attribute, default in (("bin_size", "bin_size_dict", 20), ("bin_stride", "bin_stride_dict", 10), ("bin_aggregation", "bin_aggregation_dict", "sum")):
        value = options.get(option, default)
        setattr(analyzer, attribute, value if isinstance(value, dict) else dict.fromkeys(("spectral", "waterfall", "beeswarm"), value))
    return analyzer


def _methods_figure_analyzer(computed: Any, options: dict[str, Any]) -> Any:
    """Load Methods attributions into the shared SHAP figure helpers.

    Grouped attributions are drawn per group against the mean explained value
    of each group; groups are already the bins unless binning is requested.
    """
    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    analyzer = ShapAnalyzer()
    grouped = len(computed.labels) != len(computed.explained_names)
    analyzer.shap_values = computed.values
    analyzer.data = computed.group_rows
    analyzer.base_value = computed.base_value if isinstance(computed.base_value, float) else float(np.mean(computed.base_value))
    analyzer.feature_names = computed.labels
    defaults = {"bin_size": 1 if grouped else 20, "bin_stride": 1 if grouped else 10, "bin_aggregation": "sum"}
    for option, attribute in (("bin_size", "bin_size_dict"), ("bin_stride", "bin_stride_dict"), ("bin_aggregation", "bin_aggregation_dict")):
        value = options.get(option, defaults[option])
        setattr(analyzer, attribute, value if isinstance(value, dict) else dict.fromkeys(("spectral", "waterfall", "beeswarm"), value))
    return analyzer
