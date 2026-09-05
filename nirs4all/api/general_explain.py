"""SHAP over a captured scientific predictor, without replaying a legacy runner."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np

from nirs4all.pipeline.dagml.dataset import _materialize_dataset
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage

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
    if not isinstance(result, RunResult) or result.execution_engine != "dag-ml":
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


def explain_general(
    model: Any, data: Any, *, name: str, session: Any, verbose: int,
    plots_visible: bool, n_samples: int | None, explainer_type: str, options: dict[str, Any],
) -> ExplainResult:
    """Reuse the library SHAP analyzer on the complete frozen raw-input predictor.

    Preprocessing and target inverses stay inside the captured predictor. SHAP
    therefore explains original input columns and original target units, not a
    terminal estimator with an incorrectly labelled raw feature axis. Multi-output
    requests expose a labelled selected output, preserving the 2D ExplainResult.
    """
    supported = {"feature_names", "background_samples", "max_display", "visualizations", "output_dir", "workspace_path",
                 "bin_size", "bin_stride", "bin_aggregation", "sample_indices", "output_index"}
    if set(options) - supported:
        raise TypeError(f"Unsupported general SHAP options: {sorted(set(options) - supported)}")
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
    if session is not None:
        session._ensure_open()
    loaded = _load_captured_source(model, session, options.get("workspace_path"))
    dataset = _materialize_dataset(data)
    X = np.asarray(dataset.x({}, layout="2d"))
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("SHAP requires a nonempty finite 2D feature matrix")
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

    from nirs4all.visualization.analysis.shap import ShapAnalyzer

    analyzer = ShapAnalyzer()
    analyzed = analyzer.explain_model(
        predictor, X, feature_names=names, sample_indices=options.get("sample_indices"),
        n_background=background, explainer_type=explainer_type, visualizations=[], plots_visible=False,
        bin_size=options.get("bin_size", 20), bin_stride=options.get("bin_stride", 10),
        bin_aggregation=options.get("bin_aggregation", "sum"),
    )
    values = np.asarray(analyzed["shap_values"])
    unit = "class probability" if probabilities else "original target"
    level = relation.explanation_level if relation is not None else "raw_observation"
    summary = f"SHAP for captured REFIT predictor, {level} input columns, {unit} output {output_index}; {len(values)} rows. No training performed."
    if relation is not None and relation.lineage_warning:
        summary += " " + relation.lineage_warning
    lineage = relation.feature_lineage if relation is not None else {feature: {"representation": "raw_input"} for feature in names}
    lineage = {feature: {**entry, "output_index": output_index, "output_space": unit} for feature, entry in lineage.items()}
    result = ExplainResult(
        shap_values=values, feature_names=names, base_value=analyzed["base_value"],
        explainer_type=analyzed["explainer_type"], model_name=loaded["model_name"], n_samples=len(values),
        explanation_level=level, lineage_warning=summary, feature_lineage=lineage,
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

        pd.DataFrame(analyzer.data, columns=names).to_csv(output / "input_features.csv", index=False)
        source_provenance = loaded.get("source_provenance") or {
            "source_type": "archive", "archive_fingerprint": loaded["archive_fingerprint"],
            "artifact_integrity_verified": loaded["artifact_integrity_verified"],
        }
        provenance = {"contract": "nirs4all.python.shap.v1", "training_performed": False, "dataset_name": name,
                      **source_provenance,
                      "explanation_level": level, "feature_lineage": lineage,
                      "output_index": output_index, "output_space": unit, "base_value": np.asarray(result.base_value).tolist()}
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        table = result.to_dataframe().to_html(index=False, escape=True)
        (output / "summary.html").write_text(f"<!doctype html><html lang='en'><title>SHAP explanation</title><h1>SHAP explanation</h1><p>{html.escape(summary)}</p>{table}</html>", encoding="utf-8")
        result.visualizations["summary_table"] = output / "summary.html"
        # Publish the text and exact values before rendering any optional figure.
        for kind in visualizations:
            path = output / ("force.html" if kind == "force" else f"{kind}.png")
            # A short spectrum still forms one bin; rendering must not fail
            # merely because the historical default window has 20 columns.
            analyzer.bin_size = min(analyzer.bin_size_dict.get(kind, 20), X.shape[1])
            analyzer.bin_stride = analyzer.bin_stride_dict.get(kind, 10)
            analyzer.bin_aggregation = analyzer.bin_aggregation_dict.get(kind, "sum")
            kwargs: dict[str, Any] = {"output_path": str(path), "plots_visible": plots_visible}
            if kind in {"spectral", "summary", "force"}:
                kwargs["feature_names"] = names
            if kind == "summary":
                kwargs["max_display"] = options.get("max_display", 20)
            methods: dict[str, Callable[..., Any]] = {"spectral": analyzer.plot_spectral_importance, "summary": analyzer.plot_summary,
                      "waterfall": analyzer.plot_waterfall_binned, "force": analyzer.plot_force,
                      "beeswarm": analyzer.plot_beeswarm_binned}
            methods[kind](**kwargs)
            result.visualizations[kind] = path
    return result
