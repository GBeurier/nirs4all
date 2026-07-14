"""Contract snapshot: the stable public Python API surface.

This file locks the part of the 0.9.x stable contract that downstream code
(notably nirs4all-studio) binds to directly: the module-level entry points,
their exact call signatures, the package ``__all__`` exports, and the public
shape of the result objects.

Each assertion compares the *live* library against a frozen snapshot captured
from the current code. A failure means one of:

  * a public function signature changed (param added/removed/renamed, default
    changed, kw-only-ness changed) — ``test_<fn>_signature_frozen``;
  * an export was added to or removed from ``nirs4all.__all__`` /
    ``nirs4all.api.__all__`` — ``test_package_all_frozen`` /
    ``test_api_all_frozen``;
  * a public attribute/method disappeared from a result class —
    ``test_<Result>_public_surface``.

Signatures are compared as the full ``str(inspect.signature(fn))`` string
(this is the strictest contract: it catches default-value and annotation drift,
not just name changes). The ``__all__`` lists are compared exactly. Result
classes are checked with a *subset* assertion (the frozen names must remain a
subset of the live public surface) so that adding new helpers does not break
the test, while removing or renaming a documented one does.

Snapshots were originally captured from nirs4all 0.9.1 and deliberately
advanced as the native tuning/conformal public surface became part of the
contract. Updating them requires a deliberate decision because these are stable
downstream-facing contracts.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np

import nirs4all
import nirs4all.api as nirs4all_api

# ---------------------------------------------------------------------------
# Frozen signature snapshots.
#
# Captured via: str(inspect.signature(getattr(nirs4all, name)))
# These are the exact, full signature strings including annotations and
# defaults. Any drift (rename, reorder, default change, kw-only change) fails.
# ---------------------------------------------------------------------------

EXPECTED_SIGNATURES: dict[str, str] = {
    "run": (
        "(pipeline: list[typing.Any] | dict[str, typing.Any] | str | pathlib.Path | "
        "nirs4all.pipeline.config.pipeline_config.PipelineConfigs | list[list[typing.Any] | "
        "dict[str, typing.Any] | str | pathlib.Path | "
        "nirs4all.pipeline.config.pipeline_config.PipelineConfigs], "
        "dataset: str | pathlib.Path | numpy.ndarray | tuple[numpy.ndarray, ...] | "
        "dict[str, typing.Any] | nirs4all.data.dataset.SpectroDataset | "
        "nirs4all.data.config.DatasetConfigs | list[str | pathlib.Path | numpy.ndarray | "
        "tuple[numpy.ndarray, ...] | dict[str, typing.Any] | "
        "nirs4all.data.dataset.SpectroDataset | nirs4all.data.config.DatasetConfigs], *, "
        "name: str = '', session: nirs4all.api.session.Session | None = None, "
        "verbose: int = 1, save_artifacts: bool = True, save_charts: bool = True, "
        "plots_visible: bool = False, random_state: int | None = None, "
        "refit: bool | dict[str, typing.Any] | list[dict[str, typing.Any]] | None = True, "
        "cache: typing.Any | None = None, project: str | None = None, "
        "report_naming: str = 'nirs', engine: str | None = None, "
        "tuning: typing.Any | None = None, calibration: typing.Any | None = None, "
        "results_path: str | pathlib.Path | None = None, **runner_kwargs: Any) -> "
        "'RunResult | TunedSingleEstimatorConformalResult'"
    ),
    "predict": (
        "(model: 'ModelSpec | None' = None, data: 'DataSpec | None' = None, *, "
        "chain_id: 'str | None' = None, workspace_path: 'str | Path | None' = None, "
        "name: 'str' = 'prediction_dataset', all_predictions: 'bool' = False, "
        "session: 'Session | None' = None, verbose: 'int' = 0, "
        "coverage: 'float | list[float] | tuple[float, ...] | None' = None, "
        "save_to_workspace: 'bool' = False, workspace_metadata: 'Mapping[str, Any] | None' = None, "
        "workspace_result_metadata: 'Mapping[str, Any] | None' = None, "
        "**runner_kwargs: 'Any') -> 'PredictResult'"
    ),
    "explain": (
        "(model: dict[str, typing.Any] | str | pathlib.Path, "
        "data: str | pathlib.Path | numpy.ndarray | dict[str, typing.Any] | "
        "nirs4all.data.dataset.SpectroDataset | nirs4all.data.config.DatasetConfigs, *, "
        "name: str = 'explain_dataset', "
        "session: nirs4all.api.session.Session | None = None, verbose: int = 1, "
        "plots_visible: bool = True, n_samples: int | None = None, "
        "explainer_type: str = 'auto', **shap_params: Any) -> "
        "nirs4all.api.result.ExplainResult"
    ),
    "retrain": (
        "(source: dict[str, typing.Any] | str | pathlib.Path, "
        "data: str | pathlib.Path | numpy.ndarray | tuple[numpy.ndarray, ...] | "
        "dict[str, typing.Any] | nirs4all.data.dataset.SpectroDataset | "
        "nirs4all.data.config.DatasetConfigs, *, mode: str = 'full', "
        "name: str = 'retrain_dataset', new_model: typing.Any | None = None, "
        "epochs: int | None = None, "
        "session: nirs4all.api.session.Session | None = None, verbose: int = 1, "
        "save_artifacts: bool = True, **kwargs: Any) -> nirs4all.api.result.RunResult"
    ),
    "session": ("(pipeline: list[typing.Any] | None = None, name: str = '', **kwargs: Any) -> collections.abc.Generator[nirs4all.api.session.Session, None, None]"),
    "load_session": ("(path: str | pathlib.Path) -> nirs4all.api.session.Session"),
    "generate": (
        "(n_samples: 'int' = 1000, *, random_state: 'int | None' = None, "
        "complexity: \"Literal['simple', 'realistic', 'complex']\" = 'simple', "
        "wavelength_range: 'tuple[float, float] | None' = None, "
        "components: 'list[str] | None' = None, "
        "target_range: 'tuple[float, float] | None' = None, train_ratio: 'float' = 0.8, "
        "as_dataset: 'bool' = True, name: 'str' = 'synthetic_nirs', **kwargs: 'Any') -> "
        "'SpectroDataset | tuple[np.ndarray, np.ndarray]'"
    ),
    "tune_single_estimator": (
        "(pipeline: 'Any', X: 'Any', y: 'Any', tuning: 'DagMLTuningSpec | NativeTuning | "
        "Mapping[str, Any]', *, score_extractor: 'Callable[[Any], float] | None' = None, "
        "X_score: 'Any | None' = None, y_score: 'Any | None' = None, "
        "score_metric: 'str | None' = None, score_sample_ids: 'Any' = None, "
        "score_groups: 'Any' = None, score_metadata: 'Any' = None, sample_ids: 'Any' = None, "
        "groups: 'Any' = None, metadata: 'Any' = None, clone_estimator: 'bool' = True, "
        "refit: 'bool' = True, workspace_path: 'str | Path | None' = None, "
        "workspace_name: 'str' = '', workspace_tuning_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None, run_id: 'str | None' = None, "
        "pipeline_id: 'str | None' = None, chain_id: 'str | None' = None, "
        "resume_tuning_result: 'TuningResult | None' = None, "
        "resume_tuning_id: 'str | None' = None, per_dataset: 'Mapping[str, Any] | None' = None, "
        "winner_x: 'Any | None' = None, winner_y_true: 'Any | None' = None, "
        "winner_score: 'float | None' = None, winner_metric: 'str | None' = None, "
        "winner_sample_ids: 'Sequence[Any] | None' = None, "
        "winner_dataset_name: 'str' = 'tuning_winner', "
        "winner_model_name: 'str | None' = None, winner_task_type: 'str' = 'regression', "
        "winner_metadata: 'Mapping[str, Any] | None' = None, "
        "calibration: 'TuningCalibration | Mapping[str, Any] | None' = None) -> "
        "'RunResult | TunedSingleEstimatorConformalResult'"
    ),
    "calibrate": (
        "(calibration_data: 'ConformalCalibrationData | PredictResult | Mapping[str, Any] | tuple[Any, ...] | None' = None, *, "
        "y_true: 'Any | None' = None, "
        "y_pred_calibration: 'Any | None' = None, y_pred: 'Any | None' = None, "
        "calibration_sample_ids: 'Any | None' = None, "
        "prediction_sample_ids: 'Any | None' = None, "
        "coverage: 'float | list[float] | tuple[float, ...]' = 0.9, "
        "method: 'str' = 'split_absolute_residual', unit: 'str' = 'physical_sample', "
        "group_by: 'Any | None' = None, multi_target: 'str' = 'marginal', "
        "calibration_groups: 'Any | None' = None, calibration_metadata: 'Any | None' = None, "
        "prediction_groups: 'Any | None' = None, prediction_metadata: 'Any | None' = None, "
        "result_metadata: 'Mapping[str, Any] | None' = None, target_name: 'str | None' = None, "
        "predictor_fingerprint: 'str | None' = None, store_path: 'str | Path | None' = None, "
        "bundle_path: 'str | Path | None' = None, workspace_path: 'str | Path | None' = None, "
        "workspace_name: 'str' = '', workspace_conformal_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None, "
        "overwrite_store: 'bool' = False, as_predict_result: 'bool' = False) -> "
        "'CalibratedRunResult | PredictResult'"
    ),
    "predict_calibrated": (
        "(calibrated: 'CalibratedRunResult | str | Path', *, y_pred: 'Any', "
        "prediction_sample_ids: 'Any', result_metadata: 'Mapping[str, Any] | None' = None, "
        "as_predict_result: 'bool' = True) -> 'CalibratedRunResult | PredictResult'"
    ),
    "conformal_metrics": ("(calibrated: 'CalibratedRunResult | PredictResult | str | Path', *, y_true: 'Any') -> 'dict[float, ConformalMetricSet]'"),
    "robustness": (
        "(result: 'PredictResult | CalibratedRunResult', *, y_true: 'Any', X: 'Any | None' = None, "
        "predictor: 'Any | None' = None, predictor_bundle: 'str | Path | None' = None, "
        "mode: 'RobustnessMode' = 'clean_frozen', "
        "scenarios: 'Sequence[RobustnessScenarioSpec | Mapping[str, Any]] | None' = None, "
        "slice_by: 'Sequence[str] | None' = None, "
        "metadata: 'Mapping[str, Any] | Sequence[Mapping[str, Any]] | None' = None, "
        "seed: 'int | None' = None, workspace_path: 'str | Path | None' = None, "
        "workspace_name: 'str' = '', workspace_robustness_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None) -> 'RobustnessReport'"
    ),
    "robustness_from_workspace_prediction": (
        "(workspace_path: 'str | Path', prediction_id: 'str', *, y_true: 'Any', X: 'Any | None' = None, "
        "predictor: 'Any | None' = None, predictor_bundle: 'str | Path | None' = None, "
        "mode: 'RobustnessMode' = 'clean_frozen', "
        "scenarios: 'Sequence[RobustnessScenarioSpec | Mapping[str, Any]] | None' = None, "
        "slice_by: 'Sequence[str] | None' = None, "
        "metadata: 'Mapping[str, Any] | Sequence[Mapping[str, Any]] | None' = None, "
        "seed: 'int | None' = None, save_to_workspace: 'bool' = False, "
        "workspace_name: 'str' = '', workspace_robustness_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None) -> 'RobustnessReport'"
    ),
    "get_keyword_registry": ("() -> 'KeywordRegistry'"),
    "get_keyword_registry_schema": ("() -> 'dict[str, Any]'"),
    "get_robustness_summary_schema": ("() -> 'dict[str, Any]'"),
    "get_tuning_space_schema": ("() -> 'dict[str, Any]'"),
    "get_tuning_summary_schema": ("() -> 'dict[str, Any]'"),
    "inspect_tuning_space": ("(tuning: 'DagMLTuningSpec | NativeTuning | Mapping[str, Any]') -> 'dict[str, Any]'"),
    "keyword_registry_json": ("(*, indent: 'int | None' = 2) -> 'str'"),
    "keyword_registry_schema_json": ("(*, indent: 'int | None' = 2) -> 'str'"),
    "robustness_summary_schema_json": ("(*, indent: 'int | None' = 2) -> 'str'"),
    "tuning_space_schema_json": ("(*, indent: 'int | None' = 2) -> 'str'"),
    "tuning_summary_schema_json": ("(*, indent: 'int | None' = 2) -> 'str'"),
    "load_calibrated_result": ("(path: 'str | Path') -> 'CalibratedRunResult'"),
    "export_calibrated_result": ("(result: 'CalibratedRunResult', path: 'str | Path', *, overwrite: 'bool' = False) -> 'Path'"),
    "attach_calibrated_result_to_bundle": ("(model_bundle_path: 'str | Path', calibrated: 'CalibratedRunResult | str | Path', output_path: 'str | Path | None' = None, *, overwrite: 'bool' = False) -> 'Path'"),
    "save_workspace_calibrated_result": (
        "(workspace_path: 'str | Path', result: 'CalibratedRunResult', *, name: 'str' = '', "
        "conformal_id: 'str | None' = None, metadata: 'Mapping[str, Any] | None' = None, "
        "run_id: 'str | None' = None, pipeline_id: 'str | None' = None, "
        "chain_id: 'str | None' = None, prediction_id: 'str | None' = None) -> 'str'"
    ),
    "load_workspace_calibrated_result": ("(workspace_path: 'str | Path', conformal_id: 'str') -> 'CalibratedRunResult'"),
    "load_workspace_calibrated_predict_result": ("(workspace_path: 'str | Path', conformal_id: 'str') -> 'PredictResult'"),
    "load_workspace_predict_result": ("(workspace_path: 'str | Path', prediction_id: 'str') -> 'PredictResult'"),
    "load_workspace_predict_results": ("(workspace_path: 'str | Path', *, dataset_name: 'str | None' = None) -> 'list[PredictResult]'"),
    "save_workspace_predict_result": (
        "(workspace_path: 'str | Path', result: 'PredictResult', *, dataset_name: 'str' = 'prediction_dataset', "
        "metadata: 'Mapping[str, Any] | None' = None, result_metadata: 'Mapping[str, Any] | None' = None, "
        "X: 'Any | None' = None, spectra: 'Any | None' = None, sample_indices: 'Any | None' = None, "
        "model_name: 'str | None' = None, model_classname: 'str' = '', pipeline_id: 'str | None' = None, "
        "chain_id: 'str | None' = None, partition: 'str' = 'prediction', metric: 'str' = '', "
        "task_type: 'str' = 'regression', n_features: 'int | None' = None, "
        "preprocessings: 'str | Sequence[str] | None' = None) -> 'str'"
    ),
    "save_workspace_robustness_report": (
        "(workspace_path: 'str | Path', report: 'RobustnessReport', *, name: 'str' = '', "
        "robustness_id: 'str | None' = None, metadata: 'Mapping[str, Any] | None' = None, "
        "run_id: 'str | None' = None, pipeline_id: 'str | None' = None, "
        "chain_id: 'str | None' = None, conformal_id: 'str | None' = None, "
        "prediction_id: 'str | None' = None) -> 'str'"
    ),
    "load_workspace_robustness_report": ("(workspace_path: 'str | Path', robustness_id: 'str') -> 'RobustnessReport'"),
    "save_workspace_tuning_result": (
        "(workspace_path: 'str | Path', result: 'TuningResult', *, name: 'str' = '', "
        "tuning_id: 'str | None' = None, metadata: 'Mapping[str, Any] | None' = None, "
        "run_id: 'str | None' = None, pipeline_id: 'str | None' = None, "
        "chain_id: 'str | None' = None) -> 'str'"
    ),
    "load_workspace_tuning_result": ("(workspace_path: 'str | Path', tuning_id: 'str') -> 'TuningResult'"),
}

EXPECTED_METHOD_SIGNATURES: dict[str, str] = {
    "PredictResult.robustness": (
        "(self, *, y_true: 'Any', X: 'Any | None' = None, predictor: 'Any | None' = None, "
        "predictor_bundle: 'str | Path | None' = None, "
        "mode: 'str' = 'clean_frozen', scenarios: 'Sequence[Any] | None' = None, "
        "slice_by: 'Sequence[str] | None' = None, "
        "metadata: 'Mapping[str, Any] | Sequence[Mapping[str, Any]] | None' = None, "
        "seed: 'int | None' = None, workspace_path: 'str | Path | None' = None, "
        "workspace_name: 'str' = '', workspace_robustness_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None) -> 'RobustnessReport'"
    ),
    "CalibratedRunResult.to_dict": "(self) -> 'dict[str, Any]'",
    "CalibratedRunResult.to_json": "(self, *, indent: 'int | None' = 2) -> 'str'",
    "CalibratedRunResult.save_json": "(self, path: 'str | Path') -> 'Path'",
    "CalibratedRunResult.from_dict": "(payload: 'Mapping[str, Any]') -> 'CalibratedRunResult'",
    "CalibratedRunResult.from_json": "(payload: 'str') -> 'CalibratedRunResult'",
    "CalibratedRunResult.load_json": "(path: 'str | Path') -> 'CalibratedRunResult'",
    "CalibratedRunResult.robustness": (
        "(self, *, y_true: 'Any', X: 'Any | None' = None, predictor: 'Any | None' = None, "
        "predictor_bundle: 'str | Path | None' = None, "
        "mode: 'str' = 'clean_frozen', scenarios: 'Sequence[Any] | None' = None, "
        "slice_by: 'Sequence[str] | None' = None, "
        "metadata: 'Mapping[str, Any] | Sequence[Mapping[str, Any]] | None' = None, "
        "seed: 'int | None' = None, workspace_path: 'str | Path | None' = None, "
        "workspace_name: 'str' = '', workspace_robustness_id: 'str | None' = None, "
        "workspace_metadata: 'Mapping[str, Any] | None' = None) -> 'RobustnessReport'"
    ),
    "ConformalCalibrationData.to_dict": "(self) -> 'dict[str, Any]'",
    "ConformalMetricSet.to_dict": "(self) -> 'dict[str, Any]'",
    "RobustnessScenarioSpec.to_dict": "(self) -> 'dict[str, Any]'",
    "RobustnessReport.to_dict": "(self) -> 'dict[str, Any]'",
    "RobustnessReport.to_json": "(self, *, indent: 'int | None' = 2) -> 'str'",
    "RobustnessReport.save_json": "(self, path: 'str | Path') -> 'Path'",
    "RobustnessReport.summary_artifact": "(self) -> 'dict[str, Any]'",
    "RobustnessReport.to_summary_json": "(self, *, indent: 'int | None' = 2) -> 'str'",
    "RobustnessReport.save_summary": "(self, path: 'str | Path') -> 'Path'",
    "RobustnessReport.from_dict": "(payload: 'Mapping[str, Any]') -> 'RobustnessReport'",
    "RobustnessReport.from_json": "(payload: 'str') -> 'RobustnessReport'",
    "RobustnessReport.load_json": "(path: 'str | Path') -> 'RobustnessReport'",
    "RobustnessReport.to_markdown": "(self) -> 'str'",
    "RobustnessReport.save_markdown": "(self, path: 'str | Path') -> 'Path'",
    "RobustnessReport.to_html": "(self) -> 'str'",
    "RobustnessReport.save_html": "(self, path: 'str | Path') -> 'Path'",
    "RobustnessReport.save_parquet": "(self, path: 'str | Path') -> 'Path'",
    "RobustnessReport.save_artifacts": ("(self, path: 'str | Path', *, formats: 'Sequence[str]' = ('json', 'summary', 'markdown', 'html', 'parquet')) -> 'dict[str, Path]'"),
    "RobustnessReport.load_artifacts": "(path: 'str | Path') -> 'RobustnessReport'",
    "RobustnessReport.load_parquet": "(path: 'str | Path') -> 'RobustnessReport'",
    "RobustnessReport.summary_rows": "(self, *, reference: 'int' = 0, worst_slice_metric: 'str' = 'rmse') -> 'tuple[dict[str, Any], ...]'",
    "TuningResult.summary_artifact": "(self) -> 'dict[str, Any]'",
    "TuningResult.to_summary_json": "(self, *, indent: 'int | None' = 2) -> 'str'",
    "TuningResult.save_summary": "(self, path: 'str | Path') -> 'Path'",
}

# ---------------------------------------------------------------------------
# Frozen ``__all__`` snapshots (sorted).
# ---------------------------------------------------------------------------

EXPECTED_PACKAGE_ALL: list[str] = [
    "CONFORMAL_CALIBRATION_METHODS",
    "CONFORMAL_CALIBRATION_UNITS",
    "CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES",
    "CONFORMAL_MULTI_TARGET_POLICIES",
    "CONFORMAL_TUNING_SCORE_METRICS",
    "CONTROLLER_REGISTRY",
    "CalibratedRunResult",
    "ConformalCalibrationData",
    "ConformalMethod",
    "ConformalMetricSet",
    "ConformalMultiTarget",
    "ConformalUnit",
    "ExplainResult",
    "FINETUNE_APPROACHES",
    "FINETUNE_DAGML_APPROACHES",
    "FINETUNE_DAGML_DETERMINISTIC_ENGINES",
    "FINETUNE_DAGML_EVAL_MODES",
    "FINETUNE_DAGML_META_KEYS",
    "FINETUNE_DAGML_SELECTION_METRICS",
    "FINETUNE_ENGINES",
    "FINETUNE_ENGINE_ALIASES",
    "FINETUNE_EVAL_MODES",
    "FINETUNE_EVAL_MODE_ALIASES",
    "FINETUNE_N4M_PRUNERS",
    "FINETUNE_N4M_SAMPLERS",
    "FINETUNE_OPTUNA_PRUNERS",
    "FINETUNE_OPTUNA_SAMPLERS",
    "FINETUNE_SAMPLER_KEY_ALIASES",
    "FinetuneApproach",
    "FinetuneEngine",
    "FinetuneEvalMode",
    "FinetunePruner",
    "FinetuneSampler",
    "NativeTuning",
    "Nirs4AllCalibrationNotImplementedError",
    "OrderedSearchSpaceSpec",
    "ParameterPatch",
    "PipelineConfigs",
    "PipelineRunner",
    "PredictResult",
    "ROBUSTNESS_EXECUTABLE_MODES",
    "ROBUSTNESS_MODES",
    "ROBUSTNESS_SCENARIO_DISTRIBUTIONS",
    "ROBUSTNESS_SCENARIO_KINDS",
    "ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS",
    "RobustnessMetricSet",
    "RobustnessMode",
    "RobustnessReport",
    "RobustnessScenarioDistribution",
    "RobustnessScenarioKind",
    "RobustnessScenarioResult",
    "RobustnessScenarioSpec",
    "RobustnessSliceResult",
    "Run",
    "RunConfig",
    "RunResult",
    "RunStatus",
    "SearchSpaceParameter",
    "Session",
    "TUNING_CONTRACT_KEYS",
    "TUNING_DIRECTIONS",
    "TUNING_ENGINES",
    "TUNING_OPTIMIZER_PERSISTENCE_KEYS",
    "TUNING_RUNTIME_KEYS",
    "TUNING_SPACE_SCHEMA_ID",
    "TUNING_SUMMARY_SCHEMA_ID",
    "TrialResult",
    "TunedSingleEstimatorConformalResult",
    "TuningCalibration",
    "TuningConformalScoreCalibration",
    "TuningDirection",
    "TuningEngine",
    "TuningPassthrough",
    "TuningResult",
    "TuningScoreData",
    "TuningWinner",
    "attach_calibrated_result_to_bundle",
    "calibrate",
    "conformal_metrics",
    "explain",
    "export_calibrated_result",
    "framework",
    "generate",
    "generate_run_id",
    "get_keyword_registry",
    "get_keyword_registry_schema",
    "get_robustness_summary_schema",
    "get_tuning_space_schema",
    "get_tuning_summary_schema",
    "inspect_tuning_space",
    "is_gpu_available",
    "is_tensorflow_available",
    "keyword_registry_json",
    "keyword_registry_schema_json",
    "load_calibrated_result",
    "load_session",
    "load_workspace_calibrated_predict_result",
    "load_workspace_calibrated_result",
    "load_workspace_predict_result",
    "load_workspace_predict_results",
    "load_workspace_robustness_report",
    "load_workspace_tuning_result",
    "predict",
    "predict_calibrated",
    "register_controller",
    "retrain",
    "robustness",
    "robustness_from_workspace_prediction",
    "robustness_summary_schema_json",
    "run",
    "save_workspace_calibrated_result",
    "save_workspace_predict_result",
    "save_workspace_robustness_report",
    "save_workspace_tuning_result",
    "session",
    "tune_single_estimator",
    "tuning_space_schema_json",
    "tuning_summary_schema_json",
]

EXPECTED_API_ALL: list[str] = [
    "CONFORMAL_CALIBRATION_METHODS",
    "CONFORMAL_CALIBRATION_UNITS",
    "CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES",
    "CONFORMAL_MULTI_TARGET_POLICIES",
    "CONFORMAL_TUNING_SCORE_METRICS",
    "CalibratedRunResult",
    "ConformalCalibrationData",
    "ConformalMethod",
    "ConformalMetricSet",
    "ConformalMultiTarget",
    "ConformalUnit",
    "ExplainResult",
    "FINETUNE_APPROACHES",
    "FINETUNE_DAGML_APPROACHES",
    "FINETUNE_DAGML_DETERMINISTIC_ENGINES",
    "FINETUNE_DAGML_EVAL_MODES",
    "FINETUNE_DAGML_META_KEYS",
    "FINETUNE_DAGML_SELECTION_METRICS",
    "FINETUNE_ENGINES",
    "FINETUNE_ENGINE_ALIASES",
    "FINETUNE_EVAL_MODES",
    "FINETUNE_EVAL_MODE_ALIASES",
    "FINETUNE_N4M_PRUNERS",
    "FINETUNE_N4M_SAMPLERS",
    "FINETUNE_OPTUNA_PRUNERS",
    "FINETUNE_OPTUNA_SAMPLERS",
    "FINETUNE_SAMPLER_KEY_ALIASES",
    "FinetuneApproach",
    "FinetuneEngine",
    "FinetuneEvalMode",
    "FinetunePruner",
    "FinetuneSampler",
    "LazyModelRefitResult",
    "ModelRefitResult",
    "NativeTuning",
    "Nirs4AllCalibrationNotImplementedError",
    "OrderedSearchSpaceSpec",
    "ParameterPatch",
    "PredictResult",
    "ROBUSTNESS_EXECUTABLE_MODES",
    "ROBUSTNESS_MODES",
    "ROBUSTNESS_SCENARIO_DISTRIBUTIONS",
    "ROBUSTNESS_SCENARIO_KINDS",
    "ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS",
    "RobustnessMetricSet",
    "RobustnessMode",
    "RobustnessReport",
    "RobustnessScenarioDistribution",
    "RobustnessScenarioKind",
    "RobustnessScenarioResult",
    "RobustnessScenarioSpec",
    "RobustnessSliceResult",
    "RunResult",
    "SearchSpaceParameter",
    "Session",
    "TUNING_CONTRACT_KEYS",
    "TUNING_DIRECTIONS",
    "TUNING_ENGINES",
    "TUNING_OPTIMIZER_PERSISTENCE_KEYS",
    "TUNING_RUNTIME_KEYS",
    "TUNING_SPACE_SCHEMA_ID",
    "TUNING_SUMMARY_SCHEMA_ID",
    "TrialResult",
    "TunedSingleEstimatorConformalResult",
    "TuningCalibration",
    "TuningConformalScoreCalibration",
    "TuningDirection",
    "TuningEngine",
    "TuningPassthrough",
    "TuningResult",
    "TuningScoreData",
    "TuningWinner",
    "attach_calibrated_result_to_bundle",
    "calibrate",
    "conformal_metrics",
    "explain",
    "export_calibrated_result",
    "generate",
    "get_keyword_registry",
    "get_keyword_registry_schema",
    "get_robustness_summary_schema",
    "get_tuning_space_schema",
    "get_tuning_summary_schema",
    "inspect_tuning_space",
    "keyword_registry_json",
    "keyword_registry_schema_json",
    "load_calibrated_result",
    "load_session",
    "load_workspace_calibrated_predict_result",
    "load_workspace_calibrated_result",
    "load_workspace_predict_result",
    "load_workspace_predict_results",
    "load_workspace_robustness_report",
    "load_workspace_tuning_result",
    "predict",
    "predict_calibrated",
    "retrain",
    "robustness",
    "robustness_from_workspace_prediction",
    "robustness_summary_schema_json",
    "run",
    "save_workspace_calibrated_result",
    "save_workspace_predict_result",
    "save_workspace_robustness_report",
    "save_workspace_tuning_result",
    "session",
    "tune_single_estimator",
    "tuning_space_schema_json",
    "tuning_summary_schema_json",
]

# ---------------------------------------------------------------------------
# Frozen public surface of the result classes (captured from nirs4all 0.9.1).
#
# Subset contract: these names MUST remain present in the live class' public
# surface. Adding new public members is allowed and does not fail; removing or
# renaming one of these documented members does fail.
#
# The result classes are dataclasses, so the "live" public surface is
# ``set(dir(cls)) | {f.name for f in dataclasses.fields(cls)}``. ``dir(cls)``
# alone misses public INSTANCE fields that have no class-level default (e.g.
# ``RunResult.predictions`` / ``per_dataset``, ``PredictResult.y_pred``,
# ``ExplainResult.shap_values``); the dataclass-fields union pulls those in so
# they are guarded too.
# ---------------------------------------------------------------------------

EXPECTED_RUNRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "artifacts_path",
        "best",
        "best_accuracy",
        "best_final",
        "best_r2",
        "best_rmse",
        "best_score",
        "close",
        "cv_best",
        "cv_best_score",
        "detach",
        "export",
        "export_model",
        "filter",
        "final",
        "final_score",
        "get_datasets",
        "get_models",
        "models",
        "num_predictions",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "per_dataset",
        "predictions",
        "summary",
        "top",
        "tuning_best_params",
        "tuning_best_value",
        "tuning_id",
        "tuning_result",
        "validate",
    }
)

EXPECTED_PREDICTRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "conformal_guarantee_status",
        "flatten",
        "explanation_level",
        "feature_lineage",
        "from_prediction_record",
        "get_feature_lineage",
        "interval",
        "interval_coverages",
        "is_multioutput",
        "lineage_warning",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "intervals",
        "metadata",
        "model_name",
        "preprocessing_steps",
        "relation_materialization_manifest",
        "relation_replay_manifest",
        "robustness",
        "robustness_evidence",
        "sample_indices",
        "shape",
        "spectral_replay_evidence_status",
        "to_dataframe",
        "to_list",
        "to_numpy",
        "values",
        "y_pred",
    }
)

EXPECTED_EXPLAINRESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "base_value",
        "explainer_type",
        "explanation_level",
        "feature_names",
        "feature_lineage",
        "get_feature_lineage",
        "get_feature_importance",
        "get_sample_explanation",
        "lineage_warning",
        "mean_abs_shap",
        "model_name",
        "n_samples",
        "shape",
        # dataclass instance fields (missed by dir(cls), recovered via fields())
        "shap_values",
        "to_dataframe",
        "top_features",
        "values",
        "visualizations",
    }
)

EXPECTED_NATIVE_TUNING_MEMBERS: frozenset[str] = frozenset(
    {
        "calibration",
        "direction",
        "engine",
        "force_params",
        "metric",
        "n_trials",
        "pruner",
        "resume",
        "sampler",
        "score_data",
        "seed",
        "space",
        "storage",
        "study_name",
        "to_dict",
        "to_tuning_spec",
        "winner",
        "workspace_metadata",
        "workspace_tuning_id",
    }
)

EXPECTED_TUNING_SCORE_DATA_MEMBERS: frozenset[str] = frozenset(
    {
        "X",
        "X_score",
        "conformal_calibration",
        "conformal_coverage",
        "dataset",
        "group_column",
        "groups",
        "include_augmented",
        "metadata",
        "metadata_columns",
        "metric",
        "physical_sample_ids",
        "prediction_sample_ids",
        "score_groups",
        "score_metadata",
        "score_metric",
        "score_sample_ids",
        "sample_id_column",
        "sample_ids",
        "selector",
        "to_dict",
        "y",
        "y_score",
    }
)

EXPECTED_TUNING_CONFORMAL_SCORE_CALIBRATION_MEMBERS: frozenset[str] = frozenset(
    {
        "X",
        "X_calibration",
        "calibration_groups",
        "calibration_metadata",
        "calibration_sample_ids",
        "features",
        "groups",
        "metadata",
        "physical_sample_ids",
        "sample_ids",
        "target",
        "targets",
        "to_dict",
        "y",
        "y_calibration",
        "y_true",
    }
)

EXPECTED_TUNING_WINNER_MEMBERS: frozenset[str] = frozenset(
    {
        "X",
        "dataset",
        "dataset_name",
        "group_column",
        "include_augmented",
        "metadata",
        "metadata_columns",
        "metric",
        "model_name",
        "physical_sample_ids",
        "prediction_sample_ids",
        "sample_id_column",
        "sample_ids",
        "score",
        "selector",
        "task_type",
        "to_dict",
        "winner_sample_ids",
        "y_true",
    }
)

EXPECTED_TUNING_CALIBRATION_MEMBERS: frozenset[str] = frozenset(
    {
        "as_predict_result",
        "coverage",
        "extra",
        "method",
        "prediction_sample_ids",
        "to_dict",
        "unit",
        "workspace_conformal_id",
        "workspace_metadata",
        "y_pred",
    }
)

EXPECTED_TUNING_PASSTHROUGH_MEMBERS: frozenset[str] = frozenset({"to_dict"})

EXPECTED_CONFORMAL_CALIBRATION_DATA_MEMBERS: frozenset[str] = frozenset(
    {
        "dataset",
        "group_column",
        "groups",
        "include_augmented",
        "metadata",
        "metadata_columns",
        "model_bundle",
        "model_path",
        "predictor",
        "predictor_bundle",
        "predictor_chain_id",
        "predictor_fingerprint",
        "predictor_path",
        "predictor_result",
        "sample_id_column",
        "sample_ids",
        "selector",
        "to_dict",
        "workspace_path",
        "workspace_chain_id",
        "y_pred",
        "y_true",
    }
)

EXPECTED_ROBUSTNESS_METRIC_SET_MEMBERS: frozenset[str] = frozenset(
    {
        "bias",
        "fingerprint",
        "from_dict",
        "mae",
        "max_abs_error",
        "n_samples",
        "rmse",
        "to_dict",
        "version",
    }
)

EXPECTED_ROBUSTNESS_SCENARIO_SPEC_MEMBERS: frozenset[str] = frozenset(
    {
        "distribution",
        "extra",
        "kind",
        "severity",
        "to_dict",
    }
)

EXPECTED_ROBUSTNESS_SCENARIO_RESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "conformal_metrics",
        "fingerprint",
        "from_dict",
        "metrics",
        "scenario",
        "severity",
        "slices",
        "to_dict",
        "version",
    }
)

EXPECTED_ROBUSTNESS_SLICE_RESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "conformal_metrics",
        "fingerprint",
        "from_dict",
        "metrics",
        "slice_key",
        "to_dict",
        "version",
    }
)

EXPECTED_ROBUSTNESS_REPORT_MEMBERS: frozenset[str] = frozenset(
    {
        "degradation_rows",
        "fingerprint",
        "from_dict",
        "from_json",
        "load_artifacts",
        "load_json",
        "load_parquet",
        "metadata",
        "mode",
        "save_artifacts",
        "save_html",
        "save_json",
        "save_markdown",
        "save_parquet",
        "save_summary",
        "summary_artifact",
        "summary_rows",
        "scenarios",
        "slice_by",
        "tabular_records",
        "to_dict",
        "to_html",
        "to_json",
        "to_markdown",
        "to_summary_json",
        "version",
        "worst_slices",
    }
)

EXPECTED_CALIBRATED_RUN_RESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "artifact",
        "conformal_guarantee_status",
        "fingerprint",
        "from_dict",
        "from_json",
        "load_json",
        "metadata",
        "metrics",
        "prediction",
        "robustness",
        "sample_ids",
        "save_json",
        "to_dict",
        "to_json",
        "version",
    }
)

EXPECTED_CONFORMAL_METRIC_SET_MEMBERS: frozenset[str] = frozenset(
    {
        "coverage",
        "coverage_gap",
        "fingerprint",
        "mean_interval_score",
        "mean_width",
        "median_width",
        "n_covered",
        "n_missed_above",
        "n_missed_below",
        "n_samples",
        "observed_coverage",
        "to_dict",
        "unit",
        "version",
    }
)

EXPECTED_TUNED_SINGLE_ESTIMATOR_CONFORMAL_RESULT_MEMBERS: frozenset[str] = frozenset(
    {
        "calibrated",
        "conformal_guarantee_status",
        "interval",
        "interval_coverages",
        "metrics",
        "robustness",
        "run",
        "tuning_best_params",
        "tuning_best_value",
        "tuning_id",
        "tuning_result",
    }
)


def _public_members(cls: type) -> set[str]:
    """Public (non-underscore) attribute/method/field names exposed by a class.

    Unions ``dir(cls)`` with the dataclass field names so public instance
    fields that have no class-level default (and therefore never appear in
    ``dir(cls)``) are included in the live surface.
    """
    names = set(dir(cls))
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    return {name for name in names if not name.startswith("_")}


# ---------------------------------------------------------------------------
# Signature contracts.
# ---------------------------------------------------------------------------


def test_public_functions_are_importable() -> None:
    """All documented entry points are importable from the top package."""
    for name in EXPECTED_SIGNATURES:
        assert hasattr(nirs4all, name), f"nirs4all.{name} is missing"
        assert callable(getattr(nirs4all, name)), f"nirs4all.{name} is not callable"


def _sig(fn: object) -> str:
    """Render a signature for comparison, normalized across interpreter builds.

    Python 3.13 moved ``Path`` into ``pathlib._local``, and some interpreter
    builds leak that internal module in annotation reprs. The public contract is
    the ``pathlib.Path`` spelling, so normalize before comparing.
    """
    return str(inspect.signature(fn)).replace("pathlib._local.Path", "pathlib.Path")  # type: ignore[arg-type]


def test_run_signature_frozen() -> None:
    assert _sig(nirs4all.run) == EXPECTED_SIGNATURES["run"]


def test_predict_signature_frozen() -> None:
    assert _sig(nirs4all.predict) == EXPECTED_SIGNATURES["predict"]


def test_explain_signature_frozen() -> None:
    assert _sig(nirs4all.explain) == EXPECTED_SIGNATURES["explain"]


def test_retrain_signature_frozen() -> None:
    assert _sig(nirs4all.retrain) == EXPECTED_SIGNATURES["retrain"]


def test_session_signature_frozen() -> None:
    assert _sig(nirs4all.session) == EXPECTED_SIGNATURES["session"]


def test_load_session_signature_frozen() -> None:
    assert _sig(nirs4all.load_session) == EXPECTED_SIGNATURES["load_session"]


def test_generate_signature_frozen() -> None:
    assert _sig(nirs4all.generate) == EXPECTED_SIGNATURES["generate"]


def test_tune_single_estimator_signature_frozen() -> None:
    assert _sig(nirs4all.tune_single_estimator) == EXPECTED_SIGNATURES["tune_single_estimator"]


def test_conformal_and_robustness_signatures_frozen() -> None:
    for name in (
        "calibrate",
        "predict_calibrated",
        "conformal_metrics",
        "robustness",
        "robustness_from_workspace_prediction",
        "get_keyword_registry",
        "get_keyword_registry_schema",
        "get_robustness_summary_schema",
        "get_tuning_space_schema",
        "get_tuning_summary_schema",
        "keyword_registry_json",
        "keyword_registry_schema_json",
        "robustness_summary_schema_json",
        "tuning_space_schema_json",
        "tuning_summary_schema_json",
        "load_calibrated_result",
        "export_calibrated_result",
        "attach_calibrated_result_to_bundle",
        "save_workspace_calibrated_result",
        "load_workspace_calibrated_predict_result",
        "load_workspace_calibrated_result",
        "load_workspace_predict_result",
        "load_workspace_predict_results",
        "save_workspace_predict_result",
        "save_workspace_tuning_result",
        "load_workspace_tuning_result",
    ):
        assert _sig(getattr(nirs4all, name)) == EXPECTED_SIGNATURES[name]


def test_conformal_and_robustness_artifact_method_signatures_frozen() -> None:
    methods = {
        "PredictResult.robustness": nirs4all.PredictResult.robustness,
        "CalibratedRunResult.to_dict": nirs4all.CalibratedRunResult.to_dict,
        "CalibratedRunResult.to_json": nirs4all.CalibratedRunResult.to_json,
        "CalibratedRunResult.save_json": nirs4all.CalibratedRunResult.save_json,
        "CalibratedRunResult.from_dict": nirs4all.CalibratedRunResult.from_dict,
        "CalibratedRunResult.from_json": nirs4all.CalibratedRunResult.from_json,
        "CalibratedRunResult.load_json": nirs4all.CalibratedRunResult.load_json,
        "CalibratedRunResult.robustness": nirs4all.CalibratedRunResult.robustness,
        "ConformalCalibrationData.to_dict": nirs4all.ConformalCalibrationData.to_dict,
        "ConformalMetricSet.to_dict": nirs4all.ConformalMetricSet.to_dict,
        "RobustnessScenarioSpec.to_dict": nirs4all.RobustnessScenarioSpec.to_dict,
        "RobustnessReport.to_dict": nirs4all.RobustnessReport.to_dict,
        "RobustnessReport.to_json": nirs4all.RobustnessReport.to_json,
        "RobustnessReport.save_json": nirs4all.RobustnessReport.save_json,
        "RobustnessReport.summary_artifact": nirs4all.RobustnessReport.summary_artifact,
        "RobustnessReport.to_summary_json": nirs4all.RobustnessReport.to_summary_json,
        "RobustnessReport.save_summary": nirs4all.RobustnessReport.save_summary,
        "RobustnessReport.from_dict": nirs4all.RobustnessReport.from_dict,
        "RobustnessReport.from_json": nirs4all.RobustnessReport.from_json,
        "RobustnessReport.load_json": nirs4all.RobustnessReport.load_json,
        "RobustnessReport.to_markdown": nirs4all.RobustnessReport.to_markdown,
        "RobustnessReport.save_markdown": nirs4all.RobustnessReport.save_markdown,
        "RobustnessReport.to_html": nirs4all.RobustnessReport.to_html,
        "RobustnessReport.save_html": nirs4all.RobustnessReport.save_html,
        "RobustnessReport.save_parquet": nirs4all.RobustnessReport.save_parquet,
        "RobustnessReport.save_artifacts": nirs4all.RobustnessReport.save_artifacts,
        "RobustnessReport.load_artifacts": nirs4all.RobustnessReport.load_artifacts,
        "RobustnessReport.load_parquet": nirs4all.RobustnessReport.load_parquet,
        "RobustnessReport.summary_rows": nirs4all.RobustnessReport.summary_rows,
        "TuningResult.summary_artifact": nirs4all.TuningResult.summary_artifact,
        "TuningResult.to_summary_json": nirs4all.TuningResult.to_summary_json,
        "TuningResult.save_summary": nirs4all.TuningResult.save_summary,
    }
    for name, method in methods.items():
        assert _sig(method) == EXPECTED_METHOD_SIGNATURES[name]


def test_native_tuning_conformal_helper_exports_are_public() -> None:
    """Typed native tuning/conformal helpers are intentionally public exports."""

    assert nirs4all.CalibratedRunResult is nirs4all_api.CalibratedRunResult
    assert nirs4all.ConformalCalibrationData is nirs4all_api.ConformalCalibrationData
    assert nirs4all.ConformalMetricSet is nirs4all_api.ConformalMetricSet
    assert nirs4all.CONFORMAL_CALIBRATION_METHODS is nirs4all_api.CONFORMAL_CALIBRATION_METHODS
    assert nirs4all.CONFORMAL_CALIBRATION_UNITS is nirs4all_api.CONFORMAL_CALIBRATION_UNITS
    assert nirs4all.CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES is nirs4all_api.CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES
    assert nirs4all.CONFORMAL_MULTI_TARGET_POLICIES is nirs4all_api.CONFORMAL_MULTI_TARGET_POLICIES
    assert nirs4all.CONFORMAL_TUNING_SCORE_METRICS is nirs4all_api.CONFORMAL_TUNING_SCORE_METRICS
    assert nirs4all.ConformalMethod is nirs4all_api.ConformalMethod
    assert nirs4all.ConformalMultiTarget is nirs4all_api.ConformalMultiTarget
    assert nirs4all.ConformalUnit is nirs4all_api.ConformalUnit
    assert nirs4all.FINETUNE_APPROACHES is nirs4all_api.FINETUNE_APPROACHES
    assert nirs4all.FINETUNE_DAGML_APPROACHES is nirs4all_api.FINETUNE_DAGML_APPROACHES
    assert nirs4all.FINETUNE_DAGML_DETERMINISTIC_ENGINES is nirs4all_api.FINETUNE_DAGML_DETERMINISTIC_ENGINES
    assert nirs4all.FINETUNE_DAGML_EVAL_MODES is nirs4all_api.FINETUNE_DAGML_EVAL_MODES
    assert nirs4all.FINETUNE_DAGML_META_KEYS is nirs4all_api.FINETUNE_DAGML_META_KEYS
    assert nirs4all.FINETUNE_DAGML_SELECTION_METRICS is nirs4all_api.FINETUNE_DAGML_SELECTION_METRICS
    assert nirs4all.FINETUNE_ENGINES is nirs4all_api.FINETUNE_ENGINES
    assert nirs4all.FINETUNE_ENGINE_ALIASES is nirs4all_api.FINETUNE_ENGINE_ALIASES
    assert nirs4all.FINETUNE_EVAL_MODE_ALIASES is nirs4all_api.FINETUNE_EVAL_MODE_ALIASES
    assert nirs4all.FINETUNE_EVAL_MODES is nirs4all_api.FINETUNE_EVAL_MODES
    assert nirs4all.FINETUNE_N4M_PRUNERS is nirs4all_api.FINETUNE_N4M_PRUNERS
    assert nirs4all.FINETUNE_N4M_SAMPLERS is nirs4all_api.FINETUNE_N4M_SAMPLERS
    assert nirs4all.FINETUNE_OPTUNA_PRUNERS is nirs4all_api.FINETUNE_OPTUNA_PRUNERS
    assert nirs4all.FINETUNE_OPTUNA_SAMPLERS is nirs4all_api.FINETUNE_OPTUNA_SAMPLERS
    assert nirs4all.FINETUNE_SAMPLER_KEY_ALIASES is nirs4all_api.FINETUNE_SAMPLER_KEY_ALIASES
    assert nirs4all.FinetuneApproach is nirs4all_api.FinetuneApproach
    assert nirs4all.FinetuneEngine is nirs4all_api.FinetuneEngine
    assert nirs4all.FinetuneEvalMode is nirs4all_api.FinetuneEvalMode
    assert nirs4all.FinetunePruner is nirs4all_api.FinetunePruner
    assert nirs4all.FinetuneSampler is nirs4all_api.FinetuneSampler
    assert nirs4all.get_keyword_registry is nirs4all_api.get_keyword_registry
    assert nirs4all.get_keyword_registry_schema is nirs4all_api.get_keyword_registry_schema
    assert nirs4all.get_robustness_summary_schema is nirs4all_api.get_robustness_summary_schema
    assert nirs4all.get_tuning_space_schema is nirs4all_api.get_tuning_space_schema
    assert nirs4all.get_tuning_summary_schema is nirs4all_api.get_tuning_summary_schema
    assert nirs4all.keyword_registry_json is nirs4all_api.keyword_registry_json
    assert nirs4all.keyword_registry_schema_json is nirs4all_api.keyword_registry_schema_json
    assert nirs4all.robustness_summary_schema_json is nirs4all_api.robustness_summary_schema_json
    assert nirs4all.tuning_space_schema_json is nirs4all_api.tuning_space_schema_json
    assert nirs4all.tuning_summary_schema_json is nirs4all_api.tuning_summary_schema_json
    assert nirs4all.NativeTuning is nirs4all_api.NativeTuning
    assert nirs4all.TUNING_CONTRACT_KEYS is nirs4all_api.TUNING_CONTRACT_KEYS
    assert nirs4all.TUNING_DIRECTIONS is nirs4all_api.TUNING_DIRECTIONS
    assert nirs4all.TUNING_ENGINES is nirs4all_api.TUNING_ENGINES
    assert nirs4all.TUNING_OPTIMIZER_PERSISTENCE_KEYS is nirs4all_api.TUNING_OPTIMIZER_PERSISTENCE_KEYS
    assert nirs4all.TUNING_RUNTIME_KEYS is nirs4all_api.TUNING_RUNTIME_KEYS
    assert nirs4all.TUNING_SPACE_SCHEMA_ID is nirs4all_api.TUNING_SPACE_SCHEMA_ID
    assert nirs4all.TUNING_SUMMARY_SCHEMA_ID is nirs4all_api.TUNING_SUMMARY_SCHEMA_ID
    assert nirs4all.ROBUSTNESS_EXECUTABLE_MODES is nirs4all_api.ROBUSTNESS_EXECUTABLE_MODES
    assert nirs4all.ROBUSTNESS_MODES is nirs4all_api.ROBUSTNESS_MODES
    assert nirs4all.ROBUSTNESS_SCENARIO_DISTRIBUTIONS is nirs4all_api.ROBUSTNESS_SCENARIO_DISTRIBUTIONS
    assert nirs4all.ROBUSTNESS_SCENARIO_KINDS is nirs4all_api.ROBUSTNESS_SCENARIO_KINDS
    assert nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS is nirs4all_api.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS
    assert nirs4all.RobustnessMode is nirs4all_api.RobustnessMode
    assert nirs4all.RobustnessScenarioDistribution is nirs4all_api.RobustnessScenarioDistribution
    assert nirs4all.RobustnessScenarioKind is nirs4all_api.RobustnessScenarioKind
    assert nirs4all.RobustnessScenarioSpec is nirs4all_api.RobustnessScenarioSpec
    assert nirs4all.TuningScoreData is nirs4all_api.TuningScoreData
    assert nirs4all.TuningConformalScoreCalibration is nirs4all_api.TuningConformalScoreCalibration
    assert nirs4all.TuningDirection is nirs4all_api.TuningDirection
    assert nirs4all.TuningEngine is nirs4all_api.TuningEngine
    assert nirs4all.inspect_tuning_space is nirs4all_api.inspect_tuning_space
    assert nirs4all.OrderedSearchSpaceSpec is nirs4all_api.OrderedSearchSpaceSpec
    assert nirs4all.ParameterPatch is nirs4all_api.ParameterPatch
    assert nirs4all.SearchSpaceParameter is nirs4all_api.SearchSpaceParameter
    assert nirs4all.TuningPassthrough is nirs4all_api.TuningPassthrough
    assert nirs4all.TuningResult is nirs4all_api.TuningResult
    assert nirs4all.TuningWinner is nirs4all_api.TuningWinner
    assert nirs4all.TuningCalibration is nirs4all_api.TuningCalibration
    assert nirs4all.TrialResult is nirs4all_api.TrialResult


def test_native_tuning_helper_serialization_contract() -> None:
    """Typed helper ``to_dict()`` output remains stable for downstream tooling."""

    tuning = nirs4all.NativeTuning(
        space={"scale": [nirs4all.TuningPassthrough()], "model.alpha": [0.1, 1.0]},
        force_params={"scale": nirs4all.TuningPassthrough(), "model.alpha": 0.1},
        engine="optuna",
        metric="conformal_mean_width",
        direction="minimize",
        n_trials=2,
        sampler="grid",
        seed=7,
        storage="sqlite:///study.db",
        study_name="contract-study",
        score_data=nirs4all.TuningScoreData(
            X=[[1.0], [2.0]],
            y=[0.1, 0.2],
            metric="conformal_mean_width",
            sample_ids=["s1", "s2"],
            groups=["g1", "g2"],
            metadata=[{"site": "a"}, {"site": "b"}],
            conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                X=[[3.0], [4.0], [5.0]],
                y_true=[0.3, 0.4, 0.5],
                physical_sample_ids=["c1", "c2", "c3"],
                groups=["cg1", "cg2", "cg3"],
                metadata={"role": "dev-cal"},
            ),
            conformal_coverage=0.8,
        ),
        winner=nirs4all.TuningWinner(
            X=[[6.0], [7.0]],
            y_true=[0.6, 0.7],
            score=0.12,
            metric="rmse",
            sample_ids=["w1", "w2"],
            model_name="WinnerModel",
            metadata={"fold": "holdout"},
        ),
        calibration=nirs4all.TuningCalibration(
            y_pred=[0.61, 0.72],
            prediction_sample_ids=["p1", "p2"],
            coverage=[0.8, 0.9],
            workspace_conformal_id="conf-main",
            workspace_metadata={"purpose": "contract"},
            extra={"target_name": "protein"},
        ),
        workspace_tuning_id="tune-main",
        workspace_metadata={"purpose": "contract"},
    )

    assert tuning.to_dict() == {
        "direction": "minimize",
        "engine": "optuna",
        "force_params": {"scale": {"kind": "passthrough"}, "model.alpha": 0.1},
        "metric": "conformal_mean_width",
        "n_trials": 2,
        "pruner": None,
        "resume": False,
        "space": {"scale": [{"kind": "passthrough"}], "model.alpha": [0.1, 1.0]},
        "sampler": "grid",
        "seed": 7,
        "storage": "sqlite:///study.db",
        "study_name": "contract-study",
        "score_data": {
            "X": [[1.0], [2.0]],
            "y": [0.1, 0.2],
            "metric": "conformal_mean_width",
            "sample_ids": ["s1", "s2"],
            "groups": ["g1", "g2"],
            "metadata": [{"site": "a"}, {"site": "b"}],
            "conformal_calibration": {
                "X": [[3.0], [4.0], [5.0]],
                "y_true": [0.3, 0.4, 0.5],
                "sample_ids": ["c1", "c2", "c3"],
                "groups": ["cg1", "cg2", "cg3"],
                "metadata": {"role": "dev-cal"},
            },
            "conformal_coverage": 0.8,
        },
        "winner": {
            "X": [[6.0], [7.0]],
            "y_true": [0.6, 0.7],
            "score": 0.12,
            "metric": "rmse",
            "sample_ids": ["w1", "w2"],
            "dataset_name": "tuning_winner",
            "model_name": "WinnerModel",
            "task_type": "regression",
            "metadata": {"fold": "holdout"},
        },
        "calibration": {
            "as_predict_result": True,
            "coverage": [0.8, 0.9],
            "method": "split_absolute_residual",
            "prediction_sample_ids": ["p1", "p2"],
            "unit": "physical_sample",
            "y_pred": [0.61, 0.72],
            "workspace_conformal_id": "conf-main",
            "workspace_metadata": {"purpose": "contract"},
            "target_name": "protein",
        },
        "workspace_tuning_id": "tune-main",
        "workspace_metadata": {"purpose": "contract"},
    }


def test_native_tuning_dataset_backed_helper_serialization_contract() -> None:
    """Dataset-backed typed helper payloads remain stable for forms/Studio."""

    assert nirs4all.TuningScoreData(
        dataset="dataset.json",
        selector={"partition": "score"},
        sample_id_column="Sample_ID",
        score_sample_ids=["s1", "s2"],
        group_column="Batch",
        metadata_columns=["Site"],
        include_augmented=True,
    ).to_dict() == {
        "dataset": "dataset.json",
        "include_augmented": True,
        "selector": {"partition": "score"},
        "sample_id_column": "Sample_ID",
        "sample_ids": ["s1", "s2"],
        "group_column": "Batch",
        "metadata_columns": ["Site"],
    }

    assert nirs4all.TuningWinner(
        dataset="dataset.json",
        selector={"partition": "winner"},
        sample_id_column="Sample_ID",
        physical_sample_ids=["w1", "w2"],
        metadata_columns="Site",
        include_augmented=True,
        score=0.2,
        metric="rmse",
    ).to_dict() == {
        "dataset": "dataset.json",
        "include_augmented": True,
        "selector": {"partition": "winner"},
        "sample_id_column": "Sample_ID",
        "sample_ids": ["w1", "w2"],
        "metadata_columns": "Site",
        "score": 0.2,
        "metric": "rmse",
        "dataset_name": "tuning_winner",
        "task_type": "regression",
    }


def test_conformal_calibration_data_serialization_contract() -> None:
    """Typed conformal calibration payloads remain stable for forms/Studio."""

    assert nirs4all.ConformalCalibrationData(
        y_true=[1.0, 2.0],
        y_pred=[0.9, 2.1],
        sample_ids=["c1", "c2"],
        groups=["g1", "g2"],
        metadata={"site": ["a", "b"]},
        predictor_fingerprint="predictor-contract",
    ).to_dict() == {
        "y_pred": [0.9, 2.1],
        "y_true": [1.0, 2.0],
        "sample_ids": ["c1", "c2"],
        "groups": ["g1", "g2"],
        "metadata": {"site": ["a", "b"]},
        "predictor_fingerprint": "predictor-contract",
    }

    assert nirs4all.ConformalCalibrationData(
        dataset="dataset.json",
        selector={"partition": "calibration"},
        sample_id_column="Sample_ID",
        group_column="Batch",
        metadata_columns=["Site"],
        include_augmented=True,
        predictor_bundle="model.n4a",
        workspace_path="workspace",
    ).to_dict() == {
        "dataset": "dataset.json",
        "include_augmented": True,
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "group_column": "Batch",
        "metadata_columns": ["Site"],
        "predictor_bundle": "model.n4a",
        "workspace_path": "workspace",
    }

    assert nirs4all.ConformalCalibrationData(
        dataset="dataset.json",
        selector={"partition": "calibration"},
        sample_id_column="Sample_ID",
        model_bundle="model-alias.n4a",
        workspace_chain_id=None,
    ).to_dict() == {
        "dataset": "dataset.json",
        "include_augmented": False,
        "selector": {"partition": "calibration"},
        "sample_id_column": "Sample_ID",
        "predictor_bundle": "model-alias.n4a",
    }


def test_robustness_scenario_spec_serialization_contract() -> None:
    """Typed robustness scenario payloads remain stable for forms/Studio."""

    assert nirs4all.RobustnessScenarioSpec(
        kind="prediction_noise",
        severity=0.25,
        distribution="normal",
        extra={"label": "seeded"},
    ).to_dict() == {
        "distribution": "normal",
        "kind": "prediction_noise",
        "label": "seeded",
        "severity": 0.25,
    }


def test_conformal_metric_set_serialization_contract() -> None:
    """Public conformal metric serialization remains stable for reports."""

    metric = nirs4all.ConformalMetricSet(
        coverage=0.8,
        observed_coverage=0.75,
        coverage_gap=-0.05,
        mean_width=1.25,
        median_width=1.0,
        mean_interval_score=1.5,
        n_samples=4,
        n_covered=3,
        n_missed_below=1,
        n_missed_above=0,
    )

    assert metric.to_dict() == {
        "coverage": 0.8,
        "coverage_gap": -0.05,
        "fingerprint": "6e92d26975adb97e095c9947f8ac373de132603c198908821a64439cc3fd6b4a",
        "mean_interval_score": 1.5,
        "mean_width": 1.25,
        "median_width": 1.0,
        "n_covered": 3,
        "n_missed_above": 0,
        "n_missed_below": 1,
        "n_samples": 4,
        "observed_coverage": 0.75,
        "unit": "physical_sample",
        "version": 1,
    }


def test_calibrated_run_result_serialization_contract() -> None:
    """Public calibrated result serialization remains stable for artifacts."""

    result = nirs4all.calibrate(
        y_true=[0.0, 1.0, 2.0, 3.0],
        y_pred_calibration=[0.0, 1.2, 1.8, 3.5],
        y_pred=[0.5, 1.5],
        calibration_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        prediction_sample_ids=["pred-a", "pred-b"],
        coverage=0.8,
        result_metadata={"source": "contract"},
    )

    assert isinstance(result, nirs4all.CalibratedRunResult)
    payload = result.to_dict()
    assert set(payload) == {"artifact", "fingerprint", "metadata", "prediction", "sample_ids", "version"}
    assert payload["fingerprint"] == "eeca915fe96f32ea4a5a7f4380a5c218efd5da6d02db4baf26cd720c2124586f"
    assert payload["sample_ids"] == ["pred-a", "pred-b"]
    assert payload["version"] == 1
    assert payload["artifact"]["spec"] == {
        "coverage": [0.8],
        "group_by": [],
        "method": "split_absolute_residual",
        "multi_target": "marginal",
        "unit": "physical_sample",
    }
    assert payload["artifact"]["calibration_size"] == 4
    assert payload["artifact"]["qhat_by_coverage"] == [{"coverage": 0.8, "qhat": 0.5}]
    assert payload["prediction"] == {
        "group_keys": [],
        "intervals": [{"coverage": 0.8, "lower": [0.0, 1.0], "qhat": 0.5, "upper": [1.0, 2.0]}],
        "method": "split_absolute_residual",
        "unit": "physical_sample",
        "y_pred": [0.5, 1.5],
    }
    assert payload["metadata"]["source"] == "contract"
    assert payload["metadata"]["conformal_guarantee_status"]["status"] == "active"
    assert payload["metadata"]["conformal_guarantee_status"]["coverage"] == [0.8]

    restored = nirs4all.CalibratedRunResult.from_json(result.to_json())
    assert restored.to_dict() == payload


def test_robustness_report_serialization_contract() -> None:
    """Public robustness report serialization remains stable for artifacts."""

    result = nirs4all.PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
        sample_indices=np.asarray(["s1", "s2", "s3", "s4"], dtype=object),
    )
    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        slice_by=["instrument"],
        seed=123,
    )

    payload = report.to_dict()
    assert payload["fingerprint"] == "75c1b625fd4f597e6af1bee8b3a150019e021c1c0328bd172f9d3a86350d0318"
    assert payload["mode"] == "clean_frozen"
    assert payload["slice_by"] == ["instrument"]
    assert payload["version"] == 1
    assert payload["metadata"] == {
        "audit_only": True,
        "conformal_guarantee_status": None,
        "effective_seed": 123,
        "sample_ids": ["s1", "s2", "s3", "s4"],
        "seed": 123,
        "supported_scenario_kinds": ["observed", "prediction_bias", "prediction_noise", "spectral_noise", "spectral_offset", "spectral_scale", "spectral_slope", "spectral_shift"],
    }
    assert len(payload["scenarios"]) == 1
    scenario = payload["scenarios"][0]
    assert scenario["scenario"] == {"kind": "observed", "severity": 0.0}
    assert scenario["metrics"] == {
        "bias": -0.25,
        "fingerprint": "084726be870ac69a4a18ddf5ed7a46da011188041f71b175914aa9f4b3f293f2",
        "mae": 0.75,
        "max_abs_error": 1.0,
        "n_samples": 4,
        "rmse": 0.8660254037844386,
        "version": 1,
    }
    assert [slice_payload["slice_key"] for slice_payload in scenario["slices"]] == [{"instrument": "a"}, {"instrument": "b"}]
    assert [slice_payload["metrics"]["n_samples"] for slice_payload in scenario["slices"]] == [2, 2]
    assert [slice_payload["metrics"]["bias"] for slice_payload in scenario["slices"]] == [-0.5, 0.0]

    restored = nirs4all.RobustnessReport.from_json(report.to_json(indent=None))
    assert restored.to_dict() == payload


# ---------------------------------------------------------------------------
# ``__all__`` contracts.
# ---------------------------------------------------------------------------


def test_package_all_frozen() -> None:
    """``nirs4all.__all__`` matches the frozen export set exactly."""
    assert sorted(nirs4all.__all__) == EXPECTED_PACKAGE_ALL


def test_api_all_frozen() -> None:
    """``nirs4all.api.__all__`` matches the frozen export set exactly."""
    assert sorted(nirs4all_api.__all__) == EXPECTED_API_ALL


# ---------------------------------------------------------------------------
# Result-class public surface contracts (subset / superset semantics).
# ---------------------------------------------------------------------------


def test_runresult_public_surface() -> None:
    """RunResult still exposes every frozen public member."""
    live = _public_members(nirs4all.RunResult)
    missing = EXPECTED_RUNRESULT_MEMBERS - live
    assert not missing, f"RunResult dropped public members: {sorted(missing)}"


def test_predictresult_public_surface() -> None:
    """PredictResult still exposes every frozen public member."""
    live = _public_members(nirs4all.PredictResult)
    missing = EXPECTED_PREDICTRESULT_MEMBERS - live
    assert not missing, f"PredictResult dropped public members: {sorted(missing)}"


def test_explainresult_public_surface() -> None:
    """ExplainResult still exposes every frozen public member."""
    live = _public_members(nirs4all.ExplainResult)
    missing = EXPECTED_EXPLAINRESULT_MEMBERS - live
    assert not missing, f"ExplainResult dropped public members: {sorted(missing)}"


def test_native_tuning_helper_public_surfaces() -> None:
    """Typed native tuning helpers still expose their frozen public members."""
    expectations = {
        nirs4all.ConformalCalibrationData: EXPECTED_CONFORMAL_CALIBRATION_DATA_MEMBERS,
        nirs4all.NativeTuning: EXPECTED_NATIVE_TUNING_MEMBERS,
        nirs4all.TuningScoreData: EXPECTED_TUNING_SCORE_DATA_MEMBERS,
        nirs4all.TuningConformalScoreCalibration: EXPECTED_TUNING_CONFORMAL_SCORE_CALIBRATION_MEMBERS,
        nirs4all.TuningPassthrough: EXPECTED_TUNING_PASSTHROUGH_MEMBERS,
        nirs4all.TuningWinner: EXPECTED_TUNING_WINNER_MEMBERS,
        nirs4all.TuningCalibration: EXPECTED_TUNING_CALIBRATION_MEMBERS,
    }
    for cls, expected in expectations.items():
        live = _public_members(cls)
        missing = expected - live
        assert not missing, f"{cls.__name__} dropped public members: {sorted(missing)}"


def test_robustness_report_public_surfaces() -> None:
    """Robustness report containers still expose their frozen public members."""
    expectations = {
        nirs4all.RobustnessMetricSet: EXPECTED_ROBUSTNESS_METRIC_SET_MEMBERS,
        nirs4all.RobustnessScenarioSpec: EXPECTED_ROBUSTNESS_SCENARIO_SPEC_MEMBERS,
        nirs4all.RobustnessScenarioResult: EXPECTED_ROBUSTNESS_SCENARIO_RESULT_MEMBERS,
        nirs4all.RobustnessSliceResult: EXPECTED_ROBUSTNESS_SLICE_RESULT_MEMBERS,
        nirs4all.RobustnessReport: EXPECTED_ROBUSTNESS_REPORT_MEMBERS,
    }
    for cls, expected in expectations.items():
        live = _public_members(cls)
        missing = expected - live
        assert not missing, f"{cls.__name__} dropped public members: {sorted(missing)}"


def test_conformal_result_public_surfaces() -> None:
    """Conformal result containers still expose their frozen public members."""
    expectations = {
        nirs4all.CalibratedRunResult: EXPECTED_CALIBRATED_RUN_RESULT_MEMBERS,
        nirs4all.ConformalMetricSet: EXPECTED_CONFORMAL_METRIC_SET_MEMBERS,
    }
    for cls, expected in expectations.items():
        live = _public_members(cls)
        missing = expected - live
        assert not missing, f"{cls.__name__} dropped public members: {sorted(missing)}"


def test_tuned_single_estimator_conformal_result_public_surface() -> None:
    """Composite tuned+calibrated result still exposes its proxy members."""
    live = _public_members(nirs4all.TunedSingleEstimatorConformalResult)
    missing = EXPECTED_TUNED_SINGLE_ESTIMATOR_CONFORMAL_RESULT_MEMBERS - live
    assert not missing, f"TunedSingleEstimatorConformalResult dropped public members: {sorted(missing)}"
