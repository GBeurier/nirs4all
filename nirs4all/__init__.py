"""
NIRS4All - A comprehensive package for Near-Infrared Spectroscopy data processing and analysis.

This package provides tools for spectroscopy data handling, preprocessing, model building,
and pipeline management with support for multiple ML backends.

Public API (recommended)::

    nirs4all.run(pipeline, dataset, ...)         - Train a pipeline
    nirs4all.calibrate(...)                      - Calibrate replayed predictions
    nirs4all.conformal_metrics(...)              - Evaluate interval diagnostics
    nirs4all.attach_calibrated_result_to_bundle  - Attach a calibrator to a model bundle
    nirs4all.export_calibrated_result(result)    - Export a calibrated result bundle
    nirs4all.load_calibrated_result(path)        - Load a saved calibrated result
    nirs4all.load_workspace_calibrated_result    - Load a workspace conformal result
    nirs4all.load_workspace_calibrated_predict_result - Load workspace conformal as PredictResult
    nirs4all.load_workspace_predict_result       - Load workspace prediction as PredictResult
    nirs4all.load_workspace_predict_results      - Load workspace predictions as PredictResult list
    nirs4all.save_workspace_predict_result       - Save PredictResult plus replay evidence
    nirs4all.robustness_from_workspace_prediction - Robustness report from workspace prediction
    nirs4all.load_workspace_tuning_result        - Load a workspace tuning result
    nirs4all.tune_single_estimator(...)          - Native single-estimator tuning lane
    nirs4all.predict_calibrated(...)             - Apply a loaded calibrator to predictions
    nirs4all.predict(model, data, ...)           - Make predictions
    nirs4all.explain(model, data, ...)           - Generate SHAP explanations
    nirs4all.retrain(source, data, ...)          - Retrain a pipeline
    nirs4all.session(...)                        - Create execution session
    nirs4all.load_session(path)                  - Load saved session
    nirs4all.generate(n_samples, ...)            - Generate synthetic NIRS data

Classes (for advanced usage):
    nirs4all.PipelineRunner    - Direct runner access
    nirs4all.PipelineConfigs   - Pipeline configuration
    nirs4all.DatasetConfigs    - Dataset configuration (from nirs4all.data)

Example:
    >>> import nirs4all
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.cross_decomposition import PLSRegression
    >>>
    >>> result = nirs4all.run(
    ...     pipeline=[MinMaxScaler(), PLSRegression(10)],
    ...     dataset="sample_data/regression",
    ...     verbose=1
    ... )
    >>> print(f"Best RMSE: {result.best_rmse:.4f}")
    >>> result.export("exports/best_model.n4a")

Synthetic Data Generation:
    >>> # Generate synthetic data for testing
    >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)
    >>>
    >>> # Use convenience functions
    >>> dataset = nirs4all.generate.regression(n_samples=500)
    >>> dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)

See examples/ for more usage examples.
"""

__version__ = "0.11.0"

# Module-level API (primary interface) - Phase 2
from .api import (
    CONFORMAL_CALIBRATION_METHODS,
    CONFORMAL_CALIBRATION_UNITS,
    CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES,
    CONFORMAL_MULTI_TARGET_POLICIES,
    CONFORMAL_TUNING_SCORE_METRICS,
    FINETUNE_APPROACHES,
    FINETUNE_DAGML_APPROACHES,
    FINETUNE_DAGML_DETERMINISTIC_ENGINES,
    FINETUNE_DAGML_EVAL_MODES,
    FINETUNE_DAGML_META_KEYS,
    FINETUNE_DAGML_SELECTION_METRICS,
    FINETUNE_ENGINE_ALIASES,
    FINETUNE_ENGINES,
    FINETUNE_EVAL_MODE_ALIASES,
    FINETUNE_EVAL_MODES,
    FINETUNE_N4M_PRUNERS,
    FINETUNE_N4M_SAMPLERS,
    FINETUNE_OPTUNA_PRUNERS,
    FINETUNE_OPTUNA_SAMPLERS,
    FINETUNE_SAMPLER_KEY_ALIASES,
    ROBUSTNESS_EXECUTABLE_MODES,
    ROBUSTNESS_MODES,
    ROBUSTNESS_SCENARIO_DISTRIBUTIONS,
    ROBUSTNESS_SCENARIO_KINDS,
    ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS,
    TUNING_CONTRACT_KEYS,
    TUNING_DIRECTIONS,
    TUNING_ENGINES,
    TUNING_OPTIMIZER_PERSISTENCE_KEYS,
    TUNING_RUNTIME_KEYS,
    TUNING_SPACE_SCHEMA_ID,
    TUNING_SUMMARY_SCHEMA_ID,
    CalibratedRunResult,
    ConformalCalibrationData,
    ConformalMethod,
    ConformalMetricSet,
    ConformalMultiTarget,
    ConformalUnit,
    ExplainResult,
    FinetuneApproach,
    FinetuneEngine,
    FinetuneEvalMode,
    FinetunePruner,
    FinetuneSampler,
    NativeTuning,
    Nirs4AllCalibrationNotImplementedError,
    OrderedSearchSpaceSpec,
    ParameterPatch,
    PredictResult,
    RobustnessMetricSet,
    RobustnessMode,
    RobustnessReport,
    RobustnessScenarioDistribution,
    RobustnessScenarioKind,
    RobustnessScenarioResult,
    RobustnessScenarioSpec,
    RobustnessSliceResult,
    RunResult,
    SearchSpaceParameter,
    Session,
    TrialResult,
    TunedSingleEstimatorConformalResult,
    TuningCalibration,
    TuningConformalScoreCalibration,
    TuningDirection,
    TuningEngine,
    TuningPassthrough,
    TuningResult,
    TuningScoreData,
    TuningWinner,
    attach_calibrated_result_to_bundle,
    calibrate,
    conformal_metrics,
    explain,
    export_calibrated_result,
    generate,
    get_keyword_registry,
    get_keyword_registry_schema,
    get_robustness_summary_schema,
    get_tuning_space_schema,
    get_tuning_summary_schema,
    inspect_tuning_space,
    keyword_registry_json,
    keyword_registry_schema_json,
    load_calibrated_result,
    load_session,
    load_workspace_calibrated_predict_result,
    load_workspace_calibrated_result,
    load_workspace_predict_result,
    load_workspace_predict_results,
    load_workspace_robustness_report,
    load_workspace_tuning_result,
    predict,
    predict_calibrated,
    retrain,
    robustness,
    robustness_from_workspace_prediction,
    robustness_summary_schema_json,
    run,
    save_workspace_calibrated_result,
    save_workspace_predict_result,
    save_workspace_robustness_report,
    save_workspace_tuning_result,
    session,
    tune_single_estimator,
    tuning_space_schema_json,
    tuning_summary_schema_json,
)
from .controllers import CONTROLLER_REGISTRY, register_controller

# Core pipeline components - for advanced usage
from .pipeline import (
    PipelineConfigs,
    PipelineRunner,
    Run,
    RunConfig,
    RunStatus,
    generate_run_id,
)

# Utility functions for backend detection
from .utils import (
    framework,
    # is_torch_available,
    is_gpu_available,
    is_tensorflow_available,
)

# Make commonly used classes available at package level
__all__ = [
    # Module-level API (primary interface)
    "run",
    "calibrate",
    "conformal_metrics",
    "attach_calibrated_result_to_bundle",
    "export_calibrated_result",
    "load_calibrated_result",
    "load_workspace_calibrated_predict_result",
    "load_workspace_calibrated_result",
    "load_workspace_predict_result",
    "load_workspace_predict_results",
    "load_workspace_robustness_report",
    "load_workspace_tuning_result",
    "predict_calibrated",
    "robustness",
    "robustness_from_workspace_prediction",
    "save_workspace_calibrated_result",
    "save_workspace_predict_result",
    "save_workspace_robustness_report",
    "save_workspace_tuning_result",
    "tune_single_estimator",
    "predict",
    "CONFORMAL_CALIBRATION_METHODS",
    "CONFORMAL_CALIBRATION_UNITS",
    "CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES",
    "CONFORMAL_MULTI_TARGET_POLICIES",
    "CONFORMAL_TUNING_SCORE_METRICS",
    "FINETUNE_APPROACHES",
    "FINETUNE_DAGML_APPROACHES",
    "FINETUNE_DAGML_DETERMINISTIC_ENGINES",
    "FINETUNE_DAGML_EVAL_MODES",
    "FINETUNE_DAGML_META_KEYS",
    "FINETUNE_DAGML_SELECTION_METRICS",
    "FINETUNE_ENGINES",
    "FINETUNE_ENGINE_ALIASES",
    "FINETUNE_EVAL_MODE_ALIASES",
    "FINETUNE_EVAL_MODES",
    "FINETUNE_N4M_PRUNERS",
    "FINETUNE_N4M_SAMPLERS",
    "FINETUNE_OPTUNA_PRUNERS",
    "FINETUNE_OPTUNA_SAMPLERS",
    "FINETUNE_SAMPLER_KEY_ALIASES",
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
    "ROBUSTNESS_EXECUTABLE_MODES",
    "ROBUSTNESS_MODES",
    "ROBUSTNESS_SCENARIO_DISTRIBUTIONS",
    "ROBUSTNESS_SCENARIO_KINDS",
    "ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS",
    "explain",
    "retrain",
    "session",
    "load_session",
    "Session",
    "RunResult",
    "PredictResult",
    "ExplainResult",
    "CalibratedRunResult",
    "ConformalCalibrationData",
    "ConformalMethod",
    "ConformalMetricSet",
    "ConformalMultiTarget",
    "ConformalUnit",
    "FinetuneApproach",
    "FinetuneEngine",
    "FinetuneEvalMode",
    "FinetunePruner",
    "FinetuneSampler",
    "TunedSingleEstimatorConformalResult",
    "inspect_tuning_space",
    "NativeTuning",
    "OrderedSearchSpaceSpec",
    "ParameterPatch",
    "SearchSpaceParameter",
    "TUNING_CONTRACT_KEYS",
    "TUNING_DIRECTIONS",
    "TUNING_ENGINES",
    "TUNING_OPTIMIZER_PERSISTENCE_KEYS",
    "TUNING_RUNTIME_KEYS",
    "TUNING_SPACE_SCHEMA_ID",
    "TUNING_SUMMARY_SCHEMA_ID",
    "TuningScoreData",
    "TuningConformalScoreCalibration",
    "TuningDirection",
    "TuningEngine",
    "TuningResult",
    "TuningWinner",
    "TuningCalibration",
    "TuningPassthrough",
    "TrialResult",
    "Nirs4AllCalibrationNotImplementedError",
    "RobustnessMetricSet",
    "RobustnessMode",
    "RobustnessReport",
    "RobustnessScenarioDistribution",
    "RobustnessScenarioKind",
    "RobustnessScenarioResult",
    "RobustnessScenarioSpec",
    "RobustnessSliceResult",
    # Synthetic data generation
    "generate",
    # Pipeline components (advanced usage)
    "PipelineRunner",
    "PipelineConfigs",
    "Run",
    "RunStatus",
    "RunConfig",
    "generate_run_id",
    # Controller system
    "register_controller",
    "CONTROLLER_REGISTRY",
    # Utilities
    "is_tensorflow_available",
    # "is_torch_available",
    "is_gpu_available",
    "framework",
]
