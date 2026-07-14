"""
NIRS4All API Module - High-level functional interface.

This module provides the primary public API for nirs4all, offering
simple function-based entry points that wrap the underlying PipelineRunner.

Public API:
    run(pipeline, dataset, **kwargs) -> RunResult
        Execute a training pipeline on a dataset.

    predict(model, data, **kwargs) -> PredictResult
        Make predictions with a trained model.

    explain(model, data, **kwargs) -> ExplainResult
        Generate SHAP explanations for model predictions.

    retrain(source, data, **kwargs) -> RunResult
        Retrain a pipeline on new data.

    session(**kwargs) -> Session
        Create an execution session for resource reuse.

    generate(n_samples, **kwargs) -> SpectroDataset | (X, y)
        Generate synthetic NIRS data for testing and research.

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

For more examples, see the examples/Q40_new_api.py file.
"""

# Result classes (Phase 1)
from ..pipeline.keyword_registry import (
    get_keyword_registry,
    get_keyword_registry_schema,
    keyword_registry_json,
    keyword_registry_schema_json,
)
from .calibrate import (
    CONFORMAL_CALIBRATION_METHODS,
    CONFORMAL_CALIBRATION_UNITS,
    CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES,
    CONFORMAL_MULTI_TARGET_POLICIES,
    CalibratedRunResult,
    ConformalCalibrationData,
    ConformalMethod,
    ConformalMetricSet,
    ConformalMultiTarget,
    ConformalUnit,
    Nirs4AllCalibrationNotImplementedError,
    attach_calibrated_result_to_bundle,
    calibrate,
    conformal_metrics,
    export_calibrated_result,
    load_calibrated_result,
    load_workspace_calibrated_predict_result,
    load_workspace_calibrated_result,
    predict_calibrated,
    save_workspace_calibrated_result,
)
from .explain import explain

# Synthetic data generation
from .generate import generate_namespace as generate
from .predict import predict
from .result import (
    ExplainResult,
    LazyModelRefitResult,
    ModelRefitResult,
    PredictResult,
    RunResult,
    load_workspace_predict_result,
    load_workspace_predict_results,
    save_workspace_predict_result,
)
from .retrain import retrain
from .robustness import (
    ROBUSTNESS_EXECUTABLE_MODES,
    ROBUSTNESS_MODES,
    ROBUSTNESS_SCENARIO_DISTRIBUTIONS,
    ROBUSTNESS_SCENARIO_KINDS,
    ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS,
    RobustnessMetricSet,
    RobustnessMode,
    RobustnessReport,
    RobustnessScenarioDistribution,
    RobustnessScenarioKind,
    RobustnessScenarioResult,
    RobustnessScenarioSpec,
    RobustnessSliceResult,
    get_robustness_summary_schema,
    load_workspace_robustness_report,
    robustness,
    robustness_from_workspace_prediction,
    robustness_summary_schema_json,
    save_workspace_robustness_report,
)

# Module-level functions (Phase 2)
from .run import run

# Session (Phase 3 - full implementation)
from .session import Session, load_session, session
from .tuning import (
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
    TUNING_CONTRACT_KEYS,
    TUNING_DIRECTIONS,
    TUNING_ENGINES,
    TUNING_OPTIMIZER_PERSISTENCE_KEYS,
    TUNING_RUNTIME_KEYS,
    TUNING_SPACE_SCHEMA_ID,
    TUNING_SUMMARY_SCHEMA_ID,
    FinetuneApproach,
    FinetuneEngine,
    FinetuneEvalMode,
    FinetunePruner,
    FinetuneSampler,
    NativeTuning,
    OrderedSearchSpaceSpec,
    ParameterPatch,
    SearchSpaceParameter,
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
    get_tuning_space_schema,
    get_tuning_summary_schema,
    inspect_tuning_space,
    load_workspace_tuning_result,
    save_workspace_tuning_result,
    tune_single_estimator,
    tuning_space_schema_json,
    tuning_summary_schema_json,
)

__all__ = [
    # Module-level API functions
    "run",
    "calibrate",
    "attach_calibrated_result_to_bundle",
    "export_calibrated_result",
    "conformal_metrics",
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
    "get_tuning_summary_schema",
    "keyword_registry_json",
    "keyword_registry_schema_json",
    "robustness_summary_schema_json",
    "tuning_summary_schema_json",
    "ROBUSTNESS_EXECUTABLE_MODES",
    "ROBUSTNESS_MODES",
    "ROBUSTNESS_SCENARIO_DISTRIBUTIONS",
    "ROBUSTNESS_SCENARIO_KINDS",
    "ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS",
    "explain",
    "retrain",
    # Session
    "Session",
    "session",
    "load_session",
    # Synthetic data generation
    "generate",
    # Result classes
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
    "get_tuning_space_schema",
    "tuning_space_schema_json",
    "RobustnessMetricSet",
    "RobustnessMode",
    "RobustnessReport",
    "RobustnessScenarioDistribution",
    "RobustnessScenarioKind",
    "RobustnessScenarioResult",
    "RobustnessScenarioSpec",
    "RobustnessSliceResult",
    "ModelRefitResult",
    "LazyModelRefitResult",
    "Nirs4AllCalibrationNotImplementedError",
]
