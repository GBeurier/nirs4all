"""Parity cases for the public API surface: predict / explain / retrain / session / bundle round-trip.

CLAUDE.md freezes the nirs4all 0.9.x public API. The dag-ml bridge has to
reproduce every entry point: `nirs4all.run`, `nirs4all.predict`,
`nirs4all.explain`, `nirs4all.retrain`, `nirs4all.session`,
`nirs4all.generate`. The `RunResult.export()` / `BundleLoader.load()`
round-trip is the bundle-IO contract.

Cases here look like ordinary training pipelines, but they are tagged
with the API path they're meant to exercise — the smoke tests in
`test_parity_smoke.py` use those tags to drive `predict`, `explain`,
`retrain`, `session` and bundle round-trip beyond just `run`.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register


def _factory_round_trip_baseline() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="round_trip_baseline_export_predict",
        description="Train a baseline pipeline, export the .n4a bundle, reload, predict on holdout. "
        "The bundle round-trip contract that `BundleGenerator` / `BundleLoader` must preserve.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "bundle_io",
            "predict_path",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_round_trip_baseline,
        expected_min_predictions=3,
        tags=frozenset({"round_trip", "bundle_io", "predict_path", "fast"}),
    )
)


def _factory_round_trip_with_y_processing() -> list[Any]:
    return [
        SNV(),
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="round_trip_with_y_processing_inverse",
        description="Pipeline with y_processing → Ridge — round-trip must preserve the inverse-transform "
        "on predict so predictions come back in the original target scale.",
        keywords=("y_processing", "model"),
        capabilities=(
            "preprocessing_transform",
            "y_processing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "bundle_io",
            "predict_path",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_round_trip_with_y_processing,
        expected_min_predictions=3,
        tags=frozenset({"round_trip", "bundle_io", "predict_path", "fast"}),
    )
)


def _factory_refit_params() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "name": "PLS_with_refit",
            "model": PLSRegression(n_components=8),
            "refit_params": {
                "use_all_partitions": True,
            },
        },
    ]


register(
    PipelineCase(
        name="refit_params_use_all_partitions",
        description="`refit_params: {use_all_partitions: True}` forces refit on train + val + test. "
        "Exercises the explicit refit-policy override the bridge must surface.",
        keywords=("refit_params", "model", "name"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "retrain_path",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_refit_params,
        expected_min_predictions=3,
        tags=frozenset({"refit", "fast"}),
        skip_reason="refit_params semantics depend on nirs4all 0.9.x retraining logic — "
        "confirm exact key names against api/retrain.py before flipping on.",
        skip_kind="unknown_semantics",
    )
)


def _factory_explainable_pipeline() -> list[Any]:
    # nirs4all.explain wraps the pipeline as a NIRSPipeline (SHAP-compatible)
    # so the bundle must include preprocessing + model in a form SHAP can call.
    return [
        SNV(),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="explain_path_baseline",
        description="Baseline pipeline tagged for `nirs4all.explain()` — the smoke runner exports a bundle, "
        "loads as NIRSPipeline, and runs SHAP on the regression-default sample.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "explain_path",
            "bundle_io",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_explainable_pipeline,
        expected_min_predictions=3,
        tags=frozenset({"explain", "slow"}),
    )
)


def _factory_retrain_baseline() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="retrain_path_baseline",
        description="Baseline trained pipeline that the smoke runner reloads via `nirs4all.retrain()` "
        "on a second dataset.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "retrain_path",
            "bundle_io",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_retrain_baseline,
        expected_min_predictions=3,
        tags=frozenset({"retrain", "slow"}),
    )
)


def _factory_session_baseline() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="session_path_baseline",
        description="Baseline pipeline that the smoke runner drives through `nirs4all.session()` "
        "to exercise the stateful workspace API surface.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "session_api",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_session_baseline,
        expected_min_predictions=3,
        tags=frozenset({"session", "slow"}),
    )
)
