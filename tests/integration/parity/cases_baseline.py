"""Baseline parity cases.

The simplest pipelines that the legacy backend supports — single (or two)
preprocessor(s), single model, one cross-validator. They pin the happy
path. If the dag-ml bridge cannot reproduce these, nothing else will work.

Every case here declares only canonical CLAUDE.md DSL keywords in
`keywords`. Operator families (`sklearn_model`, `nirs_splitter`, etc.) live
in `capabilities`. The exact roadmap vertical slice
(`SNV + y_processing + ShuffleSplit + PLSRegression`) is included as
`baseline_vertical_slice` so the parity suite's gate-zero case is
unambiguous.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.operators.splitters import KennardStoneSplitter, SPXYSplitter
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    SavitzkyGolay,
)
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_FAST = frozenset({"baseline", "fast"})
_MED = frozenset({"baseline"})


def _factory_vertical_slice() -> list[Any]:
    """The exact roadmap vertical-slice pipeline (Phase 1 of the integration plan)."""
    return [
        SNV(),
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="baseline_vertical_slice",
        description="Roadmap vertical-slice: SNV → y_processing(MinMaxScaler) → ShuffleSplit(3) → PLSR(10). "
        "Gate-zero for the dag-ml bridge — if this does not reproduce, nothing else does.",
        keywords=("y_processing", "model"),
        capabilities=(
            "preprocessing_transform",
            "y_processing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_vertical_slice,
        expected_min_predictions=3,
        tags=_FAST | frozenset({"vertical_slice"}),
        metric_tolerances={"rmse": 1e-6, "r2": 1e-6},
    )
)


def _factory_snv_plsr_shuffle() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="baseline_snv_plsr_shuffle",
        description="SNV → PLSR(10) under 3-fold ShuffleSplit. NIRS-typical preprocessing + model.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_snv_plsr_shuffle,
        expected_min_predictions=3,
        tags=_FAST,
    )
)


def _factory_savgol_rf_kfold() -> list[Any]:
    return [
        SavitzkyGolay(window_length=11, polyorder=2, deriv=0),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {
            "model": RandomForestRegressor(
                n_estimators=20,
                max_depth=8,
                random_state=42,
                n_jobs=1,
            )
        },
    ]


register(
    PipelineCase(
        name="baseline_savgol_rf_kfold",
        description="Savitzky-Golay smoothing → RandomForestRegressor under 3-fold KFold. "
        "Non-linear model family parity check.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_savgol_rf_kfold,
        expected_min_predictions=3,
        tags=_MED,
    )
)


def _factory_msc_yprocess_ridge() -> list[Any]:
    return [
        MSC(),
        {"y_processing": StandardScaler()},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="baseline_msc_y_processing_ridge",
        description="MSC → y_processing(StandardScaler) → Ridge under 3-fold ShuffleSplit. "
        "Exercises the y inverse-transform path on predictions.",
        keywords=("y_processing", "model"),
        capabilities=(
            "preprocessing_transform",
            "y_processing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_msc_yprocess_ridge,
        expected_min_predictions=3,
        tags=_FAST,
    )
)


def _factory_kennard_stone_plsr() -> list[Any]:
    return [
        SNV(),
        KennardStoneSplitter(test_size=0.3),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="baseline_kennard_stone_plsr",
        description="SNV → KennardStone split → PLSR(10). NIRS-specific splitter parity.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "nirs_splitter",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_kennard_stone_plsr,
        expected_min_predictions=1,
        tags=_FAST,
    )
)


def _factory_spxy_plsr() -> list[Any]:
    return [
        SNV(),
        SPXYSplitter(test_size=0.3),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="baseline_spxy_plsr",
        description="SNV → SPXY split → PLSR(10). The y-aware NIRS splitter.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "nirs_splitter",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_spxy_plsr,
        expected_min_predictions=1,
        tags=_FAST,
    )
)


def _factory_detrend_firstderiv_gbr() -> list[Any]:
    return [
        Detrend(),
        FirstDerivative(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "model": GradientBoostingRegressor(
                n_estimators=30,
                max_depth=4,
                random_state=42,
            )
        },
    ]


register(
    PipelineCase(
        name="baseline_detrend_firstderiv_gbr",
        description="Detrend → FirstDerivative → GradientBoostingRegressor under 3-fold ShuffleSplit. "
        "Two-step preprocessing chain without a sklearn-native transformer.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_detrend_firstderiv_gbr,
        expected_min_predictions=3,
        tags=_MED,
    )
)


def _factory_classification_rf_stratified() -> list[Any]:
    return [
        SNV(),
        StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        {
            "model": RandomForestClassifier(
                n_estimators=30,
                max_depth=6,
                random_state=42,
                n_jobs=1,
            )
        },
    ]


register(
    PipelineCase(
        name="baseline_classification_rf_stratified",
        description="SNV → StratifiedKFold(3) → RandomForestClassifier(30). "
        "Minimum classification parity case; covers stratified folds + classification model dispatch.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "classification_model",
        ),
        dataset_key="classification",
        pipeline_factory=_factory_classification_rf_stratified,
        task="classification",
        expected_min_predictions=3,
        tags=_FAST | frozenset({"classification"}),
    )
)
