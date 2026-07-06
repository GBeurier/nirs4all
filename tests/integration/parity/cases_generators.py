"""Parity cases for the generator DSL keywords.

Generators expand a single pipeline declaration into a set of variants the
runner trains and ranks. The dag-ml bridge must enumerate the same variant
set deterministically — otherwise selection and bundle replay diverge.

Coverage: `_or_`, `_range_`, `_log_range_`, `_grid_`, `_cartesian_`,
`_zip_`, `_chain_`, `_sample_`, plus `finetune_params` / `train_params`
which sit alongside the model step as per-variant hyperparameter blocks.

The R01/R02 references show the canonical placement:
- numeric generators (`_range_`, `_log_range_`, `_sample_`) live at the TOP
  level of a step dict together with `param` (the model attribute name) and
  `model` (the class, not an instance);
- `_or_`, `_chain_`, `_cartesian_` accept lists of classes or step-shapes;
- `finetune_params` is a sibling of `model` and uses tuple `(type, low, high)`
  parameter bounds.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

from nirs4all.operators.transforms import Detrend, FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_GEN = frozenset({"generator", "slow"})


def _factory_or_preprocessing() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_preprocessing",
        description="`_or_` over preprocessing classes (SNV / MSC / Detrend) — 3 variants × 3 folds.",
        keywords=("_or_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_or_preprocessing,
        expected_min_predictions=9,
        tags=_GEN,
    )
)


def _factory_or_with_pick() -> list[Any]:
    return [
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_or_with_pick",
        description="`_or_` with `pick: 2` combinator — C(4,2) = 6 unordered preprocessing combinations.",
        keywords=("_or_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_or_with_pick,
        expected_min_predictions=18,
        tags=_GEN,
    )
)


def _factory_range_n_components() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_range_": [5, 25, 5],
            "param": "n_components",
            "model": PLSRegression,
        },
    ]


register(
    PipelineCase(
        name="generator_range_n_components",
        description="`_range_` over PLSRegression.n_components — linear sweep [5, 25, 5] → "
        "5 variants × 3 folds. Uses the canonical `_range_`/`param`/`model` triple.",
        keywords=("_range_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_range_n_components,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_log_range_alpha() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_log_range_": [1e-4, 1e0, 5],
            "param": "alpha",
            "model": Ridge,
        },
    ]


register(
    PipelineCase(
        name="generator_log_range_alpha",
        description="`_log_range_` over Ridge.alpha in [1e-4, 1e0] with 5 points — 5 variants × 3 folds.",
        keywords=("_log_range_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_log_range_alpha,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_grid_n_components_alpha() -> list[Any]:
    # `_grid_` builds the Cartesian product of all listed params. The
    # generator node lives at top level; `model` provides the target class.
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_grid_": {
                "n_components": [5, 10, 15],
                "scale": [True, False],
            },
            "model": PLSRegression,
        },
    ]


register(
    PipelineCase(
        name="generator_grid_n_components_scale",
        description="`_grid_` Cartesian over PLSRegression(n_components × scale) — 3 × 2 = 6 variants × 3 folds.",
        keywords=("_grid_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_grid_n_components_alpha,
        expected_min_predictions=18,
        tags=_GEN,
    )
)


def _factory_cartesian_stages() -> list[Any]:
    return [
        {
            "_cartesian_": [
                {"_or_": [SNV, MSC]},
                {"_or_": [Detrend, FirstDerivative]},
            ]
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_cartesian_stages",
        description="`_cartesian_` over (preprocessing × second-preprocessing) — 2 × 2 = 4 stage combos × 3 folds.",
        keywords=("_cartesian_", "_or_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_cartesian_stages,
        expected_min_predictions=12,
        tags=_GEN,
    )
)


def _factory_zip_paired() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_zip_": {
                "n_components": [5, 10, 15],
                "scale": [True, False, True],
            },
            "model": PLSRegression,
        },
    ]


register(
    PipelineCase(
        name="generator_zip_paired",
        description="`_zip_` paired iteration: (n_components, scale) zipped triples → 3 variants × 3 folds.",
        keywords=("_zip_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_zip_paired,
        expected_min_predictions=9,
        tags=_GEN,
    )
)


def _factory_chain_sequential() -> list[Any]:
    return [
        {"_chain_": [SNV, MSC, Detrend]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="generator_chain_sequential",
        description="`_chain_` over preprocessing — sequential ordered iteration of 3 preprocessors.",
        keywords=("_chain_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_chain_sequential,
        expected_min_predictions=9,
        tags=_GEN,
    )
)


def _factory_sample_random_alpha() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "_sample_": {
                "distribution": "log_uniform",
                "from": 1e-4,
                "to": 1e0,
                "num": 5,
            },
            "param": "alpha",
            "_seed_": 123,
            "model": Ridge,
        },
    ]


register(
    PipelineCase(
        name="generator_sample_log_uniform_alpha",
        description="`_sample_` seeded sampling of Ridge.alpha from log-uniform distribution — 5 variants × 3 folds.",
        keywords=("_sample_", "_seed_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sample_random_alpha,
        expected_min_predictions=15,
        tags=_GEN,
    )
)


def _factory_finetune_params() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "name": "RF_Finetuned",
            "model": RandomForestRegressor(n_jobs=1, random_state=42),
            "finetune_params": {
                "n_trials": 4,
                "sampler": "tpe",
                "metric": "rmse",
                "approach": "single",
                "eval_mode": "best",
                "seed": 42,
                "model_params": {
                    "n_estimators": ("int", 10, 40),
                    "max_depth": ("int", 3, 8),
                },
            },
            "train_params": {"verbose": 0},
        },
    ]


register(
    PipelineCase(
        name="generator_finetune_params_optuna",
        description="`finetune_params` Optuna search on RandomForestRegressor "
        "(n_estimators × max_depth, 4 trials) — canonical tuple bounds syntax.",
        keywords=("finetune_params", "model", "name", "train_params"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_finetune_params,
        expected_min_predictions=3,
        tags=_GEN | frozenset({"optuna"}),
    )
)
