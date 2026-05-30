"""Parity cases for sample/feature augmentation and concat_transform.

Augmentation is the area with the most subtle lineage requirements: the
dag-ml bridge must record augmentation origin (which sample an augmented
row derives from) so OOF leakage refusal still applies. Each case here
forces the bridge to honor that contract.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.operators.augmentation import GaussianAdditiveNoise, MultiplicativeNoise, Rotate_Translate
from nirs4all.operators.transforms import Detrend, FirstDerivative, SavitzkyGolay
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_AUG = frozenset({"augmentation", "slow"})


def _factory_sample_aug_gaussian_noise() -> list[Any]:
    return [
        SNV(),
        {
            "sample_augmentation": {
                "transformers": [GaussianAdditiveNoise(sigma=0.005)],
                "count": 1,
                "selection": "all",
                "random_state": 42,
            }
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="sample_augmentation_gaussian",
        description="Sample-level Gaussian additive noise augmentation after SNV → ShuffleSplit → PLSR. "
        "Each training sample yields an augmented copy with tracked origin_id.",
        keywords=("sample_augmentation", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "augmentation",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sample_aug_gaussian_noise,
        expected_min_predictions=3,
        tags=_AUG,
    )
)


def _factory_sample_aug_multiplicative_then_rotate() -> list[Any]:
    return [
        SNV(),
        {
            "sample_augmentation": {
                "transformers": [
                    MultiplicativeNoise(sigma_gain=0.01),
                    Rotate_Translate(p_range=2, y_factor=3),
                ],
                "count": 2,
                "selection": "random",
                "random_state": 42,
            }
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="sample_augmentation_chained",
        description="Two chained sample augmentations (multiplicative noise → rotate/translate) → PLSR. "
        "Tests augmentation-origin chaining and that origins do not lose lineage.",
        keywords=("sample_augmentation", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "augmentation",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sample_aug_multiplicative_then_rotate,
        expected_min_predictions=3,
        tags=_AUG,
    )
)


def _factory_feature_augmentation_replace() -> list[Any]:
    return [
        {
            "feature_augmentation": [SNV, FirstDerivative, Detrend],
            "action": "replace",
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="feature_augmentation_replace_three_views",
        description="Three feature views (SNV / FirstDerivative / Detrend) with action='replace' → PLSR. "
        "Tests feature-augmentation replacement semantics.",
        keywords=("feature_augmentation", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "augmentation",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_feature_augmentation_replace,
        expected_min_predictions=3,
        tags=_AUG,
    )
)


def _factory_concat_transform_pca_svd_plsr() -> list[Any]:
    return [
        SNV(),
        {
            "concat_transform": [
                PCA(n_components=15),
                TruncatedSVD(n_components=10),
            ]
        },
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=15)},
    ]


register(
    PipelineCase(
        name="concat_transform_pca_svd_plsr",
        description="SNV → concat_transform([PCA(15), TruncatedSVD(10)]) → PLSR. "
        "Tests concat_transform with two dimensionality-reducers stacked column-wise.",
        keywords=("concat_transform", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_concat_transform_pca_svd_plsr,
        expected_min_predictions=3,
        tags=_AUG,
    )
)


def _factory_sample_aug_with_savgol() -> list[Any]:
    return [
        SavitzkyGolay(window_length=11, polyorder=2, deriv=0),
        {
            "sample_augmentation": {
                "transformers": [GaussianAdditiveNoise(sigma=0.002)],
                "count": 1,
                "selection": "all",
                "random_state": 42,
            }
        },
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="sample_augmentation_after_savgol",
        description="Savitzky-Golay smoothing → sample-level Gaussian noise augmentation → KFold → Ridge. "
        "Augmentation after a stateful smoother — exercises operator-state vs augmentation interaction.",
        keywords=("sample_augmentation", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "augmentation",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sample_aug_with_savgol,
        expected_min_predictions=3,
        tags=_AUG,
    )
)
