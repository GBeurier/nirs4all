"""
Q22: Concat Transform Example
==============================
This example demonstrates the concat_transform feature which concatenates
multiple transformer outputs horizontally. This is useful for:
- Feature engineering with multiple extraction methods
- Creating tabular-like features from spectral data
- Combining PCA, SVD, and other feature extractors

The concat_transform can be used:
1. At top-level: REPLACES each processing with concatenated version
2. Inside feature_augmentation: ADDS a new processing with concatenated output
3. With generation syntax (_or_, size, count): Generates combinations of transformers

Key feature: Generation within concat_transform
- Pool: [PCA(30), SVD(20)] with size=(1,2) generates:
  - [PCA(30)] alone
  - [SVD(20)] alone
  - [PCA(30), SVD(20)] concatenated
  - [SVD(20), PCA(30)] concatenated (order matters for permutations)

Usage:
    python Q22_concat_transform.py --plots
"""

import argparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate, FirstDerivative
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q22 Concat Augmentation Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


# ============================================================
# Example 1: Top-level concat_transform (Replace Mode)
# ============================================================
# When used at top-level, concat_transform REPLACES the
# current processing with concatenated features from multiple transformers.

print("=" * 70)
print("EXAMPLE 1: Top-level concat_transform (Replace Mode)")
print("=" * 70)
print()
print("Replaces raw features with concatenated PCA + SVD features.")
print()

pipeline_1 = [
    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Feature scaling first
    StandardScaler(),

    # Concat augmentation: concatenate PCA and SVD outputs
    # Result: 30 + 20 = 50 features
    {
        "concat_transform": [
            PCA(n_components=30),           # 30 features
            TruncatedSVD(n_components=20),  # 20 features
        ]
    },

    # Show the resulting features
    "chart_2d",

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # PLS model
    PLSRegression(n_components=10),
]

# Load regression dataset
dataset_config = DatasetConfigs("sample_data/regression_3")
pipeline_config = PipelineConfigs(pipeline_1, name="Concat_Replace_Mode")

# Run pipeline
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions_1, _ = runner.run(pipeline_config, dataset_config)

print()
print("Example 1 completed. Features were reduced to 50 (30 PCA + 20 SVD)")
print()


# ============================================================
# Example 2: Chained Transformers
# ============================================================
# Chains are executed sequentially: [A, B] means B(A(X))
# This allows preprocessing before feature extraction.

print("=" * 70)
print("EXAMPLE 2: Chained Transformers in concat_transform")
print("=" * 70)
print()
print("Using chains: [StandardScaler, PCA] means scale then reduce.")
print()

pipeline_2 = [
    {"y_processing": MinMaxScaler()},

    # Concat with chains
    {
        "concat_transform": [
            PCA(n_components=25),                      # Direct: 25 features
            [StandardScaler(), PCA(n_components=15)],  # Chain: scale->PCA = 15 features
        ]
    },

    "chart_2d",

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_2 = PipelineConfigs(pipeline_2, name="Concat_Chained")
predictions_2, _ = runner.run(pipeline_config_2, dataset_config)

print()
print("Example 2 completed. Total features: 40 (25 + 15)")
print()


# ============================================================
# Example 3: concat_transform inside feature_augmentation
# ============================================================
# When nested inside feature_augmentation, concat_transform
# ADDS a new processing dimension instead of replacing.

print("=" * 70)
print("EXAMPLE 3: Nested in feature_augmentation (Add Mode)")
print("=" * 70)
print()
print("Adds preprocessing views AND concatenated features as new processings.")
print()

pipeline_3 = [
    {"y_processing": MinMaxScaler()},

    # Feature augmentation with multiple processing views
    {
        "feature_augmentation": [
            StandardNormalVariate,           # Adds "StandardNormalVariate" processing
            FirstDerivative,                 # Adds "FirstDerivative" processing
            {
                "concat_transform": [     # Adds "concat_PCA_TruncatedSVD" processing
                    PCA(n_components=20),
                    TruncatedSVD(n_components=10),
                ]
            },
        ]
    },

    "chart_2d",

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_3 = PipelineConfigs(pipeline_3, name="Concat_Add_Mode")
predictions_3, _ = runner.run(pipeline_config_3, dataset_config)

print()
print("Example 3 completed. Multiple processings created: raw, SNV, 1stDer, concat")
print()


# ============================================================
# Example 4: Custom naming
# ============================================================
# You can specify a custom name for the output processing.

print("=" * 70)
print("EXAMPLE 4: Custom Processing Name")
print("=" * 70)
print()

pipeline_4 = [
    {"y_processing": MinMaxScaler()},

    # Concat with custom name
    {
        "concat_transform": {
            "name": "latent_features",
            "operations": [
                PCA(n_components=30),
                TruncatedSVD(n_components=20),
            ]
        }
    },

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_4 = PipelineConfigs(pipeline_4, name="Concat_Custom_Name")
predictions_4, _ = runner.run(pipeline_config_4, dataset_config)

print()
print("Example 4 completed. Processing named 'raw_latent_features'")
print()


# ============================================================
# Example 5: Generation within concat_transform
# ============================================================
# Use _or_/size generator syntax within concat_transform.
# This generates multiple PIPELINE VARIANTS, each with a different
# combination of transformers to concatenate.
#
# Starting with feature_augmented dataset:
#   (100, 3, 128) with pps = [raw, raw_SNV, raw_1stDer]
#
# concat_transform with _or_: [PCA(15), SVD(10)] and size=2 generates:
# - Pipeline variant 1: concat_transform: [PCA(15), SVD(10)]
#     -> (100, 3, 25) with pps = [raw_PCA-SVD, raw_SNV_PCA-SVD, raw_1stDer_PCA-SVD]
# - Pipeline variant 2: concat_transform: [SVD(10), PCA(15)]
#     -> (100, 3, 25) same features, different order

print("=" * 70)
print("EXAMPLE 5: Generation within concat_transform")
print("=" * 70)
print()
print("Generates multiple PIPELINE VARIANTS with different concat combinations.")
print("Each variant applies concat_transform to ALL existing processings.")
print()

pipeline_5 = [
    {"y_processing": MinMaxScaler()},

    # First create multiple processings
    {
        "feature_augmentation": [
            StandardNormalVariate,
            FirstDerivative,
        ]
    },
    # Now: (samples, 3, 128) with pps = [raw, raw_SNV, raw_1stDer]

    # concat_transform with _or_ generates multiple pipeline variants
    # Each variant applies a different concat combination to ALL 3 processings
    {
        "concat_transform": {
            "_or_": [
                PCA(n_components=15),
                TruncatedSVD(n_components=10),
            ],
            "size": 2,  # Always concatenate exactly 2 transformers
        }
    },
    # All variants: 3 processings -> [PCA, SVD] concat -> (samples, 3, 25)

    "chart_2d",

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_5 = PipelineConfigs(pipeline_5, name="Concat_Generation")
predictions_5, _ = runner.run(pipeline_config_5, dataset_config)

print()
print("Example 5 completed. Generated pipeline variants with different")
print("concat combinations applied to all processings.")
print()


# ============================================================
# Example 6: feature_augmentation with _or_ including concat_transform
# ============================================================
# Use _or_ generation at the feature_augmentation level where
# concat_transform is one of the options alongside regular transforms.
#
# This generates pipeline variants where each variant picks different
# augmentation strategies, and concat_transform is treated as one option.
#
# Pool: [SNV, FirstDer, concat_transform:[PCA, SVD]]
# size=2 generates variants like:
# - Variant 1: feature_augmentation: [SNV, FirstDer]
# - Variant 2: feature_augmentation: [SNV, concat_transform:[PCA,SVD]]
# - Variant 3: feature_augmentation: [FirstDer, concat_transform:[PCA,SVD]]

print("=" * 70)
print("EXAMPLE 6: feature_augmentation with _or_ including concat_transform")
print("=" * 70)
print()
print("Generates pipeline variants from feature_augmentation pool.")
print("concat_transform is one option alongside regular transforms.")
print()

pipeline_6 = [
    {"y_processing": MinMaxScaler()},

    # feature_augmentation with _or_ generation
    # concat_transform is one of the augmentation options
    {
        "feature_augmentation": {
            "_or_": [
                StandardNormalVariate,
                FirstDerivative,
                # concat_transform as an augmentation option
                # When selected, it ADDS a processing with concatenated features
                {
                    "concat_transform": [
                        PCA(n_components=15),
                        TruncatedSVD(n_components=10),
                    ]
                },
            ],
            "size": 2,  # Pick exactly 2 augmentations
        }
    },
    # Possible results:
    # - pps = [raw, raw_SNV, raw_1stDer]
    # - pps = [raw, raw_SNV, concat_PCA_SVD]
    # - pps = [raw, raw_1stDer, concat_PCA_SVD]

    "chart_2d",

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    PLSRegression(n_components=10),
]

pipeline_config_6 = PipelineConfigs(pipeline_6, name="FA_with_Concat_Option")
predictions_6, _ = runner.run(pipeline_config_6, dataset_config)

print()
print("Example 6 completed. Generated pipeline variants where concat_transform")
print("is one of the augmentation options alongside regular transforms.")
print()


# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("concat_transform: concatenates transformer outputs horizontally")
print()
print("Key behavior:")
print("  - Applies to ALL existing processings in parallel")
print("  - REPLACES each processing with concatenated features")
print("  - Chains [A, B] execute sequentially: B(A(X))")
print()
print("Examples shown:")
print()
print("1. Top-level (Replace Mode):")
print("   raw -> concat_transform:[PCA,SVD] -> raw_PCA-SVD")
print()
print("2. Chained transformers:")
print("   [Scaler, PCA] means PCA(Scaler(X))")
print()
print("3. Inside feature_augmentation (Add Mode):")
print("   Adds concat output as new processing alongside originals")
print()
print("4. Custom naming:")
print("   Use dict format with 'name' key")
print()
print("5. Generation within concat_transform:")
print("   _or_:[PCA,SVD] + size=(1,2) -> pipeline variants")
print("   Each variant has different concat combination")
print()
print("6. feature_augmentation with concat_transform option:")
print("   _or_:[SNV, concat_transform] -> pipeline variants")
print("   concat_transform is one augmentation choice")
print()
print("All examples completed successfully!")
