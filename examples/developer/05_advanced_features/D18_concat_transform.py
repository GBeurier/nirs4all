"""
D18 - Concat Transform: Feature Concatenation
==============================================

concat_transform combines features from different transformers
into a single feature matrix by horizontal concatenation.

This tutorial covers:

* concat_transform basics
* Concatenating multiple transformer outputs
* Using chained transformers
* Named transforms
* Comparison with branching

Prerequisites
-------------
- D01_branching_basics for branching concepts
- D03_merge_basics for merge operations

Next Steps
----------
See D19_repetition_transform for handling repeated measurements.

Duration: ~3 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    FirstDerivative,
    SavitzkyGolay
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D18 Concat Transform Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D18 - Concat Transform: Feature Concatenation")
print("=" * 60)

print("""
concat_transform horizontally concatenates features from
multiple transformers:

    Original: (n, 500)
        ↓
    PCA(30) → (n, 30)  ┐
    SVD(20) → (n, 20)  ├─→ Concat: (n, 50)
                       ┘

Use cases:
  - Combine dimensionality reduction methods
  - Multi-method feature extraction
  - Ensemble feature views
""")


# =============================================================================
# Section 1: Basic Concatenation
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic Concatenation")
print("-" * 60)

print("""
Concatenate multiple transformer outputs:

    {"concat_transform": [
        PCA(n_components=30),
        TruncatedSVD(n_components=20)
    ]}

Result: 30 + 20 = 50 features
""")

pipeline_basic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"concat_transform": [
        PCA(n_components=30),
        TruncatedSVD(n_components=20)
    ]},
    PLSRegression(n_components=5),
]

result = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    name="BasicConcat",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result.num_predictions}")
print("Features: 30 (PCA) + 20 (SVD) = 50 total")


# =============================================================================
# Section 2: Chained Transformers
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Chained Transformers")
print("-" * 60)

print("""
Use chains for sequential processing within each branch:

    {"concat_transform": [
        PCA(n_components=25),
        [StandardScaler(), PCA(n_components=15)]  # Chain: scale→PCA
    ]}

Chains execute sequentially: [A, B] means B(A(X))
""")

pipeline_chained = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"concat_transform": [
        PCA(n_components=25),                       # 25 features
        [StandardScaler(), PCA(n_components=15)],   # Scale then PCA: 15 features
    ]},
    PLSRegression(n_components=5),
]

result_chained = nirs4all.run(
    pipeline=pipeline_chained,
    dataset="sample_data/regression",
    name="ChainedConcat",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result_chained.num_predictions}")
print("Features: 25 (PCA direct) + 15 (scaled PCA) = 40 total")


# =============================================================================
# Section 3: Multiple Feature Extractors
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Multiple Feature Extractors")
print("-" * 60)

print("""
Combine different feature extraction methods:

    {"concat_transform": [
        PCA(n_components=20),
        TruncatedSVD(n_components=20),
        [SNV(), PCA(n_components=10)]  # SNV preprocessing + PCA
    ]}
""")

pipeline_multi = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"concat_transform": [
        PCA(n_components=20),
        TruncatedSVD(n_components=20),
        [SNV(), PCA(n_components=10)]
    ]},
    PLSRegression(n_components=10),
]

result_multi = nirs4all.run(
    pipeline=pipeline_multi,
    dataset="sample_data/regression",
    name="MultiConcat",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result_multi.num_predictions}")
print("Features: 20 + 20 + 10 = 50 total")


# =============================================================================
# Section 4: Comparison with Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Comparison with Branching")
print("-" * 60)

print("""
concat_transform vs branch+merge:

    concat_transform:
        - Simpler syntax for feature concatenation
        - All transforms applied in parallel
        - Single merged output
        - No separate models per branch

    branch + merge:
        - Full control over each path
        - Can have models inside branches
        - Prediction or feature merging
        - More flexible but complex
""")

# Equivalent using branching
print("Equivalent branch+merge syntax:")
print("""
    {"branch": {
        "pca": [PCA(30)],
        "svd": [TruncatedSVD(20)]
    }},
    {"merge": "features"}
""")


# =============================================================================
# Section 5: Inside Feature Augmentation
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: concat_transform in feature_augmentation")
print("-" * 60)

print("""
When nested inside feature_augmentation, concat_transform
ADDS a new processing instead of replacing:

    {"feature_augmentation": [
        SNV(),
        {"concat_transform": [PCA(30), SVD(20)]}
    ]}

Result:
  - Original "raw" processing: 2151 features
  - New "snv" processing: 2151 features
  - New "concat_PCA_SVD" processing: 50 features
""")

pipeline_augment = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"feature_augmentation": [
        SNV(),
        {"concat_transform": [
            PCA(n_components=30),
            TruncatedSVD(n_components=20)
        ]}
    ]},
    PLSRegression(n_components=10),
]

result_augment = nirs4all.run(
    pipeline=pipeline_augment,
    dataset="sample_data/regression",
    name="AugmentConcat",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result_augment.num_predictions}")
print("New processings added alongside original")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. concat_transform concatenates transformer outputs
2. Use list for multiple transformers: [PCA, SVD]
3. Use nested list for chains: [StandardScaler(), PCA()]
4. Top-level: replaces features
5. In feature_augmentation: adds new processing

Key syntax:
    {"concat_transform": [
        TransformerA(params),
        TransformerB(params),
        [Chain, Of, Transforms]
    ]}

When to use:
  - Combining dimensionality reduction methods
  - Multi-method feature extraction
  - Simple parallel feature views

When to use branch+merge:
  - Need models inside branches
  - Want prediction merging
  - Complex control flow

Next: D19_repetition_transform.py - Handling repeated measurements
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
