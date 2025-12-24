"""
Q35_metadata_branching.py - Metadata Partitioner Branching Examples

This example demonstrates the metadata_partitioner branching feature in nirs4all,
which creates disjoint sample branches based on metadata column values.

**Key Concept**: Unlike regular branches (where all branches see all samples),
metadata_partitioner creates branches where each sample exists in exactly ONE branch.
This enables site-specific, variety-specific, or instrument-specific models.

**Features demonstrated:**
1. Basic metadata partitioning by a column (e.g., site, variety, instrument)
2. Per-branch CV with independent fold generation
3. Value grouping for rare categories
4. min_samples filtering to skip small groups
5. Disjoint sample merge with model selection
6. Meta-learner stacking on per-partition predictions

**Use cases:**
- Site calibration transfer: Train separate models per site, combine with meta-learner
- Variety-specific models: Handle variety-dependent spectral signatures
- Instrument harmonization: Per-instrument preprocessing and modeling
- Temporal partitioning: Per-year or per-season models

See: docs/reports/disjoint_sample_branch_merging.md
"""

# %%
# === Setup ===
import os
import sys
import argparse

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Metadata Branching Examples")
parser.add_argument("--plots", action="store_true", help="Show plots interactively")
parser.add_argument("--show", action="store_true", help="Show all plots")
parser.add_argument("--verbose", "-v", type=int, default=1, help="Verbosity level")
args = parser.parse_args()


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# %% [markdown]
# ## Overview: Metadata Partitioner vs Regular Branches
#
# **Regular branches** (copy branches):
# - All branches see ALL samples
# - Used for comparing preprocessing variants (SNV vs MSC)
# - Merge concatenates features horizontally or stacks predictions
#
# **Metadata partitioner** (disjoint branches):
# - Each sample exists in exactly ONE branch
# - Branches are created based on metadata column values
# - Used for site/variety/instrument-specific models
# - Merge reconstructs full sample set by concatenating rows

# %% [markdown]
# ## Part 1: Basic Metadata Partitioner
#
# The simplest use case: partition samples by a metadata column (e.g., "site")
# and train a model in each partition.

# %%
print_section("Example 1: Basic Metadata Partitioner")

print("""
This example partitions samples by 'site' column in metadata.
Each site gets its own PLS model trained on only that site's samples.

Pipeline:
├── MinMaxScaler (preprocessing)
├── SNV (spectral normalization)
├── ShuffleSplit (cross-validation)
├── branch by metadata_partitioner on column='site'
│   ├── site_A: PLSRegression(5)
│   ├── site_B: PLSRegression(5)
│   └── site_C: PLSRegression(5)
├── merge predictions (disjoint samples reconstructed)
└── Ridge (meta-learner combines per-site predictions)

Expected outcome:
- 3 PLS models, one per site
- Each model trains only on its site's samples
- Predictions merged by sample_id
- Meta-learner learns site-specific weights
""")

pipeline_basic = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    # Metadata partitioner: create branches by 'site' column
    {
        "branch": [PLSRegression(n_components=5)],
        "by": "metadata_partitioner",
        "column": "site",  # Partition by this metadata column
    },
    # Merge predictions from disjoint branches
    {"merge": "predictions"},
    # Meta-learner
    {"name": "SiteMeta_Ridge", "model": Ridge(alpha=1.0)},
]

# Note: Replace with actual dataset that has 'site' metadata
dataset_config = DatasetConfigs("sample_data/regression")
pipeline_config = PipelineConfigs(pipeline_basic, name="Basic_Metadata_Partitioner")

# For demonstration, we'll just print the pipeline structure
print("Pipeline configuration created successfully.")
print(f"Total steps: {len(pipeline_basic)}")

# %% [markdown]
# ## Part 2: Per-Branch CV
#
# Each partition can have its own cross-validation, with folds generated
# independently within each partition.

# %%
print_section("Example 2: Per-Branch CV")

print("""
This example adds per-branch CV: each partition has its own CV folds.
This is useful when sample distribution varies significantly across sites.

Key difference:
- Global CV: Same fold assignments across all branches
- Per-branch CV: Independent fold generation per partition
""")

pipeline_per_branch_cv = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),  # Global CV
    {
        "branch": [PLSRegression(n_components=5)],
        "by": "metadata_partitioner",
        "column": "site",
        # Per-branch CV: each site gets its own folds
        "cv": KFold(n_splits=3, shuffle=True, random_state=42),
    },
    {"merge": "predictions"},
    Ridge(alpha=1.0),
]

print("Per-branch CV pipeline created.")
print("Each partition will have 3 internal folds for OOF prediction generation.")

# %% [markdown]
# ## Part 3: min_samples Filtering
#
# Skip partitions with too few samples to train reliably.

# %%
print_section("Example 3: min_samples Filtering")

print("""
This example uses min_samples to skip small partitions.
Partitions with fewer than 30 samples are skipped entirely.

Use case:
- Some metadata values have very few samples (rare varieties, etc.)
- These can cause unstable models or CV failures
- min_samples provides a safety threshold
""")

pipeline_min_samples = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {
        "branch": [PLSRegression(n_components=5)],
        "by": "metadata_partitioner",
        "column": "site",
        "min_samples": 30,  # Skip sites with < 30 samples
    },
    {"merge": "predictions"},
    Ridge(alpha=1.0),
]

print("min_samples pipeline created.")
print("Sites with fewer than 30 samples will be skipped.")

# %% [markdown]
# ## Part 4: Value Grouping
#
# Combine rare metadata values into larger groups.

# %%
print_section("Example 4: Value Grouping")

print("""
This example groups rare metadata values together.
Instead of creating many small partitions, we combine them.

Scenario:
- Metadata column has values: A, B, C, D, E, F
- A and B are large sites (many samples)
- C, D, E, F are small sites (few samples each)

Solution:
- Keep A and B as separate partitions
- Group C, D, E, F into "other_sites" partition
""")

pipeline_grouping = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {
        "branch": [PLSRegression(n_components=5)],
        "by": "metadata_partitioner",
        "column": "site",
        "group_values": {
            # Group rare values together
            "other_sites": ["C", "D", "E", "F"],
        },
        # A and B remain individual partitions
    },
    {"merge": "predictions"},
    Ridge(alpha=1.0),
]

print("Value grouping pipeline created.")
print("Creates 3 partitions: A, B, and other_sites (containing C, D, E, F)")

# %% [markdown]
# ## Part 5: Multiple Models per Partition
#
# Train multiple models in each partition and select the best.

# %%
print_section("Example 5: Multiple Models per Partition")

print("""
This example trains multiple models in each partition.
The merge step can select the best model(s) from each partition.

Pipeline:
├── metadata_partitioner on 'site'
│   └── Each site trains: PLS(3), PLS(5), PLS(10), RF
├── merge predictions
│   ├── n_columns=2: keep top 2 models per site
│   └── select_by='mse': rank models by MSE
└── Ridge meta-learner

Result: 2 predictions per sample (from best 2 models in that sample's site)
""")

pipeline_multi_model = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {
        "branch": [
            {"name": "PLS_3", "model": PLSRegression(n_components=3)},
            {"name": "PLS_5", "model": PLSRegression(n_components=5)},
            {"name": "PLS_10", "model": PLSRegression(n_components=10)},
            {"name": "RF_10", "model": RandomForestRegressor(n_estimators=10, random_state=42)},
        ],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    },
    # Disjoint prediction merge with model selection
    {
        "merge": "predictions",
        "n_columns": 2,      # Keep top 2 models per partition
        "select_by": "mse",  # Rank by MSE (lower is better)
    },
    {"name": "Multi_Meta", "model": Ridge(alpha=1.0)},
]

print("Multiple models per partition pipeline created.")
print("Each site trains 4 models, but only the 2 best (by MSE) are kept.")

# %% [markdown]
# ## Part 6: R² Selection Criterion
#
# Select models by R² instead of MSE.

# %%
print_section("Example 6: R² Selection Criterion")

print("""
This example selects models by R² (higher is better).
Useful when comparing models with different scales or when R² is your
primary evaluation metric.

Selection criteria available:
- mse: Mean Squared Error (lower is better) [default]
- rmse: Root Mean Squared Error (lower is better)
- mae: Mean Absolute Error (lower is better)
- r2: R² coefficient (higher is better)
- order: Use pipeline definition order (first N models)
""")

pipeline_r2_select = [
    MinMaxScaler(),
    SNV(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {
        "branch": [
            PLSRegression(n_components=3),
            PLSRegression(n_components=5),
            PLSRegression(n_components=10),
        ],
        "by": "metadata_partitioner",
        "column": "site",
    },
    {
        "merge": "predictions",
        "n_columns": 1,     # Keep only the best model
        "select_by": "r2",  # Rank by R² (higher is better)
    },
    Ridge(alpha=1.0),
]

print("R² selection pipeline created.")
print("Only the best model (by R²) from each site is kept.")

# %% [markdown]
# ## Part 7: Feature Merge from Disjoint Branches
#
# Sometimes you want features, not predictions, from disjoint branches.

# %%
print_section("Example 7: Feature Merge (Symmetric)")

print("""
This example merges features from disjoint branches.
Each partition applies the same preprocessing, then features are combined.

IMPORTANT: For feature merge to work, all partitions must produce the
same number of features. Otherwise, you'll get an error.

Use case:
- Apply site-specific preprocessing
- Combine processed features for a global model
""")

pipeline_feature_merge = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {
        "branch": [
            SNV(),
            StandardScaler(),
        ],
        "by": "metadata_partitioner",
        "column": "site",
    },
    # Merge features (not predictions)
    {"merge": "features"},
    # Now train a global model on merged features
    PLSRegression(n_components=10),
]

print("Feature merge pipeline created.")
print("All partitions apply SNV + StandardScaler, then features are merged by sample_id.")

# %% [markdown]
# ## Part 8: Nested Preprocessing + Metadata Partitioner
#
# Combine preprocessing branches with metadata partitioner.

# %%
print_section("Example 8: Nested Branching")

print("""
This example combines preprocessing branches with metadata partitioner.
First, we try different preprocessing methods, then partition by site.

Result: 2 preprocessing × 3 sites = 6 total branch paths

This is powerful for finding the best preprocessing per site.
""")

pipeline_nested = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    # First level: preprocessing branches (copy branches)
    {
        "branch": {
            "snv": [SNV()],
            "msc": [MSC()],
        }
    },
    # Second level: metadata partitioner (disjoint branches)
    {
        "branch": [PLSRegression(n_components=5)],
        "by": "metadata_partitioner",
        "column": "site",
    },
    {"merge": "predictions"},
    Ridge(alpha=1.0),
]

print("Nested branching pipeline created.")
print("Structure: 2 preprocessing variants × N sites = 2N total models")

# %% [markdown]
# ## Summary
#
# **metadata_partitioner** creates disjoint sample branches:
# - Each sample exists in exactly ONE branch
# - Branches determined by metadata column values
# - Independent CV and models per partition
# - Merge reconstructs full sample set
#
# **Key parameters**:
# - `column`: Metadata column to partition by
# - `cv`: Per-branch cross-validation (optional)
# - `min_samples`: Skip partitions below this threshold
# - `group_values`: Combine rare values into groups
#
# **Merge options**:
# - `n_columns`: Force specific output column count
# - `select_by`: Selection criterion (mse, rmse, mae, r2, order)
#
# **When to use**:
# - Site-specific calibration models
# - Variety-specific spectral modeling
# - Instrument harmonization
# - Any scenario where different sample subsets need different treatment

# %%
print_section("Summary")

print("""
Examples completed!

Key takeaways:
1. metadata_partitioner creates disjoint branches by metadata column
2. Each partition trains models independently on its samples
3. Per-branch CV generates independent folds per partition
4. min_samples and group_values handle small/rare categories
5. Disjoint merge reconstructs full sample set by sample_id
6. n_columns and select_by control model selection in merge
7. Meta-learner can combine per-partition predictions

For production use, replace 'sample_data/regression' with your
actual dataset path containing the appropriate metadata column.

See also:
- Q31_outlier_branching.py - Sample partitioner (outlier-based)
- Q30_branching.py - Regular preprocessing branches
- Q18_stacking.py - Meta-model stacking
""")

print("\n" + "=" * 70)
print("  Q35_metadata_branching.py completed!")
print("=" * 70)
