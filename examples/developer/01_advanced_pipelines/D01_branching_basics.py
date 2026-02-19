"""
D01 - Branching Basics: Introduction to Pipeline Branching
===========================================================

Pipeline branching enables running multiple parallel sub-pipelines ("branches"),
each with its own preprocessing context while sharing common upstream state.

This tutorial covers:

* Basic branching with list syntax (duplication branches)
* Named branches with dictionary syntax
* Generator-based branching with ``_or_``
* Multi-step branches with Y processing
* In-branch model training
* Branch comparison visualization
* Introduction to separation branches (by_tag, by_metadata, by_source)

Branch Types:
  - **Duplication branches**: Same samples, different preprocessing (default)
  - **Separation branches**: Different samples, parallel processing (by_tag, by_metadata, by_filter, by_source)

Prerequisites
-------------
- U02_basic_regression for pipeline basics

Next Steps
----------
After this example, see :ref:`D02_branching_advanced` for statistics and HTML reports.
See :ref:`D06_separation_branches` for full separation branch documentation.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import FirstDerivative, SavitzkyGolay, SecondDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D01 Branching Basics Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Basic Branching with List Syntax
# =============================================================================
print("\n" + "=" * 60)
print("D01 - Branching Basics")
print("=" * 60)

print("""
Pipeline branching allows testing multiple preprocessing strategies
in a single run. Each branch:
  - Shares upstream state (data loading, initial preprocessing, splits)
  - Has independent X and Y processing contexts
  - Can contain its own models

Let's start with the simplest form: list syntax.
""")

print("\n" + "-" * 60)
print("Example 1: Basic List Syntax")
print("-" * 60)

# The simplest form of branching uses a list of lists
# Each inner list defines the steps for one branch
pipeline_basic = [
    MinMaxScaler(),  # Shared preprocessing - applied once
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": [
        [SNV()],              # Branch 0: SNV preprocessing
        [MSC()],              # Branch 1: MSC preprocessing
        [FirstDerivative()],  # Branch 2: First derivative
    ]},
    PLSRegression(n_components=5),  # Executes on EACH branch
]

print("Pipeline structure:")
print("  1. MinMaxScaler (shared)")
print("  2. ShuffleSplit")
print("  3. Branch: [SNV | MSC | FirstDerivative]")
print("  4. PLSRegression (runs 3 times, once per branch)")

result_basic = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    name="BasicBranching",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nTotal predictions: {result_basic.num_predictions}")
branches = result_basic.predictions.get_unique_values('branch_name')
print(f"Branches: {branches}")

# =============================================================================
# Section 2: Named Branches with Dictionary Syntax
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Named Branches with Dictionary Syntax")
print("-" * 60)

print("""
Dictionary syntax gives meaningful names to branches.
These names appear in predictions and visualizations.
""")

pipeline_named = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "savgol": [SavitzkyGolay(window_length=11, polyorder=2)],
        "derivative": [FirstDerivative()],
    }},
    PLSRegression(n_components=5),
]

result_named = nirs4all.run(
    pipeline=pipeline_named,
    dataset="sample_data/regression",
    name="NamedBranches",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nBranch names: {result_named.predictions.get_unique_values('branch_name')}")

# =============================================================================
# Section 3: Generator-Based Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Generator-Based Branching")
print("-" * 60)

print("""
Use _or_ generators inside branches for dynamic branch creation.
This is useful when comparing many preprocessing options.
""")

pipeline_generator = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {"_or_": [
        SNV(),
        MSC(),
        FirstDerivative(),
        SecondDerivative(),
    ]}},
    PLSRegression(n_components=5),
]

result_generator = nirs4all.run(
    pipeline=pipeline_generator,
    dataset="sample_data/regression",
    name="GeneratorBranches",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nBranches from generator: {result_generator.predictions.get_unique_values('branch_name')}")

# =============================================================================
# Section 4: Multi-Step Branches with Y Processing
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Multi-Step Branches with Y Processing")
print("-" * 60)

print("""
Branches can contain multiple steps, including Y processing.
Each branch maintains independent X and Y processing state.
""")

pipeline_multistep = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv_scaled_y": [
            SNV(),
            {"y_processing": StandardScaler()},  # Branch-specific Y scaling
        ],
        "msc_raw_y": [
            MSC(),
            # No Y processing - uses numeric targets directly
        ],
        "derivative_savgol": [
            FirstDerivative(),
            SavitzkyGolay(window_length=11, polyorder=2),
        ],
    }},
    PLSRegression(n_components=5),
]

result_multistep = nirs4all.run(
    pipeline=pipeline_multistep,
    dataset="sample_data/regression",
    name="MultiStepBranches",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMulti-step branches: {result_multistep.predictions.get_unique_values('branch_name')}")

# =============================================================================
# Section 5: In-Branch Model Training
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: In-Branch Model Training")
print("-" * 60)

print("""
Models can be placed inside branches to train different models
with different preprocessing strategies.
""")

pipeline_in_branch = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv_pls5": [SNV(), PLSRegression(n_components=5)],
        "snv_pls10": [SNV(), PLSRegression(n_components=10)],
        "msc_pls5": [MSC(), PLSRegression(n_components=5)],
        "msc_pls10": [MSC(), PLSRegression(n_components=10)],
    }},
]

result_in_branch = nirs4all.run(
    pipeline=pipeline_in_branch,
    dataset="sample_data/regression",
    name="InBranchModels",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nIn-branch model predictions: {result_in_branch.num_predictions}")
print(f"Branches: {result_in_branch.predictions.get_unique_values('branch_name')}")

# =============================================================================
# Section 6: Introduction to Separation Branches
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Introduction to Separation Branches")
print("-" * 60)

print("""
All examples so far use "duplication branches" - each branch processes
the SAME samples with different preprocessing.

There's another type: "separation branches" - each branch processes
DIFFERENT samples based on some criterion:

  by_tag:      Branch by tag values (from outlier detection, etc.)
  by_metadata: Branch by metadata column (e.g., instrument, farm)
  by_filter:   Branch by filter result (pass/fail)
  by_source:   Branch by feature source (for multi-source datasets)

Example: Branch by metadata column
""")

# Simple example of separation branching by metadata
pipeline_separation = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    # This branches by the "farm" metadata column
    # Each unique farm value creates a separate branch
    {"branch": {
        "by_metadata": "farm",  # Each farm gets its own branch
        "steps": [SNV()],       # Same preprocessing for all
    }},
    PLSRegression(n_components=5),  # Trains separate model per farm
]

print("""Pipeline structure:
  1. MinMaxScaler (shared)
  2. ShuffleSplit
  3. Branch by metadata "farm": Samples with farm=A go to branch A, etc.
  4. PLSRegression (trains once per farm)

Note: Separation branches require "concat" merge to reassemble predictions.
See D06_separation_branches.py for complete examples.
""")

# =============================================================================
# Section 7: Branch Comparison Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Branch Comparison Visualization")
print("-" * 60)

print("""
PredictionAnalyzer provides methods for comparing branch performance:
- branch_summary(): Statistics table
- plot_branch_comparison(): Bar chart with CI
- plot_branch_boxplot(): Score distributions
- plot_branch_heatmap(): Branch × Fold performance
""")

# Create analyzer for the last run's predictions
analyzer = PredictionAnalyzer(result_in_branch.predictions, output_dir="charts")

# Get branch summary statistics
print("\n📊 Branch Summary:")
summary = analyzer.branch_summary(metrics=['rmse', 'r2'])
if hasattr(summary, 'to_string'):
    print(summary.to_string(index=False))
else:
    print(summary)

# Get list of branches
print(f"\nAvailable branches: {analyzer.get_branches()}")

# Create visualizations if requested
if args.plots or args.show:
    # Bar chart comparing branches with confidence intervals
    fig_comparison = analyzer.plot_branch_comparison(
        display_metric='rmse',
        display_partition='test',
        show_ci=True,
        ci_level=0.95
    )
    print("Created branch comparison bar chart")

    # Boxplot showing distribution across branches
    fig_boxplot = analyzer.plot_branch_boxplot(
        display_metric='rmse',
        display_partition='test'
    )
    print("Created branch boxplot")

    # Heatmap of branch performance across folds
    fig_heatmap = analyzer.plot_branch_heatmap(
        y_var='fold_id',
        display_metric='rmse',
        display_partition='test'
    )
    print("Created branch × fold heatmap")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Branching enables parallel preprocessing experiments
2. List syntax: [branch1, branch2, branch3]
3. Dict syntax: {"name1": branch1, "name2": branch2} for named branches
4. Generator syntax: {"_or_": [options]} for dynamic branches
5. Branches can contain multiple steps and Y processing
6. Models inside branches train independently
7. Two branch types: duplication (same samples) vs separation (different samples)

Branch Types Summary:
  DUPLICATION (default):    {"branch": [[SNV()], [MSC()]]}
  SEPARATION by_tag:        {"branch": {"by_tag": "outlier", ...}}
  SEPARATION by_metadata:   {"branch": {"by_metadata": "farm", ...}}
  SEPARATION by_filter:     {"branch": {"by_filter": Filter(), ...}}
  SEPARATION by_source:     {"branch": {"by_source": True, ...}}

Key benefits:
- Single dataset load, no redundant I/O
- Shared splits across all branches
- Easy comparison with visualization tools

Next: D02_branching_advanced.py - BranchAnalyzer and statistical comparison
      D06_separation_branches.py - Full separation branch documentation
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
