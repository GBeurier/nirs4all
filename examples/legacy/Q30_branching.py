"""
Q30_branching.py - Pipeline Branching Examples

This example demonstrates the branching feature of nirs4all, which enables
splitting a pipeline into multiple parallel sub-pipelines ("branches"),
each with its own preprocessing context while sharing common upstream state.

Branching enables efficient multi-configuration experimentation without
redundant dataset loading or split recomputation.

Key features demonstrated:
1. Basic branching with list syntax
2. Named branches with dictionary syntax
3. Generator-based branching with _or_
4. Post-branch step execution
5. In-branch model training
6. Branch comparison visualization
7. Branch diagram visualization (Phase 6)
8. HTML report generation (Phase 6)
9. LaTeX export for publications (Phase 6)
"""

# %%
# === Setup ===
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay
)
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.visualization.analysis.branch import BranchAnalyzer, BranchSummary

# %% [markdown]
# ## Example 1: Basic Branching with List Syntax
#
# The simplest form of branching uses a list of lists, where each inner list
# defines the steps for a single branch. Steps after the branch block execute
# on each branch independently.

# %%
# === Example 1: Basic List Syntax ===
print("=" * 60)
print("Example 1: Basic Branching with List Syntax")
print("=" * 60)

pipeline_basic = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),  # Shared preprocessing - applied once
    {"branch": [
        [SNV()],           # Branch 0: SNV preprocessing
        [MSC()],           # Branch 1: MSC preprocessing
        [FirstDerivative()],  # Branch 2: First derivative
    ]},
    PLSRegression(n_components=5),  # Executes on EACH branch
]

# Note: Replace with your actual dataset path
dataset_config = DatasetConfigs("sample_data/regression")
pipeline_config = PipelineConfigs(pipeline_basic)

runner = PipelineRunner(workspace_path="workspace")
predictions, pipelines = runner.run(pipeline_config, dataset_config)

print(f"\nTotal predictions: {len(predictions)}")
print(f"Branches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 2: Named Branches with Dictionary Syntax
#
# For better tracking and visualization, you can use named branches.
# The dictionary keys become the branch names in predictions and charts.

# %%
# === Example 2: Named Branches ===
print("\n" + "=" * 60)
print("Example 2: Named Branches with Dictionary Syntax")
print("=" * 60)

pipeline_named = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "savgol": [SavitzkyGolay(window_length=11, polyorder=2)],
        "derivative": [FirstDerivative()],
    }},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_named)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nBranch names: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 3: Generator-Based Branching
#
# Use `_or_` generators inside branches for dynamic branch creation.
# This is useful when you want to compare many preprocessing options.

# %%
# === Example 3: Generator Syntax ===
print("\n" + "=" * 60)
print("Example 3: Generator-Based Branching")
print("=" * 60)

pipeline_generator = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    {"branch": {"_or_": [
        SNV(),
        MSC(),
        FirstDerivative(),
        SecondDerivative(),
    ]}},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_generator)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nBranches from generator: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 4: Multi-Step Branches with Y Processing
#
# Branches can contain multiple steps, including Y processing.
# Each branch maintains independent X and Y processing state.

# %%
# === Example 4: Multi-Step Branches with Y Processing ===
print("\n" + "=" * 60)
print("Example 4: Multi-Step Branches with Y Processing")
print("=" * 60)

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

pipeline_config = PipelineConfigs(pipeline_multistep)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nMulti-step branches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 5: In-Branch Model Training
#
# Models can be placed inside branches to train different models
# with different preprocessing strategies.

# %%
# === Example 5: In-Branch Models ===
print("\n" + "=" * 60)
print("Example 5: In-Branch Model Training")
print("=" * 60)

pipeline_in_branch = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    {"branch": {
        "snv_pls5": [SNV(), PLSRegression(n_components=5)],
        "snv_pls10": [SNV(), PLSRegression(n_components=10)],
        "msc_pls5": [MSC(), PLSRegression(n_components=5)],
        "msc_pls10": [MSC(), PLSRegression(n_components=10)],
    }},
]

pipeline_config = PipelineConfigs(pipeline_in_branch)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nIn-branch model predictions: {len(predictions)}")
print(f"Branches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 6: Branch Comparison Visualization
#
# The `PredictionAnalyzer` provides methods for comparing branch performance.

# %%
# === Example 6: Branch Visualization ===
print("\n" + "=" * 60)
print("Example 6: Branch Comparison Visualization")
print("=" * 60)

# Create analyzer for the last run's predictions
analyzer = PredictionAnalyzer(predictions, output_dir="charts")

# Get branch summary statistics
print("\n--- Branch Summary ---")
summary = analyzer.branch_summary(metrics=['rmse', 'r2'])
if hasattr(summary, 'to_string'):
    print(summary.to_string(index=False))  # type: ignore[union-attr]
else:
    print(summary)

# Get list of branches
print(f"\nAvailable branches: {analyzer.get_branches()}")
print(f"Branch IDs: {analyzer.get_branch_ids()}")

# %% [markdown]
# ### Bar Chart with Confidence Intervals

# %%
# Bar chart comparing branches with confidence intervals
fig_comparison = analyzer.plot_branch_comparison(
    display_metric='rmse',
    display_partition='test',
    show_ci=True,
    ci_level=0.95
)
print("Created branch comparison bar chart")

# %% [markdown]
# ### Boxplot of Score Distributions

# %%
# Boxplot showing distribution across branches
fig_boxplot = analyzer.plot_branch_boxplot(
    display_metric='rmse',
    display_partition='test'
)
print("Created branch boxplot")

# %% [markdown]
# ### Heatmap: Branch × Fold Performance

# %%
# Heatmap of branch performance across folds
fig_heatmap = analyzer.plot_branch_heatmap(
    y_var='fold_id',
    display_metric='rmse',
    display_partition='test'
)
print("Created branch × fold heatmap")

# %% [markdown]
# ### Using Standard Heatmap with Branch Variables
#
# You can also use `plot_heatmap` directly with `branch_name` as an axis variable.

# %%
# Standard heatmap with branch_name on x-axis
fig_heatmap2 = analyzer.plot_heatmap(
    x_var='branch_name',
    y_var='model_name',
    display_metric='rmse',
    display_partition='test',
    rank_partition='val'
)
print("Created branch × model heatmap")

# %% [markdown]
# ## Example 7: Branch Diagram Visualization (Phase 6)
#
# The `plot_branch_diagram` method creates a DAG visualization showing
# the branching structure of the pipeline.

# %%
# === Example 7: Branch Diagram ===
print("\n" + "=" * 60)
print("Example 7: Branch Diagram Visualization")
print("=" * 60)

# Create branch diagram showing pipeline structure
fig_diagram = analyzer.plot_branch_diagram(
    show_metrics=True,
    metric='rmse',
    partition='test'
)
print("Created branch diagram visualization")

# %% [markdown]
# ## Example 8: BranchAnalyzer for Advanced Statistics (Phase 6)
#
# The `BranchAnalyzer` class provides advanced statistical analysis
# including pairwise comparisons and significance testing.

# %%
# === Example 8: BranchAnalyzer Statistics ===
print("\n" + "=" * 60)
print("Example 8: BranchAnalyzer Statistics")
print("=" * 60)

# Create branch analyzer
branch_analyzer = BranchAnalyzer(predictions)

# Get summary with BranchSummary object
summary = branch_analyzer.summary(metrics=['rmse', 'r2'], partition='test')

# Print as markdown
print("\n--- Summary (Markdown) ---")
print(summary.to_markdown())

# Print as LaTeX for publications
print("\n--- Summary (LaTeX) ---")
latex_output = summary.to_latex(
    caption="Branch Performance Comparison",
    label="tab:branch_comparison",
    precision=4
)
print(latex_output)

# Rank branches by performance
print("\n--- Branch Ranking ---")
rankings = branch_analyzer.rank_branches(metric='rmse', partition='test')
for rank_info in rankings:
    print(f"  #{rank_info['rank']}: {rank_info['branch_name']} "
          f"(RMSE: {rank_info['rmse_mean']:.4f} ± {rank_info['rmse_std']:.4f})")

# %% [markdown]
# ### Statistical Comparison Between Branches

# %%
# Compare two branches statistically (requires scipy)
branches = branch_analyzer.get_branch_names()
if len(branches) >= 2:
    print(f"\n--- Statistical Comparison: {branches[0]} vs {branches[1]} ---")
    try:
        comparison = branch_analyzer.compare(
            branches[0], branches[1],
            metric='rmse',
            partition='test',
            test='ttest'
        )
        print(f"  {branches[0]} mean: {comparison['branch1_mean']:.4f} ± {comparison['branch1_std']:.4f}")
        print(f"  {branches[1]} mean: {comparison['branch2_mean']:.4f} ± {comparison['branch2_std']:.4f}")
        print(f"  p-value: {comparison['p_value']:.4f}")
        print(f"  Significant (α=0.05): {comparison['significant']}")
        print(f"  Effect size (Cohen's d): {comparison['effect_size']:.4f}")
    except ImportError:
        print("  scipy not installed - skipping statistical comparison")

# %% [markdown]
# ## Example 9: HTML Report Generation (Phase 6)
#
# Generate a comprehensive HTML report with all branch visualizations.

# %%
# === Example 9: HTML Report ===
print("\n" + "=" * 60)
print("Example 9: HTML Report Generation")
print("=" * 60)

# Generate HTML report
report_path = analyzer.generate_report(
    output_path="charts/branch_comparison_report.html",
    branch_comparison=True,
    include_diagrams=True,
    include_tables=True,
    metrics=['rmse', 'r2'],
    title="Branch Comparison Report"
)
print(f"Generated report: {report_path}")

# %% [markdown]
# ## Example 10: Filtering Predictions by Branch

# %%
# === Example 10: Branch Filtering ===
print("\n" + "=" * 60)
print("Example 10: Filtering Predictions by Branch")
print("=" * 60)

# Get predictions from a specific branch by name
if branches:
    first_branch = branches[0]
    branch_preds = predictions.filter_predictions(branch_name=first_branch)
    print(f"\nPredictions for '{first_branch}' branch: {len(branch_preds)}")

# Get predictions from a specific branch by ID
branch_0_preds = predictions.filter_predictions(branch_id=0)
print(f"Predictions for branch_id=0: {len(branch_0_preds)}")

# Get top models per branch
print("\n--- Top Model per Branch ---")
for branch_name in analyzer.get_branches():
    top = predictions.top(n=1, rank_metric='rmse', display_metrics=['rmse'], branch_name=branch_name)
    if top:
        best = top[0]
        # Get val score from the result's val_score field (default partition is 'test' for display)
        score = best.get('val_score')
        if score is not None:
            print(f"  {branch_name}: RMSE = {score:.4f}")
        else:
            print(f"  {branch_name}: RMSE = N/A")

# %% [markdown]
# ## Example 11: Export Summary to CSV

# %%
# === Example 11: CSV Export ===
print("\n" + "=" * 60)
print("Example 11: CSV Export")
print("=" * 60)

# Export summary to CSV
summary = branch_analyzer.summary(metrics=['rmse', 'r2', 'mae'], partition='test')
summary.to_csv("charts/branch_summary.csv")
print("Exported branch summary to charts/branch_summary.csv")

# %% [markdown]
# ## Summary
#
# The branching feature enables:
#
# 1. **Efficient comparison** of preprocessing strategies
# 2. **Shared state** (splits, initial preprocessing) across branches
# 3. **Single dataset load** - no redundant I/O
# 4. **Independent contexts** for X and Y processing per branch
# 5. **Rich visualization** for branch comparison
# 6. **Flexible syntax** - list, dict, or generator-based
# 7. **DAG diagrams** for visualizing pipeline structure (Phase 6)
# 8. **HTML reports** for comprehensive analysis (Phase 6)
# 9. **LaTeX export** for publication-ready tables (Phase 6)
# 10. **Statistical testing** for rigorous comparisons (Phase 6)
#
# Use branching when you want to compare multiple preprocessing configurations
# with the same model, or different model-preprocessing combinations.

print("\n" + "=" * 60)
print("Branching examples completed!")
print("=" * 60)
