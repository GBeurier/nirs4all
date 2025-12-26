"""
D02 - Branching Advanced: Statistics and Export
================================================

Advanced branching techniques with statistical analysis and export.

This tutorial covers:

* BranchAnalyzer for statistical comparison
* BranchSummary for Markdown, LaTeX, and CSV export
* Pairwise statistical testing
* Branch ranking methods
* Publication-ready outputs

Prerequisites
-------------
- D01_branching_basics for core branching concepts

Next Steps
----------
See D03_merge_basics for merging branch outputs.

Duration: ~3 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SavitzkyGolay
)
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.visualization.analysis.branch import BranchAnalyzer, BranchSummary


# Parse command-line arguments
parser = argparse.ArgumentParser(description='D02 Branching Advanced Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Setup: Common Pipeline Run
# =============================================================================
print("\n" + "=" * 60)
print("D02 - Branching Advanced")
print("=" * 60)

print("""
We'll run a branched pipeline and then analyze it using:
- BranchAnalyzer: Statistical tests and rankings
- BranchSummary: Markdown, LaTeX, and CSV export
""")

# Run a branched pipeline for analysis
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "savgol": [SavitzkyGolay(window_length=11, polyorder=2)],
        "derivative": [FirstDerivative()],
        "snv+derivative": [SNV(), FirstDerivative()],
    }},
    PLSRegression(n_components=5),
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="BranchAnalysis",
    verbose=1,
    plots_visible=args.plots
)

predictions = result.predictions


# =============================================================================
# Section 1: BranchAnalyzer for Statistical Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: BranchAnalyzer for Statistical Comparison")
print("-" * 60)

print("""
BranchAnalyzer provides rigorous statistical methods:
- summary(): Mean, std, min, max for each branch
- rank_branches(): Statistical ranking by performance
- compare(): Compare two branches with statistical tests
- pairwise_comparison(): All-pairs comparison matrix
""")

# Create the analyzer
analyzer = BranchAnalyzer(predictions=predictions)

# Get summary statistics
print("\n📊 Branch Summary (RMSE, test partition):")
summary = analyzer.summary(metrics=['rmse', 'r2'], partition='test')

# Print as markdown table
print(summary.to_markdown())


# =============================================================================
# Section 2: Branch Ranking
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Branch Ranking")
print("-" * 60)

print("""
rank_branches() orders branches by mean performance.
Lower RMSE is better, so ascending=True (default for RMSE).
""")

# Rank branches by RMSE
rankings = analyzer.rank_branches(metric='rmse', partition='test')

print("\n🏆 Branch Rankings (lower RMSE is better):")
for rank_info in rankings:
    print(f"  #{rank_info['rank']}: {rank_info['branch_name']} "
          f"(RMSE: {rank_info['rmse_mean']:.4f} ± {rank_info['rmse_std']:.4f})")


# =============================================================================
# Section 3: Pairwise Statistical Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Pairwise Statistical Comparison")
print("-" * 60)

print("""
Compare two branches statistically using t-test, Wilcoxon, or Mann-Whitney.
Returns p-value, effect size (Cohen's d), and significance.
""")

# Get branch names
branches = analyzer.get_branch_names()

if len(branches) >= 2:
    print(f"\n🔍 Statistical Comparison: {branches[0]} vs {branches[1]}")
    try:
        comparison = analyzer.compare(
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
    except ValueError as e:
        print(f"  Could not compare: {e}")


# =============================================================================
# Section 4: Pairwise Comparison Matrix
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Pairwise Comparison Matrix")
print("-" * 60)

print("""
pairwise_comparison() computes p-values for all branch pairs.
Returns a DataFrame matrix useful for multiple comparison correction.
""")

try:
    pairwise_matrix = analyzer.pairwise_comparison(
        metric='rmse',
        partition='test',
        test='ttest'
    )
    print("\n📊 Pairwise p-value matrix:")
    print(pairwise_matrix.round(4).to_string())
except ImportError:
    print("scipy not installed - skipping pairwise comparison")
except Exception as e:
    print(f"Could not compute pairwise comparison: {e}")


# =============================================================================
# Section 5: LaTeX Export for Publications
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: LaTeX Export for Publications")
print("-" * 60)

print("""
BranchSummary provides export to LaTeX for publication-ready tables.
The to_latex() method formats mean ± std in math mode.
""")

# Generate LaTeX table
latex_output = summary.to_latex(
    caption="Branch Performance Comparison",
    label="tab:branch_comparison",
    precision=4,
    mean_std_combined=True
)
print("\n📄 LaTeX Output:")
print(latex_output)


# =============================================================================
# Section 6: CSV Export
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: CSV Export")
print("-" * 60)

# Ensure output directory exists
Path("reports").mkdir(exist_ok=True)

# Export to CSV
summary_full = analyzer.summary(metrics=['rmse', 'r2', 'mae'], partition='test')
summary_full.to_csv("reports/branch_summary.csv")
print("📄 Exported branch summary to reports/branch_summary.csv")


# =============================================================================
# Section 7: Visualization with PredictionAnalyzer
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Branch Visualization")
print("-" * 60)

print("""
PredictionAnalyzer provides visual comparison of branches:
- branch_summary(): Statistics table
- plot_branch_comparison(): Bar chart with error bars
- plot_branch_boxplot(): Score distributions
""")

viz_analyzer = PredictionAnalyzer(predictions)

# Get branch summary from PredictionAnalyzer
branch_df = viz_analyzer.branch_summary()
print("\n📊 Branch Summary (via PredictionAnalyzer):")
print(branch_df.to_string(index=False) if hasattr(branch_df, 'to_string') else branch_df)

if args.plots or args.show:
    # Create branch comparison chart
    fig_compare = viz_analyzer.plot_branch_comparison(
        metric='rmse',
        partition='test'
    )
    print("Created branch comparison chart")

    # Create branch boxplot
    fig_boxplot = viz_analyzer.plot_branch_boxplot(
        metric='rmse',
        partition='test'
    )
    print("Created branch boxplot")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. BranchAnalyzer provides summary(), rank_branches(), compare()
2. BranchSummary exports to Markdown, LaTeX, and CSV
3. pairwise_comparison() creates all-pairs p-value matrix
4. Cohen's d measures effect size for practical significance
5. LaTeX export with mean ± std formatting for publications
6. PredictionAnalyzer adds visual branch comparison

Key outputs:
- reports/branch_summary.csv - Branch statistics export

Next: D03_merge_basics.py - Combining branch outputs
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
