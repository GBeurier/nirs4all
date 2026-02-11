"""
Paper TabPFN

Usage:
    python full_run.py              # full configuration (default)
    python full_run.py --smoke      # smoke configuration (quick pipeline test)
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.config.cache_config import CacheConfig
from nirs4all.operators.models.pytorch.nicon import customizable_nicon as nicon
from nirs4all.operators.splitters.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    ExtendedMultiplicativeScatterCorrection as EMSC,
    SavitzkyGolay,
    ASLSBaseline,
    Detrend,
    Gaussian,
    Haar,
    AreaNormalization,
    IdentityTransformer,
    Derivate,
    Baseline,
    Normalize,
    FlexiblePCA,
)
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.visualization.analysis.branch import BranchAnalyzer
from nirs4all.visualization.analysis.branch import BranchSummary

# Third-party model imports
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Run smoke (minimal) configuration")
parser.add_argument("--plots", action="store_true", help="Generate plot files")
parser.add_argument("--show", action="store_true", help="Display plots interactively")
args = parser.parse_args()

# =============================================================================
# Configuration: smoke vs full
# =============================================================================
SMOKE = args.smoke

if SMOKE:
    CFG = dict(
        linear_cartesian_count=2,
        linear_finetune_trials=2,
        linear_refit_max_iter=100,
        ridge_refit_max_iter=500,
        nicon_train_epochs=10,
        nicon_train_patience=5,
        nicon_finetune_trials=2,
        nicon_finetune_epochs=5,
        nicon_finetune_patience=3,
        nicon_refit_epochs=20,
        nicon_refit_patience=10,
        tabular_cartesian_count=2,
        catboost_iterations=10,
        catboost_refit_iterations=20,
        tabpfn_refit_estimators=2,
    )
else:
    CFG = dict(
        linear_cartesian_count=-1,  # all combinations
        linear_finetune_trials=20,
        linear_refit_max_iter=2000,
        ridge_refit_max_iter=10000,
        nicon_train_epochs=500,
        nicon_train_patience=50,
        nicon_finetune_trials=200,
        nicon_finetune_epochs=80,
        nicon_finetune_patience=10,
        nicon_refit_epochs=5000,
        nicon_refit_patience=150,
        tabular_cartesian_count=-1,  # all combinations
        catboost_iterations=80,
        catboost_refit_iterations=800,
        tabpfn_refit_estimators=16,
    )

print(f"Running in {'SMOKE' if SMOKE else 'FULL'} mode")

# =============================================================================
# Pipeline
# =============================================================================

pipeline = [
    SPXYGFold(n_splits=3, random_state=42),
    {"branch": {
        "linear_models": [
            {"y_processing": MinMaxScaler()},
            {
                "_cartesian_": [ # 240 | 729 * 3 folds * 20 trials = 43740 * 2 models = 87480 runs
                    {"_or_": [None, StandardScaler, MinMaxScaler]},
                    {"_or_": [None, ASLSBaseline, Detrend]},
                    {"_or_": [None, SNV, EMSC]},
                    {"_or_": [None, SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
                    {"_or_": [None, Haar(), AreaNormalization()]},
                ],
                "count": CFG["linear_cartesian_count"],
            },
            {
                "model": PLSRegression(),
                "name": "PLS",
                "finetune_params": {
                    "n_trials": CFG["linear_finetune_trials"],
                    "sampler": "tpe",
                    "model_params": {
                        "n_components": ('int', 2, 25),
                    },
                },
                "refit_params": {"max_iter": CFG["linear_refit_max_iter"]},
            },
            {
                "model": Ridge(),
                "name": "Ridge",
                "finetune_params": {
                    "n_trials": CFG["linear_finetune_trials"],
                    "sampler": "tpe",
                    "model_params": {
                        "alpha": ("float_log", 1e-5, 1e4),
                        "fit_intercept": ("bool", [True, False]),
                        "solver": "auto",
                        "max_iter": 1500,
                        "tol": 1e-4,
                        "positive": False,
                    },
                },
                "refit_params": {"max_iter": CFG["ridge_refit_max_iter"], "verbose": 1},
            },
        ],

        "neural_net": [
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": [
                IdentityTransformer(),
                SavitzkyGolay(window_length=17, polyorder=2, deriv=2),
                SavitzkyGolay(window_length=5, polyorder=2),
                Gaussian(order=1, sigma=2),
                Gaussian(order=2, sigma=1),
                Gaussian(order=0, sigma=2),
                SNV(),
                EMSC(scale=False),
                Detrend(),
                Derivate(2, 1),
                Derivate(1, 2),
                Haar(),
            ]},
            {
                "model": nicon,
                "name": "Nicon-CNN",
                "train_params": {
                    "epochs": CFG["nicon_train_epochs"],
                    "patience": CFG["nicon_train_patience"],
                    "batch_size": 1024,
                    "cyclic_lr": True,
                    "cyclic_lr_mode": "triangular2",
                    "base_lr": 0.0005,
                    "max_lr": 0.02,
                    "step_size": 100,
                    "loss": "mse",
                },
                "finetune_params": {
                    "n_trials": CFG["nicon_finetune_trials"],
                    "verbose": 2,
                    "pruner": "hyperband",
                    "approach": "single",
                    "model_params": {
                        'spatial_dropout': (float, 0.01, 0.5),
                        'filters1': [4, 8, 16, 32, 64],
                        'activation1': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
                        'dropout_rate': (float, 0.01, 0.5),
                        'filters2': [4, 8, 16, 32, 64],
                        'activation2': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
                        'filters3': [4, 8, 16, 32, 64],
                        'activation3': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
                        'dense_units': [4, 8, 16, 32, 64],
                        'dense_activation': ['relu', 'selu', 'elu', 'swish', 'gelu', 'linear'],
                    },
                    "train_params": {
                        "epochs": CFG["nicon_finetune_epochs"],
                        "patience": CFG["nicon_finetune_patience"],
                        "learning_rate": 0.005,
                        "verbose": 0
                    }
                },
                "refit_params": {
                    "epochs": CFG["nicon_refit_epochs"],
                    "patience": CFG["nicon_refit_patience"],
                    "batch_size": 1024,
                    "cyclic_lr": True,
                    "cyclic_lr_mode": "triangular2",
                    "base_lr": 0.0005,
                    "max_lr": 0.02,
                    "step_size": 100,
                    "loss": "mse",
                    "verbose": 1,
                    "warm_start": True,
                    "warm_start_fold": 'best'
                },
            },
        ],

        "tabular_models": [
            {"_or_": [None, {"y_processing": StandardScaler()}]},
            {
                "_cartesian_": [
                    {"_or_": [None, StandardScaler]},
                    {"_or_": [None, ASLSBaseline]},
                    {"_or_": [None, SavitzkyGolay, Baseline, SNV, Gaussian, Normalize]},
                    {"_or_": [None, FlexiblePCA(n_components=0.25)]},
                ],
                "count": CFG["tabular_cartesian_count"],
            },
            {
                "model": CatBoostRegressor(iterations=CFG["catboost_iterations"], depth=8, learning_rate=0.1, random_state=42, verbose=0, allow_writing_files=False),
                "name": "CatBoost",
                "refit_params": {"iterations": CFG["catboost_refit_iterations"], "depth": 8, "learning_rate": 0.05, "verbose": 1},
            },
            {
                "model": TabPFNRegressor(model_path="tabpfn-v2.5-regressor-v2.5_real.ckpt", device="cuda"),
                "name": "TabPFN",
                "refit_params": {"n_estimators": CFG["tabpfn_refit_estimators"]}
            },
        ],
    }},
]

DATAPATH = ['data/regression/AMYLOSE/Rice_Amylose_313_YbasedSplit', 'data/regression/CORN/Corn_Moisture_80_WangStyle_m5spec']
result = nirs4all.run(
    pipeline=pipeline,
    dataset=DATAPATH,
    name="TABPFN_Paper",
    verbose=1,
    random_state=42,
    cache=CacheConfig(memory_warning_threshold_mb=16984),
)

# predictions = result.predictions


# # =============================================================================
# # Section 1: BranchAnalyzer for Statistical Comparison
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 1: BranchAnalyzer for Statistical Comparison")
# print("-" * 60)

# print("""
# BranchAnalyzer provides rigorous statistical methods:
# - summary(): Mean, std, min, max for each branch
# - rank_branches(): Statistical ranking by performance
# - compare(): Compare two branches with statistical tests
# - pairwise_comparison(): All-pairs comparison matrix
# """)

# # Create the analyzer
# analyzer = BranchAnalyzer(predictions=predictions)

# # Get summary statistics
# print("\n📊 Branch Summary (RMSE, test partition):")
# summary = analyzer.summary(metrics=['rmse', 'r2'], partition='test')

# # Print as markdown table
# print(summary.to_markdown())


# # =============================================================================
# # Section 2: Branch Ranking
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 2: Branch Ranking")
# print("-" * 60)

# print("""
# rank_branches() orders branches by mean performance.
# Lower RMSE is better, so ascending=True (default for RMSE).
# """)

# # Rank branches by RMSE
# rankings = analyzer.rank_branches(metric='rmse', partition='test')

# print("\n🏆 Branch Rankings (lower RMSE is better):")
# for rank_info in rankings:
#     print(f"  #{rank_info['rank']}: {rank_info['branch_name']} "
#           f"(RMSE: {rank_info['rmse_mean']:.4f} ± {rank_info['rmse_std']:.4f})")


# # =============================================================================
# # Section 3: Pairwise Statistical Comparison
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 3: Pairwise Statistical Comparison")
# print("-" * 60)

# print("""
# Compare two branches statistically using t-test, Wilcoxon, or Mann-Whitney.
# Returns p-value, effect size (Cohen's d), and significance.
# """)

# # Get branch names
# branches = analyzer.get_branch_names()

# if len(branches) >= 2:
#     print(f"\n🔍 Statistical Comparison: {branches[0]} vs {branches[1]}")
#     try:
#         comparison = analyzer.compare(
#             branches[0], branches[1],
#             metric='rmse',
#             partition='test',
#             test='ttest'
#         )
#         print(f"  {branches[0]} mean: {comparison['branch1_mean']:.4f} ± {comparison['branch1_std']:.4f}")
#         print(f"  {branches[1]} mean: {comparison['branch2_mean']:.4f} ± {comparison['branch2_std']:.4f}")
#         print(f"  p-value: {comparison['p_value']:.4f}")
#         print(f"  Significant (α=0.05): {comparison['significant']}")
#         print(f"  Effect size (Cohen's d): {comparison['effect_size']:.4f}")
#     except ImportError:
#         print("  scipy not installed - skipping statistical comparison")
#     except ValueError as e:
#         print(f"  Could not compare: {e}")


# # =============================================================================
# # Section 4: Pairwise Comparison Matrix
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 4: Pairwise Comparison Matrix")
# print("-" * 60)

# print("""
# pairwise_comparison() computes p-values for all branch pairs.
# Returns a DataFrame matrix useful for multiple comparison correction.
# """)

# try:
#     pairwise_matrix = analyzer.pairwise_comparison(
#         metric='rmse',
#         partition='test',
#         test='ttest'
#     )
#     print("\n📊 Pairwise p-value matrix:")
#     print(pairwise_matrix.round(4).to_string())
# except ImportError:
#     print("scipy not installed - skipping pairwise comparison")
# except Exception as e:
#     print(f"Could not compute pairwise comparison: {e}")


# # =============================================================================
# # Section 5: LaTeX Export for Publications
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 5: LaTeX Export for Publications")
# print("-" * 60)

# print("""
# BranchSummary provides export to LaTeX for publication-ready tables.
# The to_latex() method formats mean ± std in math mode.
# """)

# # Generate LaTeX table
# latex_output = summary.to_latex(
#     caption="Branch Performance Comparison",
#     label="tab:branch_comparison",
#     precision=4,
#     mean_std_combined=True
# )
# print("\n📄 LaTeX Output:")
# print(latex_output)


# # =============================================================================
# # Section 6: CSV Export
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 6: CSV Export")
# print("-" * 60)

# # Ensure output directory exists
# Path("reports").mkdir(exist_ok=True)

# # Export to CSV
# summary_full = analyzer.summary(metrics=['rmse', 'r2', 'mae'], partition='test')
# summary_full.to_csv("reports/branch_summary.csv")
# print("📄 Exported branch summary to reports/branch_summary.csv")


# # =============================================================================
# # Section 7: Visualization with PredictionAnalyzer
# # =============================================================================
# print("\n" + "-" * 60)
# print("Example 7: Branch Visualization")
# print("-" * 60)

# print("""
# PredictionAnalyzer provides visual comparison of branches:
# - branch_summary(): Statistics table
# - plot_branch_comparison(): Bar chart with error bars
# - plot_branch_boxplot(): Score distributions
# """)

# viz_analyzer = PredictionAnalyzer(predictions)

# # Get branch summary from PredictionAnalyzer
# branch_df = viz_analyzer.branch_summary()
# print("\n📊 Branch Summary (via PredictionAnalyzer):")
# print(branch_df.to_string(index=False) if hasattr(branch_df, 'to_string') else branch_df)

# if args.plots or args.show:
#     # Create branch comparison chart
#     fig_compare = viz_analyzer.plot_branch_comparison(
#         rank_metric='rmse',
#         display_partition='test'
#     )
#     print("Created branch comparison chart")

#     # Create branch boxplot
#     fig_boxplot = viz_analyzer.plot_branch_boxplot(
#         rank_metric='rmse',
#         display_partition='test'
#     )
#     print("Created branch boxplot")



