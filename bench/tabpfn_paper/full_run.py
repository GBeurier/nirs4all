"""
Paper TabPFN

Usage:
    python full_run.py              # full configuration (default)
    python full_run.py --smoke      # smoke configuration (quick pipeline test)
"""

# Standard library imports
import argparse
from pathlib import Path
import time

# Third-party imports

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from optuna.pruners import SuccessiveHalvingPruner

# NIRS4All imports
import nirs4all
from nirs4all.config.cache_config import CacheConfig
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.pytorch.nicon import customizable_nicon as nicon
from nirs4all.operators.splitters.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    ExtendedMultiplicativeScatterCorrection as EMSC,
    MultiplicativeScatterCorrection as MSC,
    SavitzkyGolay,
    SavitzkyGolay as SG,
    ASLSBaseline,
    Detrend,
    Gaussian,
    ToAbsorbance,
    KubelkaMunk,
    Haar,
    AreaNormalization,
    IdentityTransformer,
    Derivate,
    Baseline,
    Normalize,
    FlexiblePCA,
)
from nirs4all.operators.transforms.nirs import Wavelet
from nirs4all.operators.transforms.orthogonalization import OSC
from nirs4all.operators.transforms.wavelet_denoise import WaveletDenoise
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
        pls_finetune_trials=2,
        ridge_finetune_trials=2,
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
        pls_finetune_trials=25,
        ridge_finetune_trials=60,
        linear_refit_max_iter=2000,
        ridge_refit_max_iter=10000,
        nicon_train_epochs=600,
        nicon_train_patience=250,
        nicon_finetune_trials=50,
        nicon_finetune_epochs=100,
        nicon_finetune_patience=50,
        nicon_refit_epochs=300,
        # nicon_refit_patience=500,
        tabular_cartesian_count=-1,  # all combinations
        catboost_iterations=150,
        catboost_refit_iterations=600,
        tabpfn_refit_estimators=20,
    )

print(f"Running in {'SMOKE' if SMOKE else 'FULL'} mode")

# =============================================================================
# Pipeline
# =============================================================================

pipeline = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": {
            "linear_models": [
                # {
                #     "_cartesian_": [
                #         # {"_or_": [None, KubelkaMunk]},
                #         {"_or_": [None, SNV, MSC, EMSC(degree=1), EMSC(degree=2)]},
                #         # {"_or_": [None,
                #         #         SG(window_length=11, polyorder=2, deriv=1), SG(15,2,1), SG(21,2,1), SG(31,2,1), SG(15,3,2), SG(21,3,2), SG(31,3,2),
                #         #         # Gaussian(order=0, sigma=1), Gaussian(order=0, sigma=2),
                #         #         # WaveletDenoise('db4', level=3), WaveletDenoise('db4', level=5)
                #         #         ]},
                #         {"_or_": [None, ASLSBaseline, Detrend]},
                #         {"_or_": [None, OSC(1), OSC(2), OSC(3)]},

                #     ],
                #     "count": CFG["linear_cartesian_count"],
                # },
                # StandardScaler(with_mean=True, with_std=False),
                # {
                #     "model": PLSRegression(scale=False),
                #     "name": "PLS",
                #     "finetune_params": {
                #         "n_trials": CFG["pls_finetune_trials"],
                #         "sampler": "binary",
                #         # "pruner": 'successive_halving',
                #         # "n_jobs": 20,
                #         "model_params": {
                #             "n_components": ('int', 1, 25),
                #         },
                #     },
                # },
                # {
                #     "model": Ridge(),
                #     "name": "Ridge",
                #     "finetune_params": {
                #         "n_trials": CFG["ridge_finetune_trials"],
                #         "sampler": "tpe",
                #         "model_params": {
                #             "alpha": ("float_log", 1e-5, 1e4),
                #             "fit_intercept": ("bool", [True, False]),
                #             "solver": ("categorical", ["auto", "svd", "cholesky", "lsqr"]),  # Removed sag, saga, sparse_cg
                #             # "solver": ("categorical", ["auto", "svd", "cholesky", "lsqr", "sag", "saga", "sparse_cg"]),
                #             # "solver": "auto",
                #             # "max_iter": 1500,
                #             "tol": 1e-4,
                #             "positive": False,
                #         },
                #     },
                #     # "refit_params": {"max_iter": CFG["ridge_refit_max_iter"], "verbose": 1},
                # },
            ],

            "neural_net": [
                # {"y_processing": MinMaxScaler()},
                # {"feature_augmentation": [
                #     IdentityTransformer(),
                #     SNV(),
                #     EMSC(),
                #     SavitzkyGolay(window_length=15, deriv=0),
                #     SavitzkyGolay(deriv=1),
                #     SavitzkyGolay(deriv=2),
                #     [SNV(), SavitzkyGolay(window_length=11, deriv=0)],
                #     Derivate(1, 2),
                #     [SNV(), Haar()],
                # ]},
                # StandardScaler(with_std=True),
                # {
                #     "model": nicon,
                #     "name": "Nicon-CNN",
                #     "train_params": {
                #         "epochs": CFG["nicon_train_epochs"],
                #         "patience": CFG["nicon_train_patience"],
                #         # "batch_size": 1024,
                #         "cyclic_lr": True,
                #         "cyclic_lr_mode": "triangular2",
                #         "base_lr": 0.0005,
                #         "max_lr": 0.01,
                #         "step_size": 200,
                #         "loss": "mse",
                #     },
                #     "finetune_params": {
                #         "n_trials": CFG["nicon_finetune_trials"],
                #         "verbose": 2,
                #         "pruner": "hyperband",
                #         "sampler": "tpe",
                #         "approach": "grouped",
                #         "model_params": {
                #             'spatial_dropout': (float, 0.01, 0.5),
                #             'filters1': [4, 8, 16, 32],
                #             'dropout_rate': (float, 0.01, 0.5),
                #             'filters2': [32, 64, 128, 256],
                #             'filters3': [8, 16, 32, 64],
                #             'dense_units': [8, 16, 32, 64],
                #         },
                #         "train_params": {
                #             "epochs": CFG["nicon_finetune_epochs"],
                #             "patience": CFG["nicon_finetune_patience"],
                #             "learning_rate": 0.005,
                #             "verbose": 0
                #         }
                #     },
                #     "refit_params": {
                #         "epochs": CFG["nicon_refit_epochs"],
                #         # "patience": CFG["nicon_refit_patience"],
                #         # "batch_size": 1024,
                #         # "cyclic_lr": True,
                #         # "cyclic_lr_mode": "triangular2",
                #         "base_lr": 0.0005,
                #         # "max_lr": 0.02,
                #         # "step_size": 100,
                #         "loss": "mse",
                #         # "verbose": 1,
                #         # "warm_start": True,
                #         # "warm_start_fold": 'best'
                #     },
                # },
            ],

            "tabular_models": [
                # {"_or_": [None, {"y_processing": StandardScaler()}]},
                {
                    "_cartesian_": [
                        # {"_or_": [Baseline, , Gaussian, Normalize, AreaNormalization, KubelkaMunk, Haar]},
                        {"_or_":[None, SNV]},
                        # {"_or_": [None, SG(window_length=11, polyorder=2, deriv=1), SG(15,2,1), SG(21,2,1), SG(31,2,1), SG(15,3,2), SG(21,3,2), SG(31,3,2)]},
                        {"_or_": [None, OSC, Detrend]}, #ASLSBaseline, StandardScaler(with_mean=True, with_std=False), Detrend]},
                        # {"_or_": [ASLSBaseline, Detrend]},
                        # {"_or_": [OSC(1), OSC(2), OSC(3)]},
                        # {"_or_": [None, FlexiblePCA(n_components=0.25)]},
                    ],
                    "count": CFG["tabular_cartesian_count"],
                },
                # StandardScaler(with_mean=False, with_std=True),

                # {
                #     "model": CatBoostRegressor(iterations=CFG["catboost_iterations"], depth=8, learning_rate=0.1, random_state=42, verbose=0, allow_writing_files=False),
                #     "name": "CatBoost",
                #     "refit_params": {"iterations": CFG["catboost_refit_iterations"], "depth": 8, "learning_rate": 0.05, "verbose": 1},
                # },
                {
                    "model": TabPFNRegressor(ignore_pretraining_limits=True, model_path="tabpfn-v2.5-regressor-v2.5_real.ckpt", device="cuda"),
                    "name": "TabPFN",
                    "refit_params": {"n_estimators": CFG["tabpfn_refit_estimators"], "ignore_pretraining_limits":True}
                },
            ],
            # Parallel execution configuration
            "parallel": True,  # Enable/disable parallel execution
            "n_jobs": -1,  # Number of parallel workers (-1=auto)
        }
    },
]

start_time = time.time()
#PLS > 142sec: run 0 bch -1 opt 0, 130sec: 0 12 12, 142sec: 20 - 4, 136: 0 4 12, 60: -1 0 0, 72: 12 0 12, 150: 4 0 20
#PFN > 149sec 0 0 GPU, 170 sec 0 -1 CPU, 172 sec 0 -1 GPU, sec -1 0 CPU, 133 sec -1 0 GPU
DATASETS = [
            'AMYLOSE/Rice_Amylose_313_YbasedSplit', # PLS 1.85 - TabPFN: 1.31 +
            'IncombustibleMaterial/TIC_spxy70', # PLS 3.03 - TabPFN: 3.14 *
            'PHOSPHORUS/LP_spxyG', # PLS 0.17 - TabPFN: 0.14 *
            'PLUMS/Firmness_spxy70', # PLS 0.29 - TabPFN: 0.26 *
            'BEER/Beer_OriginalExtract_60_KS', # PLS 0.20 - TabPFN: 0.08 +
            'COLZA/N_woOutlier', # PLS 0.25- TabPFN: 0.18
            'MILK/Milk_Lactose_1224_KS', # PLS 0.0541 - TabPFN: 0.053
            'DIESEL/DIESEL_bp50_246_hlb-a', # PLS 3.22 - TabPFN: 3.3 *
            'WOOD_density/WOOD_N_402_Olale', # PLS 0.050 - TabPFN: 0.048
            #
            'MANURE21/All_manure_Total_N_SPXY_strat_Manure_type', 'COLZA/N_wOutlier', 'DarkResp/Rd25_CBtestSite', 'GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD', 'PHOSPHORUS/MP_spxyG', 'ALPINE/ALPINE_P_291_KS', 'BERRY/ta_groupSampleID_stratDateVar_balRows', 'FUSARIUM/Tleaf_grp70_30', 'PHOSPHORUS/V25_spxyG', 'GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit', 'DarkResp/Rd25_spxy70', 'COLZA/C_woOutlier', 'ALPINE/ALPINE_C_424_KS', 'FUSARIUM/Fv_Fm_grp70_30', 'MILK/Milk_Fat_1224_KS', 'MANURE21/All_manure_MgO_SPXY_strat_Manure_type', 'FUSARIUM/FinalScore_grp70_30_scoreQ', 'MANURE21/All_manure_K2O_SPXY_strat_Manure_type', 'QUARTZ/Quartz_spxy70', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'ALPINE/ALPINE_N_552_KS', 'LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS', 'MALARIA/Malaria_Sporozoite_229_Maia', 'MALARIA/Malaria_Oocist_333_Maia', 'BISCUIT/Biscuit_Fat_40_RandomSplit', 'TABLET/Escitalopramt_310_Zhao', 'PHOSPHORUS/Pi_spxyG', 'ECOSIS_LeafTraits/Chla+b_spxyG_species', 'ECOSIS_LeafTraits/Ccar_spxyG_block2deg', 'MILK/Milk_Urea_1224_KS', 'BISCUIT/Biscuit_Sucrose_40_RandomSplit', 'DIESEL/DIESEL_bp50_246_hla-b', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD', 'BERRY/brix_groupSampleID_stratDateVar_balRows', 'LUCAS/LUCAS_SOC_all_26650_NocitaKS', 'BERRY/ph_groupSampleID_stratDateVar_balRows', 'BEEFMARBLING/Beef_Marbling_RandomSplit', 'GRAPEVINES/grapevine_chloride_556_KS', 'MANURE21/All_manure_CaO_SPXY_strat_Manure_type', 'CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit', 'DIESEL/DIESEL_bp50_246_b-a', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR', 'PLUMS/Brix_spxy70', 'PEACH/Brix_spxy70', 'ECOSIS_LeafTraits/LMA_spxyG_block2deg', 'ALPINE/ALPINE_C_424_RobustnessAlps', 'WOOD_density/WOOD_Density_402_Olale', 'LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic', 'BEER/Beer_OriginalExtract_60_YbaseSplit', 'DarkResp/Rd25_GTtestSite', 'MANURE21/All_manure_P2O5_SPXY_strat_Manure_type', 'ECOSIS_LeafTraits/Chla+b_spxyG_block2deg', 'DarkResp/Rd25_XSBNtestSite', 'PHOSPHORUS/NP_spxyG'
            ]

DATASETS_PATHS = [f"data/regression/{ds}" for ds in DATASETS]
dataset_config = DatasetConfigs( DATASETS_PATHS, task_type="regression")
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset_config,
    name="TABPFN_Paper",
    verbose=1,
    random_state=42,
    # n_jobs=-1,
    cache=CacheConfig(memory_warning_threshold_mb=16984),
)

duration = time.time() - start_time
print(f"\nPipeline completed in {duration:.2f} seconds")



# # =============================================================================
# # Section 1: BranchAnalyzer for Statistical Comparison
# # =============================================================================
# predictions = result.predictions


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



