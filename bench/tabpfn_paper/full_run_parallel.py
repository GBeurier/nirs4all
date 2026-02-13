"""
Paper TabPFN - Parallel Execution Version

This version demonstrates how to use parallel branch execution to speed up
the pipeline by running branches concurrently.

Key optimizations:
1. Parallel branch execution for independent model families
2. Parallel Optuna optimization within branches (when safe)
3. Smart detection to avoid nested parallelization conflicts

Usage:
    python full_run_parallel.py              # full configuration (default)
    python full_run_parallel.py --smoke      # smoke configuration (quick pipeline test)
"""

# Standard library imports
import argparse
from pathlib import Path
import time

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
parser.add_argument("--parallel-branches", type=int, default=2, help="Number of parallel branch workers (0=sequential, -1=auto)")
parser.add_argument("--parallel-optuna", type=int, default=1, help="Number of parallel Optuna workers per branch")
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
        tabular_cartesian_count=-1,  # all combinations
        catboost_iterations=150,
        catboost_refit_iterations=600,
        tabpfn_refit_estimators=20,
    )

print(f"Running in {'SMOKE' if SMOKE else 'FULL'} mode")
print(f"Parallel branches: {args.parallel_branches} (-1=auto, 0=sequential)")
print(f"Parallel Optuna: {args.parallel_optuna} workers per branch")

# =============================================================================
# Pipeline with Parallel Execution
# =============================================================================

# PARALLELIZATION STRATEGY:
# 1. Linear models: Can run in parallel (no internal parallelization)
# 2. Neural networks: MUST run sequentially (GPU/memory intensive)
# 3. Tabular models: Can run in parallel but with care (memory)

# Determine parallel settings based on branch content
use_parallel_branches = args.parallel_branches != 0
n_jobs_branches = args.parallel_branches if args.parallel_branches > 0 else -1
n_jobs_optuna = args.parallel_optuna

# Note: If n_jobs_optuna > 1, we should NOT parallelize branches to avoid
# nested parallelization conflicts
if n_jobs_optuna > 1 and use_parallel_branches:
    print("⚠️  Warning: Disabling branch parallelization due to Optuna n_jobs > 1")
    print("   (to avoid nested parallelization conflicts)")
    use_parallel_branches = False
    n_jobs_branches = 1

pipeline = [
    SPXYGFold(n_splits=3, random_state=42),
    {
        "branch": {
            "linear_models": [
                {
                    "_cartesian_": [
                        {"_or_": [None, SNV, MSC, EMSC(degree=1), EMSC(degree=2)]},
                        {"_or_": [None,
                                  SG(window_length=11, polyorder=2, deriv=1), SG(15,2,1), SG(21,2,1), SG(31,2,1), SG(15,3,2), SG(21,3,2), SG(31,3,2),
                                  Gaussian(order=0, sigma=1), Gaussian(order=0, sigma=2),
                                  WaveletDenoise('db4', level=3), WaveletDenoise('db4', level=5)
                                 ]},
                        {"_or_": [None, ASLSBaseline, Detrend]},
                        {"_or_": [None, OSC(1), OSC(2), OSC(3)]},
                    ],
                    "count": CFG["linear_cartesian_count"],
                },
                StandardScaler(with_mean=True, with_std=False),
                {
                    "model": PLSRegression(scale=False),
                    "name": "PLS",
                    "finetune_params": {
                        "n_trials": CFG["pls_finetune_trials"],
                        "sampler": "tpe",
                        "n_jobs": n_jobs_optuna,  # Parallel Optuna optimization
                        "model_params": {
                            "n_components": ('int', 1, 25),
                        },
                    },
                },
            ],

            "neural_net": [
                # Neural networks are commented out in original - keeping same
                # Note: These would NOT be parallelized due to GPU detection
            ],

            "tabular_models": [
                # Tabular models are commented out in original - keeping same
                # Note: These have internal parallelization so would disable branch parallel
            ],
        },
        # Parallel execution configuration
        "parallel": use_parallel_branches,  # Enable/disable parallel execution
        "n_jobs": n_jobs_branches,  # Number of parallel workers (-1=auto)
    },
]

start_time = time.time()

DATASETS = ['AMYLOSE/Rice_Amylose_313_YbasedSplit', 'IncombustibleMaterial/TIC_spxy70', 'PHOSPHORUS/LP_spxyG', 'PLUMS/Firmness_spxy70', 'BEER/Beer_OriginalExtract_60_KS', 'COLZA/N_woOutlier', 'MILK/Milk_Lactose_1224_KS', 'DIESEL/DIESEL_bp50_246_hlb-a', 'WOOD_density/WOOD_N_402_Olale', 'MANURE21/All_manure_Total_N_SPXY_strat_Manure_type', 'COLZA/N_wOutlier', 'DarkResp/Rd25_CBtestSite', 'GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD', 'PHOSPHORUS/MP_spxyG', 'ALPINE/ALPINE_P_291_KS', 'BERRY/ta_groupSampleID_stratDateVar_balRows', 'FUSARIUM/Tleaf_grp70_30', 'PHOSPHORUS/V25_spxyG', 'GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit', 'DarkResp/Rd25_spxy70', 'COLZA/C_woOutlier', 'ALPINE/ALPINE_C_424_KS', 'FUSARIUM/Fv_Fm_grp70_30', 'MILK/Milk_Fat_1224_KS', 'MANURE21/All_manure_MgO_SPXY_strat_Manure_type', 'FUSARIUM/FinalScore_grp70_30_scoreQ', 'MANURE21/All_manure_K2O_SPXY_strat_Manure_type', 'QUARTZ/Quartz_spxy70', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'ALPINE/ALPINE_N_552_KS', 'LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS', 'MALARIA/Malaria_Sporozoite_229_Maia', 'MALARIA/Malaria_Oocist_333_Maia', 'BISCUIT/Biscuit_Fat_40_RandomSplit', 'TABLET/Escitalopramt_310_Zhao', 'PHOSPHORUS/Pi_spxyG', 'ECOSIS_LeafTraits/Chla+b_spxyG_species', 'ECOSIS_LeafTraits/Ccar_spxyG_block2deg', 'MILK/Milk_Urea_1224_KS', 'BISCUIT/Biscuit_Sucrose_40_RandomSplit', 'DIESEL/DIESEL_bp50_246_hla-b', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD', 'BERRY/brix_groupSampleID_stratDateVar_balRows', 'LUCAS/LUCAS_SOC_all_26650_NocitaKS', 'BERRY/ph_groupSampleID_stratDateVar_balRows', 'BEEFMARBLING/Beef_Marbling_RandomSplit', 'GRAPEVINES/grapevine_chloride_556_KS', 'MANURE21/All_manure_CaO_SPXY_strat_Manure_type', 'CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit', 'DIESEL/DIESEL_bp50_246_b-a', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR', 'PLUMS/Brix_spxy70', 'PEACH/Brix_spxy70', 'ECOSIS_LeafTraits/LMA_spxyG_block2deg', 'ALPINE/ALPINE_C_424_RobustnessAlps', 'WOOD_density/WOOD_Density_402_Olale', 'LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic', 'BEER/Beer_OriginalExtract_60_YbaseSplit', 'DarkResp/Rd25_GTtestSite', 'MANURE21/All_manure_P2O5_SPXY_strat_Manure_type', 'ECOSIS_LeafTraits/Chla+b_spxyG_block2deg', 'DarkResp/Rd25_XSBNtestSite', 'PHOSPHORUS/NP_spxyG']
DATASETS_PATHS = [f"data/regression/{ds}" for ds in DATASETS]
dataset_config = DatasetConfigs(DATASETS_PATHS, task_type="regression")
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset_config,
    name="TABPFN_Paper_Parallel",
    verbose=1,
    random_state=42,
    cache=CacheConfig(memory_warning_threshold_mb=16984),
)

duration = time.time() - start_time
print(f"\nPipeline completed in {duration:.2f} seconds")

# Print performance comparison if available
if hasattr(result, 'execution_stats'):
    print(f"\nExecution Statistics:")
    print(f"  Parallel branches: {use_parallel_branches}")
    if use_parallel_branches:
        print(f"  Workers used: {n_jobs_branches if n_jobs_branches > 0 else 'auto'}")
    print(f"  Parallel Optuna: {n_jobs_optuna} workers per branch")
