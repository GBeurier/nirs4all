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
from nirs4all.operators.splitters.splitters import SPXYFold, SPXYGFold
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
    SPXYFold(n_splits=3, random_state=42),
    {
        "_cartesian_": [
            ## DISCARDED
            # {"_or_": [Baseline, , Gaussian, Normalize, AreaNormalization, KubelkaMunk, Haar]},


            {"_or_": [None, SNV, MSC]},
            {"_or_": [None, SG(window_length=11, polyorder=2, deriv=1), SG(15,2,1), SG(21,2,1), SG(31,2,1), SG(15,3,2), SG(21,3,2), SG(31,3,2), SG(15,2,0)]},
            {"_or_": [None, OSC, Detrend]},
            {"_or_": [None, FlexiblePCA(n_components=0.25)]},

            ## DISCARDED
            #ASLSBaseline, StandardScaler(with_mean=True, with_std=False), Detrend]},
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
]

start_time = time.time()
#PLS > 142sec: run 0 bch -1 opt 0, 130sec: 0 12 12, 142sec: 20 - 4, 136: 0 4 12, 60: -1 0 0, 72: 12 0 12, 150: 4 0 20
#PFN > 0 0 GPU
DATASETS = [
            'AMYLOSE/Rice_Amylose_313_YbasedSplit', # PLS 1.85 - TabPFN: 1.31 +
            'IncombustibleMaterial/TIC_spxy70', # PLS 3.31 - TabPFN: 3.14 *
            'PHOSPHORUS/LP_spxyG', # PLS 0.17 - TabPFN: 0.14 *
            'PLUMS/Firmness_spxy70', # PLS 0.29 - TabPFN: 0.26 *
            'BEER/Beer_OriginalExtract_60_KS', # PLS 0.20 - TabPFN: 0.08 +
            'COLZA/N_woOutlier', # PLS 0.25- TabPFN: 0.18
            'MILK/Milk_Lactose_1224_KS', # PLS 0.0541 - TabPFN: 0.053
            'DIESEL/DIESEL_bp50_246_hlb-a', # PLS 3.22 - TabPFN: 3.3 *
            'WOOD_density/WOOD_N_402_Olale', # PLS 0.050 - TabPFN: 0.048
            'MANURE21/All_manure_Total_N_SPXY_strat_Manure_type',
            'COLZA/N_wOutlier',
            'DarkResp/Rd25_CBtestSite',
            #
            'GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD', 'PHOSPHORUS/MP_spxyG', 'ALPINE/ALPINE_P_291_KS', 'BERRY/ta_groupSampleID_stratDateVar_balRows', 'FUSARIUM/Tleaf_grp70_30', 'PHOSPHORUS/V25_spxyG', 'GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit', 'DarkResp/Rd25_spxy70', 'COLZA/C_woOutlier', 'ALPINE/ALPINE_C_424_KS', 'FUSARIUM/Fv_Fm_grp70_30', 'MILK/Milk_Fat_1224_KS', 'MANURE21/All_manure_MgO_SPXY_strat_Manure_type', 'FUSARIUM/FinalScore_grp70_30_scoreQ', 'MANURE21/All_manure_K2O_SPXY_strat_Manure_type', 'QUARTZ/Quartz_spxy70', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'ALPINE/ALPINE_N_552_KS', 'LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS', 'MALARIA/Malaria_Sporozoite_229_Maia', 'MALARIA/Malaria_Oocist_333_Maia', 'BISCUIT/Biscuit_Fat_40_RandomSplit', 'TABLET/Escitalopramt_310_Zhao', 'PHOSPHORUS/Pi_spxyG', 'ECOSIS_LeafTraits/Chla+b_spxyG_species', 'ECOSIS_LeafTraits/Ccar_spxyG_block2deg', 'MILK/Milk_Urea_1224_KS', 'BISCUIT/Biscuit_Sucrose_40_RandomSplit', 'DIESEL/DIESEL_bp50_246_hla-b', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD', 'BERRY/brix_groupSampleID_stratDateVar_balRows', 'LUCAS/LUCAS_SOC_all_26650_NocitaKS', 'BERRY/ph_groupSampleID_stratDateVar_balRows', 'BEEFMARBLING/Beef_Marbling_RandomSplit', 'GRAPEVINES/grapevine_chloride_556_KS', 'MANURE21/All_manure_CaO_SPXY_strat_Manure_type', 'CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit', 'DIESEL/DIESEL_bp50_246_b-a', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR', 'PLUMS/Brix_spxy70', 'PEACH/Brix_spxy70', 'ECOSIS_LeafTraits/LMA_spxyG_block2deg', 'ALPINE/ALPINE_C_424_RobustnessAlps', 'WOOD_density/WOOD_Density_402_Olale', 'LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic', 'BEER/Beer_OriginalExtract_60_YbaseSplit', 'DarkResp/Rd25_GTtestSite', 'MANURE21/All_manure_P2O5_SPXY_strat_Manure_type', 'ECOSIS_LeafTraits/Chla+b_spxyG_block2deg', 'DarkResp/Rd25_XSBNtestSite', 'PHOSPHORUS/NP_spxyG'
            ]

DATASETS_PATHS = [f"data/regression/{ds}" for ds in DATASETS]
dataset_config = DatasetConfigs( DATASETS_PATHS, task_type="regression")
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset_config,
    name="TABPFN_Paper",
    verbose=0,
    random_state=42,
    # n_jobs=10,
    cache=CacheConfig(memory_warning_threshold_mb=16984),
    refit=[
        {"top_k": 3, "ranking": "rmsecv"},
        {"top_k": 3, "ranking": "mean_val"},
    ]
)

duration = time.time() - start_time
print(f"\nPipeline completed in {duration:.2f} seconds")
