
"""
Paper TabPFN

Usage:
    python full_run.py              # full configuration (default)
    python full_run.py --smoke      # smoke configuration (quick pipeline test)
"""

# Standard library imports
import argparse
import time

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.config.cache_config import CacheConfig
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models import AOMPLSRegressor, POPPLSRegressor
from nirs4all.operators.splitters.splitters import SPXYFold
from nirs4all.operators.transforms import (
    AreaNormalization,
    ASLSBaseline,
    Baseline,
    Derivate,
    Detrend,
    FlexiblePCA,
    Gaussian,
    Haar,
    IdentityTransformer,
    KubelkaMunk,
    Normalize,
    SavitzkyGolay,
    ToAbsorbance,
)
from nirs4all.operators.transforms import (
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.operators.transforms import (
    MultiplicativeScatterCorrection as MSC,
)
from nirs4all.operators.transforms import (
    SavitzkyGolay as SG,
)
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Run smoke (minimal) configuration")
parser.add_argument("--plots", action="store_true", help="Generate plot files")
parser.add_argument("--show", action="store_true", help="Display plots interactively")
args = parser.parse_args()

from oom import OOMPLSRegressorTorch

# =============================================================================
# Configuration: smoke vs full
# =============================================================================
SMOKE = args.smoke

CFG = {"aom_finetune_trials": 2} if SMOKE else {"aom_finetune_trials": 100}

print(f"Running in {'SMOKE' if SMOKE else 'FULL'} mode")

# =============================================================================
# Pipeline
# =============================================================================

pipeline = [
    SPXYFold(n_splits=3, random_state=42),
    # {
        # "_cartesian_": [
            # {"_or_": [None, SNV, MSC, EMSC(degree=1), EMSC(degree=2)]},
            # {"_or_": [None, SG(11, 2, 1), SG(21, 2, 1)]},
        # ],
    # },
    # {"_or_": [None, SNV, MSC, EMSC(degree=1), EMSC(degree=2)]},
    {"y_processing": StandardScaler()},
    # {"_or_": [None, SNV, ASLSBaseline(lam=1e5, p=0.01)]},
    {
        "model": AOMPLSRegressor(center=True, scale=False),
        "name": "AOM-PLS",
        "finetune_params": {
            "n_trials": 200,
            "sampler": "tpe",
            "model_params": {
                "n_components": ('int', 1, 27),
                "n_orth": ('int', 0, 5),
                "operator_index": ('int', 0, 120),
            },
        },
    },
    # StandardScaler(),
    # POPPLSRegressor(),
    # {
    #     "model": OOMPLSRegressorTorch(),
    #     "name": "OOM-PLS",
    #     "finetune_params": {
    #         "n_trials": 50,
    #         "sampler": "tpe",
    #         "model_params": {
    #             # main bias/variance knob (same as PLS)
    #             "n_components": ("int", 1, 30),

    #             # sparsity / regularization knobs (should be float, not int)
    #             # use log-scale if your tuner supports it; otherwise uniform is ok
    #             "tau_kernel": ("float", 0.05, 2.0),   # smaller -> fewer kernels
    #             "tau_block":  ("float", 0.05, 2.0),   # smaller -> fewer blocks

    #             # capacity/speed knobs (pruning)
    #             "top_m_kernels": ("int", 4, 16),      # e.g. 4/8/12/16
    #             "top_m_blocks":  ("int", 4, 24),      # depends on bank size

    #             # optional discrete knobs (encode as int + map inside the model)
    #             # degrees_mode: 0->(0,), 1->(0,1), 2->(0,1,2)
    #             "degrees_mode": ("int", 0, 2),

    #             # families_mode: 0->("SG",), 1->("NW",), 2->("SG","NW")
    #             "families_mode": ("int", 0, 2),
    #         },
    #     },
    # }
]

start_time = time.time()

# DATASETS = [
#             'AMYLOSE/Rice_Amylose_313_YbasedSplit', # PLS 1.85 - TabPFN: 1.31 - AOM: 1.82 - POP: 1.82
#             'IncombustibleMaterial/TIC_spxy70', # PLS 3.31 - TabPFN: 3.14 - AOM: 3.13 - POP: 3.25
#             'PHOSPHORUS/LP_spxyG', # PLS 0.17 - TabPFN: 0.14 - AOM: 0.18 - POP: 0.17
#             'PLUMS/Firmness_spxy70', # PLS 0.29 - TabPFN: 0.26 - AOM: 0.21 - POP: 0.26
#             # 'BEER/Beer_OriginalExtract_60_KS', # PLS 0.20 - TabPFN: 0.08 - AOM: 0.19 - POP: 0.21
#             # 'COLZA/N_woOutlier', # PLS 0.25- TabPFN: 0.18 - AOM: 0.33 - POP: 0.28
#             # 'MILK/Milk_Lactose_1224_KS', # PLS 0.056 - TabPFN: 0.053 - AOM: 0.056 - POP: 0.058
#             # 'DIESEL/DIESEL_bp50_246_hlb-a', # PLS 3.02 - TabPFN: 3.3 - AOM: 3.7 - POP: 3.05
#             # 'WOOD_density/WOOD_N_402_Olale', # PLS 0.050 - TabPFN: 0.048 - AOM: 0.05 - POP: 0.049
#             # 'MANURE21/All_manure_Total_N_SPXY_strat_Manure_type', 'COLZA/N_wOutlier', 'DarkResp/Rd25_CBtestSite', 'GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD', 'PHOSPHORUS/MP_spxyG', 'ALPINE/ALPINE_P_291_KS', 'BERRY/ta_groupSampleID_stratDateVar_balRows', 'FUSARIUM/Tleaf_grp70_30', 'PHOSPHORUS/V25_spxyG', 'GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit', 'DarkResp/Rd25_spxy70', 'COLZA/C_woOutlier', 'ALPINE/ALPINE_C_424_KS', 'FUSARIUM/Fv_Fm_grp70_30', 'MILK/Milk_Fat_1224_KS', 'MANURE21/All_manure_MgO_SPXY_strat_Manure_type', 'FUSARIUM/FinalScore_grp70_30_scoreQ', 'MANURE21/All_manure_K2O_SPXY_strat_Manure_type', 'QUARTZ/Quartz_spxy70', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra', 'ALPINE/ALPINE_N_552_KS', 'LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS', 'MALARIA/Malaria_Sporozoite_229_Maia', 'MALARIA/Malaria_Oocist_333_Maia', 'BISCUIT/Biscuit_Fat_40_RandomSplit', 'TABLET/Escitalopramt_310_Zhao', 'PHOSPHORUS/Pi_spxyG', 'ECOSIS_LeafTraits/Chla+b_spxyG_species', 'ECOSIS_LeafTraits/Ccar_spxyG_block2deg', 'MILK/Milk_Urea_1224_KS', 'BISCUIT/Biscuit_Sucrose_40_RandomSplit', 'DIESEL/DIESEL_bp50_246_hla-b', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD', 'BERRY/brix_groupSampleID_stratDateVar_balRows', 'LUCAS/LUCAS_SOC_all_26650_NocitaKS', 'BERRY/ph_groupSampleID_stratDateVar_balRows', 'BEEFMARBLING/Beef_Marbling_RandomSplit', 'GRAPEVINES/grapevine_chloride_556_KS', 'MANURE21/All_manure_CaO_SPXY_strat_Manure_type', 'CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit', 'DIESEL/DIESEL_bp50_246_b-a', 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR', 'PLUMS/Brix_spxy70', 'PEACH/Brix_spxy70', 'ECOSIS_LeafTraits/LMA_spxyG_block2deg', 'ALPINE/ALPINE_C_424_RobustnessAlps', 'WOOD_density/WOOD_Density_402_Olale', 'LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic', 'BEER/Beer_OriginalExtract_60_YbaseSplit', 'DarkResp/Rd25_GTtestSite', 'MANURE21/All_manure_P2O5_SPXY_strat_Manure_type', 'ECOSIS_LeafTraits/Chla+b_spxyG_block2deg', 'DarkResp/Rd25_XSBNtestSite', 'PHOSPHORUS/NP_spxyG'
#             ]

DATASETS = [
            # 'ALPINE/ALPINE_C_424_KS', # NAN values
            # 'ALPINE/ALPINE_C_424_RobustnessAlps',  # CRASH
            # 'FUSARIUM/Tleaf_grp70_30', # NAN values
            'LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS', # SKIPPED TOO BIG
            'LUCAS/LUCAS_SOC_all_26650_NocitaKS', # SKIPPED TOO BIG
            # 'ALPINE/ALPINE_N_552_KS', # CRASH
            # 'ALPINE/ALPINE_P_291_KS',
            # 'AMYLOSE/Rice_Amylose_313_YbasedSplit', # PLS 1.85 - TabPFN: 1.31 +
            # 'BEEFMARBLING/Beef_Marbling_RandomSplit',
            # 'BEER/Beer_OriginalExtract_60_KS', # PLS 0.20 - TabPFN: 0.08 +
            # 'BEER/Beer_OriginalExtract_60_YbaseSplit',
            # 'BERRY/brix_groupSampleID_stratDateVar_balRows',
            # 'BERRY/ph_groupSampleID_stratDateVar_balRows',
            # 'BERRY/ta_groupSampleID_stratDateVar_balRows',
            # 'BISCUIT/Biscuit_Fat_40_RandomSplit',
            # 'BISCUIT/Biscuit_Sucrose_40_RandomSplit',
            # 'COLZA/C_woOutlier',
            # 'COLZA/N_wOutlier',
            # 'COLZA/N_woOutlier', # PLS 0.25- TabPFN: 0.18
            # 'CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit',
            # 'CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit',
            # 'DIESEL/DIESEL_bp50_246_b-a',
            # 'DIESEL/DIESEL_bp50_246_hla-b',
            # 'DIESEL/DIESEL_bp50_246_hlb-a', # PLS 3.22 - TabPFN: 3.3 *
            # 'DarkResp/Rd25_CBtestSite',
            # 'DarkResp/Rd25_GTtestSite',
            # 'DarkResp/Rd25_XSBNtestSite',
            # 'DarkResp/Rd25_spxy70',
            # 'ECOSIS_LeafTraits/Ccar_spxyG_block2deg',
            # 'ECOSIS_LeafTraits/Chla+b_spxyG_block2deg',
            # 'ECOSIS_LeafTraits/Chla+b_spxyG_species',
            # 'ECOSIS_LeafTraits/LMA_spxyG_block2deg',
            # 'FUSARIUM/FinalScore_grp70_30_scoreQ',
            # 'FUSARIUM/Fv_Fm_grp70_30',
            # 'GRAPEVINES/grapevine_chloride_556_KS',
            # 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD',
            # 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR',
            # 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra',
            # 'GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra',
            # 'GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD',
            # 'GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra',
            # 'IncombustibleMaterial/TIC_spxy70', # PLS 3.31 - TabPFN: 3.14 *
            # 'LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic',
            # 'MALARIA/Malaria_Oocist_333_Maia',
            # 'MALARIA/Malaria_Sporozoite_229_Maia',
            # 'MANURE21/All_manure_CaO_SPXY_strat_Manure_type',
            # 'MANURE21/All_manure_K2O_SPXY_strat_Manure_type',
            # 'MANURE21/All_manure_MgO_SPXY_strat_Manure_type',
            # 'MANURE21/All_manure_P2O5_SPXY_strat_Manure_type',
            # 'MANURE21/All_manure_Total_N_SPXY_strat_Manure_type',
            # 'MILK/Milk_Fat_1224_KS',
            # 'MILK/Milk_Lactose_1224_KS', # PLS 0.0541 - TabPFN: 0.053
            # 'MILK/Milk_Urea_1224_KS',
            # 'PEACH/Brix_spxy70',
            # 'PHOSPHORUS/LP_spxyG', # PLS 0.17 - TabPFN: 0.14 *
            # 'PHOSPHORUS/MP_spxyG',
            # 'PHOSPHORUS/NP_spxyG',
            # 'PHOSPHORUS/Pi_spxyG',
            # 'PHOSPHORUS/V25_spxyG',
            # 'PLUMS/Brix_spxy70',
            # 'PLUMS/Firmness_spxy70', # PLS 0.29 - TabPFN: 0.26 *
            # 'QUARTZ/Quartz_spxy70',
            # 'TABLET/Escitalopramt_310_Zhao',
            # 'WOOD_density/WOOD_Density_402_Olale',
            # 'WOOD_density/WOOD_N_402_Olale', # PLS 0.050 - TabPFN: 0.048
            ]

DATASETS_PATHS = [f"data/regression/{ds}" for ds in DATASETS]
dataset_config = DatasetConfigs(DATASETS_PATHS, task_type="regression")
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset_config,
    name="AOM_PLS",
    verbose=1,
    random_state=42,
    n_jobs=-1,
    cache=CacheConfig(memory_warning_threshold_mb=32768),
    refit=[
        {"top_k": 1, "ranking": "rmsecv"},
        {"top_k": 1, "ranking": "mean_val"},
    ],
    workspace_path="AOM_workspace",
)

duration = time.time() - start_time
print(f"\nPipeline completed in {duration:.2f} seconds")
