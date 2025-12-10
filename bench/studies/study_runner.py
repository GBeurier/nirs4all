"""
Study Runner - User Configuration File
=======================================
Configure your NIRS analysis study here and run with: python study_runner.py

This file inherits from StudyRunner base class and allows you to configure:
- Datasets and aggregation keys
- All training parameters (simple scalars via CLI)
- Complex objects (GLOBAL_PP, TABPFN_PP, TRANSFER_PP_CONFIG) via direct Python

Just edit the configuration below and run this file.
"""

import os
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from nirs4all.operators.transforms import (
    Wavelet, WaveletFeatures, WaveletPCA, WaveletSVD,
    StandardNormalVariate, FirstDerivative, SecondDerivative, SavitzkyGolay,
    MultiplicativeScatterCorrection, Detrend, Gaussian, Haar,
    RobustStandardNormalVariate,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)

from study_base_runner import StudyRunner


# ============================================================================
# CONFIGURATION
# ============================================================================

class MyStudy(StudyRunner):
    """Configure your study by editing this class."""

    def __init__(self):
        super().__init__()

        # ====================================================================
        # DATASET CONFIGURATION
        # ====================================================================

        DATASETS_FOLDER = '../_datasets/hiba/'
        SUB_FOLDER_LIST = [
            'LDMC',
        ]

        self.folder_list = [os.path.join(DATASETS_FOLDER, sub) for sub in SUB_FOLDER_LIST]
        self.aggregation_key_list = ["ID" for _ in self.folder_list]

        # ====================================================================
        # TEST MODE (set to True for quick testing)
        # ====================================================================

        self.test_mode = True

        # ====================================================================
        # TRAINING CONFIGURATION - Simple Parameters
        # ====================================================================

        # Transfer Preprocessing Selection (simple mode)
        self.transfer_pp_preset = None  # "fast", "balanced", "comprehensive"
        self.transfer_pp_selected = 10

        # PLS/OPLS
        self.pls_pp_count = 40
        self.pls_pp_top_selected = 10
        self.pls_trials = 20
        self.opls_trials = 30

        # Other models
        self.test_lwpls = False
        self.ridge_trials = 20

        # TabPFN (simple)
        self.tabpfn_trials = 10
        self.tabpfn_model_variants = ['default', 'real', 'low-skew', 'small-samples']
        self.tabpfn_pp_max_count = 20
        self.tabpfn_pp_max_size = 3

        # Execution
        self.device = "cuda"
        self.verbose = 1
        self.show_plots = False

        # ====================================================================
        # REPORTING CONFIGURATION
        # ====================================================================

        self.workspace_path = "wk"
        self.output_dir = "reports"
        self.report_mode = "aggregated"  # 'raw', 'aggregated', or 'both'
        self.include_dataset_viz = True

        # Report aggregation key (None = use first from aggregation_key_list if unique)
        self.report_aggregation_key = None

        # Model filtering: exclude specific models from reports (case-insensitive partial match)
        self.report_exclude_models = ["KernelPLS"]  # Default exclusions

        # Model renaming: map model names to display names (case-insensitive partial match)
        self.report_model_rename_map = {
            "dict": "nicon",
            "tabpfn": "transformer",
            "tunedtabpfn": "tuned_transformer",
        }

        # ====================================================================
        # ADVANCED CONFIGURATION - Python Objects (optional)
        # ====================================================================
        # Uncomment and configure these for advanced control.
        # When set, they override simple parameters and use direct function calls.

        self.transfer_pp_config = {
            'preset': None,  # Disable preset to use custom config
            'run_stage2': False,
            'stage2_top_k': 15,
            'stage2_max_depth': 3,
            'run_stage3': True,
            'stage3_top_k': 10,
            'stage3_max_order': 2,
            'run_stage4': False,
            'n_components': 20,
            'k_neighbors': 10,
            'n_jobs': -1,
        }

        # --- GLOBAL_PP: Preprocessing search space for TransferPreprocessingSelector ---
        # Now uses direct transformer objects instead of string names (more explicit & IDE-friendly)
        # self.global_pp = {
        #     "_cartesian_": [
        #         {"_or_": [None, MultiplicativeScatterCorrection(), StandardNormalVariate(), EMSC(), RobustStandardNormalVariate()]},
        #         {"_or_": [None, SavitzkyGolay(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2), Gaussian(order=2, sigma=2)]},
        #         {"_or_": [None, FirstDerivative(), SecondDerivative(), SavitzkyGolay(deriv=1), SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(deriv=2)]},
        #         {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("sym5"), Wavelet("coif3")]},
        #     ],
        # }

        # --- TABPFN_PP: Custom preprocessing pipelines for TabPFN ---
        # self.tabpfn_pp = [
        #     PCA(n_components=50),
        #     PCA(n_components=100),
        #     TruncatedSVD(n_components=50),
        #     SparseRandomProjection(n_components=100),
        #     GaussianRandomProjection(n_components=100),
        #     [Wavelet('haar'), PCA(n_components=50)],
        #     [Wavelet('db4'), PCA(n_components=50)],
        #     [StandardNormalVariate(), PCA(n_components=100)],
        #     [SavitzkyGolay(), PCA(n_components=100)],
        #     [FirstDerivative(), PCA(n_components=100)],
        #     WaveletFeatures(wavelet='db4', max_level=5, n_coeffs_per_level=10),
        #     WaveletPCA(wavelet='coif3', max_level=4, n_components_per_level=5),
        #     WaveletSVD(wavelet='db4', max_level=4, n_components_per_level=5),
        # ]

        # --- TRANSFER_PP_CONFIG: Full TransferPreprocessingSelector configuration ---


        # ====================================================================
        # EXECUTION CONTROL (optional)
        # ====================================================================

        # self.skip_training = False
        # self.skip_reporting = False


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    study = MyStudy()
    exit(study.run())
