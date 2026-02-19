"""
Batch Regression Analysis - Comprehensive Pipeline for Multiple Datasets
=========================================================================
Runs complex regression analysis on all datasets in the regression/ folder.
Includes ML models, DL models, and various preprocessing combinations.
Generates top_k plots, heatmaps, candlestick plots, and histograms.
"""

# Standard library imports
import argparse
import os

os.environ['DISABLE_EMOJIS'] = '0'

import tensorflow as tf
from keras.initializers import he_normal, lecun_normal
from matplotlib import pyplot as plt

# Third-party imports - ML Models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.layers import (
    Activation,
    Add,
    AlphaDropout,
    Average,
    AveragePooling1D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    Multiply,
    Reshape,
    SpatialDropout1D,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model, Sequential

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models.pytorch.nicon import customizable_decon as customizable_decon_torch
from nirs4all.operators.models.pytorch.nicon import customizable_nicon as customizable_nicon_torch
from nirs4all.operators.models.pytorch.nicon import customizable_nicon_classification as customizable_nicon_classification_torch
from nirs4all.operators.models.pytorch.nicon import nicon as nicon_torch
from nirs4all.operators.models.sklearn import IKPLS, MBPLS, OPLS, OPLSDA, PLSDA, SIMPLS, DiPLS, SparsePLS
from nirs4all.operators.models.sklearn.fckpls import FCKPLS
from nirs4all.operators.models.tensorflow.nicon import customizable_decon as customizable_decon_tf
from nirs4all.operators.models.tensorflow.nicon import customizable_nicon as customizable_nicon_tf
from nirs4all.operators.models.tensorflow.nicon import customizable_nicon_classification as customizable_nicon_classification_tf
from nirs4all.operators.models.tensorflow.nicon import nicon as nicon_tf
from nirs4all.operators.splitters import BinnedStratifiedGroupKFold, SPXYGFold, SPXYSplitter

# NIRS4All imports - Sample augmentation transforms
from nirs4all.operators.transforms import (
    CARS,
    MCUVE,
    ChannelDropout,
    Derivate,
    # Feature augmentation transforms (commented - uncomment if using feature_augmentation)
    Detrend,
    FirstDerivative,
    Gaussian,
    GaussianAdditiveNoise,
    GaussianSmoothingJitter,
    Haar,
    LinearBaselineDrift,
    LocalStandardNormalVariate,
    LocalWavelengthWarp,
    MixupAugmenter,
    MultiplicativeNoise,
    MultiplicativeScatterCorrection,
    PolynomialBaselineDrift,
    RobustStandardNormalVariate,
    Rotate_Translate,
    SavitzkyGolay,
    ScatterSimulationMSC,
    SecondDerivative,
    SmoothMagnitudeWarp,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    StandardNormalVariate,
    UnsharpSpectralMask,
    WavelengthShift,
    WavelengthStretch,
    Wavelet,
)
from nirs4all.operators.transforms import FirstDerivative as FstDer
from nirs4all.operators.transforms import LocalStandardNormalVariate as LSNV
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import RobustStandardNormalVariate as RSNV
from nirs4all.operators.transforms import SavitzkyGolay as SavGol
from nirs4all.operators.transforms import SecondDerivative as SndDer
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.operators.transforms.nirs import AreaNormalization
from nirs4all.operators.transforms.nirs import ExtendedMultiplicativeScatterCorrection as EMSC
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.utils.backend import framework
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Batch Regression Analysis')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at the end')
args = parser.parse_args()

print("=" * 80)
print("BATCH REGRESSION ANALYSIS")
print("=" * 80)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# All regression datasets
data_paths = [
    # Regression
    # 'sample_data/nitro_reg_merged/Digestibility_0.8',
    # 'sample_data/nitro_reg_merged/Hardness_0.8',
    # 'sample_data/nitro_reg_merged/Tannin_0.8'

    # Classification
    # 'sample_data/nitro_classif_merged/Hardness_custom2',
    'sample_data/nitro_classif_merged/Digestibility_custom2',
    # 'sample_data/nitro_classif_merged/Digestibility_custom3',
    # 'sample_data/nitro_classif_merged/Digestibility_custom5',
    # 'sample_data/nitro_classif_merged/Hardness_custom4',
    # 'sample_data/nitro_classif_merged/Tannin_custom2',
    # 'sample_data/nitro_classif_merged/Tannin_custom3',
]

pipeline = [
    {
        "split": SPXYGFold,
        "split_params": {
            "n_splits": 1,
            "test_size": 0.2,
            "aggregation": "mean"
        },
        "group": "ID"
    },
    # {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"},
    {"model": PLSDA}
]

# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline_config = PipelineConfigs(pipeline, name="BatchRegression")
dataset_config = DatasetConfigs(data_paths)

runner = PipelineRunner(save_artifacts=True, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

if args.show:
    plt.show()
