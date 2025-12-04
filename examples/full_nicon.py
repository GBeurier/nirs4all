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

from matplotlib import pyplot as plt

# Third-party imports - ML Models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, RandomForestClassifier
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# NIRS4All imports - Sample augmentation transforms
from nirs4all.operators.transforms import (
    Rotate_Translate,
    Spline_Y_Perturbations,
    Spline_X_Simplification,
    GaussianAdditiveNoise,
    MultiplicativeNoise,
    LinearBaselineDrift,
    PolynomialBaselineDrift,
    WavelengthShift,
    WavelengthStretch,
    LocalWavelengthWarp,
    SmoothMagnitudeWarp,
    GaussianSmoothingJitter,
    UnsharpSpectralMask,
    ChannelDropout,
    MixupAugmenter,
    ScatterSimulationMSC,
    # Feature augmentation transforms (commented - uncomment if using feature_augmentation)
    Detrend, FirstDerivative, SecondDerivative,
    Gaussian, StandardNormalVariate, SavitzkyGolay,
    Haar, MultiplicativeScatterCorrection,
    RobustStandardNormalVariate, LocalStandardNormalVariate, Wavelet,
)

from nirs4all.operators.transforms import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer,
    Gaussian, StandardNormalVariate as SNV, SavitzkyGolay as SavGol,
    Haar, MultiplicativeScatterCorrection as MSC, Derivate,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet,
    CARS,
    MCUVE
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)
from nirs4all.operators.models.sklearn.fckpls import FCKPLS

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.models.pytorch.nicon import nicon as nicon_torch, customizable_nicon as customizable_nicon_torch, customizable_decon as customizable_decon_torch, customizable_nicon_classification as customizable_nicon_classification_torch
from nirs4all.operators.models.tensorflow.nicon import nicon as nicon_tf, customizable_nicon as customizable_nicon_tf, customizable_nicon_classification as customizable_nicon_classification_tf, customizable_decon as customizable_decon_tf
from nirs4all.operators.splitters import SPXYSplitter, SPXYGFold, BinnedStratifiedGroupKFold
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Add, Multiply, Concatenate, Reshape, Average, Lambda,
    AveragePooling1D, ZeroPadding1D, Activation, AlphaDropout, Flatten
)
import tensorflow as tf
from keras.initializers import lecun_normal, he_normal
from nirs4all.utils.backend import framework
from nirs4all.operators.models.sklearn import (
    PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS
)

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

runner = PipelineRunner(save_files=True, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)



if args.show:
    plt.show()
