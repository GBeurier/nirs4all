"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
os.environ['DISABLE_EMOJIS'] = '0'

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer, Gaussian as Gauss,
    StandardNormalVariate as SNV, SavitzkyGolay as SavGol, Haar, MultiplicativeScatterCorrection as MSC,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet, Derivate
)

from nirs4all.operators.transformations.nirs import AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC

from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.models.cirad_tf import nicon, transformer_VG, transformer, decon, customizable_nicon

from custom_NN import (
    resnet_se, inception_time, tcn_noncausal, conv_transformer,
    convmixer1d, cnn_pls_head, spectraformer,
    # additional tensorflow model builders from examples/custom_NN
    sota_cnn_attention, hybrid_cnn_lstm,
    resnet1d, senet1d, inception1d, tcn1d, attention_cnn1d, deep_resnet1d,
    transformer_nirs, resnet_nirs, nirs_resnet, nirs_inception, nirs_transformer_cnn,
    nicon_enhanced, se_resnet, spectratr_transformer
)

from nicon_custom import (
    nicon_improved, nicon_lightweight, nicon_experimental, nicon_auto_norm, nicon_batch_norm
)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_path = 'sample_data/Hiba23'
pipeline = [
    {"split": RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)},
    # {"split": SPXYSplitter(test_size=0.25)},
    # {"feature_augmentation": [
    #     MSC,
    #     SNV,
    #     RSNV,
    #     LSNV,
    #     Gauss,
    #     SavGol,
    #     Haar,
    #     FstDer,
    #     SndDer,
    #     Detrend,
    #     Wavelet('coif3')
    # ]},
    {"feature_augmentation": [
        ## PROTEIN PREPROCESSING PIPELINE ##
        # MSC,
        # [SNV, SavGol],
        # [MSC, FstDer],  # Critical for protein
        # [SNV, SndDer],   # Resolves amide peaks
        # Wavelet('coif3'),
        # Haar,
        
        ## STANDARD PREPROCESSING PIPELINE ##
        EMSC,  # or EMSC if you have it
        # [SNV, SavGol],  # NOT MSC+SNV! Pick one scatter correction
        # [EMSC, FstDer],  # MSC first to reduce noise in derivative
        # [SNV, SndDer],  # or [MSC, SavGol(deriv=2)]
        # [SNV, Detrend],  # Detrend AFTER SNV, not before
        # Wavelet('db4'),  # Daubechies-4: good for NIRS peaks
        # [SNV, Wavelet('sym5')],  # Symlet-5: good for smooth baseline
        # [AreaNormalization, SavGol],  # Custom or use StandardScaler per spectrum
        # [LSNV, SavGol],  # or RSNV depending on your implementation
        # # ## Greg Addons ##
        # [Detrend, SavGol, Derivate(order=3)],  # Triple derivative for very complex spectra
        Haar,
        # [EMSC, Wavelet('coif3')],
        # Wavelet('db8'),
        # [AreaNormalization, EMSC],
        
        ## BIG BOY PREPROCESSING PIPELINE ##
        [MSC, SavGol(deriv=3)],
        [MSC, SavGol(deriv=4)],
        [SNV, SavGol(deriv=3)],
        [SNV, SavGol(deriv=4)],
        [Detrend, SavGol, Derivate(order=3)],
        [Detrend, SavGol, Derivate(order=4)],
        [MSC, Gauss, Derivate(order=3)],
        [MSC, Gauss, Derivate(order=4)],
        [SNV, Gauss, Derivate(order=3)],
        [SNV, Gauss, Derivate(order=4)],
        [LSNV, SavGol(deriv=3)],
        [RSNV, SavGol(deriv=3)],
        [MSC, SavGol, FstDer, Derivate(order=3)],  # dérivée 1 puis 3 pour pics fins
        [SNV, SavGol, SndDer, Derivate(order=3)], # d2 puis d3, à n’utiliser que si SNR élevé
        [MSC, SavGol, Wavelet('db4')], # lissage puis ondelette détails
        [SNV, SavGol, Wavelet('sym5')], # lissage puis ondelette baseline
        [Detrend, SNV, SavGol(deriv=3)],
        [Detrend, MSC, SavGol(deriv=3)],
        [AreaNormalization, SavGol, Derivate(order=3)], # si tu as AreaNormalization
        [Haar, SNV], # rupture/franges, parcimonieux
        
        
        
        ## FAT PREPROCESSING PIPELINE ##
        # MSC,  # Important for fat scatter
        # [SNV, SavGol],
        # [MSC, FstDer],
        # [SNV, SndDer],
        # Wavelet('db8'),  # Smooth wavelets for fat peaks
        # [AreaNormalization, MSC],  # Fat often needs area normalization
        
        ## MOISTURE PREPROCESSING PIPELINE ##
        # [MSC, SavGol],
        # [SNV, FstDer],  # Water peaks show up strongly
        # [MSC, SndDer],
        # [RSNV, SavGol],  # Local scatter for heterogeneous moisture
        # Wavelet('haar'),  # Sharp edges for water absorption
    ]},
    
    RobustScaler(),
    # MinMaxScaler(clip=True),
    {"y_processing": MinMaxScaler(feature_range=(0.05, 0.95))},
    # {"y_processing": StandardScaler()},
    {
        # "model": nicon,
        "model": nicon_auto_norm,
        # "model": nicon_batch_norm,
        # "model": nicon_improved,
        # "model": nicon_lightweight,
        # "model": nicon_experimental,
        # "model": customizable_nicon,
        "train_params": {
            "epochs": 2000,
            "patience": 200,
            # "learning_rate": 0.0005,
            "batch_size": 1024,
            "cyclic_lr": True,
            "cyclic_lr_mode": "triangular2",
            "base_lr": 0.0005,
            "max_lr": 0.02,
            "step_size": 200,
            "loss": "mse",
            # "verbose": 2  # 0=silent, 2=one line per epoch
        },
        # "finetune_params": {
        #     "n_trials": 100,
        #     "verbose": 2,
        #     "sample": "hyperband",
        #     "approach": "single",
        #     "model_params": {
        #         'spatial_dropout': (float, 0.01, 0.5),
        #         'filters1': [4, 8, 16, 32, 64],
        #         'activation1': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
        #         'dropout_rate': (float, 0.01, 0.5),
        #         'filters2': [4, 8, 16, 32, 64],
        #         'activation2': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
        #         'filters3': [4, 8, 16, 32, 64],
        #         'activation3': ['relu', 'selu', 'elu', 'swish', 'gelu', 'sigmoid'],
        #         'dense_units': [4, 8, 16, 32, 64],
        #         'dense_activation': ['relu', 'selu', 'elu', 'swish', 'gelu', 'linear'],
        #     },
        #     "train_params": {
        #         "epochs": 50,
        #         "learning_rate": 0.005,
        #         "verbose": 0
        #     }
        # },
    },
    # {
    #     "model": decon,
    #     "train_params": {
    #         "epochs": 2000,
    #         "patience": 200,
    #         # "learning_rate": 0.0005,
    #         "batch_size": 1024,
    #         "cyclic_lr": True,
    #         "base_lr": 0.001,
    #         "max_lr": 0.01,
    #         "cycle_steps": 200,
    #         # "verbose": 2  # 0=silent, 2=one line per epoch
    #     },
    # },
    # { #pourri
    #     "model": spectraformer,
    #     "train_params": {
    #         "epochs": 2000,
    #         "patience": 200,
    #         # "learning_rate": 0.0005,
    #         "batch_size": 1024,
    #         "cyclic_lr": True,
    #         "base_lr": 0.0005,
    #         "max_lr": 0.01,
    #         "cycle_steps": 200,
    #         # "verbose": 2  # 0=silent, 2=one line per epoch
    #     },
    # },
    # {
    #     "model": PLSRegression(),
    #     "name": "PLS-Finetuned",
    #     "finetune_params": {
    #         "n_trials": 30,
    #         "verbose": 2,
    #         "approach": "single",
    #         "eval_mode": "best",
    #         "sample": "grid",
    #         "model_params": {
    #             'n_components': ('int', 1, 40),
    #         },
    #     }
    # },
    # { # good
    #     "model": resnet_se,
    #     "name": "ResNetSE",
    #     "params": {"filters": [32, 64, 128], "se_ratio": 8, "kernel_size": 9, "dropout": 0.2},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # { # bad
    #     "model": inception_time,
    #     "name": "InceptionTime",
    #     "params": {"filters": 64, "kernels": (5, 9, 15), "blocks": 3, "dropout": 0.2},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # { # bad
    #     "model": tcn_noncausal,
    #     "name": "TCN",
    #     "params": {"filters": [64, 64, 128], "dilations": [1, 2, 4, 8], "kernel_size": 7, "spatial_dropout": 0.1},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # {
    #     # bad
    #     "model": conv_transformer,
    #     "name": "ConvTransformer",
    #     "params": {"conv_filters": [64, 128], "conv_kernel": 7, "conv_stride": 2, "heads": 4, "model_dim": 128, "blocks": 2},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # { # bof
    #     "model": convmixer1d,
    #     "name": "ConvMixer1D",
    #     "params": {"dim": 128, "depth": 6, "patch_size": 5, "dw_kernel": 9, "dropout": 0.1},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # { # bad
    #     "model": cnn_pls_head,
    #     "name": "CNN+PLSHead",
    #     "params": {"filters": [32, 64, 128], "kernel_size": 7, "pool": 2, "dropout": 0.2},
    #     "train_params": {
    #         "epochs": 2000, "patience": 200,
    #         "batch_size": 1024, "cyclic_lr": True,
    #         "base_lr": 1e-3, "max_lr": 1e-2, "cycle_steps": 200
    #     }
    # },
    # { # shit but patience to be changed
    #     "model": sota_cnn_attention,
    #     "name": "SOTA-CNN-Attention",
    #     "train_params": {"epochs": 500, "patience": 50, "batch_size": 256}
    # },
    # { #correct
    #     "model": hybrid_cnn_lstm,
    #     "name": "HybridCNN-LSTM",
    #     "train_params": {"epochs": 500, "patience": 50, "batch_size": 256}
    # },
    # { # bof
    #     "model": resnet1d,
    #     "name": "ResNet1D",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # { # pourri
    #     "model": senet1d,
    #     "name": "SENet1D",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # { # shit
    #     "model": inception1d,
    #     "name": "Inception1D",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": tcn1d,
    #     "name": "TCN1D",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": attention_cnn1d,
    #     "name": "AttentionCNN1D",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": deep_resnet1d,
    #     "name": "DeepResNet1D",
    #     "train_params": {"epochs": 1500, "patience": 150, "batch_size": 512}
    # },
    # {
    #     "model": transformer_nirs,
    #     "name": "TransformerNIRS",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 256}
    # },
    # {
    #     "model": resnet_nirs,
    #     "name": "ResNetNIRS",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": nirs_resnet,
    #     "name": "NIRS-ResNet",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": nirs_inception,
    #     "name": "NIRS-Inception",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": nirs_transformer_cnn,
    #     "name": "NIRS-Transformer-CNN",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 256}
    # },
    # {
    #     "model": nicon_enhanced,
    #     "name": "NICON-Enhanced",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": se_resnet,
    #     "name": "SE-ResNet",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 512}
    # },
    # {
    #     "model": spectratr_transformer,
    #     "name": "SpectraTr-Transformer",
    #     "train_params": {"epochs": 1000, "patience": 100, "batch_size": 256}
    # },
    # RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    # XGBRegressor(n_estimators=50, max_depth=10, random_state=42),
    # LGBMRegressor(n_estimators=50, max_depth=10, random_state=42)
]
# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Hiba")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models
top_models = predictions.top(best_model_count, ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")
top_models[0].save_to_csv("Q1_regression_best_model.csv")

# # Create visualizations
analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse')
fig2 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse', rank_partition='test')
# fig3 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse', rank_partition='train')
plt.show()
