################################################################################
##################### Import libraries #########################################
################################################################################
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import *
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

import matplotlib.pyplot as plt
import numpy as np

import os

################################################################################
##################### Datasets setting #########################################
################################################################################
# ----- 1 seul dataset à analyser -----------------
PATH = 'sample_data/regression'        # Nom du dossier contenant les données
DELIMITER=";"                                                                   # Séparateur de colonnes des fichiers csv (";" ou "," ou "\t")
DECIMAL="."                                                                     # Décimale utilisée dans le csv ("." ou ",")
datasets={
    "folder": PATH,
    "params": {"delimiter": DELIMITER, "decimal_separator": DECIMAL}
}

# ----- Plusieurs datasets dans un répertoire -----------------
# PATH='D:/R/R projects/INRAE_INSPIR_ble/out/DBcurated/'
# DELIMITER=";"
# DECIMAL=","
# def get_dataset_list(root):
#     return [
#         {
#             "folder": os.path.join(root, name),
#             "params": {"delimiter": DELIMITER, "decimal_separator": DECIMAL},
#         }
#         for name in sorted(os.listdir(root))
#         if os.path.isdir(os.path.join(root, name))
#     ]
# datasets = get_dataset_list(PATH)


################################################################################
##################### X transformations ########################################
################################################################################
X_SCALER = MinMaxScaler()                                                       # Autre scalers possibles : StandardScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), LogTransform()
LIST_OF_PREPROCESSORS = [Detrend, FirstDerivative,                              # Liste des prétraitements souhaités (possibilités: RobustNormalVariate, Wavelet, SecondDerivative, Baseline, Detrend, Gaussian)
                         StandardNormalVariate, SavitzkyGolay, Haar,
                         MultiplicativeScatterCorrection]


################################################################################
##################### Y transformations ########################################
################################################################################
Y_SCALER = MinMaxScaler()                                                       # Autre scalers possibles : StandardScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), LogTransform()


################################################################################
##################### Model definition #########################################
################################################################################
PLS_COMP_MIN=1                                                                  # Nombre minimum de composante de la PLS à tester
PLS_COMP_MAX=20                                                                 # Nombre maximum de composante de la PLS à tester
PLS_COMP_STEP=2                                                                 # Step d'incrémentation entre nombre de composantes testés

################################################################################
##################### Inner split for finetune #################################
################################################################################
CROSS_VAL_FOLDS_NB=3                                                            # Nombre de folds pour la validation croisée (CV)
CROSS_VAL_PROPORTION=0.25                                                       # Proportion d'échantillons conservés pour la validation interne dans la CV
splitting_strategy = ShuffleSplit(n_splits=CROSS_VAL_FOLDS_NB,
                                  test_size=CROSS_VAL_PROPORTION)


################################################################################
##################### Gather configs for each dataset ##########################
################################################################################
pipeline = [
    # "chart_2d",
    X_SCALER,
    # "chart_3d",
    {"y_processing": Y_SCALER},
    {"feature_augmentation": { "_or_": LIST_OF_PREPROCESSORS, "size":[1,(1,2)], "count":64 }}, # Generate all elements of size 1 and of order 1 or 2 (ie. "Gaussian", ["SavitzkyGolay", "Log"], etc.)
    splitting_strategy,
]

for i in range(PLS_COMP_MIN, PLS_COMP_MAX, PLS_COMP_STEP):
    model = {
        "name": f"PLS-{i}_cp",
        "model": PLSRegression(n_components=i)
    }
    pipeline.append(model)

pipeline_config = PipelineConfigs(pipeline, "pipeline_Q1")
dataset_config = DatasetConfigs(datasets)


################################################################################
##################### Run configs ##############################################
################################################################################
# Create pipeline
runner = PipelineRunner(save_files=True)
run_predictions, other_predictions = runner.run(pipeline_config, dataset_config)