"""
Sample configuration file for illustrating all pipeline declarations formats.
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.operators.models.cirad_tf import nicon, customizable_nicon

preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/regression'

# Build the pipeline with hyperparameter optimization
pipeline = [
    MinMaxScaler(feature_range=(0.1, 0.8)),  # instance with parameters
    "chart_2d", # string to controller
    {"y_processing": MinMaxScaler},  # class
    {"feature_augmentation": {"_or_": preprocessing_options, "size": [1, (1, 2)], "count": 5}},  # special node for generation
    ShuffleSplit(n_splits=3, test_size=0.25),  # instance
    {  # dict for operators
        "class": "sklearn.model_selection._split.ShuffleSplit",
        "params": {
            "n_splits": 3,
            "test_size": 0.25
        }
    },
    "my/super/transformer.pkl",  # string to saved transformer file
    {  # dict for generator of models
        "_range_": [1, 12, 2],
        "param": "n_components",
        "model": {
            "class": "sklearn.cross_decomposition._pls.PLSRegression"
        }
    },
    {  # dict for model
        "name": "PLS-3_components",
        "model": {
            "class": "sklearn.cross_decomposition._pls.PLSRegression",
            "params": {
                "n_components": {

                }
            }
        }
    },
    nicon,  # function as model
    {  # complex dict for model with finetuning
        "model": PLSRegression(),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 20,
            "verbose": 2,
            "approach": "single",
            "eval_mode": "best",
            "sample": "grid",
            "model_params": {
                'n_components': ('int', 1, 30),
            },
        }
    },
    {  # complex dict for model as function (NN) with finetuning
        "model": customizable_nicon,
        "name": "PLS-Default",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 10,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 250,
            "verbose": 0
        }
    },
    "My_awesome_model.pkl",  # string to saved model file
    "My_awesome_tf_model.h5",  # string to saved TensorFlow model directory
    {"model": PLSRegression(10), "name": "PLS_10_components"},  # instance with parameters
    {"source_file": "my_model.python", "class": "MyAwesomeModel"},  # custom code file with class name
    "sklearn.linear_model.Ridge"  # string to class
]
