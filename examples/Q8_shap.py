#!/usr/bin/env python3
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.models.cirad_tf import nicon
print("Q8 - SHAP Model Explanation Example")

pipeline = [
    {"y_processing": MinMaxScaler},
    MinMaxScaler,
    Gaussian,
    SavitzkyGolay,
    StandardNormalVariate,
    PLSRegression(n_components=16),
]

pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/regression_2'])

print("Training models...")
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

best_prediction = predictions.top_k(1, metric='rmse', partition="test")[0]
print(f"Best model: {best_prediction['model_name']} (RMSE: {best_prediction['rmse']:.4f})")

print("Running SHAP analysis...")
# Use the same runner instance to call explain

# shap_params = {
#     'n_samples': 200,
#     'explainer_type': 'auto',
#     'visualizations': ['spectral', 'waterfall', 'beeswarm'],  #'summary', --- IGNORE ---
#     'bin_size': 50,        # Number of wavelengths per bin
#     'bin_stride': 25,      # Step between bins (50% overlap)
#     'bin_aggregation': 'mean'  # Mean of SHAP values in each bin (possible: 'sum', 'sum_abs', 'mean', 'mean_abs')
# }


shap_params = {
    'n_samples': 200,
    'explainer_type': 'auto', # Options: 'auto', 'tree', 'kernel', 'deep', 'linear'
    'visualizations': ['spectral', 'waterfall', 'beeswarm'],

    # Different bin sizes for each visualization
    'bin_size': {
        'spectral': 10,      # Fine-grained for spectral overview
        'waterfall': 20,     # Coarser for waterfall (fewer bars)
        'beeswarm': 20       # Medium for beeswarm
    },

    # Different strides (50% overlap for all, but matches bin_size)
    'bin_stride': {
        'spectral': 2,      # 50% overlap with bin_size=20
        'waterfall': 10,     # 50% overlap with bin_size=50
        'beeswarm': 20       # 50% overlap with bin_size=30
    },

    # Different aggregation methods
    'bin_aggregation': {
        'spectral': 'mean',      # Total importance per region
        'waterfall': 'mean',    # Average per wavelength
        'beeswarm': 'mean'   # Absolute importance sum
    }
}

shap_results, output_dir = runner.explain(best_prediction, dataset_config, shap_params=shap_params)
