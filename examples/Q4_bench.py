from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# from deprec.controllers.models import data  # Comment out problematic import
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.operators.transformations import Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer


# Clear old results to ensure fresh training with metadata
from pathlib import Path
import shutil
results_path = Path("./results")
if results_path.exists():
    shutil.rmtree(results_path)
    print("🧹 Cleared old results to ensure fresh training")
list_of_preprocessors = [Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection]
pipeline = [
    # Normalize the spectra reflectance
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},
    {"feature_augmentation": list_of_preprocessors},
    StandardScaler(),
    # ShuffleSplit(n_splits=3, test_size=.25),
    # {"model": PLSRegression(10)},
    {"model": nicon, "name": "nicon_model", "train_params": {"epochs": 500, "patience": 50, "verbose": 0}},
]

# create pipeline config
pipeline_config = PipelineConfigs(pipeline)

path = ['sample_data/regression']
# {
#     "train_x": "sample_data/regression/Xcal.csv.gz"
# }
dataset_config = DatasetConfigs('sample_data/regression')

dataset = dataset_config.get_dataset_at(0)
print(f"Dataset name: {dataset.name}, number of samples: {dataset}")

# # Runner setup with spinner enabled (default is True, but let's be explicit)
runner = PipelineRunner(save_files=True, verbose=0)  # CHANGED: Enable model saving for testing
print("🔄 Running pipeline with spinner enabled - watch for loading animations during model training!")
run_predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)
