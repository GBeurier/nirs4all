import argparse
from dotenv import load_dotenv
from pathlib import Path
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import SavitzkyGolay, ASLSBaseline

from huggingface_hub import login
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q1 Regression Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Hugging Face login for TabPFN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env or environment. TabPFN may not work properly.")

# Clear GPU cache if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##========================== Main Pipeline Configuration =========================##
# Configuration variables
DATA_PATH = ['sample_data/Hiba/LDMC_vera', 'sample_data/Hiba/LDMC_hiba', 'sample_data/Hiba/SLA_vera', 'sample_data/Hiba/SLA_hiba']
AGGREGATION_KEY = "ID"  # None
TASK_TYPE = "regression"  # "classification" or "regression"

TabPFNModel = TabPFNRegressor if TASK_TYPE == "regression" else TabPFNClassifier
tabpfn_real_path = 'tabpfn-v2.5-regressor-v2.5_real.ckpt' if TASK_TYPE == "regression" else 'tabpfn-v2.5-classifier-v2.5_real.ckpt'
# Define the pipeline
pipeline = [
    ASLSBaseline(),
    {"split": SPXYGFold(n_splits=1, random_state=42), "group": AGGREGATION_KEY}, # Comment if train and test are provided
    {"split": SPXYGFold(n_splits=3, random_state=42), "group": AGGREGATION_KEY},
    {"y_processing": StandardScaler()},
    StandardScaler(),
    SavitzkyGolay(),
    PCA(n_components=0.99, random_state=42, whiten=True), # PCA(50)
    StandardScaler(),
    PowerTransformer(),
    {
        'model': {
            'framework': 'autogluon',
            'params': {
                'presets': 'extreme_quality',
                'time_limit': 3600,
                'num_bag_folds': 5,
                'random_state': 42,
            }
        },
        "name": "AutoGluon",
    },
    {
        "model": TabPFNModel(n_estimators=16, device='cuda', random_state=42),
        "name": "TabPFN",
    },
    {
        "model": TabPFNModel(n_estimators=16, device='cuda', random_state=42, model_path=tabpfn_real_path),
        "name": "TabPFN-real",
    },
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "SOTA")
dataset_config = DatasetConfigs(DATA_PATH, TASK_TYPE)

# Run the pipeline
runner = PipelineRunner(save_files=True, verbose=0, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analyze and display top performing models
best_model_count = 5
ranking_metric = ['rmse', 'r2', 'mape']

for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    dataset_predictions = dataset_prediction['run_predictions']

    # Display top performing models
    print("Top Predictions (row wise):")
    print("-" * 80)
    top_models_val = dataset_predictions.top(best_model_count, ranking_metric)
    print(f"\nTop {best_model_count} models based on validation {ranking_metric[0].upper()}:")
    for idx, prediction in enumerate(top_models_val):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=ranking_metric)} - {prediction['preprocessings']}")

    print(f"\nTop {best_model_count} models based on test {ranking_metric[0].upper()}:")
    top_models_test = dataset_predictions.top(best_model_count, ranking_metric, rank_partition='test')
    for idx, prediction in enumerate(top_models_test):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=ranking_metric)} - {prediction['preprocessings']}")

    # Print aggregated results if aggregation_key is provided
    if aggregation_key is not None:
        print("*" * 80)
        print(f"\n Top Predictions (aggregated by {aggregation_key}):")
        print("-" * 80)
        # Display top performing models
        top_models_val = dataset_predictions.top(best_model_count, ranking_metric[0], aggregate=aggregation_key)
        print(f"\nTop {best_model_count} models based on validation {ranking_metric[0].upper()}:")
        for idx, prediction in enumerate(top_models_val):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=ranking_metric)} - {prediction['preprocessings']}")

        print(f"\nTop {best_model_count} models based on test {ranking_metric[0].upper()}:")
        top_models_test = dataset_predictions.top(best_model_count, ranking_metric[0], rank_partition='test', aggregate=aggregation_key)
        for idx, prediction in enumerate(top_models_test):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=ranking_metric)} - {prediction['preprocessings']}")